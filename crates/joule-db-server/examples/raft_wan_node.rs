//! Standalone Networked TCP Raft Node + Link-Partition Proxy for Multi-Host WAN Testing.
//!
//! Boots a real OS process running `RaftNode<KvStateMachine, TcpRaftTransport>` + `RaftRpcServer`
//! with durable on-disk `raft.wal`, local TCP link-severing relays to remote WAN peers, and a
//! line-based TCP control port (`WRITE <key> <val>`, `READ <key>`, `STATUS`, `BLOCK <peer>`,
//! `ALLOW <peer>`) so `bench/raft_multihost_wan.py` can orchestrate 5-node cross-host WAN
//! partition + `SIGKILL` recovery experiments verified by official Clojure `elle/elle 0.2.8`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use joule_db_server::raft::{
    BatchReplicationConfig, ClusterConfig, Command, KvStateMachine, RaftConfig, RaftNode,
};
use joule_db_server::raft_server::{election_loop, replication_loop, RaftRpcServer};
use joule_db_server::raft_transport::TcpRaftTransport;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Args:
    // --id <node_id> --raft-bind <ip:port> --ctrl-bind <ip:port> --data <dir>
    // --peer <peer_id>=<ip:port> [--peer ...]
    let args: Vec<String> = std::env::args().collect();
    let mut node_id = String::new();
    let mut raft_bind = String::new();
    let mut ctrl_bind = String::new();
    let mut data_dir = PathBuf::new();
    let mut remote_peers: HashMap<String, String> = HashMap::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--id" if i + 1 < args.len() => {
                node_id = args[i + 1].clone();
                i += 2;
            }
            "--raft-bind" if i + 1 < args.len() => {
                raft_bind = args[i + 1].clone();
                i += 2;
            }
            "--ctrl-bind" if i + 1 < args.len() => {
                ctrl_bind = args[i + 1].clone();
                i += 2;
            }
            "--data" if i + 1 < args.len() => {
                data_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--peer" if i + 1 < args.len() => {
                if let Some((pid, addr)) = args[i + 1].split_once('=') {
                    remote_peers.insert(pid.to_string(), addr.to_string());
                }
                i += 2;
            }
            _ => i += 1,
        }
    }

    if node_id.is_empty() || raft_bind.is_empty() || ctrl_bind.is_empty() {
        eprintln!(
            "Usage: raft_wan_node --id <id> --raft-bind <ip:port> --ctrl-bind <ip:port> --data <dir> --peer <id>=<ip:port>..."
        );
        std::process::exit(2);
    }

    std::fs::create_dir_all(&data_dir)?;

    // Spawn local outbound link-fault-injection TCP relays for every peer so we can sever
    // both new and in-flight WAN TCP sockets instantaneously without root/iptables.
    let mut local_peer_addrs: HashMap<String, String> = HashMap::new();
    let mut link_controls: HashMap<String, (Arc<AtomicBool>, watch::Sender<u64>)> = HashMap::new();

    for (pid, target_addr) in &remote_peers {
        let relay = TcpListener::bind("127.0.0.1:0").await?;
        let relay_addr = relay.local_addr()?.to_string();
        local_peer_addrs.insert(pid.clone(), relay_addr);

        let allowed = Arc::new(AtomicBool::new(true));
        let (sever_tx, sever_rx) = watch::channel(0u64);
        link_controls.insert(pid.clone(), (allowed.clone(), sever_tx));

        let upstream_target = target_addr.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut client_sock, _)) = relay.accept().await else {
                    break;
                };
                if !allowed.load(Ordering::SeqCst) {
                    let _ = client_sock.shutdown().await;
                    continue;
                }
                let tgt = upstream_target.clone();
                let mut sever_watch = sever_rx.clone();
                let _ = sever_watch.borrow_and_update();
                tokio::spawn(async move {
                    let Ok(Ok(mut upstream)) = tokio::time::timeout(
                        Duration::from_millis(1200),
                        TcpStream::connect(&tgt),
                    )
                    .await
                    else {
                        let _ = client_sock.shutdown().await;
                        return;
                    };
                    let _ = client_sock.set_nodelay(true);
                    let _ = upstream.set_nodelay(true);
                    tokio::select! {
                        _ = tokio::io::copy_bidirectional(&mut client_sock, &mut upstream) => {}
                        _ = sever_watch.changed() => {
                            let _ = client_sock.shutdown().await;
                            let _ = upstream.shutdown().await;
                        }
                    }
                });
            }
        });
    }

    let mut members: HashSet<String> = remote_peers.keys().cloned().collect();
    members.insert(node_id.clone());
    let cluster_cfg = ClusterConfig::new(members);

    let mut cfg = RaftConfig::new(node_id.clone());
    // WAN-tuned election & heartbeat timers (cross-host Tailscale RTT ~15-45 ms)
    cfg.election_timeout_min = Duration::from_millis(450);
    cfg.election_timeout_max = Duration::from_millis(850);
    cfg.heartbeat_interval = Duration::from_millis(110);
    cfg.enable_pre_vote = false;
    cfg.batch_config = BatchReplicationConfig {
        enabled: false,
        ..Default::default()
    };
    cfg.data_dir = Some(data_dir);

    let transport = Arc::new(TcpRaftTransport::new(node_id.clone(), local_peer_addrs));
    let sm = KvStateMachine::default();
    let node = Arc::new(RaftNode::new(cfg, sm, transport, cluster_cfg));
    node.start().await;

    let rpc_server = RaftRpcServer::new(node.clone(), raft_bind.clone());
    tokio::spawn(async move {
        let _ = rpc_server.run().await;
    });
    let en = node.clone();
    tokio::spawn(async move {
        election_loop(en).await;
    });
    let rn = node.clone();
    tokio::spawn(async move {
        replication_loop(rn).await;
    });

    let ctrl_listener = TcpListener::bind(&ctrl_bind).await?;
    let link_controls = Arc::new(link_controls);

    loop {
        let (stream, _) = ctrl_listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let node_ref = node.clone();
        let links_ref = link_controls.clone();
        let my_id = node_id.clone();

        tokio::spawn(async move {
            let (rd, mut wr) = stream.into_split();
            let mut reader = BufReader::new(rd);
            let mut line = String::new();
            while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                let parts: Vec<&str> = line.trim().split_whitespace().collect();
                let reply = match parts.as_slice() {
                    ["STATUS"] => {
                        let role = format!("{:?}", node_ref.state().await);
                        let term = node_ref.current_term().await;
                        let commit = node_ref.commit_index().await;
                        let applied = node_ref.last_applied().await;
                        let x = node_ref
                            .with_state_machine(|sm| {
                                sm.data()
                                    .get("x")
                                    .map(|b| String::from_utf8_lossy(b).to_string())
                            })
                            .await;
                        let y = node_ref
                            .with_state_machine(|sm| {
                                sm.data()
                                    .get("y")
                                    .map(|b| String::from_utf8_lossy(b).to_string())
                            })
                            .await;
                        serde_json::json!({
                            "ok": true,
                            "id": my_id,
                            "role": role,
                            "term": term,
                            "commit": commit,
                            "applied": applied,
                            "local_x": x,
                            "local_y": y,
                        })
                    }
                    ["WRITE", key, val] => {
                        let cmd = Command::Set {
                            key: (*key).as_bytes().to_vec(),
                            value: (*val).as_bytes().to_vec(),
                        };
                        match node_ref
                            .propose_and_commit(cmd, Duration::from_millis(2200))
                            .await
                        {
                            Ok(idx) => serde_json::json!({ "ok": true, "index": idx }),
                            Err(e) => serde_json::json!({
                                "ok": false,
                                "definite_fail": matches!(
                                    e,
                                    joule_db_server::raft::RaftError::NotLeader { .. }
                                ),
                                "err": format!("{e:?}")
                            }),
                        }
                    }
                    ["READ", key] => {
                        let k = (*key).to_string();
                        match node_ref
                            .read_linearizable(
                                Duration::from_millis(1600),
                                move |sm| {
                                    sm.data()
                                        .get(&k)
                                        .map(|b| String::from_utf8_lossy(b).to_string())
                                },
                            )
                            .await
                        {
                            Ok(val) => {
                                serde_json::json!({ "ok": true, "value": val })
                            }
                            Err(e) => serde_json::json!({ "ok": false, "err": format!("{e:?}") }),
                        }
                    }
                    ["BLOCK", peer] => {
                        if let Some((flag, tx)) = links_ref.get(*peer) {
                            flag.store(false, Ordering::SeqCst);
                            let next = *tx.borrow() + 1;
                            let _ = tx.send(next);
                        }
                        serde_json::json!({ "ok": true, "blocked": peer })
                    }
                    ["ALLOW", peer] => {
                        if let Some((flag, tx)) = links_ref.get(*peer) {
                            flag.store(true, Ordering::SeqCst);
                            let next = *tx.borrow() + 1;
                            let _ = tx.send(next);
                        }
                        serde_json::json!({ "ok": true, "allowed": peer })
                    }
                    _ => serde_json::json!({ "ok": false, "err": "unknown_cmd" }),
                };
                let mut out = reply.to_string();
                out.push('\n');
                if wr.write_all(out.as_bytes()).await.is_err() {
                    break;
                }
                line.clear();
            }
        });
    }
}
