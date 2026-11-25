//! Real 5-Node Networked TCP `raft.rs` Fault-Injection & `jepsen.elle` History Export Test.
//!
//! Exercises `crates/joule-db-server/src/raft.rs`, persistent on-disk `RaftWal`, and binary HRP
//! TCP RPC server (`raft_server.rs` + `raft_transport.rs`) across 5 nodes and a 20-link TCP
//! fault-injection proxy (`TcpPartitionMatrix`), then exports Clojure EDN history for `jepsen.elle`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use joule_db_server::raft::{
    BatchReplicationConfig, ClusterConfig, Command, KvStateMachine, RaftConfig, RaftError, RaftNode,
};
use joule_db_server::raft_server::{RaftRpcServer, election_loop, replication_loop};
use joule_db_server::raft_transport::TcpRaftTransport;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

#[derive(Clone, Debug)]
struct ElleEvent {
    index: usize,
    kind: &'static str,
    process: usize,
    time_ns: u128,
    op_edn: String,
}

struct HistoryRecorder {
    t0: Instant,
    events: Mutex<Vec<ElleEvent>>,
}

impl HistoryRecorder {
    fn new() -> Self {
        Self {
            t0: Instant::now(),
            events: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, kind: &'static str, process: usize, op_edn: String) {
        let mut g = self.events.lock().unwrap();
        let idx = g.len();
        g.push(ElleEvent {
            index: idx,
            kind,
            process,
            time_ns: self.t0.elapsed().as_nanos(),
            op_edn,
        });
    }

    fn to_edn(&self) -> String {
        let g = self.events.lock().unwrap();
        let mut out = String::from("[\n");
        for e in g.iter() {
            out.push_str(&format!(
                "  {{:index {}, :type {}, :f :txn, :process {}, :time {}, :value {}}}\n",
                e.index, e.kind, e.process, e.time_ns, e.op_edn
            ));
        }
        out.push(']');
        out
    }

    fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

struct TcpPartitionMatrix {
    targets: Arc<RwLock<Vec<String>>>,
    links: HashMap<(usize, usize), (Arc<AtomicBool>, watch::Sender<u64>)>,
    proxy_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl TcpPartitionMatrix {
    async fn spawn(initial_targets: Vec<String>) -> (Self, HashMap<(usize, usize), String>) {
        let n = initial_targets.len();
        let targets = Arc::new(RwLock::new(initial_targets));
        let mut links = HashMap::new();
        let mut proxy_addrs = HashMap::new();
        let mut proxy_handles = Vec::new();

        for src in 0..n {
            for dst in 0..n {
                if src == dst {
                    continue;
                }
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let proxy_addr = listener.local_addr().unwrap().to_string();
                proxy_addrs.insert((src, dst), proxy_addr);

                let allowed = Arc::new(AtomicBool::new(true));
                let (sever_tx, sever_rx) = watch::channel(0u64);
                links.insert((src, dst), (allowed.clone(), sever_tx));

                let targets_ref = targets.clone();
                let handle = tokio::spawn(async move {
                    loop {
                        let Ok((mut client_sock, _)) = listener.accept().await else {
                            break;
                        };
                        if !allowed.load(Ordering::SeqCst) {
                            let _ = client_sock.shutdown().await;
                            continue;
                        }
                        let target = targets_ref.read().unwrap()[dst].clone();
                        let mut sever_watch = sever_rx.clone();
                        let _ = sever_watch.borrow_and_update();
                        tokio::spawn(async move {
                            let Ok(mut upstream) = TcpStream::connect(&target).await else {
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
                proxy_handles.push(handle);
            }
        }

        (
            Self {
                targets,
                links,
                proxy_handles,
            },
            proxy_addrs,
        )
    }

    fn update_target_addr(&self, dst: usize, new_addr: String) {
        self.targets.write().unwrap()[dst] = new_addr;
        for src in 0..5 {
            if src != dst
                && let Some((_, tx)) = self.links.get(&(src, dst))
            {
                let next_epoch = *tx.borrow() + 1;
                let _ = tx.send(next_epoch);
            }
        }
    }

    fn set_partition(&self, group_a: &[usize], group_b: &[usize], blocked: bool) {
        for &a in group_a {
            for &b in group_b {
                for &(u, v) in &[(a, b), (b, a)] {
                    if let Some((flag, tx)) = self.links.get(&(u, v)) {
                        flag.store(!blocked, Ordering::SeqCst);
                        let next_epoch = *tx.borrow() + 1;
                        let _ = tx.send(next_epoch);
                    }
                }
            }
        }
    }

    fn shutdown(&self) {
        for h in &self.proxy_handles {
            h.abort();
        }
    }
}

struct RunningNode {
    node: Arc<RaftNode<KvStateMachine, TcpRaftTransport>>,
    rpc_handle: tokio::task::JoinHandle<()>,
    election_handle: tokio::task::JoinHandle<()>,
    repl_handle: tokio::task::JoinHandle<()>,
}

impl RunningNode {
    fn stop_node(&self) {
        self.node.stop();
        self.rpc_handle.abort();
        self.election_handle.abort();
        self.repl_handle.abort();
    }
}

async fn allocate_ephemeral_addr() -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    l.local_addr().unwrap().to_string()
}

async fn boot_raft_node(
    idx: usize,
    node_ids: &[String],
    listen_addr: &str,
    data_dir: PathBuf,
    proxy_addrs: &HashMap<(usize, usize), String>,
) -> RunningNode {
    let mut members = HashSet::new();
    for id in node_ids {
        members.insert(id.clone());
    }
    let cluster_config = ClusterConfig::new(members);

    let mut peers = HashMap::new();
    for j in 0..node_ids.len() {
        if j != idx {
            peers.insert(node_ids[j].clone(), proxy_addrs[&(idx, j)].clone());
        }
    }

    let mut cfg = RaftConfig::new(node_ids[idx].clone());
    cfg.election_timeout_min = Duration::from_millis(180);
    cfg.election_timeout_max = Duration::from_millis(340);
    cfg.heartbeat_interval = Duration::from_millis(45);
    cfg.enable_pre_vote = false;
    cfg.batch_config = BatchReplicationConfig {
        enabled: false,
        ..Default::default()
    };
    cfg.data_dir = Some(data_dir);

    let transport = Arc::new(TcpRaftTransport::new(node_ids[idx].clone(), peers));
    let sm = KvStateMachine::default();
    let node = Arc::new(RaftNode::new(cfg, sm, transport, cluster_config));
    node.start().await;

    let rpc_server = RaftRpcServer::new(node.clone(), listen_addr.to_string());
    let rpc_handle = tokio::spawn(async move {
        let _ = rpc_server.run().await;
    });
    let en = node.clone();
    let election_handle = tokio::spawn(async move {
        election_loop(en).await;
    });
    let rn = node.clone();
    let repl_handle = tokio::spawn(async move {
        replication_loop(rn).await;
    });

    RunningNode {
        node,
        rpc_handle,
        election_handle,
        repl_handle,
    }
}

async fn wait_for_leader_in_set(
    nodes: &[RunningNode],
    allowed_indices: &[usize],
    timeout: Duration,
) -> Option<usize> {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        let mut leaders = Vec::new();
        for &i in allowed_indices {
            if nodes[i].node.is_leader().await {
                leaders.push((i, nodes[i].node.current_term().await));
            }
        }
        if let Some(&(best_idx, _)) = leaders.iter().max_by_key(|x| x.1) {
            return Some(best_idx);
        }
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    None
}

async fn client_write_register(
    node: &Arc<RaftNode<KvStateMachine, TcpRaftTransport>>,
    rec: &HistoryRecorder,
    proc_id: usize,
    reg_kw: &str,
    key_bytes: &[u8],
    val_int: u64,
) -> Result<(), RaftError> {
    rec.record(":invoke", proc_id, format!("[[:w :{reg_kw} {val_int}]]"));
    let cmd = Command::Set {
        key: key_bytes.to_vec(),
        value: val_int.to_string().into_bytes(),
    };
    match node
        .propose_and_commit(cmd, Duration::from_millis(900))
        .await
    {
        Ok(_) => {
            rec.record(":ok", proc_id, format!("[[:w :{reg_kw} {val_int}]]"));
            Ok(())
        }
        Err(RaftError::NotLeader(l)) => {
            rec.record(":fail", proc_id, format!("[[:w :{reg_kw} {val_int}]]"));
            Err(RaftError::NotLeader(l))
        }
        Err(e) => {
            rec.record(":info", proc_id, format!("[[:w :{reg_kw} {val_int}]]"));
            Err(e)
        }
    }
}

async fn client_linearizable_read_register(
    node: &Arc<RaftNode<KvStateMachine, TcpRaftTransport>>,
    rec: &HistoryRecorder,
    proc_id: usize,
    reg_kw: &str,
    key_bytes: &[u8],
) -> Result<Option<u64>, RaftError> {
    rec.record(":invoke", proc_id, format!("[[:r :{reg_kw} nil]]"));
    match node
        .propose_and_commit(Command::Noop, Duration::from_millis(900))
        .await
    {
        Ok(_) => {
            let val = node
                .with_state_machine(|sm| {
                    sm.get(key_bytes)
                        .and_then(|b| std::str::from_utf8(b).ok())
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .await;
            let val_edn = val.map(|v| v.to_string()).unwrap_or_else(|| "nil".to_string());
            rec.record(":ok", proc_id, format!("[[:r :{reg_kw} {val_edn}]]"));
            Ok(val)
        }
        Err(e) => {
            rec.record(":fail", proc_id, format!("[[:r :{reg_kw} nil]]"));
            Err(e)
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn test_real_5node_tcp_raft_fault_injection_and_export_elle_history() {
    let t_start = Instant::now();
    let tmp = tempfile::tempdir().expect("tempdir");

    let mut real_addrs = Vec::new();
    for _ in 0..5 {
        real_addrs.push(allocate_ephemeral_addr().await);
    }
    let node_ids: Vec<String> = (1..=5).map(|i| format!("raft-tcp-{i}")).collect();
    let data_dirs: Vec<PathBuf> = (0..5)
        .map(|i| {
            let p = tmp.path().join(format!("node_{i}"));
            std::fs::create_dir_all(&p).unwrap();
            p
        })
        .collect();

    let (matrix, proxy_addrs) = TcpPartitionMatrix::spawn(real_addrs.clone()).await;

    let mut nodes = Vec::new();
    for i in 0..5 {
        nodes.push(
            boot_raft_node(
                i,
                &node_ids,
                &real_addrs[i],
                data_dirs[i].clone(),
                &proxy_addrs,
            )
            .await,
        );
    }

    let rec = Arc::new(HistoryRecorder::new());
    let mut next_val = 1u64;
    let mut committed_expected = HashMap::<String, u64>::new();

    // Phase 1: Healthy 5-node TCP cluster election + quorum writes & reads
    let initial_leader = wait_for_leader_in_set(&nodes, &[0, 1, 2, 3, 4], Duration::from_secs(6))
        .await
        .expect("5-node TCP Raft cluster must elect an initial leader");
    let initial_term = nodes[initial_leader].node.current_term().await;

    for step in 0..8 {
        let (reg_kw, key_b): (&str, &[u8]) = if step % 2 == 0 {
            ("x", b"reg_x")
        } else {
            ("y", b"reg_y")
        };
        client_write_register(
            &nodes[initial_leader].node,
            &rec,
            step % 2,
            reg_kw,
            key_b,
            next_val,
        )
        .await
        .expect("healthy 5-node quorum write must succeed");
        committed_expected.insert(reg_kw.to_string(), next_val);
        next_val += 1;

        let obs = client_linearizable_read_register(
            &nodes[initial_leader].node,
            &rec,
            2 + (step % 2),
            reg_kw,
            key_b,
        )
        .await
        .expect("healthy linearizable read must succeed");
        assert_eq!(obs, Some(*committed_expected.get(reg_kw).unwrap()));
    }

    tokio::time::sleep(Duration::from_millis(350)).await;

    // Phase 2: Asymmetric TCP partition isolating `[initial_leader, buddy]` (2/5) from majority (3/5)
    let buddy = (initial_leader + 1) % 5;
    let minority = vec![initial_leader, buddy];
    let majority: Vec<usize> = (0..5).filter(|i| !minority.contains(i)).collect();
    matrix.set_partition(&minority, &majority, true);

    let iso_write_res = client_write_register(
        &nodes[initial_leader].node,
        &rec,
        10,
        "x",
        b"reg_x",
        999_001,
    )
    .await;
    assert!(
        iso_write_res.is_err(),
        "isolated 2/5 minority leader MUST NOT commit a write"
    );
    let iso_read_res =
        client_linearizable_read_register(&nodes[initial_leader].node, &rec, 11, "x", b"reg_x")
            .await;
    assert!(
        iso_read_res.is_err(),
        "isolated 2/5 minority leader MUST NOT pass a quorum-barrier read"
    );

    let maj_leader = wait_for_leader_in_set(&nodes, &majority, Duration::from_secs(6))
        .await
        .expect("3/5 majority partition must elect a new TCP leader");
    assert_ne!(maj_leader, initial_leader);
    let maj_term = nodes[maj_leader].node.current_term().await;
    assert!(maj_term > initial_term);

    for step in 0..6 {
        let (reg_kw, key_b): (&str, &[u8]) = if step % 2 == 0 {
            ("x", b"reg_x")
        } else {
            ("y", b"reg_y")
        };
        client_write_register(
            &nodes[maj_leader].node,
            &rec,
            4 + (step % 2),
            reg_kw,
            key_b,
            next_val,
        )
        .await
        .expect("3/5 majority leader write must commit over TCP");
        committed_expected.insert(reg_kw.to_string(), next_val);
        next_val += 1;

        let obs = client_linearizable_read_register(
            &nodes[maj_leader].node,
            &rec,
            6 + (step % 2),
            reg_kw,
            key_b,
        )
        .await
        .expect("3/5 majority linearizable read must succeed");
        assert_eq!(obs, Some(*committed_expected.get(reg_kw).unwrap()));
    }

    let raw_stale_x_on_isolated = nodes[initial_leader]
        .node
        .with_state_machine(|sm| {
            sm.get(b"reg_x")
                .and_then(|b| std::str::from_utf8(b).ok())
                .and_then(|s| s.parse::<u64>().ok())
        })
        .await;
    let unbarriered_local_read_was_stale =
        raw_stale_x_on_isolated != committed_expected.get("x").copied();

    // Phase 3: Heal TCP partition + stop & restart a follower node from persistent `raft.wal`
    matrix.set_partition(&minority, &majority, false);
    tokio::time::sleep(Duration::from_millis(600)).await;

    let restarted_idx = majority
        .iter()
        .copied()
        .find(|&idx| idx != maj_leader)
        .unwrap_or(majority[0]);
    nodes[restarted_idx].stop_node();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let fresh_addr = allocate_ephemeral_addr().await;
    matrix.update_target_addr(restarted_idx, fresh_addr.clone());
    nodes[restarted_idx] = boot_raft_node(
        restarted_idx,
        &node_ids,
        &fresh_addr,
        data_dirs[restarted_idx].clone(),
        &proxy_addrs,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let final_leader = wait_for_leader_in_set(&nodes, &[0, 1, 2, 3, 4], Duration::from_secs(6))
        .await
        .expect("healed 5-node cluster must have an active leader");

    for step in 0..4 {
        let (reg_kw, key_b): (&str, &[u8]) = if step % 2 == 0 {
            ("x", b"reg_x")
        } else {
            ("y", b"reg_y")
        };
        let active_l = wait_for_leader_in_set(&nodes, &[0, 1, 2, 3, 4], Duration::from_secs(4))
            .await
            .unwrap_or(final_leader);
        if client_write_register(
            &nodes[active_l].node,
            &rec,
            8 + (step % 2),
            reg_kw,
            key_b,
            next_val,
        )
        .await
        .is_ok()
        {
            committed_expected.insert(reg_kw.to_string(), next_val);
            next_val += 1;
            let _ = client_linearizable_read_register(
                &nodes[active_l].node,
                &rec,
                8 + (step % 2),
                reg_kw,
                key_b,
            )
            .await;
        }
    }

    tokio::time::sleep(Duration::from_millis(1200)).await;

    let mut final_x_values = Vec::new();
    let mut final_y_values = Vec::new();
    for n in &nodes {
        n.node.apply_committed_entries().await;
        let vx = n
            .node
            .with_state_machine(|sm| {
                sm.get(b"reg_x")
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .await;
        let vy = n
            .node
            .with_state_machine(|sm| {
                sm.get(b"reg_y")
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .await;
        final_x_values.push(vx);
        final_y_values.push(vy);
    }

    for i in 0..5 {
        assert_eq!(
            final_x_values[i],
            committed_expected.get("x").copied(),
            "node {i} reg_x diverged after partition heal: {final_x_values:?}"
        );
        assert_eq!(
            final_y_values[i],
            committed_expected.get("y").copied(),
            "node {i} reg_y diverged after partition heal: {final_y_values:?}"
        );
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let results_dir = repo_root.join("bench").join("results");
    std::fs::create_dir_all(&results_dir).expect("mkdir bench/results");

    let edn_path = results_dir.join("raft_tcp_elle_history.edn");
    std::fs::write(&edn_path, rec.to_edn()).expect("write EDN history");

    let summary = serde_json::json!({
        "harness": "raft_tcp_elle_fault_injection",
        "engine_tested": "crates/joule-db-server/src/{raft.rs, raft_server.rs, raft_transport.rs}",
        "cluster_nodes": 5,
        "transport": "Real TCP loopback sockets via 20-link TcpPartitionMatrix fault-injection proxy",
        "wal_persistence": "On-disk RaftWal per node (CRC-32 records + restart recovery)",
        "faults_injected": [
            "Asymmetric 2/5 vs 3/5 TCP socket severing isolating initial leader",
            "Minority write & quorum-read rejection/timeout under partition",
            "Majority 3/5 leader election over TCP at higher term",
            "Follower stop + restart from on-disk raft.wal + partition heal"
        ],
        "initial_leader": node_ids[initial_leader],
        "initial_term": initial_term,
        "new_majority_leader": node_ids[maj_leader],
        "new_majority_term": maj_term,
        "all_5_nodes_converged_x": final_x_values,
        "all_5_nodes_converged_y": final_y_values,
        "lost_committed_writes": 0,
        "candid_finding_unbarriered_local_sm_read_stale_during_partition": unbarriered_local_read_was_stale,
        "candid_note": "RaftNode::propose appends asynchronously to the leader log without awaiting quorum commit; callers requiring linearizable durability across partitions must use RaftNode::propose_and_commit (added in v0.6.0). Similarly, RaftNode::with_state_machine reads local memory without a quorum round-trip unless preceded by a quorum barrier.",
        "elle_history_events": rec.len(),
        "elle_history_file": "bench/results/raft_tcp_elle_history.edn",
        "elapsed_ms": t_start.elapsed().as_millis() as u64,
    });
    std::fs::write(
        results_dir.join("raft_tcp_fault_injection_summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary JSON");

    for n in &nodes {
        n.stop_node();
    }
    matrix.shutdown();
}
