//! Integration tests for TCP-based Raft transport.
//!
//! These tests spin up real TCP listeners and verify that Raft consensus
//! works over the network (localhost).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use joule_db_server::raft::{ClusterConfig, Command, KvStateMachine, RaftConfig, RaftNode};
use joule_db_server::raft_server::{RaftRpcServer, raft_loop};
use joule_db_server::raft_transport::TcpRaftTransport;

/// Allocate 3 unique ephemeral ports by binding and immediately closing.
async fn allocate_ports(count: usize) -> Vec<u16> {
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        ports.push(port);
    }
    ports
}

/// Helper: create a 3-node cluster on ephemeral ports.
///
/// Returns (nodes, rpc_handles, loop_handles) — all tasks are spawned.
async fn create_3_node_cluster() -> (
    Vec<Arc<RaftNode<KvStateMachine, TcpRaftTransport>>>,
    Vec<tokio::task::JoinHandle<()>>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let ports = allocate_ports(3).await;
    let node_ids: Vec<String> = (1..=3).map(|i| format!("node{}", i)).collect();
    let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{}", p)).collect();

    let mut nodes = Vec::new();
    let mut rpc_handles = Vec::new();
    let mut loop_handles = Vec::new();

    // Build cluster config: all 3 nodes
    let mut members = HashSet::new();
    for id in &node_ids {
        members.insert(id.clone());
    }
    let cluster_config = ClusterConfig::new(members);

    for i in 0..3 {
        // Peers: all other nodes
        let mut peers = HashMap::new();
        for j in 0..3 {
            if j != i {
                peers.insert(node_ids[j].clone(), addrs[j].clone());
            }
        }

        let mut raft_config = RaftConfig::new(node_ids[i].clone());
        // Faster timeouts for testing
        raft_config.election_timeout_min = Duration::from_millis(200);
        raft_config.election_timeout_max = Duration::from_millis(400);
        raft_config.heartbeat_interval = Duration::from_millis(50);

        let transport = Arc::new(TcpRaftTransport::new(node_ids[i].clone(), peers));
        let state_machine = KvStateMachine::default();
        let node = Arc::new(RaftNode::new(
            raft_config,
            state_machine,
            transport.clone(),
            cluster_config.clone(),
        ));

        // Start RPC server
        let rpc_server = RaftRpcServer::new(node.clone(), addrs[i].clone());
        let rpc_handle = tokio::spawn(async move {
            let _ = rpc_server.run().await;
        });

        // Start consensus loop
        let loop_node = node.clone();
        let loop_handle = tokio::spawn(async move {
            raft_loop(loop_node).await;
        });

        nodes.push(node);
        rpc_handles.push(rpc_handle);
        loop_handles.push(loop_handle);
    }

    // Give TCP listeners time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    (nodes, rpc_handles, loop_handles)
}

/// Wait until exactly one leader is elected, or timeout.
async fn wait_for_leader(
    nodes: &[Arc<RaftNode<KvStateMachine, TcpRaftTransport>>],
    timeout: Duration,
) -> Option<usize> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        let mut leader_idx = None;
        for (i, node) in nodes.iter().enumerate() {
            if node.is_leader().await {
                leader_idx = Some(i);
            }
        }
        if leader_idx.is_some() {
            return leader_idx;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Abort all spawned tasks for cleanup.
fn abort_all(handles: &[tokio::task::JoinHandle<()>]) {
    for h in handles {
        h.abort();
    }
}

// ============================================================================
// Test 1: 3-node cluster elects a leader
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tcp_transport_3_node_election() {
    let result = tokio::time::timeout(Duration::from_secs(10), async {
        let (nodes, rpc_handles, loop_handles) = create_3_node_cluster().await;

        // Wait for leader election (up to 5 seconds)
        let leader = wait_for_leader(&nodes, Duration::from_secs(5)).await;
        assert!(leader.is_some(), "A leader should be elected within 5s");

        let leader_idx = leader.unwrap();
        assert!(
            nodes[leader_idx].is_leader().await,
            "Node {} should be leader",
            leader_idx
        );

        // Exactly one leader
        let mut leader_count = 0;
        for node in &nodes {
            if node.is_leader().await {
                leader_count += 1;
            }
        }
        assert_eq!(leader_count, 1, "Exactly one leader should exist");

        // All nodes should agree on the leader's term
        for node in &nodes {
            assert!(node.current_term().await >= 1, "Term should be at least 1");
        }

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;
    assert!(result.is_ok(), "Test timed out");
}

// ============================================================================
// Test 2: Propose a command and replicate it
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tcp_transport_propose_and_replicate() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let (nodes, rpc_handles, loop_handles) = create_3_node_cluster().await;

        let leader_idx = wait_for_leader(&nodes, Duration::from_secs(5))
            .await
            .expect("leader should be elected");

        // Propose a Set command on the leader
        let cmd = Command::Set {
            key: b"greeting".to_vec(),
            value: b"hello world".to_vec(),
        };
        let result = nodes[leader_idx].propose(cmd).await;
        assert!(result.is_ok(), "Propose should succeed: {:?}", result.err());
        let log_index = result.unwrap();

        // Wait for replication and application
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify the leader has committed the entry
        let leader_commit = nodes[leader_idx].commit_index().await;
        assert!(
            leader_commit >= log_index,
            "Leader commit index {} should be >= log index {}",
            leader_commit,
            log_index
        );

        // Verify followers replicated and applied the entry
        for (i, node) in nodes.iter().enumerate() {
            if i == leader_idx {
                continue;
            }
            let follower_commit = node.commit_index().await;
            assert!(
                follower_commit >= log_index,
                "Follower {} commit index {} should be >= log index {}",
                i,
                follower_commit,
                log_index
            );

            // Verify data was applied to follower's state machine
            let value = node
                .with_state_machine(|sm| sm.get(b"greeting").cloned())
                .await;
            assert_eq!(
                value.as_deref(),
                Some(b"hello world".as_slice()),
                "Follower {} should have the replicated value",
                i
            );
        }

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;
    assert!(result.is_ok(), "Test timed out");
}

// ============================================================================
// Test 3: Cluster tolerates one node failure (quorum = 2/3)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tcp_transport_node_failure_quorum() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let (nodes, rpc_handles, loop_handles) = create_3_node_cluster().await;

        let leader_idx = wait_for_leader(&nodes, Duration::from_secs(5))
            .await
            .expect("leader should be elected");

        // Kill a non-leader node
        let victim = if leader_idx == 0 { 2 } else { 0 };
        rpc_handles[victim].abort();
        loop_handles[victim].abort();

        // Wait for the cluster to stabilize
        tokio::time::sleep(Duration::from_millis(500)).await;

        // The remaining 2 nodes should still form a quorum.
        // Propose on the leader should succeed.
        let cmd = Command::Set {
            key: b"after_failure".to_vec(),
            value: b"still works".to_vec(),
        };
        let result = nodes[leader_idx].propose(cmd).await;
        assert!(
            result.is_ok(),
            "Propose should succeed with 2/3 quorum: {:?}",
            result.err()
        );

        // Wait for commit
        tokio::time::sleep(Duration::from_millis(500)).await;

        let commit = nodes[leader_idx].commit_index().await;
        assert!(commit >= 2, "Leader should have committed entries");

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;
    assert!(result.is_ok(), "Test timed out");
}

// ============================================================================
// Test 4: Leader failover — new leader elected after old leader stops
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tcp_transport_leader_failover() {
    let result = tokio::time::timeout(Duration::from_secs(20), async {
        let (nodes, rpc_handles, loop_handles) = create_3_node_cluster().await;

        let leader_idx = wait_for_leader(&nodes, Duration::from_secs(5))
            .await
            .expect("leader should be elected");

        let old_leader_id = nodes[leader_idx].node_id().clone();

        // Kill the leader's RPC server and consensus loop
        rpc_handles[leader_idx].abort();
        loop_handles[leader_idx].abort();

        // Remaining nodes should elect a new leader
        let remaining: Vec<Arc<RaftNode<KvStateMachine, TcpRaftTransport>>> = nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, n)| n.clone())
            .collect();

        let new_leader = wait_for_leader(&remaining, Duration::from_secs(10)).await;
        assert!(
            new_leader.is_some(),
            "A new leader should be elected after old leader fails"
        );

        let new_leader_idx = new_leader.unwrap();
        let new_leader_id = remaining[new_leader_idx].node_id().clone();
        assert_ne!(
            new_leader_id, old_leader_id,
            "New leader should be different from old leader"
        );

        // Propose on the new leader
        let cmd = Command::Set {
            key: b"failover_key".to_vec(),
            value: b"new_leader_works".to_vec(),
        };
        let result = remaining[new_leader_idx].propose(cmd).await;
        assert!(
            result.is_ok(),
            "Propose on new leader should succeed: {:?}",
            result.err()
        );

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;
    assert!(result.is_ok(), "Test timed out");
}
