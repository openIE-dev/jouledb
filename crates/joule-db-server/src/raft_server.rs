//! Raft RPC server — accepts incoming TCP connections from Raft peers,
//! dispatches RPCs to the local `RaftNode`, and sends back responses.
//!
//! Also provides `election_loop` + `replication_loop` (HRP Phase 1 split)
//! for event-driven replication, plus the legacy `raft_loop` for backward
//! compatibility. The `start_raft_node` convenience function uses the split loops.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

#[cfg(feature = "tls")]
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::hrp_security::{EpochKeyManager, SequenceTracker};
use crate::raft::{
    ClusterConfig, KvStateMachine, NodeId, RaftConfig, RaftMessage, RaftNode, RaftState,
    RaftTransport, StateMachine,
};
use crate::raft_transport::{
    RaftEnvelope, RaftStream, TcpRaftTransport, read_envelope_v2, write_hrp_envelope,
    write_hrp_v2_envelope,
};

// ============================================================================
// Raft RPC Server
// ============================================================================

/// TCP server that listens for incoming Raft RPCs and dispatches them
/// to the local `RaftNode`.
///
/// When compiled with `--features tls` and configured with a `TlsAcceptor`,
/// incoming connections are automatically upgraded to TLS.
pub struct RaftRpcServer<S: StateMachine + 'static, T: RaftTransport + 'static> {
    node: Arc<RaftNode<S, T>>,
    listen_addr: String,
    #[cfg(feature = "tls")]
    tls_acceptor: Option<TlsAcceptor>,
    /// HRP Phase 3: Optional security manager for write token verification
    security: Option<Arc<EpochKeyManager>>,
    /// HRP Phase 3: Per-peer sequence tracking for replay detection
    sequence_tracker: Arc<SequenceTracker>,
    /// Maximum concurrent RPC connections — prevents file descriptor exhaustion
    conn_limit: Arc<Semaphore>,
}

impl<S: StateMachine + 'static, T: RaftTransport + 'static> RaftRpcServer<S, T> {
    /// Create a new RPC server.
    pub fn new(node: Arc<RaftNode<S, T>>, listen_addr: String) -> Self {
        Self {
            node,
            listen_addr,
            #[cfg(feature = "tls")]
            tls_acceptor: None,
            security: None,
            sequence_tracker: Arc::new(SequenceTracker::new()),
            conn_limit: Arc::new(Semaphore::new(100)),
        }
    }

    /// Enable TLS for incoming connections.
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, acceptor: TlsAcceptor) -> Self {
        self.tls_acceptor = Some(acceptor);
        self
    }

    /// Enable HRP v2 security (write token verification + replay detection).
    pub fn with_security(mut self, key_mgr: Arc<EpochKeyManager>) -> Self {
        self.security = Some(key_mgr);
        self
    }

    /// Run the RPC server — binds and accepts connections forever.
    pub async fn run(&self) -> Result<(), std::io::Error> {
        let listener = TcpListener::bind(&self.listen_addr).await?;
        tracing::info!("Raft RPC listening on {}", self.listen_addr);

        loop {
            match listener.accept().await {
                Ok((tcp_stream, peer_addr)) => {
                    tracing::debug!("Raft RPC connection from {}", peer_addr);
                    let node = self.node.clone();
                    let security = self.security.clone();
                    let seq_tracker = self.sequence_tracker.clone();
                    let conn_limit = self.conn_limit.clone();

                    // Perform TLS handshake if configured
                    #[cfg(feature = "tls")]
                    let acceptor = self.tls_acceptor.clone();

                    tokio::spawn(async move {
                        // Acquire connection permit — drop rejects excess connections
                        let _permit = match conn_limit.try_acquire() {
                            Ok(p) => p,
                            Err(_) => {
                                tracing::warn!(
                                    "Raft RPC connection limit reached, rejecting {}",
                                    peer_addr
                                );
                                return;
                            }
                        };
                        let _ = tcp_stream.set_nodelay(true);

                        #[cfg(feature = "tls")]
                        let stream = match acceptor {
                            Some(acceptor) => match acceptor.accept(tcp_stream).await {
                                Ok(tls_stream) => RaftStream::TlsServer(tls_stream),
                                Err(e) => {
                                    tracing::debug!(
                                        "Raft TLS handshake failed from {}: {}",
                                        peer_addr,
                                        e
                                    );
                                    return;
                                }
                            },
                            None => RaftStream::Plain(tcp_stream),
                        };

                        #[cfg(not(feature = "tls"))]
                        let stream = RaftStream::Plain(tcp_stream);

                        if let Err(e) =
                            Self::handle_connection(node, stream, security, seq_tracker).await
                        {
                            tracing::debug!("Raft RPC connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("Raft RPC accept error: {}", e);
                }
            }
        }
    }

    /// Handle a single connection — reads RPCs in a loop until the
    /// connection closes.
    ///
    /// HRP Phase 3: Uses v2 reader for auto-detection (JSON/v1/v2).
    /// When security is configured, verifies write tokens on incoming
    /// messages and signs outgoing responses.
    async fn handle_connection(
        node: Arc<RaftNode<S, T>>,
        mut stream: RaftStream,
        security: Option<Arc<EpochKeyManager>>,
        seq_tracker: Arc<SequenceTracker>,
    ) -> Result<(), String> {
        loop {
            // HRP Phase 3: Read with v2 auto-detect (handles JSON/v1/v2)
            let result = match read_envelope_v2(&mut stream).await {
                Ok(r) => r,
                Err(_) => return Ok(()), // connection closed or read error
            };

            let envelope = result.envelope;

            // HRP Phase 3: When security is configured, ALL messages MUST carry
            // a valid write token. Reject v1/JSON messages that lack tokens —
            // otherwise an attacker can bypass HMAC by sending legacy-format messages.
            if security.is_some() && result.write_token.is_none() {
                tracing::warn!(
                    "HRP security: REJECTING unauthenticated message from {} (no write token, v1/JSON bypass attempt)",
                    envelope.from
                );
                continue; // Drop message — require v2 with token
            }

            if let (Some(key_mgr), Some(token)) = (&security, &result.write_token) {
                // Use the Raft message's term for verification, NOT the local
                // node's term. The receiver's term may lag behind the sender's
                // (e.g., a follower at term 0 receiving from a leader at term 1).
                // Raft's own protocol handles term validation separately.
                let msg_term = match &envelope.msg {
                    RaftMessage::RequestVote(req) => req.term,
                    RaftMessage::AppendEntries(req) => req.term,
                    RaftMessage::InstallSnapshot(req) => req.term,
                    _ => token.term,
                };

                // Verify HMAC — reject on failure
                let bincode_check = bincode::serde::encode_to_vec(&envelope, bincode::config::standard()).unwrap_or_default();
                if let Err(e) = key_mgr.verify_token(token, msg_term, &bincode_check) {
                    tracing::warn!(
                        "HRP security: REJECTING message from {} — token verification failed: {}",
                        envelope.from,
                        e
                    );
                    continue; // Do NOT process this message
                }

                // Replay detection — reject on failure
                if let Err(e) = seq_tracker.check_and_update(&envelope.from, token.sequence) {
                    tracing::warn!(
                        "HRP security: REJECTING replay from {}: {}",
                        envelope.from,
                        e
                    );
                    continue; // Do NOT process replayed messages
                }
            }

            let response_msg = match envelope.msg {
                RaftMessage::RequestVote(req) => {
                    let resp = node.handle_request_vote(req).await;
                    RaftMessage::RequestVoteResponse(resp)
                }
                RaftMessage::AppendEntries(req) => {
                    let resp = node.handle_append_entries(req).await;
                    RaftMessage::AppendEntriesResponse(resp)
                }
                RaftMessage::InstallSnapshot(req) => {
                    let resp = node.handle_install_snapshot(req).await;
                    RaftMessage::InstallSnapshotResponse(resp)
                }
                // Response messages arriving on the server side are protocol errors
                _ => continue,
            };

            let response_envelope = RaftEnvelope {
                from: node.node_id().clone(),
                msg: response_msg,
            };

            // HRP Phase 3: Reply with v2 format when security is configured
            let write_result = if let Some(ref key_mgr) = security {
                let term = node.current_term().await;
                let bincode_data = bincode::serde::encode_to_vec(&response_envelope, bincode::config::standard()).unwrap_or_default();
                let token = key_mgr.generate_token(term, &bincode_data);
                let hmac_key = key_mgr.current_key();
                write_hrp_v2_envelope(
                    &mut stream,
                    &response_envelope,
                    Some(&token),
                    Some(&hmac_key),
                )
                .await
            } else {
                write_hrp_envelope(&mut stream, &response_envelope).await
            };

            if write_result.is_err() {
                return Ok(()); // connection lost
            }
        }
    }
}

// ============================================================================
// Raft Background Loops (HRP Phase 1: split into election + replication)
// ============================================================================

/// Drive elections, periodic heartbeats, and entry application on a fixed tick.
///
/// This handles leader election timeouts and serves as a fallback heartbeat
/// sender (the replication loop handles immediate replication on propose).
///
/// Runs forever — spawn as a tokio task.
pub async fn election_loop<S: StateMachine + 'static, T: RaftTransport + 'static>(
    node: Arc<RaftNode<S, T>>,
) {
    let tick = Duration::from_millis(50);

    loop {
        tokio::time::sleep(tick).await;

        // Follower/Candidate: check election timeout
        if node.election_timeout_elapsed().await {
            let state = node.state().await;
            if matches!(state, RaftState::Follower | RaftState::Candidate) {
                node.run_election().await;
            }
        }

        // Leader: periodic heartbeats (catches entries missed by replication_loop)
        if node.is_leader().await {
            node.send_heartbeats().await;
        }

        // All nodes: apply committed but not-yet-applied entries
        node.apply_committed_entries().await;
    }
}

/// Event-driven replication loop (HRP Phase 1).
///
/// Instead of waiting for the next 50ms tick, this loop wakes immediately
/// when `RaftNode::propose_internal()` or `propose_batch()` appends new
/// entries and calls `replication_notify.notify_waiters()`.
///
/// This eliminates the 0-50ms latency penalty per propose — the leader
/// replicates to followers within microseconds of the propose.
///
/// Runs forever — spawn as a tokio task.
pub async fn replication_loop<S: StateMachine + 'static, T: RaftTransport + 'static>(
    node: Arc<RaftNode<S, T>>,
) {
    loop {
        // Wait for a propose to wake us
        node.replication_notify().notified().await;

        // Leader: immediately replicate new entries to followers
        if node.is_leader().await {
            node.send_heartbeats().await;
        }

        // All nodes: apply any newly committed entries
        node.apply_committed_entries().await;
    }
}

/// Legacy raft_loop for backward compatibility with tests and non-HRP usage.
///
/// Calls both election + replication logic in a single 50ms tick loop.
pub async fn raft_loop<S: StateMachine + 'static, T: RaftTransport + 'static>(
    node: Arc<RaftNode<S, T>>,
) {
    let tick = Duration::from_millis(50);

    loop {
        tokio::time::sleep(tick).await;

        if node.election_timeout_elapsed().await {
            let state = node.state().await;
            if matches!(state, RaftState::Follower | RaftState::Candidate) {
                node.run_election().await;
            }
        }

        if node.is_leader().await {
            node.send_heartbeats().await;
        }

        node.apply_committed_entries().await;
    }
}

// ============================================================================
// Convenience: Start a Raft Node
// ============================================================================

/// Parse a peer list in the format `["node2=host:port", "node3=host:port"]`
/// into a `HashMap<NodeId, String>`.
pub fn parse_peer_list(peers: &[String]) -> HashMap<NodeId, String> {
    let mut map = HashMap::new();
    for entry in peers {
        if let Some((id, addr)) = entry.split_once('=') {
            map.insert(id.trim().to_string(), addr.trim().to_string());
        }
    }
    map
}

// ============================================================================
// TLS Configuration
// ============================================================================

/// TLS configuration for Raft inter-node transport.
///
/// Holds both the server-side acceptor (for incoming connections) and the
/// client-side connector (for outgoing connections).
#[cfg(feature = "tls")]
pub struct RaftTlsConfig {
    /// TLS acceptor for incoming peer connections.
    pub acceptor: TlsAcceptor,
    /// TLS connector for outgoing peer connections.
    pub connector: TlsConnector,
}

// ============================================================================
// Convenience: Start a Raft Node
// ============================================================================

/// Start a complete Raft node with TCP transport, RPC server, and
/// background consensus loops (HRP Phase 1: split election + replication).
///
/// Returns the `RaftNode` handle (for proposing commands, querying state)
/// and join handles for the background tasks (RPC server, election loop,
/// replication loop).
///
/// If `tls_config` is provided (requires `tls` feature), both incoming and
/// outgoing Raft connections will be encrypted with TLS.
///
/// If `security_key_mgr` is provided (HRP Phase 3), write tokens and HMAC
/// are used for message integrity and replay protection.
pub async fn start_raft_node(
    config: RaftConfig,
    peers: HashMap<NodeId, String>,
    listen_addr: String,
    #[cfg(feature = "tls")] tls_config: Option<RaftTlsConfig>,
    security_key_mgr: Option<Arc<EpochKeyManager>>,
) -> Result<
    (
        Arc<RaftNode<KvStateMachine, TcpRaftTransport>>,
        JoinHandle<()>,
        JoinHandle<()>,
        JoinHandle<()>,
    ),
    std::io::Error,
> {
    let node_id = config.node_id.clone();

    // Build cluster config: self + all peers
    let mut members = HashSet::new();
    members.insert(node_id.clone());
    for peer_id in peers.keys() {
        members.insert(peer_id.clone());
    }
    let cluster_config = ClusterConfig::new(members);

    // Create transport + state machine + node
    let term_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut transport = TcpRaftTransport::new(node_id, peers);
    #[cfg(feature = "tls")]
    if let Some(ref tls) = tls_config {
        transport = transport.with_tls(tls.connector.clone());
    }
    // HRP Phase 3: Wire security into transport
    if let Some(ref key_mgr) = security_key_mgr {
        transport = transport.with_security(key_mgr.clone(), term_counter);
    }

    let transport = Arc::new(transport);
    let state_machine = KvStateMachine::default();
    let node = Arc::new(RaftNode::new(
        config,
        state_machine,
        transport,
        cluster_config,
    ));

    // Start RPC server
    let mut rpc_server = RaftRpcServer::new(node.clone(), listen_addr);
    #[cfg(feature = "tls")]
    if let Some(tls) = tls_config {
        rpc_server = rpc_server.with_tls(tls.acceptor);
    }
    // HRP Phase 3: Wire security into RPC server
    if let Some(key_mgr) = security_key_mgr {
        rpc_server = rpc_server.with_security(key_mgr);
    }
    // Suppress unused variable warning when tls feature is disabled
    let _ = &rpc_server;

    let rpc_handle = tokio::spawn(async move {
        if let Err(e) = rpc_server.run().await {
            tracing::error!("Raft RPC server error: {}", e);
        }
    });

    // HRP Phase 1: Split consensus into election + replication loops
    // Election loop: 50ms tick for election timeouts and periodic heartbeats
    let election_node = node.clone();
    let election_handle = tokio::spawn(async move {
        election_loop(election_node).await;
    });

    // Replication loop: event-driven, wakes immediately on propose
    let replication_node = node.clone();
    let replication_handle = tokio::spawn(async move {
        replication_loop(replication_node).await;
    });

    Ok((node, rpc_handle, election_handle, replication_handle))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_peer_list() {
        let peers = vec![
            "node2=127.0.0.1:17002".to_string(),
            "node3=127.0.0.1:17003".to_string(),
        ];
        let map = parse_peer_list(&peers);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("node2").unwrap(), "127.0.0.1:17002");
        assert_eq!(map.get("node3").unwrap(), "127.0.0.1:17003");
    }

    #[test]
    fn test_parse_peer_list_empty() {
        let peers: Vec<String> = vec![];
        let map = parse_peer_list(&peers);
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_peer_list_malformed() {
        let peers = vec![
            "node2=127.0.0.1:17002".to_string(),
            "bad_entry_no_equals".to_string(), // should be skipped
            "node3=127.0.0.1:17003".to_string(),
        ];
        let map = parse_peer_list(&peers);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_parse_peer_list_with_spaces() {
        let peers = vec![" node2 = 127.0.0.1:17002 ".to_string()];
        let map = parse_peer_list(&peers);
        assert_eq!(map.get("node2").unwrap(), "127.0.0.1:17002");
    }
}
