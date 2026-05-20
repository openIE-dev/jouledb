//! Network Replication for JouleDB
//!
//! Implements leader-follower replication using WAL streaming:
//!
//! - **Leader**: Accepts connections from followers, streams WAL entries
//! - **Follower**: Connects to leader, receives and applies WAL entries
//!
//! ## Protocol
//!
//! Uses the binary protocol (from joule-db-core/src/persistence/network.rs) with
//! additional opcodes for replication:
//!
//! - `REPLICATE` (0x20): Request replication stream from LSN
//! - `WAL_ENTRY` (0x21): WAL entry from leader to follower
//! - `ACK` (0x22): Follower acknowledgment with applied LSN
//! - `HEARTBEAT` (0x23): Keep-alive between leader and follower

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;

use joule_db_core::persistence::network::{HEADER_SIZE, Message, OpCode, PROTOCOL_MAGIC};

/// Replication opcodes (extension to base protocol)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationOpCode {
    /// Request replication stream starting from LSN
    Replicate = 0x20,
    /// WAL entry from leader
    WalEntry = 0x21,
    /// Acknowledgment from follower
    Ack = 0x22,
    /// Heartbeat/keepalive
    Heartbeat = 0x23,
    /// Full sync request (for new followers)
    FullSync = 0x24,
    /// Snapshot chunk (for full sync)
    SnapshotChunk = 0x25,
    /// Promote follower to leader
    Promote = 0x26,
}

/// Replication configuration
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Local node ID
    pub node_id: String,
    /// Leader address (if follower)
    pub leader_addr: Option<String>,
    /// Listen address for replication (if leader)
    pub listen_addr: String,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Max batch size for WAL entries
    pub max_batch_size: usize,
    /// Sync mode (sync = wait for ack, async = don't wait)
    pub sync_mode: SyncMode,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            node_id: uuid_v4(),
            leader_addr: None,
            listen_addr: "0.0.0.0:6381".to_string(),
            heartbeat_interval: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(10),
            max_batch_size: 1000,
            sync_mode: SyncMode::SemiSync,
        }
    }
}

/// Synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Wait for all followers to ack
    Sync,
    /// Wait for majority to ack
    SemiSync,
    /// Don't wait for acks
    Async,
}

/// Replication error
#[derive(Debug, Clone)]
pub enum ReplicationError {
    /// Connection failed
    ConnectionFailed(String),
    /// Protocol error
    ProtocolError(String),
    /// Timeout
    Timeout,
    /// Not leader
    NotLeader,
    /// Leader not found
    LeaderNotFound,
    /// IO error
    IoError(String),
    /// No replicas available
    NoReplicasAvailable,
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(s) => write!(f, "Connection failed: {}", s),
            Self::ProtocolError(s) => write!(f, "Protocol error: {}", s),
            Self::Timeout => write!(f, "Timeout"),
            Self::NotLeader => write!(f, "Not leader"),
            Self::LeaderNotFound => write!(f, "Leader not found"),
            Self::IoError(s) => write!(f, "IO error: {}", s),
            Self::NoReplicasAvailable => write!(f, "No replicas available"),
        }
    }
}

impl std::error::Error for ReplicationError {}

/// Follower state tracked by leader
#[derive(Debug, Clone)]
pub struct FollowerState {
    /// Follower node ID
    pub node_id: String,
    /// Follower address
    pub addr: SocketAddr,
    /// Last acknowledged LSN
    pub acked_lsn: u64,
    /// Last heartbeat time
    pub last_heartbeat: Instant,
    /// Connection state
    pub connected: bool,
    /// Replication lag (entries behind)
    pub lag: u64,
}

/// WAL entry for replication
#[derive(Debug, Clone)]
pub struct ReplicationWalEntry {
    /// Log sequence number
    pub lsn: u64,
    /// Operation type
    pub op_type: ReplicationOp,
    /// Key
    pub key: Vec<u8>,
    /// Value (for Put)
    pub value: Option<Vec<u8>>,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
}

/// Replication operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationOp {
    Put,
    Delete,
    Checkpoint,
}

impl ReplicationWalEntry {
    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // LSN (8 bytes)
        buf.extend_from_slice(&self.lsn.to_le_bytes());

        // Op type (1 byte)
        buf.push(match self.op_type {
            ReplicationOp::Put => 1,
            ReplicationOp::Delete => 2,
            ReplicationOp::Checkpoint => 3,
        });

        // Timestamp (8 bytes)
        buf.extend_from_slice(&self.timestamp.to_le_bytes());

        // Key length + key
        buf.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.key);

        // Value length + value
        if let Some(ref value) = self.value {
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value);
        } else {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }

        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ReplicationError> {
        if data.len() < 21 {
            return Err(ReplicationError::ProtocolError("Entry too short".into()));
        }

        let lsn = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);

        let op_type = match data[8] {
            1 => ReplicationOp::Put,
            2 => ReplicationOp::Delete,
            3 => ReplicationOp::Checkpoint,
            _ => return Err(ReplicationError::ProtocolError("Invalid op type".into())),
        };

        let timestamp = u64::from_le_bytes([
            data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16],
        ]);

        let key_len = u32::from_le_bytes([data[17], data[18], data[19], data[20]]) as usize;
        let key = data[21..21 + key_len].to_vec();

        let value_offset = 21 + key_len;
        let value_len = u32::from_le_bytes([
            data[value_offset],
            data[value_offset + 1],
            data[value_offset + 2],
            data[value_offset + 3],
        ]) as usize;

        let value = if value_len > 0 {
            Some(data[value_offset + 4..value_offset + 4 + value_len].to_vec())
        } else {
            None
        };

        Ok(Self {
            lsn,
            op_type,
            timestamp,
            key,
            value,
        })
    }
}

// ============================================================================
// Leader Replication Server
// ============================================================================

/// Leader's replication server
///
/// Accepts connections from followers and streams WAL entries to them.
pub struct ReplicationServer {
    config: ReplicationConfig,
    /// Connected followers
    followers: Arc<RwLock<HashMap<String, FollowerState>>>,
    /// Current LSN
    current_lsn: Arc<AtomicU64>,
    /// Broadcast channel for new WAL entries
    wal_broadcast: broadcast::Sender<ReplicationWalEntry>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<ReplicationServerStats>,
}

/// Replication server statistics
#[derive(Debug, Default)]
pub struct ReplicationServerStats {
    pub entries_sent: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub acks_received: AtomicU64,
    pub followers_connected: AtomicU64,
    pub followers_disconnected: AtomicU64,
}

impl ReplicationServer {
    /// Create a new replication server
    pub fn new(config: ReplicationConfig) -> Self {
        let (wal_broadcast, _) = broadcast::channel(10000);

        Self {
            config,
            followers: Arc::new(RwLock::new(HashMap::new())),
            current_lsn: Arc::new(AtomicU64::new(0)),
            wal_broadcast,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(ReplicationServerStats::default()),
        }
    }

    /// Start the replication server
    pub async fn start(&self) -> Result<(), ReplicationError> {
        let listener = TcpListener::bind(&self.config.listen_addr)
            .await
            .map_err(|e| ReplicationError::IoError(e.to_string()))?;

        tracing::info!(
            "Replication server listening on {}",
            self.config.listen_addr
        );
        self.running.store(true, Ordering::SeqCst);

        while self.running.load(Ordering::SeqCst) {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            let followers = self.followers.clone();
                            let current_lsn = self.current_lsn.clone();
                            let wal_rx = self.wal_broadcast.subscribe();
                            let stats = self.stats.clone();
                            let config = self.config.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_follower(
                                    stream, addr, followers, current_lsn, wal_rx, stats, config
                                ).await {
                                    tracing::error!("Follower connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Accept error: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Stop the replication server
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Broadcast a WAL entry to all followers
    pub fn broadcast_entry(&self, entry: ReplicationWalEntry) -> Result<(), ReplicationError> {
        self.current_lsn.store(entry.lsn, Ordering::SeqCst);
        self.wal_broadcast
            .send(entry)
            .map_err(|_| ReplicationError::ProtocolError("No followers connected".into()))?;
        Ok(())
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Allocate the next LSN (atomically increments and returns the new value)
    pub fn next_lsn(&self) -> u64 {
        self.current_lsn.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Subscribe to the WAL broadcast channel (returns a receiver).
    /// Useful for tests and internal consumers.
    pub fn subscribe(&self) -> broadcast::Receiver<ReplicationWalEntry> {
        self.wal_broadcast.subscribe()
    }

    /// Get list of followers
    pub async fn followers(&self) -> Vec<FollowerState> {
        self.followers.read().await.values().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> ReplicationServerStatsSnapshot {
        ReplicationServerStatsSnapshot {
            entries_sent: self.stats.entries_sent.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(Ordering::Relaxed),
            acks_received: self.stats.acks_received.load(Ordering::Relaxed),
            followers_connected: self.stats.followers_connected.load(Ordering::Relaxed),
            followers_disconnected: self.stats.followers_disconnected.load(Ordering::Relaxed),
        }
    }

    /// Handle a follower connection
    async fn handle_follower(
        mut stream: TcpStream,
        addr: SocketAddr,
        followers: Arc<RwLock<HashMap<String, FollowerState>>>,
        current_lsn: Arc<AtomicU64>,
        mut wal_rx: broadcast::Receiver<ReplicationWalEntry>,
        stats: Arc<ReplicationServerStats>,
        config: ReplicationConfig,
    ) -> Result<(), ReplicationError> {
        tracing::info!("New follower connection from {}", addr);
        stats.followers_connected.fetch_add(1, Ordering::Relaxed);

        // Read initial handshake (REPLICATE request with start LSN)
        let mut header = [0u8; HEADER_SIZE];
        stream
            .read_exact(&mut header)
            .await
            .map_err(|e| ReplicationError::IoError(e.to_string()))?;

        // Verify protocol magic
        if &header[0..2] != &PROTOCOL_MAGIC {
            return Err(ReplicationError::ProtocolError(
                "Invalid protocol magic".into(),
            ));
        }

        let request_id = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let payload_len =
            u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;

        // Read payload (node_id + start_lsn)
        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            stream
                .read_exact(&mut payload)
                .await
                .map_err(|e| ReplicationError::IoError(e.to_string()))?;
        }

        // Parse follower node ID and start LSN
        let node_id_len =
            u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let node_id = String::from_utf8_lossy(&payload[4..4 + node_id_len]).to_string();
        let start_lsn = u64::from_le_bytes([
            payload[4 + node_id_len],
            payload[5 + node_id_len],
            payload[6 + node_id_len],
            payload[7 + node_id_len],
            payload[8 + node_id_len],
            payload[9 + node_id_len],
            payload[10 + node_id_len],
            payload[11 + node_id_len],
        ]);

        tracing::info!(
            "Follower {} starting replication from LSN {}",
            node_id,
            start_lsn
        );

        // Register follower
        {
            let mut followers_guard = followers.write().await;
            followers_guard.insert(
                node_id.clone(),
                FollowerState {
                    node_id: node_id.clone(),
                    addr,
                    acked_lsn: start_lsn,
                    last_heartbeat: Instant::now(),
                    connected: true,
                    lag: current_lsn.load(Ordering::SeqCst).saturating_sub(start_lsn),
                },
            );
        }

        // Send response
        let response = Message::response(
            request_id,
            OpCode::Ping,
            current_lsn.load(Ordering::SeqCst).to_le_bytes().to_vec(),
        );
        stream
            .write_all(&response.encode())
            .await
            .map_err(|e| ReplicationError::IoError(e.to_string()))?;

        // Stream WAL entries
        let mut heartbeat_interval = interval(config.heartbeat_interval);

        loop {
            tokio::select! {
                entry = wal_rx.recv() => {
                    match entry {
                        Ok(wal_entry) => {
                            if wal_entry.lsn > start_lsn {
                                let encoded = wal_entry.encode();
                                let msg = create_wal_entry_message(wal_entry.lsn as u32, &encoded);

                                if stream.write_all(&msg).await.is_err() {
                                    break;
                                }

                                stats.entries_sent.fetch_add(1, Ordering::Relaxed);
                                stats.bytes_sent.fetch_add(msg.len() as u64, Ordering::Relaxed);
                            }
                        }
                        Err(_) => break, // Channel closed
                    }
                }

                _ = heartbeat_interval.tick() => {
                    // Send heartbeat
                    let msg = create_heartbeat_message(current_lsn.load(Ordering::SeqCst));
                    if stream.write_all(&msg).await.is_err() {
                        break;
                    }
                }

                // Read incoming data (ACKs, heartbeat responses)
                result = async {
                    let mut header = [0u8; HEADER_SIZE];
                    stream.read_exact(&mut header).await
                } => {
                    match result {
                        Ok(_bytes_read) => {
                            // Parse header
                            if header[0..2] != PROTOCOL_MAGIC {
                                tracing::warn!("Invalid magic from follower {}", node_id);
                                continue;
                            }

                            let opcode = header[2];
                            let payload_len = u32::from_le_bytes([
                                header[8], header[9], header[10], header[11]
                            ]) as usize;

                            // Read payload if present
                            let mut payload = vec![0u8; payload_len];
                            if payload_len > 0 {
                                if stream.read_exact(&mut payload).await.is_err() {
                                    break;
                                }
                            }

                            match opcode {
                                x if x == ReplicationOpCode::Ack as u8 => {
                                    // ACK message contains the acked LSN
                                    if payload.len() >= 8 {
                                        let acked_lsn = u64::from_le_bytes([
                                            payload[0], payload[1], payload[2], payload[3],
                                            payload[4], payload[5], payload[6], payload[7],
                                        ]);

                                        // Update follower state
                                        {
                                            let mut followers_guard = followers.write().await;
                                            if let Some(state) = followers_guard.get_mut(&node_id) {
                                                state.acked_lsn = acked_lsn;
                                                state.last_heartbeat = Instant::now();
                                                state.lag = current_lsn.load(Ordering::SeqCst).saturating_sub(acked_lsn);
                                            }
                                        }

                                        stats.acks_received.fetch_add(1, Ordering::Relaxed);
                                        tracing::trace!("ACK from {} at LSN {}", node_id, acked_lsn);
                                    }
                                }
                                x if x == ReplicationOpCode::Heartbeat as u8 => {
                                    // Heartbeat response - update last heartbeat time
                                    {
                                        let mut followers_guard = followers.write().await;
                                        if let Some(state) = followers_guard.get_mut(&node_id) {
                                            state.last_heartbeat = Instant::now();
                                        }
                                    }
                                }
                                _ => {
                                    tracing::warn!("Unknown opcode {} from follower {}", opcode, node_id);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Error reading from follower {}: {}", node_id, e);
                            break;
                        }
                    }
                }
            }
        }

        // Remove follower
        {
            let mut followers_guard = followers.write().await;
            followers_guard.remove(&node_id);
        }
        stats.followers_disconnected.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct ReplicationServerStatsSnapshot {
    pub entries_sent: u64,
    pub bytes_sent: u64,
    pub acks_received: u64,
    pub followers_connected: u64,
    pub followers_disconnected: u64,
}

// ============================================================================
// Follower Replication Client
// ============================================================================

/// Follower's replication client
///
/// Connects to the leader and receives WAL entries.
pub struct ReplicationClient {
    config: ReplicationConfig,
    /// Current applied LSN
    applied_lsn: Arc<AtomicU64>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<ReplicationClientStats>,
}

/// Replication client statistics
#[derive(Debug, Default)]
pub struct ReplicationClientStats {
    pub entries_received: AtomicU64,
    pub bytes_received: AtomicU64,
    pub reconnections: AtomicU64,
}

impl ReplicationClient {
    /// Create a new replication client
    pub fn new(config: ReplicationConfig) -> Self {
        Self {
            config,
            applied_lsn: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(ReplicationClientStats::default()),
        }
    }

    /// Start receiving replication stream
    ///
    /// Returns a receiver for WAL entries that should be applied to the local database.
    pub async fn start(&mut self) -> Result<mpsc::Receiver<ReplicationWalEntry>, ReplicationError> {
        let leader_addr = self
            .config
            .leader_addr
            .as_ref()
            .ok_or(ReplicationError::LeaderNotFound)?;

        self.running.store(true, Ordering::SeqCst);

        let (entry_tx, entry_rx) = mpsc::channel(10000);

        let leader_addr_clone = leader_addr.clone();
        let config = self.config.clone();
        let applied_lsn = self.applied_lsn.clone();
        let running = self.running.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                match Self::connect_and_stream(
                    &leader_addr_clone,
                    &config,
                    applied_lsn.clone(),
                    entry_tx.clone(),
                    stats.clone(),
                )
                .await
                {
                    Ok(()) => break, // Clean shutdown
                    Err(e) => {
                        tracing::error!("Replication error: {}. Reconnecting...", e);
                        stats.reconnections.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        });

        Ok(entry_rx)
    }

    /// Stop the replication client
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Get current applied LSN
    pub fn applied_lsn(&self) -> u64 {
        self.applied_lsn.load(Ordering::SeqCst)
    }

    /// Set applied LSN (after applying entries)
    pub fn set_applied_lsn(&self, lsn: u64) {
        self.applied_lsn.store(lsn, Ordering::SeqCst);
    }

    /// Get statistics
    pub fn stats(&self) -> ReplicationClientStatsSnapshot {
        ReplicationClientStatsSnapshot {
            entries_received: self.stats.entries_received.load(Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(Ordering::Relaxed),
            reconnections: self.stats.reconnections.load(Ordering::Relaxed),
        }
    }

    /// Connect to leader and stream entries
    async fn connect_and_stream(
        leader_addr: &str,
        config: &ReplicationConfig,
        applied_lsn: Arc<AtomicU64>,
        entry_tx: mpsc::Sender<ReplicationWalEntry>,
        stats: Arc<ReplicationClientStats>,
    ) -> Result<(), ReplicationError> {
        let mut stream = TcpStream::connect(leader_addr)
            .await
            .map_err(|e| ReplicationError::ConnectionFailed(e.to_string()))?;

        tracing::info!("Connected to leader at {}", leader_addr);

        // Send REPLICATE handshake
        let start_lsn = applied_lsn.load(Ordering::SeqCst);
        let handshake = create_replicate_request(&config.node_id, start_lsn);
        stream
            .write_all(&handshake)
            .await
            .map_err(|e| ReplicationError::IoError(e.to_string()))?;

        // Read response
        let mut header = [0u8; HEADER_SIZE];
        stream
            .read_exact(&mut header)
            .await
            .map_err(|e| ReplicationError::IoError(e.to_string()))?;

        // Stream entries
        loop {
            let mut header = [0u8; HEADER_SIZE];
            if let Err(e) = stream.read_exact(&mut header).await {
                return Err(ReplicationError::IoError(e.to_string()));
            }

            // Parse header
            if &header[0..2] != &PROTOCOL_MAGIC {
                return Err(ReplicationError::ProtocolError("Invalid magic".into()));
            }

            let opcode = header[2];
            let payload_len =
                u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;

            // Read payload
            let mut payload = vec![0u8; payload_len];
            if payload_len > 0 {
                stream
                    .read_exact(&mut payload)
                    .await
                    .map_err(|e| ReplicationError::IoError(e.to_string()))?;
            }

            stats
                .bytes_received
                .fetch_add((HEADER_SIZE + payload_len) as u64, Ordering::Relaxed);

            match opcode {
                0x21 => {
                    // WAL_ENTRY
                    let entry = ReplicationWalEntry::decode(&payload)?;
                    stats.entries_received.fetch_add(1, Ordering::Relaxed);

                    if entry_tx.send(entry).await.is_err() {
                        return Ok(()); // Receiver dropped
                    }
                }
                0x23 => {
                    // HEARTBEAT
                    // Send ACK with our current applied LSN
                    let current_lsn = applied_lsn.load(Ordering::SeqCst);
                    let ack_msg = create_ack_message(current_lsn);
                    stream
                        .write_all(&ack_msg)
                        .await
                        .map_err(|e| ReplicationError::IoError(e.to_string()))?;
                    tracing::trace!("Sent ACK for LSN {}", current_lsn);
                }
                _ => {
                    tracing::warn!("Unknown opcode: {}", opcode);
                }
            }
        }
    }
}

/// Client statistics snapshot
#[derive(Debug, Clone)]
pub struct ReplicationClientStatsSnapshot {
    pub entries_received: u64,
    pub bytes_received: u64,
    pub reconnections: u64,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create a REPLICATE request message
fn create_replicate_request(node_id: &str, start_lsn: u64) -> Vec<u8> {
    let node_id_bytes = node_id.as_bytes();
    let payload_len = 4 + node_id_bytes.len() + 8;

    let mut msg = Vec::with_capacity(HEADER_SIZE + payload_len);

    // Header
    msg.extend_from_slice(&PROTOCOL_MAGIC);
    msg.push(ReplicationOpCode::Replicate as u8);
    msg.push(0); // Flags
    msg.extend_from_slice(&1u32.to_le_bytes()); // Request ID
    msg.extend_from_slice(&(payload_len as u32).to_le_bytes());
    msg.extend_from_slice(&[0u8; 4]); // Reserved

    // Payload
    msg.extend_from_slice(&(node_id_bytes.len() as u32).to_le_bytes());
    msg.extend_from_slice(node_id_bytes);
    msg.extend_from_slice(&start_lsn.to_le_bytes());

    msg
}

/// Create a WAL_ENTRY message
fn create_wal_entry_message(lsn: u32, entry_data: &[u8]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(HEADER_SIZE + entry_data.len());

    // Header
    msg.extend_from_slice(&PROTOCOL_MAGIC);
    msg.push(ReplicationOpCode::WalEntry as u8);
    msg.push(0); // Flags
    msg.extend_from_slice(&lsn.to_le_bytes()); // Request ID = LSN
    msg.extend_from_slice(&(entry_data.len() as u32).to_le_bytes());
    msg.extend_from_slice(&[0u8; 4]); // Reserved

    // Payload
    msg.extend_from_slice(entry_data);

    msg
}

/// Create a HEARTBEAT message
fn create_heartbeat_message(current_lsn: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(HEADER_SIZE + 8);

    // Header
    msg.extend_from_slice(&PROTOCOL_MAGIC);
    msg.push(ReplicationOpCode::Heartbeat as u8);
    msg.push(0); // Flags
    msg.extend_from_slice(&0u32.to_le_bytes()); // Request ID
    msg.extend_from_slice(&8u32.to_le_bytes()); // Payload len
    msg.extend_from_slice(&[0u8; 4]); // Reserved

    // Payload
    msg.extend_from_slice(&current_lsn.to_le_bytes());

    msg
}

/// Create an ACK message with the applied LSN
fn create_ack_message(applied_lsn: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(HEADER_SIZE + 8);

    // Header
    msg.extend_from_slice(&PROTOCOL_MAGIC);
    msg.push(ReplicationOpCode::Ack as u8);
    msg.push(0); // Flags
    msg.extend_from_slice(&0u32.to_le_bytes()); // Request ID
    msg.extend_from_slice(&8u32.to_le_bytes()); // Payload len (8 bytes for LSN)
    msg.extend_from_slice(&[0u8; 4]); // Reserved

    // Payload: applied LSN
    msg.extend_from_slice(&applied_lsn.to_le_bytes());

    msg
}

/// Generate a simple UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    format!("{:016x}-{:04x}", now, (now >> 64) as u32 & 0xFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replication_wal_entry_encode_decode() {
        let entry = ReplicationWalEntry {
            lsn: 42,
            op_type: ReplicationOp::Put,
            timestamp: 1234567890,
            key: b"test_key".to_vec(),
            value: Some(b"test_value".to_vec()),
        };

        let encoded = entry.encode();
        let decoded = ReplicationWalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 42);
        assert_eq!(decoded.op_type, ReplicationOp::Put);
        assert_eq!(decoded.key, b"test_key");
        assert_eq!(decoded.value, Some(b"test_value".to_vec()));
    }

    #[test]
    fn test_replication_wal_entry_delete() {
        let entry = ReplicationWalEntry {
            lsn: 100,
            op_type: ReplicationOp::Delete,
            timestamp: 999,
            key: b"deleted".to_vec(),
            value: None,
        };

        let encoded = entry.encode();
        let decoded = ReplicationWalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 100);
        assert_eq!(decoded.op_type, ReplicationOp::Delete);
        assert_eq!(decoded.value, None);
    }

    #[test]
    fn test_replication_config_default() {
        let config = ReplicationConfig::default();
        assert!(config.leader_addr.is_none());
        assert_eq!(config.listen_addr, "0.0.0.0:6381");
        assert_eq!(config.sync_mode, SyncMode::SemiSync);
    }

    #[test]
    fn test_create_replicate_request() {
        let msg = create_replicate_request("node1", 100);

        assert_eq!(&msg[0..2], &PROTOCOL_MAGIC);
        assert_eq!(msg[2], ReplicationOpCode::Replicate as u8);
    }

    #[test]
    fn test_create_heartbeat_message() {
        let msg = create_heartbeat_message(12345);

        assert_eq!(&msg[0..2], &PROTOCOL_MAGIC);
        assert_eq!(msg[2], ReplicationOpCode::Heartbeat as u8);
    }
}
