//! # Raft Consensus Protocol Implementation
//!
//! A complete implementation of the Raft consensus algorithm following the
//! original paper by Ongaro & Ousterhout: "In Search of an Understandable
//! Consensus Algorithm"
//!
//! ## Features
//!
//! - **Leader Election**: Randomized timeouts for leader election
//! - **Log Replication**: AppendEntries RPC for log replication
//! - **Safety Guarantees**: Election safety, leader append-only, log matching
//! - **Cluster Membership Changes**: Joint consensus for safe configuration changes
//! - **Log Compaction**: Snapshotting for log compaction
//!
//! ## Architecture
//!
//! The implementation is structured around:
//! - `RaftNode`: The main state machine managing consensus
//! - `RaftState`: The three possible states (Follower, Candidate, Leader)
//! - `LogEntry`: Individual log entries with term and command
//! - RPC messages: RequestVote, AppendEntries, InstallSnapshot

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify, RwLock, broadcast, mpsc};

// ============================================================================
// Constants
// ============================================================================

/// Minimum election timeout in milliseconds
const ELECTION_TIMEOUT_MIN_MS: u64 = 150;

/// Maximum election timeout in milliseconds
const ELECTION_TIMEOUT_MAX_MS: u64 = 300;

/// Heartbeat interval in milliseconds (must be << election timeout)
const HEARTBEAT_INTERVAL_MS: u64 = 50;

/// Maximum entries per AppendEntries RPC
const MAX_ENTRIES_PER_RPC: usize = 100;

/// Maximum snapshot chunk size in bytes
const MAX_SNAPSHOT_CHUNK_SIZE: usize = 1024 * 1024; // 1MB

/// Magic bytes prefixed to erasure-coded snapshot shard data.
/// Used by the receiver to distinguish erasure shards from standard chunks.
const ERASURE_SHARD_MAGIC: &[u8] = b"ECSH"; // "Erasure Coded SHard"

// ============================================================================
// Types and Enums
// ============================================================================

/// Unique identifier for a Raft node
pub type NodeId = String;

/// Log index (1-indexed as per Raft paper)
pub type LogIndex = u64;

/// Term number
pub type Term = u64;

/// Raft node state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftState {
    /// Passive state, responds to RPCs from leaders and candidates
    Follower,
    /// Active state, requesting votes to become leader
    Candidate,
    /// Active state, managing log replication
    Leader,
}

impl std::fmt::Display for RaftState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaftState::Follower => write!(f, "Follower"),
            RaftState::Candidate => write!(f, "Candidate"),
            RaftState::Leader => write!(f, "Leader"),
        }
    }
}

/// Command to be replicated in the state machine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Command {
    /// No-op command (used for leader establishment)
    Noop,
    /// Set a key-value pair
    Set { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key
    Delete { key: Vec<u8> },
    /// Configuration change command
    ConfigChange(ClusterConfig),
    /// HRP Phase 2: Row-level mutation delta (bincode-serialized MutationDelta).
    /// Followers apply the delta directly to storage, bypassing SQL re-execution.
    MutationDelta(Vec<u8>),
}

impl Command {
    /// Encode command to bytes
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Decode command from bytes
    pub fn decode(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }
}

/// A single log entry
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    /// The term when entry was received by leader
    pub term: Term,
    /// The log index (1-indexed)
    pub index: LogIndex,
    /// The command to apply to state machine
    pub command: Command,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(term: Term, index: LogIndex, command: Command) -> Self {
        Self {
            term,
            index,
            command,
        }
    }
}

/// Cluster configuration for membership changes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Current cluster members
    pub members: HashSet<NodeId>,
    /// New members (during joint consensus)
    pub new_members: Option<HashSet<NodeId>>,
    /// Configuration index (for tracking)
    pub config_index: LogIndex,
}

impl ClusterConfig {
    /// Create a new single-node configuration
    pub fn single(node_id: NodeId) -> Self {
        let mut members = HashSet::new();
        members.insert(node_id);
        Self {
            members,
            new_members: None,
            config_index: 0,
        }
    }

    /// Create a configuration with multiple members
    pub fn new(members: HashSet<NodeId>) -> Self {
        Self {
            members,
            new_members: None,
            config_index: 0,
        }
    }

    /// Check if in joint consensus mode
    pub fn is_joint(&self) -> bool {
        self.new_members.is_some()
    }

    /// Get all voting members (both old and new during joint consensus)
    pub fn voting_members(&self) -> HashSet<NodeId> {
        let mut all = self.members.clone();
        if let Some(ref new) = self.new_members {
            all.extend(new.iter().cloned());
        }
        all
    }

    /// Calculate majority for old configuration
    pub fn old_majority(&self) -> usize {
        (self.members.len() / 2) + 1
    }

    /// Calculate majority for new configuration
    pub fn new_majority(&self) -> Option<usize> {
        self.new_members.as_ref().map(|m| (m.len() / 2) + 1)
    }

    /// Check if we have a majority in old config
    pub fn has_old_majority(&self, voters: &HashSet<NodeId>) -> bool {
        let count = voters.iter().filter(|v| self.members.contains(*v)).count();
        count >= self.old_majority()
    }

    /// Check if we have a majority in new config (if in joint consensus)
    pub fn has_new_majority(&self, voters: &HashSet<NodeId>) -> bool {
        match &self.new_members {
            Some(new) => {
                let count = voters.iter().filter(|v| new.contains(*v)).count();
                count >= self.new_majority().unwrap_or(1)
            }
            None => true,
        }
    }

    /// Check if we have majorities in both configs (for joint consensus)
    pub fn has_quorum(&self, voters: &HashSet<NodeId>) -> bool {
        self.has_old_majority(voters) && self.has_new_majority(voters)
    }
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index
    pub last_included_index: LogIndex,
    /// Last included term
    pub last_included_term: Term,
    /// Cluster configuration at snapshot
    pub config: ClusterConfig,
    /// Total snapshot size in bytes
    pub total_size: u64,
}

/// Snapshot data
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Metadata
    pub metadata: SnapshotMetadata,
    /// Snapshot data (state machine state)
    pub data: Vec<u8>,
}

// ============================================================================
// RPC Messages
// ============================================================================

/// RequestVote RPC arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// RequestVote RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term, for candidate to update itself
    pub term: Term,
    /// True means candidate received vote
    pub vote_granted: bool,
}

/// AppendEntries RPC arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader's ID, for followers to redirect clients
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
    /// HRP Phase 4: Leader's energy state piggybacked on heartbeats
    pub energy_state: Option<NodeEnergyState>,
}

/// AppendEntries RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term, for leader to update itself
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// Optimization: hint for next index on conflict
    pub conflict_index: Option<LogIndex>,
    /// Optimization: term of conflicting entry
    pub conflict_term: Option<Term>,
    /// The index of the last entry appended (for leader tracking)
    pub match_index: LogIndex,
    /// HRP Phase 4: Follower's energy state piggybacked on heartbeat response
    pub energy_state: Option<NodeEnergyState>,
}

/// HRP Phase 4: Node energy state for energy-aware query routing.
///
/// Piggybacked on AppendEntries RPCs to avoid extra round-trips.
/// The leader tracks all peers' energy state for routing decisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeEnergyState {
    /// Current power draw in watts
    pub power_watts: f64,
    /// Active compute target ("cpu", "gpu", "npu", "auto")
    pub device_target: String,
    /// Estimated load factor (0.0 = idle, 1.0 = fully loaded)
    pub load_factor: f64,
    /// Available memory in megabytes
    pub available_memory_mb: u64,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
}

/// InstallSnapshot RPC arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotRequest {
    /// Leader's term
    pub term: Term,
    /// Leader's ID
    pub leader_id: NodeId,
    /// Last included index in snapshot
    pub last_included_index: LogIndex,
    /// Last included term in snapshot
    pub last_included_term: Term,
    /// Byte offset where chunk is positioned
    pub offset: u64,
    /// Raw bytes of snapshot chunk
    pub data: Vec<u8>,
    /// True if this is the last chunk
    pub done: bool,
    /// Cluster configuration at snapshot
    pub config: ClusterConfig,
}

/// InstallSnapshot RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotResponse {
    /// Current term, for leader to update itself
    pub term: Term,
    /// True if snapshot was accepted
    pub success: bool,
    /// Next expected offset
    pub next_offset: u64,
}

/// Message types for RPC communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    RequestVote(RequestVoteRequest),
    RequestVoteResponse(RequestVoteResponse),
    AppendEntries(AppendEntriesRequest),
    AppendEntriesResponse(AppendEntriesResponse),
    InstallSnapshot(InstallSnapshotRequest),
    InstallSnapshotResponse(InstallSnapshotResponse),
}

// ============================================================================
// Persistent State
// ============================================================================

/// Persistent state on all servers
/// (Updated on stable storage before responding to RPCs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Latest term server has seen (initialized to 0 on first boot)
    pub current_term: Term,
    /// CandidateId that received vote in current term (or None)
    pub voted_for: Option<NodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
}

impl PersistentState {
    /// Create new persistent state
    pub fn new() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
        }
    }

    /// Get the last log index
    pub fn last_log_index(&self) -> LogIndex {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Get the last log term
    pub fn last_log_term(&self) -> Term {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Get entry at index
    pub fn get_entry(&self, index: LogIndex) -> Option<&LogEntry> {
        if index == 0 || index as usize > self.log.len() {
            None
        } else {
            self.log.get((index - 1) as usize)
        }
    }

    /// Get term at index
    pub fn get_term(&self, index: LogIndex) -> Option<Term> {
        self.get_entry(index).map(|e| e.term)
    }

    /// Append entries to log
    pub fn append_entries(&mut self, entries: Vec<LogEntry>) {
        self.log.extend(entries);
    }

    /// Truncate log from index onwards
    pub fn truncate_from(&mut self, index: LogIndex) {
        if index > 0 && (index as usize) <= self.log.len() {
            self.log.truncate((index - 1) as usize);
        }
    }

    /// Get entries from start_index to end_index (inclusive)
    pub fn get_entries(&self, start_index: LogIndex, end_index: LogIndex) -> Vec<LogEntry> {
        if start_index == 0 || start_index > end_index {
            return Vec::new();
        }
        let start = (start_index - 1) as usize;
        let end = std::cmp::min(end_index as usize, self.log.len());
        self.log[start..end].to_vec()
    }

    /// Compact log up to (and including) index
    pub fn compact_until(&mut self, index: LogIndex) {
        if index > 0 && (index as usize) <= self.log.len() {
            self.log.drain(0..(index as usize));
            // Reindex remaining entries
            for (i, entry) in self.log.iter_mut().enumerate() {
                entry.index = (i + 1) as LogIndex + index;
            }
        }
    }

    /// Save persistent state to a directory (atomic write via temp + rename).
    ///
    /// Writes `raft_state.json` containing current_term, voted_for, and all log entries.
    /// This is called after every state mutation (term change, vote, log append) per the
    /// Raft specification: "Updated on stable storage before responding to RPCs."
    pub fn save_to_dir(&self, dir: &std::path::Path) -> Result<(), String> {
        use std::io::Write;
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create Raft state directory: {}", e))?;

        let json = serde_json::to_vec(self)
            .map_err(|e| format!("Failed to serialize Raft state: {}", e))?;

        let target = dir.join("raft_state.json");
        let tmp = dir.join("raft_state.json.tmp");

        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("Failed to create temp file: {}", e))?;
        file.write_all(&json)
            .map_err(|e| format!("Failed to write Raft state: {}", e))?;
        file.sync_all()
            .map_err(|e| format!("Failed to fsync Raft state: {}", e))?;

        std::fs::rename(&tmp, &target)
            .map_err(|e| format!("Failed to rename Raft state file: {}", e))?;

        Ok(())
    }

    /// Load persistent state from a directory, or return a fresh state if no file exists.
    pub fn load_from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join("raft_state.json");
        if !path.exists() {
            return Ok(Self::new());
        }

        let data =
            std::fs::read(&path).map_err(|e| format!("Failed to read Raft state file: {}", e))?;

        serde_json::from_slice(&data)
            .map_err(|e| format!("Failed to deserialize Raft state: {}", e))
    }
}

impl Default for PersistentState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Append-Only Raft WAL (HRP Phase 1.2)
// ============================================================================

/// WAL entry types for the append-only Raft log.
const WAL_TYPE_META: u8 = 1; // term + voted_for
const WAL_TYPE_ENTRY: u8 = 2; // log entry (term + index + command)

/// Append-only Write-Ahead Log for Raft state persistence.
///
/// Replaces the full-state JSON serialization (`PersistentState::save_to_dir`)
/// with O(1) appends per log entry. Each record is:
///
/// ```text
/// [1 byte type][payload bytes][4 byte CRC32]
/// ```
///
/// For `WAL_TYPE_META`:
///   `[8 byte term][1 byte has_voted][voted_for_len + voted_for_bytes]`
///
/// For `WAL_TYPE_ENTRY`:
///   `[4 byte bincode_len][bincode_bytes]`
///
/// On recovery, the WAL is replayed to reconstruct `PersistentState`.
/// A periodic checkpoint writes a full JSON snapshot and truncates the WAL.
pub struct RaftWal {
    file: std::fs::File,
    path: std::path::PathBuf,
    entries_since_checkpoint: usize,
    checkpoint_threshold: usize,
}

impl RaftWal {
    /// Open or create the WAL file at `dir/raft.wal`.
    pub fn open(dir: &std::path::Path) -> Result<Self, String> {
        use std::fs::OpenOptions;

        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create WAL directory: {}", e))?;

        let path = dir.join("raft.wal");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("Failed to open WAL: {}", e))?;

        Ok(Self {
            file,
            path,
            entries_since_checkpoint: 0,
            checkpoint_threshold: 10000, // checkpoint after 10K entries
        })
    }

    /// Append a log entry to the WAL and fsync.
    pub fn append_entry(&mut self, entry: &LogEntry) -> Result<(), String> {
        use std::io::Write;

        let bincode_data =
            bincode::serde::encode_to_vec(entry, bincode::config::standard())
                .map_err(|e| format!("WAL bincode serialize: {}", e))?;

        let len = bincode_data.len() as u32;
        let mut record = Vec::with_capacity(1 + 4 + bincode_data.len() + 4);

        record.push(WAL_TYPE_ENTRY);
        record.extend_from_slice(&len.to_le_bytes());
        record.extend_from_slice(&bincode_data);

        // CRC32 over type + len + payload
        let crc = crc32fast::hash(&record);
        record.extend_from_slice(&crc.to_le_bytes());

        self.file
            .write_all(&record)
            .map_err(|e| format!("WAL write: {}", e))?;
        self.file
            .sync_data()
            .map_err(|e| format!("WAL sync: {}", e))?;

        self.entries_since_checkpoint += 1;
        Ok(())
    }

    /// Append a metadata record (term + voted_for) to the WAL and fsync.
    pub fn append_meta(&mut self, term: Term, voted_for: &Option<NodeId>) -> Result<(), String> {
        use std::io::Write;

        let mut record = Vec::with_capacity(64);
        record.push(WAL_TYPE_META);
        record.extend_from_slice(&term.to_le_bytes());

        match voted_for {
            Some(id) => {
                record.push(1); // has_voted
                let id_bytes = id.as_bytes();
                let id_len = id_bytes.len() as u32;
                record.extend_from_slice(&id_len.to_le_bytes());
                record.extend_from_slice(id_bytes);
            }
            None => {
                record.push(0); // no vote
            }
        }

        let crc = crc32fast::hash(&record);
        record.extend_from_slice(&crc.to_le_bytes());

        self.file
            .write_all(&record)
            .map_err(|e| format!("WAL meta write: {}", e))?;
        self.file
            .sync_data()
            .map_err(|e| format!("WAL meta sync: {}", e))?;

        Ok(())
    }

    /// Check if a checkpoint is due (too many entries since last checkpoint).
    pub fn needs_checkpoint(&self) -> bool {
        self.entries_since_checkpoint >= self.checkpoint_threshold
    }

    /// Write a full checkpoint (JSON snapshot) and truncate the WAL.
    pub fn checkpoint(&mut self, state: &PersistentState) -> Result<(), String> {
        use std::io::Write;

        let dir = self.path.parent().ok_or("WAL has no parent directory")?;

        // Write JSON snapshot first (atomic via temp + rename)
        state.save_to_dir(dir)?;

        // Truncate the WAL
        drop(std::mem::replace(
            &mut self.file,
            std::fs::File::create(&self.path).map_err(|e| format!("WAL truncate: {}", e))?,
        ));

        // Re-open in append mode
        self.file = std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("WAL reopen: {}", e))?;

        self.file
            .sync_data()
            .map_err(|e| format!("WAL sync after truncate: {}", e))?;

        self.entries_since_checkpoint = 0;

        tracing::debug!("Raft WAL checkpoint complete, WAL truncated");
        Ok(())
    }

    /// Recover `PersistentState` by loading the JSON snapshot and replaying
    /// any WAL entries written after the last checkpoint.
    pub fn recover(dir: &std::path::Path) -> Result<PersistentState, String> {
        // Load base state from JSON snapshot (if exists)
        let mut state = PersistentState::load_from_dir(dir)?;

        let wal_path = dir.join("raft.wal");
        if !wal_path.exists() {
            return Ok(state);
        }

        let data = std::fs::read(&wal_path).map_err(|e| format!("WAL read: {}", e))?;

        if data.is_empty() {
            return Ok(state);
        }

        let mut cursor = 0;
        let mut entries_replayed = 0;
        let mut meta_replayed = 0;

        while cursor < data.len() {
            // Need at least type byte + CRC
            if cursor + 5 > data.len() {
                tracing::warn!(
                    "WAL: truncated record at offset {}, stopping replay",
                    cursor
                );
                break;
            }

            let record_type = data[cursor];
            cursor += 1;

            match record_type {
                WAL_TYPE_ENTRY => {
                    if cursor + 4 > data.len() {
                        tracing::warn!("WAL: truncated entry length at offset {}", cursor - 1);
                        break;
                    }
                    let len =
                        u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
                    cursor += 4;

                    if cursor + len + 4 > data.len() {
                        tracing::warn!("WAL: truncated entry payload at offset {}", cursor);
                        break;
                    }

                    // Verify CRC: covers type(1) + len(4) + payload(len)
                    let record_start = cursor - 5; // back to type byte
                    let crc_offset = cursor + len;
                    let expected_crc =
                        u32::from_le_bytes(data[crc_offset..crc_offset + 4].try_into().unwrap());
                    let actual_crc = crc32fast::hash(&data[record_start..crc_offset]);
                    if expected_crc != actual_crc {
                        tracing::warn!(
                            "WAL: CRC mismatch at offset {} (expected {:08x}, got {:08x}), stopping",
                            record_start,
                            expected_crc,
                            actual_crc
                        );
                        break;
                    }

                    let (entry, _): (LogEntry, _) = bincode::serde::decode_from_slice(&data[cursor..cursor + len], bincode::config::standard())
                        .map_err(|e| format!("WAL entry deserialize: {}", e))?;

                    // Only apply if this entry is beyond what's in the snapshot
                    if entry.index > state.last_log_index() {
                        state.log.push(entry);
                        entries_replayed += 1;
                    }

                    cursor = crc_offset + 4;
                }
                WAL_TYPE_META => {
                    if cursor + 9 > data.len() {
                        tracing::warn!("WAL: truncated meta at offset {}", cursor - 1);
                        break;
                    }

                    let record_start = cursor - 1;
                    let term = u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap());
                    cursor += 8;

                    let has_voted = data[cursor];
                    cursor += 1;

                    let voted_for = if has_voted == 1 {
                        if cursor + 4 > data.len() {
                            tracing::warn!("WAL: truncated voted_for at offset {}", cursor);
                            break;
                        }
                        let id_len =
                            u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap())
                                as usize;
                        cursor += 4;
                        if cursor + id_len > data.len() {
                            tracing::warn!("WAL: truncated voted_for string at offset {}", cursor);
                            break;
                        }
                        let id = String::from_utf8(data[cursor..cursor + id_len].to_vec())
                            .map_err(|e| format!("WAL voted_for UTF-8: {}", e))?;
                        cursor += id_len;
                        Some(id)
                    } else {
                        None
                    };

                    // Verify CRC
                    if cursor + 4 > data.len() {
                        tracing::warn!("WAL: truncated meta CRC at offset {}", cursor);
                        break;
                    }
                    let expected_crc =
                        u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap());
                    let actual_crc = crc32fast::hash(&data[record_start..cursor]);
                    if expected_crc != actual_crc {
                        tracing::warn!("WAL: meta CRC mismatch, stopping");
                        break;
                    }
                    cursor += 4;

                    state.current_term = term;
                    state.voted_for = voted_for;
                    meta_replayed += 1;
                }
                _ => {
                    tracing::warn!(
                        "WAL: unknown record type {} at offset {}, stopping",
                        record_type,
                        cursor - 1
                    );
                    break;
                }
            }
        }

        if entries_replayed > 0 || meta_replayed > 0 {
            tracing::info!(
                "WAL recovery: replayed {} entries, {} meta records (term={}, log_len={})",
                entries_replayed,
                meta_replayed,
                state.current_term,
                state.log.len()
            );
        }

        Ok(state)
    }
}

// ============================================================================
// Volatile State
// ============================================================================

/// Volatile state on all servers
#[derive(Debug, Clone)]
pub struct VolatileState {
    /// Index of highest log entry known to be committed
    pub commit_index: LogIndex,
    /// Index of highest log entry applied to state machine
    pub last_applied: LogIndex,
}

impl VolatileState {
    pub fn new() -> Self {
        Self {
            commit_index: 0,
            last_applied: 0,
        }
    }
}

impl Default for VolatileState {
    fn default() -> Self {
        Self::new()
    }
}

/// Volatile state on leaders (reinitialized after election)
#[derive(Debug, Clone)]
pub struct LeaderState {
    /// For each server, index of the next log entry to send
    pub next_index: HashMap<NodeId, LogIndex>,
    /// For each server, index of highest log entry known to be replicated
    pub match_index: HashMap<NodeId, LogIndex>,
    /// Pending snapshot transfers
    pub snapshot_transfers: HashMap<NodeId, SnapshotTransfer>,
}

impl LeaderState {
    pub fn new(peers: &HashSet<NodeId>, last_log_index: LogIndex) -> Self {
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for peer in peers {
            // Initialize next_index to leader's last log index + 1
            next_index.insert(peer.clone(), last_log_index + 1);
            // Initialize match_index to 0
            match_index.insert(peer.clone(), 0);
        }

        Self {
            next_index,
            match_index,
            snapshot_transfers: HashMap::new(),
        }
    }

    /// Update match_index and potentially advance next_index
    pub fn update_match(&mut self, node_id: &NodeId, match_index: LogIndex) {
        self.match_index.insert(node_id.clone(), match_index);
        self.next_index.insert(node_id.clone(), match_index + 1);
    }

    /// Decrement next_index for a node (on rejection)
    pub fn decrement_next(&mut self, node_id: &NodeId, hint_index: Option<LogIndex>) {
        if let Some(next) = self.next_index.get_mut(node_id) {
            if let Some(hint) = hint_index {
                *next = std::cmp::min(*next, hint);
            } else if *next > 1 {
                *next -= 1;
            }
        }
    }
}

/// State for ongoing snapshot transfer to a follower
#[derive(Debug, Clone)]
pub struct SnapshotTransfer {
    /// Snapshot being transferred
    pub snapshot: Snapshot,
    /// Current offset in snapshot data
    pub offset: u64,
    /// Last send time for timeout tracking
    pub last_sent: Instant,
}

// ============================================================================
// Raft Error Types
// ============================================================================

/// Errors that can occur during Raft operations
#[derive(Debug, Clone)]
pub enum RaftError {
    /// Not the leader, redirect to this node
    NotLeader(Option<NodeId>),
    /// Proposal failed (log not replicated)
    ProposalFailed(String),
    /// Timeout waiting for consensus
    Timeout,
    /// Node not found in cluster
    NodeNotFound(NodeId),
    /// Invalid configuration
    InvalidConfig(String),
    /// Snapshot error
    SnapshotError(String),
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for RaftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaftError::NotLeader(Some(id)) => write!(f, "Not leader, redirect to {}", id),
            RaftError::NotLeader(None) => write!(f, "Not leader, leader unknown"),
            RaftError::ProposalFailed(s) => write!(f, "Proposal failed: {}", s),
            RaftError::Timeout => write!(f, "Operation timed out"),
            RaftError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            RaftError::InvalidConfig(s) => write!(f, "Invalid configuration: {}", s),
            RaftError::SnapshotError(s) => write!(f, "Snapshot error: {}", s),
            RaftError::Internal(s) => write!(f, "Internal error: {}", s),
        }
    }
}

impl std::error::Error for RaftError {}

// ============================================================================
// Raft Node Configuration
// ============================================================================

/// Configuration for a Raft node
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// This node's ID
    pub node_id: NodeId,
    /// Minimum election timeout
    pub election_timeout_min: Duration,
    /// Maximum election timeout
    pub election_timeout_max: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Maximum entries per AppendEntries RPC
    pub max_entries_per_rpc: usize,
    /// Snapshot threshold (entries before compaction)
    pub snapshot_threshold: usize,
    /// Enable pre-vote (prevents disruption from isolated nodes)
    pub enable_pre_vote: bool,
    /// Batch replication configuration
    pub batch_config: BatchReplicationConfig,
    /// Data directory for persisting Raft state (None = in-memory only)
    pub data_dir: Option<std::path::PathBuf>,
}

/// Configuration for batch log replication.
///
/// Based on "Improved Raft with Async Batch Processing" (Springer, 2025):
/// - Accumulates entries for a configurable window before replicating
/// - Increases throughput 2-3.6x by batching log entries
/// - Processes parallel requests within the batch window
///
/// And "RaftOptima" (2025):
/// - Proxy leader delegation for large clusters (>5 nodes)
/// - Shows 60% latency reduction with up to 25 servers
#[derive(Debug, Clone)]
pub struct BatchReplicationConfig {
    /// Enable batch replication (default: true)
    pub enabled: bool,
    /// Maximum time to accumulate entries before forcing replication (default: 1ms)
    pub batch_window_ms: u64,
    /// Maximum entries to accumulate before forcing replication (default: 64)
    pub max_batch_size: usize,
    /// Enable proxy leader for clusters with >5 nodes (default: true)
    pub proxy_leader_enabled: bool,
    /// Number of proxy leaders to use (default: 2)
    pub proxy_leader_count: usize,
}

impl Default for BatchReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_window_ms: 1,
            max_batch_size: 64,
            proxy_leader_enabled: true,
            proxy_leader_count: 2,
        }
    }
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            election_timeout_min: Duration::from_millis(ELECTION_TIMEOUT_MIN_MS),
            election_timeout_max: Duration::from_millis(ELECTION_TIMEOUT_MAX_MS),
            heartbeat_interval: Duration::from_millis(HEARTBEAT_INTERVAL_MS),
            max_entries_per_rpc: MAX_ENTRIES_PER_RPC,
            snapshot_threshold: 10000,
            enable_pre_vote: true,
            batch_config: BatchReplicationConfig::default(),
            data_dir: None,
        }
    }
}

impl RaftConfig {
    /// Create a new configuration with the given node ID
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            ..Default::default()
        }
    }

    /// Generate a random election timeout
    pub fn random_election_timeout(&self) -> Duration {
        use rand::{Rng, RngExt};
        let mut rng = rand::rng();
        let min = self.election_timeout_min.as_millis() as u64;
        let max = self.election_timeout_max.as_millis() as u64;
        Duration::from_millis(rng.random_range(min..=max))
    }
}

// ============================================================================
// State Machine Interface
// ============================================================================

/// Trait for the replicated state machine
pub trait StateMachine: Send + Sync {
    /// Apply a command to the state machine
    fn apply(&mut self, command: &Command) -> Vec<u8>;

    /// Take a snapshot of the current state
    fn snapshot(&self) -> Vec<u8>;

    /// Restore state from a snapshot
    fn restore(&mut self, data: &[u8]) -> Result<(), String>;
}

/// A simple key-value state machine for testing
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct KvStateMachine {
    data: HashMap<String, Vec<u8>>,
}

impl KvStateMachine {
    /// Get a value from the state machine
    pub fn get(&self, key: &[u8]) -> Option<&Vec<u8>> {
        let key_str = String::from_utf8_lossy(key).to_string();
        self.data.get(&key_str)
    }

    /// Get all data
    pub fn data(&self) -> &HashMap<String, Vec<u8>> {
        &self.data
    }
}

impl StateMachine for KvStateMachine {
    fn apply(&mut self, command: &Command) -> Vec<u8> {
        match command {
            Command::Noop => Vec::new(),
            Command::Set { key, value } => {
                let key_str = String::from_utf8_lossy(key).to_string();
                self.data.insert(key_str, value.clone());
                value.clone()
            }
            Command::Delete { key } => {
                let key_str = String::from_utf8_lossy(key).to_string();
                self.data.remove(&key_str);
                Vec::new()
            }
            Command::ConfigChange(_) => Vec::new(),
            // MutationDelta is applied directly by the follower applier in lib.rs,
            // not through the KvStateMachine. Return the raw bytes for reference.
            Command::MutationDelta(data) => data.clone(),
        }
    }

    fn snapshot(&self) -> Vec<u8> {
        serde_json::to_vec(&self.data).unwrap_or_default()
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            self.data = HashMap::new();
            return Ok(());
        }
        self.data = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        Ok(())
    }
}

// ============================================================================
// Transport Interface
// ============================================================================

/// Trait for RPC transport between Raft nodes
#[async_trait::async_trait]
pub trait RaftTransport: Send + Sync {
    /// Send RequestVote RPC
    async fn send_request_vote(
        &self,
        target: &NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse, RaftError>;

    /// Send AppendEntries RPC
    async fn send_append_entries(
        &self,
        target: &NodeId,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse, RaftError>;

    /// Send InstallSnapshot RPC
    async fn send_install_snapshot(
        &self,
        target: &NodeId,
        request: InstallSnapshotRequest,
    ) -> Result<InstallSnapshotResponse, RaftError>;
}

/// In-memory transport for testing
pub struct InMemoryTransport {
    nodes: Arc<RwLock<HashMap<NodeId, mpsc::Sender<(RaftMessage, mpsc::Sender<RaftMessage>)>>>>,
}

impl InMemoryTransport {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(
        &self,
        node_id: NodeId,
        sender: mpsc::Sender<(RaftMessage, mpsc::Sender<RaftMessage>)>,
    ) {
        self.nodes.write().await.insert(node_id, sender);
    }

    pub async fn unregister(&self, node_id: &NodeId) {
        self.nodes.write().await.remove(node_id);
    }
}

#[async_trait::async_trait]
impl RaftTransport for InMemoryTransport {
    async fn send_request_vote(
        &self,
        target: &NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse, RaftError> {
        let nodes = self.nodes.read().await;
        let sender = nodes
            .get(target)
            .ok_or_else(|| RaftError::NodeNotFound(target.clone()))?
            .clone();
        drop(nodes);

        let (response_tx, mut response_rx) = mpsc::channel(1);
        sender
            .send((RaftMessage::RequestVote(request), response_tx))
            .await
            .map_err(|_| RaftError::NodeNotFound(target.clone()))?;

        match tokio::time::timeout(Duration::from_millis(100), response_rx.recv()).await {
            Ok(Some(RaftMessage::RequestVoteResponse(resp))) => Ok(resp),
            _ => Err(RaftError::Timeout),
        }
    }

    async fn send_append_entries(
        &self,
        target: &NodeId,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse, RaftError> {
        let nodes = self.nodes.read().await;
        let sender = nodes
            .get(target)
            .ok_or_else(|| RaftError::NodeNotFound(target.clone()))?
            .clone();
        drop(nodes);

        let (response_tx, mut response_rx) = mpsc::channel(1);
        sender
            .send((RaftMessage::AppendEntries(request), response_tx))
            .await
            .map_err(|_| RaftError::NodeNotFound(target.clone()))?;

        match tokio::time::timeout(Duration::from_millis(100), response_rx.recv()).await {
            Ok(Some(RaftMessage::AppendEntriesResponse(resp))) => Ok(resp),
            _ => Err(RaftError::Timeout),
        }
    }

    async fn send_install_snapshot(
        &self,
        target: &NodeId,
        request: InstallSnapshotRequest,
    ) -> Result<InstallSnapshotResponse, RaftError> {
        let nodes = self.nodes.read().await;
        let sender = nodes
            .get(target)
            .ok_or_else(|| RaftError::NodeNotFound(target.clone()))?
            .clone();
        drop(nodes);

        let (response_tx, mut response_rx) = mpsc::channel(1);
        sender
            .send((RaftMessage::InstallSnapshot(request), response_tx))
            .await
            .map_err(|_| RaftError::NodeNotFound(target.clone()))?;

        match tokio::time::timeout(Duration::from_millis(500), response_rx.recv()).await {
            Ok(Some(RaftMessage::InstallSnapshotResponse(resp))) => Ok(resp),
            _ => Err(RaftError::Timeout),
        }
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Raft Node
// ============================================================================

/// The main Raft consensus node
pub struct RaftNode<S: StateMachine, T: RaftTransport> {
    /// Node configuration
    config: RaftConfig,
    /// Current state (Follower, Candidate, Leader)
    state: RwLock<RaftState>,
    /// Persistent state
    persistent: RwLock<PersistentState>,
    /// Volatile state
    volatile: RwLock<VolatileState>,
    /// Leader-specific state (only valid when state is Leader)
    leader_state: RwLock<Option<LeaderState>>,
    /// Current cluster configuration
    cluster_config: RwLock<ClusterConfig>,
    /// Current known leader
    leader_id: RwLock<Option<NodeId>>,
    /// State machine
    state_machine: Mutex<S>,
    /// Transport layer
    transport: Arc<T>,
    /// Last heartbeat/election timeout reset
    last_heartbeat: RwLock<Instant>,
    /// Current election timeout
    election_timeout: RwLock<Duration>,
    /// Running flag
    running: AtomicBool,
    /// Latest snapshot
    snapshot: RwLock<Option<Snapshot>>,
    /// Snapshot offset during installation
    snapshot_offset: RwLock<u64>,
    /// Pending snapshot data during installation
    pending_snapshot: RwLock<Vec<u8>>,
    /// Channel for notifying about committed entries
    commit_notify: broadcast::Sender<LogIndex>,
    /// Notify replication loop immediately when new entries are proposed (HRP Phase 1)
    replication_notify: Notify,
    /// Append-only WAL for O(1) persistence (HRP Phase 1.2)
    wal: Option<std::sync::Mutex<RaftWal>>,
    /// HRP Phase 4: This node's energy state (updated periodically)
    energy_state: std::sync::RwLock<NodeEnergyState>,
    /// HRP Phase 4: Peer energy states (populated from heartbeat responses)
    peer_energy: std::sync::RwLock<HashMap<NodeId, NodeEnergyState>>,
    /// HRP Phase 4: Erasure coding configuration for large snapshot transfers
    erasure_config: Option<crate::hrp_erasure::ErasureConfig>,
    /// Pending erasure shards during snapshot installation (shard_index → shard)
    pending_erasure_shards: RwLock<Vec<Option<crate::hrp_erasure::ErasureShard>>>,
    /// Statistics
    stats: RaftStats,
}

/// Raft node statistics
#[derive(Debug, Default)]
pub struct RaftStats {
    pub elections_started: AtomicU64,
    pub elections_won: AtomicU64,
    pub votes_granted: AtomicU64,
    pub append_entries_sent: AtomicU64,
    pub append_entries_received: AtomicU64,
    pub snapshots_sent: AtomicU64,
    pub snapshots_installed: AtomicU64,
}

impl<S: StateMachine + 'static, T: RaftTransport + 'static> RaftNode<S, T> {
    /// Create a new Raft node.
    ///
    /// If `config.data_dir` is set, loads persisted state from disk (term, vote, log).
    /// Otherwise starts with a fresh in-memory state.
    pub fn new(
        config: RaftConfig,
        state_machine: S,
        transport: Arc<T>,
        cluster_config: ClusterConfig,
    ) -> Self {
        let (commit_notify, _) = broadcast::channel(1000);
        let election_timeout = config.random_election_timeout();

        // Load persisted state if data_dir is configured.
        // HRP Phase 1.2: Try WAL recovery first (replays delta entries on top of
        // the JSON snapshot), then fall back to JSON-only for backward compat.
        let (persistent_state, wal) = if let Some(ref dir) = config.data_dir {
            let state = match RaftWal::recover(dir) {
                Ok(state) => {
                    if state.current_term > 0 {
                        tracing::info!(
                            "Raft: Recovered persisted state (term={}, log_entries={}, voted_for={:?})",
                            state.current_term,
                            state.log.len(),
                            state.voted_for
                        );
                    }
                    state
                }
                Err(e) => {
                    tracing::error!("Raft: WAL recovery failed: {} — trying JSON fallback", e);
                    match PersistentState::load_from_dir(dir) {
                        Ok(s) => s,
                        Err(e2) => {
                            tracing::error!(
                                "Raft: JSON fallback also failed: {} — starting fresh",
                                e2
                            );
                            PersistentState::new()
                        }
                    }
                }
            };

            let wal = match RaftWal::open(dir) {
                Ok(w) => Some(std::sync::Mutex::new(w)),
                Err(e) => {
                    tracing::error!("Raft: Failed to open WAL: {} — using JSON persistence", e);
                    None
                }
            };

            (state, wal)
        } else {
            (PersistentState::new(), None)
        };

        Self {
            config,
            state: RwLock::new(RaftState::Follower),
            persistent: RwLock::new(persistent_state),
            volatile: RwLock::new(VolatileState::new()),
            leader_state: RwLock::new(None),
            cluster_config: RwLock::new(cluster_config),
            leader_id: RwLock::new(None),
            state_machine: Mutex::new(state_machine),
            transport,
            last_heartbeat: RwLock::new(Instant::now()),
            election_timeout: RwLock::new(election_timeout),
            running: AtomicBool::new(false),
            snapshot: RwLock::new(None),
            snapshot_offset: RwLock::new(0),
            pending_snapshot: RwLock::new(Vec::new()),
            commit_notify,
            replication_notify: Notify::new(),
            wal,
            energy_state: std::sync::RwLock::new(NodeEnergyState::default()),
            peer_energy: std::sync::RwLock::new(HashMap::new()),
            erasure_config: None,
            pending_erasure_shards: RwLock::new(Vec::new()),
            stats: RaftStats::default(),
        }
    }

    /// Enable erasure coding for large snapshot transfers.
    pub fn with_erasure_config(mut self, config: crate::hrp_erasure::ErasureConfig) -> Self {
        self.erasure_config = Some(config);
        self
    }

    /// Get the node ID
    pub fn node_id(&self) -> &NodeId {
        &self.config.node_id
    }

    /// Get the current state
    pub async fn state(&self) -> RaftState {
        *self.state.read().await
    }

    /// Get the current term
    pub async fn current_term(&self) -> Term {
        self.persistent.read().await.current_term
    }

    /// Get the current leader ID
    pub async fn leader_id(&self) -> Option<NodeId> {
        self.leader_id.read().await.clone()
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        *self.state.read().await == RaftState::Leader
    }

    /// Synchronous leader check (for use in non-async contexts).
    /// Uses try_read to avoid blocking; returns false if lock is contended.
    pub fn is_leader_sync(&self) -> bool {
        self.state
            .try_read()
            .map(|s| *s == RaftState::Leader)
            .unwrap_or(false)
    }

    /// Persist the current state to disk (if data_dir is configured).
    ///
    /// `new_entries` indicates how many entries from the tail of `state.log`
    /// were just appended and need to be WAL-written. Pass 0 for operations
    /// that only compact/checkpoint (e.g. snapshot install, log compaction).
    ///
    /// HRP Phase 1.2: If a WAL is available, only the new entries are appended
    /// (O(N) per batch, O(1) per single propose). Falls back to full JSON
    /// snapshot if no WAL.
    ///
    /// Per Raft spec: state must be on stable storage before responding to RPCs.
    /// Errors are propagated — callers must handle WAL failures.
    fn persist_state_sync(
        &self,
        state: &PersistentState,
        new_entries: usize,
    ) -> Result<(), RaftError> {
        if self.config.data_dir.is_none() {
            return Ok(());
        }

        // If WAL is available, append new entries (fast path)
        if let Some(ref wal_mutex) = self.wal {
            if let Ok(mut wal) = wal_mutex.lock() {
                // Append the last `new_entries` entries from the log
                let log_len = state.log.len();
                let start = log_len.saturating_sub(new_entries);
                for entry in &state.log[start..] {
                    wal.append_entry(entry)
                        .map_err(|e| RaftError::Internal(format!("WAL append failed: {}", e)))?;
                }

                // Periodic checkpoint to bound WAL size
                if wal.needs_checkpoint() {
                    if let Err(e) = wal.checkpoint(state) {
                        // Checkpoint failure is non-fatal — data is in WAL entries
                        tracing::error!("Raft WAL checkpoint failed: {}", e);
                    }
                }
                return Ok(());
            }
        }

        // Fallback: full JSON snapshot (legacy path)
        if let Some(ref dir) = self.config.data_dir {
            state
                .save_to_dir(dir)
                .map_err(|e| RaftError::Internal(format!("Failed to persist state: {}", e)))?;
        }

        Ok(())
    }

    /// Persist metadata (term, voted_for) to the WAL.
    ///
    /// Called when term or vote changes (not on every log append).
    fn persist_meta_sync(&self, term: Term, voted_for: &Option<NodeId>) -> Result<(), RaftError> {
        if let Some(ref wal_mutex) = self.wal {
            if let Ok(mut wal) = wal_mutex.lock() {
                wal.append_meta(term, voted_for)
                    .map_err(|e| RaftError::Internal(format!("WAL meta persist failed: {}", e)))?;
                return Ok(());
            }
        }

        // If no WAL, the full persist_state_sync will handle it
        Ok(())
    }

    /// Get commit index
    pub async fn commit_index(&self) -> LogIndex {
        self.volatile.read().await.commit_index
    }

    /// Get last applied index
    pub async fn last_applied(&self) -> LogIndex {
        self.volatile.read().await.last_applied
    }

    /// Access the state machine under its lock
    pub async fn with_state_machine<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&S) -> R,
    {
        let sm = self.state_machine.lock().await;
        f(&sm)
    }

    /// Subscribe to commit notifications
    pub fn subscribe_commits(&self) -> broadcast::Receiver<LogIndex> {
        self.commit_notify.subscribe()
    }

    /// Get the replication notify handle (HRP Phase 1 — event-driven replication).
    ///
    /// The replication loop awaits this Notify instead of sleeping on a fixed tick.
    /// When a new entry is proposed, the Notify wakes the replication loop immediately.
    pub fn replication_notify(&self) -> &Notify {
        &self.replication_notify
    }

    /// Get committed log entries in the given range (inclusive).
    /// Used by the follower applier to read SQL commands for local execution.
    pub async fn get_log_entries(&self, start: LogIndex, end: LogIndex) -> Vec<LogEntry> {
        let persistent = self.persistent.read().await;
        persistent.get_entries(start, end)
    }

    /// Get statistics
    pub fn stats(&self) -> &RaftStats {
        &self.stats
    }

    // ========================================================================
    // HRP Phase 4: Energy-Aware State
    // ========================================================================

    /// Update this node's energy state (called periodically from energy executor).
    pub fn update_energy_state(&self, state: NodeEnergyState) {
        if let Ok(mut es) = self.energy_state.write() {
            *es = state;
        }
    }

    /// Get a snapshot of this node's current energy state.
    pub fn get_energy_state(&self) -> NodeEnergyState {
        self.energy_state
            .read()
            .map(|es| es.clone())
            .unwrap_or_default()
    }

    /// Store a peer's energy state (called when processing heartbeat responses).
    pub fn update_peer_energy(&self, peer_id: &NodeId, state: NodeEnergyState) {
        if let Ok(mut pe) = self.peer_energy.write() {
            pe.insert(peer_id.clone(), state);
        }
    }

    /// Find the peer with the lowest energy cost (power_watts * load_factor).
    ///
    /// Returns `None` if no peer energy data is available.
    pub fn lowest_energy_peer(&self) -> Option<(NodeId, f64)> {
        let pe = self.peer_energy.read().ok()?;
        pe.iter()
            .map(|(id, state)| {
                let cost = state.power_watts * (0.1 + state.load_factor);
                (id.clone(), cost)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get all peer energy states (for monitoring/diagnostics).
    pub fn get_peer_energy_states(&self) -> HashMap<NodeId, NodeEnergyState> {
        self.peer_energy
            .read()
            .map(|pe| pe.clone())
            .unwrap_or_default()
    }

    // ========================================================================
    // State Transitions
    // ========================================================================

    /// Become a follower
    async fn become_follower(&self, term: Term, leader_id: Option<NodeId>) {
        let mut state = self.state.write().await;
        let mut persistent = self.persistent.write().await;

        if term > persistent.current_term {
            persistent.current_term = term;
            persistent.voted_for = None;
        }

        // Persist metadata (term+vote) before responding (Raft safety requirement)
        if let Err(e) = self.persist_meta_sync(persistent.current_term, &persistent.voted_for) {
            tracing::error!("Raft: failed to persist meta on become_follower: {}", e);
        }

        *state = RaftState::Follower;
        *self.leader_id.write().await = leader_id;
        *self.leader_state.write().await = None;

        // Reset election timeout
        self.reset_election_timeout().await;

        tracing::debug!("[{}] Became follower in term {}", self.config.node_id, term);
    }

    /// Become a candidate and start election
    async fn become_candidate(&self) {
        let mut state = self.state.write().await;
        let mut persistent = self.persistent.write().await;

        // Increment current term
        persistent.current_term += 1;
        let current_term = persistent.current_term;

        // Vote for self
        persistent.voted_for = Some(self.config.node_id.clone());

        // Persist metadata (term+vote) before sending RequestVote RPCs
        if let Err(e) = self.persist_meta_sync(persistent.current_term, &persistent.voted_for) {
            tracing::error!("Raft: failed to persist meta on become_candidate: {}", e);
        }

        *state = RaftState::Candidate;
        *self.leader_id.write().await = None;
        *self.leader_state.write().await = None;

        // Reset election timeout
        self.reset_election_timeout().await;

        self.stats.elections_started.fetch_add(1, Ordering::Relaxed);

        tracing::debug!(
            "[{}] Became candidate in term {}",
            self.config.node_id,
            current_term
        );
    }

    /// Become the leader
    async fn become_leader(&self) {
        // Scope the state + persistent locks so they are released before
        // propose_internal(). This prevents a deadlock where become_leader
        // holds state.write() and waits for persistent.write(), while an
        // RPC handler holds persistent.write() and waits for state.write()
        // (via become_follower).
        {
            let mut state = self.state.write().await;
            let persistent = self.persistent.read().await;

            *state = RaftState::Leader;
            *self.leader_id.write().await = Some(self.config.node_id.clone());

            // Initialize leader state
            let cluster_config = self.cluster_config.read().await;
            let mut peers = cluster_config.voting_members();
            peers.remove(&self.config.node_id);
            drop(cluster_config);

            let last_log_index = persistent.last_log_index();
            *self.leader_state.write().await = Some(LeaderState::new(&peers, last_log_index));

            self.stats.elections_won.fetch_add(1, Ordering::Relaxed);

            tracing::info!(
                "[{}] Became leader in term {}",
                self.config.node_id,
                persistent.current_term
            );
        }

        // Append a no-op entry to establish leadership (no locks held)
        let _ = self.propose_internal(Command::Noop).await;
    }

    /// Reset the election timeout with a new random value
    async fn reset_election_timeout(&self) {
        *self.last_heartbeat.write().await = Instant::now();
        *self.election_timeout.write().await = self.config.random_election_timeout();
    }

    // ========================================================================
    // Election
    // ========================================================================

    /// Check if election timeout has elapsed
    pub async fn election_timeout_elapsed(&self) -> bool {
        let last = *self.last_heartbeat.read().await;
        let timeout = *self.election_timeout.read().await;
        last.elapsed() >= timeout
    }

    /// Run an election
    pub async fn run_election(&self) {
        self.become_candidate().await;

        let persistent = self.persistent.read().await;
        let term = persistent.current_term;
        let last_log_index = persistent.last_log_index();
        let last_log_term = persistent.last_log_term();
        drop(persistent);

        let cluster_config = self.cluster_config.read().await;
        let peers: Vec<NodeId> = cluster_config
            .voting_members()
            .into_iter()
            .filter(|id| id != &self.config.node_id)
            .collect();
        let config_clone = cluster_config.clone();
        drop(cluster_config);

        // Vote for ourselves
        let mut votes_received: HashSet<NodeId> = HashSet::new();
        votes_received.insert(self.config.node_id.clone());

        // Send RequestVote RPCs to all peers
        let request = RequestVoteRequest {
            term,
            candidate_id: self.config.node_id.clone(),
            last_log_index,
            last_log_term,
        };

        // Collect votes in parallel
        let mut vote_futures = Vec::new();
        for peer in &peers {
            let transport = self.transport.clone();
            let peer_clone = peer.clone();
            let request_clone = request.clone();

            vote_futures.push(async move {
                let result = transport
                    .send_request_vote(&peer_clone, request_clone)
                    .await;
                (peer_clone, result)
            });
        }

        // Process votes as they come in
        let results = futures::future::join_all(vote_futures).await;

        for (peer, result) in results {
            match result {
                Ok(response) => {
                    // Check if we're still a candidate
                    if *self.state.read().await != RaftState::Candidate {
                        return;
                    }

                    // Check for higher term
                    if response.term > term {
                        self.become_follower(response.term, None).await;
                        return;
                    }

                    if response.vote_granted {
                        votes_received.insert(peer);
                        self.stats.votes_granted.fetch_add(1, Ordering::Relaxed);

                        // Check if we have a majority
                        if config_clone.has_quorum(&votes_received) {
                            self.become_leader().await;
                            return;
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("RequestVote to {} failed: {}", peer, e);
                }
            }
        }

        // Election failed, will try again after timeout
        tracing::debug!(
            "[{}] Election failed, got {} votes",
            self.config.node_id,
            votes_received.len()
        );
    }

    // ========================================================================
    // Log Replication
    // ========================================================================

    /// Propose a command to be replicated
    pub async fn propose(&self, command: Command) -> Result<LogIndex, RaftError> {
        if !self.is_leader().await {
            let leader = self.leader_id().await;
            return Err(RaftError::NotLeader(leader));
        }

        self.propose_internal(command).await
    }

    /// Internal propose (doesn't check leadership)
    async fn propose_internal(&self, command: Command) -> Result<LogIndex, RaftError> {
        let mut persistent = self.persistent.write().await;
        let term = persistent.current_term;
        let index = persistent.last_log_index() + 1;

        let entry = LogEntry::new(term, index, command);
        persistent.log.push(entry);

        // Persist new log entry before acknowledging
        self.persist_state_sync(&persistent, 1)?;

        tracing::debug!(
            "[{}] Proposed entry at index {} term {}",
            self.config.node_id,
            index,
            term
        );

        // HRP Phase 1: Wake the replication loop immediately instead of
        // waiting for the next 50ms tick. This eliminates the 0-50ms
        // latency penalty on every propose.
        self.replication_notify.notify_one();

        Ok(index)
    }

    /// Propose a batch of commands atomically.
    ///
    /// This is the key optimization from the batch replication research:
    /// instead of proposing one command at a time (each triggering replication),
    /// we accumulate multiple commands and append them to the log in one operation.
    ///
    /// Benefits:
    /// - 2-3.6x throughput improvement (amortized replication overhead)
    /// - Single AppendEntries RPC carries all batch entries
    /// - Reduces network round-trips for write-heavy workloads
    ///
    /// Returns the log indices of all proposed entries.
    pub async fn propose_batch(&self, commands: Vec<Command>) -> Result<Vec<LogIndex>, RaftError> {
        if !self.is_leader().await {
            let leader = self.leader_id().await;
            return Err(RaftError::NotLeader(leader));
        }

        if commands.is_empty() {
            return Ok(Vec::new());
        }

        let mut persistent = self.persistent.write().await;
        let term = persistent.current_term;
        let batch_len = commands.len();
        let mut indices = Vec::with_capacity(batch_len);

        for command in commands {
            let index = persistent.last_log_index() + 1;
            let entry = LogEntry::new(term, index, command);
            persistent.log.push(entry);
            indices.push(index);
        }

        // Persist ALL batch entries before acknowledging (Bug fix: was only writing last entry)
        self.persist_state_sync(&persistent, batch_len)?;

        let batch_size = indices.len();
        let first_index = indices[0];
        let last_index = *indices.last().unwrap_or(&first_index);

        tracing::debug!(
            "[{}] Proposed batch of {} entries at indices {}-{} term {}",
            self.config.node_id,
            batch_size,
            first_index,
            last_index,
            term
        );

        // HRP Phase 1: Wake the replication loop immediately
        self.replication_notify.notify_one();

        Ok(indices)
    }

    /// Send heartbeats/AppendEntries to all followers
    pub async fn send_heartbeats(&self) {
        if !self.is_leader().await {
            return;
        }

        let persistent = self.persistent.read().await;
        let volatile = self.volatile.read().await;
        let leader_state = self.leader_state.read().await;

        let Some(ref leader) = *leader_state else {
            return;
        };

        let cluster_config = self.cluster_config.read().await;
        let peers: Vec<NodeId> = cluster_config
            .voting_members()
            .into_iter()
            .filter(|id| id != &self.config.node_id)
            .collect();
        drop(cluster_config);

        let term = persistent.current_term;
        let commit_index = volatile.commit_index;

        // Send AppendEntries to each peer
        let mut append_futures = Vec::new();

        for peer in peers {
            let next_index = *leader.next_index.get(&peer).unwrap_or(&1);
            let prev_log_index = next_index.saturating_sub(1);
            let prev_log_term = persistent.get_term(prev_log_index).unwrap_or(0);

            // Get entries to send
            let last_index = persistent.last_log_index();
            let end_index = std::cmp::min(
                next_index + self.config.max_entries_per_rpc as u64 - 1,
                last_index,
            );

            let entries = if next_index <= last_index {
                persistent.get_entries(next_index, end_index)
            } else {
                Vec::new()
            };

            let request = AppendEntriesRequest {
                term,
                leader_id: self.config.node_id.clone(),
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit: commit_index,
                energy_state: Some(self.get_energy_state()),
            };

            let transport = self.transport.clone();
            let peer_clone = peer.clone();

            append_futures.push(async move {
                let result = transport.send_append_entries(&peer_clone, request).await;
                (peer_clone, result)
            });

            self.stats
                .append_entries_sent
                .fetch_add(1, Ordering::Relaxed);
        }

        drop(leader_state);
        drop(volatile);
        drop(persistent);

        // Process responses
        let results = futures::future::join_all(append_futures).await;

        let current_term = self.current_term().await;

        for (peer, result) in results {
            match result {
                Ok(response) => {
                    if response.term > current_term {
                        self.become_follower(response.term, None).await;
                        return;
                    }

                    // Discard stale responses from previous terms
                    if response.term != current_term {
                        tracing::debug!(
                            "Discarding stale AppendEntries response from {} (term {} != {})",
                            peer,
                            response.term,
                            current_term
                        );
                        continue;
                    }

                    // HRP Phase 4: Capture peer energy state from heartbeat response
                    if let Some(ref es) = response.energy_state {
                        self.update_peer_energy(&peer, es.clone());
                    }

                    let mut leader_state = self.leader_state.write().await;
                    if let Some(ref mut leader) = *leader_state {
                        if response.success {
                            leader.update_match(&peer, response.match_index);
                        } else {
                            leader.decrement_next(&peer, response.conflict_index);
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("AppendEntries to {} failed: {}", peer, e);
                }
            }
        }

        // Try to advance commit index
        self.maybe_advance_commit_index().await;
    }

    /// Advance commit index if possible
    async fn maybe_advance_commit_index(&self) {
        let persistent = self.persistent.read().await;
        let leader_state = self.leader_state.read().await;
        let cluster_config = self.cluster_config.read().await;

        let Some(ref leader) = *leader_state else {
            return;
        };

        let current_term = persistent.current_term;
        let last_index = persistent.last_log_index();

        // Find the highest index that has been replicated to a majority
        // Read commit_index outside the loop to avoid holding the RwLockReadGuard
        // across the loop body (which would deadlock with volatile.write() inside).
        let current_commit = self.volatile.read().await.commit_index;
        for index in (current_commit + 1)..=last_index {
            // Only commit entries from current term
            if let Some(term) = persistent.get_term(index) {
                if term != current_term {
                    continue;
                }
            } else {
                continue;
            }

            // Count replications (including self)
            let mut replicated: HashSet<NodeId> = HashSet::new();
            replicated.insert(self.config.node_id.clone());

            for (node_id, &match_index) in &leader.match_index {
                if match_index >= index {
                    replicated.insert(node_id.clone());
                }
            }

            if cluster_config.has_quorum(&replicated) {
                let mut volatile = self.volatile.write().await;
                volatile.commit_index = index;
                let _ = self.commit_notify.send(index);

                tracing::debug!(
                    "[{}] Committed index {} with {} replications",
                    self.config.node_id,
                    index,
                    replicated.len()
                );
            }
        }
    }

    /// Apply committed entries to state machine
    pub async fn apply_committed_entries(&self) {
        let volatile = self.volatile.read().await;
        let commit_index = volatile.commit_index;
        let mut last_applied = volatile.last_applied;
        drop(volatile);

        if last_applied >= commit_index {
            return;
        }

        let persistent = self.persistent.read().await;
        let entries: Vec<LogEntry> = persistent
            .get_entries(last_applied + 1, commit_index)
            .into_iter()
            .collect();
        drop(persistent);

        let mut state_machine = self.state_machine.lock().await;
        for entry in entries {
            state_machine.apply(&entry.command);
            last_applied = entry.index;

            // Handle configuration changes
            if let Command::ConfigChange(config) = &entry.command {
                let mut cluster_config = self.cluster_config.write().await;
                *cluster_config = config.clone();

                // If completing joint consensus, apply final config
                if config.new_members.is_some() {
                    // This is C_old,new - need to append C_new entry
                    if self.is_leader().await {
                        let new_config = ClusterConfig {
                            members: config
                                .new_members
                                .clone()
                                .expect("new_members confirmed Some above"),
                            new_members: None,
                            config_index: entry.index + 1,
                        };
                        drop(cluster_config);
                        let _ = self
                            .propose_internal(Command::ConfigChange(new_config))
                            .await;
                    }
                }
            }
        }

        let mut volatile = self.volatile.write().await;
        volatile.last_applied = last_applied;

        tracing::debug!(
            "[{}] Applied entries up to index {}",
            self.config.node_id,
            last_applied
        );
    }

    // ========================================================================
    // RPC Handlers
    // ========================================================================

    /// Handle RequestVote RPC
    pub async fn handle_request_vote(&self, request: RequestVoteRequest) -> RequestVoteResponse {
        let mut persistent = self.persistent.write().await;

        // Reply false if term < currentTerm (§5.1)
        if request.term < persistent.current_term {
            return RequestVoteResponse {
                term: persistent.current_term,
                vote_granted: false,
            };
        }

        // If RPC request contains term > currentTerm, update currentTerm
        // and convert to follower (§5.1)
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            drop(persistent);
            self.become_follower(request.term, None).await;
            persistent = self.persistent.write().await;
        }

        // If votedFor is null or candidateId, and candidate's log is at
        // least as up-to-date as receiver's log, grant vote (§5.2, §5.4)
        let vote_granted = {
            let can_vote = persistent.voted_for.is_none()
                || persistent.voted_for.as_ref() == Some(&request.candidate_id);

            let log_ok = {
                let last_term = persistent.last_log_term();
                let last_index = persistent.last_log_index();

                // Candidate's log is at least as up-to-date
                request.last_log_term > last_term
                    || (request.last_log_term == last_term && request.last_log_index >= last_index)
            };

            can_vote && log_ok
        };

        // Save term before dropping the write guard — acquiring a new
        // read lock while the write guard is alive would self-deadlock.
        let response_term = persistent.current_term;

        if vote_granted {
            persistent.voted_for = Some(request.candidate_id.clone());
            // Persist vote before responding
            if let Err(e) = self.persist_meta_sync(persistent.current_term, &persistent.voted_for) {
                tracing::error!("Raft: failed to persist vote: {}", e);
            }
            drop(persistent);
            self.reset_election_timeout().await;

            tracing::debug!(
                "[{}] Granted vote to {} in term {}",
                self.config.node_id,
                request.candidate_id,
                request.term
            );
        } else {
            drop(persistent);
        }

        RequestVoteResponse {
            term: response_term,
            vote_granted,
        }
    }

    /// Handle AppendEntries RPC
    pub async fn handle_append_entries(
        &self,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse {
        self.stats
            .append_entries_received
            .fetch_add(1, Ordering::Relaxed);

        // HRP Phase 4: Store leader's energy state if present
        if let Some(ref es) = request.energy_state {
            self.update_peer_energy(&request.leader_id, es.clone());
        }

        let mut persistent = self.persistent.write().await;

        // Reply false if term < currentTerm (§5.1)
        if request.term < persistent.current_term {
            return AppendEntriesResponse {
                term: persistent.current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
                match_index: 0,
                energy_state: Some(self.get_energy_state()),
            };
        }

        // If RPC request contains term >= currentTerm, recognize leader
        if request.term >= persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            drop(persistent);
            self.become_follower(request.term, Some(request.leader_id.clone()))
                .await;
            persistent = self.persistent.write().await;
        }

        // Reset election timeout (we heard from leader)
        drop(persistent);
        self.reset_election_timeout().await;
        persistent = self.persistent.write().await;

        // Reply false if log doesn't contain an entry at prevLogIndex
        // whose term matches prevLogTerm (§5.3)
        if request.prev_log_index > 0 {
            match persistent.get_term(request.prev_log_index) {
                None => {
                    // Log is too short
                    return AppendEntriesResponse {
                        term: persistent.current_term,
                        success: false,
                        conflict_index: Some(persistent.last_log_index() + 1),
                        conflict_term: None,
                        match_index: 0,
                        energy_state: Some(self.get_energy_state()),
                    };
                }
                Some(term) if term != request.prev_log_term => {
                    // Term mismatch - find first entry with conflicting term
                    let conflict_term = term;
                    let mut conflict_index = request.prev_log_index;
                    while conflict_index > 1 {
                        if let Some(t) = persistent.get_term(conflict_index - 1) {
                            if t != conflict_term {
                                break;
                            }
                        }
                        conflict_index -= 1;
                    }
                    return AppendEntriesResponse {
                        term: persistent.current_term,
                        success: false,
                        conflict_index: Some(conflict_index),
                        conflict_term: Some(conflict_term),
                        match_index: 0,
                        energy_state: Some(self.get_energy_state()),
                    };
                }
                _ => {}
            }
        }

        // If an existing entry conflicts with a new one (same index
        // but different terms), delete the existing entry and all that
        // follow it (§5.3)
        for entry in &request.entries {
            if let Some(existing_term) = persistent.get_term(entry.index) {
                if existing_term != entry.term {
                    persistent.truncate_from(entry.index);
                    break;
                }
            }
        }

        // Append any new entries not already in the log
        let last_new_index = request
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(request.prev_log_index);
        let mut appended_count = 0usize;
        for entry in request.entries {
            if entry.index > persistent.last_log_index() {
                persistent.log.push(entry);
                appended_count += 1;
            }
        }

        // Persist after log changes (append or truncate)
        if appended_count > 0 {
            if let Err(e) = self.persist_state_sync(&persistent, appended_count) {
                tracing::error!("Raft: WAL persist failed in append_entries: {}", e);
                return AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    conflict_index: None,
                    conflict_term: None,
                    match_index: 0,
                    energy_state: None,
                };
            }
        }

        let current_term = persistent.current_term;
        drop(persistent);

        // If leaderCommit > commitIndex, set commitIndex =
        // min(leaderCommit, index of last new entry)
        {
            let mut volatile = self.volatile.write().await;
            if request.leader_commit > volatile.commit_index {
                volatile.commit_index = std::cmp::min(request.leader_commit, last_new_index);
                let _ = self.commit_notify.send(volatile.commit_index);
            }
        }

        AppendEntriesResponse {
            term: current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
            match_index: last_new_index,
            energy_state: Some(self.get_energy_state()),
        }
    }

    /// Handle InstallSnapshot RPC
    pub async fn handle_install_snapshot(
        &self,
        request: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        let mut persistent = self.persistent.write().await;

        // Reply immediately if term < currentTerm
        if request.term < persistent.current_term {
            return InstallSnapshotResponse {
                term: persistent.current_term,
                success: false,
                next_offset: 0,
            };
        }

        // Update term if needed
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            drop(persistent);
            self.become_follower(request.term, Some(request.leader_id.clone()))
                .await;
            persistent = self.persistent.write().await;
        }

        drop(persistent);
        self.reset_election_timeout().await;

        // HRP Phase 4: Detect erasure-coded shards by magic prefix
        if request.data.starts_with(ERASURE_SHARD_MAGIC) {
            return self.handle_erasure_shard(request).await;
        }

        // Handle standard snapshot chunks
        let mut snapshot_offset = self.snapshot_offset.write().await;
        let mut pending_snapshot = self.pending_snapshot.write().await;

        // If offset is 0, create a new snapshot buffer
        if request.offset == 0 {
            *snapshot_offset = 0;
            pending_snapshot.clear();
        }

        // Check offset matches expected
        if request.offset != *snapshot_offset {
            return InstallSnapshotResponse {
                term: self.current_term().await,
                success: false,
                next_offset: *snapshot_offset,
            };
        }

        // Append chunk
        pending_snapshot.extend_from_slice(&request.data);
        *snapshot_offset += request.data.len() as u64;

        // If done, apply the snapshot
        if request.done {
            let snapshot_data = std::mem::take(&mut *pending_snapshot);
            *snapshot_offset = 0;
            drop(pending_snapshot);
            drop(snapshot_offset);

            // Apply snapshot to state machine
            let mut state_machine = self.state_machine.lock().await;
            if let Err(e) = state_machine.restore(&snapshot_data) {
                tracing::error!("Failed to restore snapshot: {}", e);
                return InstallSnapshotResponse {
                    term: self.current_term().await,
                    success: false,
                    next_offset: 0,
                };
            }
            drop(state_machine);

            // Update persistent state
            let mut persistent = self.persistent.write().await;
            persistent.log.clear();
            // Add a placeholder entry at the snapshot index
            persistent.log.push(LogEntry::new(
                request.last_included_term,
                request.last_included_index,
                Command::Noop,
            ));
            // Persist after snapshot log replacement (1 placeholder entry added)
            if let Err(e) = self.persist_state_sync(&persistent, 1) {
                tracing::error!("Raft: WAL persist failed in install_snapshot: {}", e);
                return InstallSnapshotResponse {
                    term: persistent.current_term,
                    success: false,
                    next_offset: 0,
                };
            }
            drop(persistent);

            // Update volatile state
            let mut volatile = self.volatile.write().await;
            volatile.commit_index = request.last_included_index;
            volatile.last_applied = request.last_included_index;
            drop(volatile);

            // Update cluster config
            let mut cluster_config = self.cluster_config.write().await;
            *cluster_config = request.config;
            drop(cluster_config);

            // Store snapshot
            let mut snapshot = self.snapshot.write().await;
            *snapshot = Some(Snapshot {
                metadata: SnapshotMetadata {
                    last_included_index: request.last_included_index,
                    last_included_term: request.last_included_term,
                    config: self.cluster_config.read().await.clone(),
                    total_size: snapshot_data.len() as u64,
                },
                data: snapshot_data,
            });

            self.stats
                .snapshots_installed
                .fetch_add(1, Ordering::Relaxed);

            tracing::info!(
                "[{}] Installed snapshot at index {}",
                self.config.node_id,
                request.last_included_index
            );
        }

        InstallSnapshotResponse {
            term: self.current_term().await,
            success: true,
            next_offset: self.snapshot_offset.read().await.clone(),
        }
    }

    /// Handle an erasure-coded snapshot shard.
    /// Accumulates shards; when k data shards are received, reconstructs the
    /// original snapshot and applies it to the state machine.
    async fn handle_erasure_shard(
        &self,
        request: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        // Deserialize the shard (skip magic prefix)
        let shard: crate::hrp_erasure::ErasureShard =
            match bincode::serde::decode_from_slice(&request.data[ERASURE_SHARD_MAGIC.len()..], bincode::config::standard()).map(|(v, _)| v) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Failed to deserialize erasure shard: {}", e);
                    return InstallSnapshotResponse {
                        term: self.current_term().await,
                        success: false,
                        next_offset: 0,
                    };
                }
            };

        let total = shard.total_shards;
        let data_shards = shard.data_shards;
        let shard_index = shard.index;

        let mut pending = self.pending_erasure_shards.write().await;
        // Initialize shard buffer if needed
        if pending.len() != total {
            *pending = vec![None; total];
        }
        pending[shard_index] = Some(shard);

        // Count received shards
        let received = pending.iter().filter(|s| s.is_some()).count();
        tracing::debug!(
            "Erasure shard {}/{} received ({}/{} needed for reconstruction)",
            shard_index,
            total,
            received,
            data_shards
        );

        // If we have enough shards (>= k), attempt reconstruction.
        // NOTE: We no longer require request.done — if the last shard is lost,
        // we can still reconstruct from k shards without it.
        if received >= data_shards {
            let shards_for_decode: Vec<Option<crate::hrp_erasure::ErasureShard>> =
                std::mem::take(&mut *pending);
            drop(pending);

            match crate::hrp_erasure::decode(&shards_for_decode) {
                Ok(snapshot_data) => {
                    tracing::info!(
                        "Erasure-decoded snapshot: {} shards → {} bytes",
                        received,
                        snapshot_data.len()
                    );
                    return self.apply_snapshot_data(snapshot_data, &request).await;
                }
                Err(e) => {
                    tracing::error!("Erasure decode failed: {}", e);
                    return InstallSnapshotResponse {
                        term: self.current_term().await,
                        success: false,
                        next_offset: 0,
                    };
                }
            }
        }

        // Shard accepted, waiting for more
        InstallSnapshotResponse {
            term: self.current_term().await,
            success: true,
            next_offset: shard_index as u64 + 1,
        }
    }

    /// Apply reconstructed snapshot data to the state machine and update
    /// persistent/volatile state. Shared by both standard and erasure-coded paths.
    async fn apply_snapshot_data(
        &self,
        snapshot_data: Vec<u8>,
        request: &InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        let mut state_machine = self.state_machine.lock().await;
        if let Err(e) = state_machine.restore(&snapshot_data) {
            tracing::error!("Failed to restore snapshot: {}", e);
            return InstallSnapshotResponse {
                term: self.current_term().await,
                success: false,
                next_offset: 0,
            };
        }
        drop(state_machine);

        let mut persistent = self.persistent.write().await;
        persistent.log.clear();
        persistent.log.push(LogEntry::new(
            request.last_included_term,
            request.last_included_index,
            Command::Noop,
        ));
        if let Err(e) = self.persist_state_sync(&persistent, 1) {
            tracing::error!("Raft: WAL persist failed in apply_snapshot_data: {}", e);
            return InstallSnapshotResponse {
                term: persistent.current_term,
                success: false,
                next_offset: 0,
            };
        }
        drop(persistent);

        let mut volatile = self.volatile.write().await;
        volatile.commit_index = request.last_included_index;
        volatile.last_applied = request.last_included_index;
        drop(volatile);

        let mut cluster_config = self.cluster_config.write().await;
        *cluster_config = request.config.clone();
        drop(cluster_config);

        let mut snapshot = self.snapshot.write().await;
        *snapshot = Some(Snapshot {
            metadata: SnapshotMetadata {
                last_included_index: request.last_included_index,
                last_included_term: request.last_included_term,
                config: self.cluster_config.read().await.clone(),
                total_size: snapshot_data.len() as u64,
            },
            data: snapshot_data,
        });

        self.stats
            .snapshots_installed
            .fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            "[{}] Installed snapshot at index {} (erasure-coded)",
            self.config.node_id,
            request.last_included_index
        );

        InstallSnapshotResponse {
            term: self.current_term().await,
            success: true,
            next_offset: 0,
        }
    }

    // ========================================================================
    // Snapshotting
    // ========================================================================

    /// Take a snapshot of the current state
    pub async fn take_snapshot(&self) -> Result<Snapshot, RaftError> {
        let volatile = self.volatile.read().await;
        let persistent = self.persistent.read().await;

        if volatile.last_applied == 0 {
            return Err(RaftError::SnapshotError("No entries applied yet".into()));
        }

        let last_applied = volatile.last_applied;
        let last_term = persistent
            .get_term(last_applied)
            .ok_or_else(|| RaftError::SnapshotError("No term for last applied".into()))?;

        drop(persistent);
        drop(volatile);

        // Get state machine snapshot
        let state_machine = self.state_machine.lock().await;
        let data = state_machine.snapshot();
        drop(state_machine);

        let cluster_config = self.cluster_config.read().await.clone();

        let snapshot = Snapshot {
            metadata: SnapshotMetadata {
                last_included_index: last_applied,
                last_included_term: last_term,
                config: cluster_config,
                total_size: data.len() as u64,
            },
            data,
        };

        // Store snapshot
        let mut stored_snapshot = self.snapshot.write().await;
        *stored_snapshot = Some(snapshot.clone());

        // Compact log
        let mut persistent = self.persistent.write().await;
        persistent.compact_until(last_applied);
        self.persist_state_sync(&persistent, 0)?; // 0 new entries, just checkpoint

        tracing::info!(
            "[{}] Created snapshot at index {}",
            self.config.node_id,
            last_applied
        );

        Ok(snapshot)
    }

    /// Send snapshot to a follower
    async fn send_snapshot(&self, follower_id: &NodeId) -> Result<(), RaftError> {
        let snapshot = self.snapshot.read().await;
        let Some(ref snap) = *snapshot else {
            return Err(RaftError::SnapshotError("No snapshot available".into()));
        };

        let snapshot_clone = snap.clone();
        drop(snapshot);

        // HRP Phase 4: If erasure coding is configured and payload exceeds threshold,
        // encode into shards and send each shard as a separate InstallSnapshot RPC.
        if let Some(ref ec) = self.erasure_config {
            if snapshot_clone.data.len() > ec.threshold_bytes {
                return self
                    .send_snapshot_erasure_coded(follower_id, &snapshot_clone, ec)
                    .await;
            }
        }

        // Standard chunk-based transfer
        let mut offset: u64 = 0;
        let total_size = snapshot_clone.data.len() as u64;

        while offset < total_size {
            let chunk_end = std::cmp::min(
                offset as usize + MAX_SNAPSHOT_CHUNK_SIZE,
                snapshot_clone.data.len(),
            );
            let chunk = snapshot_clone.data[offset as usize..chunk_end].to_vec();
            let done = chunk_end as u64 == total_size;

            let request = InstallSnapshotRequest {
                term: self.current_term().await,
                leader_id: self.config.node_id.clone(),
                last_included_index: snapshot_clone.metadata.last_included_index,
                last_included_term: snapshot_clone.metadata.last_included_term,
                offset,
                data: chunk,
                done,
                config: snapshot_clone.metadata.config.clone(),
            };

            match self
                .transport
                .send_install_snapshot(follower_id, request)
                .await
            {
                Ok(response) => {
                    if !response.success {
                        if response.term > self.current_term().await {
                            self.become_follower(response.term, None).await;
                            return Err(RaftError::NotLeader(None));
                        }
                        // Reset and retry
                        offset = response.next_offset;
                        continue;
                    }
                    offset = response.next_offset;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        self.stats.snapshots_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Send a snapshot using erasure coding: encode into k+m shards, send each
    /// as a separate InstallSnapshot RPC. The follower reconstructs when k
    /// shards are received. Magic prefix `ERASURE_SHARD_MAGIC` distinguishes
    /// erasure-coded chunks from standard chunks.
    async fn send_snapshot_erasure_coded(
        &self,
        follower_id: &NodeId,
        snapshot: &Snapshot,
        ec: &crate::hrp_erasure::ErasureConfig,
    ) -> Result<(), RaftError> {
        let shards = crate::hrp_erasure::encode(&snapshot.data, ec)
            .map_err(|e| RaftError::SnapshotError(format!("Erasure encode failed: {}", e)))?;
        let total_shards = shards.len() as u64;

        tracing::info!(
            "Erasure-coded snapshot: {} bytes → {} shards ({} data + {} parity)",
            snapshot.data.len(),
            total_shards,
            ec.data_shards,
            ec.parity_shards
        );

        for (i, shard) in shards.iter().enumerate() {
            // Serialize the shard with magic prefix so receiver can detect it
            let mut shard_data = ERASURE_SHARD_MAGIC.to_vec();
            let encoded =
                bincode::serde::encode_to_vec(shard, bincode::config::standard()).map_err(|e| RaftError::SnapshotError(e.to_string()))?;
            shard_data.extend_from_slice(&encoded);

            let done = i as u64 + 1 == total_shards;
            let request = InstallSnapshotRequest {
                term: self.current_term().await,
                leader_id: self.config.node_id.clone(),
                last_included_index: snapshot.metadata.last_included_index,
                last_included_term: snapshot.metadata.last_included_term,
                offset: i as u64, // shard index
                data: shard_data,
                done,
                config: snapshot.metadata.config.clone(),
            };

            match self
                .transport
                .send_install_snapshot(follower_id, request)
                .await
            {
                Ok(response) => {
                    if !response.success {
                        if response.term > self.current_term().await {
                            self.become_follower(response.term, None).await;
                            return Err(RaftError::NotLeader(None));
                        }
                        return Err(RaftError::SnapshotError(format!(
                            "Follower rejected erasure shard {}",
                            i
                        )));
                    }
                }
                Err(e) => return Err(e),
            }
        }

        self.stats.snapshots_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // ========================================================================
    // Cluster Membership Changes
    // ========================================================================

    /// Add a new node to the cluster (joint consensus)
    pub async fn add_node(&self, node_id: NodeId) -> Result<LogIndex, RaftError> {
        if !self.is_leader().await {
            return Err(RaftError::NotLeader(self.leader_id().await));
        }

        let cluster_config = self.cluster_config.read().await;
        if cluster_config.is_joint() {
            return Err(RaftError::InvalidConfig(
                "Already in joint consensus".into(),
            ));
        }

        // Create C_old,new configuration
        let mut new_members = cluster_config.members.clone();
        new_members.insert(node_id);

        let joint_config = ClusterConfig {
            members: cluster_config.members.clone(),
            new_members: Some(new_members),
            config_index: self.persistent.read().await.last_log_index() + 1,
        };
        drop(cluster_config);

        // Propose the joint configuration
        self.propose(Command::ConfigChange(joint_config)).await
    }

    /// Remove a node from the cluster (joint consensus)
    pub async fn remove_node(&self, node_id: NodeId) -> Result<LogIndex, RaftError> {
        if !self.is_leader().await {
            return Err(RaftError::NotLeader(self.leader_id().await));
        }

        let cluster_config = self.cluster_config.read().await;
        if cluster_config.is_joint() {
            return Err(RaftError::InvalidConfig(
                "Already in joint consensus".into(),
            ));
        }

        if !cluster_config.members.contains(&node_id) {
            return Err(RaftError::NodeNotFound(node_id));
        }

        // Create C_old,new configuration
        let mut new_members = cluster_config.members.clone();
        new_members.remove(&node_id);

        if new_members.is_empty() {
            return Err(RaftError::InvalidConfig("Cannot remove last node".into()));
        }

        let joint_config = ClusterConfig {
            members: cluster_config.members.clone(),
            new_members: Some(new_members),
            config_index: self.persistent.read().await.last_log_index() + 1,
        };
        drop(cluster_config);

        // Propose the joint configuration
        self.propose(Command::ConfigChange(joint_config)).await
    }

    /// Get current cluster configuration
    pub async fn cluster_config(&self) -> ClusterConfig {
        self.cluster_config.read().await.clone()
    }

    // ========================================================================
    // Main Loop
    // ========================================================================

    /// Start the Raft node
    pub async fn start(&self) {
        self.running.store(true, Ordering::SeqCst);

        tracing::info!("[{}] Starting Raft node", self.config.node_id);
    }

    /// Stop the Raft node
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if node is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Run the main tick (call periodically)
    pub async fn tick(&self) {
        if !self.is_running() {
            return;
        }

        let state = *self.state.read().await;

        match state {
            RaftState::Follower | RaftState::Candidate => {
                // Check election timeout
                if self.election_timeout_elapsed().await {
                    self.run_election().await;
                }
            }
            RaftState::Leader => {
                // Send heartbeats
                self.send_heartbeats().await;
            }
        }

        // Apply committed entries
        self.apply_committed_entries().await;

        // Check if we need to take a snapshot
        let persistent = self.persistent.read().await;
        let volatile = self.volatile.read().await;
        if persistent.log.len() > self.config.snapshot_threshold {
            drop(volatile);
            drop(persistent);
            let _ = self.take_snapshot().await;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_config(node_id: &str) -> RaftConfig {
        RaftConfig {
            node_id: node_id.to_string(),
            election_timeout_min: Duration::from_millis(50),
            election_timeout_max: Duration::from_millis(100),
            heartbeat_interval: Duration::from_millis(25),
            max_entries_per_rpc: 10,
            snapshot_threshold: 100,
            enable_pre_vote: false,
            batch_config: BatchReplicationConfig::default(),
            data_dir: None,
        }
    }

    fn create_test_cluster_config(nodes: Vec<&str>) -> ClusterConfig {
        ClusterConfig::new(nodes.into_iter().map(String::from).collect())
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(
            1,
            1,
            Command::Set {
                key: b"key".to_vec(),
                value: b"value".to_vec(),
            },
        );

        assert_eq!(entry.term, 1);
        assert_eq!(entry.index, 1);
        match entry.command {
            Command::Set { ref key, ref value } => {
                assert_eq!(key, b"key");
                assert_eq!(value, b"value");
            }
            _ => panic!("Expected Set command"),
        }
    }

    #[test]
    fn test_command_encode_decode() {
        let cmd = Command::Set {
            key: b"test_key".to_vec(),
            value: b"test_value".to_vec(),
        };

        let encoded = cmd.encode();
        let decoded = Command::decode(&encoded).unwrap();

        assert_eq!(cmd, decoded);
    }

    #[test]
    fn test_cluster_config_single() {
        let config = ClusterConfig::single("node1".to_string());

        assert_eq!(config.members.len(), 1);
        assert!(config.members.contains("node1"));
        assert!(!config.is_joint());
        assert_eq!(config.old_majority(), 1);
    }

    #[test]
    fn test_cluster_config_majority() {
        let config = create_test_cluster_config(vec!["n1", "n2", "n3"]);

        assert_eq!(config.old_majority(), 2);

        let mut voters = HashSet::new();
        voters.insert("n1".to_string());
        assert!(!config.has_old_majority(&voters));

        voters.insert("n2".to_string());
        assert!(config.has_old_majority(&voters));
    }

    #[test]
    fn test_cluster_config_joint_consensus() {
        let mut config = create_test_cluster_config(vec!["n1", "n2", "n3"]);

        // Start joint consensus to add n4
        let mut new_members = config.members.clone();
        new_members.insert("n4".to_string());
        config.new_members = Some(new_members);

        assert!(config.is_joint());
        assert_eq!(config.voting_members().len(), 4);
        assert_eq!(config.new_majority(), Some(3));

        // Need majority in both configs
        let mut voters = HashSet::new();
        voters.insert("n1".to_string());
        voters.insert("n2".to_string());
        assert!(!config.has_quorum(&voters)); // Has old majority, but not new

        voters.insert("n4".to_string());
        assert!(config.has_quorum(&voters)); // Now has both
    }

    #[test]
    fn test_persistent_state_log_operations() {
        let mut state = PersistentState::new();

        assert_eq!(state.last_log_index(), 0);
        assert_eq!(state.last_log_term(), 0);

        // Add entries
        state.append_entries(vec![
            LogEntry::new(1, 1, Command::Noop),
            LogEntry::new(1, 2, Command::Noop),
            LogEntry::new(2, 3, Command::Noop),
        ]);

        assert_eq!(state.last_log_index(), 3);
        assert_eq!(state.last_log_term(), 2);

        // Get entry
        let entry = state.get_entry(2).unwrap();
        assert_eq!(entry.term, 1);
        assert_eq!(entry.index, 2);

        // Get term
        assert_eq!(state.get_term(3), Some(2));
        assert_eq!(state.get_term(4), None);

        // Truncate
        state.truncate_from(2);
        assert_eq!(state.last_log_index(), 1);
    }

    #[test]
    fn test_persistent_state_entries_range() {
        let mut state = PersistentState::new();

        for i in 1..=5 {
            state.log.push(LogEntry::new(1, i, Command::Noop));
        }

        let entries = state.get_entries(2, 4);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].index, 2);
        assert_eq!(entries[2].index, 4);

        // Empty range
        let empty = state.get_entries(0, 1);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_leader_state_initialization() {
        let mut peers = HashSet::new();
        peers.insert("n1".to_string());
        peers.insert("n2".to_string());

        let leader_state = LeaderState::new(&peers, 10);

        assert_eq!(leader_state.next_index.get("n1"), Some(&11));
        assert_eq!(leader_state.next_index.get("n2"), Some(&11));
        assert_eq!(leader_state.match_index.get("n1"), Some(&0));
        assert_eq!(leader_state.match_index.get("n2"), Some(&0));
    }

    #[test]
    fn test_leader_state_update_match() {
        let mut peers = HashSet::new();
        peers.insert("n1".to_string());

        let mut leader_state = LeaderState::new(&peers, 10);

        leader_state.update_match(&"n1".to_string(), 15);

        assert_eq!(leader_state.match_index.get("n1"), Some(&15));
        assert_eq!(leader_state.next_index.get("n1"), Some(&16));
    }

    #[test]
    fn test_leader_state_decrement_next() {
        let mut peers = HashSet::new();
        peers.insert("n1".to_string());

        let mut leader_state = LeaderState::new(&peers, 10);

        leader_state.decrement_next(&"n1".to_string(), None);
        assert_eq!(leader_state.next_index.get("n1"), Some(&10));

        leader_state.decrement_next(&"n1".to_string(), Some(5));
        assert_eq!(leader_state.next_index.get("n1"), Some(&5));
    }

    #[test]
    fn test_kv_state_machine() {
        let mut sm = KvStateMachine::default();

        // Set
        sm.apply(&Command::Set {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        });

        // Snapshot and restore
        let snapshot = sm.snapshot();

        let mut sm2 = KvStateMachine::default();
        sm2.restore(&snapshot).unwrap();

        // Delete
        sm.apply(&Command::Delete {
            key: b"key1".to_vec(),
        });

        // Verify original has key deleted
        let snapshot2 = sm.snapshot();
        assert_ne!(snapshot, snapshot2);
    }

    #[test]
    fn test_request_vote_request_serialization() {
        let request = RequestVoteRequest {
            term: 5,
            candidate_id: "node1".to_string(),
            last_log_index: 10,
            last_log_term: 3,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: RequestVoteRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.term, 5);
        assert_eq!(deserialized.candidate_id, "node1");
        assert_eq!(deserialized.last_log_index, 10);
        assert_eq!(deserialized.last_log_term, 3);
    }

    #[test]
    fn test_append_entries_request_serialization() {
        let request = AppendEntriesRequest {
            term: 3,
            leader_id: "leader".to_string(),
            prev_log_index: 5,
            prev_log_term: 2,
            entries: vec![LogEntry::new(3, 6, Command::Noop)],
            leader_commit: 4,
            energy_state: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: AppendEntriesRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.term, 3);
        assert_eq!(deserialized.entries.len(), 1);
    }

    #[tokio::test]
    async fn test_raft_node_creation() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2", "node3"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);

        assert_eq!(node.node_id(), "node1");
        assert_eq!(node.state().await, RaftState::Follower);
        assert_eq!(node.current_term().await, 0);
        assert!(node.leader_id().await.is_none());
    }

    #[tokio::test]
    async fn test_raft_node_become_follower() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        node.become_follower(5, Some("leader1".to_string())).await;

        assert_eq!(node.state().await, RaftState::Follower);
        assert_eq!(node.current_term().await, 5);
        assert_eq!(node.leader_id().await, Some("leader1".to_string()));
    }

    #[tokio::test]
    async fn test_raft_node_handle_request_vote_grant() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        let request = RequestVoteRequest {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = node.handle_request_vote(request).await;

        assert!(response.vote_granted);
        assert_eq!(response.term, 1);
    }

    #[tokio::test]
    async fn test_raft_node_handle_request_vote_reject_old_term() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;
        node.become_follower(5, None).await;

        let request = RequestVoteRequest {
            term: 3, // Lower than current term 5
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = node.handle_request_vote(request).await;

        assert!(!response.vote_granted);
        assert_eq!(response.term, 5);
    }

    #[tokio::test]
    async fn test_raft_node_handle_append_entries_success() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry::new(1, 1, Command::Noop)],
            leader_commit: 0,
            energy_state: None,
        };

        let response = node.handle_append_entries(request).await;

        assert!(response.success);
        assert_eq!(response.term, 1);
        assert_eq!(response.match_index, 1);
    }

    #[tokio::test]
    async fn test_raft_node_handle_append_entries_reject_old_term() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;
        node.become_follower(5, None).await;

        let request = AppendEntriesRequest {
            term: 3,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            energy_state: None,
        };

        let response = node.handle_append_entries(request).await;

        assert!(!response.success);
        assert_eq!(response.term, 5);
    }

    #[tokio::test]
    async fn test_raft_node_log_conflict_detection() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        // Add an entry first
        {
            let mut persistent = node.persistent.write().await;
            persistent.log.push(LogEntry::new(1, 1, Command::Noop));
        }

        // Try to append with wrong prev_log_term
        let request = AppendEntriesRequest {
            term: 2,
            leader_id: "node2".to_string(),
            prev_log_index: 1,
            prev_log_term: 2, // Wrong - entry has term 1
            entries: vec![LogEntry::new(2, 2, Command::Noop)],
            leader_commit: 0,
            energy_state: None,
        };

        let response = node.handle_append_entries(request).await;

        assert!(!response.success);
        assert!(response.conflict_term.is_some());
        assert!(response.conflict_index.is_some());
    }

    #[tokio::test]
    async fn test_raft_node_propose_not_leader() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        let result = node
            .propose(Command::Set {
                key: b"key".to_vec(),
                value: b"value".to_vec(),
            })
            .await;

        assert!(matches!(result, Err(RaftError::NotLeader(_))));
    }

    #[tokio::test]
    async fn test_snapshot_metadata() {
        let config = ClusterConfig::new(
            vec!["n1", "n2", "n3"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        let metadata = SnapshotMetadata {
            last_included_index: 100,
            last_included_term: 5,
            config,
            total_size: 1024,
        };

        assert_eq!(metadata.last_included_index, 100);
        assert_eq!(metadata.last_included_term, 5);
        assert_eq!(metadata.config.members.len(), 3);
    }

    #[test]
    fn test_raft_config_random_timeout() {
        let config = RaftConfig::default();

        // Generate multiple timeouts and verify they're in range
        for _ in 0..100 {
            let timeout = config.random_election_timeout();
            assert!(timeout >= config.election_timeout_min);
            assert!(timeout <= config.election_timeout_max);
        }
    }

    #[test]
    fn test_raft_state_display() {
        assert_eq!(format!("{}", RaftState::Follower), "Follower");
        assert_eq!(format!("{}", RaftState::Candidate), "Candidate");
        assert_eq!(format!("{}", RaftState::Leader), "Leader");
    }

    #[test]
    fn test_raft_error_display() {
        let err = RaftError::NotLeader(Some("leader1".to_string()));
        assert!(err.to_string().contains("leader1"));

        let err = RaftError::Timeout;
        assert!(err.to_string().contains("timed out"));

        let err = RaftError::InvalidConfig("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[tokio::test]
    async fn test_in_memory_transport() {
        let transport = InMemoryTransport::new();

        // Create a channel for a mock node
        let (tx, mut rx) = mpsc::channel(10);
        transport.register("node1".to_string(), tx).await;

        // Spawn a handler
        tokio::spawn(async move {
            while let Some((msg, response_tx)) = rx.recv().await {
                if let RaftMessage::RequestVote(req) = msg {
                    let response = RequestVoteResponse {
                        term: req.term,
                        vote_granted: true,
                    };
                    let _ = response_tx
                        .send(RaftMessage::RequestVoteResponse(response))
                        .await;
                }
            }
        });

        // Send a request
        let request = RequestVoteRequest {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = transport
            .send_request_vote(&"node1".to_string(), request)
            .await;
        assert!(response.is_ok());
        assert!(response.unwrap().vote_granted);
    }

    #[tokio::test]
    async fn test_handle_install_snapshot() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1", "node2"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        // Create a simple snapshot
        let snapshot_data = serde_json::to_vec(&HashMap::<Vec<u8>, Vec<u8>>::new()).unwrap();

        let request = InstallSnapshotRequest {
            term: 2,
            leader_id: "node2".to_string(),
            last_included_index: 10,
            last_included_term: 2,
            offset: 0,
            data: snapshot_data.clone(),
            done: true,
            config: ClusterConfig::new(
                vec!["node1", "node2"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            ),
        };

        let response = node.handle_install_snapshot(request).await;

        assert!(response.success);
        assert_eq!(node.commit_index().await, 10);
        assert_eq!(node.last_applied().await, 10);
    }

    #[tokio::test]
    async fn test_cluster_membership_change_errors() {
        let config = create_test_config("node1");
        let cluster = create_test_cluster_config(vec!["node1"]);
        let transport = Arc::new(InMemoryTransport::new());
        let sm = KvStateMachine::default();

        let node = RaftNode::new(config, sm, transport, cluster);
        node.start().await;

        // Not leader - should fail
        let result = node.add_node("node2".to_string()).await;
        assert!(matches!(result, Err(RaftError::NotLeader(_))));

        // Remove non-existent node
        let result = node.remove_node("node99".to_string()).await;
        assert!(matches!(result, Err(RaftError::NotLeader(_))));
    }
}
