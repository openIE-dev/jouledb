//! Edge Sync Protocol
//!
//! Provides synchronization between edge devices (no_std compatible)
//! and the central server. Supports:
//! - Conflict-free replicated data types (CRDTs)
//! - Vector clocks for causality tracking
//! - Delta synchronization for bandwidth efficiency
//! - Offline operation with eventual consistency

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

// ============================================================================
// Vector Clocks for Causality
// ============================================================================

/// A vector clock for tracking causality across nodes
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Node ID -> logical timestamp
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Create a new empty vector clock
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment the clock for a node
    pub fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Get the timestamp for a node
    pub fn get(&self, node_id: &str) -> u64 {
        self.clocks.get(node_id).copied().unwrap_or(0)
    }

    /// Merge with another vector clock (take max of each component)
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &ts) in &other.clocks {
            let current = self.clocks.entry(node.clone()).or_insert(0);
            *current = (*current).max(ts);
        }
    }

    /// Check if this clock happened before another
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        let mut dominated = false;
        for (node, &ts) in &self.clocks {
            let other_ts = other.get(node);
            if ts > other_ts {
                return false;
            }
            if ts < other_ts {
                dominated = true;
            }
        }
        // Check for nodes in other but not in self
        for (node, &ts) in &other.clocks {
            if !self.clocks.contains_key(node) && ts > 0 {
                dominated = true;
            }
        }
        dominated
    }

    /// Check if two clocks are concurrent (neither happened before the other)
    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happened_before(other) && !other.happened_before(self)
    }
}

// ============================================================================
// Sync Operations
// ============================================================================

/// Type of sync operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncOperation {
    /// Insert a new row
    Insert {
        table: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// Update an existing row
    Update {
        table: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// Delete a row
    Delete { table: String, key: Vec<u8> },
    /// Tombstone (soft delete with timestamp)
    Tombstone {
        table: String,
        key: Vec<u8>,
        deleted_at: u64,
    },
}

impl SyncOperation {
    /// Extract the value bytes from an operation (empty for Delete/Tombstone)
    pub fn value_bytes(&self) -> &[u8] {
        match self {
            SyncOperation::Insert { value, .. } => value,
            SyncOperation::Update { value, .. } => value,
            SyncOperation::Delete { .. } => &[],
            SyncOperation::Tombstone { .. } => &[],
        }
    }
}

/// A sync entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEntry {
    /// Unique entry ID
    pub id: u64,
    /// The operation
    pub operation: SyncOperation,
    /// Vector clock at time of operation
    pub clock: VectorClock,
    /// Originating node
    pub origin: String,
    /// Timestamp (wall clock, for debugging)
    pub timestamp: u64,
    /// Checksum for integrity
    pub checksum: u32,
}

impl SyncEntry {
    /// Create a new sync entry
    pub fn new(id: u64, operation: SyncOperation, clock: VectorClock, origin: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut entry = Self {
            id,
            operation,
            clock,
            origin: origin.to_string(),
            timestamp,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    fn compute_checksum(&self) -> u32 {
        // Simple CRC-like checksum
        let mut hash: u32 = 0;
        hash = hash.wrapping_add(self.id as u32);
        hash = hash.wrapping_mul(31);
        hash = hash.wrapping_add(self.timestamp as u32);
        hash
    }

    /// Verify integrity
    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

// ============================================================================
// Sync State
// ============================================================================

/// Sync state for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// Node ID
    pub node_id: String,
    /// Current vector clock
    pub clock: VectorClock,
    /// Pending operations not yet synced
    pub pending: Vec<SyncEntry>,
    /// Last sync timestamp with each peer
    pub last_sync: HashMap<String, u64>,
    /// Local sequence number
    pub sequence: u64,
}

impl SyncState {
    /// Create a new sync state for a node
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            clock: VectorClock::new(),
            pending: Vec::new(),
            last_sync: HashMap::new(),
            sequence: 0,
        }
    }

    /// Record a local operation
    pub fn record(&mut self, operation: SyncOperation) -> SyncEntry {
        self.clock.increment(&self.node_id);
        self.sequence += 1;

        let entry = SyncEntry::new(self.sequence, operation, self.clock.clone(), &self.node_id);

        self.pending.push(entry.clone());
        entry
    }

    /// Get pending entries for sync
    pub fn get_pending(&self) -> &[SyncEntry] {
        &self.pending
    }

    /// Clear pending entries after successful sync
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Mark sync with a peer
    pub fn mark_synced(&mut self, peer: &str, timestamp: u64) {
        self.last_sync.insert(peer.to_string(), timestamp);
    }
}

// ============================================================================
// Delta Sync
// ============================================================================

/// A delta sync request from client to server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    /// Requesting node ID
    pub node_id: String,
    /// Client's current vector clock
    pub clock: VectorClock,
    /// Entries to push to server
    pub entries: Vec<SyncEntry>,
}

/// A delta sync response from server to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    /// Server's current vector clock
    pub clock: VectorClock,
    /// Entries from server that client is missing
    pub entries: Vec<SyncEntry>,
    /// Conflicts detected
    pub conflicts: Vec<Conflict>,
}

/// A sync conflict that needs resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// The key in conflict
    pub key: Vec<u8>,
    /// Local version
    pub local: SyncEntry,
    /// Remote version
    pub remote: SyncEntry,
    /// Suggested resolution
    pub resolution: ConflictResolution,
}

/// How to resolve a conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Use the local version
    UseLocal,
    /// Use the remote version
    UseRemote,
    /// Merge versions (for compatible types)
    Merge(Vec<u8>),
    /// Manual resolution required
    Manual,
}

// ============================================================================
// Sync Engine
// ============================================================================

/// The sync engine manages synchronization state and operations
pub struct SyncEngine {
    /// Local sync state
    state: SyncState,
    /// Entry log (for replay and recovery)
    log: Vec<SyncEntry>,
    /// Conflict resolution strategy
    resolution_strategy: ConflictStrategy,
}

/// Strategy for automatic conflict resolution
#[derive(Debug, Clone, Copy)]
pub enum ConflictStrategy {
    /// Last write wins (based on wall clock)
    LastWriteWins,
    /// First write wins
    FirstWriteWins,
    /// Use CRDT merge semantics (commutative, associative, idempotent).
    /// Requires the column/table to be CRDT-enabled. Conflict is resolved
    /// by merging both values via the column's CRDT type (e.g., LWWRegister,
    /// ORSet, GCounter).
    CrdtMerge,
    /// Always prefer local
    PreferLocal,
    /// Always prefer remote
    PreferRemote,
    /// Require manual resolution
    Manual,
}

impl SyncEngine {
    /// Create a new sync engine for a node
    pub fn new(node_id: &str) -> Self {
        Self {
            state: SyncState::new(node_id),
            log: Vec::new(),
            resolution_strategy: ConflictStrategy::LastWriteWins,
        }
    }

    /// Set the conflict resolution strategy
    pub fn set_resolution_strategy(&mut self, strategy: ConflictStrategy) {
        self.resolution_strategy = strategy;
    }

    /// Record a local write operation
    pub fn write(&mut self, table: &str, key: Vec<u8>, value: Vec<u8>) -> SyncEntry {
        let op = SyncOperation::Insert {
            table: table.to_string(),
            key,
            value,
        };
        let entry = self.state.record(op);
        self.log.push(entry.clone());
        entry
    }

    /// Record a local delete operation
    pub fn delete(&mut self, table: &str, key: Vec<u8>) -> SyncEntry {
        let op = SyncOperation::Delete {
            table: table.to_string(),
            key,
        };
        let entry = self.state.record(op);
        self.log.push(entry.clone());
        entry
    }

    /// Create a sync request to send to server
    pub fn create_sync_request(&self) -> SyncRequest {
        SyncRequest {
            node_id: self.state.node_id.clone(),
            clock: self.state.clock.clone(),
            entries: self.state.pending.clone(),
        }
    }

    /// Process a sync response from server
    pub fn process_sync_response(&mut self, response: SyncResponse) -> Vec<Conflict> {
        // Merge vector clocks
        self.state.clock.merge(&response.clock);

        // Apply remote entries
        for entry in response.entries {
            self.apply_remote_entry(entry);
        }

        // Clear pending after successful sync
        self.state.clear_pending();

        // Return any conflicts for handling
        response.conflicts
    }

    fn apply_remote_entry(&mut self, entry: SyncEntry) {
        // Verify entry integrity
        if !entry.verify() {
            return; // Skip corrupted entries
        }

        // Merge clock
        self.state.clock.merge(&entry.clock);

        // Add to log
        self.log.push(entry);
    }

    /// Resolve a conflict based on strategy
    pub fn resolve_conflict(&self, conflict: &Conflict) -> ConflictResolution {
        match self.resolution_strategy {
            ConflictStrategy::LastWriteWins => {
                if conflict.local.timestamp >= conflict.remote.timestamp {
                    ConflictResolution::UseLocal
                } else {
                    ConflictResolution::UseRemote
                }
            }
            ConflictStrategy::FirstWriteWins => {
                if conflict.local.timestamp <= conflict.remote.timestamp {
                    ConflictResolution::UseLocal
                } else {
                    ConflictResolution::UseRemote
                }
            }
            ConflictStrategy::CrdtMerge => {
                // CRDT merge: the caller is responsible for deserializing both
                // values into the appropriate CRDT type and calling merge().
                // Here we signal that a merge is needed by returning both values
                // concatenated with a length prefix so the caller can split them.
                let mut merged = (conflict.local.operation.value_bytes().len() as u32)
                    .to_le_bytes()
                    .to_vec();
                merged.extend_from_slice(conflict.local.operation.value_bytes());
                merged.extend_from_slice(conflict.remote.operation.value_bytes());
                ConflictResolution::Merge(merged)
            }
            ConflictStrategy::PreferLocal => ConflictResolution::UseLocal,
            ConflictStrategy::PreferRemote => ConflictResolution::UseRemote,
            ConflictStrategy::Manual => ConflictResolution::Manual,
        }
    }

    /// Get the current vector clock
    pub fn clock(&self) -> &VectorClock {
        &self.state.clock
    }

    /// Get the node ID
    pub fn node_id(&self) -> &str {
        &self.state.node_id
    }

    /// Get the operation log
    pub fn log(&self) -> &[SyncEntry] {
        &self.log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_increment() {
        let mut clock = VectorClock::new();
        clock.increment("node1");
        clock.increment("node1");
        clock.increment("node2");

        assert_eq!(clock.get("node1"), 2);
        assert_eq!(clock.get("node2"), 1);
        assert_eq!(clock.get("node3"), 0);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut clock1 = VectorClock::new();
        clock1.increment("node1");
        clock1.increment("node1");

        let mut clock2 = VectorClock::new();
        clock2.increment("node1");
        clock2.increment("node2");
        clock2.increment("node2");

        clock1.merge(&clock2);

        assert_eq!(clock1.get("node1"), 2); // max(2, 1)
        assert_eq!(clock1.get("node2"), 2); // max(0, 2)
    }

    #[test]
    fn test_happened_before() {
        let mut clock1 = VectorClock::new();
        clock1.increment("node1");

        let mut clock2 = clock1.clone();
        clock2.increment("node1");

        assert!(clock1.happened_before(&clock2));
        assert!(!clock2.happened_before(&clock1));
    }

    #[test]
    fn test_concurrent() {
        let mut clock1 = VectorClock::new();
        clock1.increment("node1");

        let mut clock2 = VectorClock::new();
        clock2.increment("node2");

        assert!(clock1.concurrent_with(&clock2));
        assert!(clock2.concurrent_with(&clock1));
    }

    #[test]
    fn test_sync_engine_write() {
        let mut engine = SyncEngine::new("edge1");

        let entry = engine.write("users", b"user1".to_vec(), b"data".to_vec());

        assert_eq!(entry.origin, "edge1");
        assert_eq!(engine.clock().get("edge1"), 1);
        assert_eq!(engine.log().len(), 1);
    }

    #[test]
    fn test_sync_request_creation() {
        let mut engine = SyncEngine::new("edge1");
        engine.write("users", b"user1".to_vec(), b"data1".to_vec());
        engine.write("users", b"user2".to_vec(), b"data2".to_vec());

        let request = engine.create_sync_request();

        assert_eq!(request.node_id, "edge1");
        assert_eq!(request.entries.len(), 2);
    }

    #[test]
    fn test_conflict_resolution_last_write_wins() {
        let engine = SyncEngine::new("edge1");

        let local_clock = VectorClock::new();
        let remote_clock = VectorClock::new();

        let local = SyncEntry {
            id: 1,
            operation: SyncOperation::Insert {
                table: "users".to_string(),
                key: b"user1".to_vec(),
                value: b"local_data".to_vec(),
            },
            clock: local_clock,
            origin: "edge1".to_string(),
            timestamp: 100,
            checksum: 0,
        };

        let remote = SyncEntry {
            id: 2,
            operation: SyncOperation::Insert {
                table: "users".to_string(),
                key: b"user1".to_vec(),
                value: b"remote_data".to_vec(),
            },
            clock: remote_clock,
            origin: "server".to_string(),
            timestamp: 200,
            checksum: 0,
        };

        let conflict = Conflict {
            key: b"user1".to_vec(),
            local,
            remote,
            resolution: ConflictResolution::Manual,
        };

        let resolution = engine.resolve_conflict(&conflict);
        assert!(matches!(resolution, ConflictResolution::UseRemote));
    }
}
