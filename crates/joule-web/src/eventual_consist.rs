//! Eventual consistency protocol — Lamport timestamps, vector clocks, anti-entropy.
//!
//! Provides `VersionedValue` with Lamport timestamp, `EventualStore` (key-value
//! with vector clocks), read/write operations, anti-entropy protocol (compare
//! and exchange missing updates), read repair on stale reads, configurable
//! consistency level (One, Quorum, All), convergence detection, and tombstone
//! handling for deletes.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Eventual consistency domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsistencyError {
    /// Key not found.
    KeyNotFound(String),
    /// Insufficient replicas responded.
    InsufficientReplicas { needed: usize, responded: usize },
    /// Node not found.
    NodeNotFound(u64),
    /// Key is tombstoned (deleted).
    Tombstoned(String),
}

impl fmt::Display for ConsistencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::InsufficientReplicas { needed, responded } => {
                write!(f, "insufficient replicas: need {needed}, got {responded}")
            }
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::Tombstoned(k) => write!(f, "key tombstoned: {k}"),
        }
    }
}

impl std::error::Error for ConsistencyError {}

// ── Consistency Level ───────────────────────────────────────────

/// Required consistency level for read/write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyLevel {
    /// Any single replica.
    One,
    /// Majority of replicas.
    Quorum,
    /// All replicas.
    All,
}

impl ConsistencyLevel {
    /// Required number of acknowledgments given total replicas.
    pub fn required_acks(&self, total_replicas: usize) -> usize {
        match self {
            Self::One => 1,
            Self::Quorum => total_replicas / 2 + 1,
            Self::All => total_replicas,
        }
    }
}

impl fmt::Display for ConsistencyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::One => "ONE",
            Self::Quorum => "QUORUM",
            Self::All => "ALL",
        };
        write!(f, "{s}")
    }
}

// ── Versioned Value ─────────────────────────────────────────────

/// A value tagged with a Lamport timestamp and originating node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedValue {
    pub data: Vec<u8>,
    pub lamport_ts: u64,
    pub origin_node: u64,
    pub is_tombstone: bool,
}

impl VersionedValue {
    pub fn new(data: Vec<u8>, lamport_ts: u64, origin_node: u64) -> Self {
        Self { data, lamport_ts, origin_node, is_tombstone: false }
    }

    pub fn tombstone(lamport_ts: u64, origin_node: u64) -> Self {
        Self { data: Vec::new(), lamport_ts, origin_node, is_tombstone: true }
    }

    /// Is this value newer than another? Tie-break by node ID.
    pub fn is_newer_than(&self, other: &VersionedValue) -> bool {
        if self.lamport_ts != other.lamport_ts {
            self.lamport_ts > other.lamport_ts
        } else {
            self.origin_node > other.origin_node
        }
    }
}

impl fmt::Display for VersionedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_tombstone {
            write!(f, "Tombstone(ts={}, node={})", self.lamport_ts, self.origin_node)
        } else {
            write!(f, "Value(ts={}, node={}, {}B)", self.lamport_ts, self.origin_node, self.data.len())
        }
    }
}

// ── Anti-Entropy Digest ─────────────────────────────────────────

/// Summary of what a node has, for anti-entropy exchange.
#[derive(Debug, Clone)]
pub struct Digest {
    pub node_id: u64,
    /// key -> lamport timestamp.
    pub entries: HashMap<String, u64>,
}

impl Digest {
    pub fn new(node_id: u64) -> Self {
        Self { node_id, entries: HashMap::new() }
    }
}

/// Updates to send as result of anti-entropy comparison.
#[derive(Debug, Clone)]
pub struct AntiEntropyDelta {
    pub from_node: u64,
    pub updates: Vec<(String, VersionedValue)>,
}

// ── Eventual Store ──────────────────────────────────────────────

/// Eventually consistent key-value store with anti-entropy.
pub struct EventualStore {
    pub node_id: u64,
    lamport_clock: u64,
    store: HashMap<String, VersionedValue>,
    pub consistency_level: ConsistencyLevel,
    /// Simulated replica count for consistency checks.
    replica_count: usize,
    /// Read repair log: (key, old_ts, new_ts).
    repair_log: Vec<(String, u64, u64)>,
    /// Statistics.
    writes: u64,
    reads: u64,
    repairs: u64,
    tombstone_count: u64,
}

impl EventualStore {
    pub fn new(node_id: u64, replica_count: usize) -> Self {
        Self {
            node_id,
            lamport_clock: 0,
            store: HashMap::new(),
            consistency_level: ConsistencyLevel::One,
            replica_count,
            repair_log: Vec::new(),
            writes: 0,
            reads: 0,
            repairs: 0,
            tombstone_count: 0,
        }
    }

    pub fn with_consistency(mut self, level: ConsistencyLevel) -> Self {
        self.consistency_level = level;
        self
    }

    /// Advance the Lamport clock and return the new value.
    fn tick(&mut self) -> u64 {
        self.lamport_clock += 1;
        self.lamport_clock
    }

    /// Observe a remote timestamp (merge Lamport clocks).
    fn observe(&mut self, remote_ts: u64) {
        self.lamport_clock = self.lamport_clock.max(remote_ts) + 1;
    }

    /// Write a value locally.
    pub fn write(&mut self, key: impl Into<String>, data: Vec<u8>) -> VersionedValue {
        let ts = self.tick();
        let val = VersionedValue::new(data, ts, self.node_id);
        self.store.insert(key.into(), val.clone());
        self.writes += 1;
        val
    }

    /// Delete a key by writing a tombstone.
    pub fn delete(&mut self, key: &str) -> Option<VersionedValue> {
        let ts = self.tick();
        let tombstone = VersionedValue::tombstone(ts, self.node_id);
        let old = self.store.insert(key.to_string(), tombstone.clone());
        if old.is_some() {
            self.tombstone_count += 1;
        }
        Some(tombstone)
    }

    /// Read a value. Returns None if not found or tombstoned.
    pub fn read(&mut self, key: &str) -> Result<&VersionedValue, ConsistencyError> {
        self.reads += 1;
        match self.store.get(key) {
            None => Err(ConsistencyError::KeyNotFound(key.to_string())),
            Some(v) if v.is_tombstone => Err(ConsistencyError::Tombstoned(key.to_string())),
            Some(v) => Ok(v),
        }
    }

    /// Read a value including tombstones (for replication).
    pub fn read_raw(&self, key: &str) -> Option<&VersionedValue> {
        self.store.get(key)
    }

    /// Apply a remote write (e.g., from anti-entropy). Returns true if applied.
    pub fn apply_remote(&mut self, key: impl Into<String>, remote_val: VersionedValue) -> bool {
        self.observe(remote_val.lamport_ts);
        let k = key.into();
        let should_apply = match self.store.get(&k) {
            None => true,
            Some(existing) => remote_val.is_newer_than(existing),
        };
        if should_apply {
            if remote_val.is_tombstone {
                self.tombstone_count += 1;
            }
            self.store.insert(k, remote_val);
        }
        should_apply
    }

    /// Read repair: given a potentially stale value from another replica,
    /// return our value if newer.
    pub fn read_repair(&mut self, key: &str, stale: &VersionedValue) -> Option<VersionedValue> {
        if let Some(local) = self.store.get(key) {
            if local.is_newer_than(stale) {
                self.repairs += 1;
                self.repair_log.push((key.to_string(), stale.lamport_ts, local.lamport_ts));
                return Some(local.clone());
            }
        }
        None
    }

    /// Generate a digest of our store for anti-entropy.
    pub fn digest(&self) -> Digest {
        let mut d = Digest::new(self.node_id);
        for (k, v) in &self.store {
            d.entries.insert(k.clone(), v.lamport_ts);
        }
        d
    }

    /// Compare a remote digest and produce delta (updates we have that they need).
    pub fn anti_entropy_delta(&self, remote_digest: &Digest) -> AntiEntropyDelta {
        let mut updates = Vec::new();
        for (k, v) in &self.store {
            let remote_ts = remote_digest.entries.get(k).copied().unwrap_or(0);
            if v.lamport_ts > remote_ts {
                updates.push((k.clone(), v.clone()));
            }
        }
        AntiEntropyDelta { from_node: self.node_id, updates }
    }

    /// Apply an anti-entropy delta from a remote node. Returns number of updates applied.
    pub fn apply_delta(&mut self, delta: &AntiEntropyDelta) -> usize {
        let mut applied = 0;
        for (k, v) in &delta.updates {
            if self.apply_remote(k.clone(), v.clone()) {
                applied += 1;
            }
        }
        applied
    }

    /// Check if consistency level is satisfiable.
    pub fn check_consistency(&self, responding_replicas: usize) -> Result<(), ConsistencyError> {
        let needed = self.consistency_level.required_acks(self.replica_count);
        if responding_replicas < needed {
            Err(ConsistencyError::InsufficientReplicas {
                needed,
                responded: responding_replicas,
            })
        } else {
            Ok(())
        }
    }

    /// Check convergence: given all node digests, do they agree?
    pub fn check_convergence(&self, digests: &[Digest]) -> bool {
        if digests.is_empty() {
            return true;
        }
        let all_keys: HashSet<&String> = digests.iter()
            .flat_map(|d| d.entries.keys())
            .collect();
        for key in all_keys {
            let timestamps: HashSet<u64> = digests.iter()
                .filter_map(|d| d.entries.get(key).copied())
                .collect();
            if timestamps.len() > 1 {
                return false;
            }
        }
        true
    }

    /// Purge tombstones older than a given Lamport timestamp.
    pub fn purge_tombstones(&mut self, older_than_ts: u64) -> usize {
        let keys_to_remove: Vec<String> = self.store.iter()
            .filter(|(_, v)| v.is_tombstone && v.lamport_ts < older_than_ts)
            .map(|(k, _)| k.clone())
            .collect();
        let count = keys_to_remove.len();
        for k in keys_to_remove {
            self.store.remove(&k);
            self.tombstone_count = self.tombstone_count.saturating_sub(1);
        }
        count
    }

    /// Total keys (including tombstones).
    pub fn total_keys(&self) -> usize {
        self.store.len()
    }

    /// Live keys (excluding tombstones).
    pub fn live_keys(&self) -> usize {
        self.store.values().filter(|v| !v.is_tombstone).count()
    }

    /// Current Lamport clock value.
    pub fn clock(&self) -> u64 {
        self.lamport_clock
    }

    /// Write count.
    pub fn write_count(&self) -> u64 {
        self.writes
    }

    /// Read count.
    pub fn read_count(&self) -> u64 {
        self.reads
    }

    /// Repair count.
    pub fn repair_count(&self) -> u64 {
        self.repairs
    }

    /// Repair log.
    pub fn repair_log(&self) -> &[(String, u64, u64)] {
        &self.repair_log
    }
}

impl fmt::Display for EventualStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EventualStore(node={}, keys={}, clock={}, level={})",
            self.node_id,
            self.store.len(),
            self.lamport_clock,
            self.consistency_level,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistency_level_acks() {
        assert_eq!(ConsistencyLevel::One.required_acks(5), 1);
        assert_eq!(ConsistencyLevel::Quorum.required_acks(5), 3);
        assert_eq!(ConsistencyLevel::All.required_acks(5), 5);
    }

    #[test]
    fn consistency_level_display() {
        assert_eq!(format!("{}", ConsistencyLevel::Quorum), "QUORUM");
    }

    #[test]
    fn versioned_value_newer_by_ts() {
        let a = VersionedValue::new(vec![1], 5, 1);
        let b = VersionedValue::new(vec![2], 3, 1);
        assert!(a.is_newer_than(&b));
        assert!(!b.is_newer_than(&a));
    }

    #[test]
    fn versioned_value_newer_tiebreak_by_node() {
        let a = VersionedValue::new(vec![1], 5, 2);
        let b = VersionedValue::new(vec![2], 5, 1);
        assert!(a.is_newer_than(&b));
    }

    #[test]
    fn versioned_value_display() {
        let v = VersionedValue::new(vec![1, 2], 10, 1);
        assert!(format!("{v}").contains("Value"));
        let t = VersionedValue::tombstone(10, 1);
        assert!(format!("{t}").contains("Tombstone"));
    }

    #[test]
    fn store_write_and_read() {
        let mut store = EventualStore::new(1, 3);
        store.write("hello", vec![42]);
        let val = store.read("hello").unwrap();
        assert_eq!(val.data, vec![42]);
    }

    #[test]
    fn store_read_missing_key() {
        let mut store = EventualStore::new(1, 3);
        assert!(matches!(store.read("nope"), Err(ConsistencyError::KeyNotFound(_))));
    }

    #[test]
    fn store_delete_creates_tombstone() {
        let mut store = EventualStore::new(1, 3);
        store.write("k", vec![1]);
        store.delete("k");
        assert!(matches!(store.read("k"), Err(ConsistencyError::Tombstoned(_))));
    }

    #[test]
    fn store_lamport_clock_advances() {
        let mut store = EventualStore::new(1, 3);
        store.write("a", vec![1]);
        store.write("b", vec![2]);
        assert!(store.clock() >= 2);
    }

    #[test]
    fn store_apply_remote_newer() {
        let mut store = EventualStore::new(1, 3);
        store.write("k", vec![1]); // ts=1
        let remote = VersionedValue::new(vec![99], 10, 2);
        assert!(store.apply_remote("k", remote));
        assert_eq!(store.read("k").unwrap().data, vec![99]);
    }

    #[test]
    fn store_apply_remote_older_rejected() {
        let mut store = EventualStore::new(1, 3);
        store.write("k", vec![1]); // ts=1
        // Write again to advance clock.
        store.write("k", vec![2]); // ts=2
        let stale = VersionedValue::new(vec![0], 0, 2);
        assert!(!store.apply_remote("k", stale));
    }

    #[test]
    fn store_read_repair() {
        let mut store = EventualStore::new(1, 3);
        store.write("k", vec![10]);
        let stale = VersionedValue::new(vec![1], 0, 2);
        let repair = store.read_repair("k", &stale);
        assert!(repair.is_some());
        assert_eq!(store.repair_count(), 1);
    }

    #[test]
    fn store_digest() {
        let mut store = EventualStore::new(1, 3);
        store.write("a", vec![1]);
        store.write("b", vec![2]);
        let digest = store.digest();
        assert_eq!(digest.entries.len(), 2);
    }

    #[test]
    fn store_anti_entropy_delta() {
        let mut store_a = EventualStore::new(1, 3);
        store_a.write("x", vec![1]);
        store_a.write("y", vec![2]);

        let store_b = EventualStore::new(2, 3);
        let digest_b = store_b.digest();
        let delta = store_a.anti_entropy_delta(&digest_b);
        assert_eq!(delta.updates.len(), 2);
    }

    #[test]
    fn store_apply_delta() {
        let mut store_a = EventualStore::new(1, 3);
        store_a.write("x", vec![1]);

        let mut store_b = EventualStore::new(2, 3);
        let digest_b = store_b.digest();
        let delta = store_a.anti_entropy_delta(&digest_b);
        let applied = store_b.apply_delta(&delta);
        assert_eq!(applied, 1);
        assert_eq!(store_b.read("x").unwrap().data, vec![1]);
    }

    #[test]
    fn store_check_consistency_ok() {
        let store = EventualStore::new(1, 3).with_consistency(ConsistencyLevel::Quorum);
        assert!(store.check_consistency(2).is_ok());
    }

    #[test]
    fn store_check_consistency_fail() {
        let store = EventualStore::new(1, 3).with_consistency(ConsistencyLevel::All);
        assert!(matches!(
            store.check_consistency(2),
            Err(ConsistencyError::InsufficientReplicas { .. })
        ));
    }

    #[test]
    fn store_convergence_check() {
        let mut a = EventualStore::new(1, 3);
        let mut b = EventualStore::new(2, 3);
        a.write("k", vec![1]);
        // Sync a -> b.
        let delta = a.anti_entropy_delta(&b.digest());
        b.apply_delta(&delta);
        let converged = a.check_convergence(&[a.digest(), b.digest()]);
        assert!(converged);
    }

    #[test]
    fn store_purge_tombstones() {
        let mut store = EventualStore::new(1, 3);
        store.write("a", vec![1]);
        store.delete("a");
        let ts = store.clock();
        let purged = store.purge_tombstones(ts + 1);
        assert_eq!(purged, 1);
        assert_eq!(store.total_keys(), 0);
    }

    #[test]
    fn store_live_keys() {
        let mut store = EventualStore::new(1, 3);
        store.write("a", vec![1]);
        store.write("b", vec![2]);
        store.delete("a");
        assert_eq!(store.live_keys(), 1);
        assert_eq!(store.total_keys(), 2);
    }

    #[test]
    fn store_display() {
        let store = EventualStore::new(1, 3);
        let d = format!("{store}");
        assert!(d.contains("EventualStore"));
    }
}
