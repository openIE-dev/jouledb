//! Snapshot store — save/load snapshots at version, snapshot frequency policy,
//! snapshot + events replay, snapshot pruning, snapshot metadata, and simple
//! snapshot compression.
//!
//! Replaces JS snapshot stores (EventStoreDB snapshots, Marten snapshots) with
//! a pure-Rust snapshot store for aggregate state persistence.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Snapshot store errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotStoreError {
    /// Snapshot not found.
    NotFound { aggregate_id: String },
    /// Snapshot not found at specific version.
    VersionNotFound { aggregate_id: String, version: u64 },
    /// Compression error.
    CompressionError(String),
    /// Decompression error.
    DecompressionError(String),
    /// Invalid snapshot data.
    InvalidData(String),
    /// Aggregate has no snapshots.
    NoSnapshots(String),
}

impl std::fmt::Display for SnapshotStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { aggregate_id } => {
                write!(f, "snapshot not found for aggregate: {aggregate_id}")
            }
            Self::VersionNotFound { aggregate_id, version } => {
                write!(f, "snapshot at version {version} not found for {aggregate_id}")
            }
            Self::CompressionError(msg) => write!(f, "compression error: {msg}"),
            Self::DecompressionError(msg) => write!(f, "decompression error: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid snapshot data: {msg}"),
            Self::NoSnapshots(id) => write!(f, "no snapshots for: {id}"),
        }
    }
}

impl std::error::Error for SnapshotStoreError {}

// ── Snapshot Frequency Policy ───────────────────────────────────

/// Policy that determines when to take a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotPolicy {
    /// Take a snapshot every N events.
    EveryNEvents(u64),
    /// Take a snapshot every N versions (equivalent to every N events).
    EveryNVersions(u64),
    /// Never take snapshots automatically.
    Never,
    /// Always take a snapshot after each command.
    Always,
}

impl SnapshotPolicy {
    /// Check if a snapshot should be taken at this version given the last snapshot version.
    pub fn should_snapshot(&self, current_version: u64, last_snapshot_version: u64) -> bool {
        match self {
            Self::EveryNEvents(n) | Self::EveryNVersions(n) => {
                if *n == 0 {
                    return false;
                }
                let events_since = current_version.saturating_sub(last_snapshot_version);
                events_since >= *n
            }
            Self::Never => false,
            Self::Always => true,
        }
    }
}

// ── Snapshot Metadata ───────────────────────────────────────────

/// Metadata attached to a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub aggregate_type: String,
    pub created_by: Option<String>,
    pub compressed: bool,
    pub original_size: usize,
    pub stored_size: usize,
    pub custom: HashMap<String, String>,
}

impl SnapshotMetadata {
    pub fn new(aggregate_type: impl Into<String>) -> Self {
        Self {
            aggregate_type: aggregate_type.into(),
            created_by: None,
            compressed: false,
            original_size: 0,
            stored_size: 0,
            custom: HashMap::new(),
        }
    }

    pub fn with_created_by(mut self, creator: impl Into<String>) -> Self {
        self.created_by = Some(creator.into());
        self
    }
}

// ── Snapshot ────────────────────────────────────────────────────

/// A stored snapshot of aggregate state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub aggregate_id: String,
    pub version: u64,
    pub state_data: Vec<u8>,
    pub metadata: SnapshotMetadata,
    pub taken_at: DateTime<Utc>,
}

impl Snapshot {
    /// Deserialize the state data into a HashMap.
    pub fn deserialize_state(&self) -> Result<HashMap<String, serde_json::Value>, SnapshotStoreError> {
        let data = if self.metadata.compressed {
            simple_decompress(&self.state_data)?
        } else {
            self.state_data.clone()
        };
        serde_json::from_slice(&data).map_err(|e| SnapshotStoreError::InvalidData(e.to_string()))
    }
}

// ── Simple Compression ──────────────────────────────────────────

/// Simple RLE-style compression for snapshot data.
fn simple_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut count: u8 = 1;
        while i + (count as usize) < data.len()
            && data[i + (count as usize)] == byte
            && count < 255
        {
            count += 1;
        }
        output.push(count);
        output.push(byte);
        i += count as usize;
    }

    output
}

/// Decompress RLE data.
fn simple_decompress(data: &[u8]) -> Result<Vec<u8>, SnapshotStoreError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if data.len() % 2 != 0 {
        return Err(SnapshotStoreError::DecompressionError(
            "compressed data has odd length".to_string(),
        ));
    }

    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let count = data[i] as usize;
        let byte = data[i + 1];
        for _ in 0..count {
            output.push(byte);
        }
        i += 2;
    }

    Ok(output)
}

// ── Replay Event ────────────────────────────────────────────────

/// An event used during snapshot + events replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub event_type: String,
    pub version: u64,
    pub data: HashMap<String, String>,
}

// ── Snapshot Store ──────────────────────────────────────────────

/// Store for aggregate snapshots with frequency policy and pruning.
#[derive(Debug)]
pub struct SnapshotStore {
    /// Snapshots indexed by aggregate_id, ordered by version.
    snapshots: HashMap<String, Vec<Snapshot>>,
    /// Default snapshot policy.
    policy: SnapshotPolicy,
    /// Whether to compress snapshots.
    compression_enabled: bool,
    /// Maximum snapshots to keep per aggregate (0 = unlimited).
    max_snapshots_per_aggregate: usize,
}

impl SnapshotStore {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            policy: SnapshotPolicy::EveryNEvents(10),
            compression_enabled: false,
            max_snapshots_per_aggregate: 0,
        }
    }

    /// Set the snapshot frequency policy.
    pub fn with_policy(mut self, policy: SnapshotPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Enable or disable compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Set maximum snapshots per aggregate (0 = unlimited).
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots_per_aggregate = max;
        self
    }

    /// Save a snapshot.
    pub fn save(
        &mut self,
        aggregate_id: &str,
        version: u64,
        state: &HashMap<String, serde_json::Value>,
        metadata: SnapshotMetadata,
    ) -> Result<Snapshot, SnapshotStoreError> {
        let json = serde_json::to_vec(state)
            .map_err(|e| SnapshotStoreError::CompressionError(e.to_string()))?;

        let original_size = json.len();

        let (state_data, compressed, stored_size) = if self.compression_enabled {
            let compressed_data = simple_compress(&json);
            let sz = compressed_data.len();
            (compressed_data, true, sz)
        } else {
            let sz = json.len();
            (json, false, sz)
        };

        let mut meta = metadata;
        meta.compressed = compressed;
        meta.original_size = original_size;
        meta.stored_size = stored_size;

        let snapshot = Snapshot {
            aggregate_id: aggregate_id.to_string(),
            version,
            state_data,
            metadata: meta,
            taken_at: Utc::now(),
        };

        let snapshots = self
            .snapshots
            .entry(aggregate_id.to_string())
            .or_default();
        snapshots.push(snapshot.clone());
        snapshots.sort_by_key(|s| s.version);

        // Prune if needed.
        if self.max_snapshots_per_aggregate > 0
            && snapshots.len() > self.max_snapshots_per_aggregate
        {
            let excess = snapshots.len() - self.max_snapshots_per_aggregate;
            snapshots.drain(..excess);
        }

        Ok(snapshot)
    }

    /// Load the latest snapshot for an aggregate.
    pub fn load_latest(
        &self,
        aggregate_id: &str,
    ) -> Result<Snapshot, SnapshotStoreError> {
        let snapshots = self
            .snapshots
            .get(aggregate_id)
            .ok_or_else(|| SnapshotStoreError::NotFound {
                aggregate_id: aggregate_id.to_string(),
            })?;

        snapshots
            .last()
            .cloned()
            .ok_or_else(|| SnapshotStoreError::NoSnapshots(aggregate_id.to_string()))
    }

    /// Load a snapshot at a specific version.
    pub fn load_at_version(
        &self,
        aggregate_id: &str,
        version: u64,
    ) -> Result<Snapshot, SnapshotStoreError> {
        let snapshots = self
            .snapshots
            .get(aggregate_id)
            .ok_or_else(|| SnapshotStoreError::NotFound {
                aggregate_id: aggregate_id.to_string(),
            })?;

        snapshots
            .iter()
            .find(|s| s.version == version)
            .cloned()
            .ok_or(SnapshotStoreError::VersionNotFound {
                aggregate_id: aggregate_id.to_string(),
                version,
            })
    }

    /// Load the latest snapshot at or before a given version.
    pub fn load_at_or_before(
        &self,
        aggregate_id: &str,
        version: u64,
    ) -> Result<Snapshot, SnapshotStoreError> {
        let snapshots = self
            .snapshots
            .get(aggregate_id)
            .ok_or_else(|| SnapshotStoreError::NotFound {
                aggregate_id: aggregate_id.to_string(),
            })?;

        snapshots
            .iter()
            .rev()
            .find(|s| s.version <= version)
            .cloned()
            .ok_or(SnapshotStoreError::VersionNotFound {
                aggregate_id: aggregate_id.to_string(),
                version,
            })
    }

    /// Check if a snapshot should be taken based on the policy.
    pub fn should_snapshot(&self, current_version: u64, aggregate_id: &str) -> bool {
        let last_version = self
            .snapshots
            .get(aggregate_id)
            .and_then(|snaps| snaps.last())
            .map(|s| s.version)
            .unwrap_or(0);
        self.policy.should_snapshot(current_version, last_version)
    }

    /// Prune snapshots for an aggregate, keeping only the N most recent.
    pub fn prune(&mut self, aggregate_id: &str, keep: usize) -> usize {
        if let Some(snapshots) = self.snapshots.get_mut(aggregate_id) {
            if snapshots.len() > keep {
                let removed = snapshots.len() - keep;
                snapshots.drain(..removed);
                return removed;
            }
        }
        0
    }

    /// Delete all snapshots for an aggregate.
    pub fn delete_all(&mut self, aggregate_id: &str) -> usize {
        self.snapshots
            .remove(aggregate_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Count snapshots for an aggregate.
    pub fn count(&self, aggregate_id: &str) -> usize {
        self.snapshots
            .get(aggregate_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Total snapshots across all aggregates.
    pub fn total_count(&self) -> usize {
        self.snapshots.values().map(|v| v.len()).sum()
    }

    /// List all aggregate IDs that have snapshots.
    pub fn aggregate_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.snapshots.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get snapshot versions for an aggregate.
    pub fn versions(&self, aggregate_id: &str) -> Vec<u64> {
        self.snapshots
            .get(aggregate_id)
            .map(|snaps| snaps.iter().map(|s| s.version).collect())
            .unwrap_or_default()
    }

    /// Get the current policy.
    pub fn policy(&self) -> SnapshotPolicy {
        self.policy
    }

    /// Replay: load snapshot + apply events on top.
    pub fn replay(
        &self,
        aggregate_id: &str,
        events_after_snapshot: &[ReplayEvent],
        apply_fn: fn(&mut HashMap<String, serde_json::Value>, &ReplayEvent),
    ) -> Result<(HashMap<String, serde_json::Value>, u64), SnapshotStoreError> {
        let snapshot = self.load_latest(aggregate_id)?;
        let mut state = snapshot.deserialize_state()?;
        let mut version = snapshot.version;

        for event in events_after_snapshot {
            if event.version > snapshot.version {
                apply_fn(&mut state, event);
                version = event.version;
            }
        }

        Ok((state, version))
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(name: &str, balance: i64) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert("name".to_string(), serde_json::json!(name));
        state.insert("balance".to_string(), serde_json::json!(balance));
        state
    }

    fn test_metadata() -> SnapshotMetadata {
        SnapshotMetadata::new("Account")
    }

    #[test]
    fn test_save_and_load_latest() {
        let mut store = SnapshotStore::new();
        let state = test_state("alice", 100);
        store.save("agg-1", 5, &state, test_metadata()).unwrap();

        let snap = store.load_latest("agg-1").unwrap();
        assert_eq!(snap.aggregate_id, "agg-1");
        assert_eq!(snap.version, 5);

        let loaded = snap.deserialize_state().unwrap();
        assert_eq!(loaded.get("name").unwrap(), &serde_json::json!("alice"));
        assert_eq!(loaded.get("balance").unwrap(), &serde_json::json!(100));
    }

    #[test]
    fn test_load_at_version() {
        let mut store = SnapshotStore::new();
        store.save("a1", 5, &test_state("v5", 50), test_metadata()).unwrap();
        store.save("a1", 10, &test_state("v10", 100), test_metadata()).unwrap();
        store.save("a1", 15, &test_state("v15", 150), test_metadata()).unwrap();

        let snap = store.load_at_version("a1", 10).unwrap();
        assert_eq!(snap.version, 10);
        let state = snap.deserialize_state().unwrap();
        assert_eq!(state.get("balance").unwrap(), &serde_json::json!(100));
    }

    #[test]
    fn test_load_at_version_not_found() {
        let mut store = SnapshotStore::new();
        store.save("a1", 5, &test_state("v5", 50), test_metadata()).unwrap();
        let err = store.load_at_version("a1", 99).unwrap_err();
        assert!(matches!(err, SnapshotStoreError::VersionNotFound { .. }));
    }

    #[test]
    fn test_load_at_or_before() {
        let mut store = SnapshotStore::new();
        store.save("a1", 5, &test_state("v5", 50), test_metadata()).unwrap();
        store.save("a1", 10, &test_state("v10", 100), test_metadata()).unwrap();

        let snap = store.load_at_or_before("a1", 8).unwrap();
        assert_eq!(snap.version, 5);

        let snap = store.load_at_or_before("a1", 15).unwrap();
        assert_eq!(snap.version, 10);
    }

    #[test]
    fn test_load_not_found() {
        let store = SnapshotStore::new();
        let err = store.load_latest("missing").unwrap_err();
        assert!(matches!(err, SnapshotStoreError::NotFound { .. }));
    }

    #[test]
    fn test_snapshot_policy_every_n() {
        let policy = SnapshotPolicy::EveryNEvents(5);
        assert!(!policy.should_snapshot(3, 0)); // 3 events since 0.
        assert!(policy.should_snapshot(5, 0));  // 5 events since 0.
        assert!(!policy.should_snapshot(7, 5)); // 2 events since 5.
        assert!(policy.should_snapshot(10, 5)); // 5 events since 5.
    }

    #[test]
    fn test_snapshot_policy_never() {
        let policy = SnapshotPolicy::Never;
        assert!(!policy.should_snapshot(100, 0));
    }

    #[test]
    fn test_snapshot_policy_always() {
        let policy = SnapshotPolicy::Always;
        assert!(policy.should_snapshot(1, 0));
        assert!(policy.should_snapshot(1, 1));
    }

    #[test]
    fn test_snapshot_policy_zero_interval() {
        let policy = SnapshotPolicy::EveryNEvents(0);
        assert!(!policy.should_snapshot(100, 0));
    }

    #[test]
    fn test_should_snapshot_integration() {
        let mut store = SnapshotStore::new().with_policy(SnapshotPolicy::EveryNEvents(3));

        assert!(store.should_snapshot(3, "a1")); // No snapshot yet, 3 events since 0.
        store.save("a1", 3, &test_state("v3", 30), test_metadata()).unwrap();
        assert!(!store.should_snapshot(5, "a1")); // 2 events since snapshot at 3.
        assert!(store.should_snapshot(6, "a1"));  // 3 events since snapshot at 3.
    }

    #[test]
    fn test_compression() {
        let mut store = SnapshotStore::new().with_compression(true);
        let state = test_state("alice", 100);
        let snap = store.save("a1", 1, &state, test_metadata()).unwrap();

        assert!(snap.metadata.compressed);
        assert!(snap.metadata.stored_size > 0);
        assert!(snap.metadata.original_size > 0);

        let loaded = snap.deserialize_state().unwrap();
        assert_eq!(loaded.get("name").unwrap(), &serde_json::json!("alice"));
    }

    #[test]
    fn test_compression_round_trip() {
        let original = b"aaabbbcccddd";
        let compressed = simple_compress(original);
        let decompressed = simple_decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compression_empty() {
        let compressed = simple_compress(b"");
        assert!(compressed.is_empty());
        let decompressed = simple_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_decompression_odd_length_error() {
        let err = simple_decompress(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, SnapshotStoreError::DecompressionError(_)));
    }

    #[test]
    fn test_pruning() {
        let mut store = SnapshotStore::new();
        for v in 1..=10 {
            store.save("a1", v, &test_state("x", v as i64), test_metadata()).unwrap();
        }
        assert_eq!(store.count("a1"), 10);

        let pruned = store.prune("a1", 3);
        assert_eq!(pruned, 7);
        assert_eq!(store.count("a1"), 3);

        // Remaining are the latest 3.
        let versions = store.versions("a1");
        assert_eq!(versions, vec![8, 9, 10]);
    }

    #[test]
    fn test_auto_pruning_on_save() {
        let mut store = SnapshotStore::new().with_max_snapshots(3);
        for v in 1..=5 {
            store.save("a1", v, &test_state("x", v as i64), test_metadata()).unwrap();
        }
        assert_eq!(store.count("a1"), 3);
        let versions = store.versions("a1");
        assert_eq!(versions, vec![3, 4, 5]);
    }

    #[test]
    fn test_delete_all() {
        let mut store = SnapshotStore::new();
        store.save("a1", 1, &test_state("x", 1), test_metadata()).unwrap();
        store.save("a1", 2, &test_state("x", 2), test_metadata()).unwrap();

        let deleted = store.delete_all("a1");
        assert_eq!(deleted, 2);
        assert_eq!(store.count("a1"), 0);
    }

    #[test]
    fn test_total_count() {
        let mut store = SnapshotStore::new();
        store.save("a1", 1, &test_state("x", 1), test_metadata()).unwrap();
        store.save("a2", 1, &test_state("y", 2), test_metadata()).unwrap();
        store.save("a2", 2, &test_state("y", 3), test_metadata()).unwrap();
        assert_eq!(store.total_count(), 3);
    }

    #[test]
    fn test_aggregate_ids_sorted() {
        let mut store = SnapshotStore::new();
        store.save("zulu", 1, &test_state("z", 1), test_metadata()).unwrap();
        store.save("alpha", 1, &test_state("a", 1), test_metadata()).unwrap();
        assert_eq!(store.aggregate_ids(), vec!["alpha", "zulu"]);
    }

    #[test]
    fn test_metadata_fields() {
        let meta = SnapshotMetadata::new("Order").with_created_by("system");
        assert_eq!(meta.aggregate_type, "Order");
        assert_eq!(meta.created_by.as_deref(), Some("system"));
    }

    #[test]
    fn test_replay_from_snapshot() {
        let mut store = SnapshotStore::new();
        let state = test_state("alice", 100);
        store.save("a1", 5, &state, test_metadata()).unwrap();

        let events = vec![
            ReplayEvent {
                event_type: "Deposit".to_string(),
                version: 6,
                data: {
                    let mut d = HashMap::new();
                    d.insert("amount".to_string(), "50".to_string());
                    d
                },
            },
            ReplayEvent {
                event_type: "Deposit".to_string(),
                version: 7,
                data: {
                    let mut d = HashMap::new();
                    d.insert("amount".to_string(), "25".to_string());
                    d
                },
            },
        ];

        fn apply(state: &mut HashMap<String, serde_json::Value>, event: &ReplayEvent) {
            let amount: i64 = event.data.get("amount").and_then(|s| s.parse().ok()).unwrap_or(0);
            let balance = state.get("balance").and_then(|v| v.as_i64()).unwrap_or(0);
            state.insert("balance".to_string(), serde_json::json!(balance + amount));
        }

        let (result_state, version) = store.replay("a1", &events, apply).unwrap();
        assert_eq!(version, 7);
        assert_eq!(result_state.get("balance").unwrap(), &serde_json::json!(175));
    }

    #[test]
    fn test_replay_skips_old_events() {
        let mut store = SnapshotStore::new();
        store.save("a1", 5, &test_state("alice", 100), test_metadata()).unwrap();

        let events = vec![
            ReplayEvent {
                event_type: "Old".to_string(),
                version: 3, // Before snapshot — should be skipped.
                data: HashMap::new(),
            },
        ];

        fn noop(_state: &mut HashMap<String, serde_json::Value>, _event: &ReplayEvent) {
            unreachable!("should not be called for old events");
        }

        let (state, version) = store.replay("a1", &events, noop).unwrap();
        assert_eq!(version, 5);
        assert_eq!(state.get("balance").unwrap(), &serde_json::json!(100));
    }

    #[test]
    fn test_prune_nonexistent_aggregate() {
        let mut store = SnapshotStore::new();
        assert_eq!(store.prune("missing", 5), 0);
    }

    #[test]
    fn test_versions_empty() {
        let store = SnapshotStore::new();
        assert!(store.versions("missing").is_empty());
    }

    #[test]
    fn test_policy_getter() {
        let store = SnapshotStore::new().with_policy(SnapshotPolicy::Always);
        assert_eq!(store.policy(), SnapshotPolicy::Always);
    }

    #[test]
    fn test_delete_all_nonexistent() {
        let mut store = SnapshotStore::new();
        assert_eq!(store.delete_all("missing"), 0);
    }
}
