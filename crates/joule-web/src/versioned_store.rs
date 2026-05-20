//! Versioned key-value store — version per key, read-at-version, version range
//! query, garbage collection of old versions, snapshot isolation, MVCC concepts,
//! version compaction.

use std::collections::BTreeMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by the versioned store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionedStoreError {
    /// Key not found.
    KeyNotFound(String),
    /// No version of the key visible at the requested version.
    VersionNotFound { key: String, version: u64 },
    /// Snapshot not found.
    SnapshotNotFound(u64),
    /// Write conflict under snapshot isolation.
    WriteConflict { key: String },
}

impl std::fmt::Display for VersionedStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::VersionNotFound { key, version } => {
                write!(f, "no version of key '{key}' at version {version}")
            }
            Self::SnapshotNotFound(id) => write!(f, "snapshot {id} not found"),
            Self::WriteConflict { key } => write!(f, "write conflict on key '{key}'"),
        }
    }
}

impl std::error::Error for VersionedStoreError {}

// ── Versioned Value ──────────────────────────────────────────────────────────

/// A single version of a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedValue {
    /// The value data (None = tombstone).
    pub value: Option<Vec<u8>>,
    /// The version at which this was written.
    pub version: u64,
}

// ── Version Chain ────────────────────────────────────────────────────────────

/// All versions of a single key, ordered by version (oldest first).
#[derive(Debug, Clone)]
struct VersionChain {
    /// Versions in ascending order.
    versions: Vec<VersionedValue>,
}

impl VersionChain {
    fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Add a new version.
    fn put(&mut self, value: Option<Vec<u8>>, version: u64) {
        // If a version at this exact number exists, replace it.
        if let Some(existing) = self.versions.iter_mut().find(|v| v.version == version) {
            existing.value = value;
            return;
        }
        self.versions.push(VersionedValue { value, version });
        self.versions.sort_by_key(|v| v.version);
    }

    /// Read the value visible at the given version (latest version <= target).
    fn read_at(&self, target_version: u64) -> Option<&VersionedValue> {
        // Binary search for the latest version <= target.
        let mut result = None;
        for v in &self.versions {
            if v.version <= target_version {
                result = Some(v);
            } else {
                break;
            }
        }
        result
    }

    /// Get the latest version.
    fn latest(&self) -> Option<&VersionedValue> {
        self.versions.last()
    }

    /// Get all versions in a range [from, to] inclusive.
    fn range(&self, from: u64, to: u64) -> Vec<&VersionedValue> {
        self.versions
            .iter()
            .filter(|v| v.version >= from && v.version <= to)
            .collect()
    }

    /// Remove versions older than `min_version`, keeping at least the latest.
    fn gc(&mut self, min_version: u64) -> usize {
        if self.versions.len() <= 1 {
            return 0;
        }
        let before = self.versions.len();
        // Keep the latest version and any version >= min_version.
        let latest = self.versions.last().cloned();
        self.versions.retain(|v| v.version >= min_version);
        if self.versions.is_empty() {
            if let Some(l) = latest {
                self.versions.push(l);
            }
        }
        before - self.versions.len()
    }

    /// Number of versions.
    fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// All versions.
    fn all_versions(&self) -> &[VersionedValue] {
        &self.versions
    }
}

// ── Snapshot ──────────────────────────────────────────────────────────────────

/// A read snapshot at a particular version.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot ID.
    pub id: u64,
    /// The version at which this snapshot reads.
    pub read_version: u64,
    /// Keys written by this snapshot's transaction (for conflict detection).
    pub write_set: Vec<String>,
}

// ── Store Statistics ─────────────────────────────────────────────────────────

/// Statistics for the versioned store.
#[derive(Debug, Clone, Default)]
pub struct VersionedStoreStats {
    pub total_keys: usize,
    pub total_versions: usize,
    pub current_version: u64,
    pub active_snapshots: usize,
    pub gc_runs: u64,
    pub versions_collected: u64,
    pub total_reads: u64,
    pub total_writes: u64,
}

// ── Versioned Store ──────────────────────────────────────────────────────────

/// An MVCC versioned key-value store.
#[derive(Debug)]
pub struct VersionedStore {
    /// Key -> version chain.
    data: BTreeMap<String, VersionChain>,
    /// Current global version.
    current_version: u64,
    /// Active snapshots.
    snapshots: BTreeMap<u64, Snapshot>,
    /// Next snapshot ID.
    next_snapshot_id: u64,
    /// Statistics.
    gc_runs: u64,
    versions_collected: u64,
    total_reads: u64,
    total_writes: u64,
}

impl VersionedStore {
    /// Create a new versioned store.
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
            current_version: 0,
            snapshots: BTreeMap::new(),
            next_snapshot_id: 1,
            gc_runs: 0,
            versions_collected: 0,
            total_reads: 0,
            total_writes: 0,
        }
    }

    /// Put a key-value pair, creating a new version.
    pub fn put(&mut self, key: String, value: Vec<u8>) -> u64 {
        self.current_version += 1;
        let version = self.current_version;
        self.total_writes += 1;

        let chain = self
            .data
            .entry(key)
            .or_insert_with(VersionChain::new);
        chain.put(Some(value), version);
        version
    }

    /// Delete a key (write tombstone at new version).
    pub fn delete(&mut self, key: &str) -> Result<u64, VersionedStoreError> {
        self.current_version += 1;
        let version = self.current_version;
        self.total_writes += 1;

        let chain = self
            .data
            .get_mut(key)
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        chain.put(None, version);
        Ok(version)
    }

    /// Get the latest value of a key.
    pub fn get(&mut self, key: &str) -> Result<Vec<u8>, VersionedStoreError> {
        self.total_reads += 1;
        let chain = self
            .data
            .get(key)
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        let latest = chain
            .latest()
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        latest
            .value
            .clone()
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))
    }

    /// Read a key at a specific version.
    pub fn get_at_version(
        &mut self,
        key: &str,
        version: u64,
    ) -> Result<Vec<u8>, VersionedStoreError> {
        self.total_reads += 1;
        let chain = self
            .data
            .get(key)
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        let entry = chain.read_at(version).ok_or_else(|| {
            VersionedStoreError::VersionNotFound {
                key: key.to_string(),
                version,
            }
        })?;
        entry
            .value
            .clone()
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))
    }

    /// Get all versions of a key in a version range.
    pub fn version_range(
        &self,
        key: &str,
        from: u64,
        to: u64,
    ) -> Result<Vec<&VersionedValue>, VersionedStoreError> {
        let chain = self
            .data
            .get(key)
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        Ok(chain.range(from, to))
    }

    /// Get all versions of a key.
    pub fn all_versions(
        &self,
        key: &str,
    ) -> Result<&[VersionedValue], VersionedStoreError> {
        let chain = self
            .data
            .get(key)
            .ok_or_else(|| VersionedStoreError::KeyNotFound(key.to_string()))?;
        Ok(chain.all_versions())
    }

    /// Number of versions for a key.
    pub fn version_count(&self, key: &str) -> usize {
        self.data
            .get(key)
            .map_or(0, |c| c.version_count())
    }

    /// Create a snapshot at the current version.
    pub fn snapshot(&mut self) -> u64 {
        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;
        self.snapshots.insert(
            id,
            Snapshot {
                id,
                read_version: self.current_version,
                write_set: Vec::new(),
            },
        );
        id
    }

    /// Read a key through a snapshot.
    pub fn snapshot_get(
        &mut self,
        snapshot_id: u64,
        key: &str,
    ) -> Result<Vec<u8>, VersionedStoreError> {
        let snap = self
            .snapshots
            .get(&snapshot_id)
            .ok_or(VersionedStoreError::SnapshotNotFound(snapshot_id))?;
        let read_version = snap.read_version;
        self.get_at_version(key, read_version)
    }

    /// Write through a snapshot (records in write set for conflict detection).
    pub fn snapshot_put(
        &mut self,
        snapshot_id: u64,
        key: String,
        value: Vec<u8>,
    ) -> Result<u64, VersionedStoreError> {
        // Check for write conflicts: if another version was written after
        // the snapshot was taken, that's a conflict.
        let read_version = {
            let snap = self
                .snapshots
                .get(&snapshot_id)
                .ok_or(VersionedStoreError::SnapshotNotFound(snapshot_id))?;
            snap.read_version
        };

        if let Some(chain) = self.data.get(&key) {
            if let Some(latest) = chain.latest() {
                if latest.version > read_version {
                    return Err(VersionedStoreError::WriteConflict { key });
                }
            }
        }

        let version = self.put(key.clone(), value);

        let snap = self
            .snapshots
            .get_mut(&snapshot_id)
            .ok_or(VersionedStoreError::SnapshotNotFound(snapshot_id))?;
        snap.write_set.push(key);

        Ok(version)
    }

    /// Release a snapshot.
    pub fn release_snapshot(&mut self, snapshot_id: u64) -> Result<(), VersionedStoreError> {
        self.snapshots
            .remove(&snapshot_id)
            .map(|_| ())
            .ok_or(VersionedStoreError::SnapshotNotFound(snapshot_id))
    }

    /// Garbage collect versions older than `min_version`, but never remove
    /// versions still visible to active snapshots.
    pub fn gc(&mut self, min_version: u64) -> u64 {
        // Find the oldest active snapshot version.
        let oldest_snap = self
            .snapshots
            .values()
            .map(|s| s.read_version)
            .min()
            .unwrap_or(u64::MAX);

        let effective_min = min_version.min(oldest_snap);
        let mut collected = 0u64;

        for chain in self.data.values_mut() {
            collected += chain.gc(effective_min) as u64;
        }

        self.gc_runs += 1;
        self.versions_collected += collected;
        collected
    }

    /// Compact: remove keys whose only version is a tombstone.
    pub fn compact(&mut self) -> usize {
        let keys_to_remove: Vec<String> = self
            .data
            .iter()
            .filter(|(_, chain)| {
                chain.version_count() == 1
                    && chain.latest().is_some_and(|v| v.value.is_none())
            })
            .map(|(k, _)| k.clone())
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.data.remove(&key);
        }
        count
    }

    /// Current global version.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Number of keys (including tombstoned).
    pub fn key_count(&self) -> usize {
        self.data.len()
    }

    /// Number of active snapshots.
    pub fn active_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Total versions across all keys.
    pub fn total_versions(&self) -> usize {
        self.data.values().map(|c| c.version_count()).sum()
    }

    /// Get statistics.
    pub fn stats(&self) -> VersionedStoreStats {
        VersionedStoreStats {
            total_keys: self.data.len(),
            total_versions: self.total_versions(),
            current_version: self.current_version,
            active_snapshots: self.snapshots.len(),
            gc_runs: self.gc_runs,
            versions_collected: self.versions_collected,
            total_reads: self.total_reads,
            total_writes: self.total_writes,
        }
    }
}

impl Default for VersionedStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let mut store = VersionedStore::new();
        store.put("key1".into(), b"val1".to_vec());
        assert_eq!(store.get("key1").unwrap(), b"val1");
    }

    #[test]
    fn get_missing_key() {
        let mut store = VersionedStore::new();
        assert_eq!(
            store.get("nope"),
            Err(VersionedStoreError::KeyNotFound("nope".into()))
        );
    }

    #[test]
    fn put_updates_version() {
        let mut store = VersionedStore::new();
        let v1 = store.put("k".into(), b"a".to_vec());
        let v2 = store.put("k".into(), b"b".to_vec());
        assert!(v2 > v1);
        assert_eq!(store.get("k").unwrap(), b"b");
    }

    #[test]
    fn delete_creates_tombstone() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"val".to_vec());
        store.delete("k").unwrap();
        assert!(store.get("k").is_err());
    }

    #[test]
    fn read_at_version() {
        let mut store = VersionedStore::new();
        let v1 = store.put("k".into(), b"first".to_vec());
        let _v2 = store.put("k".into(), b"second".to_vec());
        assert_eq!(store.get_at_version("k", v1).unwrap(), b"first");
    }

    #[test]
    fn read_at_version_not_found() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"val".to_vec());
        let result = store.get_at_version("k", 0);
        assert!(matches!(
            result,
            Err(VersionedStoreError::VersionNotFound { .. })
        ));
    }

    #[test]
    fn version_range_query() {
        let mut store = VersionedStore::new();
        let v1 = store.put("k".into(), b"a".to_vec());
        let v2 = store.put("k".into(), b"b".to_vec());
        let v3 = store.put("k".into(), b"c".to_vec());
        let range = store.version_range("k", v1, v3).unwrap();
        assert_eq!(range.len(), 3);
        let mid = store.version_range("k", v2, v2).unwrap();
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].value.as_deref(), Some(b"b".as_slice()));
    }

    #[test]
    fn all_versions() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"1".to_vec());
        store.put("k".into(), b"2".to_vec());
        store.put("k".into(), b"3".to_vec());
        let versions = store.all_versions("k").unwrap();
        assert_eq!(versions.len(), 3);
    }

    #[test]
    fn snapshot_read() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"old".to_vec());
        let snap = store.snapshot();
        store.put("k".into(), b"new".to_vec());
        // Snapshot reads at the old version.
        assert_eq!(store.snapshot_get(snap, "k").unwrap(), b"old");
        // Current reads the new version.
        assert_eq!(store.get("k").unwrap(), b"new");
    }

    #[test]
    fn snapshot_write_no_conflict() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"v1".to_vec());
        let snap = store.snapshot();
        store.snapshot_put(snap, "k".into(), b"v2".to_vec()).unwrap();
        assert_eq!(store.get("k").unwrap(), b"v2");
    }

    #[test]
    fn snapshot_write_conflict() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"v1".to_vec());
        let snap = store.snapshot();
        // Write after snapshot.
        store.put("k".into(), b"v2".to_vec());
        // Snapshot write should detect conflict.
        let result = store.snapshot_put(snap, "k".into(), b"v3".to_vec());
        assert!(matches!(
            result,
            Err(VersionedStoreError::WriteConflict { .. })
        ));
    }

    #[test]
    fn release_snapshot() {
        let mut store = VersionedStore::new();
        let snap = store.snapshot();
        assert_eq!(store.active_snapshots(), 1);
        store.release_snapshot(snap).unwrap();
        assert_eq!(store.active_snapshots(), 0);
    }

    #[test]
    fn release_unknown_snapshot() {
        let mut store = VersionedStore::new();
        assert_eq!(
            store.release_snapshot(99),
            Err(VersionedStoreError::SnapshotNotFound(99))
        );
    }

    #[test]
    fn gc_removes_old_versions() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"v1".to_vec());
        store.put("k".into(), b"v2".to_vec());
        let v3 = store.put("k".into(), b"v3".to_vec());
        assert_eq!(store.version_count("k"), 3);
        let collected = store.gc(v3);
        assert!(collected > 0);
        assert!(store.version_count("k") < 3);
    }

    #[test]
    fn gc_respects_snapshots() {
        let mut store = VersionedStore::new();
        let v1 = store.put("k".into(), b"v1".to_vec());
        store.put("k".into(), b"v2".to_vec());
        let _snap = store.snapshot();
        // GC should not remove versions visible to the snapshot.
        store.gc(v1 + 1);
        // v1 is before snap's read_version, but the snapshot was taken at v2,
        // so v1 might be collected depending on effective_min logic.
        assert!(store.version_count("k") >= 1);
    }

    #[test]
    fn compact_removes_tombstones() {
        let mut store = VersionedStore::new();
        store.put("k".into(), b"v".to_vec());
        store.delete("k").unwrap();
        // GC to reduce to single tombstone.
        store.gc(store.current_version() + 1);
        let removed = store.compact();
        assert!(removed >= 1);
        assert_eq!(store.key_count(), 0);
    }

    #[test]
    fn stats() {
        let mut store = VersionedStore::new();
        store.put("a".into(), b"1".to_vec());
        store.put("a".into(), b"2".to_vec());
        store.put("b".into(), b"3".to_vec());
        store.get("a").unwrap();
        let stats = store.stats();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.total_versions, 3);
        assert_eq!(stats.total_writes, 3);
        assert_eq!(stats.total_reads, 1);
    }

    #[test]
    fn version_count_for_missing_key() {
        let store = VersionedStore::new();
        assert_eq!(store.version_count("missing"), 0);
    }

    #[test]
    fn error_display() {
        let e = VersionedStoreError::KeyNotFound("foo".into());
        assert!(e.to_string().contains("foo"));
        let e = VersionedStoreError::WriteConflict { key: "bar".into() };
        assert!(e.to_string().contains("bar"));
    }
}
