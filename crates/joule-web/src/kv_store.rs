//! Key-value store — in-memory KV with get/set/delete, prefix scan, TTL expiry,
//! atomic compare-and-swap, batch operations, iteration, size tracking, and
//! snapshotting (serialize to JSON).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by KV store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// Key not found.
    NotFound,
    /// Compare-and-swap failed: expected value did not match current.
    CasFailed,
    /// Entry has expired.
    Expired,
    /// Snapshot serialization / deserialization error.
    SnapshotError(String),
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "key not found"),
            Self::CasFailed => write!(f, "compare-and-swap failed"),
            Self::Expired => write!(f, "entry expired"),
            Self::SnapshotError(msg) => write!(f, "snapshot error: {msg}"),
        }
    }
}

// ── Entry ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct KvEntry {
    value: Vec<u8>,
    version: u64,
    expires_at: Option<Instant>,
    created_at: Instant,
}

impl KvEntry {
    fn is_expired(&self, now: Instant) -> bool {
        self.expires_at.is_some_and(|exp| now >= exp)
    }
}

// ── Snapshot types ───────────────────────────────────────────────────────────

/// A serializable snapshot of a single KV entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub version: u64,
    /// TTL remaining in milliseconds (None if no expiry).
    pub ttl_remaining_ms: Option<u64>,
}

/// A serializable snapshot of the entire KV store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub entries: Vec<SnapshotEntry>,
    pub total_size_bytes: usize,
}

// ── Stats ────────────────────────────────────────────────────────────────────

/// Statistics for the KV store.
#[derive(Debug, Clone, Default)]
pub struct KvStats {
    pub total_keys: usize,
    pub total_value_bytes: usize,
    pub gets: u64,
    pub sets: u64,
    pub deletes: u64,
    pub cas_attempts: u64,
    pub cas_successes: u64,
    pub expirations: u64,
}

// ── Batch operation ──────────────────────────────────────────────────────────

/// A single operation in a batch.
#[derive(Debug, Clone)]
pub enum BatchOp {
    /// Set a key to a value.
    Set { key: String, value: Vec<u8> },
    /// Set a key with a TTL.
    SetWithTtl {
        key: String,
        value: Vec<u8>,
        ttl: Duration,
    },
    /// Delete a key.
    Delete { key: String },
}

/// Result of a batch operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub sets: usize,
    pub deletes: usize,
    pub errors: usize,
}

// ── KvStore ──────────────────────────────────────────────────────────────────

/// In-memory key-value store with TTL, CAS, prefix scan, and snapshotting.
pub struct KvStore {
    data: BTreeMap<String, KvEntry>,
    next_version: u64,
    stats: KvStats,
}

impl KvStore {
    /// Create a new empty KV store.
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
            next_version: 1,
            stats: KvStats::default(),
        }
    }

    /// Get the value for a key, returning `None` if not found or expired.
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        self.stats.gets += 1;
        let now = Instant::now();
        if let Some(entry) = self.data.get(key) {
            if entry.is_expired(now) {
                let k = key.to_string();
                let entry = self.data.remove(&k).unwrap();
                self.stats.total_value_bytes -= entry.value.len();
                self.stats.total_keys -= 1;
                self.stats.expirations += 1;
                return None;
            }
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Get the value and version for a key.
    pub fn get_versioned(&mut self, key: &str) -> Option<(Vec<u8>, u64)> {
        self.stats.gets += 1;
        let now = Instant::now();
        if let Some(entry) = self.data.get(key) {
            if entry.is_expired(now) {
                let k = key.to_string();
                let entry = self.data.remove(&k).unwrap();
                self.stats.total_value_bytes -= entry.value.len();
                self.stats.total_keys -= 1;
                self.stats.expirations += 1;
                return None;
            }
            Some((entry.value.clone(), entry.version))
        } else {
            None
        }
    }

    /// Set a key to a value (no TTL).
    pub fn set(&mut self, key: String, value: Vec<u8>) {
        self.set_inner(key, value, None);
    }

    /// Set a key with a TTL.
    pub fn set_with_ttl(&mut self, key: String, value: Vec<u8>, ttl: Duration) {
        self.set_inner(key, value, Some(ttl));
    }

    fn set_inner(&mut self, key: String, value: Vec<u8>, ttl: Option<Duration>) {
        let now = Instant::now();
        let version = self.next_version;
        self.next_version += 1;
        self.stats.sets += 1;

        let val_len = value.len();
        if let Some(old) = self.data.insert(
            key,
            KvEntry {
                value,
                version,
                expires_at: ttl.map(|d| now + d),
                created_at: now,
            },
        ) {
            // Replacing: adjust byte count.
            self.stats.total_value_bytes -= old.value.len();
            self.stats.total_value_bytes += val_len;
        } else {
            self.stats.total_keys += 1;
            self.stats.total_value_bytes += val_len;
        }
    }

    /// Delete a key. Returns true if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        self.stats.deletes += 1;
        if let Some(entry) = self.data.remove(key) {
            self.stats.total_keys -= 1;
            self.stats.total_value_bytes -= entry.value.len();
            true
        } else {
            false
        }
    }

    /// Atomic compare-and-swap: only update if the current version matches `expected_version`.
    pub fn compare_and_swap(
        &mut self,
        key: &str,
        expected_version: u64,
        new_value: Vec<u8>,
    ) -> Result<u64, KvError> {
        self.stats.cas_attempts += 1;
        let now = Instant::now();

        let entry = self.data.get(key).ok_or(KvError::NotFound)?;
        if entry.is_expired(now) {
            let k = key.to_string();
            let entry = self.data.remove(&k).unwrap();
            self.stats.total_value_bytes -= entry.value.len();
            self.stats.total_keys -= 1;
            self.stats.expirations += 1;
            return Err(KvError::Expired);
        }
        if entry.version != expected_version {
            return Err(KvError::CasFailed);
        }

        let new_version = self.next_version;
        self.next_version += 1;
        let old_len = entry.value.len();
        let new_len = new_value.len();

        let k = key.to_string();
        let entry_mut = self.data.get_mut(&k).unwrap();
        entry_mut.value = new_value;
        entry_mut.version = new_version;

        self.stats.total_value_bytes = self.stats.total_value_bytes - old_len + new_len;
        self.stats.cas_successes += 1;
        Ok(new_version)
    }

    /// Scan all keys with the given prefix, returning (key, value) pairs.
    pub fn prefix_scan(&mut self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        let now = Instant::now();
        let mut expired_keys = Vec::new();
        let mut result = Vec::new();

        for (k, entry) in self.data.range(prefix.to_string()..) {
            if !k.starts_with(prefix) {
                break;
            }
            if entry.is_expired(now) {
                expired_keys.push(k.clone());
            } else {
                result.push((k.clone(), entry.value.clone()));
            }
        }

        for k in expired_keys {
            if let Some(entry) = self.data.remove(&k) {
                self.stats.total_value_bytes -= entry.value.len();
                self.stats.total_keys -= 1;
                self.stats.expirations += 1;
            }
        }

        result
    }

    /// Execute a batch of operations atomically.
    pub fn batch(&mut self, ops: Vec<BatchOp>) -> BatchResult {
        let mut sets = 0;
        let mut deletes = 0;
        let errors = 0;

        for op in ops {
            match op {
                BatchOp::Set { key, value } => {
                    self.set(key, value);
                    sets += 1;
                }
                BatchOp::SetWithTtl { key, value, ttl } => {
                    self.set_with_ttl(key, value, ttl);
                    sets += 1;
                }
                BatchOp::Delete { key } => {
                    self.delete(&key);
                    deletes += 1;
                }
            }
        }

        BatchResult {
            sets,
            deletes,
            errors,
        }
    }

    /// Iterate over all non-expired entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u8])> {
        let now = Instant::now();
        self.data.iter().filter_map(move |(k, entry)| {
            if entry.is_expired(now) {
                None
            } else {
                Some((k.as_str(), entry.value.as_slice()))
            }
        })
    }

    /// Return all keys (non-expired).
    pub fn keys(&self) -> Vec<&str> {
        let now = Instant::now();
        self.data
            .iter()
            .filter(|(_, e)| !e.is_expired(now))
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Number of (non-expired) entries.
    pub fn len(&self) -> usize {
        self.stats.total_keys
    }

    pub fn is_empty(&self) -> bool {
        self.stats.total_keys == 0
    }

    /// Total bytes of all stored values.
    pub fn total_bytes(&self) -> usize {
        self.stats.total_value_bytes
    }

    pub fn stats(&self) -> &KvStats {
        &self.stats
    }

    /// Check if a key exists and is not expired.
    pub fn contains_key(&mut self, key: &str) -> bool {
        let now = Instant::now();
        if let Some(entry) = self.data.get(key) {
            if entry.is_expired(now) {
                let k = key.to_string();
                let entry = self.data.remove(&k).unwrap();
                self.stats.total_value_bytes -= entry.value.len();
                self.stats.total_keys -= 1;
                self.stats.expirations += 1;
                return false;
            }
            true
        } else {
            false
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.data.clear();
        self.stats.total_keys = 0;
        self.stats.total_value_bytes = 0;
    }

    /// Create a JSON snapshot of the store.
    pub fn snapshot(&self) -> Result<String, KvError> {
        let now = Instant::now();
        let entries: Vec<SnapshotEntry> = self
            .data
            .iter()
            .filter(|(_, e)| !e.is_expired(now))
            .map(|(k, e)| {
                let ttl_remaining_ms = e.expires_at.and_then(|exp| {
                    if exp > now {
                        Some(exp.duration_since(now).as_millis() as u64)
                    } else {
                        None
                    }
                });
                SnapshotEntry {
                    key: k.clone(),
                    value: e.value.clone(),
                    version: e.version,
                    ttl_remaining_ms,
                }
            })
            .collect();
        let total_size = entries.iter().map(|e| e.value.len()).sum();
        let snapshot = Snapshot {
            entries,
            total_size_bytes: total_size,
        };
        serde_json::to_string(&snapshot).map_err(|e| KvError::SnapshotError(e.to_string()))
    }

    /// Restore from a JSON snapshot. This clears existing data.
    pub fn restore_snapshot(&mut self, json: &str) -> Result<usize, KvError> {
        let snapshot: Snapshot =
            serde_json::from_str(json).map_err(|e| KvError::SnapshotError(e.to_string()))?;
        self.clear();
        let now = Instant::now();
        let count = snapshot.entries.len();
        for entry in snapshot.entries {
            let ttl = entry.ttl_remaining_ms.map(Duration::from_millis);
            let version = self.next_version;
            self.next_version += 1;
            let val_len = entry.value.len();
            self.data.insert(
                entry.key,
                KvEntry {
                    value: entry.value,
                    version,
                    expires_at: ttl.map(|d| now + d),
                    created_at: now,
                },
            );
            self.stats.total_keys += 1;
            self.stats.total_value_bytes += val_len;
        }
        Ok(count)
    }
}

impl Default for KvStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut store = KvStore::new();
        store.set("key1".into(), b"value1".to_vec());
        assert_eq!(store.get("key1"), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_get_missing() {
        let mut store = KvStore::new();
        assert_eq!(store.get("missing"), None);
    }

    #[test]
    fn test_delete() {
        let mut store = KvStore::new();
        store.set("k".into(), b"v".to_vec());
        assert!(store.delete("k"));
        assert!(!store.delete("k"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut store = KvStore::new();
        store.set_with_ttl("fast".into(), b"data".to_vec(), Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(store.get("fast"), None);
        assert_eq!(store.stats().expirations, 1);
    }

    #[test]
    fn test_compare_and_swap_success() {
        let mut store = KvStore::new();
        store.set("k".into(), b"v1".to_vec());
        let (_, v1) = store.get_versioned("k").unwrap();
        let v2 = store.compare_and_swap("k", v1, b"v2".to_vec()).unwrap();
        assert!(v2 > v1);
        assert_eq!(store.get("k"), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_compare_and_swap_fail() {
        let mut store = KvStore::new();
        store.set("k".into(), b"v1".to_vec());
        let result = store.compare_and_swap("k", 999, b"v2".to_vec());
        assert_eq!(result, Err(KvError::CasFailed));
        // Original value preserved.
        assert_eq!(store.get("k"), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_prefix_scan() {
        let mut store = KvStore::new();
        store.set("user:1".into(), b"alice".to_vec());
        store.set("user:2".into(), b"bob".to_vec());
        store.set("order:1".into(), b"pizza".to_vec());
        let users = store.prefix_scan("user:");
        assert_eq!(users.len(), 2);
        // BTreeMap is sorted, so user:1 comes first.
        assert_eq!(users[0].0, "user:1");
        assert_eq!(users[1].0, "user:2");
    }

    #[test]
    fn test_batch_operations() {
        let mut store = KvStore::new();
        store.set("existing".into(), b"old".to_vec());
        let result = store.batch(vec![
            BatchOp::Set {
                key: "a".into(),
                value: b"1".to_vec(),
            },
            BatchOp::Set {
                key: "b".into(),
                value: b"2".to_vec(),
            },
            BatchOp::Delete {
                key: "existing".into(),
            },
        ]);
        assert_eq!(result.sets, 2);
        assert_eq!(result.deletes, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_size_tracking() {
        let mut store = KvStore::new();
        store.set("k1".into(), b"abc".to_vec()); // 3 bytes
        store.set("k2".into(), b"defgh".to_vec()); // 5 bytes
        assert_eq!(store.total_bytes(), 8);
        store.delete("k1");
        assert_eq!(store.total_bytes(), 5);
    }

    #[test]
    fn test_overwrite_adjusts_size() {
        let mut store = KvStore::new();
        store.set("k".into(), b"short".to_vec()); // 5 bytes
        assert_eq!(store.total_bytes(), 5);
        store.set("k".into(), b"much longer value".to_vec()); // 17 bytes
        assert_eq!(store.total_bytes(), 17);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_keys() {
        let mut store = KvStore::new();
        store.set("b".into(), b"2".to_vec());
        store.set("a".into(), b"1".to_vec());
        store.set("c".into(), b"3".to_vec());
        let keys = store.keys();
        // BTreeMap is sorted.
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_contains_key() {
        let mut store = KvStore::new();
        store.set("exists".into(), b"yes".to_vec());
        assert!(store.contains_key("exists"));
        assert!(!store.contains_key("nope"));
    }

    #[test]
    fn test_snapshot_and_restore() {
        let mut store = KvStore::new();
        store.set("k1".into(), b"v1".to_vec());
        store.set("k2".into(), b"v2".to_vec());

        let json = store.snapshot().unwrap();
        assert!(json.contains("k1"));
        assert!(json.contains("k2"));

        let mut store2 = KvStore::new();
        let count = store2.restore_snapshot(&json).unwrap();
        assert_eq!(count, 2);
        assert_eq!(store2.get("k1"), Some(b"v1".to_vec()));
        assert_eq!(store2.get("k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_clear() {
        let mut store = KvStore::new();
        store.set("a".into(), b"1".to_vec());
        store.set("b".into(), b"2".to_vec());
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn test_iter() {
        let mut store = KvStore::new();
        store.set("x".into(), b"10".to_vec());
        store.set("y".into(), b"20".to_vec());
        let entries: Vec<_> = store.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_versioned_get() {
        let mut store = KvStore::new();
        store.set("k".into(), b"v1".to_vec());
        let (val, ver1) = store.get_versioned("k").unwrap();
        assert_eq!(val, b"v1".to_vec());
        store.set("k".into(), b"v2".to_vec());
        let (val2, ver2) = store.get_versioned("k").unwrap();
        assert_eq!(val2, b"v2".to_vec());
        assert!(ver2 > ver1);
    }

    #[test]
    fn test_batch_with_ttl() {
        let mut store = KvStore::new();
        store.batch(vec![BatchOp::SetWithTtl {
            key: "temp".into(),
            value: b"data".to_vec(),
            ttl: Duration::from_secs(60),
        }]);
        assert!(store.contains_key("temp"));
    }

    #[test]
    fn test_stats() {
        let mut store = KvStore::new();
        store.set("a".into(), b"1".to_vec());
        store.get("a");
        store.get("b");
        store.delete("a");
        assert_eq!(store.stats().sets, 1);
        assert_eq!(store.stats().gets, 2);
        assert_eq!(store.stats().deletes, 1);
    }
}
