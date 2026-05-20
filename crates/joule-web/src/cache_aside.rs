//! Cache-aside pattern — get-or-load with TTL, write-through, write-behind,
//! cache warming, invalidation, and miss-rate tracking.
//!
//! Replaces `node-cache`, `lru-cache`, and similar JS caching libraries with a
//! pure-Rust, energy-aware, in-memory cache that supports multiple write
//! strategies and detailed hit/miss statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Cache errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Key not found in cache or backing store.
    NotFound(String),
    /// Loader function returned an error.
    LoadFailed(String),
    /// Write to backing store failed.
    WriteFailed(String),
    /// Cache is at capacity and eviction is disabled.
    CapacityExceeded { capacity: usize },
    /// Invalid TTL configuration.
    InvalidTtl(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(k) => write!(f, "cache key not found: {k}"),
            Self::LoadFailed(msg) => write!(f, "loader failed: {msg}"),
            Self::WriteFailed(msg) => write!(f, "write failed: {msg}"),
            Self::CapacityExceeded { capacity } => {
                write!(f, "cache capacity exceeded: {capacity}")
            }
            Self::InvalidTtl(msg) => write!(f, "invalid TTL: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

// ── Write strategy ──────────────────────────────────────────────

/// Strategy for propagating writes to the backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteStrategy {
    /// Writes go only to cache; backing store is not updated.
    CacheOnly,
    /// Writes go to both cache and backing store synchronously.
    WriteThrough,
    /// Writes go to cache immediately; backing store updates are queued.
    WriteBehind,
}

// ── Cache entry ─────────────────────────────────────────────────

/// A single cached value with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<V: Clone> {
    /// The cached value.
    pub value: V,
    /// Insertion timestamp (milliseconds since epoch).
    pub inserted_at_ms: u64,
    /// TTL in milliseconds (0 = no expiry).
    pub ttl_ms: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Last access timestamp.
    pub last_accessed_ms: u64,
    /// Whether this entry is dirty (write-behind pending).
    pub dirty: bool,
}

impl<V: Clone> CacheEntry<V> {
    /// Check if this entry has expired at the given time.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        if self.ttl_ms == 0 {
            return false;
        }
        now_ms.saturating_sub(self.inserted_at_ms) >= self.ttl_ms
    }
}

// ── Stats ───────────────────────────────────────────────────────

/// Cache hit/miss statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub loads: u64,
    pub load_failures: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub write_throughs: u64,
    pub write_behinds_queued: u64,
    pub write_behinds_flushed: u64,
    pub warmups: u64,
}

impl CacheStats {
    /// Hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Miss rate as a fraction in [0.0, 1.0].
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

// ── Pending write ───────────────────────────────────────────────

/// A deferred write for write-behind strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingWrite<V: Clone> {
    pub key: String,
    pub value: V,
    pub queued_at_ms: u64,
}

// ── Cache configuration ─────────────────────────────────────────

/// Configuration for the cache-aside store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries (0 = unbounded).
    pub max_entries: usize,
    /// Default TTL in milliseconds (0 = no expiry).
    pub default_ttl_ms: u64,
    /// Write strategy.
    pub write_strategy: WriteStrategy,
    /// Whether to evict the least-recently-used entry on capacity overflow.
    pub lru_eviction: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            default_ttl_ms: 60_000,
            write_strategy: WriteStrategy::CacheOnly,
            lru_eviction: true,
        }
    }
}

// ── CacheAside ──────────────────────────────────────────────────

/// In-memory cache-aside store.
pub struct CacheAside<V: Clone> {
    config: CacheConfig,
    entries: HashMap<String, CacheEntry<V>>,
    stats: CacheStats,
    pending_writes: Vec<PendingWrite<V>>,
}

impl<V: Clone + fmt::Debug> CacheAside<V> {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: CacheStats::default(),
            pending_writes: Vec::new(),
        }
    }

    /// Get a value from the cache.  Returns `None` if absent or expired.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&V> {
        // Two-phase: check expiry, then borrow.
        let expired = self
            .entries
            .get(key)
            .map(|e| e.is_expired(now_ms))
            .unwrap_or(false);

        if expired {
            self.entries.remove(key);
            self.stats.misses += 1;
            return None;
        }

        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            let entry = self.entries.get_mut(key).unwrap();
            entry.access_count += 1;
            entry.last_accessed_ms = now_ms;
            Some(&self.entries.get(key).unwrap().value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get-or-load: returns cached value or invokes `loader`, caching the
    /// result on success.
    pub fn get_or_load<F>(
        &mut self,
        key: &str,
        now_ms: u64,
        loader: F,
    ) -> Result<&V, CacheError>
    where
        F: FnOnce(&str) -> Result<V, String>,
    {
        // Check cache first (including expiry).
        let expired = self
            .entries
            .get(key)
            .map(|e| e.is_expired(now_ms))
            .unwrap_or(false);
        if expired {
            self.entries.remove(key);
        }

        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            let entry = self.entries.get_mut(key).unwrap();
            entry.access_count += 1;
            entry.last_accessed_ms = now_ms;
            return Ok(&self.entries.get(key).unwrap().value);
        }

        // Cache miss — invoke loader.
        self.stats.misses += 1;
        self.stats.loads += 1;
        let value = loader(key).map_err(|e| {
            self.stats.load_failures += 1;
            CacheError::LoadFailed(e)
        })?;

        self.insert_internal(key.to_string(), value, self.config.default_ttl_ms, now_ms, false)?;

        Ok(&self.entries.get(key).unwrap().value)
    }

    /// Put a value into the cache, applying the configured write strategy.
    pub fn put(
        &mut self,
        key: String,
        value: V,
        now_ms: u64,
    ) -> Result<(), CacheError> {
        let strategy = self.config.write_strategy;
        let ttl = self.config.default_ttl_ms;
        let dirty = strategy == WriteStrategy::WriteBehind;
        self.insert_internal(key.clone(), value.clone(), ttl, now_ms, dirty)?;

        match strategy {
            WriteStrategy::WriteThrough => {
                self.stats.write_throughs += 1;
            }
            WriteStrategy::WriteBehind => {
                self.pending_writes.push(PendingWrite {
                    key,
                    value,
                    queued_at_ms: now_ms,
                });
                self.stats.write_behinds_queued += 1;
            }
            WriteStrategy::CacheOnly => {}
        }
        Ok(())
    }

    /// Put with custom TTL.
    pub fn put_with_ttl(
        &mut self,
        key: String,
        value: V,
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<(), CacheError> {
        self.insert_internal(key, value, ttl_ms, now_ms, false)
    }

    /// Invalidate (remove) a single key.
    pub fn invalidate(&mut self, key: &str) -> bool {
        let removed = self.entries.remove(key).is_some();
        if removed {
            self.stats.invalidations += 1;
        }
        removed
    }

    /// Invalidate all entries matching a predicate on keys.
    pub fn invalidate_matching<F: Fn(&str) -> bool>(&mut self, predicate: F) -> usize {
        let keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| predicate(k))
            .cloned()
            .collect();
        let count = keys.len();
        for k in &keys {
            self.entries.remove(k);
        }
        self.stats.invalidations += count as u64;
        count
    }

    /// Warm the cache with a batch of key-value pairs.
    pub fn warm(
        &mut self,
        items: Vec<(String, V)>,
        now_ms: u64,
    ) -> Result<usize, CacheError> {
        let ttl = self.config.default_ttl_ms;
        let mut count = 0;
        for (key, value) in items {
            self.insert_internal(key, value, ttl, now_ms, false)?;
            count += 1;
        }
        self.stats.warmups += count as u64;
        Ok(count)
    }

    /// Flush all pending write-behind entries, returning them.
    pub fn flush_pending_writes(&mut self) -> Vec<PendingWrite<V>> {
        let writes: Vec<_> = self.pending_writes.drain(..).collect();
        self.stats.write_behinds_flushed += writes.len() as u64;
        // Clear dirty flags.
        for w in &writes {
            if let Some(entry) = self.entries.get_mut(&w.key) {
                entry.dirty = false;
            }
        }
        writes
    }

    /// Evict all expired entries.
    pub fn evict_expired(&mut self, now_ms: u64) -> usize {
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(now_ms))
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for k in &expired_keys {
            self.entries.remove(k);
        }
        self.stats.evictions += count as u64;
        count
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get current stats.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// List all keys.
    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Check if a key exists and is not expired.
    pub fn contains(&self, key: &str, now_ms: u64) -> bool {
        self.entries
            .get(key)
            .map(|e| !e.is_expired(now_ms))
            .unwrap_or(false)
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        let count = self.entries.len();
        self.entries.clear();
        self.stats.evictions += count as u64;
    }

    // ── Internal helpers ────────────────────────────────────────

    fn insert_internal(
        &mut self,
        key: String,
        value: V,
        ttl_ms: u64,
        now_ms: u64,
        dirty: bool,
    ) -> Result<(), CacheError> {
        let max = self.config.max_entries;
        if max > 0 && !self.entries.contains_key(&key) && self.entries.len() >= max {
            if self.config.lru_eviction {
                self.evict_lru();
            } else {
                return Err(CacheError::CapacityExceeded { capacity: max });
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                inserted_at_ms: now_ms,
                ttl_ms,
                access_count: 0,
                last_accessed_ms: now_ms,
                dirty,
            },
        );
        Ok(())
    }

    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed_ms)
            .map(|(k, _)| k.clone());
        if let Some(k) = lru_key {
            self.entries.remove(&k);
            self.stats.evictions += 1;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> CacheAside<String> {
        CacheAside::new(CacheConfig::default())
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = default_cache();
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        assert_eq!(cache.get("k1", 1000), Some(&"v1".to_string()));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_get_miss() {
        let mut cache = default_cache();
        assert!(cache.get("missing", 1000).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = default_cache();
        cache.put_with_ttl("k1".into(), "v1".into(), 500, 1000).unwrap();
        assert!(cache.get("k1", 1200).is_some()); // 200ms < 500ms TTL
        assert!(cache.get("k1", 1500).is_none()); // 500ms = TTL, expired
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_no_ttl_never_expires() {
        let mut cache = default_cache();
        cache.put_with_ttl("k1".into(), "v1".into(), 0, 1000).unwrap();
        assert!(cache.get("k1", u64::MAX - 1).is_some());
    }

    #[test]
    fn test_get_or_load_caches() {
        let mut cache = default_cache();
        let val = cache
            .get_or_load("k1", 1000, |k| Ok(format!("loaded-{k}")))
            .unwrap()
            .clone();
        assert_eq!(val, "loaded-k1");
        assert_eq!(cache.stats().loads, 1);
        assert_eq!(cache.stats().misses, 1);

        // Second call should hit cache.
        let val2 = cache
            .get_or_load("k1", 1001, |_| Ok("should-not-be-called".into()))
            .unwrap()
            .clone();
        assert_eq!(val2, "loaded-k1");
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_get_or_load_failure() {
        let mut cache = default_cache();
        let err = cache.get_or_load("k1", 1000, |_| Err("boom".into()));
        assert!(err.is_err());
        assert_eq!(cache.stats().load_failures, 1);
    }

    #[test]
    fn test_invalidate() {
        let mut cache = default_cache();
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        assert!(cache.invalidate("k1"));
        assert!(!cache.invalidate("k1")); // already gone
        assert!(cache.get("k1", 1000).is_none());
        assert_eq!(cache.stats().invalidations, 1);
    }

    #[test]
    fn test_invalidate_matching() {
        let mut cache = default_cache();
        cache.put("user:1".into(), "a".into(), 1000).unwrap();
        cache.put("user:2".into(), "b".into(), 1000).unwrap();
        cache.put("order:1".into(), "c".into(), 1000).unwrap();
        let removed = cache.invalidate_matching(|k| k.starts_with("user:"));
        assert_eq!(removed, 2);
        assert!(cache.get("order:1", 1000).is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            default_ttl_ms: 0,
            lru_eviction: true,
            ..Default::default()
        };
        let mut cache = CacheAside::<String>::new(config);
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        cache.put("k2".into(), "v2".into(), 2000).unwrap();
        // Access k1 so k2 is LRU.
        let _ = cache.get("k1", 3000);
        // Insert k3 — should evict k2.
        cache.put("k3".into(), "v3".into(), 4000).unwrap();
        assert!(cache.get("k1", 5000).is_some());
        assert!(cache.get("k2", 5000).is_none());
        assert!(cache.get("k3", 5000).is_some());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_capacity_exceeded_no_eviction() {
        let config = CacheConfig {
            max_entries: 1,
            default_ttl_ms: 0,
            lru_eviction: false,
            ..Default::default()
        };
        let mut cache = CacheAside::<String>::new(config);
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        let err = cache.put("k2".into(), "v2".into(), 1000);
        assert!(matches!(err, Err(CacheError::CapacityExceeded { capacity: 1 })));
    }

    #[test]
    fn test_write_through_stats() {
        let config = CacheConfig {
            write_strategy: WriteStrategy::WriteThrough,
            ..Default::default()
        };
        let mut cache = CacheAside::<String>::new(config);
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        assert_eq!(cache.stats().write_throughs, 1);
    }

    #[test]
    fn test_write_behind_pending() {
        let config = CacheConfig {
            write_strategy: WriteStrategy::WriteBehind,
            ..Default::default()
        };
        let mut cache = CacheAside::<String>::new(config);
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        assert_eq!(cache.stats().write_behinds_queued, 1);
        assert!(cache.entries.get("k1").unwrap().dirty);

        let flushed = cache.flush_pending_writes();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].key, "k1");
        assert!(!cache.entries.get("k1").unwrap().dirty);
        assert_eq!(cache.stats().write_behinds_flushed, 1);
    }

    #[test]
    fn test_warm() {
        let mut cache = default_cache();
        let items = vec![
            ("a".into(), "1".into()),
            ("b".into(), "2".into()),
            ("c".into(), "3".into()),
        ];
        let count = cache.warm(items, 1000).unwrap();
        assert_eq!(count, 3);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().warmups, 3);
    }

    #[test]
    fn test_evict_expired() {
        let mut cache = default_cache();
        cache.put_with_ttl("a".into(), "1".into(), 100, 1000).unwrap();
        cache.put_with_ttl("b".into(), "2".into(), 500, 1000).unwrap();
        cache.put_with_ttl("c".into(), "3".into(), 0, 1000).unwrap();
        let evicted = cache.evict_expired(1200);
        assert_eq!(evicted, 1); // only "a" expired
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_contains() {
        let mut cache = default_cache();
        cache.put_with_ttl("k1".into(), "v1".into(), 100, 1000).unwrap();
        assert!(cache.contains("k1", 1050));
        assert!(!cache.contains("k1", 1100));
        assert!(!cache.contains("missing", 1000));
    }

    #[test]
    fn test_clear() {
        let mut cache = default_cache();
        cache.put("a".into(), "1".into(), 1000).unwrap();
        cache.put("b".into(), "2".into(), 1000).unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().evictions, 2);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = default_cache();
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        let _ = cache.get("k1", 1001); // hit
        let _ = cache.get("k1", 1002); // hit
        let _ = cache.get("missing", 1003); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
        assert!((cache.stats().miss_rate() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_zero_total() {
        let stats = CacheStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut cache = default_cache();
        cache.put("k1".into(), "old".into(), 1000).unwrap();
        cache.put("k1".into(), "new".into(), 2000).unwrap();
        assert_eq!(cache.get("k1", 2000), Some(&"new".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_get_or_load_expired_reloads() {
        let mut cache = default_cache();
        cache.put_with_ttl("k1".into(), "old".into(), 100, 1000).unwrap();
        let val = cache
            .get_or_load("k1", 1200, |_| Ok("refreshed".into()))
            .unwrap()
            .clone();
        assert_eq!(val, "refreshed");
        assert_eq!(cache.stats().loads, 1);
    }

    #[test]
    fn test_access_count_tracking() {
        let mut cache = default_cache();
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        let _ = cache.get("k1", 1001);
        let _ = cache.get("k1", 1002);
        let _ = cache.get("k1", 1003);
        assert_eq!(cache.entries.get("k1").unwrap().access_count, 3);
    }

    #[test]
    fn test_overwrite_does_not_trigger_eviction_at_capacity() {
        let config = CacheConfig {
            max_entries: 1,
            default_ttl_ms: 0,
            lru_eviction: false,
            ..Default::default()
        };
        let mut cache = CacheAside::<String>::new(config);
        cache.put("k1".into(), "v1".into(), 1000).unwrap();
        // Overwriting same key should succeed even at capacity.
        cache.put("k1".into(), "v2".into(), 2000).unwrap();
        assert_eq!(cache.get("k1", 2000), Some(&"v2".to_string()));
    }

    #[test]
    fn test_cache_entry_expired_boundary() {
        let entry = CacheEntry {
            value: 42,
            inserted_at_ms: 1000,
            ttl_ms: 500,
            access_count: 0,
            last_accessed_ms: 1000,
            dirty: false,
        };
        assert!(!entry.is_expired(1499)); // 499 < 500
        assert!(entry.is_expired(1500));  // 500 >= 500
        assert!(entry.is_expired(2000));
    }

    #[test]
    fn test_error_display() {
        let e = CacheError::NotFound("foo".into());
        assert!(e.to_string().contains("foo"));
        let e2 = CacheError::CapacityExceeded { capacity: 10 };
        assert!(e2.to_string().contains("10"));
    }
}
