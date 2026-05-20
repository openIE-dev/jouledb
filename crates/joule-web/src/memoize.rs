//! Memoization utilities — single-arg, multi-arg, LRU, TTL, cache statistics.
//!
//! Replaces memoize-one, lodash.memoize, lru-cache, and micro-memoize with
//! pure-Rust memoization primitives supporting LRU eviction, TTL expiry,
//! cache stats, and selective invalidation.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

// ── Cache Statistics ────────────────────────────────────────────

/// Statistics for cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }
}

// ── Simple Memo ─────────────────────────────────────────────────

/// Simple single-key memoization cache (unbounded).
#[derive(Debug)]
pub struct Memo<K, V> {
    cache: HashMap<K, V>,
    stats: CacheStats,
}

impl<K: Eq + Hash, V: Clone> Memo<K, V> {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Get or compute a value.
    pub fn get_or_insert_with(&mut self, key: K, f: impl FnOnce(&K) -> V) -> V
    where
        K: Clone,
    {
        if let Some(v) = self.cache.get(&key) {
            self.stats.hits += 1;
            return v.clone();
        }
        self.stats.misses += 1;
        let value = f(&key);
        self.cache.insert(key, value.clone());
        value
    }

    /// Get a cached value without computing.
    pub fn get(&mut self, key: &K) -> Option<V> {
        match self.cache.get(key) {
            Some(v) => {
                self.stats.hits += 1;
                Some(v.clone())
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Invalidate a specific key.
    pub fn invalidate(&mut self, key: &K) -> bool {
        self.cache.remove(key).is_some()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

impl<K: Eq + Hash, V: Clone> Default for Memo<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── LRU Memo ────────────────────────────────────────────────────

/// Entry in the LRU cache.
#[derive(Debug, Clone)]
struct LruEntry<V> {
    value: V,
    /// Access counter for LRU ordering.
    last_access: u64,
}

/// LRU cache-backed memoization.
#[derive(Debug)]
pub struct LruMemo<K, V> {
    cache: HashMap<K, LruEntry<V>>,
    capacity: usize,
    access_counter: u64,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone, V: Clone> LruMemo<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            access_counter: 0,
            stats: CacheStats::default(),
        }
    }

    /// Get or compute a value with LRU eviction.
    pub fn get_or_insert_with(&mut self, key: K, f: impl FnOnce(&K) -> V) -> V {
        self.access_counter += 1;
        let counter = self.access_counter;

        if let Some(entry) = self.cache.get_mut(&key) {
            entry.last_access = counter;
            self.stats.hits += 1;
            return entry.value.clone();
        }

        self.stats.misses += 1;
        let value = f(&key);

        // Evict LRU if at capacity.
        if self.cache.len() >= self.capacity {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            LruEntry {
                value: value.clone(),
                last_access: counter,
            },
        );

        value
    }

    /// Get a cached value.
    pub fn get(&mut self, key: &K) -> Option<V> {
        self.access_counter += 1;
        let counter = self.access_counter;

        match self.cache.get_mut(key) {
            Some(entry) => {
                entry.last_access = counter;
                self.stats.hits += 1;
                Some(entry.value.clone())
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    pub fn invalidate(&mut self, key: &K) -> bool {
        self.cache.remove(key).is_some()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .cache
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(k, _)| k.clone())
        {
            self.cache.remove(&lru_key);
            self.stats.evictions += 1;
        }
    }
}

// ── TTL Memo ────────────────────────────────────────────────────

/// Entry with expiration time.
#[derive(Debug, Clone)]
struct TtlEntry<V> {
    value: V,
    expires_at: Instant,
}

/// TTL-based memoization cache.
#[derive(Debug)]
pub struct TtlMemo<K, V> {
    cache: HashMap<K, TtlEntry<V>>,
    ttl: Duration,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone, V: Clone> TtlMemo<K, V> {
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            ttl,
            stats: CacheStats::default(),
        }
    }

    /// Get or compute a value, respecting TTL.
    pub fn get_or_insert_with(&mut self, key: K, now: Instant, f: impl FnOnce(&K) -> V) -> V {
        // Check for existing non-expired entry.
        if let Some(entry) = self.cache.get(&key) {
            if now < entry.expires_at {
                self.stats.hits += 1;
                return entry.value.clone();
            } else {
                self.stats.expirations += 1;
            }
        }

        self.stats.misses += 1;
        let value = f(&key);
        self.cache.insert(
            key,
            TtlEntry {
                value: value.clone(),
                expires_at: now + self.ttl,
            },
        );
        value
    }

    /// Get a cached value if it hasn't expired.
    pub fn get(&mut self, key: &K, now: Instant) -> Option<V> {
        match self.cache.get(key) {
            Some(entry) => {
                if now < entry.expires_at {
                    self.stats.hits += 1;
                    Some(entry.value.clone())
                } else {
                    self.stats.expirations += 1;
                    self.stats.misses += 1;
                    self.cache.remove(key);
                    None
                }
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Remove expired entries.
    pub fn prune(&mut self, now: Instant) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, entry| now < entry.expires_at);
        let removed = before - self.cache.len();
        self.stats.expirations += removed as u64;
        removed
    }

    pub fn invalidate(&mut self, key: &K) -> bool {
        self.cache.remove(key).is_some()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

// ── Weak Memo ───────────────────────────────────────────────────

/// Memoization using weak references for large values.
///
/// Values are stored via `std::sync::Arc`, and the cache holds weak refs.
/// If all strong references are dropped, the value is automatically reclaimed.
#[derive(Debug)]
pub struct WeakMemo<K, V> {
    cache: HashMap<K, std::sync::Weak<V>>,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone, V> WeakMemo<K, V> {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Try to get a cached value. Returns None if expired or missing.
    pub fn get(&mut self, key: &K) -> Option<std::sync::Arc<V>> {
        match self.cache.get(key) {
            Some(weak) => match weak.upgrade() {
                Some(arc) => {
                    self.stats.hits += 1;
                    Some(arc)
                }
                None => {
                    self.cache.remove(key);
                    self.stats.misses += 1;
                    None
                }
            },
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert a value, returning an Arc to it.
    pub fn insert(&mut self, key: K, value: V) -> std::sync::Arc<V> {
        let arc = std::sync::Arc::new(value);
        self.cache.insert(key, std::sync::Arc::downgrade(&arc));
        arc
    }

    /// Get or insert.
    pub fn get_or_insert_with(&mut self, key: K, f: impl FnOnce(&K) -> V) -> std::sync::Arc<V> {
        if let Some(arc) = self.get(&key) {
            return arc;
        }
        self.stats.misses = self.stats.misses.saturating_sub(1); // get() already counted a miss
        self.stats.misses += 1;
        let value = f(&key);
        self.insert(key, value)
    }

    /// Remove dead weak refs.
    pub fn prune(&mut self) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, weak| weak.strong_count() > 0);
        before - self.cache.len()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

impl<K: Eq + Hash + Clone, V> Default for WeakMemo<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_basic() {
        let mut memo = Memo::<u32, u64>::new();
        let val = memo.get_or_insert_with(5, |k| (*k as u64) * (*k as u64));
        assert_eq!(val, 25);

        // Second call should hit cache.
        let val2 = memo.get_or_insert_with(5, |_| panic!("should not be called"));
        assert_eq!(val2, 25);
        assert_eq!(memo.stats().hits, 1);
        assert_eq!(memo.stats().misses, 1);
    }

    #[test]
    fn memo_invalidate() {
        let mut memo = Memo::<&str, i32>::new();
        memo.get_or_insert_with("x", |_| 42);
        assert!(memo.invalidate(&"x"));
        assert!(memo.get(&"x").is_none());
    }

    #[test]
    fn memo_clear() {
        let mut memo = Memo::<i32, i32>::new();
        memo.get_or_insert_with(1, |_| 10);
        memo.get_or_insert_with(2, |_| 20);
        memo.clear();
        assert!(memo.is_empty());
    }

    #[test]
    fn lru_eviction() {
        let mut lru = LruMemo::<&str, i32>::new(2);
        lru.get_or_insert_with("a", |_| 1);
        lru.get_or_insert_with("b", |_| 2);
        // "a" was accessed first, so it's the LRU.
        lru.get_or_insert_with("c", |_| 3);
        assert_eq!(lru.len(), 2);
        assert!(lru.get(&"a").is_none()); // evicted
        assert_eq!(lru.stats().evictions, 1);
    }

    #[test]
    fn lru_access_updates_order() {
        let mut lru = LruMemo::<&str, i32>::new(2);
        lru.get_or_insert_with("a", |_| 1);
        lru.get_or_insert_with("b", |_| 2);
        // Access "a" to make it recently used.
        lru.get(&"a");
        // Insert "c" — should evict "b" (now LRU).
        lru.get_or_insert_with("c", |_| 3);
        assert!(lru.get(&"b").is_none());
        assert!(lru.get(&"a").is_some());
    }

    #[test]
    fn ttl_basic() {
        let start = Instant::now();
        let mut ttl = TtlMemo::<&str, i32>::new(Duration::from_secs(1));

        ttl.get_or_insert_with("x", start, |_| 42);
        // Still valid.
        assert_eq!(ttl.get(&"x", start + Duration::from_millis(500)), Some(42));
        // Expired.
        assert!(ttl.get(&"x", start + Duration::from_secs(2)).is_none());
    }

    #[test]
    fn ttl_recompute_after_expiry() {
        let start = Instant::now();
        let mut ttl = TtlMemo::<&str, i32>::new(Duration::from_millis(100));

        ttl.get_or_insert_with("k", start, |_| 1);
        let val = ttl.get_or_insert_with("k", start + Duration::from_millis(200), |_| 2);
        assert_eq!(val, 2); // Recomputed.
    }

    #[test]
    fn ttl_prune() {
        let start = Instant::now();
        let mut ttl = TtlMemo::<i32, i32>::new(Duration::from_millis(100));

        ttl.get_or_insert_with(1, start, |_| 10);
        ttl.get_or_insert_with(2, start, |_| 20);
        ttl.get_or_insert_with(3, start + Duration::from_millis(200), |_| 30);

        let removed = ttl.prune(start + Duration::from_millis(150));
        assert_eq!(removed, 2);
        assert_eq!(ttl.len(), 1);
    }

    #[test]
    fn weak_memo_basic() {
        let mut weak = WeakMemo::<&str, Vec<u8>>::new();
        let arc = weak.insert("big", vec![1, 2, 3]);
        assert_eq!(arc.len(), 3);

        // Should be retrievable while arc is alive.
        let retrieved = weak.get(&"big");
        assert!(retrieved.is_some());

        // Drop the strong reference.
        drop(arc);
        drop(retrieved);
        let gone = weak.get(&"big");
        assert!(gone.is_none());
    }

    #[test]
    fn weak_memo_prune() {
        let mut weak = WeakMemo::<i32, String>::new();
        let _a = weak.insert(1, "hello".into());
        let b = weak.insert(2, "world".into());
        drop(_a);
        let removed = weak.prune();
        assert_eq!(removed, 1);
        assert_eq!(weak.len(), 1);
        drop(b);
    }

    #[test]
    fn cache_stats_hit_rate() {
        let mut memo = Memo::<i32, i32>::new();
        memo.get_or_insert_with(1, |_| 10); // miss
        memo.get_or_insert_with(1, |_| 10); // hit
        memo.get_or_insert_with(1, |_| 10); // hit
        assert!((memo.stats().hit_rate() - 0.666).abs() < 0.01);
        assert_eq!(memo.stats().total_lookups(), 3);
    }

    #[test]
    fn multi_arg_with_tuple_key() {
        let mut memo = Memo::<(i32, i32), i32>::new();
        let val = memo.get_or_insert_with((3, 4), |(a, b)| a + b);
        assert_eq!(val, 7);
        let val2 = memo.get_or_insert_with((3, 4), |_| panic!("cached"));
        assert_eq!(val2, 7);
    }

    #[test]
    fn lru_capacity() {
        let lru = LruMemo::<i32, i32>::new(10);
        assert_eq!(lru.capacity(), 10);
        assert!(lru.is_empty());
    }

    #[test]
    fn selective_invalidation() {
        let mut memo = Memo::<&str, i32>::new();
        memo.get_or_insert_with("a", |_| 1);
        memo.get_or_insert_with("b", |_| 2);
        memo.get_or_insert_with("c", |_| 3);

        memo.invalidate(&"b");
        assert_eq!(memo.len(), 2);
        assert!(memo.get(&"b").is_none());
        assert!(memo.get(&"a").is_some());
    }

    #[test]
    fn weak_memo_get_or_insert() {
        let mut weak = WeakMemo::<&str, i32>::new();
        let arc = weak.get_or_insert_with("key", |_| 99);
        assert_eq!(*arc, 99);
        // Second call should return cached.
        let arc2 = weak.get_or_insert_with("key", |_| 100);
        assert_eq!(*arc2, 99);
        drop(arc);
        drop(arc2);
    }
}
