//! Caching strategies — write-through, write-back, write-around, read-through,
//! cache-aside patterns with TTL expiry, LFU eviction, tag-based invalidation,
//! stampede prevention (singleflight), and cache warming.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by cache strategy operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// The key was not found in cache.
    NotFound,
    /// The entry has expired.
    Expired,
    /// A flight for this key is already in progress (stampede guard).
    FlightInProgress,
    /// The backing store produced an error.
    StoreError(String),
    /// The cache has reached its capacity limit.
    CapacityExceeded,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "key not found in cache"),
            Self::Expired => write!(f, "cache entry expired"),
            Self::FlightInProgress => write!(f, "singleflight in progress for this key"),
            Self::StoreError(msg) => write!(f, "store error: {msg}"),
            Self::CapacityExceeded => write!(f, "cache capacity exceeded"),
        }
    }
}

// ── Write Policy ─────────────────────────────────────────────────────────────

/// Write policy for cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePolicy {
    /// Write to cache and backing store simultaneously.
    WriteThrough,
    /// Write to cache only; flush to store later.
    WriteBack,
    /// Write to backing store only; skip cache (read will repopulate on miss).
    WriteAround,
}

/// Read policy for cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPolicy {
    /// On miss, load from store and populate cache before returning.
    ReadThrough,
    /// Application manages cache population (cache-aside / lazy loading).
    CacheAside,
}

// ── Cache Entry ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: Instant,
    ttl: Duration,
    frequency: u64,
    tags: Vec<String>,
    dirty: bool,
}

impl<V> CacheEntry<V> {
    fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.created_at) > self.ttl
    }

    fn touch(&mut self, now: Instant) {
        self.last_accessed = now;
        self.frequency += 1;
    }
}

// ── Singleflight ─────────────────────────────────────────────────────────────

/// Tracks in-flight requests to prevent cache stampedes.
#[derive(Debug)]
struct Singleflight<K: Eq + Hash + Clone> {
    in_flight: HashMap<K, Instant>,
    timeout: Duration,
}

impl<K: Eq + Hash + Clone> Singleflight<K> {
    fn new(timeout: Duration) -> Self {
        Self {
            in_flight: HashMap::new(),
            timeout,
        }
    }

    fn try_acquire(&mut self, key: &K, now: Instant) -> bool {
        if let Some(started) = self.in_flight.get(key) {
            if now.duration_since(*started) < self.timeout {
                return false; // still in progress
            }
            // Timed out — allow new flight
        }
        self.in_flight.insert(key.clone(), now);
        true
    }

    fn release(&mut self, key: &K) {
        self.in_flight.remove(key);
    }

    fn is_in_flight(&self, key: &K, now: Instant) -> bool {
        if let Some(started) = self.in_flight.get(key) {
            now.duration_since(*started) < self.timeout
        } else {
            false
        }
    }
}

// ── Cache Statistics ─────────────────────────────────────────────────────────

/// Statistics for cache operations.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub stampede_blocks: u64,
    pub dirty_flushes: u64,
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
}

// ── StrategyCache ────────────────────────────────────────────────────────────

/// A cache implementing multiple write/read policies, LFU eviction, TTL,
/// tag-based invalidation, stampede prevention, and warming.
pub struct StrategyCache<K: Eq + Hash + Clone, V: Clone> {
    entries: HashMap<K, CacheEntry<V>>,
    capacity: usize,
    default_ttl: Duration,
    write_policy: WritePolicy,
    read_policy: ReadPolicy,
    singleflight: Singleflight<K>,
    stats: CacheStats,
    /// Simulated backing store (for testing / demonstration).
    store: HashMap<K, V>,
    /// Dirty keys awaiting flush (for write-back).
    dirty_keys: Vec<K>,
}

impl<K: Eq + Hash + Clone, V: Clone> StrategyCache<K, V> {
    /// Create a new cache with the given capacity, TTL, and policies.
    pub fn new(
        capacity: usize,
        default_ttl: Duration,
        write_policy: WritePolicy,
        read_policy: ReadPolicy,
    ) -> Self {
        assert!(capacity > 0, "cache capacity must be > 0");
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            default_ttl,
            write_policy,
            read_policy,
            singleflight: Singleflight::new(Duration::from_secs(5)),
            stats: CacheStats::default(),
            store: HashMap::new(),
            dirty_keys: Vec::new(),
        }
    }

    /// Create a write-through, read-through cache.
    pub fn write_through_read_through(capacity: usize, ttl: Duration) -> Self {
        Self::new(capacity, ttl, WritePolicy::WriteThrough, ReadPolicy::ReadThrough)
    }

    /// Create a write-back, read-through cache.
    pub fn write_back(capacity: usize, ttl: Duration) -> Self {
        Self::new(capacity, ttl, WritePolicy::WriteBack, ReadPolicy::ReadThrough)
    }

    /// Create a write-around, cache-aside cache.
    pub fn write_around(capacity: usize, ttl: Duration) -> Self {
        Self::new(capacity, ttl, WritePolicy::WriteAround, ReadPolicy::CacheAside)
    }

    /// Create a cache-aside cache (lazy caching).
    pub fn cache_aside(capacity: usize, ttl: Duration) -> Self {
        Self::new(capacity, ttl, WritePolicy::WriteThrough, ReadPolicy::CacheAside)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn write_policy(&self) -> WritePolicy {
        self.write_policy
    }

    pub fn read_policy(&self) -> ReadPolicy {
        self.read_policy
    }

    /// Seed the backing store (for read-through / testing).
    pub fn seed_store(&mut self, key: K, value: V) {
        self.store.insert(key, value);
    }

    /// Read from the cache with the configured read policy.
    pub fn get(&mut self, key: &K) -> Result<V, CacheError> {
        let now = Instant::now();

        // Check cache first.
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired(now) {
                // Remove expired entry.
                let k = key.clone();
                self.entries.remove(&k);
                self.stats.misses += 1;
            } else {
                entry.touch(now);
                self.stats.hits += 1;
                return Ok(entry.value.clone());
            }
        } else {
            self.stats.misses += 1;
        }

        // Cache miss — apply read policy.
        match self.read_policy {
            ReadPolicy::ReadThrough => {
                // Load from store and populate cache.
                let value = self.store.get(key).cloned().ok_or(CacheError::NotFound)?;
                self.insert_entry(key.clone(), value.clone(), now);
                Ok(value)
            }
            ReadPolicy::CacheAside => Err(CacheError::NotFound),
        }
    }

    /// Write a value using the configured write policy.
    pub fn put(&mut self, key: K, value: V) -> Result<(), CacheError> {
        let now = Instant::now();
        self.stats.writes += 1;

        match self.write_policy {
            WritePolicy::WriteThrough => {
                // Write to both cache and store.
                self.store.insert(key.clone(), value.clone());
                self.insert_entry(key, value, now);
            }
            WritePolicy::WriteBack => {
                // Write to cache only; mark dirty.
                self.insert_entry_dirty(key.clone(), value, now);
                self.dirty_keys.push(key);
            }
            WritePolicy::WriteAround => {
                // Write to store only; skip cache.
                self.store.insert(key, value);
            }
        }
        Ok(())
    }

    /// Write a value with a custom TTL.
    pub fn put_with_ttl(&mut self, key: K, value: V, ttl: Duration) -> Result<(), CacheError> {
        let now = Instant::now();
        self.stats.writes += 1;

        match self.write_policy {
            WritePolicy::WriteThrough => {
                self.store.insert(key.clone(), value.clone());
                self.insert_entry_ttl(key, value, now, ttl, false);
            }
            WritePolicy::WriteBack => {
                self.insert_entry_ttl(key.clone(), value, now, ttl, true);
                self.dirty_keys.push(key);
            }
            WritePolicy::WriteAround => {
                self.store.insert(key, value);
            }
        }
        Ok(())
    }

    /// Write a value with associated tags for group invalidation.
    pub fn put_tagged(&mut self, key: K, value: V, tags: Vec<String>) -> Result<(), CacheError> {
        let now = Instant::now();
        self.stats.writes += 1;

        let is_dirty = self.write_policy == WritePolicy::WriteBack;
        if self.write_policy != WritePolicy::WriteAround {
            if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
                self.evict_lfu();
            }
            self.entries.insert(
                key.clone(),
                CacheEntry {
                    value: value.clone(),
                    created_at: now,
                    last_accessed: now,
                    ttl: self.default_ttl,
                    frequency: 0,
                    tags,
                    dirty: is_dirty,
                },
            );
        }
        // Always write to store for write-through and write-around.
        if self.write_policy != WritePolicy::WriteBack {
            self.store.insert(key.clone(), value);
        }
        if is_dirty {
            self.dirty_keys.push(key);
        }
        Ok(())
    }

    /// Invalidate a specific key.
    pub fn invalidate(&mut self, key: &K) -> bool {
        if self.entries.remove(key).is_some() {
            self.stats.invalidations += 1;
            true
        } else {
            false
        }
    }

    /// Invalidate all entries with the given tag.
    pub fn invalidate_by_tag(&mut self, tag: &str) -> usize {
        let keys_to_remove: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.tags.iter().any(|t| t == tag))
            .map(|(k, _)| k.clone())
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        self.stats.invalidations += count as u64;
        count
    }

    /// Remove all expired entries.
    pub fn purge_expired(&mut self) -> usize {
        let now = Instant::now();
        let keys: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();
        let count = keys.len();
        for key in keys {
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
        count
    }

    /// Try to acquire a singleflight lock for a key (stampede prevention).
    pub fn try_flight(&mut self, key: &K) -> bool {
        let now = Instant::now();
        let acquired = self.singleflight.try_acquire(key, now);
        if !acquired {
            self.stats.stampede_blocks += 1;
        }
        acquired
    }

    /// Release the singleflight lock for a key.
    pub fn release_flight(&mut self, key: &K) {
        self.singleflight.release(key);
    }

    /// Check if a flight is currently in progress for a key.
    pub fn is_flight_in_progress(&self, key: &K) -> bool {
        self.singleflight.is_in_flight(key, Instant::now())
    }

    /// Flush dirty entries to the backing store (write-back only).
    pub fn flush_dirty(&mut self) -> usize {
        let dirty: Vec<K> = self.dirty_keys.drain(..).collect();
        let mut flushed = 0;
        for key in dirty {
            if let Some(entry) = self.entries.get_mut(&key) {
                if entry.dirty {
                    self.store.insert(key, entry.value.clone());
                    entry.dirty = false;
                    flushed += 1;
                }
            }
        }
        self.stats.dirty_flushes += flushed as u64;
        flushed
    }

    /// Return the number of dirty entries.
    pub fn dirty_count(&self) -> usize {
        self.entries.values().filter(|e| e.dirty).count()
    }

    /// Warm the cache with key/value pairs.
    pub fn warm(&mut self, items: impl IntoIterator<Item = (K, V)>) -> usize {
        let now = Instant::now();
        let mut count = 0;
        for (key, value) in items {
            self.insert_entry(key, value, now);
            count += 1;
        }
        count
    }

    /// Warm the cache from the backing store for the given keys.
    pub fn warm_from_store(&mut self, keys: &[K]) -> usize {
        let now = Instant::now();
        let mut count = 0;
        for key in keys {
            if let Some(value) = self.store.get(key).cloned() {
                self.insert_entry(key.clone(), value, now);
                count += 1;
            }
        }
        count
    }

    /// Clear the cache entirely.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty_keys.clear();
    }

    /// Clear the backing store.
    pub fn clear_store(&mut self) {
        self.store.clear();
    }

    /// Peek at a value without updating frequency or access time.
    pub fn peek(&self, key: &K) -> Option<&V> {
        let entry = self.entries.get(key)?;
        if entry.is_expired(Instant::now()) {
            return None;
        }
        Some(&entry.value)
    }

    /// Get the frequency count for a cached key.
    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.entries.get(key).map(|e| e.frequency)
    }

    /// Check if the backing store contains a key.
    pub fn store_contains(&self, key: &K) -> bool {
        self.store.contains_key(key)
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn insert_entry(&mut self, key: K, value: V, now: Instant) {
        self.insert_entry_ttl(key, value, now, self.default_ttl, false);
    }

    fn insert_entry_dirty(&mut self, key: K, value: V, now: Instant) {
        self.insert_entry_ttl(key, value, now, self.default_ttl, true);
    }

    fn insert_entry_ttl(&mut self, key: K, value: V, now: Instant, ttl: Duration, dirty: bool) {
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.evict_lfu();
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                created_at: now,
                last_accessed: now,
                ttl,
                frequency: 0,
                tags: Vec::new(),
                dirty,
            },
        );
    }

    /// Evict the least-frequently-used entry. Ties broken by oldest last access.
    fn evict_lfu(&mut self) {
        let victim = self
            .entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.frequency
                    .cmp(&b.frequency)
                    .then_with(|| a.last_accessed.cmp(&b.last_accessed))
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = victim {
            // If dirty, flush first.
            if let Some(entry) = self.entries.get(&key) {
                if entry.dirty {
                    self.store.insert(key.clone(), entry.value.clone());
                    self.stats.dirty_flushes += 1;
                }
            }
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ttl_1s() -> Duration {
        Duration::from_secs(1)
    }

    #[test]
    fn test_write_through_read_through() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.put("a".to_string(), 42).unwrap();
        // Should be in both cache and store.
        assert_eq!(cache.get(&"a".to_string()).unwrap(), 42);
        assert!(cache.store_contains(&"a".to_string()));
    }

    #[test]
    fn test_write_back_deferred_store() {
        let mut cache = StrategyCache::write_back(10, ttl_1s());
        cache.put("x".to_string(), 99).unwrap();
        // In cache but NOT in store yet.
        assert_eq!(cache.get(&"x".to_string()).unwrap(), 99);
        assert!(!cache.store_contains(&"x".to_string()));
        // Flush.
        let flushed = cache.flush_dirty();
        assert_eq!(flushed, 1);
        assert!(cache.store_contains(&"x".to_string()));
    }

    #[test]
    fn test_write_around_skips_cache() {
        let mut cache = StrategyCache::write_around(10, ttl_1s());
        cache.put("y".to_string(), 7).unwrap();
        // Should be in store but NOT in cache (cache-aside read policy).
        assert!(cache.store_contains(&"y".to_string()));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_read_through_populates_on_miss() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.seed_store("k".to_string(), 55);
        // First read is a "miss" that populates.
        let v = cache.get(&"k".to_string()).unwrap();
        assert_eq!(v, 55);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_aside_returns_not_found() {
        let mut cache = StrategyCache::cache_aside(10, ttl_1s());
        cache.seed_store("k".to_string(), 55);
        // Cache-aside does NOT auto-load from store.
        assert_eq!(cache.get(&"k".to_string()), Err(CacheError::NotFound));
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = StrategyCache::write_through_read_through(10, Duration::from_millis(0));
        cache.put("t".to_string(), 1).unwrap();
        // Entry expires immediately (0ms TTL).
        std::thread::sleep(Duration::from_millis(1));
        // Should get a miss, then re-read from store.
        let v = cache.get(&"t".to_string()).unwrap();
        assert_eq!(v, 1);
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = StrategyCache::write_through_read_through(2, ttl_1s());
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        // Access "a" to increase its frequency.
        let _ = cache.get(&"a".to_string());
        let _ = cache.get(&"a".to_string());
        // Adding "c" should evict "b" (lower frequency).
        cache.put("c".to_string(), 3).unwrap();
        assert_eq!(cache.len(), 2);
        // "a" should survive.
        assert_eq!(cache.get(&"a".to_string()).unwrap(), 1);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_invalidate_key() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.put("k".to_string(), 10).unwrap();
        assert!(cache.invalidate(&"k".to_string()));
        assert!(!cache.invalidate(&"k".to_string()));
        assert_eq!(cache.stats().invalidations, 1);
    }

    #[test]
    fn test_invalidate_by_tag() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache
            .put_tagged(
                "a".to_string(),
                1,
                vec!["product".to_string()],
            )
            .unwrap();
        cache
            .put_tagged(
                "b".to_string(),
                2,
                vec!["product".to_string(), "sale".to_string()],
            )
            .unwrap();
        cache
            .put_tagged("c".to_string(), 3, vec!["user".to_string()])
            .unwrap();
        let removed = cache.invalidate_by_tag("product");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_singleflight_prevents_stampede() {
        let mut cache = StrategyCache::<String, i32>::write_through_read_through(10, ttl_1s());
        let key = "expensive".to_string();
        assert!(cache.try_flight(&key));
        // Second attempt blocked.
        assert!(!cache.try_flight(&key));
        assert!(cache.is_flight_in_progress(&key));
        assert_eq!(cache.stats().stampede_blocks, 1);
        // Release.
        cache.release_flight(&key);
        assert!(!cache.is_flight_in_progress(&key));
        // Now can acquire again.
        assert!(cache.try_flight(&key));
    }

    #[test]
    fn test_warm_cache() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        let items = vec![
            ("a".to_string(), 1),
            ("b".to_string(), 2),
            ("c".to_string(), 3),
        ];
        let warmed = cache.warm(items);
        assert_eq!(warmed, 3);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_warm_from_store() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.seed_store("x".to_string(), 10);
        cache.seed_store("y".to_string(), 20);
        let keys = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let warmed = cache.warm_from_store(&keys);
        assert_eq!(warmed, 2); // "z" not in store
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_dirty_count() {
        let mut cache = StrategyCache::write_back(10, ttl_1s());
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        assert_eq!(cache.dirty_count(), 2);
        cache.flush_dirty();
        assert_eq!(cache.dirty_count(), 0);
    }

    #[test]
    fn test_purge_expired() {
        let mut cache = StrategyCache::write_through_read_through(10, Duration::from_millis(0));
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let purged = cache.purge_expired();
        assert_eq!(purged, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_peek_does_not_update() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.put("k".to_string(), 42).unwrap();
        assert_eq!(cache.peek(&"k".to_string()), Some(&42));
        // Frequency should still be 0 after peek.
        assert_eq!(cache.frequency(&"k".to_string()), Some(0));
    }

    #[test]
    fn test_hit_rate_statistics() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.put("a".to_string(), 1).unwrap();
        let _ = cache.get(&"a".to_string()); // hit
        let _ = cache.get(&"b".to_string()); // miss (not found in store either)
        let _ = cache.get(&"a".to_string()); // hit
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 1);
        let rate = cache.stats().hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_custom_ttl_per_key() {
        let mut cache = StrategyCache::write_through_read_through(10, Duration::from_secs(60));
        cache
            .put_with_ttl("fast".to_string(), 1, Duration::from_millis(0))
            .unwrap();
        cache
            .put_with_ttl("slow".to_string(), 2, Duration::from_secs(60))
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));
        // "fast" expired, "slow" still valid.
        let fast = cache.get(&"fast".to_string()); // will re-read from store
        assert_eq!(fast.unwrap(), 1);
        assert_eq!(cache.get(&"slow".to_string()).unwrap(), 2);
    }

    #[test]
    fn test_clear() {
        let mut cache = StrategyCache::write_through_read_through(10, ttl_1s());
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_write_back_eviction_flushes_dirty() {
        let mut cache = StrategyCache::write_back(2, ttl_1s());
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        // Both dirty. Adding "c" should evict one and flush it.
        cache.put("c".to_string(), 3).unwrap();
        // The evicted entry should have been flushed to store.
        assert_eq!(cache.stats().dirty_flushes, 1);
        assert_eq!(cache.len(), 2);
    }
}
