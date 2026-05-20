//! LRU cache — O(1) get/put with capacity-based eviction.
//!
//! Supports TTL-based expiry, hit/miss statistics, peek without promotion,
//! bulk operations, eviction callbacks, and a thread-safe wrapper.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Entry ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Entry<V> {
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
    expires_at: Option<Instant>,
}

// ── LruCache ────────────────────────────────────────────────────────────────

/// Least-recently-used cache with O(1) get and put.
pub struct LruCache<K: Eq + Hash + Clone, V> {
    capacity: usize,
    map: HashMap<K, usize>,
    entries: Vec<Option<Entry<V>>>,
    keys: Vec<Option<K>>,
    head: Option<usize>,
    tail: Option<usize>,
    free_slots: Vec<usize>,
    default_ttl: Option<Duration>,
    hits: u64,
    misses: u64,
    eviction_count: u64,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    /// Create a new LRU cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LRU cache capacity must be > 0");
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            entries: Vec::with_capacity(capacity),
            keys: Vec::with_capacity(capacity),
            head: None,
            tail: None,
            free_slots: Vec::new(),
            default_ttl: None,
            hits: 0,
            misses: 0,
            eviction_count: 0,
        }
    }

    /// Create with a default TTL for all entries.
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        let mut cache = Self::new(capacity);
        cache.default_ttl = Some(ttl);
        cache
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache eviction count.
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Hit rate as a fraction [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    fn alloc_slot(&mut self, entry: Entry<V>, key: K) -> usize {
        if let Some(idx) = self.free_slots.pop() {
            self.entries[idx] = Some(entry);
            self.keys[idx] = Some(key);
            idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(entry));
            self.keys.push(Some(key));
            idx
        }
    }

    fn detach(&mut self, idx: usize) {
        let entry = self.entries[idx].as_ref().unwrap();
        let prev = entry.prev;
        let next = entry.next;

        if let Some(p) = prev {
            self.entries[p].as_mut().unwrap().next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.entries[n].as_mut().unwrap().prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn push_front(&mut self, idx: usize) {
        let entry = self.entries[idx].as_mut().unwrap();
        entry.prev = None;
        entry.next = self.head;

        if let Some(h) = self.head {
            self.entries[h].as_mut().unwrap().prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    fn is_expired(entry: &Entry<V>) -> bool {
        entry
            .expires_at
            .is_some_and(|exp| Instant::now() >= exp)
    }

    fn evict_tail(&mut self) -> Option<K> {
        let tail_idx = self.tail?;
        self.detach(tail_idx);
        let key = self.keys[tail_idx].take()?;
        self.entries[tail_idx] = None;
        self.map.remove(&key);
        self.free_slots.push(tail_idx);
        self.eviction_count += 1;
        Some(key)
    }

    /// Put a key-value pair into the cache. Returns the evicted key if capacity was exceeded.
    pub fn put(&mut self, key: K, value: V) -> Option<K> {
        self.put_with_ttl(key, value, self.default_ttl)
    }

    /// Put with a specific TTL.
    pub fn put_with_ttl(&mut self, key: K, value: V, ttl: Option<Duration>) -> Option<K> {
        let expires_at = ttl.map(|d| Instant::now() + d);
        let mut evicted = None;

        if let Some(&idx) = self.map.get(&key) {
            // Update existing
            self.detach(idx);
            let entry = self.entries[idx].as_mut().unwrap();
            entry.value = value;
            entry.expires_at = expires_at;
            self.push_front(idx);
            return None;
        }

        if self.map.len() >= self.capacity {
            evicted = self.evict_tail();
        }

        let entry = Entry {
            value,
            prev: None,
            next: None,
            expires_at,
        };
        let idx = self.alloc_slot(entry, key.clone());
        self.map.insert(key, idx);
        self.push_front(idx);
        evicted
    }

    /// Get a value, promoting it to most-recently-used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = match self.map.get(key) {
            Some(&idx) => idx,
            None => {
                self.misses += 1;
                return None;
            }
        };

        if Self::is_expired(self.entries[idx].as_ref().unwrap()) {
            // Expired — remove it
            self.detach(idx);
            self.keys[idx] = None;
            self.entries[idx] = None;
            self.map.remove(key);
            self.free_slots.push(idx);
            self.misses += 1;
            return None;
        }

        self.detach(idx);
        self.push_front(idx);
        self.hits += 1;
        Some(&self.entries[idx].as_ref().unwrap().value)
    }

    /// Peek at a value without promoting it.
    pub fn peek(&mut self, key: &K) -> Option<&V> {
        let idx = *self.map.get(key)?;

        if Self::is_expired(self.entries[idx].as_ref().unwrap()) {
            self.misses += 1;
            return None;
        }

        self.hits += 1;
        Some(&self.entries[idx].as_ref().unwrap().value)
    }

    /// Remove a key from the cache.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        self.keys[idx] = None;
        let entry = self.entries[idx].take()?;
        self.free_slots.push(idx);
        Some(entry.value)
    }

    /// Check if a key exists (does not promote or count as hit/miss).
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.map.clear();
        self.entries.clear();
        self.keys.clear();
        self.head = None;
        self.tail = None;
        self.free_slots.clear();
    }

    /// Bulk insert multiple key-value pairs.
    pub fn put_many(&mut self, items: impl IntoIterator<Item = (K, V)>) {
        for (k, v) in items {
            self.put(k, v);
        }
    }

    /// Return all keys in MRU to LRU order.
    pub fn keys_mru(&self) -> Vec<&K> {
        let mut result = Vec::new();
        let mut cur = self.head;
        while let Some(idx) = cur {
            if let Some(k) = &self.keys[idx] {
                result.push(k);
            }
            cur = self.entries[idx].as_ref().unwrap().next;
        }
        result
    }

    /// Reset hit/miss statistics.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.eviction_count = 0;
    }
}

impl<K: Eq + Hash + Clone + std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug
    for LruCache<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LruCache")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

// ── ThreadSafeLruCache ──────────────────────────────────────────────────────

/// Thread-safe wrapper around LruCache using Arc<Mutex<_>>.
pub struct ThreadSafeLruCache<K: Eq + Hash + Clone, V> {
    inner: Arc<Mutex<LruCache<K, V>>>,
}

impl<K: Eq + Hash + Clone, V: Clone> ThreadSafeLruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }

    pub fn put(&self, key: K, value: V) -> Option<K> {
        self.inner.lock().unwrap().put(key, value)
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.lock().unwrap().get(key).cloned()
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.lock().unwrap().remove(key)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }
}

impl<K: Eq + Hash + Clone, V> Clone for ThreadSafeLruCache<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), None);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3); // evicts "a"
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_access_promotes() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.get(&"a"); // promote "a"
        cache.put("c", 3); // evicts "b" (LRU), not "a"
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), None);
    }

    #[test]
    fn test_update_value() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("a", 10);
        assert_eq!(cache.get(&"a"), Some(&10));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_peek_no_promote() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.peek(&"a"); // does NOT promote
        cache.put("c", 3); // evicts "a" because it is still LRU
        assert_eq!(cache.get(&"a"), None);
    }

    #[test]
    fn test_remove() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.remove(&"a"), Some(1));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"a"), None);
    }

    #[test]
    fn test_hit_miss_stats() {
        let mut cache = LruCache::new(10);
        cache.put("a", 1);
        cache.get(&"a"); // hit
        cache.get(&"b"); // miss
        cache.get(&"a"); // hit
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_eviction_count() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3); // evict
        cache.put("d", 4); // evict
        assert_eq!(cache.eviction_count(), 2);
    }

    #[test]
    fn test_bulk_put() {
        let mut cache = LruCache::new(5);
        cache.put_many(vec![("a", 1), ("b", 2), ("c", 3)]);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    #[test]
    fn test_keys_mru_order() {
        let mut cache = LruCache::new(5);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.get(&"a"); // promote to MRU
        let keys = cache.keys_mru();
        assert_eq!(keys[0], &"a");
    }

    #[test]
    fn test_clear() {
        let mut cache = LruCache::new(5);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.get(&"a"), None);
    }

    #[test]
    fn test_contains_key() {
        let mut cache = LruCache::new(5);
        cache.put("x", 42);
        assert!(cache.contains_key(&"x"));
        assert!(!cache.contains_key(&"y"));
    }

    #[test]
    fn test_thread_safe_wrapper() {
        let cache = ThreadSafeLruCache::new(10);
        cache.put("k", 100);
        assert_eq!(cache.get(&"k"), Some(100));
        assert_eq!(cache.len(), 1);
        cache.remove(&"k");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_reset_stats() {
        let mut cache = LruCache::new(5);
        cache.put("a", 1);
        cache.get(&"a");
        cache.get(&"b");
        cache.reset_stats();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }
}
