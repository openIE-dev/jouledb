//! Game resource caching with LRU eviction.
//!
//! Generic typed cache: store/retrieve by key, LRU eviction when memory budget
//! exceeded, per-entry size tracking, hit/miss/eviction statistics, preload hints,
//! priority-pinned entries, cache warming, and time-to-live expiry.

use std::collections::HashMap;
use std::fmt;

// ── Cache priority ─────────────────────────────────────────────

/// Priority level for a cache entry. Pinned entries are never evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CachePriority {
    Low,
    Normal,
    High,
    Pinned,
}

impl CachePriority {
    pub fn is_evictable(&self) -> bool {
        *self != CachePriority::Pinned
    }
}

impl fmt::Display for CachePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CachePriority::Low => write!(f, "low"),
            CachePriority::Normal => write!(f, "normal"),
            CachePriority::High => write!(f, "high"),
            CachePriority::Pinned => write!(f, "pinned"),
        }
    }
}

// ── Cache statistics ───────────────────────────────────────────

/// Aggregate cache statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub memory_used: usize,
    pub memory_budget: usize,
    pub entry_count: usize,
}

impl CacheStats {
    /// Hit rate as a ratio in [0, 1]. Returns 0 if no accesses.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ── Internal entry ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheEntry<V: Clone> {
    value: V,
    size_bytes: usize,
    priority: CachePriority,
    access_order: u64,
    insert_time: u64,
    ttl: Option<u64>,
}

impl<V: Clone> CacheEntry<V> {
    fn is_expired(&self, now: u64) -> bool {
        if let Some(ttl) = self.ttl {
            now.saturating_sub(self.insert_time) > ttl
        } else {
            false
        }
    }
}

// ── Resource cache ─────────────────────────────────────────────

/// Generic typed cache with LRU eviction, pinning, TTL, and statistics.
pub struct ResourceCache<K: Eq + std::hash::Hash + Clone + fmt::Debug, V: Clone> {
    entries: HashMap<K, CacheEntry<V>>,
    memory_budget: usize,
    memory_used: usize,
    access_counter: u64,
    current_time: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    preload_hints: Vec<K>,
}

impl<K: Eq + std::hash::Hash + Clone + fmt::Debug, V: Clone> fmt::Debug for ResourceCache<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceCache")
            .field("entries", &self.entries.len())
            .field("memory", &self.memory_used)
            .field("budget", &self.memory_budget)
            .finish()
    }
}

impl<K: Eq + std::hash::Hash + Clone + fmt::Debug, V: Clone> ResourceCache<K, V> {
    /// Create a cache with the given memory budget in bytes.
    pub fn new(memory_budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            memory_budget,
            memory_used: 0,
            access_counter: 0,
            current_time: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            preload_hints: Vec::new(),
        }
    }

    /// Advance the logical clock (for TTL expiry).
    pub fn tick(&mut self, delta: u64) {
        self.current_time += delta;
    }

    /// Set the logical time directly.
    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    /// Insert a value with estimated size and priority.
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize, priority: CachePriority) {
        self.insert_with_ttl(key, value, size_bytes, priority, None);
    }

    /// Insert with a time-to-live (in logical time units).
    pub fn insert_with_ttl(
        &mut self,
        key: K,
        value: V,
        size_bytes: usize,
        priority: CachePriority,
        ttl: Option<u64>,
    ) {
        // If key exists, remove old entry first.
        if let Some(old) = self.entries.remove(&key) {
            self.memory_used = self.memory_used.saturating_sub(old.size_bytes);
        }

        // Evict until we have room (or no evictable entries remain).
        while self.memory_used + size_bytes > self.memory_budget {
            if !self.evict_lru() {
                break;
            }
        }

        self.access_counter += 1;
        let entry = CacheEntry {
            value,
            size_bytes,
            priority,
            access_order: self.access_counter,
            insert_time: self.current_time,
            ttl,
        };
        self.memory_used += size_bytes;
        self.entries.insert(key, entry);
    }

    /// Retrieve a value by key, updating LRU order. Returns None on miss or expired.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        // Check expiry first.
        let expired = self.entries.get(key).map(|e| e.is_expired(self.current_time)).unwrap_or(false);
        if expired {
            self.remove(key);
            self.misses += 1;
            return None;
        }
        self.access_counter += 1;
        let counter = self.access_counter;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_order = counter;
            self.hits += 1;
            Some(&entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Peek at a value without updating LRU order or statistics.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|e| &e.value)
    }

    /// Remove an entry by key.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.memory_used = self.memory_used.saturating_sub(entry.size_bytes);
            Some(entry.value)
        } else {
            None
        }
    }

    /// Whether the cache contains a key (does not count as hit/miss).
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Evict the least-recently-used non-pinned entry. Returns true if something was evicted.
    fn evict_lru(&mut self) -> bool {
        let victim_key = {
            let mut best: Option<(&K, u64)> = None;
            for (k, e) in &self.entries {
                if !e.priority.is_evictable() {
                    continue;
                }
                match best {
                    None => best = Some((k, e.access_order)),
                    Some((_, bo)) if e.access_order < bo => best = Some((k, e.access_order)),
                    _ => {}
                }
            }
            best.map(|(k, _)| k.clone())
        };
        if let Some(key) = victim_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.memory_used = self.memory_used.saturating_sub(entry.size_bytes);
                self.evictions += 1;
                return true;
            }
        }
        false
    }

    /// Expire all entries that have exceeded their TTL.
    pub fn expire_stale(&mut self) -> usize {
        let now = self.current_time;
        let expired_keys: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for k in expired_keys {
            self.remove(&k);
        }
        count
    }

    /// Add a preload hint (key to load before it is needed).
    pub fn add_preload_hint(&mut self, key: K) {
        if !self.preload_hints.contains(&key) {
            self.preload_hints.push(key);
        }
    }

    /// Drain preload hints.
    pub fn take_preload_hints(&mut self) -> Vec<K> {
        std::mem::take(&mut self.preload_hints)
    }

    /// Set priority for an existing entry.
    pub fn set_priority(&mut self, key: &K, priority: CachePriority) -> bool {
        if let Some(e) = self.entries.get_mut(key) {
            e.priority = priority;
            true
        } else {
            false
        }
    }

    /// Pin an entry (never evict).
    pub fn pin(&mut self, key: &K) -> bool {
        self.set_priority(key, CachePriority::Pinned)
    }

    /// Unpin an entry (make evictable again at Normal priority).
    pub fn unpin(&mut self, key: &K) -> bool {
        self.set_priority(key, CachePriority::Normal)
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory currently used.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Memory budget.
    pub fn memory_budget(&self) -> usize {
        self.memory_budget
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            memory_used: self.memory_used,
            memory_budget: self.memory_budget,
            entry_count: self.entries.len(),
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.memory_used = 0;
    }

    /// Warm the cache by inserting a batch of (key, value, size) tuples at Normal priority.
    pub fn warm(&mut self, items: Vec<(K, V, usize)>) {
        for (k, v, sz) in items {
            self.insert(k, v, sz, CachePriority::Normal);
        }
    }

    /// Resize the memory budget. May trigger evictions.
    pub fn set_budget(&mut self, budget: usize) {
        self.memory_budget = budget;
        while self.memory_used > self.memory_budget {
            if !self.evict_lru() {
                break;
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut cache: ResourceCache<String, u32> = ResourceCache::new(1000);
        cache.insert("a".into(), 42, 100, CachePriority::Normal);
        assert_eq!(cache.get(&"a".into()), Some(&42));
    }

    #[test]
    fn miss_tracking() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn hit_tracking() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        cache.get(&1);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn lru_eviction() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(250);
        cache.insert(1, 10, 100, CachePriority::Normal);
        cache.insert(2, 20, 100, CachePriority::Normal);
        // Access 1 to make it more recent.
        cache.get(&1);
        // This should evict 2 (least recently used).
        cache.insert(3, 30, 100, CachePriority::Normal);
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn pinned_never_evicted() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(200);
        cache.insert(1, 10, 100, CachePriority::Pinned);
        cache.insert(2, 20, 100, CachePriority::Normal);
        // Insert third — should evict 2 (not 1 which is pinned).
        cache.insert(3, 30, 100, CachePriority::Normal);
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
    }

    #[test]
    fn remove_entry() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        let v = cache.remove(&1);
        assert_eq!(v, Some(10));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_used(), 0);
    }

    #[test]
    fn ttl_expiry() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert_with_ttl(1, 10, 100, CachePriority::Normal, Some(5));
        assert!(cache.contains(&1));
        cache.set_time(6);
        // Access should see expired entry.
        assert_eq!(cache.get(&1), None);
        assert!(!cache.contains(&1));
    }

    #[test]
    fn expire_stale_batch() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert_with_ttl(1, 10, 100, CachePriority::Normal, Some(3));
        cache.insert_with_ttl(2, 20, 100, CachePriority::Normal, Some(3));
        cache.insert(3, 30, 100, CachePriority::Normal); // no TTL
        cache.set_time(4);
        let expired = cache.expire_stale();
        assert_eq!(expired, 2);
        assert!(!cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
    }

    #[test]
    fn peek_does_not_affect_stats() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        let _ = cache.peek(&1);
        let _ = cache.peek(&99);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn preload_hints() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.add_preload_hint(1);
        cache.add_preload_hint(2);
        cache.add_preload_hint(1); // duplicate ignored
        let hints = cache.take_preload_hints();
        assert_eq!(hints, vec![1, 2]);
        assert!(cache.take_preload_hints().is_empty());
    }

    #[test]
    fn pin_and_unpin() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(200);
        cache.insert(1, 10, 100, CachePriority::Normal);
        assert!(cache.pin(&1));
        cache.insert(2, 20, 100, CachePriority::Normal);
        // Inserting third should evict 2, not pinned 1.
        cache.insert(3, 30, 100, CachePriority::Normal);
        assert!(cache.contains(&1));
        assert!(cache.unpin(&1));
    }

    #[test]
    fn warm_cache() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(10000);
        cache.warm(vec![(1, 10, 100), (2, 20, 100), (3, 30, 100)]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn clear_cache() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        cache.insert(2, 20, 100, CachePriority::Normal);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_used(), 0);
    }

    #[test]
    fn set_budget_triggers_eviction() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 400, CachePriority::Normal);
        cache.insert(2, 20, 400, CachePriority::Normal);
        cache.set_budget(500);
        // One entry evicted to fit within 500.
        assert_eq!(cache.len(), 1);
        assert!(cache.memory_used() <= 500);
    }

    #[test]
    fn hit_rate_calculation() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        cache.get(&1); // hit
        cache.get(&2); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hit_rate_zero_accesses() {
        let cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        assert!((cache.stats().hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn overwrite_existing_key() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        cache.insert(1, 10, 100, CachePriority::Normal);
        cache.insert(1, 20, 200, CachePriority::Normal);
        assert_eq!(cache.get(&1), Some(&20));
        assert_eq!(cache.memory_used(), 200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn priority_display() {
        assert_eq!(CachePriority::Pinned.to_string(), "pinned");
        assert_eq!(CachePriority::Low.to_string(), "low");
    }

    #[test]
    fn memory_budget_getter() {
        let cache: ResourceCache<u32, u32> = ResourceCache::new(4096);
        assert_eq!(cache.memory_budget(), 4096);
    }

    #[test]
    fn is_empty() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        assert!(cache.is_empty());
        cache.insert(1, 10, 10, CachePriority::Normal);
        assert!(!cache.is_empty());
    }

    #[test]
    fn eviction_with_all_pinned_does_not_loop() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(100);
        cache.insert(1, 10, 60, CachePriority::Pinned);
        // Adding another that exceeds budget — eviction attempt fails (all pinned).
        cache.insert(2, 20, 60, CachePriority::Pinned);
        // Both present even though over budget (pinned entries can't be evicted).
        assert!(cache.contains(&1));
        assert!(cache.contains(&2));
    }

    #[test]
    fn set_priority_on_nonexistent_returns_false() {
        let mut cache: ResourceCache<u32, u32> = ResourceCache::new(1000);
        assert!(!cache.set_priority(&1, CachePriority::High));
    }
}
