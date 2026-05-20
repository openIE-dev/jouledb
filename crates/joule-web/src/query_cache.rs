//! Data fetching cache inspired by React Query / TanStack Query.
//!
//! Provides cache entries with TTL, stale-while-revalidate, deduplication
//! of in-flight requests, cache invalidation, optimistic updates, and
//! garbage collection of unused entries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── CacheStatus ──────────────────────────────────────────────

/// The status of a cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStatus {
    /// Fresh data, not yet stale.
    Fresh,
    /// Data is stale but still usable while revalidating.
    Stale,
    /// Data is being fetched.
    Fetching,
    /// Fetch failed.
    Error,
    /// No data available.
    Empty,
}

// ── CacheEntry ───────────────────────────────────────────────

/// A single cache entry with metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached data.
    pub data: Option<T>,
    /// Current status.
    pub status: CacheStatus,
    /// Timestamp when the data was last fetched (ms since epoch).
    pub fetched_at: u64,
    /// Time-to-live in milliseconds. After this, data is considered stale.
    pub ttl_ms: u64,
    /// Number of active subscribers/observers.
    pub observer_count: usize,
    /// Error message if status is Error.
    pub error: Option<String>,
    /// Retry count for failed fetches.
    pub retry_count: u32,
    /// Maximum retries before giving up.
    pub max_retries: u32,
    /// Last accessed timestamp for GC.
    pub last_accessed: u64,
    /// Whether an optimistic update has been applied.
    pub is_optimistic: bool,
    /// Snapshot before optimistic update (for rollback).
    previous_data: Option<T>,
}

impl<T: Clone> CacheEntry<T> {
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            data: None,
            status: CacheStatus::Empty,
            fetched_at: 0,
            ttl_ms,
            observer_count: 0,
            error: None,
            retry_count: 0,
            max_retries: 3,
            last_accessed: 0,
            is_optimistic: false,
            previous_data: None,
        }
    }

    /// Check if the entry is stale at the given timestamp.
    pub fn is_stale(&self, now_ms: u64) -> bool {
        if self.fetched_at == 0 {
            return true;
        }
        now_ms.saturating_sub(self.fetched_at) > self.ttl_ms
    }

    /// Check if the entry is fresh at the given timestamp.
    pub fn is_fresh(&self, now_ms: u64) -> bool {
        self.data.is_some() && !self.is_stale(now_ms)
    }

    /// Set the data and mark as fresh.
    pub fn set_data(&mut self, data: T, now_ms: u64) {
        self.data = Some(data);
        self.status = CacheStatus::Fresh;
        self.fetched_at = now_ms;
        self.last_accessed = now_ms;
        self.error = None;
        self.retry_count = 0;
        self.is_optimistic = false;
        self.previous_data = None;
    }

    /// Apply an optimistic update, saving the previous data for rollback.
    pub fn optimistic_update(&mut self, data: T) {
        self.previous_data = self.data.clone();
        self.data = Some(data);
        self.is_optimistic = true;
    }

    /// Rollback an optimistic update.
    pub fn rollback(&mut self) {
        if self.is_optimistic {
            self.data = self.previous_data.take();
            self.is_optimistic = false;
        }
    }

    /// Mark as fetching.
    pub fn mark_fetching(&mut self) {
        self.status = CacheStatus::Fetching;
    }

    /// Mark as errored.
    pub fn mark_error(&mut self, error: impl Into<String>) {
        self.status = CacheStatus::Error;
        self.error = Some(error.into());
        self.retry_count += 1;
    }

    /// Check if we can retry.
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Mark as stale.
    pub fn mark_stale(&mut self) {
        if self.data.is_some() {
            self.status = CacheStatus::Stale;
        }
    }

    /// Touch the entry (update last_accessed).
    pub fn touch(&mut self, now_ms: u64) {
        self.last_accessed = now_ms;
    }

    /// Add an observer.
    pub fn add_observer(&mut self) {
        self.observer_count += 1;
    }

    /// Remove an observer.
    pub fn remove_observer(&mut self) {
        self.observer_count = self.observer_count.saturating_sub(1);
    }

    /// Check if the entry has no observers (candidate for GC).
    pub fn is_unused(&self) -> bool {
        self.observer_count == 0
    }
}

// ── QueryCache ───────────────────────────────────────────────

/// A cache for data queries with TTL, deduplication, and GC.
pub struct QueryCache<T> {
    entries: HashMap<String, CacheEntry<T>>,
    /// Default TTL for new entries.
    default_ttl_ms: u64,
    /// Stale-while-revalidate window (ms beyond TTL).
    stale_time_ms: u64,
    /// GC threshold: entries unused for this long are removed.
    gc_time_ms: u64,
    /// In-flight request deduplication: keys currently being fetched.
    in_flight: HashMap<String, u64>,
    /// Number of GC runs performed.
    gc_runs: usize,
}

impl<T: Clone> Default for QueryCache<T> {
    fn default() -> Self {
        Self::new(5 * 60 * 1000) // 5 minutes default TTL
    }
}

impl<T: Clone> QueryCache<T> {
    /// Create a cache with a default TTL in milliseconds.
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_ms,
            stale_time_ms: 30_000, // 30 seconds stale window
            gc_time_ms: 5 * 60 * 1000, // 5 minutes
            in_flight: HashMap::new(),
            gc_runs: 0,
        }
    }

    /// Set the stale-while-revalidate window.
    pub fn set_stale_time(&mut self, ms: u64) {
        self.stale_time_ms = ms;
    }

    /// Set the GC time threshold.
    pub fn set_gc_time(&mut self, ms: u64) {
        self.gc_time_ms = ms;
    }

    /// Get a cache entry by key.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry<T>> {
        // Update stale status
        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch(now_ms);
            if entry.is_stale(now_ms) && entry.status == CacheStatus::Fresh {
                entry.status = CacheStatus::Stale;
            }
        }
        self.entries.get(key)
    }

    /// Get data directly, if available and not empty.
    pub fn get_data(&mut self, key: &str, now_ms: u64) -> Option<&T> {
        self.get(key, now_ms).and_then(|e| e.data.as_ref())
    }

    /// Get the status of a cache entry.
    pub fn get_status(&self, key: &str) -> CacheStatus {
        self.entries
            .get(key)
            .map_or(CacheStatus::Empty, |e| e.status)
    }

    /// Set data for a key.
    pub fn set(&mut self, key: impl Into<String>, data: T, now_ms: u64) {
        let key = key.into();
        let entry = self
            .entries
            .entry(key.clone())
            .or_insert_with(|| CacheEntry::new(self.default_ttl_ms));
        entry.set_data(data, now_ms);
        self.in_flight.remove(&key);
    }

    /// Set data with a custom TTL.
    pub fn set_with_ttl(
        &mut self,
        key: impl Into<String>,
        data: T,
        now_ms: u64,
        ttl_ms: u64,
    ) {
        let key = key.into();
        let entry = self
            .entries
            .entry(key.clone())
            .or_insert_with(|| CacheEntry::new(ttl_ms));
        entry.ttl_ms = ttl_ms;
        entry.set_data(data, now_ms);
        self.in_flight.remove(&key);
    }

    /// Check if we should fetch (revalidate) data for a key.
    /// Returns true if data is missing, stale, or within the stale-while-revalidate window.
    pub fn should_fetch(&self, key: &str, now_ms: u64) -> bool {
        // Don't duplicate in-flight requests
        if self.in_flight.contains_key(key) {
            return false;
        }

        match self.entries.get(key) {
            None => true,
            Some(entry) => {
                if entry.status == CacheStatus::Fetching {
                    return false;
                }
                entry.is_stale(now_ms)
            }
        }
    }

    /// Mark a key as being fetched (for deduplication).
    pub fn mark_fetching(&mut self, key: impl Into<String>, now_ms: u64) {
        let key = key.into();
        self.in_flight.insert(key.clone(), now_ms);
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| CacheEntry::new(self.default_ttl_ms));
        entry.mark_fetching();
    }

    /// Mark a fetch as failed.
    pub fn mark_error(&mut self, key: &str, error: impl Into<String>) {
        self.in_flight.remove(key);
        if let Some(entry) = self.entries.get_mut(key) {
            entry.mark_error(error);
        }
    }

    /// Check if a request for this key is in flight.
    pub fn is_fetching(&self, key: &str) -> bool {
        self.in_flight.contains_key(key)
    }

    /// Invalidate a cache entry (mark as stale).
    pub fn invalidate(&mut self, key: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.mark_stale();
        }
    }

    /// Invalidate all entries whose keys match a prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) {
        let keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for key in keys {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.mark_stale();
            }
        }
    }

    /// Invalidate all entries.
    pub fn invalidate_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.mark_stale();
        }
    }

    /// Apply an optimistic update.
    pub fn optimistic_update(&mut self, key: &str, data: T) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.optimistic_update(data);
        }
    }

    /// Rollback an optimistic update.
    pub fn rollback(&mut self, key: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.rollback();
        }
    }

    /// Add an observer for a key.
    pub fn observe(&mut self, key: impl Into<String>) {
        let key = key.into();
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| CacheEntry::new(self.default_ttl_ms));
        entry.add_observer();
    }

    /// Remove an observer for a key.
    pub fn unobserve(&mut self, key: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.remove_observer();
        }
    }

    /// Remove a specific cache entry.
    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);
        self.in_flight.remove(key);
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.in_flight.clear();
    }

    /// Run garbage collection: remove unused entries older than gc_time_ms.
    pub fn gc(&mut self, now_ms: u64) -> usize {
        let gc_time = self.gc_time_ms;
        let keys_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.is_unused()
                    && now_ms.saturating_sub(entry.last_accessed) > gc_time
            })
            .map(|(key, _)| key.clone())
            .collect();

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
            self.in_flight.remove(&key);
        }
        self.gc_runs += 1;
        removed
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of in-flight requests.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Number of GC runs performed.
    pub fn gc_runs(&self) -> usize {
        self.gc_runs
    }

    /// Get all cache keys.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let mut fresh = 0;
        let mut stale = 0;
        let mut fetching = 0;
        let mut errored = 0;
        let mut empty = 0;

        for entry in self.entries.values() {
            match entry.status {
                CacheStatus::Fresh => fresh += 1,
                CacheStatus::Stale => stale += 1,
                CacheStatus::Fetching => fetching += 1,
                CacheStatus::Error => errored += 1,
                CacheStatus::Empty => empty += 1,
            }
        }

        CacheStats {
            total: self.entries.len(),
            fresh,
            stale,
            fetching,
            errored,
            empty,
            in_flight: self.in_flight.len(),
            gc_runs: self.gc_runs,
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    pub total: usize,
    pub fresh: usize,
    pub stale: usize,
    pub fetching: usize,
    pub errored: usize,
    pub empty: usize,
    pub in_flight: usize,
    pub gc_runs: usize,
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "value1".into(), 100);
        let entry = cache.get("key1", 100).unwrap();
        assert_eq!(entry.data.as_deref(), Some("value1"));
        assert_eq!(entry.status, CacheStatus::Fresh);
    }

    #[test]
    fn entry_becomes_stale() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "value1".into(), 100);

        // Before TTL
        let entry = cache.get("key1", 500).unwrap();
        assert_eq!(entry.status, CacheStatus::Fresh);

        // After TTL
        let entry = cache.get("key1", 1200).unwrap();
        assert_eq!(entry.status, CacheStatus::Stale);
    }

    #[test]
    fn should_fetch_logic() {
        let mut cache = QueryCache::<String>::new(1000);

        // No entry -> should fetch
        assert!(cache.should_fetch("key1", 0));

        // Fresh entry -> should not fetch
        cache.set("key1", "value1".into(), 100);
        assert!(!cache.should_fetch("key1", 500));

        // Stale entry -> should fetch
        assert!(cache.should_fetch("key1", 1200));
    }

    #[test]
    fn deduplication_of_in_flight() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.mark_fetching("key1", 100);
        assert!(cache.is_fetching("key1"));
        assert!(!cache.should_fetch("key1", 100));

        cache.set("key1", "value1".into(), 200);
        assert!(!cache.is_fetching("key1"));
    }

    #[test]
    fn optimistic_update_and_rollback() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "original".into(), 100);
        cache.optimistic_update("key1", "optimistic".into());

        let entry = cache.get("key1", 100).unwrap();
        assert_eq!(entry.data.as_deref(), Some("optimistic"));
        assert!(entry.is_optimistic);

        cache.rollback("key1");
        let entry = cache.get("key1", 100).unwrap();
        assert_eq!(entry.data.as_deref(), Some("original"));
        assert!(!entry.is_optimistic);
    }

    #[test]
    fn invalidation() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "value1".into(), 100);
        cache.invalidate("key1");
        assert_eq!(cache.get_status("key1"), CacheStatus::Stale);
    }

    #[test]
    fn invalidate_prefix() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("users/1", "alice".into(), 100);
        cache.set("users/2", "bob".into(), 100);
        cache.set("posts/1", "hello".into(), 100);
        cache.invalidate_prefix("users/");

        assert_eq!(cache.get_status("users/1"), CacheStatus::Stale);
        assert_eq!(cache.get_status("users/2"), CacheStatus::Stale);
        assert_eq!(cache.get_status("posts/1"), CacheStatus::Fresh);
    }

    #[test]
    fn invalidate_all() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("a", "1".into(), 100);
        cache.set("b", "2".into(), 100);
        cache.invalidate_all();
        assert_eq!(cache.get_status("a"), CacheStatus::Stale);
        assert_eq!(cache.get_status("b"), CacheStatus::Stale);
    }

    #[test]
    fn gc_removes_unused_entries() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set_gc_time(100);
        cache.set("key1", "value1".into(), 100);
        // key1 has no observers and last_accessed = 100

        let removed = cache.gc(300); // 200ms since last access > gc_time 100ms
        assert_eq!(removed, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn gc_preserves_observed_entries() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set_gc_time(100);
        cache.set("key1", "value1".into(), 100);
        cache.observe("key1");

        let removed = cache.gc(300);
        assert_eq!(removed, 0);
        assert!(!cache.is_empty());
    }

    #[test]
    fn observer_count() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.observe("key1");
        cache.observe("key1");

        let entry = cache.get("key1", 0).unwrap();
        assert_eq!(entry.observer_count, 2);

        cache.unobserve("key1");
        let entry = cache.get("key1", 0).unwrap();
        assert_eq!(entry.observer_count, 1);
    }

    #[test]
    fn error_handling() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.mark_fetching("key1", 100);
        cache.mark_error("key1", "network error");

        let entry = cache.get("key1", 100).unwrap();
        assert_eq!(entry.status, CacheStatus::Error);
        assert_eq!(entry.error.as_deref(), Some("network error"));
        assert_eq!(entry.retry_count, 1);
        assert!(entry.can_retry());
    }

    #[test]
    fn max_retries() {
        let mut entry = CacheEntry::<String>::new(1000);
        entry.max_retries = 2;
        entry.mark_error("err1");
        assert!(entry.can_retry());
        entry.mark_error("err2");
        assert!(!entry.can_retry());
    }

    #[test]
    fn custom_ttl() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set_with_ttl("key1", "value1".into(), 100, 500);

        // Should be fresh at 400ms
        assert!(cache.get("key1", 400).unwrap().status == CacheStatus::Fresh);
        // Should be stale at 700ms
        assert!(cache.get("key1", 700).unwrap().status == CacheStatus::Stale);
    }

    #[test]
    fn remove_entry() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "value1".into(), 100);
        cache.remove("key1");
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_cache() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("key1", "value1".into(), 100);
        cache.set("key2", "value2".into(), 100);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.in_flight_count(), 0);
    }

    #[test]
    fn cache_stats() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("a", "1".into(), 100);
        cache.set("b", "2".into(), 100);
        cache.mark_fetching("c", 100);
        cache.invalidate("a");

        let stats = cache.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.fresh, 1);
        assert_eq!(stats.stale, 1);
        assert_eq!(stats.fetching, 1);
        assert_eq!(stats.in_flight, 1);
    }

    #[test]
    fn get_data_convenience() {
        let mut cache = QueryCache::<String>::new(1000);
        assert!(cache.get_data("key1", 0).is_none());
        cache.set("key1", "hello".into(), 100);
        assert_eq!(cache.get_data("key1", 100), Some(&"hello".to_string()));
    }

    #[test]
    fn keys_sorted() {
        let mut cache = QueryCache::<String>::new(1000);
        cache.set("c", "3".into(), 0);
        cache.set("a", "1".into(), 0);
        cache.set("b", "2".into(), 0);
        assert_eq!(cache.keys(), vec!["a", "b", "c"]);
    }

    #[test]
    fn gc_runs_counter() {
        let mut cache = QueryCache::<String>::new(1000);
        assert_eq!(cache.gc_runs(), 0);
        cache.gc(0);
        assert_eq!(cache.gc_runs(), 1);
        cache.gc(0);
        assert_eq!(cache.gc_runs(), 2);
    }
}
