//! Async data fetching state machine: cache-first with stale-while-revalidate.
//!
//! Replaces TanStack Query, SWR, and Apollo Client with a synchronous
//! state machine that any async executor (or manual polling) can drive.

use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc, TimeDelta};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

// ── Status ──────────────────────────────────────────────────────

/// Current status of a query or mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    Idle,
    Loading,
    Success,
    Error(String),
    Stale,
}

// ── Cache ───────────────────────────────────────────────────────

/// Type-erased cache entry stored as `serde_json::Value`.
#[derive(Debug, Clone)]
pub struct CacheEntryRaw {
    pub value: Value,
    pub fetched_at: DateTime<Utc>,
    pub stale_after: Option<DateTime<Utc>>,
}

/// Typed cache entry (returned to callers after deserialization).
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub data: T,
    pub fetched_at: DateTime<Utc>,
    pub stale_after: Option<DateTime<Utc>>,
    pub error_count: u32,
}

// ── Config ──────────────────────────────────────────────────────

/// Tuning knobs for the query cache.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    pub stale_time_ms: u64,
    pub cache_time_ms: u64,
    pub retry_count: u32,
    pub retry_delay_ms: u64,
    pub refetch_on_mount: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            stale_time_ms: 5 * 60 * 1000,   // 5 min
            cache_time_ms: 30 * 60 * 1000,  // 30 min
            retry_count: 3,
            retry_delay_ms: 1000,
            refetch_on_mount: true,
        }
    }
}

// ── QueryState ──────────────────────────────────────────────────

/// Observable state for a single query.
#[derive(Debug, Clone)]
pub struct QueryState<T: Clone> {
    pub status: QueryStatus,
    pub data: Option<T>,
    pub error: Option<String>,
    pub is_fetching: bool,
    pub data_updated_at: Option<DateTime<Utc>>,
    pub fetch_count: u32,
}

impl<T: Clone> Default for QueryState<T> {
    fn default() -> Self {
        Self {
            status: QueryStatus::Idle,
            data: None,
            error: None,
            is_fetching: false,
            data_updated_at: None,
            fetch_count: 0,
        }
    }
}

// ── QueryClient ─────────────────────────────────────────────────

/// Central cache and coordinator for all queries.
pub struct QueryClient {
    cache: HashMap<String, CacheEntryRaw>,
    config: QueryConfig,
    invalidated: HashSet<String>,
}

impl QueryClient {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            config: QueryConfig::default(),
            invalidated: HashSet::new(),
        }
    }

    pub fn with_config(config: QueryConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
            invalidated: HashSet::new(),
        }
    }

    pub fn get_cached(&self, key: &str) -> Option<&CacheEntryRaw> {
        self.cache.get(key)
    }

    pub fn set_cached(&mut self, key: &str, value: Value) {
        let now = Utc::now();
        let stale_after = Some(now + TimeDelta::milliseconds(self.config.stale_time_ms as i64));
        self.cache.insert(key.to_string(), CacheEntryRaw {
            value,
            fetched_at: now,
            stale_after,
        });
        self.invalidated.remove(key);
    }

    pub fn is_stale(&self, key: &str, now: &DateTime<Utc>) -> bool {
        if self.invalidated.contains(key) {
            return true;
        }
        match self.cache.get(key) {
            Some(entry) => match entry.stale_after {
                Some(sa) => *now >= sa,
                None => false,
            },
            None => true,
        }
    }

    pub fn invalidate(&mut self, key: &str) {
        self.invalidated.insert(key.to_string());
    }

    pub fn invalidate_prefix(&mut self, prefix: &str) {
        let keys: Vec<String> = self.cache.keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for k in keys {
            self.invalidated.insert(k);
        }
    }

    pub fn remove(&mut self, key: &str) -> bool {
        self.invalidated.remove(key);
        self.cache.remove(key).is_some()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.invalidated.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn should_fetch(&self, key: &str, now: &DateTime<Utc>) -> bool {
        if !self.cache.contains_key(key) {
            return true;
        }
        if self.invalidated.contains(key) {
            return true;
        }
        if self.is_stale(key, now) {
            return true;
        }
        self.config.refetch_on_mount
    }

    /// Store a typed result in the cache.
    pub fn resolve_query<T: Serialize + DeserializeOwned>(&mut self, key: &str, data: T) {
        let value = serde_json::to_value(&data).unwrap_or(Value::Null);
        let now = Utc::now();
        let stale_after = Some(now + TimeDelta::milliseconds(self.config.stale_time_ms as i64));
        self.cache.insert(key.to_string(), CacheEntryRaw {
            value,
            fetched_at: now,
            stale_after,
        });
        self.invalidated.remove(key);
    }

    /// Garbage-collect entries older than `cache_time_ms`. Returns count removed.
    pub fn gc(&mut self, now: &DateTime<Utc>) -> usize {
        let cutoff = *now - TimeDelta::milliseconds(self.config.cache_time_ms as i64);
        let old_len = self.cache.len();
        self.cache.retain(|_, entry| entry.fetched_at > cutoff);
        let removed = old_len - self.cache.len();
        removed
    }
}

impl Default for QueryClient {
    fn default() -> Self { Self::new() }
}

// ── MutationState ───────────────────────────────────────────────

/// Observable state for a mutation (create/update/delete).
#[derive(Debug, Clone)]
pub struct MutationState<T: Clone> {
    pub status: QueryStatus,
    pub data: Option<T>,
    pub error: Option<String>,
    pub submit_count: u32,
}

impl<T: Clone> MutationState<T> {
    pub fn new() -> Self {
        Self {
            status: QueryStatus::Idle,
            data: None,
            error: None,
            submit_count: 0,
        }
    }

    pub fn start(&mut self) {
        self.status = QueryStatus::Loading;
        self.error = None;
        self.submit_count += 1;
    }

    pub fn succeed(&mut self, data: T) {
        self.status = QueryStatus::Success;
        self.data = Some(data);
        self.error = None;
    }

    pub fn fail(&mut self, error: &str) {
        self.status = QueryStatus::Error(error.to_string());
        self.error = Some(error.to_string());
    }

    pub fn reset(&mut self) {
        self.status = QueryStatus::Idle;
        self.data = None;
        self.error = None;
        self.submit_count = 0;
    }

    pub fn is_loading(&self) -> bool { self.status == QueryStatus::Loading }
    pub fn is_success(&self) -> bool { self.status == QueryStatus::Success }
    pub fn is_error(&self) -> bool { matches!(self.status, QueryStatus::Error(_)) }
}

impl<T: Clone> Default for MutationState<T> {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use chrono::TimeDelta;

    #[test]
    fn new_client_empty() {
        let c = QueryClient::new();
        assert_eq!(c.cache_size(), 0);
    }

    #[test]
    fn set_get_cached() {
        let mut c = QueryClient::new();
        c.set_cached("users", json!([1, 2, 3]));
        assert!(c.get_cached("users").is_some());
        assert_eq!(c.get_cached("users").unwrap().value, json!([1, 2, 3]));
    }

    #[test]
    fn is_stale_after_time() {
        let mut c = QueryClient::with_config(QueryConfig {
            stale_time_ms: 1000,
            ..Default::default()
        });
        c.set_cached("k", json!(1));
        let future = Utc::now() + TimeDelta::seconds(2);
        assert!(c.is_stale("k", &future));
    }

    #[test]
    fn invalidate_marks_stale() {
        let mut c = QueryClient::new();
        c.set_cached("k", json!(1));
        let now = Utc::now();
        assert!(!c.is_stale("k", &now));
        c.invalidate("k");
        assert!(c.is_stale("k", &now));
    }

    #[test]
    fn invalidate_prefix_works() {
        let mut c = QueryClient::new();
        c.set_cached("user:1", json!(1));
        c.set_cached("user:2", json!(2));
        c.set_cached("post:1", json!(3));
        c.invalidate_prefix("user:");
        let now = Utc::now();
        assert!(c.is_stale("user:1", &now));
        assert!(c.is_stale("user:2", &now));
        assert!(!c.is_stale("post:1", &now));
    }

    #[test]
    fn should_fetch_when_not_cached() {
        let c = QueryClient::new();
        assert!(c.should_fetch("x", &Utc::now()));
    }

    #[test]
    fn should_fetch_when_stale() {
        let mut c = QueryClient::with_config(QueryConfig {
            stale_time_ms: 1,
            refetch_on_mount: false,
            ..Default::default()
        });
        c.set_cached("k", json!(1));
        let future = Utc::now() + TimeDelta::seconds(1);
        assert!(c.should_fetch("k", &future));
    }

    #[test]
    fn resolve_caches_data() {
        let mut c = QueryClient::new();
        c.resolve_query("k", vec![1, 2, 3]);
        let entry = c.get_cached("k").unwrap();
        assert_eq!(entry.value, json!([1, 2, 3]));
    }

    #[test]
    fn gc_removes_old() {
        let mut c = QueryClient::with_config(QueryConfig {
            cache_time_ms: 1000,
            ..Default::default()
        });
        c.set_cached("k", json!(1));
        // Manually backdate
        if let Some(e) = c.cache.get_mut("k") {
            e.fetched_at = Utc::now() - TimeDelta::seconds(60);
        }
        let removed = c.gc(&Utc::now());
        assert_eq!(removed, 1);
        assert_eq!(c.cache_size(), 0);
    }

    #[test]
    fn gc_keeps_fresh() {
        let mut c = QueryClient::new();
        c.set_cached("k", json!(1));
        let removed = c.gc(&Utc::now());
        assert_eq!(removed, 0);
        assert_eq!(c.cache_size(), 1);
    }

    #[test]
    fn mutation_state_transitions() {
        let mut m = MutationState::<String>::new();
        assert!(m.status == QueryStatus::Idle);
        m.start();
        assert!(m.is_loading());
        assert_eq!(m.submit_count, 1);
        m.succeed("done".to_string());
        assert!(m.is_success());
        assert_eq!(m.data, Some("done".to_string()));
        m.reset();
        assert!(m.status == QueryStatus::Idle);
    }

    #[test]
    fn mutation_fail() {
        let mut m = MutationState::<i32>::new();
        m.start();
        m.fail("network error");
        assert!(m.is_error());
        assert_eq!(m.error, Some("network error".to_string()));
    }

    #[test]
    fn clear_empties_all() {
        let mut c = QueryClient::new();
        c.set_cached("a", json!(1));
        c.set_cached("b", json!(2));
        c.invalidate("a");
        c.clear();
        assert_eq!(c.cache_size(), 0);
    }

    #[test]
    fn remove_single_key() {
        let mut c = QueryClient::new();
        c.set_cached("a", json!(1));
        c.set_cached("b", json!(2));
        assert!(c.remove("a"));
        assert!(!c.remove("a"));
        assert_eq!(c.cache_size(), 1);
    }
}
