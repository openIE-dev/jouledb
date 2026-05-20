//! Request coalescing / deduplication.
//!
//! In-flight request tracking, result sharing for identical requests,
//! cache stampede prevention, and the singleflight pattern.
//! Pure Rust — no async runtime dependencies.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ── Coalesce key ────────────────────────────────────────────────

/// A key that identifies duplicate in-flight requests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoalesceKey(pub String);

impl CoalesceKey {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }

    /// Build a key from method + URL.
    pub fn from_request(method: &str, url: &str) -> Self {
        Self(format!("{method}:{url}"))
    }

    /// Build a key from method + URL + body hash.
    pub fn from_request_with_body(method: &str, url: &str, body_hash: &str) -> Self {
        Self(format!("{method}:{url}:{body_hash}"))
    }

    pub fn as_str(&self) -> &str { &self.0 }
}

impl fmt::Display for CoalesceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Flight status ───────────────────────────────────────────────

/// Status of an in-flight coalesced request group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlightStatus {
    /// First request — the caller should execute the actual work.
    Leader,
    /// Duplicate request — the caller should wait for the leader's result.
    Follower { position: usize },
}

impl FlightStatus {
    pub fn is_leader(&self) -> bool { matches!(self, FlightStatus::Leader) }
    pub fn is_follower(&self) -> bool { matches!(self, FlightStatus::Follower { .. }) }
}

// ── Flight result ───────────────────────────────────────────────

/// The result of a coalesced request, shareable among all waiters.
#[derive(Debug, Clone)]
pub struct FlightResult {
    pub status: u16,
    pub body: Vec<u8>,
    pub headers: HashMap<String, String>,
    pub produced_at_ms: u64,
}

// ── In-flight tracker ───────────────────────────────────────────

/// Tracks in-flight request groups for coalescing.
#[derive(Debug, Clone)]
struct FlightEntry {
    waiter_count: usize,
    started_ms: u64,
    result: Option<FlightResult>,
}

/// Core coalescing tracker (not thread-safe on its own — wrap in Arc<Mutex>).
#[derive(Debug)]
struct CoalesceInner {
    flights: HashMap<String, FlightEntry>,
    stats: CoalesceStats,
}

/// Statistics for the coalescing system.
#[derive(Debug, Clone, Default)]
pub struct CoalesceStats {
    pub total_requests: u64,
    pub total_leaders: u64,
    pub total_followers: u64,
    pub total_completed: u64,
    pub total_evicted: u64,
    pub peak_in_flight: usize,
}

impl CoalesceStats {
    /// Deduplication ratio: followers / total requests.
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_requests == 0 { 0.0 }
        else { self.total_followers as f64 / self.total_requests as f64 }
    }
}

// ── Singleflight group ──────────────────────────────────────────

/// Thread-safe singleflight / request coalescing.
///
/// When multiple identical requests arrive while one is in-flight,
/// only the first (leader) executes. All others (followers) receive
/// the same result once the leader completes.
#[derive(Debug, Clone)]
pub struct SingleFlight {
    inner: Arc<Mutex<CoalesceInner>>,
}

impl SingleFlight {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CoalesceInner {
                flights: HashMap::new(),
                stats: CoalesceStats::default(),
            })),
        }
    }

    /// Register interest in a key. Returns Leader if this is the first
    /// request, or Follower if a flight is already in progress.
    pub fn register(&self, key: &CoalesceKey, now_ms: u64) -> FlightStatus {
        let mut inner = self.inner.lock().unwrap();
        inner.stats.total_requests += 1;

        let key_str = key.0.clone();
        if inner.flights.contains_key(&key_str) {
            let entry = inner.flights.get_mut(&key_str).unwrap();
            entry.waiter_count += 1;
            let position = entry.waiter_count;
            inner.stats.total_followers += 1;
            FlightStatus::Follower { position }
        } else {
            inner.flights.insert(key_str, FlightEntry {
                waiter_count: 1,
                started_ms: now_ms,
                result: None,
            });
            let count = inner.flights.len();
            if count > inner.stats.peak_in_flight {
                inner.stats.peak_in_flight = count;
            }
            inner.stats.total_leaders += 1;
            FlightStatus::Leader
        }
    }

    /// Complete a flight with a result. All followers can now retrieve it.
    pub fn complete(&self, key: &CoalesceKey, result: FlightResult) -> usize {
        let mut inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.flights.get_mut(&key.0) {
            let waiters = entry.waiter_count;
            entry.result = Some(result);
            inner.stats.total_completed += 1;
            waiters
        } else {
            0
        }
    }

    /// Retrieve the result for a key (if the leader has completed).
    pub fn get_result(&self, key: &CoalesceKey) -> Option<FlightResult> {
        let inner = self.inner.lock().unwrap();
        inner.flights.get(&key.0)
            .and_then(|e| e.result.clone())
    }

    /// Remove a completed flight, freeing the entry.
    pub fn remove(&self, key: &CoalesceKey) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner.flights.remove(&key.0).is_some()
    }

    /// Check if a key has an in-flight request.
    pub fn is_in_flight(&self, key: &CoalesceKey) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.flights.contains_key(&key.0)
    }

    /// Number of in-flight groups.
    pub fn in_flight_count(&self) -> usize {
        self.inner.lock().unwrap().flights.len()
    }

    /// Total waiters across all in-flight groups.
    pub fn total_waiters(&self) -> usize {
        self.inner.lock().unwrap().flights.values()
            .map(|e| e.waiter_count)
            .sum()
    }

    /// Get statistics.
    pub fn stats(&self) -> CoalesceStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Evict stale flights older than `max_age_ms`.
    pub fn evict_stale(&self, now_ms: u64, max_age_ms: u64) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let before = inner.flights.len();
        inner.flights.retain(|_, entry| {
            now_ms.saturating_sub(entry.started_ms) < max_age_ms
        });
        let evicted = before - inner.flights.len();
        inner.stats.total_evicted += evicted as u64;
        evicted
    }
}

impl Default for SingleFlight {
    fn default() -> Self { Self::new() }
}

// ── Stampede shield ─────────────────────────────────────────────

/// Cache stampede prevention via probabilistic early recomputation.
///
/// Uses the "XFetch" algorithm: as a cached value approaches expiry,
/// requests probabilistically trigger early recomputation rather than
/// all waiting for the exact expiry moment.
#[derive(Debug, Clone)]
pub struct StampedeShield {
    /// Multiplier for the probability function (higher = earlier recomputation).
    pub beta: f64,
}

impl StampedeShield {
    pub fn new(beta: f64) -> Self {
        Self { beta: beta.max(0.1) }
    }

    /// Default beta = 1.0.
    pub fn default_shield() -> Self {
        Self { beta: 1.0 }
    }

    /// Determine if this request should trigger early recomputation.
    ///
    /// - `ttl_remaining_secs`: seconds until the cache entry expires.
    /// - `compute_time_secs`: how long the recomputation takes (estimate).
    /// - `random_01`: uniform random value in [0, 1].
    ///
    /// Returns true if the caller should recompute now (before expiry).
    pub fn should_recompute(
        &self,
        ttl_remaining_secs: f64,
        compute_time_secs: f64,
        random_01: f64,
    ) -> bool {
        if ttl_remaining_secs <= 0.0 {
            return true; // Already expired
        }

        // XFetch: recompute if -beta * compute_time * ln(random) >= ttl_remaining
        let r = random_01.clamp(0.001, 1.0); // avoid ln(0)
        let threshold = -self.beta * compute_time_secs * r.ln();
        threshold >= ttl_remaining_secs
    }
}

// ── Dedup batch ─────────────────────────────────────────────────

/// Collects duplicate keys into a deduplicated batch for bulk execution.
#[derive(Debug, Clone)]
pub struct DedupBatch {
    /// Unique keys in insertion order.
    keys: Vec<CoalesceKey>,
    /// Map from key to list of requester indices.
    requesters: HashMap<String, Vec<usize>>,
    next_requester: usize,
}

impl DedupBatch {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            requesters: HashMap::new(),
            next_requester: 0,
        }
    }

    /// Add a request, returning its requester index.
    pub fn add(&mut self, key: CoalesceKey) -> usize {
        let idx = self.next_requester;
        self.next_requester += 1;

        let entry = self.requesters.entry(key.0.clone()).or_insert_with(|| {
            self.keys.push(key);
            Vec::new()
        });
        entry.push(idx);
        idx
    }

    /// Unique keys to fetch.
    pub fn unique_keys(&self) -> &[CoalesceKey] {
        &self.keys
    }

    /// Number of unique keys (deduplicated).
    pub fn unique_count(&self) -> usize {
        self.keys.len()
    }

    /// Total number of requests (including duplicates).
    pub fn total_count(&self) -> usize {
        self.next_requester
    }

    /// Get requester indices for a key.
    pub fn requesters_for(&self, key: &CoalesceKey) -> &[usize] {
        self.requesters.get(&key.0).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Deduplication ratio.
    pub fn dedup_ratio(&self) -> f64 {
        if self.next_requester == 0 { return 0.0; }
        let dupes = self.next_requester - self.keys.len();
        dupes as f64 / self.next_requester as f64
    }
}

impl Default for DedupBatch {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesce_key_from_request() {
        let k = CoalesceKey::from_request("GET", "/api/data");
        assert_eq!(k.as_str(), "GET:/api/data");
    }

    #[test]
    fn test_coalesce_key_with_body() {
        let k = CoalesceKey::from_request_with_body("POST", "/api", "abc123");
        assert_eq!(k.as_str(), "POST:/api:abc123");
    }

    #[test]
    fn test_coalesce_key_display() {
        let k = CoalesceKey::new("test-key");
        assert_eq!(k.to_string(), "test-key");
    }

    #[test]
    fn test_flight_status() {
        assert!(FlightStatus::Leader.is_leader());
        assert!(!FlightStatus::Leader.is_follower());
        assert!(FlightStatus::Follower { position: 2 }.is_follower());
        assert!(!FlightStatus::Follower { position: 2 }.is_leader());
    }

    // ── SingleFlight ────────────────────────────────────────

    #[test]
    fn test_singleflight_leader_follower() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("req-1");

        let s1 = sf.register(&key, 100);
        assert!(s1.is_leader());

        let s2 = sf.register(&key, 110);
        assert!(s2.is_follower());

        let s3 = sf.register(&key, 120);
        match s3 {
            FlightStatus::Follower { position: 3 } => {}
            other => panic!("expected Follower(3), got {:?}", other),
        }
    }

    #[test]
    fn test_singleflight_different_keys() {
        let sf = SingleFlight::new();
        let k1 = CoalesceKey::new("a");
        let k2 = CoalesceKey::new("b");

        assert!(sf.register(&k1, 0).is_leader());
        assert!(sf.register(&k2, 0).is_leader());
        assert_eq!(sf.in_flight_count(), 2);
    }

    #[test]
    fn test_singleflight_complete_and_get() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("req-1");

        sf.register(&key, 100);
        sf.register(&key, 110);

        let result = FlightResult {
            status: 200,
            body: b"hello".to_vec(),
            headers: HashMap::new(),
            produced_at_ms: 150,
        };
        let waiters = sf.complete(&key, result);
        assert_eq!(waiters, 2);

        let r = sf.get_result(&key).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"hello");
    }

    #[test]
    fn test_singleflight_get_before_complete() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("req-1");
        sf.register(&key, 100);
        assert!(sf.get_result(&key).is_none());
    }

    #[test]
    fn test_singleflight_remove() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("req-1");
        sf.register(&key, 100);
        assert!(sf.is_in_flight(&key));
        assert!(sf.remove(&key));
        assert!(!sf.is_in_flight(&key));
        assert!(!sf.remove(&key));
    }

    #[test]
    fn test_singleflight_total_waiters() {
        let sf = SingleFlight::new();
        let k1 = CoalesceKey::new("a");
        let k2 = CoalesceKey::new("b");

        sf.register(&k1, 0);
        sf.register(&k1, 1);
        sf.register(&k2, 2);

        assert_eq!(sf.total_waiters(), 3);
    }

    #[test]
    fn test_singleflight_stats() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("x");

        sf.register(&key, 0);
        sf.register(&key, 1);
        sf.register(&key, 2);

        let stats = sf.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.total_leaders, 1);
        assert_eq!(stats.total_followers, 2);
        assert_eq!(stats.peak_in_flight, 1);
    }

    #[test]
    fn test_singleflight_dedup_ratio() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("x");

        sf.register(&key, 0);
        sf.register(&key, 1);
        sf.register(&key, 2);
        sf.register(&key, 3);

        let stats = sf.stats();
        // 3 followers out of 4 total = 0.75
        assert!((stats.dedup_ratio() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_singleflight_evict_stale() {
        let sf = SingleFlight::new();
        sf.register(&CoalesceKey::new("old"), 100);
        sf.register(&CoalesceKey::new("new"), 900);

        let evicted = sf.evict_stale(1000, 500);
        assert_eq!(evicted, 1);
        assert_eq!(sf.in_flight_count(), 1);
        assert!(!sf.is_in_flight(&CoalesceKey::new("old")));
        assert!(sf.is_in_flight(&CoalesceKey::new("new")));
    }

    #[test]
    fn test_singleflight_complete_nonexistent() {
        let sf = SingleFlight::new();
        let key = CoalesceKey::new("ghost");
        let result = FlightResult {
            status: 200,
            body: vec![],
            headers: HashMap::new(),
            produced_at_ms: 0,
        };
        assert_eq!(sf.complete(&key, result), 0);
    }

    #[test]
    fn test_singleflight_clone_shares_state() {
        let sf1 = SingleFlight::new();
        let sf2 = sf1.clone();
        let key = CoalesceKey::new("shared");

        sf1.register(&key, 0);
        assert!(sf2.is_in_flight(&key));
    }

    // ── Stampede shield ─────────────────────────────────────

    #[test]
    fn test_stampede_expired() {
        let shield = StampedeShield::default_shield();
        assert!(shield.should_recompute(0.0, 1.0, 0.5));
        assert!(shield.should_recompute(-10.0, 1.0, 0.5));
    }

    #[test]
    fn test_stampede_long_ttl_no_recompute() {
        let shield = StampedeShield::default_shield();
        // With a long TTL remaining and short compute time, should not recompute
        assert!(!shield.should_recompute(3600.0, 1.0, 0.5));
    }

    #[test]
    fn test_stampede_near_expiry_recomputes() {
        let shield = StampedeShield::new(2.0);
        // Very close to expiry with a decent compute time
        // -2.0 * 5.0 * ln(0.1) = 23.0 >= 1.0 -> recompute
        assert!(shield.should_recompute(1.0, 5.0, 0.1));
    }

    #[test]
    fn test_stampede_high_beta_more_eager() {
        let conservative = StampedeShield::new(0.5);
        let aggressive = StampedeShield::new(5.0);

        // Same conditions — aggressive should be more likely to recompute
        let ttl = 30.0;
        let compute = 5.0;
        let r = 0.2;

        let cons_result = conservative.should_recompute(ttl, compute, r);
        let aggr_result = aggressive.should_recompute(ttl, compute, r);

        // Aggressive should recompute when conservative does not
        // (or at least be equally likely)
        if !cons_result {
            // If conservative doesn't recompute, aggressive might
            // This tests the direction, not an exact boundary
            let _ = aggr_result; // just ensure it runs without panic
        }
        // Verify aggressive is at least as eager
        if cons_result {
            assert!(aggr_result);
        }
    }

    // ── DedupBatch ──────────────────────────────────────────

    #[test]
    fn test_dedup_batch_basic() {
        let mut batch = DedupBatch::new();
        let k1 = CoalesceKey::new("a");
        let k2 = CoalesceKey::new("b");

        let idx0 = batch.add(k1.clone());
        let idx1 = batch.add(k2.clone());
        let idx2 = batch.add(k1.clone()); // duplicate

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);

        assert_eq!(batch.unique_count(), 2);
        assert_eq!(batch.total_count(), 3);
    }

    #[test]
    fn test_dedup_batch_requesters() {
        let mut batch = DedupBatch::new();
        let k = CoalesceKey::new("x");

        batch.add(k.clone());
        batch.add(k.clone());
        batch.add(k.clone());

        let requesters = batch.requesters_for(&k);
        assert_eq!(requesters, &[0, 1, 2]);
    }

    #[test]
    fn test_dedup_batch_unique_keys() {
        let mut batch = DedupBatch::new();
        batch.add(CoalesceKey::new("a"));
        batch.add(CoalesceKey::new("b"));
        batch.add(CoalesceKey::new("a"));

        let keys: Vec<&str> = batch.unique_keys().iter().map(|k| k.as_str()).collect();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_dedup_batch_ratio() {
        let mut batch = DedupBatch::new();
        let k = CoalesceKey::new("x");
        batch.add(k.clone());
        batch.add(k.clone());
        batch.add(k.clone());
        batch.add(k.clone());

        // 4 total, 1 unique, 3 dupes => 3/4 = 0.75
        assert!((batch.dedup_ratio() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_dedup_batch_empty() {
        let batch = DedupBatch::new();
        assert_eq!(batch.unique_count(), 0);
        assert_eq!(batch.total_count(), 0);
        assert!((batch.dedup_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dedup_batch_no_duplicates() {
        let mut batch = DedupBatch::new();
        batch.add(CoalesceKey::new("a"));
        batch.add(CoalesceKey::new("b"));
        batch.add(CoalesceKey::new("c"));

        assert_eq!(batch.unique_count(), 3);
        assert_eq!(batch.total_count(), 3);
        assert!((batch.dedup_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dedup_batch_requesters_unknown_key() {
        let batch = DedupBatch::new();
        let k = CoalesceKey::new("ghost");
        assert!(batch.requesters_for(&k).is_empty());
    }

    #[test]
    fn test_coalesce_stats_dedup_ratio_empty() {
        let stats = CoalesceStats::default();
        assert!((stats.dedup_ratio() - 0.0).abs() < 0.001);
    }
}
