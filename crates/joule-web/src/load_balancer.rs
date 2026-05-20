//! Load balancer algorithms — round-robin, weighted, least-conn, consistent hash.
//!
//! Pure Rust implementation of common load balancing strategies. Supports
//! round-robin, weighted round-robin, least connections, consistent hashing
//! (ring), random (seeded), IP hash, health checking, backend add/remove,
//! and session affinity (sticky sessions).

use std::collections::HashMap;
use std::fmt;

// ── Backend ───────────────────────────────────────────────────

/// Health status of a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

/// A backend server in the pool.
#[derive(Debug, Clone)]
pub struct Backend {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub weight: u32,
    pub health: HealthStatus,
    pub active_connections: u64,
    pub total_requests: u64,
    pub metadata: HashMap<String, String>,
}

impl Backend {
    pub fn new(id: impl Into<String>, address: impl Into<String>, port: u16) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            port,
            weight: 1,
            health: HealthStatus::Healthy,
            active_connections: 0,
            total_requests: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    pub fn is_healthy(&self) -> bool {
        self.health == HealthStatus::Healthy
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}({})", self.address, self.port, self.id)
    }
}

// ── LB Strategy ───────────────────────────────────────────────

/// Load balancing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ConsistentHash,
    Random,
    IpHash,
}

// ── Consistent Hash Ring ──────────────────────────────────────

/// A point on the hash ring.
#[derive(Debug, Clone)]
struct RingEntry {
    hash: u64,
    backend_id: String,
}

/// Consistent hash ring for load balancing.
#[derive(Debug)]
struct HashRing {
    entries: Vec<RingEntry>,
    replicas: usize,
}

impl HashRing {
    fn new(replicas: usize) -> Self {
        Self {
            entries: Vec::new(),
            replicas,
        }
    }

    fn add(&mut self, backend_id: &str) {
        for i in 0..self.replicas {
            let key = format!("{}:{}", backend_id, i);
            let hash = Self::hash_key(&key);
            self.entries.push(RingEntry {
                hash,
                backend_id: backend_id.to_string(),
            });
        }
        self.entries.sort_by_key(|e| e.hash);
    }

    fn remove(&mut self, backend_id: &str) {
        self.entries.retain(|e| e.backend_id != backend_id);
    }

    fn get(&self, key: &str) -> Option<&str> {
        if self.entries.is_empty() {
            return None;
        }
        let hash = Self::hash_key(key);
        // Find the first entry with hash >= key hash (clockwise).
        let idx = match self.entries.binary_search_by_key(&hash, |e| e.hash) {
            Ok(i) => i,
            Err(i) => {
                if i >= self.entries.len() {
                    0 // Wrap around.
                } else {
                    i
                }
            }
        };
        Some(&self.entries[idx].backend_id)
    }

    /// FNV-1a hash for consistent hashing.
    fn hash_key(key: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

// ── Simple RNG ────────────────────────────────────────────────

/// A simple xorshift64 PRNG for the Random strategy.
#[derive(Debug)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() % (max as u64)) as usize
    }
}

// ── Load Balancer ─────────────────────────────────────────────

/// A load balancer managing a pool of backends.
pub struct LoadBalancer {
    backends: Vec<Backend>,
    strategy: Strategy,
    rr_index: usize,
    wrr_index: usize,
    wrr_current_weight: i32,
    ring: HashRing,
    rng: SimpleRng,
    /// Session affinity: session_key -> backend_id.
    sticky_sessions: HashMap<String, String>,
    /// Whether session affinity is enabled.
    pub sticky_enabled: bool,
}

impl LoadBalancer {
    pub fn new(strategy: Strategy) -> Self {
        Self {
            backends: Vec::new(),
            strategy,
            rr_index: 0,
            wrr_index: 0,
            wrr_current_weight: 0,
            ring: HashRing::new(150),
            rng: SimpleRng::new(42),
            sticky_sessions: HashMap::new(),
            sticky_enabled: false,
        }
    }

    /// Add a backend to the pool.
    pub fn add_backend(&mut self, backend: Backend) {
        let id = backend.id.clone();
        self.backends.push(backend);
        self.ring.add(&id);
    }

    /// Remove a backend by ID.
    pub fn remove_backend(&mut self, id: &str) -> Option<Backend> {
        let pos = self.backends.iter().position(|b| b.id == id)?;
        self.ring.remove(id);
        self.sticky_sessions.retain(|_, v| v != id);
        Some(self.backends.remove(pos))
    }

    /// Get all backends.
    pub fn backends(&self) -> &[Backend] {
        &self.backends
    }

    /// Get a mutable backend by ID.
    pub fn backend_mut(&mut self, id: &str) -> Option<&mut Backend> {
        self.backends.iter_mut().find(|b| b.id == id)
    }

    /// Get healthy backends.
    fn healthy_backends(&self) -> Vec<usize> {
        self.backends
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_healthy())
            .map(|(i, _)| i)
            .collect()
    }

    /// Select a backend for the given request key (used for hashing/affinity).
    pub fn select(&mut self, key: &str) -> Option<String> {
        // Check sticky session first.
        if self.sticky_enabled {
            if let Some(id) = self.sticky_sessions.get(key) {
                if let Some(b) = self.backends.iter().find(|b| b.id == *id) {
                    if b.is_healthy() {
                        return Some(id.clone());
                    }
                }
                // Sticky target is unhealthy; fall through.
                self.sticky_sessions.remove(key);
            }
        }

        let healthy = self.healthy_backends();
        if healthy.is_empty() {
            return None;
        }

        let idx = match self.strategy {
            Strategy::RoundRobin => self.select_round_robin(&healthy),
            Strategy::WeightedRoundRobin => self.select_weighted_rr(&healthy),
            Strategy::LeastConnections => self.select_least_conn(&healthy),
            Strategy::ConsistentHash => return self.select_consistent_hash(key, &healthy),
            Strategy::Random => self.select_random(&healthy),
            Strategy::IpHash => self.select_ip_hash(key, &healthy),
        };

        let id = self.backends[idx].id.clone();
        self.backends[idx].total_requests += 1;

        if self.sticky_enabled {
            self.sticky_sessions.insert(key.to_string(), id.clone());
        }

        Some(id)
    }

    fn select_round_robin(&mut self, healthy: &[usize]) -> usize {
        let result = healthy[self.rr_index % healthy.len()];
        self.rr_index = self.rr_index.wrapping_add(1);
        result
    }

    fn select_weighted_rr(&mut self, healthy: &[usize]) -> usize {
        // Smooth Weighted Round-Robin (Nginx style).
        loop {
            self.wrr_index = (self.wrr_index + 1) % healthy.len();
            if self.wrr_index == 0 {
                let max_weight = healthy.iter()
                    .map(|i| self.backends[*i].weight)
                    .max()
                    .unwrap_or(1);
                let gcd = healthy.iter()
                    .map(|i| self.backends[*i].weight)
                    .fold(0u32, gcd);
                self.wrr_current_weight -= gcd as i32;
                if self.wrr_current_weight <= 0 {
                    self.wrr_current_weight = max_weight as i32;
                }
            }
            let idx = healthy[self.wrr_index];
            if self.backends[idx].weight as i32 >= self.wrr_current_weight {
                return idx;
            }
        }
    }

    fn select_least_conn(&self, healthy: &[usize]) -> usize {
        let mut min_idx = healthy[0];
        let mut min_conn = self.backends[healthy[0]].active_connections;
        for i in healthy.iter().skip(1) {
            let conn = self.backends[*i].active_connections;
            if conn < min_conn {
                min_conn = conn;
                min_idx = *i;
            }
        }
        min_idx
    }

    fn select_consistent_hash(&mut self, key: &str, healthy: &[usize]) -> Option<String> {
        // Get from ring, but only return healthy backends.
        let target_id = self.ring.get(key)?;
        // Verify it's healthy.
        for idx in healthy {
            if self.backends[*idx].id == target_id {
                self.backends[*idx].total_requests += 1;
                let id = target_id.to_string();
                if self.sticky_enabled {
                    self.sticky_sessions.insert(key.to_string(), id.clone());
                }
                return Some(id);
            }
        }
        // Target unhealthy — fall back to next healthy on ring.
        let healthy_ids: Vec<&str> = healthy.iter().map(|i| self.backends[*i].id.as_str()).collect();
        // Walk the ring to find the next healthy.
        let hash = HashRing::hash_key(key);
        let ring = &self.ring;
        for entry in ring.entries.iter() {
            if entry.hash >= hash && healthy_ids.contains(&entry.backend_id.as_str()) {
                let chosen_id = entry.backend_id.clone();
                if let Some(b) = self.backends.iter_mut().find(|b| b.id == chosen_id) {
                    b.total_requests += 1;
                }
                return Some(chosen_id);
            }
        }
        // Wrap around.
        for entry in ring.entries.iter() {
            if healthy_ids.contains(&entry.backend_id.as_str()) {
                let chosen_id = entry.backend_id.clone();
                if let Some(b) = self.backends.iter_mut().find(|b| b.id == chosen_id) {
                    b.total_requests += 1;
                }
                return Some(chosen_id);
            }
        }
        None
    }

    fn select_random(&mut self, healthy: &[usize]) -> usize {
        let pick = self.rng.next_usize(healthy.len());
        healthy[pick]
    }

    fn select_ip_hash(&self, key: &str, healthy: &[usize]) -> usize {
        let hash = HashRing::hash_key(key);
        healthy[(hash as usize) % healthy.len()]
    }

    /// Record a connection being opened to a backend.
    pub fn connect(&mut self, id: &str) {
        if let Some(b) = self.backends.iter_mut().find(|b| b.id == id) {
            b.active_connections = b.active_connections.saturating_add(1);
        }
    }

    /// Record a connection being closed.
    pub fn disconnect(&mut self, id: &str) {
        if let Some(b) = self.backends.iter_mut().find(|b| b.id == id) {
            b.active_connections = b.active_connections.saturating_sub(1);
        }
    }

    /// Set health status for a backend.
    pub fn set_health(&mut self, id: &str, status: HealthStatus) {
        if let Some(b) = self.backends.iter_mut().find(|b| b.id == id) {
            b.health = status;
        }
    }

    /// Number of healthy backends.
    pub fn healthy_count(&self) -> usize {
        self.backends.iter().filter(|b| b.is_healthy()).count()
    }

    /// Total backends.
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Clear all sticky sessions.
    pub fn clear_sticky(&mut self) {
        self.sticky_sessions.clear();
    }
}

/// Compute GCD of two numbers.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ── Health Checker ────────────────────────────────────────────

/// Configuration for health checking.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Interval between checks in milliseconds.
    pub interval_ms: u64,
    /// Timeout for a single check in milliseconds.
    pub timeout_ms: u64,
    /// Number of consecutive successes to mark healthy.
    pub healthy_threshold: u32,
    /// Number of consecutive failures to mark unhealthy.
    pub unhealthy_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval_ms: 10_000,
            timeout_ms: 5_000,
            healthy_threshold: 2,
            unhealthy_threshold: 3,
        }
    }
}

/// Per-backend health check state.
#[derive(Debug, Clone)]
pub struct HealthCheckState {
    pub backend_id: String,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    pub last_check_ms: u64,
    pub status: HealthStatus,
}

impl HealthCheckState {
    pub fn new(backend_id: impl Into<String>) -> Self {
        Self {
            backend_id: backend_id.into(),
            consecutive_successes: 0,
            consecutive_failures: 0,
            last_check_ms: 0,
            status: HealthStatus::Unknown,
        }
    }

    /// Record a successful health check.
    pub fn record_success(&mut self, config: &HealthCheckConfig) {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
        if self.consecutive_successes >= config.healthy_threshold {
            self.status = HealthStatus::Healthy;
        }
    }

    /// Record a failed health check.
    pub fn record_failure(&mut self, config: &HealthCheckConfig) {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        if self.consecutive_failures >= config.unhealthy_threshold {
            self.status = HealthStatus::Unhealthy;
        }
    }

    /// Whether a check should run now.
    pub fn should_check(&self, now_ms: u64, config: &HealthCheckConfig) -> bool {
        now_ms.saturating_sub(self.last_check_ms) >= config.interval_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lb(strategy: Strategy) -> LoadBalancer {
        let mut lb = LoadBalancer::new(strategy);
        lb.add_backend(Backend::new("b1", "10.0.0.1", 80));
        lb.add_backend(Backend::new("b2", "10.0.0.2", 80));
        lb.add_backend(Backend::new("b3", "10.0.0.3", 80));
        lb
    }

    #[test]
    fn test_round_robin() {
        let mut lb = make_lb(Strategy::RoundRobin);
        let s1 = lb.select("r1").unwrap();
        let s2 = lb.select("r2").unwrap();
        let s3 = lb.select("r3").unwrap();
        let s4 = lb.select("r4").unwrap();
        // Should cycle.
        assert_eq!(s1, s4);
        // All three should appear.
        let mut seen = vec![s1, s2, s3];
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_round_robin_skip_unhealthy() {
        let mut lb = make_lb(Strategy::RoundRobin);
        lb.set_health("b2", HealthStatus::Unhealthy);
        // Should only return b1 and b3.
        for _ in 0..6 {
            let id = lb.select("x").unwrap();
            assert!(id == "b1" || id == "b3", "got {}", id);
        }
    }

    #[test]
    fn test_least_connections() {
        let mut lb = make_lb(Strategy::LeastConnections);
        lb.connect("b1");
        lb.connect("b1");
        lb.connect("b2");
        // b3 has 0 connections, should be selected.
        assert_eq!(lb.select("x").unwrap(), "b3");
    }

    #[test]
    fn test_consistent_hash_stability() {
        let mut lb = make_lb(Strategy::ConsistentHash);
        let a = lb.select("user-123").unwrap();
        let b = lb.select("user-123").unwrap();
        assert_eq!(a, b, "same key should map to same backend");
    }

    #[test]
    fn test_consistent_hash_different_keys() {
        let mut lb = make_lb(Strategy::ConsistentHash);
        // Different keys may or may not map to different backends, but
        // we verify the function returns something.
        assert!(lb.select("key-alpha").is_some());
        assert!(lb.select("key-beta").is_some());
    }

    #[test]
    fn test_random_selects_from_pool() {
        let mut lb = make_lb(Strategy::Random);
        for _ in 0..20 {
            let id = lb.select("x").unwrap();
            assert!(id == "b1" || id == "b2" || id == "b3");
        }
    }

    #[test]
    fn test_ip_hash_deterministic() {
        let mut lb = make_lb(Strategy::IpHash);
        let a = lb.select("192.168.1.1").unwrap();
        let b = lb.select("192.168.1.1").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_add_remove_backend() {
        let mut lb = LoadBalancer::new(Strategy::RoundRobin);
        lb.add_backend(Backend::new("b1", "10.0.0.1", 80));
        assert_eq!(lb.backend_count(), 1);
        lb.remove_backend("b1");
        assert_eq!(lb.backend_count(), 0);
        assert!(lb.select("x").is_none());
    }

    #[test]
    fn test_all_unhealthy_returns_none() {
        let mut lb = make_lb(Strategy::RoundRobin);
        lb.set_health("b1", HealthStatus::Unhealthy);
        lb.set_health("b2", HealthStatus::Unhealthy);
        lb.set_health("b3", HealthStatus::Unhealthy);
        assert!(lb.select("x").is_none());
    }

    #[test]
    fn test_sticky_sessions() {
        let mut lb = make_lb(Strategy::RoundRobin);
        lb.sticky_enabled = true;
        let first = lb.select("session-abc").unwrap();
        // Subsequent requests with same key should go to same backend.
        for _ in 0..5 {
            assert_eq!(lb.select("session-abc").unwrap(), first);
        }
    }

    #[test]
    fn test_sticky_unhealthy_failover() {
        let mut lb = make_lb(Strategy::RoundRobin);
        lb.sticky_enabled = true;
        let first = lb.select("sess-1").unwrap();
        lb.set_health(&first, HealthStatus::Unhealthy);
        let second = lb.select("sess-1").unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn test_connect_disconnect() {
        let mut lb = make_lb(Strategy::LeastConnections);
        lb.connect("b1");
        lb.connect("b1");
        assert_eq!(lb.backends().iter().find(|b| b.id == "b1").unwrap().active_connections, 2);
        lb.disconnect("b1");
        assert_eq!(lb.backends().iter().find(|b| b.id == "b1").unwrap().active_connections, 1);
    }

    #[test]
    fn test_backend_display() {
        let b = Backend::new("web1", "10.0.0.1", 8080);
        assert_eq!(format!("{}", b), "10.0.0.1:8080(web1)");
    }

    #[test]
    fn test_weighted_round_robin() {
        let mut lb = LoadBalancer::new(Strategy::WeightedRoundRobin);
        lb.add_backend(Backend::new("b1", "10.0.0.1", 80).with_weight(3));
        lb.add_backend(Backend::new("b2", "10.0.0.2", 80).with_weight(1));

        let mut counts: HashMap<String, u32> = HashMap::new();
        for i in 0..40 {
            let key = format!("r{}", i);
            let id = lb.select(&key).unwrap();
            *counts.entry(id).or_insert(0) += 1;
        }
        // b1 (weight 3) should get significantly more than b2 (weight 1).
        let b1_count = counts.get("b1").copied().unwrap_or(0);
        let b2_count = counts.get("b2").copied().unwrap_or(0);
        assert!(b1_count > b2_count, "b1={}, b2={}", b1_count, b2_count);
    }

    #[test]
    fn test_health_check_state() {
        let config = HealthCheckConfig::default();
        let mut state = HealthCheckState::new("b1");
        assert_eq!(state.status, HealthStatus::Unknown);
        state.record_success(&config);
        state.record_success(&config);
        assert_eq!(state.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_check_failure_threshold() {
        let config = HealthCheckConfig { unhealthy_threshold: 2, ..Default::default() };
        let mut state = HealthCheckState::new("b1");
        state.record_failure(&config);
        assert_ne!(state.status, HealthStatus::Unhealthy);
        state.record_failure(&config);
        assert_eq!(state.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_check_should_check() {
        let config = HealthCheckConfig { interval_ms: 5000, ..Default::default() };
        let state = HealthCheckState::new("b1");
        assert!(state.should_check(5000, &config));
        assert!(!state.should_check(3000, &config));
    }

    #[test]
    fn test_clear_sticky() {
        let mut lb = make_lb(Strategy::RoundRobin);
        lb.sticky_enabled = true;
        lb.select("s1").unwrap();
        lb.clear_sticky();
        // After clearing, affinity is gone.
        // We can't assert a different backend, but we verify it doesn't crash.
        assert!(lb.select("s1").is_some());
    }
}
