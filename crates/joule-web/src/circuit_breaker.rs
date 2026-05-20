//! Circuit breaker pattern — closed/open/half-open states with failure tracking.
//!
//! Pure Rust implementation of the circuit breaker pattern for resilience.
//! Tracks failure rates, transitions between states with configurable
//! thresholds, and supports per-operation breakers and event callbacks.

use std::collections::HashMap;

// ── Circuit State ──────────────────────────────────────────────

/// The three states of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests pass through.
    Closed,
    /// Tripped — requests are rejected.
    Open,
    /// Testing — a limited number of requests pass to probe recovery.
    HalfOpen,
}

// ── Event ──────────────────────────────────────────────────────

/// Events emitted by the circuit breaker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitEvent {
    /// Transitioned from Closed to Open.
    Opened { failures: u64, threshold: u64 },
    /// Transitioned from Open to HalfOpen.
    HalfOpened,
    /// Transitioned from HalfOpen to Closed (recovered).
    Recovered { successes: u64 },
    /// Transitioned from HalfOpen to Open (still failing).
    RejectedProbe,
    /// A call was rejected because the circuit is open.
    Rejected,
    /// A call succeeded.
    Success,
    /// A call failed.
    Failure,
}

// ── Call Result ─────────────────────────────────────────────────

/// Whether the circuit breaker allows a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallPermission {
    Allowed,
    Rejected,
}

// ── Config ─────────────────────────────────────────────────────

/// Configuration for a circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trip open.
    pub failure_threshold: u64,
    /// Number of consecutive successes in half-open to recover.
    pub success_threshold: u64,
    /// Time in ms before transitioning from open to half-open.
    pub open_timeout_ms: u64,
    /// Maximum number of probe requests in half-open state.
    pub half_open_max_probes: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            open_timeout_ms: 30_000,
            half_open_max_probes: 3,
        }
    }
}

// ── Circuit Breaker ────────────────────────────────────────────

/// A single circuit breaker instance.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    /// Consecutive failure count (reset on success).
    consecutive_failures: u64,
    /// Consecutive success count in half-open (reset on failure).
    half_open_successes: u64,
    /// Probes sent in current half-open phase.
    half_open_probes: u64,
    /// Simulated time when the circuit was opened.
    opened_at_ms: u64,
    /// Current simulated time.
    current_time_ms: u64,
    /// Total counts.
    total_successes: u64,
    total_failures: u64,
    total_rejected: u64,
    /// Rolling window of recent results for failure rate (true=success, false=failure).
    recent_results: std::collections::VecDeque<bool>,
    /// Window size for failure rate calculation.
    window_size: usize,
    /// Event log.
    events: Vec<CircuitEvent>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            half_open_successes: 0,
            half_open_probes: 0,
            opened_at_ms: 0,
            current_time_ms: 0,
            total_successes: 0,
            total_failures: 0,
            total_rejected: 0,
            recent_results: std::collections::VecDeque::new(),
            window_size: 100,
            events: Vec::new(),
        }
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(1);
        self
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    pub fn total_successes(&self) -> u64 {
        self.total_successes
    }

    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    pub fn events(&self) -> &[CircuitEvent] {
        &self.events
    }

    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Failure rate over the rolling window (0.0 to 1.0).
    pub fn failure_rate(&self) -> f64 {
        if self.recent_results.is_empty() {
            return 0.0;
        }
        let failures = self.recent_results.iter().filter(|r| !**r).count();
        failures as f64 / self.recent_results.len() as f64
    }

    /// Advance simulated time.
    pub fn advance_time(&mut self, ms: u64) {
        self.current_time_ms += ms;
        // Check if open→half-open transition is due.
        if self.state == CircuitState::Open {
            let elapsed = self.current_time_ms.saturating_sub(self.opened_at_ms);
            if elapsed >= self.config.open_timeout_ms {
                self.state = CircuitState::HalfOpen;
                self.half_open_successes = 0;
                self.half_open_probes = 0;
                self.events.push(CircuitEvent::HalfOpened);
            }
        }
    }

    /// Check if a call is allowed.
    pub fn check(&mut self) -> CallPermission {
        match self.state {
            CircuitState::Closed => CallPermission::Allowed,
            CircuitState::Open => {
                self.total_rejected += 1;
                self.events.push(CircuitEvent::Rejected);
                CallPermission::Rejected
            }
            CircuitState::HalfOpen => {
                if self.half_open_probes < self.config.half_open_max_probes {
                    self.half_open_probes += 1;
                    CallPermission::Allowed
                } else {
                    self.total_rejected += 1;
                    self.events.push(CircuitEvent::Rejected);
                    CallPermission::Rejected
                }
            }
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.total_successes += 1;
        self.consecutive_failures = 0;
        self.push_result(true);
        self.events.push(CircuitEvent::Success);

        match self.state {
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.events.push(CircuitEvent::Recovered {
                        successes: self.half_open_successes,
                    });
                }
            }
            _ => {}
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.total_failures += 1;
        self.consecutive_failures += 1;
        self.push_result(false);
        self.events.push(CircuitEvent::Failure);

        match self.state {
            CircuitState::Closed => {
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.trip_open();
                }
            }
            CircuitState::HalfOpen => {
                self.trip_open();
                self.events.push(CircuitEvent::RejectedProbe);
            }
            _ => {}
        }
    }

    fn trip_open(&mut self) {
        let failures = self.consecutive_failures;
        self.state = CircuitState::Open;
        self.opened_at_ms = self.current_time_ms;
        self.events.push(CircuitEvent::Opened {
            failures,
            threshold: self.config.failure_threshold,
        });
    }

    fn push_result(&mut self, success: bool) {
        self.recent_results.push_back(success);
        if self.recent_results.len() > self.window_size {
            self.recent_results.pop_front();
        }
    }

    /// Reset to closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.half_open_successes = 0;
        self.half_open_probes = 0;
        self.recent_results.clear();
    }
}

// ── Per-Operation Circuit Breakers ─────────────────────────────

/// Registry of circuit breakers keyed by operation name.
#[derive(Debug)]
pub struct CircuitBreakerRegistry {
    breakers: HashMap<String, CircuitBreaker>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: HashMap::new(),
            default_config,
        }
    }

    /// Get or create a circuit breaker for the given operation.
    pub fn get_or_create(&mut self, operation: &str) -> &mut CircuitBreaker {
        let config = self.default_config.clone();
        self.breakers
            .entry(operation.to_string())
            .or_insert_with(|| CircuitBreaker::new(config))
    }

    /// Get a circuit breaker if it exists.
    pub fn get(&self, operation: &str) -> Option<&CircuitBreaker> {
        self.breakers.get(operation)
    }

    /// Get a mutable circuit breaker if it exists.
    pub fn get_mut(&mut self, operation: &str) -> Option<&mut CircuitBreaker> {
        self.breakers.get_mut(operation)
    }

    /// Number of registered breakers.
    pub fn len(&self) -> usize {
        self.breakers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.breakers.is_empty()
    }

    /// Advance time for all breakers.
    pub fn advance_time(&mut self, ms: u64) {
        for cb in self.breakers.values_mut() {
            cb.advance_time(ms);
        }
    }

    /// List all operation names.
    pub fn operations(&self) -> Vec<&str> {
        self.breakers.keys().map(|s| s.as_str()).collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cb() -> CircuitBreaker {
        CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            open_timeout_ms: 1000,
            half_open_max_probes: 2,
        })
    }

    #[test]
    fn test_starts_closed() {
        let cb = default_cb();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_closed_allows_calls() {
        let mut cb = default_cb();
        assert_eq!(cb.check(), CallPermission::Allowed);
    }

    #[test]
    fn test_trips_open_on_failures() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_open_rejects_calls() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        assert_eq!(cb.check(), CallPermission::Rejected);
        assert_eq!(cb.total_rejected(), 1);
    }

    #[test]
    fn test_open_to_half_open_on_timeout() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        cb.advance_time(1000);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_recovery() {
        let mut cb = default_cb();
        // Trip open.
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        cb.advance_time(1000); // → HalfOpen.
        // Two successes → recover.
        cb.check();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.check();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_failure_trips_open() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        cb.advance_time(1000);
        cb.check();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_half_open_probe_limit() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        cb.advance_time(1000);
        assert_eq!(cb.check(), CallPermission::Allowed); // probe 1
        assert_eq!(cb.check(), CallPermission::Allowed); // probe 2
        assert_eq!(cb.check(), CallPermission::Rejected); // max probes exceeded
    }

    #[test]
    fn test_failure_rate() {
        let mut cb = default_cb().with_window_size(10);
        for _ in 0..3 {
            cb.record_success();
        }
        for _ in 0..7 {
            cb.push_result(false);
        }
        let rate = cb.failure_rate();
        assert!((rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_success_resets_failure_count() {
        let mut cb = default_cb();
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // Resets consecutive failures.
        cb.record_failure();
        cb.record_failure();
        // Only 2 consecutive failures, threshold is 3 — should still be closed.
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_event_log() {
        let mut cb = default_cb();
        cb.check();
        cb.record_failure();
        assert!(cb.events().iter().any(|e| matches!(e, CircuitEvent::Failure)));
        cb.clear_events();
        assert!(cb.events().is_empty());
    }

    #[test]
    fn test_registry() {
        let mut reg = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        let cb = reg.get_or_create("api_call");
        cb.record_failure();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("api_call").is_some());
        assert!(reg.get("other").is_none());
    }

    #[test]
    fn test_registry_advance_time() {
        let mut reg = CircuitBreakerRegistry::new(CircuitBreakerConfig {
            failure_threshold: 1,
            open_timeout_ms: 500,
            ..Default::default()
        });
        let cb = reg.get_or_create("op");
        cb.check();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        reg.advance_time(500);
        assert_eq!(reg.get("op").unwrap().state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_reset() {
        let mut cb = default_cb();
        for _ in 0..3 {
            cb.check();
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.check(), CallPermission::Allowed);
    }
}
