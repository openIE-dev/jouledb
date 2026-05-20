//! Advanced circuit breaker — configurable thresholds, fallbacks, metrics, callbacks.
//!
//! Extends the basic circuit breaker with sliding window failure rates,
//! configurable success/failure thresholds, timeout-based recovery,
//! fallback support, detailed metrics, and event callback hooks.

use std::collections::VecDeque;
use std::fmt;

// ── State ─────────────────────────────────────────────────────

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl fmt::Display for BreakerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Closed => "CLOSED",
            Self::Open => "OPEN",
            Self::HalfOpen => "HALF_OPEN",
        };
        f.write_str(s)
    }
}

// ── Call Outcome ──────────────────────────────────────────────

/// Result of a single call through the breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallOutcome {
    Success,
    Failure,
    Timeout,
    Rejected,
}

// ── Events ────────────────────────────────────────────────────

/// Events emitted by the circuit breaker.
#[derive(Debug, Clone, PartialEq)]
pub enum BreakerEvent {
    StateChanged { from: BreakerState, to: BreakerState, timestamp_ms: u64 },
    CallSucceeded { duration_ms: u64 },
    CallFailed { duration_ms: u64 },
    CallTimedOut { duration_ms: u64 },
    CallRejected,
    FallbackExecuted,
    ThresholdReached { failure_rate: f64, threshold: f64 },
    RecoverySucceeded { consecutive_successes: u32 },
}

// ── Config ────────────────────────────────────────────────────

/// Advanced circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct BreakerConfig {
    /// Failure rate threshold (0.0–1.0) to trip open.
    pub failure_rate_threshold: f64,
    /// Minimum number of calls in the window before evaluating failure rate.
    pub minimum_calls: u32,
    /// Sliding window size (number of calls).
    pub window_size: u32,
    /// Number of consecutive successes in half-open to recover.
    pub success_threshold: u32,
    /// Maximum number of probe calls allowed in half-open.
    pub half_open_max_calls: u32,
    /// Duration in ms to remain open before transitioning to half-open.
    pub open_duration_ms: u64,
    /// Call timeout in ms (calls exceeding this are counted as timeout failures).
    pub call_timeout_ms: u64,
    /// Whether to count timeout as failure.
    pub timeout_is_failure: bool,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_rate_threshold: 0.5,
            minimum_calls: 5,
            window_size: 20,
            success_threshold: 3,
            half_open_max_calls: 3,
            open_duration_ms: 30_000,
            call_timeout_ms: 10_000,
            timeout_is_failure: true,
        }
    }
}

// ── Metrics ───────────────────────────────────────────────────

/// Circuit breaker metrics.
#[derive(Debug, Clone)]
pub struct BreakerMetrics {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub timed_out_calls: u64,
    pub rejected_calls: u64,
    pub fallback_calls: u64,
    pub state_changes: u64,
    pub current_failure_rate: f64,
    pub total_call_duration_ms: u64,
}

impl BreakerMetrics {
    fn new() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            timed_out_calls: 0,
            rejected_calls: 0,
            fallback_calls: 0,
            state_changes: 0,
            current_failure_rate: 0.0,
            total_call_duration_ms: 0,
        }
    }

    /// Average call duration in ms. Returns 0 if no calls.
    pub fn avg_duration_ms(&self) -> u64 {
        let completed = self.successful_calls + self.failed_calls + self.timed_out_calls;
        if completed == 0 {
            0
        } else {
            self.total_call_duration_ms / completed
        }
    }
}

// ── Sliding Window ────────────────────────────────────────────

/// A sliding window tracking call outcomes.
#[derive(Debug, Clone)]
struct SlidingWindow {
    outcomes: VecDeque<CallOutcome>,
    max_size: usize,
}

impl SlidingWindow {
    fn new(max_size: u32) -> Self {
        Self {
            outcomes: VecDeque::with_capacity(max_size as usize),
            max_size: max_size as usize,
        }
    }

    fn push(&mut self, outcome: CallOutcome) {
        if self.outcomes.len() >= self.max_size {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(outcome);
    }

    fn len(&self) -> usize {
        self.outcomes.len()
    }

    fn failure_count(&self) -> usize {
        self.outcomes.iter().filter(|o| matches!(o, CallOutcome::Failure | CallOutcome::Timeout)).count()
    }

    fn failure_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        self.failure_count() as f64 / self.outcomes.len() as f64
    }

    fn clear(&mut self) {
        self.outcomes.clear();
    }
}

// ── Circuit Breaker ───────────────────────────────────────────

/// Advanced circuit breaker.
pub struct CircuitBreakerAdv {
    config: BreakerConfig,
    state: BreakerState,
    window: SlidingWindow,
    metrics: BreakerMetrics,
    /// When the breaker entered the OPEN state (ms).
    opened_at_ms: u64,
    /// Consecutive successes in half-open.
    half_open_successes: u32,
    /// Number of calls allowed through in half-open.
    half_open_calls: u32,
    /// Event log.
    events: Vec<BreakerEvent>,
    /// Maximum events to keep.
    max_events: usize,
}

impl CircuitBreakerAdv {
    pub fn new(config: BreakerConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            state: BreakerState::Closed,
            window: SlidingWindow::new(window_size),
            metrics: BreakerMetrics::new(),
            opened_at_ms: 0,
            half_open_successes: 0,
            half_open_calls: 0,
            events: Vec::new(),
            max_events: 1000,
        }
    }

    /// Current state.
    pub fn state(&self) -> BreakerState {
        self.state
    }

    /// Configuration.
    pub fn config(&self) -> &BreakerConfig {
        &self.config
    }

    /// Metrics snapshot.
    pub fn metrics(&self) -> &BreakerMetrics {
        &self.metrics
    }

    /// Get buffered events.
    pub fn events(&self) -> &[BreakerEvent] {
        &self.events
    }

    /// Drain all events.
    pub fn drain_events(&mut self) -> Vec<BreakerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if a call is allowed at the given timestamp.
    /// Automatically transitions OPEN -> HALF_OPEN when timeout expires.
    pub fn allow_call(&mut self, now_ms: u64) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => {
                if now_ms.saturating_sub(self.opened_at_ms) >= self.config.open_duration_ms {
                    self.transition(BreakerState::HalfOpen, now_ms);
                    self.half_open_calls < self.config.half_open_max_calls
                } else {
                    self.metrics.rejected_calls += 1;
                    self.emit(BreakerEvent::CallRejected);
                    false
                }
            }
            BreakerState::HalfOpen => {
                if self.half_open_calls < self.config.half_open_max_calls {
                    true
                } else {
                    self.metrics.rejected_calls += 1;
                    self.emit(BreakerEvent::CallRejected);
                    false
                }
            }
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self, duration_ms: u64, now_ms: u64) {
        self.metrics.total_calls += 1;
        self.metrics.successful_calls += 1;
        self.metrics.total_call_duration_ms = self.metrics.total_call_duration_ms.saturating_add(duration_ms);
        self.emit(BreakerEvent::CallSucceeded { duration_ms });

        match self.state {
            BreakerState::Closed => {
                self.window.push(CallOutcome::Success);
                self.update_failure_rate();
            }
            BreakerState::HalfOpen => {
                self.half_open_calls += 1;
                self.half_open_successes += 1;
                if self.half_open_successes >= self.config.success_threshold {
                    self.emit(BreakerEvent::RecoverySucceeded {
                        consecutive_successes: self.half_open_successes,
                    });
                    self.transition(BreakerState::Closed, now_ms);
                    self.window.clear();
                }
            }
            BreakerState::Open => {} // Shouldn't happen if allow_call was checked.
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self, duration_ms: u64, now_ms: u64) {
        self.metrics.total_calls += 1;
        self.metrics.failed_calls += 1;
        self.metrics.total_call_duration_ms = self.metrics.total_call_duration_ms.saturating_add(duration_ms);
        self.emit(BreakerEvent::CallFailed { duration_ms });

        match self.state {
            BreakerState::Closed => {
                self.window.push(CallOutcome::Failure);
                self.update_failure_rate();
                self.maybe_trip(now_ms);
            }
            BreakerState::HalfOpen => {
                self.half_open_calls += 1;
                // Any failure in half-open trips back to open.
                self.transition(BreakerState::Open, now_ms);
            }
            BreakerState::Open => {}
        }
    }

    /// Record a timed-out call.
    pub fn record_timeout(&mut self, duration_ms: u64, now_ms: u64) {
        self.metrics.total_calls += 1;
        self.metrics.timed_out_calls += 1;
        self.metrics.total_call_duration_ms = self.metrics.total_call_duration_ms.saturating_add(duration_ms);
        self.emit(BreakerEvent::CallTimedOut { duration_ms });

        if self.config.timeout_is_failure {
            match self.state {
                BreakerState::Closed => {
                    self.window.push(CallOutcome::Timeout);
                    self.update_failure_rate();
                    self.maybe_trip(now_ms);
                }
                BreakerState::HalfOpen => {
                    self.half_open_calls += 1;
                    self.transition(BreakerState::Open, now_ms);
                }
                BreakerState::Open => {}
            }
        }
    }

    /// Record a fallback execution.
    pub fn record_fallback(&mut self) {
        self.metrics.fallback_calls += 1;
        self.emit(BreakerEvent::FallbackExecuted);
    }

    /// Execute with the circuit breaker, using a fallback if rejected.
    /// Returns the outcome and whether the fallback was used.
    pub fn execute<T, F, G>(
        &mut self,
        now_ms: u64,
        primary: F,
        fallback: G,
    ) -> (T, bool)
    where
        F: FnOnce() -> Result<T, T>,
        G: FnOnce() -> T,
    {
        if !self.allow_call(now_ms) {
            self.record_fallback();
            return (fallback(), true);
        }

        match primary() {
            Ok(val) => {
                self.record_success(0, now_ms);
                (val, false)
            }
            Err(val) => {
                self.record_failure(0, now_ms);
                (val, false)
            }
        }
    }

    /// Force the breaker into a specific state (for testing/manual override).
    pub fn force_state(&mut self, state: BreakerState, now_ms: u64) {
        self.transition(state, now_ms);
    }

    /// Reset the breaker to initial closed state.
    pub fn reset(&mut self) {
        self.state = BreakerState::Closed;
        self.window.clear();
        self.metrics = BreakerMetrics::new();
        self.opened_at_ms = 0;
        self.half_open_successes = 0;
        self.half_open_calls = 0;
        self.events.clear();
    }

    // ── Internal ──────────────────────────────────────────────

    fn transition(&mut self, new_state: BreakerState, now_ms: u64) {
        let old = self.state;
        if old == new_state {
            return;
        }
        self.state = new_state;
        self.metrics.state_changes += 1;

        match new_state {
            BreakerState::Open => {
                self.opened_at_ms = now_ms;
                self.half_open_successes = 0;
                self.half_open_calls = 0;
            }
            BreakerState::HalfOpen => {
                self.half_open_successes = 0;
                self.half_open_calls = 0;
            }
            BreakerState::Closed => {
                self.half_open_successes = 0;
                self.half_open_calls = 0;
            }
        }

        self.emit(BreakerEvent::StateChanged { from: old, to: new_state, timestamp_ms: now_ms });
    }

    fn update_failure_rate(&mut self) {
        self.metrics.current_failure_rate = self.window.failure_rate();
    }

    fn maybe_trip(&mut self, now_ms: u64) {
        if self.window.len() < self.config.minimum_calls as usize {
            return;
        }
        let rate = self.window.failure_rate();
        if rate >= self.config.failure_rate_threshold {
            self.emit(BreakerEvent::ThresholdReached {
                failure_rate: rate,
                threshold: self.config.failure_rate_threshold,
            });
            self.transition(BreakerState::Open, now_ms);
        }
    }

    fn emit(&mut self, event: BreakerEvent) {
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }
}

impl fmt::Debug for CircuitBreakerAdv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CircuitBreakerAdv")
            .field("state", &self.state)
            .field("failure_rate", &self.metrics.current_failure_rate)
            .field("total_calls", &self.metrics.total_calls)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_breaker() -> CircuitBreakerAdv {
        CircuitBreakerAdv::new(BreakerConfig::default())
    }

    #[test]
    fn test_initial_state() {
        let cb = default_breaker();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert_eq!(cb.metrics().total_calls, 0);
    }

    #[test]
    fn test_allow_call_closed() {
        let mut cb = default_breaker();
        assert!(cb.allow_call(1000));
    }

    #[test]
    fn test_success_stays_closed() {
        let mut cb = default_breaker();
        for i in 0..10 {
            assert!(cb.allow_call(i * 100));
            cb.record_success(10, i * 100);
        }
        assert_eq!(cb.state(), BreakerState::Closed);
        assert_eq!(cb.metrics().successful_calls, 10);
    }

    #[test]
    fn test_trip_on_failures() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 4,
            window_size: 10,
            ..Default::default()
        });

        for i in 0..4 {
            cb.allow_call(i * 100);
            cb.record_failure(10, i * 100);
        }
        assert_eq!(cb.state(), BreakerState::Open);
    }

    #[test]
    fn test_open_rejects_calls() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);

        assert_eq!(cb.state(), BreakerState::Open);
        assert!(!cb.allow_call(200));
        assert!(cb.metrics().rejected_calls > 0);
    }

    #[test]
    fn test_open_to_half_open_on_timeout() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            open_duration_ms: 5000,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);
        assert_eq!(cb.state(), BreakerState::Open);

        // After open duration, should transition to half-open.
        assert!(cb.allow_call(6000));
        assert_eq!(cb.state(), BreakerState::HalfOpen);
    }

    #[test]
    fn test_half_open_recovery() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            success_threshold: 2,
            open_duration_ms: 1000,
            window_size: 10,
            ..Default::default()
        });

        // Trip open.
        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);

        // Transition to half-open.
        cb.allow_call(2000);
        assert_eq!(cb.state(), BreakerState::HalfOpen);

        // Succeed enough times.
        cb.record_success(10, 2000);
        cb.allow_call(2100);
        cb.record_success(10, 2100);

        assert_eq!(cb.state(), BreakerState::Closed);
    }

    #[test]
    fn test_half_open_failure_trips_open() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            open_duration_ms: 1000,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);

        cb.allow_call(2000);
        assert_eq!(cb.state(), BreakerState::HalfOpen);

        cb.record_failure(10, 2000);
        assert_eq!(cb.state(), BreakerState::Open);
    }

    #[test]
    fn test_half_open_max_calls() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            half_open_max_calls: 1,
            success_threshold: 3,
            open_duration_ms: 1000,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);

        cb.allow_call(2000);
        assert_eq!(cb.state(), BreakerState::HalfOpen);
        cb.record_success(10, 2000);

        // Max calls reached (1), further calls rejected.
        assert!(!cb.allow_call(2100));
    }

    #[test]
    fn test_timeout_counts_as_failure() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            timeout_is_failure: true,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_timeout(15000, 0);
        cb.allow_call(100);
        cb.record_timeout(15000, 100);
        assert_eq!(cb.state(), BreakerState::Open);
    }

    #[test]
    fn test_timeout_not_failure() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            timeout_is_failure: false,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_timeout(15000, 0);
        cb.allow_call(100);
        cb.record_timeout(15000, 100);
        assert_eq!(cb.state(), BreakerState::Closed); // Timeouts not counted.
    }

    #[test]
    fn test_execute_with_fallback() {
        let mut cb = default_breaker();
        cb.force_state(BreakerState::Open, 0);

        let (val, used_fallback) = cb.execute(100, || Ok::<i32, i32>(42), || -1);
        assert_eq!(val, -1);
        assert!(used_fallback);
    }

    #[test]
    fn test_execute_primary_success() {
        let mut cb = default_breaker();
        let (val, used_fallback) = cb.execute(0, || Ok::<i32, i32>(42), || -1);
        assert_eq!(val, 42);
        assert!(!used_fallback);
    }

    #[test]
    fn test_execute_primary_failure() {
        let mut cb = default_breaker();
        let (val, used_fallback) = cb.execute(0, || Err::<i32, i32>(99), || -1);
        assert_eq!(val, 99);
        assert!(!used_fallback);
        assert_eq!(cb.metrics().failed_calls, 1);
    }

    #[test]
    fn test_metrics_avg_duration() {
        let mut cb = default_breaker();
        cb.allow_call(0);
        cb.record_success(100, 0);
        cb.allow_call(100);
        cb.record_success(200, 100);
        assert_eq!(cb.metrics().avg_duration_ms(), 150);
    }

    #[test]
    fn test_force_state() {
        let mut cb = default_breaker();
        cb.force_state(BreakerState::Open, 1000);
        assert_eq!(cb.state(), BreakerState::Open);
        cb.force_state(BreakerState::Closed, 2000);
        assert_eq!(cb.state(), BreakerState::Closed);
    }

    #[test]
    fn test_reset() {
        let mut cb = default_breaker();
        cb.allow_call(0);
        cb.record_success(10, 0);
        cb.reset();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert_eq!(cb.metrics().total_calls, 0);
        assert!(cb.events().is_empty());
    }

    #[test]
    fn test_events_emitted() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            failure_rate_threshold: 0.5,
            minimum_calls: 2,
            window_size: 10,
            ..Default::default()
        });

        cb.allow_call(0);
        cb.record_failure(10, 0);
        cb.allow_call(100);
        cb.record_failure(10, 100);

        let events = cb.drain_events();
        assert!(!events.is_empty());
        // Should include failure events and state change.
        let has_state_change = events.iter().any(|e| matches!(e, BreakerEvent::StateChanged { .. }));
        assert!(has_state_change);
    }

    #[test]
    fn test_breaker_state_display() {
        assert_eq!(format!("{}", BreakerState::Closed), "CLOSED");
        assert_eq!(format!("{}", BreakerState::Open), "OPEN");
        assert_eq!(format!("{}", BreakerState::HalfOpen), "HALF_OPEN");
    }

    #[test]
    fn test_sliding_window_eviction() {
        let mut cb = CircuitBreakerAdv::new(BreakerConfig {
            window_size: 5,
            minimum_calls: 5,
            failure_rate_threshold: 0.6,
            ..Default::default()
        });

        // Fill window with successes.
        for i in 0..5 {
            cb.allow_call(i * 10);
            cb.record_success(5, i * 10);
        }
        assert_eq!(cb.state(), BreakerState::Closed);

        // Add failures — old successes slide out.
        for i in 5..10 {
            cb.allow_call(i * 10);
            cb.record_failure(5, i * 10);
        }
        // Window now has most recent 5 calls, all failures.
        assert_eq!(cb.state(), BreakerState::Open);
    }
}
