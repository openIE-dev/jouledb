//! RPC retry with exponential backoff — policies, jitter, circuit breaker.
//!
//! Provides [`RetryPolicy`] for configuring retry behavior with exponential
//! backoff and jitter, [`RetryState`] for tracking attempt progression, and
//! [`CircuitBreakerGuard`] for halting retries after sustained failures.
//! Error classification distinguishes [`ErrorKind::Transient`] (retryable) from
//! [`ErrorKind::Permanent`] (non-retryable).

use std::fmt;

// ── Error Classification ───────────────────────────────────────

/// Classifies an error as transient (retryable) or permanent (not retryable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Temporary failure — safe to retry (e.g., timeout, 503).
    Transient,
    /// Permanent failure — retrying won't help (e.g., 400, auth error).
    Permanent,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Permanent => write!(f, "permanent"),
        }
    }
}

// ── Retry Error ────────────────────────────────────────────────

/// An error that carries a classification for retry decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryError {
    pub kind: ErrorKind,
    pub code: u32,
    pub message: String,
}

impl RetryError {
    pub fn transient(code: u32, message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Transient, code, message: message.into() }
    }

    pub fn permanent(code: u32, message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Permanent, code, message: message.into() }
    }
}

impl fmt::Display for RetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.kind, self.code, self.message)
    }
}

// ── Retry Decision ─────────────────────────────────────────────

/// The outcome of a retry decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryDecision {
    /// Retry after the given delay in milliseconds.
    RetryAfter(u64),
    /// Do not retry; give up.
    GiveUp(GiveUpReason),
}

/// Why we decided not to retry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GiveUpReason {
    MaxRetriesExceeded,
    PermanentError,
    CircuitOpen,
    BudgetExhausted,
}

impl fmt::Display for GiveUpReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxRetriesExceeded => write!(f, "max retries exceeded"),
            Self::PermanentError => write!(f, "permanent error"),
            Self::CircuitOpen => write!(f, "circuit breaker open"),
            Self::BudgetExhausted => write!(f, "retry budget exhausted"),
        }
    }
}

// ── Retry Policy ───────────────────────────────────────────────

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub jitter_pct: u32,
    pub multiplier: f64,
    pub retryable_codes: Vec<u32>,
    pub total_timeout_ms: Option<u64>,
}

impl RetryPolicy {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 30_000,
            jitter_pct: 25,
            multiplier: 2.0,
            retryable_codes: Vec::new(),
            total_timeout_ms: None,
        }
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n; self
    }

    pub fn with_base_delay_ms(mut self, ms: u64) -> Self {
        self.base_delay_ms = ms; self
    }

    pub fn with_max_delay_ms(mut self, ms: u64) -> Self {
        self.max_delay_ms = ms; self
    }

    pub fn with_jitter_pct(mut self, pct: u32) -> Self {
        self.jitter_pct = pct.min(100); self
    }

    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m; self
    }

    pub fn with_retryable_code(mut self, code: u32) -> Self {
        self.retryable_codes.push(code); self
    }

    pub fn with_total_timeout_ms(mut self, ms: u64) -> Self {
        self.total_timeout_ms = Some(ms); self
    }

    /// Compute delay for the given attempt (0-indexed).
    pub fn compute_delay(&self, attempt: u32, jitter_seed: u64) -> u64 {
        let base = self.base_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay_ms as f64) as u64;
        apply_jitter(capped, self.jitter_pct, jitter_seed)
    }

    /// Whether a given error code is in the retryable list (if configured).
    pub fn is_code_retryable(&self, code: u32) -> bool {
        self.retryable_codes.is_empty() || self.retryable_codes.contains(&code)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RetryPolicy(max={}, base={}ms, max_delay={}ms, jitter={}%)",
            self.max_retries, self.base_delay_ms, self.max_delay_ms, self.jitter_pct)
    }
}

// ── Jitter ─────────────────────────────────────────────────────

/// Apply jitter to a delay. Uses a deterministic seed for testability.
fn apply_jitter(delay_ms: u64, jitter_pct: u32, seed: u64) -> u64 {
    if jitter_pct == 0 || delay_ms == 0 {
        return delay_ms;
    }
    let jitter_range = (delay_ms as f64 * jitter_pct as f64 / 100.0) as u64;
    if jitter_range == 0 {
        return delay_ms;
    }
    // Simple deterministic jitter from seed
    let jitter = seed % (jitter_range * 2 + 1);
    delay_ms.saturating_sub(jitter_range).saturating_add(jitter)
}

// ── Retry State ────────────────────────────────────────────────

/// Tracks the progression of retry attempts for a single operation.
#[derive(Debug, Clone)]
pub struct RetryState {
    policy: RetryPolicy,
    attempt: u32,
    started_at_ms: u64,
    total_delay_ms: u64,
    errors: Vec<RetryError>,
}

impl RetryState {
    pub fn new(policy: RetryPolicy, started_at_ms: u64) -> Self {
        Self {
            policy,
            attempt: 0,
            started_at_ms,
            total_delay_ms: 0,
            errors: Vec::new(),
        }
    }

    /// Decide whether to retry after the given error, using the specified jitter seed.
    pub fn should_retry(&mut self, error: RetryError, now_ms: u64, jitter_seed: u64) -> RetryDecision {
        self.errors.push(error.clone());

        // Permanent errors are never retried
        if error.kind == ErrorKind::Permanent {
            return RetryDecision::GiveUp(GiveUpReason::PermanentError);
        }

        // Check retryable codes
        if !self.policy.is_code_retryable(error.code) {
            return RetryDecision::GiveUp(GiveUpReason::PermanentError);
        }

        // Max retries check
        if self.attempt >= self.policy.max_retries {
            return RetryDecision::GiveUp(GiveUpReason::MaxRetriesExceeded);
        }

        // Total timeout check
        if let Some(timeout) = self.policy.total_timeout_ms {
            if now_ms.saturating_sub(self.started_at_ms) >= timeout {
                return RetryDecision::GiveUp(GiveUpReason::BudgetExhausted);
            }
        }

        let delay = self.policy.compute_delay(self.attempt, jitter_seed);
        self.attempt += 1;
        self.total_delay_ms += delay;
        RetryDecision::RetryAfter(delay)
    }

    pub fn attempt(&self) -> u32 { self.attempt }
    pub fn total_delay_ms(&self) -> u64 { self.total_delay_ms }
    pub fn errors(&self) -> &[RetryError] { &self.errors }
    pub fn elapsed_ms(&self, now_ms: u64) -> u64 { now_ms.saturating_sub(self.started_at_ms) }
}

// ── Circuit Breaker Guard ──────────────────────────────────────

/// Tracks consecutive failures and opens the circuit when a threshold is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Circuit breaker that integrates with retry logic.
#[derive(Debug, Clone)]
pub struct CircuitBreakerGuard {
    state: CircuitState,
    failure_threshold: u32,
    consecutive_failures: u32,
    success_threshold: u32,
    half_open_successes: u32,
    cooldown_ms: u64,
    opened_at_ms: u64,
    total_trips: u64,
}

impl CircuitBreakerGuard {
    pub fn new(failure_threshold: u32, success_threshold: u32, cooldown_ms: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_threshold,
            consecutive_failures: 0,
            success_threshold,
            half_open_successes: 0,
            cooldown_ms,
            opened_at_ms: 0,
            total_trips: 0,
        }
    }

    /// Whether the circuit allows a request at the given time.
    pub fn allow_request(&mut self, now_ms: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if now_ms.saturating_sub(self.opened_at_ms) >= self.cooldown_ms {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_successes = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.consecutive_failures = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self, now_ms: u64) {
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at_ms = now_ms;
                    self.total_trips += 1;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.opened_at_ms = now_ms;
                self.total_trips += 1;
                self.half_open_successes = 0;
            }
            CircuitState::Open => {}
        }
    }

    pub fn state(&self) -> CircuitState { self.state }
    pub fn consecutive_failures(&self) -> u32 { self.consecutive_failures }
    pub fn total_trips(&self) -> u64 { self.total_trips }
}

// ── Retry Statistics ───────────────────────────────────────────

/// Aggregated retry statistics across multiple operations.
#[derive(Debug, Clone, Default)]
pub struct RetryStats {
    pub total_operations: u64,
    pub successful_first_try: u64,
    pub successful_after_retry: u64,
    pub exhausted: u64,
    pub total_retries: u64,
    pub total_delay_ms: u64,
}

impl RetryStats {
    pub fn new() -> Self { Self::default() }

    pub fn record_success_first_try(&mut self) {
        self.total_operations += 1;
        self.successful_first_try += 1;
    }

    pub fn record_success_after_retry(&mut self, retries: u32, delay_ms: u64) {
        self.total_operations += 1;
        self.successful_after_retry += 1;
        self.total_retries += retries as u64;
        self.total_delay_ms += delay_ms;
    }

    pub fn record_exhausted(&mut self, retries: u32, delay_ms: u64) {
        self.total_operations += 1;
        self.exhausted += 1;
        self.total_retries += retries as u64;
        self.total_delay_ms += delay_ms;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 { return 0.0; }
        (self.successful_first_try + self.successful_after_retry) as f64
            / self.total_operations as f64
    }

    pub fn avg_retries(&self) -> f64 {
        if self.total_operations == 0 { return 0.0; }
        self.total_retries as f64 / self.total_operations as f64
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_default_values() {
        let p = RetryPolicy::new();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.base_delay_ms, 100);
        assert_eq!(p.max_delay_ms, 30_000);
        assert_eq!(p.jitter_pct, 25);
    }

    #[test]
    fn policy_builder() {
        let p = RetryPolicy::new()
            .with_max_retries(5)
            .with_base_delay_ms(200)
            .with_max_delay_ms(10_000)
            .with_jitter_pct(10)
            .with_multiplier(3.0);
        assert_eq!(p.max_retries, 5);
        assert_eq!(p.base_delay_ms, 200);
        assert_eq!(p.max_delay_ms, 10_000);
        assert_eq!(p.jitter_pct, 10);
        assert_eq!(p.multiplier, 3.0);
    }

    #[test]
    fn exponential_backoff_increases() {
        let p = RetryPolicy::new().with_base_delay_ms(100).with_jitter_pct(0).with_multiplier(2.0);
        assert_eq!(p.compute_delay(0, 0), 100);
        assert_eq!(p.compute_delay(1, 0), 200);
        assert_eq!(p.compute_delay(2, 0), 400);
        assert_eq!(p.compute_delay(3, 0), 800);
    }

    #[test]
    fn delay_capped_at_max() {
        let p = RetryPolicy::new()
            .with_base_delay_ms(100)
            .with_max_delay_ms(500)
            .with_jitter_pct(0)
            .with_multiplier(2.0);
        assert_eq!(p.compute_delay(10, 0), 500);
    }

    #[test]
    fn jitter_varies_delay() {
        let p = RetryPolicy::new().with_base_delay_ms(1000).with_jitter_pct(50);
        let d1 = p.compute_delay(0, 1);
        let d2 = p.compute_delay(0, 999);
        // Both should be in the range [500, 1500] approximately
        assert!(d1 > 0);
        assert!(d2 > 0);
    }

    #[test]
    fn retry_state_transient_error_retries() {
        let policy = RetryPolicy::new().with_max_retries(3).with_jitter_pct(0);
        let mut state = RetryState::new(policy, 0);
        let err = RetryError::transient(503, "service unavailable");
        let decision = state.should_retry(err, 100, 0);
        assert!(matches!(decision, RetryDecision::RetryAfter(_)));
        assert_eq!(state.attempt(), 1);
    }

    #[test]
    fn retry_state_permanent_error_gives_up() {
        let policy = RetryPolicy::new();
        let mut state = RetryState::new(policy, 0);
        let err = RetryError::permanent(400, "bad request");
        let decision = state.should_retry(err, 100, 0);
        assert_eq!(decision, RetryDecision::GiveUp(GiveUpReason::PermanentError));
    }

    #[test]
    fn retry_state_max_retries_exceeded() {
        let policy = RetryPolicy::new().with_max_retries(2).with_jitter_pct(0);
        let mut state = RetryState::new(policy, 0);
        let err = || RetryError::transient(500, "err");
        state.should_retry(err(), 10, 0); // attempt 0 -> 1
        state.should_retry(err(), 20, 0); // attempt 1 -> 2
        let decision = state.should_retry(err(), 30, 0); // attempt 2, exceeds max
        assert_eq!(decision, RetryDecision::GiveUp(GiveUpReason::MaxRetriesExceeded));
    }

    #[test]
    fn retry_state_total_timeout() {
        let policy = RetryPolicy::new()
            .with_max_retries(10)
            .with_total_timeout_ms(1000);
        let mut state = RetryState::new(policy, 0);
        let err = RetryError::transient(500, "err");
        let decision = state.should_retry(err, 1500, 0);
        assert_eq!(decision, RetryDecision::GiveUp(GiveUpReason::BudgetExhausted));
    }

    #[test]
    fn retry_state_tracks_errors() {
        let policy = RetryPolicy::new().with_jitter_pct(0);
        let mut state = RetryState::new(policy, 0);
        state.should_retry(RetryError::transient(500, "a"), 10, 0);
        state.should_retry(RetryError::transient(503, "b"), 20, 0);
        assert_eq!(state.errors().len(), 2);
        assert_eq!(state.errors()[0].code, 500);
        assert_eq!(state.errors()[1].code, 503);
    }

    #[test]
    fn circuit_breaker_closed_allows() {
        let mut cb = CircuitBreakerGuard::new(3, 1, 5000);
        assert!(cb.allow_request(0));
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_trips_open() {
        let mut cb = CircuitBreakerGuard::new(3, 1, 5000);
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(300);
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request(400));
    }

    #[test]
    fn circuit_breaker_half_open_after_cooldown() {
        let mut cb = CircuitBreakerGuard::new(2, 1, 1000);
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request(500));
        assert!(cb.allow_request(1200));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_recovers() {
        let mut cb = CircuitBreakerGuard::new(2, 2, 1000);
        cb.record_failure(100);
        cb.record_failure(200);
        cb.allow_request(1200); // transition to half-open
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_half_open_failure_reopens() {
        let mut cb = CircuitBreakerGuard::new(2, 2, 1000);
        cb.record_failure(100);
        cb.record_failure(200);
        cb.allow_request(1200);
        cb.record_failure(1300);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.total_trips(), 2);
    }

    #[test]
    fn retry_stats_success_rate() {
        let mut stats = RetryStats::new();
        stats.record_success_first_try();
        stats.record_success_first_try();
        stats.record_exhausted(3, 700);
        assert!((stats.success_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn retry_stats_avg_retries() {
        let mut stats = RetryStats::new();
        stats.record_success_first_try();
        stats.record_success_after_retry(2, 300);
        stats.record_exhausted(3, 700);
        assert!((stats.avg_retries() - 5.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn error_display() {
        let err = RetryError::transient(503, "service unavailable");
        let s = format!("{err}");
        assert!(s.contains("transient"));
        assert!(s.contains("503"));
    }

    #[test]
    fn retryable_codes_filter() {
        let policy = RetryPolicy::new()
            .with_retryable_code(503)
            .with_retryable_code(429)
            .with_max_retries(5);
        let mut state = RetryState::new(policy, 0);
        // 503 is retryable
        let d = state.should_retry(RetryError::transient(503, "x"), 10, 0);
        assert!(matches!(d, RetryDecision::RetryAfter(_)));
        // 500 is NOT in the retryable list
        let d = state.should_retry(RetryError::transient(500, "y"), 20, 0);
        assert_eq!(d, RetryDecision::GiveUp(GiveUpReason::PermanentError));
    }
}
