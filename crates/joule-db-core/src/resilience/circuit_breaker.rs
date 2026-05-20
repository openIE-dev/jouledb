//! Circuit Breaker Pattern Implementation
//!
//! Prevents cascading failures by temporarily disabling failing operations.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitBreakerState {
    /// Normal operation - requests are allowed
    Closed = 0,
    /// Circuit is open - requests are rejected
    Open = 1,
    /// Testing if the service has recovered
    HalfOpen = 2,
}

impl From<u8> for CircuitBreakerState {
    fn from(value: u8) -> Self {
        match value {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trip the circuit
    pub failure_threshold: u32,
    /// Time to wait before testing if service recovered
    pub recovery_timeout: Duration,
    /// Number of successful requests needed to close circuit from half-open
    pub success_threshold: u32,
    /// Optional: failure rate threshold (0.0 to 1.0)
    pub failure_rate_threshold: Option<f32>,
    /// Window size for failure rate calculation
    pub failure_rate_window: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 3,
            failure_rate_threshold: None,
            failure_rate_window: 100,
        }
    }
}

/// Circuit breaker for protecting against cascading failures
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: AtomicU8,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: RwLock<Option<Instant>>,
    // For failure rate tracking
    recent_results: RwLock<Vec<bool>>, // true = success, false = failure
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: AtomicU8::new(CircuitBreakerState::Closed as u8),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure_time: RwLock::new(None),
            recent_results: RwLock::new(Vec::new()),
        }
    }

    /// Builder: set failure threshold
    pub fn failure_threshold(mut self, threshold: u32) -> Self {
        self.config.failure_threshold = threshold;
        self
    }

    /// Builder: set recovery timeout
    pub fn recovery_timeout(mut self, timeout: Duration) -> Self {
        self.config.recovery_timeout = timeout;
        self
    }

    /// Builder: set success threshold
    pub fn success_threshold(mut self, threshold: u32) -> Self {
        self.config.success_threshold = threshold;
        self
    }

    /// Get current state
    pub fn state(&self) -> CircuitBreakerState {
        CircuitBreakerState::from(self.state.load(Ordering::SeqCst))
    }

    /// Check if a request should be allowed
    pub fn allow_request(&self) -> bool {
        match self.state() {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if recovery timeout has passed
                let last_failure = self
                    .last_failure_time
                    .read()
                    .expect("lock poisoned: last_failure_time read");
                if let Some(time) = *last_failure {
                    if time.elapsed() >= self.config.recovery_timeout {
                        // Transition to half-open
                        self.state
                            .store(CircuitBreakerState::HalfOpen as u8, Ordering::SeqCst);
                        self.success_count.store(0, Ordering::SeqCst);
                        return true;
                    }
                }
                false
            }
            CircuitBreakerState::HalfOpen => {
                // Allow limited requests in half-open state
                true
            }
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        // Track for failure rate
        self.track_result(true);

        match self.state() {
            CircuitBreakerState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitBreakerState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.config.success_threshold {
                    // Close the circuit
                    self.state
                        .store(CircuitBreakerState::Closed as u8, Ordering::SeqCst);
                    self.failure_count.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                }
            }
            CircuitBreakerState::Open => {
                // Shouldn't happen, but handle gracefully
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        // Track for failure rate
        self.track_result(false);

        // Update last failure time
        {
            let mut last_failure = self
                .last_failure_time
                .write()
                .expect("lock poisoned: last_failure_time write");
            *last_failure = Some(Instant::now());
        }

        match self.state() {
            CircuitBreakerState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

                // Check failure threshold
                if count >= self.config.failure_threshold {
                    self.trip();
                }

                // Check failure rate threshold
                if let Some(rate_threshold) = self.config.failure_rate_threshold {
                    if self.failure_rate() >= rate_threshold {
                        self.trip();
                    }
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Immediately trip back to open on failure
                self.trip();
            }
            CircuitBreakerState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Trip the circuit to open state
    fn trip(&self) {
        self.state
            .store(CircuitBreakerState::Open as u8, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
    }

    /// Track result for failure rate calculation
    fn track_result(&self, success: bool) {
        let mut results = self
            .recent_results
            .write()
            .expect("lock poisoned: recent_results write");
        results.push(success);

        // Keep only the window size
        while results.len() > self.config.failure_rate_window as usize {
            results.remove(0);
        }
    }

    /// Calculate current failure rate
    pub fn failure_rate(&self) -> f32 {
        let results = self
            .recent_results
            .read()
            .expect("lock poisoned: recent_results read");
        if results.is_empty() {
            return 0.0;
        }

        let failures = results.iter().filter(|&&success| !success).count();
        failures as f32 / results.len() as f32
    }

    /// Get failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::SeqCst)
    }

    /// Manually reset the circuit breaker
    pub fn reset(&self) {
        self.state
            .store(CircuitBreakerState::Closed as u8, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        *self
            .last_failure_time
            .write()
            .expect("lock poisoned: last_failure_time write") = None;
        self.recent_results
            .write()
            .expect("lock poisoned: recent_results write")
            .clear();
    }

    /// Execute an operation through the circuit breaker
    pub fn execute<T, E, F>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if !self.allow_request() {
            return Err(CircuitBreakerError::Open);
        }

        match f() {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(err) => {
                self.record_failure();
                Err(CircuitBreakerError::Inner(err))
            }
        }
    }

    /// Execute an async operation through the circuit breaker
    pub async fn execute_async<T, E, F, Fut>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        if !self.allow_request() {
            return Err(CircuitBreakerError::Open);
        }

        match f().await {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(err) => {
                self.record_failure();
                Err(CircuitBreakerError::Inner(err))
            }
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Error from circuit breaker operations
#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    /// Circuit is open, request rejected
    Open,
    /// Inner operation error
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::Open => write!(f, "Circuit breaker is open"),
            CircuitBreakerError::Inner(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for CircuitBreakerError<E> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn test_trip_on_failures() {
        let cb = CircuitBreaker::new().failure_threshold(3);

        // First two failures don't trip
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        // Third failure trips the circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
    }

    #[test]
    fn test_success_resets_failures() {
        let cb = CircuitBreaker::new().failure_threshold(3);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);

        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_open_rejects_requests() {
        let cb = CircuitBreaker::new().failure_threshold(1);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_execute_success() {
        let cb = CircuitBreaker::new();

        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Ok(42));
        assert!(matches!(result, Ok(42)));
    }

    #[test]
    fn test_execute_failure() {
        let cb = CircuitBreaker::new();

        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Err("error"));
        assert!(matches!(result, Err(CircuitBreakerError::Inner("error"))));
    }

    #[test]
    fn test_execute_rejected() {
        let cb = CircuitBreaker::new().failure_threshold(1);
        cb.record_failure();

        let result: Result<i32, CircuitBreakerError<&str>> = cb.execute(|| Ok(42));
        assert!(matches!(result, Err(CircuitBreakerError::Open)));
    }

    #[test]
    fn test_manual_reset() {
        let cb = CircuitBreaker::new().failure_threshold(1);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn test_failure_rate() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_rate_window: 10,
            ..Default::default()
        });

        // 3 failures, 7 successes = 30% failure rate
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        for _ in 0..7 {
            cb.record_success();
        }

        assert!((cb.failure_rate() - 0.3).abs() < 0.01);
    }
}
