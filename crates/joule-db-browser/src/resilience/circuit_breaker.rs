//! Circuit breaker pattern for fault tolerance

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
    failure_threshold: u32,
    timeout: Duration,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            failure_threshold,
            timeout,
        }
    }

    /// Check if request should be allowed
    pub fn allow_request(&self) -> bool {
        let state = *self.state.read().unwrap();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_fail) = *self.last_failure.read().unwrap() {
                    if last_fail.elapsed() >= self.timeout {
                        // Transition to half-open
                        *self.state.write().unwrap() = CircuitState::HalfOpen;
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record success
    pub fn record_success(&self) {
        let mut state = self.state.write().unwrap();
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
            *self.failure_count.write().unwrap() = 0;
        }
    }

    /// Record failure
    pub fn record_failure(&self) {
        let mut failure_count = self.failure_count.write().unwrap();
        *failure_count += 1;
        *self.last_failure.write().unwrap() = Some(Instant::now());

        if *failure_count >= self.failure_threshold {
            *self.state.write().unwrap() = CircuitState::Open;
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.read().unwrap()
    }
}
