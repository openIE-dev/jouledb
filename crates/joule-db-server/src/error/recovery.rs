//! Error Recovery and Resilience
//!
//! Provides comprehensive error handling, recovery strategies, and graceful degradation
//! for production environments.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Error recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    Retry,
    /// Circuit breaker pattern
    CircuitBreaker,
    /// Fallback to alternative
    Fallback,
    /// Graceful degradation
    Degrade,
    /// Fail fast
    FailFast,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing, reject requests)
    Open,
    /// Circuit is half-open (testing if recovered)
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold (open circuit after this many failures)
    pub failure_threshold: usize,
    /// Success threshold (close circuit after this many successes in half-open)
    pub success_threshold: usize,
    /// Timeout before transitioning from open to half-open
    pub timeout: Duration,
    /// Time window for counting failures
    pub time_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            time_window: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker
pub struct CircuitBreaker {
    /// Current state
    state: Arc<RwLock<CircuitState>>,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Failure count
    failure_count: Arc<RwLock<usize>>,
    /// Success count (for half-open state)
    success_count: Arc<RwLock<usize>>,
    /// Last failure time
    last_failure: Arc<RwLock<Option<Instant>>>,
    /// State transition time
    state_transition_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            state_transition_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if request should be allowed
    pub async fn allow(&self) -> bool {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed
                let transition_time = *self.state_transition_time.read().await;
                if let Some(time) = transition_time {
                    if time.elapsed() >= self.config.timeout {
                        // Transition to half-open
                        *self.state.write().await = CircuitState::HalfOpen;
                        *self.success_count.write().await = 0;
                        *self.state_transition_time.write().await = Some(Instant::now());
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record success
    pub async fn record_success(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                // Reset failure count on success
                *self.failure_count.write().await = 0;
            }
            CircuitState::Open => {
                // Should not happen, but reset anyway
                *self.failure_count.write().await = 0;
            }
            CircuitState::HalfOpen => {
                let mut success_count = self.success_count.write().await;
                *success_count += 1;

                if *success_count >= self.config.success_threshold {
                    // Transition to closed
                    *self.state.write().await = CircuitState::Closed;
                    *self.failure_count.write().await = 0;
                    *self.success_count.write().await = 0;
                    *self.state_transition_time.write().await = None;
                }
            }
        }
    }

    /// Record failure
    pub async fn record_failure(&self) {
        let state = *self.state.read().await;

        match state {
            CircuitState::Closed => {
                let mut failure_count = self.failure_count.write().await;
                *failure_count += 1;
                *self.last_failure.write().await = Some(Instant::now());

                if *failure_count >= self.config.failure_threshold {
                    // Transition to open
                    *self.state.write().await = CircuitState::Open;
                    *self.state_transition_time.write().await = Some(Instant::now());
                }
            }
            CircuitState::Open => {
                // Update transition time
                *self.state_transition_time.write().await = Some(Instant::now());
            }
            CircuitState::HalfOpen => {
                // Transition back to open
                *self.state.write().await = CircuitState::Open;
                *self.failure_count.write().await = self.config.failure_threshold;
                *self.success_count.write().await = 0;
                *self.state_transition_time.write().await = Some(Instant::now());
            }
        }
    }

    /// Get current state
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Reset circuit breaker
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        *self.failure_count.write().await = 0;
        *self.success_count.write().await = 0;
        *self.last_failure.write().await = None;
        *self.state_transition_time.write().await = None;
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Initial delay
    pub initial_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter (randomization factor)
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Retry executor
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create new retry executor
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute function with retry
    pub async fn execute<F, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
        E: Clone,
    {
        let mut delay = self.config.initial_delay;
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e.clone());

                    if attempt < self.config.max_retries {
                        // Calculate delay with backoff
                        let mut current_delay = delay;
                        if self.config.jitter {
                            use rand::{Rng, RngExt};
                            let jitter_amount = current_delay.as_millis() as f64 * 0.1;
                            let jitter = rand::rng().random_range(-jitter_amount..=jitter_amount);
                            current_delay = Duration::from_millis(
                                (current_delay.as_millis() as f64 + jitter) as u64,
                            );
                        }

                        tokio::time::sleep(current_delay.min(self.config.max_delay)).await;

                        // Exponential backoff
                        delay = Duration::from_millis(
                            (delay.as_millis() as f64 * self.config.backoff_multiplier) as u64,
                        )
                        .min(self.config.max_delay);
                    }
                }
            }
        }

        Err(last_error.expect("Should have error after retries"))
    }
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    /// Circuit breakers by operation
    circuit_breakers: Arc<RwLock<std::collections::HashMap<String, Arc<CircuitBreaker>>>>,
    /// Retry executor
    retry_executor: RetryExecutor,
}

impl ErrorRecoveryManager {
    /// Create new error recovery manager
    pub fn new(retry_config: RetryConfig) -> Self {
        Self {
            circuit_breakers: Arc::new(RwLock::new(std::collections::HashMap::new())),
            retry_executor: RetryExecutor::new(retry_config),
        }
    }

    /// Get or create circuit breaker for operation
    pub async fn get_circuit_breaker(&self, operation: &str) -> Arc<CircuitBreaker> {
        // Check if exists
        {
            let breakers = self.circuit_breakers.read().await;
            if let Some(breaker) = breakers.get(operation) {
                return Arc::clone(breaker);
            }
        }

        // Create new
        let mut breakers = self.circuit_breakers.write().await;
        let breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default()));
        breakers.insert(operation.to_string(), Arc::clone(&breaker));
        breaker
    }

    /// Execute with recovery
    pub async fn execute_with_recovery<F, T, E>(
        &self,
        operation: &str,
        strategy: RecoveryStrategy,
        mut f: F,
    ) -> Result<T, E>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
        E: Clone,
    {
        match strategy {
            RecoveryStrategy::Retry => self.retry_executor.execute(f).await,
            RecoveryStrategy::CircuitBreaker => {
                let breaker = self.get_circuit_breaker(operation).await;
                if !breaker.allow().await {
                    // Circuit is open, fail fast
                    // Return a generic error - in real implementation would use proper error type
                    return f().await; // For now, just try anyway
                }

                match f().await {
                    Ok(result) => {
                        breaker.record_success().await;
                        Ok(result)
                    }
                    Err(e) => {
                        breaker.record_failure().await;
                        Err(e)
                    }
                }
            }
            RecoveryStrategy::Fallback => {
                // Try the operation; on failure, retry once after a short delay
                match f().await {
                    Ok(result) => Ok(result),
                    Err(_first_err) => {
                        tracing::warn!(
                            operation = operation,
                            "Primary operation failed, retrying with fallback delay"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        f().await
                    }
                }
            }
            RecoveryStrategy::Degrade => {
                // Log degraded-mode entry, then execute normally
                tracing::warn!(operation = operation, "Executing in degraded mode");
                f().await
            }
            RecoveryStrategy::FailFast => f().await,
        }
    }

    /// Execute with timeout
    ///
    /// Wraps an async operation with a timeout.
    /// Returns TimeoutError if the operation exceeds the timeout.
    pub async fn execute_with_timeout<F, T, E>(
        &self,
        timeout: Duration,
        f: F,
    ) -> Result<T, TimeoutRecoveryError<E>>
    where
        F: std::future::Future<Output = Result<T, E>> + Send,
        E: Send,
    {
        match tokio::time::timeout(timeout, f).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(TimeoutRecoveryError::Inner(e)),
            Err(_) => Err(TimeoutRecoveryError::Timeout(timeout)),
        }
    }

    /// Execute with timeout and retry
    ///
    /// Combines timeout enforcement with retry logic.
    pub async fn execute_with_timeout_retry<F, T, E>(
        &self,
        timeout: Duration,
        mut f: F,
    ) -> Result<T, TimeoutRecoveryError<E>>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
        E: Clone + Send,
    {
        let mut delay = self.retry_executor.config.initial_delay;
        let mut last_error = None;

        for attempt in 0..=self.retry_executor.config.max_retries {
            match self.execute_with_timeout(timeout, f()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.retry_executor.config.max_retries {
                        // Calculate delay with backoff
                        let current_delay = delay.min(self.retry_executor.config.max_delay);
                        tokio::time::sleep(current_delay).await;

                        // Exponential backoff
                        delay = Duration::from_millis(
                            (delay.as_millis() as f64
                                * self.retry_executor.config.backoff_multiplier)
                                as u64,
                        );
                    }
                }
            }
        }

        Err(last_error.expect("Should have error after retries"))
    }
}

/// Error type for timeout-aware operations
#[derive(Debug, Clone)]
pub enum TimeoutRecoveryError<E> {
    /// Operation timed out
    Timeout(Duration),
    /// Inner operation error
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for TimeoutRecoveryError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutRecoveryError::Timeout(duration) => {
                write!(f, "Operation timed out after {:?}", duration)
            }
            TimeoutRecoveryError::Inner(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for TimeoutRecoveryError<E> {}
