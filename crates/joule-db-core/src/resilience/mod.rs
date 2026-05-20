//! Resilience Patterns
//!
//! Production-grade error handling, retry logic, and circuit breakers.
//!
//! # Features
//!
//! - `async-tokio`: Enables tokio-based async utilities (retry with sleep, timeouts)
//!
//! # Example
//!
//! ```
//! use joule_db_core::resilience::{RetryPolicy, RetryResult};
//! use std::time::Duration;
//!
//! let policy = RetryPolicy::new()
//!     .max_attempts(3)
//!     .base_delay(Duration::from_millis(100));
//!
//! let result: RetryResult<i32, &str> = policy.execute(|| {
//!     Ok(42)
//! });
//!
//! assert!(matches!(result, RetryResult::Success(42)));
//! ```

mod circuit_breaker;
mod retry;
mod timeout;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState};
pub use retry::{RetryPolicy, RetryResult, RetryableError};
pub use timeout::{TimeoutConfig, TimeoutError, with_timeout};

use std::future::Future;
use std::time::Duration;

/// Execute an async operation with all resilience patterns applied.
///
/// Requires the `async-tokio` feature for automatic sleep between retries.
#[cfg(feature = "async-tokio")]
pub async fn execute_with_resilience<T, E, F, Fut>(
    _operation_name: &str,
    retry_policy: &RetryPolicy,
    circuit_breaker: Option<&CircuitBreaker>,
    timeout: Duration,
    f: F,
) -> Result<T, ResilienceError<E>>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Debug + Clone,
{
    // Check circuit breaker
    if let Some(cb) = circuit_breaker {
        if !cb.allow_request() {
            return Err(ResilienceError::CircuitOpen);
        }
    }

    // Execute with retry and timeout
    let result = retry_policy
        .execute_async(|| async {
            with_timeout(timeout, f()).await.map_err(|e| match e {
                TimeoutError::Timeout => ResilienceError::Timeout,
                TimeoutError::Inner(inner) => ResilienceError::Operation(inner),
            })
        })
        .await;

    // Update circuit breaker
    if let Some(cb) = circuit_breaker {
        match &result {
            RetryResult::Success(_) => cb.record_success(),
            RetryResult::Exhausted(_) | RetryResult::NonRetryable(_) => cb.record_failure(),
        }
    }

    match result {
        RetryResult::Success(value) => Ok(value),
        RetryResult::Exhausted(err) => Err(err),
        RetryResult::NonRetryable(err) => Err(err),
    }
}

/// Execute an async operation with all resilience patterns applied.
///
/// This version uses a custom sleep function, allowing any async runtime.
pub async fn execute_with_resilience_custom<T, E, F, Fut, S, SFut>(
    _operation_name: &str,
    retry_policy: &RetryPolicy,
    circuit_breaker: Option<&CircuitBreaker>,
    timeout: Duration,
    f: F,
    sleep: S,
) -> Result<T, ResilienceError<E>>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    S: Fn(Duration) -> SFut,
    SFut: Future<Output = ()>,
    E: std::fmt::Debug + Clone,
{
    // Check circuit breaker
    if let Some(cb) = circuit_breaker {
        if !cb.allow_request() {
            return Err(ResilienceError::CircuitOpen);
        }
    }

    // Execute with retry and timeout (custom sleep)
    let result = retry_policy
        .execute_async_with_sleep(
            || async {
                with_timeout(timeout, f()).await.map_err(|e| match e {
                    TimeoutError::Timeout => ResilienceError::Timeout,
                    TimeoutError::Inner(inner) => ResilienceError::Operation(inner),
                })
            },
            &sleep,
        )
        .await;

    // Update circuit breaker
    if let Some(cb) = circuit_breaker {
        match &result {
            RetryResult::Success(_) => cb.record_success(),
            RetryResult::Exhausted(_) | RetryResult::NonRetryable(_) => cb.record_failure(),
        }
    }

    match result {
        RetryResult::Success(value) => Ok(value),
        RetryResult::Exhausted(err) => Err(err),
        RetryResult::NonRetryable(err) => Err(err),
    }
}

/// Errors that can occur in resilience-wrapped operations
#[derive(Debug, Clone)]
pub enum ResilienceError<E> {
    /// The circuit breaker is open
    CircuitOpen,
    /// Operation timed out
    Timeout,
    /// Underlying operation error
    Operation(E),
    /// Retry exhausted after all attempts failed
    RetryExhausted {
        /// Number of attempts made
        attempts: u32,
        /// The last error encountered
        last_error: E,
    },
}

impl<E: std::fmt::Display> std::fmt::Display for ResilienceError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResilienceError::CircuitOpen => write!(f, "Circuit breaker is open"),
            ResilienceError::Timeout => write!(f, "Operation timed out"),
            ResilienceError::Operation(e) => write!(f, "Operation failed: {}", e),
            ResilienceError::RetryExhausted {
                attempts,
                last_error,
            } => {
                write!(
                    f,
                    "Retry exhausted after {} attempts: {}",
                    attempts, last_error
                )
            }
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for ResilienceError<E> {}
