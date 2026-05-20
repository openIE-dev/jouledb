//! Retry Policy Implementation
//!
//! Provides configurable retry logic with exponential backoff and jitter.

use std::future::Future;
use std::time::Duration;

/// Result of a retry operation
pub enum RetryResult<T, E> {
    /// Operation succeeded
    Success(T),
    /// All retries exhausted
    Exhausted(E),
    /// Error is not retryable
    NonRetryable(E),
}

/// Trait for errors that can be classified as retryable
pub trait RetryableError {
    /// Returns true if this error should trigger a retry
    fn is_retryable(&self) -> bool;
}

// Note: We can't use specialization in stable Rust
// Users should implement RetryableError for their error types
// Default: assume all errors are retryable

/// Retry policy with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Enable jitter to prevent thundering herd
    pub jitter: bool,
    /// Backoff multiplier (default 2.0)
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: true,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a policy that never retries
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }

    /// Create a policy with custom max attempts
    pub fn with_max_attempts(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }

    /// Builder: set max attempts
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    /// Builder: set base delay
    pub fn base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Builder: set max delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Builder: enable/disable jitter
    pub fn jitter(mut self, enable: bool) -> Self {
        self.jitter = enable;
        self
    }

    /// Builder: set backoff multiplier
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Calculate delay for a given attempt (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let delay_ms = base_ms * self.backoff_multiplier.powi(attempt as i32);
        let capped_ms = delay_ms.min(self.max_delay.as_millis() as f64);

        let final_ms = if self.jitter {
            // Add up to 25% jitter
            let jitter_range = capped_ms * 0.25;
            let jitter = (rand_u64() % (jitter_range as u64 + 1)) as f64;
            capped_ms + jitter - jitter_range / 2.0
        } else {
            capped_ms
        };

        Duration::from_millis(final_ms.max(0.0) as u64)
    }

    /// Execute an operation with retry (sync version)
    pub fn execute<T, E, F>(&self, mut f: F) -> RetryResult<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut last_error = None;

        for attempt in 0..self.max_attempts {
            match f() {
                Ok(value) => return RetryResult::Success(value),
                Err(err) => {
                    last_error = Some(err);

                    // Don't sleep after the last attempt
                    if attempt + 1 < self.max_attempts {
                        let delay = self.delay_for_attempt(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        RetryResult::Exhausted(last_error.expect("max_attempts > 0 guarantees at least one error"))
    }

    /// Execute an async operation with retry (requires `async-tokio` feature)
    #[cfg(feature = "async-tokio")]
    pub async fn execute_async<T, E, F, Fut>(&self, f: F) -> RetryResult<T, E>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let mut last_error = None;

        for attempt in 0..self.max_attempts {
            match f().await {
                Ok(value) => return RetryResult::Success(value),
                Err(err) => {
                    last_error = Some(err);

                    // Don't sleep after the last attempt
                    if attempt + 1 < self.max_attempts {
                        let delay = self.delay_for_attempt(attempt);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        RetryResult::Exhausted(last_error.expect("max_attempts > 0 guarantees at least one error"))
    }

    /// Execute an async operation with retry using a custom sleep function.
    /// This allows using any async runtime.
    pub async fn execute_async_with_sleep<T, E, F, Fut, S, SFut>(
        &self,
        f: F,
        sleep: S,
    ) -> RetryResult<T, E>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        S: Fn(Duration) -> SFut,
        SFut: Future<Output = ()>,
    {
        let mut last_error = None;

        for attempt in 0..self.max_attempts {
            match f().await {
                Ok(value) => return RetryResult::Success(value),
                Err(err) => {
                    last_error = Some(err);

                    // Don't sleep after the last attempt
                    if attempt + 1 < self.max_attempts {
                        let delay = self.delay_for_attempt(attempt);
                        sleep(delay).await;
                    }
                }
            }
        }

        RetryResult::Exhausted(last_error.expect("max_attempts > 0 guarantees at least one error"))
    }
}

/// Simple random number generator (not cryptographically secure)
fn rand_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    // Simple xorshift
    let mut x = seed.wrapping_add(counter);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_calculation() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_millis(100))
            .backoff_multiplier(2.0)
            .jitter(false);

        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn test_max_delay_cap() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_secs(1))
            .max_delay(Duration::from_secs(5))
            .backoff_multiplier(10.0)
            .jitter(false);

        // 1 * 10^3 = 1000 seconds, should be capped to 5
        assert_eq!(policy.delay_for_attempt(3), Duration::from_secs(5));
    }

    #[test]
    fn test_retry_success_first_attempt() {
        let policy = RetryPolicy::with_max_attempts(3);
        let mut attempts = 0;

        let result: RetryResult<i32, &str> = policy.execute(|| {
            attempts += 1;
            Ok(42)
        });

        assert!(matches!(result, RetryResult::Success(42)));
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_retry_success_after_failures() {
        let policy = RetryPolicy::with_max_attempts(5)
            .base_delay(Duration::from_millis(1))
            .jitter(false);
        let mut attempts = 0;

        let result: RetryResult<i32, &str> = policy.execute(|| {
            attempts += 1;
            if attempts < 3 {
                Err("temporary failure")
            } else {
                Ok(42)
            }
        });

        assert!(matches!(result, RetryResult::Success(42)));
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_retry_exhausted() {
        let policy = RetryPolicy::with_max_attempts(3)
            .base_delay(Duration::from_millis(1))
            .jitter(false);
        let mut attempts = 0;

        let result: RetryResult<i32, &str> = policy.execute(|| {
            attempts += 1;
            Err("persistent failure")
        });

        assert!(matches!(
            result,
            RetryResult::Exhausted("persistent failure")
        ));
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_no_retry() {
        let policy = RetryPolicy::no_retry();
        let mut attempts = 0;

        let result: RetryResult<i32, &str> = policy.execute(|| {
            attempts += 1;
            Err("failure")
        });

        assert!(matches!(result, RetryResult::Exhausted(_)));
        assert_eq!(attempts, 1);
    }
}
