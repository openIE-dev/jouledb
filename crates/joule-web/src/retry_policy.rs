//! Retry strategies — configurable retry policies with backoff algorithms.
//!
//! Replaces retry / async-retry / p-retry with a pure-Rust retry framework.
//! Supports constant, linear, exponential, and Fibonacci backoff with jitter,
//! custom predicates, max attempts, and max total duration tracking.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ── Retry Error ───────────────────────────────────────────────

/// Errors from the retry system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryError {
    /// All retry attempts exhausted.
    Exhausted {
        attempts: u32,
        last_error: String,
    },
    /// Total duration budget exceeded.
    DurationExceeded {
        elapsed_ms: u64,
        budget_ms: u64,
        last_error: String,
    },
    /// Retry predicate rejected the error (non-retryable).
    NonRetryable {
        error: String,
    },
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted { attempts, last_error } => {
                write!(f, "exhausted after {} attempts: {}", attempts, last_error)
            }
            Self::DurationExceeded { elapsed_ms, budget_ms, last_error } => {
                write!(f, "duration exceeded ({}/{}ms): {}", elapsed_ms, budget_ms, last_error)
            }
            Self::NonRetryable { error } => {
                write!(f, "non-retryable: {}", error)
            }
        }
    }
}

// ── Backoff Strategy ──────────────────────────────────────────

/// The backoff algorithm to use between retries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Constant { delay_ms: u64 },
    /// Linear increase: delay_ms * attempt.
    Linear { base_delay_ms: u64 },
    /// Exponential: base_delay_ms * 2^attempt, capped at max_delay_ms.
    Exponential {
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Exponential with jitter: adds random component up to jitter_ms.
    ExponentialWithJitter {
        base_delay_ms: u64,
        max_delay_ms: u64,
        jitter_ms: u64,
    },
    /// Fibonacci sequence of delays.
    Fibonacci { base_delay_ms: u64 },
}

impl BackoffStrategy {
    /// Calculate delay for a given attempt number (0-based).
    pub fn delay(&self, attempt: u32) -> Duration {
        let ms = self.delay_ms(attempt);
        Duration::from_millis(ms)
    }

    /// Calculate delay in milliseconds for a given attempt number (0-based).
    pub fn delay_ms(&self, attempt: u32) -> u64 {
        match self {
            Self::Constant { delay_ms } => *delay_ms,
            Self::Linear { base_delay_ms } => {
                base_delay_ms.saturating_mul((attempt + 1) as u64)
            }
            Self::Exponential { base_delay_ms, max_delay_ms } => {
                let exp = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
                let delay = base_delay_ms.saturating_mul(exp);
                delay.min(*max_delay_ms)
            }
            Self::ExponentialWithJitter {
                base_delay_ms,
                max_delay_ms,
                jitter_ms,
            } => {
                let exp = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
                let base = base_delay_ms.saturating_mul(exp).min(*max_delay_ms);
                // Deterministic jitter based on attempt number for testability.
                let jitter = if *jitter_ms > 0 {
                    (attempt as u64 * 7 + 13) % jitter_ms
                } else {
                    0
                };
                base.saturating_add(jitter)
            }
            Self::Fibonacci { base_delay_ms } => {
                let fib = fibonacci(attempt);
                base_delay_ms.saturating_mul(fib)
            }
        }
    }
}

/// Compute the nth Fibonacci number (0-indexed: fib(0)=1, fib(1)=1, fib(2)=2...).
fn fibonacci(n: u32) -> u64 {
    if n == 0 {
        return 1;
    }
    let mut a: u64 = 1;
    let mut b: u64 = 1;
    for _ in 1..n {
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    b
}

// ── Retry Context ─────────────────────────────────────────────

/// Context passed through each retry attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryContext {
    /// Current attempt number (0-based).
    pub attempt: u32,
    /// Last error message (empty on first attempt).
    pub last_error: String,
    /// Total elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Delays applied so far in milliseconds.
    pub delays_applied: Vec<u64>,
}

impl RetryContext {
    /// Create a new context for the first attempt.
    pub fn new() -> Self {
        Self {
            attempt: 0,
            last_error: String::new(),
            elapsed_ms: 0,
            delays_applied: Vec::new(),
        }
    }

    /// Total delay applied so far.
    pub fn total_delay_ms(&self) -> u64 {
        self.delays_applied.iter().sum()
    }

    /// Is this the first attempt?
    pub fn is_first_attempt(&self) -> bool {
        self.attempt == 0
    }
}

impl Default for RetryContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Retry Outcome ─────────────────────────────────────────────

/// Outcome of a single attempt within a retry loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The operation succeeded.
    Success,
    /// The operation failed with a retryable error.
    RetryableFailure(String),
    /// The operation failed with a non-retryable error.
    NonRetryableFailure(String),
}

// ── Retry Decision ────────────────────────────────────────────

/// Decision from the retry policy on what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryDecision {
    /// Retry after waiting the specified delay.
    RetryAfter(Duration),
    /// Stop retrying — policy limits reached.
    Stop(RetryError),
}

// ── Retry Policy ──────────────────────────────────────────────

/// A configurable retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Backoff strategy.
    pub strategy: BackoffStrategy,
    /// Maximum number of retry attempts (not including the initial call).
    pub max_attempts: u32,
    /// Maximum total duration budget in milliseconds (0 = unlimited).
    pub max_total_duration_ms: u64,
    /// Error classifications that are retryable. Empty = all are retryable.
    pub retryable_errors: Vec<String>,
}

impl RetryPolicy {
    /// Create a policy with constant delay.
    pub fn constant(delay_ms: u64, max_attempts: u32) -> Self {
        Self {
            strategy: BackoffStrategy::Constant { delay_ms },
            max_attempts,
            max_total_duration_ms: 0,
            retryable_errors: Vec::new(),
        }
    }

    /// Create a policy with linear backoff.
    pub fn linear(base_delay_ms: u64, max_attempts: u32) -> Self {
        Self {
            strategy: BackoffStrategy::Linear { base_delay_ms },
            max_attempts,
            max_total_duration_ms: 0,
            retryable_errors: Vec::new(),
        }
    }

    /// Create a policy with exponential backoff.
    pub fn exponential(base_delay_ms: u64, max_delay_ms: u64, max_attempts: u32) -> Self {
        Self {
            strategy: BackoffStrategy::Exponential {
                base_delay_ms,
                max_delay_ms,
            },
            max_attempts,
            max_total_duration_ms: 0,
            retryable_errors: Vec::new(),
        }
    }

    /// Create a policy with exponential backoff plus jitter.
    pub fn exponential_jitter(
        base_delay_ms: u64,
        max_delay_ms: u64,
        jitter_ms: u64,
        max_attempts: u32,
    ) -> Self {
        Self {
            strategy: BackoffStrategy::ExponentialWithJitter {
                base_delay_ms,
                max_delay_ms,
                jitter_ms,
            },
            max_attempts,
            max_total_duration_ms: 0,
            retryable_errors: Vec::new(),
        }
    }

    /// Create a policy with Fibonacci backoff.
    pub fn fibonacci(base_delay_ms: u64, max_attempts: u32) -> Self {
        Self {
            strategy: BackoffStrategy::Fibonacci { base_delay_ms },
            max_attempts,
            max_total_duration_ms: 0,
            retryable_errors: Vec::new(),
        }
    }

    /// Set maximum total duration budget.
    pub fn with_max_duration(mut self, max_ms: u64) -> Self {
        self.max_total_duration_ms = max_ms;
        self
    }

    /// Add retryable error patterns. If non-empty, only matching errors are retried.
    pub fn with_retryable_errors(mut self, errors: Vec<String>) -> Self {
        self.retryable_errors = errors;
        self
    }

    /// Check if an error message is retryable under this policy.
    pub fn is_retryable(&self, error: &str) -> bool {
        if self.retryable_errors.is_empty() {
            return true; // All errors are retryable if no filter set.
        }
        self.retryable_errors.iter().any(|pat| error.contains(pat))
    }

    /// Given the current context and a failure, decide whether to retry.
    pub fn decide(&self, ctx: &RetryContext, error: &str) -> RetryDecision {
        // Check if the error is retryable.
        if !self.is_retryable(error) {
            return RetryDecision::Stop(RetryError::NonRetryable {
                error: error.to_string(),
            });
        }

        // Check max attempts.
        if ctx.attempt >= self.max_attempts {
            return RetryDecision::Stop(RetryError::Exhausted {
                attempts: ctx.attempt + 1,
                last_error: error.to_string(),
            });
        }

        // Calculate delay for next attempt.
        let delay_ms = self.strategy.delay_ms(ctx.attempt);

        // Check total duration budget.
        if self.max_total_duration_ms > 0 {
            let projected_total = ctx.total_delay_ms() + delay_ms;
            if projected_total > self.max_total_duration_ms {
                return RetryDecision::Stop(RetryError::DurationExceeded {
                    elapsed_ms: ctx.total_delay_ms(),
                    budget_ms: self.max_total_duration_ms,
                    last_error: error.to_string(),
                });
            }
        }

        RetryDecision::RetryAfter(Duration::from_millis(delay_ms))
    }
}

// ── Retry Executor ────────────────────────────────────────────

/// Synchronous retry executor that drives attempts via an outcome callback.
///
/// Returns the attempt index on success or the final RetryError on failure.
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    /// Create a new executor with the given policy.
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Execute a retry loop. The `attempt_fn` closure is called for each attempt.
    /// Returns Ok(attempt_number) on success or Err(RetryError) on final failure.
    pub fn execute<F>(&self, mut attempt_fn: F) -> Result<RetryContext, RetryError>
    where
        F: FnMut(&RetryContext) -> AttemptOutcome,
    {
        let mut ctx = RetryContext::new();

        loop {
            let outcome = attempt_fn(&ctx);

            match outcome {
                AttemptOutcome::Success => return Ok(ctx),
                AttemptOutcome::NonRetryableFailure(err) => {
                    return Err(RetryError::NonRetryable { error: err });
                }
                AttemptOutcome::RetryableFailure(err) => {
                    let decision = self.policy.decide(&ctx, &err);
                    match decision {
                        RetryDecision::RetryAfter(delay) => {
                            ctx.delays_applied.push(delay.as_millis() as u64);
                            ctx.elapsed_ms += delay.as_millis() as u64;
                            ctx.last_error = err;
                            ctx.attempt += 1;
                        }
                        RetryDecision::Stop(retry_err) => {
                            return Err(retry_err);
                        }
                    }
                }
            }
        }
    }

    /// Get a reference to the policy.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }
}

// ── Schedule Preview ──────────────────────────────────────────

/// Preview the retry schedule without executing.
pub fn preview_schedule(policy: &RetryPolicy) -> Vec<u64> {
    let mut delays = Vec::new();
    for attempt in 0..policy.max_attempts {
        delays.push(policy.strategy.delay_ms(attempt));
    }
    delays
}

/// Total delay of a retry schedule preview.
pub fn total_schedule_delay_ms(policy: &RetryPolicy) -> u64 {
    preview_schedule(policy).iter().sum()
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_backoff() {
        let strategy = BackoffStrategy::Constant { delay_ms: 100 };
        assert_eq!(strategy.delay_ms(0), 100);
        assert_eq!(strategy.delay_ms(5), 100);
        assert_eq!(strategy.delay_ms(100), 100);
    }

    #[test]
    fn test_linear_backoff() {
        let strategy = BackoffStrategy::Linear { base_delay_ms: 100 };
        assert_eq!(strategy.delay_ms(0), 100);
        assert_eq!(strategy.delay_ms(1), 200);
        assert_eq!(strategy.delay_ms(2), 300);
        assert_eq!(strategy.delay_ms(9), 1000);
    }

    #[test]
    fn test_exponential_backoff() {
        let strategy = BackoffStrategy::Exponential {
            base_delay_ms: 100,
            max_delay_ms: 10000,
        };
        assert_eq!(strategy.delay_ms(0), 100);   // 100 * 1
        assert_eq!(strategy.delay_ms(1), 200);   // 100 * 2
        assert_eq!(strategy.delay_ms(2), 400);   // 100 * 4
        assert_eq!(strategy.delay_ms(3), 800);   // 100 * 8
        assert_eq!(strategy.delay_ms(10), 10000); // capped at max
    }

    #[test]
    fn test_exponential_jitter() {
        let strategy = BackoffStrategy::ExponentialWithJitter {
            base_delay_ms: 100,
            max_delay_ms: 10000,
            jitter_ms: 50,
        };
        let d0 = strategy.delay_ms(0);
        let d1 = strategy.delay_ms(1);
        // Jitter adds a deterministic offset.
        assert!(d0 >= 100);
        assert!(d1 >= 200);
    }

    #[test]
    fn test_fibonacci_backoff() {
        let strategy = BackoffStrategy::Fibonacci { base_delay_ms: 100 };
        // fib: 1, 1, 2, 3, 5, 8, 13, 21
        assert_eq!(strategy.delay_ms(0), 100);  // 1*100
        assert_eq!(strategy.delay_ms(1), 100);  // 1*100
        assert_eq!(strategy.delay_ms(2), 200);  // 2*100
        assert_eq!(strategy.delay_ms(3), 300);  // 3*100
        assert_eq!(strategy.delay_ms(4), 500);  // 5*100
        assert_eq!(strategy.delay_ms(5), 800);  // 8*100
    }

    #[test]
    fn test_fibonacci_function() {
        assert_eq!(fibonacci(0), 1);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 2);
        assert_eq!(fibonacci(3), 3);
        assert_eq!(fibonacci(4), 5);
        assert_eq!(fibonacci(5), 8);
    }

    #[test]
    fn test_retry_context_new() {
        let ctx = RetryContext::new();
        assert_eq!(ctx.attempt, 0);
        assert!(ctx.last_error.is_empty());
        assert!(ctx.is_first_attempt());
        assert_eq!(ctx.total_delay_ms(), 0);
    }

    #[test]
    fn test_retry_policy_max_attempts() {
        let policy = RetryPolicy::constant(100, 3);
        let mut ctx = RetryContext::new();
        // Attempts 0, 1, 2 should be allowed. Attempt 3 should stop.
        for i in 0..3 {
            ctx.attempt = i;
            let decision = policy.decide(&ctx, "error");
            assert!(matches!(decision, RetryDecision::RetryAfter(_)));
        }
        ctx.attempt = 3;
        let decision = policy.decide(&ctx, "error");
        assert!(matches!(decision, RetryDecision::Stop(RetryError::Exhausted { .. })));
    }

    #[test]
    fn test_retry_policy_duration_budget() {
        let policy = RetryPolicy::constant(100, 10).with_max_duration(250);
        let mut ctx = RetryContext::new();
        // First two retries: 100 + 100 = 200ms, within budget.
        ctx.delays_applied = vec![100, 100];
        ctx.attempt = 2;
        // Next delay 100 would bring total to 300, exceeding 250.
        let decision = policy.decide(&ctx, "error");
        assert!(matches!(decision, RetryDecision::Stop(RetryError::DurationExceeded { .. })));
    }

    #[test]
    fn test_retry_policy_retryable_errors() {
        let policy = RetryPolicy::constant(100, 3)
            .with_retryable_errors(vec!["timeout".to_string(), "503".to_string()]);
        assert!(policy.is_retryable("connection timeout"));
        assert!(policy.is_retryable("status 503"));
        assert!(!policy.is_retryable("permission denied"));
    }

    #[test]
    fn test_non_retryable_decision() {
        let policy = RetryPolicy::constant(100, 3)
            .with_retryable_errors(vec!["timeout".to_string()]);
        let ctx = RetryContext::new();
        let decision = policy.decide(&ctx, "permission denied");
        assert!(matches!(decision, RetryDecision::Stop(RetryError::NonRetryable { .. })));
    }

    #[test]
    fn test_retry_executor_success_first_try() {
        let policy = RetryPolicy::constant(100, 3);
        let executor = RetryExecutor::new(policy);
        let result = executor.execute(|_ctx| AttemptOutcome::Success);
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert_eq!(ctx.attempt, 0);
    }

    #[test]
    fn test_retry_executor_success_after_retries() {
        let policy = RetryPolicy::constant(100, 5);
        let executor = RetryExecutor::new(policy);
        let result = executor.execute(|ctx| {
            if ctx.attempt < 3 {
                AttemptOutcome::RetryableFailure("fail".to_string())
            } else {
                AttemptOutcome::Success
            }
        });
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert_eq!(ctx.attempt, 3);
    }

    #[test]
    fn test_retry_executor_exhausted() {
        let policy = RetryPolicy::constant(100, 2);
        let executor = RetryExecutor::new(policy);
        let result = executor.execute(|_ctx| {
            AttemptOutcome::RetryableFailure("always fail".to_string())
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RetryError::Exhausted { attempts: 3, .. }));
    }

    #[test]
    fn test_retry_executor_non_retryable() {
        let policy = RetryPolicy::constant(100, 5);
        let executor = RetryExecutor::new(policy);
        let result = executor.execute(|_ctx| {
            AttemptOutcome::NonRetryableFailure("fatal".to_string())
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RetryError::NonRetryable { .. }));
    }

    #[test]
    fn test_preview_schedule() {
        let policy = RetryPolicy::exponential(100, 5000, 5);
        let schedule = preview_schedule(&policy);
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule[0], 100);
        assert_eq!(schedule[1], 200);
        assert_eq!(schedule[2], 400);
    }

    #[test]
    fn test_total_schedule_delay() {
        let policy = RetryPolicy::constant(100, 3);
        let total = total_schedule_delay_ms(&policy);
        assert_eq!(total, 300);
    }

    #[test]
    fn test_delay_as_duration() {
        let strategy = BackoffStrategy::Constant { delay_ms: 250 };
        let dur = strategy.delay(0);
        assert_eq!(dur, Duration::from_millis(250));
    }

    #[test]
    fn test_retry_error_display() {
        let err = RetryError::Exhausted { attempts: 5, last_error: "timeout".into() };
        let msg = format!("{}", err);
        assert!(msg.contains("5 attempts"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn test_executor_tracks_delays() {
        let policy = RetryPolicy::linear(50, 5);
        let executor = RetryExecutor::new(policy);
        let result = executor.execute(|ctx| {
            if ctx.attempt < 2 {
                AttemptOutcome::RetryableFailure("err".to_string())
            } else {
                AttemptOutcome::Success
            }
        });
        let ctx = result.unwrap();
        assert_eq!(ctx.delays_applied.len(), 2);
        assert_eq!(ctx.delays_applied[0], 50);  // linear: 50 * 1
        assert_eq!(ctx.delays_applied[1], 100); // linear: 50 * 2
    }
}
