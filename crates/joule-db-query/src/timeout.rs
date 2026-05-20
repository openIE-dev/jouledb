//! Query Timeout and Cancellation System
//!
//! This module provides comprehensive timeout management and cooperative
//! cancellation for query execution in JouleDB.
//!
//! # Features
//!
//! - Configurable query timeouts
//! - Cooperative cancellation via cancellation tokens
//! - Graceful query interruption at checkpoint locations
//! - Resource cleanup on cancellation
//! - Detailed timeout and cancellation statistics
//!
//! # Example
//!
//! ```ignore
//! use joule_db_query::timeout::{CancellationToken, QueryTimeout, TimeoutConfig};
//! use std::time::Duration;
//!
//! let config = TimeoutConfig::new(Duration::from_secs(30));
//! let token = CancellationToken::new();
//! let timeout = QueryTimeout::new(config, token.clone());
//!
//! // Execute with timeout
//! let result = timeout.execute(|| {
//!     // Long running query
//!     Ok(42)
//! });
//! ```

use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Reason for cancellation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancellationReason {
    /// User requested cancellation
    UserRequested,
    /// Query exceeded timeout
    Timeout,
    /// System shutdown
    Shutdown,
    /// Resource limit exceeded
    ResourceLimit(String),
    /// Parent query was cancelled
    ParentCancelled,
    /// Custom reason
    Custom(String),
}

impl fmt::Display for CancellationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserRequested => write!(f, "user requested cancellation"),
            Self::Timeout => write!(f, "query timeout exceeded"),
            Self::Shutdown => write!(f, "system shutdown"),
            Self::ResourceLimit(msg) => write!(f, "resource limit exceeded: {}", msg),
            Self::ParentCancelled => write!(f, "parent query was cancelled"),
            Self::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

/// A token that can be used to signal cancellation to running operations.
///
/// CancellationToken provides cooperative cancellation - operations must
/// periodically check the token and respond to cancellation requests.
#[derive(Clone)]
pub struct CancellationToken {
    inner: Arc<CancellationTokenInner>,
}

struct CancellationTokenInner {
    /// Whether cancellation has been requested
    cancelled: AtomicBool,
    /// The reason for cancellation
    reason: RwLock<Option<CancellationReason>>,
    /// Time when cancellation was requested
    cancelled_at: RwLock<Option<Instant>>,
    /// Child tokens that should be cancelled when this token is cancelled
    children: Mutex<Vec<CancellationToken>>,
    /// Cleanup callbacks to run on cancellation
    cleanup_callbacks: Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CancellationTokenInner {
                cancelled: AtomicBool::new(false),
                reason: RwLock::new(None),
                cancelled_at: RwLock::new(None),
                children: Mutex::new(Vec::new()),
                cleanup_callbacks: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Create a child token that will be cancelled when the parent is cancelled
    pub fn child(&self) -> CancellationToken {
        let child = CancellationToken::new();

        // If parent is already cancelled, cancel child immediately
        if self.is_cancelled() {
            if self.cancellation_reason().is_some() {
                child.cancel_with_reason(CancellationReason::ParentCancelled);
            } else {
                child.cancel();
            }
        } else {
            // Register child for future cancellation
            if let Ok(mut children) = self.inner.children.lock() {
                children.push(child.clone());
            }
        }

        child
    }

    /// Request cancellation without a specific reason
    pub fn cancel(&self) {
        self.cancel_with_reason(CancellationReason::UserRequested);
    }

    /// Request cancellation with a specific reason
    pub fn cancel_with_reason(&self, reason: CancellationReason) {
        // Only set if not already cancelled
        if self
            .inner
            .cancelled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Set the reason and time
            if let Ok(mut r) = self.inner.reason.write() {
                *r = Some(reason);
            }
            if let Ok(mut t) = self.inner.cancelled_at.write() {
                *t = Some(Instant::now());
            }

            // Cancel all children
            if let Ok(children) = self.inner.children.lock() {
                for child in children.iter() {
                    child.cancel_with_reason(CancellationReason::ParentCancelled);
                }
            }

            // Run cleanup callbacks
            self.run_cleanup();
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Get the reason for cancellation, if cancelled
    pub fn cancellation_reason(&self) -> Option<CancellationReason> {
        if let Ok(reason) = self.inner.reason.read() {
            reason.clone()
        } else {
            None
        }
    }

    /// Get the time when cancellation was requested
    pub fn cancelled_at(&self) -> Option<Instant> {
        if let Ok(time) = self.inner.cancelled_at.read() {
            *time
        } else {
            None
        }
    }

    /// Register a cleanup callback to run when cancellation is requested
    pub fn on_cancel<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if self.is_cancelled() {
            // Already cancelled, run immediately
            callback();
        } else {
            if let Ok(mut callbacks) = self.inner.cleanup_callbacks.lock() {
                callbacks.push(Box::new(callback));
            }
        }
    }

    /// Check cancellation and return error if cancelled
    pub fn check(&self) -> QueryResult<()> {
        if self.is_cancelled() {
            let reason = self
                .cancellation_reason()
                .unwrap_or(CancellationReason::UserRequested);
            Err(QueryError::ExecutionError(format!(
                "Query cancelled: {}",
                reason
            )))
        } else {
            Ok(())
        }
    }

    /// Run cleanup callbacks
    fn run_cleanup(&self) {
        if let Ok(mut callbacks) = self.inner.cleanup_callbacks.lock() {
            // Take ownership of callbacks and run them
            let callbacks: Vec<_> = std::mem::take(&mut *callbacks);
            for callback in callbacks {
                callback();
            }
        }
    }

    /// Reset the token for reuse (primarily for testing)
    pub fn reset(&self) {
        self.inner.cancelled.store(false, Ordering::SeqCst);
        if let Ok(mut reason) = self.inner.reason.write() {
            *reason = None;
        }
        if let Ok(mut time) = self.inner.cancelled_at.write() {
            *time = None;
        }
        if let Ok(mut children) = self.inner.children.lock() {
            children.clear();
        }
        if let Ok(mut callbacks) = self.inner.cleanup_callbacks.lock() {
            callbacks.clear();
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .field("reason", &self.cancellation_reason())
            .finish()
    }
}

/// Configuration for query timeout behavior
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Maximum execution time for a query
    pub query_timeout: Duration,
    /// Timeout for individual operations within a query
    pub operation_timeout: Option<Duration>,
    /// Interval between checkpoint checks
    pub checkpoint_interval: Duration,
    /// Whether to allow timeout extension
    pub allow_extension: bool,
    /// Maximum allowed extension time
    pub max_extension: Duration,
    /// Soft timeout threshold (for warnings)
    pub soft_timeout: Option<Duration>,
    /// Grace period after timeout before force termination
    pub grace_period: Duration,
}

impl TimeoutConfig {
    /// Create a new timeout configuration with the specified query timeout
    pub fn new(query_timeout: Duration) -> Self {
        Self {
            query_timeout,
            operation_timeout: None,
            checkpoint_interval: Duration::from_millis(100),
            allow_extension: false,
            max_extension: Duration::from_secs(0),
            soft_timeout: None,
            grace_period: Duration::from_millis(500),
        }
    }

    /// Set the operation timeout
    pub fn with_operation_timeout(mut self, timeout: Duration) -> Self {
        self.operation_timeout = Some(timeout);
        self
    }

    /// Set the checkpoint interval
    pub fn with_checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Allow timeout extensions
    pub fn with_extension(mut self, max_extension: Duration) -> Self {
        self.allow_extension = true;
        self.max_extension = max_extension;
        self
    }

    /// Set soft timeout threshold
    pub fn with_soft_timeout(mut self, soft_timeout: Duration) -> Self {
        self.soft_timeout = Some(soft_timeout);
        self
    }

    /// Set grace period
    pub fn with_grace_period(mut self, grace_period: Duration) -> Self {
        self.grace_period = grace_period;
        self
    }

    /// Create a lenient configuration for long-running queries
    pub fn lenient(query_timeout: Duration) -> Self {
        Self::new(query_timeout)
            .with_checkpoint_interval(Duration::from_millis(500))
            .with_extension(Duration::from_secs(60))
            .with_grace_period(Duration::from_secs(5))
    }

    /// Create a strict configuration for quick queries
    pub fn strict(query_timeout: Duration) -> Self {
        Self::new(query_timeout)
            .with_checkpoint_interval(Duration::from_millis(10))
            .with_grace_period(Duration::from_millis(100))
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

/// Statistics about timeout and cancellation events
#[derive(Debug, Default)]
pub struct TimeoutStatistics {
    /// Total number of queries executed
    pub total_queries: AtomicU64,
    /// Number of queries that completed successfully
    pub completed_queries: AtomicU64,
    /// Number of queries that timed out
    pub timed_out_queries: AtomicU64,
    /// Number of queries that were cancelled
    pub cancelled_queries: AtomicU64,
    /// Total execution time across all queries (in microseconds)
    pub total_execution_time_us: AtomicU64,
    /// Number of checkpoint checks performed
    pub checkpoint_checks: AtomicU64,
    /// Number of times soft timeout was reached
    pub soft_timeout_warnings: AtomicU64,
    /// Number of timeout extensions granted
    pub extensions_granted: AtomicU64,
    /// Cancellation reasons breakdown
    cancellation_reasons: Mutex<HashMap<String, u64>>,
}

impl TimeoutStatistics {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed query
    pub fn record_completed(&self, execution_time: Duration) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.completed_queries.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_us
            .fetch_add(execution_time.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a timed out query
    pub fn record_timeout(&self, execution_time: Duration) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.timed_out_queries.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_us
            .fetch_add(execution_time.as_micros() as u64, Ordering::Relaxed);
        self.record_cancellation_reason("timeout");
    }

    /// Record a cancelled query
    pub fn record_cancelled(&self, execution_time: Duration, reason: &CancellationReason) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.cancelled_queries.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_us
            .fetch_add(execution_time.as_micros() as u64, Ordering::Relaxed);
        self.record_cancellation_reason(&reason.to_string());
    }

    /// Record a checkpoint check
    pub fn record_checkpoint(&self) {
        self.checkpoint_checks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a soft timeout warning
    pub fn record_soft_timeout(&self) {
        self.soft_timeout_warnings.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an extension granted
    pub fn record_extension(&self) {
        self.extensions_granted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_cancellation_reason(&self, reason: &str) {
        if let Ok(mut reasons) = self.cancellation_reasons.lock() {
            *reasons.entry(reason.to_string()).or_insert(0) += 1;
        }
    }

    /// Get average execution time in milliseconds
    pub fn average_execution_time_ms(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let total_us = self.total_execution_time_us.load(Ordering::Relaxed);
        (total_us as f64 / total as f64) / 1000.0
    }

    /// Get timeout rate as a percentage
    pub fn timeout_rate(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let timed_out = self.timed_out_queries.load(Ordering::Relaxed);
        (timed_out as f64 / total as f64) * 100.0
    }

    /// Get cancellation reasons breakdown
    pub fn cancellation_reasons(&self) -> HashMap<String, u64> {
        if let Ok(reasons) = self.cancellation_reasons.lock() {
            reasons.clone()
        } else {
            HashMap::new()
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_queries.store(0, Ordering::Relaxed);
        self.completed_queries.store(0, Ordering::Relaxed);
        self.timed_out_queries.store(0, Ordering::Relaxed);
        self.cancelled_queries.store(0, Ordering::Relaxed);
        self.total_execution_time_us.store(0, Ordering::Relaxed);
        self.checkpoint_checks.store(0, Ordering::Relaxed);
        self.soft_timeout_warnings.store(0, Ordering::Relaxed);
        self.extensions_granted.store(0, Ordering::Relaxed);
        if let Ok(mut reasons) = self.cancellation_reasons.lock() {
            reasons.clear();
        }
    }
}

impl Clone for TimeoutStatistics {
    fn clone(&self) -> Self {
        Self {
            total_queries: AtomicU64::new(self.total_queries.load(Ordering::Relaxed)),
            completed_queries: AtomicU64::new(self.completed_queries.load(Ordering::Relaxed)),
            timed_out_queries: AtomicU64::new(self.timed_out_queries.load(Ordering::Relaxed)),
            cancelled_queries: AtomicU64::new(self.cancelled_queries.load(Ordering::Relaxed)),
            total_execution_time_us: AtomicU64::new(
                self.total_execution_time_us.load(Ordering::Relaxed),
            ),
            checkpoint_checks: AtomicU64::new(self.checkpoint_checks.load(Ordering::Relaxed)),
            soft_timeout_warnings: AtomicU64::new(
                self.soft_timeout_warnings.load(Ordering::Relaxed),
            ),
            extensions_granted: AtomicU64::new(self.extensions_granted.load(Ordering::Relaxed)),
            cancellation_reasons: Mutex::new(self.cancellation_reasons()),
        }
    }
}

/// Checkpoint context for cooperative cancellation
pub struct CheckpointContext {
    token: CancellationToken,
    config: TimeoutConfig,
    start_time: Instant,
    last_checkpoint: Mutex<Instant>,
    deadline: Instant,
    extended_deadline: Mutex<Option<Instant>>,
    stats: Arc<TimeoutStatistics>,
    soft_timeout_triggered: AtomicBool,
}

impl CheckpointContext {
    /// Create a new checkpoint context
    pub fn new(
        token: CancellationToken,
        config: TimeoutConfig,
        stats: Arc<TimeoutStatistics>,
    ) -> Self {
        let now = Instant::now();
        Self {
            token,
            config: config.clone(),
            start_time: now,
            last_checkpoint: Mutex::new(now),
            deadline: now + config.query_timeout,
            extended_deadline: Mutex::new(None),
            stats,
            soft_timeout_triggered: AtomicBool::new(false),
        }
    }

    /// Check if the operation should continue at a checkpoint
    ///
    /// This method should be called periodically during long-running operations.
    /// It checks for cancellation, timeout, and updates statistics.
    pub fn checkpoint(&self) -> QueryResult<()> {
        self.stats.record_checkpoint();

        // Check cancellation token first
        self.token.check()?;

        let now = Instant::now();

        // Check soft timeout
        if let Some(soft) = self.config.soft_timeout {
            if !self.soft_timeout_triggered.load(Ordering::Relaxed)
                && now.duration_since(self.start_time) >= soft
            {
                self.soft_timeout_triggered.store(true, Ordering::Relaxed);
                self.stats.record_soft_timeout();
            }
        }

        // Check hard timeout
        let effective_deadline = self
            .extended_deadline
            .lock()
            .ok()
            .and_then(|d| *d)
            .unwrap_or(self.deadline);

        if now >= effective_deadline {
            // Check grace period
            if now >= effective_deadline + self.config.grace_period {
                self.token.cancel_with_reason(CancellationReason::Timeout);
                return Err(QueryError::Timeout);
            }
        }

        // Update last checkpoint time
        if let Ok(mut last) = self.last_checkpoint.lock() {
            *last = now;
        }

        Ok(())
    }

    /// Check checkpoint only if enough time has passed since last check
    ///
    /// This is more efficient for very tight loops where checking every
    /// iteration would be too expensive.
    pub fn checkpoint_throttled(&self) -> QueryResult<()> {
        let should_check = if let Ok(last) = self.last_checkpoint.lock() {
            Instant::now().duration_since(*last) >= self.config.checkpoint_interval
        } else {
            true
        };

        if should_check {
            self.checkpoint()
        } else {
            Ok(())
        }
    }

    /// Request a timeout extension
    ///
    /// Returns true if the extension was granted.
    pub fn request_extension(&self, additional_time: Duration) -> bool {
        if !self.config.allow_extension {
            return false;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time);
        let requested_total = elapsed + additional_time;

        if requested_total > self.config.query_timeout + self.config.max_extension {
            return false;
        }

        if let Ok(mut deadline) = self.extended_deadline.lock() {
            *deadline = Some(now + additional_time);
            self.stats.record_extension();
            true
        } else {
            false
        }
    }

    /// Get elapsed time since query started
    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }

    /// Get remaining time until timeout
    pub fn remaining(&self) -> Duration {
        let effective_deadline = self
            .extended_deadline
            .lock()
            .ok()
            .and_then(|d| *d)
            .unwrap_or(self.deadline);

        let now = Instant::now();
        if now >= effective_deadline {
            Duration::ZERO
        } else {
            effective_deadline - now
        }
    }

    /// Check if soft timeout has been triggered
    pub fn soft_timeout_triggered(&self) -> bool {
        self.soft_timeout_triggered.load(Ordering::Relaxed)
    }

    /// Get the cancellation token
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }
}

/// Query timeout wrapper that manages timeout and cancellation for query execution
pub struct QueryTimeout {
    config: TimeoutConfig,
    token: CancellationToken,
    stats: Arc<TimeoutStatistics>,
}

impl QueryTimeout {
    /// Create a new query timeout wrapper
    pub fn new(config: TimeoutConfig, token: CancellationToken) -> Self {
        Self {
            config,
            token,
            stats: Arc::new(TimeoutStatistics::new()),
        }
    }

    /// Create with shared statistics
    pub fn with_stats(
        config: TimeoutConfig,
        token: CancellationToken,
        stats: Arc<TimeoutStatistics>,
    ) -> Self {
        Self {
            config,
            token,
            stats,
        }
    }

    /// Execute a function with timeout and cancellation support
    pub fn execute<F, T>(&self, f: F) -> QueryResult<T>
    where
        F: FnOnce() -> QueryResult<T>,
    {
        let start = Instant::now();

        // Check if already cancelled before starting
        self.token.check()?;

        // Execute the function
        let result = f();
        let elapsed = start.elapsed();

        // Record statistics
        match &result {
            Ok(_) => self.stats.record_completed(elapsed),
            Err(QueryError::Timeout) => self.stats.record_timeout(elapsed),
            Err(_) if self.token.is_cancelled() => {
                let reason = self
                    .token
                    .cancellation_reason()
                    .unwrap_or(CancellationReason::UserRequested);
                self.stats.record_cancelled(elapsed, &reason);
            }
            Err(_) => {
                // Other error, still count as completed
                self.stats.record_completed(elapsed);
            }
        }

        result
    }

    /// Execute a function with checkpoint support
    ///
    /// The function receives a CheckpointContext that should be used to
    /// periodically check for timeout/cancellation.
    pub fn execute_with_checkpoints<F, T>(&self, f: F) -> QueryResult<T>
    where
        F: FnOnce(&CheckpointContext) -> QueryResult<T>,
    {
        let start = Instant::now();

        // Check if already cancelled before starting
        self.token.check()?;

        // Create checkpoint context
        let ctx =
            CheckpointContext::new(self.token.clone(), self.config.clone(), self.stats.clone());

        // Execute with checkpoint context
        let result = f(&ctx);
        let elapsed = start.elapsed();

        // Record statistics
        match &result {
            Ok(_) => self.stats.record_completed(elapsed),
            Err(QueryError::Timeout) => self.stats.record_timeout(elapsed),
            Err(_) if self.token.is_cancelled() => {
                let reason = self
                    .token
                    .cancellation_reason()
                    .unwrap_or(CancellationReason::UserRequested);
                self.stats.record_cancelled(elapsed, &reason);
            }
            Err(_) => {
                self.stats.record_completed(elapsed);
            }
        }

        result
    }

    /// Get the cancellation token
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// Get the configuration
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &Arc<TimeoutStatistics> {
        &self.stats
    }

    /// Cancel the associated token
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Cancel with a specific reason
    pub fn cancel_with_reason(&self, reason: CancellationReason) {
        self.token.cancel_with_reason(reason);
    }
}

impl Clone for QueryTimeout {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token: self.token.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl fmt::Debug for QueryTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryTimeout")
            .field("config", &self.config)
            .field("token", &self.token)
            .finish()
    }
}

/// A guard that ensures cleanup runs when dropped
pub struct CleanupGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

impl<F: FnOnce()> CleanupGuard<F> {
    /// Create a new cleanup guard
    pub fn new(cleanup: F) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }

    /// Disarm the guard, preventing cleanup from running
    pub fn disarm(&mut self) {
        self.cleanup = None;
    }
}

impl<F: FnOnce()> Drop for CleanupGuard<F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

/// Execute an operation with automatic resource cleanup on cancellation
pub fn with_cleanup<T, F, C>(token: &CancellationToken, operation: F, cleanup: C) -> QueryResult<T>
where
    F: FnOnce() -> QueryResult<T>,
    C: FnOnce() + Send + 'static,
{
    // Register cleanup callback
    let cleanup_registered = Arc::new(AtomicBool::new(true));
    let cleanup_ref = cleanup_registered.clone();

    token.on_cancel(move || {
        if cleanup_ref.load(Ordering::SeqCst) {
            cleanup();
        }
    });

    // Execute operation
    let result = operation();

    // If successful, don't run cleanup
    if result.is_ok() {
        cleanup_registered.store(false, Ordering::SeqCst);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cancellation_token_basic() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        assert!(token.cancellation_reason().is_none());

        token.cancel();
        assert!(token.is_cancelled());
        assert_eq!(
            token.cancellation_reason(),
            Some(CancellationReason::UserRequested)
        );
    }

    #[test]
    fn test_cancellation_token_with_reason() {
        let token = CancellationToken::new();
        token.cancel_with_reason(CancellationReason::Timeout);

        assert!(token.is_cancelled());
        assert_eq!(
            token.cancellation_reason(),
            Some(CancellationReason::Timeout)
        );
    }

    #[test]
    fn test_cancellation_token_child() {
        let parent = CancellationToken::new();
        let child = parent.child();

        assert!(!child.is_cancelled());

        parent.cancel();
        assert!(parent.is_cancelled());
        assert!(child.is_cancelled());
        assert_eq!(
            child.cancellation_reason(),
            Some(CancellationReason::ParentCancelled)
        );
    }

    #[test]
    fn test_cancellation_token_check() {
        let token = CancellationToken::new();
        assert!(token.check().is_ok());

        token.cancel();
        assert!(token.check().is_err());
    }

    #[test]
    fn test_cancellation_callback() {
        let token = CancellationToken::new();
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        token.on_cancel(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        assert!(!called.load(Ordering::SeqCst));
        token.cancel();
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_timeout_config() {
        let config = TimeoutConfig::new(Duration::from_secs(30))
            .with_operation_timeout(Duration::from_secs(5))
            .with_soft_timeout(Duration::from_secs(20))
            .with_extension(Duration::from_secs(10));

        assert_eq!(config.query_timeout, Duration::from_secs(30));
        assert_eq!(config.operation_timeout, Some(Duration::from_secs(5)));
        assert_eq!(config.soft_timeout, Some(Duration::from_secs(20)));
        assert!(config.allow_extension);
        assert_eq!(config.max_extension, Duration::from_secs(10));
    }

    #[test]
    fn test_timeout_config_presets() {
        let lenient = TimeoutConfig::lenient(Duration::from_secs(60));
        assert!(lenient.allow_extension);
        assert!(lenient.grace_period >= Duration::from_secs(1));

        let strict = TimeoutConfig::strict(Duration::from_secs(5));
        assert!(!strict.allow_extension);
        assert!(strict.checkpoint_interval <= Duration::from_millis(50));
    }

    #[test]
    fn test_query_timeout_execute() {
        let config = TimeoutConfig::new(Duration::from_secs(30));
        let token = CancellationToken::new();
        let timeout = QueryTimeout::new(config, token);

        let result = timeout.execute(|| Ok(42));
        assert_eq!(result.unwrap(), 42);

        assert_eq!(timeout.stats.completed_queries.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_query_timeout_cancelled() {
        let config = TimeoutConfig::new(Duration::from_secs(30));
        let token = CancellationToken::new();
        token.cancel();

        let timeout = QueryTimeout::new(config, token);
        let result: QueryResult<i32> = timeout.execute(|| Ok(42));

        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_context() {
        let token = CancellationToken::new();
        let config = TimeoutConfig::new(Duration::from_secs(30));
        let stats = Arc::new(TimeoutStatistics::new());

        let ctx = CheckpointContext::new(token, config, stats.clone());

        // Initial checkpoint should succeed
        assert!(ctx.checkpoint().is_ok());

        // Check elapsed time
        assert!(ctx.elapsed() < Duration::from_secs(1));

        // Check remaining time
        assert!(ctx.remaining() > Duration::from_secs(25));
    }

    #[test]
    fn test_checkpoint_with_cancellation() {
        let token = CancellationToken::new();
        let config = TimeoutConfig::new(Duration::from_secs(30));
        let stats = Arc::new(TimeoutStatistics::new());

        let ctx = CheckpointContext::new(token.clone(), config, stats);

        assert!(ctx.checkpoint().is_ok());

        token.cancel();
        assert!(ctx.checkpoint().is_err());
    }

    #[test]
    fn test_execute_with_checkpoints() {
        let config = TimeoutConfig::new(Duration::from_secs(30));
        let token = CancellationToken::new();
        let timeout = QueryTimeout::new(config, token);

        let result = timeout.execute_with_checkpoints(|ctx| {
            let mut sum = 0;
            for i in 0..100 {
                ctx.checkpoint_throttled()?;
                sum += i;
            }
            Ok(sum)
        });

        assert_eq!(result.unwrap(), 4950);
    }

    #[test]
    fn test_timeout_statistics() {
        let stats = TimeoutStatistics::new();

        stats.record_completed(Duration::from_millis(100));
        stats.record_completed(Duration::from_millis(200));
        stats.record_timeout(Duration::from_millis(5000));

        assert_eq!(stats.total_queries.load(Ordering::Relaxed), 3);
        assert_eq!(stats.completed_queries.load(Ordering::Relaxed), 2);
        assert_eq!(stats.timed_out_queries.load(Ordering::Relaxed), 1);

        // Average should be around 1766.67 ms
        let avg = stats.average_execution_time_ms();
        assert!(avg > 1700.0 && avg < 1850.0);

        // Timeout rate should be ~33%
        let rate = stats.timeout_rate();
        assert!(rate > 30.0 && rate < 35.0);
    }

    #[test]
    fn test_cleanup_guard() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_clone = cleaned.clone();

        {
            let _guard = CleanupGuard::new(move || {
                cleaned_clone.store(true, Ordering::SeqCst);
            });
        }

        assert!(cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_cleanup_guard_disarm() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_clone = cleaned.clone();

        {
            let mut guard = CleanupGuard::new(move || {
                cleaned_clone.store(true, Ordering::SeqCst);
            });
            guard.disarm();
        }

        assert!(!cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_with_cleanup_success() {
        let token = CancellationToken::new();
        let cleanup_called = Arc::new(AtomicBool::new(false));
        let cleanup_clone = cleanup_called.clone();

        let result = with_cleanup(
            &token,
            || Ok(42),
            move || {
                cleanup_clone.store(true, Ordering::SeqCst);
            },
        );

        assert_eq!(result.unwrap(), 42);
        // Cleanup should not be called on success
        // Note: Due to the callback mechanism, we can't easily prevent
        // the callback from being stored, but it won't execute until cancellation
    }

    #[test]
    fn test_cancellation_reason_display() {
        assert_eq!(
            CancellationReason::UserRequested.to_string(),
            "user requested cancellation"
        );
        assert_eq!(
            CancellationReason::Timeout.to_string(),
            "query timeout exceeded"
        );
        assert_eq!(
            CancellationReason::ResourceLimit("memory".to_string()).to_string(),
            "resource limit exceeded: memory"
        );
    }

    #[test]
    fn test_statistics_reset() {
        let stats = TimeoutStatistics::new();
        stats.record_completed(Duration::from_millis(100));
        stats.record_timeout(Duration::from_millis(5000));

        assert_eq!(stats.total_queries.load(Ordering::Relaxed), 2);

        stats.reset();

        assert_eq!(stats.total_queries.load(Ordering::Relaxed), 0);
        assert_eq!(stats.completed_queries.load(Ordering::Relaxed), 0);
        assert_eq!(stats.timed_out_queries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_checkpoint_extension() {
        let token = CancellationToken::new();
        let config =
            TimeoutConfig::new(Duration::from_secs(1)).with_extension(Duration::from_secs(10));
        let stats = Arc::new(TimeoutStatistics::new());

        let ctx = CheckpointContext::new(token, config, stats.clone());

        // Initial remaining should be ~1 second
        let initial_remaining = ctx.remaining();
        assert!(initial_remaining <= Duration::from_secs(1));

        // Request extension
        assert!(ctx.request_extension(Duration::from_secs(5)));

        // Remaining should now be more
        let new_remaining = ctx.remaining();
        assert!(new_remaining > initial_remaining);

        // Stats should show extension
        assert_eq!(stats.extensions_granted.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multiple_child_tokens() {
        let parent = CancellationToken::new();
        let child1 = parent.child();
        let child2 = parent.child();
        let grandchild = child1.child();

        assert!(!parent.is_cancelled());
        assert!(!child1.is_cancelled());
        assert!(!child2.is_cancelled());
        assert!(!grandchild.is_cancelled());

        parent.cancel();

        assert!(parent.is_cancelled());
        assert!(child1.is_cancelled());
        assert!(child2.is_cancelled());
        assert!(grandchild.is_cancelled());
    }

    #[test]
    fn test_soft_timeout_tracking() {
        let token = CancellationToken::new();
        let config = TimeoutConfig::new(Duration::from_secs(10))
            .with_soft_timeout(Duration::from_millis(10));
        let stats = Arc::new(TimeoutStatistics::new());

        let ctx = CheckpointContext::new(token, config, stats.clone());

        // First checkpoint won't trigger soft timeout
        ctx.checkpoint().unwrap();
        assert!(!ctx.soft_timeout_triggered());

        // Wait a bit and check again
        thread::sleep(Duration::from_millis(15));
        ctx.checkpoint().unwrap();

        assert!(ctx.soft_timeout_triggered());
        assert_eq!(stats.soft_timeout_warnings.load(Ordering::Relaxed), 1);
    }
}
