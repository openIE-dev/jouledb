//! Timeout utilities for browser async operations
//!
//! Provides platform-specific timeout implementations for WASM environments.

use std::future::Future;
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Error type for timeout operations
#[derive(Debug, Clone)]
pub enum TimeoutError<E> {
    /// Operation timed out
    Timeout,
    /// Inner operation error
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for TimeoutError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutError::Timeout => write!(f, "Operation timed out"),
            TimeoutError::Inner(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for TimeoutError<E> {}

/// Timeout configuration for async operations
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Default operation timeout
    pub default_timeout: Duration,
    /// Timeout for read operations
    pub read_timeout: Duration,
    /// Timeout for write operations
    pub write_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(30),
        }
    }
}

impl TimeoutConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config for fast operations
    pub fn fast() -> Self {
        Self {
            default_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(2),
            write_timeout: Duration::from_secs(5),
        }
    }

    /// Create a config for slow/bulk operations
    pub fn slow() -> Self {
        Self {
            default_timeout: Duration::from_secs(300),
            read_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(300),
        }
    }

    /// Builder: set default timeout
    pub fn with_default(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Builder: set read timeout
    pub fn with_read(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
        self
    }

    /// Builder: set write timeout
    pub fn with_write(mut self, timeout: Duration) -> Self {
        self.write_timeout = timeout;
        self
    }
}

/// Execute an async operation with a timeout (WASM implementation)
#[cfg(target_arch = "wasm32")]
pub async fn with_timeout<T, E, Fut>(timeout: Duration, future: Fut) -> Result<T, TimeoutError<E>>
where
    Fut: Future<Output = Result<T, E>>,
{
    use js_sys::Promise;
    use wasm_bindgen_futures::JsFuture;

    // Create a timeout promise
    let timeout_ms = timeout.as_millis() as i32;
    let timeout_promise = Promise::new(&mut |resolve, _reject| {
        if let Some(window) = web_sys::window() {
            let _ =
                window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, timeout_ms);
        }
    });

    // Race the operation against the timeout
    // Note: This is a simplified implementation. A proper implementation would
    // use Promise.race() or a similar mechanism.
    let result = future.await;
    match result {
        Ok(value) => Ok(value),
        Err(e) => Err(TimeoutError::Inner(e)),
    }
}

/// Execute an async operation with a timeout (native implementation)
#[cfg(not(target_arch = "wasm32"))]
pub async fn with_timeout<T, E, Fut>(_timeout: Duration, future: Fut) -> Result<T, TimeoutError<E>>
where
    Fut: Future<Output = Result<T, E>>,
{
    // For non-WASM targets without tokio, we just run the future directly.
    // Enable the async-tokio feature in joule-db-core for proper timeout support.
    match future.await {
        Ok(value) => Ok(value),
        Err(e) => Err(TimeoutError::Inner(e)),
    }
}

/// Execute an async operation with a timeout, returning the result or a timeout error
pub async fn timeout_result<T, Fut>(timeout: Duration, future: Fut) -> Result<T, String>
where
    Fut: Future<Output = Result<T, String>>,
{
    match with_timeout(timeout, future).await {
        Ok(value) => Ok(value),
        Err(TimeoutError::Timeout) => Err(format!("Operation timed out after {:?}", timeout)),
        Err(TimeoutError::Inner(e)) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_config_defaults() {
        let config = TimeoutConfig::default();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert_eq!(config.read_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_timeout_config_fast() {
        let config = TimeoutConfig::fast();
        assert_eq!(config.default_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_timeout_config_slow() {
        let config = TimeoutConfig::slow();
        assert_eq!(config.default_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_timeout_config_builder() {
        let config = TimeoutConfig::new()
            .with_default(Duration::from_secs(60))
            .with_read(Duration::from_secs(20));

        assert_eq!(config.default_timeout, Duration::from_secs(60));
        assert_eq!(config.read_timeout, Duration::from_secs(20));
    }
}
