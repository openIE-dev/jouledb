//! Timeout Utilities
//!
//! Provides configurable timeouts for async operations.

use std::future::Future;
use std::time::Duration;

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

/// Execute an async operation with a timeout (requires `async-tokio` feature)
#[cfg(feature = "async-tokio")]
pub async fn with_timeout<T, E, Fut>(timeout: Duration, future: Fut) -> Result<T, TimeoutError<E>>
where
    Fut: Future<Output = Result<T, E>>,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(TimeoutError::Inner(e)),
        Err(_) => Err(TimeoutError::Timeout),
    }
}

/// Execute an async operation without timeout enforcement.
/// This is a fallback for when `async-tokio` feature is not enabled.
///
/// Note: This does NOT actually enforce a timeout - it just runs the future.
/// Enable the `async-tokio` feature for proper timeout support.
#[cfg(not(feature = "async-tokio"))]
pub async fn with_timeout<T, E, Fut>(_timeout: Duration, future: Fut) -> Result<T, TimeoutError<E>>
where
    Fut: Future<Output = Result<T, E>>,
{
    // Without tokio, we cannot enforce timeouts.
    // This is a passthrough that just runs the future.
    match future.await {
        Ok(value) => Ok(value),
        Err(e) => Err(TimeoutError::Inner(e)),
    }
}

/// Configuration for timeout behavior
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Default operation timeout
    pub default_timeout: Duration,
    /// Timeout for read operations
    pub read_timeout: Duration,
    /// Timeout for write operations
    pub write_timeout: Duration,
    /// Timeout for connection establishment
    pub connect_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
        }
    }
}

impl TimeoutConfig {
    /// Create a new config with conservative defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config for fast operations
    pub fn fast() -> Self {
        Self {
            default_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(2),
            write_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(2),
        }
    }

    /// Create a config for slow/bulk operations
    pub fn slow() -> Self {
        Self {
            default_timeout: Duration::from_secs(300),
            read_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(30),
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

    /// Builder: set connect timeout
    pub fn with_connect(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
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
