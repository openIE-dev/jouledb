//! Error recovery and resilience patterns
//!
//! Provides retry logic, circuit breakers, graceful degradation, and timeouts.

pub mod circuit_breaker;
pub mod fallback;
pub mod retry;
pub mod timeout;

pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use fallback::GracefulDegradation;
pub use retry::RetryPolicy;
pub use timeout::{TimeoutConfig, TimeoutError, timeout_result, with_timeout};
