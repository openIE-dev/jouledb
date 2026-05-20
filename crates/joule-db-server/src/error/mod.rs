//! Error Handling and Recovery Module
//!
//! Provides comprehensive error handling, recovery strategies, and resilience patterns.

pub mod recovery;

pub use recovery::{
    CircuitBreaker, CircuitBreakerConfig, CircuitState, ErrorRecoveryManager, RecoveryStrategy,
    RetryConfig, RetryExecutor,
};
