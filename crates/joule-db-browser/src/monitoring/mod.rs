//! Performance monitoring and observability
//!
//! Tracks metrics, performance, and provides DevTools integration.

pub mod devtools;
pub mod metrics;

pub use devtools::DevToolsIntegration;
pub use metrics::{Metrics, PerformanceMonitor};
