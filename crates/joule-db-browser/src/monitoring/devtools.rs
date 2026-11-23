//! Browser DevTools integration
//!
//! Provides performance metrics and debugging information to browser DevTools.

use crate::monitoring::PerformanceMonitor;
use std::sync::Arc;

/// DevTools integration
pub struct DevToolsIntegration {
    monitor: Arc<PerformanceMonitor>,
}

impl DevToolsIntegration {
    /// Create new DevTools integration
    pub fn new(monitor: Arc<PerformanceMonitor>) -> Self {
        Self { monitor }
    }

    /// Export metrics to console (for DevTools)
    #[cfg(target_arch = "wasm32")]
    pub fn export_metrics(&self) {
        use wasm_bindgen::prelude::*;
        use web_sys::console;

        let metrics = self.monitor.get_metrics();
        let avg_time = self.monitor.average_query_time();

        let msg = format!(
            "joule-db Metrics:\n\
             Total Queries: {}\n\
             Total Errors: {}\n\
             Error Rate: {:.2}%\n\
             GPU Utilization: {:.1}%\n\
             Memory Usage: {} MB\n\
             Avg Query Time: {:?}",
            metrics.total_queries,
            metrics.total_errors,
            metrics.error_rate * 100.0,
            metrics.gpu_utilization * 100.0,
            metrics.memory_usage / 1024 / 1024,
            avg_time.unwrap_or_default(),
        );

        console::log_1(&JsValue::from_str(&msg));
    }

    /// Export metrics (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_metrics(&self) {
        let metrics = self.monitor.get_metrics();
        println!("joule-db Metrics: {:?}", metrics);
    }
}
