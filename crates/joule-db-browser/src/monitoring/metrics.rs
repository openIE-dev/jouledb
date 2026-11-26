//! Performance metrics collection

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Performance metrics
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Query execution times
    pub query_times: Vec<Duration>,
    /// GPU utilization (0.0 - 1.0)
    pub gpu_utilization: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Error rate (errors per second)
    pub error_rate: f32,
    /// Total queries executed
    pub total_queries: u64,
    /// Total errors
    pub total_errors: u64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            query_times: Vec::new(),
            gpu_utilization: 0.0,
            memory_usage: 0,
            error_rate: 0.0,
            total_queries: 0,
            total_errors: 0,
        }
    }
}

/// Performance monitor
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<Metrics>>,
    query_time_history: Arc<RwLock<VecDeque<Duration>>>,
    max_history: usize,
}

impl PerformanceMonitor {
    /// Create new monitor
    pub fn new(max_history: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Metrics::default())),
            query_time_history: Arc::new(RwLock::new(VecDeque::new())),
            max_history,
        }
    }

    /// Record query execution time
    pub fn record_query_time(&self, duration: Duration) {
        let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
        metrics.total_queries += 1;
        metrics.query_times.push(duration);

        let mut history = self
            .query_time_history
            .write()
            .unwrap_or_else(|e| e.into_inner());
        history.push_back(duration);
        if history.len() > self.max_history {
            history.pop_front();
        }
    }

    /// Record error
    pub fn record_error(&self) {
        let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
        metrics.total_errors += 1;
        // Calculate error rate (simplified)
        if metrics.total_queries > 0 {
            metrics.error_rate = metrics.total_errors as f32 / metrics.total_queries as f32;
        }
    }

    /// Update GPU utilization
    pub fn update_gpu_utilization(&self, utilization: f32) {
        let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
        metrics.gpu_utilization = utilization;
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: u64) {
        let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
        metrics.memory_usage = bytes;
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> Metrics {
        self.metrics
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get average query time
    pub fn average_query_time(&self) -> Option<Duration> {
        let history = self
            .query_time_history
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if history.is_empty() {
            return None;
        }

        let sum: Duration = history.iter().sum();
        Some(sum / history.len() as u32)
    }
}
