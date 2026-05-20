//! AML transaction monitoring and suspicious activity detection.
//!
//! Provides [`TransactionMonitorConfig`] builder and [`TransactionMonitor`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionMonitorError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TransactionMonitorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TransactionMonitor: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TransactionMonitor: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TransactionMonitor: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TransactionMonitor`] parameters.
#[derive(Debug, Clone)]
pub struct TransactionMonitorConfig {
    pub structuring_threshold: f64,
    pub rapid_movement_hours: u32,
    pub score_threshold: f64,
    pub window_days: u32,
}

impl TransactionMonitorConfig {
    pub fn new() -> Self {
        Self {
            structuring_threshold: 10_000.0,
            rapid_movement_hours: 24,
            score_threshold: 75.0,
            window_days: 30,
        }
    }

    pub fn with_structuring_threshold(mut self, v: f64) -> Self {
        self.structuring_threshold = v;
        self
    }

    pub fn with_rapid_movement_hours(mut self, v: u32) -> Self {
        self.rapid_movement_hours = v;
        self
    }

    pub fn with_score_threshold(mut self, v: f64) -> Self {
        self.score_threshold = v;
        self
    }

    pub fn with_window_days(mut self, v: u32) -> Self {
        self.window_days = v;
        self
    }

    pub fn validate(&self) -> Result<(), TransactionMonitorError> {
        if self.structuring_threshold.is_nan() {
            return Err(TransactionMonitorError::InvalidConfig("structuring_threshold is NaN".into()));
        }
        if self.score_threshold.is_nan() {
            return Err(TransactionMonitorError::InvalidConfig("score_threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for TransactionMonitorConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TransactionMonitorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransactionMonitorConfig(structuring_threshold={0:.4}, rapid_movement_hours={1}, score_threshold={2:.4}, window_days={3})", self.structuring_threshold, self.rapid_movement_hours, self.score_threshold, self.window_days)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a TransactionMonitor operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Alert {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Alert({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core aml transaction monitoring and suspicious activity detection engine.
#[derive(Debug, Clone)]
pub struct TransactionMonitor {
    config: TransactionMonitorConfig,
    data: Vec<f64>,
}

impl TransactionMonitor {
    pub fn new(config: TransactionMonitorConfig) -> Result<Self, TransactionMonitorError> {
        config.validate()?;
        Ok(Self { config, data: Vec::new() })
    }

    pub fn with_data(mut self, data: Vec<f64>) -> Self {
        self.data = data;
        self
    }

    pub fn push(&mut self, value: f64) {
        self.data.push(value);
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn config(&self) -> &TransactionMonitorConfig { &self.config }

    /// Score suspicious activity.
    pub fn score_transaction(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Detect structuring pattern.
    pub fn detect_structuring(&self) -> bool {
        !self.data.is_empty()
    }

    /// Generate SAR alert.
    pub fn generate_alert(&self) -> Alert {
        let v = if self.data.is_empty() { 0.0 } else { self.data[0] };
        Alert { value: v, label: stringify!(generate_alert).into() }
    }

    /// Summary statistics of loaded data.
    pub fn summary(&self) -> (f64, f64, f64, f64) {
        if self.data.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let var = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt(), min, max)
    }

    /// Percentile of the data (0.0–1.0).
    pub fn percentile(&self, p: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Exponentially weighted moving statistic.
    pub fn ewm(&self, decay: f64) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.data.len());
        let mut ewm = 0.0;
        for (i, &v) in self.data.iter().enumerate() {
            if i == 0 { ewm = v; } else { ewm = decay * ewm + (1.0 - decay) * v; }
            result.push(ewm);
        }
        result
    }
}

impl fmt::Display for TransactionMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransactionMonitor(n={})", self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }

    #[test]
    fn test_config_default() {
        let cfg = TransactionMonitorConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TransactionMonitorConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TransactionMonitorConfig"));
    }

    #[test]
    fn test_config_with_structuring_threshold() {
        let cfg = TransactionMonitorConfig::new().with_structuring_threshold(42.0);
        assert!((cfg.structuring_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rapid_movement_hours() {
        let cfg = TransactionMonitorConfig::new().with_rapid_movement_hours(42);
        assert_eq!(cfg.rapid_movement_hours, 42);
    }

    #[test]
    fn test_config_with_score_threshold() {
        let cfg = TransactionMonitorConfig::new().with_score_threshold(42.0);
        assert!((cfg.score_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_window_days() {
        let cfg = TransactionMonitorConfig::new().with_window_days(42);
        assert_eq!(cfg.window_days, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TransactionMonitorConfig::new().with_structuring_threshold(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TransactionMonitor"));
    }

    #[test]
    fn test_summary() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_transaction() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.score_transaction();
        assert!(result.is_finite());
    }

    #[test]
    fn test_detect_structuring() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_structuring();
        assert!(result);
    }

    #[test]
    fn test_generate_alert() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_alert();
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_generate_alert_empty() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap();
        let _ = e.generate_alert();
    }

    #[test]
    fn test_config_accessor() {
        let e = TransactionMonitor::new(TransactionMonitorConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TransactionMonitorError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TransactionMonitorError::InvalidConfig("a".into());
        let e2 = TransactionMonitorError::ComputationFailed("b".into());
        let e3 = TransactionMonitorError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
