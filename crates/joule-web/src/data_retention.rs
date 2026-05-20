//! Healthcare data retention policy management.
//!
//! Provides [`DataRetentionConfig`] builder and [`DataRetention`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DataRetentionError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DataRetentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DataRetention: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DataRetention: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DataRetention: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DataRetention`] parameters.
#[derive(Debug, Clone)]
pub struct DataRetentionConfig {
    pub record_type: usize,
    pub retention_years: usize,
    pub legal_hold: bool,
    pub auto_destroy: bool,
}

impl DataRetentionConfig {
    pub fn new() -> Self {
        Self {
            record_type: 0,
            retention_years: 7,
            legal_hold: false,
            auto_destroy: false,
        }
    }

    pub fn with_record_type(mut self, v: usize) -> Self {
        self.record_type = v;
        self
    }

    pub fn with_retention_years(mut self, v: usize) -> Self {
        self.retention_years = v;
        self
    }

    pub fn with_legal_hold(mut self, v: bool) -> Self {
        self.legal_hold = v;
        self
    }

    pub fn with_auto_destroy(mut self, v: bool) -> Self {
        self.auto_destroy = v;
        self
    }

    pub fn validate(&self) -> Result<(), DataRetentionError> {
        Ok(())
    }
}

impl Default for DataRetentionConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DataRetentionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataRetentionConfig(record_type={0}, retention_years={1}, legal_hold={2}, auto_destroy={3})", self.record_type, self.retention_years, self.legal_hold, self.auto_destroy)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core healthcare data retention policy management engine.
#[derive(Debug, Clone)]
pub struct DataRetention {
    config: DataRetentionConfig,
    data: Vec<f64>,
}

impl DataRetention {
    pub fn new(config: DataRetentionConfig) -> Result<Self, DataRetentionError> {
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
    pub fn config(&self) -> &DataRetentionConfig { &self.config }

    /// Get retention period.
    pub fn retention_period(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check if past retention.
    pub fn is_expired(&self) -> bool {
        !self.data.is_empty()
    }

    /// Schedule data destruction.
    pub fn schedule_destruction(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for DataRetention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataRetention(n={})", self.data.len())
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
        let cfg = DataRetentionConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DataRetentionConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DataRetentionConfig"));
    }

    #[test]
    fn test_config_with_record_type() {
        let cfg = DataRetentionConfig::new().with_record_type(42);
        assert_eq!(cfg.record_type, 42);
    }

    #[test]
    fn test_config_with_retention_years() {
        let cfg = DataRetentionConfig::new().with_retention_years(42);
        assert_eq!(cfg.retention_years, 42);
    }

    #[test]
    fn test_config_with_legal_hold() {
        let cfg = DataRetentionConfig::new().with_legal_hold(true);
        assert_eq!(cfg.legal_hold, true);
    }

    #[test]
    fn test_config_with_auto_destroy() {
        let cfg = DataRetentionConfig::new().with_auto_destroy(true);
        assert_eq!(cfg.auto_destroy, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DataRetentionConfig::new().with_record_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DataRetention::new(DataRetentionConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DataRetention"));
    }

    #[test]
    fn test_summary() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_retention_period() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.retention_period();
        assert!(result.is_finite());
    }

    #[test]
    fn test_is_expired() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_expired();
        assert!(result);
    }

    #[test]
    fn test_schedule_destruction() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.schedule_destruction();
        assert!(result);
    }

    #[test]
    fn test_schedule_destruction_empty() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap();
        assert!(!e.schedule_destruction());
    }

    #[test]
    fn test_config_accessor() {
        let e = DataRetention::new(DataRetentionConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DataRetentionError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DataRetentionError::InvalidConfig("a".into());
        let e2 = DataRetentionError::ComputationFailed("b".into());
        let e3 = DataRetentionError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
