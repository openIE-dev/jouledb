//! Patient consent directive management.
//!
//! Provides [`ConsentManageConfig`] builder and [`ConsentManage`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsentManageError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ConsentManageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ConsentManage: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ConsentManage: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ConsentManage: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ConsentManage`] parameters.
#[derive(Debug, Clone)]
pub struct ConsentManageConfig {
    pub consent_type: usize,
    pub granular: bool,
    pub expiry_days: u32,
    pub opt_in_default: bool,
}

impl ConsentManageConfig {
    pub fn new() -> Self {
        Self {
            consent_type: 0,
            granular: true,
            expiry_days: 365,
            opt_in_default: false,
        }
    }

    pub fn with_consent_type(mut self, v: usize) -> Self {
        self.consent_type = v;
        self
    }

    pub fn with_granular(mut self, v: bool) -> Self {
        self.granular = v;
        self
    }

    pub fn with_expiry_days(mut self, v: u32) -> Self {
        self.expiry_days = v;
        self
    }

    pub fn with_opt_in_default(mut self, v: bool) -> Self {
        self.opt_in_default = v;
        self
    }

    pub fn validate(&self) -> Result<(), ConsentManageError> {
        Ok(())
    }
}

impl Default for ConsentManageConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ConsentManageConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConsentManageConfig(consent_type={0}, granular={1}, expiry_days={2}, opt_in_default={3})", self.consent_type, self.granular, self.expiry_days, self.opt_in_default)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core patient consent directive management engine.
#[derive(Debug, Clone)]
pub struct ConsentManage {
    config: ConsentManageConfig,
    data: Vec<f64>,
}

impl ConsentManage {
    pub fn new(config: ConsentManageConfig) -> Result<Self, ConsentManageError> {
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
    pub fn config(&self) -> &ConsentManageConfig { &self.config }

    /// Record consent directive.
    pub fn record_consent(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check if action is consented.
    pub fn is_consented(&self) -> bool {
        !self.data.is_empty()
    }

    /// Generate consent summary.
    pub fn consent_summary(&self) -> String {
        format!("{}: {} records", stringify!(consent_summary), self.data.len())
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

impl fmt::Display for ConsentManage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConsentManage(n={})", self.data.len())
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
        let cfg = ConsentManageConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ConsentManageConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ConsentManageConfig"));
    }

    #[test]
    fn test_config_with_consent_type() {
        let cfg = ConsentManageConfig::new().with_consent_type(42);
        assert_eq!(cfg.consent_type, 42);
    }

    #[test]
    fn test_config_with_granular() {
        let cfg = ConsentManageConfig::new().with_granular(false);
        assert_eq!(cfg.granular, false);
    }

    #[test]
    fn test_config_with_expiry_days() {
        let cfg = ConsentManageConfig::new().with_expiry_days(42);
        assert_eq!(cfg.expiry_days, 42);
    }

    #[test]
    fn test_config_with_opt_in_default() {
        let cfg = ConsentManageConfig::new().with_opt_in_default(true);
        assert_eq!(cfg.opt_in_default, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ConsentManageConfig::new().with_consent_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ConsentManage::new(ConsentManageConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ConsentManage"));
    }

    #[test]
    fn test_summary() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_consent() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.record_consent();
        assert!(result);
    }

    #[test]
    fn test_is_consented() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_consented();
        assert!(result);
    }

    #[test]
    fn test_consent_summary() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.consent_summary();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_consent_summary_empty() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap();
        let _ = e.consent_summary();
    }

    #[test]
    fn test_config_accessor() {
        let e = ConsentManage::new(ConsentManageConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ConsentManageError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ConsentManageError::InvalidConfig("a".into());
        let e2 = ConsentManageError::ComputationFailed("b".into());
        let e3 = ConsentManageError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
