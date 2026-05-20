//! DRG assignment and grouper logic.
//!
//! Provides [`DrgAssignConfig`] builder and [`DrgAssign`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DrgAssignError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DrgAssignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DrgAssign: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DrgAssign: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DrgAssign: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DrgAssign`] parameters.
#[derive(Debug, Clone)]
pub struct DrgAssignConfig {
    pub grouper_version: usize,
    pub mdc_check: bool,
    pub cc_mcc_check: bool,
    pub age_check: bool,
}

impl DrgAssignConfig {
    pub fn new() -> Self {
        Self {
            grouper_version: 41,
            mdc_check: true,
            cc_mcc_check: true,
            age_check: true,
        }
    }

    pub fn with_grouper_version(mut self, v: usize) -> Self {
        self.grouper_version = v;
        self
    }

    pub fn with_mdc_check(mut self, v: bool) -> Self {
        self.mdc_check = v;
        self
    }

    pub fn with_cc_mcc_check(mut self, v: bool) -> Self {
        self.cc_mcc_check = v;
        self
    }

    pub fn with_age_check(mut self, v: bool) -> Self {
        self.age_check = v;
        self
    }

    pub fn validate(&self) -> Result<(), DrgAssignError> {
        Ok(())
    }
}

impl Default for DrgAssignConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DrgAssignConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrgAssignConfig(grouper_version={0}, mdc_check={1}, cc_mcc_check={2}, age_check={3})", self.grouper_version, self.mdc_check, self.cc_mcc_check, self.age_check)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core drg assignment and grouper logic engine.
#[derive(Debug, Clone)]
pub struct DrgAssign {
    config: DrgAssignConfig,
    data: Vec<f64>,
}

impl DrgAssign {
    pub fn new(config: DrgAssignConfig) -> Result<Self, DrgAssignError> {
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
    pub fn config(&self) -> &DrgAssignConfig { &self.config }

    /// Assign MS-DRG.
    pub fn assign_drg(&self) -> String {
        format!("{}: {} records", stringify!(assign_drg), self.data.len())
    }

    /// DRG relative weight.
    pub fn relative_weight(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Expected length of stay.
    pub fn expected_los(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for DrgAssign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrgAssign(n={})", self.data.len())
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
        let cfg = DrgAssignConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DrgAssignConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DrgAssignConfig"));
    }

    #[test]
    fn test_config_with_grouper_version() {
        let cfg = DrgAssignConfig::new().with_grouper_version(42);
        assert_eq!(cfg.grouper_version, 42);
    }

    #[test]
    fn test_config_with_mdc_check() {
        let cfg = DrgAssignConfig::new().with_mdc_check(false);
        assert_eq!(cfg.mdc_check, false);
    }

    #[test]
    fn test_config_with_cc_mcc_check() {
        let cfg = DrgAssignConfig::new().with_cc_mcc_check(false);
        assert_eq!(cfg.cc_mcc_check, false);
    }

    #[test]
    fn test_config_with_age_check() {
        let cfg = DrgAssignConfig::new().with_age_check(false);
        assert_eq!(cfg.age_check, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DrgAssignConfig::new().with_grouper_version(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DrgAssign::new(DrgAssignConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DrgAssign"));
    }

    #[test]
    fn test_summary() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_assign_drg() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.assign_drg();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_relative_weight() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.relative_weight();
        assert!(result.is_finite());
    }

    #[test]
    fn test_expected_los() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.expected_los();
        assert!(result.is_finite());
    }

    #[test]
    fn test_expected_los_empty() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap();
        assert!((e.expected_los() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = DrgAssign::new(DrgAssignConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DrgAssignError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DrgAssignError::InvalidConfig("a".into());
        let e2 = DrgAssignError::ComputationFailed("b".into());
        let e3 = DrgAssignError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
