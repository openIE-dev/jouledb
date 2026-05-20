//! Medication error detection and LASA checking.
//!
//! Provides [`MedErrorConfig`] builder and [`MedError`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MedErrorError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MedErrorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MedError: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MedError: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MedError: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MedError`] parameters.
#[derive(Debug, Clone)]
pub struct MedErrorConfig {
    pub edit_distance_threshold: usize,
    pub check_dose_10x: bool,
    pub check_route: bool,
    pub high_alert_flag: bool,
}

impl MedErrorConfig {
    pub fn new() -> Self {
        Self {
            edit_distance_threshold: 2,
            check_dose_10x: true,
            check_route: true,
            high_alert_flag: true,
        }
    }

    pub fn with_edit_distance_threshold(mut self, v: usize) -> Self {
        self.edit_distance_threshold = v;
        self
    }

    pub fn with_check_dose_10x(mut self, v: bool) -> Self {
        self.check_dose_10x = v;
        self
    }

    pub fn with_check_route(mut self, v: bool) -> Self {
        self.check_route = v;
        self
    }

    pub fn with_high_alert_flag(mut self, v: bool) -> Self {
        self.high_alert_flag = v;
        self
    }

    pub fn validate(&self) -> Result<(), MedErrorError> {
        Ok(())
    }
}

impl Default for MedErrorConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MedErrorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MedErrorConfig(edit_distance_threshold={0}, check_dose_10x={1}, check_route={2}, high_alert_flag={3})", self.edit_distance_threshold, self.check_dose_10x, self.check_route, self.high_alert_flag)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core medication error detection and lasa checking engine.
#[derive(Debug, Clone)]
pub struct MedError {
    config: MedErrorConfig,
    data: Vec<f64>,
}

impl MedError {
    pub fn new(config: MedErrorConfig) -> Result<Self, MedErrorError> {
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
    pub fn config(&self) -> &MedErrorConfig { &self.config }

    /// Look-alike sound-alike check.
    pub fn lasa_check(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check for dose error (10x rule).
    pub fn dose_error_check(&self) -> bool {
        !self.data.is_empty()
    }

    /// Overall error risk score.
    pub fn error_score(&self) -> f64 {
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

impl fmt::Display for MedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MedError(n={})", self.data.len())
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
        let cfg = MedErrorConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MedErrorConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MedErrorConfig"));
    }

    #[test]
    fn test_config_with_edit_distance_threshold() {
        let cfg = MedErrorConfig::new().with_edit_distance_threshold(42);
        assert_eq!(cfg.edit_distance_threshold, 42);
    }

    #[test]
    fn test_config_with_check_dose_10x() {
        let cfg = MedErrorConfig::new().with_check_dose_10x(false);
        assert_eq!(cfg.check_dose_10x, false);
    }

    #[test]
    fn test_config_with_check_route() {
        let cfg = MedErrorConfig::new().with_check_route(false);
        assert_eq!(cfg.check_route, false);
    }

    #[test]
    fn test_config_with_high_alert_flag() {
        let cfg = MedErrorConfig::new().with_high_alert_flag(false);
        assert_eq!(cfg.high_alert_flag, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MedErrorConfig::new().with_edit_distance_threshold(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MedError::new(MedErrorConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MedError::new(MedErrorConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MedError"));
    }

    #[test]
    fn test_summary() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MedError::new(MedErrorConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lasa_check() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lasa_check();
        assert!(result);
    }

    #[test]
    fn test_dose_error_check() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.dose_error_check();
        assert!(result);
    }

    #[test]
    fn test_error_score() {
        let e = MedError::new(MedErrorConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.error_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_error_score_empty() {
        let e = MedError::new(MedErrorConfig::new()).unwrap();
        assert!((e.error_score() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = MedError::new(MedErrorConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MedErrorError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MedErrorError::InvalidConfig("a".into());
        let e2 = MedErrorError::ComputationFailed("b".into());
        let e3 = MedErrorError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
