//! Exposure calculation for counterparty credit risk.
//!
//! Provides [`ExposureCalcConfig`] builder and [`ExposureCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ExposureCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ExposureCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ExposureCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ExposureCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ExposureCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ExposureCalc`] parameters.
#[derive(Debug, Clone)]
pub struct ExposureCalcConfig {
    pub notional: f64,
    pub maturity_years: f64,
    pub recovery_rate: f64,
    pub netting: bool,
}

impl ExposureCalcConfig {
    pub fn new() -> Self {
        Self {
            notional: 1_000_000.0,
            maturity_years: 5.0,
            recovery_rate: 0.4,
            netting: true,
        }
    }

    pub fn with_notional(mut self, v: f64) -> Self {
        self.notional = v;
        self
    }

    pub fn with_maturity_years(mut self, v: f64) -> Self {
        self.maturity_years = v;
        self
    }

    pub fn with_recovery_rate(mut self, v: f64) -> Self {
        self.recovery_rate = v;
        self
    }

    pub fn with_netting(mut self, v: bool) -> Self {
        self.netting = v;
        self
    }

    pub fn validate(&self) -> Result<(), ExposureCalcError> {
        if self.notional.is_nan() {
            return Err(ExposureCalcError::InvalidConfig("notional is NaN".into()));
        }
        if self.maturity_years.is_nan() {
            return Err(ExposureCalcError::InvalidConfig("maturity_years is NaN".into()));
        }
        if self.recovery_rate.is_nan() {
            return Err(ExposureCalcError::InvalidConfig("recovery_rate is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ExposureCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ExposureCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExposureCalcConfig(notional={0:.4}, maturity_years={1:.4}, recovery_rate={2:.4}, netting={3})", self.notional, self.maturity_years, self.recovery_rate, self.netting)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core exposure calculation for counterparty credit risk engine.
#[derive(Debug, Clone)]
pub struct ExposureCalc {
    config: ExposureCalcConfig,
    data: Vec<f64>,
}

impl ExposureCalc {
    pub fn new(config: ExposureCalcConfig) -> Result<Self, ExposureCalcError> {
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
    pub fn config(&self) -> &ExposureCalcConfig { &self.config }

    /// Calculate current exposure.
    pub fn current_exposure(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Potential future exposure.
    pub fn potential_future(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Expected positive exposure.
    pub fn expected_positive(&self) -> f64 {
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

impl fmt::Display for ExposureCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExposureCalc(n={})", self.data.len())
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
        let cfg = ExposureCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ExposureCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ExposureCalcConfig"));
    }

    #[test]
    fn test_config_with_notional() {
        let cfg = ExposureCalcConfig::new().with_notional(42.0);
        assert!((cfg.notional - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_maturity_years() {
        let cfg = ExposureCalcConfig::new().with_maturity_years(42.0);
        assert!((cfg.maturity_years - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_recovery_rate() {
        let cfg = ExposureCalcConfig::new().with_recovery_rate(42.0);
        assert!((cfg.recovery_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_netting() {
        let cfg = ExposureCalcConfig::new().with_netting(false);
        assert_eq!(cfg.netting, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ExposureCalcConfig::new().with_notional(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ExposureCalc"));
    }

    #[test]
    fn test_summary() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_current_exposure() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.current_exposure();
        assert!(result.is_finite());
    }

    #[test]
    fn test_potential_future() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.potential_future();
        assert!(result.is_finite());
    }

    #[test]
    fn test_expected_positive() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.expected_positive();
        assert!(result.is_finite());
    }

    #[test]
    fn test_expected_positive_empty() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap();
        assert!((e.expected_positive() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = ExposureCalc::new(ExposureCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ExposureCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ExposureCalcError::InvalidConfig("a".into());
        let e2 = ExposureCalcError::ComputationFailed("b".into());
        let e3 = ExposureCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
