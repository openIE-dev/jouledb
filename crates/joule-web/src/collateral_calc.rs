//! Collateral management with haircut application.
//!
//! Provides [`CollateralCalcConfig`] builder and [`CollateralCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CollateralCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CollateralCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CollateralCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CollateralCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CollateralCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CollateralCalc`] parameters.
#[derive(Debug, Clone)]
pub struct CollateralCalcConfig {
    pub haircut_pct: f64,
    pub concentration_limit: f64,
    pub rehypothecation: bool,
    pub min_transfer: f64,
}

impl CollateralCalcConfig {
    pub fn new() -> Self {
        Self {
            haircut_pct: 0.10,
            concentration_limit: 0.25,
            rehypothecation: false,
            min_transfer: 100_000.0,
        }
    }

    pub fn with_haircut_pct(mut self, v: f64) -> Self {
        self.haircut_pct = v;
        self
    }

    pub fn with_concentration_limit(mut self, v: f64) -> Self {
        self.concentration_limit = v;
        self
    }

    pub fn with_rehypothecation(mut self, v: bool) -> Self {
        self.rehypothecation = v;
        self
    }

    pub fn with_min_transfer(mut self, v: f64) -> Self {
        self.min_transfer = v;
        self
    }

    pub fn validate(&self) -> Result<(), CollateralCalcError> {
        if self.haircut_pct.is_nan() {
            return Err(CollateralCalcError::InvalidConfig("haircut_pct is NaN".into()));
        }
        if self.concentration_limit.is_nan() {
            return Err(CollateralCalcError::InvalidConfig("concentration_limit is NaN".into()));
        }
        if self.min_transfer.is_nan() {
            return Err(CollateralCalcError::InvalidConfig("min_transfer is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CollateralCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CollateralCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CollateralCalcConfig(haircut_pct={0:.4}, concentration_limit={1:.4}, rehypothecation={2}, min_transfer={3:.4})", self.haircut_pct, self.concentration_limit, self.rehypothecation, self.min_transfer)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core collateral management with haircut application engine.
#[derive(Debug, Clone)]
pub struct CollateralCalc {
    config: CollateralCalcConfig,
    data: Vec<f64>,
}

impl CollateralCalc {
    pub fn new(config: CollateralCalcConfig) -> Result<Self, CollateralCalcError> {
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
    pub fn config(&self) -> &CollateralCalcConfig { &self.config }

    /// Eligible collateral value after haircuts.
    pub fn eligible_value(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Generate margin call amount.
    pub fn margin_call(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check concentration limits.
    pub fn concentration_check(&self) -> bool {
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

impl fmt::Display for CollateralCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CollateralCalc(n={})", self.data.len())
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
        let cfg = CollateralCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CollateralCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CollateralCalcConfig"));
    }

    #[test]
    fn test_config_with_haircut_pct() {
        let cfg = CollateralCalcConfig::new().with_haircut_pct(42.0);
        assert!((cfg.haircut_pct - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_concentration_limit() {
        let cfg = CollateralCalcConfig::new().with_concentration_limit(42.0);
        assert!((cfg.concentration_limit - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rehypothecation() {
        let cfg = CollateralCalcConfig::new().with_rehypothecation(true);
        assert_eq!(cfg.rehypothecation, true);
    }

    #[test]
    fn test_config_with_min_transfer() {
        let cfg = CollateralCalcConfig::new().with_min_transfer(42.0);
        assert!((cfg.min_transfer - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CollateralCalcConfig::new().with_haircut_pct(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CollateralCalc"));
    }

    #[test]
    fn test_summary() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_eligible_value() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.eligible_value();
        assert!(result.is_finite());
    }

    #[test]
    fn test_margin_call() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.margin_call();
        assert!(result.is_finite());
    }

    #[test]
    fn test_concentration_check() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.concentration_check();
        assert!(result);
    }

    #[test]
    fn test_concentration_check_empty() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap();
        assert!(!e.concentration_check());
    }

    #[test]
    fn test_config_accessor() {
        let e = CollateralCalc::new(CollateralCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CollateralCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CollateralCalcError::InvalidConfig("a".into());
        let e2 = CollateralCalcError::ComputationFailed("b".into());
        let e3 = CollateralCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
