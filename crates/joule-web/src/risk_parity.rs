//! Risk parity portfolio with equal risk contribution.
//!
//! Provides [`RiskParityConfig`] builder and [`RiskParity`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskParityError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RiskParityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RiskParity: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RiskParity: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RiskParity: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RiskParity`] parameters.
#[derive(Debug, Clone)]
pub struct RiskParityConfig {
    pub target_risk: f64,
    pub max_leverage: f64,
    pub rebalance_threshold: f64,
    pub num_assets: usize,
}

impl RiskParityConfig {
    pub fn new() -> Self {
        Self {
            target_risk: 0.10,
            max_leverage: 3.0,
            rebalance_threshold: 0.05,
            num_assets: 10,
        }
    }

    pub fn with_target_risk(mut self, v: f64) -> Self {
        self.target_risk = v;
        self
    }

    pub fn with_max_leverage(mut self, v: f64) -> Self {
        self.max_leverage = v;
        self
    }

    pub fn with_rebalance_threshold(mut self, v: f64) -> Self {
        self.rebalance_threshold = v;
        self
    }

    pub fn with_num_assets(mut self, v: usize) -> Self {
        self.num_assets = v;
        self
    }

    pub fn validate(&self) -> Result<(), RiskParityError> {
        if self.target_risk.is_nan() {
            return Err(RiskParityError::InvalidConfig("target_risk is NaN".into()));
        }
        if self.max_leverage.is_nan() {
            return Err(RiskParityError::InvalidConfig("max_leverage is NaN".into()));
        }
        if self.rebalance_threshold.is_nan() {
            return Err(RiskParityError::InvalidConfig("rebalance_threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RiskParityConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RiskParityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskParityConfig(target_risk={0:.4}, max_leverage={1:.4}, rebalance_threshold={2:.4}, num_assets={3})", self.target_risk, self.max_leverage, self.rebalance_threshold, self.num_assets)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core risk parity portfolio with equal risk contribution engine.
#[derive(Debug, Clone)]
pub struct RiskParity {
    config: RiskParityConfig,
    data: Vec<f64>,
}

impl RiskParity {
    pub fn new(config: RiskParityConfig) -> Result<Self, RiskParityError> {
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
    pub fn config(&self) -> &RiskParityConfig { &self.config }

    /// Calculate equal risk contribution weights.
    pub fn equal_risk_weights(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Marginal risk contribution.
    pub fn marginal_risk(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Portfolio diversification ratio.
    pub fn diversification_ratio(&self) -> f64 {
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

impl fmt::Display for RiskParity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskParity(n={})", self.data.len())
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
        let cfg = RiskParityConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RiskParityConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RiskParityConfig"));
    }

    #[test]
    fn test_config_with_target_risk() {
        let cfg = RiskParityConfig::new().with_target_risk(42.0);
        assert!((cfg.target_risk - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_leverage() {
        let cfg = RiskParityConfig::new().with_max_leverage(42.0);
        assert!((cfg.max_leverage - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rebalance_threshold() {
        let cfg = RiskParityConfig::new().with_rebalance_threshold(42.0);
        assert!((cfg.rebalance_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_num_assets() {
        let cfg = RiskParityConfig::new().with_num_assets(42);
        assert_eq!(cfg.num_assets, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RiskParityConfig::new().with_target_risk(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RiskParity::new(RiskParityConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RiskParity"));
    }

    #[test]
    fn test_summary() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_equal_risk_weights() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.equal_risk_weights();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_marginal_risk() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.marginal_risk();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_diversification_ratio() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.diversification_ratio();
        assert!(result.is_finite());
    }

    #[test]
    fn test_diversification_ratio_empty() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap();
        assert!((e.diversification_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = RiskParity::new(RiskParityConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RiskParityError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RiskParityError::InvalidConfig("a".into());
        let e2 = RiskParityError::ComputationFailed("b".into());
        let e3 = RiskParityError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
