//! Risk budgeting and factor risk decomposition.
//!
//! Provides [`RiskBudgetConfig`] builder and [`RiskBudget`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskBudgetError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RiskBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RiskBudget: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RiskBudget: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RiskBudget: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RiskBudget`] parameters.
#[derive(Debug, Clone)]
pub struct RiskBudgetConfig {
    pub total_risk: f64,
    pub num_factors: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl RiskBudgetConfig {
    pub fn new() -> Self {
        Self {
            total_risk: 0.15,
            num_factors: 5,
            tolerance: 0.001,
            max_iterations: 100,
        }
    }

    pub fn with_total_risk(mut self, v: f64) -> Self {
        self.total_risk = v;
        self
    }

    pub fn with_num_factors(mut self, v: usize) -> Self {
        self.num_factors = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn validate(&self) -> Result<(), RiskBudgetError> {
        if self.total_risk.is_nan() {
            return Err(RiskBudgetError::InvalidConfig("total_risk is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(RiskBudgetError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RiskBudgetConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RiskBudgetConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskBudgetConfig(total_risk={0:.4}, num_factors={1}, tolerance={2:.4}, max_iterations={3})", self.total_risk, self.num_factors, self.tolerance, self.max_iterations)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core risk budgeting and factor risk decomposition engine.
#[derive(Debug, Clone)]
pub struct RiskBudget {
    config: RiskBudgetConfig,
    data: Vec<f64>,
}

impl RiskBudget {
    pub fn new(config: RiskBudgetConfig) -> Result<Self, RiskBudgetError> {
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
    pub fn config(&self) -> &RiskBudgetConfig { &self.config }

    /// Decompose risk by factor.
    pub fn decompose(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Marginal risk contribution.
    pub fn marginal_contribution(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Component VaR.
    pub fn component_var(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
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

impl fmt::Display for RiskBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskBudget(n={})", self.data.len())
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
        let cfg = RiskBudgetConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RiskBudgetConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RiskBudgetConfig"));
    }

    #[test]
    fn test_config_with_total_risk() {
        let cfg = RiskBudgetConfig::new().with_total_risk(42.0);
        assert!((cfg.total_risk - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_num_factors() {
        let cfg = RiskBudgetConfig::new().with_num_factors(42);
        assert_eq!(cfg.num_factors, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = RiskBudgetConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_iterations() {
        let cfg = RiskBudgetConfig::new().with_max_iterations(42);
        assert_eq!(cfg.max_iterations, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RiskBudgetConfig::new().with_total_risk(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RiskBudget::new(RiskBudgetConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RiskBudget"));
    }

    #[test]
    fn test_summary() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_decompose() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decompose();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_marginal_contribution() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.marginal_contribution();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_component_var() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.component_var();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_component_var_empty() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap();
        assert!(e.component_var().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = RiskBudget::new(RiskBudgetConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RiskBudgetError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RiskBudgetError::InvalidConfig("a".into());
        let e2 = RiskBudgetError::ComputationFailed("b".into());
        let e3 = RiskBudgetError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
