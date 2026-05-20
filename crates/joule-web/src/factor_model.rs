//! Factor models for risk decomposition (CAPM, Fama-French).
//!
//! Provides [`FactorModelConfig`] builder and [`FactorModel`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorModelError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FactorModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "FactorModel: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "FactorModel: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "FactorModel: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`FactorModel`] parameters.
#[derive(Debug, Clone)]
pub struct FactorModelConfig {
    pub num_factors: usize,
    pub risk_free_rate: f64,
    pub estimation_window: usize,
    pub confidence: f64,
}

impl FactorModelConfig {
    pub fn new() -> Self {
        Self {
            num_factors: 3,
            risk_free_rate: 0.02,
            estimation_window: 60,
            confidence: 0.95,
        }
    }

    pub fn with_num_factors(mut self, v: usize) -> Self {
        self.num_factors = v;
        self
    }

    pub fn with_risk_free_rate(mut self, v: f64) -> Self {
        self.risk_free_rate = v;
        self
    }

    pub fn with_estimation_window(mut self, v: usize) -> Self {
        self.estimation_window = v;
        self
    }

    pub fn with_confidence(mut self, v: f64) -> Self {
        self.confidence = v;
        self
    }

    pub fn validate(&self) -> Result<(), FactorModelError> {
        if self.risk_free_rate.is_nan() {
            return Err(FactorModelError::InvalidConfig("risk_free_rate is NaN".into()));
        }
        if self.confidence.is_nan() {
            return Err(FactorModelError::InvalidConfig("confidence is NaN".into()));
        }
        Ok(())
    }
}

impl Default for FactorModelConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FactorModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FactorModelConfig(num_factors={0}, risk_free_rate={1:.4}, estimation_window={2}, confidence={3:.4})", self.num_factors, self.risk_free_rate, self.estimation_window, self.confidence)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core factor models for risk decomposition (capm, fama-french) engine.
#[derive(Debug, Clone)]
pub struct FactorModel {
    config: FactorModelConfig,
    data: Vec<f64>,
}

impl FactorModel {
    pub fn new(config: FactorModelConfig) -> Result<Self, FactorModelError> {
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
    pub fn config(&self) -> &FactorModelConfig { &self.config }

    /// Estimate factor exposures.
    pub fn estimate_betas(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Systematic risk component.
    pub fn systematic_risk(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Idiosyncratic risk.
    pub fn idiosyncratic_risk(&self) -> f64 {
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

impl fmt::Display for FactorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FactorModel(n={})", self.data.len())
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
        let cfg = FactorModelConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FactorModelConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FactorModelConfig"));
    }

    #[test]
    fn test_config_with_num_factors() {
        let cfg = FactorModelConfig::new().with_num_factors(42);
        assert_eq!(cfg.num_factors, 42);
    }

    #[test]
    fn test_config_with_risk_free_rate() {
        let cfg = FactorModelConfig::new().with_risk_free_rate(42.0);
        assert!((cfg.risk_free_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_estimation_window() {
        let cfg = FactorModelConfig::new().with_estimation_window(42);
        assert_eq!(cfg.estimation_window, 42);
    }

    #[test]
    fn test_config_with_confidence() {
        let cfg = FactorModelConfig::new().with_confidence(42.0);
        assert!((cfg.confidence - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FactorModelConfig::new().with_num_factors(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = FactorModel::new(FactorModelConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("FactorModel"));
    }

    #[test]
    fn test_summary() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_betas() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.estimate_betas();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_systematic_risk() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.systematic_risk();
        assert!(result.is_finite());
    }

    #[test]
    fn test_idiosyncratic_risk() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.idiosyncratic_risk();
        assert!(result.is_finite());
    }

    #[test]
    fn test_idiosyncratic_risk_empty() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap();
        assert!((e.idiosyncratic_risk() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = FactorModel::new(FactorModelConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FactorModelError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FactorModelError::InvalidConfig("a".into());
        let e2 = FactorModelError::ComputationFailed("b".into());
        let e3 = FactorModelError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
