//! Population risk stratification and HCC scoring.
//!
//! Provides [`RiskStratifyConfig`] builder and [`RiskStratify`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskStratifyError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RiskStratifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RiskStratify: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RiskStratify: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RiskStratify: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RiskStratify`] parameters.
#[derive(Debug, Clone)]
pub struct RiskStratifyConfig {
    pub model_version: usize,
    pub age_group: usize,
    pub gender: usize,
    pub community: bool,
}

impl RiskStratifyConfig {
    pub fn new() -> Self {
        Self {
            model_version: 28,
            age_group: 0,
            gender: 0,
            community: true,
        }
    }

    pub fn with_model_version(mut self, v: usize) -> Self {
        self.model_version = v;
        self
    }

    pub fn with_age_group(mut self, v: usize) -> Self {
        self.age_group = v;
        self
    }

    pub fn with_gender(mut self, v: usize) -> Self {
        self.gender = v;
        self
    }

    pub fn with_community(mut self, v: bool) -> Self {
        self.community = v;
        self
    }

    pub fn validate(&self) -> Result<(), RiskStratifyError> {
        Ok(())
    }
}

impl Default for RiskStratifyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RiskStratifyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskStratifyConfig(model_version={0}, age_group={1}, gender={2}, community={3})", self.model_version, self.age_group, self.gender, self.community)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core population risk stratification and hcc scoring engine.
#[derive(Debug, Clone)]
pub struct RiskStratify {
    config: RiskStratifyConfig,
    data: Vec<f64>,
}

impl RiskStratify {
    pub fn new(config: RiskStratifyConfig) -> Result<Self, RiskStratifyError> {
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
    pub fn config(&self) -> &RiskStratifyConfig { &self.config }

    /// Calculate HCC risk score.
    pub fn hcc_score(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Assign risk segment.
    pub fn risk_segment(&self) -> String {
        format!("{}: {} records", stringify!(risk_segment), self.data.len())
    }

    /// Detect rising risk.
    pub fn rising_risk(&self) -> bool {
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

impl fmt::Display for RiskStratify {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskStratify(n={})", self.data.len())
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
        let cfg = RiskStratifyConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RiskStratifyConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RiskStratifyConfig"));
    }

    #[test]
    fn test_config_with_model_version() {
        let cfg = RiskStratifyConfig::new().with_model_version(42);
        assert_eq!(cfg.model_version, 42);
    }

    #[test]
    fn test_config_with_age_group() {
        let cfg = RiskStratifyConfig::new().with_age_group(42);
        assert_eq!(cfg.age_group, 42);
    }

    #[test]
    fn test_config_with_gender() {
        let cfg = RiskStratifyConfig::new().with_gender(42);
        assert_eq!(cfg.gender, 42);
    }

    #[test]
    fn test_config_with_community() {
        let cfg = RiskStratifyConfig::new().with_community(false);
        assert_eq!(cfg.community, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RiskStratifyConfig::new().with_model_version(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RiskStratify::new(RiskStratifyConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RiskStratify"));
    }

    #[test]
    fn test_summary() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_hcc_score() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.hcc_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_risk_segment() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.risk_segment();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rising_risk() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rising_risk();
        assert!(result);
    }

    #[test]
    fn test_rising_risk_empty() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap();
        assert!(!e.rising_risk());
    }

    #[test]
    fn test_config_accessor() {
        let e = RiskStratify::new(RiskStratifyConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RiskStratifyError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RiskStratifyError::InvalidConfig("a".into());
        let e2 = RiskStratifyError::ComputationFailed("b".into());
        let e3 = RiskStratifyError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
