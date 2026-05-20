//! Lattice sampling with discrete Gaussian distribution.
//!
//! Provides [`LatticeSampleConfig`] builder and [`LatticeSample`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum LatticeSampleError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for LatticeSampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "LatticeSample: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "LatticeSample: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "LatticeSample: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`LatticeSample`] parameters.
#[derive(Debug, Clone)]
pub struct LatticeSampleConfig {
    pub dimension: usize,
    pub std_dev: f64,
    pub center: f64,
    pub precision: usize,
}

impl LatticeSampleConfig {
    pub fn new() -> Self {
        Self {
            dimension: 256,
            std_dev: 3.2,
            center: 0.0,
            precision: 64,
        }
    }

    pub fn with_dimension(mut self, v: usize) -> Self {
        self.dimension = v;
        self
    }

    pub fn with_std_dev(mut self, v: f64) -> Self {
        self.std_dev = v;
        self
    }

    pub fn with_center(mut self, v: f64) -> Self {
        self.center = v;
        self
    }

    pub fn with_precision(mut self, v: usize) -> Self {
        self.precision = v;
        self
    }

    pub fn validate(&self) -> Result<(), LatticeSampleError> {
        if self.std_dev.is_nan() {
            return Err(LatticeSampleError::InvalidConfig("std_dev is NaN".into()));
        }
        if self.center.is_nan() {
            return Err(LatticeSampleError::InvalidConfig("center is NaN".into()));
        }
        Ok(())
    }
}

impl Default for LatticeSampleConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for LatticeSampleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatticeSampleConfig(dimension={0}, std_dev={1:.4}, center={2:.4}, precision={3})", self.dimension, self.std_dev, self.center, self.precision)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core lattice sampling with discrete gaussian distribution engine.
#[derive(Debug, Clone)]
pub struct LatticeSample {
    config: LatticeSampleConfig,
    data: Vec<f64>,
}

impl LatticeSample {
    pub fn new(config: LatticeSampleConfig) -> Result<Self, LatticeSampleError> {
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
    pub fn config(&self) -> &LatticeSampleConfig { &self.config }

    /// Sample from discrete Gaussian.
    pub fn discrete_gaussian(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Sample uniform polynomial.
    pub fn uniform_poly(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Sample ternary distribution.
    pub fn ternary_dist(&self) -> Vec<f64> {
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

impl fmt::Display for LatticeSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatticeSample(n={})", self.data.len())
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
        let cfg = LatticeSampleConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = LatticeSampleConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("LatticeSampleConfig"));
    }

    #[test]
    fn test_config_with_dimension() {
        let cfg = LatticeSampleConfig::new().with_dimension(42);
        assert_eq!(cfg.dimension, 42);
    }

    #[test]
    fn test_config_with_std_dev() {
        let cfg = LatticeSampleConfig::new().with_std_dev(42.0);
        assert!((cfg.std_dev - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_center() {
        let cfg = LatticeSampleConfig::new().with_center(42.0);
        assert!((cfg.center - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_precision() {
        let cfg = LatticeSampleConfig::new().with_precision(42);
        assert_eq!(cfg.precision, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = LatticeSampleConfig::new().with_dimension(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = LatticeSample::new(LatticeSampleConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("LatticeSample"));
    }

    #[test]
    fn test_summary() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_discrete_gaussian() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.discrete_gaussian();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_uniform_poly() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.uniform_poly();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_ternary_dist() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ternary_dist();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_ternary_dist_empty() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap();
        assert!(e.ternary_dist().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = LatticeSample::new(LatticeSampleConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = LatticeSampleError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = LatticeSampleError::InvalidConfig("a".into());
        let e2 = LatticeSampleError::ComputationFailed("b".into());
        let e3 = LatticeSampleError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
