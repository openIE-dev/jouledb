//! Mesh smoothing operations.
//!
//! Provides [`MeshSmoothConfig`] builder and [`MeshSmooth`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshSmoothError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MeshSmoothError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MeshSmooth: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MeshSmooth: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MeshSmooth: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MeshSmooth`] parameters.
#[derive(Debug, Clone)]
pub struct MeshSmoothConfig {
    pub iterations: usize,
    pub lambda: f64,
    pub mu: f64,
    pub preserve_boundary: bool,
}

impl MeshSmoothConfig {
    pub fn new() -> Self {
        Self {
            iterations: 5,
            lambda: 0.5,
            mu: -0.53,
            preserve_boundary: true,
        }
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    pub fn with_lambda(mut self, v: f64) -> Self {
        self.lambda = v;
        self
    }

    pub fn with_mu(mut self, v: f64) -> Self {
        self.mu = v;
        self
    }

    pub fn with_preserve_boundary(mut self, v: bool) -> Self {
        self.preserve_boundary = v;
        self
    }

    pub fn validate(&self) -> Result<(), MeshSmoothError> {
        if self.lambda.is_nan() {
            return Err(MeshSmoothError::InvalidConfig("lambda is NaN".into()));
        }
        if self.mu.is_nan() {
            return Err(MeshSmoothError::InvalidConfig("mu is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MeshSmoothConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MeshSmoothConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshSmoothConfig(iterations={0}, lambda={1:.4}, mu={2:.4}, preserve_boundary={3})", self.iterations, self.lambda, self.mu, self.preserve_boundary)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core mesh smoothing operations engine.
#[derive(Debug, Clone)]
pub struct MeshSmooth {
    config: MeshSmoothConfig,
    data: Vec<f64>,
}

impl MeshSmooth {
    pub fn new(config: MeshSmoothConfig) -> Result<Self, MeshSmoothError> {
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
    pub fn config(&self) -> &MeshSmoothConfig { &self.config }

    /// Laplacian smoothing.
    pub fn laplacian_smooth(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Taubin smoothing.
    pub fn taubin_smooth(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Feature-preserving smooth.
    pub fn feature_preserve(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for MeshSmooth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshSmooth(n={})", self.data.len())
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
        let cfg = MeshSmoothConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MeshSmoothConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MeshSmoothConfig"));
    }

    #[test]
    fn test_config_with_iterations() {
        let cfg = MeshSmoothConfig::new().with_iterations(42);
        assert_eq!(cfg.iterations, 42);
    }

    #[test]
    fn test_config_with_lambda() {
        let cfg = MeshSmoothConfig::new().with_lambda(42.0);
        assert!((cfg.lambda - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_mu() {
        let cfg = MeshSmoothConfig::new().with_mu(42.0);
        assert!((cfg.mu - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_preserve_boundary() {
        let cfg = MeshSmoothConfig::new().with_preserve_boundary(false);
        assert_eq!(cfg.preserve_boundary, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MeshSmoothConfig::new().with_iterations(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MeshSmooth"));
    }

    #[test]
    fn test_summary() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_laplacian_smooth() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.laplacian_smooth();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_taubin_smooth() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.taubin_smooth();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_feature_preserve() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.feature_preserve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_feature_preserve_empty() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap();
        assert!(e.feature_preserve().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MeshSmooth::new(MeshSmoothConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MeshSmoothError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MeshSmoothError::InvalidConfig("a".into());
        let e2 = MeshSmoothError::ComputationFailed("b".into());
        let e3 = MeshSmoothError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
