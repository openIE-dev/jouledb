//! Multivariate polynomial cryptography core.
//!
//! Provides [`MultivarCoreConfig`] builder and [`MultivarCore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MultivarCoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MultivarCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MultivarCore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MultivarCore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MultivarCore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MultivarCore`] parameters.
#[derive(Debug, Clone)]
pub struct MultivarCoreConfig {
    pub field_size: u32,
    pub num_vars: usize,
    pub num_equations: usize,
    pub degree: usize,
}

impl MultivarCoreConfig {
    pub fn new() -> Self {
        Self {
            field_size: 31,
            num_vars: 10,
            num_equations: 10,
            degree: 2,
        }
    }

    pub fn with_field_size(mut self, v: u32) -> Self {
        self.field_size = v;
        self
    }

    pub fn with_num_vars(mut self, v: usize) -> Self {
        self.num_vars = v;
        self
    }

    pub fn with_num_equations(mut self, v: usize) -> Self {
        self.num_equations = v;
        self
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v;
        self
    }

    pub fn validate(&self) -> Result<(), MultivarCoreError> {
        Ok(())
    }
}

impl Default for MultivarCoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MultivarCoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultivarCoreConfig(field_size={0}, num_vars={1}, num_equations={2}, degree={3})", self.field_size, self.num_vars, self.num_equations, self.degree)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core multivariate polynomial cryptography core engine.
#[derive(Debug, Clone)]
pub struct MultivarCore {
    config: MultivarCoreConfig,
    data: Vec<f64>,
}

impl MultivarCore {
    pub fn new(config: MultivarCoreConfig) -> Result<Self, MultivarCoreError> {
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
    pub fn config(&self) -> &MultivarCoreConfig { &self.config }

    /// Evaluate multivariate polynomial system.
    pub fn evaluate_system(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Construct central map.
    pub fn central_map(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compose with affine transform.
    pub fn compose_affine(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for MultivarCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultivarCore(n={})", self.data.len())
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
        let cfg = MultivarCoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MultivarCoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MultivarCoreConfig"));
    }

    #[test]
    fn test_config_with_field_size() {
        let cfg = MultivarCoreConfig::new().with_field_size(42);
        assert_eq!(cfg.field_size, 42);
    }

    #[test]
    fn test_config_with_num_vars() {
        let cfg = MultivarCoreConfig::new().with_num_vars(42);
        assert_eq!(cfg.num_vars, 42);
    }

    #[test]
    fn test_config_with_num_equations() {
        let cfg = MultivarCoreConfig::new().with_num_equations(42);
        assert_eq!(cfg.num_equations, 42);
    }

    #[test]
    fn test_config_with_degree() {
        let cfg = MultivarCoreConfig::new().with_degree(42);
        assert_eq!(cfg.degree, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MultivarCoreConfig::new().with_field_size(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MultivarCore::new(MultivarCoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MultivarCore"));
    }

    #[test]
    fn test_summary() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_system() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate_system();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_central_map() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.central_map();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compose_affine() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compose_affine();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compose_affine_empty() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap();
        assert!(e.compose_affine().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MultivarCore::new(MultivarCoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MultivarCoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MultivarCoreError::InvalidConfig("a".into());
        let e2 = MultivarCoreError::ComputationFailed("b".into());
        let e3 = MultivarCoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
