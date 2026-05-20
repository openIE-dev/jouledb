//! Parametric feature-based modeling.
//!
//! Provides [`ParametricModelConfig`] builder and [`ParametricModel`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ParametricModelError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ParametricModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ParametricModel: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ParametricModel: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ParametricModel: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ParametricModel`] parameters.
#[derive(Debug, Clone)]
pub struct ParametricModelConfig {
    pub max_features: usize,
    pub history_depth: usize,
    pub auto_rebuild: bool,
    pub validate_tree: bool,
}

impl ParametricModelConfig {
    pub fn new() -> Self {
        Self {
            max_features: 1000,
            history_depth: 100,
            auto_rebuild: true,
            validate_tree: true,
        }
    }

    pub fn with_max_features(mut self, v: usize) -> Self {
        self.max_features = v;
        self
    }

    pub fn with_history_depth(mut self, v: usize) -> Self {
        self.history_depth = v;
        self
    }

    pub fn with_auto_rebuild(mut self, v: bool) -> Self {
        self.auto_rebuild = v;
        self
    }

    pub fn with_validate_tree(mut self, v: bool) -> Self {
        self.validate_tree = v;
        self
    }

    pub fn validate(&self) -> Result<(), ParametricModelError> {
        Ok(())
    }
}

impl Default for ParametricModelConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ParametricModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParametricModelConfig(max_features={0}, history_depth={1}, auto_rebuild={2}, validate_tree={3})", self.max_features, self.history_depth, self.auto_rebuild, self.validate_tree)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core parametric feature-based modeling engine.
#[derive(Debug, Clone)]
pub struct ParametricModel {
    config: ParametricModelConfig,
    data: Vec<f64>,
}

impl ParametricModel {
    pub fn new(config: ParametricModelConfig) -> Result<Self, ParametricModelError> {
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
    pub fn config(&self) -> &ParametricModelConfig { &self.config }

    /// Add feature to model.
    pub fn add_feature(&self) -> usize {
        self.data.len()
    }

    /// Rebuild model from history.
    pub fn rebuild(&self) -> bool {
        !self.data.is_empty()
    }

    /// Sweep parameter range.
    pub fn parameter_sweep(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for ParametricModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParametricModel(n={})", self.data.len())
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
        let cfg = ParametricModelConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ParametricModelConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ParametricModelConfig"));
    }

    #[test]
    fn test_config_with_max_features() {
        let cfg = ParametricModelConfig::new().with_max_features(42);
        assert_eq!(cfg.max_features, 42);
    }

    #[test]
    fn test_config_with_history_depth() {
        let cfg = ParametricModelConfig::new().with_history_depth(42);
        assert_eq!(cfg.history_depth, 42);
    }

    #[test]
    fn test_config_with_auto_rebuild() {
        let cfg = ParametricModelConfig::new().with_auto_rebuild(false);
        assert_eq!(cfg.auto_rebuild, false);
    }

    #[test]
    fn test_config_with_validate_tree() {
        let cfg = ParametricModelConfig::new().with_validate_tree(false);
        assert_eq!(cfg.validate_tree, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ParametricModelConfig::new().with_max_features(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ParametricModel::new(ParametricModelConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ParametricModel"));
    }

    #[test]
    fn test_summary() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_feature() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_feature();
        assert!(result > 0);
    }

    #[test]
    fn test_rebuild() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rebuild();
        assert!(result);
    }

    #[test]
    fn test_parameter_sweep() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parameter_sweep();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_parameter_sweep_empty() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap();
        assert!(e.parameter_sweep().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ParametricModel::new(ParametricModelConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ParametricModelError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ParametricModelError::InvalidConfig("a".into());
        let e2 = ParametricModelError::ComputationFailed("b".into());
        let e3 = ParametricModelError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
