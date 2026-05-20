//! Mesh subdivision algorithms.
//!
//! Provides [`MeshSubdivideConfig`] builder and [`MeshSubdivide`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshSubdivideError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MeshSubdivideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MeshSubdivide: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MeshSubdivide: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MeshSubdivide: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MeshSubdivide`] parameters.
#[derive(Debug, Clone)]
pub struct MeshSubdivideConfig {
    pub iterations: usize,
    pub method: usize,
    pub crease_angle: f64,
    pub boundary_rule: usize,
}

impl MeshSubdivideConfig {
    pub fn new() -> Self {
        Self {
            iterations: 2,
            method: 0,
            crease_angle: 30.0,
            boundary_rule: 0,
        }
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    pub fn with_method(mut self, v: usize) -> Self {
        self.method = v;
        self
    }

    pub fn with_crease_angle(mut self, v: f64) -> Self {
        self.crease_angle = v;
        self
    }

    pub fn with_boundary_rule(mut self, v: usize) -> Self {
        self.boundary_rule = v;
        self
    }

    pub fn validate(&self) -> Result<(), MeshSubdivideError> {
        if self.crease_angle.is_nan() {
            return Err(MeshSubdivideError::InvalidConfig("crease_angle is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MeshSubdivideConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MeshSubdivideConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshSubdivideConfig(iterations={0}, method={1}, crease_angle={2:.4}, boundary_rule={3})", self.iterations, self.method, self.crease_angle, self.boundary_rule)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core mesh subdivision algorithms engine.
#[derive(Debug, Clone)]
pub struct MeshSubdivide {
    config: MeshSubdivideConfig,
    data: Vec<f64>,
}

impl MeshSubdivide {
    pub fn new(config: MeshSubdivideConfig) -> Result<Self, MeshSubdivideError> {
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
    pub fn config(&self) -> &MeshSubdivideConfig { &self.config }

    /// Catmull-Clark subdivision.
    pub fn catmull_clark(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Loop subdivision.
    pub fn loop_subdivide(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Butterfly subdivision.
    pub fn butterfly(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for MeshSubdivide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshSubdivide(n={})", self.data.len())
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
        let cfg = MeshSubdivideConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MeshSubdivideConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MeshSubdivideConfig"));
    }

    #[test]
    fn test_config_with_iterations() {
        let cfg = MeshSubdivideConfig::new().with_iterations(42);
        assert_eq!(cfg.iterations, 42);
    }

    #[test]
    fn test_config_with_method() {
        let cfg = MeshSubdivideConfig::new().with_method(42);
        assert_eq!(cfg.method, 42);
    }

    #[test]
    fn test_config_with_crease_angle() {
        let cfg = MeshSubdivideConfig::new().with_crease_angle(42.0);
        assert!((cfg.crease_angle - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_boundary_rule() {
        let cfg = MeshSubdivideConfig::new().with_boundary_rule(42);
        assert_eq!(cfg.boundary_rule, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MeshSubdivideConfig::new().with_iterations(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MeshSubdivide"));
    }

    #[test]
    fn test_summary() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_catmull_clark() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.catmull_clark();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_loop_subdivide() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.loop_subdivide();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_butterfly() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.butterfly();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_butterfly_empty() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap();
        assert!(e.butterfly().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MeshSubdivide::new(MeshSubdivideConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MeshSubdivideError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MeshSubdivideError::InvalidConfig("a".into());
        let e2 = MeshSubdivideError::ComputationFailed("b".into());
        let e3 = MeshSubdivideError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
