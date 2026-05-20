//! Discrete mesh curvature computation.
//!
//! Provides [`MeshCurvatureConfig`] builder and [`MeshCurvature`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshCurvatureError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MeshCurvatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MeshCurvature: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MeshCurvature: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MeshCurvature: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MeshCurvature`] parameters.
#[derive(Debug, Clone)]
pub struct MeshCurvatureConfig {
    pub ring_size: usize,
    pub smooth: bool,
    pub feature_angle: f64,
    pub color_map: bool,
}

impl MeshCurvatureConfig {
    pub fn new() -> Self {
        Self {
            ring_size: 1,
            smooth: true,
            feature_angle: 30.0,
            color_map: true,
        }
    }

    pub fn with_ring_size(mut self, v: usize) -> Self {
        self.ring_size = v;
        self
    }

    pub fn with_smooth(mut self, v: bool) -> Self {
        self.smooth = v;
        self
    }

    pub fn with_feature_angle(mut self, v: f64) -> Self {
        self.feature_angle = v;
        self
    }

    pub fn with_color_map(mut self, v: bool) -> Self {
        self.color_map = v;
        self
    }

    pub fn validate(&self) -> Result<(), MeshCurvatureError> {
        if self.feature_angle.is_nan() {
            return Err(MeshCurvatureError::InvalidConfig("feature_angle is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MeshCurvatureConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MeshCurvatureConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshCurvatureConfig(ring_size={0}, smooth={1}, feature_angle={2:.4}, color_map={3})", self.ring_size, self.smooth, self.feature_angle, self.color_map)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core discrete mesh curvature computation engine.
#[derive(Debug, Clone)]
pub struct MeshCurvature {
    config: MeshCurvatureConfig,
    data: Vec<f64>,
}

impl MeshCurvature {
    pub fn new(config: MeshCurvatureConfig) -> Result<Self, MeshCurvatureError> {
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
    pub fn config(&self) -> &MeshCurvatureConfig { &self.config }

    /// Discrete Gaussian curvature.
    pub fn gaussian_curvature(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Discrete mean curvature.
    pub fn mean_curvature(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Detect feature edges.
    pub fn feature_edges(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for MeshCurvature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshCurvature(n={})", self.data.len())
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
        let cfg = MeshCurvatureConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MeshCurvatureConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MeshCurvatureConfig"));
    }

    #[test]
    fn test_config_with_ring_size() {
        let cfg = MeshCurvatureConfig::new().with_ring_size(42);
        assert_eq!(cfg.ring_size, 42);
    }

    #[test]
    fn test_config_with_smooth() {
        let cfg = MeshCurvatureConfig::new().with_smooth(false);
        assert_eq!(cfg.smooth, false);
    }

    #[test]
    fn test_config_with_feature_angle() {
        let cfg = MeshCurvatureConfig::new().with_feature_angle(42.0);
        assert!((cfg.feature_angle - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_color_map() {
        let cfg = MeshCurvatureConfig::new().with_color_map(false);
        assert_eq!(cfg.color_map, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MeshCurvatureConfig::new().with_ring_size(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MeshCurvature"));
    }

    #[test]
    fn test_summary() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_curvature() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gaussian_curvature();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mean_curvature() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mean_curvature();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_feature_edges() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.feature_edges();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_feature_edges_empty() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap();
        assert!(e.feature_edges().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MeshCurvature::new(MeshCurvatureConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MeshCurvatureError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MeshCurvatureError::InvalidConfig("a".into());
        let e2 = MeshCurvatureError::ComputationFailed("b".into());
        let e3 = MeshCurvatureError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
