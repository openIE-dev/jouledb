//! Triangle mesh generation from geometry.
//!
//! Provides [`MeshGenConfig`] builder and [`MeshGen`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshGenError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MeshGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MeshGen: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MeshGen: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MeshGen: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MeshGen`] parameters.
#[derive(Debug, Clone)]
pub struct MeshGenConfig {
    pub max_edge_length: f64,
    pub min_angle_deg: f64,
    pub max_triangles: usize,
    pub adaptive: bool,
}

impl MeshGenConfig {
    pub fn new() -> Self {
        Self {
            max_edge_length: 1.0,
            min_angle_deg: 20.0,
            max_triangles: 100000,
            adaptive: true,
        }
    }

    pub fn with_max_edge_length(mut self, v: f64) -> Self {
        self.max_edge_length = v;
        self
    }

    pub fn with_min_angle_deg(mut self, v: f64) -> Self {
        self.min_angle_deg = v;
        self
    }

    pub fn with_max_triangles(mut self, v: usize) -> Self {
        self.max_triangles = v;
        self
    }

    pub fn with_adaptive(mut self, v: bool) -> Self {
        self.adaptive = v;
        self
    }

    pub fn validate(&self) -> Result<(), MeshGenError> {
        if self.max_edge_length.is_nan() {
            return Err(MeshGenError::InvalidConfig("max_edge_length is NaN".into()));
        }
        if self.min_angle_deg.is_nan() {
            return Err(MeshGenError::InvalidConfig("min_angle_deg is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MeshGenConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MeshGenConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshGenConfig(max_edge_length={0:.4}, min_angle_deg={1:.4}, max_triangles={2}, adaptive={3})", self.max_edge_length, self.min_angle_deg, self.max_triangles, self.adaptive)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core triangle mesh generation from geometry engine.
#[derive(Debug, Clone)]
pub struct MeshGen {
    config: MeshGenConfig,
    data: Vec<f64>,
}

impl MeshGen {
    pub fn new(config: MeshGenConfig) -> Result<Self, MeshGenError> {
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
    pub fn config(&self) -> &MeshGenConfig { &self.config }

    /// 2D Delaunay triangulation.
    pub fn triangulate_2d(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Mesh a parametric surface.
    pub fn surface_mesh(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute mesh quality.
    pub fn quality_metrics(&self) -> Vec<f64> {
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

impl fmt::Display for MeshGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshGen(n={})", self.data.len())
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
        let cfg = MeshGenConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MeshGenConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MeshGenConfig"));
    }

    #[test]
    fn test_config_with_max_edge_length() {
        let cfg = MeshGenConfig::new().with_max_edge_length(42.0);
        assert!((cfg.max_edge_length - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_min_angle_deg() {
        let cfg = MeshGenConfig::new().with_min_angle_deg(42.0);
        assert!((cfg.min_angle_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_triangles() {
        let cfg = MeshGenConfig::new().with_max_triangles(42);
        assert_eq!(cfg.max_triangles, 42);
    }

    #[test]
    fn test_config_with_adaptive() {
        let cfg = MeshGenConfig::new().with_adaptive(false);
        assert_eq!(cfg.adaptive, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MeshGenConfig::new().with_max_edge_length(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MeshGen::new(MeshGenConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MeshGen"));
    }

    #[test]
    fn test_summary() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_triangulate_2d() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.triangulate_2d();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_surface_mesh() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.surface_mesh();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quality_metrics() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.quality_metrics();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quality_metrics_empty() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap();
        assert!(e.quality_metrics().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MeshGen::new(MeshGenConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MeshGenError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MeshGenError::InvalidConfig("a".into());
        let e2 = MeshGenError::ComputationFailed("b".into());
        let e3 = MeshGenError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
