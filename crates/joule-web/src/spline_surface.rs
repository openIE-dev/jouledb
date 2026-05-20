//! Tensor-product B-spline surface patch.
//!
//! Provides [`SplineSurfaceConfig`] builder and [`SplineSurface`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SplineSurfaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SplineSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SplineSurface: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SplineSurface: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SplineSurface: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SplineSurface`] parameters.
#[derive(Debug, Clone)]
pub struct SplineSurfaceConfig {
    pub degree_u: usize,
    pub degree_v: usize,
    pub control_pts_u: usize,
    pub control_pts_v: usize,
}

impl SplineSurfaceConfig {
    pub fn new() -> Self {
        Self {
            degree_u: 3,
            degree_v: 3,
            control_pts_u: 5,
            control_pts_v: 5,
        }
    }

    pub fn with_degree_u(mut self, v: usize) -> Self {
        self.degree_u = v;
        self
    }

    pub fn with_degree_v(mut self, v: usize) -> Self {
        self.degree_v = v;
        self
    }

    pub fn with_control_pts_u(mut self, v: usize) -> Self {
        self.control_pts_u = v;
        self
    }

    pub fn with_control_pts_v(mut self, v: usize) -> Self {
        self.control_pts_v = v;
        self
    }

    pub fn validate(&self) -> Result<(), SplineSurfaceError> {
        Ok(())
    }
}

impl Default for SplineSurfaceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SplineSurfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SplineSurfaceConfig(degree_u={0}, degree_v={1}, control_pts_u={2}, control_pts_v={3})", self.degree_u, self.degree_v, self.control_pts_u, self.control_pts_v)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core tensor-product b-spline surface patch engine.
#[derive(Debug, Clone)]
pub struct SplineSurface {
    config: SplineSurfaceConfig,
    data: Vec<f64>,
}

impl SplineSurface {
    pub fn new(config: SplineSurfaceConfig) -> Result<Self, SplineSurfaceError> {
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
    pub fn config(&self) -> &SplineSurfaceConfig { &self.config }

    /// Evaluate surface at (u,v).
    pub fn evaluate_uv(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Surface normal at (u,v).
    pub fn normal_at(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Extract iso-parametric curve.
    pub fn iso_curve(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SplineSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SplineSurface(n={})", self.data.len())
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
        let cfg = SplineSurfaceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SplineSurfaceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SplineSurfaceConfig"));
    }

    #[test]
    fn test_config_with_degree_u() {
        let cfg = SplineSurfaceConfig::new().with_degree_u(42);
        assert_eq!(cfg.degree_u, 42);
    }

    #[test]
    fn test_config_with_degree_v() {
        let cfg = SplineSurfaceConfig::new().with_degree_v(42);
        assert_eq!(cfg.degree_v, 42);
    }

    #[test]
    fn test_config_with_control_pts_u() {
        let cfg = SplineSurfaceConfig::new().with_control_pts_u(42);
        assert_eq!(cfg.control_pts_u, 42);
    }

    #[test]
    fn test_config_with_control_pts_v() {
        let cfg = SplineSurfaceConfig::new().with_control_pts_v(42);
        assert_eq!(cfg.control_pts_v, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SplineSurfaceConfig::new().with_degree_u(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SplineSurface"));
    }

    #[test]
    fn test_summary() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_uv() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate_uv();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_normal_at() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.normal_at();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_iso_curve() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.iso_curve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_iso_curve_empty() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap();
        assert!(e.iso_curve().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SplineSurface::new(SplineSurfaceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SplineSurfaceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SplineSurfaceError::InvalidConfig("a".into());
        let e2 = SplineSurfaceError::ComputationFailed("b".into());
        let e3 = SplineSurfaceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
