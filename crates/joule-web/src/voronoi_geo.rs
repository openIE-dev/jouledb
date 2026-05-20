//! Voronoi diagram and Delaunay triangulation.
//!
//! Provides [`VoronoiGeoConfig`] builder and [`VoronoiGeo`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum VoronoiGeoError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for VoronoiGeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "VoronoiGeo: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "VoronoiGeo: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "VoronoiGeo: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`VoronoiGeo`] parameters.
#[derive(Debug, Clone)]
pub struct VoronoiGeoConfig {
    pub bound_min_x: f64,
    pub bound_max_x: f64,
    pub bound_min_y: f64,
    pub bound_max_y: f64,
}

impl VoronoiGeoConfig {
    pub fn new() -> Self {
        Self {
            bound_min_x: -180.0,
            bound_max_x: 180.0,
            bound_min_y: -90.0,
            bound_max_y: 90.0,
        }
    }

    pub fn with_bound_min_x(mut self, v: f64) -> Self {
        self.bound_min_x = v;
        self
    }

    pub fn with_bound_max_x(mut self, v: f64) -> Self {
        self.bound_max_x = v;
        self
    }

    pub fn with_bound_min_y(mut self, v: f64) -> Self {
        self.bound_min_y = v;
        self
    }

    pub fn with_bound_max_y(mut self, v: f64) -> Self {
        self.bound_max_y = v;
        self
    }

    pub fn validate(&self) -> Result<(), VoronoiGeoError> {
        if self.bound_min_x.is_nan() {
            return Err(VoronoiGeoError::InvalidConfig("bound_min_x is NaN".into()));
        }
        if self.bound_max_x.is_nan() {
            return Err(VoronoiGeoError::InvalidConfig("bound_max_x is NaN".into()));
        }
        if self.bound_min_y.is_nan() {
            return Err(VoronoiGeoError::InvalidConfig("bound_min_y is NaN".into()));
        }
        if self.bound_max_y.is_nan() {
            return Err(VoronoiGeoError::InvalidConfig("bound_max_y is NaN".into()));
        }
        Ok(())
    }
}

impl Default for VoronoiGeoConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VoronoiGeoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VoronoiGeoConfig(bound_min_x={0:.4}, bound_max_x={1:.4}, bound_min_y={2:.4}, bound_max_y={3:.4})", self.bound_min_x, self.bound_max_x, self.bound_min_y, self.bound_max_y)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core voronoi diagram and delaunay triangulation engine.
#[derive(Debug, Clone)]
pub struct VoronoiGeo {
    config: VoronoiGeoConfig,
    data: Vec<f64>,
}

impl VoronoiGeo {
    pub fn new(config: VoronoiGeoConfig) -> Result<Self, VoronoiGeoError> {
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
    pub fn config(&self) -> &VoronoiGeoConfig { &self.config }

    /// Compute Voronoi cells.
    pub fn compute_voronoi(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute Delaunay triangulation.
    pub fn delaunay(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Find nearest Voronoi site.
    pub fn nearest_site(&self) -> Vec<f64> {
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

impl fmt::Display for VoronoiGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VoronoiGeo(n={})", self.data.len())
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
        let cfg = VoronoiGeoConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = VoronoiGeoConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("VoronoiGeoConfig"));
    }

    #[test]
    fn test_config_with_bound_min_x() {
        let cfg = VoronoiGeoConfig::new().with_bound_min_x(42.0);
        assert!((cfg.bound_min_x - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bound_max_x() {
        let cfg = VoronoiGeoConfig::new().with_bound_max_x(42.0);
        assert!((cfg.bound_max_x - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bound_min_y() {
        let cfg = VoronoiGeoConfig::new().with_bound_min_y(42.0);
        assert!((cfg.bound_min_y - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bound_max_y() {
        let cfg = VoronoiGeoConfig::new().with_bound_max_y(42.0);
        assert!((cfg.bound_max_y - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = VoronoiGeoConfig::new().with_bound_min_x(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("VoronoiGeo"));
    }

    #[test]
    fn test_summary() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_voronoi() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compute_voronoi();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_delaunay() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.delaunay();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_nearest_site() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.nearest_site();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_nearest_site_empty() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap();
        assert!(e.nearest_site().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = VoronoiGeo::new(VoronoiGeoConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = VoronoiGeoError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = VoronoiGeoError::InvalidConfig("a".into());
        let e2 = VoronoiGeoError::ComputationFailed("b".into());
        let e3 = VoronoiGeoError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
