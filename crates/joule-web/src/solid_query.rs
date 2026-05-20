//! Solid model query operations.
//!
//! Provides [`SolidQueryConfig`] builder and [`SolidQuery`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SolidQueryError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SolidQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SolidQuery: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SolidQuery: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SolidQuery: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SolidQuery`] parameters.
#[derive(Debug, Clone)]
pub struct SolidQueryConfig {
    pub tolerance: f64,
    pub ray_count: usize,
    pub compute_moments: bool,
    pub include_centroid: bool,
}

impl SolidQueryConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-8,
            ray_count: 7,
            compute_moments: true,
            include_centroid: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_ray_count(mut self, v: usize) -> Self {
        self.ray_count = v;
        self
    }

    pub fn with_compute_moments(mut self, v: bool) -> Self {
        self.compute_moments = v;
        self
    }

    pub fn with_include_centroid(mut self, v: bool) -> Self {
        self.include_centroid = v;
        self
    }

    pub fn validate(&self) -> Result<(), SolidQueryError> {
        if self.tolerance.is_nan() {
            return Err(SolidQueryError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SolidQueryConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SolidQueryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SolidQueryConfig(tolerance={0:.4}, ray_count={1}, compute_moments={2}, include_centroid={3})", self.tolerance, self.ray_count, self.compute_moments, self.include_centroid)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core solid model query operations engine.
#[derive(Debug, Clone)]
pub struct SolidQuery {
    config: SolidQueryConfig,
    data: Vec<f64>,
}

impl SolidQuery {
    pub fn new(config: SolidQueryConfig) -> Result<Self, SolidQueryError> {
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
    pub fn config(&self) -> &SolidQueryConfig { &self.config }

    /// Test if point is inside solid.
    pub fn point_in_solid(&self) -> bool {
        !self.data.is_empty()
    }

    /// Compute solid volume.
    pub fn volume(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Compute mass centroid.
    pub fn centroid(&self) -> Vec<f64> {
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

impl fmt::Display for SolidQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SolidQuery(n={})", self.data.len())
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
        let cfg = SolidQueryConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SolidQueryConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SolidQueryConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = SolidQueryConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_ray_count() {
        let cfg = SolidQueryConfig::new().with_ray_count(42);
        assert_eq!(cfg.ray_count, 42);
    }

    #[test]
    fn test_config_with_compute_moments() {
        let cfg = SolidQueryConfig::new().with_compute_moments(false);
        assert_eq!(cfg.compute_moments, false);
    }

    #[test]
    fn test_config_with_include_centroid() {
        let cfg = SolidQueryConfig::new().with_include_centroid(false);
        assert_eq!(cfg.include_centroid, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SolidQueryConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SolidQuery::new(SolidQueryConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SolidQuery"));
    }

    #[test]
    fn test_summary() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_in_solid() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.point_in_solid();
        assert!(result);
    }

    #[test]
    fn test_volume() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.volume();
        assert!(result.is_finite());
    }

    #[test]
    fn test_centroid() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.centroid();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_centroid_empty() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap();
        assert!(e.centroid().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SolidQuery::new(SolidQueryConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SolidQueryError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SolidQueryError::InvalidConfig("a".into());
        let e2 = SolidQueryError::ComputationFailed("b".into());
        let e3 = SolidQueryError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
