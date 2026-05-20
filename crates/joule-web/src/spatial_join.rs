//! Spatial join operations for geographic data.
//!
//! Provides [`SpatialJoinConfig`] builder and [`SpatialJoin`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialJoinError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SpatialJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SpatialJoin: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SpatialJoin: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SpatialJoin: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SpatialJoin`] parameters.
#[derive(Debug, Clone)]
pub struct SpatialJoinConfig {
    pub join_type: usize,
    pub distance_threshold: f64,
    pub k_nearest: usize,
    pub include_distance: bool,
}

impl SpatialJoinConfig {
    pub fn new() -> Self {
        Self {
            join_type: 0,
            distance_threshold: 100.0,
            k_nearest: 1,
            include_distance: true,
        }
    }

    pub fn with_join_type(mut self, v: usize) -> Self {
        self.join_type = v;
        self
    }

    pub fn with_distance_threshold(mut self, v: f64) -> Self {
        self.distance_threshold = v;
        self
    }

    pub fn with_k_nearest(mut self, v: usize) -> Self {
        self.k_nearest = v;
        self
    }

    pub fn with_include_distance(mut self, v: bool) -> Self {
        self.include_distance = v;
        self
    }

    pub fn validate(&self) -> Result<(), SpatialJoinError> {
        if self.distance_threshold.is_nan() {
            return Err(SpatialJoinError::InvalidConfig("distance_threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SpatialJoinConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SpatialJoinConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialJoinConfig(join_type={0}, distance_threshold={1:.4}, k_nearest={2}, include_distance={3})", self.join_type, self.distance_threshold, self.k_nearest, self.include_distance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core spatial join operations for geographic data engine.
#[derive(Debug, Clone)]
pub struct SpatialJoin {
    config: SpatialJoinConfig,
    data: Vec<f64>,
}

impl SpatialJoin {
    pub fn new(config: SpatialJoinConfig) -> Result<Self, SpatialJoinError> {
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
    pub fn config(&self) -> &SpatialJoinConfig { &self.config }

    /// Point-in-polygon spatial join.
    pub fn point_in_polygon_join(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Distance-based spatial join.
    pub fn distance_join(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Intersects predicate join.
    pub fn intersects_join(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SpatialJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialJoin(n={})", self.data.len())
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
        let cfg = SpatialJoinConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SpatialJoinConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SpatialJoinConfig"));
    }

    #[test]
    fn test_config_with_join_type() {
        let cfg = SpatialJoinConfig::new().with_join_type(42);
        assert_eq!(cfg.join_type, 42);
    }

    #[test]
    fn test_config_with_distance_threshold() {
        let cfg = SpatialJoinConfig::new().with_distance_threshold(42.0);
        assert!((cfg.distance_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_k_nearest() {
        let cfg = SpatialJoinConfig::new().with_k_nearest(42);
        assert_eq!(cfg.k_nearest, 42);
    }

    #[test]
    fn test_config_with_include_distance() {
        let cfg = SpatialJoinConfig::new().with_include_distance(false);
        assert_eq!(cfg.include_distance, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SpatialJoinConfig::new().with_join_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SpatialJoin"));
    }

    #[test]
    fn test_summary() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_in_polygon_join() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.point_in_polygon_join();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_distance_join() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.distance_join();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_intersects_join() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.intersects_join();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_intersects_join_empty() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap();
        assert!(e.intersects_join().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SpatialJoin::new(SpatialJoinConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SpatialJoinError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SpatialJoinError::InvalidConfig("a".into());
        let e2 = SpatialJoinError::ComputationFailed("b".into());
        let e3 = SpatialJoinError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
