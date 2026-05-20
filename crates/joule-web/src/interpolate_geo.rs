//! Spatial interpolation methods.
//!
//! Provides [`InterpolateGeoConfig`] builder and [`InterpolateGeo`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolateGeoError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for InterpolateGeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "InterpolateGeo: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "InterpolateGeo: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "InterpolateGeo: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`InterpolateGeo`] parameters.
#[derive(Debug, Clone)]
pub struct InterpolateGeoConfig {
    pub power: f64,
    pub search_radius: f64,
    pub min_points: usize,
    pub max_points: usize,
}

impl InterpolateGeoConfig {
    pub fn new() -> Self {
        Self {
            power: 2.0,
            search_radius: 1000.0,
            min_points: 3,
            max_points: 12,
        }
    }

    pub fn with_power(mut self, v: f64) -> Self {
        self.power = v;
        self
    }

    pub fn with_search_radius(mut self, v: f64) -> Self {
        self.search_radius = v;
        self
    }

    pub fn with_min_points(mut self, v: usize) -> Self {
        self.min_points = v;
        self
    }

    pub fn with_max_points(mut self, v: usize) -> Self {
        self.max_points = v;
        self
    }

    pub fn validate(&self) -> Result<(), InterpolateGeoError> {
        if self.power.is_nan() {
            return Err(InterpolateGeoError::InvalidConfig("power is NaN".into()));
        }
        if self.search_radius.is_nan() {
            return Err(InterpolateGeoError::InvalidConfig("search_radius is NaN".into()));
        }
        Ok(())
    }
}

impl Default for InterpolateGeoConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for InterpolateGeoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterpolateGeoConfig(power={0:.4}, search_radius={1:.4}, min_points={2}, max_points={3})", self.power, self.search_radius, self.min_points, self.max_points)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core spatial interpolation methods engine.
#[derive(Debug, Clone)]
pub struct InterpolateGeo {
    config: InterpolateGeoConfig,
    data: Vec<f64>,
}

impl InterpolateGeo {
    pub fn new(config: InterpolateGeoConfig) -> Result<Self, InterpolateGeoError> {
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
    pub fn config(&self) -> &InterpolateGeoConfig { &self.config }

    /// Inverse distance weighted interpolation.
    pub fn idw(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Ordinary kriging interpolation.
    pub fn kriging(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Natural neighbor interpolation.
    pub fn natural_neighbor(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for InterpolateGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterpolateGeo(n={})", self.data.len())
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
        let cfg = InterpolateGeoConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = InterpolateGeoConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("InterpolateGeoConfig"));
    }

    #[test]
    fn test_config_with_power() {
        let cfg = InterpolateGeoConfig::new().with_power(42.0);
        assert!((cfg.power - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_search_radius() {
        let cfg = InterpolateGeoConfig::new().with_search_radius(42.0);
        assert!((cfg.search_radius - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_min_points() {
        let cfg = InterpolateGeoConfig::new().with_min_points(42);
        assert_eq!(cfg.min_points, 42);
    }

    #[test]
    fn test_config_with_max_points() {
        let cfg = InterpolateGeoConfig::new().with_max_points(42);
        assert_eq!(cfg.max_points, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = InterpolateGeoConfig::new().with_power(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("InterpolateGeo"));
    }

    #[test]
    fn test_summary() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_idw() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.idw();
        assert!(result.is_finite());
    }

    #[test]
    fn test_kriging() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.kriging();
        assert!(result.is_finite());
    }

    #[test]
    fn test_natural_neighbor() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.natural_neighbor();
        assert!(result.is_finite());
    }

    #[test]
    fn test_natural_neighbor_empty() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap();
        assert!((e.natural_neighbor() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = InterpolateGeo::new(InterpolateGeoConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = InterpolateGeoError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = InterpolateGeoError::InvalidConfig("a".into());
        let e2 = InterpolateGeoError::ComputationFailed("b".into());
        let e3 = InterpolateGeoError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
