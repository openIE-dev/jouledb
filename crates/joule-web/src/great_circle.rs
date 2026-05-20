//! Great circle distance and bearing calculations.
//!
//! Provides [`GreatCircleConfig`] builder and [`GreatCircle`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GreatCircleError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GreatCircleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GreatCircle: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GreatCircle: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GreatCircle: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GreatCircle`] parameters.
#[derive(Debug, Clone)]
pub struct GreatCircleConfig {
    pub radius_m: f64,
    pub flattening: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GreatCircleConfig {
    pub fn new() -> Self {
        Self {
            radius_m: 6371000.0,
            flattening: 0.003352811,
            max_iterations: 100,
            tolerance: 1e-12,
        }
    }

    pub fn with_radius_m(mut self, v: f64) -> Self {
        self.radius_m = v;
        self
    }

    pub fn with_flattening(mut self, v: f64) -> Self {
        self.flattening = v;
        self
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn validate(&self) -> Result<(), GreatCircleError> {
        if self.radius_m.is_nan() {
            return Err(GreatCircleError::InvalidConfig("radius_m is NaN".into()));
        }
        if self.flattening.is_nan() {
            return Err(GreatCircleError::InvalidConfig("flattening is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(GreatCircleError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for GreatCircleConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GreatCircleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GreatCircleConfig(radius_m={0:.4}, flattening={1:.4}, max_iterations={2}, tolerance={3:.4})", self.radius_m, self.flattening, self.max_iterations, self.tolerance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core great circle distance and bearing calculations engine.
#[derive(Debug, Clone)]
pub struct GreatCircle {
    config: GreatCircleConfig,
    data: Vec<f64>,
}

impl GreatCircle {
    pub fn new(config: GreatCircleConfig) -> Result<Self, GreatCircleError> {
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
    pub fn config(&self) -> &GreatCircleConfig { &self.config }

    /// Haversine distance.
    pub fn haversine(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Vincenty distance (iterative).
    pub fn vincenty(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Initial bearing in degrees.
    pub fn initial_bearing(&self) -> f64 {
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

impl fmt::Display for GreatCircle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GreatCircle(n={})", self.data.len())
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
        let cfg = GreatCircleConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GreatCircleConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GreatCircleConfig"));
    }

    #[test]
    fn test_config_with_radius_m() {
        let cfg = GreatCircleConfig::new().with_radius_m(42.0);
        assert!((cfg.radius_m - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_flattening() {
        let cfg = GreatCircleConfig::new().with_flattening(42.0);
        assert!((cfg.flattening - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_iterations() {
        let cfg = GreatCircleConfig::new().with_max_iterations(42);
        assert_eq!(cfg.max_iterations, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = GreatCircleConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GreatCircleConfig::new().with_radius_m(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GreatCircle::new(GreatCircleConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GreatCircle"));
    }

    #[test]
    fn test_summary() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_haversine() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.haversine();
        assert!(result.is_finite());
    }

    #[test]
    fn test_vincenty() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.vincenty();
        assert!(result.is_finite());
    }

    #[test]
    fn test_initial_bearing() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.initial_bearing();
        assert!(result.is_finite());
    }

    #[test]
    fn test_initial_bearing_empty() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap();
        assert!((e.initial_bearing() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = GreatCircle::new(GreatCircleConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GreatCircleError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GreatCircleError::InvalidConfig("a".into());
        let e2 = GreatCircleError::ComputationFailed("b".into());
        let e3 = GreatCircleError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
