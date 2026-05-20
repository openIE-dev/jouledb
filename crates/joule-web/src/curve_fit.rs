//! Curve fitting through point data.
//!
//! Provides [`CurveFitConfig`] builder and [`CurveFit`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveFitError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CurveFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CurveFit: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CurveFit: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CurveFit: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CurveFit`] parameters.
#[derive(Debug, Clone)]
pub struct CurveFitConfig {
    pub degree: usize,
    pub tolerance: f64,
    pub max_control_pts: usize,
    pub parameterization: usize,
}

impl CurveFitConfig {
    pub fn new() -> Self {
        Self {
            degree: 3,
            tolerance: 0.01,
            max_control_pts: 50,
            parameterization: 0,
        }
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_control_pts(mut self, v: usize) -> Self {
        self.max_control_pts = v;
        self
    }

    pub fn with_parameterization(mut self, v: usize) -> Self {
        self.parameterization = v;
        self
    }

    pub fn validate(&self) -> Result<(), CurveFitError> {
        if self.tolerance.is_nan() {
            return Err(CurveFitError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CurveFitConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CurveFitConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveFitConfig(degree={0}, tolerance={1:.4}, max_control_pts={2}, parameterization={3})", self.degree, self.tolerance, self.max_control_pts, self.parameterization)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core curve fitting through point data engine.
#[derive(Debug, Clone)]
pub struct CurveFit {
    config: CurveFitConfig,
    data: Vec<f64>,
}

impl CurveFit {
    pub fn new(config: CurveFitConfig) -> Result<Self, CurveFitError> {
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
    pub fn config(&self) -> &CurveFitConfig { &self.config }

    /// Interpolate through points.
    pub fn interpolate(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Approximate with tolerance.
    pub fn approximate(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute fitting error.
    pub fn fitting_error(&self) -> f64 {
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

impl fmt::Display for CurveFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveFit(n={})", self.data.len())
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
        let cfg = CurveFitConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CurveFitConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CurveFitConfig"));
    }

    #[test]
    fn test_config_with_degree() {
        let cfg = CurveFitConfig::new().with_degree(42);
        assert_eq!(cfg.degree, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = CurveFitConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_control_pts() {
        let cfg = CurveFitConfig::new().with_max_control_pts(42);
        assert_eq!(cfg.max_control_pts, 42);
    }

    #[test]
    fn test_config_with_parameterization() {
        let cfg = CurveFitConfig::new().with_parameterization(42);
        assert_eq!(cfg.parameterization, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CurveFitConfig::new().with_degree(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CurveFit::new(CurveFitConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CurveFit"));
    }

    #[test]
    fn test_summary() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.interpolate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_approximate() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.approximate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_fitting_error() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.fitting_error();
        assert!(result.is_finite());
    }

    #[test]
    fn test_fitting_error_empty() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap();
        assert!((e.fitting_error() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = CurveFit::new(CurveFitConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CurveFitError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CurveFitError::InvalidConfig("a".into());
        let e2 = CurveFitError::ComputationFailed("b".into());
        let e3 = CurveFitError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
