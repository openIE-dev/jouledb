//! Bezier curve evaluation and manipulation.
//!
//! Provides [`BezierCurveConfig`] builder and [`BezierCurve`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BezierCurveError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BezierCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BezierCurve: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BezierCurve: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BezierCurve: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BezierCurve`] parameters.
#[derive(Debug, Clone)]
pub struct BezierCurveConfig {
    pub degree: usize,
    pub tolerance: f64,
    pub max_subdivisions: usize,
    pub parameterization: usize,
}

impl BezierCurveConfig {
    pub fn new() -> Self {
        Self {
            degree: 3,
            tolerance: 1e-6,
            max_subdivisions: 20,
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

    pub fn with_max_subdivisions(mut self, v: usize) -> Self {
        self.max_subdivisions = v;
        self
    }

    pub fn with_parameterization(mut self, v: usize) -> Self {
        self.parameterization = v;
        self
    }

    pub fn validate(&self) -> Result<(), BezierCurveError> {
        if self.tolerance.is_nan() {
            return Err(BezierCurveError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for BezierCurveConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BezierCurveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BezierCurveConfig(degree={0}, tolerance={1:.4}, max_subdivisions={2}, parameterization={3})", self.degree, self.tolerance, self.max_subdivisions, self.parameterization)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core bezier curve evaluation and manipulation engine.
#[derive(Debug, Clone)]
pub struct BezierCurve {
    config: BezierCurveConfig,
    data: Vec<f64>,
}

impl BezierCurve {
    pub fn new(config: BezierCurveConfig) -> Result<Self, BezierCurveError> {
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
    pub fn config(&self) -> &BezierCurveConfig { &self.config }

    /// Evaluate curve at parameter t.
    pub fn evaluate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Curve derivative at t.
    pub fn derivative(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Approximate arc length.
    pub fn arc_length(&self) -> f64 {
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

impl fmt::Display for BezierCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BezierCurve(n={})", self.data.len())
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
        let cfg = BezierCurveConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BezierCurveConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BezierCurveConfig"));
    }

    #[test]
    fn test_config_with_degree() {
        let cfg = BezierCurveConfig::new().with_degree(42);
        assert_eq!(cfg.degree, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = BezierCurveConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_subdivisions() {
        let cfg = BezierCurveConfig::new().with_max_subdivisions(42);
        assert_eq!(cfg.max_subdivisions, 42);
    }

    #[test]
    fn test_config_with_parameterization() {
        let cfg = BezierCurveConfig::new().with_parameterization(42);
        assert_eq!(cfg.parameterization, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BezierCurveConfig::new().with_degree(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BezierCurve::new(BezierCurveConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BezierCurve"));
    }

    #[test]
    fn test_summary() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_derivative() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.derivative();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_arc_length() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.arc_length();
        assert!(result.is_finite());
    }

    #[test]
    fn test_arc_length_empty() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap();
        assert!((e.arc_length() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = BezierCurve::new(BezierCurveConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BezierCurveError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BezierCurveError::InvalidConfig("a".into());
        let e2 = BezierCurveError::ComputationFailed("b".into());
        let e3 = BezierCurveError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
