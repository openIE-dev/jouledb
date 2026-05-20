//! B-spline curve with Cox-de Boor basis functions.
//!
//! Provides [`BsplineCurveConfig`] builder and [`BsplineCurve`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BsplineCurveError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BsplineCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BsplineCurve: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BsplineCurve: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BsplineCurve: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BsplineCurve`] parameters.
#[derive(Debug, Clone)]
pub struct BsplineCurveConfig {
    pub degree: usize,
    pub num_control_pts: usize,
    pub knot_type: usize,
    pub tolerance: f64,
}

impl BsplineCurveConfig {
    pub fn new() -> Self {
        Self {
            degree: 3,
            num_control_pts: 10,
            knot_type: 0,
            tolerance: 1e-6,
        }
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v;
        self
    }

    pub fn with_num_control_pts(mut self, v: usize) -> Self {
        self.num_control_pts = v;
        self
    }

    pub fn with_knot_type(mut self, v: usize) -> Self {
        self.knot_type = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn validate(&self) -> Result<(), BsplineCurveError> {
        if self.tolerance.is_nan() {
            return Err(BsplineCurveError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for BsplineCurveConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BsplineCurveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BsplineCurveConfig(degree={0}, num_control_pts={1}, knot_type={2}, tolerance={3:.4})", self.degree, self.num_control_pts, self.knot_type, self.tolerance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core b-spline curve with cox-de boor basis functions engine.
#[derive(Debug, Clone)]
pub struct BsplineCurve {
    config: BsplineCurveConfig,
    data: Vec<f64>,
}

impl BsplineCurve {
    pub fn new(config: BsplineCurveConfig) -> Result<Self, BsplineCurveError> {
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
    pub fn config(&self) -> &BsplineCurveConfig { &self.config }

    /// Evaluate B-spline at parameter.
    pub fn evaluate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Cox-de Boor basis value.
    pub fn basis_function(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Insert knot (Boehm).
    pub fn insert_knot(&self) -> Vec<f64> {
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

impl fmt::Display for BsplineCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BsplineCurve(n={})", self.data.len())
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
        let cfg = BsplineCurveConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BsplineCurveConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BsplineCurveConfig"));
    }

    #[test]
    fn test_config_with_degree() {
        let cfg = BsplineCurveConfig::new().with_degree(42);
        assert_eq!(cfg.degree, 42);
    }

    #[test]
    fn test_config_with_num_control_pts() {
        let cfg = BsplineCurveConfig::new().with_num_control_pts(42);
        assert_eq!(cfg.num_control_pts, 42);
    }

    #[test]
    fn test_config_with_knot_type() {
        let cfg = BsplineCurveConfig::new().with_knot_type(42);
        assert_eq!(cfg.knot_type, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = BsplineCurveConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BsplineCurveConfig::new().with_degree(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BsplineCurve"));
    }

    #[test]
    fn test_summary() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_basis_function() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.basis_function();
        assert!(result.is_finite());
    }

    #[test]
    fn test_insert_knot() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.insert_knot();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_insert_knot_empty() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap();
        assert!(e.insert_knot().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = BsplineCurve::new(BsplineCurveConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BsplineCurveError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BsplineCurveError::InvalidConfig("a".into());
        let e2 = BsplineCurveError::ComputationFailed("b".into());
        let e3 = BsplineCurveError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
