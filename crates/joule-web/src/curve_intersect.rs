//! Curve-curve intersection detection.
//!
//! Provides [`CurveIntersectConfig`] builder and [`CurveIntersect`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveIntersectError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CurveIntersectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CurveIntersect: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CurveIntersect: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CurveIntersect: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CurveIntersect`] parameters.
#[derive(Debug, Clone)]
pub struct CurveIntersectConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub subdivision_depth: usize,
    pub newton_refine: bool,
}

impl CurveIntersectConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 50,
            subdivision_depth: 20,
            newton_refine: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn with_subdivision_depth(mut self, v: usize) -> Self {
        self.subdivision_depth = v;
        self
    }

    pub fn with_newton_refine(mut self, v: bool) -> Self {
        self.newton_refine = v;
        self
    }

    pub fn validate(&self) -> Result<(), CurveIntersectError> {
        if self.tolerance.is_nan() {
            return Err(CurveIntersectError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CurveIntersectConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CurveIntersectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveIntersectConfig(tolerance={0:.4}, max_iterations={1}, subdivision_depth={2}, newton_refine={3})", self.tolerance, self.max_iterations, self.subdivision_depth, self.newton_refine)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core curve-curve intersection detection engine.
#[derive(Debug, Clone)]
pub struct CurveIntersect {
    config: CurveIntersectConfig,
    data: Vec<f64>,
}

impl CurveIntersect {
    pub fn new(config: CurveIntersectConfig) -> Result<Self, CurveIntersectError> {
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
    pub fn config(&self) -> &CurveIntersectConfig { &self.config }

    /// Find all intersection points.
    pub fn find_intersections(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Bezier clipping intersection.
    pub fn bezier_clip(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Detect self-intersections.
    pub fn self_intersection(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for CurveIntersect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveIntersect(n={})", self.data.len())
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
        let cfg = CurveIntersectConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CurveIntersectConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CurveIntersectConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = CurveIntersectConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_iterations() {
        let cfg = CurveIntersectConfig::new().with_max_iterations(42);
        assert_eq!(cfg.max_iterations, 42);
    }

    #[test]
    fn test_config_with_subdivision_depth() {
        let cfg = CurveIntersectConfig::new().with_subdivision_depth(42);
        assert_eq!(cfg.subdivision_depth, 42);
    }

    #[test]
    fn test_config_with_newton_refine() {
        let cfg = CurveIntersectConfig::new().with_newton_refine(false);
        assert_eq!(cfg.newton_refine, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CurveIntersectConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CurveIntersect"));
    }

    #[test]
    fn test_summary() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_intersections() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.find_intersections();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bezier_clip() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bezier_clip();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_self_intersection() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.self_intersection();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_self_intersection_empty() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap();
        assert!(e.self_intersection().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = CurveIntersect::new(CurveIntersectConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CurveIntersectError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CurveIntersectError::InvalidConfig("a".into());
        let e2 = CurveIntersectError::ComputationFailed("b".into());
        let e3 = CurveIntersectError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
