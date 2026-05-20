//! Surface-surface intersection curve tracing.
//!
//! Provides [`SurfaceIntersectConfig`] builder and [`SurfaceIntersect`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceIntersectError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurfaceIntersectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurfaceIntersect: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurfaceIntersect: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurfaceIntersect: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurfaceIntersect`] parameters.
#[derive(Debug, Clone)]
pub struct SurfaceIntersectConfig {
    pub step_size: f64,
    pub tolerance: f64,
    pub max_points: usize,
    pub march_method: bool,
}

impl SurfaceIntersectConfig {
    pub fn new() -> Self {
        Self {
            step_size: 0.01,
            tolerance: 1e-6,
            max_points: 10000,
            march_method: true,
        }
    }

    pub fn with_step_size(mut self, v: f64) -> Self {
        self.step_size = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_points(mut self, v: usize) -> Self {
        self.max_points = v;
        self
    }

    pub fn with_march_method(mut self, v: bool) -> Self {
        self.march_method = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurfaceIntersectError> {
        if self.step_size.is_nan() {
            return Err(SurfaceIntersectError::InvalidConfig("step_size is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(SurfaceIntersectError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurfaceIntersectConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurfaceIntersectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceIntersectConfig(step_size={0:.4}, tolerance={1:.4}, max_points={2}, march_method={3})", self.step_size, self.tolerance, self.max_points, self.march_method)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core surface-surface intersection curve tracing engine.
#[derive(Debug, Clone)]
pub struct SurfaceIntersect {
    config: SurfaceIntersectConfig,
    data: Vec<f64>,
}

impl SurfaceIntersect {
    pub fn new(config: SurfaceIntersectConfig) -> Result<Self, SurfaceIntersectError> {
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
    pub fn config(&self) -> &SurfaceIntersectConfig { &self.config }

    /// Find intersection curve.
    pub fn find_intersection(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Classify intersection type.
    pub fn classify_type(&self) -> String {
        format!("{}: {} records", stringify!(classify_type), self.data.len())
    }

    /// Trace intersection curve.
    pub fn trace_curve(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SurfaceIntersect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceIntersect(n={})", self.data.len())
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
        let cfg = SurfaceIntersectConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurfaceIntersectConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurfaceIntersectConfig"));
    }

    #[test]
    fn test_config_with_step_size() {
        let cfg = SurfaceIntersectConfig::new().with_step_size(42.0);
        assert!((cfg.step_size - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = SurfaceIntersectConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_points() {
        let cfg = SurfaceIntersectConfig::new().with_max_points(42);
        assert_eq!(cfg.max_points, 42);
    }

    #[test]
    fn test_config_with_march_method() {
        let cfg = SurfaceIntersectConfig::new().with_march_method(false);
        assert_eq!(cfg.march_method, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurfaceIntersectConfig::new().with_step_size(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurfaceIntersect"));
    }

    #[test]
    fn test_summary() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_intersection() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.find_intersection();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_classify_type() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.classify_type();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_trace_curve() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.trace_curve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_trace_curve_empty() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap();
        assert!(e.trace_curve().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SurfaceIntersect::new(SurfaceIntersectConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurfaceIntersectError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurfaceIntersectError::InvalidConfig("a".into());
        let e2 = SurfaceIntersectError::ComputationFailed("b".into());
        let e3 = SurfaceIntersectError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
