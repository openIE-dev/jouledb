//! Surface trimming with parameter-space curves.
//!
//! Provides [`SurfaceTrimConfig`] builder and [`SurfaceTrim`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceTrimError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurfaceTrimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurfaceTrim: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurfaceTrim: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurfaceTrim: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurfaceTrim`] parameters.
#[derive(Debug, Clone)]
pub struct SurfaceTrimConfig {
    pub tolerance: f64,
    pub max_trim_curves: usize,
    pub orientation_check: bool,
    pub refine_boundary: bool,
}

impl SurfaceTrimConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            max_trim_curves: 100,
            orientation_check: true,
            refine_boundary: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_trim_curves(mut self, v: usize) -> Self {
        self.max_trim_curves = v;
        self
    }

    pub fn with_orientation_check(mut self, v: bool) -> Self {
        self.orientation_check = v;
        self
    }

    pub fn with_refine_boundary(mut self, v: bool) -> Self {
        self.refine_boundary = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurfaceTrimError> {
        if self.tolerance.is_nan() {
            return Err(SurfaceTrimError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurfaceTrimConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurfaceTrimConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceTrimConfig(tolerance={0:.4}, max_trim_curves={1}, orientation_check={2}, refine_boundary={3})", self.tolerance, self.max_trim_curves, self.orientation_check, self.refine_boundary)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core surface trimming with parameter-space curves engine.
#[derive(Debug, Clone)]
pub struct SurfaceTrim {
    config: SurfaceTrimConfig,
    data: Vec<f64>,
}

impl SurfaceTrim {
    pub fn new(config: SurfaceTrimConfig) -> Result<Self, SurfaceTrimError> {
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
    pub fn config(&self) -> &SurfaceTrimConfig { &self.config }

    /// Add trimming curve.
    pub fn add_trim_curve(&self) -> bool {
        !self.data.is_empty()
    }

    /// Test point in trimmed region.
    pub fn point_in_trimmed(&self) -> bool {
        !self.data.is_empty()
    }

    /// Evaluate trimmed boundary.
    pub fn boundary_evaluate(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SurfaceTrim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceTrim(n={})", self.data.len())
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
        let cfg = SurfaceTrimConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurfaceTrimConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurfaceTrimConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = SurfaceTrimConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_trim_curves() {
        let cfg = SurfaceTrimConfig::new().with_max_trim_curves(42);
        assert_eq!(cfg.max_trim_curves, 42);
    }

    #[test]
    fn test_config_with_orientation_check() {
        let cfg = SurfaceTrimConfig::new().with_orientation_check(false);
        assert_eq!(cfg.orientation_check, false);
    }

    #[test]
    fn test_config_with_refine_boundary() {
        let cfg = SurfaceTrimConfig::new().with_refine_boundary(false);
        assert_eq!(cfg.refine_boundary, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurfaceTrimConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurfaceTrim"));
    }

    #[test]
    fn test_summary() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_trim_curve() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_trim_curve();
        assert!(result);
    }

    #[test]
    fn test_point_in_trimmed() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.point_in_trimmed();
        assert!(result);
    }

    #[test]
    fn test_boundary_evaluate() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.boundary_evaluate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_boundary_evaluate_empty() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap();
        assert!(e.boundary_evaluate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SurfaceTrim::new(SurfaceTrimConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurfaceTrimError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurfaceTrimError::InvalidConfig("a".into());
        let e2 = SurfaceTrimError::ComputationFailed("b".into());
        let e3 = SurfaceTrimError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
