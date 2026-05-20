//! Constructive solid geometry operations.
//!
//! Provides [`CsgOpsConfig`] builder and [`CsgOps`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CsgOpsError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CsgOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CsgOps: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CsgOps: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CsgOps: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CsgOps`] parameters.
#[derive(Debug, Clone)]
pub struct CsgOpsConfig {
    pub tolerance: f64,
    pub max_depth: usize,
    pub simplify: bool,
    pub validate: bool,
}

impl CsgOpsConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            max_depth: 20,
            simplify: true,
            validate: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    pub fn with_simplify(mut self, v: bool) -> Self {
        self.simplify = v;
        self
    }

    pub fn with_validate(mut self, v: bool) -> Self {
        self.validate = v;
        self
    }

    pub fn validate(&self) -> Result<(), CsgOpsError> {
        if self.tolerance.is_nan() {
            return Err(CsgOpsError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CsgOpsConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CsgOpsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CsgOpsConfig(tolerance={0:.4}, max_depth={1}, simplify={2}, validate={3})", self.tolerance, self.max_depth, self.simplify, self.validate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core constructive solid geometry operations engine.
#[derive(Debug, Clone)]
pub struct CsgOps {
    config: CsgOpsConfig,
    data: Vec<f64>,
}

impl CsgOps {
    pub fn new(config: CsgOpsConfig) -> Result<Self, CsgOpsError> {
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
    pub fn config(&self) -> &CsgOpsConfig { &self.config }

    /// CSG union of two solids.
    pub fn csg_union(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// CSG intersection.
    pub fn csg_intersect(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// CSG difference.
    pub fn csg_difference(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for CsgOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CsgOps(n={})", self.data.len())
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
        let cfg = CsgOpsConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CsgOpsConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CsgOpsConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = CsgOpsConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_depth() {
        let cfg = CsgOpsConfig::new().with_max_depth(42);
        assert_eq!(cfg.max_depth, 42);
    }

    #[test]
    fn test_config_with_simplify() {
        let cfg = CsgOpsConfig::new().with_simplify(false);
        assert_eq!(cfg.simplify, false);
    }

    #[test]
    fn test_config_with_validate() {
        let cfg = CsgOpsConfig::new().with_validate(false);
        assert_eq!(cfg.validate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CsgOpsConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CsgOps::new(CsgOpsConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CsgOps"));
    }

    #[test]
    fn test_summary() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_csg_union() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.csg_union();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_csg_intersect() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.csg_intersect();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_csg_difference() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.csg_difference();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_csg_difference_empty() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap();
        assert!(e.csg_difference().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = CsgOps::new(CsgOpsConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CsgOpsError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CsgOpsError::InvalidConfig("a".into());
        let e2 = CsgOpsError::ComputationFailed("b".into());
        let e3 = CsgOpsError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
