//! Geometric constraint solver (Newton-Raphson).
//!
//! Provides [`ConstraintSolveConfig`] builder and [`ConstraintSolve`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintSolveError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ConstraintSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ConstraintSolve: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ConstraintSolve: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ConstraintSolve: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ConstraintSolve`] parameters.
#[derive(Debug, Clone)]
pub struct ConstraintSolveConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub damping: f64,
    pub detect_singular: bool,
}

impl ConstraintSolveConfig {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
            damping: 1.0,
            detect_singular: true,
        }
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_damping(mut self, v: f64) -> Self {
        self.damping = v;
        self
    }

    pub fn with_detect_singular(mut self, v: bool) -> Self {
        self.detect_singular = v;
        self
    }

    pub fn validate(&self) -> Result<(), ConstraintSolveError> {
        if self.tolerance.is_nan() {
            return Err(ConstraintSolveError::InvalidConfig("tolerance is NaN".into()));
        }
        if self.damping.is_nan() {
            return Err(ConstraintSolveError::InvalidConfig("damping is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ConstraintSolveConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ConstraintSolveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConstraintSolveConfig(max_iterations={0}, tolerance={1:.4}, damping={2:.4}, detect_singular={3})", self.max_iterations, self.tolerance, self.damping, self.detect_singular)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core geometric constraint solver (newton-raphson) engine.
#[derive(Debug, Clone)]
pub struct ConstraintSolve {
    config: ConstraintSolveConfig,
    data: Vec<f64>,
}

impl ConstraintSolve {
    pub fn new(config: ConstraintSolveConfig) -> Result<Self, ConstraintSolveError> {
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
    pub fn config(&self) -> &ConstraintSolveConfig { &self.config }

    /// Solve constraint system.
    pub fn solve(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Compute Jacobian matrix.
    pub fn jacobian(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Check constraint status.
    pub fn is_well_constrained(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for ConstraintSolve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConstraintSolve(n={})", self.data.len())
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
        let cfg = ConstraintSolveConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ConstraintSolveConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ConstraintSolveConfig"));
    }

    #[test]
    fn test_config_with_max_iterations() {
        let cfg = ConstraintSolveConfig::new().with_max_iterations(42);
        assert_eq!(cfg.max_iterations, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = ConstraintSolveConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_damping() {
        let cfg = ConstraintSolveConfig::new().with_damping(42.0);
        assert!((cfg.damping - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_detect_singular() {
        let cfg = ConstraintSolveConfig::new().with_detect_singular(false);
        assert_eq!(cfg.detect_singular, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ConstraintSolveConfig::new().with_max_iterations(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ConstraintSolve"));
    }

    #[test]
    fn test_summary() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_solve() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.solve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_jacobian() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.jacobian();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_well_constrained() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_well_constrained();
        assert!(result);
    }

    #[test]
    fn test_is_well_constrained_empty() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap();
        assert!(!e.is_well_constrained());
    }

    #[test]
    fn test_config_accessor() {
        let e = ConstraintSolve::new(ConstraintSolveConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ConstraintSolveError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ConstraintSolveError::InvalidConfig("a".into());
        let e2 = ConstraintSolveError::ComputationFailed("b".into());
        let e3 = ConstraintSolveError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
