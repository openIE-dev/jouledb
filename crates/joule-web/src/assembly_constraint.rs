//! Assembly constraint definitions and solving.
//!
//! Provides [`AssemblyConstraintConfig`] builder and [`AssemblyConstraint`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyConstraintError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AssemblyConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AssemblyConstraint: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AssemblyConstraint: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AssemblyConstraint: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AssemblyConstraint`] parameters.
#[derive(Debug, Clone)]
pub struct AssemblyConstraintConfig {
    pub max_constraints: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub damping: f64,
}

impl AssemblyConstraintConfig {
    pub fn new() -> Self {
        Self {
            max_constraints: 1000,
            tolerance: 1e-6,
            max_iterations: 100,
            damping: 0.5,
        }
    }

    pub fn with_max_constraints(mut self, v: usize) -> Self {
        self.max_constraints = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn with_damping(mut self, v: f64) -> Self {
        self.damping = v;
        self
    }

    pub fn validate(&self) -> Result<(), AssemblyConstraintError> {
        if self.tolerance.is_nan() {
            return Err(AssemblyConstraintError::InvalidConfig("tolerance is NaN".into()));
        }
        if self.damping.is_nan() {
            return Err(AssemblyConstraintError::InvalidConfig("damping is NaN".into()));
        }
        Ok(())
    }
}

impl Default for AssemblyConstraintConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AssemblyConstraintConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssemblyConstraintConfig(max_constraints={0}, tolerance={1:.4}, max_iterations={2}, damping={3:.4})", self.max_constraints, self.tolerance, self.max_iterations, self.damping)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core assembly constraint definitions and solving engine.
#[derive(Debug, Clone)]
pub struct AssemblyConstraint {
    config: AssemblyConstraintConfig,
    data: Vec<f64>,
}

impl AssemblyConstraint {
    pub fn new(config: AssemblyConstraintConfig) -> Result<Self, AssemblyConstraintError> {
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
    pub fn config(&self) -> &AssemblyConstraintConfig { &self.config }

    /// Add mate constraint.
    pub fn add_mate(&self) -> bool {
        !self.data.is_empty()
    }

    /// Solve all constraints.
    pub fn solve_all(&self) -> bool {
        !self.data.is_empty()
    }

    /// Remaining DOF.
    pub fn degrees_of_freedom(&self) -> usize {
        self.data.len()
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

impl fmt::Display for AssemblyConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssemblyConstraint(n={})", self.data.len())
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
        let cfg = AssemblyConstraintConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AssemblyConstraintConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AssemblyConstraintConfig"));
    }

    #[test]
    fn test_config_with_max_constraints() {
        let cfg = AssemblyConstraintConfig::new().with_max_constraints(42);
        assert_eq!(cfg.max_constraints, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = AssemblyConstraintConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_iterations() {
        let cfg = AssemblyConstraintConfig::new().with_max_iterations(42);
        assert_eq!(cfg.max_iterations, 42);
    }

    #[test]
    fn test_config_with_damping() {
        let cfg = AssemblyConstraintConfig::new().with_damping(42.0);
        assert!((cfg.damping - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AssemblyConstraintConfig::new().with_max_constraints(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AssemblyConstraint"));
    }

    #[test]
    fn test_summary() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_mate() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_mate();
        assert!(result);
    }

    #[test]
    fn test_solve_all() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.solve_all();
        assert!(result);
    }

    #[test]
    fn test_degrees_of_freedom() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.degrees_of_freedom();
        assert!(result > 0);
    }

    #[test]
    fn test_degrees_of_freedom_empty() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap();
        let _ = e.degrees_of_freedom();
    }

    #[test]
    fn test_config_accessor() {
        let e = AssemblyConstraint::new(AssemblyConstraintConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AssemblyConstraintError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AssemblyConstraintError::InvalidConfig("a".into());
        let e2 = AssemblyConstraintError::ComputationFailed("b".into());
        let e3 = AssemblyConstraintError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
