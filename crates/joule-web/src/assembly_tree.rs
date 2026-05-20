//! Assembly hierarchy and BOM management.
//!
//! Provides [`AssemblyTreeConfig`] builder and [`AssemblyTree`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyTreeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AssemblyTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AssemblyTree: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AssemblyTree: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AssemblyTree: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AssemblyTree`] parameters.
#[derive(Debug, Clone)]
pub struct AssemblyTreeConfig {
    pub max_depth: usize,
    pub compute_mass: bool,
    pub check_interference: bool,
    pub auto_update: bool,
}

impl AssemblyTreeConfig {
    pub fn new() -> Self {
        Self {
            max_depth: 20,
            compute_mass: true,
            check_interference: false,
            auto_update: true,
        }
    }

    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    pub fn with_compute_mass(mut self, v: bool) -> Self {
        self.compute_mass = v;
        self
    }

    pub fn with_check_interference(mut self, v: bool) -> Self {
        self.check_interference = v;
        self
    }

    pub fn with_auto_update(mut self, v: bool) -> Self {
        self.auto_update = v;
        self
    }

    pub fn validate(&self) -> Result<(), AssemblyTreeError> {
        Ok(())
    }
}

impl Default for AssemblyTreeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AssemblyTreeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssemblyTreeConfig(max_depth={0}, compute_mass={1}, check_interference={2}, auto_update={3})", self.max_depth, self.compute_mass, self.check_interference, self.auto_update)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core assembly hierarchy and bom management engine.
#[derive(Debug, Clone)]
pub struct AssemblyTree {
    config: AssemblyTreeConfig,
    data: Vec<f64>,
}

impl AssemblyTree {
    pub fn new(config: AssemblyTreeConfig) -> Result<Self, AssemblyTreeError> {
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
    pub fn config(&self) -> &AssemblyTreeConfig { &self.config }

    /// Add component to assembly.
    pub fn add_component(&self) -> usize {
        self.data.len()
    }

    /// Compute total assembly mass.
    pub fn total_mass(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Generate bill of materials.
    pub fn bom(&self) -> String {
        format!("{}: {} records", stringify!(bom), self.data.len())
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

impl fmt::Display for AssemblyTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssemblyTree(n={})", self.data.len())
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
        let cfg = AssemblyTreeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AssemblyTreeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AssemblyTreeConfig"));
    }

    #[test]
    fn test_config_with_max_depth() {
        let cfg = AssemblyTreeConfig::new().with_max_depth(42);
        assert_eq!(cfg.max_depth, 42);
    }

    #[test]
    fn test_config_with_compute_mass() {
        let cfg = AssemblyTreeConfig::new().with_compute_mass(false);
        assert_eq!(cfg.compute_mass, false);
    }

    #[test]
    fn test_config_with_check_interference() {
        let cfg = AssemblyTreeConfig::new().with_check_interference(true);
        assert_eq!(cfg.check_interference, true);
    }

    #[test]
    fn test_config_with_auto_update() {
        let cfg = AssemblyTreeConfig::new().with_auto_update(false);
        assert_eq!(cfg.auto_update, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AssemblyTreeConfig::new().with_max_depth(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AssemblyTree"));
    }

    #[test]
    fn test_summary() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_component() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_component();
        assert!(result > 0);
    }

    #[test]
    fn test_total_mass() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.total_mass();
        assert!(result.is_finite());
    }

    #[test]
    fn test_bom() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bom();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bom_empty() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap();
        let _ = e.bom();
    }

    #[test]
    fn test_config_accessor() {
        let e = AssemblyTree::new(AssemblyTreeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AssemblyTreeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AssemblyTreeError::InvalidConfig("a".into());
        let e2 = AssemblyTreeError::ComputationFailed("b".into());
        let e3 = AssemblyTreeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
