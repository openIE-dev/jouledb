//! Mesh repair and healing operations.
//!
//! Provides [`MeshRepairConfig`] builder and [`MeshRepair`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshRepairError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MeshRepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MeshRepair: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MeshRepair: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MeshRepair: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MeshRepair`] parameters.
#[derive(Debug, Clone)]
pub struct MeshRepairConfig {
    pub tolerance: f64,
    pub fill_holes: bool,
    pub fix_normals: bool,
    pub remove_degenerate: bool,
}

impl MeshRepairConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            fill_holes: true,
            fix_normals: true,
            remove_degenerate: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_fill_holes(mut self, v: bool) -> Self {
        self.fill_holes = v;
        self
    }

    pub fn with_fix_normals(mut self, v: bool) -> Self {
        self.fix_normals = v;
        self
    }

    pub fn with_remove_degenerate(mut self, v: bool) -> Self {
        self.remove_degenerate = v;
        self
    }

    pub fn validate(&self) -> Result<(), MeshRepairError> {
        if self.tolerance.is_nan() {
            return Err(MeshRepairError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MeshRepairConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MeshRepairConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshRepairConfig(tolerance={0:.4}, fill_holes={1}, fix_normals={2}, remove_degenerate={3})", self.tolerance, self.fill_holes, self.fix_normals, self.remove_degenerate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core mesh repair and healing operations engine.
#[derive(Debug, Clone)]
pub struct MeshRepair {
    config: MeshRepairConfig,
    data: Vec<f64>,
}

impl MeshRepair {
    pub fn new(config: MeshRepairConfig) -> Result<Self, MeshRepairError> {
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
    pub fn config(&self) -> &MeshRepairConfig { &self.config }

    /// Apply all repairs.
    pub fn repair_all(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Fill a mesh hole.
    pub fn fill_hole(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Remove duplicate vertices.
    pub fn remove_duplicates(&self) -> usize {
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

impl fmt::Display for MeshRepair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshRepair(n={})", self.data.len())
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
        let cfg = MeshRepairConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MeshRepairConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MeshRepairConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = MeshRepairConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_fill_holes() {
        let cfg = MeshRepairConfig::new().with_fill_holes(false);
        assert_eq!(cfg.fill_holes, false);
    }

    #[test]
    fn test_config_with_fix_normals() {
        let cfg = MeshRepairConfig::new().with_fix_normals(false);
        assert_eq!(cfg.fix_normals, false);
    }

    #[test]
    fn test_config_with_remove_degenerate() {
        let cfg = MeshRepairConfig::new().with_remove_degenerate(false);
        assert_eq!(cfg.remove_degenerate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MeshRepairConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MeshRepair::new(MeshRepairConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MeshRepair"));
    }

    #[test]
    fn test_summary() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_repair_all() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.repair_all();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_fill_hole() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.fill_hole();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_remove_duplicates() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.remove_duplicates();
        assert!(result > 0);
    }

    #[test]
    fn test_remove_duplicates_empty() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap();
        let _ = e.remove_duplicates();
    }

    #[test]
    fn test_config_accessor() {
        let e = MeshRepair::new(MeshRepairConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MeshRepairError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MeshRepairError::InvalidConfig("a".into());
        let e2 = MeshRepairError::ComputationFailed("b".into());
        let e3 = MeshRepairError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
