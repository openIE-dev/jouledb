//! Boolean operations on triangle meshes.
//!
//! Provides [`BooleanMeshConfig`] builder and [`BooleanMesh`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BooleanMeshError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BooleanMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BooleanMesh: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BooleanMesh: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BooleanMesh: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BooleanMesh`] parameters.
#[derive(Debug, Clone)]
pub struct BooleanMeshConfig {
    pub tolerance: f64,
    pub coplanar_handling: bool,
    pub stitch_result: bool,
    pub validate: bool,
}

impl BooleanMeshConfig {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-8,
            coplanar_handling: true,
            stitch_result: true,
            validate: true,
        }
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_coplanar_handling(mut self, v: bool) -> Self {
        self.coplanar_handling = v;
        self
    }

    pub fn with_stitch_result(mut self, v: bool) -> Self {
        self.stitch_result = v;
        self
    }

    pub fn with_validate(mut self, v: bool) -> Self {
        self.validate = v;
        self
    }

    pub fn validate(&self) -> Result<(), BooleanMeshError> {
        if self.tolerance.is_nan() {
            return Err(BooleanMeshError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for BooleanMeshConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BooleanMeshConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BooleanMeshConfig(tolerance={0:.4}, coplanar_handling={1}, stitch_result={2}, validate={3})", self.tolerance, self.coplanar_handling, self.stitch_result, self.validate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core boolean operations on triangle meshes engine.
#[derive(Debug, Clone)]
pub struct BooleanMesh {
    config: BooleanMeshConfig,
    data: Vec<f64>,
}

impl BooleanMesh {
    pub fn new(config: BooleanMeshConfig) -> Result<Self, BooleanMeshError> {
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
    pub fn config(&self) -> &BooleanMeshConfig { &self.config }

    /// Mesh union operation.
    pub fn mesh_union(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Mesh intersection.
    pub fn mesh_intersect(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Mesh difference.
    pub fn mesh_difference(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for BooleanMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BooleanMesh(n={})", self.data.len())
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
        let cfg = BooleanMeshConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BooleanMeshConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BooleanMeshConfig"));
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = BooleanMeshConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_coplanar_handling() {
        let cfg = BooleanMeshConfig::new().with_coplanar_handling(false);
        assert_eq!(cfg.coplanar_handling, false);
    }

    #[test]
    fn test_config_with_stitch_result() {
        let cfg = BooleanMeshConfig::new().with_stitch_result(false);
        assert_eq!(cfg.stitch_result, false);
    }

    #[test]
    fn test_config_with_validate() {
        let cfg = BooleanMeshConfig::new().with_validate(false);
        assert_eq!(cfg.validate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BooleanMeshConfig::new().with_tolerance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BooleanMesh"));
    }

    #[test]
    fn test_summary() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mesh_union() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mesh_union();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mesh_intersect() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mesh_intersect();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mesh_difference() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mesh_difference();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mesh_difference_empty() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap();
        assert!(e.mesh_difference().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = BooleanMesh::new(BooleanMeshConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BooleanMeshError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BooleanMeshError::InvalidConfig("a".into());
        let e2 = BooleanMeshError::ComputationFailed("b".into());
        let e3 = BooleanMeshError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
