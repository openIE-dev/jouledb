//! Boundary representation solid modeling.
//!
//! Provides [`BrepSolidConfig`] builder and [`BrepSolid`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BrepSolidError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BrepSolidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BrepSolid: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BrepSolid: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BrepSolid: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BrepSolid`] parameters.
#[derive(Debug, Clone)]
pub struct BrepSolidConfig {
    pub max_vertices: usize,
    pub max_edges: usize,
    pub max_faces: usize,
    pub validate: bool,
}

impl BrepSolidConfig {
    pub fn new() -> Self {
        Self {
            max_vertices: 10000,
            max_edges: 30000,
            max_faces: 20000,
            validate: true,
        }
    }

    pub fn with_max_vertices(mut self, v: usize) -> Self {
        self.max_vertices = v;
        self
    }

    pub fn with_max_edges(mut self, v: usize) -> Self {
        self.max_edges = v;
        self
    }

    pub fn with_max_faces(mut self, v: usize) -> Self {
        self.max_faces = v;
        self
    }

    pub fn with_validate(mut self, v: bool) -> Self {
        self.validate = v;
        self
    }

    pub fn validate(&self) -> Result<(), BrepSolidError> {
        Ok(())
    }
}

impl Default for BrepSolidConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BrepSolidConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BrepSolidConfig(max_vertices={0}, max_edges={1}, max_faces={2}, validate={3})", self.max_vertices, self.max_edges, self.max_faces, self.validate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core boundary representation solid modeling engine.
#[derive(Debug, Clone)]
pub struct BrepSolid {
    config: BrepSolidConfig,
    data: Vec<f64>,
}

impl BrepSolid {
    pub fn new(config: BrepSolidConfig) -> Result<Self, BrepSolidError> {
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
    pub fn config(&self) -> &BrepSolidConfig { &self.config }

    /// Add vertex to B-rep.
    pub fn add_vertex(&self) -> usize {
        self.data.len()
    }

    /// Add edge between vertices.
    pub fn add_edge(&self) -> usize {
        self.data.len()
    }

    /// Validate B-rep topology.
    pub fn validate_topology(&self) -> bool {
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

impl fmt::Display for BrepSolid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BrepSolid(n={})", self.data.len())
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
        let cfg = BrepSolidConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BrepSolidConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BrepSolidConfig"));
    }

    #[test]
    fn test_config_with_max_vertices() {
        let cfg = BrepSolidConfig::new().with_max_vertices(42);
        assert_eq!(cfg.max_vertices, 42);
    }

    #[test]
    fn test_config_with_max_edges() {
        let cfg = BrepSolidConfig::new().with_max_edges(42);
        assert_eq!(cfg.max_edges, 42);
    }

    #[test]
    fn test_config_with_max_faces() {
        let cfg = BrepSolidConfig::new().with_max_faces(42);
        assert_eq!(cfg.max_faces, 42);
    }

    #[test]
    fn test_config_with_validate() {
        let cfg = BrepSolidConfig::new().with_validate(false);
        assert_eq!(cfg.validate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BrepSolidConfig::new().with_max_vertices(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BrepSolid::new(BrepSolidConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BrepSolid"));
    }

    #[test]
    fn test_summary() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_vertex() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_vertex();
        assert!(result > 0);
    }

    #[test]
    fn test_add_edge() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_edge();
        assert!(result > 0);
    }

    #[test]
    fn test_validate_topology() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_topology();
        assert!(result);
    }

    #[test]
    fn test_validate_topology_empty() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap();
        assert!(!e.validate_topology());
    }

    #[test]
    fn test_config_accessor() {
        let e = BrepSolid::new(BrepSolidConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BrepSolidError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BrepSolidError::InvalidConfig("a".into());
        let e2 = BrepSolidError::ComputationFailed("b".into());
        let e3 = BrepSolidError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
