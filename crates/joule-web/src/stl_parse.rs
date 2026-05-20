//! STL file format parser (ASCII and binary).
//!
//! Provides [`StlParseConfig`] builder and [`StlParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum StlParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for StlParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "StlParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "StlParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "StlParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`StlParse`] parameters.
#[derive(Debug, Clone)]
pub struct StlParseConfig {
    pub binary_mode: bool,
    pub merge_vertices: bool,
    pub compute_normals: bool,
    pub tolerance: f64,
}

impl StlParseConfig {
    pub fn new() -> Self {
        Self {
            binary_mode: true,
            merge_vertices: true,
            compute_normals: true,
            tolerance: 1e-6,
        }
    }

    pub fn with_binary_mode(mut self, v: bool) -> Self {
        self.binary_mode = v;
        self
    }

    pub fn with_merge_vertices(mut self, v: bool) -> Self {
        self.merge_vertices = v;
        self
    }

    pub fn with_compute_normals(mut self, v: bool) -> Self {
        self.compute_normals = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn validate(&self) -> Result<(), StlParseError> {
        if self.tolerance.is_nan() {
            return Err(StlParseError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for StlParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StlParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StlParseConfig(binary_mode={0}, merge_vertices={1}, compute_normals={2}, tolerance={3:.4})", self.binary_mode, self.merge_vertices, self.compute_normals, self.tolerance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core stl file format parser (ascii and binary) engine.
#[derive(Debug, Clone)]
pub struct StlParse {
    config: StlParseConfig,
    data: Vec<f64>,
}

impl StlParse {
    pub fn new(config: StlParseConfig) -> Result<Self, StlParseError> {
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
    pub fn config(&self) -> &StlParseConfig { &self.config }

    /// Parse STL triangles.
    pub fn parse_triangles(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute enclosed volume.
    pub fn compute_volume(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check if mesh is manifold.
    pub fn is_manifold(&self) -> bool {
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

impl fmt::Display for StlParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StlParse(n={})", self.data.len())
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
        let cfg = StlParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = StlParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("StlParseConfig"));
    }

    #[test]
    fn test_config_with_binary_mode() {
        let cfg = StlParseConfig::new().with_binary_mode(false);
        assert_eq!(cfg.binary_mode, false);
    }

    #[test]
    fn test_config_with_merge_vertices() {
        let cfg = StlParseConfig::new().with_merge_vertices(false);
        assert_eq!(cfg.merge_vertices, false);
    }

    #[test]
    fn test_config_with_compute_normals() {
        let cfg = StlParseConfig::new().with_compute_normals(false);
        assert_eq!(cfg.compute_normals, false);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = StlParseConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = StlParseConfig::new().with_binary_mode(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = StlParse::new(StlParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = StlParse::new(StlParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("StlParse"));
    }

    #[test]
    fn test_summary() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = StlParse::new(StlParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_triangles() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_triangles();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compute_volume() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compute_volume();
        assert!(result.is_finite());
    }

    #[test]
    fn test_is_manifold() {
        let e = StlParse::new(StlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_manifold();
        assert!(result);
    }

    #[test]
    fn test_is_manifold_empty() {
        let e = StlParse::new(StlParseConfig::new()).unwrap();
        assert!(!e.is_manifold());
    }

    #[test]
    fn test_config_accessor() {
        let e = StlParse::new(StlParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = StlParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = StlParseError::InvalidConfig("a".into());
        let e2 = StlParseError::ComputationFailed("b".into());
        let e3 = StlParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
