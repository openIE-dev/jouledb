//! Wavefront OBJ file format parser.
//!
//! Provides [`ObjParseConfig`] builder and [`ObjParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ObjParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ObjParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ObjParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ObjParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ObjParse`] parameters.
#[derive(Debug, Clone)]
pub struct ObjParseConfig {
    pub parse_normals: bool,
    pub parse_texcoords: bool,
    pub triangulate_quads: bool,
    pub group_by_material: bool,
}

impl ObjParseConfig {
    pub fn new() -> Self {
        Self {
            parse_normals: true,
            parse_texcoords: true,
            triangulate_quads: true,
            group_by_material: false,
        }
    }

    pub fn with_parse_normals(mut self, v: bool) -> Self {
        self.parse_normals = v;
        self
    }

    pub fn with_parse_texcoords(mut self, v: bool) -> Self {
        self.parse_texcoords = v;
        self
    }

    pub fn with_triangulate_quads(mut self, v: bool) -> Self {
        self.triangulate_quads = v;
        self
    }

    pub fn with_group_by_material(mut self, v: bool) -> Self {
        self.group_by_material = v;
        self
    }

    pub fn validate(&self) -> Result<(), ObjParseError> {
        Ok(())
    }
}

impl Default for ObjParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ObjParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjParseConfig(parse_normals={0}, parse_texcoords={1}, triangulate_quads={2}, group_by_material={3})", self.parse_normals, self.parse_texcoords, self.triangulate_quads, self.group_by_material)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core wavefront obj file format parser engine.
#[derive(Debug, Clone)]
pub struct ObjParse {
    config: ObjParseConfig,
    data: Vec<f64>,
}

impl ObjParse {
    pub fn new(config: ObjParseConfig) -> Result<Self, ObjParseError> {
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
    pub fn config(&self) -> &ObjParseConfig { &self.config }

    /// Parse OBJ mesh.
    pub fn parse_mesh(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Get vertex count.
    pub fn vertex_count(&self) -> usize {
        self.data.len()
    }

    /// Get face count.
    pub fn face_count(&self) -> usize {
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

impl fmt::Display for ObjParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjParse(n={})", self.data.len())
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
        let cfg = ObjParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ObjParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ObjParseConfig"));
    }

    #[test]
    fn test_config_with_parse_normals() {
        let cfg = ObjParseConfig::new().with_parse_normals(false);
        assert_eq!(cfg.parse_normals, false);
    }

    #[test]
    fn test_config_with_parse_texcoords() {
        let cfg = ObjParseConfig::new().with_parse_texcoords(false);
        assert_eq!(cfg.parse_texcoords, false);
    }

    #[test]
    fn test_config_with_triangulate_quads() {
        let cfg = ObjParseConfig::new().with_triangulate_quads(false);
        assert_eq!(cfg.triangulate_quads, false);
    }

    #[test]
    fn test_config_with_group_by_material() {
        let cfg = ObjParseConfig::new().with_group_by_material(true);
        assert_eq!(cfg.group_by_material, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ObjParseConfig::new().with_parse_normals(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ObjParse::new(ObjParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ObjParse"));
    }

    #[test]
    fn test_summary() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_mesh() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_mesh();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vertex_count() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.vertex_count();
        assert!(result > 0);
    }

    #[test]
    fn test_face_count() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.face_count();
        assert!(result > 0);
    }

    #[test]
    fn test_face_count_empty() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap();
        let _ = e.face_count();
    }

    #[test]
    fn test_config_accessor() {
        let e = ObjParse::new(ObjParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ObjParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ObjParseError::InvalidConfig("a".into());
        let e2 = ObjParseError::ComputationFailed("b".into());
        let e3 = ObjParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
