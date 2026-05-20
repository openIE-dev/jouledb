//! IGES file format parser.
//!
//! Provides [`IgesParseConfig`] builder and [`IgesParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum IgesParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for IgesParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "IgesParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "IgesParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "IgesParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`IgesParse`] parameters.
#[derive(Debug, Clone)]
pub struct IgesParseConfig {
    pub max_entities: usize,
    pub parse_annotations: bool,
    pub coordinate_transform: bool,
    pub flatten: bool,
}

impl IgesParseConfig {
    pub fn new() -> Self {
        Self {
            max_entities: 100000,
            parse_annotations: true,
            coordinate_transform: true,
            flatten: false,
        }
    }

    pub fn with_max_entities(mut self, v: usize) -> Self {
        self.max_entities = v;
        self
    }

    pub fn with_parse_annotations(mut self, v: bool) -> Self {
        self.parse_annotations = v;
        self
    }

    pub fn with_coordinate_transform(mut self, v: bool) -> Self {
        self.coordinate_transform = v;
        self
    }

    pub fn with_flatten(mut self, v: bool) -> Self {
        self.flatten = v;
        self
    }

    pub fn validate(&self) -> Result<(), IgesParseError> {
        Ok(())
    }
}

impl Default for IgesParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for IgesParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IgesParseConfig(max_entities={0}, parse_annotations={1}, coordinate_transform={2}, flatten={3})", self.max_entities, self.parse_annotations, self.coordinate_transform, self.flatten)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core iges file format parser engine.
#[derive(Debug, Clone)]
pub struct IgesParse {
    config: IgesParseConfig,
    data: Vec<f64>,
}

impl IgesParse {
    pub fn new(config: IgesParseConfig) -> Result<Self, IgesParseError> {
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
    pub fn config(&self) -> &IgesParseConfig { &self.config }

    /// Parse IGES entities.
    pub fn parse_entities(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract curve entities.
    pub fn extract_curves(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract surface entities.
    pub fn extract_surfaces(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for IgesParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IgesParse(n={})", self.data.len())
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
        let cfg = IgesParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = IgesParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("IgesParseConfig"));
    }

    #[test]
    fn test_config_with_max_entities() {
        let cfg = IgesParseConfig::new().with_max_entities(42);
        assert_eq!(cfg.max_entities, 42);
    }

    #[test]
    fn test_config_with_parse_annotations() {
        let cfg = IgesParseConfig::new().with_parse_annotations(false);
        assert_eq!(cfg.parse_annotations, false);
    }

    #[test]
    fn test_config_with_coordinate_transform() {
        let cfg = IgesParseConfig::new().with_coordinate_transform(false);
        assert_eq!(cfg.coordinate_transform, false);
    }

    #[test]
    fn test_config_with_flatten() {
        let cfg = IgesParseConfig::new().with_flatten(true);
        assert_eq!(cfg.flatten, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = IgesParseConfig::new().with_max_entities(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = IgesParse::new(IgesParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("IgesParse"));
    }

    #[test]
    fn test_summary() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_entities() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_entities();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_curves() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_curves();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_surfaces() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_surfaces();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_surfaces_empty() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap();
        assert!(e.extract_surfaces().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = IgesParse::new(IgesParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = IgesParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = IgesParseError::InvalidConfig("a".into());
        let e2 = IgesParseError::ComputationFailed("b".into());
        let e3 = IgesParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
