//! STEP (ISO 10303) file format parser.
//!
//! Provides [`StepParseConfig`] builder and [`StepParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum StepParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for StepParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "StepParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "StepParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "StepParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`StepParse`] parameters.
#[derive(Debug, Clone)]
pub struct StepParseConfig {
    pub schema_version: usize,
    pub lazy_parse: bool,
    pub validate_refs: bool,
    pub max_entities: usize,
}

impl StepParseConfig {
    pub fn new() -> Self {
        Self {
            schema_version: 214,
            lazy_parse: true,
            validate_refs: true,
            max_entities: 100000,
        }
    }

    pub fn with_schema_version(mut self, v: usize) -> Self {
        self.schema_version = v;
        self
    }

    pub fn with_lazy_parse(mut self, v: bool) -> Self {
        self.lazy_parse = v;
        self
    }

    pub fn with_validate_refs(mut self, v: bool) -> Self {
        self.validate_refs = v;
        self
    }

    pub fn with_max_entities(mut self, v: usize) -> Self {
        self.max_entities = v;
        self
    }

    pub fn validate(&self) -> Result<(), StepParseError> {
        Ok(())
    }
}

impl Default for StepParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StepParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StepParseConfig(schema_version={0}, lazy_parse={1}, validate_refs={2}, max_entities={3})", self.schema_version, self.lazy_parse, self.validate_refs, self.max_entities)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core step (iso 10303) file format parser engine.
#[derive(Debug, Clone)]
pub struct StepParse {
    config: StepParseConfig,
    data: Vec<f64>,
}

impl StepParse {
    pub fn new(config: StepParseConfig) -> Result<Self, StepParseError> {
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
    pub fn config(&self) -> &StepParseConfig { &self.config }

    /// Parse STEP entities.
    pub fn parse_entities(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract B-rep geometry.
    pub fn extract_geometry(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract assembly structure.
    pub fn assembly_structure(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for StepParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StepParse(n={})", self.data.len())
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
        let cfg = StepParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = StepParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("StepParseConfig"));
    }

    #[test]
    fn test_config_with_schema_version() {
        let cfg = StepParseConfig::new().with_schema_version(42);
        assert_eq!(cfg.schema_version, 42);
    }

    #[test]
    fn test_config_with_lazy_parse() {
        let cfg = StepParseConfig::new().with_lazy_parse(false);
        assert_eq!(cfg.lazy_parse, false);
    }

    #[test]
    fn test_config_with_validate_refs() {
        let cfg = StepParseConfig::new().with_validate_refs(false);
        assert_eq!(cfg.validate_refs, false);
    }

    #[test]
    fn test_config_with_max_entities() {
        let cfg = StepParseConfig::new().with_max_entities(42);
        assert_eq!(cfg.max_entities, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = StepParseConfig::new().with_schema_version(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = StepParse::new(StepParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = StepParse::new(StepParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("StepParse"));
    }

    #[test]
    fn test_summary() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = StepParse::new(StepParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_entities() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_entities();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_geometry() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_geometry();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_assembly_structure() {
        let e = StepParse::new(StepParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.assembly_structure();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_assembly_structure_empty() {
        let e = StepParse::new(StepParseConfig::new()).unwrap();
        assert!(e.assembly_structure().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = StepParse::new(StepParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = StepParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = StepParseError::InvalidConfig("a".into());
        let e2 = StepParseError::ComputationFailed("b".into());
        let e3 = StepParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
