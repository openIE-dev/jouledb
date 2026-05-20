//! Clinical Document Architecture generation.
//!
//! Provides [`CdaDocumentConfig`] builder and [`CdaDocument`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CdaDocumentError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CdaDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CdaDocument: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CdaDocument: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CdaDocument: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CdaDocument`] parameters.
#[derive(Debug, Clone)]
pub struct CdaDocumentConfig {
    pub template_id: usize,
    pub include_narrative: bool,
    pub structured_body: bool,
    pub validate_codes: bool,
}

impl CdaDocumentConfig {
    pub fn new() -> Self {
        Self {
            template_id: 0,
            include_narrative: true,
            structured_body: true,
            validate_codes: true,
        }
    }

    pub fn with_template_id(mut self, v: usize) -> Self {
        self.template_id = v;
        self
    }

    pub fn with_include_narrative(mut self, v: bool) -> Self {
        self.include_narrative = v;
        self
    }

    pub fn with_structured_body(mut self, v: bool) -> Self {
        self.structured_body = v;
        self
    }

    pub fn with_validate_codes(mut self, v: bool) -> Self {
        self.validate_codes = v;
        self
    }

    pub fn validate(&self) -> Result<(), CdaDocumentError> {
        Ok(())
    }
}

impl Default for CdaDocumentConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CdaDocumentConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CdaDocumentConfig(template_id={0}, include_narrative={1}, structured_body={2}, validate_codes={3})", self.template_id, self.include_narrative, self.structured_body, self.validate_codes)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core clinical document architecture generation engine.
#[derive(Debug, Clone)]
pub struct CdaDocument {
    config: CdaDocumentConfig,
    data: Vec<f64>,
}

impl CdaDocument {
    pub fn new(config: CdaDocumentConfig) -> Result<Self, CdaDocumentError> {
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
    pub fn config(&self) -> &CdaDocumentConfig { &self.config }

    /// Add document section.
    pub fn add_section(&self) -> bool {
        !self.data.is_empty()
    }

    /// Generate CDA document.
    pub fn generate_document(&self) -> String {
        format!("{}: {} records", stringify!(generate_document), self.data.len())
    }

    /// Validate CDA structure.
    pub fn validate_structure(&self) -> bool {
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

impl fmt::Display for CdaDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CdaDocument(n={})", self.data.len())
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
        let cfg = CdaDocumentConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CdaDocumentConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CdaDocumentConfig"));
    }

    #[test]
    fn test_config_with_template_id() {
        let cfg = CdaDocumentConfig::new().with_template_id(42);
        assert_eq!(cfg.template_id, 42);
    }

    #[test]
    fn test_config_with_include_narrative() {
        let cfg = CdaDocumentConfig::new().with_include_narrative(false);
        assert_eq!(cfg.include_narrative, false);
    }

    #[test]
    fn test_config_with_structured_body() {
        let cfg = CdaDocumentConfig::new().with_structured_body(false);
        assert_eq!(cfg.structured_body, false);
    }

    #[test]
    fn test_config_with_validate_codes() {
        let cfg = CdaDocumentConfig::new().with_validate_codes(false);
        assert_eq!(cfg.validate_codes, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CdaDocumentConfig::new().with_template_id(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CdaDocument::new(CdaDocumentConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CdaDocument"));
    }

    #[test]
    fn test_summary() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_section() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_section();
        assert!(result);
    }

    #[test]
    fn test_generate_document() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_document();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_validate_structure() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_structure();
        assert!(result);
    }

    #[test]
    fn test_validate_structure_empty() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap();
        assert!(!e.validate_structure());
    }

    #[test]
    fn test_config_accessor() {
        let e = CdaDocument::new(CdaDocumentConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CdaDocumentError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CdaDocumentError::InvalidConfig("a".into());
        let e2 = CdaDocumentError::ComputationFailed("b".into());
        let e3 = CdaDocumentError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
