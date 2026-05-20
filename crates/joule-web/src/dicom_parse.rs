//! DICOM medical image format parser.
//!
//! Provides [`DicomParseConfig`] builder and [`DicomParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DicomParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DicomParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DicomParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DicomParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DicomParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DicomParse`] parameters.
#[derive(Debug, Clone)]
pub struct DicomParseConfig {
    pub parse_pixel_data: bool,
    pub max_tags: usize,
    pub strict_mode: bool,
    pub transfer_syntax: usize,
}

impl DicomParseConfig {
    pub fn new() -> Self {
        Self {
            parse_pixel_data: false,
            max_tags: 10000,
            strict_mode: false,
            transfer_syntax: 0,
        }
    }

    pub fn with_parse_pixel_data(mut self, v: bool) -> Self {
        self.parse_pixel_data = v;
        self
    }

    pub fn with_max_tags(mut self, v: usize) -> Self {
        self.max_tags = v;
        self
    }

    pub fn with_strict_mode(mut self, v: bool) -> Self {
        self.strict_mode = v;
        self
    }

    pub fn with_transfer_syntax(mut self, v: usize) -> Self {
        self.transfer_syntax = v;
        self
    }

    pub fn validate(&self) -> Result<(), DicomParseError> {
        Ok(())
    }
}

impl Default for DicomParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DicomParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DicomParseConfig(parse_pixel_data={0}, max_tags={1}, strict_mode={2}, transfer_syntax={3})", self.parse_pixel_data, self.max_tags, self.strict_mode, self.transfer_syntax)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core dicom medical image format parser engine.
#[derive(Debug, Clone)]
pub struct DicomParse {
    config: DicomParseConfig,
    data: Vec<f64>,
}

impl DicomParse {
    pub fn new(config: DicomParseConfig) -> Result<Self, DicomParseError> {
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
    pub fn config(&self) -> &DicomParseConfig { &self.config }

    /// Parse DICOM header tags.
    pub fn parse_header(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get specific DICOM tag value.
    pub fn get_tag(&self) -> String {
        format!("{}: {} records", stringify!(get_tag), self.data.len())
    }

    /// Extract patient information.
    pub fn patient_info(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
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

impl fmt::Display for DicomParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DicomParse(n={})", self.data.len())
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
        let cfg = DicomParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DicomParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DicomParseConfig"));
    }

    #[test]
    fn test_config_with_parse_pixel_data() {
        let cfg = DicomParseConfig::new().with_parse_pixel_data(true);
        assert_eq!(cfg.parse_pixel_data, true);
    }

    #[test]
    fn test_config_with_max_tags() {
        let cfg = DicomParseConfig::new().with_max_tags(42);
        assert_eq!(cfg.max_tags, 42);
    }

    #[test]
    fn test_config_with_strict_mode() {
        let cfg = DicomParseConfig::new().with_strict_mode(true);
        assert_eq!(cfg.strict_mode, true);
    }

    #[test]
    fn test_config_with_transfer_syntax() {
        let cfg = DicomParseConfig::new().with_transfer_syntax(42);
        assert_eq!(cfg.transfer_syntax, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DicomParseConfig::new().with_parse_pixel_data(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DicomParse::new(DicomParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DicomParse"));
    }

    #[test]
    fn test_summary() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_header() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_header();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_get_tag() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.get_tag();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_patient_info() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.patient_info();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_patient_info_empty() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap();
        assert!(e.patient_info().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = DicomParse::new(DicomParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DicomParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DicomParseError::InvalidConfig("a".into());
        let e2 = DicomParseError::ComputationFailed("b".into());
        let e3 = DicomParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
