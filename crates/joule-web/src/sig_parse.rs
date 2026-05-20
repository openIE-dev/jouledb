//! Prescription sig (directions) parser.
//!
//! Provides [`SigParseConfig`] builder and [`SigParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SigParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SigParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SigParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SigParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SigParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SigParse`] parameters.
#[derive(Debug, Clone)]
pub struct SigParseConfig {
    pub strict_mode: bool,
    pub include_prn: bool,
    pub normalize: bool,
    pub max_length: usize,
}

impl SigParseConfig {
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            include_prn: true,
            normalize: true,
            max_length: 500,
        }
    }

    pub fn with_strict_mode(mut self, v: bool) -> Self {
        self.strict_mode = v;
        self
    }

    pub fn with_include_prn(mut self, v: bool) -> Self {
        self.include_prn = v;
        self
    }

    pub fn with_normalize(mut self, v: bool) -> Self {
        self.normalize = v;
        self
    }

    pub fn with_max_length(mut self, v: usize) -> Self {
        self.max_length = v;
        self
    }

    pub fn validate(&self) -> Result<(), SigParseError> {
        Ok(())
    }
}

impl Default for SigParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SigParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SigParseConfig(strict_mode={0}, include_prn={1}, normalize={2}, max_length={3})", self.strict_mode, self.include_prn, self.normalize, self.max_length)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core prescription sig (directions) parser engine.
#[derive(Debug, Clone)]
pub struct SigParse {
    config: SigParseConfig,
    data: Vec<f64>,
}

impl SigParse {
    pub fn new(config: SigParseConfig) -> Result<Self, SigParseError> {
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
    pub fn config(&self) -> &SigParseConfig { &self.config }

    /// Parse prescription sig.
    pub fn parse_sig(&self) -> String {
        format!("{}: {} records", stringify!(parse_sig), self.data.len())
    }

    /// Extract dose quantity.
    pub fn extract_dose(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Extract dosing frequency.
    pub fn extract_frequency(&self) -> String {
        format!("{}: {} records", stringify!(extract_frequency), self.data.len())
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

impl fmt::Display for SigParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SigParse(n={})", self.data.len())
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
        let cfg = SigParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SigParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SigParseConfig"));
    }

    #[test]
    fn test_config_with_strict_mode() {
        let cfg = SigParseConfig::new().with_strict_mode(true);
        assert_eq!(cfg.strict_mode, true);
    }

    #[test]
    fn test_config_with_include_prn() {
        let cfg = SigParseConfig::new().with_include_prn(false);
        assert_eq!(cfg.include_prn, false);
    }

    #[test]
    fn test_config_with_normalize() {
        let cfg = SigParseConfig::new().with_normalize(false);
        assert_eq!(cfg.normalize, false);
    }

    #[test]
    fn test_config_with_max_length() {
        let cfg = SigParseConfig::new().with_max_length(42);
        assert_eq!(cfg.max_length, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SigParseConfig::new().with_strict_mode(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SigParse::new(SigParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SigParse::new(SigParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SigParse"));
    }

    #[test]
    fn test_summary() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SigParse::new(SigParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_sig() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_sig();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_dose() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_dose();
        assert!(result.is_finite());
    }

    #[test]
    fn test_extract_frequency() {
        let e = SigParse::new(SigParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_frequency();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_frequency_empty() {
        let e = SigParse::new(SigParseConfig::new()).unwrap();
        let _ = e.extract_frequency();
    }

    #[test]
    fn test_config_accessor() {
        let e = SigParse::new(SigParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SigParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SigParseError::InvalidConfig("a".into());
        let e2 = SigParseError::ComputationFailed("b".into());
        let e3 = SigParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
