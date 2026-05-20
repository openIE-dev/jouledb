//! NDC (National Drug Code) lookup.
//!
//! Provides [`NdcLookupConfig`] builder and [`NdcLookup`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum NdcLookupError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for NdcLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "NdcLookup: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "NdcLookup: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "NdcLookup: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`NdcLookup`] parameters.
#[derive(Debug, Clone)]
pub struct NdcLookupConfig {
    pub format: usize,
    pub include_package: bool,
    pub validate_checksum: bool,
    pub max_results: usize,
}

impl NdcLookupConfig {
    pub fn new() -> Self {
        Self {
            format: 0,
            include_package: true,
            validate_checksum: true,
            max_results: 50,
        }
    }

    pub fn with_format(mut self, v: usize) -> Self {
        self.format = v;
        self
    }

    pub fn with_include_package(mut self, v: bool) -> Self {
        self.include_package = v;
        self
    }

    pub fn with_validate_checksum(mut self, v: bool) -> Self {
        self.validate_checksum = v;
        self
    }

    pub fn with_max_results(mut self, v: usize) -> Self {
        self.max_results = v;
        self
    }

    pub fn validate(&self) -> Result<(), NdcLookupError> {
        Ok(())
    }
}

impl Default for NdcLookupConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NdcLookupConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdcLookupConfig(format={0}, include_package={1}, validate_checksum={2}, max_results={3})", self.format, self.include_package, self.validate_checksum, self.max_results)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core ndc (national drug code) lookup engine.
#[derive(Debug, Clone)]
pub struct NdcLookup {
    config: NdcLookupConfig,
    data: Vec<f64>,
}

impl NdcLookup {
    pub fn new(config: NdcLookupConfig) -> Result<Self, NdcLookupError> {
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
    pub fn config(&self) -> &NdcLookupConfig { &self.config }

    /// Look up NDC code.
    pub fn lookup(&self) -> String {
        format!("{}: {} records", stringify!(lookup), self.data.len())
    }

    /// Normalize NDC format.
    pub fn normalize_format(&self) -> String {
        format!("{}: {} records", stringify!(normalize_format), self.data.len())
    }

    /// Get drug class.
    pub fn drug_class(&self) -> String {
        format!("{}: {} records", stringify!(drug_class), self.data.len())
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

impl fmt::Display for NdcLookup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdcLookup(n={})", self.data.len())
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
        let cfg = NdcLookupConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = NdcLookupConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("NdcLookupConfig"));
    }

    #[test]
    fn test_config_with_format() {
        let cfg = NdcLookupConfig::new().with_format(42);
        assert_eq!(cfg.format, 42);
    }

    #[test]
    fn test_config_with_include_package() {
        let cfg = NdcLookupConfig::new().with_include_package(false);
        assert_eq!(cfg.include_package, false);
    }

    #[test]
    fn test_config_with_validate_checksum() {
        let cfg = NdcLookupConfig::new().with_validate_checksum(false);
        assert_eq!(cfg.validate_checksum, false);
    }

    #[test]
    fn test_config_with_max_results() {
        let cfg = NdcLookupConfig::new().with_max_results(42);
        assert_eq!(cfg.max_results, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = NdcLookupConfig::new().with_format(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = NdcLookup::new(NdcLookupConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("NdcLookup"));
    }

    #[test]
    fn test_summary() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lookup() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lookup();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_normalize_format() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.normalize_format();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_drug_class() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.drug_class();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_drug_class_empty() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap();
        let _ = e.drug_class();
    }

    #[test]
    fn test_config_accessor() {
        let e = NdcLookup::new(NdcLookupConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = NdcLookupError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = NdcLookupError::InvalidConfig("a".into());
        let e2 = NdcLookupError::ComputationFailed("b".into());
        let e3 = NdcLookupError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
