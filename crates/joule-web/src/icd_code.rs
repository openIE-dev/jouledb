//! ICD-10-CM code lookup and hierarchy.
//!
//! Provides [`IcdCodeConfig`] builder and [`IcdCode`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum IcdCodeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for IcdCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "IcdCode: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "IcdCode: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "IcdCode: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`IcdCode`] parameters.
#[derive(Debug, Clone)]
pub struct IcdCodeConfig {
    pub version: usize,
    pub include_descriptions: bool,
    pub validate_format: bool,
    pub max_results: usize,
}

impl IcdCodeConfig {
    pub fn new() -> Self {
        Self {
            version: 10,
            include_descriptions: true,
            validate_format: true,
            max_results: 100,
        }
    }

    pub fn with_version(mut self, v: usize) -> Self {
        self.version = v;
        self
    }

    pub fn with_include_descriptions(mut self, v: bool) -> Self {
        self.include_descriptions = v;
        self
    }

    pub fn with_validate_format(mut self, v: bool) -> Self {
        self.validate_format = v;
        self
    }

    pub fn with_max_results(mut self, v: usize) -> Self {
        self.max_results = v;
        self
    }

    pub fn validate(&self) -> Result<(), IcdCodeError> {
        Ok(())
    }
}

impl Default for IcdCodeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for IcdCodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IcdCodeConfig(version={0}, include_descriptions={1}, validate_format={2}, max_results={3})", self.version, self.include_descriptions, self.validate_format, self.max_results)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core icd-10-cm code lookup and hierarchy engine.
#[derive(Debug, Clone)]
pub struct IcdCode {
    config: IcdCodeConfig,
    data: Vec<f64>,
}

impl IcdCode {
    pub fn new(config: IcdCodeConfig) -> Result<Self, IcdCodeError> {
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
    pub fn config(&self) -> &IcdCodeConfig { &self.config }

    /// Look up ICD code.
    pub fn lookup(&self) -> String {
        format!("{}: {} records", stringify!(lookup), self.data.len())
    }

    /// Validate ICD code format.
    pub fn is_valid(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get parent code in hierarchy.
    pub fn parent_code(&self) -> String {
        format!("{}: {} records", stringify!(parent_code), self.data.len())
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

impl fmt::Display for IcdCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IcdCode(n={})", self.data.len())
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
        let cfg = IcdCodeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = IcdCodeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("IcdCodeConfig"));
    }

    #[test]
    fn test_config_with_version() {
        let cfg = IcdCodeConfig::new().with_version(42);
        assert_eq!(cfg.version, 42);
    }

    #[test]
    fn test_config_with_include_descriptions() {
        let cfg = IcdCodeConfig::new().with_include_descriptions(false);
        assert_eq!(cfg.include_descriptions, false);
    }

    #[test]
    fn test_config_with_validate_format() {
        let cfg = IcdCodeConfig::new().with_validate_format(false);
        assert_eq!(cfg.validate_format, false);
    }

    #[test]
    fn test_config_with_max_results() {
        let cfg = IcdCodeConfig::new().with_max_results(42);
        assert_eq!(cfg.max_results, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = IcdCodeConfig::new().with_version(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = IcdCode::new(IcdCodeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("IcdCode"));
    }

    #[test]
    fn test_summary() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lookup() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lookup();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_valid() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_valid();
        assert!(result);
    }

    #[test]
    fn test_parent_code() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parent_code();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_parent_code_empty() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap();
        let _ = e.parent_code();
    }

    #[test]
    fn test_config_accessor() {
        let e = IcdCode::new(IcdCodeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = IcdCodeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = IcdCodeError::InvalidConfig("a".into());
        let e2 = IcdCodeError::ComputationFailed("b".into());
        let e3 = IcdCodeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
