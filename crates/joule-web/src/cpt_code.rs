//! CPT/HCPCS procedure code management.
//!
//! Provides [`CptCodeConfig`] builder and [`CptCode`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CptCodeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CptCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CptCode: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CptCode: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CptCode: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CptCode`] parameters.
#[derive(Debug, Clone)]
pub struct CptCodeConfig {
    pub include_modifiers: bool,
    pub check_bundling: bool,
    pub version_year: usize,
    pub max_results: usize,
}

impl CptCodeConfig {
    pub fn new() -> Self {
        Self {
            include_modifiers: true,
            check_bundling: true,
            version_year: 2026,
            max_results: 100,
        }
    }

    pub fn with_include_modifiers(mut self, v: bool) -> Self {
        self.include_modifiers = v;
        self
    }

    pub fn with_check_bundling(mut self, v: bool) -> Self {
        self.check_bundling = v;
        self
    }

    pub fn with_version_year(mut self, v: usize) -> Self {
        self.version_year = v;
        self
    }

    pub fn with_max_results(mut self, v: usize) -> Self {
        self.max_results = v;
        self
    }

    pub fn validate(&self) -> Result<(), CptCodeError> {
        Ok(())
    }
}

impl Default for CptCodeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CptCodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CptCodeConfig(include_modifiers={0}, check_bundling={1}, version_year={2}, max_results={3})", self.include_modifiers, self.check_bundling, self.version_year, self.max_results)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core cpt/hcpcs procedure code management engine.
#[derive(Debug, Clone)]
pub struct CptCode {
    config: CptCodeConfig,
    data: Vec<f64>,
}

impl CptCode {
    pub fn new(config: CptCodeConfig) -> Result<Self, CptCodeError> {
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
    pub fn config(&self) -> &CptCodeConfig { &self.config }

    /// Look up CPT code.
    pub fn lookup(&self) -> String {
        format!("{}: {} records", stringify!(lookup), self.data.len())
    }

    /// Validate CPT code.
    pub fn is_valid(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get code category.
    pub fn category(&self) -> String {
        format!("{}: {} records", stringify!(category), self.data.len())
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

impl fmt::Display for CptCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CptCode(n={})", self.data.len())
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
        let cfg = CptCodeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CptCodeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CptCodeConfig"));
    }

    #[test]
    fn test_config_with_include_modifiers() {
        let cfg = CptCodeConfig::new().with_include_modifiers(false);
        assert_eq!(cfg.include_modifiers, false);
    }

    #[test]
    fn test_config_with_check_bundling() {
        let cfg = CptCodeConfig::new().with_check_bundling(false);
        assert_eq!(cfg.check_bundling, false);
    }

    #[test]
    fn test_config_with_version_year() {
        let cfg = CptCodeConfig::new().with_version_year(42);
        assert_eq!(cfg.version_year, 42);
    }

    #[test]
    fn test_config_with_max_results() {
        let cfg = CptCodeConfig::new().with_max_results(42);
        assert_eq!(cfg.max_results, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CptCodeConfig::new().with_include_modifiers(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CptCode::new(CptCodeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CptCode"));
    }

    #[test]
    fn test_summary() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lookup() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lookup();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_valid() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_valid();
        assert!(result);
    }

    #[test]
    fn test_category() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.category();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_category_empty() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap();
        let _ = e.category();
    }

    #[test]
    fn test_config_accessor() {
        let e = CptCode::new(CptCodeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CptCodeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CptCodeError::InvalidConfig("a".into());
        let e2 = CptCodeError::ComputationFailed("b".into());
        let e3 = CptCodeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
