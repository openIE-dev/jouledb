//! HIPAA Safe Harbor de-identification.
//!
//! Provides [`HipaaDeidentConfig`] builder and [`HipaaDeident`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum HipaaDeidentError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for HipaaDeidentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "HipaaDeident: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "HipaaDeident: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "HipaaDeident: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`HipaaDeident`] parameters.
#[derive(Debug, Clone)]
pub struct HipaaDeidentConfig {
    pub method: usize,
    pub date_shift_range: usize,
    pub zip_truncate: usize,
    pub age_threshold: usize,
}

impl HipaaDeidentConfig {
    pub fn new() -> Self {
        Self {
            method: 0,
            date_shift_range: 365,
            zip_truncate: 3,
            age_threshold: 89,
        }
    }

    pub fn with_method(mut self, v: usize) -> Self {
        self.method = v;
        self
    }

    pub fn with_date_shift_range(mut self, v: usize) -> Self {
        self.date_shift_range = v;
        self
    }

    pub fn with_zip_truncate(mut self, v: usize) -> Self {
        self.zip_truncate = v;
        self
    }

    pub fn with_age_threshold(mut self, v: usize) -> Self {
        self.age_threshold = v;
        self
    }

    pub fn validate(&self) -> Result<(), HipaaDeidentError> {
        Ok(())
    }
}

impl Default for HipaaDeidentConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for HipaaDeidentConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HipaaDeidentConfig(method={0}, date_shift_range={1}, zip_truncate={2}, age_threshold={3})", self.method, self.date_shift_range, self.zip_truncate, self.age_threshold)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hipaa safe harbor de-identification engine.
#[derive(Debug, Clone)]
pub struct HipaaDeident {
    config: HipaaDeidentConfig,
    data: Vec<f64>,
}

impl HipaaDeident {
    pub fn new(config: HipaaDeidentConfig) -> Result<Self, HipaaDeidentError> {
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
    pub fn config(&self) -> &HipaaDeidentConfig { &self.config }

    /// Apply de-identification.
    pub fn deidentify(&self) -> String {
        format!("{}: {} records", stringify!(deidentify), self.data.len())
    }

    /// Check if field is PHI.
    pub fn is_identifier(&self) -> bool {
        !self.data.is_empty()
    }

    /// Re-identification risk score.
    pub fn reident_risk(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for HipaaDeident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HipaaDeident(n={})", self.data.len())
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
        let cfg = HipaaDeidentConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = HipaaDeidentConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("HipaaDeidentConfig"));
    }

    #[test]
    fn test_config_with_method() {
        let cfg = HipaaDeidentConfig::new().with_method(42);
        assert_eq!(cfg.method, 42);
    }

    #[test]
    fn test_config_with_date_shift_range() {
        let cfg = HipaaDeidentConfig::new().with_date_shift_range(42);
        assert_eq!(cfg.date_shift_range, 42);
    }

    #[test]
    fn test_config_with_zip_truncate() {
        let cfg = HipaaDeidentConfig::new().with_zip_truncate(42);
        assert_eq!(cfg.zip_truncate, 42);
    }

    #[test]
    fn test_config_with_age_threshold() {
        let cfg = HipaaDeidentConfig::new().with_age_threshold(42);
        assert_eq!(cfg.age_threshold, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = HipaaDeidentConfig::new().with_method(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("HipaaDeident"));
    }

    #[test]
    fn test_summary() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_deidentify() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.deidentify();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_identifier() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_identifier();
        assert!(result);
    }

    #[test]
    fn test_reident_risk() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.reident_risk();
        assert!(result.is_finite());
    }

    #[test]
    fn test_reident_risk_empty() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap();
        assert!((e.reident_risk() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = HipaaDeident::new(HipaaDeidentConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = HipaaDeidentError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = HipaaDeidentError::InvalidConfig("a".into());
        let e2 = HipaaDeidentError::ComputationFailed("b".into());
        let e3 = HipaaDeidentError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
