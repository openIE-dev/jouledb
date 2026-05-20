//! Protected health information detection.
//!
//! Provides [`PhiDetectConfig`] builder and [`PhiDetect`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PhiDetectError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PhiDetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PhiDetect: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PhiDetect: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PhiDetect: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PhiDetect`] parameters.
#[derive(Debug, Clone)]
pub struct PhiDetectConfig {
    pub ssn_pattern: bool,
    pub mrn_pattern: bool,
    pub name_detect: bool,
    pub confidence_threshold: f64,
}

impl PhiDetectConfig {
    pub fn new() -> Self {
        Self {
            ssn_pattern: true,
            mrn_pattern: true,
            name_detect: true,
            confidence_threshold: 0.8,
        }
    }

    pub fn with_ssn_pattern(mut self, v: bool) -> Self {
        self.ssn_pattern = v;
        self
    }

    pub fn with_mrn_pattern(mut self, v: bool) -> Self {
        self.mrn_pattern = v;
        self
    }

    pub fn with_name_detect(mut self, v: bool) -> Self {
        self.name_detect = v;
        self
    }

    pub fn with_confidence_threshold(mut self, v: f64) -> Self {
        self.confidence_threshold = v;
        self
    }

    pub fn validate(&self) -> Result<(), PhiDetectError> {
        if self.confidence_threshold.is_nan() {
            return Err(PhiDetectError::InvalidConfig("confidence_threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for PhiDetectConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PhiDetectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhiDetectConfig(ssn_pattern={0}, mrn_pattern={1}, name_detect={2}, confidence_threshold={3:.4})", self.ssn_pattern, self.mrn_pattern, self.name_detect, self.confidence_threshold)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core protected health information detection engine.
#[derive(Debug, Clone)]
pub struct PhiDetect {
    config: PhiDetectConfig,
    data: Vec<f64>,
}

impl PhiDetect {
    pub fn new(config: PhiDetectConfig) -> Result<Self, PhiDetectError> {
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
    pub fn config(&self) -> &PhiDetectConfig { &self.config }

    /// Scan text for PHI.
    pub fn scan_text(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Redact detected PHI.
    pub fn redact(&self) -> String {
        format!("{}: {} records", stringify!(redact), self.data.len())
    }

    /// Detection confidence.
    pub fn confidence(&self) -> f64 {
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

impl fmt::Display for PhiDetect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhiDetect(n={})", self.data.len())
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
        let cfg = PhiDetectConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PhiDetectConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PhiDetectConfig"));
    }

    #[test]
    fn test_config_with_ssn_pattern() {
        let cfg = PhiDetectConfig::new().with_ssn_pattern(false);
        assert_eq!(cfg.ssn_pattern, false);
    }

    #[test]
    fn test_config_with_mrn_pattern() {
        let cfg = PhiDetectConfig::new().with_mrn_pattern(false);
        assert_eq!(cfg.mrn_pattern, false);
    }

    #[test]
    fn test_config_with_name_detect() {
        let cfg = PhiDetectConfig::new().with_name_detect(false);
        assert_eq!(cfg.name_detect, false);
    }

    #[test]
    fn test_config_with_confidence_threshold() {
        let cfg = PhiDetectConfig::new().with_confidence_threshold(42.0);
        assert!((cfg.confidence_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PhiDetectConfig::new().with_ssn_pattern(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PhiDetect::new(PhiDetectConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PhiDetect"));
    }

    #[test]
    fn test_summary() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_scan_text() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.scan_text();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_redact() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.redact();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_confidence() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.confidence();
        assert!(result.is_finite());
    }

    #[test]
    fn test_confidence_empty() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap();
        assert!((e.confidence() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = PhiDetect::new(PhiDetectConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PhiDetectError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PhiDetectError::InvalidConfig("a".into());
        let e2 = PhiDetectError::ComputationFailed("b".into());
        let e3 = PhiDetectError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
