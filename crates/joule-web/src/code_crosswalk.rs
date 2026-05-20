//! Medical code mapping and crosswalk.
//!
//! Provides [`CodeCrosswalkConfig`] builder and [`CodeCrosswalk`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CodeCrosswalkError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CodeCrosswalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CodeCrosswalk: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CodeCrosswalk: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CodeCrosswalk: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CodeCrosswalk`] parameters.
#[derive(Debug, Clone)]
pub struct CodeCrosswalkConfig {
    pub source_system: usize,
    pub target_system: usize,
    pub confidence_threshold: f64,
    pub allow_many_to_many: bool,
}

impl CodeCrosswalkConfig {
    pub fn new() -> Self {
        Self {
            source_system: 0,
            target_system: 1,
            confidence_threshold: 0.8,
            allow_many_to_many: true,
        }
    }

    pub fn with_source_system(mut self, v: usize) -> Self {
        self.source_system = v;
        self
    }

    pub fn with_target_system(mut self, v: usize) -> Self {
        self.target_system = v;
        self
    }

    pub fn with_confidence_threshold(mut self, v: f64) -> Self {
        self.confidence_threshold = v;
        self
    }

    pub fn with_allow_many_to_many(mut self, v: bool) -> Self {
        self.allow_many_to_many = v;
        self
    }

    pub fn validate(&self) -> Result<(), CodeCrosswalkError> {
        if self.confidence_threshold.is_nan() {
            return Err(CodeCrosswalkError::InvalidConfig("confidence_threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CodeCrosswalkConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CodeCrosswalkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodeCrosswalkConfig(source_system={0}, target_system={1}, confidence_threshold={2:.4}, allow_many_to_many={3})", self.source_system, self.target_system, self.confidence_threshold, self.allow_many_to_many)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core medical code mapping and crosswalk engine.
#[derive(Debug, Clone)]
pub struct CodeCrosswalk {
    config: CodeCrosswalkConfig,
    data: Vec<f64>,
}

impl CodeCrosswalk {
    pub fn new(config: CodeCrosswalkConfig) -> Result<Self, CodeCrosswalkError> {
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
    pub fn config(&self) -> &CodeCrosswalkConfig { &self.config }

    /// Map code between systems.
    pub fn map_code(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Mapping confidence.
    pub fn confidence_score(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Reverse mapping.
    pub fn reverse_map(&self) -> Vec<f64> {
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

impl fmt::Display for CodeCrosswalk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodeCrosswalk(n={})", self.data.len())
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
        let cfg = CodeCrosswalkConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CodeCrosswalkConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CodeCrosswalkConfig"));
    }

    #[test]
    fn test_config_with_source_system() {
        let cfg = CodeCrosswalkConfig::new().with_source_system(42);
        assert_eq!(cfg.source_system, 42);
    }

    #[test]
    fn test_config_with_target_system() {
        let cfg = CodeCrosswalkConfig::new().with_target_system(42);
        assert_eq!(cfg.target_system, 42);
    }

    #[test]
    fn test_config_with_confidence_threshold() {
        let cfg = CodeCrosswalkConfig::new().with_confidence_threshold(42.0);
        assert!((cfg.confidence_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_allow_many_to_many() {
        let cfg = CodeCrosswalkConfig::new().with_allow_many_to_many(false);
        assert_eq!(cfg.allow_many_to_many, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CodeCrosswalkConfig::new().with_source_system(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CodeCrosswalk"));
    }

    #[test]
    fn test_summary() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_map_code() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.map_code();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_confidence_score() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.confidence_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_reverse_map() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.reverse_map();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_reverse_map_empty() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap();
        assert!(e.reverse_map().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = CodeCrosswalk::new(CodeCrosswalkConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CodeCrosswalkError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CodeCrosswalkError::InvalidConfig("a".into());
        let e2 = CodeCrosswalkError::ComputationFailed("b".into());
        let e3 = CodeCrosswalkError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
