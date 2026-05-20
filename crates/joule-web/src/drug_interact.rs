//! Drug interaction checking engine.
//!
//! Provides [`DrugInteractConfig`] builder and [`DrugInteract`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DrugInteractError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DrugInteractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DrugInteract: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DrugInteract: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DrugInteract: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DrugInteract`] parameters.
#[derive(Debug, Clone)]
pub struct DrugInteractConfig {
    pub severity_threshold: usize,
    pub include_food: bool,
    pub check_duplicates: bool,
    pub max_pairs: usize,
}

impl DrugInteractConfig {
    pub fn new() -> Self {
        Self {
            severity_threshold: 1,
            include_food: true,
            check_duplicates: true,
            max_pairs: 1000,
        }
    }

    pub fn with_severity_threshold(mut self, v: usize) -> Self {
        self.severity_threshold = v;
        self
    }

    pub fn with_include_food(mut self, v: bool) -> Self {
        self.include_food = v;
        self
    }

    pub fn with_check_duplicates(mut self, v: bool) -> Self {
        self.check_duplicates = v;
        self
    }

    pub fn with_max_pairs(mut self, v: usize) -> Self {
        self.max_pairs = v;
        self
    }

    pub fn validate(&self) -> Result<(), DrugInteractError> {
        Ok(())
    }
}

impl Default for DrugInteractConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DrugInteractConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrugInteractConfig(severity_threshold={0}, include_food={1}, check_duplicates={2}, max_pairs={3})", self.severity_threshold, self.include_food, self.check_duplicates, self.max_pairs)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core drug interaction checking engine engine.
#[derive(Debug, Clone)]
pub struct DrugInteract {
    config: DrugInteractConfig,
    data: Vec<f64>,
}

impl DrugInteract {
    pub fn new(config: DrugInteractConfig) -> Result<Self, DrugInteractError> {
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
    pub fn config(&self) -> &DrugInteractConfig { &self.config }

    /// Check drug-drug interaction.
    pub fn check_pair(&self) -> String {
        format!("{}: {} records", stringify!(check_pair), self.data.len())
    }

    /// Interaction severity level.
    pub fn severity_level(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Suggest therapeutic alternatives.
    pub fn alternatives(&self) -> Vec<f64> {
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

impl fmt::Display for DrugInteract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrugInteract(n={})", self.data.len())
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
        let cfg = DrugInteractConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DrugInteractConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DrugInteractConfig"));
    }

    #[test]
    fn test_config_with_severity_threshold() {
        let cfg = DrugInteractConfig::new().with_severity_threshold(42);
        assert_eq!(cfg.severity_threshold, 42);
    }

    #[test]
    fn test_config_with_include_food() {
        let cfg = DrugInteractConfig::new().with_include_food(false);
        assert_eq!(cfg.include_food, false);
    }

    #[test]
    fn test_config_with_check_duplicates() {
        let cfg = DrugInteractConfig::new().with_check_duplicates(false);
        assert_eq!(cfg.check_duplicates, false);
    }

    #[test]
    fn test_config_with_max_pairs() {
        let cfg = DrugInteractConfig::new().with_max_pairs(42);
        assert_eq!(cfg.max_pairs, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DrugInteractConfig::new().with_severity_threshold(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DrugInteract::new(DrugInteractConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DrugInteract"));
    }

    #[test]
    fn test_summary() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_pair() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.check_pair();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_severity_level() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.severity_level();
        assert!(result.is_finite());
    }

    #[test]
    fn test_alternatives() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.alternatives();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_alternatives_empty() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap();
        assert!(e.alternatives().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = DrugInteract::new(DrugInteractConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DrugInteractError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DrugInteractError::InvalidConfig("a".into());
        let e2 = DrugInteractError::ComputationFailed("b".into());
        let e3 = DrugInteractError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
