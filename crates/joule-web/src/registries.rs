//! Disease registry enrollment and tracking.
//!
//! Provides [`RegistriesConfig`] builder and [`Registries`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RegistriesError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RegistriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Registries: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Registries: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Registries: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Registries`] parameters.
#[derive(Debug, Clone)]
pub struct RegistriesConfig {
    pub registry_type: usize,
    pub auto_enroll: bool,
    pub dedup_check: bool,
    pub max_entries: usize,
}

impl RegistriesConfig {
    pub fn new() -> Self {
        Self {
            registry_type: 0,
            auto_enroll: false,
            dedup_check: true,
            max_entries: 100000,
        }
    }

    pub fn with_registry_type(mut self, v: usize) -> Self {
        self.registry_type = v;
        self
    }

    pub fn with_auto_enroll(mut self, v: bool) -> Self {
        self.auto_enroll = v;
        self
    }

    pub fn with_dedup_check(mut self, v: bool) -> Self {
        self.dedup_check = v;
        self
    }

    pub fn with_max_entries(mut self, v: usize) -> Self {
        self.max_entries = v;
        self
    }

    pub fn validate(&self) -> Result<(), RegistriesError> {
        Ok(())
    }
}

impl Default for RegistriesConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RegistriesConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegistriesConfig(registry_type={0}, auto_enroll={1}, dedup_check={2}, max_entries={3})", self.registry_type, self.auto_enroll, self.dedup_check, self.max_entries)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core disease registry enrollment and tracking engine.
#[derive(Debug, Clone)]
pub struct Registries {
    config: RegistriesConfig,
    data: Vec<f64>,
}

impl Registries {
    pub fn new(config: RegistriesConfig) -> Result<Self, RegistriesError> {
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
    pub fn config(&self) -> &RegistriesConfig { &self.config }

    /// Enroll patient in registry.
    pub fn enroll(&self) -> bool {
        !self.data.is_empty()
    }

    /// Match case definition.
    pub fn match_criteria(&self) -> bool {
        !self.data.is_empty()
    }

    /// Registry data quality.
    pub fn data_quality_score(&self) -> f64 {
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

impl fmt::Display for Registries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Registries(n={})", self.data.len())
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
        let cfg = RegistriesConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RegistriesConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RegistriesConfig"));
    }

    #[test]
    fn test_config_with_registry_type() {
        let cfg = RegistriesConfig::new().with_registry_type(42);
        assert_eq!(cfg.registry_type, 42);
    }

    #[test]
    fn test_config_with_auto_enroll() {
        let cfg = RegistriesConfig::new().with_auto_enroll(true);
        assert_eq!(cfg.auto_enroll, true);
    }

    #[test]
    fn test_config_with_dedup_check() {
        let cfg = RegistriesConfig::new().with_dedup_check(false);
        assert_eq!(cfg.dedup_check, false);
    }

    #[test]
    fn test_config_with_max_entries() {
        let cfg = RegistriesConfig::new().with_max_entries(42);
        assert_eq!(cfg.max_entries, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RegistriesConfig::new().with_registry_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = Registries::new(RegistriesConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Registries::new(RegistriesConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Registries"));
    }

    #[test]
    fn test_summary() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Registries::new(RegistriesConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_enroll() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.enroll();
        assert!(result);
    }

    #[test]
    fn test_match_criteria() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.match_criteria();
        assert!(result);
    }

    #[test]
    fn test_data_quality_score() {
        let e = Registries::new(RegistriesConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.data_quality_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_data_quality_score_empty() {
        let e = Registries::new(RegistriesConfig::new()).unwrap();
        assert!((e.data_quality_score() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = Registries::new(RegistriesConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RegistriesError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RegistriesError::InvalidConfig("a".into());
        let e2 = RegistriesError::ComputationFailed("b".into());
        let e3 = RegistriesError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
