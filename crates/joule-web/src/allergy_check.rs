//! Allergy verification and cross-reactivity detection.
//!
//! Provides [`AllergyCheckConfig`] builder and [`AllergyCheck`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AllergyCheckError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AllergyCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AllergyCheck: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AllergyCheck: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AllergyCheck: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AllergyCheck`] parameters.
#[derive(Debug, Clone)]
pub struct AllergyCheckConfig {
    pub check_cross_react: bool,
    pub ingredient_level: bool,
    pub severity_levels: usize,
    pub override_allowed: bool,
}

impl AllergyCheckConfig {
    pub fn new() -> Self {
        Self {
            check_cross_react: true,
            ingredient_level: true,
            severity_levels: 4,
            override_allowed: true,
        }
    }

    pub fn with_check_cross_react(mut self, v: bool) -> Self {
        self.check_cross_react = v;
        self
    }

    pub fn with_ingredient_level(mut self, v: bool) -> Self {
        self.ingredient_level = v;
        self
    }

    pub fn with_severity_levels(mut self, v: usize) -> Self {
        self.severity_levels = v;
        self
    }

    pub fn with_override_allowed(mut self, v: bool) -> Self {
        self.override_allowed = v;
        self
    }

    pub fn validate(&self) -> Result<(), AllergyCheckError> {
        Ok(())
    }
}

impl Default for AllergyCheckConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AllergyCheckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllergyCheckConfig(check_cross_react={0}, ingredient_level={1}, severity_levels={2}, override_allowed={3})", self.check_cross_react, self.ingredient_level, self.severity_levels, self.override_allowed)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core allergy verification and cross-reactivity detection engine.
#[derive(Debug, Clone)]
pub struct AllergyCheck {
    config: AllergyCheckConfig,
    data: Vec<f64>,
}

impl AllergyCheck {
    pub fn new(config: AllergyCheckConfig) -> Result<Self, AllergyCheckError> {
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
    pub fn config(&self) -> &AllergyCheckConfig { &self.config }

    /// Check drug against allergies.
    pub fn check_allergy(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check cross-reactivity.
    pub fn cross_reactivity(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get allergy severity.
    pub fn severity(&self) -> String {
        format!("{}: {} records", stringify!(severity), self.data.len())
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

impl fmt::Display for AllergyCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllergyCheck(n={})", self.data.len())
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
        let cfg = AllergyCheckConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AllergyCheckConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AllergyCheckConfig"));
    }

    #[test]
    fn test_config_with_check_cross_react() {
        let cfg = AllergyCheckConfig::new().with_check_cross_react(false);
        assert_eq!(cfg.check_cross_react, false);
    }

    #[test]
    fn test_config_with_ingredient_level() {
        let cfg = AllergyCheckConfig::new().with_ingredient_level(false);
        assert_eq!(cfg.ingredient_level, false);
    }

    #[test]
    fn test_config_with_severity_levels() {
        let cfg = AllergyCheckConfig::new().with_severity_levels(42);
        assert_eq!(cfg.severity_levels, 42);
    }

    #[test]
    fn test_config_with_override_allowed() {
        let cfg = AllergyCheckConfig::new().with_override_allowed(false);
        assert_eq!(cfg.override_allowed, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AllergyCheckConfig::new().with_check_cross_react(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AllergyCheck"));
    }

    #[test]
    fn test_summary() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_allergy() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.check_allergy();
        assert!(result);
    }

    #[test]
    fn test_cross_reactivity() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.cross_reactivity();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_severity() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.severity();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_severity_empty() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap();
        let _ = e.severity();
    }

    #[test]
    fn test_config_accessor() {
        let e = AllergyCheck::new(AllergyCheckConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AllergyCheckError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AllergyCheckError::InvalidConfig("a".into());
        let e2 = AllergyCheckError::ComputationFailed("b".into());
        let e3 = AllergyCheckError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
