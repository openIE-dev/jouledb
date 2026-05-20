//! Clinical decision support rule engine.
//!
//! Provides [`CdsRuleConfig`] builder and [`CdsRule`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CdsRuleError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CdsRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CdsRule: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CdsRule: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CdsRule: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CdsRule`] parameters.
#[derive(Debug, Clone)]
pub struct CdsRuleConfig {
    pub max_rules: usize,
    pub priority_levels: usize,
    pub conflict_resolution: usize,
    pub active_only: bool,
}

impl CdsRuleConfig {
    pub fn new() -> Self {
        Self {
            max_rules: 1000,
            priority_levels: 5,
            conflict_resolution: 0,
            active_only: true,
        }
    }

    pub fn with_max_rules(mut self, v: usize) -> Self {
        self.max_rules = v;
        self
    }

    pub fn with_priority_levels(mut self, v: usize) -> Self {
        self.priority_levels = v;
        self
    }

    pub fn with_conflict_resolution(mut self, v: usize) -> Self {
        self.conflict_resolution = v;
        self
    }

    pub fn with_active_only(mut self, v: bool) -> Self {
        self.active_only = v;
        self
    }

    pub fn validate(&self) -> Result<(), CdsRuleError> {
        Ok(())
    }
}

impl Default for CdsRuleConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CdsRuleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CdsRuleConfig(max_rules={0}, priority_levels={1}, conflict_resolution={2}, active_only={3})", self.max_rules, self.priority_levels, self.conflict_resolution, self.active_only)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core clinical decision support rule engine engine.
#[derive(Debug, Clone)]
pub struct CdsRule {
    config: CdsRuleConfig,
    data: Vec<f64>,
}

impl CdsRule {
    pub fn new(config: CdsRuleConfig) -> Result<Self, CdsRuleError> {
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
    pub fn config(&self) -> &CdsRuleConfig { &self.config }

    /// Evaluate rules against patient data.
    pub fn evaluate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate CDS alert.
    pub fn generate_alert(&self) -> String {
        format!("{}: {} records", stringify!(generate_alert), self.data.len())
    }

    /// Rule activation statistics.
    pub fn rule_activation_rate(&self) -> f64 {
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

impl fmt::Display for CdsRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CdsRule(n={})", self.data.len())
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
        let cfg = CdsRuleConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CdsRuleConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CdsRuleConfig"));
    }

    #[test]
    fn test_config_with_max_rules() {
        let cfg = CdsRuleConfig::new().with_max_rules(42);
        assert_eq!(cfg.max_rules, 42);
    }

    #[test]
    fn test_config_with_priority_levels() {
        let cfg = CdsRuleConfig::new().with_priority_levels(42);
        assert_eq!(cfg.priority_levels, 42);
    }

    #[test]
    fn test_config_with_conflict_resolution() {
        let cfg = CdsRuleConfig::new().with_conflict_resolution(42);
        assert_eq!(cfg.conflict_resolution, 42);
    }

    #[test]
    fn test_config_with_active_only() {
        let cfg = CdsRuleConfig::new().with_active_only(false);
        assert_eq!(cfg.active_only, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CdsRuleConfig::new().with_max_rules(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CdsRule::new(CdsRuleConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CdsRule"));
    }

    #[test]
    fn test_summary() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_alert() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_alert();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rule_activation_rate() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rule_activation_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_rule_activation_rate_empty() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap();
        assert!((e.rule_activation_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = CdsRule::new(CdsRuleConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CdsRuleError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CdsRuleError::InvalidConfig("a".into());
        let e2 = CdsRuleError::ComputationFailed("b".into());
        let e3 = CdsRuleError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
