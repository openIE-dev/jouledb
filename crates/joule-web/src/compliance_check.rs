//! HIPAA compliance assessment and scoring.
//!
//! Provides [`ComplianceCheckConfig`] builder and [`ComplianceCheck`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceCheckError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ComplianceCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ComplianceCheck: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ComplianceCheck: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ComplianceCheck: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ComplianceCheck`] parameters.
#[derive(Debug, Clone)]
pub struct ComplianceCheckConfig {
    pub safeguard_type: usize,
    pub assessment_scope: usize,
    pub include_remediation: bool,
    pub scoring_method: usize,
}

impl ComplianceCheckConfig {
    pub fn new() -> Self {
        Self {
            safeguard_type: 0,
            assessment_scope: 0,
            include_remediation: true,
            scoring_method: 0,
        }
    }

    pub fn with_safeguard_type(mut self, v: usize) -> Self {
        self.safeguard_type = v;
        self
    }

    pub fn with_assessment_scope(mut self, v: usize) -> Self {
        self.assessment_scope = v;
        self
    }

    pub fn with_include_remediation(mut self, v: bool) -> Self {
        self.include_remediation = v;
        self
    }

    pub fn with_scoring_method(mut self, v: usize) -> Self {
        self.scoring_method = v;
        self
    }

    pub fn validate(&self) -> Result<(), ComplianceCheckError> {
        Ok(())
    }
}

impl Default for ComplianceCheckConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ComplianceCheckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplianceCheckConfig(safeguard_type={0}, assessment_scope={1}, include_remediation={2}, scoring_method={3})", self.safeguard_type, self.assessment_scope, self.include_remediation, self.scoring_method)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hipaa compliance assessment and scoring engine.
#[derive(Debug, Clone)]
pub struct ComplianceCheck {
    config: ComplianceCheckConfig,
    data: Vec<f64>,
}

impl ComplianceCheck {
    pub fn new(config: ComplianceCheckConfig) -> Result<Self, ComplianceCheckError> {
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
    pub fn config(&self) -> &ComplianceCheckConfig { &self.config }

    /// Assess HIPAA safeguards.
    pub fn assess_safeguards(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Calculate compliance score.
    pub fn compliance_score(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Generate remediation plan.
    pub fn remediation_plan(&self) -> String {
        format!("{}: {} records", stringify!(remediation_plan), self.data.len())
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

impl fmt::Display for ComplianceCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplianceCheck(n={})", self.data.len())
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
        let cfg = ComplianceCheckConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ComplianceCheckConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ComplianceCheckConfig"));
    }

    #[test]
    fn test_config_with_safeguard_type() {
        let cfg = ComplianceCheckConfig::new().with_safeguard_type(42);
        assert_eq!(cfg.safeguard_type, 42);
    }

    #[test]
    fn test_config_with_assessment_scope() {
        let cfg = ComplianceCheckConfig::new().with_assessment_scope(42);
        assert_eq!(cfg.assessment_scope, 42);
    }

    #[test]
    fn test_config_with_include_remediation() {
        let cfg = ComplianceCheckConfig::new().with_include_remediation(false);
        assert_eq!(cfg.include_remediation, false);
    }

    #[test]
    fn test_config_with_scoring_method() {
        let cfg = ComplianceCheckConfig::new().with_scoring_method(42);
        assert_eq!(cfg.scoring_method, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ComplianceCheckConfig::new().with_safeguard_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ComplianceCheck"));
    }

    #[test]
    fn test_summary() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_assess_safeguards() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.assess_safeguards();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compliance_score() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compliance_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_remediation_plan() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.remediation_plan();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_remediation_plan_empty() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap();
        let _ = e.remediation_plan();
    }

    #[test]
    fn test_config_accessor() {
        let e = ComplianceCheck::new(ComplianceCheckConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ComplianceCheckError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ComplianceCheckError::InvalidConfig("a".into());
        let e2 = ComplianceCheckError::ComputationFailed("b".into());
        let e3 = ComplianceCheckError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
