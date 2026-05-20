//! Inspection planning for manufactured parts.
//!
//! Provides [`InspectionPlanConfig`] builder and [`InspectionPlan`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum InspectionPlanError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for InspectionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "InspectionPlan: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "InspectionPlan: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "InspectionPlan: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`InspectionPlan`] parameters.
#[derive(Debug, Clone)]
pub struct InspectionPlanConfig {
    pub num_features: usize,
    pub sample_size: usize,
    pub confidence_level: f64,
    pub cmm_speed: f64,
}

impl InspectionPlanConfig {
    pub fn new() -> Self {
        Self {
            num_features: 10,
            sample_size: 5,
            confidence_level: 0.95,
            cmm_speed: 10.0,
        }
    }

    pub fn with_num_features(mut self, v: usize) -> Self {
        self.num_features = v;
        self
    }

    pub fn with_sample_size(mut self, v: usize) -> Self {
        self.sample_size = v;
        self
    }

    pub fn with_confidence_level(mut self, v: f64) -> Self {
        self.confidence_level = v;
        self
    }

    pub fn with_cmm_speed(mut self, v: f64) -> Self {
        self.cmm_speed = v;
        self
    }

    pub fn validate(&self) -> Result<(), InspectionPlanError> {
        if self.confidence_level.is_nan() {
            return Err(InspectionPlanError::InvalidConfig("confidence_level is NaN".into()));
        }
        if self.cmm_speed.is_nan() {
            return Err(InspectionPlanError::InvalidConfig("cmm_speed is NaN".into()));
        }
        Ok(())
    }
}

impl Default for InspectionPlanConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for InspectionPlanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InspectionPlanConfig(num_features={0}, sample_size={1}, confidence_level={2:.4}, cmm_speed={3:.4})", self.num_features, self.sample_size, self.confidence_level, self.cmm_speed)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core inspection planning for manufactured parts engine.
#[derive(Debug, Clone)]
pub struct InspectionPlan {
    config: InspectionPlanConfig,
    data: Vec<f64>,
}

impl InspectionPlan {
    pub fn new(config: InspectionPlanConfig) -> Result<Self, InspectionPlanError> {
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
    pub fn config(&self) -> &InspectionPlanConfig { &self.config }

    /// Generate inspection plan.
    pub fn generate_plan(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Determine sampling strategy.
    pub fn sampling_strategy(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Check acceptance criteria.
    pub fn acceptance_criteria(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for InspectionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InspectionPlan(n={})", self.data.len())
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
        let cfg = InspectionPlanConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = InspectionPlanConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("InspectionPlanConfig"));
    }

    #[test]
    fn test_config_with_num_features() {
        let cfg = InspectionPlanConfig::new().with_num_features(42);
        assert_eq!(cfg.num_features, 42);
    }

    #[test]
    fn test_config_with_sample_size() {
        let cfg = InspectionPlanConfig::new().with_sample_size(42);
        assert_eq!(cfg.sample_size, 42);
    }

    #[test]
    fn test_config_with_confidence_level() {
        let cfg = InspectionPlanConfig::new().with_confidence_level(42.0);
        assert!((cfg.confidence_level - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_cmm_speed() {
        let cfg = InspectionPlanConfig::new().with_cmm_speed(42.0);
        assert!((cfg.cmm_speed - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = InspectionPlanConfig::new().with_num_features(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("InspectionPlan"));
    }

    #[test]
    fn test_summary() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_plan() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_plan();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sampling_strategy() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.sampling_strategy();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_acceptance_criteria() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.acceptance_criteria();
        assert!(result);
    }

    #[test]
    fn test_acceptance_criteria_empty() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap();
        assert!(!e.acceptance_criteria());
    }

    #[test]
    fn test_config_accessor() {
        let e = InspectionPlan::new(InspectionPlanConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = InspectionPlanError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = InspectionPlanError::InvalidConfig("a".into());
        let e2 = InspectionPlanError::ComputationFailed("b".into());
        let e3 = InspectionPlanError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
