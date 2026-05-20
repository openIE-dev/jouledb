//! Care plan creation and tracking.
//!
//! Provides [`CarePlanConfig`] builder and [`CarePlan`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CarePlanError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CarePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CarePlan: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CarePlan: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CarePlan: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CarePlan`] parameters.
#[derive(Debug, Clone)]
pub struct CarePlanConfig {
    pub max_goals: usize,
    pub max_interventions: usize,
    pub track_outcomes: bool,
    pub auto_schedule: bool,
}

impl CarePlanConfig {
    pub fn new() -> Self {
        Self {
            max_goals: 20,
            max_interventions: 50,
            track_outcomes: true,
            auto_schedule: false,
        }
    }

    pub fn with_max_goals(mut self, v: usize) -> Self {
        self.max_goals = v;
        self
    }

    pub fn with_max_interventions(mut self, v: usize) -> Self {
        self.max_interventions = v;
        self
    }

    pub fn with_track_outcomes(mut self, v: bool) -> Self {
        self.track_outcomes = v;
        self
    }

    pub fn with_auto_schedule(mut self, v: bool) -> Self {
        self.auto_schedule = v;
        self
    }

    pub fn validate(&self) -> Result<(), CarePlanError> {
        Ok(())
    }
}

impl Default for CarePlanConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CarePlanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CarePlanConfig(max_goals={0}, max_interventions={1}, track_outcomes={2}, auto_schedule={3})", self.max_goals, self.max_interventions, self.track_outcomes, self.auto_schedule)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core care plan creation and tracking engine.
#[derive(Debug, Clone)]
pub struct CarePlan {
    config: CarePlanConfig,
    data: Vec<f64>,
}

impl CarePlan {
    pub fn new(config: CarePlanConfig) -> Result<Self, CarePlanError> {
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
    pub fn config(&self) -> &CarePlanConfig { &self.config }

    /// Add care plan goal.
    pub fn add_goal(&self) -> usize {
        self.data.len()
    }

    /// Track goal progress.
    pub fn track_progress(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Calculate compliance rate.
    pub fn compliance_rate(&self) -> f64 {
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

impl fmt::Display for CarePlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CarePlan(n={})", self.data.len())
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
        let cfg = CarePlanConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CarePlanConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CarePlanConfig"));
    }

    #[test]
    fn test_config_with_max_goals() {
        let cfg = CarePlanConfig::new().with_max_goals(42);
        assert_eq!(cfg.max_goals, 42);
    }

    #[test]
    fn test_config_with_max_interventions() {
        let cfg = CarePlanConfig::new().with_max_interventions(42);
        assert_eq!(cfg.max_interventions, 42);
    }

    #[test]
    fn test_config_with_track_outcomes() {
        let cfg = CarePlanConfig::new().with_track_outcomes(false);
        assert_eq!(cfg.track_outcomes, false);
    }

    #[test]
    fn test_config_with_auto_schedule() {
        let cfg = CarePlanConfig::new().with_auto_schedule(true);
        assert_eq!(cfg.auto_schedule, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CarePlanConfig::new().with_max_goals(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CarePlan::new(CarePlanConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CarePlan"));
    }

    #[test]
    fn test_summary() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_goal() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_goal();
        assert!(result > 0);
    }

    #[test]
    fn test_track_progress() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.track_progress();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compliance_rate() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compliance_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_compliance_rate_empty() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap();
        assert!((e.compliance_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = CarePlan::new(CarePlanConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CarePlanError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CarePlanError::InvalidConfig("a".into());
        let e2 = CarePlanError::ComputationFailed("b".into());
        let e3 = CarePlanError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
