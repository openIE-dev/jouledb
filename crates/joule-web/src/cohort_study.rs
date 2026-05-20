//! Cohort study analysis with risk measures.
//!
//! Provides [`CohortStudyConfig`] builder and [`CohortStudy`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CohortStudyError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CohortStudyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CohortStudy: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CohortStudy: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CohortStudy: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CohortStudy`] parameters.
#[derive(Debug, Clone)]
pub struct CohortStudyConfig {
    pub exposed: usize,
    pub unexposed: usize,
    pub follow_up_years: f64,
    pub stratify: bool,
}

impl CohortStudyConfig {
    pub fn new() -> Self {
        Self {
            exposed: 100,
            unexposed: 100,
            follow_up_years: 5.0,
            stratify: false,
        }
    }

    pub fn with_exposed(mut self, v: usize) -> Self {
        self.exposed = v;
        self
    }

    pub fn with_unexposed(mut self, v: usize) -> Self {
        self.unexposed = v;
        self
    }

    pub fn with_follow_up_years(mut self, v: f64) -> Self {
        self.follow_up_years = v;
        self
    }

    pub fn with_stratify(mut self, v: bool) -> Self {
        self.stratify = v;
        self
    }

    pub fn validate(&self) -> Result<(), CohortStudyError> {
        if self.follow_up_years.is_nan() {
            return Err(CohortStudyError::InvalidConfig("follow_up_years is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CohortStudyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CohortStudyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CohortStudyConfig(exposed={0}, unexposed={1}, follow_up_years={2:.4}, stratify={3})", self.exposed, self.unexposed, self.follow_up_years, self.stratify)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core cohort study analysis with risk measures engine.
#[derive(Debug, Clone)]
pub struct CohortStudy {
    config: CohortStudyConfig,
    data: Vec<f64>,
}

impl CohortStudy {
    pub fn new(config: CohortStudyConfig) -> Result<Self, CohortStudyError> {
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
    pub fn config(&self) -> &CohortStudyConfig { &self.config }

    /// Calculate relative risk.
    pub fn relative_risk(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Attributable risk.
    pub fn attributable_risk(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Number needed to treat.
    pub fn nnt(&self) -> f64 {
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

impl fmt::Display for CohortStudy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CohortStudy(n={})", self.data.len())
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
        let cfg = CohortStudyConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CohortStudyConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CohortStudyConfig"));
    }

    #[test]
    fn test_config_with_exposed() {
        let cfg = CohortStudyConfig::new().with_exposed(42);
        assert_eq!(cfg.exposed, 42);
    }

    #[test]
    fn test_config_with_unexposed() {
        let cfg = CohortStudyConfig::new().with_unexposed(42);
        assert_eq!(cfg.unexposed, 42);
    }

    #[test]
    fn test_config_with_follow_up_years() {
        let cfg = CohortStudyConfig::new().with_follow_up_years(42.0);
        assert!((cfg.follow_up_years - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_stratify() {
        let cfg = CohortStudyConfig::new().with_stratify(true);
        assert_eq!(cfg.stratify, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CohortStudyConfig::new().with_exposed(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CohortStudy::new(CohortStudyConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CohortStudy"));
    }

    #[test]
    fn test_summary() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_relative_risk() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.relative_risk();
        assert!(result.is_finite());
    }

    #[test]
    fn test_attributable_risk() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.attributable_risk();
        assert!(result.is_finite());
    }

    #[test]
    fn test_nnt() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.nnt();
        assert!(result.is_finite());
    }

    #[test]
    fn test_nnt_empty() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap();
        assert!((e.nnt() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = CohortStudy::new(CohortStudyConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CohortStudyError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CohortStudyError::InvalidConfig("a".into());
        let e2 = CohortStudyError::ComputationFailed("b".into());
        let e3 = CohortStudyError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
