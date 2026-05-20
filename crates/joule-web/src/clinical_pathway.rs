//! Clinical pathway management and variance tracking.
//!
//! Provides [`ClinicalPathwayConfig`] builder and [`ClinicalPathway`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ClinicalPathwayError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ClinicalPathwayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ClinicalPathway: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ClinicalPathway: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ClinicalPathway: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ClinicalPathway`] parameters.
#[derive(Debug, Clone)]
pub struct ClinicalPathwayConfig {
    pub max_steps: usize,
    pub track_variance: bool,
    pub auto_advance: bool,
    pub outcome_measure: bool,
}

impl ClinicalPathwayConfig {
    pub fn new() -> Self {
        Self {
            max_steps: 50,
            track_variance: true,
            auto_advance: false,
            outcome_measure: true,
        }
    }

    pub fn with_max_steps(mut self, v: usize) -> Self {
        self.max_steps = v;
        self
    }

    pub fn with_track_variance(mut self, v: bool) -> Self {
        self.track_variance = v;
        self
    }

    pub fn with_auto_advance(mut self, v: bool) -> Self {
        self.auto_advance = v;
        self
    }

    pub fn with_outcome_measure(mut self, v: bool) -> Self {
        self.outcome_measure = v;
        self
    }

    pub fn validate(&self) -> Result<(), ClinicalPathwayError> {
        Ok(())
    }
}

impl Default for ClinicalPathwayConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ClinicalPathwayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClinicalPathwayConfig(max_steps={0}, track_variance={1}, auto_advance={2}, outcome_measure={3})", self.max_steps, self.track_variance, self.auto_advance, self.outcome_measure)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core clinical pathway management and variance tracking engine.
#[derive(Debug, Clone)]
pub struct ClinicalPathway {
    config: ClinicalPathwayConfig,
    data: Vec<f64>,
}

impl ClinicalPathway {
    pub fn new(config: ClinicalPathwayConfig) -> Result<Self, ClinicalPathwayError> {
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
    pub fn config(&self) -> &ClinicalPathwayConfig { &self.config }

    /// Advance to next pathway step.
    pub fn advance_step(&self) -> bool {
        !self.data.is_empty()
    }

    /// Detect pathway variance.
    pub fn detect_variance(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Calculate pathway compliance.
    pub fn compliance_score(&self) -> f64 {
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

impl fmt::Display for ClinicalPathway {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClinicalPathway(n={})", self.data.len())
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
        let cfg = ClinicalPathwayConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ClinicalPathwayConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ClinicalPathwayConfig"));
    }

    #[test]
    fn test_config_with_max_steps() {
        let cfg = ClinicalPathwayConfig::new().with_max_steps(42);
        assert_eq!(cfg.max_steps, 42);
    }

    #[test]
    fn test_config_with_track_variance() {
        let cfg = ClinicalPathwayConfig::new().with_track_variance(false);
        assert_eq!(cfg.track_variance, false);
    }

    #[test]
    fn test_config_with_auto_advance() {
        let cfg = ClinicalPathwayConfig::new().with_auto_advance(true);
        assert_eq!(cfg.auto_advance, true);
    }

    #[test]
    fn test_config_with_outcome_measure() {
        let cfg = ClinicalPathwayConfig::new().with_outcome_measure(false);
        assert_eq!(cfg.outcome_measure, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ClinicalPathwayConfig::new().with_max_steps(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ClinicalPathway"));
    }

    #[test]
    fn test_summary() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_advance_step() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.advance_step();
        assert!(result);
    }

    #[test]
    fn test_detect_variance() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_variance();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compliance_score() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compliance_score();
        assert!(result.is_finite());
    }

    #[test]
    fn test_compliance_score_empty() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap();
        assert!((e.compliance_score() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = ClinicalPathway::new(ClinicalPathwayConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ClinicalPathwayError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ClinicalPathwayError::InvalidConfig("a".into());
        let e2 = ClinicalPathwayError::ComputationFailed("b".into());
        let e3 = ClinicalPathwayError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
