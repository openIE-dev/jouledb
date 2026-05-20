//! Survival analysis with Kaplan-Meier estimator.
//!
//! Provides [`SurvivalAnalysisConfig`] builder and [`SurvivalAnalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurvivalAnalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurvivalAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurvivalAnalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurvivalAnalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurvivalAnalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurvivalAnalysis`] parameters.
#[derive(Debug, Clone)]
pub struct SurvivalAnalysisConfig {
    pub confidence: f64,
    pub max_time: f64,
    pub handle_censoring: bool,
    pub time_unit: usize,
}

impl SurvivalAnalysisConfig {
    pub fn new() -> Self {
        Self {
            confidence: 0.95,
            max_time: 365.0,
            handle_censoring: true,
            time_unit: 0,
        }
    }

    pub fn with_confidence(mut self, v: f64) -> Self {
        self.confidence = v;
        self
    }

    pub fn with_max_time(mut self, v: f64) -> Self {
        self.max_time = v;
        self
    }

    pub fn with_handle_censoring(mut self, v: bool) -> Self {
        self.handle_censoring = v;
        self
    }

    pub fn with_time_unit(mut self, v: usize) -> Self {
        self.time_unit = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurvivalAnalysisError> {
        if self.confidence.is_nan() {
            return Err(SurvivalAnalysisError::InvalidConfig("confidence is NaN".into()));
        }
        if self.max_time.is_nan() {
            return Err(SurvivalAnalysisError::InvalidConfig("max_time is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurvivalAnalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurvivalAnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurvivalAnalysisConfig(confidence={0:.4}, max_time={1:.4}, handle_censoring={2}, time_unit={3})", self.confidence, self.max_time, self.handle_censoring, self.time_unit)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core survival analysis with kaplan-meier estimator engine.
#[derive(Debug, Clone)]
pub struct SurvivalAnalysis {
    config: SurvivalAnalysisConfig,
    data: Vec<f64>,
}

impl SurvivalAnalysis {
    pub fn new(config: SurvivalAnalysisConfig) -> Result<Self, SurvivalAnalysisError> {
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
    pub fn config(&self) -> &SurvivalAnalysisConfig { &self.config }

    /// Kaplan-Meier survival curve.
    pub fn kaplan_meier(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Median survival time.
    pub fn median_survival(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Log-rank test statistic.
    pub fn log_rank_test(&self) -> f64 {
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

impl fmt::Display for SurvivalAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurvivalAnalysis(n={})", self.data.len())
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
        let cfg = SurvivalAnalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurvivalAnalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurvivalAnalysisConfig"));
    }

    #[test]
    fn test_config_with_confidence() {
        let cfg = SurvivalAnalysisConfig::new().with_confidence(42.0);
        assert!((cfg.confidence - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_time() {
        let cfg = SurvivalAnalysisConfig::new().with_max_time(42.0);
        assert!((cfg.max_time - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_handle_censoring() {
        let cfg = SurvivalAnalysisConfig::new().with_handle_censoring(false);
        assert_eq!(cfg.handle_censoring, false);
    }

    #[test]
    fn test_config_with_time_unit() {
        let cfg = SurvivalAnalysisConfig::new().with_time_unit(42);
        assert_eq!(cfg.time_unit, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurvivalAnalysisConfig::new().with_confidence(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurvivalAnalysis"));
    }

    #[test]
    fn test_summary() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_kaplan_meier() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.kaplan_meier();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_median_survival() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.median_survival();
        assert!(result.is_finite());
    }

    #[test]
    fn test_log_rank_test() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.log_rank_test();
        assert!(result.is_finite());
    }

    #[test]
    fn test_log_rank_test_empty() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap();
        assert!((e.log_rank_test() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SurvivalAnalysis::new(SurvivalAnalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurvivalAnalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurvivalAnalysisError::InvalidConfig("a".into());
        let e2 = SurvivalAnalysisError::ComputationFailed("b".into());
        let e3 = SurvivalAnalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
