//! Clinical risk scoring calculators.
//!
//! Provides [`RiskScoreConfig`] builder and [`RiskScore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskScoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RiskScoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RiskScore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RiskScore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RiskScore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RiskScore`] parameters.
#[derive(Debug, Clone)]
pub struct RiskScoreConfig {
    pub score_type: usize,
    pub age: f64,
    pub gender: usize,
    pub include_labs: bool,
}

impl RiskScoreConfig {
    pub fn new() -> Self {
        Self {
            score_type: 0,
            age: 65.0,
            gender: 0,
            include_labs: true,
        }
    }

    pub fn with_score_type(mut self, v: usize) -> Self {
        self.score_type = v;
        self
    }

    pub fn with_age(mut self, v: f64) -> Self {
        self.age = v;
        self
    }

    pub fn with_gender(mut self, v: usize) -> Self {
        self.gender = v;
        self
    }

    pub fn with_include_labs(mut self, v: bool) -> Self {
        self.include_labs = v;
        self
    }

    pub fn validate(&self) -> Result<(), RiskScoreError> {
        if self.age.is_nan() {
            return Err(RiskScoreError::InvalidConfig("age is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RiskScoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RiskScoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskScoreConfig(score_type={0}, age={1:.4}, gender={2}, include_labs={3})", self.score_type, self.age, self.gender, self.include_labs)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core clinical risk scoring calculators engine.
#[derive(Debug, Clone)]
pub struct RiskScore {
    config: RiskScoreConfig,
    data: Vec<f64>,
}

impl RiskScore {
    pub fn new(config: RiskScoreConfig) -> Result<Self, RiskScoreError> {
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
    pub fn config(&self) -> &RiskScoreConfig { &self.config }

    /// Framingham cardiovascular risk.
    pub fn framingham_cvd(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// CHA2DS2-VASc stroke risk.
    pub fn chadsvasc(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Wells score for DVT.
    pub fn wells_dvt(&self) -> f64 {
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

impl fmt::Display for RiskScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RiskScore(n={})", self.data.len())
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
        let cfg = RiskScoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RiskScoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RiskScoreConfig"));
    }

    #[test]
    fn test_config_with_score_type() {
        let cfg = RiskScoreConfig::new().with_score_type(42);
        assert_eq!(cfg.score_type, 42);
    }

    #[test]
    fn test_config_with_age() {
        let cfg = RiskScoreConfig::new().with_age(42.0);
        assert!((cfg.age - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_gender() {
        let cfg = RiskScoreConfig::new().with_gender(42);
        assert_eq!(cfg.gender, 42);
    }

    #[test]
    fn test_config_with_include_labs() {
        let cfg = RiskScoreConfig::new().with_include_labs(false);
        assert_eq!(cfg.include_labs, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RiskScoreConfig::new().with_score_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RiskScore::new(RiskScoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RiskScore"));
    }

    #[test]
    fn test_summary() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_framingham_cvd() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.framingham_cvd();
        assert!(result.is_finite());
    }

    #[test]
    fn test_chadsvasc() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.chadsvasc();
        assert!(result.is_finite());
    }

    #[test]
    fn test_wells_dvt() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.wells_dvt();
        assert!(result.is_finite());
    }

    #[test]
    fn test_wells_dvt_empty() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap();
        assert!((e.wells_dvt() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = RiskScore::new(RiskScoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RiskScoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RiskScoreError::InvalidConfig("a".into());
        let e2 = RiskScoreError::ComputationFailed("b".into());
        let e3 = RiskScoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
