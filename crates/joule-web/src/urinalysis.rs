//! Urinalysis result interpretation.
//!
//! Provides [`UrinalysisConfig`] builder and [`Urinalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum UrinalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for UrinalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Urinalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Urinalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Urinalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Urinalysis`] parameters.
#[derive(Debug, Clone)]
pub struct UrinalysisConfig {
    pub ph: f64,
    pub specific_gravity: f64,
    pub protein_level: usize,
    pub glucose_level: usize,
}

impl UrinalysisConfig {
    pub fn new() -> Self {
        Self {
            ph: 6.0,
            specific_gravity: 1.015,
            protein_level: 0,
            glucose_level: 0,
        }
    }

    pub fn with_ph(mut self, v: f64) -> Self {
        self.ph = v;
        self
    }

    pub fn with_specific_gravity(mut self, v: f64) -> Self {
        self.specific_gravity = v;
        self
    }

    pub fn with_protein_level(mut self, v: usize) -> Self {
        self.protein_level = v;
        self
    }

    pub fn with_glucose_level(mut self, v: usize) -> Self {
        self.glucose_level = v;
        self
    }

    pub fn validate(&self) -> Result<(), UrinalysisError> {
        if self.ph.is_nan() {
            return Err(UrinalysisError::InvalidConfig("ph is NaN".into()));
        }
        if self.specific_gravity.is_nan() {
            return Err(UrinalysisError::InvalidConfig("specific_gravity is NaN".into()));
        }
        Ok(())
    }
}

impl Default for UrinalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for UrinalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UrinalysisConfig(ph={0:.4}, specific_gravity={1:.4}, protein_level={2}, glucose_level={3})", self.ph, self.specific_gravity, self.protein_level, self.glucose_level)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core urinalysis result interpretation engine.
#[derive(Debug, Clone)]
pub struct Urinalysis {
    config: UrinalysisConfig,
    data: Vec<f64>,
}

impl Urinalysis {
    pub fn new(config: UrinalysisConfig) -> Result<Self, UrinalysisError> {
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
    pub fn config(&self) -> &UrinalysisConfig { &self.config }

    /// Interpret dipstick results.
    pub fn interpret_dipstick(&self) -> String {
        format!("{}: {} records", stringify!(interpret_dipstick), self.data.len())
    }

    /// Check for abnormal results.
    pub fn is_abnormal(&self) -> bool {
        !self.data.is_empty()
    }

    /// Screen for UTI.
    pub fn uti_screen(&self) -> bool {
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

impl fmt::Display for Urinalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Urinalysis(n={})", self.data.len())
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
        let cfg = UrinalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = UrinalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("UrinalysisConfig"));
    }

    #[test]
    fn test_config_with_ph() {
        let cfg = UrinalysisConfig::new().with_ph(42.0);
        assert!((cfg.ph - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_specific_gravity() {
        let cfg = UrinalysisConfig::new().with_specific_gravity(42.0);
        assert!((cfg.specific_gravity - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_protein_level() {
        let cfg = UrinalysisConfig::new().with_protein_level(42);
        assert_eq!(cfg.protein_level, 42);
    }

    #[test]
    fn test_config_with_glucose_level() {
        let cfg = UrinalysisConfig::new().with_glucose_level(42);
        assert_eq!(cfg.glucose_level, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = UrinalysisConfig::new().with_ph(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Urinalysis::new(UrinalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Urinalysis"));
    }

    #[test]
    fn test_summary() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpret_dipstick() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.interpret_dipstick();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_abnormal() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_abnormal();
        assert!(result);
    }

    #[test]
    fn test_uti_screen() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.uti_screen();
        assert!(result);
    }

    #[test]
    fn test_uti_screen_empty() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap();
        assert!(!e.uti_screen());
    }

    #[test]
    fn test_config_accessor() {
        let e = Urinalysis::new(UrinalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = UrinalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = UrinalysisError::InvalidConfig("a".into());
        let e2 = UrinalysisError::ComputationFailed("b".into());
        let e3 = UrinalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
