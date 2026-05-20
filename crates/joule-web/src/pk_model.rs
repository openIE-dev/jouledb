//! Pharmacokinetic one-compartment modeling.
//!
//! Provides [`PkModelConfig`] builder and [`PkModel`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PkModelError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PkModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PkModel: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PkModel: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PkModel: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PkModel`] parameters.
#[derive(Debug, Clone)]
pub struct PkModelConfig {
    pub volume_dist: f64,
    pub clearance: f64,
    pub bioavailability: f64,
    pub dose_interval_hr: f64,
}

impl PkModelConfig {
    pub fn new() -> Self {
        Self {
            volume_dist: 50.0,
            clearance: 5.0,
            bioavailability: 1.0,
            dose_interval_hr: 8.0,
        }
    }

    pub fn with_volume_dist(mut self, v: f64) -> Self {
        self.volume_dist = v;
        self
    }

    pub fn with_clearance(mut self, v: f64) -> Self {
        self.clearance = v;
        self
    }

    pub fn with_bioavailability(mut self, v: f64) -> Self {
        self.bioavailability = v;
        self
    }

    pub fn with_dose_interval_hr(mut self, v: f64) -> Self {
        self.dose_interval_hr = v;
        self
    }

    pub fn validate(&self) -> Result<(), PkModelError> {
        if self.volume_dist.is_nan() {
            return Err(PkModelError::InvalidConfig("volume_dist is NaN".into()));
        }
        if self.clearance.is_nan() {
            return Err(PkModelError::InvalidConfig("clearance is NaN".into()));
        }
        if self.bioavailability.is_nan() {
            return Err(PkModelError::InvalidConfig("bioavailability is NaN".into()));
        }
        if self.dose_interval_hr.is_nan() {
            return Err(PkModelError::InvalidConfig("dose_interval_hr is NaN".into()));
        }
        Ok(())
    }
}

impl Default for PkModelConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PkModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PkModelConfig(volume_dist={0:.4}, clearance={1:.4}, bioavailability={2:.4}, dose_interval_hr={3:.4})", self.volume_dist, self.clearance, self.bioavailability, self.dose_interval_hr)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core pharmacokinetic one-compartment modeling engine.
#[derive(Debug, Clone)]
pub struct PkModel {
    config: PkModelConfig,
    data: Vec<f64>,
}

impl PkModel {
    pub fn new(config: PkModelConfig) -> Result<Self, PkModelError> {
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
    pub fn config(&self) -> &PkModelConfig { &self.config }

    /// Calculate elimination half-life.
    pub fn half_life(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Peak at steady state.
    pub fn steady_state_peak(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// AUC by trapezoidal rule.
    pub fn auc_trapezoidal(&self) -> f64 {
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

impl fmt::Display for PkModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PkModel(n={})", self.data.len())
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
        let cfg = PkModelConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PkModelConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PkModelConfig"));
    }

    #[test]
    fn test_config_with_volume_dist() {
        let cfg = PkModelConfig::new().with_volume_dist(42.0);
        assert!((cfg.volume_dist - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_clearance() {
        let cfg = PkModelConfig::new().with_clearance(42.0);
        assert!((cfg.clearance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bioavailability() {
        let cfg = PkModelConfig::new().with_bioavailability(42.0);
        assert!((cfg.bioavailability - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_dose_interval_hr() {
        let cfg = PkModelConfig::new().with_dose_interval_hr(42.0);
        assert!((cfg.dose_interval_hr - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PkModelConfig::new().with_volume_dist(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = PkModel::new(PkModelConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PkModel::new(PkModelConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PkModel"));
    }

    #[test]
    fn test_summary() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PkModel::new(PkModelConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_half_life() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.half_life();
        assert!(result.is_finite());
    }

    #[test]
    fn test_steady_state_peak() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.steady_state_peak();
        assert!(result.is_finite());
    }

    #[test]
    fn test_auc_trapezoidal() {
        let e = PkModel::new(PkModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.auc_trapezoidal();
        assert!(result.is_finite());
    }

    #[test]
    fn test_auc_trapezoidal_empty() {
        let e = PkModel::new(PkModelConfig::new()).unwrap();
        assert!((e.auc_trapezoidal() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = PkModel::new(PkModelConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PkModelError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PkModelError::InvalidConfig("a".into());
        let e2 = PkModelError::ComputationFailed("b".into());
        let e3 = PkModelError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
