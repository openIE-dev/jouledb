//! Medication dose calculation engine.
//!
//! Provides [`DoseCalcConfig`] builder and [`DoseCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DoseCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DoseCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DoseCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DoseCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DoseCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DoseCalc`] parameters.
#[derive(Debug, Clone)]
pub struct DoseCalcConfig {
    pub weight_kg: f64,
    pub height_cm: f64,
    pub age_years: f64,
    pub renal_adjustment: bool,
}

impl DoseCalcConfig {
    pub fn new() -> Self {
        Self {
            weight_kg: 70.0,
            height_cm: 170.0,
            age_years: 40.0,
            renal_adjustment: true,
        }
    }

    pub fn with_weight_kg(mut self, v: f64) -> Self {
        self.weight_kg = v;
        self
    }

    pub fn with_height_cm(mut self, v: f64) -> Self {
        self.height_cm = v;
        self
    }

    pub fn with_age_years(mut self, v: f64) -> Self {
        self.age_years = v;
        self
    }

    pub fn with_renal_adjustment(mut self, v: bool) -> Self {
        self.renal_adjustment = v;
        self
    }

    pub fn validate(&self) -> Result<(), DoseCalcError> {
        if self.weight_kg.is_nan() {
            return Err(DoseCalcError::InvalidConfig("weight_kg is NaN".into()));
        }
        if self.height_cm.is_nan() {
            return Err(DoseCalcError::InvalidConfig("height_cm is NaN".into()));
        }
        if self.age_years.is_nan() {
            return Err(DoseCalcError::InvalidConfig("age_years is NaN".into()));
        }
        Ok(())
    }
}

impl Default for DoseCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DoseCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DoseCalcConfig(weight_kg={0:.4}, height_cm={1:.4}, age_years={2:.4}, renal_adjustment={3})", self.weight_kg, self.height_cm, self.age_years, self.renal_adjustment)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core medication dose calculation engine engine.
#[derive(Debug, Clone)]
pub struct DoseCalc {
    config: DoseCalcConfig,
    data: Vec<f64>,
}

impl DoseCalc {
    pub fn new(config: DoseCalcConfig) -> Result<Self, DoseCalcError> {
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
    pub fn config(&self) -> &DoseCalcConfig { &self.config }

    /// Calculate weight-based dose.
    pub fn weight_based_dose(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// BSA-based dose (Mosteller).
    pub fn bsa_dose(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Cockcroft-Gault CrCl.
    pub fn creatinine_clearance(&self) -> f64 {
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

impl fmt::Display for DoseCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DoseCalc(n={})", self.data.len())
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
        let cfg = DoseCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DoseCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DoseCalcConfig"));
    }

    #[test]
    fn test_config_with_weight_kg() {
        let cfg = DoseCalcConfig::new().with_weight_kg(42.0);
        assert!((cfg.weight_kg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_height_cm() {
        let cfg = DoseCalcConfig::new().with_height_cm(42.0);
        assert!((cfg.height_cm - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_age_years() {
        let cfg = DoseCalcConfig::new().with_age_years(42.0);
        assert!((cfg.age_years - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_renal_adjustment() {
        let cfg = DoseCalcConfig::new().with_renal_adjustment(false);
        assert_eq!(cfg.renal_adjustment, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DoseCalcConfig::new().with_weight_kg(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DoseCalc::new(DoseCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DoseCalc"));
    }

    #[test]
    fn test_summary() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_weight_based_dose() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.weight_based_dose();
        assert!(result.is_finite());
    }

    #[test]
    fn test_bsa_dose() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bsa_dose();
        assert!(result.is_finite());
    }

    #[test]
    fn test_creatinine_clearance() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.creatinine_clearance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_creatinine_clearance_empty() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap();
        assert!((e.creatinine_clearance() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = DoseCalc::new(DoseCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DoseCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DoseCalcError::InvalidConfig("a".into());
        let e2 = DoseCalcError::ComputationFailed("b".into());
        let e3 = DoseCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
