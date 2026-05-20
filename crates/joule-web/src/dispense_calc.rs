//! Pharmacy dispensing calculations.
//!
//! Provides [`DispenseCalcConfig`] builder and [`DispenseCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DispenseCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DispenseCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DispenseCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DispenseCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DispenseCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DispenseCalc`] parameters.
#[derive(Debug, Clone)]
pub struct DispenseCalcConfig {
    pub dose_mg: f64,
    pub frequency_daily: usize,
    pub days_supply: usize,
    pub tablet_strength: f64,
}

impl DispenseCalcConfig {
    pub fn new() -> Self {
        Self {
            dose_mg: 500.0,
            frequency_daily: 2,
            days_supply: 30,
            tablet_strength: 250.0,
        }
    }

    pub fn with_dose_mg(mut self, v: f64) -> Self {
        self.dose_mg = v;
        self
    }

    pub fn with_frequency_daily(mut self, v: usize) -> Self {
        self.frequency_daily = v;
        self
    }

    pub fn with_days_supply(mut self, v: usize) -> Self {
        self.days_supply = v;
        self
    }

    pub fn with_tablet_strength(mut self, v: f64) -> Self {
        self.tablet_strength = v;
        self
    }

    pub fn validate(&self) -> Result<(), DispenseCalcError> {
        if self.dose_mg.is_nan() {
            return Err(DispenseCalcError::InvalidConfig("dose_mg is NaN".into()));
        }
        if self.tablet_strength.is_nan() {
            return Err(DispenseCalcError::InvalidConfig("tablet_strength is NaN".into()));
        }
        Ok(())
    }
}

impl Default for DispenseCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DispenseCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DispenseCalcConfig(dose_mg={0:.4}, frequency_daily={1}, days_supply={2}, tablet_strength={3:.4})", self.dose_mg, self.frequency_daily, self.days_supply, self.tablet_strength)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core pharmacy dispensing calculations engine.
#[derive(Debug, Clone)]
pub struct DispenseCalc {
    config: DispenseCalcConfig,
    data: Vec<f64>,
}

impl DispenseCalc {
    pub fn new(config: DispenseCalcConfig) -> Result<Self, DispenseCalcError> {
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
    pub fn config(&self) -> &DispenseCalcConfig { &self.config }

    /// Calculate quantity to dispense.
    pub fn quantity_to_dispense(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Calculate days supply.
    pub fn days_supply_calc(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check for early refill.
    pub fn early_refill_check(&self) -> bool {
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

impl fmt::Display for DispenseCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DispenseCalc(n={})", self.data.len())
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
        let cfg = DispenseCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DispenseCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DispenseCalcConfig"));
    }

    #[test]
    fn test_config_with_dose_mg() {
        let cfg = DispenseCalcConfig::new().with_dose_mg(42.0);
        assert!((cfg.dose_mg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_frequency_daily() {
        let cfg = DispenseCalcConfig::new().with_frequency_daily(42);
        assert_eq!(cfg.frequency_daily, 42);
    }

    #[test]
    fn test_config_with_days_supply() {
        let cfg = DispenseCalcConfig::new().with_days_supply(42);
        assert_eq!(cfg.days_supply, 42);
    }

    #[test]
    fn test_config_with_tablet_strength() {
        let cfg = DispenseCalcConfig::new().with_tablet_strength(42.0);
        assert!((cfg.tablet_strength - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DispenseCalcConfig::new().with_dose_mg(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DispenseCalc"));
    }

    #[test]
    fn test_summary() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_quantity_to_dispense() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.quantity_to_dispense();
        assert!(result.is_finite());
    }

    #[test]
    fn test_days_supply_calc() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.days_supply_calc();
        assert!(result.is_finite());
    }

    #[test]
    fn test_early_refill_check() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.early_refill_check();
        assert!(result);
    }

    #[test]
    fn test_early_refill_check_empty() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap();
        assert!(!e.early_refill_check());
    }

    #[test]
    fn test_config_accessor() {
        let e = DispenseCalc::new(DispenseCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DispenseCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DispenseCalcError::InvalidConfig("a".into());
        let e2 = DispenseCalcError::ComputationFailed("b".into());
        let e3 = DispenseCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
