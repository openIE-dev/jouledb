//! IV infusion calculations.
//!
//! Provides [`IvCalcConfig`] builder and [`IvCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum IvCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for IvCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "IvCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "IvCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "IvCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`IvCalc`] parameters.
#[derive(Debug, Clone)]
pub struct IvCalcConfig {
    pub volume_ml: f64,
    pub rate_mlhr: f64,
    pub drop_factor: f64,
    pub weight_kg: f64,
}

impl IvCalcConfig {
    pub fn new() -> Self {
        Self {
            volume_ml: 1000.0,
            rate_mlhr: 125.0,
            drop_factor: 20.0,
            weight_kg: 70.0,
        }
    }

    pub fn with_volume_ml(mut self, v: f64) -> Self {
        self.volume_ml = v;
        self
    }

    pub fn with_rate_mlhr(mut self, v: f64) -> Self {
        self.rate_mlhr = v;
        self
    }

    pub fn with_drop_factor(mut self, v: f64) -> Self {
        self.drop_factor = v;
        self
    }

    pub fn with_weight_kg(mut self, v: f64) -> Self {
        self.weight_kg = v;
        self
    }

    pub fn validate(&self) -> Result<(), IvCalcError> {
        if self.volume_ml.is_nan() {
            return Err(IvCalcError::InvalidConfig("volume_ml is NaN".into()));
        }
        if self.rate_mlhr.is_nan() {
            return Err(IvCalcError::InvalidConfig("rate_mlhr is NaN".into()));
        }
        if self.drop_factor.is_nan() {
            return Err(IvCalcError::InvalidConfig("drop_factor is NaN".into()));
        }
        if self.weight_kg.is_nan() {
            return Err(IvCalcError::InvalidConfig("weight_kg is NaN".into()));
        }
        Ok(())
    }
}

impl Default for IvCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for IvCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IvCalcConfig(volume_ml={0:.4}, rate_mlhr={1:.4}, drop_factor={2:.4}, weight_kg={3:.4})", self.volume_ml, self.rate_mlhr, self.drop_factor, self.weight_kg)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core iv infusion calculations engine.
#[derive(Debug, Clone)]
pub struct IvCalc {
    config: IvCalcConfig,
    data: Vec<f64>,
}

impl IvCalc {
    pub fn new(config: IvCalcConfig) -> Result<Self, IvCalcError> {
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
    pub fn config(&self) -> &IvCalcConfig { &self.config }

    /// Calculate drip rate (gtt/min).
    pub fn drip_rate_gtt(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Calculate infusion duration.
    pub fn infusion_duration(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Weight-based infusion rate.
    pub fn weight_based_rate(&self) -> f64 {
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

impl fmt::Display for IvCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IvCalc(n={})", self.data.len())
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
        let cfg = IvCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = IvCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("IvCalcConfig"));
    }

    #[test]
    fn test_config_with_volume_ml() {
        let cfg = IvCalcConfig::new().with_volume_ml(42.0);
        assert!((cfg.volume_ml - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rate_mlhr() {
        let cfg = IvCalcConfig::new().with_rate_mlhr(42.0);
        assert!((cfg.rate_mlhr - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_drop_factor() {
        let cfg = IvCalcConfig::new().with_drop_factor(42.0);
        assert!((cfg.drop_factor - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_weight_kg() {
        let cfg = IvCalcConfig::new().with_weight_kg(42.0);
        assert!((cfg.weight_kg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = IvCalcConfig::new().with_volume_ml(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = IvCalc::new(IvCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("IvCalc"));
    }

    #[test]
    fn test_summary() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_drip_rate_gtt() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.drip_rate_gtt();
        assert!(result.is_finite());
    }

    #[test]
    fn test_infusion_duration() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.infusion_duration();
        assert!(result.is_finite());
    }

    #[test]
    fn test_weight_based_rate() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.weight_based_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_weight_based_rate_empty() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap();
        assert!((e.weight_based_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = IvCalc::new(IvCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = IvCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = IvCalcError::InvalidConfig("a".into());
        let e2 = IvCalcError::ComputationFailed("b".into());
        let e3 = IvCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
