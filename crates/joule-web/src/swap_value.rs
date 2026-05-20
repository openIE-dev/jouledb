//! Interest rate swap valuation.
//!
//! Provides [`SwapValueConfig`] builder and [`SwapValue`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SwapValueError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SwapValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SwapValue: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SwapValue: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SwapValue: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SwapValue`] parameters.
#[derive(Debug, Clone)]
pub struct SwapValueConfig {
    pub notional: f64,
    pub fixed_rate: f64,
    pub float_spread: f64,
    pub tenor_years: f64,
}

impl SwapValueConfig {
    pub fn new() -> Self {
        Self {
            notional: 10_000_000.0,
            fixed_rate: 0.03,
            float_spread: 0.001,
            tenor_years: 5.0,
        }
    }

    pub fn with_notional(mut self, v: f64) -> Self {
        self.notional = v;
        self
    }

    pub fn with_fixed_rate(mut self, v: f64) -> Self {
        self.fixed_rate = v;
        self
    }

    pub fn with_float_spread(mut self, v: f64) -> Self {
        self.float_spread = v;
        self
    }

    pub fn with_tenor_years(mut self, v: f64) -> Self {
        self.tenor_years = v;
        self
    }

    pub fn validate(&self) -> Result<(), SwapValueError> {
        if self.notional.is_nan() {
            return Err(SwapValueError::InvalidConfig("notional is NaN".into()));
        }
        if self.fixed_rate.is_nan() {
            return Err(SwapValueError::InvalidConfig("fixed_rate is NaN".into()));
        }
        if self.float_spread.is_nan() {
            return Err(SwapValueError::InvalidConfig("float_spread is NaN".into()));
        }
        if self.tenor_years.is_nan() {
            return Err(SwapValueError::InvalidConfig("tenor_years is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SwapValueConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SwapValueConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SwapValueConfig(notional={0:.4}, fixed_rate={1:.4}, float_spread={2:.4}, tenor_years={3:.4})", self.notional, self.fixed_rate, self.float_spread, self.tenor_years)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core interest rate swap valuation engine.
#[derive(Debug, Clone)]
pub struct SwapValue {
    config: SwapValueConfig,
    data: Vec<f64>,
}

impl SwapValue {
    pub fn new(config: SwapValueConfig) -> Result<Self, SwapValueError> {
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
    pub fn config(&self) -> &SwapValueConfig { &self.config }

    /// Net present value.
    pub fn npv(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Dollar value of a basis point.
    pub fn dv01(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Par swap rate.
    pub fn par_rate(&self) -> f64 {
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

impl fmt::Display for SwapValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SwapValue(n={})", self.data.len())
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
        let cfg = SwapValueConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SwapValueConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SwapValueConfig"));
    }

    #[test]
    fn test_config_with_notional() {
        let cfg = SwapValueConfig::new().with_notional(42.0);
        assert!((cfg.notional - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_fixed_rate() {
        let cfg = SwapValueConfig::new().with_fixed_rate(42.0);
        assert!((cfg.fixed_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_float_spread() {
        let cfg = SwapValueConfig::new().with_float_spread(42.0);
        assert!((cfg.float_spread - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tenor_years() {
        let cfg = SwapValueConfig::new().with_tenor_years(42.0);
        assert!((cfg.tenor_years - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SwapValueConfig::new().with_notional(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SwapValue::new(SwapValueConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SwapValue"));
    }

    #[test]
    fn test_summary() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_npv() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.npv();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dv01() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.dv01();
        assert!(result.is_finite());
    }

    #[test]
    fn test_par_rate() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.par_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_par_rate_empty() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap();
        assert!((e.par_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SwapValue::new(SwapValueConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SwapValueError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SwapValueError::InvalidConfig("a".into());
        let e2 = SwapValueError::ComputationFailed("b".into());
        let e3 = SwapValueError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
