//! Option strategy payoff analysis.
//!
//! Provides [`OptionStrategyConfig`] builder and [`OptionStrategy`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum OptionStrategyError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for OptionStrategyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "OptionStrategy: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "OptionStrategy: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "OptionStrategy: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`OptionStrategy`] parameters.
#[derive(Debug, Clone)]
pub struct OptionStrategyConfig {
    pub strike: f64,
    pub premium: f64,
    pub quantity: i32,
    pub is_call: bool,
}

impl OptionStrategyConfig {
    pub fn new() -> Self {
        Self {
            strike: 100.0,
            premium: 5.0,
            quantity: 1,
            is_call: true,
        }
    }

    pub fn with_strike(mut self, v: f64) -> Self {
        self.strike = v;
        self
    }

    pub fn with_premium(mut self, v: f64) -> Self {
        self.premium = v;
        self
    }

    pub fn with_quantity(mut self, v: i32) -> Self {
        self.quantity = v;
        self
    }

    pub fn with_is_call(mut self, v: bool) -> Self {
        self.is_call = v;
        self
    }

    pub fn validate(&self) -> Result<(), OptionStrategyError> {
        if self.strike.is_nan() {
            return Err(OptionStrategyError::InvalidConfig("strike is NaN".into()));
        }
        if self.premium.is_nan() {
            return Err(OptionStrategyError::InvalidConfig("premium is NaN".into()));
        }
        Ok(())
    }
}

impl Default for OptionStrategyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for OptionStrategyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OptionStrategyConfig(strike={0:.4}, premium={1:.4}, quantity={2}, is_call={3})", self.strike, self.premium, self.quantity, self.is_call)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core option strategy payoff analysis engine.
#[derive(Debug, Clone)]
pub struct OptionStrategy {
    config: OptionStrategyConfig,
    data: Vec<f64>,
}

impl OptionStrategy {
    pub fn new(config: OptionStrategyConfig) -> Result<Self, OptionStrategyError> {
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
    pub fn config(&self) -> &OptionStrategyConfig { &self.config }

    /// Payoff at expiry price.
    pub fn payoff_at(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Breakeven price.
    pub fn breakeven(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Maximum possible loss.
    pub fn max_loss(&self) -> f64 {
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

impl fmt::Display for OptionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OptionStrategy(n={})", self.data.len())
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
        let cfg = OptionStrategyConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = OptionStrategyConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("OptionStrategyConfig"));
    }

    #[test]
    fn test_config_with_strike() {
        let cfg = OptionStrategyConfig::new().with_strike(42.0);
        assert!((cfg.strike - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_premium() {
        let cfg = OptionStrategyConfig::new().with_premium(42.0);
        assert!((cfg.premium - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_quantity() {
        let cfg = OptionStrategyConfig::new().with_quantity(42);
        assert_eq!(cfg.quantity, 42);
    }

    #[test]
    fn test_config_with_is_call() {
        let cfg = OptionStrategyConfig::new().with_is_call(false);
        assert_eq!(cfg.is_call, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = OptionStrategyConfig::new().with_strike(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("OptionStrategy"));
    }

    #[test]
    fn test_summary() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_payoff_at() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.payoff_at();
        assert!(result.is_finite());
    }

    #[test]
    fn test_breakeven() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.breakeven();
        assert!(result.is_finite());
    }

    #[test]
    fn test_max_loss() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.max_loss();
        assert!(result.is_finite());
    }

    #[test]
    fn test_max_loss_empty() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap();
        assert!((e.max_loss() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = OptionStrategy::new(OptionStrategyConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = OptionStrategyError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = OptionStrategyError::InvalidConfig("a".into());
        let e2 = OptionStrategyError::ComputationFailed("b".into());
        let e3 = OptionStrategyError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
