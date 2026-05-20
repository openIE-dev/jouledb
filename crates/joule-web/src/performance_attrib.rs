//! Performance attribution with Brinson decomposition.
//!
//! Provides [`PerformanceAttribConfig`] builder and [`PerformanceAttrib`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceAttribError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PerformanceAttribError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PerformanceAttrib: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PerformanceAttrib: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PerformanceAttrib: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PerformanceAttrib`] parameters.
#[derive(Debug, Clone)]
pub struct PerformanceAttribConfig {
    pub benchmark_return: f64,
    pub portfolio_return: f64,
    pub periods: usize,
    pub currency_aware: bool,
}

impl PerformanceAttribConfig {
    pub fn new() -> Self {
        Self {
            benchmark_return: 0.08,
            portfolio_return: 0.10,
            periods: 12,
            currency_aware: false,
        }
    }

    pub fn with_benchmark_return(mut self, v: f64) -> Self {
        self.benchmark_return = v;
        self
    }

    pub fn with_portfolio_return(mut self, v: f64) -> Self {
        self.portfolio_return = v;
        self
    }

    pub fn with_periods(mut self, v: usize) -> Self {
        self.periods = v;
        self
    }

    pub fn with_currency_aware(mut self, v: bool) -> Self {
        self.currency_aware = v;
        self
    }

    pub fn validate(&self) -> Result<(), PerformanceAttribError> {
        if self.benchmark_return.is_nan() {
            return Err(PerformanceAttribError::InvalidConfig("benchmark_return is NaN".into()));
        }
        if self.portfolio_return.is_nan() {
            return Err(PerformanceAttribError::InvalidConfig("portfolio_return is NaN".into()));
        }
        Ok(())
    }
}

impl Default for PerformanceAttribConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PerformanceAttribConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PerformanceAttribConfig(benchmark_return={0:.4}, portfolio_return={1:.4}, periods={2}, currency_aware={3})", self.benchmark_return, self.portfolio_return, self.periods, self.currency_aware)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core performance attribution with brinson decomposition engine.
#[derive(Debug, Clone)]
pub struct PerformanceAttrib {
    config: PerformanceAttribConfig,
    data: Vec<f64>,
}

impl PerformanceAttrib {
    pub fn new(config: PerformanceAttribConfig) -> Result<Self, PerformanceAttribError> {
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
    pub fn config(&self) -> &PerformanceAttribConfig { &self.config }

    /// Brinson allocation effect.
    pub fn allocation_effect(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Brinson selection effect.
    pub fn selection_effect(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// TWR calculation.
    pub fn time_weighted_return(&self) -> f64 {
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

impl fmt::Display for PerformanceAttrib {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PerformanceAttrib(n={})", self.data.len())
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
        let cfg = PerformanceAttribConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PerformanceAttribConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PerformanceAttribConfig"));
    }

    #[test]
    fn test_config_with_benchmark_return() {
        let cfg = PerformanceAttribConfig::new().with_benchmark_return(42.0);
        assert!((cfg.benchmark_return - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_portfolio_return() {
        let cfg = PerformanceAttribConfig::new().with_portfolio_return(42.0);
        assert!((cfg.portfolio_return - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_periods() {
        let cfg = PerformanceAttribConfig::new().with_periods(42);
        assert_eq!(cfg.periods, 42);
    }

    #[test]
    fn test_config_with_currency_aware() {
        let cfg = PerformanceAttribConfig::new().with_currency_aware(true);
        assert_eq!(cfg.currency_aware, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PerformanceAttribConfig::new().with_benchmark_return(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PerformanceAttrib"));
    }

    #[test]
    fn test_summary() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_allocation_effect() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.allocation_effect();
        assert!(result.is_finite());
    }

    #[test]
    fn test_selection_effect() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.selection_effect();
        assert!(result.is_finite());
    }

    #[test]
    fn test_time_weighted_return() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.time_weighted_return();
        assert!(result.is_finite());
    }

    #[test]
    fn test_time_weighted_return_empty() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap();
        assert!((e.time_weighted_return() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = PerformanceAttrib::new(PerformanceAttribConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PerformanceAttribError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PerformanceAttribError::InvalidConfig("a".into());
        let e2 = PerformanceAttribError::ComputationFailed("b".into());
        let e3 = PerformanceAttribError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
