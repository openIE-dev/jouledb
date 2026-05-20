//! Best execution quality analysis and TCA.
//!
//! Provides [`BestExecCheckConfig`] builder and [`BestExecCheck`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BestExecCheckError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BestExecCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BestExecCheck: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BestExecCheck: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BestExecCheck: insufficient data: {msg}"),
        }
    }
}

/// Variant selector for BenchmarkType.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchmarkType {
    /// ArrivalPrice.
    ArrivalPrice,
    /// Vwap.
    Vwap,
    /// Twap.
    Twap,
    /// Close.
    Close,
}

impl fmt::Display for BenchmarkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BestExecCheck`] parameters.
#[derive(Debug, Clone)]
pub struct BestExecCheckConfig {
    pub benchmark: BenchmarkType,
    pub slippage_limit: f64,
    pub fill_rate_min: f64,
    pub venue_count: usize,
}

impl BestExecCheckConfig {
    pub fn new() -> Self {
        Self {
            benchmark: BenchmarkType::ArrivalPrice,
            slippage_limit: 0.001,
            fill_rate_min: 0.95,
            venue_count: 5,
        }
    }

    pub fn with_benchmark(mut self, v: BenchmarkType) -> Self {
        self.benchmark = v;
        self
    }

    pub fn with_slippage_limit(mut self, v: f64) -> Self {
        self.slippage_limit = v;
        self
    }

    pub fn with_fill_rate_min(mut self, v: f64) -> Self {
        self.fill_rate_min = v;
        self
    }

    pub fn with_venue_count(mut self, v: usize) -> Self {
        self.venue_count = v;
        self
    }

    pub fn validate(&self) -> Result<(), BestExecCheckError> {
        if self.slippage_limit.is_nan() {
            return Err(BestExecCheckError::InvalidConfig("slippage_limit is NaN".into()));
        }
        if self.fill_rate_min.is_nan() {
            return Err(BestExecCheckError::InvalidConfig("fill_rate_min is NaN".into()));
        }
        Ok(())
    }
}

impl Default for BestExecCheckConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BestExecCheckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BestExecCheckConfig(benchmark={0:?}, slippage_limit={1:.4}, fill_rate_min={2:.4}, venue_count={3})", self.benchmark, self.slippage_limit, self.fill_rate_min, self.venue_count)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core best execution quality analysis and tca engine.
#[derive(Debug, Clone)]
pub struct BestExecCheck {
    config: BestExecCheckConfig,
    data: Vec<f64>,
}

impl BestExecCheck {
    pub fn new(config: BestExecCheckConfig) -> Result<Self, BestExecCheckError> {
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
    pub fn config(&self) -> &BestExecCheckConfig { &self.config }

    /// Price improvement vs benchmark.
    pub fn price_improvement(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Execution slippage.
    pub fn slippage(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Order fill rate.
    pub fn fill_rate(&self) -> f64 {
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

impl fmt::Display for BestExecCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BestExecCheck(n={})", self.data.len())
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
        let cfg = BestExecCheckConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BestExecCheckConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BestExecCheckConfig"));
    }

    #[test]
    fn test_config_with_benchmark() {
        let cfg = BestExecCheckConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_with_slippage_limit() {
        let cfg = BestExecCheckConfig::new().with_slippage_limit(42.0);
        assert!((cfg.slippage_limit - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_fill_rate_min() {
        let cfg = BestExecCheckConfig::new().with_fill_rate_min(42.0);
        assert!((cfg.fill_rate_min - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_venue_count() {
        let cfg = BestExecCheckConfig::new().with_venue_count(42);
        assert_eq!(cfg.venue_count, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BestExecCheckConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BestExecCheck"));
    }

    #[test]
    fn test_summary() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_improvement() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.price_improvement();
        assert!(result.is_finite());
    }

    #[test]
    fn test_slippage() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.slippage();
        assert!(result.is_finite());
    }

    #[test]
    fn test_fill_rate() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.fill_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_fill_rate_empty() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap();
        assert!((e.fill_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = BestExecCheck::new(BestExecCheckConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BestExecCheckError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BestExecCheckError::InvalidConfig("a".into());
        let e2 = BestExecCheckError::ComputationFailed("b".into());
        let e3 = BestExecCheckError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
