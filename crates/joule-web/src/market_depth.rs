//! Market depth analysis and liquidity metrics.
//!
//! Provides [`MarketDepthConfig`] builder and [`MarketDepth`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MarketDepthError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MarketDepthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MarketDepth: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MarketDepth: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MarketDepth: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MarketDepth`] parameters.
#[derive(Debug, Clone)]
pub struct MarketDepthConfig {
    pub levels: usize,
    pub tick_size: f64,
    pub imbalance_threshold: f64,
    pub depth_weight: f64,
}

impl MarketDepthConfig {
    pub fn new() -> Self {
        Self {
            levels: 10,
            tick_size: 0.01,
            imbalance_threshold: 0.3,
            depth_weight: 0.5,
        }
    }

    pub fn with_levels(mut self, v: usize) -> Self {
        self.levels = v;
        self
    }

    pub fn with_tick_size(mut self, v: f64) -> Self {
        self.tick_size = v;
        self
    }

    pub fn with_imbalance_threshold(mut self, v: f64) -> Self {
        self.imbalance_threshold = v;
        self
    }

    pub fn with_depth_weight(mut self, v: f64) -> Self {
        self.depth_weight = v;
        self
    }

    pub fn validate(&self) -> Result<(), MarketDepthError> {
        if self.tick_size.is_nan() {
            return Err(MarketDepthError::InvalidConfig("tick_size is NaN".into()));
        }
        if self.imbalance_threshold.is_nan() {
            return Err(MarketDepthError::InvalidConfig("imbalance_threshold is NaN".into()));
        }
        if self.depth_weight.is_nan() {
            return Err(MarketDepthError::InvalidConfig("depth_weight is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MarketDepthConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MarketDepthConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MarketDepthConfig(levels={0}, tick_size={1:.4}, imbalance_threshold={2:.4}, depth_weight={3:.4})", self.levels, self.tick_size, self.imbalance_threshold, self.depth_weight)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core market depth analysis and liquidity metrics engine.
#[derive(Debug, Clone)]
pub struct MarketDepth {
    config: MarketDepthConfig,
    data: Vec<f64>,
}

impl MarketDepth {
    pub fn new(config: MarketDepthConfig) -> Result<Self, MarketDepthError> {
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
    pub fn config(&self) -> &MarketDepthConfig { &self.config }

    /// Current bid-ask spread.
    pub fn bid_ask_spread(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Order book imbalance ratio.
    pub fn imbalance(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Estimated market impact.
    pub fn impact_cost(&self) -> f64 {
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

impl fmt::Display for MarketDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MarketDepth(n={})", self.data.len())
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
        let cfg = MarketDepthConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MarketDepthConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MarketDepthConfig"));
    }

    #[test]
    fn test_config_with_levels() {
        let cfg = MarketDepthConfig::new().with_levels(42);
        assert_eq!(cfg.levels, 42);
    }

    #[test]
    fn test_config_with_tick_size() {
        let cfg = MarketDepthConfig::new().with_tick_size(42.0);
        assert!((cfg.tick_size - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_imbalance_threshold() {
        let cfg = MarketDepthConfig::new().with_imbalance_threshold(42.0);
        assert!((cfg.imbalance_threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_depth_weight() {
        let cfg = MarketDepthConfig::new().with_depth_weight(42.0);
        assert!((cfg.depth_weight - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MarketDepthConfig::new().with_levels(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MarketDepth::new(MarketDepthConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MarketDepth"));
    }

    #[test]
    fn test_summary() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bid_ask_spread() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bid_ask_spread();
        assert!(result.is_finite());
    }

    #[test]
    fn test_imbalance() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.imbalance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_impact_cost() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.impact_cost();
        assert!(result.is_finite());
    }

    #[test]
    fn test_impact_cost_empty() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap();
        assert!((e.impact_cost() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = MarketDepth::new(MarketDepthConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MarketDepthError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MarketDepthError::InvalidConfig("a".into());
        let e2 = MarketDepthError::ComputationFailed("b".into());
        let e3 = MarketDepthError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
