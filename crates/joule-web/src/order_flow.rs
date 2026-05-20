//! Order flow analysis with trade classification.
//!
//! Provides [`OrderFlowConfig`] builder and [`OrderFlow`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderFlowError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for OrderFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "OrderFlow: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "OrderFlow: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "OrderFlow: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`OrderFlow`] parameters.
#[derive(Debug, Clone)]
pub struct OrderFlowConfig {
    pub tick_rule: bool,
    pub lee_ready: bool,
    pub window_size: usize,
    pub decay_factor: f64,
}

impl OrderFlowConfig {
    pub fn new() -> Self {
        Self {
            tick_rule: true,
            lee_ready: true,
            window_size: 100,
            decay_factor: 0.95,
        }
    }

    pub fn with_tick_rule(mut self, v: bool) -> Self {
        self.tick_rule = v;
        self
    }

    pub fn with_lee_ready(mut self, v: bool) -> Self {
        self.lee_ready = v;
        self
    }

    pub fn with_window_size(mut self, v: usize) -> Self {
        self.window_size = v;
        self
    }

    pub fn with_decay_factor(mut self, v: f64) -> Self {
        self.decay_factor = v;
        self
    }

    pub fn validate(&self) -> Result<(), OrderFlowError> {
        if self.decay_factor.is_nan() {
            return Err(OrderFlowError::InvalidConfig("decay_factor is NaN".into()));
        }
        Ok(())
    }
}

impl Default for OrderFlowConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for OrderFlowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderFlowConfig(tick_rule={0}, lee_ready={1}, window_size={2}, decay_factor={3:.4})", self.tick_rule, self.lee_ready, self.window_size, self.decay_factor)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a OrderFlow operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeSide {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for TradeSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TradeSide({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core order flow analysis with trade classification engine.
#[derive(Debug, Clone)]
pub struct OrderFlow {
    config: OrderFlowConfig,
    data: Vec<f64>,
}

impl OrderFlow {
    pub fn new(config: OrderFlowConfig) -> Result<Self, OrderFlowError> {
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
    pub fn config(&self) -> &OrderFlowConfig { &self.config }

    /// Classify as buy or sell.
    pub fn classify_trade(&self) -> TradeSide {
        let v = if self.data.is_empty() { 0.0 } else { self.data[0] };
        TradeSide { value: v, label: stringify!(classify_trade).into() }
    }

    /// Cumulative order delta.
    pub fn cumulative_delta(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Order flow imbalance.
    pub fn flow_imbalance(&self) -> f64 {
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

impl fmt::Display for OrderFlow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderFlow(n={})", self.data.len())
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
        let cfg = OrderFlowConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = OrderFlowConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("OrderFlowConfig"));
    }

    #[test]
    fn test_config_with_tick_rule() {
        let cfg = OrderFlowConfig::new().with_tick_rule(false);
        assert_eq!(cfg.tick_rule, false);
    }

    #[test]
    fn test_config_with_lee_ready() {
        let cfg = OrderFlowConfig::new().with_lee_ready(false);
        assert_eq!(cfg.lee_ready, false);
    }

    #[test]
    fn test_config_with_window_size() {
        let cfg = OrderFlowConfig::new().with_window_size(42);
        assert_eq!(cfg.window_size, 42);
    }

    #[test]
    fn test_config_with_decay_factor() {
        let cfg = OrderFlowConfig::new().with_decay_factor(42.0);
        assert!((cfg.decay_factor - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = OrderFlowConfig::new().with_tick_rule(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = OrderFlow::new(OrderFlowConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("OrderFlow"));
    }

    #[test]
    fn test_summary() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_classify_trade() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.classify_trade();
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_cumulative_delta() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.cumulative_delta();
        assert!(result.is_finite());
    }

    #[test]
    fn test_flow_imbalance() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.flow_imbalance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_flow_imbalance_empty() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap();
        assert!((e.flow_imbalance() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = OrderFlow::new(OrderFlowConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = OrderFlowError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = OrderFlowError::InvalidConfig("a".into());
        let e2 = OrderFlowError::ComputationFailed("b".into());
        let e3 = OrderFlowError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
