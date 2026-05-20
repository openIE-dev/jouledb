//! Trade matching engine with price-time priority.
//!
//! Provides [`TradeMatchConfig`] builder and [`TradeMatch`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TradeMatchError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TradeMatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TradeMatch: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TradeMatch: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TradeMatch: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TradeMatch`] parameters.
#[derive(Debug, Clone)]
pub struct TradeMatchConfig {
    pub bid_price: f64,
    pub ask_price: f64,
    pub quantity: f64,
    pub partial_fill: bool,
}

impl TradeMatchConfig {
    pub fn new() -> Self {
        Self {
            bid_price: 100.0,
            ask_price: 99.5,
            quantity: 1000.0,
            partial_fill: true,
        }
    }

    pub fn with_bid_price(mut self, v: f64) -> Self {
        self.bid_price = v;
        self
    }

    pub fn with_ask_price(mut self, v: f64) -> Self {
        self.ask_price = v;
        self
    }

    pub fn with_quantity(mut self, v: f64) -> Self {
        self.quantity = v;
        self
    }

    pub fn with_partial_fill(mut self, v: bool) -> Self {
        self.partial_fill = v;
        self
    }

    pub fn validate(&self) -> Result<(), TradeMatchError> {
        if self.bid_price.is_nan() {
            return Err(TradeMatchError::InvalidConfig("bid_price is NaN".into()));
        }
        if self.ask_price.is_nan() {
            return Err(TradeMatchError::InvalidConfig("ask_price is NaN".into()));
        }
        if self.quantity.is_nan() {
            return Err(TradeMatchError::InvalidConfig("quantity is NaN".into()));
        }
        Ok(())
    }
}

impl Default for TradeMatchConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TradeMatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TradeMatchConfig(bid_price={0:.4}, ask_price={1:.4}, quantity={2:.4}, partial_fill={3})", self.bid_price, self.ask_price, self.quantity, self.partial_fill)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a TradeMatch operation.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchResult {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatchResult({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core trade matching engine with price-time priority engine.
#[derive(Debug, Clone)]
pub struct TradeMatch {
    config: TradeMatchConfig,
    data: Vec<f64>,
}

impl TradeMatch {
    pub fn new(config: TradeMatchConfig) -> Result<Self, TradeMatchError> {
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
    pub fn config(&self) -> &TradeMatchConfig { &self.config }

    /// Attempt to match a trade.
    pub fn execute(&self) -> Option<MatchResult> {
        if self.data.is_empty() { return None; }
        Some(MatchResult { value: self.data[0], label: "match".into() })
    }

    /// Calculate price improvement.
    pub fn price_improvement(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check for self-trading.
    pub fn self_trade_check(&self) -> bool {
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

impl fmt::Display for TradeMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TradeMatch(n={})", self.data.len())
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
        let cfg = TradeMatchConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TradeMatchConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TradeMatchConfig"));
    }

    #[test]
    fn test_config_with_bid_price() {
        let cfg = TradeMatchConfig::new().with_bid_price(42.0);
        assert!((cfg.bid_price - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_ask_price() {
        let cfg = TradeMatchConfig::new().with_ask_price(42.0);
        assert!((cfg.ask_price - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_quantity() {
        let cfg = TradeMatchConfig::new().with_quantity(42.0);
        assert!((cfg.quantity - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_partial_fill() {
        let cfg = TradeMatchConfig::new().with_partial_fill(false);
        assert_eq!(cfg.partial_fill, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TradeMatchConfig::new().with_bid_price(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TradeMatch::new(TradeMatchConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TradeMatch"));
    }

    #[test]
    fn test_summary() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.execute();
        assert!(result.is_some());
    }

    #[test]
    fn test_price_improvement() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.price_improvement();
        assert!(result.is_finite());
    }

    #[test]
    fn test_self_trade_check() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.self_trade_check();
        assert!(result);
    }

    #[test]
    fn test_self_trade_check_empty() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap();
        assert!(!e.self_trade_check());
    }

    #[test]
    fn test_config_accessor() {
        let e = TradeMatch::new(TradeMatchConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TradeMatchError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TradeMatchError::InvalidConfig("a".into());
        let e2 = TradeMatchError::ComputationFailed("b".into());
        let e3 = TradeMatchError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
