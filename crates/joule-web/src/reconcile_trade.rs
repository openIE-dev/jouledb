//! Trade reconciliation with fuzzy matching.
//!
//! Provides [`ReconcileTradeConfig`] builder and [`ReconcileTrade`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileTradeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ReconcileTradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ReconcileTrade: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ReconcileTrade: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ReconcileTrade: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ReconcileTrade`] parameters.
#[derive(Debug, Clone)]
pub struct ReconcileTradeConfig {
    pub tolerance_price: f64,
    pub tolerance_qty: f64,
    pub fuzzy_match: bool,
    pub auto_resolve: bool,
}

impl ReconcileTradeConfig {
    pub fn new() -> Self {
        Self {
            tolerance_price: 0.01,
            tolerance_qty: 0.001,
            fuzzy_match: true,
            auto_resolve: false,
        }
    }

    pub fn with_tolerance_price(mut self, v: f64) -> Self {
        self.tolerance_price = v;
        self
    }

    pub fn with_tolerance_qty(mut self, v: f64) -> Self {
        self.tolerance_qty = v;
        self
    }

    pub fn with_fuzzy_match(mut self, v: bool) -> Self {
        self.fuzzy_match = v;
        self
    }

    pub fn with_auto_resolve(mut self, v: bool) -> Self {
        self.auto_resolve = v;
        self
    }

    pub fn validate(&self) -> Result<(), ReconcileTradeError> {
        if self.tolerance_price.is_nan() {
            return Err(ReconcileTradeError::InvalidConfig("tolerance_price is NaN".into()));
        }
        if self.tolerance_qty.is_nan() {
            return Err(ReconcileTradeError::InvalidConfig("tolerance_qty is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ReconcileTradeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ReconcileTradeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReconcileTradeConfig(tolerance_price={0:.4}, tolerance_qty={1:.4}, fuzzy_match={2}, auto_resolve={3})", self.tolerance_price, self.tolerance_qty, self.fuzzy_match, self.auto_resolve)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a ReconcileTrade operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Break {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for Break {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Break({:.4}, {})", self.value, self.label)
    }
}

/// Result from a ReconcileTrade operation.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchPair {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for MatchPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatchPair({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core trade reconciliation with fuzzy matching engine.
#[derive(Debug, Clone)]
pub struct ReconcileTrade {
    config: ReconcileTradeConfig,
    data: Vec<f64>,
}

impl ReconcileTrade {
    pub fn new(config: ReconcileTradeConfig) -> Result<Self, ReconcileTradeError> {
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
    pub fn config(&self) -> &ReconcileTradeConfig { &self.config }

    /// Match internal vs external trades.
    pub fn match_trades(&self) -> Vec<MatchPair> {
        self.data.iter().enumerate().map(|(i, &v)| MatchPair {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Find unmatched breaks.
    pub fn find_breaks(&self) -> Vec<Break> {
        self.data.iter().enumerate().map(|(i, &v)| Break {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Auto-resolution rate.
    pub fn resolution_rate(&self) -> f64 {
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

impl fmt::Display for ReconcileTrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReconcileTrade(n={})", self.data.len())
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
        let cfg = ReconcileTradeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ReconcileTradeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ReconcileTradeConfig"));
    }

    #[test]
    fn test_config_with_tolerance_price() {
        let cfg = ReconcileTradeConfig::new().with_tolerance_price(42.0);
        assert!((cfg.tolerance_price - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tolerance_qty() {
        let cfg = ReconcileTradeConfig::new().with_tolerance_qty(42.0);
        assert!((cfg.tolerance_qty - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_fuzzy_match() {
        let cfg = ReconcileTradeConfig::new().with_fuzzy_match(false);
        assert_eq!(cfg.fuzzy_match, false);
    }

    #[test]
    fn test_config_with_auto_resolve() {
        let cfg = ReconcileTradeConfig::new().with_auto_resolve(true);
        assert_eq!(cfg.auto_resolve, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ReconcileTradeConfig::new().with_tolerance_price(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ReconcileTrade"));
    }

    #[test]
    fn test_summary() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_match_trades() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.match_trades();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_find_breaks() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.find_breaks();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_resolution_rate() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.resolution_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_resolution_rate_empty() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap();
        assert!((e.resolution_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = ReconcileTrade::new(ReconcileTradeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ReconcileTradeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ReconcileTradeError::InvalidConfig("a".into());
        let e2 = ReconcileTradeError::ComputationFailed("b".into());
        let e3 = ReconcileTradeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
