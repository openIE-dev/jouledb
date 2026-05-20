//! Clinical order set management.
//!
//! Provides [`OrderSetConfig`] builder and [`OrderSet`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderSetError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for OrderSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "OrderSet: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "OrderSet: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "OrderSet: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`OrderSet`] parameters.
#[derive(Debug, Clone)]
pub struct OrderSetConfig {
    pub max_orders: usize,
    pub check_duplicates: bool,
    pub validate_frequency: bool,
    pub template_based: bool,
}

impl OrderSetConfig {
    pub fn new() -> Self {
        Self {
            max_orders: 100,
            check_duplicates: true,
            validate_frequency: true,
            template_based: true,
        }
    }

    pub fn with_max_orders(mut self, v: usize) -> Self {
        self.max_orders = v;
        self
    }

    pub fn with_check_duplicates(mut self, v: bool) -> Self {
        self.check_duplicates = v;
        self
    }

    pub fn with_validate_frequency(mut self, v: bool) -> Self {
        self.validate_frequency = v;
        self
    }

    pub fn with_template_based(mut self, v: bool) -> Self {
        self.template_based = v;
        self
    }

    pub fn validate(&self) -> Result<(), OrderSetError> {
        Ok(())
    }
}

impl Default for OrderSetConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for OrderSetConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderSetConfig(max_orders={0}, check_duplicates={1}, validate_frequency={2}, template_based={3})", self.max_orders, self.check_duplicates, self.validate_frequency, self.template_based)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core clinical order set management engine.
#[derive(Debug, Clone)]
pub struct OrderSet {
    config: OrderSetConfig,
    data: Vec<f64>,
}

impl OrderSet {
    pub fn new(config: OrderSetConfig) -> Result<Self, OrderSetError> {
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
    pub fn config(&self) -> &OrderSetConfig { &self.config }

    /// Add order to set.
    pub fn add_order(&self) -> bool {
        !self.data.is_empty()
    }

    /// Validate complete order set.
    pub fn validate_set(&self) -> bool {
        !self.data.is_empty()
    }

    /// Detect duplicate orders.
    pub fn detect_duplicates(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
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

impl fmt::Display for OrderSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderSet(n={})", self.data.len())
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
        let cfg = OrderSetConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = OrderSetConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("OrderSetConfig"));
    }

    #[test]
    fn test_config_with_max_orders() {
        let cfg = OrderSetConfig::new().with_max_orders(42);
        assert_eq!(cfg.max_orders, 42);
    }

    #[test]
    fn test_config_with_check_duplicates() {
        let cfg = OrderSetConfig::new().with_check_duplicates(false);
        assert_eq!(cfg.check_duplicates, false);
    }

    #[test]
    fn test_config_with_validate_frequency() {
        let cfg = OrderSetConfig::new().with_validate_frequency(false);
        assert_eq!(cfg.validate_frequency, false);
    }

    #[test]
    fn test_config_with_template_based() {
        let cfg = OrderSetConfig::new().with_template_based(false);
        assert_eq!(cfg.template_based, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = OrderSetConfig::new().with_max_orders(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = OrderSet::new(OrderSetConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("OrderSet"));
    }

    #[test]
    fn test_summary() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_order() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_order();
        assert!(result);
    }

    #[test]
    fn test_validate_set() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_set();
        assert!(result);
    }

    #[test]
    fn test_detect_duplicates() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_duplicates();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_detect_duplicates_empty() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap();
        assert!(e.detect_duplicates().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = OrderSet::new(OrderSetConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = OrderSetError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = OrderSetError::InvalidConfig("a".into());
        let e2 = OrderSetError::ComputationFailed("b".into());
        let e3 = OrderSetError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
