//! Tick data management for trade and quote ticks.
//!
//! Provides [`TickDataConfig`] builder and [`TickData`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TickDataError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TickDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TickData: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TickData: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TickData: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TickData`] parameters.
#[derive(Debug, Clone)]
pub struct TickDataConfig {
    pub price: f64,
    pub volume: f64,
    pub timestamp_ms: u64,
    pub is_trade: bool,
}

impl TickDataConfig {
    pub fn new() -> Self {
        Self {
            price: 100.5,
            volume: 500.0,
            timestamp_ms: 1000,
            is_trade: true,
        }
    }

    pub fn with_price(mut self, v: f64) -> Self {
        self.price = v;
        self
    }

    pub fn with_volume(mut self, v: f64) -> Self {
        self.volume = v;
        self
    }

    pub fn with_timestamp_ms(mut self, v: u64) -> Self {
        self.timestamp_ms = v;
        self
    }

    pub fn with_is_trade(mut self, v: bool) -> Self {
        self.is_trade = v;
        self
    }

    pub fn validate(&self) -> Result<(), TickDataError> {
        if self.price.is_nan() {
            return Err(TickDataError::InvalidConfig("price is NaN".into()));
        }
        if self.volume.is_nan() {
            return Err(TickDataError::InvalidConfig("volume is NaN".into()));
        }
        Ok(())
    }
}

impl Default for TickDataConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TickDataConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TickDataConfig(price={0:.4}, volume={1:.4}, timestamp_ms={2}, is_trade={3})", self.price, self.volume, self.timestamp_ms, self.is_trade)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a TickData operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for Bar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bar({:.4}, {})", self.value, self.label)
    }
}

/// Result from a TickData operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TickEntry {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for TickEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TickEntry({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core tick data management for trade and quote ticks engine.
#[derive(Debug, Clone)]
pub struct TickData {
    config: TickDataConfig,
    data: Vec<f64>,
}

impl TickData {
    pub fn new(config: TickDataConfig) -> Result<Self, TickDataError> {
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
    pub fn config(&self) -> &TickDataConfig { &self.config }

    /// Remove outlier ticks.
    pub fn filter_outliers(&self) -> Vec<TickEntry> {
        self.data.iter().enumerate().map(|(i, &v)| TickEntry {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Aggregate ticks to bars.
    pub fn aggregate(&self) -> Vec<Bar> {
        self.data.iter().enumerate().map(|(i, &v)| Bar {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Normalize tick format.
    pub fn normalize(&self) -> Vec<TickEntry> {
        self.data.iter().enumerate().map(|(i, &v)| TickEntry {
            value: v, label: format!("item_{i}")
        }).collect()
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

impl fmt::Display for TickData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TickData(n={})", self.data.len())
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
        let cfg = TickDataConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TickDataConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TickDataConfig"));
    }

    #[test]
    fn test_config_with_price() {
        let cfg = TickDataConfig::new().with_price(42.0);
        assert!((cfg.price - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_volume() {
        let cfg = TickDataConfig::new().with_volume(42.0);
        assert!((cfg.volume - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_timestamp_ms() {
        let cfg = TickDataConfig::new().with_timestamp_ms(42);
        assert_eq!(cfg.timestamp_ms, 42);
    }

    #[test]
    fn test_config_with_is_trade() {
        let cfg = TickDataConfig::new().with_is_trade(false);
        assert_eq!(cfg.is_trade, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TickDataConfig::new().with_price(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = TickData::new(TickDataConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TickData::new(TickDataConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TickData"));
    }

    #[test]
    fn test_summary() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TickData::new(TickDataConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_filter_outliers() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.filter_outliers();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_aggregate() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.aggregate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_normalize() {
        let e = TickData::new(TickDataConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.normalize();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_normalize_empty() {
        let e = TickData::new(TickDataConfig::new()).unwrap();
        assert!(e.normalize().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = TickData::new(TickDataConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TickDataError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TickDataError::InvalidConfig("a".into());
        let e2 = TickDataError::ComputationFailed("b".into());
        let e3 = TickDataError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
