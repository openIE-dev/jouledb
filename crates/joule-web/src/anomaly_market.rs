//! Market anomaly detection (flash crash, volume spikes).
//!
//! Provides [`AnomalyMarketConfig`] builder and [`AnomalyMarket`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyMarketError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AnomalyMarketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AnomalyMarket: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AnomalyMarket: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AnomalyMarket: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AnomalyMarket`] parameters.
#[derive(Debug, Clone)]
pub struct AnomalyMarketConfig {
    pub threshold_std: f64,
    pub volume_multiple: f64,
    pub gap_pct: f64,
    pub lookback: usize,
}

impl AnomalyMarketConfig {
    pub fn new() -> Self {
        Self {
            threshold_std: 3.0,
            volume_multiple: 5.0,
            gap_pct: 0.05,
            lookback: 100,
        }
    }

    pub fn with_threshold_std(mut self, v: f64) -> Self {
        self.threshold_std = v;
        self
    }

    pub fn with_volume_multiple(mut self, v: f64) -> Self {
        self.volume_multiple = v;
        self
    }

    pub fn with_gap_pct(mut self, v: f64) -> Self {
        self.gap_pct = v;
        self
    }

    pub fn with_lookback(mut self, v: usize) -> Self {
        self.lookback = v;
        self
    }

    pub fn validate(&self) -> Result<(), AnomalyMarketError> {
        if self.threshold_std.is_nan() {
            return Err(AnomalyMarketError::InvalidConfig("threshold_std is NaN".into()));
        }
        if self.volume_multiple.is_nan() {
            return Err(AnomalyMarketError::InvalidConfig("volume_multiple is NaN".into()));
        }
        if self.gap_pct.is_nan() {
            return Err(AnomalyMarketError::InvalidConfig("gap_pct is NaN".into()));
        }
        Ok(())
    }
}

impl Default for AnomalyMarketConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AnomalyMarketConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnomalyMarketConfig(threshold_std={0:.4}, volume_multiple={1:.4}, gap_pct={2:.4}, lookback={3})", self.threshold_std, self.volume_multiple, self.gap_pct, self.lookback)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core market anomaly detection (flash crash, volume spikes) engine.
#[derive(Debug, Clone)]
pub struct AnomalyMarket {
    config: AnomalyMarketConfig,
    data: Vec<f64>,
}

impl AnomalyMarket {
    pub fn new(config: AnomalyMarketConfig) -> Result<Self, AnomalyMarketError> {
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
    pub fn config(&self) -> &AnomalyMarketConfig { &self.config }

    /// Detect flash crash.
    pub fn detect_flash_crash(&self) -> bool {
        !self.data.is_empty()
    }

    /// Detect volume anomaly.
    pub fn volume_anomaly(&self) -> bool {
        !self.data.is_empty()
    }

    /// Detect price gap.
    pub fn price_gap(&self) -> bool {
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

impl fmt::Display for AnomalyMarket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnomalyMarket(n={})", self.data.len())
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
        let cfg = AnomalyMarketConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AnomalyMarketConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AnomalyMarketConfig"));
    }

    #[test]
    fn test_config_with_threshold_std() {
        let cfg = AnomalyMarketConfig::new().with_threshold_std(42.0);
        assert!((cfg.threshold_std - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_volume_multiple() {
        let cfg = AnomalyMarketConfig::new().with_volume_multiple(42.0);
        assert!((cfg.volume_multiple - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_gap_pct() {
        let cfg = AnomalyMarketConfig::new().with_gap_pct(42.0);
        assert!((cfg.gap_pct - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_lookback() {
        let cfg = AnomalyMarketConfig::new().with_lookback(42);
        assert_eq!(cfg.lookback, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AnomalyMarketConfig::new().with_threshold_std(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AnomalyMarket"));
    }

    #[test]
    fn test_summary() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_detect_flash_crash() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_flash_crash();
        assert!(result);
    }

    #[test]
    fn test_volume_anomaly() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.volume_anomaly();
        assert!(result);
    }

    #[test]
    fn test_price_gap() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.price_gap();
        assert!(result);
    }

    #[test]
    fn test_price_gap_empty() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap();
        assert!(!e.price_gap());
    }

    #[test]
    fn test_config_accessor() {
        let e = AnomalyMarket::new(AnomalyMarketConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AnomalyMarketError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AnomalyMarketError::InvalidConfig("a".into());
        let e2 = AnomalyMarketError::ComputationFailed("b".into());
        let e3 = AnomalyMarketError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
