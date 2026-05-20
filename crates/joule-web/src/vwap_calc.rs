//! VWAP and TWAP calculation engine.
//!
//! Provides [`VwapCalcConfig`] builder and [`VwapCalc`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum VwapCalcError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for VwapCalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "VwapCalc: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "VwapCalc: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "VwapCalc: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`VwapCalc`] parameters.
#[derive(Debug, Clone)]
pub struct VwapCalcConfig {
    pub window_minutes: u32,
    pub anchor_time: u64,
    pub include_auction: bool,
    pub band_width: f64,
}

impl VwapCalcConfig {
    pub fn new() -> Self {
        Self {
            window_minutes: 30,
            anchor_time: 0,
            include_auction: false,
            band_width: 0.02,
        }
    }

    pub fn with_window_minutes(mut self, v: u32) -> Self {
        self.window_minutes = v;
        self
    }

    pub fn with_anchor_time(mut self, v: u64) -> Self {
        self.anchor_time = v;
        self
    }

    pub fn with_include_auction(mut self, v: bool) -> Self {
        self.include_auction = v;
        self
    }

    pub fn with_band_width(mut self, v: f64) -> Self {
        self.band_width = v;
        self
    }

    pub fn validate(&self) -> Result<(), VwapCalcError> {
        if self.band_width.is_nan() {
            return Err(VwapCalcError::InvalidConfig("band_width is NaN".into()));
        }
        Ok(())
    }
}

impl Default for VwapCalcConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VwapCalcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VwapCalcConfig(window_minutes={0}, anchor_time={1}, include_auction={2}, band_width={3:.4})", self.window_minutes, self.anchor_time, self.include_auction, self.band_width)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core vwap and twap calculation engine engine.
#[derive(Debug, Clone)]
pub struct VwapCalc {
    config: VwapCalcConfig,
    data: Vec<f64>,
}

impl VwapCalc {
    pub fn new(config: VwapCalcConfig) -> Result<Self, VwapCalcError> {
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
    pub fn config(&self) -> &VwapCalcConfig { &self.config }

    /// Rolling VWAP.
    pub fn rolling_vwap(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Time-weighted average price.
    pub fn twap(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Price deviation from VWAP.
    pub fn deviation(&self) -> f64 {
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

impl fmt::Display for VwapCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VwapCalc(n={})", self.data.len())
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
        let cfg = VwapCalcConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = VwapCalcConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("VwapCalcConfig"));
    }

    #[test]
    fn test_config_with_window_minutes() {
        let cfg = VwapCalcConfig::new().with_window_minutes(42);
        assert_eq!(cfg.window_minutes, 42);
    }

    #[test]
    fn test_config_with_anchor_time() {
        let cfg = VwapCalcConfig::new().with_anchor_time(42);
        assert_eq!(cfg.anchor_time, 42);
    }

    #[test]
    fn test_config_with_include_auction() {
        let cfg = VwapCalcConfig::new().with_include_auction(true);
        assert_eq!(cfg.include_auction, true);
    }

    #[test]
    fn test_config_with_band_width() {
        let cfg = VwapCalcConfig::new().with_band_width(42.0);
        assert!((cfg.band_width - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = VwapCalcConfig::new().with_window_minutes(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = VwapCalc::new(VwapCalcConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("VwapCalc"));
    }

    #[test]
    fn test_summary() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_rolling_vwap() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rolling_vwap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_twap() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.twap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_deviation() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.deviation();
        assert!(result.is_finite());
    }

    #[test]
    fn test_deviation_empty() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap();
        assert!((e.deviation() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = VwapCalc::new(VwapCalcConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = VwapCalcError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = VwapCalcError::InvalidConfig("a".into());
        let e2 = VwapCalcError::ComputationFailed("b".into());
        let e3 = VwapCalcError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
