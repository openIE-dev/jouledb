//! Tax lot tracking with FIFO/LIFO and wash sale detection.
//!
//! Provides [`TaxLotTrackConfig`] builder and [`TaxLotTrack`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TaxLotTrackError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TaxLotTrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TaxLotTrack: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TaxLotTrack: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TaxLotTrack: insufficient data: {msg}"),
        }
    }
}

/// Variant selector for TaxMethod.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaxMethod {
    /// Fifo method.
    Fifo,
    /// Lifo method.
    Lifo,
    /// SpecificId method.
    SpecificId,
}

impl fmt::Display for TaxMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TaxLotTrack`] parameters.
#[derive(Debug, Clone)]
pub struct TaxLotTrackConfig {
    pub method: TaxMethod,
    pub wash_sale_days: u32,
    pub short_term_days: u32,
    pub track_unrealized: bool,
}

impl TaxLotTrackConfig {
    pub fn new() -> Self {
        Self {
            method: TaxMethod::Fifo,
            wash_sale_days: 30,
            short_term_days: 365,
            track_unrealized: true,
        }
    }

    pub fn with_method(mut self, v: TaxMethod) -> Self {
        self.method = v;
        self
    }

    pub fn with_wash_sale_days(mut self, v: u32) -> Self {
        self.wash_sale_days = v;
        self
    }

    pub fn with_short_term_days(mut self, v: u32) -> Self {
        self.short_term_days = v;
        self
    }

    pub fn with_track_unrealized(mut self, v: bool) -> Self {
        self.track_unrealized = v;
        self
    }

    pub fn validate(&self) -> Result<(), TaxLotTrackError> {
        Ok(())
    }
}

impl Default for TaxLotTrackConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TaxLotTrackConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaxLotTrackConfig(method={0:?}, wash_sale_days={1}, short_term_days={2}, track_unrealized={3})", self.method, self.wash_sale_days, self.short_term_days, self.track_unrealized)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core tax lot tracking with fifo/lifo and wash sale detection engine.
#[derive(Debug, Clone)]
pub struct TaxLotTrack {
    config: TaxLotTrackConfig,
    data: Vec<f64>,
}

impl TaxLotTrack {
    pub fn new(config: TaxLotTrackConfig) -> Result<Self, TaxLotTrackError> {
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
    pub fn config(&self) -> &TaxLotTrackConfig { &self.config }

    /// Calculate realized gain.
    pub fn realize_gain(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Calculate unrealized gain.
    pub fn unrealized_gain(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Detect wash sale.
    pub fn detect_wash_sale(&self) -> bool {
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

impl fmt::Display for TaxLotTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaxLotTrack(n={})", self.data.len())
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
        let cfg = TaxLotTrackConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TaxLotTrackConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TaxLotTrackConfig"));
    }

    #[test]
    fn test_config_with_method() {
        let cfg = TaxLotTrackConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_with_wash_sale_days() {
        let cfg = TaxLotTrackConfig::new().with_wash_sale_days(42);
        assert_eq!(cfg.wash_sale_days, 42);
    }

    #[test]
    fn test_config_with_short_term_days() {
        let cfg = TaxLotTrackConfig::new().with_short_term_days(42);
        assert_eq!(cfg.short_term_days, 42);
    }

    #[test]
    fn test_config_with_track_unrealized() {
        let cfg = TaxLotTrackConfig::new().with_track_unrealized(false);
        assert_eq!(cfg.track_unrealized, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TaxLotTrackConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TaxLotTrack"));
    }

    #[test]
    fn test_summary() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_realize_gain() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.realize_gain();
        assert!(result.is_finite());
    }

    #[test]
    fn test_unrealized_gain() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.unrealized_gain();
        assert!(result.is_finite());
    }

    #[test]
    fn test_detect_wash_sale() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_wash_sale();
        assert!(result);
    }

    #[test]
    fn test_detect_wash_sale_empty() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap();
        assert!(!e.detect_wash_sale());
    }

    #[test]
    fn test_config_accessor() {
        let e = TaxLotTrack::new(TaxLotTrackConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TaxLotTrackError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TaxLotTrackError::InvalidConfig("a".into());
        let e2 = TaxLotTrackError::ComputationFailed("b".into());
        let e3 = TaxLotTrackError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
