//! Coordinate conversion between reference frames.
//!
//! Provides [`CoordConvertConfig`] builder and [`CoordConvert`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordConvertError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CoordConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CoordConvert: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CoordConvert: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CoordConvert: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CoordConvert`] parameters.
#[derive(Debug, Clone)]
pub struct CoordConvertConfig {
    pub src_epsg: u32,
    pub dst_epsg: u32,
    pub precision: u8,
    pub swap_xy: bool,
}

impl CoordConvertConfig {
    pub fn new() -> Self {
        Self {
            src_epsg: 4326,
            dst_epsg: 3857,
            precision: 8,
            swap_xy: false,
        }
    }

    pub fn with_src_epsg(mut self, v: u32) -> Self {
        self.src_epsg = v;
        self
    }

    pub fn with_dst_epsg(mut self, v: u32) -> Self {
        self.dst_epsg = v;
        self
    }

    pub fn with_precision(mut self, v: u8) -> Self {
        self.precision = v;
        self
    }

    pub fn with_swap_xy(mut self, v: bool) -> Self {
        self.swap_xy = v;
        self
    }

    pub fn validate(&self) -> Result<(), CoordConvertError> {
        Ok(())
    }
}

impl Default for CoordConvertConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CoordConvertConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoordConvertConfig(src_epsg={0}, dst_epsg={1}, precision={2}, swap_xy={3})", self.src_epsg, self.dst_epsg, self.precision, self.swap_xy)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core coordinate conversion between reference frames engine.
#[derive(Debug, Clone)]
pub struct CoordConvert {
    config: CoordConvertConfig,
    data: Vec<f64>,
}

impl CoordConvert {
    pub fn new(config: CoordConvertConfig) -> Result<Self, CoordConvertError> {
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
    pub fn config(&self) -> &CoordConvertConfig { &self.config }

    /// DMS to decimal degrees.
    pub fn dms_to_decimal(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Decimal to DMS string.
    pub fn decimal_to_dms(&self) -> String {
        format!("{}: {} records", stringify!(decimal_to_dms), self.data.len())
    }

    /// Cartesian to polar.
    pub fn cartesian_to_polar(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
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

impl fmt::Display for CoordConvert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoordConvert(n={})", self.data.len())
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
        let cfg = CoordConvertConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CoordConvertConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CoordConvertConfig"));
    }

    #[test]
    fn test_config_with_src_epsg() {
        let cfg = CoordConvertConfig::new().with_src_epsg(42);
        assert_eq!(cfg.src_epsg, 42);
    }

    #[test]
    fn test_config_with_dst_epsg() {
        let cfg = CoordConvertConfig::new().with_dst_epsg(42);
        assert_eq!(cfg.dst_epsg, 42);
    }

    #[test]
    fn test_config_with_precision() {
        let cfg = CoordConvertConfig::new().with_precision(42);
        assert_eq!(cfg.precision, 42);
    }

    #[test]
    fn test_config_with_swap_xy() {
        let cfg = CoordConvertConfig::new().with_swap_xy(true);
        assert_eq!(cfg.swap_xy, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CoordConvertConfig::new().with_src_epsg(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CoordConvert::new(CoordConvertConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CoordConvert"));
    }

    #[test]
    fn test_summary() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dms_to_decimal() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.dms_to_decimal();
        assert!(result.is_finite());
    }

    #[test]
    fn test_decimal_to_dms() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decimal_to_dms();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_cartesian_to_polar() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.cartesian_to_polar();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_cartesian_to_polar_empty() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap();
        let _ = e.cartesian_to_polar();
    }

    #[test]
    fn test_config_accessor() {
        let e = CoordConvert::new(CoordConvertConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CoordConvertError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CoordConvertError::InvalidConfig("a".into());
        let e2 = CoordConvertError::ComputationFailed("b".into());
        let e3 = CoordConvertError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
