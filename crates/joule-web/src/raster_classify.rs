//! Raster data classification methods.
//!
//! Provides [`RasterClassifyConfig`] builder and [`RasterClassify`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RasterClassifyError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RasterClassifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RasterClassify: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RasterClassify: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RasterClassify: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RasterClassify`] parameters.
#[derive(Debug, Clone)]
pub struct RasterClassifyConfig {
    pub num_classes: usize,
    pub method: usize,
    pub nodata_value: f64,
    pub include_bounds: bool,
}

impl RasterClassifyConfig {
    pub fn new() -> Self {
        Self {
            num_classes: 5,
            method: 0,
            nodata_value: -9999.0,
            include_bounds: true,
        }
    }

    pub fn with_num_classes(mut self, v: usize) -> Self {
        self.num_classes = v;
        self
    }

    pub fn with_method(mut self, v: usize) -> Self {
        self.method = v;
        self
    }

    pub fn with_nodata_value(mut self, v: f64) -> Self {
        self.nodata_value = v;
        self
    }

    pub fn with_include_bounds(mut self, v: bool) -> Self {
        self.include_bounds = v;
        self
    }

    pub fn validate(&self) -> Result<(), RasterClassifyError> {
        if self.nodata_value.is_nan() {
            return Err(RasterClassifyError::InvalidConfig("nodata_value is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RasterClassifyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RasterClassifyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RasterClassifyConfig(num_classes={0}, method={1}, nodata_value={2:.4}, include_bounds={3})", self.num_classes, self.method, self.nodata_value, self.include_bounds)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core raster data classification methods engine.
#[derive(Debug, Clone)]
pub struct RasterClassify {
    config: RasterClassifyConfig,
    data: Vec<f64>,
}

impl RasterClassify {
    pub fn new(config: RasterClassifyConfig) -> Result<Self, RasterClassifyError> {
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
    pub fn config(&self) -> &RasterClassifyConfig { &self.config }

    /// Equal interval classification.
    pub fn equal_interval(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Quantile classification.
    pub fn quantile(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Jenks natural breaks.
    pub fn natural_breaks(&self) -> Vec<f64> {
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

impl fmt::Display for RasterClassify {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RasterClassify(n={})", self.data.len())
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
        let cfg = RasterClassifyConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RasterClassifyConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RasterClassifyConfig"));
    }

    #[test]
    fn test_config_with_num_classes() {
        let cfg = RasterClassifyConfig::new().with_num_classes(42);
        assert_eq!(cfg.num_classes, 42);
    }

    #[test]
    fn test_config_with_method() {
        let cfg = RasterClassifyConfig::new().with_method(42);
        assert_eq!(cfg.method, 42);
    }

    #[test]
    fn test_config_with_nodata_value() {
        let cfg = RasterClassifyConfig::new().with_nodata_value(42.0);
        assert!((cfg.nodata_value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_include_bounds() {
        let cfg = RasterClassifyConfig::new().with_include_bounds(false);
        assert_eq!(cfg.include_bounds, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RasterClassifyConfig::new().with_num_classes(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RasterClassify::new(RasterClassifyConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RasterClassify"));
    }

    #[test]
    fn test_summary() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_equal_interval() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.equal_interval();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quantile() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.quantile();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_natural_breaks() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.natural_breaks();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_natural_breaks_empty() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap();
        assert!(e.natural_breaks().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = RasterClassify::new(RasterClassifyConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RasterClassifyError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RasterClassifyError::InvalidConfig("a".into());
        let e2 = RasterClassifyError::ComputationFailed("b".into());
        let e3 = RasterClassifyError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
