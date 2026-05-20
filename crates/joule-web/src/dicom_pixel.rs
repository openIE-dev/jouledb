//! DICOM pixel data processing.
//!
//! Provides [`DicomPixelConfig`] builder and [`DicomPixel`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DicomPixelError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DicomPixelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DicomPixel: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DicomPixel: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DicomPixel: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DicomPixel`] parameters.
#[derive(Debug, Clone)]
pub struct DicomPixelConfig {
    pub window_center: f64,
    pub window_width: f64,
    pub rescale_slope: f64,
    pub rescale_intercept: f64,
}

impl DicomPixelConfig {
    pub fn new() -> Self {
        Self {
            window_center: 40.0,
            window_width: 80.0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    pub fn with_window_center(mut self, v: f64) -> Self {
        self.window_center = v;
        self
    }

    pub fn with_window_width(mut self, v: f64) -> Self {
        self.window_width = v;
        self
    }

    pub fn with_rescale_slope(mut self, v: f64) -> Self {
        self.rescale_slope = v;
        self
    }

    pub fn with_rescale_intercept(mut self, v: f64) -> Self {
        self.rescale_intercept = v;
        self
    }

    pub fn validate(&self) -> Result<(), DicomPixelError> {
        if self.window_center.is_nan() {
            return Err(DicomPixelError::InvalidConfig("window_center is NaN".into()));
        }
        if self.window_width.is_nan() {
            return Err(DicomPixelError::InvalidConfig("window_width is NaN".into()));
        }
        if self.rescale_slope.is_nan() {
            return Err(DicomPixelError::InvalidConfig("rescale_slope is NaN".into()));
        }
        if self.rescale_intercept.is_nan() {
            return Err(DicomPixelError::InvalidConfig("rescale_intercept is NaN".into()));
        }
        Ok(())
    }
}

impl Default for DicomPixelConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DicomPixelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DicomPixelConfig(window_center={0:.4}, window_width={1:.4}, rescale_slope={2:.4}, rescale_intercept={3:.4})", self.window_center, self.window_width, self.rescale_slope, self.rescale_intercept)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core dicom pixel data processing engine.
#[derive(Debug, Clone)]
pub struct DicomPixel {
    config: DicomPixelConfig,
    data: Vec<f64>,
}

impl DicomPixel {
    pub fn new(config: DicomPixelConfig) -> Result<Self, DicomPixelError> {
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
    pub fn config(&self) -> &DicomPixelConfig { &self.config }

    /// Apply window/level transform.
    pub fn apply_window_level(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Rescale pixel values.
    pub fn rescale_pixels(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Convert to Hounsfield units.
    pub fn hounsfield_value(&self) -> f64 {
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

impl fmt::Display for DicomPixel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DicomPixel(n={})", self.data.len())
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
        let cfg = DicomPixelConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DicomPixelConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DicomPixelConfig"));
    }

    #[test]
    fn test_config_with_window_center() {
        let cfg = DicomPixelConfig::new().with_window_center(42.0);
        assert!((cfg.window_center - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_window_width() {
        let cfg = DicomPixelConfig::new().with_window_width(42.0);
        assert!((cfg.window_width - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rescale_slope() {
        let cfg = DicomPixelConfig::new().with_rescale_slope(42.0);
        assert!((cfg.rescale_slope - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rescale_intercept() {
        let cfg = DicomPixelConfig::new().with_rescale_intercept(42.0);
        assert!((cfg.rescale_intercept - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DicomPixelConfig::new().with_window_center(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DicomPixel::new(DicomPixelConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DicomPixel"));
    }

    #[test]
    fn test_summary() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_window_level() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.apply_window_level();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rescale_pixels() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rescale_pixels();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_hounsfield_value() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.hounsfield_value();
        assert!(result.is_finite());
    }

    #[test]
    fn test_hounsfield_value_empty() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap();
        assert!((e.hounsfield_value() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = DicomPixel::new(DicomPixelConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DicomPixelError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DicomPixelError::InvalidConfig("a".into());
        let e2 = DicomPixelError::ComputationFailed("b".into());
        let e3 = DicomPixelError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
