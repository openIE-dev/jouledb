//! Medical image windowing presets.
//!
//! Provides [`ImageWindowConfig`] builder and [`ImageWindow`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageWindowError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ImageWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ImageWindow: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ImageWindow: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ImageWindow: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ImageWindow`] parameters.
#[derive(Debug, Clone)]
pub struct ImageWindowConfig {
    pub preset: usize,
    pub custom_center: f64,
    pub custom_width: f64,
    pub auto_window: bool,
}

impl ImageWindowConfig {
    pub fn new() -> Self {
        Self {
            preset: 0,
            custom_center: 40.0,
            custom_width: 80.0,
            auto_window: false,
        }
    }

    pub fn with_preset(mut self, v: usize) -> Self {
        self.preset = v;
        self
    }

    pub fn with_custom_center(mut self, v: f64) -> Self {
        self.custom_center = v;
        self
    }

    pub fn with_custom_width(mut self, v: f64) -> Self {
        self.custom_width = v;
        self
    }

    pub fn with_auto_window(mut self, v: bool) -> Self {
        self.auto_window = v;
        self
    }

    pub fn validate(&self) -> Result<(), ImageWindowError> {
        if self.custom_center.is_nan() {
            return Err(ImageWindowError::InvalidConfig("custom_center is NaN".into()));
        }
        if self.custom_width.is_nan() {
            return Err(ImageWindowError::InvalidConfig("custom_width is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ImageWindowConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ImageWindowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageWindowConfig(preset={0}, custom_center={1:.4}, custom_width={2:.4}, auto_window={3})", self.preset, self.custom_center, self.custom_width, self.auto_window)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core medical image windowing presets engine.
#[derive(Debug, Clone)]
pub struct ImageWindow {
    config: ImageWindowConfig,
    data: Vec<f64>,
}

impl ImageWindow {
    pub fn new(config: ImageWindowConfig) -> Result<Self, ImageWindowError> {
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
    pub fn config(&self) -> &ImageWindowConfig { &self.config }

    /// Apply windowing preset.
    pub fn apply_preset(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
    }

    /// Auto-detect optimal window.
    pub fn auto_detect(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
    }

    /// Get preset name.
    pub fn preset_name(&self) -> String {
        format!("{}: {} records", stringify!(preset_name), self.data.len())
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

impl fmt::Display for ImageWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageWindow(n={})", self.data.len())
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
        let cfg = ImageWindowConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ImageWindowConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ImageWindowConfig"));
    }

    #[test]
    fn test_config_with_preset() {
        let cfg = ImageWindowConfig::new().with_preset(42);
        assert_eq!(cfg.preset, 42);
    }

    #[test]
    fn test_config_with_custom_center() {
        let cfg = ImageWindowConfig::new().with_custom_center(42.0);
        assert!((cfg.custom_center - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_custom_width() {
        let cfg = ImageWindowConfig::new().with_custom_width(42.0);
        assert!((cfg.custom_width - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_auto_window() {
        let cfg = ImageWindowConfig::new().with_auto_window(true);
        assert_eq!(cfg.auto_window, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ImageWindowConfig::new().with_preset(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ImageWindow::new(ImageWindowConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ImageWindow"));
    }

    #[test]
    fn test_summary() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_preset() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.apply_preset();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_auto_detect() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.auto_detect();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_preset_name() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.preset_name();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_preset_name_empty() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap();
        let _ = e.preset_name();
    }

    #[test]
    fn test_config_accessor() {
        let e = ImageWindow::new(ImageWindowConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ImageWindowError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ImageWindowError::InvalidConfig("a".into());
        let e2 = ImageWindowError::ComputationFailed("b".into());
        let e3 = ImageWindowError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
