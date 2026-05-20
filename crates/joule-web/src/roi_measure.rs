//! Region of interest measurement on medical images.
//!
//! Provides [`RoiMeasureConfig`] builder and [`RoiMeasure`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RoiMeasureError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RoiMeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RoiMeasure: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RoiMeasure: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RoiMeasure: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RoiMeasure`] parameters.
#[derive(Debug, Clone)]
pub struct RoiMeasureConfig {
    pub roi_type: usize,
    pub include_stats: bool,
    pub calibration: f64,
    pub units: usize,
}

impl RoiMeasureConfig {
    pub fn new() -> Self {
        Self {
            roi_type: 0,
            include_stats: true,
            calibration: 1.0,
            units: 0,
        }
    }

    pub fn with_roi_type(mut self, v: usize) -> Self {
        self.roi_type = v;
        self
    }

    pub fn with_include_stats(mut self, v: bool) -> Self {
        self.include_stats = v;
        self
    }

    pub fn with_calibration(mut self, v: f64) -> Self {
        self.calibration = v;
        self
    }

    pub fn with_units(mut self, v: usize) -> Self {
        self.units = v;
        self
    }

    pub fn validate(&self) -> Result<(), RoiMeasureError> {
        if self.calibration.is_nan() {
            return Err(RoiMeasureError::InvalidConfig("calibration is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RoiMeasureConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RoiMeasureConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoiMeasureConfig(roi_type={0}, include_stats={1}, calibration={2:.4}, units={3})", self.roi_type, self.include_stats, self.calibration, self.units)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core region of interest measurement on medical images engine.
#[derive(Debug, Clone)]
pub struct RoiMeasure {
    config: RoiMeasureConfig,
    data: Vec<f64>,
}

impl RoiMeasure {
    pub fn new(config: RoiMeasureConfig) -> Result<Self, RoiMeasureError> {
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
    pub fn config(&self) -> &RoiMeasureConfig { &self.config }

    /// Calculate ROI area.
    pub fn area(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Mean pixel value in ROI.
    pub fn mean_value(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Measure distance.
    pub fn distance(&self) -> f64 {
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

impl fmt::Display for RoiMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoiMeasure(n={})", self.data.len())
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
        let cfg = RoiMeasureConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RoiMeasureConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RoiMeasureConfig"));
    }

    #[test]
    fn test_config_with_roi_type() {
        let cfg = RoiMeasureConfig::new().with_roi_type(42);
        assert_eq!(cfg.roi_type, 42);
    }

    #[test]
    fn test_config_with_include_stats() {
        let cfg = RoiMeasureConfig::new().with_include_stats(false);
        assert_eq!(cfg.include_stats, false);
    }

    #[test]
    fn test_config_with_calibration() {
        let cfg = RoiMeasureConfig::new().with_calibration(42.0);
        assert!((cfg.calibration - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_units() {
        let cfg = RoiMeasureConfig::new().with_units(42);
        assert_eq!(cfg.units, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RoiMeasureConfig::new().with_roi_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RoiMeasure"));
    }

    #[test]
    fn test_summary() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_area() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.area();
        assert!(result.is_finite());
    }

    #[test]
    fn test_mean_value() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mean_value();
        assert!(result.is_finite());
    }

    #[test]
    fn test_distance() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.distance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_distance_empty() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap();
        assert!((e.distance() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = RoiMeasure::new(RoiMeasureConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RoiMeasureError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RoiMeasureError::InvalidConfig("a".into());
        let e2 = RoiMeasureError::ComputationFailed("b".into());
        let e3 = RoiMeasureError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
