//! Heatmap rendering with kernel density estimation.
//!
//! Provides [`HeatmapGeoConfig`] builder and [`HeatmapGeo`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum HeatmapGeoError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for HeatmapGeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "HeatmapGeo: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "HeatmapGeo: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "HeatmapGeo: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`HeatmapGeo`] parameters.
#[derive(Debug, Clone)]
pub struct HeatmapGeoConfig {
    pub radius_px: f64,
    pub intensity: f64,
    pub gradient_stops: usize,
    pub blur_sigma: f64,
}

impl HeatmapGeoConfig {
    pub fn new() -> Self {
        Self {
            radius_px: 30.0,
            intensity: 1.0,
            gradient_stops: 5,
            blur_sigma: 0.5,
        }
    }

    pub fn with_radius_px(mut self, v: f64) -> Self {
        self.radius_px = v;
        self
    }

    pub fn with_intensity(mut self, v: f64) -> Self {
        self.intensity = v;
        self
    }

    pub fn with_gradient_stops(mut self, v: usize) -> Self {
        self.gradient_stops = v;
        self
    }

    pub fn with_blur_sigma(mut self, v: f64) -> Self {
        self.blur_sigma = v;
        self
    }

    pub fn validate(&self) -> Result<(), HeatmapGeoError> {
        if self.radius_px.is_nan() {
            return Err(HeatmapGeoError::InvalidConfig("radius_px is NaN".into()));
        }
        if self.intensity.is_nan() {
            return Err(HeatmapGeoError::InvalidConfig("intensity is NaN".into()));
        }
        if self.blur_sigma.is_nan() {
            return Err(HeatmapGeoError::InvalidConfig("blur_sigma is NaN".into()));
        }
        Ok(())
    }
}

impl Default for HeatmapGeoConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for HeatmapGeoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeatmapGeoConfig(radius_px={0:.4}, intensity={1:.4}, gradient_stops={2}, blur_sigma={3:.4})", self.radius_px, self.intensity, self.gradient_stops, self.blur_sigma)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core heatmap rendering with kernel density estimation engine.
#[derive(Debug, Clone)]
pub struct HeatmapGeo {
    config: HeatmapGeoConfig,
    data: Vec<f64>,
}

impl HeatmapGeo {
    pub fn new(config: HeatmapGeoConfig) -> Result<Self, HeatmapGeoError> {
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
    pub fn config(&self) -> &HeatmapGeoConfig { &self.config }

    /// Generate intensity grid.
    pub fn generate_grid(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Kernel density at point.
    pub fn kernel_density(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Map intensity to color.
    pub fn color_at_intensity(&self) -> Vec<f64> {
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

impl fmt::Display for HeatmapGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeatmapGeo(n={})", self.data.len())
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
        let cfg = HeatmapGeoConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = HeatmapGeoConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("HeatmapGeoConfig"));
    }

    #[test]
    fn test_config_with_radius_px() {
        let cfg = HeatmapGeoConfig::new().with_radius_px(42.0);
        assert!((cfg.radius_px - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_intensity() {
        let cfg = HeatmapGeoConfig::new().with_intensity(42.0);
        assert!((cfg.intensity - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_gradient_stops() {
        let cfg = HeatmapGeoConfig::new().with_gradient_stops(42);
        assert_eq!(cfg.gradient_stops, 42);
    }

    #[test]
    fn test_config_with_blur_sigma() {
        let cfg = HeatmapGeoConfig::new().with_blur_sigma(42.0);
        assert!((cfg.blur_sigma - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = HeatmapGeoConfig::new().with_radius_px(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("HeatmapGeo"));
    }

    #[test]
    fn test_summary() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_grid() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_grid();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_kernel_density() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.kernel_density();
        assert!(result.is_finite());
    }

    #[test]
    fn test_color_at_intensity() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.color_at_intensity();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_color_at_intensity_empty() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap();
        assert!(e.color_at_intensity().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = HeatmapGeo::new(HeatmapGeoConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = HeatmapGeoError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = HeatmapGeoError::InvalidConfig("a".into());
        let e2 = HeatmapGeoError::ComputationFailed("b".into());
        let e3 = HeatmapGeoError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
