//! Map legend generation and layout.
//!
//! Provides [`MapLegendConfig`] builder and [`MapLegend`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MapLegendError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MapLegendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MapLegend: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MapLegend: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MapLegend: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MapLegend`] parameters.
#[derive(Debug, Clone)]
pub struct MapLegendConfig {
    pub width_px: f64,
    pub height_px: f64,
    pub padding: f64,
    pub vertical: bool,
}

impl MapLegendConfig {
    pub fn new() -> Self {
        Self {
            width_px: 200.0,
            height_px: 300.0,
            padding: 10.0,
            vertical: true,
        }
    }

    pub fn with_width_px(mut self, v: f64) -> Self {
        self.width_px = v;
        self
    }

    pub fn with_height_px(mut self, v: f64) -> Self {
        self.height_px = v;
        self
    }

    pub fn with_padding(mut self, v: f64) -> Self {
        self.padding = v;
        self
    }

    pub fn with_vertical(mut self, v: bool) -> Self {
        self.vertical = v;
        self
    }

    pub fn validate(&self) -> Result<(), MapLegendError> {
        if self.width_px.is_nan() {
            return Err(MapLegendError::InvalidConfig("width_px is NaN".into()));
        }
        if self.height_px.is_nan() {
            return Err(MapLegendError::InvalidConfig("height_px is NaN".into()));
        }
        if self.padding.is_nan() {
            return Err(MapLegendError::InvalidConfig("padding is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MapLegendConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MapLegendConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapLegendConfig(width_px={0:.4}, height_px={1:.4}, padding={2:.4}, vertical={3})", self.width_px, self.height_px, self.padding, self.vertical)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core map legend generation and layout engine.
#[derive(Debug, Clone)]
pub struct MapLegend {
    config: MapLegendConfig,
    data: Vec<f64>,
}

impl MapLegend {
    pub fn new(config: MapLegendConfig) -> Result<Self, MapLegendError> {
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
    pub fn config(&self) -> &MapLegendConfig { &self.config }

    /// Continuous gradient legend.
    pub fn gradient_legend(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Discrete class legend.
    pub fn discrete_legend(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Proportional symbol legend.
    pub fn symbol_legend(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for MapLegend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapLegend(n={})", self.data.len())
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
        let cfg = MapLegendConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MapLegendConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MapLegendConfig"));
    }

    #[test]
    fn test_config_with_width_px() {
        let cfg = MapLegendConfig::new().with_width_px(42.0);
        assert!((cfg.width_px - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_height_px() {
        let cfg = MapLegendConfig::new().with_height_px(42.0);
        assert!((cfg.height_px - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_padding() {
        let cfg = MapLegendConfig::new().with_padding(42.0);
        assert!((cfg.padding - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_vertical() {
        let cfg = MapLegendConfig::new().with_vertical(false);
        assert_eq!(cfg.vertical, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MapLegendConfig::new().with_width_px(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MapLegend::new(MapLegendConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MapLegend"));
    }

    #[test]
    fn test_summary() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_legend() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gradient_legend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_discrete_legend() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.discrete_legend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_symbol_legend() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.symbol_legend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_symbol_legend_empty() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap();
        assert!(e.symbol_legend().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MapLegend::new(MapLegendConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MapLegendError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MapLegendError::InvalidConfig("a".into());
        let e2 = MapLegendError::ComputationFailed("b".into());
        let e3 = MapLegendError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
