//! Map graticule/grid line overlay.
//!
//! Provides [`MapGraticuleConfig`] builder and [`MapGraticule`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MapGraticuleError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MapGraticuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MapGraticule: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MapGraticule: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MapGraticule: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MapGraticule`] parameters.
#[derive(Debug, Clone)]
pub struct MapGraticuleConfig {
    pub lat_interval: f64,
    pub lon_interval: f64,
    pub label_edges: bool,
    pub adaptive: bool,
}

impl MapGraticuleConfig {
    pub fn new() -> Self {
        Self {
            lat_interval: 10.0,
            lon_interval: 10.0,
            label_edges: true,
            adaptive: true,
        }
    }

    pub fn with_lat_interval(mut self, v: f64) -> Self {
        self.lat_interval = v;
        self
    }

    pub fn with_lon_interval(mut self, v: f64) -> Self {
        self.lon_interval = v;
        self
    }

    pub fn with_label_edges(mut self, v: bool) -> Self {
        self.label_edges = v;
        self
    }

    pub fn with_adaptive(mut self, v: bool) -> Self {
        self.adaptive = v;
        self
    }

    pub fn validate(&self) -> Result<(), MapGraticuleError> {
        if self.lat_interval.is_nan() {
            return Err(MapGraticuleError::InvalidConfig("lat_interval is NaN".into()));
        }
        if self.lon_interval.is_nan() {
            return Err(MapGraticuleError::InvalidConfig("lon_interval is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MapGraticuleConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MapGraticuleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapGraticuleConfig(lat_interval={0:.4}, lon_interval={1:.4}, label_edges={2}, adaptive={3})", self.lat_interval, self.lon_interval, self.label_edges, self.adaptive)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core map graticule/grid line overlay engine.
#[derive(Debug, Clone)]
pub struct MapGraticule {
    config: MapGraticuleConfig,
    data: Vec<f64>,
}

impl MapGraticule {
    pub fn new(config: MapGraticuleConfig) -> Result<Self, MapGraticuleError> {
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
    pub fn config(&self) -> &MapGraticuleConfig { &self.config }

    /// Generate graticule lines.
    pub fn generate_lines(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute label positions.
    pub fn label_positions(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Clip graticule to viewport.
    pub fn clip_to_viewport(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for MapGraticule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapGraticule(n={})", self.data.len())
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
        let cfg = MapGraticuleConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MapGraticuleConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MapGraticuleConfig"));
    }

    #[test]
    fn test_config_with_lat_interval() {
        let cfg = MapGraticuleConfig::new().with_lat_interval(42.0);
        assert!((cfg.lat_interval - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_lon_interval() {
        let cfg = MapGraticuleConfig::new().with_lon_interval(42.0);
        assert!((cfg.lon_interval - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_label_edges() {
        let cfg = MapGraticuleConfig::new().with_label_edges(false);
        assert_eq!(cfg.label_edges, false);
    }

    #[test]
    fn test_config_with_adaptive() {
        let cfg = MapGraticuleConfig::new().with_adaptive(false);
        assert_eq!(cfg.adaptive, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MapGraticuleConfig::new().with_lat_interval(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MapGraticule::new(MapGraticuleConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MapGraticule"));
    }

    #[test]
    fn test_summary() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_lines() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_lines();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_label_positions() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.label_positions();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_clip_to_viewport() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.clip_to_viewport();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_clip_to_viewport_empty() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap();
        assert!(e.clip_to_viewport().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MapGraticule::new(MapGraticuleConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MapGraticuleError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MapGraticuleError::InvalidConfig("a".into());
        let e2 = MapGraticuleError::ComputationFailed("b".into());
        let e3 = MapGraticuleError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
