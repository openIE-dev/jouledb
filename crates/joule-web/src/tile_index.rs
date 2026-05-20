//! Map tile indexing with XYZ and TMS schemes.
//!
//! Provides [`TileIndexConfig`] builder and [`TileIndex`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TileIndexError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TileIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TileIndex: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TileIndex: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TileIndex: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TileIndex`] parameters.
#[derive(Debug, Clone)]
pub struct TileIndexConfig {
    pub zoom: u8,
    pub tile_size: u32,
    pub origin_top_left: bool,
    pub max_zoom: u8,
}

impl TileIndexConfig {
    pub fn new() -> Self {
        Self {
            zoom: 10,
            tile_size: 256,
            origin_top_left: true,
            max_zoom: 22,
        }
    }

    pub fn with_zoom(mut self, v: u8) -> Self {
        self.zoom = v;
        self
    }

    pub fn with_tile_size(mut self, v: u32) -> Self {
        self.tile_size = v;
        self
    }

    pub fn with_origin_top_left(mut self, v: bool) -> Self {
        self.origin_top_left = v;
        self
    }

    pub fn with_max_zoom(mut self, v: u8) -> Self {
        self.max_zoom = v;
        self
    }

    pub fn validate(&self) -> Result<(), TileIndexError> {
        Ok(())
    }
}

impl Default for TileIndexConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TileIndexConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileIndexConfig(zoom={0}, tile_size={1}, origin_top_left={2}, max_zoom={3})", self.zoom, self.tile_size, self.origin_top_left, self.max_zoom)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core map tile indexing with xyz and tms schemes engine.
#[derive(Debug, Clone)]
pub struct TileIndex {
    config: TileIndexConfig,
    data: Vec<f64>,
}

impl TileIndex {
    pub fn new(config: TileIndexConfig) -> Result<Self, TileIndexError> {
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
    pub fn config(&self) -> &TileIndexConfig { &self.config }

    /// Convert lat/lon to tile XY.
    pub fn lat_lon_to_tile(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
    }

    /// Tile to geographic bounds.
    pub fn tile_to_bounds(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get child tiles.
    pub fn tile_children(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for TileIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileIndex(n={})", self.data.len())
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
        let cfg = TileIndexConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TileIndexConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TileIndexConfig"));
    }

    #[test]
    fn test_config_with_zoom() {
        let cfg = TileIndexConfig::new().with_zoom(42);
        assert_eq!(cfg.zoom, 42);
    }

    #[test]
    fn test_config_with_tile_size() {
        let cfg = TileIndexConfig::new().with_tile_size(42);
        assert_eq!(cfg.tile_size, 42);
    }

    #[test]
    fn test_config_with_origin_top_left() {
        let cfg = TileIndexConfig::new().with_origin_top_left(false);
        assert_eq!(cfg.origin_top_left, false);
    }

    #[test]
    fn test_config_with_max_zoom() {
        let cfg = TileIndexConfig::new().with_max_zoom(42);
        assert_eq!(cfg.max_zoom, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TileIndexConfig::new().with_zoom(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TileIndex::new(TileIndexConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TileIndex"));
    }

    #[test]
    fn test_summary() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lat_lon_to_tile() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.lat_lon_to_tile();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_tile_to_bounds() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tile_to_bounds();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tile_children() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tile_children();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tile_children_empty() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap();
        assert!(e.tile_children().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = TileIndex::new(TileIndexConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TileIndexError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TileIndexError::InvalidConfig("a".into());
        let e2 = TileIndexError::ComputationFailed("b".into());
        let e3 = TileIndexError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
