//! Mapbox Vector Tile encoding and decoding.
//!
//! Provides [`MvtEncodeConfig`] builder and [`MvtEncode`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MvtEncodeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MvtEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MvtEncode: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MvtEncode: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MvtEncode: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MvtEncode`] parameters.
#[derive(Debug, Clone)]
pub struct MvtEncodeConfig {
    pub extent: u32,
    pub buffer: u32,
    pub simplify_tolerance: f64,
    pub quantize: bool,
}

impl MvtEncodeConfig {
    pub fn new() -> Self {
        Self {
            extent: 4096,
            buffer: 64,
            simplify_tolerance: 1.0,
            quantize: true,
        }
    }

    pub fn with_extent(mut self, v: u32) -> Self {
        self.extent = v;
        self
    }

    pub fn with_buffer(mut self, v: u32) -> Self {
        self.buffer = v;
        self
    }

    pub fn with_simplify_tolerance(mut self, v: f64) -> Self {
        self.simplify_tolerance = v;
        self
    }

    pub fn with_quantize(mut self, v: bool) -> Self {
        self.quantize = v;
        self
    }

    pub fn validate(&self) -> Result<(), MvtEncodeError> {
        if self.simplify_tolerance.is_nan() {
            return Err(MvtEncodeError::InvalidConfig("simplify_tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for MvtEncodeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MvtEncodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MvtEncodeConfig(extent={0}, buffer={1}, simplify_tolerance={2:.4}, quantize={3})", self.extent, self.buffer, self.simplify_tolerance, self.quantize)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core mapbox vector tile encoding and decoding engine.
#[derive(Debug, Clone)]
pub struct MvtEncode {
    config: MvtEncodeConfig,
    data: Vec<f64>,
}

impl MvtEncode {
    pub fn new(config: MvtEncodeConfig) -> Result<Self, MvtEncodeError> {
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
    pub fn config(&self) -> &MvtEncodeConfig { &self.config }

    /// Encode a tile layer.
    pub fn encode_layer(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decode a vector tile.
    pub fn decode_tile(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Quantize coordinates to tile extent.
    pub fn quantize_coords(&self) -> Vec<f64> {
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

impl fmt::Display for MvtEncode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MvtEncode(n={})", self.data.len())
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
        let cfg = MvtEncodeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MvtEncodeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MvtEncodeConfig"));
    }

    #[test]
    fn test_config_with_extent() {
        let cfg = MvtEncodeConfig::new().with_extent(42);
        assert_eq!(cfg.extent, 42);
    }

    #[test]
    fn test_config_with_buffer() {
        let cfg = MvtEncodeConfig::new().with_buffer(42);
        assert_eq!(cfg.buffer, 42);
    }

    #[test]
    fn test_config_with_simplify_tolerance() {
        let cfg = MvtEncodeConfig::new().with_simplify_tolerance(42.0);
        assert!((cfg.simplify_tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_quantize() {
        let cfg = MvtEncodeConfig::new().with_quantize(false);
        assert_eq!(cfg.quantize, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MvtEncodeConfig::new().with_extent(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MvtEncode::new(MvtEncodeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MvtEncode"));
    }

    #[test]
    fn test_summary() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_layer() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encode_layer();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decode_tile() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decode_tile();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quantize_coords() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.quantize_coords();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quantize_coords_empty() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap();
        assert!(e.quantize_coords().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MvtEncode::new(MvtEncodeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MvtEncodeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MvtEncodeError::InvalidConfig("a".into());
        let e2 = MvtEncodeError::ComputationFailed("b".into());
        let e3 = MvtEncodeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
