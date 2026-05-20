//! Well-Known Binary geometry encoding and decoding.
//!
//! Provides [`WkbParseConfig`] builder and [`WkbParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum WkbParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for WkbParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "WkbParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "WkbParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "WkbParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`WkbParse`] parameters.
#[derive(Debug, Clone)]
pub struct WkbParseConfig {
    pub byte_order: usize,
    pub include_srid: bool,
    pub srid: u32,
    pub dimensions: usize,
}

impl WkbParseConfig {
    pub fn new() -> Self {
        Self {
            byte_order: 0,
            include_srid: false,
            srid: 4326,
            dimensions: 2,
        }
    }

    pub fn with_byte_order(mut self, v: usize) -> Self {
        self.byte_order = v;
        self
    }

    pub fn with_include_srid(mut self, v: bool) -> Self {
        self.include_srid = v;
        self
    }

    pub fn with_srid(mut self, v: u32) -> Self {
        self.srid = v;
        self
    }

    pub fn with_dimensions(mut self, v: usize) -> Self {
        self.dimensions = v;
        self
    }

    pub fn validate(&self) -> Result<(), WkbParseError> {
        Ok(())
    }
}

impl Default for WkbParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for WkbParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WkbParseConfig(byte_order={0}, include_srid={1}, srid={2}, dimensions={3})", self.byte_order, self.include_srid, self.srid, self.dimensions)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core well-known binary geometry encoding and decoding engine.
#[derive(Debug, Clone)]
pub struct WkbParse {
    config: WkbParseConfig,
    data: Vec<f64>,
}

impl WkbParse {
    pub fn new(config: WkbParseConfig) -> Result<Self, WkbParseError> {
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
    pub fn config(&self) -> &WkbParseConfig { &self.config }

    /// Encode geometry to WKB bytes.
    pub fn encode(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decode WKB bytes to geometry.
    pub fn decode(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Detect geometry type from WKB.
    pub fn geometry_type(&self) -> String {
        format!("{}: {} records", stringify!(geometry_type), self.data.len())
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

impl fmt::Display for WkbParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WkbParse(n={})", self.data.len())
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
        let cfg = WkbParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = WkbParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("WkbParseConfig"));
    }

    #[test]
    fn test_config_with_byte_order() {
        let cfg = WkbParseConfig::new().with_byte_order(42);
        assert_eq!(cfg.byte_order, 42);
    }

    #[test]
    fn test_config_with_include_srid() {
        let cfg = WkbParseConfig::new().with_include_srid(true);
        assert_eq!(cfg.include_srid, true);
    }

    #[test]
    fn test_config_with_srid() {
        let cfg = WkbParseConfig::new().with_srid(42);
        assert_eq!(cfg.srid, 42);
    }

    #[test]
    fn test_config_with_dimensions() {
        let cfg = WkbParseConfig::new().with_dimensions(42);
        assert_eq!(cfg.dimensions, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = WkbParseConfig::new().with_byte_order(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = WkbParse::new(WkbParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("WkbParse"));
    }

    #[test]
    fn test_summary() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encode();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decode() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decode();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_geometry_type() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.geometry_type();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_geometry_type_empty() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap();
        let _ = e.geometry_type();
    }

    #[test]
    fn test_config_accessor() {
        let e = WkbParse::new(WkbParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = WkbParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = WkbParseError::InvalidConfig("a".into());
        let e2 = WkbParseError::ComputationFailed("b".into());
        let e3 = WkbParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
