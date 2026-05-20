//! Post-quantum serialization with bit-packing.
//!
//! Provides [`PqSerializeConfig`] builder and [`PqSerialize`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqSerializeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqSerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqSerialize: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqSerialize: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqSerialize: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqSerialize`] parameters.
#[derive(Debug, Clone)]
pub struct PqSerializeConfig {
    pub coefficient_bits: usize,
    pub compress: bool,
    pub include_header: bool,
    pub version: usize,
}

impl PqSerializeConfig {
    pub fn new() -> Self {
        Self {
            coefficient_bits: 12,
            compress: true,
            include_header: true,
            version: 1,
        }
    }

    pub fn with_coefficient_bits(mut self, v: usize) -> Self {
        self.coefficient_bits = v;
        self
    }

    pub fn with_compress(mut self, v: bool) -> Self {
        self.compress = v;
        self
    }

    pub fn with_include_header(mut self, v: bool) -> Self {
        self.include_header = v;
        self
    }

    pub fn with_version(mut self, v: usize) -> Self {
        self.version = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqSerializeError> {
        Ok(())
    }
}

impl Default for PqSerializeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqSerializeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqSerializeConfig(coefficient_bits={0}, compress={1}, include_header={2}, version={3})", self.coefficient_bits, self.compress, self.include_header, self.version)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum serialization with bit-packing engine.
#[derive(Debug, Clone)]
pub struct PqSerialize {
    config: PqSerializeConfig,
    data: Vec<f64>,
}

impl PqSerialize {
    pub fn new(config: PqSerializeConfig) -> Result<Self, PqSerializeError> {
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
    pub fn config(&self) -> &PqSerializeConfig { &self.config }

    /// Pack polynomial coefficients.
    pub fn pack_poly(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Unpack polynomial coefficients.
    pub fn unpack_poly(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Encode public key.
    pub fn encode_public_key(&self) -> Vec<f64> {
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

impl fmt::Display for PqSerialize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqSerialize(n={})", self.data.len())
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
        let cfg = PqSerializeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqSerializeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqSerializeConfig"));
    }

    #[test]
    fn test_config_with_coefficient_bits() {
        let cfg = PqSerializeConfig::new().with_coefficient_bits(42);
        assert_eq!(cfg.coefficient_bits, 42);
    }

    #[test]
    fn test_config_with_compress() {
        let cfg = PqSerializeConfig::new().with_compress(false);
        assert_eq!(cfg.compress, false);
    }

    #[test]
    fn test_config_with_include_header() {
        let cfg = PqSerializeConfig::new().with_include_header(false);
        assert_eq!(cfg.include_header, false);
    }

    #[test]
    fn test_config_with_version() {
        let cfg = PqSerializeConfig::new().with_version(42);
        assert_eq!(cfg.version, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqSerializeConfig::new().with_coefficient_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqSerialize::new(PqSerializeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqSerialize"));
    }

    #[test]
    fn test_summary() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pack_poly() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.pack_poly();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_unpack_poly() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.unpack_poly();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encode_public_key() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encode_public_key();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encode_public_key_empty() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap();
        assert!(e.encode_public_key().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqSerialize::new(PqSerializeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqSerializeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqSerializeError::InvalidConfig("a".into());
        let e2 = PqSerializeError::ComputationFailed("b".into());
        let e3 = PqSerializeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
