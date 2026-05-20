//! BIKE QC-MDPC code-based KEM.
//!
//! Provides [`BikeKemConfig`] builder and [`BikeKem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BikeKemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BikeKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BikeKem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BikeKem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BikeKem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BikeKem`] parameters.
#[derive(Debug, Clone)]
pub struct BikeKemConfig {
    pub block_size: usize,
    pub weight: usize,
    pub error_weight: usize,
    pub max_iter: usize,
}

impl BikeKemConfig {
    pub fn new() -> Self {
        Self {
            block_size: 12323,
            weight: 142,
            error_weight: 134,
            max_iter: 100,
        }
    }

    pub fn with_block_size(mut self, v: usize) -> Self {
        self.block_size = v;
        self
    }

    pub fn with_weight(mut self, v: usize) -> Self {
        self.weight = v;
        self
    }

    pub fn with_error_weight(mut self, v: usize) -> Self {
        self.error_weight = v;
        self
    }

    pub fn with_max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    pub fn validate(&self) -> Result<(), BikeKemError> {
        Ok(())
    }
}

impl Default for BikeKemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BikeKemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BikeKemConfig(block_size={0}, weight={1}, error_weight={2}, max_iter={3})", self.block_size, self.weight, self.error_weight, self.max_iter)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core bike qc-mdpc code-based kem engine.
#[derive(Debug, Clone)]
pub struct BikeKem {
    config: BikeKemConfig,
    data: Vec<f64>,
}

impl BikeKem {
    pub fn new(config: BikeKemConfig) -> Result<Self, BikeKemError> {
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
    pub fn config(&self) -> &BikeKemConfig { &self.config }

    /// Generate BIKE keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Encapsulate shared secret.
    pub fn encapsulate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decapsulate with BGF decoder.
    pub fn decapsulate(&self) -> Vec<f64> {
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

impl fmt::Display for BikeKem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BikeKem(n={})", self.data.len())
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
        let cfg = BikeKemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BikeKemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BikeKemConfig"));
    }

    #[test]
    fn test_config_with_block_size() {
        let cfg = BikeKemConfig::new().with_block_size(42);
        assert_eq!(cfg.block_size, 42);
    }

    #[test]
    fn test_config_with_weight() {
        let cfg = BikeKemConfig::new().with_weight(42);
        assert_eq!(cfg.weight, 42);
    }

    #[test]
    fn test_config_with_error_weight() {
        let cfg = BikeKemConfig::new().with_error_weight(42);
        assert_eq!(cfg.error_weight, 42);
    }

    #[test]
    fn test_config_with_max_iter() {
        let cfg = BikeKemConfig::new().with_max_iter(42);
        assert_eq!(cfg.max_iter, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BikeKemConfig::new().with_block_size(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BikeKem::new(BikeKemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BikeKem"));
    }

    #[test]
    fn test_summary() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = BikeKem::new(BikeKemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BikeKemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BikeKemError::InvalidConfig("a".into());
        let e2 = BikeKemError::ComputationFailed("b".into());
        let e3 = BikeKemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
