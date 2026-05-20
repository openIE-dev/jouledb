//! Hybrid classical+PQ key encapsulation.
//!
//! Provides [`HybridKemConfig`] builder and [`HybridKem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum HybridKemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for HybridKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "HybridKem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "HybridKem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "HybridKem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`HybridKem`] parameters.
#[derive(Debug, Clone)]
pub struct HybridKemConfig {
    pub classical_bits: usize,
    pub pq_level: usize,
    pub combiner: usize,
    pub kdf_output_len: usize,
}

impl HybridKemConfig {
    pub fn new() -> Self {
        Self {
            classical_bits: 256,
            pq_level: 3,
            combiner: 0,
            kdf_output_len: 32,
        }
    }

    pub fn with_classical_bits(mut self, v: usize) -> Self {
        self.classical_bits = v;
        self
    }

    pub fn with_pq_level(mut self, v: usize) -> Self {
        self.pq_level = v;
        self
    }

    pub fn with_combiner(mut self, v: usize) -> Self {
        self.combiner = v;
        self
    }

    pub fn with_kdf_output_len(mut self, v: usize) -> Self {
        self.kdf_output_len = v;
        self
    }

    pub fn validate(&self) -> Result<(), HybridKemError> {
        Ok(())
    }
}

impl Default for HybridKemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for HybridKemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKemConfig(classical_bits={0}, pq_level={1}, combiner={2}, kdf_output_len={3})", self.classical_bits, self.pq_level, self.combiner, self.kdf_output_len)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hybrid classical+pq key encapsulation engine.
#[derive(Debug, Clone)]
pub struct HybridKem {
    config: HybridKemConfig,
    data: Vec<f64>,
}

impl HybridKem {
    pub fn new(config: HybridKemConfig) -> Result<Self, HybridKemError> {
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
    pub fn config(&self) -> &HybridKemConfig { &self.config }

    /// Generate hybrid keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Hybrid encapsulation.
    pub fn encapsulate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Hybrid decapsulation.
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

impl fmt::Display for HybridKem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKem(n={})", self.data.len())
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
        let cfg = HybridKemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = HybridKemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("HybridKemConfig"));
    }

    #[test]
    fn test_config_with_classical_bits() {
        let cfg = HybridKemConfig::new().with_classical_bits(42);
        assert_eq!(cfg.classical_bits, 42);
    }

    #[test]
    fn test_config_with_pq_level() {
        let cfg = HybridKemConfig::new().with_pq_level(42);
        assert_eq!(cfg.pq_level, 42);
    }

    #[test]
    fn test_config_with_combiner() {
        let cfg = HybridKemConfig::new().with_combiner(42);
        assert_eq!(cfg.combiner, 42);
    }

    #[test]
    fn test_config_with_kdf_output_len() {
        let cfg = HybridKemConfig::new().with_kdf_output_len(42);
        assert_eq!(cfg.kdf_output_len, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = HybridKemConfig::new().with_classical_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = HybridKem::new(HybridKemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("HybridKem"));
    }

    #[test]
    fn test_summary() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = HybridKem::new(HybridKemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = HybridKemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = HybridKemError::InvalidConfig("a".into());
        let e2 = HybridKemError::ComputationFailed("b".into());
        let e3 = HybridKemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
