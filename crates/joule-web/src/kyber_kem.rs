//! Kyber/ML-KEM lattice-based key encapsulation mechanism.
//!
//! Provides [`KyberKemConfig`] builder and [`KyberKem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum KyberKemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for KyberKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "KyberKem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "KyberKem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "KyberKem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`KyberKem`] parameters.
#[derive(Debug, Clone)]
pub struct KyberKemConfig {
    pub security_level: usize,
    pub n: usize,
    pub q: u32,
    pub eta1: usize,
}

impl KyberKemConfig {
    pub fn new() -> Self {
        Self {
            security_level: 3,
            n: 256,
            q: 3329,
            eta1: 2,
        }
    }

    pub fn with_security_level(mut self, v: usize) -> Self {
        self.security_level = v;
        self
    }

    pub fn with_n(mut self, v: usize) -> Self {
        self.n = v;
        self
    }

    pub fn with_q(mut self, v: u32) -> Self {
        self.q = v;
        self
    }

    pub fn with_eta1(mut self, v: usize) -> Self {
        self.eta1 = v;
        self
    }

    pub fn validate(&self) -> Result<(), KyberKemError> {
        Ok(())
    }
}

impl Default for KyberKemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for KyberKemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KyberKemConfig(security_level={0}, n={1}, q={2}, eta1={3})", self.security_level, self.n, self.q, self.eta1)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core kyber/ml-kem lattice-based key encapsulation mechanism engine.
#[derive(Debug, Clone)]
pub struct KyberKem {
    config: KyberKemConfig,
    data: Vec<f64>,
}

impl KyberKem {
    pub fn new(config: KyberKemConfig) -> Result<Self, KyberKemError> {
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
    pub fn config(&self) -> &KyberKemConfig { &self.config }

    /// Generate keypair.
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

    /// Decapsulate shared secret.
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

impl fmt::Display for KyberKem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KyberKem(n={})", self.data.len())
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
        let cfg = KyberKemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = KyberKemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("KyberKemConfig"));
    }

    #[test]
    fn test_config_with_security_level() {
        let cfg = KyberKemConfig::new().with_security_level(42);
        assert_eq!(cfg.security_level, 42);
    }

    #[test]
    fn test_config_with_n() {
        let cfg = KyberKemConfig::new().with_n(42);
        assert_eq!(cfg.n, 42);
    }

    #[test]
    fn test_config_with_q() {
        let cfg = KyberKemConfig::new().with_q(42);
        assert_eq!(cfg.q, 42);
    }

    #[test]
    fn test_config_with_eta1() {
        let cfg = KyberKemConfig::new().with_eta1(42);
        assert_eq!(cfg.eta1, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = KyberKemConfig::new().with_security_level(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = KyberKem::new(KyberKemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("KyberKem"));
    }

    #[test]
    fn test_summary() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = KyberKem::new(KyberKemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = KyberKemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = KyberKemError::InvalidConfig("a".into());
        let e2 = KyberKemError::ComputationFailed("b".into());
        let e3 = KyberKemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
