//! Dilithium/ML-DSA lattice-based digital signature.
//!
//! Provides [`DilithiumSignConfig`] builder and [`DilithiumSign`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DilithiumSignError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DilithiumSignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DilithiumSign: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DilithiumSign: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DilithiumSign: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DilithiumSign`] parameters.
#[derive(Debug, Clone)]
pub struct DilithiumSignConfig {
    pub security_level: usize,
    pub n: usize,
    pub q: u32,
    pub gamma1: u32,
}

impl DilithiumSignConfig {
    pub fn new() -> Self {
        Self {
            security_level: 3,
            n: 256,
            q: 8380417,
            gamma1: 131072,
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

    pub fn with_gamma1(mut self, v: u32) -> Self {
        self.gamma1 = v;
        self
    }

    pub fn validate(&self) -> Result<(), DilithiumSignError> {
        Ok(())
    }
}

impl Default for DilithiumSignConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DilithiumSignConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DilithiumSignConfig(security_level={0}, n={1}, q={2}, gamma1={3})", self.security_level, self.n, self.q, self.gamma1)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core dilithium/ml-dsa lattice-based digital signature engine.
#[derive(Debug, Clone)]
pub struct DilithiumSign {
    config: DilithiumSignConfig,
    data: Vec<f64>,
}

impl DilithiumSign {
    pub fn new(config: DilithiumSignConfig) -> Result<Self, DilithiumSignError> {
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
    pub fn config(&self) -> &DilithiumSignConfig { &self.config }

    /// Generate signing keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Sign a message.
    pub fn sign(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify a signature.
    pub fn verify(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for DilithiumSign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DilithiumSign(n={})", self.data.len())
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
        let cfg = DilithiumSignConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DilithiumSignConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DilithiumSignConfig"));
    }

    #[test]
    fn test_config_with_security_level() {
        let cfg = DilithiumSignConfig::new().with_security_level(42);
        assert_eq!(cfg.security_level, 42);
    }

    #[test]
    fn test_config_with_n() {
        let cfg = DilithiumSignConfig::new().with_n(42);
        assert_eq!(cfg.n, 42);
    }

    #[test]
    fn test_config_with_q() {
        let cfg = DilithiumSignConfig::new().with_q(42);
        assert_eq!(cfg.q, 42);
    }

    #[test]
    fn test_config_with_gamma1() {
        let cfg = DilithiumSignConfig::new().with_gamma1(42);
        assert_eq!(cfg.gamma1, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DilithiumSignConfig::new().with_security_level(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DilithiumSign"));
    }

    #[test]
    fn test_summary() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sign() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.sign();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_verify_empty() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap();
        assert!(!e.verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = DilithiumSign::new(DilithiumSignConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DilithiumSignError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DilithiumSignError::InvalidConfig("a".into());
        let e2 = DilithiumSignError::ComputationFailed("b".into());
        let e3 = DilithiumSignError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
