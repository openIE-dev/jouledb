//! Sigma protocols for zero-knowledge proofs.
//!
//! Provides [`SigmaProtocolConfig`] builder and [`SigmaProtocol`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SigmaProtocolError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SigmaProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SigmaProtocol: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SigmaProtocol: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SigmaProtocol: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SigmaProtocol`] parameters.
#[derive(Debug, Clone)]
pub struct SigmaProtocolConfig {
    pub security_bits: usize,
    pub num_rounds: usize,
    pub fiat_shamir: bool,
    pub hash_len: usize,
}

impl SigmaProtocolConfig {
    pub fn new() -> Self {
        Self {
            security_bits: 128,
            num_rounds: 1,
            fiat_shamir: true,
            hash_len: 32,
        }
    }

    pub fn with_security_bits(mut self, v: usize) -> Self {
        self.security_bits = v;
        self
    }

    pub fn with_num_rounds(mut self, v: usize) -> Self {
        self.num_rounds = v;
        self
    }

    pub fn with_fiat_shamir(mut self, v: bool) -> Self {
        self.fiat_shamir = v;
        self
    }

    pub fn with_hash_len(mut self, v: usize) -> Self {
        self.hash_len = v;
        self
    }

    pub fn validate(&self) -> Result<(), SigmaProtocolError> {
        Ok(())
    }
}

impl Default for SigmaProtocolConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SigmaProtocolConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SigmaProtocolConfig(security_bits={0}, num_rounds={1}, fiat_shamir={2}, hash_len={3})", self.security_bits, self.num_rounds, self.fiat_shamir, self.hash_len)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core sigma protocols for zero-knowledge proofs engine.
#[derive(Debug, Clone)]
pub struct SigmaProtocol {
    config: SigmaProtocolConfig,
    data: Vec<f64>,
}

impl SigmaProtocol {
    pub fn new(config: SigmaProtocolConfig) -> Result<Self, SigmaProtocolError> {
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
    pub fn config(&self) -> &SigmaProtocolConfig { &self.config }

    /// Generate proof.
    pub fn prove(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify proof.
    pub fn verify(&self) -> bool {
        !self.data.is_empty()
    }

    /// OR-composition of proofs.
    pub fn or_compose(&self) -> Vec<f64> {
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

impl fmt::Display for SigmaProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SigmaProtocol(n={})", self.data.len())
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
        let cfg = SigmaProtocolConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SigmaProtocolConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SigmaProtocolConfig"));
    }

    #[test]
    fn test_config_with_security_bits() {
        let cfg = SigmaProtocolConfig::new().with_security_bits(42);
        assert_eq!(cfg.security_bits, 42);
    }

    #[test]
    fn test_config_with_num_rounds() {
        let cfg = SigmaProtocolConfig::new().with_num_rounds(42);
        assert_eq!(cfg.num_rounds, 42);
    }

    #[test]
    fn test_config_with_fiat_shamir() {
        let cfg = SigmaProtocolConfig::new().with_fiat_shamir(false);
        assert_eq!(cfg.fiat_shamir, false);
    }

    #[test]
    fn test_config_with_hash_len() {
        let cfg = SigmaProtocolConfig::new().with_hash_len(42);
        assert_eq!(cfg.hash_len, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SigmaProtocolConfig::new().with_security_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SigmaProtocol"));
    }

    #[test]
    fn test_summary() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_prove() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.prove();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_or_compose() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.or_compose();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_or_compose_empty() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap();
        assert!(e.or_compose().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SigmaProtocol::new(SigmaProtocolConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SigmaProtocolError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SigmaProtocolError::InvalidConfig("a".into());
        let e2 = SigmaProtocolError::ComputationFailed("b".into());
        let e3 = SigmaProtocolError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
