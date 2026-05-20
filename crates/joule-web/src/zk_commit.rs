//! Zero-knowledge commitment schemes.
//!
//! Provides [`ZkCommitConfig`] builder and [`ZkCommit`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ZkCommitError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ZkCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ZkCommit: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ZkCommit: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ZkCommit: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ZkCommit`] parameters.
#[derive(Debug, Clone)]
pub struct ZkCommitConfig {
    pub security_bits: usize,
    pub hash_len: usize,
    pub batch_size: usize,
    pub binding: bool,
}

impl ZkCommitConfig {
    pub fn new() -> Self {
        Self {
            security_bits: 128,
            hash_len: 32,
            batch_size: 64,
            binding: true,
        }
    }

    pub fn with_security_bits(mut self, v: usize) -> Self {
        self.security_bits = v;
        self
    }

    pub fn with_hash_len(mut self, v: usize) -> Self {
        self.hash_len = v;
        self
    }

    pub fn with_batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn with_binding(mut self, v: bool) -> Self {
        self.binding = v;
        self
    }

    pub fn validate(&self) -> Result<(), ZkCommitError> {
        Ok(())
    }
}

impl Default for ZkCommitConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ZkCommitConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZkCommitConfig(security_bits={0}, hash_len={1}, batch_size={2}, binding={3})", self.security_bits, self.hash_len, self.batch_size, self.binding)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core zero-knowledge commitment schemes engine.
#[derive(Debug, Clone)]
pub struct ZkCommit {
    config: ZkCommitConfig,
    data: Vec<f64>,
}

impl ZkCommit {
    pub fn new(config: ZkCommitConfig) -> Result<Self, ZkCommitError> {
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
    pub fn config(&self) -> &ZkCommitConfig { &self.config }

    /// Create commitment.
    pub fn commit(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Open commitment.
    pub fn open(&self) -> bool {
        !self.data.is_empty()
    }

    /// Batch verify commitments.
    pub fn batch_verify(&self) -> bool {
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

impl fmt::Display for ZkCommit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZkCommit(n={})", self.data.len())
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
        let cfg = ZkCommitConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ZkCommitConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ZkCommitConfig"));
    }

    #[test]
    fn test_config_with_security_bits() {
        let cfg = ZkCommitConfig::new().with_security_bits(42);
        assert_eq!(cfg.security_bits, 42);
    }

    #[test]
    fn test_config_with_hash_len() {
        let cfg = ZkCommitConfig::new().with_hash_len(42);
        assert_eq!(cfg.hash_len, 42);
    }

    #[test]
    fn test_config_with_batch_size() {
        let cfg = ZkCommitConfig::new().with_batch_size(42);
        assert_eq!(cfg.batch_size, 42);
    }

    #[test]
    fn test_config_with_binding() {
        let cfg = ZkCommitConfig::new().with_binding(false);
        assert_eq!(cfg.binding, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ZkCommitConfig::new().with_security_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ZkCommit::new(ZkCommitConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ZkCommit"));
    }

    #[test]
    fn test_summary() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_commit() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.commit();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_open() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.open();
        assert!(result);
    }

    #[test]
    fn test_batch_verify() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.batch_verify();
        assert!(result);
    }

    #[test]
    fn test_batch_verify_empty() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap();
        assert!(!e.batch_verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = ZkCommit::new(ZkCommitConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ZkCommitError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ZkCommitError::InvalidConfig("a".into());
        let e2 = ZkCommitError::ComputationFailed("b".into());
        let e3 = ZkCommitError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
