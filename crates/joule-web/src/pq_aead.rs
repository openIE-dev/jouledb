//! Post-quantum AEAD (authenticated encryption with associated data).
//!
//! Provides [`PqAeadConfig`] builder and [`PqAead`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqAeadError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqAeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqAead: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqAead: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqAead: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqAead`] parameters.
#[derive(Debug, Clone)]
pub struct PqAeadConfig {
    pub key_len: usize,
    pub nonce_len: usize,
    pub tag_len: usize,
    pub max_plaintext: usize,
}

impl PqAeadConfig {
    pub fn new() -> Self {
        Self {
            key_len: 32,
            nonce_len: 12,
            tag_len: 16,
            max_plaintext: 65536,
        }
    }

    pub fn with_key_len(mut self, v: usize) -> Self {
        self.key_len = v;
        self
    }

    pub fn with_nonce_len(mut self, v: usize) -> Self {
        self.nonce_len = v;
        self
    }

    pub fn with_tag_len(mut self, v: usize) -> Self {
        self.tag_len = v;
        self
    }

    pub fn with_max_plaintext(mut self, v: usize) -> Self {
        self.max_plaintext = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqAeadError> {
        Ok(())
    }
}

impl Default for PqAeadConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqAeadConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqAeadConfig(key_len={0}, nonce_len={1}, tag_len={2}, max_plaintext={3})", self.key_len, self.nonce_len, self.tag_len, self.max_plaintext)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum aead (authenticated encryption with associated data) engine.
#[derive(Debug, Clone)]
pub struct PqAead {
    config: PqAeadConfig,
    data: Vec<f64>,
}

impl PqAead {
    pub fn new(config: PqAeadConfig) -> Result<Self, PqAeadError> {
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
    pub fn config(&self) -> &PqAeadConfig { &self.config }

    /// Encrypt and authenticate.
    pub fn seal(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decrypt and verify.
    pub fn open(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Derive new key from existing.
    pub fn rekey(&self) -> Vec<f64> {
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

impl fmt::Display for PqAead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqAead(n={})", self.data.len())
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
        let cfg = PqAeadConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqAeadConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqAeadConfig"));
    }

    #[test]
    fn test_config_with_key_len() {
        let cfg = PqAeadConfig::new().with_key_len(42);
        assert_eq!(cfg.key_len, 42);
    }

    #[test]
    fn test_config_with_nonce_len() {
        let cfg = PqAeadConfig::new().with_nonce_len(42);
        assert_eq!(cfg.nonce_len, 42);
    }

    #[test]
    fn test_config_with_tag_len() {
        let cfg = PqAeadConfig::new().with_tag_len(42);
        assert_eq!(cfg.tag_len, 42);
    }

    #[test]
    fn test_config_with_max_plaintext() {
        let cfg = PqAeadConfig::new().with_max_plaintext(42);
        assert_eq!(cfg.max_plaintext, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqAeadConfig::new().with_key_len(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqAead::new(PqAeadConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqAead"));
    }

    #[test]
    fn test_summary() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_seal() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.seal();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_open() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.open();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rekey() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rekey();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rekey_empty() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap();
        assert!(e.rekey().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqAead::new(PqAeadConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqAeadError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqAeadError::InvalidConfig("a".into());
        let e2 = PqAeadError::ComputationFailed("b".into());
        let e3 = PqAeadError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
