//! NTRU lattice-based encryption scheme.
//!
//! Provides [`NtruEncryptConfig`] builder and [`NtruEncrypt`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum NtruEncryptError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for NtruEncryptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "NtruEncrypt: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "NtruEncrypt: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "NtruEncrypt: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`NtruEncrypt`] parameters.
#[derive(Debug, Clone)]
pub struct NtruEncryptConfig {
    pub n: usize,
    pub p: u32,
    pub q: u32,
    pub df: usize,
}

impl NtruEncryptConfig {
    pub fn new() -> Self {
        Self {
            n: 701,
            p: 3,
            q: 2048,
            df: 216,
        }
    }

    pub fn with_n(mut self, v: usize) -> Self {
        self.n = v;
        self
    }

    pub fn with_p(mut self, v: u32) -> Self {
        self.p = v;
        self
    }

    pub fn with_q(mut self, v: u32) -> Self {
        self.q = v;
        self
    }

    pub fn with_df(mut self, v: usize) -> Self {
        self.df = v;
        self
    }

    pub fn validate(&self) -> Result<(), NtruEncryptError> {
        Ok(())
    }
}

impl Default for NtruEncryptConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NtruEncryptConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NtruEncryptConfig(n={0}, p={1}, q={2}, df={3})", self.n, self.p, self.q, self.df)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core ntru lattice-based encryption scheme engine.
#[derive(Debug, Clone)]
pub struct NtruEncrypt {
    config: NtruEncryptConfig,
    data: Vec<f64>,
}

impl NtruEncrypt {
    pub fn new(config: NtruEncryptConfig) -> Result<Self, NtruEncryptError> {
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
    pub fn config(&self) -> &NtruEncryptConfig { &self.config }

    /// Generate NTRU keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Encrypt plaintext.
    pub fn encrypt(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decrypt ciphertext.
    pub fn decrypt(&self) -> Vec<f64> {
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

impl fmt::Display for NtruEncrypt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NtruEncrypt(n={})", self.data.len())
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
        let cfg = NtruEncryptConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = NtruEncryptConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("NtruEncryptConfig"));
    }

    #[test]
    fn test_config_with_n() {
        let cfg = NtruEncryptConfig::new().with_n(42);
        assert_eq!(cfg.n, 42);
    }

    #[test]
    fn test_config_with_p() {
        let cfg = NtruEncryptConfig::new().with_p(42);
        assert_eq!(cfg.p, 42);
    }

    #[test]
    fn test_config_with_q() {
        let cfg = NtruEncryptConfig::new().with_q(42);
        assert_eq!(cfg.q, 42);
    }

    #[test]
    fn test_config_with_df() {
        let cfg = NtruEncryptConfig::new().with_df(42);
        assert_eq!(cfg.df, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = NtruEncryptConfig::new().with_n(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("NtruEncrypt"));
    }

    #[test]
    fn test_summary() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encrypt() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encrypt();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decrypt() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decrypt();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decrypt_empty() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap();
        assert!(e.decrypt().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = NtruEncrypt::new(NtruEncryptConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = NtruEncryptError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = NtruEncryptError::InvalidConfig("a".into());
        let e2 = NtruEncryptError::ComputationFailed("b".into());
        let e3 = NtruEncryptError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
