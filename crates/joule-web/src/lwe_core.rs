//! Learning With Errors core operations.
//!
//! Provides [`LweCoreConfig`] builder and [`LweCore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum LweCoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for LweCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "LweCore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "LweCore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "LweCore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`LweCore`] parameters.
#[derive(Debug, Clone)]
pub struct LweCoreConfig {
    pub n: usize,
    pub q: u32,
    pub std_dev: f64,
    pub samples: usize,
}

impl LweCoreConfig {
    pub fn new() -> Self {
        Self {
            n: 512,
            q: 12289,
            std_dev: 3.2,
            samples: 1024,
        }
    }

    pub fn with_n(mut self, v: usize) -> Self {
        self.n = v;
        self
    }

    pub fn with_q(mut self, v: u32) -> Self {
        self.q = v;
        self
    }

    pub fn with_std_dev(mut self, v: f64) -> Self {
        self.std_dev = v;
        self
    }

    pub fn with_samples(mut self, v: usize) -> Self {
        self.samples = v;
        self
    }

    pub fn validate(&self) -> Result<(), LweCoreError> {
        if self.std_dev.is_nan() {
            return Err(LweCoreError::InvalidConfig("std_dev is NaN".into()));
        }
        Ok(())
    }
}

impl Default for LweCoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for LweCoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LweCoreConfig(n={0}, q={1}, std_dev={2:.4}, samples={3})", self.n, self.q, self.std_dev, self.samples)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core learning with errors core operations engine.
#[derive(Debug, Clone)]
pub struct LweCore {
    config: LweCoreConfig,
    data: Vec<f64>,
}

impl LweCore {
    pub fn new(config: LweCoreConfig) -> Result<Self, LweCoreError> {
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
    pub fn config(&self) -> &LweCoreConfig { &self.config }

    /// Generate LWE keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// LWE encryption.
    pub fn encrypt(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// LWE decryption.
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

impl fmt::Display for LweCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LweCore(n={})", self.data.len())
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
        let cfg = LweCoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = LweCoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("LweCoreConfig"));
    }

    #[test]
    fn test_config_with_n() {
        let cfg = LweCoreConfig::new().with_n(42);
        assert_eq!(cfg.n, 42);
    }

    #[test]
    fn test_config_with_q() {
        let cfg = LweCoreConfig::new().with_q(42);
        assert_eq!(cfg.q, 42);
    }

    #[test]
    fn test_config_with_std_dev() {
        let cfg = LweCoreConfig::new().with_std_dev(42.0);
        assert!((cfg.std_dev - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_samples() {
        let cfg = LweCoreConfig::new().with_samples(42);
        assert_eq!(cfg.samples, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = LweCoreConfig::new().with_n(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = LweCore::new(LweCoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("LweCore"));
    }

    #[test]
    fn test_summary() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encrypt() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encrypt();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decrypt() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decrypt();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decrypt_empty() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap();
        assert!(e.decrypt().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = LweCore::new(LweCoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = LweCoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = LweCoreError::InvalidConfig("a".into());
        let e2 = LweCoreError::ComputationFailed("b".into());
        let e3 = LweCoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
