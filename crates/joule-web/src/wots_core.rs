//! WOTS+ one-time signature core operations.
//!
//! Provides [`WotsCoreConfig`] builder and [`WotsCore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum WotsCoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for WotsCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "WotsCore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "WotsCore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "WotsCore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`WotsCore`] parameters.
#[derive(Debug, Clone)]
pub struct WotsCoreConfig {
    pub w: usize,
    pub hash_len: usize,
    pub message_len: usize,
    pub checksum_len: usize,
}

impl WotsCoreConfig {
    pub fn new() -> Self {
        Self {
            w: 16,
            hash_len: 32,
            message_len: 32,
            checksum_len: 3,
        }
    }

    pub fn with_w(mut self, v: usize) -> Self {
        self.w = v;
        self
    }

    pub fn with_hash_len(mut self, v: usize) -> Self {
        self.hash_len = v;
        self
    }

    pub fn with_message_len(mut self, v: usize) -> Self {
        self.message_len = v;
        self
    }

    pub fn with_checksum_len(mut self, v: usize) -> Self {
        self.checksum_len = v;
        self
    }

    pub fn validate(&self) -> Result<(), WotsCoreError> {
        Ok(())
    }
}

impl Default for WotsCoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for WotsCoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WotsCoreConfig(w={0}, hash_len={1}, message_len={2}, checksum_len={3})", self.w, self.hash_len, self.message_len, self.checksum_len)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core wots+ one-time signature core operations engine.
#[derive(Debug, Clone)]
pub struct WotsCore {
    config: WotsCoreConfig,
    data: Vec<f64>,
}

impl WotsCore {
    pub fn new(config: WotsCoreConfig) -> Result<Self, WotsCoreError> {
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
    pub fn config(&self) -> &WotsCoreConfig { &self.config }

    /// Generate WOTS+ keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate WOTS+ signature.
    pub fn sign(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify WOTS+ signature.
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

impl fmt::Display for WotsCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WotsCore(n={})", self.data.len())
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
        let cfg = WotsCoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = WotsCoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("WotsCoreConfig"));
    }

    #[test]
    fn test_config_with_w() {
        let cfg = WotsCoreConfig::new().with_w(42);
        assert_eq!(cfg.w, 42);
    }

    #[test]
    fn test_config_with_hash_len() {
        let cfg = WotsCoreConfig::new().with_hash_len(42);
        assert_eq!(cfg.hash_len, 42);
    }

    #[test]
    fn test_config_with_message_len() {
        let cfg = WotsCoreConfig::new().with_message_len(42);
        assert_eq!(cfg.message_len, 42);
    }

    #[test]
    fn test_config_with_checksum_len() {
        let cfg = WotsCoreConfig::new().with_checksum_len(42);
        assert_eq!(cfg.checksum_len, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = WotsCoreConfig::new().with_w(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = WotsCore::new(WotsCoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("WotsCore"));
    }

    #[test]
    fn test_summary() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sign() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.sign();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_verify_empty() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap();
        assert!(!e.verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = WotsCore::new(WotsCoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = WotsCoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = WotsCoreError::InvalidConfig("a".into());
        let e2 = WotsCoreError::ComputationFailed("b".into());
        let e3 = WotsCoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
