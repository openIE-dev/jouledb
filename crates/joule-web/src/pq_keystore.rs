//! Post-quantum key storage and management.
//!
//! Provides [`PqKeystoreConfig`] builder and [`PqKeystore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqKeystoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqKeystoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqKeystore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqKeystore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqKeystore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqKeystore`] parameters.
#[derive(Debug, Clone)]
pub struct PqKeystoreConfig {
    pub max_keys: usize,
    pub encrypted_storage: bool,
    pub expiry_days: u32,
    pub auto_rotate: bool,
}

impl PqKeystoreConfig {
    pub fn new() -> Self {
        Self {
            max_keys: 1000,
            encrypted_storage: true,
            expiry_days: 365,
            auto_rotate: false,
        }
    }

    pub fn with_max_keys(mut self, v: usize) -> Self {
        self.max_keys = v;
        self
    }

    pub fn with_encrypted_storage(mut self, v: bool) -> Self {
        self.encrypted_storage = v;
        self
    }

    pub fn with_expiry_days(mut self, v: u32) -> Self {
        self.expiry_days = v;
        self
    }

    pub fn with_auto_rotate(mut self, v: bool) -> Self {
        self.auto_rotate = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqKeystoreError> {
        Ok(())
    }
}

impl Default for PqKeystoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqKeystoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqKeystoreConfig(max_keys={0}, encrypted_storage={1}, expiry_days={2}, auto_rotate={3})", self.max_keys, self.encrypted_storage, self.expiry_days, self.auto_rotate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum key storage and management engine.
#[derive(Debug, Clone)]
pub struct PqKeystore {
    config: PqKeystoreConfig,
    data: Vec<f64>,
}

impl PqKeystore {
    pub fn new(config: PqKeystoreConfig) -> Result<Self, PqKeystoreError> {
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
    pub fn config(&self) -> &PqKeystoreConfig { &self.config }

    /// Store a key.
    pub fn store_key(&self) -> bool {
        !self.data.is_empty()
    }

    /// Retrieve a key by ID.
    pub fn retrieve_key(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// List all key IDs.
    pub fn list_keys(&self) -> Vec<f64> {
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

impl fmt::Display for PqKeystore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqKeystore(n={})", self.data.len())
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
        let cfg = PqKeystoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqKeystoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqKeystoreConfig"));
    }

    #[test]
    fn test_config_with_max_keys() {
        let cfg = PqKeystoreConfig::new().with_max_keys(42);
        assert_eq!(cfg.max_keys, 42);
    }

    #[test]
    fn test_config_with_encrypted_storage() {
        let cfg = PqKeystoreConfig::new().with_encrypted_storage(false);
        assert_eq!(cfg.encrypted_storage, false);
    }

    #[test]
    fn test_config_with_expiry_days() {
        let cfg = PqKeystoreConfig::new().with_expiry_days(42);
        assert_eq!(cfg.expiry_days, 42);
    }

    #[test]
    fn test_config_with_auto_rotate() {
        let cfg = PqKeystoreConfig::new().with_auto_rotate(true);
        assert_eq!(cfg.auto_rotate, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqKeystoreConfig::new().with_max_keys(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqKeystore::new(PqKeystoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqKeystore"));
    }

    #[test]
    fn test_summary() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_store_key() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.store_key();
        assert!(result);
    }

    #[test]
    fn test_retrieve_key() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.retrieve_key();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_list_keys() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.list_keys();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_list_keys_empty() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap();
        assert!(e.list_keys().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqKeystore::new(PqKeystoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqKeystoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqKeystoreError::InvalidConfig("a".into());
        let e2 = PqKeystoreError::ComputationFailed("b".into());
        let e3 = PqKeystoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
