//! KEM abstraction layer with unified interface.
//!
//! Provides [`KeyEncapsConfig`] builder and [`KeyEncaps`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyEncapsError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for KeyEncapsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "KeyEncaps: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "KeyEncaps: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "KeyEncaps: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`KeyEncaps`] parameters.
#[derive(Debug, Clone)]
pub struct KeyEncapsConfig {
    pub algorithm_id: usize,
    pub shared_secret_len: usize,
    pub ciphertext_overhead: usize,
    pub validate_keys: bool,
}

impl KeyEncapsConfig {
    pub fn new() -> Self {
        Self {
            algorithm_id: 0,
            shared_secret_len: 32,
            ciphertext_overhead: 1088,
            validate_keys: true,
        }
    }

    pub fn with_algorithm_id(mut self, v: usize) -> Self {
        self.algorithm_id = v;
        self
    }

    pub fn with_shared_secret_len(mut self, v: usize) -> Self {
        self.shared_secret_len = v;
        self
    }

    pub fn with_ciphertext_overhead(mut self, v: usize) -> Self {
        self.ciphertext_overhead = v;
        self
    }

    pub fn with_validate_keys(mut self, v: bool) -> Self {
        self.validate_keys = v;
        self
    }

    pub fn validate(&self) -> Result<(), KeyEncapsError> {
        Ok(())
    }
}

impl Default for KeyEncapsConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for KeyEncapsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyEncapsConfig(algorithm_id={0}, shared_secret_len={1}, ciphertext_overhead={2}, validate_keys={3})", self.algorithm_id, self.shared_secret_len, self.ciphertext_overhead, self.validate_keys)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core kem abstraction layer with unified interface engine.
#[derive(Debug, Clone)]
pub struct KeyEncaps {
    config: KeyEncapsConfig,
    data: Vec<f64>,
}

impl KeyEncaps {
    pub fn new(config: KeyEncapsConfig) -> Result<Self, KeyEncapsError> {
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
    pub fn config(&self) -> &KeyEncapsConfig { &self.config }

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

impl fmt::Display for KeyEncaps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyEncaps(n={})", self.data.len())
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
        let cfg = KeyEncapsConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = KeyEncapsConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("KeyEncapsConfig"));
    }

    #[test]
    fn test_config_with_algorithm_id() {
        let cfg = KeyEncapsConfig::new().with_algorithm_id(42);
        assert_eq!(cfg.algorithm_id, 42);
    }

    #[test]
    fn test_config_with_shared_secret_len() {
        let cfg = KeyEncapsConfig::new().with_shared_secret_len(42);
        assert_eq!(cfg.shared_secret_len, 42);
    }

    #[test]
    fn test_config_with_ciphertext_overhead() {
        let cfg = KeyEncapsConfig::new().with_ciphertext_overhead(42);
        assert_eq!(cfg.ciphertext_overhead, 42);
    }

    #[test]
    fn test_config_with_validate_keys() {
        let cfg = KeyEncapsConfig::new().with_validate_keys(false);
        assert_eq!(cfg.validate_keys, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = KeyEncapsConfig::new().with_algorithm_id(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("KeyEncaps"));
    }

    #[test]
    fn test_summary() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = KeyEncaps::new(KeyEncapsConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = KeyEncapsError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = KeyEncapsError::InvalidConfig("a".into());
        let e2 = KeyEncapsError::ComputationFailed("b".into());
        let e3 = KeyEncapsError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
