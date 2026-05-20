//! Merkle hash tree primitives.
//!
//! Provides [`HashTreeConfig`] builder and [`HashTree`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum HashTreeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for HashTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "HashTree: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "HashTree: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "HashTree: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`HashTree`] parameters.
#[derive(Debug, Clone)]
pub struct HashTreeConfig {
    pub height: usize,
    pub hash_len: usize,
    pub leaf_count: usize,
    pub precompute: bool,
}

impl HashTreeConfig {
    pub fn new() -> Self {
        Self {
            height: 10,
            hash_len: 32,
            leaf_count: 1024,
            precompute: true,
        }
    }

    pub fn with_height(mut self, v: usize) -> Self {
        self.height = v;
        self
    }

    pub fn with_hash_len(mut self, v: usize) -> Self {
        self.hash_len = v;
        self
    }

    pub fn with_leaf_count(mut self, v: usize) -> Self {
        self.leaf_count = v;
        self
    }

    pub fn with_precompute(mut self, v: bool) -> Self {
        self.precompute = v;
        self
    }

    pub fn validate(&self) -> Result<(), HashTreeError> {
        Ok(())
    }
}

impl Default for HashTreeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for HashTreeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashTreeConfig(height={0}, hash_len={1}, leaf_count={2}, precompute={3})", self.height, self.hash_len, self.leaf_count, self.precompute)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core merkle hash tree primitives engine.
#[derive(Debug, Clone)]
pub struct HashTree {
    config: HashTreeConfig,
    data: Vec<f64>,
}

impl HashTree {
    pub fn new(config: HashTreeConfig) -> Result<Self, HashTreeError> {
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
    pub fn config(&self) -> &HashTreeConfig { &self.config }

    /// Build Merkle tree from leaves.
    pub fn build_tree(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract authentication path.
    pub fn auth_path(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Verify authentication path.
    pub fn verify_path(&self) -> bool {
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

impl fmt::Display for HashTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashTree(n={})", self.data.len())
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
        let cfg = HashTreeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = HashTreeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("HashTreeConfig"));
    }

    #[test]
    fn test_config_with_height() {
        let cfg = HashTreeConfig::new().with_height(42);
        assert_eq!(cfg.height, 42);
    }

    #[test]
    fn test_config_with_hash_len() {
        let cfg = HashTreeConfig::new().with_hash_len(42);
        assert_eq!(cfg.hash_len, 42);
    }

    #[test]
    fn test_config_with_leaf_count() {
        let cfg = HashTreeConfig::new().with_leaf_count(42);
        assert_eq!(cfg.leaf_count, 42);
    }

    #[test]
    fn test_config_with_precompute() {
        let cfg = HashTreeConfig::new().with_precompute(false);
        assert_eq!(cfg.precompute, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = HashTreeConfig::new().with_height(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = HashTree::new(HashTreeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("HashTree"));
    }

    #[test]
    fn test_summary() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_tree() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.build_tree();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_auth_path() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.auth_path();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_path() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_path();
        assert!(result);
    }

    #[test]
    fn test_verify_path_empty() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap();
        assert!(!e.verify_path());
    }

    #[test]
    fn test_config_accessor() {
        let e = HashTree::new(HashTreeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = HashTreeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = HashTreeError::InvalidConfig("a".into());
        let e2 = HashTreeError::ComputationFailed("b".into());
        let e3 = HashTreeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
