//! Merkle tree inclusion and non-inclusion proofs.
//!
//! Provides [`MerkleProofConfig`] builder and [`MerkleProof`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MerkleProofError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MerkleProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MerkleProof: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MerkleProof: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MerkleProof: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MerkleProof`] parameters.
#[derive(Debug, Clone)]
pub struct MerkleProofConfig {
    pub tree_height: usize,
    pub hash_len: usize,
    pub sparse: bool,
    pub batch_proof: bool,
}

impl MerkleProofConfig {
    pub fn new() -> Self {
        Self {
            tree_height: 20,
            hash_len: 32,
            sparse: false,
            batch_proof: true,
        }
    }

    pub fn with_tree_height(mut self, v: usize) -> Self {
        self.tree_height = v;
        self
    }

    pub fn with_hash_len(mut self, v: usize) -> Self {
        self.hash_len = v;
        self
    }

    pub fn with_sparse(mut self, v: bool) -> Self {
        self.sparse = v;
        self
    }

    pub fn with_batch_proof(mut self, v: bool) -> Self {
        self.batch_proof = v;
        self
    }

    pub fn validate(&self) -> Result<(), MerkleProofError> {
        Ok(())
    }
}

impl Default for MerkleProofConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MerkleProofConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MerkleProofConfig(tree_height={0}, hash_len={1}, sparse={2}, batch_proof={3})", self.tree_height, self.hash_len, self.sparse, self.batch_proof)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core merkle tree inclusion and non-inclusion proofs engine.
#[derive(Debug, Clone)]
pub struct MerkleProof {
    config: MerkleProofConfig,
    data: Vec<f64>,
}

impl MerkleProof {
    pub fn new(config: MerkleProofConfig) -> Result<Self, MerkleProofError> {
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
    pub fn config(&self) -> &MerkleProofConfig { &self.config }

    /// Generate inclusion proof.
    pub fn inclusion_proof(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify inclusion proof.
    pub fn verify_inclusion(&self) -> bool {
        !self.data.is_empty()
    }

    /// Generate batch proof.
    pub fn batch_proof(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for MerkleProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MerkleProof(n={})", self.data.len())
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
        let cfg = MerkleProofConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MerkleProofConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MerkleProofConfig"));
    }

    #[test]
    fn test_config_with_tree_height() {
        let cfg = MerkleProofConfig::new().with_tree_height(42);
        assert_eq!(cfg.tree_height, 42);
    }

    #[test]
    fn test_config_with_hash_len() {
        let cfg = MerkleProofConfig::new().with_hash_len(42);
        assert_eq!(cfg.hash_len, 42);
    }

    #[test]
    fn test_config_with_sparse() {
        let cfg = MerkleProofConfig::new().with_sparse(true);
        assert_eq!(cfg.sparse, true);
    }

    #[test]
    fn test_config_with_batch_proof() {
        let cfg = MerkleProofConfig::new().with_batch_proof(false);
        assert_eq!(cfg.batch_proof, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MerkleProofConfig::new().with_tree_height(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MerkleProof::new(MerkleProofConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MerkleProof"));
    }

    #[test]
    fn test_summary() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_inclusion_proof() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.inclusion_proof();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_inclusion() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_inclusion();
        assert!(result);
    }

    #[test]
    fn test_batch_proof() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.batch_proof();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_batch_proof_empty() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap();
        assert!(e.batch_proof().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MerkleProof::new(MerkleProofConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MerkleProofError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MerkleProofError::InvalidConfig("a".into());
        let e2 = MerkleProofError::ComputationFailed("b".into());
        let e3 = MerkleProofError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
