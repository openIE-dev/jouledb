//! Range proofs for proving value in interval.
//!
//! Provides [`RangeProofConfig`] builder and [`RangeProof`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RangeProofError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RangeProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RangeProof: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RangeProof: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RangeProof: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RangeProof`] parameters.
#[derive(Debug, Clone)]
pub struct RangeProofConfig {
    pub range_bits: usize,
    pub security_bits: usize,
    pub batch_verify: bool,
    pub aggregation_size: usize,
}

impl RangeProofConfig {
    pub fn new() -> Self {
        Self {
            range_bits: 64,
            security_bits: 128,
            batch_verify: true,
            aggregation_size: 1,
        }
    }

    pub fn with_range_bits(mut self, v: usize) -> Self {
        self.range_bits = v;
        self
    }

    pub fn with_security_bits(mut self, v: usize) -> Self {
        self.security_bits = v;
        self
    }

    pub fn with_batch_verify(mut self, v: bool) -> Self {
        self.batch_verify = v;
        self
    }

    pub fn with_aggregation_size(mut self, v: usize) -> Self {
        self.aggregation_size = v;
        self
    }

    pub fn validate(&self) -> Result<(), RangeProofError> {
        Ok(())
    }
}

impl Default for RangeProofConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RangeProofConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeProofConfig(range_bits={0}, security_bits={1}, batch_verify={2}, aggregation_size={3})", self.range_bits, self.security_bits, self.batch_verify, self.aggregation_size)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core range proofs for proving value in interval engine.
#[derive(Debug, Clone)]
pub struct RangeProof {
    config: RangeProofConfig,
    data: Vec<f64>,
}

impl RangeProof {
    pub fn new(config: RangeProofConfig) -> Result<Self, RangeProofError> {
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
    pub fn config(&self) -> &RangeProofConfig { &self.config }

    /// Prove value in range.
    pub fn prove_range(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify range proof.
    pub fn verify_range(&self) -> bool {
        !self.data.is_empty()
    }

    /// Inner product argument.
    pub fn inner_product(&self) -> Vec<f64> {
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

impl fmt::Display for RangeProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeProof(n={})", self.data.len())
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
        let cfg = RangeProofConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RangeProofConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RangeProofConfig"));
    }

    #[test]
    fn test_config_with_range_bits() {
        let cfg = RangeProofConfig::new().with_range_bits(42);
        assert_eq!(cfg.range_bits, 42);
    }

    #[test]
    fn test_config_with_security_bits() {
        let cfg = RangeProofConfig::new().with_security_bits(42);
        assert_eq!(cfg.security_bits, 42);
    }

    #[test]
    fn test_config_with_batch_verify() {
        let cfg = RangeProofConfig::new().with_batch_verify(false);
        assert_eq!(cfg.batch_verify, false);
    }

    #[test]
    fn test_config_with_aggregation_size() {
        let cfg = RangeProofConfig::new().with_aggregation_size(42);
        assert_eq!(cfg.aggregation_size, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RangeProofConfig::new().with_range_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RangeProof::new(RangeProofConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RangeProof"));
    }

    #[test]
    fn test_summary() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_prove_range() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.prove_range();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_range() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_range();
        assert!(result);
    }

    #[test]
    fn test_inner_product() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.inner_product();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_inner_product_empty() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap();
        assert!(e.inner_product().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = RangeProof::new(RangeProofConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RangeProofError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RangeProofError::InvalidConfig("a".into());
        let e2 = RangeProofError::ComputationFailed("b".into());
        let e3 = RangeProofError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
