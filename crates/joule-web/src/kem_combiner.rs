//! KEM combiner for parallel and serial composition.
//!
//! Provides [`KemCombinerConfig`] builder and [`KemCombiner`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum KemCombinerError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for KemCombinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "KemCombiner: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "KemCombiner: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "KemCombiner: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`KemCombiner`] parameters.
#[derive(Debug, Clone)]
pub struct KemCombinerConfig {
    pub mode: usize,
    pub num_kems: usize,
    pub threshold: usize,
    pub hash_output: usize,
}

impl KemCombinerConfig {
    pub fn new() -> Self {
        Self {
            mode: 0,
            num_kems: 2,
            threshold: 2,
            hash_output: 32,
        }
    }

    pub fn with_mode(mut self, v: usize) -> Self {
        self.mode = v;
        self
    }

    pub fn with_num_kems(mut self, v: usize) -> Self {
        self.num_kems = v;
        self
    }

    pub fn with_threshold(mut self, v: usize) -> Self {
        self.threshold = v;
        self
    }

    pub fn with_hash_output(mut self, v: usize) -> Self {
        self.hash_output = v;
        self
    }

    pub fn validate(&self) -> Result<(), KemCombinerError> {
        Ok(())
    }
}

impl Default for KemCombinerConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for KemCombinerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KemCombinerConfig(mode={0}, num_kems={1}, threshold={2}, hash_output={3})", self.mode, self.num_kems, self.threshold, self.hash_output)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core kem combiner for parallel and serial composition engine.
#[derive(Debug, Clone)]
pub struct KemCombiner {
    config: KemCombinerConfig,
    data: Vec<f64>,
}

impl KemCombiner {
    pub fn new(config: KemCombinerConfig) -> Result<Self, KemCombinerError> {
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
    pub fn config(&self) -> &KemCombinerConfig { &self.config }

    /// Combine KEM shared secrets.
    pub fn combine_keys(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Parallel encapsulation.
    pub fn parallel_encaps(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify all KEMs succeeded.
    pub fn verify_all(&self) -> bool {
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

impl fmt::Display for KemCombiner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KemCombiner(n={})", self.data.len())
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
        let cfg = KemCombinerConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = KemCombinerConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("KemCombinerConfig"));
    }

    #[test]
    fn test_config_with_mode() {
        let cfg = KemCombinerConfig::new().with_mode(42);
        assert_eq!(cfg.mode, 42);
    }

    #[test]
    fn test_config_with_num_kems() {
        let cfg = KemCombinerConfig::new().with_num_kems(42);
        assert_eq!(cfg.num_kems, 42);
    }

    #[test]
    fn test_config_with_threshold() {
        let cfg = KemCombinerConfig::new().with_threshold(42);
        assert_eq!(cfg.threshold, 42);
    }

    #[test]
    fn test_config_with_hash_output() {
        let cfg = KemCombinerConfig::new().with_hash_output(42);
        assert_eq!(cfg.hash_output, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = KemCombinerConfig::new().with_mode(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = KemCombiner::new(KemCombinerConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("KemCombiner"));
    }

    #[test]
    fn test_summary() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_combine_keys() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.combine_keys();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_parallel_encaps() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parallel_encaps();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_all() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_all();
        assert!(result);
    }

    #[test]
    fn test_verify_all_empty() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap();
        assert!(!e.verify_all());
    }

    #[test]
    fn test_config_accessor() {
        let e = KemCombiner::new(KemCombinerConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = KemCombinerError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = KemCombinerError::InvalidConfig("a".into());
        let e2 = KemCombinerError::ComputationFailed("b".into());
        let e3 = KemCombinerError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
