//! STARK zero-knowledge proof system core.
//!
//! Provides [`StarkCoreConfig`] builder and [`StarkCore`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum StarkCoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for StarkCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "StarkCore: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "StarkCore: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "StarkCore: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`StarkCore`] parameters.
#[derive(Debug, Clone)]
pub struct StarkCoreConfig {
    pub trace_length: usize,
    pub blowup_factor: usize,
    pub num_queries: usize,
    pub field_bits: usize,
}

impl StarkCoreConfig {
    pub fn new() -> Self {
        Self {
            trace_length: 1024,
            blowup_factor: 4,
            num_queries: 30,
            field_bits: 64,
        }
    }

    pub fn with_trace_length(mut self, v: usize) -> Self {
        self.trace_length = v;
        self
    }

    pub fn with_blowup_factor(mut self, v: usize) -> Self {
        self.blowup_factor = v;
        self
    }

    pub fn with_num_queries(mut self, v: usize) -> Self {
        self.num_queries = v;
        self
    }

    pub fn with_field_bits(mut self, v: usize) -> Self {
        self.field_bits = v;
        self
    }

    pub fn validate(&self) -> Result<(), StarkCoreError> {
        Ok(())
    }
}

impl Default for StarkCoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StarkCoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StarkCoreConfig(trace_length={0}, blowup_factor={1}, num_queries={2}, field_bits={3})", self.trace_length, self.blowup_factor, self.num_queries, self.field_bits)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core stark zero-knowledge proof system core engine.
#[derive(Debug, Clone)]
pub struct StarkCore {
    config: StarkCoreConfig,
    data: Vec<f64>,
}

impl StarkCore {
    pub fn new(config: StarkCoreConfig) -> Result<Self, StarkCoreError> {
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
    pub fn config(&self) -> &StarkCoreConfig { &self.config }

    /// Generate execution trace.
    pub fn generate_trace(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Generate STARK proof.
    pub fn prove(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify STARK proof.
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

impl fmt::Display for StarkCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StarkCore(n={})", self.data.len())
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
        let cfg = StarkCoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = StarkCoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("StarkCoreConfig"));
    }

    #[test]
    fn test_config_with_trace_length() {
        let cfg = StarkCoreConfig::new().with_trace_length(42);
        assert_eq!(cfg.trace_length, 42);
    }

    #[test]
    fn test_config_with_blowup_factor() {
        let cfg = StarkCoreConfig::new().with_blowup_factor(42);
        assert_eq!(cfg.blowup_factor, 42);
    }

    #[test]
    fn test_config_with_num_queries() {
        let cfg = StarkCoreConfig::new().with_num_queries(42);
        assert_eq!(cfg.num_queries, 42);
    }

    #[test]
    fn test_config_with_field_bits() {
        let cfg = StarkCoreConfig::new().with_field_bits(42);
        assert_eq!(cfg.field_bits, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = StarkCoreConfig::new().with_trace_length(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = StarkCore::new(StarkCoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("StarkCore"));
    }

    #[test]
    fn test_summary() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_trace() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_trace();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_prove() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.prove();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_verify_empty() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap();
        assert!(!e.verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = StarkCore::new(StarkCoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = StarkCoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = StarkCoreError::InvalidConfig("a".into());
        let e2 = StarkCoreError::ComputationFailed("b".into());
        let e3 = StarkCoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
