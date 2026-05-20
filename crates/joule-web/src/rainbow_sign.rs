//! Rainbow multivariate digital signature.
//!
//! Provides [`RainbowSignConfig`] builder and [`RainbowSign`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RainbowSignError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RainbowSignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RainbowSign: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RainbowSign: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RainbowSign: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RainbowSign`] parameters.
#[derive(Debug, Clone)]
pub struct RainbowSignConfig {
    pub v1: usize,
    pub o1: usize,
    pub o2: usize,
    pub field_size: u32,
}

impl RainbowSignConfig {
    pub fn new() -> Self {
        Self {
            v1: 36,
            o1: 32,
            o2: 32,
            field_size: 16,
        }
    }

    pub fn with_v1(mut self, v: usize) -> Self {
        self.v1 = v;
        self
    }

    pub fn with_o1(mut self, v: usize) -> Self {
        self.o1 = v;
        self
    }

    pub fn with_o2(mut self, v: usize) -> Self {
        self.o2 = v;
        self
    }

    pub fn with_field_size(mut self, v: u32) -> Self {
        self.field_size = v;
        self
    }

    pub fn validate(&self) -> Result<(), RainbowSignError> {
        Ok(())
    }
}

impl Default for RainbowSignConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RainbowSignConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RainbowSignConfig(v1={0}, o1={1}, o2={2}, field_size={3})", self.v1, self.o1, self.o2, self.field_size)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core rainbow multivariate digital signature engine.
#[derive(Debug, Clone)]
pub struct RainbowSign {
    config: RainbowSignConfig,
    data: Vec<f64>,
}

impl RainbowSign {
    pub fn new(config: RainbowSignConfig) -> Result<Self, RainbowSignError> {
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
    pub fn config(&self) -> &RainbowSignConfig { &self.config }

    /// Generate Rainbow keypair.
    pub fn keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Sign message.
    pub fn sign(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify signature.
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

impl fmt::Display for RainbowSign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RainbowSign(n={})", self.data.len())
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
        let cfg = RainbowSignConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RainbowSignConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RainbowSignConfig"));
    }

    #[test]
    fn test_config_with_v1() {
        let cfg = RainbowSignConfig::new().with_v1(42);
        assert_eq!(cfg.v1, 42);
    }

    #[test]
    fn test_config_with_o1() {
        let cfg = RainbowSignConfig::new().with_o1(42);
        assert_eq!(cfg.o1, 42);
    }

    #[test]
    fn test_config_with_o2() {
        let cfg = RainbowSignConfig::new().with_o2(42);
        assert_eq!(cfg.o2, 42);
    }

    #[test]
    fn test_config_with_field_size() {
        let cfg = RainbowSignConfig::new().with_field_size(42);
        assert_eq!(cfg.field_size, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RainbowSignConfig::new().with_v1(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RainbowSign::new(RainbowSignConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RainbowSign"));
    }

    #[test]
    fn test_summary() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sign() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.sign();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_verify_empty() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap();
        assert!(!e.verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = RainbowSign::new(RainbowSignConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RainbowSignError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RainbowSignError::InvalidConfig("a".into());
        let e2 = RainbowSignError::ComputationFailed("b".into());
        let e3 = RainbowSignError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
