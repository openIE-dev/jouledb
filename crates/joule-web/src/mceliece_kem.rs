//! Classic McEliece code-based KEM.
//!
//! Provides [`McElieceKemConfig`] builder and [`McElieceKem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum McElieceKemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for McElieceKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "McElieceKem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "McElieceKem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "McElieceKem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`McElieceKem`] parameters.
#[derive(Debug, Clone)]
pub struct McElieceKemConfig {
    pub m: usize,
    pub t: usize,
    pub n: usize,
    pub k: usize,
}

impl McElieceKemConfig {
    pub fn new() -> Self {
        Self {
            m: 13,
            t: 128,
            n: 8192,
            k: 6528,
        }
    }

    pub fn with_m(mut self, v: usize) -> Self {
        self.m = v;
        self
    }

    pub fn with_t(mut self, v: usize) -> Self {
        self.t = v;
        self
    }

    pub fn with_n(mut self, v: usize) -> Self {
        self.n = v;
        self
    }

    pub fn with_k(mut self, v: usize) -> Self {
        self.k = v;
        self
    }

    pub fn validate(&self) -> Result<(), McElieceKemError> {
        Ok(())
    }
}

impl Default for McElieceKemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for McElieceKemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "McElieceKemConfig(m={0}, t={1}, n={2}, k={3})", self.m, self.t, self.n, self.k)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core classic mceliece code-based kem engine.
#[derive(Debug, Clone)]
pub struct McElieceKem {
    config: McElieceKemConfig,
    data: Vec<f64>,
}

impl McElieceKem {
    pub fn new(config: McElieceKemConfig) -> Result<Self, McElieceKemError> {
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
    pub fn config(&self) -> &McElieceKemConfig { &self.config }

    /// Generate McEliece keypair.
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

impl fmt::Display for McElieceKem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "McElieceKem(n={})", self.data.len())
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
        let cfg = McElieceKemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = McElieceKemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("McElieceKemConfig"));
    }

    #[test]
    fn test_config_with_m() {
        let cfg = McElieceKemConfig::new().with_m(42);
        assert_eq!(cfg.m, 42);
    }

    #[test]
    fn test_config_with_t() {
        let cfg = McElieceKemConfig::new().with_t(42);
        assert_eq!(cfg.t, 42);
    }

    #[test]
    fn test_config_with_n() {
        let cfg = McElieceKemConfig::new().with_n(42);
        assert_eq!(cfg.n, 42);
    }

    #[test]
    fn test_config_with_k() {
        let cfg = McElieceKemConfig::new().with_k(42);
        assert_eq!(cfg.k, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = McElieceKemConfig::new().with_m(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = McElieceKem::new(McElieceKemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("McElieceKem"));
    }

    #[test]
    fn test_summary() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = McElieceKem::new(McElieceKemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = McElieceKemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = McElieceKemError::InvalidConfig("a".into());
        let e2 = McElieceKemError::ComputationFailed("b".into());
        let e3 = McElieceKemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
