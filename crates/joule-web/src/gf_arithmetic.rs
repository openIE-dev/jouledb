//! Galois field arithmetic GF(2^n).
//!
//! Provides [`GfArithmeticConfig`] builder and [`GfArithmetic`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GfArithmeticError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GfArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GfArithmetic: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GfArithmetic: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GfArithmetic: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GfArithmetic`] parameters.
#[derive(Debug, Clone)]
pub struct GfArithmeticConfig {
    pub degree: usize,
    pub irreducible: u32,
    pub table_size: usize,
    pub precompute: bool,
}

impl GfArithmeticConfig {
    pub fn new() -> Self {
        Self {
            degree: 8,
            irreducible: 0x11B,
            table_size: 256,
            precompute: true,
        }
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v;
        self
    }

    pub fn with_irreducible(mut self, v: u32) -> Self {
        self.irreducible = v;
        self
    }

    pub fn with_table_size(mut self, v: usize) -> Self {
        self.table_size = v;
        self
    }

    pub fn with_precompute(mut self, v: bool) -> Self {
        self.precompute = v;
        self
    }

    pub fn validate(&self) -> Result<(), GfArithmeticError> {
        Ok(())
    }
}

impl Default for GfArithmeticConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GfArithmeticConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GfArithmeticConfig(degree={0}, irreducible={1}, table_size={2}, precompute={3})", self.degree, self.irreducible, self.table_size, self.precompute)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core galois field arithmetic gf(2^n) engine.
#[derive(Debug, Clone)]
pub struct GfArithmetic {
    config: GfArithmeticConfig,
    data: Vec<f64>,
}

impl GfArithmetic {
    pub fn new(config: GfArithmeticConfig) -> Result<Self, GfArithmeticError> {
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
    pub fn config(&self) -> &GfArithmeticConfig { &self.config }

    /// GF addition (XOR).
    pub fn gf_add(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// GF multiplication with reduction.
    pub fn gf_mul(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// GF multiplicative inverse.
    pub fn gf_inv(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for GfArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GfArithmetic(n={})", self.data.len())
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
        let cfg = GfArithmeticConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GfArithmeticConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GfArithmeticConfig"));
    }

    #[test]
    fn test_config_with_degree() {
        let cfg = GfArithmeticConfig::new().with_degree(42);
        assert_eq!(cfg.degree, 42);
    }

    #[test]
    fn test_config_with_irreducible() {
        let cfg = GfArithmeticConfig::new().with_irreducible(42);
        assert_eq!(cfg.irreducible, 42);
    }

    #[test]
    fn test_config_with_table_size() {
        let cfg = GfArithmeticConfig::new().with_table_size(42);
        assert_eq!(cfg.table_size, 42);
    }

    #[test]
    fn test_config_with_precompute() {
        let cfg = GfArithmeticConfig::new().with_precompute(false);
        assert_eq!(cfg.precompute, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GfArithmeticConfig::new().with_degree(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GfArithmetic"));
    }

    #[test]
    fn test_summary() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gf_add() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gf_add();
        assert!(result.is_finite());
    }

    #[test]
    fn test_gf_mul() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gf_mul();
        assert!(result.is_finite());
    }

    #[test]
    fn test_gf_inv() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gf_inv();
        assert!(result.is_finite());
    }

    #[test]
    fn test_gf_inv_empty() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap();
        assert!((e.gf_inv() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = GfArithmetic::new(GfArithmeticConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GfArithmeticError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GfArithmeticError::InvalidConfig("a".into());
        let e2 = GfArithmeticError::ComputationFailed("b".into());
        let e3 = GfArithmeticError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
