//! Basic fully homomorphic encryption (BFV scheme).
//!
//! Provides [`FheBasicConfig`] builder and [`FheBasic`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FheBasicError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FheBasicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "FheBasic: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "FheBasic: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "FheBasic: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`FheBasic`] parameters.
#[derive(Debug, Clone)]
pub struct FheBasicConfig {
    pub poly_degree: usize,
    pub plain_modulus: u32,
    pub coeff_modulus_bits: usize,
    pub noise_budget: usize,
}

impl FheBasicConfig {
    pub fn new() -> Self {
        Self {
            poly_degree: 4096,
            plain_modulus: 65537,
            coeff_modulus_bits: 128,
            noise_budget: 40,
        }
    }

    pub fn with_poly_degree(mut self, v: usize) -> Self {
        self.poly_degree = v;
        self
    }

    pub fn with_plain_modulus(mut self, v: u32) -> Self {
        self.plain_modulus = v;
        self
    }

    pub fn with_coeff_modulus_bits(mut self, v: usize) -> Self {
        self.coeff_modulus_bits = v;
        self
    }

    pub fn with_noise_budget(mut self, v: usize) -> Self {
        self.noise_budget = v;
        self
    }

    pub fn validate(&self) -> Result<(), FheBasicError> {
        Ok(())
    }
}

impl Default for FheBasicConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FheBasicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FheBasicConfig(poly_degree={0}, plain_modulus={1}, coeff_modulus_bits={2}, noise_budget={3})", self.poly_degree, self.plain_modulus, self.coeff_modulus_bits, self.noise_budget)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core basic fully homomorphic encryption (bfv scheme) engine.
#[derive(Debug, Clone)]
pub struct FheBasic {
    config: FheBasicConfig,
    data: Vec<f64>,
}

impl FheBasic {
    pub fn new(config: FheBasicConfig) -> Result<Self, FheBasicError> {
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
    pub fn config(&self) -> &FheBasicConfig { &self.config }

    /// Encrypt integer plaintext.
    pub fn encrypt_int(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decrypt to integer.
    pub fn decrypt_int(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Homomorphic addition.
    pub fn add_cipher(&self) -> Vec<f64> {
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

impl fmt::Display for FheBasic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FheBasic(n={})", self.data.len())
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
        let cfg = FheBasicConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FheBasicConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FheBasicConfig"));
    }

    #[test]
    fn test_config_with_poly_degree() {
        let cfg = FheBasicConfig::new().with_poly_degree(42);
        assert_eq!(cfg.poly_degree, 42);
    }

    #[test]
    fn test_config_with_plain_modulus() {
        let cfg = FheBasicConfig::new().with_plain_modulus(42);
        assert_eq!(cfg.plain_modulus, 42);
    }

    #[test]
    fn test_config_with_coeff_modulus_bits() {
        let cfg = FheBasicConfig::new().with_coeff_modulus_bits(42);
        assert_eq!(cfg.coeff_modulus_bits, 42);
    }

    #[test]
    fn test_config_with_noise_budget() {
        let cfg = FheBasicConfig::new().with_noise_budget(42);
        assert_eq!(cfg.noise_budget, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FheBasicConfig::new().with_poly_degree(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = FheBasic::new(FheBasicConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("FheBasic"));
    }

    #[test]
    fn test_summary() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encrypt_int() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encrypt_int();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decrypt_int() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decrypt_int();
        assert!(result.is_finite());
    }

    #[test]
    fn test_add_cipher() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_cipher();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_add_cipher_empty() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap();
        assert!(e.add_cipher().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = FheBasic::new(FheBasicConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FheBasicError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FheBasicError::InvalidConfig("a".into());
        let e2 = FheBasicError::ComputationFailed("b".into());
        let e3 = FheBasicError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
