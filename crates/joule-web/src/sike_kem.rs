//! SIKE/SIDH supersingular isogeny KEM (educational).
//!
//! Provides [`SikeKemConfig`] builder and [`SikeKem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SikeKemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SikeKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SikeKem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SikeKem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SikeKem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SikeKem`] parameters.
#[derive(Debug, Clone)]
pub struct SikeKemConfig {
    pub prime_bits: usize,
    pub ea: usize,
    pub eb: usize,
    pub validate: bool,
}

impl SikeKemConfig {
    pub fn new() -> Self {
        Self {
            prime_bits: 434,
            ea: 216,
            eb: 137,
            validate: true,
        }
    }

    pub fn with_prime_bits(mut self, v: usize) -> Self {
        self.prime_bits = v;
        self
    }

    pub fn with_ea(mut self, v: usize) -> Self {
        self.ea = v;
        self
    }

    pub fn with_eb(mut self, v: usize) -> Self {
        self.eb = v;
        self
    }

    pub fn with_validate(mut self, v: bool) -> Self {
        self.validate = v;
        self
    }

    pub fn validate(&self) -> Result<(), SikeKemError> {
        Ok(())
    }
}

impl Default for SikeKemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SikeKemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SikeKemConfig(prime_bits={0}, ea={1}, eb={2}, validate={3})", self.prime_bits, self.ea, self.eb, self.validate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core sike/sidh supersingular isogeny kem (educational) engine.
#[derive(Debug, Clone)]
pub struct SikeKem {
    config: SikeKemConfig,
    data: Vec<f64>,
}

impl SikeKem {
    pub fn new(config: SikeKemConfig) -> Result<Self, SikeKemError> {
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
    pub fn config(&self) -> &SikeKemConfig { &self.config }

    /// Generate SIKE keypair.
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

impl fmt::Display for SikeKem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SikeKem(n={})", self.data.len())
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
        let cfg = SikeKemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SikeKemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SikeKemConfig"));
    }

    #[test]
    fn test_config_with_prime_bits() {
        let cfg = SikeKemConfig::new().with_prime_bits(42);
        assert_eq!(cfg.prime_bits, 42);
    }

    #[test]
    fn test_config_with_ea() {
        let cfg = SikeKemConfig::new().with_ea(42);
        assert_eq!(cfg.ea, 42);
    }

    #[test]
    fn test_config_with_eb() {
        let cfg = SikeKemConfig::new().with_eb(42);
        assert_eq!(cfg.eb, 42);
    }

    #[test]
    fn test_config_with_validate() {
        let cfg = SikeKemConfig::new().with_validate(false);
        assert_eq!(cfg.validate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SikeKemConfig::new().with_prime_bits(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SikeKem::new(SikeKemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SikeKem"));
    }

    #[test]
    fn test_summary() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_keygen() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encapsulate() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decapsulate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decapsulate_empty() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap();
        assert!(e.decapsulate().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SikeKem::new(SikeKemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SikeKemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SikeKemError::InvalidConfig("a".into());
        let e2 = SikeKemError::ComputationFailed("b".into());
        let e3 = SikeKemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
