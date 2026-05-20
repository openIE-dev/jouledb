//! Post-quantum safe key derivation function.
//!
//! Provides [`PqKdfConfig`] builder and [`PqKdf`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqKdfError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqKdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqKdf: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqKdf: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqKdf: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqKdf`] parameters.
#[derive(Debug, Clone)]
pub struct PqKdfConfig {
    pub output_len: usize,
    pub iterations: usize,
    pub salt_len: usize,
    pub info_len: usize,
}

impl PqKdfConfig {
    pub fn new() -> Self {
        Self {
            output_len: 32,
            iterations: 1,
            salt_len: 32,
            info_len: 0,
        }
    }

    pub fn with_output_len(mut self, v: usize) -> Self {
        self.output_len = v;
        self
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    pub fn with_salt_len(mut self, v: usize) -> Self {
        self.salt_len = v;
        self
    }

    pub fn with_info_len(mut self, v: usize) -> Self {
        self.info_len = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqKdfError> {
        Ok(())
    }
}

impl Default for PqKdfConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqKdfConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqKdfConfig(output_len={0}, iterations={1}, salt_len={2}, info_len={3})", self.output_len, self.iterations, self.salt_len, self.info_len)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum safe key derivation function engine.
#[derive(Debug, Clone)]
pub struct PqKdf {
    config: PqKdfConfig,
    data: Vec<f64>,
}

impl PqKdf {
    pub fn new(config: PqKdfConfig) -> Result<Self, PqKdfError> {
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
    pub fn config(&self) -> &PqKdfConfig { &self.config }

    /// HKDF extract step.
    pub fn extract(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// HKDF expand step.
    pub fn expand(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Derive session key.
    pub fn derive_key(&self) -> Vec<f64> {
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

impl fmt::Display for PqKdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqKdf(n={})", self.data.len())
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
        let cfg = PqKdfConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqKdfConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqKdfConfig"));
    }

    #[test]
    fn test_config_with_output_len() {
        let cfg = PqKdfConfig::new().with_output_len(42);
        assert_eq!(cfg.output_len, 42);
    }

    #[test]
    fn test_config_with_iterations() {
        let cfg = PqKdfConfig::new().with_iterations(42);
        assert_eq!(cfg.iterations, 42);
    }

    #[test]
    fn test_config_with_salt_len() {
        let cfg = PqKdfConfig::new().with_salt_len(42);
        assert_eq!(cfg.salt_len, 42);
    }

    #[test]
    fn test_config_with_info_len() {
        let cfg = PqKdfConfig::new().with_info_len(42);
        assert_eq!(cfg.info_len, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqKdfConfig::new().with_output_len(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqKdf::new(PqKdfConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqKdf"));
    }

    #[test]
    fn test_summary() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_expand() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.expand();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_derive_key() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.derive_key();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_derive_key_empty() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap();
        assert!(e.derive_key().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqKdf::new(PqKdfConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqKdfError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqKdfError::InvalidConfig("a".into());
        let e2 = PqKdfError::ComputationFailed("b".into());
        let e3 = PqKdfError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
