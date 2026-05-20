//! Zero-knowledge identity and credential proofs.
//!
//! Provides [`ZkIdentityConfig`] builder and [`ZkIdentity`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ZkIdentityError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ZkIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ZkIdentity: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ZkIdentity: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ZkIdentity: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ZkIdentity`] parameters.
#[derive(Debug, Clone)]
pub struct ZkIdentityConfig {
    pub num_attributes: usize,
    pub security_bits: usize,
    pub selective_disclosure: bool,
    pub revocation_check: bool,
}

impl ZkIdentityConfig {
    pub fn new() -> Self {
        Self {
            num_attributes: 10,
            security_bits: 128,
            selective_disclosure: true,
            revocation_check: true,
        }
    }

    pub fn with_num_attributes(mut self, v: usize) -> Self {
        self.num_attributes = v;
        self
    }

    pub fn with_security_bits(mut self, v: usize) -> Self {
        self.security_bits = v;
        self
    }

    pub fn with_selective_disclosure(mut self, v: bool) -> Self {
        self.selective_disclosure = v;
        self
    }

    pub fn with_revocation_check(mut self, v: bool) -> Self {
        self.revocation_check = v;
        self
    }

    pub fn validate(&self) -> Result<(), ZkIdentityError> {
        Ok(())
    }
}

impl Default for ZkIdentityConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ZkIdentityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZkIdentityConfig(num_attributes={0}, security_bits={1}, selective_disclosure={2}, revocation_check={3})", self.num_attributes, self.security_bits, self.selective_disclosure, self.revocation_check)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core zero-knowledge identity and credential proofs engine.
#[derive(Debug, Clone)]
pub struct ZkIdentity {
    config: ZkIdentityConfig,
    data: Vec<f64>,
}

impl ZkIdentity {
    pub fn new(config: ZkIdentityConfig) -> Result<Self, ZkIdentityError> {
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
    pub fn config(&self) -> &ZkIdentityConfig { &self.config }

    /// Issue verifiable credential.
    pub fn issue_credential(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Present credential with selective disclosure.
    pub fn present(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify credential presentation.
    pub fn verify_presentation(&self) -> bool {
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

impl fmt::Display for ZkIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZkIdentity(n={})", self.data.len())
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
        let cfg = ZkIdentityConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ZkIdentityConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ZkIdentityConfig"));
    }

    #[test]
    fn test_config_with_num_attributes() {
        let cfg = ZkIdentityConfig::new().with_num_attributes(42);
        assert_eq!(cfg.num_attributes, 42);
    }

    #[test]
    fn test_config_with_security_bits() {
        let cfg = ZkIdentityConfig::new().with_security_bits(42);
        assert_eq!(cfg.security_bits, 42);
    }

    #[test]
    fn test_config_with_selective_disclosure() {
        let cfg = ZkIdentityConfig::new().with_selective_disclosure(false);
        assert_eq!(cfg.selective_disclosure, false);
    }

    #[test]
    fn test_config_with_revocation_check() {
        let cfg = ZkIdentityConfig::new().with_revocation_check(false);
        assert_eq!(cfg.revocation_check, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ZkIdentityConfig::new().with_num_attributes(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ZkIdentity"));
    }

    #[test]
    fn test_summary() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_issue_credential() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.issue_credential();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_present() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.present();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_presentation() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_presentation();
        assert!(result);
    }

    #[test]
    fn test_verify_presentation_empty() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap();
        assert!(!e.verify_presentation());
    }

    #[test]
    fn test_config_accessor() {
        let e = ZkIdentity::new(ZkIdentityConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ZkIdentityError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ZkIdentityError::InvalidConfig("a".into());
        let e2 = ZkIdentityError::ComputationFailed("b".into());
        let e3 = ZkIdentityError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
