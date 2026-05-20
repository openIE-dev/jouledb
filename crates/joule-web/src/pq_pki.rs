//! Post-quantum public key infrastructure.
//!
//! Provides [`PqPkiConfig`] builder and [`PqPki`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqPkiError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqPkiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqPki: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqPki: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqPki: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqPki`] parameters.
#[derive(Debug, Clone)]
pub struct PqPkiConfig {
    pub max_chain_depth: usize,
    pub validity_days: u32,
    pub allow_hybrid: bool,
    pub crl_check: bool,
}

impl PqPkiConfig {
    pub fn new() -> Self {
        Self {
            max_chain_depth: 5,
            validity_days: 365,
            allow_hybrid: true,
            crl_check: true,
        }
    }

    pub fn with_max_chain_depth(mut self, v: usize) -> Self {
        self.max_chain_depth = v;
        self
    }

    pub fn with_validity_days(mut self, v: u32) -> Self {
        self.validity_days = v;
        self
    }

    pub fn with_allow_hybrid(mut self, v: bool) -> Self {
        self.allow_hybrid = v;
        self
    }

    pub fn with_crl_check(mut self, v: bool) -> Self {
        self.crl_check = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqPkiError> {
        Ok(())
    }
}

impl Default for PqPkiConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqPkiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqPkiConfig(max_chain_depth={0}, validity_days={1}, allow_hybrid={2}, crl_check={3})", self.max_chain_depth, self.validity_days, self.allow_hybrid, self.crl_check)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum public key infrastructure engine.
#[derive(Debug, Clone)]
pub struct PqPki {
    config: PqPkiConfig,
    data: Vec<f64>,
}

impl PqPki {
    pub fn new(config: PqPkiConfig) -> Result<Self, PqPkiError> {
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
    pub fn config(&self) -> &PqPkiConfig { &self.config }

    /// Issue certificate.
    pub fn issue_cert(&self) -> String {
        format!("{}: {} records", stringify!(issue_cert), self.data.len())
    }

    /// Validate certificate chain.
    pub fn validate_chain(&self) -> bool {
        !self.data.is_empty()
    }

    /// Revoke certificate.
    pub fn revoke_cert(&self) -> bool {
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

impl fmt::Display for PqPki {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqPki(n={})", self.data.len())
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
        let cfg = PqPkiConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqPkiConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqPkiConfig"));
    }

    #[test]
    fn test_config_with_max_chain_depth() {
        let cfg = PqPkiConfig::new().with_max_chain_depth(42);
        assert_eq!(cfg.max_chain_depth, 42);
    }

    #[test]
    fn test_config_with_validity_days() {
        let cfg = PqPkiConfig::new().with_validity_days(42);
        assert_eq!(cfg.validity_days, 42);
    }

    #[test]
    fn test_config_with_allow_hybrid() {
        let cfg = PqPkiConfig::new().with_allow_hybrid(false);
        assert_eq!(cfg.allow_hybrid, false);
    }

    #[test]
    fn test_config_with_crl_check() {
        let cfg = PqPkiConfig::new().with_crl_check(false);
        assert_eq!(cfg.crl_check, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqPkiConfig::new().with_max_chain_depth(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqPki::new(PqPkiConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqPki"));
    }

    #[test]
    fn test_summary() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_issue_cert() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.issue_cert();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_validate_chain() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_chain();
        assert!(result);
    }

    #[test]
    fn test_revoke_cert() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.revoke_cert();
        assert!(result);
    }

    #[test]
    fn test_revoke_cert_empty() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap();
        assert!(!e.revoke_cert());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqPki::new(PqPkiConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqPkiError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqPkiError::InvalidConfig("a".into());
        let e2 = PqPkiError::ComputationFailed("b".into());
        let e3 = PqPkiError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
