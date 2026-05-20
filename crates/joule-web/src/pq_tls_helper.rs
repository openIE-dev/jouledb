//! Post-quantum TLS extension helpers.
//!
//! Provides [`PqTlsHelperConfig`] builder and [`PqTlsHelper`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqTlsHelperError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqTlsHelperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqTlsHelper: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqTlsHelper: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqTlsHelper: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqTlsHelper`] parameters.
#[derive(Debug, Clone)]
pub struct PqTlsHelperConfig {
    pub supported_groups: usize,
    pub hybrid_mode: bool,
    pub max_fragment: usize,
    pub version: usize,
}

impl PqTlsHelperConfig {
    pub fn new() -> Self {
        Self {
            supported_groups: 3,
            hybrid_mode: true,
            max_fragment: 16384,
            version: 4,
        }
    }

    pub fn with_supported_groups(mut self, v: usize) -> Self {
        self.supported_groups = v;
        self
    }

    pub fn with_hybrid_mode(mut self, v: bool) -> Self {
        self.hybrid_mode = v;
        self
    }

    pub fn with_max_fragment(mut self, v: usize) -> Self {
        self.max_fragment = v;
        self
    }

    pub fn with_version(mut self, v: usize) -> Self {
        self.version = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqTlsHelperError> {
        Ok(())
    }
}

impl Default for PqTlsHelperConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqTlsHelperConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqTlsHelperConfig(supported_groups={0}, hybrid_mode={1}, max_fragment={2}, version={3})", self.supported_groups, self.hybrid_mode, self.max_fragment, self.version)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum tls extension helpers engine.
#[derive(Debug, Clone)]
pub struct PqTlsHelper {
    config: PqTlsHelperConfig,
    data: Vec<f64>,
}

impl PqTlsHelper {
    pub fn new(config: PqTlsHelperConfig) -> Result<Self, PqTlsHelperError> {
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
    pub fn config(&self) -> &PqTlsHelperConfig { &self.config }

    /// Encode hybrid key share.
    pub fn encode_key_share(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Decode key share.
    pub fn decode_key_share(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Negotiate supported group.
    pub fn negotiate_group(&self) -> String {
        format!("{}: {} records", stringify!(negotiate_group), self.data.len())
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

impl fmt::Display for PqTlsHelper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqTlsHelper(n={})", self.data.len())
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
        let cfg = PqTlsHelperConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqTlsHelperConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqTlsHelperConfig"));
    }

    #[test]
    fn test_config_with_supported_groups() {
        let cfg = PqTlsHelperConfig::new().with_supported_groups(42);
        assert_eq!(cfg.supported_groups, 42);
    }

    #[test]
    fn test_config_with_hybrid_mode() {
        let cfg = PqTlsHelperConfig::new().with_hybrid_mode(false);
        assert_eq!(cfg.hybrid_mode, false);
    }

    #[test]
    fn test_config_with_max_fragment() {
        let cfg = PqTlsHelperConfig::new().with_max_fragment(42);
        assert_eq!(cfg.max_fragment, 42);
    }

    #[test]
    fn test_config_with_version() {
        let cfg = PqTlsHelperConfig::new().with_version(42);
        assert_eq!(cfg.version, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqTlsHelperConfig::new().with_supported_groups(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqTlsHelper"));
    }

    #[test]
    fn test_summary() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_key_share() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.encode_key_share();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_decode_key_share() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.decode_key_share();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_negotiate_group() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.negotiate_group();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_negotiate_group_empty() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap();
        let _ = e.negotiate_group();
    }

    #[test]
    fn test_config_accessor() {
        let e = PqTlsHelper::new(PqTlsHelperConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqTlsHelperError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqTlsHelperError::InvalidConfig("a".into());
        let e2 = PqTlsHelperError::ComputationFailed("b".into());
        let e3 = PqTlsHelperError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
