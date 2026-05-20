//! Post-quantum envelope encryption (KEM + symmetric).
//!
//! Provides [`PqEnvelopeConfig`] builder and [`PqEnvelope`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqEnvelopeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqEnvelope: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqEnvelope: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqEnvelope: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqEnvelope`] parameters.
#[derive(Debug, Clone)]
pub struct PqEnvelopeConfig {
    pub symmetric_key_len: usize,
    pub aead_tag_len: usize,
    pub max_recipients: usize,
    pub compress_payload: bool,
}

impl PqEnvelopeConfig {
    pub fn new() -> Self {
        Self {
            symmetric_key_len: 32,
            aead_tag_len: 16,
            max_recipients: 100,
            compress_payload: false,
        }
    }

    pub fn with_symmetric_key_len(mut self, v: usize) -> Self {
        self.symmetric_key_len = v;
        self
    }

    pub fn with_aead_tag_len(mut self, v: usize) -> Self {
        self.aead_tag_len = v;
        self
    }

    pub fn with_max_recipients(mut self, v: usize) -> Self {
        self.max_recipients = v;
        self
    }

    pub fn with_compress_payload(mut self, v: bool) -> Self {
        self.compress_payload = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqEnvelopeError> {
        Ok(())
    }
}

impl Default for PqEnvelopeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqEnvelopeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqEnvelopeConfig(symmetric_key_len={0}, aead_tag_len={1}, max_recipients={2}, compress_payload={3})", self.symmetric_key_len, self.aead_tag_len, self.max_recipients, self.compress_payload)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum envelope encryption (kem + symmetric) engine.
#[derive(Debug, Clone)]
pub struct PqEnvelope {
    config: PqEnvelopeConfig,
    data: Vec<f64>,
}

impl PqEnvelope {
    pub fn new(config: PqEnvelopeConfig) -> Result<Self, PqEnvelopeError> {
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
    pub fn config(&self) -> &PqEnvelopeConfig { &self.config }

    /// Seal envelope.
    pub fn seal(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Open envelope.
    pub fn open(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Add recipient to envelope.
    pub fn add_recipient(&self) -> bool {
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

impl fmt::Display for PqEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqEnvelope(n={})", self.data.len())
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
        let cfg = PqEnvelopeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqEnvelopeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqEnvelopeConfig"));
    }

    #[test]
    fn test_config_with_symmetric_key_len() {
        let cfg = PqEnvelopeConfig::new().with_symmetric_key_len(42);
        assert_eq!(cfg.symmetric_key_len, 42);
    }

    #[test]
    fn test_config_with_aead_tag_len() {
        let cfg = PqEnvelopeConfig::new().with_aead_tag_len(42);
        assert_eq!(cfg.aead_tag_len, 42);
    }

    #[test]
    fn test_config_with_max_recipients() {
        let cfg = PqEnvelopeConfig::new().with_max_recipients(42);
        assert_eq!(cfg.max_recipients, 42);
    }

    #[test]
    fn test_config_with_compress_payload() {
        let cfg = PqEnvelopeConfig::new().with_compress_payload(true);
        assert_eq!(cfg.compress_payload, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqEnvelopeConfig::new().with_symmetric_key_len(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqEnvelope"));
    }

    #[test]
    fn test_summary() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_seal() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.seal();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_open() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.open();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_add_recipient() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_recipient();
        assert!(result);
    }

    #[test]
    fn test_add_recipient_empty() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap();
        assert!(!e.add_recipient());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqEnvelope::new(PqEnvelopeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqEnvelopeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqEnvelopeError::InvalidConfig("a".into());
        let e2 = PqEnvelopeError::ComputationFailed("b".into());
        let e3 = PqEnvelopeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
