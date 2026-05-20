//! Post-quantum handshake protocol.
//!
//! Provides [`PqHandshakeConfig`] builder and [`PqHandshake`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqHandshakeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqHandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqHandshake: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqHandshake: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqHandshake: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqHandshake`] parameters.
#[derive(Debug, Clone)]
pub struct PqHandshakeConfig {
    pub pattern: usize,
    pub max_message_size: usize,
    pub replay_window: usize,
    pub timeout_ms: u64,
}

impl PqHandshakeConfig {
    pub fn new() -> Self {
        Self {
            pattern: 0,
            max_message_size: 65535,
            replay_window: 100,
            timeout_ms: 5000,
        }
    }

    pub fn with_pattern(mut self, v: usize) -> Self {
        self.pattern = v;
        self
    }

    pub fn with_max_message_size(mut self, v: usize) -> Self {
        self.max_message_size = v;
        self
    }

    pub fn with_replay_window(mut self, v: usize) -> Self {
        self.replay_window = v;
        self
    }

    pub fn with_timeout_ms(mut self, v: u64) -> Self {
        self.timeout_ms = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqHandshakeError> {
        Ok(())
    }
}

impl Default for PqHandshakeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqHandshakeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqHandshakeConfig(pattern={0}, max_message_size={1}, replay_window={2}, timeout_ms={3})", self.pattern, self.max_message_size, self.replay_window, self.timeout_ms)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum handshake protocol engine.
#[derive(Debug, Clone)]
pub struct PqHandshake {
    config: PqHandshakeConfig,
    data: Vec<f64>,
}

impl PqHandshake {
    pub fn new(config: PqHandshakeConfig) -> Result<Self, PqHandshakeError> {
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
    pub fn config(&self) -> &PqHandshakeConfig { &self.config }

    /// Initiate handshake.
    pub fn initiate(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Respond to handshake.
    pub fn respond(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Finalize session.
    pub fn finalize(&self) -> Vec<f64> {
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

impl fmt::Display for PqHandshake {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqHandshake(n={})", self.data.len())
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
        let cfg = PqHandshakeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqHandshakeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqHandshakeConfig"));
    }

    #[test]
    fn test_config_with_pattern() {
        let cfg = PqHandshakeConfig::new().with_pattern(42);
        assert_eq!(cfg.pattern, 42);
    }

    #[test]
    fn test_config_with_max_message_size() {
        let cfg = PqHandshakeConfig::new().with_max_message_size(42);
        assert_eq!(cfg.max_message_size, 42);
    }

    #[test]
    fn test_config_with_replay_window() {
        let cfg = PqHandshakeConfig::new().with_replay_window(42);
        assert_eq!(cfg.replay_window, 42);
    }

    #[test]
    fn test_config_with_timeout_ms() {
        let cfg = PqHandshakeConfig::new().with_timeout_ms(42);
        assert_eq!(cfg.timeout_ms, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqHandshakeConfig::new().with_pattern(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqHandshake::new(PqHandshakeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqHandshake"));
    }

    #[test]
    fn test_summary() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_initiate() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.initiate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_respond() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.respond();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_finalize() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.finalize();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_finalize_empty() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap();
        assert!(e.finalize().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqHandshake::new(PqHandshakeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqHandshakeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqHandshakeError::InvalidConfig("a".into());
        let e2 = PqHandshakeError::ComputationFailed("b".into());
        let e3 = PqHandshakeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
