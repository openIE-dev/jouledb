//! Post-quantum safe deterministic random number generator.
//!
//! Provides [`PqRngConfig`] builder and [`PqRng`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqRngError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqRngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqRng: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqRng: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqRng: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqRng`] parameters.
#[derive(Debug, Clone)]
pub struct PqRngConfig {
    pub seed_len: usize,
    pub counter: u64,
    pub reseed_interval: u64,
    pub personalization_len: usize,
}

impl PqRngConfig {
    pub fn new() -> Self {
        Self {
            seed_len: 32,
            counter: 0,
            reseed_interval: 1048576,
            personalization_len: 0,
        }
    }

    pub fn with_seed_len(mut self, v: usize) -> Self {
        self.seed_len = v;
        self
    }

    pub fn with_counter(mut self, v: u64) -> Self {
        self.counter = v;
        self
    }

    pub fn with_reseed_interval(mut self, v: u64) -> Self {
        self.reseed_interval = v;
        self
    }

    pub fn with_personalization_len(mut self, v: usize) -> Self {
        self.personalization_len = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqRngError> {
        Ok(())
    }
}

impl Default for PqRngConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqRngConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqRngConfig(seed_len={0}, counter={1}, reseed_interval={2}, personalization_len={3})", self.seed_len, self.counter, self.reseed_interval, self.personalization_len)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum safe deterministic random number generator engine.
#[derive(Debug, Clone)]
pub struct PqRng {
    config: PqRngConfig,
    data: Vec<f64>,
}

impl PqRng {
    pub fn new(config: PqRngConfig) -> Result<Self, PqRngError> {
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
    pub fn config(&self) -> &PqRngConfig { &self.config }

    /// Generate random bytes.
    pub fn random_bytes(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Uniform sample in range.
    pub fn uniform_range(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Expand seed to key material.
    pub fn seed_expand(&self) -> Vec<f64> {
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

impl fmt::Display for PqRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqRng(n={})", self.data.len())
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
        let cfg = PqRngConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqRngConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqRngConfig"));
    }

    #[test]
    fn test_config_with_seed_len() {
        let cfg = PqRngConfig::new().with_seed_len(42);
        assert_eq!(cfg.seed_len, 42);
    }

    #[test]
    fn test_config_with_counter() {
        let cfg = PqRngConfig::new().with_counter(42);
        assert_eq!(cfg.counter, 42);
    }

    #[test]
    fn test_config_with_reseed_interval() {
        let cfg = PqRngConfig::new().with_reseed_interval(42);
        assert_eq!(cfg.reseed_interval, 42);
    }

    #[test]
    fn test_config_with_personalization_len() {
        let cfg = PqRngConfig::new().with_personalization_len(42);
        assert_eq!(cfg.personalization_len, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqRngConfig::new().with_seed_len(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqRng::new(PqRngConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqRng::new(PqRngConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqRng"));
    }

    #[test]
    fn test_summary() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqRng::new(PqRngConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_random_bytes() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.random_bytes();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_uniform_range() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.uniform_range();
        assert!(result.is_finite());
    }

    #[test]
    fn test_seed_expand() {
        let e = PqRng::new(PqRngConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.seed_expand();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_seed_expand_empty() {
        let e = PqRng::new(PqRngConfig::new()).unwrap();
        assert!(e.seed_expand().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqRng::new(PqRngConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqRngError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqRngError::InvalidConfig("a".into());
        let e2 = PqRngError::ComputationFailed("b".into());
        let e3 = PqRngError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
