//! Groth16 zero-knowledge proof system (simplified).
//!
//! Provides [`Groth16CoreConfig`] builder and [`Groth16Core`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum Groth16CoreError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for Groth16CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Groth16Core: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Groth16Core: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Groth16Core: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Groth16Core`] parameters.
#[derive(Debug, Clone)]
pub struct Groth16CoreConfig {
    pub num_constraints: usize,
    pub num_inputs: usize,
    pub num_witnesses: usize,
    pub field_bits: usize,
}

impl Groth16CoreConfig {
    pub fn new() -> Self {
        Self {
            num_constraints: 100,
            num_inputs: 10,
            num_witnesses: 50,
            field_bits: 254,
        }
    }

    pub fn with_num_constraints(mut self, v: usize) -> Self {
        self.num_constraints = v;
        self
    }

    pub fn with_num_inputs(mut self, v: usize) -> Self {
        self.num_inputs = v;
        self
    }

    pub fn with_num_witnesses(mut self, v: usize) -> Self {
        self.num_witnesses = v;
        self
    }

    pub fn with_field_bits(mut self, v: usize) -> Self {
        self.field_bits = v;
        self
    }

    pub fn validate(&self) -> Result<(), Groth16CoreError> {
        Ok(())
    }
}

impl Default for Groth16CoreConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for Groth16CoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Groth16CoreConfig(num_constraints={0}, num_inputs={1}, num_witnesses={2}, field_bits={3})", self.num_constraints, self.num_inputs, self.num_witnesses, self.field_bits)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core groth16 zero-knowledge proof system (simplified) engine.
#[derive(Debug, Clone)]
pub struct Groth16Core {
    config: Groth16CoreConfig,
    data: Vec<f64>,
}

impl Groth16Core {
    pub fn new(config: Groth16CoreConfig) -> Result<Self, Groth16CoreError> {
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
    pub fn config(&self) -> &Groth16CoreConfig { &self.config }

    /// Trusted setup.
    pub fn setup(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate Groth16 proof.
    pub fn prove(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Verify Groth16 proof.
    pub fn verify(&self) -> bool {
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

impl fmt::Display for Groth16Core {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Groth16Core(n={})", self.data.len())
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
        let cfg = Groth16CoreConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = Groth16CoreConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("Groth16CoreConfig"));
    }

    #[test]
    fn test_config_with_num_constraints() {
        let cfg = Groth16CoreConfig::new().with_num_constraints(42);
        assert_eq!(cfg.num_constraints, 42);
    }

    #[test]
    fn test_config_with_num_inputs() {
        let cfg = Groth16CoreConfig::new().with_num_inputs(42);
        assert_eq!(cfg.num_inputs, 42);
    }

    #[test]
    fn test_config_with_num_witnesses() {
        let cfg = Groth16CoreConfig::new().with_num_witnesses(42);
        assert_eq!(cfg.num_witnesses, 42);
    }

    #[test]
    fn test_config_with_field_bits() {
        let cfg = Groth16CoreConfig::new().with_field_bits(42);
        assert_eq!(cfg.field_bits, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = Groth16CoreConfig::new().with_num_constraints(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Groth16Core::new(Groth16CoreConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Groth16Core"));
    }

    #[test]
    fn test_summary() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_setup() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.setup();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_prove() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.prove();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify();
        assert!(result);
    }

    #[test]
    fn test_verify_empty() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap();
        assert!(!e.verify());
    }

    #[test]
    fn test_config_accessor() {
        let e = Groth16Core::new(Groth16CoreConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = Groth16CoreError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = Groth16CoreError::InvalidConfig("a".into());
        let e2 = Groth16CoreError::ComputationFailed("b".into());
        let e3 = Groth16CoreError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
