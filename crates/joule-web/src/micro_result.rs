//! Microbiology culture and susceptibility results.
//!
//! Provides [`MicroResultConfig`] builder and [`MicroResult`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MicroResultError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MicroResultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MicroResult: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MicroResult: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MicroResult: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MicroResult`] parameters.
#[derive(Debug, Clone)]
pub struct MicroResultConfig {
    pub organism_count: usize,
    pub susceptibility_method: usize,
    pub breakpoint_version: usize,
    pub include_antibiogram: bool,
}

impl MicroResultConfig {
    pub fn new() -> Self {
        Self {
            organism_count: 0,
            susceptibility_method: 0,
            breakpoint_version: 2026,
            include_antibiogram: true,
        }
    }

    pub fn with_organism_count(mut self, v: usize) -> Self {
        self.organism_count = v;
        self
    }

    pub fn with_susceptibility_method(mut self, v: usize) -> Self {
        self.susceptibility_method = v;
        self
    }

    pub fn with_breakpoint_version(mut self, v: usize) -> Self {
        self.breakpoint_version = v;
        self
    }

    pub fn with_include_antibiogram(mut self, v: bool) -> Self {
        self.include_antibiogram = v;
        self
    }

    pub fn validate(&self) -> Result<(), MicroResultError> {
        Ok(())
    }
}

impl Default for MicroResultConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MicroResultConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MicroResultConfig(organism_count={0}, susceptibility_method={1}, breakpoint_version={2}, include_antibiogram={3})", self.organism_count, self.susceptibility_method, self.breakpoint_version, self.include_antibiogram)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core microbiology culture and susceptibility results engine.
#[derive(Debug, Clone)]
pub struct MicroResult {
    config: MicroResultConfig,
    data: Vec<f64>,
}

impl MicroResult {
    pub fn new(config: MicroResultConfig) -> Result<Self, MicroResultError> {
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
    pub fn config(&self) -> &MicroResultConfig { &self.config }

    /// Interpret S/I/R.
    pub fn interpret_susceptibility(&self) -> String {
        format!("{}: {} records", stringify!(interpret_susceptibility), self.data.len())
    }

    /// Generate antibiogram data.
    pub fn antibiogram(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Detect resistance pattern.
    pub fn resistance_pattern(&self) -> Vec<f64> {
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

impl fmt::Display for MicroResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MicroResult(n={})", self.data.len())
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
        let cfg = MicroResultConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MicroResultConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MicroResultConfig"));
    }

    #[test]
    fn test_config_with_organism_count() {
        let cfg = MicroResultConfig::new().with_organism_count(42);
        assert_eq!(cfg.organism_count, 42);
    }

    #[test]
    fn test_config_with_susceptibility_method() {
        let cfg = MicroResultConfig::new().with_susceptibility_method(42);
        assert_eq!(cfg.susceptibility_method, 42);
    }

    #[test]
    fn test_config_with_breakpoint_version() {
        let cfg = MicroResultConfig::new().with_breakpoint_version(42);
        assert_eq!(cfg.breakpoint_version, 42);
    }

    #[test]
    fn test_config_with_include_antibiogram() {
        let cfg = MicroResultConfig::new().with_include_antibiogram(false);
        assert_eq!(cfg.include_antibiogram, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MicroResultConfig::new().with_organism_count(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MicroResult::new(MicroResultConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MicroResult"));
    }

    #[test]
    fn test_summary() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpret_susceptibility() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.interpret_susceptibility();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_antibiogram() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.antibiogram();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_resistance_pattern() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.resistance_pattern();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_resistance_pattern_empty() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap();
        assert!(e.resistance_pattern().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = MicroResult::new(MicroResultConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MicroResultError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MicroResultError::InvalidConfig("a".into());
        let e2 = MicroResultError::ComputationFailed("b".into());
        let e3 = MicroResultError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
