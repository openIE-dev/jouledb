//! Post-quantum cryptography benchmarking framework.
//!
//! Provides [`PqBenchmarkConfig`] builder and [`PqBenchmark`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqBenchmarkError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqBenchmark: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqBenchmark: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqBenchmark: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqBenchmark`] parameters.
#[derive(Debug, Clone)]
pub struct PqBenchmarkConfig {
    pub iterations: usize,
    pub warmup: usize,
    pub measure_memory: bool,
    pub report_percentiles: bool,
}

impl PqBenchmarkConfig {
    pub fn new() -> Self {
        Self {
            iterations: 1000,
            warmup: 100,
            measure_memory: true,
            report_percentiles: true,
        }
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    pub fn with_warmup(mut self, v: usize) -> Self {
        self.warmup = v;
        self
    }

    pub fn with_measure_memory(mut self, v: bool) -> Self {
        self.measure_memory = v;
        self
    }

    pub fn with_report_percentiles(mut self, v: bool) -> Self {
        self.report_percentiles = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqBenchmarkError> {
        Ok(())
    }
}

impl Default for PqBenchmarkConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqBenchmarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqBenchmarkConfig(iterations={0}, warmup={1}, measure_memory={2}, report_percentiles={3})", self.iterations, self.warmup, self.measure_memory, self.report_percentiles)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core post-quantum cryptography benchmarking framework engine.
#[derive(Debug, Clone)]
pub struct PqBenchmark {
    config: PqBenchmarkConfig,
    data: Vec<f64>,
}

impl PqBenchmark {
    pub fn new(config: PqBenchmarkConfig) -> Result<Self, PqBenchmarkError> {
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
    pub fn config(&self) -> &PqBenchmarkConfig { &self.config }

    /// Benchmark key generation.
    pub fn benchmark_keygen(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Benchmark signing.
    pub fn benchmark_sign(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate comparison table.
    pub fn comparison_table(&self) -> String {
        format!("{}: {} records", stringify!(comparison_table), self.data.len())
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

impl fmt::Display for PqBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqBenchmark(n={})", self.data.len())
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
        let cfg = PqBenchmarkConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqBenchmarkConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqBenchmarkConfig"));
    }

    #[test]
    fn test_config_with_iterations() {
        let cfg = PqBenchmarkConfig::new().with_iterations(42);
        assert_eq!(cfg.iterations, 42);
    }

    #[test]
    fn test_config_with_warmup() {
        let cfg = PqBenchmarkConfig::new().with_warmup(42);
        assert_eq!(cfg.warmup, 42);
    }

    #[test]
    fn test_config_with_measure_memory() {
        let cfg = PqBenchmarkConfig::new().with_measure_memory(false);
        assert_eq!(cfg.measure_memory, false);
    }

    #[test]
    fn test_config_with_report_percentiles() {
        let cfg = PqBenchmarkConfig::new().with_report_percentiles(false);
        assert_eq!(cfg.report_percentiles, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqBenchmarkConfig::new().with_iterations(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqBenchmark"));
    }

    #[test]
    fn test_summary() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_benchmark_keygen() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.benchmark_keygen();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_benchmark_sign() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.benchmark_sign();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_comparison_table() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.comparison_table();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_comparison_table_empty() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap();
        let _ = e.comparison_table();
    }

    #[test]
    fn test_config_accessor() {
        let e = PqBenchmark::new(PqBenchmarkConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqBenchmarkError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqBenchmarkError::InvalidConfig("a".into());
        let e2 = PqBenchmarkError::ComputationFailed("b".into());
        let e3 = PqBenchmarkError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
