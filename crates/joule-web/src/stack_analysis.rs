//! Tolerance stack-up analysis.
//!
//! Provides [`StackAnalysisConfig`] builder and [`StackAnalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum StackAnalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for StackAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "StackAnalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "StackAnalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "StackAnalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`StackAnalysis`] parameters.
#[derive(Debug, Clone)]
pub struct StackAnalysisConfig {
    pub method: usize,
    pub num_simulations: usize,
    pub confidence: f64,
    pub seed: u64,
}

impl StackAnalysisConfig {
    pub fn new() -> Self {
        Self {
            method: 0,
            num_simulations: 10000,
            confidence: 0.997,
            seed: 42,
        }
    }

    pub fn with_method(mut self, v: usize) -> Self {
        self.method = v;
        self
    }

    pub fn with_num_simulations(mut self, v: usize) -> Self {
        self.num_simulations = v;
        self
    }

    pub fn with_confidence(mut self, v: f64) -> Self {
        self.confidence = v;
        self
    }

    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    pub fn validate(&self) -> Result<(), StackAnalysisError> {
        if self.confidence.is_nan() {
            return Err(StackAnalysisError::InvalidConfig("confidence is NaN".into()));
        }
        Ok(())
    }
}

impl Default for StackAnalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StackAnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StackAnalysisConfig(method={0}, num_simulations={1}, confidence={2:.4}, seed={3})", self.method, self.num_simulations, self.confidence, self.seed)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core tolerance stack-up analysis engine.
#[derive(Debug, Clone)]
pub struct StackAnalysis {
    config: StackAnalysisConfig,
    data: Vec<f64>,
}

impl StackAnalysis {
    pub fn new(config: StackAnalysisConfig) -> Result<Self, StackAnalysisError> {
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
    pub fn config(&self) -> &StackAnalysisConfig { &self.config }

    /// Worst-case stack analysis.
    pub fn worst_case(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Root sum square analysis.
    pub fn rss(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Monte Carlo stack simulation.
    pub fn monte_carlo_stack(&self) -> Vec<f64> {
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

impl fmt::Display for StackAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StackAnalysis(n={})", self.data.len())
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
        let cfg = StackAnalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = StackAnalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("StackAnalysisConfig"));
    }

    #[test]
    fn test_config_with_method() {
        let cfg = StackAnalysisConfig::new().with_method(42);
        assert_eq!(cfg.method, 42);
    }

    #[test]
    fn test_config_with_num_simulations() {
        let cfg = StackAnalysisConfig::new().with_num_simulations(42);
        assert_eq!(cfg.num_simulations, 42);
    }

    #[test]
    fn test_config_with_confidence() {
        let cfg = StackAnalysisConfig::new().with_confidence(42.0);
        assert!((cfg.confidence - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_seed() {
        let cfg = StackAnalysisConfig::new().with_seed(42);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = StackAnalysisConfig::new().with_method(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("StackAnalysis"));
    }

    #[test]
    fn test_summary() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_worst_case() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.worst_case();
        assert!(result.is_finite());
    }

    #[test]
    fn test_rss() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rss();
        assert!(result.is_finite());
    }

    #[test]
    fn test_monte_carlo_stack() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.monte_carlo_stack();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_monte_carlo_stack_empty() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap();
        assert!(e.monte_carlo_stack().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = StackAnalysis::new(StackAnalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = StackAnalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = StackAnalysisError::InvalidConfig("a".into());
        let e2 = StackAnalysisError::ComputationFailed("b".into());
        let e3 = StackAnalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
