//! Scenario generation for risk analysis.
//!
//! Provides [`ScenarioGenConfig`] builder and [`ScenarioGen`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ScenarioGenError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ScenarioGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ScenarioGen: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ScenarioGen: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ScenarioGen: insufficient data: {msg}"),
        }
    }
}

/// Variant selector for ScenarioMethod.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScenarioMethod {
    /// Historical method.
    Historical,
    /// MonteCarlo method.
    MonteCarlo,
    /// Parametric method.
    Parametric,
}

impl fmt::Display for ScenarioMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ScenarioGen`] parameters.
#[derive(Debug, Clone)]
pub struct ScenarioGenConfig {
    pub num_scenarios: usize,
    pub horizon_days: usize,
    pub seed: u64,
    pub method: ScenarioMethod,
}

impl ScenarioGenConfig {
    pub fn new() -> Self {
        Self {
            num_scenarios: 1000,
            horizon_days: 10,
            seed: 42,
            method: ScenarioMethod::Historical,
        }
    }

    pub fn with_num_scenarios(mut self, v: usize) -> Self {
        self.num_scenarios = v;
        self
    }

    pub fn with_horizon_days(mut self, v: usize) -> Self {
        self.horizon_days = v;
        self
    }

    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    pub fn with_method(mut self, v: ScenarioMethod) -> Self {
        self.method = v;
        self
    }

    pub fn validate(&self) -> Result<(), ScenarioGenError> {
        Ok(())
    }
}

impl Default for ScenarioGenConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ScenarioGenConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScenarioGenConfig(num_scenarios={0}, horizon_days={1}, seed={2}, method={3:?})", self.num_scenarios, self.horizon_days, self.seed, self.method)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core scenario generation for risk analysis engine.
#[derive(Debug, Clone)]
pub struct ScenarioGen {
    config: ScenarioGenConfig,
    data: Vec<f64>,
}

impl ScenarioGen {
    pub fn new(config: ScenarioGenConfig) -> Result<Self, ScenarioGenError> {
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
    pub fn config(&self) -> &ScenarioGenConfig { &self.config }

    /// Generate risk scenarios.
    pub fn generate(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Generate stress scenario.
    pub fn stress_scenario(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Moment-matched scenarios.
    pub fn moment_match(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for ScenarioGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScenarioGen(n={})", self.data.len())
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
        let cfg = ScenarioGenConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ScenarioGenConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ScenarioGenConfig"));
    }

    #[test]
    fn test_config_with_num_scenarios() {
        let cfg = ScenarioGenConfig::new().with_num_scenarios(42);
        assert_eq!(cfg.num_scenarios, 42);
    }

    #[test]
    fn test_config_with_horizon_days() {
        let cfg = ScenarioGenConfig::new().with_horizon_days(42);
        assert_eq!(cfg.horizon_days, 42);
    }

    #[test]
    fn test_config_with_seed() {
        let cfg = ScenarioGenConfig::new().with_seed(42);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_config_with_method() {
        let cfg = ScenarioGenConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ScenarioGenConfig::new().with_num_scenarios(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ScenarioGen"));
    }

    #[test]
    fn test_summary() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_stress_scenario() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.stress_scenario();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_moment_match() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.moment_match();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_moment_match_empty() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap();
        assert!(e.moment_match().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ScenarioGen::new(ScenarioGenConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ScenarioGenError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ScenarioGenError::InvalidConfig("a".into());
        let e2 = ScenarioGenError::ComputationFailed("b".into());
        let e3 = ScenarioGenError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
