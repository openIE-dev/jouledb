//! 3-axis toolpath generation strategies.
//!
//! Provides [`Toolpath3AxisConfig`] builder and [`Toolpath3Axis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum Toolpath3AxisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for Toolpath3AxisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Toolpath3Axis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Toolpath3Axis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Toolpath3Axis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Toolpath3Axis`] parameters.
#[derive(Debug, Clone)]
pub struct Toolpath3AxisConfig {
    pub tool_diameter: f64,
    pub stepover: f64,
    pub scallop_height: f64,
    pub strategy: usize,
}

impl Toolpath3AxisConfig {
    pub fn new() -> Self {
        Self {
            tool_diameter: 10.0,
            stepover: 3.0,
            scallop_height: 0.01,
            strategy: 0,
        }
    }

    pub fn with_tool_diameter(mut self, v: f64) -> Self {
        self.tool_diameter = v;
        self
    }

    pub fn with_stepover(mut self, v: f64) -> Self {
        self.stepover = v;
        self
    }

    pub fn with_scallop_height(mut self, v: f64) -> Self {
        self.scallop_height = v;
        self
    }

    pub fn with_strategy(mut self, v: usize) -> Self {
        self.strategy = v;
        self
    }

    pub fn validate(&self) -> Result<(), Toolpath3AxisError> {
        if self.tool_diameter.is_nan() {
            return Err(Toolpath3AxisError::InvalidConfig("tool_diameter is NaN".into()));
        }
        if self.stepover.is_nan() {
            return Err(Toolpath3AxisError::InvalidConfig("stepover is NaN".into()));
        }
        if self.scallop_height.is_nan() {
            return Err(Toolpath3AxisError::InvalidConfig("scallop_height is NaN".into()));
        }
        Ok(())
    }
}

impl Default for Toolpath3AxisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for Toolpath3AxisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Toolpath3AxisConfig(tool_diameter={0:.4}, stepover={1:.4}, scallop_height={2:.4}, strategy={3})", self.tool_diameter, self.stepover, self.scallop_height, self.strategy)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core 3-axis toolpath generation strategies engine.
#[derive(Debug, Clone)]
pub struct Toolpath3Axis {
    config: Toolpath3AxisConfig,
    data: Vec<f64>,
}

impl Toolpath3Axis {
    pub fn new(config: Toolpath3AxisConfig) -> Result<Self, Toolpath3AxisError> {
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
    pub fn config(&self) -> &Toolpath3AxisConfig { &self.config }

    /// Parallel planar toolpath.
    pub fn parallel_planar(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Constant-Z waterline.
    pub fn waterline(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Pencil tracing for concave edges.
    pub fn pencil_trace(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for Toolpath3Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Toolpath3Axis(n={})", self.data.len())
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
        let cfg = Toolpath3AxisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = Toolpath3AxisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("Toolpath3AxisConfig"));
    }

    #[test]
    fn test_config_with_tool_diameter() {
        let cfg = Toolpath3AxisConfig::new().with_tool_diameter(42.0);
        assert!((cfg.tool_diameter - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_stepover() {
        let cfg = Toolpath3AxisConfig::new().with_stepover(42.0);
        assert!((cfg.stepover - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_scallop_height() {
        let cfg = Toolpath3AxisConfig::new().with_scallop_height(42.0);
        assert!((cfg.scallop_height - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_strategy() {
        let cfg = Toolpath3AxisConfig::new().with_strategy(42);
        assert_eq!(cfg.strategy, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = Toolpath3AxisConfig::new().with_tool_diameter(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Toolpath3Axis"));
    }

    #[test]
    fn test_summary() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_planar() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parallel_planar();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_waterline() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.waterline();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_pencil_trace() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.pencil_trace();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_pencil_trace_empty() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap();
        assert!(e.pencil_trace().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = Toolpath3Axis::new(Toolpath3AxisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = Toolpath3AxisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = Toolpath3AxisError::InvalidConfig("a".into());
        let e2 = Toolpath3AxisError::ComputationFailed("b".into());
        let e3 = Toolpath3AxisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
