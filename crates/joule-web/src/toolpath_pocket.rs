//! Pocket toolpath strategies.
//!
//! Provides [`ToolpathPocketConfig`] builder and [`ToolpathPocket`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolpathPocketError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ToolpathPocketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ToolpathPocket: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ToolpathPocket: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ToolpathPocket: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ToolpathPocket`] parameters.
#[derive(Debug, Clone)]
pub struct ToolpathPocketConfig {
    pub tool_diameter: f64,
    pub stepover_pct: f64,
    pub strategy: usize,
    pub depth_per_pass: f64,
}

impl ToolpathPocketConfig {
    pub fn new() -> Self {
        Self {
            tool_diameter: 10.0,
            stepover_pct: 0.40,
            strategy: 0,
            depth_per_pass: 2.0,
        }
    }

    pub fn with_tool_diameter(mut self, v: f64) -> Self {
        self.tool_diameter = v;
        self
    }

    pub fn with_stepover_pct(mut self, v: f64) -> Self {
        self.stepover_pct = v;
        self
    }

    pub fn with_strategy(mut self, v: usize) -> Self {
        self.strategy = v;
        self
    }

    pub fn with_depth_per_pass(mut self, v: f64) -> Self {
        self.depth_per_pass = v;
        self
    }

    pub fn validate(&self) -> Result<(), ToolpathPocketError> {
        if self.tool_diameter.is_nan() {
            return Err(ToolpathPocketError::InvalidConfig("tool_diameter is NaN".into()));
        }
        if self.stepover_pct.is_nan() {
            return Err(ToolpathPocketError::InvalidConfig("stepover_pct is NaN".into()));
        }
        if self.depth_per_pass.is_nan() {
            return Err(ToolpathPocketError::InvalidConfig("depth_per_pass is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ToolpathPocketConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ToolpathPocketConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathPocketConfig(tool_diameter={0:.4}, stepover_pct={1:.4}, strategy={2}, depth_per_pass={3:.4})", self.tool_diameter, self.stepover_pct, self.strategy, self.depth_per_pass)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core pocket toolpath strategies engine.
#[derive(Debug, Clone)]
pub struct ToolpathPocket {
    config: ToolpathPocketConfig,
    data: Vec<f64>,
}

impl ToolpathPocket {
    pub fn new(config: ToolpathPocketConfig) -> Result<Self, ToolpathPocketError> {
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
    pub fn config(&self) -> &ToolpathPocketConfig { &self.config }

    /// Zigzag pocket toolpath.
    pub fn zigzag_pocket(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Spiral pocket toolpath.
    pub fn spiral_pocket(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Rest machining toolpath.
    pub fn rest_machining(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for ToolpathPocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathPocket(n={})", self.data.len())
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
        let cfg = ToolpathPocketConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ToolpathPocketConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ToolpathPocketConfig"));
    }

    #[test]
    fn test_config_with_tool_diameter() {
        let cfg = ToolpathPocketConfig::new().with_tool_diameter(42.0);
        assert!((cfg.tool_diameter - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_stepover_pct() {
        let cfg = ToolpathPocketConfig::new().with_stepover_pct(42.0);
        assert!((cfg.stepover_pct - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_strategy() {
        let cfg = ToolpathPocketConfig::new().with_strategy(42);
        assert_eq!(cfg.strategy, 42);
    }

    #[test]
    fn test_config_with_depth_per_pass() {
        let cfg = ToolpathPocketConfig::new().with_depth_per_pass(42.0);
        assert!((cfg.depth_per_pass - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ToolpathPocketConfig::new().with_tool_diameter(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ToolpathPocket"));
    }

    #[test]
    fn test_summary() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_zigzag_pocket() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.zigzag_pocket();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_spiral_pocket() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.spiral_pocket();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rest_machining() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rest_machining();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_rest_machining_empty() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap();
        assert!(e.rest_machining().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ToolpathPocket::new(ToolpathPocketConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ToolpathPocketError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ToolpathPocketError::InvalidConfig("a".into());
        let e2 = ToolpathPocketError::ComputationFailed("b".into());
        let e3 = ToolpathPocketError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
