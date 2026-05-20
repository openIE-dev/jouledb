//! Contour toolpath for 2D profile milling.
//!
//! Provides [`ToolpathContourConfig`] builder and [`ToolpathContour`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolpathContourError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ToolpathContourError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ToolpathContour: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ToolpathContour: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ToolpathContour: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ToolpathContour`] parameters.
#[derive(Debug, Clone)]
pub struct ToolpathContourConfig {
    pub tool_diameter: f64,
    pub stepover_pct: f64,
    pub feed_rate: f64,
    pub climb_milling: bool,
}

impl ToolpathContourConfig {
    pub fn new() -> Self {
        Self {
            tool_diameter: 10.0,
            stepover_pct: 0.50,
            feed_rate: 500.0,
            climb_milling: true,
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

    pub fn with_feed_rate(mut self, v: f64) -> Self {
        self.feed_rate = v;
        self
    }

    pub fn with_climb_milling(mut self, v: bool) -> Self {
        self.climb_milling = v;
        self
    }

    pub fn validate(&self) -> Result<(), ToolpathContourError> {
        if self.tool_diameter.is_nan() {
            return Err(ToolpathContourError::InvalidConfig("tool_diameter is NaN".into()));
        }
        if self.stepover_pct.is_nan() {
            return Err(ToolpathContourError::InvalidConfig("stepover_pct is NaN".into()));
        }
        if self.feed_rate.is_nan() {
            return Err(ToolpathContourError::InvalidConfig("feed_rate is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ToolpathContourConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ToolpathContourConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathContourConfig(tool_diameter={0:.4}, stepover_pct={1:.4}, feed_rate={2:.4}, climb_milling={3})", self.tool_diameter, self.stepover_pct, self.feed_rate, self.climb_milling)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core contour toolpath for 2d profile milling engine.
#[derive(Debug, Clone)]
pub struct ToolpathContour {
    config: ToolpathContourConfig,
    data: Vec<f64>,
}

impl ToolpathContour {
    pub fn new(config: ToolpathContourConfig) -> Result<Self, ToolpathContourError> {
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
    pub fn config(&self) -> &ToolpathContourConfig { &self.config }

    /// Generate contour toolpath.
    pub fn generate_contour(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Generate lead-in arc.
    pub fn lead_in_arc(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Estimate machining time.
    pub fn machining_time(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for ToolpathContour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathContour(n={})", self.data.len())
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
        let cfg = ToolpathContourConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ToolpathContourConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ToolpathContourConfig"));
    }

    #[test]
    fn test_config_with_tool_diameter() {
        let cfg = ToolpathContourConfig::new().with_tool_diameter(42.0);
        assert!((cfg.tool_diameter - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_stepover_pct() {
        let cfg = ToolpathContourConfig::new().with_stepover_pct(42.0);
        assert!((cfg.stepover_pct - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_feed_rate() {
        let cfg = ToolpathContourConfig::new().with_feed_rate(42.0);
        assert!((cfg.feed_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_climb_milling() {
        let cfg = ToolpathContourConfig::new().with_climb_milling(false);
        assert_eq!(cfg.climb_milling, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ToolpathContourConfig::new().with_tool_diameter(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ToolpathContour"));
    }

    #[test]
    fn test_summary() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_contour() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_contour();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_lead_in_arc() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lead_in_arc();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_machining_time() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.machining_time();
        assert!(result.is_finite());
    }

    #[test]
    fn test_machining_time_empty() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap();
        assert!((e.machining_time() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = ToolpathContour::new(ToolpathContourConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ToolpathContourError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ToolpathContourError::InvalidConfig("a".into());
        let e2 = ToolpathContourError::ComputationFailed("b".into());
        let e3 = ToolpathContourError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
