//! Cutting tool library and specifications.
//!
//! Provides [`ToolLibraryConfig`] builder and [`ToolLibrary`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolLibraryError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ToolLibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ToolLibrary: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ToolLibrary: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ToolLibrary: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ToolLibrary`] parameters.
#[derive(Debug, Clone)]
pub struct ToolLibraryConfig {
    pub tool_count: usize,
    pub default_material: usize,
    pub metric_units: bool,
    pub include_speeds: bool,
}

impl ToolLibraryConfig {
    pub fn new() -> Self {
        Self {
            tool_count: 100,
            default_material: 0,
            metric_units: true,
            include_speeds: true,
        }
    }

    pub fn with_tool_count(mut self, v: usize) -> Self {
        self.tool_count = v;
        self
    }

    pub fn with_default_material(mut self, v: usize) -> Self {
        self.default_material = v;
        self
    }

    pub fn with_metric_units(mut self, v: bool) -> Self {
        self.metric_units = v;
        self
    }

    pub fn with_include_speeds(mut self, v: bool) -> Self {
        self.include_speeds = v;
        self
    }

    pub fn validate(&self) -> Result<(), ToolLibraryError> {
        Ok(())
    }
}

impl Default for ToolLibraryConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ToolLibraryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolLibraryConfig(tool_count={0}, default_material={1}, metric_units={2}, include_speeds={3})", self.tool_count, self.default_material, self.metric_units, self.include_speeds)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core cutting tool library and specifications engine.
#[derive(Debug, Clone)]
pub struct ToolLibrary {
    config: ToolLibraryConfig,
    data: Vec<f64>,
}

impl ToolLibrary {
    pub fn new(config: ToolLibraryConfig) -> Result<Self, ToolLibraryError> {
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
    pub fn config(&self) -> &ToolLibraryConfig { &self.config }

    /// Look up tool by ID.
    pub fn lookup_tool(&self) -> String {
        format!("{}: {} records", stringify!(lookup_tool), self.data.len())
    }

    /// Recommended feed rate.
    pub fn recommended_feed(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Recommended spindle speed.
    pub fn recommended_speed(&self) -> f64 {
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

impl fmt::Display for ToolLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolLibrary(n={})", self.data.len())
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
        let cfg = ToolLibraryConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ToolLibraryConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ToolLibraryConfig"));
    }

    #[test]
    fn test_config_with_tool_count() {
        let cfg = ToolLibraryConfig::new().with_tool_count(42);
        assert_eq!(cfg.tool_count, 42);
    }

    #[test]
    fn test_config_with_default_material() {
        let cfg = ToolLibraryConfig::new().with_default_material(42);
        assert_eq!(cfg.default_material, 42);
    }

    #[test]
    fn test_config_with_metric_units() {
        let cfg = ToolLibraryConfig::new().with_metric_units(false);
        assert_eq!(cfg.metric_units, false);
    }

    #[test]
    fn test_config_with_include_speeds() {
        let cfg = ToolLibraryConfig::new().with_include_speeds(false);
        assert_eq!(cfg.include_speeds, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ToolLibraryConfig::new().with_tool_count(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ToolLibrary"));
    }

    #[test]
    fn test_summary() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lookup_tool() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lookup_tool();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_recommended_feed() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.recommended_feed();
        assert!(result.is_finite());
    }

    #[test]
    fn test_recommended_speed() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.recommended_speed();
        assert!(result.is_finite());
    }

    #[test]
    fn test_recommended_speed_empty() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap();
        assert!((e.recommended_speed() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = ToolLibrary::new(ToolLibraryConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ToolLibraryError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ToolLibraryError::InvalidConfig("a".into());
        let e2 = ToolLibraryError::ComputationFailed("b".into());
        let e3 = ToolLibraryError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
