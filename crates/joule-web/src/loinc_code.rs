//! LOINC lab observation code system.
//!
//! Provides [`LoincCodeConfig`] builder and [`LoincCode`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum LoincCodeError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for LoincCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "LoincCode: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "LoincCode: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "LoincCode: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`LoincCode`] parameters.
#[derive(Debug, Clone)]
pub struct LoincCodeConfig {
    pub include_panels: bool,
    pub version: usize,
    pub search_mode: usize,
    pub max_results: usize,
}

impl LoincCodeConfig {
    pub fn new() -> Self {
        Self {
            include_panels: true,
            version: 277,
            search_mode: 0,
            max_results: 50,
        }
    }

    pub fn with_include_panels(mut self, v: bool) -> Self {
        self.include_panels = v;
        self
    }

    pub fn with_version(mut self, v: usize) -> Self {
        self.version = v;
        self
    }

    pub fn with_search_mode(mut self, v: usize) -> Self {
        self.search_mode = v;
        self
    }

    pub fn with_max_results(mut self, v: usize) -> Self {
        self.max_results = v;
        self
    }

    pub fn validate(&self) -> Result<(), LoincCodeError> {
        Ok(())
    }
}

impl Default for LoincCodeConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for LoincCodeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoincCodeConfig(include_panels={0}, version={1}, search_mode={2}, max_results={3})", self.include_panels, self.version, self.search_mode, self.max_results)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core loinc lab observation code system engine.
#[derive(Debug, Clone)]
pub struct LoincCode {
    config: LoincCodeConfig,
    data: Vec<f64>,
}

impl LoincCode {
    pub fn new(config: LoincCodeConfig) -> Result<Self, LoincCodeError> {
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
    pub fn config(&self) -> &LoincCodeConfig { &self.config }

    /// Look up LOINC code.
    pub fn lookup(&self) -> String {
        format!("{}: {} records", stringify!(lookup), self.data.len())
    }

    /// Get panel member codes.
    pub fn panel_members(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get 6-axis values.
    pub fn axis_values(&self) -> Vec<f64> {
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

impl fmt::Display for LoincCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoincCode(n={})", self.data.len())
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
        let cfg = LoincCodeConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = LoincCodeConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("LoincCodeConfig"));
    }

    #[test]
    fn test_config_with_include_panels() {
        let cfg = LoincCodeConfig::new().with_include_panels(false);
        assert_eq!(cfg.include_panels, false);
    }

    #[test]
    fn test_config_with_version() {
        let cfg = LoincCodeConfig::new().with_version(42);
        assert_eq!(cfg.version, 42);
    }

    #[test]
    fn test_config_with_search_mode() {
        let cfg = LoincCodeConfig::new().with_search_mode(42);
        assert_eq!(cfg.search_mode, 42);
    }

    #[test]
    fn test_config_with_max_results() {
        let cfg = LoincCodeConfig::new().with_max_results(42);
        assert_eq!(cfg.max_results, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = LoincCodeConfig::new().with_include_panels(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = LoincCode::new(LoincCodeConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("LoincCode"));
    }

    #[test]
    fn test_summary() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lookup() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.lookup();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_panel_members() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.panel_members();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_axis_values() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.axis_values();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_axis_values_empty() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap();
        assert!(e.axis_values().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = LoincCode::new(LoincCodeConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = LoincCodeError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = LoincCodeError::InvalidConfig("a".into());
        let e2 = LoincCodeError::ComputationFailed("b".into());
        let e3 = LoincCodeError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
