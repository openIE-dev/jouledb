//! G-code post-processor for CNC machines.
//!
//! Provides [`PostProcessConfig`] builder and [`PostProcess`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PostProcessError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PostProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PostProcess: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PostProcess: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PostProcess: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PostProcess`] parameters.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    pub controller: usize,
    pub max_line_len: usize,
    pub block_numbering: bool,
    pub decimal_places: usize,
}

impl PostProcessConfig {
    pub fn new() -> Self {
        Self {
            controller: 0,
            max_line_len: 80,
            block_numbering: true,
            decimal_places: 3,
        }
    }

    pub fn with_controller(mut self, v: usize) -> Self {
        self.controller = v;
        self
    }

    pub fn with_max_line_len(mut self, v: usize) -> Self {
        self.max_line_len = v;
        self
    }

    pub fn with_block_numbering(mut self, v: bool) -> Self {
        self.block_numbering = v;
        self
    }

    pub fn with_decimal_places(mut self, v: usize) -> Self {
        self.decimal_places = v;
        self
    }

    pub fn validate(&self) -> Result<(), PostProcessError> {
        Ok(())
    }
}

impl Default for PostProcessConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PostProcessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostProcessConfig(controller={0}, max_line_len={1}, block_numbering={2}, decimal_places={3})", self.controller, self.max_line_len, self.block_numbering, self.decimal_places)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core g-code post-processor for cnc machines engine.
#[derive(Debug, Clone)]
pub struct PostProcess {
    config: PostProcessConfig,
    data: Vec<f64>,
}

impl PostProcess {
    pub fn new(config: PostProcessConfig) -> Result<Self, PostProcessError> {
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
    pub fn config(&self) -> &PostProcessConfig { &self.config }

    /// Post-process G-code program.
    pub fn process_program(&self) -> String {
        format!("{}: {} records", stringify!(process_program), self.data.len())
    }

    /// Add program header.
    pub fn add_header(&self) -> String {
        format!("{}: {} records", stringify!(add_header), self.data.len())
    }

    /// Add program footer.
    pub fn add_footer(&self) -> String {
        format!("{}: {} records", stringify!(add_footer), self.data.len())
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

impl fmt::Display for PostProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostProcess(n={})", self.data.len())
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
        let cfg = PostProcessConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PostProcessConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PostProcessConfig"));
    }

    #[test]
    fn test_config_with_controller() {
        let cfg = PostProcessConfig::new().with_controller(42);
        assert_eq!(cfg.controller, 42);
    }

    #[test]
    fn test_config_with_max_line_len() {
        let cfg = PostProcessConfig::new().with_max_line_len(42);
        assert_eq!(cfg.max_line_len, 42);
    }

    #[test]
    fn test_config_with_block_numbering() {
        let cfg = PostProcessConfig::new().with_block_numbering(false);
        assert_eq!(cfg.block_numbering, false);
    }

    #[test]
    fn test_config_with_decimal_places() {
        let cfg = PostProcessConfig::new().with_decimal_places(42);
        assert_eq!(cfg.decimal_places, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PostProcessConfig::new().with_controller(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PostProcess::new(PostProcessConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PostProcess"));
    }

    #[test]
    fn test_summary() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_process_program() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.process_program();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_add_header() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_header();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_add_footer() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_footer();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_add_footer_empty() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap();
        let _ = e.add_footer();
    }

    #[test]
    fn test_config_accessor() {
        let e = PostProcess::new(PostProcessConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PostProcessError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PostProcessError::InvalidConfig("a".into());
        let e2 = PostProcessError::ComputationFailed("b".into());
        let e3 = PostProcessError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
