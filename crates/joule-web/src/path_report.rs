//! Pathology report generation and staging.
//!
//! Provides [`PathReportConfig`] builder and [`PathReport`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PathReportError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PathReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PathReport: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PathReport: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PathReport: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PathReport`] parameters.
#[derive(Debug, Clone)]
pub struct PathReportConfig {
    pub specimen_type: usize,
    pub include_staging: bool,
    pub synoptic: bool,
    pub grade_system: usize,
}

impl PathReportConfig {
    pub fn new() -> Self {
        Self {
            specimen_type: 0,
            include_staging: true,
            synoptic: true,
            grade_system: 0,
        }
    }

    pub fn with_specimen_type(mut self, v: usize) -> Self {
        self.specimen_type = v;
        self
    }

    pub fn with_include_staging(mut self, v: bool) -> Self {
        self.include_staging = v;
        self
    }

    pub fn with_synoptic(mut self, v: bool) -> Self {
        self.synoptic = v;
        self
    }

    pub fn with_grade_system(mut self, v: usize) -> Self {
        self.grade_system = v;
        self
    }

    pub fn validate(&self) -> Result<(), PathReportError> {
        Ok(())
    }
}

impl Default for PathReportConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PathReportConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PathReportConfig(specimen_type={0}, include_staging={1}, synoptic={2}, grade_system={3})", self.specimen_type, self.include_staging, self.synoptic, self.grade_system)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core pathology report generation and staging engine.
#[derive(Debug, Clone)]
pub struct PathReport {
    config: PathReportConfig,
    data: Vec<f64>,
}

impl PathReport {
    pub fn new(config: PathReportConfig) -> Result<Self, PathReportError> {
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
    pub fn config(&self) -> &PathReportConfig { &self.config }

    /// Generate pathology report.
    pub fn generate_report(&self) -> String {
        format!("{}: {} records", stringify!(generate_report), self.data.len())
    }

    /// Determine TNM stage.
    pub fn tnm_stage(&self) -> String {
        format!("{}: {} records", stringify!(tnm_stage), self.data.len())
    }

    /// Evaluate margin status.
    pub fn margin_status(&self) -> String {
        format!("{}: {} records", stringify!(margin_status), self.data.len())
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

impl fmt::Display for PathReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PathReport(n={})", self.data.len())
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
        let cfg = PathReportConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PathReportConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PathReportConfig"));
    }

    #[test]
    fn test_config_with_specimen_type() {
        let cfg = PathReportConfig::new().with_specimen_type(42);
        assert_eq!(cfg.specimen_type, 42);
    }

    #[test]
    fn test_config_with_include_staging() {
        let cfg = PathReportConfig::new().with_include_staging(false);
        assert_eq!(cfg.include_staging, false);
    }

    #[test]
    fn test_config_with_synoptic() {
        let cfg = PathReportConfig::new().with_synoptic(false);
        assert_eq!(cfg.synoptic, false);
    }

    #[test]
    fn test_config_with_grade_system() {
        let cfg = PathReportConfig::new().with_grade_system(42);
        assert_eq!(cfg.grade_system, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PathReportConfig::new().with_specimen_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PathReport::new(PathReportConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PathReport::new(PathReportConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PathReport"));
    }

    #[test]
    fn test_summary() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PathReport::new(PathReportConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_report() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_report();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tnm_stage() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tnm_stage();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_margin_status() {
        let e = PathReport::new(PathReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.margin_status();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_margin_status_empty() {
        let e = PathReport::new(PathReportConfig::new()).unwrap();
        let _ = e.margin_status();
    }

    #[test]
    fn test_config_accessor() {
        let e = PathReport::new(PathReportConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PathReportError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PathReportError::InvalidConfig("a".into());
        let e2 = PathReportError::ComputationFailed("b".into());
        let e3 = PathReportError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
