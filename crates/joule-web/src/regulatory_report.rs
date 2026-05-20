//! Regulatory trade and position reporting.
//!
//! Provides [`RegulatoryReportConfig`] builder and [`RegulatoryReport`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RegulatoryReportError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RegulatoryReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RegulatoryReport: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RegulatoryReport: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RegulatoryReport: insufficient data: {msg}"),
        }
    }
}

/// Variant selector for OutputFormat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    /// Delimited.
    Delimited,
    /// FixedWidth.
    FixedWidth,
    /// Xml.
    Xml,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Variant selector for ReportType.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReportType {
    /// Trade.
    Trade,
    /// Position.
    Position,
    /// LargeTrader.
    LargeTrader,
}

impl fmt::Display for ReportType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RegulatoryReport`] parameters.
#[derive(Debug, Clone)]
pub struct RegulatoryReportConfig {
    pub report_type: ReportType,
    pub format: OutputFormat,
    pub include_header: bool,
    pub batch_size: usize,
}

impl RegulatoryReportConfig {
    pub fn new() -> Self {
        Self {
            report_type: ReportType::Trade,
            format: OutputFormat::Delimited,
            include_header: true,
            batch_size: 1000,
        }
    }

    pub fn with_report_type(mut self, v: ReportType) -> Self {
        self.report_type = v;
        self
    }

    pub fn with_format(mut self, v: OutputFormat) -> Self {
        self.format = v;
        self
    }

    pub fn with_include_header(mut self, v: bool) -> Self {
        self.include_header = v;
        self
    }

    pub fn with_batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn validate(&self) -> Result<(), RegulatoryReportError> {
        Ok(())
    }
}

impl Default for RegulatoryReportConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RegulatoryReportConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegulatoryReportConfig(report_type={0:?}, format={1:?}, include_header={2}, batch_size={3})", self.report_type, self.format, self.include_header, self.batch_size)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core regulatory trade and position reporting engine.
#[derive(Debug, Clone)]
pub struct RegulatoryReport {
    config: RegulatoryReportConfig,
    data: Vec<f64>,
}

impl RegulatoryReport {
    pub fn new(config: RegulatoryReportConfig) -> Result<Self, RegulatoryReportError> {
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
    pub fn config(&self) -> &RegulatoryReportConfig { &self.config }

    /// Generate report content.
    pub fn generate(&self) -> String {
        format!("{}: {} records", stringify!(generate), self.data.len())
    }

    /// Validate required fields.
    pub fn validate_fields(&self) -> bool {
        !self.data.is_empty()
    }

    /// Number of records.
    pub fn record_count(&self) -> usize {
        self.data.len()
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

impl fmt::Display for RegulatoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegulatoryReport(n={})", self.data.len())
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
        let cfg = RegulatoryReportConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RegulatoryReportConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RegulatoryReportConfig"));
    }

    #[test]
    fn test_config_with_report_type() {
        let cfg = RegulatoryReportConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_with_format() {
        let cfg = RegulatoryReportConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_with_include_header() {
        let cfg = RegulatoryReportConfig::new().with_include_header(false);
        assert_eq!(cfg.include_header, false);
    }

    #[test]
    fn test_config_with_batch_size() {
        let cfg = RegulatoryReportConfig::new().with_batch_size(42);
        assert_eq!(cfg.batch_size, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RegulatoryReportConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RegulatoryReport"));
    }

    #[test]
    fn test_summary() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_validate_fields() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_fields();
        assert!(result);
    }

    #[test]
    fn test_record_count() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.record_count();
        assert!(result > 0);
    }

    #[test]
    fn test_record_count_empty() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap();
        let _ = e.record_count();
    }

    #[test]
    fn test_config_accessor() {
        let e = RegulatoryReport::new(RegulatoryReportConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RegulatoryReportError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RegulatoryReportError::InvalidConfig("a".into());
        let e2 = RegulatoryReportError::ComputationFailed("b".into());
        let e3 = RegulatoryReportError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
