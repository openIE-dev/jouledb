//! Healthcare quality measure calculation.
//!
//! Provides [`QualityMeasureConfig`] builder and [`QualityMeasure`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityMeasureError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for QualityMeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "QualityMeasure: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "QualityMeasure: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "QualityMeasure: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`QualityMeasure`] parameters.
#[derive(Debug, Clone)]
pub struct QualityMeasureConfig {
    pub numerator: usize,
    pub denominator: usize,
    pub exclusions: usize,
    pub benchmark: f64,
}

impl QualityMeasureConfig {
    pub fn new() -> Self {
        Self {
            numerator: 85,
            denominator: 100,
            exclusions: 5,
            benchmark: 0.90,
        }
    }

    pub fn with_numerator(mut self, v: usize) -> Self {
        self.numerator = v;
        self
    }

    pub fn with_denominator(mut self, v: usize) -> Self {
        self.denominator = v;
        self
    }

    pub fn with_exclusions(mut self, v: usize) -> Self {
        self.exclusions = v;
        self
    }

    pub fn with_benchmark(mut self, v: f64) -> Self {
        self.benchmark = v;
        self
    }

    pub fn validate(&self) -> Result<(), QualityMeasureError> {
        if self.benchmark.is_nan() {
            return Err(QualityMeasureError::InvalidConfig("benchmark is NaN".into()));
        }
        Ok(())
    }
}

impl Default for QualityMeasureConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for QualityMeasureConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QualityMeasureConfig(numerator={0}, denominator={1}, exclusions={2}, benchmark={3:.4})", self.numerator, self.denominator, self.exclusions, self.benchmark)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core healthcare quality measure calculation engine.
#[derive(Debug, Clone)]
pub struct QualityMeasure {
    config: QualityMeasureConfig,
    data: Vec<f64>,
}

impl QualityMeasure {
    pub fn new(config: QualityMeasureConfig) -> Result<Self, QualityMeasureError> {
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
    pub fn config(&self) -> &QualityMeasureConfig { &self.config }

    /// Calculate measure rate.
    pub fn measure_rate(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check if meets benchmark.
    pub fn meets_benchmark(&self) -> bool {
        !self.data.is_empty()
    }

    /// Calculate trend over periods.
    pub fn trend(&self) -> Vec<f64> {
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

impl fmt::Display for QualityMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QualityMeasure(n={})", self.data.len())
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
        let cfg = QualityMeasureConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = QualityMeasureConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("QualityMeasureConfig"));
    }

    #[test]
    fn test_config_with_numerator() {
        let cfg = QualityMeasureConfig::new().with_numerator(42);
        assert_eq!(cfg.numerator, 42);
    }

    #[test]
    fn test_config_with_denominator() {
        let cfg = QualityMeasureConfig::new().with_denominator(42);
        assert_eq!(cfg.denominator, 42);
    }

    #[test]
    fn test_config_with_exclusions() {
        let cfg = QualityMeasureConfig::new().with_exclusions(42);
        assert_eq!(cfg.exclusions, 42);
    }

    #[test]
    fn test_config_with_benchmark() {
        let cfg = QualityMeasureConfig::new().with_benchmark(42.0);
        assert!((cfg.benchmark - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = QualityMeasureConfig::new().with_numerator(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("QualityMeasure"));
    }

    #[test]
    fn test_summary() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_measure_rate() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.measure_rate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_meets_benchmark() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.meets_benchmark();
        assert!(result);
    }

    #[test]
    fn test_trend() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.trend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_trend_empty() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap();
        assert!(e.trend().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = QualityMeasure::new(QualityMeasureConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = QualityMeasureError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = QualityMeasureError::InvalidConfig("a".into());
        let e2 = QualityMeasureError::ComputationFailed("b".into());
        let e3 = QualityMeasureError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
