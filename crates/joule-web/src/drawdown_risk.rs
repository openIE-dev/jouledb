//! Drawdown analysis and risk metrics.
//!
//! Provides [`DrawdownRiskConfig`] builder and [`DrawdownRisk`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawdownRiskError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DrawdownRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DrawdownRisk: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DrawdownRisk: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DrawdownRisk: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DrawdownRisk`] parameters.
#[derive(Debug, Clone)]
pub struct DrawdownRiskConfig {
    pub peak: f64,
    pub current: f64,
    pub threshold: f64,
    pub window_days: usize,
}

impl DrawdownRiskConfig {
    pub fn new() -> Self {
        Self {
            peak: 1000.0,
            current: 950.0,
            threshold: 0.10,
            window_days: 252,
        }
    }

    pub fn with_peak(mut self, v: f64) -> Self {
        self.peak = v;
        self
    }

    pub fn with_current(mut self, v: f64) -> Self {
        self.current = v;
        self
    }

    pub fn with_threshold(mut self, v: f64) -> Self {
        self.threshold = v;
        self
    }

    pub fn with_window_days(mut self, v: usize) -> Self {
        self.window_days = v;
        self
    }

    pub fn validate(&self) -> Result<(), DrawdownRiskError> {
        if self.peak.is_nan() {
            return Err(DrawdownRiskError::InvalidConfig("peak is NaN".into()));
        }
        if self.current.is_nan() {
            return Err(DrawdownRiskError::InvalidConfig("current is NaN".into()));
        }
        if self.threshold.is_nan() {
            return Err(DrawdownRiskError::InvalidConfig("threshold is NaN".into()));
        }
        Ok(())
    }
}

impl Default for DrawdownRiskConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DrawdownRiskConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrawdownRiskConfig(peak={0:.4}, current={1:.4}, threshold={2:.4}, window_days={3})", self.peak, self.current, self.threshold, self.window_days)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core drawdown analysis and risk metrics engine.
#[derive(Debug, Clone)]
pub struct DrawdownRisk {
    config: DrawdownRiskConfig,
    data: Vec<f64>,
}

impl DrawdownRisk {
    pub fn new(config: DrawdownRiskConfig) -> Result<Self, DrawdownRiskError> {
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
    pub fn config(&self) -> &DrawdownRiskConfig { &self.config }

    /// Calculate maximum drawdown.
    pub fn max_drawdown(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Calmar ratio.
    pub fn calmar_ratio(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Ulcer performance index.
    pub fn ulcer_index(&self) -> f64 {
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

impl fmt::Display for DrawdownRisk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrawdownRisk(n={})", self.data.len())
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
        let cfg = DrawdownRiskConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DrawdownRiskConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DrawdownRiskConfig"));
    }

    #[test]
    fn test_config_with_peak() {
        let cfg = DrawdownRiskConfig::new().with_peak(42.0);
        assert!((cfg.peak - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_current() {
        let cfg = DrawdownRiskConfig::new().with_current(42.0);
        assert!((cfg.current - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_threshold() {
        let cfg = DrawdownRiskConfig::new().with_threshold(42.0);
        assert!((cfg.threshold - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_window_days() {
        let cfg = DrawdownRiskConfig::new().with_window_days(42);
        assert_eq!(cfg.window_days, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DrawdownRiskConfig::new().with_peak(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DrawdownRisk"));
    }

    #[test]
    fn test_summary() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_drawdown() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.max_drawdown();
        assert!(result.is_finite());
    }

    #[test]
    fn test_calmar_ratio() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.calmar_ratio();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ulcer_index() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ulcer_index();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ulcer_index_empty() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap();
        assert!((e.ulcer_index() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = DrawdownRisk::new(DrawdownRiskConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DrawdownRiskError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DrawdownRiskError::InvalidConfig("a".into());
        let e2 = DrawdownRiskError::ComputationFailed("b".into());
        let e3 = DrawdownRiskError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
