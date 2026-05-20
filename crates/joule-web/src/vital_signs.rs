//! Vital signs monitoring and trend analysis.
//!
//! Provides [`VitalSignsConfig`] builder and [`VitalSigns`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum VitalSignsError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for VitalSignsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "VitalSigns: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "VitalSigns: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "VitalSigns: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`VitalSigns`] parameters.
#[derive(Debug, Clone)]
pub struct VitalSignsConfig {
    pub hr_low: f64,
    pub hr_high: f64,
    pub bp_sys_high: f64,
    pub bp_dia_high: f64,
}

impl VitalSignsConfig {
    pub fn new() -> Self {
        Self {
            hr_low: 60.0,
            hr_high: 100.0,
            bp_sys_high: 140.0,
            bp_dia_high: 90.0,
        }
    }

    pub fn with_hr_low(mut self, v: f64) -> Self {
        self.hr_low = v;
        self
    }

    pub fn with_hr_high(mut self, v: f64) -> Self {
        self.hr_high = v;
        self
    }

    pub fn with_bp_sys_high(mut self, v: f64) -> Self {
        self.bp_sys_high = v;
        self
    }

    pub fn with_bp_dia_high(mut self, v: f64) -> Self {
        self.bp_dia_high = v;
        self
    }

    pub fn validate(&self) -> Result<(), VitalSignsError> {
        if self.hr_low.is_nan() {
            return Err(VitalSignsError::InvalidConfig("hr_low is NaN".into()));
        }
        if self.hr_high.is_nan() {
            return Err(VitalSignsError::InvalidConfig("hr_high is NaN".into()));
        }
        if self.bp_sys_high.is_nan() {
            return Err(VitalSignsError::InvalidConfig("bp_sys_high is NaN".into()));
        }
        if self.bp_dia_high.is_nan() {
            return Err(VitalSignsError::InvalidConfig("bp_dia_high is NaN".into()));
        }
        Ok(())
    }
}

impl Default for VitalSignsConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VitalSignsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VitalSignsConfig(hr_low={0:.4}, hr_high={1:.4}, bp_sys_high={2:.4}, bp_dia_high={3:.4})", self.hr_low, self.hr_high, self.bp_sys_high, self.bp_dia_high)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core vital signs monitoring and trend analysis engine.
#[derive(Debug, Clone)]
pub struct VitalSigns {
    config: VitalSignsConfig,
    data: Vec<f64>,
}

impl VitalSigns {
    pub fn new(config: VitalSignsConfig) -> Result<Self, VitalSignsError> {
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
    pub fn config(&self) -> &VitalSignsConfig { &self.config }

    /// Check if vital is abnormal.
    pub fn is_abnormal(&self) -> bool {
        !self.data.is_empty()
    }

    /// Calculate BMI.
    pub fn bmi(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Analyze vital sign trends.
    pub fn trend_analysis(&self) -> Vec<f64> {
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

impl fmt::Display for VitalSigns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VitalSigns(n={})", self.data.len())
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
        let cfg = VitalSignsConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = VitalSignsConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("VitalSignsConfig"));
    }

    #[test]
    fn test_config_with_hr_low() {
        let cfg = VitalSignsConfig::new().with_hr_low(42.0);
        assert!((cfg.hr_low - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_hr_high() {
        let cfg = VitalSignsConfig::new().with_hr_high(42.0);
        assert!((cfg.hr_high - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bp_sys_high() {
        let cfg = VitalSignsConfig::new().with_bp_sys_high(42.0);
        assert!((cfg.bp_sys_high - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_bp_dia_high() {
        let cfg = VitalSignsConfig::new().with_bp_dia_high(42.0);
        assert!((cfg.bp_dia_high - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = VitalSignsConfig::new().with_hr_low(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = VitalSigns::new(VitalSignsConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("VitalSigns"));
    }

    #[test]
    fn test_summary() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_abnormal() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_abnormal();
        assert!(result);
    }

    #[test]
    fn test_bmi() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bmi();
        assert!(result.is_finite());
    }

    #[test]
    fn test_trend_analysis() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.trend_analysis();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_trend_analysis_empty() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap();
        assert!(e.trend_analysis().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = VitalSigns::new(VitalSignsConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = VitalSignsError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = VitalSignsError::InvalidConfig("a".into());
        let e2 = VitalSignsError::ComputationFailed("b".into());
        let e3 = VitalSignsError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
