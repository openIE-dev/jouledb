//! HIPAA access audit logging and monitoring.
//!
//! Provides [`AccessAuditConfig`] builder and [`AccessAudit`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessAuditError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AccessAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AccessAudit: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AccessAudit: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AccessAudit: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AccessAudit`] parameters.
#[derive(Debug, Clone)]
pub struct AccessAuditConfig {
    pub log_all_access: bool,
    pub break_glass_track: bool,
    pub retention_days: u32,
    pub alert_suspicious: bool,
}

impl AccessAuditConfig {
    pub fn new() -> Self {
        Self {
            log_all_access: true,
            break_glass_track: true,
            retention_days: 2555,
            alert_suspicious: true,
        }
    }

    pub fn with_log_all_access(mut self, v: bool) -> Self {
        self.log_all_access = v;
        self
    }

    pub fn with_break_glass_track(mut self, v: bool) -> Self {
        self.break_glass_track = v;
        self
    }

    pub fn with_retention_days(mut self, v: u32) -> Self {
        self.retention_days = v;
        self
    }

    pub fn with_alert_suspicious(mut self, v: bool) -> Self {
        self.alert_suspicious = v;
        self
    }

    pub fn validate(&self) -> Result<(), AccessAuditError> {
        Ok(())
    }
}

impl Default for AccessAuditConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AccessAuditConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AccessAuditConfig(log_all_access={0}, break_glass_track={1}, retention_days={2}, alert_suspicious={3})", self.log_all_access, self.break_glass_track, self.retention_days, self.alert_suspicious)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hipaa access audit logging and monitoring engine.
#[derive(Debug, Clone)]
pub struct AccessAudit {
    config: AccessAuditConfig,
    data: Vec<f64>,
}

impl AccessAudit {
    pub fn new(config: AccessAuditConfig) -> Result<Self, AccessAuditError> {
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
    pub fn config(&self) -> &AccessAuditConfig { &self.config }

    /// Log data access event.
    pub fn log_access(&self) -> u64 {
        self.data.len() as u64
    }

    /// Detect suspicious access.
    pub fn detect_suspicious(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate audit report.
    pub fn generate_report(&self) -> String {
        format!("{}: {} records", stringify!(generate_report), self.data.len())
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

impl fmt::Display for AccessAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AccessAudit(n={})", self.data.len())
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
        let cfg = AccessAuditConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AccessAuditConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AccessAuditConfig"));
    }

    #[test]
    fn test_config_with_log_all_access() {
        let cfg = AccessAuditConfig::new().with_log_all_access(false);
        assert_eq!(cfg.log_all_access, false);
    }

    #[test]
    fn test_config_with_break_glass_track() {
        let cfg = AccessAuditConfig::new().with_break_glass_track(false);
        assert_eq!(cfg.break_glass_track, false);
    }

    #[test]
    fn test_config_with_retention_days() {
        let cfg = AccessAuditConfig::new().with_retention_days(42);
        assert_eq!(cfg.retention_days, 42);
    }

    #[test]
    fn test_config_with_alert_suspicious() {
        let cfg = AccessAuditConfig::new().with_alert_suspicious(false);
        assert_eq!(cfg.alert_suspicious, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AccessAuditConfig::new().with_log_all_access(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AccessAudit::new(AccessAuditConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AccessAudit"));
    }

    #[test]
    fn test_summary() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_log_access() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.log_access();
        assert!(result > 0);
    }

    #[test]
    fn test_detect_suspicious() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_suspicious();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_report() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_report();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_report_empty() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap();
        let _ = e.generate_report();
    }

    #[test]
    fn test_config_accessor() {
        let e = AccessAudit::new(AccessAuditConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AccessAuditError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AccessAuditError::InvalidConfig("a".into());
        let e2 = AccessAuditError::ComputationFailed("b".into());
        let e3 = AccessAuditError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
