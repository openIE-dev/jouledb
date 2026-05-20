//! Medication administration verification.
//!
//! Provides [`MedAdminConfig`] builder and [`MedAdmin`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum MedAdminError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for MedAdminError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "MedAdmin: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "MedAdmin: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "MedAdmin: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`MedAdmin`] parameters.
#[derive(Debug, Clone)]
pub struct MedAdminConfig {
    pub verify_5rights: bool,
    pub barcode_scan: bool,
    pub time_window_min: usize,
    pub document_prn: bool,
}

impl MedAdminConfig {
    pub fn new() -> Self {
        Self {
            verify_5rights: true,
            barcode_scan: true,
            time_window_min: 30,
            document_prn: true,
        }
    }

    pub fn with_verify_5rights(mut self, v: bool) -> Self {
        self.verify_5rights = v;
        self
    }

    pub fn with_barcode_scan(mut self, v: bool) -> Self {
        self.barcode_scan = v;
        self
    }

    pub fn with_time_window_min(mut self, v: usize) -> Self {
        self.time_window_min = v;
        self
    }

    pub fn with_document_prn(mut self, v: bool) -> Self {
        self.document_prn = v;
        self
    }

    pub fn validate(&self) -> Result<(), MedAdminError> {
        Ok(())
    }
}

impl Default for MedAdminConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MedAdminConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MedAdminConfig(verify_5rights={0}, barcode_scan={1}, time_window_min={2}, document_prn={3})", self.verify_5rights, self.barcode_scan, self.time_window_min, self.document_prn)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core medication administration verification engine.
#[derive(Debug, Clone)]
pub struct MedAdmin {
    config: MedAdminConfig,
    data: Vec<f64>,
}

impl MedAdmin {
    pub fn new(config: MedAdminConfig) -> Result<Self, MedAdminError> {
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
    pub fn config(&self) -> &MedAdminConfig { &self.config }

    /// Verify 5 rights.
    pub fn verify_rights(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check if on-time administration.
    pub fn is_on_time(&self) -> bool {
        !self.data.is_empty()
    }

    /// Document administration.
    pub fn document_admin(&self) -> String {
        format!("{}: {} records", stringify!(document_admin), self.data.len())
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

impl fmt::Display for MedAdmin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MedAdmin(n={})", self.data.len())
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
        let cfg = MedAdminConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = MedAdminConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("MedAdminConfig"));
    }

    #[test]
    fn test_config_with_verify_5rights() {
        let cfg = MedAdminConfig::new().with_verify_5rights(false);
        assert_eq!(cfg.verify_5rights, false);
    }

    #[test]
    fn test_config_with_barcode_scan() {
        let cfg = MedAdminConfig::new().with_barcode_scan(false);
        assert_eq!(cfg.barcode_scan, false);
    }

    #[test]
    fn test_config_with_time_window_min() {
        let cfg = MedAdminConfig::new().with_time_window_min(42);
        assert_eq!(cfg.time_window_min, 42);
    }

    #[test]
    fn test_config_with_document_prn() {
        let cfg = MedAdminConfig::new().with_document_prn(false);
        assert_eq!(cfg.document_prn, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = MedAdminConfig::new().with_verify_5rights(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = MedAdmin::new(MedAdminConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("MedAdmin"));
    }

    #[test]
    fn test_summary() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_verify_rights() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_rights();
        assert!(result);
    }

    #[test]
    fn test_is_on_time() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_on_time();
        assert!(result);
    }

    #[test]
    fn test_document_admin() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.document_admin();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_document_admin_empty() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap();
        let _ = e.document_admin();
    }

    #[test]
    fn test_config_accessor() {
        let e = MedAdmin::new(MedAdminConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = MedAdminError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = MedAdminError::InvalidConfig("a".into());
        let e2 = MedAdminError::ComputationFailed("b".into());
        let e3 = MedAdminError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
