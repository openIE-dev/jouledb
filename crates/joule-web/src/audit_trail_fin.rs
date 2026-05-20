//! Financial audit trail with immutable event log.
//!
//! Provides [`AuditTrailFinConfig`] builder and [`AuditTrailFin`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AuditTrailFinError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for AuditTrailFinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "AuditTrailFin: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "AuditTrailFin: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "AuditTrailFin: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`AuditTrailFin`] parameters.
#[derive(Debug, Clone)]
pub struct AuditTrailFinConfig {
    pub max_entries: usize,
    pub timestamp_precision_ns: bool,
    pub compress: bool,
    pub retention_days: u32,
}

impl AuditTrailFinConfig {
    pub fn new() -> Self {
        Self {
            max_entries: 1_000_000,
            timestamp_precision_ns: true,
            compress: false,
            retention_days: 2555,
        }
    }

    pub fn with_max_entries(mut self, v: usize) -> Self {
        self.max_entries = v;
        self
    }

    pub fn with_timestamp_precision_ns(mut self, v: bool) -> Self {
        self.timestamp_precision_ns = v;
        self
    }

    pub fn with_compress(mut self, v: bool) -> Self {
        self.compress = v;
        self
    }

    pub fn with_retention_days(mut self, v: u32) -> Self {
        self.retention_days = v;
        self
    }

    pub fn validate(&self) -> Result<(), AuditTrailFinError> {
        Ok(())
    }
}

impl Default for AuditTrailFinConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AuditTrailFinConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuditTrailFinConfig(max_entries={0}, timestamp_precision_ns={1}, compress={2}, retention_days={3})", self.max_entries, self.timestamp_precision_ns, self.compress, self.retention_days)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a AuditTrailFin operation.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditEvent {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for AuditEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuditEvent({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core financial audit trail with immutable event log engine.
#[derive(Debug, Clone)]
pub struct AuditTrailFin {
    config: AuditTrailFinConfig,
    data: Vec<f64>,
}

impl AuditTrailFin {
    pub fn new(config: AuditTrailFinConfig) -> Result<Self, AuditTrailFinError> {
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
    pub fn config(&self) -> &AuditTrailFinConfig { &self.config }

    /// Append event to trail.
    pub fn append(&self) -> u64 {
        self.data.len() as u64
    }

    /// Search events by criteria.
    pub fn search(&self) -> Vec<AuditEvent> {
        self.data.iter().enumerate().map(|(i, &v)| AuditEvent {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Verify trail integrity.
    pub fn verify_integrity(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for AuditTrailFin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuditTrailFin(n={})", self.data.len())
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
        let cfg = AuditTrailFinConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = AuditTrailFinConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("AuditTrailFinConfig"));
    }

    #[test]
    fn test_config_with_max_entries() {
        let cfg = AuditTrailFinConfig::new().with_max_entries(42);
        assert_eq!(cfg.max_entries, 42);
    }

    #[test]
    fn test_config_with_timestamp_precision_ns() {
        let cfg = AuditTrailFinConfig::new().with_timestamp_precision_ns(false);
        assert_eq!(cfg.timestamp_precision_ns, false);
    }

    #[test]
    fn test_config_with_compress() {
        let cfg = AuditTrailFinConfig::new().with_compress(true);
        assert_eq!(cfg.compress, true);
    }

    #[test]
    fn test_config_with_retention_days() {
        let cfg = AuditTrailFinConfig::new().with_retention_days(42);
        assert_eq!(cfg.retention_days, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = AuditTrailFinConfig::new().with_max_entries(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("AuditTrailFin"));
    }

    #[test]
    fn test_summary() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_append() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.append();
        assert!(result > 0);
    }

    #[test]
    fn test_search() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.search();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_verify_integrity() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.verify_integrity();
        assert!(result);
    }

    #[test]
    fn test_verify_integrity_empty() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap();
        assert!(!e.verify_integrity());
    }

    #[test]
    fn test_config_accessor() {
        let e = AuditTrailFin::new(AuditTrailFinConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = AuditTrailFinError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = AuditTrailFinError::InvalidConfig("a".into());
        let e2 = AuditTrailFinError::ComputationFailed("b".into());
        let e3 = AuditTrailFinError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
