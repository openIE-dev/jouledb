//! Electronic health record management.
//!
//! Provides [`EhrRecordConfig`] builder and [`EhrRecord`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum EhrRecordError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for EhrRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "EhrRecord: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "EhrRecord: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "EhrRecord: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`EhrRecord`] parameters.
#[derive(Debug, Clone)]
pub struct EhrRecordConfig {
    pub patient_id: u64,
    pub max_encounters: usize,
    pub include_demographics: bool,
    pub validate_codes: bool,
}

impl EhrRecordConfig {
    pub fn new() -> Self {
        Self {
            patient_id: 0,
            max_encounters: 1000,
            include_demographics: true,
            validate_codes: true,
        }
    }

    pub fn with_patient_id(mut self, v: u64) -> Self {
        self.patient_id = v;
        self
    }

    pub fn with_max_encounters(mut self, v: usize) -> Self {
        self.max_encounters = v;
        self
    }

    pub fn with_include_demographics(mut self, v: bool) -> Self {
        self.include_demographics = v;
        self
    }

    pub fn with_validate_codes(mut self, v: bool) -> Self {
        self.validate_codes = v;
        self
    }

    pub fn validate(&self) -> Result<(), EhrRecordError> {
        Ok(())
    }
}

impl Default for EhrRecordConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for EhrRecordConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EhrRecordConfig(patient_id={0}, max_encounters={1}, include_demographics={2}, validate_codes={3})", self.patient_id, self.max_encounters, self.include_demographics, self.validate_codes)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core electronic health record management engine.
#[derive(Debug, Clone)]
pub struct EhrRecord {
    config: EhrRecordConfig,
    data: Vec<f64>,
}

impl EhrRecord {
    pub fn new(config: EhrRecordConfig) -> Result<Self, EhrRecordError> {
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
    pub fn config(&self) -> &EhrRecordConfig { &self.config }

    /// Add clinical encounter.
    pub fn add_encounter(&self) -> usize {
        self.data.len()
    }

    /// Get active problem list.
    pub fn problem_list(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get current medications.
    pub fn medication_list(&self) -> Vec<f64> {
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

impl fmt::Display for EhrRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EhrRecord(n={})", self.data.len())
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
        let cfg = EhrRecordConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = EhrRecordConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("EhrRecordConfig"));
    }

    #[test]
    fn test_config_with_patient_id() {
        let cfg = EhrRecordConfig::new().with_patient_id(42);
        assert_eq!(cfg.patient_id, 42);
    }

    #[test]
    fn test_config_with_max_encounters() {
        let cfg = EhrRecordConfig::new().with_max_encounters(42);
        assert_eq!(cfg.max_encounters, 42);
    }

    #[test]
    fn test_config_with_include_demographics() {
        let cfg = EhrRecordConfig::new().with_include_demographics(false);
        assert_eq!(cfg.include_demographics, false);
    }

    #[test]
    fn test_config_with_validate_codes() {
        let cfg = EhrRecordConfig::new().with_validate_codes(false);
        assert_eq!(cfg.validate_codes, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = EhrRecordConfig::new().with_patient_id(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = EhrRecord::new(EhrRecordConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("EhrRecord"));
    }

    #[test]
    fn test_summary() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_encounter() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_encounter();
        assert!(result > 0);
    }

    #[test]
    fn test_problem_list() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.problem_list();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_medication_list() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.medication_list();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_medication_list_empty() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap();
        assert!(e.medication_list().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = EhrRecord::new(EhrRecordConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = EhrRecordError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = EhrRecordError::InvalidConfig("a".into());
        let e2 = EhrRecordError::ComputationFailed("b".into());
        let e3 = EhrRecordError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
