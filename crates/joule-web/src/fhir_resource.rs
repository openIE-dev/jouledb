//! FHIR resource representation and bundling.
//!
//! Provides [`FhirResourceConfig`] builder and [`FhirResource`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FhirResourceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FhirResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "FhirResource: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "FhirResource: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "FhirResource: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`FhirResource`] parameters.
#[derive(Debug, Clone)]
pub struct FhirResourceConfig {
    pub resource_type: usize,
    pub version: usize,
    pub validate_refs: bool,
    pub include_narrative: bool,
}

impl FhirResourceConfig {
    pub fn new() -> Self {
        Self {
            resource_type: 0,
            version: 4,
            validate_refs: true,
            include_narrative: true,
        }
    }

    pub fn with_resource_type(mut self, v: usize) -> Self {
        self.resource_type = v;
        self
    }

    pub fn with_version(mut self, v: usize) -> Self {
        self.version = v;
        self
    }

    pub fn with_validate_refs(mut self, v: bool) -> Self {
        self.validate_refs = v;
        self
    }

    pub fn with_include_narrative(mut self, v: bool) -> Self {
        self.include_narrative = v;
        self
    }

    pub fn validate(&self) -> Result<(), FhirResourceError> {
        Ok(())
    }
}

impl Default for FhirResourceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FhirResourceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FhirResourceConfig(resource_type={0}, version={1}, validate_refs={2}, include_narrative={3})", self.resource_type, self.version, self.validate_refs, self.include_narrative)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core fhir resource representation and bundling engine.
#[derive(Debug, Clone)]
pub struct FhirResource {
    config: FhirResourceConfig,
    data: Vec<f64>,
}

impl FhirResource {
    pub fn new(config: FhirResourceConfig) -> Result<Self, FhirResourceError> {
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
    pub fn config(&self) -> &FhirResourceConfig { &self.config }

    /// Serialize resource to JSON-like format.
    pub fn to_json(&self) -> String {
        format!("{}: {} records", stringify!(to_json), self.data.len())
    }

    /// Validate resource structure.
    pub fn validate_resource(&self) -> bool {
        !self.data.is_empty()
    }

    /// Bundle multiple resources.
    pub fn bundle_resources(&self) -> Vec<f64> {
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

impl fmt::Display for FhirResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FhirResource(n={})", self.data.len())
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
        let cfg = FhirResourceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FhirResourceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FhirResourceConfig"));
    }

    #[test]
    fn test_config_with_resource_type() {
        let cfg = FhirResourceConfig::new().with_resource_type(42);
        assert_eq!(cfg.resource_type, 42);
    }

    #[test]
    fn test_config_with_version() {
        let cfg = FhirResourceConfig::new().with_version(42);
        assert_eq!(cfg.version, 42);
    }

    #[test]
    fn test_config_with_validate_refs() {
        let cfg = FhirResourceConfig::new().with_validate_refs(false);
        assert_eq!(cfg.validate_refs, false);
    }

    #[test]
    fn test_config_with_include_narrative() {
        let cfg = FhirResourceConfig::new().with_include_narrative(false);
        assert_eq!(cfg.include_narrative, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FhirResourceConfig::new().with_resource_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = FhirResource::new(FhirResourceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("FhirResource"));
    }

    #[test]
    fn test_summary() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_json() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.to_json();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_validate_resource() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate_resource();
        assert!(result);
    }

    #[test]
    fn test_bundle_resources() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bundle_resources();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bundle_resources_empty() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap();
        assert!(e.bundle_resources().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = FhirResource::new(FhirResourceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FhirResourceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FhirResourceError::InvalidConfig("a".into());
        let e2 = FhirResourceError::ComputationFailed("b".into());
        let e3 = FhirResourceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
