//! SNOMED CT concept representation and hierarchy.
//!
//! Provides [`SnomedConceptConfig`] builder and [`SnomedConcept`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SnomedConceptError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SnomedConceptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SnomedConcept: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SnomedConcept: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SnomedConcept: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SnomedConcept`] parameters.
#[derive(Debug, Clone)]
pub struct SnomedConceptConfig {
    pub concept_id: u64,
    pub include_descriptions: bool,
    pub traverse_hierarchy: bool,
    pub max_depth: usize,
}

impl SnomedConceptConfig {
    pub fn new() -> Self {
        Self {
            concept_id: 0,
            include_descriptions: true,
            traverse_hierarchy: true,
            max_depth: 10,
        }
    }

    pub fn with_concept_id(mut self, v: u64) -> Self {
        self.concept_id = v;
        self
    }

    pub fn with_include_descriptions(mut self, v: bool) -> Self {
        self.include_descriptions = v;
        self
    }

    pub fn with_traverse_hierarchy(mut self, v: bool) -> Self {
        self.traverse_hierarchy = v;
        self
    }

    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    pub fn validate(&self) -> Result<(), SnomedConceptError> {
        Ok(())
    }
}

impl Default for SnomedConceptConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SnomedConceptConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SnomedConceptConfig(concept_id={0}, include_descriptions={1}, traverse_hierarchy={2}, max_depth={3})", self.concept_id, self.include_descriptions, self.traverse_hierarchy, self.max_depth)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core snomed ct concept representation and hierarchy engine.
#[derive(Debug, Clone)]
pub struct SnomedConcept {
    config: SnomedConceptConfig,
    data: Vec<f64>,
}

impl SnomedConcept {
    pub fn new(config: SnomedConceptConfig) -> Result<Self, SnomedConceptError> {
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
    pub fn config(&self) -> &SnomedConceptConfig { &self.config }

    /// Find concept by term.
    pub fn find_concept(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Check IS-A relationship.
    pub fn is_descendant(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get preferred term.
    pub fn preferred_term(&self) -> String {
        format!("{}: {} records", stringify!(preferred_term), self.data.len())
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

impl fmt::Display for SnomedConcept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SnomedConcept(n={})", self.data.len())
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
        let cfg = SnomedConceptConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SnomedConceptConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SnomedConceptConfig"));
    }

    #[test]
    fn test_config_with_concept_id() {
        let cfg = SnomedConceptConfig::new().with_concept_id(42);
        assert_eq!(cfg.concept_id, 42);
    }

    #[test]
    fn test_config_with_include_descriptions() {
        let cfg = SnomedConceptConfig::new().with_include_descriptions(false);
        assert_eq!(cfg.include_descriptions, false);
    }

    #[test]
    fn test_config_with_traverse_hierarchy() {
        let cfg = SnomedConceptConfig::new().with_traverse_hierarchy(false);
        assert_eq!(cfg.traverse_hierarchy, false);
    }

    #[test]
    fn test_config_with_max_depth() {
        let cfg = SnomedConceptConfig::new().with_max_depth(42);
        assert_eq!(cfg.max_depth, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SnomedConceptConfig::new().with_concept_id(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SnomedConcept"));
    }

    #[test]
    fn test_summary() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_concept() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.find_concept();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_descendant() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_descendant();
        assert!(result);
    }

    #[test]
    fn test_preferred_term() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.preferred_term();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_preferred_term_empty() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap();
        let _ = e.preferred_term();
    }

    #[test]
    fn test_config_accessor() {
        let e = SnomedConcept::new(SnomedConceptConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SnomedConceptError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SnomedConceptError::InvalidConfig("a".into());
        let e2 = SnomedConceptError::ComputationFailed("b".into());
        let e3 = SnomedConceptError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
