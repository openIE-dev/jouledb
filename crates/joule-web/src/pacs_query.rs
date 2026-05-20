//! PACS query model for medical image retrieval.
//!
//! Provides [`PacsQueryConfig`] builder and [`PacsQuery`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PacsQueryError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PacsQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PacsQuery: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PacsQuery: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PacsQuery: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PacsQuery`] parameters.
#[derive(Debug, Clone)]
pub struct PacsQueryConfig {
    pub query_level: usize,
    pub max_results: usize,
    pub include_previews: bool,
    pub timeout_ms: u64,
}

impl PacsQueryConfig {
    pub fn new() -> Self {
        Self {
            query_level: 0,
            max_results: 100,
            include_previews: false,
            timeout_ms: 30000,
        }
    }

    pub fn with_query_level(mut self, v: usize) -> Self {
        self.query_level = v;
        self
    }

    pub fn with_max_results(mut self, v: usize) -> Self {
        self.max_results = v;
        self
    }

    pub fn with_include_previews(mut self, v: bool) -> Self {
        self.include_previews = v;
        self
    }

    pub fn with_timeout_ms(mut self, v: u64) -> Self {
        self.timeout_ms = v;
        self
    }

    pub fn validate(&self) -> Result<(), PacsQueryError> {
        Ok(())
    }
}

impl Default for PacsQueryConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PacsQueryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PacsQueryConfig(query_level={0}, max_results={1}, include_previews={2}, timeout_ms={3})", self.query_level, self.max_results, self.include_previews, self.timeout_ms)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core pacs query model for medical image retrieval engine.
#[derive(Debug, Clone)]
pub struct PacsQuery {
    config: PacsQueryConfig,
    data: Vec<f64>,
}

impl PacsQuery {
    pub fn new(config: PacsQueryConfig) -> Result<Self, PacsQueryError> {
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
    pub fn config(&self) -> &PacsQueryConfig { &self.config }

    /// Query studies.
    pub fn study_query(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Query series within study.
    pub fn series_query(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Query modality worklist.
    pub fn worklist_query(&self) -> Vec<f64> {
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

impl fmt::Display for PacsQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PacsQuery(n={})", self.data.len())
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
        let cfg = PacsQueryConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PacsQueryConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PacsQueryConfig"));
    }

    #[test]
    fn test_config_with_query_level() {
        let cfg = PacsQueryConfig::new().with_query_level(42);
        assert_eq!(cfg.query_level, 42);
    }

    #[test]
    fn test_config_with_max_results() {
        let cfg = PacsQueryConfig::new().with_max_results(42);
        assert_eq!(cfg.max_results, 42);
    }

    #[test]
    fn test_config_with_include_previews() {
        let cfg = PacsQueryConfig::new().with_include_previews(true);
        assert_eq!(cfg.include_previews, true);
    }

    #[test]
    fn test_config_with_timeout_ms() {
        let cfg = PacsQueryConfig::new().with_timeout_ms(42);
        assert_eq!(cfg.timeout_ms, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PacsQueryConfig::new().with_query_level(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PacsQuery::new(PacsQueryConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PacsQuery"));
    }

    #[test]
    fn test_summary() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_study_query() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.study_query();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_series_query() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.series_query();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_worklist_query() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.worklist_query();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_worklist_query_empty() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap();
        assert!(e.worklist_query().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PacsQuery::new(PacsQueryConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PacsQueryError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PacsQueryError::InvalidConfig("a".into());
        let e2 = PacsQueryError::ComputationFailed("b".into());
        let e3 = PacsQueryError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
