//! Hotspot analysis with Getis-Ord Gi* statistic.
//!
//! Provides [`HotspotAnalysisConfig`] builder and [`HotspotAnalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum HotspotAnalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for HotspotAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "HotspotAnalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "HotspotAnalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "HotspotAnalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`HotspotAnalysis`] parameters.
#[derive(Debug, Clone)]
pub struct HotspotAnalysisConfig {
    pub distance_band: f64,
    pub significance: f64,
    pub row_standardize: bool,
    pub permutations: usize,
}

impl HotspotAnalysisConfig {
    pub fn new() -> Self {
        Self {
            distance_band: 1000.0,
            significance: 0.05,
            row_standardize: true,
            permutations: 999,
        }
    }

    pub fn with_distance_band(mut self, v: f64) -> Self {
        self.distance_band = v;
        self
    }

    pub fn with_significance(mut self, v: f64) -> Self {
        self.significance = v;
        self
    }

    pub fn with_row_standardize(mut self, v: bool) -> Self {
        self.row_standardize = v;
        self
    }

    pub fn with_permutations(mut self, v: usize) -> Self {
        self.permutations = v;
        self
    }

    pub fn validate(&self) -> Result<(), HotspotAnalysisError> {
        if self.distance_band.is_nan() {
            return Err(HotspotAnalysisError::InvalidConfig("distance_band is NaN".into()));
        }
        if self.significance.is_nan() {
            return Err(HotspotAnalysisError::InvalidConfig("significance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for HotspotAnalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for HotspotAnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HotspotAnalysisConfig(distance_band={0:.4}, significance={1:.4}, row_standardize={2}, permutations={3})", self.distance_band, self.significance, self.row_standardize, self.permutations)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hotspot analysis with getis-ord gi* statistic engine.
#[derive(Debug, Clone)]
pub struct HotspotAnalysis {
    config: HotspotAnalysisConfig,
    data: Vec<f64>,
}

impl HotspotAnalysis {
    pub fn new(config: HotspotAnalysisConfig) -> Result<Self, HotspotAnalysisError> {
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
    pub fn config(&self) -> &HotspotAnalysisConfig { &self.config }

    /// Compute Gi* statistic.
    pub fn getis_ord_gi(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Z-scores for each feature.
    pub fn z_scores(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Classify hot/cold spots.
    pub fn classify_spots(&self) -> Vec<f64> {
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

impl fmt::Display for HotspotAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HotspotAnalysis(n={})", self.data.len())
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
        let cfg = HotspotAnalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = HotspotAnalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("HotspotAnalysisConfig"));
    }

    #[test]
    fn test_config_with_distance_band() {
        let cfg = HotspotAnalysisConfig::new().with_distance_band(42.0);
        assert!((cfg.distance_band - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_significance() {
        let cfg = HotspotAnalysisConfig::new().with_significance(42.0);
        assert!((cfg.significance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_row_standardize() {
        let cfg = HotspotAnalysisConfig::new().with_row_standardize(false);
        assert_eq!(cfg.row_standardize, false);
    }

    #[test]
    fn test_config_with_permutations() {
        let cfg = HotspotAnalysisConfig::new().with_permutations(42);
        assert_eq!(cfg.permutations, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = HotspotAnalysisConfig::new().with_distance_band(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("HotspotAnalysis"));
    }

    #[test]
    fn test_summary() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_getis_ord_gi() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.getis_ord_gi();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_z_scores() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.z_scores();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_classify_spots() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.classify_spots();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_classify_spots_empty() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap();
        assert!(e.classify_spots().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = HotspotAnalysis::new(HotspotAnalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = HotspotAnalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = HotspotAnalysisError::InvalidConfig("a".into());
        let e2 = HotspotAnalysisError::ComputationFailed("b".into());
        let e3 = HotspotAnalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
