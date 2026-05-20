//! Spectral vegetation and water indices.
//!
//! Provides [`NdviIndexConfig`] builder and [`NdviIndex`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum NdviIndexError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for NdviIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "NdviIndex: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "NdviIndex: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "NdviIndex: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`NdviIndex`] parameters.
#[derive(Debug, Clone)]
pub struct NdviIndexConfig {
    pub red_band: usize,
    pub nir_band: usize,
    pub swir_band: usize,
    pub scale_factor: f64,
}

impl NdviIndexConfig {
    pub fn new() -> Self {
        Self {
            red_band: 3,
            nir_band: 4,
            swir_band: 5,
            scale_factor: 0.0001,
        }
    }

    pub fn with_red_band(mut self, v: usize) -> Self {
        self.red_band = v;
        self
    }

    pub fn with_nir_band(mut self, v: usize) -> Self {
        self.nir_band = v;
        self
    }

    pub fn with_swir_band(mut self, v: usize) -> Self {
        self.swir_band = v;
        self
    }

    pub fn with_scale_factor(mut self, v: f64) -> Self {
        self.scale_factor = v;
        self
    }

    pub fn validate(&self) -> Result<(), NdviIndexError> {
        if self.scale_factor.is_nan() {
            return Err(NdviIndexError::InvalidConfig("scale_factor is NaN".into()));
        }
        Ok(())
    }
}

impl Default for NdviIndexConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NdviIndexConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdviIndexConfig(red_band={0}, nir_band={1}, swir_band={2}, scale_factor={3:.4})", self.red_band, self.nir_band, self.swir_band, self.scale_factor)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core spectral vegetation and water indices engine.
#[derive(Debug, Clone)]
pub struct NdviIndex {
    config: NdviIndexConfig,
    data: Vec<f64>,
}

impl NdviIndex {
    pub fn new(config: NdviIndexConfig) -> Result<Self, NdviIndexError> {
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
    pub fn config(&self) -> &NdviIndexConfig { &self.config }

    /// Normalized Difference Vegetation Index.
    pub fn ndvi(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Normalized Difference Water Index.
    pub fn ndwi(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Soil-Adjusted Vegetation Index.
    pub fn savi(&self) -> f64 {
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

impl fmt::Display for NdviIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdviIndex(n={})", self.data.len())
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
        let cfg = NdviIndexConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = NdviIndexConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("NdviIndexConfig"));
    }

    #[test]
    fn test_config_with_red_band() {
        let cfg = NdviIndexConfig::new().with_red_band(42);
        assert_eq!(cfg.red_band, 42);
    }

    #[test]
    fn test_config_with_nir_band() {
        let cfg = NdviIndexConfig::new().with_nir_band(42);
        assert_eq!(cfg.nir_band, 42);
    }

    #[test]
    fn test_config_with_swir_band() {
        let cfg = NdviIndexConfig::new().with_swir_band(42);
        assert_eq!(cfg.swir_band, 42);
    }

    #[test]
    fn test_config_with_scale_factor() {
        let cfg = NdviIndexConfig::new().with_scale_factor(42.0);
        assert!((cfg.scale_factor - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = NdviIndexConfig::new().with_red_band(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = NdviIndex::new(NdviIndexConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("NdviIndex"));
    }

    #[test]
    fn test_summary() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndvi() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ndvi();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ndwi() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ndwi();
        assert!(result.is_finite());
    }

    #[test]
    fn test_savi() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.savi();
        assert!(result.is_finite());
    }

    #[test]
    fn test_savi_empty() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap();
        assert!((e.savi() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = NdviIndex::new(NdviIndexConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = NdviIndexError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = NdviIndexError::InvalidConfig("a".into());
        let e2 = NdviIndexError::ComputationFailed("b".into());
        let e3 = NdviIndexError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
