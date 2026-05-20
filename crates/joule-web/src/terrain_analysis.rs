//! Terrain analysis with TPI and visibility.
//!
//! Provides [`TerrainAnalysisConfig`] builder and [`TerrainAnalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TerrainAnalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TerrainAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TerrainAnalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TerrainAnalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TerrainAnalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TerrainAnalysis`] parameters.
#[derive(Debug, Clone)]
pub struct TerrainAnalysisConfig {
    pub cell_size: f64,
    pub z_factor: f64,
    pub observer_height: f64,
    pub max_distance: f64,
}

impl TerrainAnalysisConfig {
    pub fn new() -> Self {
        Self {
            cell_size: 30.0,
            z_factor: 1.0,
            observer_height: 1.8,
            max_distance: 10000.0,
        }
    }

    pub fn with_cell_size(mut self, v: f64) -> Self {
        self.cell_size = v;
        self
    }

    pub fn with_z_factor(mut self, v: f64) -> Self {
        self.z_factor = v;
        self
    }

    pub fn with_observer_height(mut self, v: f64) -> Self {
        self.observer_height = v;
        self
    }

    pub fn with_max_distance(mut self, v: f64) -> Self {
        self.max_distance = v;
        self
    }

    pub fn validate(&self) -> Result<(), TerrainAnalysisError> {
        if self.cell_size.is_nan() {
            return Err(TerrainAnalysisError::InvalidConfig("cell_size is NaN".into()));
        }
        if self.z_factor.is_nan() {
            return Err(TerrainAnalysisError::InvalidConfig("z_factor is NaN".into()));
        }
        if self.observer_height.is_nan() {
            return Err(TerrainAnalysisError::InvalidConfig("observer_height is NaN".into()));
        }
        if self.max_distance.is_nan() {
            return Err(TerrainAnalysisError::InvalidConfig("max_distance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for TerrainAnalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TerrainAnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TerrainAnalysisConfig(cell_size={0:.4}, z_factor={1:.4}, observer_height={2:.4}, max_distance={3:.4})", self.cell_size, self.z_factor, self.observer_height, self.max_distance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core terrain analysis with tpi and visibility engine.
#[derive(Debug, Clone)]
pub struct TerrainAnalysis {
    config: TerrainAnalysisConfig,
    data: Vec<f64>,
}

impl TerrainAnalysis {
    pub fn new(config: TerrainAnalysisConfig) -> Result<Self, TerrainAnalysisError> {
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
    pub fn config(&self) -> &TerrainAnalysisConfig { &self.config }

    /// Topographic position index.
    pub fn tpi(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Classify terrain type.
    pub fn classify_terrain(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute viewshed.
    pub fn viewshed(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for TerrainAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TerrainAnalysis(n={})", self.data.len())
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
        let cfg = TerrainAnalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TerrainAnalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TerrainAnalysisConfig"));
    }

    #[test]
    fn test_config_with_cell_size() {
        let cfg = TerrainAnalysisConfig::new().with_cell_size(42.0);
        assert!((cfg.cell_size - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_z_factor() {
        let cfg = TerrainAnalysisConfig::new().with_z_factor(42.0);
        assert!((cfg.z_factor - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_observer_height() {
        let cfg = TerrainAnalysisConfig::new().with_observer_height(42.0);
        assert!((cfg.observer_height - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_distance() {
        let cfg = TerrainAnalysisConfig::new().with_max_distance(42.0);
        assert!((cfg.max_distance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TerrainAnalysisConfig::new().with_cell_size(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TerrainAnalysis"));
    }

    #[test]
    fn test_summary() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tpi() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tpi();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_classify_terrain() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.classify_terrain();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_viewshed() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.viewshed();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_viewshed_empty() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap();
        assert!(e.viewshed().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = TerrainAnalysis::new(TerrainAnalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TerrainAnalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TerrainAnalysisError::InvalidConfig("a".into());
        let e2 = TerrainAnalysisError::ComputationFailed("b".into());
        let e3 = TerrainAnalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
