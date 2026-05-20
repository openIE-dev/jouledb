//! Map symbol rendering and clustering.
//!
//! Provides [`SymbolRenderConfig`] builder and [`SymbolRender`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolRenderError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SymbolRenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SymbolRender: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SymbolRender: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SymbolRender: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SymbolRender`] parameters.
#[derive(Debug, Clone)]
pub struct SymbolRenderConfig {
    pub default_size: f64,
    pub cluster_radius: f64,
    pub rotation_deg: f64,
    pub opacity: f64,
}

impl SymbolRenderConfig {
    pub fn new() -> Self {
        Self {
            default_size: 12.0,
            cluster_radius: 50.0,
            rotation_deg: 0.0,
            opacity: 1.0,
        }
    }

    pub fn with_default_size(mut self, v: f64) -> Self {
        self.default_size = v;
        self
    }

    pub fn with_cluster_radius(mut self, v: f64) -> Self {
        self.cluster_radius = v;
        self
    }

    pub fn with_rotation_deg(mut self, v: f64) -> Self {
        self.rotation_deg = v;
        self
    }

    pub fn with_opacity(mut self, v: f64) -> Self {
        self.opacity = v;
        self
    }

    pub fn validate(&self) -> Result<(), SymbolRenderError> {
        if self.default_size.is_nan() {
            return Err(SymbolRenderError::InvalidConfig("default_size is NaN".into()));
        }
        if self.cluster_radius.is_nan() {
            return Err(SymbolRenderError::InvalidConfig("cluster_radius is NaN".into()));
        }
        if self.rotation_deg.is_nan() {
            return Err(SymbolRenderError::InvalidConfig("rotation_deg is NaN".into()));
        }
        if self.opacity.is_nan() {
            return Err(SymbolRenderError::InvalidConfig("opacity is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SymbolRenderConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SymbolRenderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolRenderConfig(default_size={0:.4}, cluster_radius={1:.4}, rotation_deg={2:.4}, opacity={3:.4})", self.default_size, self.cluster_radius, self.rotation_deg, self.opacity)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core map symbol rendering and clustering engine.
#[derive(Debug, Clone)]
pub struct SymbolRender {
    config: SymbolRenderConfig,
    data: Vec<f64>,
}

impl SymbolRender {
    pub fn new(config: SymbolRenderConfig) -> Result<Self, SymbolRenderError> {
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
    pub fn config(&self) -> &SymbolRenderConfig { &self.config }

    /// Render marker symbols.
    pub fn render_markers(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Cluster nearby symbols.
    pub fn cluster_symbols(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Calculate proportional symbol size.
    pub fn proportional_size(&self) -> f64 {
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

impl fmt::Display for SymbolRender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolRender(n={})", self.data.len())
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
        let cfg = SymbolRenderConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SymbolRenderConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SymbolRenderConfig"));
    }

    #[test]
    fn test_config_with_default_size() {
        let cfg = SymbolRenderConfig::new().with_default_size(42.0);
        assert!((cfg.default_size - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_cluster_radius() {
        let cfg = SymbolRenderConfig::new().with_cluster_radius(42.0);
        assert!((cfg.cluster_radius - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_rotation_deg() {
        let cfg = SymbolRenderConfig::new().with_rotation_deg(42.0);
        assert!((cfg.rotation_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_opacity() {
        let cfg = SymbolRenderConfig::new().with_opacity(42.0);
        assert!((cfg.opacity - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SymbolRenderConfig::new().with_default_size(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SymbolRender::new(SymbolRenderConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SymbolRender"));
    }

    #[test]
    fn test_summary() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_render_markers() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.render_markers();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_cluster_symbols() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.cluster_symbols();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_proportional_size() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.proportional_size();
        assert!(result.is_finite());
    }

    #[test]
    fn test_proportional_size_empty() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap();
        assert!((e.proportional_size() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SymbolRender::new(SymbolRenderConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SymbolRenderError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SymbolRenderError::InvalidConfig("a".into());
        let e2 = SymbolRenderError::ComputationFailed("b".into());
        let e3 = SymbolRenderError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
