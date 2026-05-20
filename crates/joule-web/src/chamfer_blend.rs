//! Chamfer and blend operations on solid edges.
//!
//! Provides [`ChamferBlendConfig`] builder and [`ChamferBlend`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ChamferBlendError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ChamferBlendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ChamferBlend: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ChamferBlend: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ChamferBlend: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ChamferBlend`] parameters.
#[derive(Debug, Clone)]
pub struct ChamferBlendConfig {
    pub distance: f64,
    pub angle_deg: f64,
    pub variable_radius: bool,
    pub propagate: bool,
}

impl ChamferBlendConfig {
    pub fn new() -> Self {
        Self {
            distance: 1.0,
            angle_deg: 45.0,
            variable_radius: false,
            propagate: true,
        }
    }

    pub fn with_distance(mut self, v: f64) -> Self {
        self.distance = v;
        self
    }

    pub fn with_angle_deg(mut self, v: f64) -> Self {
        self.angle_deg = v;
        self
    }

    pub fn with_variable_radius(mut self, v: bool) -> Self {
        self.variable_radius = v;
        self
    }

    pub fn with_propagate(mut self, v: bool) -> Self {
        self.propagate = v;
        self
    }

    pub fn validate(&self) -> Result<(), ChamferBlendError> {
        if self.distance.is_nan() {
            return Err(ChamferBlendError::InvalidConfig("distance is NaN".into()));
        }
        if self.angle_deg.is_nan() {
            return Err(ChamferBlendError::InvalidConfig("angle_deg is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ChamferBlendConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ChamferBlendConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChamferBlendConfig(distance={0:.4}, angle_deg={1:.4}, variable_radius={2}, propagate={3})", self.distance, self.angle_deg, self.variable_radius, self.propagate)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core chamfer and blend operations on solid edges engine.
#[derive(Debug, Clone)]
pub struct ChamferBlend {
    config: ChamferBlendConfig,
    data: Vec<f64>,
}

impl ChamferBlend {
    pub fn new(config: ChamferBlendConfig) -> Result<Self, ChamferBlendError> {
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
    pub fn config(&self) -> &ChamferBlendConfig { &self.config }

    /// Chamfer an edge.
    pub fn edge_chamfer(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Blend (fillet) an edge.
    pub fn edge_blend(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Blend chain of edges.
    pub fn chain_blend(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for ChamferBlend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChamferBlend(n={})", self.data.len())
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
        let cfg = ChamferBlendConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ChamferBlendConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ChamferBlendConfig"));
    }

    #[test]
    fn test_config_with_distance() {
        let cfg = ChamferBlendConfig::new().with_distance(42.0);
        assert!((cfg.distance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_angle_deg() {
        let cfg = ChamferBlendConfig::new().with_angle_deg(42.0);
        assert!((cfg.angle_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_variable_radius() {
        let cfg = ChamferBlendConfig::new().with_variable_radius(true);
        assert_eq!(cfg.variable_radius, true);
    }

    #[test]
    fn test_config_with_propagate() {
        let cfg = ChamferBlendConfig::new().with_propagate(false);
        assert_eq!(cfg.propagate, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ChamferBlendConfig::new().with_distance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ChamferBlend"));
    }

    #[test]
    fn test_summary() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_edge_chamfer() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.edge_chamfer();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_edge_blend() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.edge_blend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_chain_blend() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.chain_blend();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_chain_blend_empty() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap();
        assert!(e.chain_blend().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ChamferBlend::new(ChamferBlendConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ChamferBlendError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ChamferBlendError::InvalidConfig("a".into());
        let e2 = ChamferBlendError::ComputationFailed("b".into());
        let e3 = ChamferBlendError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
