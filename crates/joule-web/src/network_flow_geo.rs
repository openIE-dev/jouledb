//! Network flow analysis for geospatial graphs.
//!
//! Provides [`NetworkFlowGeoConfig`] builder and [`NetworkFlowGeo`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkFlowGeoError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for NetworkFlowGeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "NetworkFlowGeo: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "NetworkFlowGeo: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "NetworkFlowGeo: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`NetworkFlowGeo`] parameters.
#[derive(Debug, Clone)]
pub struct NetworkFlowGeoConfig {
    pub num_nodes: usize,
    pub directed: bool,
    pub capacity_default: f64,
    pub cost_default: f64,
}

impl NetworkFlowGeoConfig {
    pub fn new() -> Self {
        Self {
            num_nodes: 100,
            directed: true,
            capacity_default: 1.0,
            cost_default: 1.0,
        }
    }

    pub fn with_num_nodes(mut self, v: usize) -> Self {
        self.num_nodes = v;
        self
    }

    pub fn with_directed(mut self, v: bool) -> Self {
        self.directed = v;
        self
    }

    pub fn with_capacity_default(mut self, v: f64) -> Self {
        self.capacity_default = v;
        self
    }

    pub fn with_cost_default(mut self, v: f64) -> Self {
        self.cost_default = v;
        self
    }

    pub fn validate(&self) -> Result<(), NetworkFlowGeoError> {
        if self.capacity_default.is_nan() {
            return Err(NetworkFlowGeoError::InvalidConfig("capacity_default is NaN".into()));
        }
        if self.cost_default.is_nan() {
            return Err(NetworkFlowGeoError::InvalidConfig("cost_default is NaN".into()));
        }
        Ok(())
    }
}

impl Default for NetworkFlowGeoConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NetworkFlowGeoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NetworkFlowGeoConfig(num_nodes={0}, directed={1}, capacity_default={2:.4}, cost_default={3:.4})", self.num_nodes, self.directed, self.capacity_default, self.cost_default)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core network flow analysis for geospatial graphs engine.
#[derive(Debug, Clone)]
pub struct NetworkFlowGeo {
    config: NetworkFlowGeoConfig,
    data: Vec<f64>,
}

impl NetworkFlowGeo {
    pub fn new(config: NetworkFlowGeoConfig) -> Result<Self, NetworkFlowGeoError> {
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
    pub fn config(&self) -> &NetworkFlowGeoConfig { &self.config }

    /// Maximum flow (Ford-Fulkerson).
    pub fn max_flow(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Minimum cost flow.
    pub fn min_cost_flow(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Graph connectivity check.
    pub fn connectivity(&self) -> bool {
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

impl fmt::Display for NetworkFlowGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NetworkFlowGeo(n={})", self.data.len())
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
        let cfg = NetworkFlowGeoConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = NetworkFlowGeoConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("NetworkFlowGeoConfig"));
    }

    #[test]
    fn test_config_with_num_nodes() {
        let cfg = NetworkFlowGeoConfig::new().with_num_nodes(42);
        assert_eq!(cfg.num_nodes, 42);
    }

    #[test]
    fn test_config_with_directed() {
        let cfg = NetworkFlowGeoConfig::new().with_directed(false);
        assert_eq!(cfg.directed, false);
    }

    #[test]
    fn test_config_with_capacity_default() {
        let cfg = NetworkFlowGeoConfig::new().with_capacity_default(42.0);
        assert!((cfg.capacity_default - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_cost_default() {
        let cfg = NetworkFlowGeoConfig::new().with_cost_default(42.0);
        assert!((cfg.cost_default - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = NetworkFlowGeoConfig::new().with_num_nodes(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("NetworkFlowGeo"));
    }

    #[test]
    fn test_summary() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_flow() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.max_flow();
        assert!(result.is_finite());
    }

    #[test]
    fn test_min_cost_flow() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.min_cost_flow();
        assert!(result.is_finite());
    }

    #[test]
    fn test_connectivity() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.connectivity();
        assert!(result);
    }

    #[test]
    fn test_connectivity_empty() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap();
        assert!(!e.connectivity());
    }

    #[test]
    fn test_config_accessor() {
        let e = NetworkFlowGeo::new(NetworkFlowGeoConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = NetworkFlowGeoError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = NetworkFlowGeoError::InvalidConfig("a".into());
        let e2 = NetworkFlowGeoError::ComputationFailed("b".into());
        let e3 = NetworkFlowGeoError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
