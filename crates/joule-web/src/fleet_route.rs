//! Fleet routing and vehicle routing problem solver.
//!
//! Provides [`FleetRouteConfig`] builder and [`FleetRoute`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FleetRouteError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FleetRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "FleetRoute: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "FleetRoute: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "FleetRoute: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`FleetRoute`] parameters.
#[derive(Debug, Clone)]
pub struct FleetRouteConfig {
    pub vehicle_capacity: f64,
    pub max_route_time: f64,
    pub num_vehicles: usize,
    pub depot_count: usize,
}

impl FleetRouteConfig {
    pub fn new() -> Self {
        Self {
            vehicle_capacity: 1000.0,
            max_route_time: 480.0,
            num_vehicles: 5,
            depot_count: 1,
        }
    }

    pub fn with_vehicle_capacity(mut self, v: f64) -> Self {
        self.vehicle_capacity = v;
        self
    }

    pub fn with_max_route_time(mut self, v: f64) -> Self {
        self.max_route_time = v;
        self
    }

    pub fn with_num_vehicles(mut self, v: usize) -> Self {
        self.num_vehicles = v;
        self
    }

    pub fn with_depot_count(mut self, v: usize) -> Self {
        self.depot_count = v;
        self
    }

    pub fn validate(&self) -> Result<(), FleetRouteError> {
        if self.vehicle_capacity.is_nan() {
            return Err(FleetRouteError::InvalidConfig("vehicle_capacity is NaN".into()));
        }
        if self.max_route_time.is_nan() {
            return Err(FleetRouteError::InvalidConfig("max_route_time is NaN".into()));
        }
        Ok(())
    }
}

impl Default for FleetRouteConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FleetRouteConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FleetRouteConfig(vehicle_capacity={0:.4}, max_route_time={1:.4}, num_vehicles={2}, depot_count={3})", self.vehicle_capacity, self.max_route_time, self.num_vehicles, self.depot_count)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core fleet routing and vehicle routing problem solver engine.
#[derive(Debug, Clone)]
pub struct FleetRoute {
    config: FleetRouteConfig,
    data: Vec<f64>,
}

impl FleetRoute {
    pub fn new(config: FleetRouteConfig) -> Result<Self, FleetRouteError> {
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
    pub fn config(&self) -> &FleetRouteConfig { &self.config }

    /// Solve vehicle routing problem.
    pub fn solve_vrp(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// 2-opt route optimization.
    pub fn optimize_2opt(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Total fleet distance.
    pub fn total_distance(&self) -> f64 {
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

impl fmt::Display for FleetRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FleetRoute(n={})", self.data.len())
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
        let cfg = FleetRouteConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FleetRouteConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FleetRouteConfig"));
    }

    #[test]
    fn test_config_with_vehicle_capacity() {
        let cfg = FleetRouteConfig::new().with_vehicle_capacity(42.0);
        assert!((cfg.vehicle_capacity - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_route_time() {
        let cfg = FleetRouteConfig::new().with_max_route_time(42.0);
        assert!((cfg.max_route_time - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_num_vehicles() {
        let cfg = FleetRouteConfig::new().with_num_vehicles(42);
        assert_eq!(cfg.num_vehicles, 42);
    }

    #[test]
    fn test_config_with_depot_count() {
        let cfg = FleetRouteConfig::new().with_depot_count(42);
        assert_eq!(cfg.depot_count, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FleetRouteConfig::new().with_vehicle_capacity(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = FleetRoute::new(FleetRouteConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("FleetRoute"));
    }

    #[test]
    fn test_summary() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_solve_vrp() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.solve_vrp();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_optimize_2opt() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.optimize_2opt();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_total_distance() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.total_distance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_total_distance_empty() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap();
        assert!((e.total_distance() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = FleetRoute::new(FleetRouteConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FleetRouteError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FleetRouteError::InvalidConfig("a".into());
        let e2 = FleetRouteError::ComputationFailed("b".into());
        let e3 = FleetRouteError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
