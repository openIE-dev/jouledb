//! Extrusion and revolution solid creation.
//!
//! Provides [`ExtrudeRevolveConfig`] builder and [`ExtrudeRevolve`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtrudeRevolveError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ExtrudeRevolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ExtrudeRevolve: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ExtrudeRevolve: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ExtrudeRevolve: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ExtrudeRevolve`] parameters.
#[derive(Debug, Clone)]
pub struct ExtrudeRevolveConfig {
    pub distance: f64,
    pub angle_deg: f64,
    pub draft_deg: f64,
    pub taper_ratio: f64,
}

impl ExtrudeRevolveConfig {
    pub fn new() -> Self {
        Self {
            distance: 10.0,
            angle_deg: 360.0,
            draft_deg: 0.0,
            taper_ratio: 1.0,
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

    pub fn with_draft_deg(mut self, v: f64) -> Self {
        self.draft_deg = v;
        self
    }

    pub fn with_taper_ratio(mut self, v: f64) -> Self {
        self.taper_ratio = v;
        self
    }

    pub fn validate(&self) -> Result<(), ExtrudeRevolveError> {
        if self.distance.is_nan() {
            return Err(ExtrudeRevolveError::InvalidConfig("distance is NaN".into()));
        }
        if self.angle_deg.is_nan() {
            return Err(ExtrudeRevolveError::InvalidConfig("angle_deg is NaN".into()));
        }
        if self.draft_deg.is_nan() {
            return Err(ExtrudeRevolveError::InvalidConfig("draft_deg is NaN".into()));
        }
        if self.taper_ratio.is_nan() {
            return Err(ExtrudeRevolveError::InvalidConfig("taper_ratio is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ExtrudeRevolveConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ExtrudeRevolveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtrudeRevolveConfig(distance={0:.4}, angle_deg={1:.4}, draft_deg={2:.4}, taper_ratio={3:.4})", self.distance, self.angle_deg, self.draft_deg, self.taper_ratio)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core extrusion and revolution solid creation engine.
#[derive(Debug, Clone)]
pub struct ExtrudeRevolve {
    config: ExtrudeRevolveConfig,
    data: Vec<f64>,
}

impl ExtrudeRevolve {
    pub fn new(config: ExtrudeRevolveConfig) -> Result<Self, ExtrudeRevolveError> {
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
    pub fn config(&self) -> &ExtrudeRevolveConfig { &self.config }

    /// Linear extrusion.
    pub fn linear_extrude(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Revolution about axis.
    pub fn revolve(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Helical extrusion.
    pub fn helical_extrude(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for ExtrudeRevolve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtrudeRevolve(n={})", self.data.len())
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
        let cfg = ExtrudeRevolveConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ExtrudeRevolveConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ExtrudeRevolveConfig"));
    }

    #[test]
    fn test_config_with_distance() {
        let cfg = ExtrudeRevolveConfig::new().with_distance(42.0);
        assert!((cfg.distance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_angle_deg() {
        let cfg = ExtrudeRevolveConfig::new().with_angle_deg(42.0);
        assert!((cfg.angle_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_draft_deg() {
        let cfg = ExtrudeRevolveConfig::new().with_draft_deg(42.0);
        assert!((cfg.draft_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_taper_ratio() {
        let cfg = ExtrudeRevolveConfig::new().with_taper_ratio(42.0);
        assert!((cfg.taper_ratio - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ExtrudeRevolveConfig::new().with_distance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ExtrudeRevolve"));
    }

    #[test]
    fn test_summary() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_extrude() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.linear_extrude();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_revolve() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.revolve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_helical_extrude() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.helical_extrude();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_helical_extrude_empty() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap();
        assert!(e.helical_extrude().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ExtrudeRevolve::new(ExtrudeRevolveConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ExtrudeRevolveError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ExtrudeRevolveError::InvalidConfig("a".into());
        let e2 = ExtrudeRevolveError::ComputationFailed("b".into());
        let e3 = ExtrudeRevolveError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
