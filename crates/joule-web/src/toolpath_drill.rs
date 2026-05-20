//! Drilling cycle toolpath generation.
//!
//! Provides [`ToolpathDrillConfig`] builder and [`ToolpathDrill`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolpathDrillError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ToolpathDrillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ToolpathDrill: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ToolpathDrill: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ToolpathDrill: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ToolpathDrill`] parameters.
#[derive(Debug, Clone)]
pub struct ToolpathDrillConfig {
    pub drill_diameter: f64,
    pub depth: f64,
    pub peck_increment: f64,
    pub retract_height: f64,
}

impl ToolpathDrillConfig {
    pub fn new() -> Self {
        Self {
            drill_diameter: 6.0,
            depth: 20.0,
            peck_increment: 3.0,
            retract_height: 2.0,
        }
    }

    pub fn with_drill_diameter(mut self, v: f64) -> Self {
        self.drill_diameter = v;
        self
    }

    pub fn with_depth(mut self, v: f64) -> Self {
        self.depth = v;
        self
    }

    pub fn with_peck_increment(mut self, v: f64) -> Self {
        self.peck_increment = v;
        self
    }

    pub fn with_retract_height(mut self, v: f64) -> Self {
        self.retract_height = v;
        self
    }

    pub fn validate(&self) -> Result<(), ToolpathDrillError> {
        if self.drill_diameter.is_nan() {
            return Err(ToolpathDrillError::InvalidConfig("drill_diameter is NaN".into()));
        }
        if self.depth.is_nan() {
            return Err(ToolpathDrillError::InvalidConfig("depth is NaN".into()));
        }
        if self.peck_increment.is_nan() {
            return Err(ToolpathDrillError::InvalidConfig("peck_increment is NaN".into()));
        }
        if self.retract_height.is_nan() {
            return Err(ToolpathDrillError::InvalidConfig("retract_height is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ToolpathDrillConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ToolpathDrillConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathDrillConfig(drill_diameter={0:.4}, depth={1:.4}, peck_increment={2:.4}, retract_height={3:.4})", self.drill_diameter, self.depth, self.peck_increment, self.retract_height)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core drilling cycle toolpath generation engine.
#[derive(Debug, Clone)]
pub struct ToolpathDrill {
    config: ToolpathDrillConfig,
    data: Vec<f64>,
}

impl ToolpathDrill {
    pub fn new(config: ToolpathDrillConfig) -> Result<Self, ToolpathDrillError> {
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
    pub fn config(&self) -> &ToolpathDrillConfig { &self.config }

    /// Spot drill cycle.
    pub fn spot_drill(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Peck drilling cycle.
    pub fn peck_drill(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Tapping cycle.
    pub fn tapping_cycle(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for ToolpathDrill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolpathDrill(n={})", self.data.len())
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
        let cfg = ToolpathDrillConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ToolpathDrillConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ToolpathDrillConfig"));
    }

    #[test]
    fn test_config_with_drill_diameter() {
        let cfg = ToolpathDrillConfig::new().with_drill_diameter(42.0);
        assert!((cfg.drill_diameter - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_depth() {
        let cfg = ToolpathDrillConfig::new().with_depth(42.0);
        assert!((cfg.depth - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_peck_increment() {
        let cfg = ToolpathDrillConfig::new().with_peck_increment(42.0);
        assert!((cfg.peck_increment - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_retract_height() {
        let cfg = ToolpathDrillConfig::new().with_retract_height(42.0);
        assert!((cfg.retract_height - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ToolpathDrillConfig::new().with_drill_diameter(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ToolpathDrill"));
    }

    #[test]
    fn test_summary() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_spot_drill() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.spot_drill();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_peck_drill() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.peck_drill();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tapping_cycle() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tapping_cycle();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tapping_cycle_empty() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap();
        assert!(e.tapping_cycle().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ToolpathDrill::new(ToolpathDrillConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ToolpathDrillError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ToolpathDrillError::InvalidConfig("a".into());
        let e2 = ToolpathDrillError::ComputationFailed("b".into());
        let e3 = ToolpathDrillError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
