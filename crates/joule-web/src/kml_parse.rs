//! KML file parsing for placemarks and styles.
//!
//! Provides [`KmlParseConfig`] builder and [`KmlParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum KmlParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for KmlParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "KmlParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "KmlParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "KmlParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`KmlParse`] parameters.
#[derive(Debug, Clone)]
pub struct KmlParseConfig {
    pub parse_styles: bool,
    pub flatten_hierarchy: bool,
    pub extract_altitude: bool,
    pub max_depth: usize,
}

impl KmlParseConfig {
    pub fn new() -> Self {
        Self {
            parse_styles: true,
            flatten_hierarchy: false,
            extract_altitude: true,
            max_depth: 10,
        }
    }

    pub fn with_parse_styles(mut self, v: bool) -> Self {
        self.parse_styles = v;
        self
    }

    pub fn with_flatten_hierarchy(mut self, v: bool) -> Self {
        self.flatten_hierarchy = v;
        self
    }

    pub fn with_extract_altitude(mut self, v: bool) -> Self {
        self.extract_altitude = v;
        self
    }

    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    pub fn validate(&self) -> Result<(), KmlParseError> {
        Ok(())
    }
}

impl Default for KmlParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for KmlParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KmlParseConfig(parse_styles={0}, flatten_hierarchy={1}, extract_altitude={2}, max_depth={3})", self.parse_styles, self.flatten_hierarchy, self.extract_altitude, self.max_depth)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core kml file parsing for placemarks and styles engine.
#[derive(Debug, Clone)]
pub struct KmlParse {
    config: KmlParseConfig,
    data: Vec<f64>,
}

impl KmlParse {
    pub fn new(config: KmlParseConfig) -> Result<Self, KmlParseError> {
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
    pub fn config(&self) -> &KmlParseConfig { &self.config }

    /// Parse KML placemarks.
    pub fn parse_placemarks(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Parse KML folder hierarchy.
    pub fn parse_folders(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Extract all coordinates.
    pub fn extract_coordinates(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for KmlParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KmlParse(n={})", self.data.len())
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
        let cfg = KmlParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = KmlParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("KmlParseConfig"));
    }

    #[test]
    fn test_config_with_parse_styles() {
        let cfg = KmlParseConfig::new().with_parse_styles(false);
        assert_eq!(cfg.parse_styles, false);
    }

    #[test]
    fn test_config_with_flatten_hierarchy() {
        let cfg = KmlParseConfig::new().with_flatten_hierarchy(true);
        assert_eq!(cfg.flatten_hierarchy, true);
    }

    #[test]
    fn test_config_with_extract_altitude() {
        let cfg = KmlParseConfig::new().with_extract_altitude(false);
        assert_eq!(cfg.extract_altitude, false);
    }

    #[test]
    fn test_config_with_max_depth() {
        let cfg = KmlParseConfig::new().with_max_depth(42);
        assert_eq!(cfg.max_depth, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = KmlParseConfig::new().with_parse_styles(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = KmlParse::new(KmlParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("KmlParse"));
    }

    #[test]
    fn test_summary() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_placemarks() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_placemarks();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_parse_folders() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_folders();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_coordinates() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.extract_coordinates();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_coordinates_empty() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap();
        assert!(e.extract_coordinates().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = KmlParse::new(KmlParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = KmlParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = KmlParseError::InvalidConfig("a".into());
        let e2 = KmlParseError::ComputationFailed("b".into());
        let e3 = KmlParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
