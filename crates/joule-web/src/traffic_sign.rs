//! Traffic Sign Recognition — HOG (Histogram of Oriented Gradients) feature
//! extraction, template matching, sign classification, and speed limit
//! extraction.
//!
//! Processes image patches to classify common traffic signs (stop, yield,
//! speed limit, no entry, etc.) using gradient-based features and
//! distance-based matching against a template library.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Traffic sign recognition errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SignError {
    /// Image patch is too small for feature extraction.
    PatchTooSmall(String),
    /// Template library is empty.
    EmptyLibrary,
    /// No match found above confidence threshold.
    NoMatch,
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for SignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PatchTooSmall(m) => write!(f, "patch too small: {m}"),
            Self::EmptyLibrary => write!(f, "template library is empty"),
            Self::NoMatch => write!(f, "no match above threshold"),
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
        }
    }
}

impl std::error::Error for SignError {}

// ── Sign Classes ────────────────────────────────────────────────

/// Known traffic sign classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignClass {
    Stop,
    Yield,
    SpeedLimit(u32),
    NoEntry,
    OneWay,
    PedestrianCrossing,
    SchoolZone,
    RoadWork,
    NoOvertaking,
    RightOfWay,
    Unknown,
}

impl fmt::Display for SignClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "STOP"),
            Self::Yield => write!(f, "YIELD"),
            Self::SpeedLimit(v) => write!(f, "SPEED_LIMIT_{v}"),
            Self::NoEntry => write!(f, "NO_ENTRY"),
            Self::OneWay => write!(f, "ONE_WAY"),
            Self::PedestrianCrossing => write!(f, "PEDESTRIAN_CROSSING"),
            Self::SchoolZone => write!(f, "SCHOOL_ZONE"),
            Self::RoadWork => write!(f, "ROAD_WORK"),
            Self::NoOvertaking => write!(f, "NO_OVERTAKING"),
            Self::RightOfWay => write!(f, "RIGHT_OF_WAY"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ── Image Patch ─────────────────────────────────────────────────

/// A grayscale image patch for processing.
#[derive(Debug, Clone)]
pub struct ImagePatch {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f64>,
}

impl ImagePatch {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, pixels: vec![0.0; width * height] }
    }

    pub fn from_data(width: usize, height: usize, pixels: Vec<f64>) -> Result<Self, SignError> {
        if width < 8 || height < 8 {
            return Err(SignError::PatchTooSmall(format!("{width}x{height}")));
        }
        if pixels.len() != width * height {
            return Err(SignError::InvalidConfig(format!(
                "expected {} pixels, got {}",
                width * height,
                pixels.len()
            )));
        }
        Ok(Self { width, height, pixels })
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> f64 {
        self.pixels[y * self.width + x]
    }

    /// Resize to target dimensions using nearest-neighbour interpolation.
    pub fn resize(&self, tw: usize, th: usize) -> Self {
        let mut out = ImagePatch::new(tw, th);
        for y in 0..th {
            for x in 0..tw {
                let sx = x * self.width / tw;
                let sy = y * self.height / th;
                out.pixels[y * tw + x] = self.get(sx, sy);
            }
        }
        out
    }
}

impl fmt::Display for ImagePatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Patch({}x{})", self.width, self.height)
    }
}

// ── HOG Feature Extractor ───────────────────────────────────────

/// Histogram of Oriented Gradients feature extractor.
///
/// Divides the image into cells, computes gradient magnitude and orientation
/// histograms per cell, then normalises across blocks.
#[derive(Debug, Clone)]
pub struct HogExtractor {
    cell_size: usize,
    block_size: usize,
    num_bins: usize,
    target_size: usize,
}

impl HogExtractor {
    pub fn new() -> Self {
        Self { cell_size: 8, block_size: 2, num_bins: 9, target_size: 64 }
    }

    pub fn with_cell_size(mut self, s: usize) -> Self {
        self.cell_size = s.max(2);
        self
    }

    pub fn with_block_size(mut self, s: usize) -> Self {
        self.block_size = s.max(1);
        self
    }

    pub fn with_num_bins(mut self, b: usize) -> Self {
        self.num_bins = b.max(2);
        self
    }

    pub fn with_target_size(mut self, s: usize) -> Self {
        self.target_size = s.max(16);
        self
    }

    /// Extract HOG feature vector from an image patch.
    pub fn extract(&self, patch: &ImagePatch) -> Result<Vec<f64>, SignError> {
        if patch.width < self.cell_size * 2 || patch.height < self.cell_size * 2 {
            return Err(SignError::PatchTooSmall(format!(
                "{}x{} < {}",
                patch.width,
                patch.height,
                self.cell_size * 2
            )));
        }

        // Resize to standard size.
        let resized = patch.resize(self.target_size, self.target_size);

        // Compute gradients.
        let (mag, orient) = self.compute_gradients(&resized);

        // Build cell histograms.
        let cells_x = self.target_size / self.cell_size;
        let cells_y = self.target_size / self.cell_size;
        let mut cell_hists = vec![vec![0.0f64; self.num_bins]; cells_x * cells_y];

        for cy in 0..cells_y {
            for cx in 0..cells_x {
                let hist = &mut cell_hists[cy * cells_x + cx];
                for dy in 0..self.cell_size {
                    for dx in 0..self.cell_size {
                        let px = cx * self.cell_size + dx;
                        let py = cy * self.cell_size + dy;
                        let idx = py * self.target_size + px;
                        let angle = orient[idx];
                        let m = mag[idx];
                        // Map angle [0, π) to bin.
                        let bin_f = angle / std::f64::consts::PI * self.num_bins as f64;
                        let bin = (bin_f as usize).min(self.num_bins - 1);
                        hist[bin] += m;
                    }
                }
            }
        }

        // Block normalisation (L2-norm).
        let mut features = Vec::new();
        let blocks_x = cells_x.saturating_sub(self.block_size - 1);
        let blocks_y = cells_y.saturating_sub(self.block_size - 1);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut block_vec = Vec::new();
                for dy in 0..self.block_size {
                    for dx in 0..self.block_size {
                        let ci = (by + dy) * cells_x + (bx + dx);
                        block_vec.extend_from_slice(&cell_hists[ci]);
                    }
                }
                let norm = block_vec.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-6;
                for v in &block_vec {
                    features.push(v / norm);
                }
            }
        }

        Ok(features)
    }

    fn compute_gradients(&self, patch: &ImagePatch) -> (Vec<f64>, Vec<f64>) {
        let w = patch.width;
        let h = patch.height;
        let mut mag = vec![0.0f64; w * h];
        let mut orient = vec![0.0f64; w * h];

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = patch.get(x + 1, y) - patch.get(x - 1, y);
                let gy = patch.get(x, y + 1) - patch.get(x, y - 1);
                let idx = y * w + x;
                mag[idx] = (gx * gx + gy * gy).sqrt();
                let mut angle = gy.atan2(gx);
                if angle < 0.0 {
                    angle += std::f64::consts::PI;
                }
                orient[idx] = angle.min(std::f64::consts::PI - 1e-9);
            }
        }
        (mag, orient)
    }
}

impl fmt::Display for HogExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HOG(cell={}, block={}, bins={}, size={})",
            self.cell_size, self.block_size, self.num_bins, self.target_size
        )
    }
}

// ── Template ────────────────────────────────────────────────────

/// A sign template: a reference HOG feature vector with a class label.
#[derive(Debug, Clone)]
pub struct SignTemplate {
    pub class: SignClass,
    pub features: Vec<f64>,
}

impl SignTemplate {
    pub fn new(class: SignClass, features: Vec<f64>) -> Self {
        Self { class, features }
    }
}

impl fmt::Display for SignTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Template({}, dim={})", self.class, self.features.len())
    }
}

// ── Template Matcher ────────────────────────────────────────────

/// Distance metric for template matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    ChiSquared,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::Cosine => write!(f, "Cosine"),
            Self::ChiSquared => write!(f, "Chi-Squared"),
        }
    }
}

/// Matches HOG features against a template library.
#[derive(Debug, Clone)]
pub struct TemplateMatcher {
    metric: DistanceMetric,
    confidence_threshold: f64,
}

impl TemplateMatcher {
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric, confidence_threshold: 0.6 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.confidence_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Match a feature vector against the template library.
    /// Returns (class, confidence) if above threshold.
    pub fn classify(
        &self,
        features: &[f64],
        templates: &[SignTemplate],
    ) -> Result<(SignClass, f64), SignError> {
        if templates.is_empty() {
            return Err(SignError::EmptyLibrary);
        }

        let mut best_class = SignClass::Unknown;
        let mut best_conf = 0.0f64;

        for tmpl in templates {
            let sim = self.similarity(features, &tmpl.features);
            if sim > best_conf {
                best_conf = sim;
                best_class = tmpl.class;
            }
        }

        if best_conf >= self.confidence_threshold {
            Ok((best_class, best_conf))
        } else {
            Err(SignError::NoMatch)
        }
    }

    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        match self.metric {
            DistanceMetric::Euclidean => {
                let dist: f64 = (0..n).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
                1.0 / (1.0 + dist)
            }
            DistanceMetric::Cosine => {
                let dot: f64 = (0..n).map(|i| a[i] * b[i]).sum();
                let na = (0..n).map(|i| a[i] * a[i]).sum::<f64>().sqrt();
                let nb = (0..n).map(|i| b[i] * b[i]).sum::<f64>().sqrt();
                if na < 1e-12 || nb < 1e-12 {
                    return 0.0;
                }
                (dot / (na * nb)).max(0.0)
            }
            DistanceMetric::ChiSquared => {
                let chi: f64 = (0..n)
                    .map(|i| {
                        let denom = a[i] + b[i];
                        if denom.abs() < 1e-12 {
                            0.0
                        } else {
                            (a[i] - b[i]).powi(2) / denom
                        }
                    })
                    .sum();
                1.0 / (1.0 + chi)
            }
        }
    }
}

impl fmt::Display for TemplateMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TemplateMatcher({}, thresh={:.2})",
            self.metric, self.confidence_threshold
        )
    }
}

// ── Speed Limit Extractor ───────────────────────────────────────

/// Extracts numeric speed limits from sign classification results.
#[derive(Debug, Clone)]
pub struct SpeedLimitExtractor {
    valid_limits: Vec<u32>,
    unit: SpeedUnit,
}

/// Speed unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedUnit {
    Kmh,
    Mph,
}

impl fmt::Display for SpeedUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kmh => write!(f, "km/h"),
            Self::Mph => write!(f, "mph"),
        }
    }
}

impl SpeedLimitExtractor {
    pub fn new(unit: SpeedUnit) -> Self {
        let valid = match unit {
            SpeedUnit::Kmh => vec![20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
            SpeedUnit::Mph => vec![15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        };
        Self { valid_limits: valid, unit }
    }

    pub fn with_valid_limits(mut self, limits: Vec<u32>) -> Self {
        self.valid_limits = limits;
        self
    }

    /// Extract speed limit from a sign class, if it's a speed limit sign.
    pub fn extract(&self, class: SignClass) -> Option<SpeedLimitInfo> {
        if let SignClass::SpeedLimit(v) = class {
            let snapped = self.snap_to_valid(v);
            Some(SpeedLimitInfo { value: snapped, unit: self.unit, raw_value: v })
        } else {
            None
        }
    }

    /// Snap a raw value to the nearest valid speed limit.
    fn snap_to_valid(&self, raw: u32) -> u32 {
        let mut best = raw;
        let mut best_diff = u32::MAX;
        for &v in &self.valid_limits {
            let diff = if raw > v { raw - v } else { v - raw };
            if diff < best_diff {
                best_diff = diff;
                best = v;
            }
        }
        best
    }

    /// Aggregate detections: use a simple voting scheme.
    pub fn aggregate(&self, detections: &[SignClass]) -> HashMap<SignClass, usize> {
        let mut counts: HashMap<SignClass, usize> = HashMap::new();
        for &det in detections {
            *counts.entry(det).or_insert(0) += 1;
        }
        counts
    }
}

impl fmt::Display for SpeedLimitExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpeedLimitExtractor(unit={})", self.unit)
    }
}

/// Extracted speed limit information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedLimitInfo {
    pub value: u32,
    pub unit: SpeedUnit,
    pub raw_value: u32,
}

impl fmt::Display for SpeedLimitInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_patch(w: usize, h: usize) -> ImagePatch {
        let mut pixels = vec![0.0; w * h];
        for (i, p) in pixels.iter_mut().enumerate() {
            *p = (i % 256) as f64;
        }
        ImagePatch::from_data(w, h, pixels).unwrap()
    }

    #[test]
    fn test_sign_class_display() {
        assert_eq!(format!("{}", SignClass::Stop), "STOP");
        assert_eq!(format!("{}", SignClass::SpeedLimit(60)), "SPEED_LIMIT_60");
    }

    #[test]
    fn test_image_patch_resize() {
        let patch = test_patch(32, 32);
        let resized = patch.resize(16, 16);
        assert_eq!(resized.width, 16);
        assert_eq!(resized.height, 16);
    }

    #[test]
    fn test_patch_too_small() {
        let r = ImagePatch::from_data(4, 4, vec![0.0; 16]);
        assert!(r.is_err());
    }

    #[test]
    fn test_patch_display() {
        let patch = test_patch(32, 32);
        assert!(format!("{patch}").contains("32x32"));
    }

    #[test]
    fn test_hog_extract() {
        let patch = test_patch(64, 64);
        let hog = HogExtractor::new();
        let features = hog.extract(&patch).unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_hog_small_patch() {
        let patch = test_patch(8, 8);
        let hog = HogExtractor::new().with_cell_size(8);
        // 8x8 patch with cell_size=8 → too small after resize checks
        let result = hog.extract(&patch);
        // May succeed or fail depending on resize; just no panic.
        let _ = result;
    }

    #[test]
    fn test_hog_display() {
        let hog = HogExtractor::new();
        assert!(format!("{hog}").contains("HOG"));
    }

    #[test]
    fn test_template_display() {
        let t = SignTemplate::new(SignClass::Stop, vec![1.0, 2.0, 3.0]);
        assert!(format!("{t}").contains("STOP"));
    }

    #[test]
    fn test_euclidean_similarity_identical() {
        let matcher = TemplateMatcher::new(DistanceMetric::Euclidean).with_threshold(0.0);
        let feat = vec![1.0, 2.0, 3.0];
        let tmpl = vec![SignTemplate::new(SignClass::Stop, feat.clone())];
        let (cls, conf) = matcher.classify(&feat, &tmpl).unwrap();
        assert_eq!(cls, SignClass::Stop);
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let matcher = TemplateMatcher::new(DistanceMetric::Cosine).with_threshold(0.0);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let tmpl = vec![SignTemplate::new(SignClass::Yield, b)];
        let (cls, conf) = matcher.classify(&a, &tmpl).unwrap();
        assert_eq!(cls, SignClass::Yield);
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared() {
        let matcher = TemplateMatcher::new(DistanceMetric::ChiSquared).with_threshold(0.0);
        let a = vec![1.0, 2.0, 3.0];
        let tmpl = vec![SignTemplate::new(SignClass::NoEntry, a.clone())];
        let (_, conf) = matcher.classify(&a, &tmpl).unwrap();
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_match() {
        let matcher = TemplateMatcher::new(DistanceMetric::Euclidean).with_threshold(0.99);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 100.0];
        let tmpl = vec![SignTemplate::new(SignClass::Stop, b)];
        assert_eq!(matcher.classify(&a, &tmpl), Err(SignError::NoMatch));
    }

    #[test]
    fn test_empty_library() {
        let matcher = TemplateMatcher::new(DistanceMetric::Euclidean);
        assert_eq!(matcher.classify(&[1.0], &[]), Err(SignError::EmptyLibrary));
    }

    #[test]
    fn test_speed_limit_extract() {
        let ext = SpeedLimitExtractor::new(SpeedUnit::Kmh);
        let info = ext.extract(SignClass::SpeedLimit(62)).unwrap();
        assert_eq!(info.value, 60);
        assert_eq!(info.raw_value, 62);
    }

    #[test]
    fn test_speed_limit_not_speed() {
        let ext = SpeedLimitExtractor::new(SpeedUnit::Mph);
        assert!(ext.extract(SignClass::Stop).is_none());
    }

    #[test]
    fn test_aggregate_detections() {
        let ext = SpeedLimitExtractor::new(SpeedUnit::Kmh);
        let dets = vec![
            SignClass::Stop,
            SignClass::Stop,
            SignClass::SpeedLimit(50),
            SignClass::Stop,
        ];
        let counts = ext.aggregate(&dets);
        assert_eq!(*counts.get(&SignClass::Stop).unwrap(), 3);
    }

    #[test]
    fn test_speed_limit_info_display() {
        let info = SpeedLimitInfo { value: 60, unit: SpeedUnit::Kmh, raw_value: 62 };
        assert_eq!(format!("{info}"), "60 km/h");
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(format!("{}", DistanceMetric::Cosine), "Cosine");
    }

    #[test]
    fn test_matcher_display() {
        let m = TemplateMatcher::new(DistanceMetric::Euclidean);
        assert!(format!("{m}").contains("Euclidean"));
    }
}
