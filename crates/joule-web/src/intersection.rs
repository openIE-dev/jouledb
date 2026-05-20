//! Intersection Observer pattern.
//!
//! Headless implementation of the Intersection Observer API for
//! detecting visibility of elements relative to a root viewport.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Rect ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}

// ── IntersectionEntry ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionEntry {
    pub target_id: String,
    pub is_intersecting: bool,
    pub intersection_ratio: f64,
    pub bounding_rect: Rect,
    pub time_ms: u64,
}

// ── IntersectionConfig ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IntersectionConfig {
    pub root_margin: f64,
    pub thresholds: Vec<f64>,
}

impl Default for IntersectionConfig {
    fn default() -> Self {
        Self {
            root_margin: 0.0,
            thresholds: vec![0.0],
        }
    }
}

// ── Core computation ────────────────────────────────────────────────────────

/// Compute the intersection ratio of `target` against `root`.
/// Returns a value in `[0.0, 1.0]`.
pub fn intersection_ratio(target: &Rect, root: &Rect) -> f64 {
    let target_area = target.area();
    if target_area <= 0.0 {
        return 0.0;
    }

    let ix = target.x.max(root.x);
    let iy = target.y.max(root.y);
    let ix2 = target.right().min(root.right());
    let iy2 = target.bottom().min(root.bottom());

    let iw = (ix2 - ix).max(0.0);
    let ih = (iy2 - iy).max(0.0);
    let intersection_area = iw * ih;

    (intersection_area / target_area).clamp(0.0, 1.0)
}

// ── Observer ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct IntersectionObserver {
    pub config: IntersectionConfig,
    targets: HashMap<String, Rect>,
    root: Rect,
    /// Previous ratios for threshold-crossing detection.
    prev_ratios: HashMap<String, f64>,
    entries: Vec<IntersectionEntry>,
    time_ms: u64,
}

impl IntersectionObserver {
    pub fn new(root: Rect, config: IntersectionConfig) -> Self {
        Self {
            config,
            targets: HashMap::new(),
            root,
            prev_ratios: HashMap::new(),
            entries: Vec::new(),
            time_ms: 0,
        }
    }

    pub fn observe(&mut self, id: impl Into<String>, rect: Rect) {
        let id = id.into();
        self.targets.insert(id.clone(), rect);
        self.prev_ratios.insert(id, -1.0); // force first compute to emit
    }

    pub fn unobserve(&mut self, id: &str) {
        self.targets.remove(id);
        self.prev_ratios.remove(id);
    }

    pub fn update_target(&mut self, id: &str, rect: Rect) {
        if self.targets.contains_key(id) {
            self.targets.insert(id.to_string(), rect);
        }
    }

    pub fn update_root(&mut self, rect: Rect) {
        self.root = rect;
    }

    /// Compute intersection entries for all observed targets. Only emits an
    /// entry when the ratio crosses a threshold boundary compared to the
    /// previous computation.
    pub fn compute(&mut self) -> Vec<IntersectionEntry> {
        self.time_ms += 16; // simulated frame tick
        self.entries.clear();

        // Expand root by root_margin.
        let effective_root = Rect::new(
            self.root.x - self.config.root_margin,
            self.root.y - self.config.root_margin,
            self.root.width + self.config.root_margin * 2.0,
            self.root.height + self.config.root_margin * 2.0,
        );

        for (id, rect) in &self.targets {
            let ratio = intersection_ratio(rect, &effective_root);
            let prev = self.prev_ratios.get(id).copied().unwrap_or(-1.0);

            // Determine if a threshold was crossed or the intersecting state changed.
            let prev_intersecting = prev > 0.0;
            let now_intersecting = ratio > 0.0;
            let state_changed = prev_intersecting != now_intersecting;
            let crossed = state_changed || self.config.thresholds.iter().any(|t| {
                (prev < *t && ratio >= *t) || (prev >= *t && ratio < *t)
            });

            if crossed || prev < 0.0 {
                self.entries.push(IntersectionEntry {
                    target_id: id.clone(),
                    is_intersecting: ratio > 0.0,
                    intersection_ratio: ratio,
                    bounding_rect: *rect,
                    time_ms: self.time_ms,
                });
            }

            self.prev_ratios.insert(id.clone(), ratio);
        }

        self.entries.clone()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Rect {
        Rect::new(0.0, 0.0, 100.0, 100.0)
    }

    #[test]
    fn fully_visible() {
        let r = intersection_ratio(
            &Rect::new(10.0, 10.0, 20.0, 20.0),
            &root(),
        );
        assert!((r - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fully_outside() {
        let r = intersection_ratio(
            &Rect::new(200.0, 200.0, 20.0, 20.0),
            &root(),
        );
        assert!((r - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_intersection() {
        // Target is 20x20 at (90,90), so only 10x10 = 100 sq px visible
        // out of 400 total = 0.25.
        let r = intersection_ratio(
            &Rect::new(90.0, 90.0, 20.0, 20.0),
            &root(),
        );
        assert!((r - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn half_visible() {
        let r = intersection_ratio(
            &Rect::new(50.0, 0.0, 100.0, 100.0),
            &root(),
        );
        assert!((r - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn observer_first_compute_emits() {
        let mut obs = IntersectionObserver::new(root(), IntersectionConfig::default());
        obs.observe("box1", Rect::new(10.0, 10.0, 20.0, 20.0));
        let entries = obs.compute();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_intersecting);
        assert!((entries[0].intersection_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn threshold_crossing() {
        let config = IntersectionConfig {
            root_margin: 0.0,
            thresholds: vec![0.5],
        };
        let mut obs = IntersectionObserver::new(root(), config);
        // Start fully visible.
        obs.observe("box", Rect::new(10.0, 10.0, 20.0, 20.0));
        let _e1 = obs.compute(); // initial emit

        // Move so ratio drops below 0.5 (e.g., only 25% visible).
        obs.update_target("box", Rect::new(90.0, 90.0, 20.0, 20.0));
        let e2 = obs.compute();
        assert_eq!(e2.len(), 1);
        assert!(e2[0].intersection_ratio < 0.5);
    }

    #[test]
    fn no_emit_when_no_crossing() {
        let config = IntersectionConfig {
            root_margin: 0.0,
            thresholds: vec![0.5],
        };
        let mut obs = IntersectionObserver::new(root(), config);
        obs.observe("box", Rect::new(10.0, 10.0, 20.0, 20.0));
        obs.compute(); // initial

        // Still fully visible — no crossing of 0.5 threshold.
        obs.update_target("box", Rect::new(20.0, 20.0, 20.0, 20.0));
        let entries = obs.compute();
        assert!(entries.is_empty());
    }

    #[test]
    fn observe_unobserve() {
        let mut obs = IntersectionObserver::new(root(), IntersectionConfig::default());
        obs.observe("a", Rect::new(10.0, 10.0, 10.0, 10.0));
        obs.observe("b", Rect::new(20.0, 20.0, 10.0, 10.0));
        obs.unobserve("a");
        let entries = obs.compute();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].target_id, "b");
    }

    #[test]
    fn root_margin_expands() {
        let config = IntersectionConfig {
            root_margin: 10.0,
            thresholds: vec![0.0],
        };
        let mut obs = IntersectionObserver::new(root(), config);
        // Just outside root, but within margin.
        obs.observe("box", Rect::new(105.0, 0.0, 5.0, 5.0));
        let entries = obs.compute();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_intersecting);
    }

    #[test]
    fn update_root() {
        let mut obs = IntersectionObserver::new(root(), IntersectionConfig::default());
        obs.observe("box", Rect::new(150.0, 150.0, 10.0, 10.0));
        let e1 = obs.compute();
        assert!(!e1[0].is_intersecting);

        obs.update_root(Rect::new(0.0, 0.0, 200.0, 200.0));
        let e2 = obs.compute();
        assert_eq!(e2.len(), 1);
        assert!(e2[0].is_intersecting);
    }

    #[test]
    fn zero_area_target() {
        let r = intersection_ratio(&Rect::new(0.0, 0.0, 0.0, 0.0), &root());
        assert!((r - 0.0).abs() < f64::EPSILON);
    }
}
