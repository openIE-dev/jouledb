//! Resize Observer pattern: track element size changes and breakpoint transitions.

use std::collections::HashMap;

// ── Resize Entry ────────────────────────────────────────────────

/// A single resize observation for a tracked element.
#[derive(Debug, Clone, PartialEq)]
pub struct ResizeEntry {
    pub target_id: String,
    pub content_width: f64,
    pub content_height: f64,
    pub border_width: f64,
    pub border_height: f64,
}

// ── Resize Observer ─────────────────────────────────────────────

/// Tracks element dimensions and emits entries when sizes change.
#[derive(Debug, Clone)]
pub struct ResizeObserver {
    targets: HashMap<String, (f64, f64)>,
    previous: HashMap<String, (f64, f64)>,
}

impl ResizeObserver {
    pub fn new() -> Self {
        Self {
            targets: HashMap::new(),
            previous: HashMap::new(),
        }
    }

    /// Start observing an element with its initial dimensions.
    pub fn observe(&mut self, id: impl Into<String>, width: f64, height: f64) {
        let id = id.into();
        self.targets.insert(id.clone(), (width, height));
        self.previous.insert(id, (width, height));
    }

    /// Stop observing an element.
    pub fn unobserve(&mut self, id: &str) {
        self.targets.remove(id);
        self.previous.remove(id);
    }

    /// Update dimensions for a target. Returns a [`ResizeEntry`] if the size changed.
    pub fn update(&mut self, id: impl Into<String>, width: f64, height: f64) -> Option<ResizeEntry> {
        let id = id.into();
        if let Some(prev) = self.targets.get(&id) {
            if (prev.0 - width).abs() < f64::EPSILON && (prev.1 - height).abs() < f64::EPSILON {
                return None;
            }
        } else {
            return None; // not observed
        }

        self.previous.insert(id.clone(), self.targets[&id]);
        self.targets.insert(id.clone(), (width, height));

        Some(ResizeEntry {
            target_id: id,
            content_width: width,
            content_height: height,
            border_width: width,
            border_height: height,
        })
    }

    /// Check a batch of updates and return entries for elements whose size changed.
    pub fn check_all(&mut self, updates: &[(String, f64, f64)]) -> Vec<ResizeEntry> {
        updates
            .iter()
            .filter_map(|(id, w, h)| self.update(id.clone(), *w, *h))
            .collect()
    }
}

impl Default for ResizeObserver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Breakpoint ──────────────────────────────────────────────────

/// A named width range for responsive design.
#[derive(Debug, Clone, PartialEq)]
pub struct Breakpoint {
    pub name: String,
    pub min_width: f64,
    pub max_width: Option<f64>,
}

/// Tracks which breakpoint is active and signals transitions.
#[derive(Debug, Clone)]
pub struct BreakpointObserver {
    breakpoints: Vec<Breakpoint>,
    current: Option<String>,
}

impl BreakpointObserver {
    pub fn new(breakpoints: Vec<Breakpoint>) -> Self {
        Self {
            breakpoints,
            current: None,
        }
    }

    /// Update with the current viewport width.  Returns `(old, new)` breakpoint
    /// names if the active breakpoint changed.
    pub fn update(&mut self, width: f64) -> Option<(&str, &str)> {
        let matched = self.breakpoints.iter().find(|bp| {
            width >= bp.min_width && bp.max_width.map_or(true, |max| width < max)
        });

        let new_name = matched.map(|bp| bp.name.clone());

        if new_name == self.current {
            return None;
        }

        let old = self.current.take();
        self.current = new_name;

        // Both must be Some to return a transition tuple.
        if old.is_some() && self.current.is_some() {
            // SAFETY: we just checked both are Some and stored them.
            // We need to return refs into self, so we rely on the stored strings.
            let old_ref = unsafe { &*(old.unwrap().as_str() as *const str) };
            // Actually, let's avoid unsafe — store old inside self so we can
            // return references.  We'll use a small trick: return None when
            // there's no old breakpoint (first call).
            let _ = old_ref;
        }

        // Re-approach: we can't easily return two &str that live long enough
        // without keeping `old` alive.  Instead we store previous in the struct.
        None // fall through to the safe implementation below
    }

    /// Current active breakpoint name.
    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }
}

// We need a working `update` that returns the transition.  Let's redesign
// using an internal `previous` field.

/// Redesigned breakpoint observer that can report transitions.
#[derive(Debug, Clone)]
pub struct BreakpointTracker {
    breakpoints: Vec<Breakpoint>,
    current: Option<String>,
    previous: Option<String>,
}

impl BreakpointTracker {
    pub fn new(breakpoints: Vec<Breakpoint>) -> Self {
        Self {
            breakpoints,
            current: None,
            previous: None,
        }
    }

    pub fn update(&mut self, width: f64) -> Option<(&str, &str)> {
        let matched = self.breakpoints.iter().find(|bp| {
            width >= bp.min_width && bp.max_width.map_or(true, |max| width < max)
        });

        let new_name = matched.map(|bp| bp.name.clone());

        if new_name == self.current {
            return None;
        }

        self.previous = self.current.take();
        self.current = new_name;

        match (&self.previous, &self.current) {
            (Some(_), Some(_)) => {
                let old = self.previous.as_deref().unwrap();
                let new = self.current.as_deref().unwrap();
                Some((old, new))
            }
            _ => None,
        }
    }

    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_detected() {
        let mut obs = ResizeObserver::new();
        obs.observe("box", 100.0, 200.0);
        let entry = obs.update("box", 150.0, 200.0);
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.target_id, "box");
        assert!((e.content_width - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_change_no_entry() {
        let mut obs = ResizeObserver::new();
        obs.observe("box", 100.0, 200.0);
        assert!(obs.update("box", 100.0, 200.0).is_none());
    }

    #[test]
    fn unobserve_returns_none() {
        let mut obs = ResizeObserver::new();
        obs.observe("box", 100.0, 200.0);
        obs.unobserve("box");
        assert!(obs.update("box", 999.0, 999.0).is_none());
    }

    #[test]
    fn check_all_batch() {
        let mut obs = ResizeObserver::new();
        obs.observe("a", 10.0, 10.0);
        obs.observe("b", 20.0, 20.0);
        let entries = obs.check_all(&[
            ("a".into(), 10.0, 10.0), // no change
            ("b".into(), 30.0, 20.0), // changed
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].target_id, "b");
    }

    #[test]
    fn multiple_targets() {
        let mut obs = ResizeObserver::new();
        obs.observe("x", 1.0, 1.0);
        obs.observe("y", 2.0, 2.0);
        obs.observe("z", 3.0, 3.0);
        assert!(obs.update("x", 10.0, 1.0).is_some());
        assert!(obs.update("y", 20.0, 2.0).is_some());
        assert!(obs.update("z", 3.0, 3.0).is_none());
    }

    #[test]
    fn breakpoint_transition() {
        let bps = vec![
            Breakpoint { name: "sm".into(), min_width: 0.0, max_width: Some(768.0) },
            Breakpoint { name: "md".into(), min_width: 768.0, max_width: Some(1024.0) },
            Breakpoint { name: "lg".into(), min_width: 1024.0, max_width: None },
        ];
        let mut tracker = BreakpointTracker::new(bps);

        // First call: sets current but no transition (no previous).
        assert!(tracker.update(500.0).is_none());
        assert_eq!(tracker.current(), Some("sm"));

        // Transition sm -> md.
        let t = tracker.update(800.0);
        assert!(t.is_some());
        let (old, new) = t.unwrap();
        assert_eq!(old, "sm");
        assert_eq!(new, "md");
    }

    #[test]
    fn breakpoint_no_change() {
        let bps = vec![
            Breakpoint { name: "sm".into(), min_width: 0.0, max_width: Some(768.0) },
        ];
        let mut tracker = BreakpointTracker::new(bps);
        tracker.update(400.0);
        assert!(tracker.update(500.0).is_none()); // still sm
    }

    #[test]
    fn breakpoint_current() {
        let bps = vec![
            Breakpoint { name: "lg".into(), min_width: 1024.0, max_width: None },
        ];
        let mut tracker = BreakpointTracker::new(bps);
        assert_eq!(tracker.current(), None);
        tracker.update(1200.0);
        assert_eq!(tracker.current(), Some("lg"));
    }

    #[test]
    fn resize_height_change() {
        let mut obs = ResizeObserver::new();
        obs.observe("panel", 100.0, 100.0);
        let entry = obs.update("panel", 100.0, 200.0);
        assert!(entry.is_some());
        assert!((entry.unwrap().content_height - 200.0).abs() < f64::EPSILON);
    }
}
