//! Scroll-driven animation engine.
//!
//! Replaces GSAP ScrollTrigger / Intersection Observer API for
//! headless computation. Scroll position drives animation progress,
//! with support for viewport intersection triggers, pinning,
//! scrubbing, and snap points.

use std::fmt;

// ── Element Rect ───────────────────────────────────────────────

/// Simulated bounding rectangle of an element relative to the document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementRect {
    /// Top edge (document coordinates).
    pub top: f64,
    /// Bottom edge (document coordinates).
    pub bottom: f64,
    /// Left edge (document coordinates).
    pub left: f64,
    /// Right edge (document coordinates).
    pub right: f64,
}

impl ElementRect {
    pub fn new(top: f64, left: f64, width: f64, height: f64) -> Self {
        Self {
            top,
            bottom: top + height,
            left,
            right: left + width,
        }
    }

    pub fn height(&self) -> f64 {
        self.bottom - self.top
    }

    pub fn width(&self) -> f64 {
        self.right - self.left
    }
}

// ── Viewport ───────────────────────────────────────────────────

/// The visible viewport (scroll position + dimensions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub scroll_y: f64,
    pub scroll_x: f64,
    pub height: f64,
    pub width: f64,
}

impl Viewport {
    pub fn new(scroll_y: f64, height: f64) -> Self {
        Self { scroll_y, scroll_x: 0.0, height, width: 0.0 }
    }

    pub fn full(scroll_x: f64, scroll_y: f64, width: f64, height: f64) -> Self {
        Self { scroll_y, scroll_x, height, width }
    }

    /// Top of the viewport in document coordinates.
    pub fn top(&self) -> f64 {
        self.scroll_y
    }

    /// Bottom of the viewport in document coordinates.
    pub fn bottom(&self) -> f64 {
        self.scroll_y + self.height
    }

    /// Whether an element intersects the viewport vertically.
    pub fn intersects_y(&self, rect: &ElementRect) -> bool {
        rect.bottom > self.top() && rect.top < self.bottom()
    }

    /// Intersection ratio [0, 1] for an element (vertical).
    pub fn intersection_ratio_y(&self, rect: &ElementRect) -> f64 {
        if !self.intersects_y(rect) {
            return 0.0;
        }
        let visible_top = rect.top.max(self.top());
        let visible_bottom = rect.bottom.min(self.bottom());
        let visible_height = (visible_bottom - visible_top).max(0.0);
        let element_height = rect.height();
        if element_height <= 0.0 { 0.0 } else { (visible_height / element_height).clamp(0.0, 1.0) }
    }
}

// ── Scroll Trigger ─────────────────────────────────────────────

/// Trigger state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerState {
    /// Element has not yet entered.
    Before,
    /// Element is in the active range.
    Active,
    /// Element has passed the active range.
    After,
}

/// Events emitted by triggers.
#[derive(Debug, Clone, PartialEq)]
pub enum ScrollEvent {
    Enter(String),
    Leave(String),
    Progress(String, f64),
}

/// A scroll-driven trigger bound to an element.
#[derive(Debug, Clone)]
pub struct ScrollTrigger {
    /// Element identifier.
    pub element_id: String,
    /// Offset from viewport top where trigger starts (pixels or fraction).
    /// 0.0 = top of viewport, 1.0 = bottom.
    pub start_offset: f64,
    /// Offset from viewport top where trigger ends.
    pub end_offset: f64,
    /// Current state.
    state: TriggerState,
    /// Last computed progress.
    progress: f64,
    /// Whether the element is pinned during the trigger range.
    pub pin: bool,
    /// Pin offset (how much to translate the element when pinned).
    pin_offset: f64,
}

impl ScrollTrigger {
    pub fn new(element_id: impl Into<String>, start_offset: f64, end_offset: f64) -> Self {
        Self {
            element_id: element_id.into(),
            start_offset,
            end_offset,
            state: TriggerState::Before,
            progress: 0.0,
            pin: false,
            pin_offset: 0.0,
        }
    }

    pub fn with_pin(mut self, pin: bool) -> Self {
        self.pin = pin;
        self
    }

    pub fn state(&self) -> TriggerState {
        self.state
    }

    pub fn progress(&self) -> f64 {
        self.progress
    }

    pub fn pin_offset(&self) -> f64 {
        self.pin_offset
    }

    /// Update trigger state given the element rect and viewport.
    /// Returns any events generated.
    pub fn update(&mut self, rect: &ElementRect, viewport: &Viewport) -> Vec<ScrollEvent> {
        let mut events = Vec::new();

        // Compute the start and end scroll positions.
        let trigger_start = rect.top - viewport.height * self.start_offset;
        let trigger_end = rect.top - viewport.height * self.end_offset + rect.height();

        let scroll = viewport.scroll_y;
        let range = trigger_end - trigger_start;

        let new_state;
        let new_progress;

        if range.abs() < 1e-10 {
            new_state = if scroll >= trigger_start { TriggerState::After } else { TriggerState::Before };
            new_progress = if scroll >= trigger_start { 1.0 } else { 0.0 };
        } else if scroll < trigger_start {
            new_state = TriggerState::Before;
            new_progress = 0.0;
        } else if scroll > trigger_end {
            new_state = TriggerState::After;
            new_progress = 1.0;
        } else {
            new_state = TriggerState::Active;
            new_progress = ((scroll - trigger_start) / range).clamp(0.0, 1.0);
        }

        // Emit events.
        let old_state = self.state;
        if old_state != TriggerState::Active && new_state == TriggerState::Active {
            events.push(ScrollEvent::Enter(self.element_id.clone()));
        }
        if old_state == TriggerState::Active && new_state != TriggerState::Active {
            events.push(ScrollEvent::Leave(self.element_id.clone()));
        }
        if new_state == TriggerState::Active && (new_progress - self.progress).abs() > 1e-6 {
            events.push(ScrollEvent::Progress(self.element_id.clone(), new_progress));
        }

        self.state = new_state;
        self.progress = new_progress;

        // Pinning: when active, compute translate offset.
        if self.pin && new_state == TriggerState::Active {
            self.pin_offset = scroll - trigger_start;
        } else {
            self.pin_offset = 0.0;
        }

        events
    }
}

impl fmt::Display for ScrollTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScrollTrigger({}, {:.2}%)", self.element_id, self.progress * 100.0)
    }
}

// ── Snap Points ────────────────────────────────────────────────

/// Snap point configuration.
#[derive(Debug, Clone)]
pub struct SnapPoint {
    /// Scroll position (document coordinates).
    pub position: f64,
    /// Label for identification.
    pub label: String,
}

impl SnapPoint {
    pub fn new(position: f64, label: impl Into<String>) -> Self {
        Self { position, label: label.into() }
    }
}

// ── Timeline ───────────────────────────────────────────────────

/// A scroll timeline managing multiple triggers.
#[derive(Debug, Clone)]
pub struct ScrollTimeline {
    triggers: Vec<ScrollTrigger>,
    snap_points: Vec<SnapPoint>,
}

impl ScrollTimeline {
    pub fn new() -> Self {
        Self { triggers: Vec::new(), snap_points: Vec::new() }
    }

    pub fn add_trigger(&mut self, trigger: ScrollTrigger) {
        self.triggers.push(trigger);
    }

    pub fn add_snap_point(&mut self, snap: SnapPoint) {
        self.snap_points.push(snap);
        self.snap_points.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
    }

    pub fn triggers(&self) -> &[ScrollTrigger] {
        &self.triggers
    }

    pub fn snap_points(&self) -> &[SnapPoint] {
        &self.snap_points
    }

    /// Update all triggers given element rects and viewport.
    /// `rects` maps element_id -> rect.
    pub fn update(
        &mut self,
        rects: &std::collections::HashMap<String, ElementRect>,
        viewport: &Viewport,
    ) -> Vec<ScrollEvent> {
        let mut all_events = Vec::new();
        for trigger in &mut self.triggers {
            if let Some(rect) = rects.get(&trigger.element_id) {
                let events = trigger.update(rect, viewport);
                all_events.extend(events);
            }
        }
        all_events
    }

    /// Find the nearest snap point to a given scroll position.
    pub fn nearest_snap(&self, scroll_y: f64) -> Option<&SnapPoint> {
        self.snap_points.iter().min_by(|a, b| {
            let da = (a.position - scroll_y).abs();
            let db = (b.position - scroll_y).abs();
            da.partial_cmp(&db).unwrap()
        })
    }

    /// Find snap point within a given threshold distance.
    pub fn snap_within(&self, scroll_y: f64, threshold: f64) -> Option<&SnapPoint> {
        self.nearest_snap(scroll_y)
            .filter(|s| (s.position - scroll_y).abs() <= threshold)
    }

    /// Get a trigger by element ID.
    pub fn trigger_by_id(&self, id: &str) -> Option<&ScrollTrigger> {
        self.triggers.iter().find(|t| t.element_id == id)
    }

    /// Get progress for a specific trigger.
    pub fn progress_of(&self, id: &str) -> Option<f64> {
        self.trigger_by_id(id).map(|t| t.progress())
    }
}

impl Default for ScrollTimeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn viewport_at(scroll_y: f64) -> Viewport {
        Viewport::new(scroll_y, 800.0)
    }

    fn element_rect() -> ElementRect {
        // Element at y=1000, 200px tall.
        ElementRect::new(1000.0, 0.0, 300.0, 200.0)
    }

    #[test]
    fn viewport_intersection() {
        let vp = viewport_at(900.0);
        let rect = element_rect();
        assert!(vp.intersects_y(&rect));

        let vp_far = viewport_at(0.0);
        assert!(!vp_far.intersects_y(&rect));
    }

    #[test]
    fn intersection_ratio() {
        let rect = ElementRect::new(100.0, 0.0, 100.0, 200.0);
        // Viewport fully containing the element.
        let vp = Viewport::new(0.0, 1000.0);
        assert!((vp.intersection_ratio_y(&rect) - 1.0).abs() < 0.01);

        // Viewport showing half.
        let vp2 = Viewport::new(200.0, 200.0);
        assert!((vp2.intersection_ratio_y(&rect) - 0.5).abs() < 0.01);
    }

    #[test]
    fn trigger_before_active_after() {
        let rect = element_rect();
        let mut trigger = ScrollTrigger::new("el", 1.0, 0.0);

        // Before: scroll well above the element.
        trigger.update(&rect, &viewport_at(0.0));
        assert_eq!(trigger.state(), TriggerState::Before);
        assert!((trigger.progress() - 0.0).abs() < 0.01);

        // Active: scrolling into range.
        trigger.update(&rect, &viewport_at(500.0));
        assert_eq!(trigger.state(), TriggerState::Active);
        assert!(trigger.progress() > 0.0);

        // After: scrolled well past.
        trigger.update(&rect, &viewport_at(2000.0));
        assert_eq!(trigger.state(), TriggerState::After);
        assert!((trigger.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn trigger_emits_enter_leave_events() {
        let rect = element_rect();
        let mut trigger = ScrollTrigger::new("box", 1.0, 0.0);

        // Move from before to active.
        let events = trigger.update(&rect, &viewport_at(500.0));
        assert!(events.iter().any(|e| matches!(e, ScrollEvent::Enter(id) if id == "box")));

        // Move from active to after.
        let events = trigger.update(&rect, &viewport_at(2000.0));
        assert!(events.iter().any(|e| matches!(e, ScrollEvent::Leave(id) if id == "box")));
    }

    #[test]
    fn scrub_progress() {
        let rect = element_rect();
        let mut trigger = ScrollTrigger::new("scrub", 1.0, 0.0);

        // Middle of range.
        trigger.update(&rect, &viewport_at(600.0));
        let p = trigger.progress();
        assert!(p > 0.0 && p < 1.0, "Progress should be between 0 and 1, got {p}");
    }

    #[test]
    fn pin_offset() {
        let rect = element_rect();
        let mut trigger = ScrollTrigger::new("pinned", 1.0, 0.0).with_pin(true);

        trigger.update(&rect, &viewport_at(500.0));
        assert_eq!(trigger.state(), TriggerState::Active);
        assert!(trigger.pin_offset() > 0.0);

        // When not active, pin offset is 0.
        trigger.update(&rect, &viewport_at(0.0));
        assert!((trigger.pin_offset() - 0.0).abs() < 0.01);
    }

    #[test]
    fn snap_points() {
        let mut timeline = ScrollTimeline::new();
        timeline.add_snap_point(SnapPoint::new(0.0, "top"));
        timeline.add_snap_point(SnapPoint::new(500.0, "section1"));
        timeline.add_snap_point(SnapPoint::new(1000.0, "section2"));

        let nearest = timeline.nearest_snap(480.0).unwrap();
        assert_eq!(nearest.label, "section1");

        let within = timeline.snap_within(498.0, 10.0).unwrap();
        assert_eq!(within.label, "section1");

        assert!(timeline.snap_within(250.0, 10.0).is_none());
    }

    #[test]
    fn timeline_update_multiple() {
        let mut timeline = ScrollTimeline::new();
        timeline.add_trigger(ScrollTrigger::new("a", 1.0, 0.0));
        timeline.add_trigger(ScrollTrigger::new("b", 1.0, 0.0));

        let mut rects = HashMap::new();
        rects.insert("a".to_string(), ElementRect::new(500.0, 0.0, 100.0, 100.0));
        rects.insert("b".to_string(), ElementRect::new(2000.0, 0.0, 100.0, 100.0));

        let vp = viewport_at(200.0);
        let events = timeline.update(&rects, &vp);
        // "a" should be entering/active, "b" still before.
        assert!(events.iter().any(|e| matches!(e, ScrollEvent::Enter(id) if id == "a")));
    }

    #[test]
    fn trigger_display() {
        let trigger = ScrollTrigger::new("el", 1.0, 0.0);
        let s = format!("{trigger}");
        assert!(s.contains("ScrollTrigger(el"));
    }

    #[test]
    fn timeline_trigger_by_id() {
        let mut timeline = ScrollTimeline::new();
        timeline.add_trigger(ScrollTrigger::new("hero", 1.0, 0.0));
        assert!(timeline.trigger_by_id("hero").is_some());
        assert!(timeline.trigger_by_id("missing").is_none());
    }

    #[test]
    fn element_rect_dimensions() {
        let rect = ElementRect::new(100.0, 50.0, 300.0, 200.0);
        assert!((rect.height() - 200.0).abs() < 1e-10);
        assert!((rect.width() - 300.0).abs() < 1e-10);
    }
}
