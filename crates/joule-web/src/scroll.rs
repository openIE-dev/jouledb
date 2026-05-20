//! Scroll state management: position tracking, infinite scroll, restoration,
//! anchors, and smooth interpolation.
//!
//! Replaces react-infinite-scroll / locomotive-scroll. Headless (no DOM) —
//! provides pure state logic that any rendering layer can drive.

use std::collections::HashMap;

// ── Scroll Position & Direction ─────────────────────────────────

/// 2-D scroll position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollPosition {
    pub x: f64,
    pub y: f64,
}

impl ScrollPosition {
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

/// Direction of scroll movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
    None_,
}

// ── Scroll State ────────────────────────────────────────────────

/// Full scroll viewport state with velocity tracking.
#[derive(Debug, Clone)]
pub struct ScrollState {
    pub position: ScrollPosition,
    pub previous: ScrollPosition,
    pub viewport_width: f64,
    pub viewport_height: f64,
    pub content_width: f64,
    pub content_height: f64,
    pub is_scrolling: bool,
    pub direction: ScrollDirection,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub last_update_ms: u64,
}

impl ScrollState {
    pub fn new(viewport_w: f64, viewport_h: f64, content_w: f64, content_h: f64) -> Self {
        Self {
            position: ScrollPosition::zero(),
            previous: ScrollPosition::zero(),
            viewport_width: viewport_w,
            viewport_height: viewport_h,
            content_width: content_w,
            content_height: content_h,
            is_scrolling: false,
            direction: ScrollDirection::None_,
            velocity_x: 0.0,
            velocity_y: 0.0,
            last_update_ms: 0,
        }
    }

    /// Update scroll position and compute direction / velocity.
    pub fn update(&mut self, x: f64, y: f64, timestamp_ms: u64) {
        self.previous = self.position;
        self.position = ScrollPosition { x, y };

        let dt = if timestamp_ms > self.last_update_ms {
            (timestamp_ms - self.last_update_ms) as f64
        } else {
            1.0
        };

        let dx = x - self.previous.x;
        let dy = y - self.previous.y;

        self.velocity_x = dx / dt * 1000.0; // px/s
        self.velocity_y = dy / dt * 1000.0;

        self.direction = if dy.abs() > dx.abs() {
            if dy > 0.0 {
                ScrollDirection::Down
            } else if dy < 0.0 {
                ScrollDirection::Up
            } else {
                ScrollDirection::None_
            }
        } else if dx > 0.0 {
            ScrollDirection::Right
        } else if dx < 0.0 {
            ScrollDirection::Left
        } else {
            ScrollDirection::None_
        };

        self.is_scrolling = dx.abs() > 0.0 || dy.abs() > 0.0;
        self.last_update_ms = timestamp_ms;
    }

    pub fn at_top(&self) -> bool {
        self.position.y <= 0.0
    }

    pub fn at_bottom(&self) -> bool {
        let max_y = (self.content_height - self.viewport_height).max(0.0);
        self.position.y >= max_y
    }

    pub fn at_left(&self) -> bool {
        self.position.x <= 0.0
    }

    pub fn at_right(&self) -> bool {
        let max_x = (self.content_width - self.viewport_width).max(0.0);
        self.position.x >= max_x
    }

    /// Vertical scroll progress as 0.0 to 100.0.
    pub fn scroll_percent_y(&self) -> f64 {
        let max_y = self.content_height - self.viewport_height;
        if max_y <= 0.0 {
            return 0.0;
        }
        (self.position.y / max_y * 100.0).clamp(0.0, 100.0)
    }

    /// Horizontal scroll progress as 0.0 to 100.0.
    pub fn scroll_percent_x(&self) -> f64 {
        let max_x = self.content_width - self.viewport_width;
        if max_x <= 0.0 {
            return 0.0;
        }
        (self.position.x / max_x * 100.0).clamp(0.0, 100.0)
    }

    pub fn can_scroll_y(&self) -> bool {
        self.content_height > self.viewport_height
    }

    pub fn can_scroll_x(&self) -> bool {
        self.content_width > self.viewport_width
    }
}

// ── Infinite Scroll ─────────────────────────────────────────────

/// Loading direction for infinite scroll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfiniteDirection {
    Down,
    Up,
}

/// Infinite scroll state machine.
#[derive(Debug, Clone)]
pub struct InfiniteScroll {
    pub page: usize,
    pub loading: bool,
    pub has_more: bool,
    pub threshold_px: f64,
    pub direction: InfiniteDirection,
}

impl InfiniteScroll {
    pub fn new(threshold: f64) -> Self {
        Self {
            page: 0,
            loading: false,
            has_more: true,
            threshold_px: threshold,
            direction: InfiniteDirection::Down,
        }
    }

    /// Should we trigger another page load?
    pub fn should_load_more(&self, scroll: &ScrollState) -> bool {
        if self.loading || !self.has_more {
            return false;
        }
        match self.direction {
            InfiniteDirection::Down => {
                let max_y = (scroll.content_height - scroll.viewport_height).max(0.0);
                let remaining = max_y - scroll.position.y;
                remaining <= self.threshold_px
            }
            InfiniteDirection::Up => scroll.position.y <= self.threshold_px,
        }
    }

    pub fn start_loading(&mut self) {
        self.loading = true;
    }

    pub fn finish_loading(&mut self, has_more: bool) {
        self.loading = false;
        self.has_more = has_more;
        self.page += 1;
    }

    pub fn reset(&mut self) {
        self.page = 0;
        self.loading = false;
        self.has_more = true;
    }
}

// ── Scroll Restoration ──────────────────────────────────────────

/// Save / restore scroll positions per route (browser-style).
#[derive(Debug, Clone, Default)]
pub struct ScrollRestoration {
    positions: HashMap<String, ScrollPosition>,
}

impl ScrollRestoration {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save(&mut self, route: &str, position: ScrollPosition) {
        self.positions.insert(route.to_string(), position);
    }

    pub fn restore(&self, route: &str) -> Option<ScrollPosition> {
        self.positions.get(route).copied()
    }

    pub fn clear(&mut self, route: &str) {
        self.positions.remove(route);
    }

    pub fn clear_all(&mut self) {
        self.positions.clear();
    }
}

// ── Scroll Anchor ───────────────────────────────────────────────

/// Track named anchor positions for scroll-spy / table-of-contents.
#[derive(Debug, Clone, Default)]
pub struct ScrollAnchor {
    /// (id, y_position), kept sorted by y.
    anchors: Vec<(String, f64)>,
}

impl ScrollAnchor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, id: &str, y: f64) {
        // Remove existing entry with the same id.
        self.anchors.retain(|(existing, _)| existing != id);
        self.anchors.push((id.to_string(), y));
        self.anchors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    pub fn unregister(&mut self, id: &str) {
        self.anchors.retain(|(existing, _)| existing != id);
    }

    /// Return the id of the anchor closest to (but above) `scroll_y + offset`.
    pub fn active_anchor(&self, scroll_y: f64, offset: f64) -> Option<&str> {
        let target = scroll_y + offset;
        let mut active: Option<&str> = None;
        for (id, y) in &self.anchors {
            if *y <= target {
                active = Some(id.as_str());
            } else {
                break;
            }
        }
        active
    }

    /// Return the y position to scroll to for the given anchor.
    pub fn scroll_to_anchor(&self, id: &str) -> Option<f64> {
        self.anchors.iter().find(|(aid, _)| aid == id).map(|(_, y)| *y)
    }
}

// ── Smooth Scroll ───────────────────────────────────────────────

/// Ease-out cubic (default easing).
fn ease_out_cubic(t: f64) -> f64 {
    let inv = 1.0 - t;
    1.0 - inv * inv * inv
}

/// Smooth scroll interpolation.
pub struct SmoothScroll {
    pub target: ScrollPosition,
    pub current: ScrollPosition,
    from: ScrollPosition,
    pub duration_ms: u64,
    pub elapsed_ms: u64,
    pub easing: fn(f64) -> f64,
}

impl SmoothScroll {
    pub fn new(from: ScrollPosition, to: ScrollPosition, duration_ms: u64) -> Self {
        Self {
            target: to,
            current: from,
            from,
            duration_ms,
            elapsed_ms: 0,
            easing: ease_out_cubic,
        }
    }

    /// Advance by `dt_ms` and return the interpolated position.
    pub fn tick(&mut self, dt_ms: u64) -> ScrollPosition {
        self.elapsed_ms = self.elapsed_ms.saturating_add(dt_ms).min(self.duration_ms);
        let t = if self.duration_ms == 0 {
            1.0
        } else {
            self.elapsed_ms as f64 / self.duration_ms as f64
        };
        let eased = (self.easing)(t);
        self.current = ScrollPosition {
            x: self.from.x + (self.target.x - self.from.x) * eased,
            y: self.from.y + (self.target.y - self.from.y) * eased,
        };
        self.current
    }

    pub fn is_finished(&self) -> bool {
        self.elapsed_ms >= self.duration_ms
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_state_direction_detection() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        state.update(0.0, 100.0, 100);
        assert_eq!(state.direction, ScrollDirection::Down);
        state.update(0.0, 50.0, 200);
        assert_eq!(state.direction, ScrollDirection::Up);
    }

    #[test]
    fn at_bottom_when_position_matches() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        // max_y = 2000 - 600 = 1400
        state.update(0.0, 1400.0, 100);
        assert!(state.at_bottom());
    }

    #[test]
    fn scroll_percent() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        state.update(0.0, 700.0, 100);
        let pct = state.scroll_percent_y();
        assert!((pct - 50.0).abs() < 0.1);
    }

    #[test]
    fn velocity_calculation() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        state.update(0.0, 0.0, 0);
        state.update(0.0, 100.0, 100); // 100px in 100ms = 1000 px/s
        assert!((state.velocity_y - 1000.0).abs() < 1.0);
    }

    #[test]
    fn infinite_scroll_triggers_near_bottom() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        let inf = InfiniteScroll::new(200.0);
        // max_y = 1400, scroll to 1250 → remaining = 150 < 200
        state.update(0.0, 1250.0, 100);
        assert!(inf.should_load_more(&state));
    }

    #[test]
    fn infinite_scroll_loading_blocks() {
        let mut state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        state.update(0.0, 1400.0, 100);
        let mut inf = InfiniteScroll::new(200.0);
        inf.start_loading();
        assert!(!inf.should_load_more(&state));
    }

    #[test]
    fn scroll_restoration_save_restore_roundtrip() {
        let mut rest = ScrollRestoration::new();
        let pos = ScrollPosition { x: 10.0, y: 250.0 };
        rest.save("/home", pos);
        let restored = rest.restore("/home").unwrap();
        assert!((restored.x - 10.0).abs() < f64::EPSILON);
        assert!((restored.y - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scroll_anchor_active_detection() {
        let mut anchor = ScrollAnchor::new();
        anchor.register("intro", 0.0);
        anchor.register("chapter1", 500.0);
        anchor.register("chapter2", 1200.0);

        assert_eq!(anchor.active_anchor(600.0, 0.0), Some("chapter1"));
        assert_eq!(anchor.active_anchor(1500.0, 0.0), Some("chapter2"));
        assert_eq!(anchor.active_anchor(0.0, 0.0), Some("intro"));
    }

    #[test]
    fn smooth_scroll_interpolates() {
        let from = ScrollPosition { x: 0.0, y: 0.0 };
        let to = ScrollPosition { x: 0.0, y: 1000.0 };
        let mut smooth = SmoothScroll::new(from, to, 1000);

        let mid = smooth.tick(500);
        assert!(mid.y > 0.0);
        assert!(mid.y < 1000.0);
    }

    #[test]
    fn smooth_scroll_finishes() {
        let from = ScrollPosition { x: 0.0, y: 0.0 };
        let to = ScrollPosition { x: 0.0, y: 500.0 };
        let mut smooth = SmoothScroll::new(from, to, 300);

        smooth.tick(300);
        assert!(smooth.is_finished());
        assert!((smooth.current.y - 500.0).abs() < 0.01);
    }

    #[test]
    fn at_top() {
        let state = ScrollState::new(800.0, 600.0, 800.0, 2000.0);
        assert!(state.at_top());
    }

    #[test]
    fn clear_all_restoration() {
        let mut rest = ScrollRestoration::new();
        rest.save("/a", ScrollPosition { x: 0.0, y: 100.0 });
        rest.save("/b", ScrollPosition { x: 0.0, y: 200.0 });
        rest.clear_all();
        assert!(rest.restore("/a").is_none());
        assert!(rest.restore("/b").is_none());
    }

    #[test]
    fn infinite_reset() {
        let mut inf = InfiniteScroll::new(100.0);
        inf.start_loading();
        inf.finish_loading(true);
        assert_eq!(inf.page, 1);
        inf.reset();
        assert_eq!(inf.page, 0);
        assert!(!inf.loading);
        assert!(inf.has_more);
    }

    #[test]
    fn scroll_state_no_scrollable_content() {
        let state = ScrollState::new(800.0, 600.0, 800.0, 400.0);
        assert!(!state.can_scroll_y());
        assert!(state.at_bottom()); // content smaller than viewport
        assert_eq!(state.scroll_percent_y(), 0.0);
    }
}
