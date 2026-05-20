//! Touch input for mobile games.
//!
//! Multi-touch tracking (up to 10 touch points). Gesture recognition: tap,
//! double-tap, long press, swipe (direction + velocity), pinch (scale factor),
//! rotate (angle delta), and pan (delta). Gesture state machine with
//! configurable thresholds.

use std::collections::HashMap;

// ── Constants ───────────────────────────────────────────────────

/// Maximum simultaneous touch points.
pub const MAX_TOUCH_POINTS: usize = 10;

// ── Touch Point ─────────────────────────────────────────────────

/// A single touch point.
#[derive(Debug, Clone, PartialEq)]
pub struct TouchPoint {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub start_x: f64,
    pub start_y: f64,
    pub start_time_ms: f64,
    pub last_time_ms: f64,
    pub active: bool,
}

impl TouchPoint {
    fn new(id: u64, x: f64, y: f64, time_ms: f64) -> Self {
        Self {
            id, x, y,
            start_x: x, start_y: y,
            start_time_ms: time_ms,
            last_time_ms: time_ms,
            active: true,
        }
    }

    /// Distance from start to current position.
    pub fn distance_from_start(&self) -> f64 {
        let dx = self.x - self.start_x;
        let dy = self.y - self.start_y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Duration in ms since touch started.
    pub fn duration_ms(&self) -> f64 {
        self.last_time_ms - self.start_time_ms
    }
}

// ── Swipe Direction ─────────────────────────────────────────────

/// Swipe direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwipeDirection {
    Up,
    Down,
    Left,
    Right,
}

// ── Gesture ─────────────────────────────────────────────────────

/// A recognized gesture.
#[derive(Debug, Clone, PartialEq)]
pub enum Gesture {
    Tap { x: f64, y: f64 },
    DoubleTap { x: f64, y: f64 },
    LongPress { x: f64, y: f64, duration_ms: f64 },
    Swipe { direction: SwipeDirection, velocity: f64, start_x: f64, start_y: f64, end_x: f64, end_y: f64 },
    Pinch { scale_factor: f64, center_x: f64, center_y: f64 },
    Rotate { angle_delta: f64, center_x: f64, center_y: f64 },
    Pan { dx: f64, dy: f64, x: f64, y: f64 },
}

// ── Gesture State ───────────────────────────────────────────────

/// Internal state for gesture recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GesturePhase {
    Idle,
    TouchDown,
    Moving,
    MultiTouch,
}

// ── Gesture Config ──────────────────────────────────────────────

/// Configurable thresholds for gesture recognition.
#[derive(Debug, Clone, PartialEq)]
pub struct GestureConfig {
    /// Max distance (px) for a touch to count as a tap.
    pub tap_max_distance: f64,
    /// Max duration (ms) for a touch to count as a tap.
    pub tap_max_duration_ms: f64,
    /// Max interval (ms) between two taps for a double-tap.
    pub double_tap_interval_ms: f64,
    /// Min duration (ms) for a long press.
    pub long_press_min_ms: f64,
    /// Min distance (px) for a swipe.
    pub swipe_min_distance: f64,
    /// Min velocity (px/ms) for a swipe.
    pub swipe_min_velocity: f64,
    /// Min distance (px) for a pan.
    pub pan_min_distance: f64,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            tap_max_distance: 10.0,
            tap_max_duration_ms: 300.0,
            double_tap_interval_ms: 300.0,
            long_press_min_ms: 500.0,
            swipe_min_distance: 50.0,
            swipe_min_velocity: 0.3,
            pan_min_distance: 5.0,
        }
    }
}

// ── Touch Tracker ───────────────────────────────────────────────

/// Tracks touch points and recognizes gestures.
pub struct TouchTracker {
    config: GestureConfig,
    touches: HashMap<u64, TouchPoint>,
    phase: GesturePhase,
    gestures: Vec<Gesture>,
    last_tap_time_ms: f64,
    last_tap_x: f64,
    last_tap_y: f64,
    prev_pinch_distance: Option<f64>,
    prev_rotation_angle: Option<f64>,
}

impl TouchTracker {
    pub fn new(config: GestureConfig) -> Self {
        Self {
            config,
            touches: HashMap::new(),
            phase: GesturePhase::Idle,
            gestures: Vec::new(),
            last_tap_time_ms: 0.0,
            last_tap_x: 0.0,
            last_tap_y: 0.0,
            prev_pinch_distance: None,
            prev_rotation_angle: None,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(GestureConfig::default())
    }

    /// Begin a new frame: clear recognized gestures.
    pub fn begin_frame(&mut self) {
        self.gestures.clear();
    }

    /// Number of active touch points.
    pub fn active_touch_count(&self) -> usize {
        self.touches.values().filter(|t| t.active).count()
    }

    /// Get a specific touch point by id.
    pub fn touch(&self, id: u64) -> Option<&TouchPoint> {
        self.touches.get(&id).filter(|t| t.active)
    }

    /// All active touch points.
    pub fn active_touches(&self) -> Vec<&TouchPoint> {
        self.touches.values().filter(|t| t.active).collect()
    }

    /// Recognized gestures this frame.
    pub fn gestures(&self) -> &[Gesture] {
        &self.gestures
    }

    /// Record a touch-start event.
    pub fn touch_start(&mut self, id: u64, x: f64, y: f64, time_ms: f64) {
        if self.touches.values().filter(|t| t.active).count() >= MAX_TOUCH_POINTS {
            return;
        }
        self.touches.insert(id, TouchPoint::new(id, x, y, time_ms));
        let active = self.active_touch_count();
        if active == 1 {
            self.phase = GesturePhase::TouchDown;
        } else {
            self.phase = GesturePhase::MultiTouch;
            self.prev_pinch_distance = None;
            self.prev_rotation_angle = None;
        }
    }

    /// Record a touch-move event.
    pub fn touch_move(&mut self, id: u64, x: f64, y: f64, time_ms: f64) {
        // Update the touch point and extract data before any further borrows.
        let move_data = if let Some(tp) = self.touches.get_mut(&id) {
            let prev_x = tp.x;
            let prev_y = tp.y;
            tp.x = x;
            tp.y = y;
            tp.last_time_ms = time_ms;
            let dist = tp.distance_from_start();
            Some((prev_x, prev_y, dist))
        } else {
            None
        };

        if let Some((prev_x, prev_y, dist)) = move_data {
            let active_count = self.touches.values().filter(|t| t.active).count();
            if active_count == 1 && dist > self.config.pan_min_distance {
                self.phase = GesturePhase::Moving;
                self.gestures.push(Gesture::Pan {
                    dx: x - prev_x,
                    dy: y - prev_y,
                    x, y,
                });
            } else if active_count >= 2 {
                self.recognize_pinch_rotate();
            }
        }
    }

    /// Record a touch-end event.
    pub fn touch_end(&mut self, id: u64, time_ms: f64) {
        let touch_data = if let Some(tp) = self.touches.get_mut(&id) {
            tp.last_time_ms = time_ms;
            tp.active = false;
            Some((tp.start_x, tp.start_y, tp.x, tp.y,
                  tp.distance_from_start(), tp.duration_ms(), tp.start_time_ms))
        } else {
            None
        };

        if let Some((start_x, start_y, end_x, end_y, distance, duration, _start_time)) = touch_data {
            match self.phase {
                GesturePhase::TouchDown => {
                    // Potential tap
                    if distance <= self.config.tap_max_distance
                        && duration <= self.config.tap_max_duration_ms
                    {
                        // Check for double-tap
                        let dt = time_ms - self.last_tap_time_ms;
                        let tap_distance = ((end_x - self.last_tap_x).powi(2) + (end_y - self.last_tap_y).powi(2)).sqrt();
                        if dt <= self.config.double_tap_interval_ms && tap_distance <= self.config.tap_max_distance * 2.0 && self.last_tap_time_ms > 0.0 {
                            self.gestures.push(Gesture::DoubleTap { x: end_x, y: end_y });
                            self.last_tap_time_ms = 0.0;
                        } else {
                            self.gestures.push(Gesture::Tap { x: end_x, y: end_y });
                            self.last_tap_time_ms = time_ms;
                            self.last_tap_x = end_x;
                            self.last_tap_y = end_y;
                        }
                    } else if duration >= self.config.long_press_min_ms
                        && distance <= self.config.tap_max_distance
                    {
                        self.gestures.push(Gesture::LongPress {
                            x: end_x, y: end_y,
                            duration_ms: duration,
                        });
                    }
                }
                GesturePhase::Moving => {
                    // Potential swipe
                    if distance >= self.config.swipe_min_distance {
                        let velocity = if duration > 0.0 { distance / duration } else { 0.0 };
                        if velocity >= self.config.swipe_min_velocity {
                            let dx = end_x - start_x;
                            let dy = end_y - start_y;
                            let direction = if dx.abs() > dy.abs() {
                                if dx > 0.0 { SwipeDirection::Right } else { SwipeDirection::Left }
                            } else {
                                if dy > 0.0 { SwipeDirection::Down } else { SwipeDirection::Up }
                            };
                            self.gestures.push(Gesture::Swipe {
                                direction, velocity,
                                start_x, start_y, end_x, end_y,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        // Clean up inactive touches
        self.touches.retain(|_, t| t.active);
        if self.touches.is_empty() {
            self.phase = GesturePhase::Idle;
            self.prev_pinch_distance = None;
            self.prev_rotation_angle = None;
        }
    }

    /// Check for long press on active touches (call each frame).
    pub fn check_long_press(&mut self, current_time_ms: f64) {
        let mut long_presses = Vec::new();
        for tp in self.touches.values() {
            if tp.active
                && tp.distance_from_start() <= self.config.tap_max_distance
                && (current_time_ms - tp.start_time_ms) >= self.config.long_press_min_ms
                && self.phase == GesturePhase::TouchDown
            {
                long_presses.push(Gesture::LongPress {
                    x: tp.x, y: tp.y,
                    duration_ms: current_time_ms - tp.start_time_ms,
                });
            }
        }
        for g in long_presses {
            self.gestures.push(g);
        }
    }

    fn recognize_pinch_rotate(&mut self) {
        let active: Vec<&TouchPoint> = self.touches.values().filter(|t| t.active).collect();
        if active.len() < 2 { return; }

        let t1 = active[0];
        let t2 = active[1];

        // Pinch distance
        let dx = t2.x - t1.x;
        let dy = t2.y - t1.y;
        let distance = (dx * dx + dy * dy).sqrt();
        let center_x = (t1.x + t2.x) / 2.0;
        let center_y = (t1.y + t2.y) / 2.0;

        if let Some(prev_dist) = self.prev_pinch_distance {
            if prev_dist > 1e-9 {
                let scale = distance / prev_dist;
                self.gestures.push(Gesture::Pinch {
                    scale_factor: scale,
                    center_x, center_y,
                });
            }
        }
        self.prev_pinch_distance = Some(distance);

        // Rotation
        let angle = dy.atan2(dx);
        if let Some(prev_angle) = self.prev_rotation_angle {
            let mut delta = angle - prev_angle;
            // Normalize to -PI..PI
            while delta > std::f64::consts::PI { delta -= 2.0 * std::f64::consts::PI; }
            while delta < -std::f64::consts::PI { delta += 2.0 * std::f64::consts::PI; }
            if delta.abs() > 1e-9 {
                self.gestures.push(Gesture::Rotate {
                    angle_delta: delta,
                    center_x, center_y,
                });
            }
        }
        self.prev_rotation_angle = Some(angle);
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: GestureConfig) {
        self.config = config;
    }

    /// Get current configuration.
    pub fn config(&self) -> &GestureConfig {
        &self.config
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.touches.clear();
        self.gestures.clear();
        self.phase = GesturePhase::Idle;
        self.last_tap_time_ms = 0.0;
        self.prev_pinch_distance = None;
        self.prev_rotation_angle = None;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tracker() -> TouchTracker {
        TouchTracker::with_defaults()
    }

    #[test]
    fn test_touch_start_end_tap() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        assert_eq!(t.active_touch_count(), 1);
        t.touch_end(1, 100.0);
        assert_eq!(t.active_touch_count(), 0);
        assert_eq!(t.gestures().len(), 1);
        match &t.gestures()[0] {
            Gesture::Tap { x, y } => {
                assert!((*x - 100.0).abs() < 1e-9);
                assert!((*y - 100.0).abs() < 1e-9);
            }
            other => panic!("Expected Tap, got {:?}", other),
        }
    }

    #[test]
    fn test_double_tap() {
        let mut t = default_tracker();
        t.begin_frame();
        // First tap
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_end(1, 100.0);
        // Second tap
        t.touch_start(2, 102.0, 102.0, 200.0);
        t.touch_end(2, 300.0);
        let double_taps: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::DoubleTap { .. }))
            .collect();
        assert_eq!(double_taps.len(), 1);
    }

    #[test]
    fn test_long_press_on_end() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 50.0, 50.0, 0.0);
        t.touch_end(1, 600.0); // 600ms > 500ms threshold
        let lp: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::LongPress { .. }))
            .collect();
        assert_eq!(lp.len(), 1);
    }

    #[test]
    fn test_long_press_check() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 50.0, 50.0, 0.0);
        t.check_long_press(600.0);
        let lp: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::LongPress { .. }))
            .collect();
        assert_eq!(lp.len(), 1);
    }

    #[test]
    fn test_swipe_right() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 10.0, 100.0, 0.0);
        t.touch_move(1, 40.0, 100.0, 50.0);
        t.touch_move(1, 80.0, 100.0, 100.0);
        t.touch_end(1, 150.0); // distance=70, velocity=70/150≈0.47>0.3
        let swipes: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Swipe { .. }))
            .collect();
        assert_eq!(swipes.len(), 1);
        match &swipes[0] {
            Gesture::Swipe { direction, .. } => assert_eq!(*direction, SwipeDirection::Right),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_swipe_up() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 200.0, 0.0);
        t.touch_move(1, 100.0, 100.0, 100.0);
        t.touch_end(1, 150.0); // dy=-100, distance=100, velocity=100/150≈0.67
        let swipes: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Swipe { .. }))
            .collect();
        assert_eq!(swipes.len(), 1);
        match &swipes[0] {
            Gesture::Swipe { direction, .. } => assert_eq!(*direction, SwipeDirection::Up),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_pan() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_move(1, 120.0, 105.0, 50.0);
        let pans: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Pan { .. }))
            .collect();
        assert_eq!(pans.len(), 1);
        match &pans[0] {
            Gesture::Pan { dx, dy, .. } => {
                assert!((*dx - 20.0).abs() < 1e-9);
                assert!((*dy - 5.0).abs() < 1e-9);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_pinch() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_start(2, 200.0, 100.0, 0.0);
        // Move them apart
        t.touch_move(1, 50.0, 100.0, 50.0);
        t.touch_move(2, 250.0, 100.0, 50.0);
        let pinches: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Pinch { .. }))
            .collect();
        assert!(!pinches.is_empty());
        match &pinches[0] {
            Gesture::Pinch { scale_factor, .. } => {
                assert!(*scale_factor > 1.0); // spread apart = zoom in
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_rotate() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_start(2, 200.0, 100.0, 0.0);
        // First move establishes prev_rotation_angle
        t.touch_move(2, 200.0, 95.0, 25.0);
        // Second move produces a detectable rotation
        t.touch_move(2, 200.0, 50.0, 50.0);
        let rotations: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Rotate { .. }))
            .collect();
        assert!(!rotations.is_empty());
    }

    #[test]
    fn test_max_touch_points() {
        let mut t = default_tracker();
        t.begin_frame();
        for i in 0..(MAX_TOUCH_POINTS + 3) {
            t.touch_start(i as u64, 10.0 * i as f64, 10.0, 0.0);
        }
        assert_eq!(t.active_touch_count(), MAX_TOUCH_POINTS);
    }

    #[test]
    fn test_touch_point_distance() {
        let tp = TouchPoint::new(0, 0.0, 0.0, 0.0);
        assert!((tp.distance_from_start() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_touch_point_duration() {
        let mut tp = TouchPoint::new(0, 0.0, 0.0, 100.0);
        tp.last_time_ms = 350.0;
        assert!((tp.duration_ms() - 250.0).abs() < 1e-9);
    }

    #[test]
    fn test_reset() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 50.0, 50.0, 0.0);
        t.reset();
        assert_eq!(t.active_touch_count(), 0);
        assert!(t.gestures().is_empty());
    }

    #[test]
    fn test_config_change() {
        let mut t = default_tracker();
        let mut cfg = GestureConfig::default();
        cfg.tap_max_distance = 5.0;
        t.set_config(cfg.clone());
        assert!((t.config().tap_max_distance - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_swipe_below_min_distance() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_move(1, 120.0, 100.0, 50.0);
        t.touch_end(1, 80.0); // distance=20, below default 50
        let swipes: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Swipe { .. }))
            .collect();
        assert!(swipes.is_empty());
    }

    #[test]
    fn test_no_double_tap_too_slow() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 100.0, 100.0, 0.0);
        t.touch_end(1, 100.0);
        // Wait too long
        t.touch_start(2, 100.0, 100.0, 500.0);
        t.touch_end(2, 600.0);
        let double_taps: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::DoubleTap { .. }))
            .collect();
        assert!(double_taps.is_empty());
    }

    #[test]
    fn test_begin_frame_clears_gestures() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 50.0, 50.0, 0.0);
        t.touch_end(1, 50.0);
        assert!(!t.gestures().is_empty());
        t.begin_frame();
        assert!(t.gestures().is_empty());
    }

    #[test]
    fn test_get_touch_by_id() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(42, 75.0, 80.0, 0.0);
        let tp = t.touch(42).unwrap();
        assert!((tp.x - 75.0).abs() < 1e-9);
        assert!(t.touch(99).is_none());
    }

    #[test]
    fn test_swipe_left() {
        let mut t = default_tracker();
        t.begin_frame();
        t.touch_start(1, 200.0, 100.0, 0.0);
        t.touch_move(1, 100.0, 100.0, 100.0);
        t.touch_end(1, 150.0);
        let swipes: Vec<_> = t.gestures().iter()
            .filter(|g| matches!(g, Gesture::Swipe { .. }))
            .collect();
        assert_eq!(swipes.len(), 1);
        match &swipes[0] {
            Gesture::Swipe { direction, .. } => assert_eq!(*direction, SwipeDirection::Left),
            _ => unreachable!(),
        }
    }
}
