//! Touch/pointer gesture recognizer.
//!
//! Pure math gesture recognition — no DOM dependency. Processes coordinate
//! streams and emits `GestureType` events. Replaces Hammer.js and use-gesture.

use std::collections::HashMap;

// ── Types ───────────────────────────────────────────────────────

/// A raw pointer event (touch, mouse, or pen).
#[derive(Debug, Clone)]
pub struct PointerEvent {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub timestamp_ms: u64,
    pub pressure: f64,
}

/// Direction of a swipe gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwipeDirection {
    Left,
    Right,
    Up,
    Down,
}

/// A recognized gesture.
#[derive(Debug, Clone, PartialEq)]
pub enum GestureType {
    Tap,
    DoubleTap,
    LongPress,
    Pan {
        dx: f64,
        dy: f64,
        velocity_x: f64,
        velocity_y: f64,
    },
    Swipe {
        direction: SwipeDirection,
        velocity: f64,
    },
    Pinch {
        scale: f64,
        center_x: f64,
        center_y: f64,
    },
    Rotate {
        angle_deg: f64,
        center_x: f64,
        center_y: f64,
    },
}

/// Internal state of the gesture recognizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureState {
    Idle,
    Tracking,
    Recognizing,
}

/// Configuration for gesture detection thresholds.
#[derive(Debug, Clone)]
pub struct GestureConfig {
    /// Maximum duration in ms for a tap (default 300).
    pub tap_max_duration_ms: u64,
    /// Maximum gap between two taps to trigger double-tap (default 300).
    pub double_tap_max_gap_ms: u64,
    /// Minimum hold time in ms for a long press (default 500).
    pub long_press_min_ms: u64,
    /// Minimum velocity (px/ms) to register as a swipe (default 0.5).
    pub swipe_min_velocity: f64,
    /// Minimum distance for a swipe (default 50).
    pub swipe_min_distance: f64,
    /// Minimum distance before pan starts (default 10).
    pub pan_threshold: f64,
    /// Minimum scale change for pinch (default 0.05).
    pub pinch_threshold: f64,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            tap_max_duration_ms: 300,
            double_tap_max_gap_ms: 300,
            long_press_min_ms: 500,
            swipe_min_velocity: 0.5,
            swipe_min_distance: 50.0,
            pan_threshold: 10.0,
            pinch_threshold: 0.05,
        }
    }
}

// ── GestureRecognizer ───────────────────────────────────────────

/// Stateful gesture recognizer that processes pointer events and
/// emits `GestureType` events.
pub struct GestureRecognizer {
    pub config: GestureConfig,
    active_pointers: HashMap<u32, Vec<PointerEvent>>,
    state: GestureState,
    last_tap: Option<(f64, f64, u64)>,
    /// Initial distance between two pointers when pinch/rotate started.
    initial_two_pointer_distance: Option<f64>,
    /// Initial angle between two pointers when rotation started.
    initial_two_pointer_angle: Option<f64>,
}

impl GestureRecognizer {
    /// Create a recognizer with default thresholds.
    pub fn new() -> Self {
        Self::with_config(GestureConfig::default())
    }

    /// Create a recognizer with custom thresholds.
    pub fn with_config(config: GestureConfig) -> Self {
        Self {
            config,
            active_pointers: HashMap::new(),
            state: GestureState::Idle,
            last_tap: None,
            initial_two_pointer_distance: None,
            initial_two_pointer_angle: None,
        }
    }

    /// Number of pointers currently being tracked.
    pub fn active_pointer_count(&self) -> usize {
        self.active_pointers.len()
    }

    /// Record a pointer-down event. May produce gestures immediately (rare).
    pub fn pointer_down(&mut self, event: PointerEvent) -> Vec<GestureType> {
        self.active_pointers
            .entry(event.id)
            .or_default()
            .push(event);
        self.state = GestureState::Tracking;

        // If we now have exactly 2 pointers, record initial distance/angle.
        if self.active_pointers.len() == 2 {
            let pts = self.latest_two_points();
            if let Some(((x1, y1), (x2, y2))) = pts {
                self.initial_two_pointer_distance = Some(distance(x1, y1, x2, y2));
                self.initial_two_pointer_angle = Some(angle_deg(x1, y1, x2, y2));
            }
        }

        Vec::new()
    }

    /// Record a pointer-move event. May emit Pan, Pinch, or Rotate.
    pub fn pointer_move(&mut self, event: PointerEvent) -> Vec<GestureType> {
        let mut gestures = Vec::new();

        let id = event.id;
        self.active_pointers.entry(id).or_default().push(event);

        // Two pointers → pinch / rotate
        if self.active_pointers.len() == 2 {
            if let Some(((x1, y1), (x2, y2))) = self.latest_two_points() {
                let current_dist = distance(x1, y1, x2, y2);
                let current_angle = angle_deg(x1, y1, x2, y2);
                let cx = (x1 + x2) / 2.0;
                let cy = (y1 + y2) / 2.0;

                if let Some(init_dist) = self.initial_two_pointer_distance {
                    if init_dist > 0.0 {
                        let scale = current_dist / init_dist;
                        if (scale - 1.0).abs() >= self.config.pinch_threshold {
                            self.state = GestureState::Recognizing;
                            gestures.push(GestureType::Pinch {
                                scale,
                                center_x: cx,
                                center_y: cy,
                            });
                        }
                    }
                }

                if let Some(init_angle) = self.initial_two_pointer_angle {
                    let delta = current_angle - init_angle;
                    if delta.abs() > 1.0 {
                        self.state = GestureState::Recognizing;
                        gestures.push(GestureType::Rotate {
                            angle_deg: delta,
                            center_x: cx,
                            center_y: cy,
                        });
                    }
                }
            }
            return gestures;
        }

        // Single pointer → pan
        if let Some(trail) = self.active_pointers.get(&id) {
            if trail.len() >= 2 {
                let first = &trail[0];
                let last = &trail[trail.len() - 1];
                let dx = last.x - first.x;
                let dy = last.y - first.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist >= self.config.pan_threshold {
                    self.state = GestureState::Recognizing;
                    let dt = (last.timestamp_ms.saturating_sub(first.timestamp_ms)) as f64;
                    let vx = if dt > 0.0 { dx / dt } else { 0.0 };
                    let vy = if dt > 0.0 { dy / dt } else { 0.0 };
                    gestures.push(GestureType::Pan {
                        dx,
                        dy,
                        velocity_x: vx,
                        velocity_y: vy,
                    });
                }
            }
        }

        gestures
    }

    /// Record a pointer-up event. May emit Tap, DoubleTap, LongPress, or Swipe.
    pub fn pointer_up(&mut self, event: PointerEvent) -> Vec<GestureType> {
        let mut gestures = Vec::new();
        let id = event.id;

        // Add this final event to the trail.
        self.active_pointers.entry(id).or_default().push(event);

        if let Some(trail) = self.active_pointers.remove(&id) {
            if trail.len() >= 2 {
                let first = &trail[0];
                let last = &trail[trail.len() - 1];
                let dx = last.x - first.x;
                let dy = last.y - first.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let duration_ms = last.timestamp_ms.saturating_sub(first.timestamp_ms);
                let dt = duration_ms as f64;

                // Long press: held long enough without significant movement
                if duration_ms >= self.config.long_press_min_ms
                    && dist < self.config.pan_threshold
                {
                    gestures.push(GestureType::LongPress);
                }
                // Swipe: fast, long movement
                else if dist >= self.config.swipe_min_distance && dt > 0.0 {
                    let velocity = dist / dt;
                    if velocity >= self.config.swipe_min_velocity {
                        let direction = if dx.abs() > dy.abs() {
                            if dx > 0.0 {
                                SwipeDirection::Right
                            } else {
                                SwipeDirection::Left
                            }
                        } else if dy > 0.0 {
                            SwipeDirection::Down
                        } else {
                            SwipeDirection::Up
                        };
                        gestures.push(GestureType::Swipe {
                            direction,
                            velocity,
                        });
                    }
                }
                // Tap: short, small movement
                else if duration_ms <= self.config.tap_max_duration_ms
                    && dist < self.config.pan_threshold
                {
                    // Check double tap
                    if let Some((lx, ly, lt)) = self.last_tap {
                        let gap = first.timestamp_ms.saturating_sub(lt);
                        let tap_dist = distance(lx, ly, first.x, first.y);
                        if gap <= self.config.double_tap_max_gap_ms
                            && tap_dist < self.config.pan_threshold
                        {
                            gestures.push(GestureType::DoubleTap);
                            self.last_tap = None;
                        } else {
                            gestures.push(GestureType::Tap);
                            self.last_tap = Some((first.x, first.y, first.timestamp_ms));
                        }
                    } else {
                        gestures.push(GestureType::Tap);
                        self.last_tap = Some((first.x, first.y, first.timestamp_ms));
                    }
                }
            }
        }

        if self.active_pointers.is_empty() {
            self.state = GestureState::Idle;
            self.initial_two_pointer_distance = None;
            self.initial_two_pointer_angle = None;
        }

        gestures
    }

    /// Cancel tracking for a pointer.
    pub fn pointer_cancel(&mut self, id: u32) {
        self.active_pointers.remove(&id);
        if self.active_pointers.is_empty() {
            self.state = GestureState::Idle;
            self.initial_two_pointer_distance = None;
            self.initial_two_pointer_angle = None;
        }
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn latest_two_points(&self) -> Option<((f64, f64), (f64, f64))> {
        let mut iter = self.active_pointers.values();
        let a = iter.next()?.last()?;
        let b = iter.next()?.last()?;
        Some(((a.x, a.y), (b.x, b.y)))
    }
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Math helpers ────────────────────────────────────────────────

fn distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

fn angle_deg(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dy = y2 - y1;
    let dx = x2 - x1;
    dy.atan2(dx).to_degrees()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pe(id: u32, x: f64, y: f64, ts: u64) -> PointerEvent {
        PointerEvent {
            id,
            x,
            y,
            timestamp_ms: ts,
            pressure: 1.0,
        }
    }

    #[test]
    fn tap_detection() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 100.0, 100.0, 0));
        let g = r.pointer_up(pe(0, 100.0, 100.0, 100));
        assert!(g.contains(&GestureType::Tap));
    }

    #[test]
    fn double_tap() {
        let mut r = GestureRecognizer::new();
        // First tap
        r.pointer_down(pe(0, 100.0, 100.0, 0));
        r.pointer_up(pe(0, 100.0, 100.0, 50));
        // Second tap
        r.pointer_down(pe(0, 100.0, 100.0, 100));
        let g = r.pointer_up(pe(0, 100.0, 100.0, 150));
        assert!(g.contains(&GestureType::DoubleTap));
    }

    #[test]
    fn pan_with_velocity() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        let g = r.pointer_move(pe(0, 50.0, 0.0, 100));
        assert!(!g.is_empty());
        match &g[0] {
            GestureType::Pan {
                dx,
                velocity_x,
                ..
            } => {
                assert!(*dx > 0.0);
                assert!(*velocity_x > 0.0);
            }
            other => panic!("Expected Pan, got {:?}", other),
        }
    }

    #[test]
    fn swipe_direction_detection() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        // Large fast movement to the right
        let g = r.pointer_up(pe(0, 200.0, 0.0, 100));
        let swipe = g
            .iter()
            .find(|g| matches!(g, GestureType::Swipe { .. }));
        assert!(swipe.is_some());
        match swipe.unwrap() {
            GestureType::Swipe { direction, .. } => {
                assert_eq!(*direction, SwipeDirection::Right);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn pinch_scale_computation() {
        let mut r = GestureRecognizer::new();
        // Two pointers 100 px apart
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        r.pointer_down(pe(1, 100.0, 0.0, 0));
        // Move them 200 px apart → scale ≈ 2.0
        r.pointer_move(pe(0, -50.0, 0.0, 100));
        let g = r.pointer_move(pe(1, 150.0, 0.0, 100));
        let pinch = g.iter().find(|g| matches!(g, GestureType::Pinch { .. }));
        assert!(pinch.is_some(), "Expected a Pinch gesture");
        match pinch.unwrap() {
            GestureType::Pinch { scale, .. } => {
                assert!(*scale > 1.5);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn single_pointer_no_pinch() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        let g = r.pointer_move(pe(0, 50.0, 0.0, 100));
        for gesture in &g {
            assert!(!matches!(gesture, GestureType::Pinch { .. }));
        }
    }

    #[test]
    fn cancel_clears_state() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        assert_eq!(r.active_pointer_count(), 1);
        r.pointer_cancel(0);
        assert_eq!(r.active_pointer_count(), 0);
    }

    #[test]
    fn default_config_values() {
        let cfg = GestureConfig::default();
        assert_eq!(cfg.tap_max_duration_ms, 300);
        assert_eq!(cfg.double_tap_max_gap_ms, 300);
        assert_eq!(cfg.long_press_min_ms, 500);
        assert!((cfg.swipe_min_velocity - 0.5).abs() < f64::EPSILON);
        assert!((cfg.swipe_min_distance - 50.0).abs() < f64::EPSILON);
        assert!((cfg.pan_threshold - 10.0).abs() < f64::EPSILON);
        assert!((cfg.pinch_threshold - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn long_press_detection() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 100.0, 100.0, 0));
        let g = r.pointer_up(pe(0, 100.0, 100.0, 600));
        assert!(g.contains(&GestureType::LongPress));
    }

    #[test]
    fn swipe_requires_min_velocity() {
        let mut r = GestureRecognizer::new();
        r.pointer_down(pe(0, 0.0, 0.0, 0));
        // 60 px in 5000 ms → velocity = 0.012, well below 0.5
        let g = r.pointer_up(pe(0, 60.0, 0.0, 5000));
        let swipe = g
            .iter()
            .find(|g| matches!(g, GestureType::Swipe { .. }));
        assert!(swipe.is_none(), "Should not detect swipe at low velocity");
    }
}
