//! Unified input system for games.
//!
//! Tracks keyboard (key down/up/held), mouse (position, delta, buttons, scroll),
//! and abstract input events. Per-frame state with previous-frame comparison for
//! just_pressed / just_released detection. Buffered input for fighting-game style
//! combos. Dead zone support for analog inputs.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Key Codes ───────────────────────────────────────────────────

/// A keyboard key, identified by its lowercase name.
pub type KeyCode = String;

// ── Mouse Button ────────────────────────────────────────────────

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

// ── Input Event ─────────────────────────────────────────────────

/// An abstract input event with a timestamp (frame number).
#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    KeyDown(KeyCode),
    KeyUp(KeyCode),
    MouseButtonDown(MouseButton),
    MouseButtonUp(MouseButton),
    MouseMove { x: f64, y: f64 },
    MouseScroll { dx: f64, dy: f64 },
    AnalogAxis { id: String, value: f64 },
}

/// A timestamped input event stored in the buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampedEvent {
    pub frame: u64,
    pub event: InputEvent,
}

// ── Mouse State ─────────────────────────────────────────────────

/// Current mouse state.
#[derive(Debug, Clone, PartialEq)]
pub struct MouseState {
    pub x: f64,
    pub y: f64,
    pub prev_x: f64,
    pub prev_y: f64,
    pub scroll_x: f64,
    pub scroll_y: f64,
    pub buttons: HashSet<MouseButton>,
    pub prev_buttons: HashSet<MouseButton>,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            x: 0.0, y: 0.0,
            prev_x: 0.0, prev_y: 0.0,
            scroll_x: 0.0, scroll_y: 0.0,
            buttons: HashSet::new(),
            prev_buttons: HashSet::new(),
        }
    }
}

impl MouseState {
    /// Delta X since last frame.
    pub fn delta_x(&self) -> f64 { self.x - self.prev_x }

    /// Delta Y since last frame.
    pub fn delta_y(&self) -> f64 { self.y - self.prev_y }

    /// True if the button was just pressed this frame.
    pub fn just_pressed(&self, button: MouseButton) -> bool {
        self.buttons.contains(&button) && !self.prev_buttons.contains(&button)
    }

    /// True if the button was just released this frame.
    pub fn just_released(&self, button: MouseButton) -> bool {
        !self.buttons.contains(&button) && self.prev_buttons.contains(&button)
    }

    /// True if the button is currently held.
    pub fn is_held(&self, button: MouseButton) -> bool {
        self.buttons.contains(&button)
    }
}

// ── Dead Zone ───────────────────────────────────────────────────

/// Dead zone configuration for analog inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadZone {
    /// Threshold below which input is zeroed.
    pub threshold: f64,
    /// Whether to rescale the remaining range to 0.0..1.0.
    pub rescale: bool,
}

impl Default for DeadZone {
    fn default() -> Self {
        Self { threshold: 0.1, rescale: true }
    }
}

impl DeadZone {
    pub fn new(threshold: f64, rescale: bool) -> Self {
        Self { threshold: threshold.abs(), rescale }
    }

    /// Apply dead zone filtering to a value in -1.0..1.0.
    pub fn apply(&self, value: f64) -> f64 {
        let abs = value.abs();
        if abs < self.threshold {
            return 0.0;
        }
        if self.rescale {
            let sign = value.signum();
            let rescaled = (abs - self.threshold) / (1.0 - self.threshold);
            sign * rescaled.min(1.0)
        } else {
            value
        }
    }
}

// ── Combo Buffer ────────────────────────────────────────────────

/// A combo pattern: a sequence of keys that must be pressed within a time window.
#[derive(Debug, Clone, PartialEq)]
pub struct ComboPattern {
    pub name: String,
    pub sequence: Vec<KeyCode>,
    /// Maximum frames between first and last input.
    pub window_frames: u64,
}

/// A detected combo match.
#[derive(Debug, Clone, PartialEq)]
pub struct ComboMatch {
    pub name: String,
    pub frame: u64,
}

// ── Input Manager ───────────────────────────────────────────────

/// Unified input manager tracking keyboard, mouse, and abstract events.
pub struct InputManager {
    current_frame: u64,
    keys_down: HashSet<KeyCode>,
    prev_keys_down: HashSet<KeyCode>,
    mouse: MouseState,
    analog_axes: HashMap<String, f64>,
    dead_zones: HashMap<String, DeadZone>,
    event_buffer: VecDeque<TimestampedEvent>,
    buffer_capacity: usize,
    combos: Vec<ComboPattern>,
    matched_combos: Vec<ComboMatch>,
}

impl InputManager {
    /// Create a new input manager with the given event buffer capacity.
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            current_frame: 0,
            keys_down: HashSet::new(),
            prev_keys_down: HashSet::new(),
            mouse: MouseState::default(),
            analog_axes: HashMap::new(),
            dead_zones: HashMap::new(),
            event_buffer: VecDeque::with_capacity(buffer_capacity),
            buffer_capacity,
            combos: Vec::new(),
            matched_combos: Vec::new(),
        }
    }

    /// Begin a new frame: snapshot previous state and reset per-frame accumulators.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.prev_keys_down = self.keys_down.clone();
        self.mouse.prev_x = self.mouse.x;
        self.mouse.prev_y = self.mouse.y;
        self.mouse.prev_buttons = self.mouse.buttons.clone();
        self.mouse.scroll_x = 0.0;
        self.mouse.scroll_y = 0.0;
        self.matched_combos.clear();
    }

    /// Current frame number.
    pub fn frame(&self) -> u64 { self.current_frame }

    // ── Keyboard ────────────────────────────────────────────

    /// Record a key-down event.
    pub fn key_down(&mut self, key: &str) {
        let k = key.to_lowercase();
        self.keys_down.insert(k.clone());
        self.push_event(InputEvent::KeyDown(k));
    }

    /// Record a key-up event.
    pub fn key_up(&mut self, key: &str) {
        let k = key.to_lowercase();
        self.keys_down.remove(&k);
        self.push_event(InputEvent::KeyUp(k));
    }

    /// True if the key was just pressed this frame.
    pub fn key_just_pressed(&self, key: &str) -> bool {
        let k = key.to_lowercase();
        self.keys_down.contains(&k) && !self.prev_keys_down.contains(&k)
    }

    /// True if the key was just released this frame.
    pub fn key_just_released(&self, key: &str) -> bool {
        let k = key.to_lowercase();
        !self.keys_down.contains(&k) && self.prev_keys_down.contains(&k)
    }

    /// True if the key is currently held down.
    pub fn key_held(&self, key: &str) -> bool {
        self.keys_down.contains(&key.to_lowercase())
    }

    /// All currently held keys.
    pub fn keys_held(&self) -> Vec<KeyCode> {
        let mut v: Vec<_> = self.keys_down.iter().cloned().collect();
        v.sort();
        v
    }

    // ── Mouse ───────────────────────────────────────────────

    /// Set mouse position.
    pub fn mouse_move(&mut self, x: f64, y: f64) {
        self.mouse.x = x;
        self.mouse.y = y;
        self.push_event(InputEvent::MouseMove { x, y });
    }

    /// Record mouse button press.
    pub fn mouse_button_down(&mut self, button: MouseButton) {
        self.mouse.buttons.insert(button);
        self.push_event(InputEvent::MouseButtonDown(button));
    }

    /// Record mouse button release.
    pub fn mouse_button_up(&mut self, button: MouseButton) {
        self.mouse.buttons.remove(&button);
        self.push_event(InputEvent::MouseButtonUp(button));
    }

    /// Record scroll input.
    pub fn mouse_scroll(&mut self, dx: f64, dy: f64) {
        self.mouse.scroll_x += dx;
        self.mouse.scroll_y += dy;
        self.push_event(InputEvent::MouseScroll { dx, dy });
    }

    /// Reference to the current mouse state.
    pub fn mouse(&self) -> &MouseState { &self.mouse }

    // ── Analog Axes ─────────────────────────────────────────

    /// Set a dead zone for the named analog axis.
    pub fn set_dead_zone(&mut self, axis_id: &str, dead_zone: DeadZone) {
        self.dead_zones.insert(axis_id.to_string(), dead_zone);
    }

    /// Set the raw value for an analog axis. Dead zone is applied on read.
    pub fn set_analog(&mut self, axis_id: &str, raw_value: f64) {
        self.analog_axes.insert(axis_id.to_string(), raw_value);
        self.push_event(InputEvent::AnalogAxis {
            id: axis_id.to_string(),
            value: raw_value,
        });
    }

    /// Read an analog axis value with dead zone applied.
    pub fn analog(&self, axis_id: &str) -> f64 {
        let raw = self.analog_axes.get(axis_id).copied().unwrap_or(0.0);
        if let Some(dz) = self.dead_zones.get(axis_id) {
            dz.apply(raw)
        } else {
            raw
        }
    }

    /// Read the raw (un-filtered) value for an analog axis.
    pub fn analog_raw(&self, axis_id: &str) -> f64 {
        self.analog_axes.get(axis_id).copied().unwrap_or(0.0)
    }

    // ── Combos ──────────────────────────────────────────────

    /// Register a combo pattern.
    pub fn register_combo(&mut self, pattern: ComboPattern) {
        self.combos.push(pattern);
    }

    /// Check all registered combos against the event buffer.
    /// Call this after processing all events for the frame (e.g. at end of begin_frame or manually).
    pub fn check_combos(&mut self) {
        let combos = self.combos.clone();
        for combo in &combos {
            if self.buffer_matches_combo(combo) {
                self.matched_combos.push(ComboMatch {
                    name: combo.name.clone(),
                    frame: self.current_frame,
                });
            }
        }
    }

    /// Return combos matched this frame.
    pub fn matched_combos(&self) -> &[ComboMatch] {
        &self.matched_combos
    }

    fn buffer_matches_combo(&self, combo: &ComboPattern) -> bool {
        if combo.sequence.is_empty() {
            return false;
        }
        // Walk the buffer backwards looking for the sequence in order
        let key_downs: Vec<&TimestampedEvent> = self.event_buffer.iter()
            .filter(|e| matches!(&e.event, InputEvent::KeyDown(_)))
            .collect();

        if key_downs.is_empty() {
            return false;
        }

        let mut seq_idx = combo.sequence.len();
        let mut last_frame = None;
        let mut first_frame = None;

        for ev in key_downs.iter().rev() {
            if seq_idx == 0 { break; }
            if let InputEvent::KeyDown(ref k) = ev.event {
                if *k == combo.sequence[seq_idx - 1] {
                    if last_frame.is_none() {
                        last_frame = Some(ev.frame);
                    }
                    first_frame = Some(ev.frame);
                    seq_idx -= 1;
                }
            }
        }

        if seq_idx != 0 { return false; }

        match (first_frame, last_frame) {
            (Some(first), Some(last)) => last - first <= combo.window_frames,
            _ => false,
        }
    }

    // ── Buffer ──────────────────────────────────────────────

    fn push_event(&mut self, event: InputEvent) {
        if self.event_buffer.len() >= self.buffer_capacity {
            self.event_buffer.pop_front();
        }
        self.event_buffer.push_back(TimestampedEvent {
            frame: self.current_frame,
            event,
        });
    }

    /// The number of events currently in the buffer.
    pub fn buffer_len(&self) -> usize { self.event_buffer.len() }

    /// Drain all events from the buffer.
    pub fn drain_buffer(&mut self) -> Vec<TimestampedEvent> {
        self.event_buffer.drain(..).collect()
    }

    /// Clear all input state (keys, mouse, axes, buffer).
    pub fn clear(&mut self) {
        self.keys_down.clear();
        self.prev_keys_down.clear();
        self.mouse = MouseState::default();
        self.analog_axes.clear();
        self.event_buffer.clear();
        self.matched_combos.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_down_up() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("w");
        assert!(mgr.key_held("w"));
        assert!(mgr.key_just_pressed("w"));
        mgr.key_up("w");
        assert!(!mgr.key_held("w"));
    }

    #[test]
    fn test_key_just_pressed_only_one_frame() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("space");
        assert!(mgr.key_just_pressed("space"));
        mgr.begin_frame();
        // Still held but not just pressed
        assert!(!mgr.key_just_pressed("space"));
        assert!(mgr.key_held("space"));
    }

    #[test]
    fn test_key_just_released() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("a");
        mgr.begin_frame();
        mgr.key_up("a");
        assert!(mgr.key_just_released("a"));
        mgr.begin_frame();
        assert!(!mgr.key_just_released("a"));
    }

    #[test]
    fn test_case_insensitive_keys() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("W");
        assert!(mgr.key_held("w"));
        assert!(mgr.key_held("W"));
    }

    #[test]
    fn test_keys_held_sorted() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("z");
        mgr.key_down("a");
        mgr.key_down("m");
        let held = mgr.keys_held();
        assert_eq!(held, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_mouse_position_and_delta() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.mouse_move(100.0, 200.0);
        assert!((mgr.mouse().x - 100.0).abs() < 1e-9);
        assert!((mgr.mouse().y - 200.0).abs() < 1e-9);
        mgr.begin_frame();
        mgr.mouse_move(110.0, 215.0);
        assert!((mgr.mouse().delta_x() - 10.0).abs() < 1e-9);
        assert!((mgr.mouse().delta_y() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_mouse_button_just_pressed() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.mouse_button_down(MouseButton::Left);
        assert!(mgr.mouse().just_pressed(MouseButton::Left));
        assert!(!mgr.mouse().just_pressed(MouseButton::Right));
        mgr.begin_frame();
        assert!(!mgr.mouse().just_pressed(MouseButton::Left));
        assert!(mgr.mouse().is_held(MouseButton::Left));
    }

    #[test]
    fn test_mouse_button_just_released() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.mouse_button_down(MouseButton::Right);
        mgr.begin_frame();
        mgr.mouse_button_up(MouseButton::Right);
        assert!(mgr.mouse().just_released(MouseButton::Right));
    }

    #[test]
    fn test_mouse_scroll() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.mouse_scroll(0.0, 3.5);
        mgr.mouse_scroll(0.0, 1.5);
        assert!((mgr.mouse().scroll_y - 5.0).abs() < 1e-9);
        mgr.begin_frame();
        assert!((mgr.mouse().scroll_y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dead_zone_zero_below_threshold() {
        let dz = DeadZone::new(0.2, true);
        assert!((dz.apply(0.1) - 0.0).abs() < 1e-9);
        assert!((dz.apply(-0.15) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dead_zone_rescale() {
        let dz = DeadZone::new(0.2, true);
        // At threshold = 0.2, value 0.6 → (0.6 - 0.2) / (1.0 - 0.2) = 0.5
        assert!((dz.apply(0.6) - 0.5).abs() < 1e-9);
        assert!((dz.apply(-0.6) - -0.5).abs() < 1e-9);
    }

    #[test]
    fn test_dead_zone_no_rescale() {
        let dz = DeadZone::new(0.2, false);
        assert!((dz.apply(0.6) - 0.6).abs() < 1e-9);
        assert!((dz.apply(0.1) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dead_zone_full_range() {
        let dz = DeadZone::new(0.2, true);
        assert!((dz.apply(1.0) - 1.0).abs() < 1e-9);
        assert!((dz.apply(-1.0) - -1.0).abs() < 1e-9);
    }

    #[test]
    fn test_analog_axis_with_dead_zone() {
        let mut mgr = InputManager::new(64);
        mgr.set_dead_zone("lx", DeadZone::new(0.15, true));
        mgr.begin_frame();
        mgr.set_analog("lx", 0.1);
        assert!((mgr.analog("lx") - 0.0).abs() < 1e-9);
        assert!((mgr.analog_raw("lx") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_analog_axis_above_dead_zone() {
        let mut mgr = InputManager::new(64);
        mgr.set_dead_zone("lx", DeadZone::new(0.2, true));
        mgr.begin_frame();
        mgr.set_analog("lx", 0.6);
        assert!((mgr.analog("lx") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_analog_axis_no_dead_zone() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.set_analog("slider", 0.05);
        assert!((mgr.analog("slider") - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_event_buffer_capacity() {
        let mut mgr = InputManager::new(4);
        mgr.begin_frame();
        mgr.key_down("a");
        mgr.key_down("b");
        mgr.key_down("c");
        mgr.key_down("d");
        assert_eq!(mgr.buffer_len(), 4);
        mgr.key_down("e");
        assert_eq!(mgr.buffer_len(), 4);
    }

    #[test]
    fn test_combo_detection() {
        let mut mgr = InputManager::new(64);
        mgr.register_combo(ComboPattern {
            name: "hadouken".to_string(),
            sequence: vec!["down".into(), "right".into(), "a".into()],
            window_frames: 10,
        });
        mgr.begin_frame();
        mgr.key_down("down");
        mgr.begin_frame();
        mgr.key_down("right");
        mgr.begin_frame();
        mgr.key_down("a");
        mgr.check_combos();
        assert_eq!(mgr.matched_combos().len(), 1);
        assert_eq!(mgr.matched_combos()[0].name, "hadouken");
    }

    #[test]
    fn test_combo_expired_window() {
        let mut mgr = InputManager::new(64);
        mgr.register_combo(ComboPattern {
            name: "combo".to_string(),
            sequence: vec!["a".into(), "b".into()],
            window_frames: 2,
        });
        mgr.begin_frame();
        mgr.key_down("a");
        // Skip many frames
        for _ in 0..5 {
            mgr.begin_frame();
        }
        mgr.key_down("b");
        mgr.check_combos();
        assert!(mgr.matched_combos().is_empty());
    }

    #[test]
    fn test_drain_buffer() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("x");
        mgr.key_up("x");
        let events = mgr.drain_buffer();
        assert_eq!(events.len(), 2);
        assert_eq!(mgr.buffer_len(), 0);
    }

    #[test]
    fn test_clear_resets_all() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("w");
        mgr.mouse_move(50.0, 50.0);
        mgr.set_analog("lx", 0.5);
        mgr.clear();
        assert!(!mgr.key_held("w"));
        assert!((mgr.mouse().x - 0.0).abs() < 1e-9);
        assert!((mgr.analog("lx") - 0.0).abs() < 1e-9);
        assert_eq!(mgr.buffer_len(), 0);
    }

    #[test]
    fn test_frame_counter() {
        let mut mgr = InputManager::new(64);
        assert_eq!(mgr.frame(), 0);
        mgr.begin_frame();
        assert_eq!(mgr.frame(), 1);
        mgr.begin_frame();
        assert_eq!(mgr.frame(), 2);
    }

    #[test]
    fn test_multiple_keys_held() {
        let mut mgr = InputManager::new(64);
        mgr.begin_frame();
        mgr.key_down("w");
        mgr.key_down("shift");
        assert!(mgr.key_held("w"));
        assert!(mgr.key_held("shift"));
        mgr.key_up("w");
        assert!(!mgr.key_held("w"));
        assert!(mgr.key_held("shift"));
    }

    #[test]
    fn test_mouse_no_delta_before_move() {
        let mgr = InputManager::new(64);
        assert!((mgr.mouse().delta_x() - 0.0).abs() < 1e-9);
        assert!((mgr.mouse().delta_y() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dead_zone_at_exactly_threshold() {
        let dz = DeadZone::new(0.2, true);
        // At exactly threshold, abs < threshold is false (equals), so it should pass through
        // 0.2 is NOT < 0.2, so it should rescale: (0.2 - 0.2) / 0.8 = 0.0
        assert!((dz.apply(0.2) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_combo_cleared_each_frame() {
        let mut mgr = InputManager::new(64);
        mgr.register_combo(ComboPattern {
            name: "quick".to_string(),
            sequence: vec!["a".into()],
            window_frames: 5,
        });
        mgr.begin_frame();
        mgr.key_down("a");
        mgr.check_combos();
        assert_eq!(mgr.matched_combos().len(), 1);
        mgr.begin_frame();
        // Combos cleared on begin_frame
        assert!(mgr.matched_combos().is_empty());
    }
}
