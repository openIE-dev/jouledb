//! Gamepad API: button/axis state, dead-zone filtering, standard mapping.

use std::collections::HashMap;

// ── Standard Mapping Constants ──────────────────────────────────

pub const BUTTON_A: usize = 0;
pub const BUTTON_B: usize = 1;
pub const BUTTON_X: usize = 2;
pub const BUTTON_Y: usize = 3;
pub const LEFT_BUMPER: usize = 4;
pub const RIGHT_BUMPER: usize = 5;
pub const LEFT_TRIGGER: usize = 6;
pub const RIGHT_TRIGGER: usize = 7;
pub const SELECT: usize = 8;
pub const START: usize = 9;
pub const LEFT_STICK: usize = 10;
pub const RIGHT_STICK: usize = 11;
pub const DPAD_UP: usize = 12;
pub const DPAD_DOWN: usize = 13;
pub const DPAD_LEFT: usize = 14;
pub const DPAD_RIGHT: usize = 15;

pub const AXIS_LEFT_X: usize = 0;
pub const AXIS_LEFT_Y: usize = 1;
pub const AXIS_RIGHT_X: usize = 2;
pub const AXIS_RIGHT_Y: usize = 3;

// ── Types ───────────────────────────────────────────────────────

/// State of a single gamepad button.
#[derive(Debug, Clone, PartialEq)]
pub struct GamepadButton {
    pub pressed: bool,
    pub value: f64,
    pub touched: bool,
}

/// Controller mapping type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamepadMapping {
    Standard,
    Unknown,
}

/// A connected gamepad.
#[derive(Debug, Clone)]
pub struct Gamepad {
    pub id: String,
    pub index: u32,
    pub connected: bool,
    pub buttons: Vec<GamepadButton>,
    pub axes: Vec<f64>,
    pub mapping: GamepadMapping,
    pub timestamp: u64,
}

// ── Manager ─────────────────────────────────────────────────────

/// Manages connected gamepads and provides dead-zone–filtered reads.
#[derive(Debug)]
pub struct GamepadManager {
    gamepads: HashMap<u32, Gamepad>,
    deadzone: f64,
}

impl GamepadManager {
    pub fn new(deadzone: f64) -> Self {
        Self {
            gamepads: HashMap::new(),
            deadzone,
        }
    }

    pub fn connect(&mut self, gamepad: Gamepad) {
        self.gamepads.insert(gamepad.index, gamepad);
    }

    pub fn disconnect(&mut self, index: u32) {
        if let Some(gp) = self.gamepads.get_mut(&index) {
            gp.connected = false;
        }
        self.gamepads.remove(&index);
    }

    pub fn update(&mut self, index: u32, buttons: Vec<GamepadButton>, axes: Vec<f64>) {
        if let Some(gp) = self.gamepads.get_mut(&index) {
            gp.buttons = buttons;
            gp.axes = axes;
        }
    }

    pub fn get(&self, index: u32) -> Option<&Gamepad> {
        self.gamepads.get(&index)
    }

    pub fn connected_count(&self) -> usize {
        self.gamepads.values().filter(|g| g.connected).count()
    }

    /// Apply dead-zone: values within `[-deadzone, deadzone]` become 0.
    pub fn apply_deadzone(value: f64, deadzone: f64) -> f64 {
        if value.abs() < deadzone {
            0.0
        } else {
            value
        }
    }

    /// Check if a button is pressed on a specific gamepad.
    pub fn is_button_pressed(&self, index: u32, button: usize) -> bool {
        self.gamepads
            .get(&index)
            .and_then(|gp| gp.buttons.get(button))
            .is_some_and(|b| b.pressed)
    }

    /// Get an axis value with dead-zone applied.
    pub fn get_axis(&self, index: u32, axis: usize) -> f64 {
        let raw = self
            .gamepads
            .get(&index)
            .and_then(|gp| gp.axes.get(axis).copied())
            .unwrap_or(0.0);
        Self::apply_deadzone(raw, self.deadzone)
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn make_gamepad(index: u32) -> Gamepad {
    Gamepad {
        id: format!("Gamepad {index}"),
        index,
        connected: true,
        buttons: vec![
            GamepadButton { pressed: false, value: 0.0, touched: false };
            16
        ],
        axes: vec![0.0; 4],
        mapping: GamepadMapping::Standard,
        timestamp: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_disconnect() {
        let mut mgr = GamepadManager::new(0.1);
        let gp = make_gamepad(0);
        mgr.connect(gp);
        assert_eq!(mgr.connected_count(), 1);
        mgr.disconnect(0);
        assert_eq!(mgr.connected_count(), 0);
    }

    #[test]
    fn button_press_detection() {
        let mut mgr = GamepadManager::new(0.1);
        let mut gp = make_gamepad(0);
        gp.buttons[BUTTON_A].pressed = true;
        mgr.connect(gp);
        assert!(mgr.is_button_pressed(0, BUTTON_A));
        assert!(!mgr.is_button_pressed(0, BUTTON_B));
    }

    #[test]
    fn axis_with_deadzone() {
        let mut mgr = GamepadManager::new(0.15);
        let mut gp = make_gamepad(0);
        gp.axes[AXIS_LEFT_X] = 0.05; // within deadzone
        gp.axes[AXIS_LEFT_Y] = 0.8;  // outside deadzone
        mgr.connect(gp);
        assert!((mgr.get_axis(0, AXIS_LEFT_X) - 0.0).abs() < f64::EPSILON);
        assert!((mgr.get_axis(0, AXIS_LEFT_Y) - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn deadzone_static() {
        assert!((GamepadManager::apply_deadzone(0.05, 0.1) - 0.0).abs() < f64::EPSILON);
        assert!((GamepadManager::apply_deadzone(0.5, 0.1) - 0.5).abs() < f64::EPSILON);
        assert!((GamepadManager::apply_deadzone(-0.05, 0.1) - 0.0).abs() < f64::EPSILON);
        assert!((GamepadManager::apply_deadzone(-0.5, 0.1) - -0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn standard_mapping_constants() {
        assert_eq!(BUTTON_A, 0);
        assert_eq!(BUTTON_Y, 3);
        assert_eq!(DPAD_RIGHT, 15);
        assert_eq!(AXIS_RIGHT_Y, 3);
    }

    #[test]
    fn multiple_gamepads() {
        let mut mgr = GamepadManager::new(0.1);
        mgr.connect(make_gamepad(0));
        mgr.connect(make_gamepad(1));
        assert_eq!(mgr.connected_count(), 2);
        assert!(mgr.get(0).is_some());
        assert!(mgr.get(1).is_some());
        assert!(mgr.get(2).is_none());
    }

    #[test]
    fn update_buttons_and_axes() {
        let mut mgr = GamepadManager::new(0.1);
        mgr.connect(make_gamepad(0));
        assert!(!mgr.is_button_pressed(0, BUTTON_X));

        let mut new_buttons = vec![
            GamepadButton { pressed: false, value: 0.0, touched: false };
            16
        ];
        new_buttons[BUTTON_X].pressed = true;
        let new_axes = vec![0.9, 0.0, 0.0, 0.0];
        mgr.update(0, new_buttons, new_axes);

        assert!(mgr.is_button_pressed(0, BUTTON_X));
        assert!((mgr.get_axis(0, AXIS_LEFT_X) - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn get_nonexistent_gamepad() {
        let mgr = GamepadManager::new(0.1);
        assert!(mgr.get(99).is_none());
        assert!(!mgr.is_button_pressed(99, BUTTON_A));
        assert!((mgr.get_axis(99, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gamepad_mapping_type() {
        let gp = make_gamepad(0);
        assert_eq!(gp.mapping, GamepadMapping::Standard);
    }
}
