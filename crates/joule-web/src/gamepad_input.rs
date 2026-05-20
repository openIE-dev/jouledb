//! Gamepad/controller abstraction for games.
//!
//! Supports up to 4 gamepads, each with standard buttons (A/B/X/Y, bumpers,
//! triggers, sticks, d-pad), analog axes (left/right stick X/Y, triggers 0.0-1.0),
//! dead zone filtering (radial and axial), button mapping profiles, and
//! rumble/vibration commands.

use std::collections::HashMap;

// ── Constants ───────────────────────────────────────────────────

/// Maximum number of simultaneously connected gamepads.
pub const MAX_GAMEPADS: usize = 4;

// ── Gamepad Button ──────────────────────────────────────────────

/// Standard gamepad button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadBtn {
    A,
    B,
    X,
    Y,
    LeftBumper,
    RightBumper,
    LeftTriggerBtn,
    RightTriggerBtn,
    Select,
    Start,
    LeftStickClick,
    RightStickClick,
    DpadUp,
    DpadDown,
    DpadLeft,
    DpadRight,
    Home,
}

// ── Gamepad Axis ────────────────────────────────────────────────

/// Standard gamepad axis identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

// ── Dead Zone Mode ──────────────────────────────────────────────

/// Dead zone filtering mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeadZoneMode {
    /// Per-axis: each axis is independently filtered.
    Axial { threshold: f64 },
    /// Radial: the magnitude of the (x, y) vector is filtered.
    Radial { threshold: f64 },
}

impl DeadZoneMode {
    /// Apply axial dead zone to a single value.
    pub fn apply_axial(value: f64, threshold: f64) -> f64 {
        let abs = value.abs();
        if abs < threshold {
            0.0
        } else {
            let sign = value.signum();
            let rescaled = (abs - threshold) / (1.0 - threshold);
            sign * rescaled.min(1.0)
        }
    }

    /// Apply radial dead zone to an (x, y) pair. Returns filtered (x, y).
    pub fn apply_radial(x: f64, y: f64, threshold: f64) -> (f64, f64) {
        let mag = (x * x + y * y).sqrt();
        if mag < threshold {
            (0.0, 0.0)
        } else {
            let scale = ((mag - threshold) / (1.0 - threshold)).min(1.0) / mag;
            (x * scale, y * scale)
        }
    }
}

// ── Button Mapping Profile ──────────────────────────────────────

/// A mapping from physical button to logical button, supporting remapping.
#[derive(Debug, Clone, PartialEq)]
pub struct ButtonMappingProfile {
    pub name: String,
    mappings: HashMap<GamepadBtn, GamepadBtn>,
}

impl ButtonMappingProfile {
    /// Create a default (identity) mapping.
    pub fn identity(name: &str) -> Self {
        Self {
            name: name.to_string(),
            mappings: HashMap::new(),
        }
    }

    /// Remap a physical button to a different logical button.
    pub fn remap(&mut self, physical: GamepadBtn, logical: GamepadBtn) {
        self.mappings.insert(physical, logical);
    }

    /// Resolve a physical button to its logical equivalent.
    pub fn resolve(&self, physical: GamepadBtn) -> GamepadBtn {
        self.mappings.get(&physical).copied().unwrap_or(physical)
    }

    /// Remove a remapping.
    pub fn clear_remap(&mut self, physical: GamepadBtn) {
        self.mappings.remove(&physical);
    }

    /// Number of active remappings.
    pub fn remap_count(&self) -> usize {
        self.mappings.len()
    }
}

// ── Rumble Command ──────────────────────────────────────────────

/// A vibration/rumble command.
#[derive(Debug, Clone, PartialEq)]
pub struct RumbleCommand {
    /// Intensity of the strong (low-frequency) motor, 0.0 to 1.0.
    pub strong_magnitude: f64,
    /// Intensity of the weak (high-frequency) motor, 0.0 to 1.0.
    pub weak_magnitude: f64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

impl RumbleCommand {
    pub fn new(strong: f64, weak: f64, duration_ms: u64) -> Self {
        Self {
            strong_magnitude: strong.clamp(0.0, 1.0),
            weak_magnitude: weak.clamp(0.0, 1.0),
            duration_ms,
        }
    }

    /// Light tap feedback.
    pub fn light_tap() -> Self { Self::new(0.2, 0.0, 50) }

    /// Medium impact feedback.
    pub fn medium_impact() -> Self { Self::new(0.5, 0.3, 150) }

    /// Heavy explosion feedback.
    pub fn heavy_explosion() -> Self { Self::new(1.0, 0.8, 400) }
}

// ── Single Gamepad State ────────────────────────────────────────

/// State of a single gamepad.
#[derive(Debug, Clone)]
pub struct GamepadState {
    pub slot: usize,
    pub connected: bool,
    pub name: String,
    buttons: HashMap<GamepadBtn, bool>,
    prev_buttons: HashMap<GamepadBtn, bool>,
    axes: HashMap<GamepadAxis, f64>,
    dead_zone_left: DeadZoneMode,
    dead_zone_right: DeadZoneMode,
    trigger_dead_zone: f64,
    mapping_profile: ButtonMappingProfile,
    pending_rumble: Option<RumbleCommand>,
}

impl GamepadState {
    fn new(slot: usize) -> Self {
        Self {
            slot,
            connected: false,
            name: String::new(),
            buttons: HashMap::new(),
            prev_buttons: HashMap::new(),
            axes: HashMap::new(),
            dead_zone_left: DeadZoneMode::Radial { threshold: 0.15 },
            dead_zone_right: DeadZoneMode::Radial { threshold: 0.15 },
            trigger_dead_zone: 0.05,
            mapping_profile: ButtonMappingProfile::identity("default"),
            pending_rumble: None,
        }
    }

    /// Snapshot current buttons to previous for frame comparison.
    pub fn begin_frame(&mut self) {
        self.prev_buttons = self.buttons.clone();
        self.pending_rumble = None;
    }

    /// Set a button state (physical button).
    pub fn set_button(&mut self, button: GamepadBtn, pressed: bool) {
        let logical = self.mapping_profile.resolve(button);
        self.buttons.insert(logical, pressed);
    }

    /// Set a raw axis value.
    pub fn set_axis(&mut self, axis: GamepadAxis, value: f64) {
        self.axes.insert(axis, value);
    }

    /// Is the logical button currently pressed?
    pub fn is_pressed(&self, button: GamepadBtn) -> bool {
        self.buttons.get(&button).copied().unwrap_or(false)
    }

    /// Was the button just pressed this frame?
    pub fn just_pressed(&self, button: GamepadBtn) -> bool {
        let cur = self.buttons.get(&button).copied().unwrap_or(false);
        let prev = self.prev_buttons.get(&button).copied().unwrap_or(false);
        cur && !prev
    }

    /// Was the button just released this frame?
    pub fn just_released(&self, button: GamepadBtn) -> bool {
        let cur = self.buttons.get(&button).copied().unwrap_or(false);
        let prev = self.prev_buttons.get(&button).copied().unwrap_or(false);
        !cur && prev
    }

    /// Get the filtered left stick as (x, y).
    pub fn left_stick(&self) -> (f64, f64) {
        let rx = self.axes.get(&GamepadAxis::LeftStickX).copied().unwrap_or(0.0);
        let ry = self.axes.get(&GamepadAxis::LeftStickY).copied().unwrap_or(0.0);
        match self.dead_zone_left {
            DeadZoneMode::Axial { threshold } => (
                DeadZoneMode::apply_axial(rx, threshold),
                DeadZoneMode::apply_axial(ry, threshold),
            ),
            DeadZoneMode::Radial { threshold } => {
                DeadZoneMode::apply_radial(rx, ry, threshold)
            }
        }
    }

    /// Get the filtered right stick as (x, y).
    pub fn right_stick(&self) -> (f64, f64) {
        let rx = self.axes.get(&GamepadAxis::RightStickX).copied().unwrap_or(0.0);
        let ry = self.axes.get(&GamepadAxis::RightStickY).copied().unwrap_or(0.0);
        match self.dead_zone_right {
            DeadZoneMode::Axial { threshold } => (
                DeadZoneMode::apply_axial(rx, threshold),
                DeadZoneMode::apply_axial(ry, threshold),
            ),
            DeadZoneMode::Radial { threshold } => {
                DeadZoneMode::apply_radial(rx, ry, threshold)
            }
        }
    }

    /// Get the filtered trigger value (0.0 to 1.0).
    pub fn trigger(&self, axis: GamepadAxis) -> f64 {
        let raw = self.axes.get(&axis).copied().unwrap_or(0.0);
        if raw < self.trigger_dead_zone { 0.0 }
        else { ((raw - self.trigger_dead_zone) / (1.0 - self.trigger_dead_zone)).min(1.0) }
    }

    /// Set dead zone for left stick.
    pub fn set_left_dead_zone(&mut self, mode: DeadZoneMode) {
        self.dead_zone_left = mode;
    }

    /// Set dead zone for right stick.
    pub fn set_right_dead_zone(&mut self, mode: DeadZoneMode) {
        self.dead_zone_right = mode;
    }

    /// Set trigger dead zone threshold.
    pub fn set_trigger_dead_zone(&mut self, threshold: f64) {
        self.trigger_dead_zone = threshold.abs();
    }

    /// Set button mapping profile.
    pub fn set_mapping_profile(&mut self, profile: ButtonMappingProfile) {
        self.mapping_profile = profile;
    }

    /// Queue a rumble command.
    pub fn rumble(&mut self, command: RumbleCommand) {
        self.pending_rumble = Some(command);
    }

    /// Take the pending rumble command (if any).
    pub fn take_rumble(&mut self) -> Option<RumbleCommand> {
        self.pending_rumble.take()
    }
}

// ── Gamepad System ──────────────────────────────────────────────

/// Manages up to MAX_GAMEPADS gamepads.
pub struct GamepadSystem {
    gamepads: Vec<GamepadState>,
}

impl GamepadSystem {
    pub fn new() -> Self {
        let mut gamepads = Vec::with_capacity(MAX_GAMEPADS);
        for i in 0..MAX_GAMEPADS {
            gamepads.push(GamepadState::new(i));
        }
        Self { gamepads }
    }

    /// Begin frame for all gamepads.
    pub fn begin_frame(&mut self) {
        for gp in &mut self.gamepads {
            gp.begin_frame();
        }
    }

    /// Connect a gamepad to a slot.
    pub fn connect(&mut self, slot: usize, name: &str) -> bool {
        if slot >= MAX_GAMEPADS { return false; }
        self.gamepads[slot].connected = true;
        self.gamepads[slot].name = name.to_string();
        true
    }

    /// Disconnect a gamepad.
    pub fn disconnect(&mut self, slot: usize) -> bool {
        if slot >= MAX_GAMEPADS { return false; }
        self.gamepads[slot].connected = false;
        self.gamepads[slot].buttons.clear();
        self.gamepads[slot].axes.clear();
        true
    }

    /// Get gamepad state by slot.
    pub fn gamepad(&self, slot: usize) -> Option<&GamepadState> {
        self.gamepads.get(slot).filter(|gp| gp.connected)
    }

    /// Get mutable gamepad state by slot.
    pub fn gamepad_mut(&mut self, slot: usize) -> Option<&mut GamepadState> {
        self.gamepads.get_mut(slot).filter(|gp| gp.connected)
    }

    /// Number of connected gamepads.
    pub fn connected_count(&self) -> usize {
        self.gamepads.iter().filter(|gp| gp.connected).count()
    }

    /// Indices of connected gamepads.
    pub fn connected_slots(&self) -> Vec<usize> {
        self.gamepads.iter()
            .filter(|gp| gp.connected)
            .map(|gp| gp.slot)
            .collect()
    }

    /// Collect pending rumble commands from all gamepads.
    pub fn collect_rumbles(&mut self) -> Vec<(usize, RumbleCommand)> {
        let mut rumbles = Vec::new();
        for gp in &mut self.gamepads {
            if let Some(cmd) = gp.take_rumble() {
                rumbles.push((gp.slot, cmd));
            }
        }
        rumbles
    }
}

impl Default for GamepadSystem {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamepad_connect_disconnect() {
        let mut sys = GamepadSystem::new();
        assert_eq!(sys.connected_count(), 0);
        sys.connect(0, "Xbox Controller");
        assert_eq!(sys.connected_count(), 1);
        sys.disconnect(0);
        assert_eq!(sys.connected_count(), 0);
    }

    #[test]
    fn test_max_gamepads() {
        let mut sys = GamepadSystem::new();
        for i in 0..MAX_GAMEPADS {
            assert!(sys.connect(i, &format!("Pad {}", i)));
        }
        assert!(!sys.connect(MAX_GAMEPADS, "overflow"));
        assert_eq!(sys.connected_count(), MAX_GAMEPADS);
    }

    #[test]
    fn test_button_press_release() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        sys.begin_frame();
        sys.gamepad_mut(0).unwrap().set_button(GamepadBtn::A, true);
        assert!(sys.gamepad(0).unwrap().is_pressed(GamepadBtn::A));
        assert!(sys.gamepad(0).unwrap().just_pressed(GamepadBtn::A));
    }

    #[test]
    fn test_just_pressed_one_frame() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        sys.begin_frame();
        sys.gamepad_mut(0).unwrap().set_button(GamepadBtn::B, true);
        assert!(sys.gamepad(0).unwrap().just_pressed(GamepadBtn::B));
        sys.begin_frame();
        assert!(!sys.gamepad(0).unwrap().just_pressed(GamepadBtn::B));
        assert!(sys.gamepad(0).unwrap().is_pressed(GamepadBtn::B));
    }

    #[test]
    fn test_just_released() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        sys.begin_frame();
        sys.gamepad_mut(0).unwrap().set_button(GamepadBtn::X, true);
        sys.begin_frame();
        sys.gamepad_mut(0).unwrap().set_button(GamepadBtn::X, false);
        assert!(sys.gamepad(0).unwrap().just_released(GamepadBtn::X));
    }

    #[test]
    fn test_axial_dead_zone() {
        let result = DeadZoneMode::apply_axial(0.1, 0.15);
        assert!((result - 0.0).abs() < 1e-9);
        let result2 = DeadZoneMode::apply_axial(0.5, 0.15);
        let expected = (0.5 - 0.15) / (1.0 - 0.15);
        assert!((result2 - expected).abs() < 1e-9);
    }

    #[test]
    fn test_radial_dead_zone_below() {
        let (x, y) = DeadZoneMode::apply_radial(0.05, 0.05, 0.15);
        assert!((x - 0.0).abs() < 1e-9);
        assert!((y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_radial_dead_zone_above() {
        let (x, y) = DeadZoneMode::apply_radial(0.7, 0.0, 0.15);
        let mag: f64 = 0.7;
        let expected_scale = ((mag - 0.15) / (1.0 - 0.15)).min(1.0) / mag;
        assert!((x - 0.7 * expected_scale).abs() < 1e-6);
        assert!((y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_left_stick_radial() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        let gp = sys.gamepad_mut(0).unwrap();
        gp.set_left_dead_zone(DeadZoneMode::Radial { threshold: 0.2 });
        gp.set_axis(GamepadAxis::LeftStickX, 0.1);
        gp.set_axis(GamepadAxis::LeftStickY, 0.1);
        let (x, y) = gp.left_stick();
        // Magnitude ~ 0.141, below 0.2 threshold
        assert!((x - 0.0).abs() < 1e-9);
        assert!((y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_trigger_dead_zone() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        let gp = sys.gamepad_mut(0).unwrap();
        gp.set_trigger_dead_zone(0.1);
        gp.set_axis(GamepadAxis::LeftTrigger, 0.05);
        assert!((gp.trigger(GamepadAxis::LeftTrigger) - 0.0).abs() < 1e-9);
        gp.set_axis(GamepadAxis::LeftTrigger, 0.55);
        let expected = (0.55 - 0.1) / (1.0 - 0.1);
        assert!((gp.trigger(GamepadAxis::LeftTrigger) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_button_mapping_profile() {
        let mut profile = ButtonMappingProfile::identity("southpaw");
        profile.remap(GamepadBtn::A, GamepadBtn::B);
        profile.remap(GamepadBtn::B, GamepadBtn::A);
        assert_eq!(profile.resolve(GamepadBtn::A), GamepadBtn::B);
        assert_eq!(profile.resolve(GamepadBtn::B), GamepadBtn::A);
        assert_eq!(profile.resolve(GamepadBtn::X), GamepadBtn::X);
        assert_eq!(profile.remap_count(), 2);
    }

    #[test]
    fn test_mapping_profile_on_gamepad() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        let gp = sys.gamepad_mut(0).unwrap();
        let mut profile = ButtonMappingProfile::identity("swap");
        profile.remap(GamepadBtn::A, GamepadBtn::B);
        gp.set_mapping_profile(profile);
        gp.begin_frame();
        gp.set_button(GamepadBtn::A, true);
        // Physical A is mapped to logical B
        assert!(gp.is_pressed(GamepadBtn::B));
        assert!(!gp.is_pressed(GamepadBtn::A));
    }

    #[test]
    fn test_rumble_command() {
        let cmd = RumbleCommand::new(0.8, 0.3, 200);
        assert!((cmd.strong_magnitude - 0.8).abs() < 1e-9);
        assert!((cmd.weak_magnitude - 0.3).abs() < 1e-9);
        assert_eq!(cmd.duration_ms, 200);
    }

    #[test]
    fn test_rumble_clamp() {
        let cmd = RumbleCommand::new(1.5, -0.2, 100);
        assert!((cmd.strong_magnitude - 1.0).abs() < 1e-9);
        assert!((cmd.weak_magnitude - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_rumble_presets() {
        let tap = RumbleCommand::light_tap();
        assert!((tap.strong_magnitude - 0.2).abs() < 1e-9);
        assert_eq!(tap.duration_ms, 50);
        let heavy = RumbleCommand::heavy_explosion();
        assert!((heavy.strong_magnitude - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_collect_rumbles() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad0");
        sys.connect(1, "pad1");
        sys.gamepad_mut(0).unwrap().rumble(RumbleCommand::light_tap());
        sys.gamepad_mut(1).unwrap().rumble(RumbleCommand::heavy_explosion());
        let rumbles = sys.collect_rumbles();
        assert_eq!(rumbles.len(), 2);
    }

    #[test]
    fn test_disconnected_gamepad_returns_none() {
        let sys = GamepadSystem::new();
        assert!(sys.gamepad(0).is_none());
    }

    #[test]
    fn test_connected_slots() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad0");
        sys.connect(2, "pad2");
        let slots = sys.connected_slots();
        assert_eq!(slots.len(), 2);
        assert!(slots.contains(&0));
        assert!(slots.contains(&2));
    }

    #[test]
    fn test_right_stick_axial() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        let gp = sys.gamepad_mut(0).unwrap();
        gp.set_right_dead_zone(DeadZoneMode::Axial { threshold: 0.1 });
        gp.set_axis(GamepadAxis::RightStickX, 0.05);
        gp.set_axis(GamepadAxis::RightStickY, 0.8);
        let (x, y) = gp.right_stick();
        assert!((x - 0.0).abs() < 1e-9);
        let expected_y = (0.8 - 0.1) / (1.0 - 0.1);
        assert!((y - expected_y).abs() < 1e-9);
    }

    #[test]
    fn test_clear_remap() {
        let mut profile = ButtonMappingProfile::identity("test");
        profile.remap(GamepadBtn::A, GamepadBtn::X);
        assert_eq!(profile.remap_count(), 1);
        profile.clear_remap(GamepadBtn::A);
        assert_eq!(profile.remap_count(), 0);
        assert_eq!(profile.resolve(GamepadBtn::A), GamepadBtn::A);
    }

    #[test]
    fn test_disconnect_clears_state() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        sys.gamepad_mut(0).unwrap().set_button(GamepadBtn::A, true);
        sys.disconnect(0);
        // After reconnecting, state should be clean
        sys.connect(0, "pad");
        assert!(!sys.gamepad(0).unwrap().is_pressed(GamepadBtn::A));
    }

    #[test]
    fn test_trigger_full_range() {
        let mut sys = GamepadSystem::new();
        sys.connect(0, "pad");
        let gp = sys.gamepad_mut(0).unwrap();
        gp.set_trigger_dead_zone(0.05);
        gp.set_axis(GamepadAxis::RightTrigger, 1.0);
        assert!((gp.trigger(GamepadAxis::RightTrigger) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_default_system() {
        let sys = GamepadSystem::default();
        assert_eq!(sys.connected_count(), 0);
    }
}
