//! High-level action/axis mapping layer for games.
//!
//! Actions are boolean (pressed/not), axes are f32 (-1.0 to 1.0). Multiple
//! inputs can feed one action (keyboard OR gamepad). Axis composition
//! (W/S -> vertical axis). Input contexts (gameplay vs menu vs vehicle) that
//! enable/disable action sets. Priority-based consumption.

use std::collections::HashMap;

// ── Abstract Input ──────────────────────────────────────────────

/// An abstract input identifier that can be a key, button, or axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputId {
    Key(String),
    MouseButton(u8),
    GamepadButton { pad: u8, button: String },
    GamepadAxis { pad: u8, axis: String },
}

impl InputId {
    pub fn key(k: &str) -> Self { Self::Key(k.to_lowercase()) }
    pub fn mouse(b: u8) -> Self { Self::MouseButton(b) }
    pub fn gamepad_btn(pad: u8, btn: &str) -> Self {
        Self::GamepadButton { pad, button: btn.to_string() }
    }
    pub fn gamepad_axis(pad: u8, axis: &str) -> Self {
        Self::GamepadAxis { pad, axis: axis.to_string() }
    }
}

// ── Action Binding ──────────────────────────────────────────────

/// Maps an input to contribute to an action (boolean).
#[derive(Debug, Clone, PartialEq)]
pub struct ActionBinding {
    pub input: InputId,
}

impl ActionBinding {
    pub fn new(input: InputId) -> Self { Self { input } }
}

// ── Axis Binding ────────────────────────────────────────────────

/// Maps an input to contribute to an axis (f32).
#[derive(Debug, Clone, PartialEq)]
pub struct AxisBinding {
    pub input: InputId,
    /// Scale factor applied to the input value.
    pub scale: f32,
}

impl AxisBinding {
    pub fn new(input: InputId, scale: f32) -> Self { Self { input, scale } }
}

// ── Action Definition ───────────────────────────────────────────

/// A named boolean action with one or more input bindings.
#[derive(Debug, Clone)]
pub struct ActionDef {
    pub name: String,
    pub bindings: Vec<ActionBinding>,
    pub consumed: bool,
}

impl ActionDef {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), bindings: Vec::new(), consumed: false }
    }

    pub fn add_binding(&mut self, input: InputId) {
        self.bindings.push(ActionBinding::new(input));
    }
}

// ── Axis Definition ─────────────────────────────────────────────

/// A named float axis with one or more input bindings.
#[derive(Debug, Clone)]
pub struct AxisDef {
    pub name: String,
    pub bindings: Vec<AxisBinding>,
}

impl AxisDef {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), bindings: Vec::new() }
    }

    pub fn add_binding(&mut self, input: InputId, scale: f32) {
        self.bindings.push(AxisBinding::new(input, scale));
    }
}

// ── Input Context ───────────────────────────────────────────────

/// A named set of actions and axes that can be enabled/disabled together.
#[derive(Debug, Clone)]
pub struct InputContext {
    pub name: String,
    pub priority: i32,
    pub enabled: bool,
    pub actions: HashMap<String, ActionDef>,
    pub axes: HashMap<String, AxisDef>,
    /// If true, matched inputs are consumed and won't propagate to lower-priority contexts.
    pub consumes_input: bool,
}

impl InputContext {
    pub fn new(name: &str, priority: i32) -> Self {
        Self {
            name: name.to_string(),
            priority,
            enabled: true,
            actions: HashMap::new(),
            axes: HashMap::new(),
            consumes_input: false,
        }
    }

    /// Add an action definition to this context.
    pub fn add_action(&mut self, action: ActionDef) {
        self.actions.insert(action.name.clone(), action);
    }

    /// Add an axis definition to this context.
    pub fn add_axis(&mut self, axis: AxisDef) {
        self.axes.insert(axis.name.clone(), axis);
    }

    /// Remove an action from this context.
    pub fn remove_action(&mut self, name: &str) -> bool {
        self.actions.remove(name).is_some()
    }

    /// Remove an axis from this context.
    pub fn remove_axis(&mut self, name: &str) -> bool {
        self.axes.remove(name).is_some()
    }
}

// ── Action State ────────────────────────────────────────────────

/// Resolved action state for a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionState {
    pub active: bool,
    pub just_activated: bool,
    pub just_deactivated: bool,
}

impl Default for ActionState {
    fn default() -> Self {
        Self { active: false, just_activated: false, just_deactivated: false }
    }
}

// ── Action Mapping System ───────────────────────────────────────

/// The main action mapping system.
pub struct ActionMappingSystem {
    contexts: Vec<InputContext>,
    /// Current raw input states: InputId -> value (bool as 0.0/1.0, axes as f32).
    input_states: HashMap<InputId, f32>,
    prev_input_states: HashMap<InputId, f32>,
    /// Cached resolved actions.
    action_cache: HashMap<String, ActionState>,
    /// Cached resolved axes.
    axis_cache: HashMap<String, f32>,
    /// Set of consumed input IDs this frame.
    consumed: HashMap<InputId, bool>,
}

impl ActionMappingSystem {
    pub fn new() -> Self {
        Self {
            contexts: Vec::new(),
            input_states: HashMap::new(),
            prev_input_states: HashMap::new(),
            action_cache: HashMap::new(),
            axis_cache: HashMap::new(),
            consumed: HashMap::new(),
        }
    }

    /// Add an input context. Contexts are auto-sorted by priority (higher first).
    pub fn add_context(&mut self, context: InputContext) {
        self.contexts.push(context);
        self.contexts.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get a context by name.
    pub fn context(&self, name: &str) -> Option<&InputContext> {
        self.contexts.iter().find(|c| c.name == name)
    }

    /// Get a mutable context by name.
    pub fn context_mut(&mut self, name: &str) -> Option<&mut InputContext> {
        self.contexts.iter_mut().find(|c| c.name == name)
    }

    /// Enable a context by name.
    pub fn enable_context(&mut self, name: &str) -> bool {
        if let Some(ctx) = self.context_mut(name) {
            ctx.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a context by name.
    pub fn disable_context(&mut self, name: &str) -> bool {
        if let Some(ctx) = self.context_mut(name) {
            ctx.enabled = false;
            true
        } else {
            false
        }
    }

    /// Set a raw input state (1.0 for pressed keys/buttons, axis values for axes).
    pub fn set_input(&mut self, input: InputId, value: f32) {
        self.input_states.insert(input, value);
    }

    /// Set a key as pressed (1.0) or released (0.0).
    pub fn set_key(&mut self, key: &str, pressed: bool) {
        self.set_input(InputId::key(key), if pressed { 1.0 } else { 0.0 });
    }

    /// Begin a new frame: resolve all actions/axes against previous state, then snapshot.
    pub fn update(&mut self) {
        self.action_cache.clear();
        self.axis_cache.clear();
        self.consumed.clear();
        self.resolve();
        // Snapshot current inputs as prev for next frame's comparison
        self.prev_input_states = self.input_states.clone();
    }

    fn resolve(&mut self) {
        // Process contexts in priority order (already sorted, highest first)
        let contexts = self.contexts.clone();
        for ctx in &contexts {
            if !ctx.enabled { continue; }

            // Resolve actions
            for (name, action_def) in &ctx.actions {
                if self.action_cache.contains_key(name) { continue; }

                let mut active = false;
                for binding in &action_def.bindings {
                    if self.consumed.get(&binding.input).copied().unwrap_or(false) {
                        continue;
                    }
                    let val = self.input_states.get(&binding.input).copied().unwrap_or(0.0);
                    if val > 0.5 {
                        active = true;
                        if ctx.consumes_input {
                            self.consumed.insert(binding.input.clone(), true);
                        }
                        break;
                    }
                }

                let prev_active = {
                    let mut prev = false;
                    for binding in &action_def.bindings {
                        let val = self.prev_input_states.get(&binding.input).copied().unwrap_or(0.0);
                        if val > 0.5 { prev = true; break; }
                    }
                    prev
                };

                self.action_cache.insert(name.clone(), ActionState {
                    active,
                    just_activated: active && !prev_active,
                    just_deactivated: !active && prev_active,
                });
            }

            // Resolve axes
            for (name, axis_def) in &ctx.axes {
                if self.axis_cache.contains_key(name) { continue; }

                let mut total: f32 = 0.0;
                for binding in &axis_def.bindings {
                    if self.consumed.get(&binding.input).copied().unwrap_or(false) {
                        continue;
                    }
                    let val = self.input_states.get(&binding.input).copied().unwrap_or(0.0);
                    total += val * binding.scale;
                    if ctx.consumes_input && val.abs() > 0.01 {
                        self.consumed.insert(binding.input.clone(), true);
                    }
                }

                self.axis_cache.insert(name.clone(), total.clamp(-1.0, 1.0));
            }
        }
    }

    /// Get the resolved state of an action.
    pub fn action(&self, name: &str) -> ActionState {
        self.action_cache.get(name).cloned().unwrap_or_default()
    }

    /// Get the resolved value of an axis.
    pub fn axis(&self, name: &str) -> f32 {
        self.axis_cache.get(name).copied().unwrap_or(0.0)
    }

    /// Is an action currently active?
    pub fn is_action_active(&self, name: &str) -> bool {
        self.action(name).active
    }

    /// Was an action just activated this frame?
    pub fn is_action_just_activated(&self, name: &str) -> bool {
        self.action(name).just_activated
    }

    /// Number of registered contexts.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// Names of enabled contexts, sorted by priority (highest first).
    pub fn enabled_context_names(&self) -> Vec<String> {
        self.contexts.iter()
            .filter(|c| c.enabled)
            .map(|c| c.name.clone())
            .collect()
    }
}

impl Default for ActionMappingSystem {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gameplay_context() -> InputContext {
        let mut ctx = InputContext::new("gameplay", 10);
        let mut jump = ActionDef::new("jump");
        jump.add_binding(InputId::key("space"));
        jump.add_binding(InputId::gamepad_btn(0, "A"));
        ctx.add_action(jump);

        let mut fire = ActionDef::new("fire");
        fire.add_binding(InputId::mouse(0));
        ctx.add_action(fire);

        let mut vertical = AxisDef::new("vertical");
        vertical.add_binding(InputId::key("w"), 1.0);
        vertical.add_binding(InputId::key("s"), -1.0);
        ctx.add_axis(vertical);

        let mut horizontal = AxisDef::new("horizontal");
        horizontal.add_binding(InputId::key("d"), 1.0);
        horizontal.add_binding(InputId::key("a"), -1.0);
        ctx.add_axis(horizontal);

        ctx
    }

    #[test]
    fn test_action_press() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_key("space", true);
        sys.update();
        assert!(sys.is_action_active("jump"));
    }

    #[test]
    fn test_action_not_pressed() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.update();
        assert!(!sys.is_action_active("jump"));
    }

    #[test]
    fn test_action_just_activated() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.update();
        sys.set_key("space", true);
        sys.update();
        assert!(sys.is_action_just_activated("jump"));
        sys.update();
        assert!(!sys.is_action_just_activated("jump"));
        assert!(sys.is_action_active("jump"));
    }

    #[test]
    fn test_action_just_deactivated() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_key("space", true);
        sys.update();
        sys.set_key("space", false);
        sys.update();
        assert!(sys.action("jump").just_deactivated);
    }

    #[test]
    fn test_axis_composition() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_key("w", true);
        sys.update();
        assert!((sys.axis("vertical") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_axis_negative() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_key("s", true);
        sys.update();
        assert!((sys.axis("vertical") - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_axis_cancel_out() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_key("w", true);
        sys.set_key("s", true);
        sys.update();
        assert!((sys.axis("vertical") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_axis_clamp() {
        let mut sys = ActionMappingSystem::new();
        let mut ctx = InputContext::new("test", 10);
        let mut axis = AxisDef::new("overloaded");
        axis.add_binding(InputId::key("a"), 1.0);
        axis.add_binding(InputId::key("b"), 1.0);
        ctx.add_axis(axis);
        sys.add_context(ctx);
        sys.set_key("a", true);
        sys.set_key("b", true);
        sys.update();
        // 1.0 + 1.0 = 2.0, clamped to 1.0
        assert!((sys.axis("overloaded") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_inputs_for_action() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_input(InputId::gamepad_btn(0, "A"), 1.0);
        sys.update();
        assert!(sys.is_action_active("jump"));
    }

    #[test]
    fn test_context_disable() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.disable_context("gameplay");
        sys.set_key("space", true);
        sys.update();
        assert!(!sys.is_action_active("jump"));
    }

    #[test]
    fn test_context_re_enable() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.disable_context("gameplay");
        sys.enable_context("gameplay");
        sys.set_key("space", true);
        sys.update();
        assert!(sys.is_action_active("jump"));
    }

    #[test]
    fn test_context_priority_order() {
        let mut sys = ActionMappingSystem::new();
        let mut menu = InputContext::new("menu", 20);
        let mut confirm = ActionDef::new("confirm");
        confirm.add_binding(InputId::key("space"));
        menu.add_action(confirm);
        menu.consumes_input = true;
        sys.add_context(menu);
        sys.add_context(make_gameplay_context());
        sys.set_key("space", true);
        sys.update();
        // Menu (priority 20) consumes space before gameplay (priority 10)
        assert!(sys.is_action_active("confirm"));
        assert!(!sys.is_action_active("jump"));
    }

    #[test]
    fn test_no_consumption_by_default() {
        let mut sys = ActionMappingSystem::new();
        let mut ctx1 = InputContext::new("ctx1", 20);
        let mut a1 = ActionDef::new("a1");
        a1.add_binding(InputId::key("x"));
        ctx1.add_action(a1);
        // consumes_input is false by default
        sys.add_context(ctx1);

        let mut ctx2 = InputContext::new("ctx2", 10);
        let mut a2 = ActionDef::new("a2");
        a2.add_binding(InputId::key("x"));
        ctx2.add_action(a2);
        sys.add_context(ctx2);

        sys.set_key("x", true);
        sys.update();
        assert!(sys.is_action_active("a1"));
        assert!(sys.is_action_active("a2"));
    }

    #[test]
    fn test_gamepad_axis() {
        let mut sys = ActionMappingSystem::new();
        let mut ctx = InputContext::new("game", 10);
        let mut look = AxisDef::new("look_x");
        look.add_binding(InputId::gamepad_axis(0, "RightStickX"), 1.0);
        ctx.add_axis(look);
        sys.add_context(ctx);
        sys.set_input(InputId::gamepad_axis(0, "RightStickX"), 0.75);
        sys.update();
        assert!((sys.axis("look_x") - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_enabled_context_names() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(InputContext::new("game", 10));
        sys.add_context(InputContext::new("menu", 20));
        sys.disable_context("game");
        let names = sys.enabled_context_names();
        assert_eq!(names, vec!["menu"]);
    }

    #[test]
    fn test_context_count() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(InputContext::new("a", 10));
        sys.add_context(InputContext::new("b", 20));
        assert_eq!(sys.context_count(), 2);
    }

    #[test]
    fn test_remove_action_from_context() {
        let mut ctx = make_gameplay_context();
        assert!(ctx.remove_action("jump"));
        assert!(!ctx.remove_action("nonexistent"));
        assert!(ctx.actions.get("jump").is_none());
    }

    #[test]
    fn test_remove_axis_from_context() {
        let mut ctx = make_gameplay_context();
        assert!(ctx.remove_axis("vertical"));
        assert!(!ctx.remove_axis("nonexistent"));
    }

    #[test]
    fn test_unknown_action_defaults() {
        let sys = ActionMappingSystem::new();
        let state = sys.action("nonexistent");
        assert!(!state.active);
        assert!(!state.just_activated);
        assert!(!state.just_deactivated);
    }

    #[test]
    fn test_unknown_axis_defaults() {
        let sys = ActionMappingSystem::new();
        assert!((sys.axis("nonexistent") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mouse_action() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(make_gameplay_context());
        sys.set_input(InputId::mouse(0), 1.0);
        sys.update();
        assert!(sys.is_action_active("fire"));
    }

    #[test]
    fn test_context_get() {
        let mut sys = ActionMappingSystem::new();
        sys.add_context(InputContext::new("test", 5));
        assert!(sys.context("test").is_some());
        assert!(sys.context("missing").is_none());
    }
}
