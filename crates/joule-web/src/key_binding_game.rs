//! Rebindable key bindings for games.
//!
//! Map action names ("jump", "fire") to one or more input sources (keyboard key,
//! mouse button, gamepad button). Conflict detection when the same key is bound
//! to multiple actions. Save/load binding configs as serializable data.
//! Default + custom profiles. Modifier key support (Shift+X).

use std::collections::HashMap;

// ── Input Source ────────────────────────────────────────────────

/// Modifier keys that can combine with a key or button.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl Default for Modifiers {
    fn default() -> Self {
        Self { shift: false, ctrl: false, alt: false }
    }
}

impl Modifiers {
    pub fn none() -> Self { Self::default() }
    pub fn shift() -> Self { Self { shift: true, ..Self::default() } }
    pub fn ctrl() -> Self { Self { ctrl: true, ..Self::default() } }
    pub fn alt() -> Self { Self { alt: true, ..Self::default() } }

    pub fn is_empty(&self) -> bool {
        !self.shift && !self.ctrl && !self.alt
    }

    /// Serialize to a compact string representation.
    pub fn to_string_repr(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl { parts.push("Ctrl"); }
        if self.shift { parts.push("Shift"); }
        if self.alt { parts.push("Alt"); }
        parts.join("+")
    }
}

/// An input source that can be bound to an action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputSource {
    /// Keyboard key with optional modifiers.
    Key { key: String, modifiers: Modifiers },
    /// Mouse button (0=left, 1=right, 2=middle, etc.).
    MouseButton { button: u8, modifiers: Modifiers },
    /// Gamepad button on a specific pad.
    GamepadButton { pad: u8, button: String },
}

impl InputSource {
    /// Create a simple key binding with no modifiers.
    pub fn key(key: &str) -> Self {
        Self::Key { key: key.to_lowercase(), modifiers: Modifiers::none() }
    }

    /// Create a key binding with modifiers.
    pub fn key_with_mods(key: &str, modifiers: Modifiers) -> Self {
        Self::Key { key: key.to_lowercase(), modifiers }
    }

    /// Create a mouse button binding.
    pub fn mouse(button: u8) -> Self {
        Self::MouseButton { button, modifiers: Modifiers::none() }
    }

    /// Create a mouse button binding with modifiers.
    pub fn mouse_with_mods(button: u8, modifiers: Modifiers) -> Self {
        Self::MouseButton { button, modifiers }
    }

    /// Create a gamepad button binding.
    pub fn gamepad(pad: u8, button: &str) -> Self {
        Self::GamepadButton { pad, button: button.to_string() }
    }

    /// Display string for UI presentation.
    pub fn display_string(&self) -> String {
        match self {
            Self::Key { key, modifiers } => {
                let mods = modifiers.to_string_repr();
                if mods.is_empty() {
                    key.to_uppercase()
                } else {
                    format!("{}+{}", mods, key.to_uppercase())
                }
            }
            Self::MouseButton { button, modifiers } => {
                let name = match button {
                    0 => "Left Click",
                    1 => "Right Click",
                    2 => "Middle Click",
                    n => return format!("Mouse{}", n),
                };
                let mods = modifiers.to_string_repr();
                if mods.is_empty() { name.to_string() }
                else { format!("{}+{}", mods, name) }
            }
            Self::GamepadButton { pad, button } => {
                format!("Pad{}:{}", pad, button)
            }
        }
    }
}

// ── Binding ─────────────────────────────────────────────────────

/// A single action-to-input binding.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub action: String,
    pub sources: Vec<InputSource>,
}

impl Binding {
    pub fn new(action: &str) -> Self {
        Self { action: action.to_string(), sources: Vec::new() }
    }

    pub fn add_source(&mut self, source: InputSource) {
        if !self.sources.contains(&source) {
            self.sources.push(source);
        }
    }

    pub fn remove_source(&mut self, source: &InputSource) {
        self.sources.retain(|s| s != source);
    }

    pub fn has_source(&self, source: &InputSource) -> bool {
        self.sources.contains(source)
    }
}

// ── Conflict ────────────────────────────────────────────────────

/// A detected binding conflict.
#[derive(Debug, Clone, PartialEq)]
pub struct BindingConflict {
    pub source: InputSource,
    pub action_a: String,
    pub action_b: String,
}

// ── Serializable Config ─────────────────────────────────────────

/// A serializable binding entry for save/load.
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedBinding {
    pub action: String,
    pub source_type: String,
    pub source_key: String,
    pub modifiers: Option<String>,
    pub pad: Option<u8>,
}

// ── Binding Profile ─────────────────────────────────────────────

/// A named set of key bindings.
#[derive(Debug, Clone)]
pub struct BindingProfile {
    pub name: String,
    bindings: HashMap<String, Binding>,
}

impl BindingProfile {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), bindings: HashMap::new() }
    }

    /// Bind an input source to an action. Creates the action if it doesn't exist.
    pub fn bind(&mut self, action: &str, source: InputSource) {
        let entry = self.bindings.entry(action.to_string())
            .or_insert_with(|| Binding::new(action));
        entry.add_source(source);
    }

    /// Unbind a specific source from an action.
    pub fn unbind(&mut self, action: &str, source: &InputSource) {
        if let Some(binding) = self.bindings.get_mut(action) {
            binding.remove_source(source);
        }
    }

    /// Unbind all sources from an action.
    pub fn unbind_all(&mut self, action: &str) {
        if let Some(binding) = self.bindings.get_mut(action) {
            binding.sources.clear();
        }
    }

    /// Get bindings for an action.
    pub fn get_binding(&self, action: &str) -> Option<&Binding> {
        self.bindings.get(action)
    }

    /// Check if an input source triggers an action.
    pub fn is_source_active(&self, action: &str, source: &InputSource) -> bool {
        self.bindings.get(action)
            .map(|b| b.has_source(source))
            .unwrap_or(false)
    }

    /// Find which action(s) an input source is bound to.
    pub fn actions_for_source(&self, source: &InputSource) -> Vec<String> {
        let mut result: Vec<String> = self.bindings.iter()
            .filter(|(_, b)| b.has_source(source))
            .map(|(name, _)| name.clone())
            .collect();
        result.sort();
        result
    }

    /// Detect all conflicts (same source bound to multiple actions).
    pub fn detect_conflicts(&self) -> Vec<BindingConflict> {
        let mut source_map: HashMap<&InputSource, Vec<&str>> = HashMap::new();
        for (action, binding) in &self.bindings {
            for source in &binding.sources {
                source_map.entry(source).or_default().push(action.as_str());
            }
        }

        let mut conflicts = Vec::new();
        for (source, actions) in &source_map {
            if actions.len() >= 2 {
                let mut sorted = actions.clone();
                sorted.sort();
                for i in 0..sorted.len() {
                    for j in (i + 1)..sorted.len() {
                        conflicts.push(BindingConflict {
                            source: (*source).clone(),
                            action_a: sorted[i].to_string(),
                            action_b: sorted[j].to_string(),
                        });
                    }
                }
            }
        }
        conflicts.sort_by(|a, b| a.action_a.cmp(&b.action_a).then(a.action_b.cmp(&b.action_b)));
        conflicts
    }

    /// Number of actions in the profile.
    pub fn action_count(&self) -> usize {
        self.bindings.len()
    }

    /// All action names, sorted.
    pub fn action_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.bindings.keys().cloned().collect();
        names.sort();
        names
    }

    /// Serialize all bindings for save.
    pub fn serialize(&self) -> Vec<SerializedBinding> {
        let mut result = Vec::new();
        let mut actions: Vec<_> = self.bindings.keys().cloned().collect();
        actions.sort();
        for action in actions {
            if let Some(binding) = self.bindings.get(&action) {
                for source in &binding.sources {
                    let entry = match source {
                        InputSource::Key { key, modifiers } => SerializedBinding {
                            action: action.clone(),
                            source_type: "key".to_string(),
                            source_key: key.clone(),
                            modifiers: if modifiers.is_empty() { None } else { Some(modifiers.to_string_repr()) },
                            pad: None,
                        },
                        InputSource::MouseButton { button, modifiers } => SerializedBinding {
                            action: action.clone(),
                            source_type: "mouse".to_string(),
                            source_key: button.to_string(),
                            modifiers: if modifiers.is_empty() { None } else { Some(modifiers.to_string_repr()) },
                            pad: None,
                        },
                        InputSource::GamepadButton { pad, button } => SerializedBinding {
                            action: action.clone(),
                            source_type: "gamepad".to_string(),
                            source_key: button.clone(),
                            modifiers: None,
                            pad: Some(*pad),
                        },
                    };
                    result.push(entry);
                }
            }
        }
        result
    }

    /// Deserialize and load bindings (replaces current).
    pub fn deserialize(name: &str, entries: &[SerializedBinding]) -> Self {
        let mut profile = Self::new(name);
        for entry in entries {
            let source = match entry.source_type.as_str() {
                "key" => {
                    let mods = parse_modifiers(entry.modifiers.as_deref());
                    InputSource::Key { key: entry.source_key.clone(), modifiers: mods }
                }
                "mouse" => {
                    let button: u8 = entry.source_key.parse().unwrap_or(0);
                    let mods = parse_modifiers(entry.modifiers.as_deref());
                    InputSource::MouseButton { button, modifiers: mods }
                }
                "gamepad" => {
                    let pad = entry.pad.unwrap_or(0);
                    InputSource::GamepadButton { pad, button: entry.source_key.clone() }
                }
                _ => continue,
            };
            profile.bind(&entry.action, source);
        }
        profile
    }
}

fn parse_modifiers(s: Option<&str>) -> Modifiers {
    let Some(s) = s else { return Modifiers::none(); };
    let mut m = Modifiers::none();
    for part in s.split('+') {
        match part.trim().to_lowercase().as_str() {
            "ctrl" => m.ctrl = true,
            "shift" => m.shift = true,
            "alt" => m.alt = true,
            _ => {}
        }
    }
    m
}

// ── Binding Manager ─────────────────────────────────────────────

/// Manages multiple binding profiles with a default + custom override.
pub struct BindingManager {
    default_profile: BindingProfile,
    custom_profile: Option<BindingProfile>,
    active_profile_name: String,
}

impl BindingManager {
    pub fn new(default_profile: BindingProfile) -> Self {
        let name = default_profile.name.clone();
        Self {
            default_profile,
            custom_profile: None,
            active_profile_name: name,
        }
    }

    /// Set a custom override profile.
    pub fn set_custom(&mut self, profile: BindingProfile) {
        self.active_profile_name = profile.name.clone();
        self.custom_profile = Some(profile);
    }

    /// Clear custom profile, reverting to default.
    pub fn clear_custom(&mut self) {
        self.custom_profile = None;
        self.active_profile_name = self.default_profile.name.clone();
    }

    /// Get the active profile (custom if set, else default).
    pub fn active(&self) -> &BindingProfile {
        self.custom_profile.as_ref().unwrap_or(&self.default_profile)
    }

    /// Get the default profile.
    pub fn default_profile(&self) -> &BindingProfile {
        &self.default_profile
    }

    /// Active profile name.
    pub fn active_profile_name(&self) -> &str {
        &self.active_profile_name
    }

    /// Check if a source triggers an action in the active profile.
    pub fn is_active(&self, action: &str, source: &InputSource) -> bool {
        self.active().is_source_active(action, source)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_key_binding() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        assert!(p.is_source_active("jump", &InputSource::key("space")));
    }

    #[test]
    fn test_key_with_modifier() {
        let mut p = BindingProfile::new("default");
        p.bind("sprint", InputSource::key_with_mods("w", Modifiers::shift()));
        let src = InputSource::key_with_mods("w", Modifiers::shift());
        assert!(p.is_source_active("sprint", &src));
        // Without modifier should NOT match
        assert!(!p.is_source_active("sprint", &InputSource::key("w")));
    }

    #[test]
    fn test_mouse_binding() {
        let mut p = BindingProfile::new("default");
        p.bind("fire", InputSource::mouse(0));
        assert!(p.is_source_active("fire", &InputSource::mouse(0)));
    }

    #[test]
    fn test_gamepad_binding() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::gamepad(0, "A"));
        assert!(p.is_source_active("jump", &InputSource::gamepad(0, "A")));
    }

    #[test]
    fn test_multiple_sources() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("jump", InputSource::gamepad(0, "A"));
        let binding = p.get_binding("jump").unwrap();
        assert_eq!(binding.sources.len(), 2);
    }

    #[test]
    fn test_unbind() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("jump", InputSource::key("w"));
        p.unbind("jump", &InputSource::key("space"));
        let binding = p.get_binding("jump").unwrap();
        assert_eq!(binding.sources.len(), 1);
        assert!(!p.is_source_active("jump", &InputSource::key("space")));
    }

    #[test]
    fn test_unbind_all() {
        let mut p = BindingProfile::new("default");
        p.bind("fire", InputSource::key("f"));
        p.bind("fire", InputSource::mouse(0));
        p.unbind_all("fire");
        let binding = p.get_binding("fire").unwrap();
        assert_eq!(binding.sources.len(), 0);
    }

    #[test]
    fn test_conflict_detection() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("interact", InputSource::key("space"));
        let conflicts = p.detect_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].action_a, "interact");
        assert_eq!(conflicts[0].action_b, "jump");
    }

    #[test]
    fn test_no_conflict() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("fire", InputSource::mouse(0));
        let conflicts = p.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_actions_for_source() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("confirm", InputSource::key("space"));
        let actions = p.actions_for_source(&InputSource::key("space"));
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&"jump".to_string()));
        assert!(actions.contains(&"confirm".to_string()));
    }

    #[test]
    fn test_serialize_deserialize() {
        let mut p = BindingProfile::new("original");
        p.bind("jump", InputSource::key("space"));
        p.bind("fire", InputSource::mouse(0));
        p.bind("reload", InputSource::key_with_mods("r", Modifiers::ctrl()));
        p.bind("move", InputSource::gamepad(0, "LeftStick"));

        let data = p.serialize();
        assert_eq!(data.len(), 4);

        let restored = BindingProfile::deserialize("restored", &data);
        assert!(restored.is_source_active("jump", &InputSource::key("space")));
        assert!(restored.is_source_active("fire", &InputSource::mouse(0)));
        assert!(restored.is_source_active("reload", &InputSource::key_with_mods("r", Modifiers::ctrl())));
        assert!(restored.is_source_active("move", &InputSource::gamepad(0, "LeftStick")));
    }

    #[test]
    fn test_display_string_key() {
        let src = InputSource::key("a");
        assert_eq!(src.display_string(), "A");
    }

    #[test]
    fn test_display_string_key_with_mods() {
        let src = InputSource::key_with_mods("s", Modifiers::ctrl());
        assert_eq!(src.display_string(), "Ctrl+S");
    }

    #[test]
    fn test_display_string_mouse() {
        let src = InputSource::mouse(0);
        assert_eq!(src.display_string(), "Left Click");
    }

    #[test]
    fn test_display_string_gamepad() {
        let src = InputSource::gamepad(0, "A");
        assert_eq!(src.display_string(), "Pad0:A");
    }

    #[test]
    fn test_binding_manager_default() {
        let mut default_p = BindingProfile::new("default");
        default_p.bind("jump", InputSource::key("space"));
        let mgr = BindingManager::new(default_p);
        assert!(mgr.is_active("jump", &InputSource::key("space")));
        assert_eq!(mgr.active_profile_name(), "default");
    }

    #[test]
    fn test_binding_manager_custom_override() {
        let mut default_p = BindingProfile::new("default");
        default_p.bind("jump", InputSource::key("space"));
        let mut mgr = BindingManager::new(default_p);
        let mut custom = BindingProfile::new("custom");
        custom.bind("jump", InputSource::key("w"));
        mgr.set_custom(custom);
        assert!(mgr.is_active("jump", &InputSource::key("w")));
        assert!(!mgr.is_active("jump", &InputSource::key("space")));
        assert_eq!(mgr.active_profile_name(), "custom");
    }

    #[test]
    fn test_binding_manager_revert() {
        let mut default_p = BindingProfile::new("default");
        default_p.bind("jump", InputSource::key("space"));
        let mut mgr = BindingManager::new(default_p);
        let custom = BindingProfile::new("custom");
        mgr.set_custom(custom);
        mgr.clear_custom();
        assert!(mgr.is_active("jump", &InputSource::key("space")));
        assert_eq!(mgr.active_profile_name(), "default");
    }

    #[test]
    fn test_action_names_sorted() {
        let mut p = BindingProfile::new("default");
        p.bind("zoom", InputSource::key("z"));
        p.bind("attack", InputSource::key("a"));
        p.bind("move", InputSource::key("m"));
        let names = p.action_names();
        assert_eq!(names, vec!["attack", "move", "zoom"]);
    }

    #[test]
    fn test_no_duplicate_sources() {
        let mut p = BindingProfile::new("default");
        p.bind("jump", InputSource::key("space"));
        p.bind("jump", InputSource::key("space"));
        let binding = p.get_binding("jump").unwrap();
        assert_eq!(binding.sources.len(), 1);
    }

    #[test]
    fn test_modifiers_empty() {
        let m = Modifiers::none();
        assert!(m.is_empty());
        let m2 = Modifiers::shift();
        assert!(!m2.is_empty());
    }

    #[test]
    fn test_action_count() {
        let mut p = BindingProfile::new("test");
        p.bind("a", InputSource::key("a"));
        p.bind("b", InputSource::key("b"));
        assert_eq!(p.action_count(), 2);
    }
}
