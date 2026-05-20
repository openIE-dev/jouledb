//! Keyboard shortcuts: parsing, scope isolation, and chord sequences.
//!
//! Replaces hotkeys-js, Mousetrap, and cmdk with a pure-Rust manager
//! that handles modifier keys, scoped bindings, and multi-key chords.

use std::time::Instant;

// ── Modifiers ───────────────────────────────────────────────────

/// Modifier key state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Modifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub meta: bool,
}

impl Modifiers {
    pub fn none() -> Self { Self::default() }

    pub fn ctrl() -> Self { Self { ctrl: true, ..Self::default() } }

    pub fn meta() -> Self { Self { meta: true, ..Self::default() } }

    pub fn shift() -> Self { Self { shift: true, ..Self::default() } }

    pub fn alt() -> Self { Self { alt: true, ..Self::default() } }
}

// ── KeyCombo ────────────────────────────────────────────────────

/// A single key + modifier combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyCombo {
    pub key: String,
    pub modifiers: Modifiers,
}

impl KeyCombo {
    /// Parse a shortcut string like "Ctrl+S", "Meta+Shift+P", "Escape".
    /// Case-insensitive. Supports Ctrl/Control, Meta/Cmd/Command, Alt/Option.
    pub fn parse(shortcut: &str) -> Option<KeyCombo> {
        let parts: Vec<&str> = shortcut.split('+').collect();
        if parts.is_empty() {
            return None;
        }

        let mut modifiers = Modifiers::none();
        let mut key = None;

        for (i, part) in parts.iter().enumerate() {
            let lower = part.trim().to_lowercase();
            match lower.as_str() {
                "ctrl" | "control" => modifiers.ctrl = true,
                "shift" => modifiers.shift = true,
                "alt" | "option" => modifiers.alt = true,
                "meta" | "cmd" | "command" => modifiers.meta = true,
                _ => {
                    // Last unrecognized part is the key
                    if i == parts.len() - 1 || key.is_none() {
                        key = Some(lower);
                    }
                }
            }
        }

        Some(KeyCombo {
            key: key.unwrap_or_default(),
            modifiers,
        })
    }

    /// Check whether an incoming keystroke matches this combo.
    pub fn matches(&self, key: &str, modifiers: &Modifiers) -> bool {
        self.key.eq_ignore_ascii_case(key) && self.modifiers == *modifiers
    }

    /// Display string in platform-appropriate style.
    pub fn to_display_string(&self) -> String {
        let mut parts = Vec::new();
        if self.modifiers.ctrl { parts.push("Ctrl"); }
        if self.modifiers.alt { parts.push("Alt"); }
        if self.modifiers.shift { parts.push("Shift"); }
        if self.modifiers.meta { parts.push("Meta"); }

        // Capitalize the key for display
        let key_display = if self.key.len() == 1 {
            self.key.to_uppercase()
        } else {
            let mut chars = self.key.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            }
        };
        parts.push(&key_display);

        // We need to own the joined string
        let owned_parts: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
        owned_parts.join("+")
    }
}

// ── Bindings ────────────────────────────────────────────────────

/// A registered hotkey binding.
#[derive(Debug, Clone)]
pub struct HotkeyBinding {
    pub combo: KeyCombo,
    pub action_id: u64,
    pub scope: String,
    pub description: String,
    pub enabled: bool,
}

/// A registered chord (multi-key) sequence.
#[derive(Debug, Clone)]
struct ChordBinding {
    combos: Vec<KeyCombo>,
    action_id: u64,
    scope: String,
    #[allow(dead_code)]
    description: String,
}

// ── HotkeyManager ──────────────────────────────────────────────

/// Central registry for keyboard shortcuts.
pub struct HotkeyManager {
    bindings: Vec<HotkeyBinding>,
    chords: Vec<ChordBinding>,
    active_scopes: Vec<String>,
    sequence_buffer: Vec<(String, Modifiers, Instant)>,
    pub sequence_timeout_ms: u64,
}

impl HotkeyManager {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            chords: Vec::new(),
            active_scopes: vec!["global".to_string()],
            sequence_buffer: Vec::new(),
            sequence_timeout_ms: 1000,
        }
    }

    /// Register a single-key binding. Returns false if shortcut cannot be parsed.
    pub fn register(
        &mut self,
        shortcut: &str,
        action_id: u64,
        scope: &str,
        description: &str,
    ) -> bool {
        let Some(combo) = KeyCombo::parse(shortcut) else { return false };
        self.bindings.push(HotkeyBinding {
            combo,
            action_id,
            scope: scope.to_string(),
            description: description.to_string(),
            enabled: true,
        });
        true
    }

    /// Unregister a binding by shortcut and scope.
    pub fn unregister(&mut self, shortcut: &str, scope: &str) -> bool {
        let Some(combo) = KeyCombo::parse(shortcut) else { return false };
        let before = self.bindings.len();
        self.bindings.retain(|b| !(b.combo == combo && b.scope == scope));
        self.bindings.len() < before
    }

    /// Push a scope onto the active stack.
    pub fn push_scope(&mut self, scope: &str) {
        self.active_scopes.push(scope.to_string());
    }

    /// Pop the topmost scope.
    pub fn pop_scope(&mut self) -> Option<String> {
        if self.active_scopes.len() > 1 {
            self.active_scopes.pop()
        } else {
            None // never pop "global"
        }
    }

    /// The currently active (topmost) scope.
    pub fn active_scope(&self) -> &str {
        self.active_scopes.last().map(|s| s.as_str()).unwrap_or("global")
    }

    /// Handle a key press. Returns the `action_id` if a binding matches.
    pub fn handle_key(&mut self, key: &str, modifiers: &Modifiers) -> Option<u64> {
        let now = Instant::now();

        // Prune stale sequence buffer entries
        let timeout = std::time::Duration::from_millis(self.sequence_timeout_ms);
        self.sequence_buffer.retain(|(_, _, t)| now.duration_since(*t) < timeout);

        // Add current key to buffer
        self.sequence_buffer.push((key.to_lowercase(), modifiers.clone(), now));

        // Check chord sequences first (most specific)
        for chord in &self.chords {
            if !self.active_scopes.contains(&chord.scope) {
                continue;
            }
            let buf_len = self.sequence_buffer.len();
            let chord_len = chord.combos.len();
            if buf_len >= chord_len {
                let tail = &self.sequence_buffer[buf_len - chord_len..];
                let matched = tail.iter().zip(&chord.combos).all(|((k, m, _), combo)| {
                    combo.matches(k, m)
                });
                if matched {
                    self.sequence_buffer.clear();
                    return Some(chord.action_id);
                }
            }
        }

        // Check single-key bindings (search active scopes from most-recent first)
        let key_lower = key.to_lowercase();
        for scope in self.active_scopes.iter().rev() {
            for binding in self.bindings.iter().rev() {
                if binding.enabled && binding.scope == *scope && binding.combo.matches(&key_lower, modifiers) {
                    return Some(binding.action_id);
                }
            }
        }

        None
    }

    /// Enable or disable a binding.
    pub fn set_enabled(&mut self, shortcut: &str, scope: &str, enabled: bool) {
        if let Some(combo) = KeyCombo::parse(shortcut) {
            for b in &mut self.bindings {
                if b.combo == combo && b.scope == scope {
                    b.enabled = enabled;
                }
            }
        }
    }

    /// All bindings in a given scope.
    pub fn bindings_for_scope(&self, scope: &str) -> Vec<&HotkeyBinding> {
        self.bindings.iter().filter(|b| b.scope == scope).collect()
    }

    /// All registered bindings.
    pub fn all_bindings(&self) -> Vec<&HotkeyBinding> {
        self.bindings.iter().collect()
    }

    /// Register a chord sequence (e.g., "Ctrl+K" then "Ctrl+C").
    pub fn register_sequence(
        &mut self,
        shortcuts: &[&str],
        action_id: u64,
        scope: &str,
        description: &str,
    ) -> bool {
        let combos: Vec<KeyCombo> = shortcuts
            .iter()
            .filter_map(|s| KeyCombo::parse(s))
            .collect();
        if combos.len() != shortcuts.len() {
            return false;
        }
        self.chords.push(ChordBinding {
            combos,
            action_id,
            scope: scope.to_string(),
            description: description.to_string(),
        });
        true
    }
}

impl Default for HotkeyManager {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_handle_ctrl_s() {
        let mut m = HotkeyManager::new();
        assert!(m.register("Ctrl+S", 1, "global", "Save"));
        let result = m.handle_key("s", &Modifiers::ctrl());
        assert_eq!(result, Some(1));
    }

    #[test]
    fn parse_case_insensitive() {
        let a = KeyCombo::parse("ctrl+s").unwrap();
        let b = KeyCombo::parse("CTRL+S").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn scope_isolation() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        m.register("Ctrl+S", 2, "editor", "Editor Save");
        // Only global is active
        assert_eq!(m.handle_key("s", &Modifiers::ctrl()), Some(1));
        // Activate editor scope
        m.push_scope("editor");
        // Editor scope binding takes priority (searched last = most recent)
        assert_eq!(m.handle_key("s", &Modifiers::ctrl()), Some(2));
    }

    #[test]
    fn push_pop_scope() {
        let mut m = HotkeyManager::new();
        assert_eq!(m.active_scope(), "global");
        m.push_scope("modal");
        assert_eq!(m.active_scope(), "modal");
        assert_eq!(m.pop_scope(), Some("modal".to_string()));
        assert_eq!(m.active_scope(), "global");
        // Cannot pop global
        assert_eq!(m.pop_scope(), None);
    }

    #[test]
    fn unregister_removes() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        assert!(m.unregister("Ctrl+S", "global"));
        assert_eq!(m.handle_key("s", &Modifiers::ctrl()), None);
    }

    #[test]
    fn disabled_binding_skipped() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        m.set_enabled("Ctrl+S", "global", false);
        assert_eq!(m.handle_key("s", &Modifiers::ctrl()), None);
    }

    #[test]
    fn to_display_string_format() {
        let combo = KeyCombo::parse("Ctrl+Shift+P").unwrap();
        let display = combo.to_display_string();
        assert!(display.contains("Ctrl"));
        assert!(display.contains("Shift"));
        assert!(display.contains("P"));
    }

    #[test]
    fn parse_complex_meta_shift() {
        let combo = KeyCombo::parse("Meta+Shift+P").unwrap();
        assert!(combo.modifiers.meta);
        assert!(combo.modifiers.shift);
        assert_eq!(combo.key, "p");
    }

    #[test]
    fn chord_sequence() {
        let mut m = HotkeyManager::new();
        assert!(m.register_sequence(&["Ctrl+K", "Ctrl+C"], 99, "global", "Comment"));
        // First key — no match yet
        let r1 = m.handle_key("k", &Modifiers::ctrl());
        assert_eq!(r1, None);
        // Second key — chord completes
        let r2 = m.handle_key("c", &Modifiers::ctrl());
        assert_eq!(r2, Some(99));
    }

    #[test]
    fn no_match_returns_none() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        assert_eq!(m.handle_key("x", &Modifiers::ctrl()), None);
    }

    #[test]
    fn multiple_bindings_same_scope() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        m.register("Ctrl+Z", 2, "global", "Undo");
        assert_eq!(m.handle_key("s", &Modifiers::ctrl()), Some(1));
        assert_eq!(m.handle_key("z", &Modifiers::ctrl()), Some(2));
    }

    #[test]
    fn bindings_for_scope_filter() {
        let mut m = HotkeyManager::new();
        m.register("Ctrl+S", 1, "global", "Save");
        m.register("Ctrl+E", 2, "editor", "Edit");
        assert_eq!(m.bindings_for_scope("global").len(), 1);
        assert_eq!(m.bindings_for_scope("editor").len(), 1);
        assert_eq!(m.all_bindings().len(), 2);
    }

    #[test]
    fn parse_escape() {
        let combo = KeyCombo::parse("Escape").unwrap();
        assert_eq!(combo.key, "escape");
        assert_eq!(combo.modifiers, Modifiers::none());
    }
}
