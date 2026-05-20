//! Keyboard navigation manager: arrow key navigation (linear, grid, tree),
//! roving tabindex, typeahead search, shortcut registry, focus-visible
//! tracking, and composite widget patterns.
//!
//! Pure data — no browser dependency. Computes focus targets so renderers
//! can apply `tabindex` and focus management.

use std::collections::HashMap;

// ── Key Events ────────────────────────────────────────────────

/// A keyboard key relevant to navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavKey {
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Home,
    End,
    Enter,
    Space,
    Escape,
    Tab,
}

/// Modifier keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

impl Modifiers {
    pub const NONE: Modifiers = Modifiers { shift: false, ctrl: false, alt: false, meta: false };
    pub const SHIFT: Modifiers = Modifiers { shift: true, ctrl: false, alt: false, meta: false };
}

/// A keyboard event for navigation.
#[derive(Debug, Clone)]
pub struct NavKeyEvent {
    pub key: NavKey,
    pub modifiers: Modifiers,
}

impl NavKeyEvent {
    pub fn new(key: NavKey) -> Self {
        Self { key, modifiers: Modifiers::NONE }
    }

    pub fn with_shift(mut self) -> Self {
        self.modifiers.shift = true;
        self
    }
}

// ── Focus Visible ─────────────────────────────────────────────

/// Tracks whether focus should be visually indicated.
#[derive(Debug, Clone)]
pub struct FocusVisibleTracker {
    /// True when the last interaction was keyboard-based.
    pub keyboard_active: bool,
}

impl FocusVisibleTracker {
    pub fn new() -> Self {
        Self { keyboard_active: false }
    }

    /// Record a keyboard interaction.
    pub fn on_keyboard(&mut self) {
        self.keyboard_active = true;
    }

    /// Record a mouse/pointer interaction.
    pub fn on_pointer(&mut self) {
        self.keyboard_active = false;
    }

    /// Whether focus indicators should be visible.
    pub fn should_show_focus(&self) -> bool {
        self.keyboard_active
    }
}

impl Default for FocusVisibleTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Navigation Pattern ────────────────────────────────────────

/// The navigation pattern for a composite widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavPattern {
    /// Linear list (up/down or left/right).
    Linear,
    /// 2D grid.
    Grid { columns: usize },
    /// Tree (expand/collapse with left/right).
    Tree,
}

// ── Nav Item ──────────────────────────────────────────────────

/// An item in a navigable widget.
#[derive(Debug, Clone)]
pub struct NavItem {
    pub id: String,
    pub label: String,
    pub disabled: bool,
    /// For tree pattern: nesting depth (0 = root).
    pub depth: usize,
    /// For tree pattern: whether this node is expanded.
    pub expanded: bool,
    /// For tree pattern: whether this node has children.
    pub has_children: bool,
}

impl NavItem {
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            disabled: false,
            depth: 0,
            expanded: false,
            has_children: false,
        }
    }

    pub fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn with_tree(mut self, depth: usize, has_children: bool) -> Self {
        self.depth = depth;
        self.has_children = has_children;
        self
    }
}

// ── Navigation Action ─────────────────────────────────────────

/// Action to perform after processing a key event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavAction {
    /// Focus moved to element with this ID.
    Focus(String),
    /// Activate (select/click) the current item.
    Activate(String),
    /// Toggle expanded state on a tree node.
    ToggleExpand(String),
    /// No action (key not handled).
    None,
}

// ── Keyboard Navigator ────────────────────────────────────────

/// Manages keyboard navigation for a composite widget.
#[derive(Debug)]
pub struct KeyboardNavigator {
    pub pattern: NavPattern,
    items: Vec<NavItem>,
    current_index: Option<usize>,
    /// Wrap around at boundaries.
    pub wrap: bool,
    /// Typeahead buffer.
    typeahead_buffer: String,
    typeahead_timeout_ms: u64,
    typeahead_last_ms: u64,
}

impl KeyboardNavigator {
    pub fn new(pattern: NavPattern) -> Self {
        Self {
            pattern,
            items: Vec::new(),
            current_index: None,
            wrap: true,
            typeahead_buffer: String::new(),
            typeahead_timeout_ms: 500,
            typeahead_last_ms: 0,
        }
    }

    /// Set typeahead timeout in milliseconds.
    pub fn set_typeahead_timeout(&mut self, ms: u64) {
        self.typeahead_timeout_ms = ms;
    }

    /// Add an item.
    pub fn add_item(&mut self, item: NavItem) {
        self.items.push(item);
    }

    /// Set all items at once.
    pub fn set_items(&mut self, items: Vec<NavItem>) {
        self.items = items;
        self.current_index = None;
    }

    /// Set the current focused index.
    pub fn set_current(&mut self, index: usize) {
        if index < self.items.len() {
            self.current_index = Some(index);
        }
    }

    /// Set current by ID.
    pub fn set_current_by_id(&mut self, id: &str) {
        if let Some(pos) = self.items.iter().position(|i| i.id == id) {
            self.current_index = Some(pos);
        }
    }

    /// Get the currently focused item.
    pub fn current_item(&self) -> Option<&NavItem> {
        self.current_index.and_then(|i| self.items.get(i))
    }

    /// Get the current index.
    pub fn current_index(&self) -> Option<usize> {
        self.current_index
    }

    /// Get enabled (non-disabled) item indices.
    fn enabled_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| !item.disabled)
            .map(|(i, _)| i)
            .collect()
    }

    /// Handle a navigation key event. Returns the action to perform.
    pub fn handle_key(&mut self, event: &NavKeyEvent) -> NavAction {
        match self.pattern {
            NavPattern::Linear => self.handle_linear(event),
            NavPattern::Grid { columns } => self.handle_grid(event, columns),
            NavPattern::Tree => self.handle_tree(event),
        }
    }

    fn handle_linear(&mut self, event: &NavKeyEvent) -> NavAction {
        let enabled = self.enabled_indices();
        if enabled.is_empty() {
            return NavAction::None;
        }
        match event.key {
            NavKey::ArrowDown | NavKey::ArrowRight => self.move_next(&enabled),
            NavKey::ArrowUp | NavKey::ArrowLeft => self.move_prev(&enabled),
            NavKey::Home => self.move_to_first(&enabled),
            NavKey::End => self.move_to_last(&enabled),
            NavKey::Enter | NavKey::Space => self.activate_current(),
            _ => NavAction::None,
        }
    }

    fn handle_grid(&mut self, event: &NavKeyEvent, columns: usize) -> NavAction {
        let enabled = self.enabled_indices();
        if enabled.is_empty() {
            return NavAction::None;
        }
        match event.key {
            NavKey::ArrowRight => self.move_next(&enabled),
            NavKey::ArrowLeft => self.move_prev(&enabled),
            NavKey::ArrowDown => self.move_by_offset(&enabled, columns as isize),
            NavKey::ArrowUp => self.move_by_offset(&enabled, -(columns as isize)),
            NavKey::Home => {
                if event.modifiers.ctrl {
                    self.move_to_first(&enabled)
                } else {
                    // Move to start of row.
                    let cur = self.current_index.unwrap_or(0);
                    let row_start = (cur / columns) * columns;
                    self.move_to_index(&enabled, row_start)
                }
            }
            NavKey::End => {
                if event.modifiers.ctrl {
                    self.move_to_last(&enabled)
                } else {
                    let cur = self.current_index.unwrap_or(0);
                    let row_end = ((cur / columns) + 1) * columns - 1;
                    let target = row_end.min(self.items.len().saturating_sub(1));
                    self.move_to_index(&enabled, target)
                }
            }
            NavKey::Enter | NavKey::Space => self.activate_current(),
            _ => NavAction::None,
        }
    }

    fn handle_tree(&mut self, event: &NavKeyEvent) -> NavAction {
        let enabled = self.enabled_indices();
        if enabled.is_empty() {
            return NavAction::None;
        }
        match event.key {
            NavKey::ArrowDown => self.move_next(&enabled),
            NavKey::ArrowUp => self.move_prev(&enabled),
            NavKey::ArrowRight => {
                if let Some(item) = self.current_item() {
                    if item.has_children && !item.expanded {
                        let id = item.id.clone();
                        return NavAction::ToggleExpand(id);
                    }
                }
                // Move to first child (next item).
                self.move_next(&enabled)
            }
            NavKey::ArrowLeft => {
                if let Some(item) = self.current_item() {
                    if item.has_children && item.expanded {
                        let id = item.id.clone();
                        return NavAction::ToggleExpand(id);
                    }
                }
                // Move to parent — find previous item with lower depth.
                if let Some(cur) = self.current_index {
                    let cur_depth = self.items[cur].depth;
                    if cur_depth > 0 {
                        for i in (0..cur).rev() {
                            if self.items[i].depth < cur_depth && !self.items[i].disabled {
                                self.current_index = Some(i);
                                return NavAction::Focus(self.items[i].id.clone());
                            }
                        }
                    }
                }
                NavAction::None
            }
            NavKey::Home => self.move_to_first(&enabled),
            NavKey::End => self.move_to_last(&enabled),
            NavKey::Enter | NavKey::Space => self.activate_current(),
            _ => NavAction::None,
        }
    }

    fn move_next(&mut self, enabled: &[usize]) -> NavAction {
        let cur = self.current_index.unwrap_or(0);
        let next = enabled.iter().find(|&&i| i > cur).copied();
        let target = match next {
            Some(i) => i,
            None if self.wrap => enabled[0],
            None => return NavAction::None,
        };
        self.current_index = Some(target);
        NavAction::Focus(self.items[target].id.clone())
    }

    fn move_prev(&mut self, enabled: &[usize]) -> NavAction {
        let cur = self.current_index.unwrap_or(0);
        let prev = enabled.iter().rev().find(|&&i| i < cur).copied();
        let target = match prev {
            Some(i) => i,
            None if self.wrap => *enabled.last().unwrap(),
            None => return NavAction::None,
        };
        self.current_index = Some(target);
        NavAction::Focus(self.items[target].id.clone())
    }

    fn move_to_first(&mut self, enabled: &[usize]) -> NavAction {
        let target = enabled[0];
        self.current_index = Some(target);
        NavAction::Focus(self.items[target].id.clone())
    }

    fn move_to_last(&mut self, enabled: &[usize]) -> NavAction {
        let target = *enabled.last().unwrap();
        self.current_index = Some(target);
        NavAction::Focus(self.items[target].id.clone())
    }

    fn move_by_offset(&mut self, enabled: &[usize], offset: isize) -> NavAction {
        let cur = self.current_index.unwrap_or(0) as isize;
        let target = cur + offset;
        if target < 0 || target as usize >= self.items.len() {
            if self.wrap {
                let wrapped = ((target % self.items.len() as isize) + self.items.len() as isize)
                    as usize
                    % self.items.len();
                return self.move_to_index(enabled, wrapped);
            }
            return NavAction::None;
        }
        self.move_to_index(enabled, target as usize)
    }

    fn move_to_index(&mut self, enabled: &[usize], target: usize) -> NavAction {
        // Find nearest enabled index at or after target.
        let nearest = enabled
            .iter()
            .min_by_key(|&&i| (i as isize - target as isize).unsigned_abs())
            .copied();
        match nearest {
            Some(i) => {
                self.current_index = Some(i);
                NavAction::Focus(self.items[i].id.clone())
            }
            None => NavAction::None,
        }
    }

    fn activate_current(&self) -> NavAction {
        match self.current_item() {
            Some(item) => NavAction::Activate(item.id.clone()),
            None => NavAction::None,
        }
    }

    /// Typeahead: search items by label prefix. Advances to next match.
    pub fn typeahead(&mut self, ch: char, now_ms: u64) -> NavAction {
        // Reset buffer if timeout elapsed.
        if now_ms.saturating_sub(self.typeahead_last_ms) > self.typeahead_timeout_ms {
            self.typeahead_buffer.clear();
        }
        self.typeahead_last_ms = now_ms;
        self.typeahead_buffer.push(ch.to_ascii_lowercase());

        let start = self.current_index.map(|i| i + 1).unwrap_or(0);
        let len = self.items.len();
        for offset in 0..len {
            let idx = (start + offset) % len;
            let item = &self.items[idx];
            if item.disabled {
                continue;
            }
            if item.label.to_lowercase().starts_with(&self.typeahead_buffer) {
                self.current_index = Some(idx);
                return NavAction::Focus(item.id.clone());
            }
        }
        NavAction::None
    }

    /// Get roving tabindex values: the focused item gets 0, all others get -1.
    pub fn tabindex_map(&self) -> HashMap<String, i32> {
        let mut map = HashMap::new();
        for (i, item) in self.items.iter().enumerate() {
            let ti = if Some(i) == self.current_index { 0 } else { -1 };
            map.insert(item.id.clone(), ti);
        }
        map
    }

    /// Number of items.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

// ── Keyboard Shortcut Registry ────────────────────────────────

/// A registered keyboard shortcut.
#[derive(Debug, Clone)]
pub struct KeyboardShortcut {
    pub key: String,
    pub modifiers: Modifiers,
    pub action_id: String,
    pub description: String,
    pub enabled: bool,
}

/// Registry of keyboard shortcuts.
#[derive(Debug, Default)]
pub struct ShortcutRegistry {
    shortcuts: Vec<KeyboardShortcut>,
}

impl ShortcutRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, shortcut: KeyboardShortcut) {
        self.shortcuts.push(shortcut);
    }

    /// Find matching shortcut for a key + modifiers.
    pub fn find(&self, key: &str, modifiers: Modifiers) -> Option<&KeyboardShortcut> {
        self.shortcuts
            .iter()
            .find(|s| s.enabled && s.key == key && s.modifiers == modifiers)
    }

    pub fn all(&self) -> &[KeyboardShortcut] {
        &self.shortcuts
    }

    pub fn count(&self) -> usize {
        self.shortcuts.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_nav() -> KeyboardNavigator {
        let mut nav = KeyboardNavigator::new(NavPattern::Linear);
        nav.add_item(NavItem::new("item-0", "Apple"));
        nav.add_item(NavItem::new("item-1", "Banana"));
        nav.add_item(NavItem::new("item-2", "Cherry"));
        nav.set_current(0);
        nav
    }

    #[test]
    fn linear_arrow_down() {
        let mut nav = linear_nav();
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::ArrowDown)), NavAction::Focus("item-1".into()));
    }

    #[test]
    fn linear_arrow_up_wraps() {
        let mut nav = linear_nav();
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::ArrowUp)), NavAction::Focus("item-2".into()));
    }

    #[test]
    fn linear_home_end() {
        let mut nav = linear_nav();
        nav.set_current(1);
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::Home)), NavAction::Focus("item-0".into()));
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::End)), NavAction::Focus("item-2".into()));
    }

    #[test]
    fn linear_activate() {
        let mut nav = linear_nav();
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::Enter)), NavAction::Activate("item-0".into()));
    }

    #[test]
    fn skips_disabled() {
        let mut nav = KeyboardNavigator::new(NavPattern::Linear);
        nav.add_item(NavItem::new("a", "A"));
        nav.add_item(NavItem::new("b", "B").with_disabled(true));
        nav.add_item(NavItem::new("c", "C"));
        nav.set_current(0);
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::ArrowDown)), NavAction::Focus("c".into()));
    }

    #[test]
    fn grid_navigation() {
        let mut nav = KeyboardNavigator::new(NavPattern::Grid { columns: 3 });
        for i in 0..9 {
            nav.add_item(NavItem::new(&format!("cell-{}", i), &format!("Cell {}", i)));
        }
        nav.set_current(0);
        // Move right.
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::ArrowRight)), NavAction::Focus("cell-1".into()));
        // Move down from cell-1 (index 1) should go to cell-4 (index 4).
        assert_eq!(nav.handle_key(&NavKeyEvent::new(NavKey::ArrowDown)), NavAction::Focus("cell-4".into()));
    }

    #[test]
    fn tree_expand_collapse() {
        let mut nav = KeyboardNavigator::new(NavPattern::Tree);
        nav.add_item(NavItem::new("folder", "Documents").with_tree(0, true));
        nav.add_item(NavItem::new("file", "readme.txt").with_tree(1, false));
        nav.set_current(0);
        // Right arrow on collapsed parent = toggle expand.
        let action = nav.handle_key(&NavKeyEvent::new(NavKey::ArrowRight));
        assert_eq!(action, NavAction::ToggleExpand("folder".into()));
    }

    #[test]
    fn tree_left_to_parent() {
        let mut nav = KeyboardNavigator::new(NavPattern::Tree);
        nav.add_item(NavItem::new("root", "Root").with_tree(0, true));
        nav.add_item(NavItem::new("child", "Child").with_tree(1, false));
        nav.set_current(1);
        let action = nav.handle_key(&NavKeyEvent::new(NavKey::ArrowLeft));
        assert_eq!(action, NavAction::Focus("root".into()));
    }

    #[test]
    fn roving_tabindex() {
        let nav = linear_nav();
        let map = nav.tabindex_map();
        assert_eq!(map["item-0"], 0);
        assert_eq!(map["item-1"], -1);
        assert_eq!(map["item-2"], -1);
    }

    #[test]
    fn typeahead_search() {
        let mut nav = linear_nav();
        let action = nav.typeahead('b', 1000);
        assert_eq!(action, NavAction::Focus("item-1".into()));
    }

    #[test]
    fn typeahead_multi_char() {
        let mut nav = linear_nav();
        nav.typeahead('c', 1000);
        let action = nav.typeahead('h', 1100); // "ch" matches "Cherry"
        assert_eq!(action, NavAction::Focus("item-2".into()));
    }

    #[test]
    fn typeahead_timeout_resets() {
        let mut nav = linear_nav();
        nav.typeahead('b', 1000); // matches Banana
        let action = nav.typeahead('a', 2000); // timeout, resets buffer, "a" matches Apple
        assert_eq!(action, NavAction::Focus("item-0".into()));
    }

    #[test]
    fn focus_visible_tracker() {
        let mut tracker = FocusVisibleTracker::new();
        assert!(!tracker.should_show_focus());
        tracker.on_keyboard();
        assert!(tracker.should_show_focus());
        tracker.on_pointer();
        assert!(!tracker.should_show_focus());
    }

    #[test]
    fn shortcut_registry() {
        let mut reg = ShortcutRegistry::new();
        reg.register(KeyboardShortcut {
            key: "s".into(),
            modifiers: Modifiers { shift: false, ctrl: true, alt: false, meta: false },
            action_id: "save".into(),
            description: "Save".into(),
            enabled: true,
        });
        let found = reg.find("s", Modifiers { shift: false, ctrl: true, alt: false, meta: false });
        assert!(found.is_some());
        assert_eq!(found.unwrap().action_id, "save");
    }

    #[test]
    fn shortcut_disabled_not_found() {
        let mut reg = ShortcutRegistry::new();
        reg.register(KeyboardShortcut {
            key: "x".into(),
            modifiers: Modifiers::NONE,
            action_id: "cut".into(),
            description: "Cut".into(),
            enabled: false,
        });
        assert!(reg.find("x", Modifiers::NONE).is_none());
    }
}
