//! Focus trap manager: trap activation/deactivation, focusable element tracking,
//! tab order management, initial focus, return focus, escape handling, nesting.
//!
//! Pure data — no browser dependency. Tracks focusable element IDs and computes
//! tab-order navigation so renderers can apply focus management.

use std::collections::HashMap;

// ── Focusable Element ─────────────────────────────────────────

/// A focusable element within a trap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FocusableElement {
    pub id: String,
    pub tab_index: i32,
    pub disabled: bool,
}

impl FocusableElement {
    pub fn new(id: &str, tab_index: i32) -> Self {
        Self { id: id.into(), tab_index, disabled: false }
    }

    pub fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

// ── Focus Trap ─────────────────────────────────────────────────

/// Configuration for escape key behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeBehavior {
    /// Deactivate the trap on Escape.
    Deactivate,
    /// Do nothing on Escape.
    Ignore,
}

/// A single focus trap scope.
#[derive(Debug, Clone)]
pub struct FocusTrap {
    pub id: String,
    elements: Vec<FocusableElement>,
    active: bool,
    initial_focus_id: Option<String>,
    return_focus_id: Option<String>,
    current_index: Option<usize>,
    escape_behavior: EscapeBehavior,
}

impl FocusTrap {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            elements: Vec::new(),
            active: false,
            initial_focus_id: None,
            return_focus_id: None,
            current_index: None,
            escape_behavior: EscapeBehavior::Deactivate,
        }
    }

    /// Set which element receives focus when the trap activates.
    pub fn set_initial_focus(&mut self, element_id: &str) {
        self.initial_focus_id = Some(element_id.into());
    }

    /// Set which element receives focus when the trap deactivates.
    pub fn set_return_focus(&mut self, element_id: &str) {
        self.return_focus_id = Some(element_id.into());
    }

    /// Set escape key behavior.
    pub fn set_escape_behavior(&mut self, behavior: EscapeBehavior) {
        self.escape_behavior = behavior;
    }

    /// Add a focusable element.
    pub fn add_element(&mut self, elem: FocusableElement) {
        self.elements.push(elem);
    }

    /// Remove an element by ID.
    pub fn remove_element(&mut self, id: &str) {
        self.elements.retain(|e| e.id != id);
        // Reset current index if it's now out of bounds.
        if let Some(idx) = self.current_index {
            let focusable = self.focusable_elements();
            if idx >= focusable.len() {
                self.current_index = if focusable.is_empty() { None } else { Some(0) };
            }
        }
    }

    /// Get tab-ordered focusable (non-disabled) elements.
    fn focusable_elements(&self) -> Vec<FocusableElement> {
        let mut elems: Vec<_> = self.elements.iter().filter(|e| !e.disabled).cloned().collect();
        // Positive tab_index first (ascending), then zero tab_index in DOM order.
        elems.sort_by(|a, b| {
            let a_pos = if a.tab_index > 0 { 0 } else { 1 };
            let b_pos = if b.tab_index > 0 { 0 } else { 1 };
            a_pos.cmp(&b_pos).then_with(|| a.tab_index.cmp(&b.tab_index))
        });
        elems
    }

    /// Activate the trap — returns the ID of the element that should receive focus.
    pub fn activate(&mut self) -> Option<String> {
        self.active = true;
        let focusable = self.focusable_elements();
        if focusable.is_empty() {
            return None;
        }
        // Use initial_focus_id if it exists and is focusable.
        if let Some(init_id) = &self.initial_focus_id {
            if let Some(pos) = focusable.iter().position(|e| e.id == *init_id) {
                self.current_index = Some(pos);
                return Some(focusable[pos].id.clone());
            }
        }
        self.current_index = Some(0);
        Some(focusable[0].id.clone())
    }

    /// Deactivate the trap — returns the return-focus element ID if set.
    pub fn deactivate(&mut self) -> Option<String> {
        self.active = false;
        self.current_index = None;
        self.return_focus_id.clone()
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Move focus forward (Tab). Wraps around. Returns the newly focused ID.
    pub fn focus_next(&mut self) -> Option<String> {
        if !self.active {
            return None;
        }
        let focusable = self.focusable_elements();
        if focusable.is_empty() {
            return None;
        }
        let idx = match self.current_index {
            Some(i) => (i + 1) % focusable.len(),
            None => 0,
        };
        self.current_index = Some(idx);
        Some(focusable[idx].id.clone())
    }

    /// Move focus backward (Shift+Tab). Wraps around.
    pub fn focus_prev(&mut self) -> Option<String> {
        if !self.active {
            return None;
        }
        let focusable = self.focusable_elements();
        if focusable.is_empty() {
            return None;
        }
        let idx = match self.current_index {
            Some(0) | None => focusable.len() - 1,
            Some(i) => i - 1,
        };
        self.current_index = Some(idx);
        Some(focusable[idx].id.clone())
    }

    /// Handle Escape key press. Returns `Some(return_id)` if the trap deactivated.
    pub fn handle_escape(&mut self) -> Option<String> {
        if self.escape_behavior == EscapeBehavior::Deactivate {
            self.deactivate()
        } else {
            None
        }
    }

    /// Current focused element ID.
    pub fn current_focus(&self) -> Option<String> {
        let focusable = self.focusable_elements();
        self.current_index.and_then(|i| focusable.get(i).map(|e| e.id.clone()))
    }

    /// Number of focusable elements.
    pub fn focusable_count(&self) -> usize {
        self.focusable_elements().len()
    }
}

// ── Nested Focus Trap Manager ─────────────────────────────────

/// Manages a stack of nested focus traps.
#[derive(Debug, Default)]
pub struct FocusTrapStack {
    traps: HashMap<String, FocusTrap>,
    stack: Vec<String>,
}

impl FocusTrapStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trap (does not activate it).
    pub fn register(&mut self, trap: FocusTrap) {
        self.traps.insert(trap.id.clone(), trap);
    }

    /// Push and activate a trap. Returns the initial focus element ID.
    pub fn push(&mut self, trap_id: &str) -> Option<String> {
        // Deactivate current top if any.
        if let Some(top_id) = self.stack.last() {
            if let Some(trap) = self.traps.get_mut(top_id) {
                trap.deactivate();
            }
        }
        self.stack.push(trap_id.into());
        if let Some(trap) = self.traps.get_mut(trap_id) {
            trap.activate()
        } else {
            None
        }
    }

    /// Pop and deactivate the topmost trap. Returns the return-focus element ID.
    pub fn pop(&mut self) -> Option<String> {
        let top_id = self.stack.pop()?;
        let return_id = if let Some(trap) = self.traps.get_mut(&top_id) {
            trap.deactivate()
        } else {
            None
        };
        // Re-activate the new top.
        if let Some(new_top_id) = self.stack.last() {
            if let Some(trap) = self.traps.get_mut(new_top_id) {
                trap.activate();
            }
        }
        return_id
    }

    /// Get the currently active trap.
    pub fn active_trap(&self) -> Option<&FocusTrap> {
        self.stack.last().and_then(|id| self.traps.get(id))
    }

    /// Get the currently active trap mutably.
    pub fn active_trap_mut(&mut self) -> Option<&mut FocusTrap> {
        let id = self.stack.last()?.clone();
        self.traps.get_mut(&id)
    }

    /// Depth of the trap stack.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trap() -> FocusTrap {
        let mut trap = FocusTrap::new("dialog");
        trap.add_element(FocusableElement::new("input1", 0));
        trap.add_element(FocusableElement::new("btn-cancel", 0));
        trap.add_element(FocusableElement::new("btn-ok", 0));
        trap
    }

    #[test]
    fn activate_focuses_first() {
        let mut trap = sample_trap();
        let focused = trap.activate();
        assert_eq!(focused, Some("input1".into()));
        assert!(trap.is_active());
    }

    #[test]
    fn activate_with_initial_focus() {
        let mut trap = sample_trap();
        trap.set_initial_focus("btn-ok");
        let focused = trap.activate();
        assert_eq!(focused, Some("btn-ok".into()));
    }

    #[test]
    fn tab_wraps_forward() {
        let mut trap = sample_trap();
        trap.activate();
        assert_eq!(trap.focus_next(), Some("btn-cancel".into()));
        assert_eq!(trap.focus_next(), Some("btn-ok".into()));
        assert_eq!(trap.focus_next(), Some("input1".into())); // wrap
    }

    #[test]
    fn shift_tab_wraps_backward() {
        let mut trap = sample_trap();
        trap.activate();
        assert_eq!(trap.focus_prev(), Some("btn-ok".into())); // wrap to end
        assert_eq!(trap.focus_prev(), Some("btn-cancel".into()));
    }

    #[test]
    fn deactivate_returns_focus() {
        let mut trap = sample_trap();
        trap.set_return_focus("trigger-btn");
        trap.activate();
        let ret = trap.deactivate();
        assert_eq!(ret, Some("trigger-btn".into()));
        assert!(!trap.is_active());
    }

    #[test]
    fn escape_deactivates() {
        let mut trap = sample_trap();
        trap.set_return_focus("trigger");
        trap.activate();
        let ret = trap.handle_escape();
        assert_eq!(ret, Some("trigger".into()));
        assert!(!trap.is_active());
    }

    #[test]
    fn escape_ignore() {
        let mut trap = sample_trap();
        trap.set_escape_behavior(EscapeBehavior::Ignore);
        trap.activate();
        let ret = trap.handle_escape();
        assert_eq!(ret, None);
        assert!(trap.is_active());
    }

    #[test]
    fn disabled_elements_skipped() {
        let mut trap = FocusTrap::new("t");
        trap.add_element(FocusableElement::new("a", 0));
        trap.add_element(FocusableElement::new("b", 0).with_disabled(true));
        trap.add_element(FocusableElement::new("c", 0));
        trap.activate();
        assert_eq!(trap.focusable_count(), 2);
        assert_eq!(trap.focus_next(), Some("c".into()));
        assert_eq!(trap.focus_next(), Some("a".into())); // wrap, skip b
    }

    #[test]
    fn remove_element() {
        let mut trap = sample_trap();
        trap.activate();
        trap.remove_element("btn-cancel");
        assert_eq!(trap.focusable_count(), 2);
    }

    #[test]
    fn tab_order_positive_first() {
        let mut trap = FocusTrap::new("t");
        trap.add_element(FocusableElement::new("dom-first", 0));
        trap.add_element(FocusableElement::new("tab2", 2));
        trap.add_element(FocusableElement::new("tab1", 1));
        let focused = trap.activate();
        assert_eq!(focused, Some("tab1".into())); // tab_index 1 comes first
    }

    #[test]
    fn nested_trap_stack() {
        let mut stack = FocusTrapStack::new();
        let mut outer = FocusTrap::new("outer");
        outer.add_element(FocusableElement::new("outer-input", 0));
        let mut inner = FocusTrap::new("inner");
        inner.add_element(FocusableElement::new("inner-input", 0));
        inner.set_return_focus("outer-input");

        stack.register(outer);
        stack.register(inner);

        let f1 = stack.push("outer");
        assert_eq!(f1, Some("outer-input".into()));
        assert_eq!(stack.depth(), 1);

        let f2 = stack.push("inner");
        assert_eq!(f2, Some("inner-input".into()));
        assert_eq!(stack.depth(), 2);

        let ret = stack.pop();
        assert_eq!(ret, Some("outer-input".into()));
        assert_eq!(stack.depth(), 1);
        assert!(stack.active_trap().unwrap().is_active());
    }

    #[test]
    fn inactive_trap_no_navigation() {
        let mut trap = sample_trap();
        assert_eq!(trap.focus_next(), None);
        assert_eq!(trap.focus_prev(), None);
    }

    #[test]
    fn current_focus_tracks() {
        let mut trap = sample_trap();
        assert_eq!(trap.current_focus(), None);
        trap.activate();
        assert_eq!(trap.current_focus(), Some("input1".into()));
        trap.focus_next();
        assert_eq!(trap.current_focus(), Some("btn-cancel".into()));
    }
}
