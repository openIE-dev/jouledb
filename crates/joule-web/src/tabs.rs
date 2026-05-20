//! Tab Management: dynamic tab groups with activation, reordering, and close.
//!
//! Provides a tab-strip model with keyboard navigation (next/prev wrapping),
//! closable tabs, disabled-tab skipping, and drag-to-reorder support.

// ── Types ───────────────────────────────────────────────────────

/// A single tab.
#[derive(Debug, Clone)]
pub struct Tab {
    pub id: String,
    pub label: String,
    pub content: String,
    pub closable: bool,
    pub disabled: bool,
    pub icon: Option<String>,
    pub badge: Option<String>,
}

impl Tab {
    pub fn new(id: impl Into<String>, label: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            content: content.into(),
            closable: true,
            disabled: false,
            icon: None,
            badge: None,
        }
    }

    pub fn closable(mut self, v: bool) -> Self { self.closable = v; self }
    pub fn disabled(mut self, v: bool) -> Self { self.disabled = v; self }
    pub fn icon(mut self, i: impl Into<String>) -> Self { self.icon = Some(i.into()); self }
    pub fn badge(mut self, b: impl Into<String>) -> Self { self.badge = Some(b.into()); self }
}

/// Manages an ordered collection of tabs.
#[derive(Debug, Clone)]
pub struct TabGroup {
    tabs: Vec<Tab>,
    active_id: Option<String>,
    pub closable_default: bool,
}

// ── Implementation ──────────────────────────────────────────────

impl TabGroup {
    pub fn new() -> Self {
        Self { tabs: Vec::new(), active_id: None, closable_default: true }
    }

    pub fn add_tab(&mut self, tab: Tab) {
        let activate = self.tabs.is_empty() && !tab.disabled;
        let id = tab.id.clone();
        self.tabs.push(tab);
        if activate { self.active_id = Some(id); }
    }

    /// Remove a tab by id. If it was active, activate nearest enabled neighbor.
    pub fn remove_tab(&mut self, id: &str) -> Option<Tab> {
        let pos = self.tabs.iter().position(|t| t.id == id)?;
        let tab = self.tabs.remove(pos);
        if self.active_id.as_deref() == Some(id) {
            self.active_id = None;
            self.activate_nearest(pos);
        }
        Some(tab)
    }

    pub fn activate(&mut self, id: &str) {
        if self.tabs.iter().any(|t| t.id == id && !t.disabled) {
            self.active_id = Some(id.to_string());
        }
    }

    pub fn activate_index(&mut self, idx: usize) {
        if let Some(tab) = self.tabs.get(idx) {
            if !tab.disabled {
                self.active_id = Some(tab.id.clone());
            }
        }
    }

    pub fn active_tab(&self) -> Option<&Tab> {
        self.active_id.as_ref().and_then(|id| self.tabs.iter().find(|t| t.id == *id))
    }

    pub fn active_index(&self) -> Option<usize> {
        self.active_id.as_ref().and_then(|id| self.tabs.iter().position(|t| t.id == *id))
    }

    pub fn next_tab(&mut self) {
        let Some(cur) = self.active_index() else { return };
        let n = self.tabs.len();
        for offset in 1..=n {
            let idx = (cur + offset) % n;
            if !self.tabs[idx].disabled {
                self.active_id = Some(self.tabs[idx].id.clone());
                return;
            }
        }
    }

    pub fn previous_tab(&mut self) {
        let Some(cur) = self.active_index() else { return };
        let n = self.tabs.len();
        for offset in 1..=n {
            let idx = (cur + n - offset) % n;
            if !self.tabs[idx].disabled {
                self.active_id = Some(self.tabs[idx].id.clone());
                return;
            }
        }
    }

    pub fn move_tab(&mut self, from_idx: usize, to_idx: usize) {
        if from_idx >= self.tabs.len() || to_idx >= self.tabs.len() { return; }
        let tab = self.tabs.remove(from_idx);
        self.tabs.insert(to_idx, tab);
    }

    /// Close a tab only if it is closable. Returns the removed tab.
    pub fn close_tab(&mut self, id: &str) -> Option<Tab> {
        let closable = self.tabs.iter().any(|t| t.id == id && t.closable);
        if closable { self.remove_tab(id) } else { None }
    }

    pub fn len(&self) -> usize { self.tabs.len() }
    pub fn is_empty(&self) -> bool { self.tabs.is_empty() }

    pub fn tab_ids(&self) -> Vec<&str> {
        self.tabs.iter().map(|t| t.id.as_str()).collect()
    }

    pub fn tabs(&self) -> &[Tab] { &self.tabs }

    // ── Internal ────────────────────────────────────────────────

    fn activate_nearest(&mut self, removed_pos: usize) {
        if self.tabs.is_empty() { return; }
        let n = self.tabs.len();
        let start = removed_pos.min(n - 1);
        for i in start..n {
            if !self.tabs[i].disabled {
                self.active_id = Some(self.tabs[i].id.clone());
                return;
            }
        }
        for i in (0..start).rev() {
            if !self.tabs[i].disabled {
                self.active_id = Some(self.tabs[i].id.clone());
                return;
            }
        }
    }
}

impl Default for TabGroup {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tabs3() -> TabGroup {
        let mut g = TabGroup::new();
        g.add_tab(Tab::new("a", "Alpha", "A content"));
        g.add_tab(Tab::new("b", "Beta", "B content"));
        g.add_tab(Tab::new("c", "Gamma", "C content"));
        g
    }

    #[test]
    fn activate_by_id() {
        let mut g = tabs3();
        g.activate("b");
        assert_eq!(g.active_tab().unwrap().id, "b");
        assert_eq!(g.active_index(), Some(1));
    }

    #[test]
    fn remove_activates_neighbor() {
        let mut g = tabs3();
        g.activate("b");
        g.remove_tab("b");
        assert_eq!(g.active_tab().unwrap().id, "c");
    }

    #[test]
    fn remove_last_activates_previous() {
        let mut g = tabs3();
        g.activate("c");
        g.remove_tab("c");
        assert!(g.active_tab().is_some());
    }

    #[test]
    fn next_previous_wrap() {
        let mut g = tabs3();
        assert_eq!(g.active_tab().unwrap().id, "a");
        g.next_tab();
        assert_eq!(g.active_tab().unwrap().id, "b");
        g.next_tab();
        assert_eq!(g.active_tab().unwrap().id, "c");
        g.next_tab();
        assert_eq!(g.active_tab().unwrap().id, "a");
        g.previous_tab();
        assert_eq!(g.active_tab().unwrap().id, "c");
    }

    #[test]
    fn skip_disabled() {
        let mut g = TabGroup::new();
        g.add_tab(Tab::new("a", "A", ""));
        g.add_tab(Tab::new("b", "B", "").disabled(true));
        g.add_tab(Tab::new("c", "C", ""));
        g.next_tab();
        assert_eq!(g.active_tab().unwrap().id, "c");
    }

    #[test]
    fn close_only_closable() {
        let mut g = TabGroup::new();
        g.add_tab(Tab::new("a", "A", "").closable(false));
        g.add_tab(Tab::new("b", "B", "").closable(true));
        assert!(g.close_tab("a").is_none());
        assert!(g.close_tab("b").is_some());
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn move_reorders() {
        let mut g = tabs3();
        g.move_tab(0, 2);
        assert_eq!(g.tab_ids(), vec!["b", "c", "a"]);
    }

    #[test]
    fn empty_group() {
        let g = TabGroup::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.active_tab().is_none());
    }

    #[test]
    fn activate_index() {
        let mut g = tabs3();
        g.activate_index(2);
        assert_eq!(g.active_tab().unwrap().id, "c");
    }

    #[test]
    fn first_tab_auto_activated() {
        let g = tabs3();
        assert_eq!(g.active_tab().unwrap().id, "a");
    }
}
