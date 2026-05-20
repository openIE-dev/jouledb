//! Accordion / Collapsible: expand/collapse panels with single or multiple mode.
//!
//! Supports single-open (accordion) and multi-open (collapsible) modes,
//! disabled items, and bulk expand/collapse.

// ── Types ───────────────────────────────────────────────────────

/// A single collapsible panel.
#[derive(Debug, Clone)]
pub struct AccordionItem {
    pub id: String,
    pub title: String,
    pub content: String,
    pub expanded: bool,
    pub disabled: bool,
}

/// Controls how many panels can be open simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccordionMode {
    /// Only one panel may be expanded at a time.
    Single,
    /// Any number of panels may be expanded.
    Multiple,
}

/// Manages a list of accordion items.
#[derive(Debug, Clone)]
pub struct Accordion {
    items: Vec<AccordionItem>,
    mode: AccordionMode,
}

// ── Implementation ──────────────────────────────────────────────

impl Accordion {
    pub fn new(mode: AccordionMode) -> Self {
        Self { items: Vec::new(), mode }
    }

    pub fn add_item(&mut self, id: impl Into<String>, title: impl Into<String>, content: impl Into<String>) {
        self.items.push(AccordionItem {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            expanded: false,
            disabled: false,
        });
    }

    pub fn toggle(&mut self, id: &str) {
        let Some(pos) = self.items.iter().position(|i| i.id == id) else { return };
        if self.items[pos].disabled { return; }
        let currently_expanded = self.items[pos].expanded;
        if self.mode == AccordionMode::Single && !currently_expanded {
            for item in &mut self.items { item.expanded = false; }
        }
        self.items[pos].expanded = !currently_expanded;
    }

    pub fn expand(&mut self, id: &str) {
        let Some(pos) = self.items.iter().position(|i| i.id == id) else { return };
        if self.items[pos].disabled { return; }
        if self.mode == AccordionMode::Single {
            for item in &mut self.items { item.expanded = false; }
        }
        self.items[pos].expanded = true;
    }

    pub fn collapse(&mut self, id: &str) {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.expanded = false;
        }
    }

    pub fn expand_all(&mut self) {
        match self.mode {
            AccordionMode::Multiple => {
                for item in &mut self.items {
                    if !item.disabled { item.expanded = true; }
                }
            }
            AccordionMode::Single => {
                let mut found = false;
                for item in &mut self.items {
                    if !found && !item.disabled {
                        item.expanded = true;
                        found = true;
                    } else {
                        item.expanded = false;
                    }
                }
            }
        }
    }

    pub fn collapse_all(&mut self) {
        for item in &mut self.items { item.expanded = false; }
    }

    pub fn expanded_ids(&self) -> Vec<&str> {
        self.items.iter().filter(|i| i.expanded).map(|i| i.id.as_str()).collect()
    }

    pub fn is_expanded(&self, id: &str) -> bool {
        self.items.iter().any(|i| i.id == id && i.expanded)
    }

    pub fn item_count(&self) -> usize { self.items.len() }

    pub fn remove_item(&mut self, id: &str) -> bool {
        let before = self.items.len();
        self.items.retain(|i| i.id != id);
        self.items.len() < before
    }

    pub fn set_disabled(&mut self, id: &str, disabled: bool) {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.disabled = disabled;
            if disabled { item.expanded = false; }
        }
    }

    pub fn items(&self) -> &[AccordionItem] { &self.items }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_accordion(mode: AccordionMode) -> Accordion {
        let mut a = Accordion::new(mode);
        a.add_item("a", "Alpha", "Content A");
        a.add_item("b", "Beta", "Content B");
        a.add_item("c", "Gamma", "Content C");
        a
    }

    #[test]
    fn single_mode_collapses_others() {
        let mut a = make_accordion(AccordionMode::Single);
        a.toggle("a");
        assert!(a.is_expanded("a"));
        a.toggle("b");
        assert!(!a.is_expanded("a"));
        assert!(a.is_expanded("b"));
    }

    #[test]
    fn multiple_mode_allows_many() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.toggle("a");
        a.toggle("b");
        assert!(a.is_expanded("a"));
        assert!(a.is_expanded("b"));
    }

    #[test]
    fn toggle_collapses_expanded() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.toggle("a");
        assert!(a.is_expanded("a"));
        a.toggle("a");
        assert!(!a.is_expanded("a"));
    }

    #[test]
    fn expand_all_multiple() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.expand_all();
        assert_eq!(a.expanded_ids().len(), 3);
    }

    #[test]
    fn expand_all_single() {
        let mut a = make_accordion(AccordionMode::Single);
        a.expand_all();
        assert_eq!(a.expanded_ids().len(), 1);
        assert_eq!(a.expanded_ids()[0], "a");
    }

    #[test]
    fn collapse_all() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.expand_all();
        a.collapse_all();
        assert!(a.expanded_ids().is_empty());
    }

    #[test]
    fn disabled_cannot_toggle() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.set_disabled("b", true);
        a.toggle("b");
        assert!(!a.is_expanded("b"));
    }

    #[test]
    fn disabled_collapses_if_expanded() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.toggle("b");
        assert!(a.is_expanded("b"));
        a.set_disabled("b", true);
        assert!(!a.is_expanded("b"));
    }

    #[test]
    fn remove_item() {
        let mut a = make_accordion(AccordionMode::Multiple);
        assert!(a.remove_item("b"));
        assert_eq!(a.item_count(), 2);
        assert!(!a.remove_item("b"));
    }

    #[test]
    fn expand_and_collapse() {
        let mut a = make_accordion(AccordionMode::Multiple);
        a.expand("a");
        assert!(a.is_expanded("a"));
        a.collapse("a");
        assert!(!a.is_expanded("a"));
    }
}
