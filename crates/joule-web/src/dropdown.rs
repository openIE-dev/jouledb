//! Dropdown menu component: open/close toggle, item selection, keyboard
//! navigation (up/down/enter/escape), search/filter, multi-select mode,
//! grouped items, disabled items, placement (above/below/auto).

// ── Placement ──────────────────────────────────────────────────────

/// Where the dropdown renders relative to its trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Above,
    Below,
    Auto,
}

// ── Dropdown item ──────────────────────────────────────────────────

/// A single item in a dropdown menu.
#[derive(Debug, Clone)]
pub struct DropdownItem {
    pub id: String,
    pub label: String,
    pub value: String,
    pub disabled: bool,
    pub group: Option<String>,
    pub icon: Option<String>,
}

impl DropdownItem {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            id: id.into(),
            value: label.clone(),
            label,
            disabled: false,
            group: None,
            icon: None,
        }
    }

    pub fn value(mut self, v: impl Into<String>) -> Self {
        self.value = v.into();
        self
    }

    pub fn disabled(mut self, d: bool) -> Self {
        self.disabled = d;
        self
    }

    pub fn group(mut self, g: impl Into<String>) -> Self {
        self.group = Some(g.into());
        self
    }

    pub fn icon(mut self, i: impl Into<String>) -> Self {
        self.icon = Some(i.into());
        self
    }
}

// ── Keyboard key ───────────────────────────────────────────────────

/// Keyboard events that the dropdown handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    ArrowUp,
    ArrowDown,
    Enter,
    Escape,
    Home,
    End,
}

// ── Dropdown ───────────────────────────────────────────────────────

/// A dropdown menu with optional multi-select, filtering, and keyboard nav.
#[derive(Debug)]
pub struct Dropdown {
    pub items: Vec<DropdownItem>,
    pub open: bool,
    pub multi_select: bool,
    pub placement: Placement,
    /// Index of the currently highlighted item (keyboard focus).
    pub highlighted: Option<usize>,
    /// Indices of selected items.
    pub selected: Vec<usize>,
    /// Current filter/search string.
    pub filter: String,
    /// Available space above trigger (for auto placement).
    pub space_above: f64,
    /// Available space below trigger (for auto placement).
    pub space_below: f64,
}

impl Default for Dropdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Dropdown {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            open: false,
            multi_select: false,
            placement: Placement::Below,
            highlighted: None,
            selected: Vec::new(),
            filter: String::new(),
            space_above: 200.0,
            space_below: 200.0,
        }
    }

    pub fn with_items(mut self, items: Vec<DropdownItem>) -> Self {
        self.items = items;
        self
    }

    pub fn multi_select(mut self, enabled: bool) -> Self {
        self.multi_select = enabled;
        self
    }

    pub fn placement(mut self, p: Placement) -> Self {
        self.placement = p;
        self
    }

    /// Toggle the dropdown open/closed.
    pub fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.filter.clear();
            // Highlight first enabled visible item
            self.highlighted = self.first_enabled_visible();
        } else {
            self.highlighted = None;
        }
    }

    /// Set filter text and reset highlight.
    pub fn set_filter(&mut self, text: &str) {
        self.filter = text.to_lowercase();
        self.highlighted = self.first_enabled_visible();
    }

    /// Indices of items visible after filtering.
    pub fn visible_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                if self.filter.is_empty() {
                    true
                } else {
                    item.label.to_lowercase().contains(&self.filter)
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn first_enabled_visible(&self) -> Option<usize> {
        self.visible_indices()
            .into_iter()
            .find(|i| !self.items[*i].disabled)
    }

    /// Handle a keyboard event. Returns true if the event was consumed.
    pub fn handle_key(&mut self, key: Key) -> bool {
        if !self.open && key != Key::ArrowDown && key != Key::Enter {
            return false;
        }

        match key {
            Key::Escape => {
                self.open = false;
                self.highlighted = None;
                true
            }
            Key::ArrowDown => {
                if !self.open {
                    self.toggle();
                    return true;
                }
                let vis = self.visible_indices();
                if vis.is_empty() {
                    return false;
                }
                let current_pos = self
                    .highlighted
                    .and_then(|h| vis.iter().position(|v| *v == h))
                    .unwrap_or(vis.len().wrapping_sub(1));
                // Move forward, skipping disabled
                for offset in 1..=vis.len() {
                    let idx = vis[(current_pos + offset) % vis.len()];
                    if !self.items[idx].disabled {
                        self.highlighted = Some(idx);
                        return true;
                    }
                }
                false
            }
            Key::ArrowUp => {
                let vis = self.visible_indices();
                if vis.is_empty() {
                    return false;
                }
                let current_pos = self
                    .highlighted
                    .and_then(|h| vis.iter().position(|v| *v == h))
                    .unwrap_or(1);
                for offset in 1..=vis.len() {
                    let idx = vis[(current_pos + vis.len() - offset) % vis.len()];
                    if !self.items[idx].disabled {
                        self.highlighted = Some(idx);
                        return true;
                    }
                }
                false
            }
            Key::Enter => {
                if !self.open {
                    self.toggle();
                    return true;
                }
                if let Some(hi) = self.highlighted {
                    if !self.items[hi].disabled {
                        self.select_index(hi);
                        if !self.multi_select {
                            self.open = false;
                        }
                        return true;
                    }
                }
                false
            }
            Key::Home => {
                self.highlighted = self.first_enabled_visible();
                self.highlighted.is_some()
            }
            Key::End => {
                let vis = self.visible_indices();
                self.highlighted = vis.into_iter().rev().find(|i| !self.items[*i].disabled);
                self.highlighted.is_some()
            }
        }
    }

    /// Select item at index.
    pub fn select_index(&mut self, index: usize) {
        if index >= self.items.len() || self.items[index].disabled {
            return;
        }
        if self.multi_select {
            if let Some(pos) = self.selected.iter().position(|s| *s == index) {
                self.selected.remove(pos);
            } else {
                self.selected.push(index);
            }
        } else {
            self.selected = vec![index];
        }
    }

    /// Get selected items.
    pub fn selected_items(&self) -> Vec<&DropdownItem> {
        self.selected
            .iter()
            .filter_map(|i| self.items.get(*i))
            .collect()
    }

    /// Resolved placement considering auto.
    pub fn resolved_placement(&self) -> Placement {
        match self.placement {
            Placement::Auto => {
                if self.space_below >= self.space_above {
                    Placement::Below
                } else {
                    Placement::Above
                }
            }
            other => other,
        }
    }

    /// Group items by their group field.
    pub fn grouped_items(&self) -> Vec<(Option<String>, Vec<usize>)> {
        let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
        for (i, item) in self.items.iter().enumerate() {
            if let Some(existing) = groups.iter_mut().find(|(g, _)| *g == item.group) {
                existing.1.push(i);
            } else {
                groups.push((item.group.clone(), vec![i]));
            }
        }
        groups
    }

    /// Render to HTML.
    pub fn render(&self) -> String {
        let placement_class = match self.resolved_placement() {
            Placement::Above => "dropdown--above",
            Placement::Below => "dropdown--below",
            Placement::Auto => "dropdown--below",
        };
        let open_class = if self.open { " dropdown--open" } else { "" };

        let mut html = format!(
            "<div class=\"dropdown {}{}\" role=\"listbox\" aria-expanded=\"{}\">",
            placement_class, open_class, self.open
        );

        if self.open {
            html.push_str("<ul class=\"dropdown-menu\">");
            for (i, item) in self.items.iter().enumerate() {
                let vis = self.visible_indices();
                if !vis.contains(&i) {
                    continue;
                }
                let selected = self.selected.contains(&i);
                let highlighted = self.highlighted == Some(i);
                let disabled = item.disabled;
                html.push_str(&format!(
                    "<li class=\"dropdown-item{}{}{}\" data-value=\"{}\" aria-selected=\"{}\" aria-disabled=\"{}\">{}</li>",
                    if selected { " selected" } else { "" },
                    if highlighted { " highlighted" } else { "" },
                    if disabled { " disabled" } else { "" },
                    item.value,
                    selected,
                    disabled,
                    item.label,
                ));
            }
            html.push_str("</ul>");
        }

        html.push_str("</div>");
        html
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_items() -> Vec<DropdownItem> {
        vec![
            DropdownItem::new("1", "Apple"),
            DropdownItem::new("2", "Banana"),
            DropdownItem::new("3", "Cherry").disabled(true),
            DropdownItem::new("4", "Date"),
        ]
    }

    #[test]
    fn test_toggle_open_close() {
        let mut dd = Dropdown::new().with_items(sample_items());
        assert!(!dd.open);
        dd.toggle();
        assert!(dd.open);
        dd.toggle();
        assert!(!dd.open);
    }

    #[test]
    fn test_single_select() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        dd.select_index(0);
        assert_eq!(dd.selected, vec![0]);
        dd.select_index(1);
        assert_eq!(dd.selected, vec![1]);
    }

    #[test]
    fn test_multi_select() {
        let mut dd = Dropdown::new().with_items(sample_items()).multi_select(true);
        dd.toggle();
        dd.select_index(0);
        dd.select_index(1);
        assert_eq!(dd.selected, vec![0, 1]);
        // Deselect
        dd.select_index(0);
        assert_eq!(dd.selected, vec![1]);
    }

    #[test]
    fn test_disabled_item_not_selectable() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        dd.select_index(2); // Cherry is disabled
        assert!(dd.selected.is_empty());
    }

    #[test]
    fn test_keyboard_arrow_down() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        assert_eq!(dd.highlighted, Some(0));
        dd.handle_key(Key::ArrowDown);
        assert_eq!(dd.highlighted, Some(1));
        // Skip disabled item (index 2)
        dd.handle_key(Key::ArrowDown);
        assert_eq!(dd.highlighted, Some(3));
    }

    #[test]
    fn test_keyboard_arrow_up() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        dd.highlighted = Some(3);
        dd.handle_key(Key::ArrowUp);
        // Skip disabled index 2
        assert_eq!(dd.highlighted, Some(1));
    }

    #[test]
    fn test_keyboard_enter_selects() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        dd.highlighted = Some(1);
        dd.handle_key(Key::Enter);
        assert_eq!(dd.selected, vec![1]);
        assert!(!dd.open); // Closes after single select
    }

    #[test]
    fn test_keyboard_escape_closes() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        assert!(dd.open);
        dd.handle_key(Key::Escape);
        assert!(!dd.open);
    }

    #[test]
    fn test_filter_items() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.toggle();
        dd.set_filter("an");
        let vis = dd.visible_indices();
        // "Banana" matches "an"
        assert!(vis.contains(&1));
        // "Apple" does not
        assert!(!vis.contains(&0));
    }

    #[test]
    fn test_auto_placement() {
        let mut dd = Dropdown::new().placement(Placement::Auto);
        dd.space_above = 50.0;
        dd.space_below = 200.0;
        assert_eq!(dd.resolved_placement(), Placement::Below);
        dd.space_above = 300.0;
        dd.space_below = 100.0;
        assert_eq!(dd.resolved_placement(), Placement::Above);
    }

    #[test]
    fn test_grouped_items() {
        let items = vec![
            DropdownItem::new("1", "A").group("Fruits"),
            DropdownItem::new("2", "B").group("Fruits"),
            DropdownItem::new("3", "C").group("Vegs"),
        ];
        let dd = Dropdown::new().with_items(items);
        let groups = dd.grouped_items();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, Some("Fruits".into()));
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn test_render_html_contains_role() {
        let dd = Dropdown::new().with_items(sample_items());
        let html = dd.render();
        assert!(html.contains("role=\"listbox\""));
        assert!(html.contains("aria-expanded=\"false\""));
    }

    #[test]
    fn test_selected_items() {
        let mut dd = Dropdown::new().with_items(sample_items());
        dd.select_index(0);
        let selected = dd.selected_items();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].label, "Apple");
    }
}
