//! Row selection model: single, multiple, range, select-all, checkbox state.
//!
//! Pure headless logic — the UI layer reads selection state and renders
//! checkboxes, highlights, etc.

use std::collections::HashSet;

// ── SelectionMode ───────────────────────────────────────────────

/// How rows can be selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Selection is disabled.
    None,
    /// Only one row at a time.
    Single,
    /// Multiple individual rows via ctrl-click.
    Multiple,
    /// Contiguous range via shift-click.
    Range,
}

// ── CheckboxState ───────────────────────────────────────────────

/// Visual state for a header "select all" checkbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckboxState {
    Checked,
    Unchecked,
    Indeterminate,
}

// ── RowSelection ────────────────────────────────────────────────

/// Manages which rows are currently selected.
#[derive(Debug, Clone)]
pub struct RowSelection {
    pub mode: SelectionMode,
    selected: HashSet<String>,
    /// Anchor row id for range selection (shift-click start).
    anchor: Option<String>,
}

impl RowSelection {
    pub fn new(mode: SelectionMode) -> Self {
        Self {
            mode,
            selected: HashSet::new(),
            anchor: None,
        }
    }

    /// Number of selected rows.
    pub fn count(&self) -> usize {
        self.selected.len()
    }

    /// Whether a particular row is selected.
    pub fn is_selected(&self, row_id: &str) -> bool {
        self.selected.contains(row_id)
    }

    /// The set of selected row ids.
    pub fn selected_ids(&self) -> &HashSet<String> {
        &self.selected
    }

    /// Select a row (single-mode replaces previous selection).
    pub fn select(&mut self, row_id: impl Into<String>) {
        let id = row_id.into();
        match self.mode {
            SelectionMode::None => {}
            SelectionMode::Single => {
                self.selected.clear();
                self.selected.insert(id.clone());
                self.anchor = Some(id);
            }
            SelectionMode::Multiple | SelectionMode::Range => {
                self.selected.insert(id.clone());
                self.anchor = Some(id);
            }
        }
    }

    /// Deselect a row.
    pub fn deselect(&mut self, row_id: &str) {
        self.selected.remove(row_id);
    }

    /// Toggle a row's selection state.
    pub fn toggle(&mut self, row_id: impl Into<String>) {
        let id = row_id.into();
        if self.selected.contains(&id) {
            self.selected.remove(&id);
        } else {
            self.select(id);
        }
    }

    /// Select a range of rows from the anchor to the target.
    /// `ordered_ids` is the full list of visible row ids in display order.
    pub fn select_range(&mut self, target_id: &str, ordered_ids: &[&str]) {
        if self.mode == SelectionMode::None {
            return;
        }
        let anchor = match &self.anchor {
            Some(a) => a.clone(),
            Option::None => {
                self.select(target_id.to_string());
                return;
            }
        };
        let anchor_pos = ordered_ids.iter().position(|id| *id == anchor);
        let target_pos = ordered_ids.iter().position(|id| *id == target_id);
        if let (Some(a), Some(t)) = (anchor_pos, target_pos) {
            let (start, end) = if a <= t { (a, t) } else { (t, a) };
            for id in &ordered_ids[start..=end] {
                self.selected.insert(id.to_string());
            }
        }
    }

    /// Select all rows.
    pub fn select_all(&mut self, all_ids: &[&str]) {
        if self.mode == SelectionMode::None {
            return;
        }
        for id in all_ids {
            self.selected.insert(id.to_string());
        }
    }

    /// Deselect all rows.
    pub fn deselect_all(&mut self) {
        self.selected.clear();
        self.anchor = None;
    }

    /// Get the header checkbox state given the total row count.
    pub fn header_checkbox_state(&self, total_rows: usize) -> CheckboxState {
        if self.selected.is_empty() || total_rows == 0 {
            CheckboxState::Unchecked
        } else if self.selected.len() >= total_rows {
            CheckboxState::Checked
        } else {
            CheckboxState::Indeterminate
        }
    }

    /// Change selection mode (clears current selection).
    pub fn set_mode(&mut self, mode: SelectionMode) {
        self.mode = mode;
        self.selected.clear();
        self.anchor = None;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_mode_replaces() {
        let mut sel = RowSelection::new(SelectionMode::Single);
        sel.select("r1");
        sel.select("r2");
        assert_eq!(sel.count(), 1);
        assert!(sel.is_selected("r2"));
        assert!(!sel.is_selected("r1"));
    }

    #[test]
    fn multiple_mode_accumulates() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        sel.select("r1");
        sel.select("r2");
        assert_eq!(sel.count(), 2);
    }

    #[test]
    fn toggle_selection() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        sel.toggle("r1");
        assert!(sel.is_selected("r1"));
        sel.toggle("r1");
        assert!(!sel.is_selected("r1"));
    }

    #[test]
    fn deselect() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        sel.select("r1");
        sel.deselect("r1");
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn select_range_forward() {
        let mut sel = RowSelection::new(SelectionMode::Range);
        let ids = vec!["r1", "r2", "r3", "r4", "r5"];
        sel.select("r2"); // anchor
        sel.select_range("r4", &ids);
        assert!(sel.is_selected("r2"));
        assert!(sel.is_selected("r3"));
        assert!(sel.is_selected("r4"));
        assert!(!sel.is_selected("r1"));
    }

    #[test]
    fn select_range_backward() {
        let mut sel = RowSelection::new(SelectionMode::Range);
        let ids = vec!["r1", "r2", "r3", "r4", "r5"];
        sel.select("r4");
        sel.select_range("r2", &ids);
        assert!(sel.is_selected("r2"));
        assert!(sel.is_selected("r3"));
        assert!(sel.is_selected("r4"));
    }

    #[test]
    fn select_all_deselect_all() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        let ids = vec!["r1", "r2", "r3"];
        sel.select_all(&ids);
        assert_eq!(sel.count(), 3);
        sel.deselect_all();
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn header_checkbox_states() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        assert_eq!(sel.header_checkbox_state(3), CheckboxState::Unchecked);
        sel.select("r1");
        assert_eq!(sel.header_checkbox_state(3), CheckboxState::Indeterminate);
        sel.select("r2");
        sel.select("r3");
        assert_eq!(sel.header_checkbox_state(3), CheckboxState::Checked);
    }

    #[test]
    fn none_mode_ignores_select() {
        let mut sel = RowSelection::new(SelectionMode::None);
        sel.select("r1");
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn set_mode_clears() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        sel.select("r1");
        sel.select("r2");
        sel.set_mode(SelectionMode::Single);
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn selected_ids_returns_set() {
        let mut sel = RowSelection::new(SelectionMode::Multiple);
        sel.select("r1");
        sel.select("r2");
        let ids = sel.selected_ids();
        assert!(ids.contains("r1"));
        assert!(ids.contains("r2"));
    }
}
