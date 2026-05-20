//! In-cell editing: editor types, validation, commit/cancel, tab navigation,
//! dirty-cell tracking.
//!
//! Pure headless model — the renderer reads `editing_state()` and shows the
//! appropriate input widget.

use std::collections::{HashMap, HashSet};

// ── CellEditorType ──────────────────────────────────────────────

/// The kind of editor to show for a cell.
#[derive(Debug, Clone, PartialEq)]
pub enum CellEditorType {
    Text,
    Number,
    Select(Vec<String>),
    Checkbox,
    Date,
}

// ── CellValue ───────────────────────────────────────────────────

/// A loosely-typed cell value used during editing.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Text(String),
    Number(f64),
    Bool(bool),
    Empty,
}

impl CellValue {
    pub fn as_text(&self) -> String {
        match self {
            CellValue::Text(s) => s.clone(),
            CellValue::Number(n) => n.to_string(),
            CellValue::Bool(b) => b.to_string(),
            CellValue::Empty => String::new(),
        }
    }
}

// ── Validation ──────────────────────────────────────────────────

/// Validation rule for a cell editor.
#[derive(Debug, Clone)]
pub enum ValidationRule {
    Required,
    /// Regex pattern the text value must match.
    Pattern(String),
    /// Minimum value for Number cells.
    Min(f64),
    /// Maximum value for Number cells.
    Max(f64),
}

/// Result of validating a cell value.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    pub valid: bool,
    pub messages: Vec<String>,
}

impl ValidationResult {
    pub fn ok() -> Self {
        Self { valid: true, messages: Vec::new() }
    }

    pub fn fail(msg: impl Into<String>) -> Self {
        Self { valid: false, messages: vec![msg.into()] }
    }
}

/// Validate a `CellValue` against a set of rules.
pub fn validate(value: &CellValue, rules: &[ValidationRule]) -> ValidationResult {
    let mut messages = Vec::new();
    for rule in rules {
        match rule {
            ValidationRule::Required => {
                let empty = matches!(value, CellValue::Empty)
                    || matches!(value, CellValue::Text(s) if s.is_empty());
                if empty {
                    messages.push("Value is required".into());
                }
            }
            ValidationRule::Pattern(pat) => {
                let text = value.as_text();
                // Simple pattern matching: treat pat as a contains check when
                // full regex isn't available (no regex crate).  We implement
                // basic anchored-glob: `^...$` means exact match.
                if pat.starts_with('^') && pat.ends_with('$') {
                    let inner = &pat[1..pat.len() - 1];
                    if text != inner {
                        messages.push(format!("Value must match pattern {pat}"));
                    }
                } else if !text.contains(pat.as_str()) {
                    messages.push(format!("Value must match pattern {pat}"));
                }
            }
            ValidationRule::Min(min) => {
                if let CellValue::Number(n) = value {
                    if n < min {
                        messages.push(format!("Value must be >= {min}"));
                    }
                }
            }
            ValidationRule::Max(max) => {
                if let CellValue::Number(n) = value {
                    if n > max {
                        messages.push(format!("Value must be <= {max}"));
                    }
                }
            }
        }
    }
    ValidationResult {
        valid: messages.is_empty(),
        messages,
    }
}

// ── EditingState ────────────────────────────────────────────────

/// Tracks the currently editing cell.
#[derive(Debug, Clone, PartialEq)]
pub struct EditingState {
    pub row_id: String,
    pub column_id: String,
    pub original_value: CellValue,
    pub current_value: CellValue,
}

// ── CellCoord ───────────────────────────────────────────────────

/// A (row, column) coordinate for tab navigation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub row_id: String,
    pub column_id: String,
}

// ── CellEditorManager ───────────────────────────────────────────

/// Manages cell editing across the grid.
#[derive(Debug, Clone)]
pub struct CellEditorManager {
    /// Editor type per column.
    pub editors: HashMap<String, CellEditorType>,
    /// Validation rules per column.
    pub rules: HashMap<String, Vec<ValidationRule>>,
    /// Current editing state (at most one cell at a time).
    editing: Option<EditingState>,
    /// Set of cells that have been modified since last "save".
    dirty: HashSet<CellCoord>,
    /// Ordered list of editable column ids for tab navigation.
    pub editable_columns: Vec<String>,
    /// Ordered list of row ids for tab navigation.
    pub row_order: Vec<String>,
}

impl CellEditorManager {
    pub fn new() -> Self {
        Self {
            editors: HashMap::new(),
            rules: HashMap::new(),
            editing: None,
            dirty: HashSet::new(),
            editable_columns: Vec::new(),
            row_order: Vec::new(),
        }
    }

    /// Register an editor type for a column.
    pub fn set_editor(&mut self, column_id: impl Into<String>, editor: CellEditorType) {
        let col = column_id.into();
        self.editors.insert(col.clone(), editor);
        if !self.editable_columns.contains(&col) {
            self.editable_columns.push(col);
        }
    }

    /// Register validation rules for a column.
    pub fn set_rules(&mut self, column_id: impl Into<String>, rules: Vec<ValidationRule>) {
        self.rules.insert(column_id.into(), rules);
    }

    /// Start editing a cell.  Returns `false` if the column has no editor.
    pub fn start_editing(
        &mut self,
        row_id: impl Into<String>,
        column_id: impl Into<String>,
        current_value: CellValue,
    ) -> bool {
        let col = column_id.into();
        if !self.editors.contains_key(&col) {
            return false;
        }
        self.editing = Some(EditingState {
            row_id: row_id.into(),
            column_id: col,
            original_value: current_value.clone(),
            current_value,
        });
        true
    }

    /// Update the value being edited.
    pub fn update_value(&mut self, value: CellValue) {
        if let Some(state) = &mut self.editing {
            state.current_value = value;
        }
    }

    /// Commit the current edit.  Returns `Err` with validation messages if invalid.
    /// On success returns the committed `EditingState`.
    pub fn commit(&mut self) -> Result<EditingState, Vec<String>> {
        let state = match self.editing.take() {
            Some(s) => s,
            Option::None => return Err(vec!["No cell is being edited".into()]),
        };
        // Validate.
        if let Some(rules) = self.rules.get(&state.column_id) {
            let result = validate(&state.current_value, rules);
            if !result.valid {
                let msgs = result.messages;
                // Put it back so the user can fix.
                self.editing = Some(state);
                return Err(msgs);
            }
        }
        // Mark dirty if changed.
        if state.current_value != state.original_value {
            self.dirty.insert(CellCoord {
                row_id: state.row_id.clone(),
                column_id: state.column_id.clone(),
            });
        }
        Ok(state)
    }

    /// Cancel the current edit, reverting to original value.
    pub fn cancel(&mut self) -> Option<EditingState> {
        self.editing.take()
    }

    /// Get current editing state.
    pub fn editing_state(&self) -> Option<&EditingState> {
        self.editing.as_ref()
    }

    /// Is a specific cell dirty (modified)?
    pub fn is_dirty(&self, row_id: &str, column_id: &str) -> bool {
        self.dirty.contains(&CellCoord {
            row_id: row_id.to_string(),
            column_id: column_id.to_string(),
        })
    }

    /// Count of dirty cells.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Compute the next editable cell after the current one (tab navigation).
    /// Returns `None` if there are no editable columns or rows.
    pub fn next_cell(&self, row_id: &str, column_id: &str) -> Option<CellCoord> {
        if self.editable_columns.is_empty() || self.row_order.is_empty() {
            return None;
        }
        let col_idx = self.editable_columns.iter().position(|c| c == column_id).unwrap_or(0);
        let row_idx = self.row_order.iter().position(|r| r == row_id).unwrap_or(0);

        let next_col = col_idx + 1;
        if next_col < self.editable_columns.len() {
            Some(CellCoord {
                row_id: row_id.to_string(),
                column_id: self.editable_columns[next_col].clone(),
            })
        } else if row_idx + 1 < self.row_order.len() {
            Some(CellCoord {
                row_id: self.row_order[row_idx + 1].clone(),
                column_id: self.editable_columns[0].clone(),
            })
        } else {
            // Wrap around to first cell.
            Some(CellCoord {
                row_id: self.row_order[0].clone(),
                column_id: self.editable_columns[0].clone(),
            })
        }
    }

    /// Compute the previous editable cell (shift-tab).
    pub fn prev_cell(&self, row_id: &str, column_id: &str) -> Option<CellCoord> {
        if self.editable_columns.is_empty() || self.row_order.is_empty() {
            return None;
        }
        let col_idx = self.editable_columns.iter().position(|c| c == column_id).unwrap_or(0);
        let row_idx = self.row_order.iter().position(|r| r == row_id).unwrap_or(0);

        if col_idx > 0 {
            Some(CellCoord {
                row_id: row_id.to_string(),
                column_id: self.editable_columns[col_idx - 1].clone(),
            })
        } else if row_idx > 0 {
            Some(CellCoord {
                row_id: self.row_order[row_idx - 1].clone(),
                column_id: self.editable_columns.last().unwrap().clone(),
            })
        } else {
            Some(CellCoord {
                row_id: self.row_order.last().unwrap().clone(),
                column_id: self.editable_columns.last().unwrap().clone(),
            })
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> CellEditorManager {
        let mut mgr = CellEditorManager::new();
        mgr.set_editor("name", CellEditorType::Text);
        mgr.set_editor("age", CellEditorType::Number);
        mgr.set_editor("active", CellEditorType::Checkbox);
        mgr.row_order = vec!["r1".into(), "r2".into(), "r3".into()];
        mgr
    }

    #[test]
    fn start_editing_known_column() {
        let mut mgr = make_manager();
        assert!(mgr.start_editing("r1", "name", CellValue::Text("Alice".into())));
        assert!(mgr.editing_state().is_some());
    }

    #[test]
    fn start_editing_unknown_column_fails() {
        let mut mgr = make_manager();
        assert!(!mgr.start_editing("r1", "unknown", CellValue::Empty));
    }

    #[test]
    fn commit_marks_dirty() {
        let mut mgr = make_manager();
        mgr.start_editing("r1", "name", CellValue::Text("Alice".into()));
        mgr.update_value(CellValue::Text("Bob".into()));
        let result = mgr.commit();
        assert!(result.is_ok());
        assert!(mgr.is_dirty("r1", "name"));
        assert_eq!(mgr.dirty_count(), 1);
    }

    #[test]
    fn commit_unchanged_not_dirty() {
        let mut mgr = make_manager();
        mgr.start_editing("r1", "name", CellValue::Text("Alice".into()));
        let result = mgr.commit();
        assert!(result.is_ok());
        assert!(!mgr.is_dirty("r1", "name"));
    }

    #[test]
    fn cancel_reverts() {
        let mut mgr = make_manager();
        mgr.start_editing("r1", "name", CellValue::Text("Alice".into()));
        mgr.update_value(CellValue::Text("Bob".into()));
        let reverted = mgr.cancel();
        assert!(reverted.is_some());
        assert!(mgr.editing_state().is_none());
    }

    #[test]
    fn validation_required() {
        let result = validate(&CellValue::Empty, &[ValidationRule::Required]);
        assert!(!result.valid);
    }

    #[test]
    fn validation_min_max() {
        let rules = vec![ValidationRule::Min(0.0), ValidationRule::Max(100.0)];
        assert!(validate(&CellValue::Number(50.0), &rules).valid);
        assert!(!validate(&CellValue::Number(-1.0), &rules).valid);
        assert!(!validate(&CellValue::Number(101.0), &rules).valid);
    }

    #[test]
    fn validation_pattern() {
        let rules = vec![ValidationRule::Pattern("hello".into())];
        assert!(validate(&CellValue::Text("say hello".into()), &rules).valid);
        assert!(!validate(&CellValue::Text("goodbye".into()), &rules).valid);
    }

    #[test]
    fn tab_navigation_next() {
        let mgr = make_manager();
        let next = mgr.next_cell("r1", "name").unwrap();
        assert_eq!(next.column_id, "age");
        assert_eq!(next.row_id, "r1");
    }

    #[test]
    fn tab_navigation_wraps_to_next_row() {
        let mgr = make_manager();
        let next = mgr.next_cell("r1", "active").unwrap();
        assert_eq!(next.row_id, "r2");
        assert_eq!(next.column_id, "name");
    }

    #[test]
    fn tab_navigation_prev() {
        let mgr = make_manager();
        let prev = mgr.prev_cell("r2", "name").unwrap();
        assert_eq!(prev.row_id, "r1");
        assert_eq!(prev.column_id, "active");
    }

    #[test]
    fn clear_dirty() {
        let mut mgr = make_manager();
        mgr.start_editing("r1", "name", CellValue::Text("A".into()));
        mgr.update_value(CellValue::Text("B".into()));
        let _ = mgr.commit();
        assert_eq!(mgr.dirty_count(), 1);
        mgr.clear_dirty();
        assert_eq!(mgr.dirty_count(), 0);
    }

    #[test]
    fn commit_with_validation_failure() {
        let mut mgr = make_manager();
        mgr.set_rules("name", vec![ValidationRule::Required]);
        mgr.start_editing("r1", "name", CellValue::Text("Alice".into()));
        mgr.update_value(CellValue::Empty);
        let result = mgr.commit();
        assert!(result.is_err());
        // Still editing — user can fix.
        assert!(mgr.editing_state().is_some());
    }
}
