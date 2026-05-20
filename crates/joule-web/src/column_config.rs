//! Column configuration: definitions, runtime state, resize, auto-size,
//! show/hide, pinning, grouping, and state persistence.
//!
//! Replaces column-config layers of AG Grid / TanStack with pure Rust.

use std::collections::HashMap;

// ── Pin position ────────────────────────────────────────────────

/// Where a column is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinPosition {
    Left,
    Right,
    None,
}

impl Default for PinPosition {
    fn default() -> Self { PinPosition::None }
}

// ── ColumnDef ───────────────────────────────────────────────────

/// Persistent column definition.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub field: String,
    pub header: String,
    pub width: f64,
    pub min_width: f64,
    pub max_width: f64,
    pub hidden: bool,
    pub pinned: PinPosition,
    /// Flex factor for distributing remaining space.  0 = fixed width.
    pub flex: f64,
}

impl ColumnDef {
    pub fn new(field: impl Into<String>, header: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            header: header.into(),
            width: 150.0,
            min_width: 50.0,
            max_width: 800.0,
            hidden: false,
            pinned: PinPosition::None,
            flex: 0.0,
        }
    }
}

// ── ColumnState (runtime) ───────────────────────────────────────

/// Runtime state for a single column.
#[derive(Debug, Clone)]
pub struct ColumnState {
    pub field: String,
    pub width: f64,
    pub hidden: bool,
    pub pinned: PinPosition,
    pub order: usize,
}

// ── ColumnGroup ─────────────────────────────────────────────────

/// A group header that spans multiple child columns.
#[derive(Debug, Clone)]
pub struct ColumnGroup {
    pub header: String,
    pub children: Vec<String>, // field names
}

// ── ColumnConfig ────────────────────────────────────────────────

/// Manages column definitions and their runtime state.
#[derive(Debug, Clone)]
pub struct ColumnConfig {
    pub defs: Vec<ColumnDef>,
    pub groups: Vec<ColumnGroup>,
    /// Runtime widths (field → width).  Lazily initialised from defs.
    runtime_widths: HashMap<String, f64>,
}

impl ColumnConfig {
    pub fn new(defs: Vec<ColumnDef>) -> Self {
        let runtime_widths: HashMap<String, f64> = defs
            .iter()
            .map(|d| (d.field.clone(), d.width))
            .collect();
        Self { defs, groups: Vec::new(), runtime_widths }
    }

    /// Get the current runtime width of a column.
    pub fn width(&self, field: &str) -> Option<f64> {
        self.runtime_widths.get(field).copied()
    }

    /// Resize a column, clamping to its min/max bounds.
    pub fn resize(&mut self, field: &str, new_width: f64) {
        if let Some(def) = self.defs.iter().find(|d| d.field == field) {
            let clamped = new_width.clamp(def.min_width, def.max_width);
            self.runtime_widths.insert(field.to_string(), clamped);
        }
    }

    /// Auto-size a column to the widest content value.
    /// `content_widths` provides the measured width for each row in that column.
    pub fn auto_size(&mut self, field: &str, content_widths: &[f64]) {
        let max_content = content_widths.iter().cloned().fold(0.0_f64, f64::max);
        self.resize(field, max_content);
    }

    /// Hide a column.
    pub fn hide(&mut self, field: &str) {
        if let Some(def) = self.defs.iter_mut().find(|d| d.field == field) {
            def.hidden = true;
        }
    }

    /// Show a previously hidden column.
    pub fn show(&mut self, field: &str) {
        if let Some(def) = self.defs.iter_mut().find(|d| d.field == field) {
            def.hidden = false;
        }
    }

    /// Toggle column visibility.
    pub fn toggle_visibility(&mut self, field: &str) {
        if let Some(def) = self.defs.iter_mut().find(|d| d.field == field) {
            def.hidden = !def.hidden;
        }
    }

    /// Pin a column.
    pub fn pin(&mut self, field: &str, position: PinPosition) {
        if let Some(def) = self.defs.iter_mut().find(|d| d.field == field) {
            def.pinned = position;
        }
    }

    /// Return visible (non-hidden) column definitions in order.
    pub fn visible_columns(&self) -> Vec<&ColumnDef> {
        self.defs.iter().filter(|d| !d.hidden).collect()
    }

    /// Add a column group.
    pub fn add_group(&mut self, header: impl Into<String>, children: Vec<String>) {
        self.groups.push(ColumnGroup { header: header.into(), children });
    }

    /// Columns belonging to a named group.
    pub fn group_columns(&self, header: &str) -> Vec<&ColumnDef> {
        if let Some(group) = self.groups.iter().find(|g| g.header == header) {
            self.defs
                .iter()
                .filter(|d| group.children.contains(&d.field))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Save state to a serialisable snapshot.
    pub fn save_state(&self) -> Vec<ColumnState> {
        self.defs
            .iter()
            .enumerate()
            .map(|(i, d)| ColumnState {
                field: d.field.clone(),
                width: self.runtime_widths.get(&d.field).copied().unwrap_or(d.width),
                hidden: d.hidden,
                pinned: d.pinned,
                order: i,
            })
            .collect()
    }

    /// Restore from a previously saved state snapshot.
    pub fn restore_state(&mut self, states: &[ColumnState]) {
        let field_order: HashMap<&str, &ColumnState> = states
            .iter()
            .map(|s| (s.field.as_str(), s))
            .collect();

        // Apply individual states.
        for def in &mut self.defs {
            if let Some(st) = field_order.get(def.field.as_str()) {
                def.hidden = st.hidden;
                def.pinned = st.pinned;
                self.runtime_widths.insert(def.field.clone(), st.width);
            }
        }

        // Reorder by saved order.
        let mut ordered: Vec<(usize, ColumnDef)> = self.defs
            .drain(..)
            .map(|d| {
                let order = field_order.get(d.field.as_str()).map(|s| s.order).unwrap_or(usize::MAX);
                (order, d)
            })
            .collect();
        ordered.sort_by_key(|(o, _)| *o);
        self.defs = ordered.into_iter().map(|(_, d)| d).collect();
    }

    /// Distribute remaining space among flex columns.
    pub fn apply_flex(&mut self, total_width: f64) {
        let fixed_total: f64 = self.defs.iter()
            .filter(|d| !d.hidden && d.flex == 0.0)
            .map(|d| self.runtime_widths.get(&d.field).copied().unwrap_or(d.width))
            .sum();
        let remaining = (total_width - fixed_total).max(0.0);
        let flex_sum: f64 = self.defs.iter()
            .filter(|d| !d.hidden && d.flex > 0.0)
            .map(|d| d.flex)
            .sum();
        if flex_sum <= 0.0 {
            return;
        }
        let flex_fields: Vec<(String, f64)> = self.defs.iter()
            .filter(|d| !d.hidden && d.flex > 0.0)
            .map(|d| (d.field.clone(), d.flex))
            .collect();
        for (field, flex) in flex_fields {
            let w = remaining * (flex / flex_sum);
            self.resize(&field, w);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ColumnConfig {
        ColumnConfig::new(vec![
            ColumnDef::new("name", "Name"),
            ColumnDef::new("age", "Age"),
            ColumnDef::new("email", "Email"),
        ])
    }

    #[test]
    fn initial_widths() {
        let cfg = make_config();
        assert_eq!(cfg.width("name"), Some(150.0));
    }

    #[test]
    fn resize_clamps_min() {
        let mut cfg = make_config();
        cfg.resize("name", 10.0);
        assert_eq!(cfg.width("name"), Some(50.0));
    }

    #[test]
    fn resize_clamps_max() {
        let mut cfg = make_config();
        cfg.resize("name", 9000.0);
        assert_eq!(cfg.width("name"), Some(800.0));
    }

    #[test]
    fn auto_size_sets_max_content() {
        let mut cfg = make_config();
        cfg.auto_size("name", &[50.0, 120.0, 90.0]);
        assert_eq!(cfg.width("name"), Some(120.0));
    }

    #[test]
    fn hide_show_toggle() {
        let mut cfg = make_config();
        assert_eq!(cfg.visible_columns().len(), 3);
        cfg.hide("age");
        assert_eq!(cfg.visible_columns().len(), 2);
        cfg.show("age");
        assert_eq!(cfg.visible_columns().len(), 3);
        cfg.toggle_visibility("email");
        assert_eq!(cfg.visible_columns().len(), 2);
    }

    #[test]
    fn pin_column() {
        let mut cfg = make_config();
        cfg.pin("name", PinPosition::Left);
        assert_eq!(cfg.defs[0].pinned, PinPosition::Left);
    }

    #[test]
    fn column_groups() {
        let mut cfg = make_config();
        cfg.add_group("Personal", vec!["name".into(), "age".into()]);
        let cols = cfg.group_columns("Personal");
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn save_restore_state() {
        let mut cfg = make_config();
        cfg.hide("age");
        cfg.resize("name", 200.0);
        let state = cfg.save_state();

        let mut cfg2 = make_config();
        cfg2.restore_state(&state);
        assert!(cfg2.defs.iter().find(|d| d.field == "age").unwrap().hidden);
        assert_eq!(cfg2.width("name"), Some(200.0));
    }

    #[test]
    fn restore_preserves_order() {
        let mut cfg = make_config();
        let mut state = cfg.save_state();
        state.reverse();
        for (i, s) in state.iter_mut().enumerate() {
            s.order = i;
        }
        cfg.restore_state(&state);
        assert_eq!(cfg.defs[0].field, "email");
        assert_eq!(cfg.defs[2].field, "name");
    }

    #[test]
    fn flex_distributes_space() {
        let mut defs = vec![
            ColumnDef::new("a", "A"),
            ColumnDef::new("b", "B"),
        ];
        defs[0].width = 100.0;
        defs[0].flex = 0.0;
        defs[1].flex = 1.0;
        let mut cfg = ColumnConfig::new(defs);
        cfg.apply_flex(500.0);
        // b should get 400
        assert!((cfg.width("b").unwrap() - 400.0).abs() < 1.0);
    }

    #[test]
    fn visible_columns_excludes_hidden() {
        let mut cfg = make_config();
        cfg.hide("name");
        cfg.hide("email");
        assert_eq!(cfg.visible_columns().len(), 1);
        assert_eq!(cfg.visible_columns()[0].field, "age");
    }
}
