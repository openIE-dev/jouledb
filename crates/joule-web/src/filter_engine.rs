//! Data filtering engine: operators, groups (AND/OR), quick filter,
//! column-specific filters, presets.
//!
//! Replaces AG Grid's filter layer with pure Rust logic.

use std::collections::HashMap;

// ── FilterOp ────────────────────────────────────────────────────

/// A filter comparison operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterOp {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    /// Inclusive range.
    Between,
    /// Value is one of a set.
    In,
    IsEmpty,
    IsNotEmpty,
    /// Pattern match (simplified: treated as "contains" since we have no regex crate).
    Regex,
}

// ── Filter ──────────────────────────────────────────────────────

/// A single filter condition on a field.
#[derive(Debug, Clone)]
pub struct Filter {
    pub field: String,
    pub op: FilterOp,
    /// Primary filter value.  For `Between`, use `value` as low and `value2` as high.
    /// For `In`, values are comma-separated in `value`.
    pub value: String,
    /// Secondary value for `Between`.
    pub value2: Option<String>,
}

impl Filter {
    pub fn new(field: impl Into<String>, op: FilterOp, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            op,
            value: value.into(),
            value2: None,
        }
    }

    pub fn between(
        field: impl Into<String>,
        low: impl Into<String>,
        high: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            op: FilterOp::Between,
            value: low.into(),
            value2: Some(high.into()),
        }
    }

    /// Test whether a cell value passes this filter.
    pub fn matches(&self, cell: &str) -> bool {
        let cell_lower = cell.to_lowercase();
        let val_lower = self.value.to_lowercase();

        match &self.op {
            FilterOp::Equals => cell_lower == val_lower,
            FilterOp::NotEquals => cell_lower != val_lower,
            FilterOp::Contains | FilterOp::Regex => cell_lower.contains(&val_lower),
            FilterOp::StartsWith => cell_lower.starts_with(&val_lower),
            FilterOp::EndsWith => cell_lower.ends_with(&val_lower),
            FilterOp::GreaterThan => {
                compare_numeric_or_string(cell, &self.value)
                    .map(|ord| ord == std::cmp::Ordering::Greater)
                    .unwrap_or(false)
            }
            FilterOp::LessThan => {
                compare_numeric_or_string(cell, &self.value)
                    .map(|ord| ord == std::cmp::Ordering::Less)
                    .unwrap_or(false)
            }
            FilterOp::Between => {
                let high = self.value2.as_deref().unwrap_or("");
                let ge = compare_numeric_or_string(cell, &self.value)
                    .map(|o| o != std::cmp::Ordering::Less)
                    .unwrap_or(false);
                let le = compare_numeric_or_string(cell, high)
                    .map(|o| o != std::cmp::Ordering::Greater)
                    .unwrap_or(false);
                ge && le
            }
            FilterOp::In => {
                let set: Vec<String> = self.value.split(',')
                    .map(|s| s.trim().to_lowercase())
                    .collect();
                set.contains(&cell_lower)
            }
            FilterOp::IsEmpty => cell.is_empty(),
            FilterOp::IsNotEmpty => !cell.is_empty(),
        }
    }
}

/// Compare two values: try numeric first, fall back to string.
fn compare_numeric_or_string(a: &str, b: &str) -> Option<std::cmp::Ordering> {
    if let (Ok(na), Ok(nb)) = (a.parse::<f64>(), b.parse::<f64>()) {
        na.partial_cmp(&nb)
    } else {
        Some(a.to_lowercase().cmp(&b.to_lowercase()))
    }
}

// ── Combinator ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    And,
    Or,
}

// ── FilterGroup ─────────────────────────────────────────────────

/// A group of filters combined with AND or OR.
#[derive(Debug, Clone)]
pub struct FilterGroup {
    pub filters: Vec<Filter>,
    pub combinator: Combinator,
}

impl FilterGroup {
    pub fn and(filters: Vec<Filter>) -> Self {
        Self { filters, combinator: Combinator::And }
    }

    pub fn or(filters: Vec<Filter>) -> Self {
        Self { filters, combinator: Combinator::Or }
    }

    /// Test whether a data row matches this group.
    /// `row` maps field → cell string value.
    pub fn matches(&self, row: &HashMap<String, String>) -> bool {
        if self.filters.is_empty() {
            return true;
        }
        match self.combinator {
            Combinator::And => self.filters.iter().all(|f| {
                let cell = row.get(&f.field).map(|s| s.as_str()).unwrap_or("");
                f.matches(cell)
            }),
            Combinator::Or => self.filters.iter().any(|f| {
                let cell = row.get(&f.field).map(|s| s.as_str()).unwrap_or("");
                f.matches(cell)
            }),
        }
    }
}

// ── FilterEngine ────────────────────────────────────────────────

/// Manages active filters, quick filter, and presets.
#[derive(Debug, Clone)]
pub struct FilterEngine {
    /// Active filter group.
    pub group: FilterGroup,
    /// Quick filter text (searches all columns).
    pub quick_filter: String,
    /// Column-specific filters.
    pub column_filters: HashMap<String, Filter>,
    /// Named presets.
    pub presets: HashMap<String, FilterGroup>,
}

impl FilterEngine {
    pub fn new() -> Self {
        Self {
            group: FilterGroup::and(Vec::new()),
            quick_filter: String::new(),
            column_filters: HashMap::new(),
            presets: HashMap::new(),
        }
    }

    /// Set the main filter group.
    pub fn set_group(&mut self, group: FilterGroup) {
        self.group = group;
    }

    /// Set a quick-filter text (empty clears).
    pub fn set_quick_filter(&mut self, text: impl Into<String>) {
        self.quick_filter = text.into();
    }

    /// Set a column-specific filter.
    pub fn set_column_filter(&mut self, filter: Filter) {
        self.column_filters.insert(filter.field.clone(), filter);
    }

    /// Remove a column-specific filter.
    pub fn remove_column_filter(&mut self, field: &str) {
        self.column_filters.remove(field);
    }

    /// Clear all filters.
    pub fn clear(&mut self) {
        self.group = FilterGroup::and(Vec::new());
        self.quick_filter.clear();
        self.column_filters.clear();
    }

    /// Save current group as a named preset.
    pub fn save_preset(&mut self, name: impl Into<String>) {
        self.presets.insert(name.into(), self.group.clone());
    }

    /// Load a named preset (replaces current group).
    pub fn load_preset(&mut self, name: &str) -> bool {
        if let Some(group) = self.presets.get(name).cloned() {
            self.group = group;
            true
        } else {
            false
        }
    }

    /// Apply all filters to a set of rows.  Returns indices of matching rows.
    pub fn apply(&self, rows: &[HashMap<String, String>]) -> Vec<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, row)| self.matches_row(row))
            .map(|(i, _)| i)
            .collect()
    }

    /// Test a single row against all active filters.
    pub fn matches_row(&self, row: &HashMap<String, String>) -> bool {
        // Main group.
        if !self.group.matches(row) {
            return false;
        }
        // Column-specific filters (AND).
        for filter in self.column_filters.values() {
            let cell = row.get(&filter.field).map(|s| s.as_str()).unwrap_or("");
            if !filter.matches(cell) {
                return false;
            }
        }
        // Quick filter: any column contains the text.
        if !self.quick_filter.is_empty() {
            let lower = self.quick_filter.to_lowercase();
            if !row.values().any(|v| v.to_lowercase().contains(&lower)) {
                return false;
            }
        }
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, age: &str) -> HashMap<String, String> {
        HashMap::from([
            ("name".into(), name.into()),
            ("age".into(), age.into()),
        ])
    }

    #[test]
    fn equals_filter() {
        let f = Filter::new("name", FilterOp::Equals, "Alice");
        assert!(f.matches("alice"));
        assert!(!f.matches("Bob"));
    }

    #[test]
    fn not_equals_filter() {
        let f = Filter::new("name", FilterOp::NotEquals, "Alice");
        assert!(!f.matches("Alice"));
        assert!(f.matches("Bob"));
    }

    #[test]
    fn contains_filter() {
        let f = Filter::new("name", FilterOp::Contains, "lic");
        assert!(f.matches("Alice"));
        assert!(!f.matches("Bob"));
    }

    #[test]
    fn starts_ends_with() {
        let sw = Filter::new("name", FilterOp::StartsWith, "Al");
        assert!(sw.matches("Alice"));
        let ew = Filter::new("name", FilterOp::EndsWith, "ce");
        assert!(ew.matches("Alice"));
    }

    #[test]
    fn greater_less_than() {
        let gt = Filter::new("age", FilterOp::GreaterThan, "25");
        assert!(gt.matches("30"));
        assert!(!gt.matches("20"));
        let lt = Filter::new("age", FilterOp::LessThan, "25");
        assert!(lt.matches("20"));
    }

    #[test]
    fn between_filter() {
        let f = Filter::between("age", "20", "30");
        assert!(f.matches("25"));
        assert!(f.matches("20"));
        assert!(f.matches("30"));
        assert!(!f.matches("31"));
    }

    #[test]
    fn in_filter() {
        let f = Filter::new("name", FilterOp::In, "alice, bob");
        assert!(f.matches("Alice"));
        assert!(f.matches("Bob"));
        assert!(!f.matches("Charlie"));
    }

    #[test]
    fn is_empty_not_empty() {
        let ie = Filter::new("name", FilterOp::IsEmpty, "");
        assert!(ie.matches(""));
        assert!(!ie.matches("x"));
        let ine = Filter::new("name", FilterOp::IsNotEmpty, "");
        assert!(ine.matches("x"));
        assert!(!ine.matches(""));
    }

    #[test]
    fn filter_group_and() {
        let group = FilterGroup::and(vec![
            Filter::new("name", FilterOp::Contains, "a"),
            Filter::new("age", FilterOp::GreaterThan, "20"),
        ]);
        assert!(group.matches(&row("Sarah", "30")));
        assert!(!group.matches(&row("Bob", "30")));
    }

    #[test]
    fn filter_group_or() {
        let group = FilterGroup::or(vec![
            Filter::new("name", FilterOp::Equals, "Alice"),
            Filter::new("name", FilterOp::Equals, "Bob"),
        ]);
        assert!(group.matches(&row("Alice", "30")));
        assert!(group.matches(&row("Bob", "25")));
        assert!(!group.matches(&row("Charlie", "35")));
    }

    #[test]
    fn engine_apply() {
        let rows = vec![
            row("Alice", "30"),
            row("Bob", "25"),
            row("Charlie", "35"),
        ];
        let mut engine = FilterEngine::new();
        engine.set_group(FilterGroup::and(vec![
            Filter::new("age", FilterOp::GreaterThan, "26"),
        ]));
        let indices = engine.apply(&rows);
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn quick_filter() {
        let rows = vec![row("Alice", "30"), row("Bob", "25")];
        let mut engine = FilterEngine::new();
        engine.set_quick_filter("bob");
        let indices = engine.apply(&rows);
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn column_specific_filter() {
        let rows = vec![row("Alice", "30"), row("Bob", "25")];
        let mut engine = FilterEngine::new();
        engine.set_column_filter(Filter::new("name", FilterOp::Equals, "Alice"));
        let indices = engine.apply(&rows);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn preset_save_load() {
        let mut engine = FilterEngine::new();
        engine.set_group(FilterGroup::and(vec![
            Filter::new("age", FilterOp::GreaterThan, "30"),
        ]));
        engine.save_preset("over30");
        engine.clear();
        assert!(engine.group.filters.is_empty());
        assert!(engine.load_preset("over30"));
        assert_eq!(engine.group.filters.len(), 1);
    }

    #[test]
    fn preset_load_missing_returns_false() {
        let mut engine = FilterEngine::new();
        assert!(!engine.load_preset("nope"));
    }

    #[test]
    fn regex_as_contains() {
        let f = Filter::new("name", FilterOp::Regex, "ali");
        assert!(f.matches("Alice"));
    }
}
