//! Data grid model: columns, rows, cell values, pagination, sort, filter, freeze.
//!
//! Replaces heavy JS data-grid widgets (AG Grid, Handsontable) with a pure-Rust
//! headless model.  The caller reads `visible_rows()` and renders however it wants.

use std::collections::HashMap;

// ── CellValue ───────────────────────────────────────────────────

/// A typed cell value inside a grid row.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Text(String),
    Number(f64),
    Bool(bool),
    /// ISO-8601 date string, e.g. `"2026-03-08"`.
    Date(String),
    Empty,
}

impl CellValue {
    /// Return a sortable string representation for comparisons.
    pub fn sort_key(&self) -> String {
        match self {
            CellValue::Text(s) => s.clone(),
            CellValue::Number(n) => format!("{n:020.6}"),
            CellValue::Bool(b) => if *b { "1".into() } else { "0".into() },
            CellValue::Date(d) => d.clone(),
            CellValue::Empty => String::new(),
        }
    }

    /// Loose text match for quick-filter purposes.
    pub fn contains_text(&self, needle: &str) -> bool {
        let lower = needle.to_lowercase();
        match self {
            CellValue::Text(s) => s.to_lowercase().contains(&lower),
            CellValue::Number(n) => n.to_string().contains(&lower),
            CellValue::Bool(b) => b.to_string().contains(&lower),
            CellValue::Date(d) => d.to_lowercase().contains(&lower),
            CellValue::Empty => false,
        }
    }
}

// ── GridColumn ──────────────────────────────────────────────────

/// Column definition for the data grid.
#[derive(Debug, Clone)]
pub struct GridColumn {
    pub id: String,
    pub header: String,
    pub field: String,
    pub width: f64,
    pub sortable: bool,
    pub filterable: bool,
    pub resizable: bool,
}

impl GridColumn {
    pub fn new(id: impl Into<String>, header: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            header: header.into(),
            field: field.into(),
            width: 150.0,
            sortable: true,
            filterable: true,
            resizable: true,
        }
    }
}

// ── GridRow ─────────────────────────────────────────────────────

/// A single row in the data grid.
#[derive(Debug, Clone)]
pub struct GridRow {
    pub id: String,
    pub cells: HashMap<String, CellValue>,
}

impl GridRow {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            cells: HashMap::new(),
        }
    }

    pub fn set(mut self, field: impl Into<String>, value: CellValue) -> Self {
        self.cells.insert(field.into(), value);
        self
    }

    pub fn get(&self, field: &str) -> &CellValue {
        self.cells.get(field).unwrap_or(&CellValue::Empty)
    }
}

// ── SortDirection ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Active sort specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridSort {
    pub field: String,
    pub direction: SortDirection,
}

// ── QuickFilter ─────────────────────────────────────────────────

/// Simple text filter applied across all columns.
#[derive(Debug, Clone, Default)]
pub struct QuickFilter {
    pub text: String,
}

// ── Pagination ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Pagination {
    pub page: usize,
    pub page_size: usize,
}

impl Default for Pagination {
    fn default() -> Self {
        Self { page: 0, page_size: 50 }
    }
}

// ── DataGrid ────────────────────────────────────────────────────

/// The top-level data grid model.
#[derive(Debug, Clone)]
pub struct DataGrid {
    pub columns: Vec<GridColumn>,
    pub rows: Vec<GridRow>,
    pub sort: Option<GridSort>,
    pub quick_filter: QuickFilter,
    pub pagination: Pagination,
    /// Number of columns frozen on the left side.
    pub frozen_columns: usize,
}

impl DataGrid {
    pub fn new(columns: Vec<GridColumn>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            sort: None,
            quick_filter: QuickFilter::default(),
            pagination: Pagination::default(),
            frozen_columns: 0,
        }
    }

    /// Add a row.
    pub fn add_row(&mut self, row: GridRow) {
        self.rows.push(row);
    }

    /// Set the quick-filter text (empty string clears).
    pub fn set_quick_filter(&mut self, text: impl Into<String>) {
        self.quick_filter.text = text.into();
        self.pagination.page = 0;
    }

    /// Set sort on a single column.
    pub fn set_sort(&mut self, field: impl Into<String>, direction: SortDirection) {
        self.sort = Some(GridSort { field: field.into(), direction });
    }

    /// Clear current sort.
    pub fn clear_sort(&mut self) {
        self.sort = None;
    }

    /// Freeze the left-most `n` columns.
    pub fn freeze_columns(&mut self, n: usize) {
        self.frozen_columns = n.min(self.columns.len());
    }

    /// Reorder a column from `from_index` to `to_index`.
    pub fn reorder_column(&mut self, from_index: usize, to_index: usize) {
        if from_index >= self.columns.len() || to_index >= self.columns.len() {
            return;
        }
        let col = self.columns.remove(from_index);
        self.columns.insert(to_index, col);
    }

    /// Total number of rows after filtering (before pagination).
    pub fn filtered_row_count(&self) -> usize {
        self.filtered_rows().len()
    }

    /// Total number of pages.
    pub fn total_pages(&self) -> usize {
        let count = self.filtered_row_count();
        if self.pagination.page_size == 0 {
            return 1;
        }
        (count + self.pagination.page_size - 1) / self.pagination.page_size
    }

    /// Navigate to a specific page (0-indexed).
    pub fn go_to_page(&mut self, page: usize) {
        let max = self.total_pages().saturating_sub(1);
        self.pagination.page = page.min(max);
    }

    /// Go to the next page if available.
    pub fn next_page(&mut self) {
        let max = self.total_pages().saturating_sub(1);
        if self.pagination.page < max {
            self.pagination.page += 1;
        }
    }

    /// Go to the previous page if available.
    pub fn prev_page(&mut self) {
        if self.pagination.page > 0 {
            self.pagination.page -= 1;
        }
    }

    /// Return the frozen (left-pinned) columns.
    pub fn frozen_cols(&self) -> &[GridColumn] {
        &self.columns[..self.frozen_columns.min(self.columns.len())]
    }

    /// Return the scrollable (non-frozen) columns.
    pub fn scrollable_cols(&self) -> &[GridColumn] {
        let start = self.frozen_columns.min(self.columns.len());
        &self.columns[start..]
    }

    // ── internal helpers ────────────────────────────────────────

    fn filtered_rows(&self) -> Vec<&GridRow> {
        let needle = &self.quick_filter.text;
        if needle.is_empty() {
            self.rows.iter().collect()
        } else {
            self.rows
                .iter()
                .filter(|row| {
                    row.cells.values().any(|cv| cv.contains_text(needle))
                })
                .collect()
        }
    }

    fn sorted_rows(&self) -> Vec<&GridRow> {
        let mut rows = self.filtered_rows();
        if let Some(sort) = &self.sort {
            rows.sort_by(|a, b| {
                let ka = a.get(&sort.field).sort_key();
                let kb = b.get(&sort.field).sort_key();
                match sort.direction {
                    SortDirection::Asc => ka.cmp(&kb),
                    SortDirection::Desc => kb.cmp(&ka),
                }
            });
        }
        rows
    }

    /// The rows visible on the current page after filter + sort + pagination.
    pub fn visible_rows(&self) -> Vec<&GridRow> {
        let sorted = self.sorted_rows();
        let ps = self.pagination.page_size;
        if ps == 0 {
            return sorted;
        }
        let start = self.pagination.page * ps;
        if start >= sorted.len() {
            return Vec::new();
        }
        let end = (start + ps).min(sorted.len());
        sorted[start..end].to_vec()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid() -> DataGrid {
        let cols = vec![
            GridColumn::new("c1", "Name", "name"),
            GridColumn::new("c2", "Age", "age"),
            GridColumn::new("c3", "Active", "active"),
        ];
        let mut grid = DataGrid::new(cols);
        grid.add_row(GridRow::new("r1")
            .set("name", CellValue::Text("Alice".into()))
            .set("age", CellValue::Number(30.0))
            .set("active", CellValue::Bool(true)));
        grid.add_row(GridRow::new("r2")
            .set("name", CellValue::Text("Bob".into()))
            .set("age", CellValue::Number(25.0))
            .set("active", CellValue::Bool(false)));
        grid.add_row(GridRow::new("r3")
            .set("name", CellValue::Text("Charlie".into()))
            .set("age", CellValue::Number(35.0))
            .set("active", CellValue::Bool(true)));
        grid.pagination.page_size = 10;
        grid
    }

    #[test]
    fn visible_rows_returns_all_when_no_filter() {
        let grid = sample_grid();
        assert_eq!(grid.visible_rows().len(), 3);
    }

    #[test]
    fn quick_filter_narrows_rows() {
        let mut grid = sample_grid();
        grid.set_quick_filter("bob");
        let vis = grid.visible_rows();
        assert_eq!(vis.len(), 1);
        assert_eq!(vis[0].id, "r2");
    }

    #[test]
    fn sort_asc() {
        let mut grid = sample_grid();
        grid.set_sort("name", SortDirection::Asc);
        let vis = grid.visible_rows();
        assert_eq!(vis[0].id, "r1"); // Alice
        assert_eq!(vis[2].id, "r3"); // Charlie
    }

    #[test]
    fn sort_desc() {
        let mut grid = sample_grid();
        grid.set_sort("name", SortDirection::Desc);
        let vis = grid.visible_rows();
        assert_eq!(vis[0].id, "r3"); // Charlie
    }

    #[test]
    fn pagination_basics() {
        let mut grid = sample_grid();
        grid.pagination.page_size = 2;
        assert_eq!(grid.total_pages(), 2);
        assert_eq!(grid.visible_rows().len(), 2);
        grid.next_page();
        assert_eq!(grid.pagination.page, 1);
        assert_eq!(grid.visible_rows().len(), 1);
        grid.next_page(); // should not exceed
        assert_eq!(grid.pagination.page, 1);
        grid.prev_page();
        assert_eq!(grid.pagination.page, 0);
    }

    #[test]
    fn go_to_page_clamps() {
        let mut grid = sample_grid();
        grid.pagination.page_size = 2;
        grid.go_to_page(100);
        assert_eq!(grid.pagination.page, 1);
    }

    #[test]
    fn freeze_columns() {
        let mut grid = sample_grid();
        grid.freeze_columns(1);
        assert_eq!(grid.frozen_cols().len(), 1);
        assert_eq!(grid.scrollable_cols().len(), 2);
    }

    #[test]
    fn reorder_column() {
        let mut grid = sample_grid();
        grid.reorder_column(0, 2);
        assert_eq!(grid.columns[0].id, "c2");
        assert_eq!(grid.columns[2].id, "c1");
    }

    #[test]
    fn cell_value_sort_keys() {
        assert!(CellValue::Number(5.0).sort_key() < CellValue::Number(10.0).sort_key());
        assert_eq!(CellValue::Bool(true).sort_key(), "1");
        assert_eq!(CellValue::Empty.sort_key(), "");
    }

    #[test]
    fn cell_value_contains_text() {
        assert!(CellValue::Text("Hello World".into()).contains_text("hello"));
        assert!(CellValue::Number(42.0).contains_text("42"));
        assert!(!CellValue::Empty.contains_text("x"));
    }

    #[test]
    fn filtered_row_count() {
        let mut grid = sample_grid();
        assert_eq!(grid.filtered_row_count(), 3);
        grid.set_quick_filter("alice");
        assert_eq!(grid.filtered_row_count(), 1);
    }

    #[test]
    fn freeze_clamps_to_column_count() {
        let mut grid = sample_grid();
        grid.freeze_columns(999);
        assert_eq!(grid.frozen_columns, 3);
    }
}
