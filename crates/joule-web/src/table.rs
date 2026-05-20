//! Headless data table engine: sort, filter, paginate, column management.
//!
//! Replaces AG Grid, TanStack Table with pure-Rust logic.
//! No DOM — the caller renders based on `visible_rows()` output.

use std::collections::{HashMap, HashSet};

// ── Row type ─────────────────────────────────────────────────────

/// A table row is a map of column accessor → value.
pub type Row = HashMap<String, String>;

// ── ColumnDef ────────────────────────────────────────────────────

/// Column definition.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub id: String,
    pub header: String,
    pub sortable: bool,
    pub filterable: bool,
    pub width: f64,
    pub min_width: f64,
    pub visible: bool,
    pub accessor: String,
}

// ── Sort ─────────────────────────────────────────────────────────

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Active sort on a column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortState {
    pub column_id: String,
    pub direction: SortDirection,
}

// ── Filter ───────────────────────────────────────────────────────

/// Filter operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    NotEq,
    Contains,
    StartsWith,
    EndsWith,
    Gt,
    Lt,
    Gte,
    Lte,
    IsEmpty,
    IsNotEmpty,
}

/// A filter applied to a column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnFilter {
    pub column_id: String,
    pub op: FilterOp,
    pub value: String,
}

// ── PageState ────────────────────────────────────────────────────

/// Pagination state.
#[derive(Debug, Clone)]
pub struct PageState {
    pub page: usize,
    pub page_size: usize,
    pub total_rows: usize,
}

impl PageState {
    pub fn total_pages(&self) -> usize {
        if self.page_size == 0 {
            return 0;
        }
        (self.total_rows + self.page_size - 1) / self.page_size
    }

    pub fn has_next(&self) -> bool {
        self.page + 1 < self.total_pages()
    }

    pub fn has_previous(&self) -> bool {
        self.page > 0
    }

    pub fn offset(&self) -> usize {
        self.page * self.page_size
    }
}

// ── TableState ───────────────────────────────────────────────────

/// Full table state: columns, rows, sort, filter, pagination, selection.
pub struct TableState {
    pub columns: Vec<ColumnDef>,
    pub rows: Vec<Row>,
    pub sort: Vec<SortState>,
    pub filters: Vec<ColumnFilter>,
    pub page_state: PageState,
    pub selected_rows: HashSet<usize>,
}

impl TableState {
    pub fn new(columns: Vec<ColumnDef>, rows: Vec<Row>) -> Self {
        let total = rows.len();
        Self {
            columns,
            rows,
            sort: Vec::new(),
            filters: Vec::new(),
            page_state: PageState {
                page: 0,
                page_size: 25,
                total_rows: total,
            },
            selected_rows: HashSet::new(),
        }
    }

    // ── Sort ─────────────────────────────────────────────────────

    /// Set sort on a column (replaces any existing sort on that column).
    pub fn set_sort(&mut self, column_id: &str, direction: SortDirection) {
        self.sort.retain(|s| s.column_id != column_id);
        self.sort.push(SortState {
            column_id: column_id.to_string(),
            direction,
        });
    }

    /// Toggle sort: None → Asc → Desc → None.
    pub fn toggle_sort(&mut self, column_id: &str) {
        let existing = self.sort.iter().position(|s| s.column_id == column_id);
        match existing {
            None => self.set_sort(column_id, SortDirection::Asc),
            Some(idx) => {
                if self.sort[idx].direction == SortDirection::Asc {
                    self.sort[idx].direction = SortDirection::Desc;
                } else {
                    self.sort.remove(idx);
                }
            }
        }
    }

    // ── Filter ───────────────────────────────────────────────────

    pub fn add_filter(&mut self, filter: ColumnFilter) {
        // Replace existing filter on same column
        self.filters.retain(|f| f.column_id != filter.column_id);
        self.filters.push(filter);
    }

    pub fn remove_filter(&mut self, column_id: &str) {
        self.filters.retain(|f| f.column_id != column_id);
    }

    pub fn clear_filters(&mut self) {
        self.filters.clear();
    }

    // ── Pagination ───────────────────────────────────────────────

    pub fn set_page(&mut self, page: usize) {
        self.page_state.page = page;
    }

    pub fn set_page_size(&mut self, size: usize) {
        self.page_state.page_size = size;
        self.page_state.page = 0; // reset to first page
    }

    pub fn next_page(&mut self) {
        if self.page_state.has_next() {
            self.page_state.page += 1;
        }
    }

    pub fn previous_page(&mut self) {
        if self.page_state.has_previous() {
            self.page_state.page -= 1;
        }
    }

    // ── Selection ────────────────────────────────────────────────

    pub fn select_row(&mut self, index: usize) {
        self.selected_rows.insert(index);
    }

    pub fn deselect_row(&mut self, index: usize) {
        self.selected_rows.remove(&index);
    }

    pub fn select_all(&mut self) {
        for i in 0..self.rows.len() {
            self.selected_rows.insert(i);
        }
    }

    pub fn deselect_all(&mut self) {
        self.selected_rows.clear();
    }

    pub fn selected_count(&self) -> usize {
        self.selected_rows.len()
    }

    // ── Column management ────────────────────────────────────────

    pub fn resize_column(&mut self, column_id: &str, width: f64) {
        if let Some(col) = self.columns.iter_mut().find(|c| c.id == column_id) {
            col.width = width.max(col.min_width);
        }
    }

    pub fn toggle_column_visibility(&mut self, column_id: &str) {
        if let Some(col) = self.columns.iter_mut().find(|c| c.id == column_id) {
            col.visible = !col.visible;
        }
    }

    pub fn reorder_columns(&mut self, from: usize, to: usize) {
        if from < self.columns.len() && to < self.columns.len() && from != to {
            let col = self.columns.remove(from);
            self.columns.insert(to, col);
        }
    }

    // ── Core query ───────────────────────────────────────────────

    /// Apply filters → sort → paginate. Returns the current page of rows.
    pub fn visible_rows(&self) -> Vec<&Row> {
        // 1. Filter
        let mut indices: Vec<usize> = (0..self.rows.len())
            .filter(|i| self.row_passes_filters(&self.rows[*i]))
            .collect();

        // Update total for pagination awareness (non-mutable, so we
        // compute total pages from filtered count).
        let filtered_total = indices.len();

        // 2. Sort
        if !self.sort.is_empty() {
            indices.sort_by(|&a, &b| self.compare_rows(&self.rows[a], &self.rows[b]));
        }

        // 3. Paginate
        let offset = self.page_state.page * self.page_state.page_size;
        let end = (offset + self.page_state.page_size).min(filtered_total);
        if offset >= filtered_total {
            return Vec::new();
        }

        indices[offset..end]
            .iter()
            .map(|i| &self.rows[*i])
            .collect()
    }

    /// Number of rows after filters applied (before pagination).
    pub fn filtered_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| self.row_passes_filters(r))
            .count()
    }

    /// Export visible columns/filtered rows as CSV.
    pub fn export_csv(&self) -> String {
        let vis_cols: Vec<&ColumnDef> = self.columns.iter().filter(|c| c.visible).collect();
        let mut out = String::new();

        // Header
        let headers: Vec<&str> = vis_cols.iter().map(|c| c.header.as_str()).collect();
        out.push_str(&csv_escape_row(&headers));
        out.push('\n');

        // Rows (filtered, sorted — all pages)
        let mut indices: Vec<usize> = (0..self.rows.len())
            .filter(|i| self.row_passes_filters(&self.rows[*i]))
            .collect();
        if !self.sort.is_empty() {
            indices.sort_by(|&a, &b| self.compare_rows(&self.rows[a], &self.rows[b]));
        }

        for &i in &indices {
            let vals: Vec<&str> = vis_cols
                .iter()
                .map(|c| {
                    self.rows[i]
                        .get(&c.accessor)
                        .map(|s| s.as_str())
                        .unwrap_or("")
                })
                .collect();
            out.push_str(&csv_escape_row(&vals));
            out.push('\n');
        }

        out
    }

    // ── helpers ──────────────────────────────────────────────────

    fn row_passes_filters(&self, row: &Row) -> bool {
        self.filters.iter().all(|f| {
            let accessor = self
                .columns
                .iter()
                .find(|c| c.id == f.column_id)
                .map(|c| c.accessor.as_str())
                .unwrap_or(&f.column_id);
            let val = row.get(accessor).map(|s| s.as_str()).unwrap_or("");
            match f.op {
                FilterOp::Eq => val == f.value,
                FilterOp::NotEq => val != f.value,
                FilterOp::Contains => val.contains(&f.value),
                FilterOp::StartsWith => val.starts_with(&f.value),
                FilterOp::EndsWith => val.ends_with(&f.value),
                FilterOp::Gt => parse_f64(val) > parse_f64(&f.value),
                FilterOp::Lt => parse_f64(val) < parse_f64(&f.value),
                FilterOp::Gte => parse_f64(val) >= parse_f64(&f.value),
                FilterOp::Lte => parse_f64(val) <= parse_f64(&f.value),
                FilterOp::IsEmpty => val.is_empty(),
                FilterOp::IsNotEmpty => !val.is_empty(),
            }
        })
    }

    fn compare_rows(&self, a: &Row, b: &Row) -> std::cmp::Ordering {
        for s in &self.sort {
            let accessor = self
                .columns
                .iter()
                .find(|c| c.id == s.column_id)
                .map(|c| c.accessor.as_str())
                .unwrap_or(&s.column_id);
            let va = a.get(accessor).map(|s| s.as_str()).unwrap_or("");
            let vb = b.get(accessor).map(|s| s.as_str()).unwrap_or("");

            // Try numeric comparison first
            let ord = match (va.parse::<f64>(), vb.parse::<f64>()) {
                (Ok(na), Ok(nb)) => na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal),
                _ => va.cmp(vb),
            };

            let ord = match s.direction {
                SortDirection::Asc => ord,
                SortDirection::Desc => ord.reverse(),
            };

            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    }
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

fn csv_escape_row(fields: &[&str]) -> String {
    fields
        .iter()
        .map(|f| {
            if f.contains(',') || f.contains('"') || f.contains('\n') {
                format!("\"{}\"", f.replace('"', "\"\""))
            } else {
                f.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn col(id: &str, header: &str) -> ColumnDef {
        ColumnDef {
            id: id.to_string(),
            header: header.to_string(),
            sortable: true,
            filterable: true,
            width: 100.0,
            min_width: 50.0,
            visible: true,
            accessor: id.to_string(),
        }
    }

    fn row(pairs: &[(&str, &str)]) -> Row {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn sample_table() -> TableState {
        let cols = vec![col("name", "Name"), col("age", "Age"), col("city", "City")];
        let rows = vec![
            row(&[("name", "Alice"), ("age", "30"), ("city", "NYC")]),
            row(&[("name", "Bob"), ("age", "25"), ("city", "LA")]),
            row(&[("name", "Charlie"), ("age", "35"), ("city", "NYC")]),
            row(&[("name", "Diana"), ("age", "28"), ("city", "SF")]),
            row(&[("name", "Eve"), ("age", "22"), ("city", "LA")]),
        ];
        let mut ts = TableState::new(cols, rows);
        ts.set_page_size(25); // all on one page
        ts
    }

    #[test]
    fn sort_asc() {
        let mut ts = sample_table();
        ts.set_sort("age", SortDirection::Asc);
        let rows = ts.visible_rows();
        assert_eq!(rows[0]["name"], "Eve"); // 22
        assert_eq!(rows[4]["name"], "Charlie"); // 35
    }

    #[test]
    fn sort_desc() {
        let mut ts = sample_table();
        ts.set_sort("age", SortDirection::Desc);
        let rows = ts.visible_rows();
        assert_eq!(rows[0]["name"], "Charlie"); // 35
        assert_eq!(rows[4]["name"], "Eve"); // 22
    }

    #[test]
    fn multi_sort() {
        let mut ts = sample_table();
        ts.set_sort("city", SortDirection::Asc);
        ts.set_sort("age", SortDirection::Asc);
        let rows = ts.visible_rows();
        // city sort first, then age within same city
        // LA: Eve(22), Bob(25) | NYC: Alice(30), Charlie(35) | SF: Diana(28)
        assert_eq!(rows[0]["name"], "Eve");
        assert_eq!(rows[1]["name"], "Bob");
    }

    #[test]
    fn filter_contains() {
        let mut ts = sample_table();
        ts.add_filter(ColumnFilter {
            column_id: "city".into(),
            op: FilterOp::Contains,
            value: "NY".into(),
        });
        let rows = ts.visible_rows();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["city"] == "NYC"));
    }

    #[test]
    fn filter_gt_lt() {
        let mut ts = sample_table();
        ts.add_filter(ColumnFilter {
            column_id: "age".into(),
            op: FilterOp::Gt,
            value: "28".into(),
        });
        assert_eq!(ts.filtered_count(), 2); // Alice(30), Charlie(35)

        ts.clear_filters();
        ts.add_filter(ColumnFilter {
            column_id: "age".into(),
            op: FilterOp::Lt,
            value: "25".into(),
        });
        assert_eq!(ts.filtered_count(), 1); // Eve(22)
    }

    #[test]
    fn paginate() {
        let mut ts = sample_table();
        ts.set_page_size(2);
        let page0 = ts.visible_rows();
        assert_eq!(page0.len(), 2);

        ts.next_page();
        let page1 = ts.visible_rows();
        assert_eq!(page1.len(), 2);

        ts.next_page();
        let page2 = ts.visible_rows();
        assert_eq!(page2.len(), 1);
    }

    #[test]
    fn next_previous_page() {
        let mut ts = sample_table();
        ts.set_page_size(2);
        assert_eq!(ts.page_state.page, 0);
        ts.next_page();
        assert_eq!(ts.page_state.page, 1);
        ts.previous_page();
        assert_eq!(ts.page_state.page, 0);
        // Previous at page 0 stays at 0
        ts.previous_page();
        assert_eq!(ts.page_state.page, 0);
    }

    #[test]
    fn toggle_sort_cycles() {
        let mut ts = sample_table();
        assert!(ts.sort.is_empty());
        ts.toggle_sort("name");
        assert_eq!(ts.sort.len(), 1);
        assert_eq!(ts.sort[0].direction, SortDirection::Asc);
        ts.toggle_sort("name");
        assert_eq!(ts.sort[0].direction, SortDirection::Desc);
        ts.toggle_sort("name");
        assert!(ts.sort.is_empty());
    }

    #[test]
    fn select_deselect() {
        let mut ts = sample_table();
        ts.select_row(0);
        ts.select_row(2);
        assert_eq!(ts.selected_count(), 2);
        ts.deselect_row(0);
        assert_eq!(ts.selected_count(), 1);
        ts.select_all();
        assert_eq!(ts.selected_count(), 5);
        ts.deselect_all();
        assert_eq!(ts.selected_count(), 0);
    }

    #[test]
    fn visible_rows_applies_all() {
        let mut ts = sample_table();
        // Filter to NYC, sort by age desc, page size 1
        ts.add_filter(ColumnFilter {
            column_id: "city".into(),
            op: FilterOp::Eq,
            value: "NYC".into(),
        });
        ts.set_sort("age", SortDirection::Desc);
        ts.set_page_size(1);

        let page0 = ts.visible_rows();
        assert_eq!(page0.len(), 1);
        assert_eq!(page0[0]["name"], "Charlie"); // 35, NYC

        ts.next_page();
        let page1 = ts.visible_rows();
        assert_eq!(page1.len(), 1);
        assert_eq!(page1[0]["name"], "Alice"); // 30, NYC
    }

    #[test]
    fn export_csv() {
        let mut ts = sample_table();
        ts.set_sort("name", SortDirection::Asc);
        let csv = ts.export_csv();
        let lines: Vec<&str> = csv.trim().lines().collect();
        assert_eq!(lines[0], "Name,Age,City");
        assert!(lines[1].starts_with("Alice,"));
        assert_eq!(lines.len(), 6); // header + 5 rows
    }

    #[test]
    fn column_resize() {
        let mut ts = sample_table();
        ts.resize_column("name", 200.0);
        let c = ts.columns.iter().find(|c| c.id == "name").unwrap();
        assert_eq!(c.width, 200.0);

        // Respects min_width
        ts.resize_column("name", 10.0);
        let c = ts.columns.iter().find(|c| c.id == "name").unwrap();
        assert_eq!(c.width, 50.0);
    }

    #[test]
    fn column_visibility() {
        let mut ts = sample_table();
        ts.toggle_column_visibility("age");
        let c = ts.columns.iter().find(|c| c.id == "age").unwrap();
        assert!(!c.visible);
        // CSV should not include age column
        let csv = ts.export_csv();
        assert!(csv.starts_with("Name,City"));
    }

    #[test]
    fn reorder_columns() {
        let mut ts = sample_table();
        // name(0), age(1), city(2) → city(0), name(1), age(2)
        ts.reorder_columns(2, 0);
        assert_eq!(ts.columns[0].id, "city");
        assert_eq!(ts.columns[1].id, "name");
        assert_eq!(ts.columns[2].id, "age");
    }

    #[test]
    fn page_state_helpers() {
        let ps = PageState {
            page: 1,
            page_size: 10,
            total_rows: 25,
        };
        assert_eq!(ps.total_pages(), 3);
        assert!(ps.has_next());
        assert!(ps.has_previous());
        assert_eq!(ps.offset(), 10);
    }
}
