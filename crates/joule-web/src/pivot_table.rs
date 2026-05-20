//! Pivot table: group, aggregate, and cross-tabulate flat data.
//!
//! Replaces PivotTable.js / AG Grid pivot mode with pure Rust logic.

use std::collections::HashMap;

// ── Value ───────────────────────────────────────────────────────

/// A loosely-typed value in flat source data.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Text(String),
    Number(f64),
    Bool(bool),
    Null,
}

impl Value {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::Text(s) => s.parse::<f64>().ok(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Null => None,
        }
    }

    pub fn as_text(&self) -> String {
        match self {
            Value::Text(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => String::new(),
        }
    }
}

/// A source data row.
pub type DataRow = HashMap<String, Value>;

// ── Aggregation ─────────────────────────────────────────────────

/// Aggregation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    Sum,
    Count,
    Average,
    Min,
    Max,
}

/// Apply an aggregation to a list of numeric values.
fn aggregate(values: &[f64], agg: Aggregation) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match agg {
        Aggregation::Sum => values.iter().sum(),
        Aggregation::Count => values.len() as f64,
        Aggregation::Average => {
            let s: f64 = values.iter().sum();
            s / values.len() as f64
        }
        Aggregation::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        Aggregation::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

// ── PivotConfig ─────────────────────────────────────────────────

/// Configuration for a pivot computation.
#[derive(Debug, Clone)]
pub struct PivotConfig {
    /// Fields that form row groups (hierarchical).
    pub rows: Vec<String>,
    /// Fields that form column groups.
    pub columns: Vec<String>,
    /// (field, aggregation) pairs to compute.
    pub values: Vec<(String, Aggregation)>,
}

// ── PivotCell ───────────────────────────────────────────────────

/// A single aggregated cell in the pivot result.
#[derive(Debug, Clone, PartialEq)]
pub struct PivotCell {
    pub value: f64,
    pub aggregation: Aggregation,
    pub field: String,
}

// ── PivotResult ─────────────────────────────────────────────────

/// The result of a pivot computation.
#[derive(Debug, Clone)]
pub struct PivotResult {
    /// Unique row header paths (each entry is one level in the hierarchy).
    pub row_headers: Vec<Vec<String>>,
    /// Unique column header paths.
    pub column_headers: Vec<Vec<String>>,
    /// Aggregated cells: `cells[row_idx][col_header_key][value_field]`.
    pub cells: Vec<HashMap<String, HashMap<String, f64>>>,
    /// Grand totals per column header + value field.
    pub column_totals: HashMap<String, HashMap<String, f64>>,
    /// Grand totals per row header.
    pub row_totals: Vec<HashMap<String, f64>>,
    /// Which row groups are collapsed (row header path joined by `|`).
    pub collapsed: std::collections::HashSet<String>,
}

impl PivotResult {
    /// Toggle expansion of a row group.
    pub fn toggle_collapse(&mut self, row_path: &[String]) {
        let key = row_path.join("|");
        if self.collapsed.contains(&key) {
            self.collapsed.remove(&key);
        } else {
            self.collapsed.insert(key);
        }
    }

    /// Check if a row group is collapsed.
    pub fn is_collapsed(&self, row_path: &[String]) -> bool {
        self.collapsed.contains(&row_path.join("|"))
    }

    /// Visible row headers (respecting collapsed state).
    pub fn visible_rows(&self) -> Vec<&Vec<String>> {
        self.row_headers
            .iter()
            .filter(|path| {
                // A row is visible if none of its *ancestor* paths are collapsed.
                for i in 1..path.len() {
                    let ancestor = &path[..i];
                    let key = ancestor.join("|");
                    if self.collapsed.contains(&key) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

// ── compute ─────────────────────────────────────────────────────

fn col_key(row: &DataRow, col_fields: &[String]) -> String {
    col_fields
        .iter()
        .map(|f| row.get(f).map(|v| v.as_text()).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("|")
}

fn row_key(row: &DataRow, row_fields: &[String]) -> Vec<String> {
    row_fields
        .iter()
        .map(|f| row.get(f).map(|v| v.as_text()).unwrap_or_default())
        .collect()
}

/// Compute a pivot table from flat data.
pub fn compute(data: &[DataRow], config: &PivotConfig) -> PivotResult {
    // Collect unique row and column headers.
    let mut row_set: Vec<Vec<String>> = Vec::new();
    let mut col_set: Vec<Vec<String>> = Vec::new();

    // Group data: (row_key_str, col_key_str) → Vec<&DataRow>
    let mut groups: HashMap<(String, String), Vec<&DataRow>> = HashMap::new();

    for row in data {
        let rk = row_key(row, &config.rows);
        let ck_vec: Vec<String> = config.columns.iter()
            .map(|f| row.get(f).map(|v| v.as_text()).unwrap_or_default())
            .collect();
        let ck = col_key(row, &config.columns);
        let rk_str = rk.join("|");

        if !row_set.iter().any(|r| r == &rk) {
            row_set.push(rk.clone());
        }
        if !col_set.iter().any(|c| c == &ck_vec) {
            col_set.push(ck_vec);
        }

        groups.entry((rk_str, ck)).or_default().push(row);
    }

    // Build cells.
    let mut cells: Vec<HashMap<String, HashMap<String, f64>>> = Vec::new();
    let mut row_totals: Vec<HashMap<String, f64>> = Vec::new();

    for rk in &row_set {
        let rk_str = rk.join("|");
        let mut row_cells: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut rt: HashMap<String, f64> = HashMap::new();

        // Collect all values for this row across all columns (for row totals).
        let mut row_all_values: HashMap<String, Vec<f64>> = HashMap::new();

        for ck_vec in &col_set {
            let ck = ck_vec.join("|");
            let mut col_vals: HashMap<String, f64> = HashMap::new();

            if let Some(rows) = groups.get(&(rk_str.clone(), ck.clone())) {
                for (field, agg) in &config.values {
                    let nums: Vec<f64> = rows.iter()
                        .filter_map(|r| r.get(field).and_then(|v| v.as_number()))
                        .collect();
                    let val = aggregate(&nums, *agg);
                    col_vals.insert(field.clone(), val);
                    row_all_values.entry(field.clone()).or_default().extend(&nums);
                }
            }
            row_cells.insert(ck, col_vals);
        }

        // Row totals.
        for (field, agg) in &config.values {
            if let Some(nums) = row_all_values.get(field) {
                rt.insert(field.clone(), aggregate(nums, *agg));
            }
        }

        cells.push(row_cells);
        row_totals.push(rt);
    }

    // Column totals (grand totals per column).
    let mut column_totals: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for ck_vec in &col_set {
        let ck = ck_vec.join("|");
        let mut ct: HashMap<String, f64> = HashMap::new();

        let mut all_vals: HashMap<String, Vec<f64>> = HashMap::new();
        for rk in &row_set {
            let rk_str = rk.join("|");
            if let Some(rows) = groups.get(&(rk_str, ck.clone())) {
                for (field, _agg) in &config.values {
                    let nums: Vec<f64> = rows.iter()
                        .filter_map(|r| r.get(field).and_then(|v| v.as_number()))
                        .collect();
                    all_vals.entry(field.clone()).or_default().extend(nums);
                }
            }
        }
        for (field, agg) in &config.values {
            if let Some(nums) = all_vals.get(field) {
                ct.insert(field.clone(), aggregate(nums, *agg));
            }
        }
        column_totals.insert(ck, ct);
    }

    PivotResult {
        row_headers: row_set,
        column_headers: col_set,
        cells,
        column_totals,
        row_totals,
        collapsed: std::collections::HashSet::new(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<DataRow> {
        vec![
            HashMap::from([
                ("region".into(), Value::Text("North".into())),
                ("product".into(), Value::Text("A".into())),
                ("sales".into(), Value::Number(100.0)),
            ]),
            HashMap::from([
                ("region".into(), Value::Text("North".into())),
                ("product".into(), Value::Text("B".into())),
                ("sales".into(), Value::Number(200.0)),
            ]),
            HashMap::from([
                ("region".into(), Value::Text("South".into())),
                ("product".into(), Value::Text("A".into())),
                ("sales".into(), Value::Number(150.0)),
            ]),
            HashMap::from([
                ("region".into(), Value::Text("South".into())),
                ("product".into(), Value::Text("B".into())),
                ("sales".into(), Value::Number(250.0)),
            ]),
        ]
    }

    fn config() -> PivotConfig {
        PivotConfig {
            rows: vec!["region".into()],
            columns: vec!["product".into()],
            values: vec![("sales".into(), Aggregation::Sum)],
        }
    }

    #[test]
    fn basic_pivot() {
        let result = compute(&sample_data(), &config());
        assert_eq!(result.row_headers.len(), 2);
        assert_eq!(result.column_headers.len(), 2);
    }

    #[test]
    fn sum_aggregation() {
        let result = compute(&sample_data(), &config());
        // North + A = 100
        let north_idx = result.row_headers.iter().position(|r| r[0] == "North").unwrap();
        let north_a = &result.cells[north_idx]["A"]["sales"];
        assert_eq!(*north_a, 100.0);
    }

    #[test]
    fn row_totals() {
        let result = compute(&sample_data(), &config());
        let north_idx = result.row_headers.iter().position(|r| r[0] == "North").unwrap();
        assert_eq!(result.row_totals[north_idx]["sales"], 300.0);
    }

    #[test]
    fn column_totals() {
        let result = compute(&sample_data(), &config());
        // Product A total: 100 + 150 = 250
        assert_eq!(result.column_totals["A"]["sales"], 250.0);
    }

    #[test]
    fn count_aggregation() {
        let cfg = PivotConfig {
            rows: vec!["region".into()],
            columns: vec![],
            values: vec![("sales".into(), Aggregation::Count)],
        };
        let result = compute(&sample_data(), &cfg);
        let north_idx = result.row_headers.iter().position(|r| r[0] == "North").unwrap();
        assert_eq!(result.row_totals[north_idx]["sales"], 2.0);
    }

    #[test]
    fn average_aggregation() {
        let vals = vec![10.0, 20.0, 30.0];
        assert_eq!(aggregate(&vals, Aggregation::Average), 20.0);
    }

    #[test]
    fn min_max_aggregation() {
        let vals = vec![10.0, 20.0, 30.0];
        assert_eq!(aggregate(&vals, Aggregation::Min), 10.0);
        assert_eq!(aggregate(&vals, Aggregation::Max), 30.0);
    }

    #[test]
    fn empty_data() {
        let result = compute(&[], &config());
        assert!(result.row_headers.is_empty());
        assert!(result.cells.is_empty());
    }

    #[test]
    fn collapse_expand() {
        let mut result = compute(&sample_data(), &config());
        result.toggle_collapse(&["North".to_string()]);
        assert!(result.is_collapsed(&["North".to_string()]));
        result.toggle_collapse(&["North".to_string()]);
        assert!(!result.is_collapsed(&["North".to_string()]));
    }

    #[test]
    fn visible_rows_respects_collapse() {
        // Create hierarchical data.
        let data: Vec<DataRow> = vec![
            HashMap::from([
                ("region".into(), Value::Text("North".into())),
                ("city".into(), Value::Text("NYC".into())),
                ("sales".into(), Value::Number(100.0)),
            ]),
            HashMap::from([
                ("region".into(), Value::Text("North".into())),
                ("city".into(), Value::Text("Boston".into())),
                ("sales".into(), Value::Number(50.0)),
            ]),
            HashMap::from([
                ("region".into(), Value::Text("South".into())),
                ("city".into(), Value::Text("Miami".into())),
                ("sales".into(), Value::Number(200.0)),
            ]),
        ];
        let cfg = PivotConfig {
            rows: vec!["region".into(), "city".into()],
            columns: vec![],
            values: vec![("sales".into(), Aggregation::Sum)],
        };
        let mut result = compute(&data, &cfg);
        let all = result.visible_rows().len();
        assert!(all >= 3);

        // Collapse "North" — its children ("North|NYC", "North|Boston") should hide.
        result.toggle_collapse(&["North".to_string()]);
        let after = result.visible_rows().len();
        assert!(after < all);
    }

    #[test]
    fn aggregate_empty_values() {
        assert_eq!(aggregate(&[], Aggregation::Sum), 0.0);
        assert_eq!(aggregate(&[], Aggregation::Count), 0.0);
    }
}
