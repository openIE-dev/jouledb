//! Report builder — data source binding, column definitions, grouping/aggregation,
//! sorting, filtering, header/footer/summary, page breaks, export formats
//! (text table, CSV, JSON), and parameterized reports.
//!
//! Replaces JavaScript reporting libraries (birt, jasper-reports, pdfmake) with
//! a pure-Rust report builder that generates structured output from data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Report builder domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportError {
    /// Report definition not found.
    ReportNotFound(String),
    /// Column not found in data.
    ColumnNotFound(String),
    /// No data provided.
    NoData,
    /// Invalid parameter.
    InvalidParameter { name: String, message: String },
    /// Invalid aggregation on non-numeric column.
    NonNumericColumn(String),
    /// Duplicate report ID.
    DuplicateReport(String),
    /// Export format error.
    ExportError(String),
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReportNotFound(id) => write!(f, "report not found: {id}"),
            Self::ColumnNotFound(col) => write!(f, "column not found: {col}"),
            Self::NoData => write!(f, "no data provided"),
            Self::InvalidParameter { name, message } => {
                write!(f, "invalid parameter {name}: {message}")
            }
            Self::NonNumericColumn(col) => write!(f, "column {col} is not numeric"),
            Self::DuplicateReport(id) => write!(f, "duplicate report: {id}"),
            Self::ExportError(msg) => write!(f, "export error: {msg}"),
        }
    }
}

impl std::error::Error for ReportError {}

// ── Enums ───────────────────────────────────────────────────────

/// Export format for report output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    TextTable,
    Csv,
    Json,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Aggregation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Average,
    Min,
    Max,
}

/// Column alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

impl Default for Alignment {
    fn default() -> Self {
        Self::Left
    }
}

/// Filter operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    Equals(String),
    NotEquals(String),
    Contains(String),
    GreaterThan(String),
    LessThan(String),
    IsEmpty,
    IsNotEmpty,
}

// ── Data Structures ─────────────────────────────────────────────

/// Column definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub field: String,
    pub label: String,
    pub width: Option<usize>,
    pub alignment: Alignment,
    pub format: Option<ColumnFormat>,
    pub visible: bool,
}

/// Column formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnFormat {
    /// Fixed decimal places.
    Decimal(usize),
    /// Prefix string.
    Prefix(String),
    /// Suffix string.
    Suffix(String),
    /// Uppercase.
    Uppercase,
    /// Lowercase.
    Lowercase,
}

/// Sort specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortSpec {
    pub column: String,
    pub direction: SortDirection,
}

/// Filter specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSpec {
    pub column: String,
    pub op: FilterOp,
}

/// Grouping specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSpec {
    pub column: String,
    pub aggregates: Vec<AggregateSpec>,
}

/// Aggregate specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateSpec {
    pub column: String,
    pub function: AggregateFunction,
    pub label: Option<String>,
}

/// Report parameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportParam {
    pub name: String,
    pub label: String,
    pub default_value: Option<String>,
    pub required: bool,
}

/// A data row for the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRow {
    pub values: HashMap<String, String>,
}

impl DataRow {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn set(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    pub fn get(&self, key: &str) -> &str {
        self.values.get(key).map(|s| s.as_str()).unwrap_or("")
    }
}

impl Default for DataRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Report definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDef {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub columns: Vec<ColumnDef>,
    pub sorts: Vec<SortSpec>,
    pub filters: Vec<FilterSpec>,
    pub groups: Vec<GroupSpec>,
    pub parameters: Vec<ReportParam>,
    pub header: Option<String>,
    pub footer: Option<String>,
    pub show_summary: bool,
    pub page_size: Option<usize>,
    pub created_at: DateTime<Utc>,
}

/// A generated report group with optional aggregated values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGroup {
    pub group_value: String,
    pub rows: Vec<DataRow>,
    pub aggregates: HashMap<String, String>,
}

/// Generated report output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportOutput {
    pub title: String,
    pub header: Option<String>,
    pub footer: Option<String>,
    pub columns: Vec<ColumnDef>,
    pub groups: Vec<ReportGroup>,
    pub summary: HashMap<String, String>,
    pub total_rows: usize,
    pub total_pages: usize,
    pub generated_at: DateTime<Utc>,
    pub parameters_used: HashMap<String, String>,
}

// ── Engine ──────────────────────────────────────────────────────

/// Report builder engine.
pub struct ReportBuilder {
    reports: HashMap<String, ReportDef>,
}

impl ReportBuilder {
    pub fn new() -> Self {
        Self {
            reports: HashMap::new(),
        }
    }

    /// Register a report definition.
    pub fn register(&mut self, report: ReportDef) -> Result<(), ReportError> {
        if self.reports.contains_key(&report.id) {
            return Err(ReportError::DuplicateReport(report.id.clone()));
        }
        self.reports.insert(report.id.clone(), report);
        Ok(())
    }

    /// Get a report definition.
    pub fn get_report(&self, id: &str) -> Option<&ReportDef> {
        self.reports.get(id)
    }

    /// Build and generate a report.
    pub fn generate(
        &self,
        report_id: &str,
        data: &[DataRow],
        params: &HashMap<String, String>,
    ) -> Result<ReportOutput, ReportError> {
        let report = self
            .reports
            .get(report_id)
            .ok_or_else(|| ReportError::ReportNotFound(report_id.to_string()))?;

        if data.is_empty() {
            return Err(ReportError::NoData);
        }

        // Validate required parameters.
        for p in &report.parameters {
            if p.required && !params.contains_key(&p.name) && p.default_value.is_none() {
                return Err(ReportError::InvalidParameter {
                    name: p.name.clone(),
                    message: "required parameter missing".to_string(),
                });
            }
        }

        // Resolve parameters (merge defaults + provided).
        let mut resolved_params = HashMap::new();
        for p in &report.parameters {
            if let Some(val) = params.get(&p.name) {
                resolved_params.insert(p.name.clone(), val.clone());
            } else if let Some(def) = &p.default_value {
                resolved_params.insert(p.name.clone(), def.clone());
            }
        }

        // Apply parameterized filters.
        let mut working_data: Vec<DataRow> = data.to_vec();

        // Apply filters.
        for filter in &report.filters {
            working_data.retain(|row| apply_filter(row, filter, &resolved_params));
        }

        // Apply sorts.
        for sort in report.sorts.iter().rev() {
            let col = sort.column.clone();
            let dir = sort.direction;
            working_data.sort_by(|a, b| {
                let va = a.get(&col);
                let vb = b.get(&col);
                let cmp = compare_values(va, vb);
                match dir {
                    SortDirection::Ascending => cmp,
                    SortDirection::Descending => cmp.reverse(),
                }
            });
        }

        // Group data.
        let visible_columns: Vec<ColumnDef> = report
            .columns
            .iter()
            .filter(|c| c.visible)
            .cloned()
            .collect();

        let groups = if report.groups.is_empty() {
            // No grouping — single group.
            vec![ReportGroup {
                group_value: String::new(),
                rows: working_data.clone(),
                aggregates: HashMap::new(),
            }]
        } else {
            build_groups(&working_data, &report.groups)
        };

        // Build summary aggregates.
        let mut summary = HashMap::new();
        if report.show_summary {
            for group_spec in &report.groups {
                for agg in &group_spec.aggregates {
                    let values = collect_numeric_values(&working_data, &agg.column);
                    let result = compute_aggregate(&agg.function, &values);
                    let label = agg
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("{:?}({})", agg.function, agg.column));
                    summary.insert(label, result);
                }
            }
        }

        let total_rows = working_data.len();
        let total_pages = match report.page_size {
            Some(ps) if ps > 0 => (total_rows + ps - 1) / ps,
            _ => 1,
        };

        let header = report.header.as_ref().map(|h| substitute_params(h, &resolved_params));
        let footer = report.footer.as_ref().map(|ft| substitute_params(ft, &resolved_params));

        Ok(ReportOutput {
            title: substitute_params(&report.title, &resolved_params),
            header,
            footer,
            columns: visible_columns,
            groups,
            summary,
            total_rows,
            total_pages,
            generated_at: Utc::now(),
            parameters_used: resolved_params,
        })
    }

    /// Export a generated report to the specified format.
    pub fn export(
        &self,
        output: &ReportOutput,
        format: ExportFormat,
    ) -> Result<String, ReportError> {
        match format {
            ExportFormat::TextTable => Ok(export_text_table(output)),
            ExportFormat::Csv => Ok(export_csv(output)),
            ExportFormat::Json => export_json(output),
        }
    }
}

impl Default for ReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Filter Application ──────────────────────────────────────────

fn apply_filter(
    row: &DataRow,
    filter: &FilterSpec,
    params: &HashMap<String, String>,
) -> bool {
    let val = row.get(&filter.column);
    match &filter.op {
        FilterOp::Equals(expected) => {
            let resolved = substitute_params(expected, params);
            val == resolved
        }
        FilterOp::NotEquals(expected) => {
            let resolved = substitute_params(expected, params);
            val != resolved
        }
        FilterOp::Contains(substr) => {
            let resolved = substitute_params(substr, params);
            val.contains(&resolved)
        }
        FilterOp::GreaterThan(threshold) => {
            let resolved = substitute_params(threshold, params);
            compare_values(val, &resolved) == std::cmp::Ordering::Greater
        }
        FilterOp::LessThan(threshold) => {
            let resolved = substitute_params(threshold, params);
            compare_values(val, &resolved) == std::cmp::Ordering::Less
        }
        FilterOp::IsEmpty => val.is_empty(),
        FilterOp::IsNotEmpty => !val.is_empty(),
    }
}

fn compare_values(a: &str, b: &str) -> std::cmp::Ordering {
    // Try numeric comparison first.
    if let (Ok(na), Ok(nb)) = (a.parse::<f64>(), b.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
        a.cmp(b)
    }
}

fn substitute_params(text: &str, params: &HashMap<String, String>) -> String {
    let mut result = text.to_string();
    for (key, value) in params {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, value);
    }
    result
}

// ── Grouping ────────────────────────────────────────────────────

fn build_groups(data: &[DataRow], group_specs: &[GroupSpec]) -> Vec<ReportGroup> {
    if group_specs.is_empty() {
        return vec![ReportGroup {
            group_value: String::new(),
            rows: data.to_vec(),
            aggregates: HashMap::new(),
        }];
    }

    let group_col = &group_specs[0].column;
    let mut group_map: Vec<(String, Vec<DataRow>)> = Vec::new();

    for row in data {
        let key = row.get(group_col).to_string();
        if let Some(entry) = group_map.iter_mut().find(|(k, _)| k == &key) {
            entry.1.push(row.clone());
        } else {
            group_map.push((key, vec![row.clone()]));
        }
    }

    group_map
        .into_iter()
        .map(|(key, rows)| {
            let mut aggregates = HashMap::new();
            for agg in &group_specs[0].aggregates {
                let values = collect_numeric_values(&rows, &agg.column);
                let result = compute_aggregate(&agg.function, &values);
                let label = agg
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("{:?}({})", agg.function, agg.column));
                aggregates.insert(label, result);
            }
            ReportGroup {
                group_value: key,
                rows,
                aggregates,
            }
        })
        .collect()
}

fn collect_numeric_values(rows: &[DataRow], column: &str) -> Vec<f64> {
    rows.iter()
        .filter_map(|r| r.get(column).parse::<f64>().ok())
        .collect()
}

fn compute_aggregate(func: &AggregateFunction, values: &[f64]) -> String {
    if values.is_empty() {
        return "0".to_string();
    }
    match func {
        AggregateFunction::Count => values.len().to_string(),
        AggregateFunction::Sum => {
            let sum: f64 = values.iter().sum();
            format_num(sum)
        }
        AggregateFunction::Average => {
            let sum: f64 = values.iter().sum();
            let avg = sum / values.len() as f64;
            format!("{avg:.2}")
        }
        AggregateFunction::Min => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            format_num(min)
        }
        AggregateFunction::Max => {
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            format_num(max)
        }
    }
}

fn format_num(n: f64) -> String {
    if n == n.floor() && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n:.2}")
    }
}

// ── Format Column Value ─────────────────────────────────────────

fn format_column_value(value: &str, col: &ColumnDef) -> String {
    let base = match &col.format {
        Some(ColumnFormat::Decimal(dec)) => {
            if let Ok(n) = value.parse::<f64>() {
                format!("{:.prec$}", n, prec = *dec)
            } else {
                value.to_string()
            }
        }
        Some(ColumnFormat::Prefix(p)) => format!("{p}{value}"),
        Some(ColumnFormat::Suffix(s)) => format!("{value}{s}"),
        Some(ColumnFormat::Uppercase) => value.to_uppercase(),
        Some(ColumnFormat::Lowercase) => value.to_lowercase(),
        None => value.to_string(),
    };
    base
}

// ── Export: Text Table ──────────────────────────────────────────

fn export_text_table(output: &ReportOutput) -> String {
    let mut out = String::new();

    // Title.
    out.push_str(&output.title);
    out.push('\n');
    let title_len = output.title.len();
    for _ in 0..title_len {
        out.push('=');
    }
    out.push('\n');

    // Header.
    if let Some(header) = &output.header {
        out.push_str(header);
        out.push('\n');
    }

    // Compute column widths.
    let widths: Vec<usize> = output
        .columns
        .iter()
        .map(|c| {
            let label_width = c.label.len();
            let data_max = output
                .groups
                .iter()
                .flat_map(|g| g.rows.iter())
                .map(|r| format_column_value(r.get(&c.field), c).len())
                .max()
                .unwrap_or(0);
            c.width.unwrap_or(label_width.max(data_max).max(4))
        })
        .collect();

    // Header row.
    let header_row: String = output
        .columns
        .iter()
        .zip(widths.iter())
        .map(|(c, w)| pad_str(&c.label, *w, c.alignment))
        .collect::<Vec<_>>()
        .join(" | ");
    out.push_str(&header_row);
    out.push('\n');

    // Separator.
    let sep: String = widths
        .iter()
        .map(|w| "-".repeat(*w))
        .collect::<Vec<_>>()
        .join("-+-");
    out.push_str(&sep);
    out.push('\n');

    // Data rows.
    for group in &output.groups {
        if !group.group_value.is_empty() {
            out.push_str(&format!("[{}]\n", group.group_value));
        }
        for row in &group.rows {
            let row_str: String = output
                .columns
                .iter()
                .zip(widths.iter())
                .map(|(c, w)| {
                    let val = format_column_value(row.get(&c.field), c);
                    pad_str(&val, *w, c.alignment)
                })
                .collect::<Vec<_>>()
                .join(" | ");
            out.push_str(&row_str);
            out.push('\n');
        }
    }

    // Summary.
    if !output.summary.is_empty() {
        out.push_str(&sep);
        out.push('\n');
        // Sort summary keys for deterministic output.
        let mut summary_entries: Vec<_> = output.summary.iter().collect();
        summary_entries.sort_by_key(|(k, _)| (*k).clone());
        for (label, val) in &summary_entries {
            out.push_str(&format!("{label}: {val}\n"));
        }
    }

    // Footer.
    if let Some(footer) = &output.footer {
        out.push_str(footer);
        out.push('\n');
    }

    out
}

fn pad_str(s: &str, width: usize, alignment: Alignment) -> String {
    if s.len() >= width {
        return s.to_string();
    }
    match alignment {
        Alignment::Left => format!("{:<width$}", s),
        Alignment::Right => format!("{:>width$}", s),
        Alignment::Center => {
            let padding = width - s.len();
            let left = padding / 2;
            let right = padding - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }
}

// ── Export: CSV ──────────────────────────────────────────────────

fn export_csv(output: &ReportOutput) -> String {
    let mut out = String::new();

    // Header.
    let headers: Vec<String> = output
        .columns
        .iter()
        .map(|c| csv_escape(&c.label))
        .collect();
    out.push_str(&headers.join(","));
    out.push('\n');

    // Data.
    for group in &output.groups {
        for row in &group.rows {
            let vals: Vec<String> = output
                .columns
                .iter()
                .map(|c| {
                    let val = format_column_value(row.get(&c.field), c);
                    csv_escape(&val)
                })
                .collect();
            out.push_str(&vals.join(","));
            out.push('\n');
        }
    }

    out
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ── Export: JSON ─────────────────────────────────────────────────

fn export_json(output: &ReportOutput) -> Result<String, ReportError> {
    serde_json::to_string_pretty(output)
        .map_err(|e| ReportError::ExportError(e.to_string()))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_columns() -> Vec<ColumnDef> {
        vec![
            ColumnDef {
                field: "name".into(),
                label: "Name".into(),
                width: None,
                alignment: Alignment::Left,
                format: None,
                visible: true,
            },
            ColumnDef {
                field: "dept".into(),
                label: "Department".into(),
                width: None,
                alignment: Alignment::Left,
                format: None,
                visible: true,
            },
            ColumnDef {
                field: "salary".into(),
                label: "Salary".into(),
                width: None,
                alignment: Alignment::Right,
                format: Some(ColumnFormat::Decimal(2)),
                visible: true,
            },
        ]
    }

    fn sample_data() -> Vec<DataRow> {
        vec![
            DataRow::new().set("name", "Alice").set("dept", "Engineering").set("salary", "90000"),
            DataRow::new().set("name", "Bob").set("dept", "Marketing").set("salary", "75000"),
            DataRow::new().set("name", "Charlie").set("dept", "Engineering").set("salary", "95000"),
            DataRow::new().set("name", "Diana").set("dept", "Marketing").set("salary", "80000"),
        ]
    }

    fn make_report(id: &str) -> ReportDef {
        ReportDef {
            id: id.into(),
            title: "Test Report".into(),
            description: None,
            columns: sample_columns(),
            sorts: vec![],
            filters: vec![],
            groups: vec![],
            parameters: vec![],
            header: None,
            footer: None,
            show_summary: false,
            page_size: None,
            created_at: Utc::now(),
        }
    }

    fn setup() -> ReportBuilder {
        let mut rb = ReportBuilder::new();
        rb.register(make_report("r1")).unwrap();
        rb
    }

    #[test]
    fn test_register_report() {
        let rb = setup();
        assert!(rb.get_report("r1").is_some());
    }

    #[test]
    fn test_duplicate_report() {
        let mut rb = setup();
        let err = rb.register(make_report("r1")).unwrap_err();
        assert!(matches!(err, ReportError::DuplicateReport(_)));
    }

    #[test]
    fn test_generate_basic_report() {
        let rb = setup();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.total_rows, 4);
        assert_eq!(output.title, "Test Report");
    }

    #[test]
    fn test_no_data_error() {
        let rb = setup();
        let err = rb.generate("r1", &[], &HashMap::new()).unwrap_err();
        assert_eq!(err, ReportError::NoData);
    }

    #[test]
    fn test_sorting_ascending() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.sorts = vec![SortSpec {
            column: "name".into(),
            direction: SortDirection::Ascending,
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        let names: Vec<&str> = output.groups[0]
            .rows
            .iter()
            .map(|r| r.get("name"))
            .collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie", "Diana"]);
    }

    #[test]
    fn test_sorting_descending() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.sorts = vec![SortSpec {
            column: "salary".into(),
            direction: SortDirection::Descending,
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        let salaries: Vec<&str> = output.groups[0]
            .rows
            .iter()
            .map(|r| r.get("salary"))
            .collect();
        assert_eq!(salaries, vec!["95000", "90000", "80000", "75000"]);
    }

    #[test]
    fn test_filter_equals() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.filters = vec![FilterSpec {
            column: "dept".into(),
            op: FilterOp::Equals("Engineering".into()),
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.total_rows, 2);
    }

    #[test]
    fn test_filter_contains() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.filters = vec![FilterSpec {
            column: "name".into(),
            op: FilterOp::Contains("li".into()),
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.total_rows, 2); // Alice, Charlie
    }

    #[test]
    fn test_grouping() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.groups = vec![GroupSpec {
            column: "dept".into(),
            aggregates: vec![AggregateSpec {
                column: "salary".into(),
                function: AggregateFunction::Sum,
                label: Some("Total Salary".into()),
            }],
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert!(output.groups.len() >= 2);
        let eng_group = output.groups.iter().find(|g| g.group_value == "Engineering");
        assert!(eng_group.is_some());
        let eng = eng_group.unwrap();
        assert_eq!(eng.aggregates.get("Total Salary"), Some(&"185000".to_string()));
    }

    #[test]
    fn test_aggregate_average() {
        let values = vec![10.0, 20.0, 30.0];
        let result = compute_aggregate(&AggregateFunction::Average, &values);
        assert_eq!(result, "20.00");
    }

    #[test]
    fn test_aggregate_min_max() {
        let values = vec![5.0, 15.0, 10.0];
        assert_eq!(compute_aggregate(&AggregateFunction::Min, &values), "5");
        assert_eq!(compute_aggregate(&AggregateFunction::Max, &values), "15");
    }

    #[test]
    fn test_aggregate_count() {
        let values = vec![1.0, 2.0, 3.0];
        assert_eq!(compute_aggregate(&AggregateFunction::Count, &values), "3");
    }

    #[test]
    fn test_export_text_table() {
        let rb = setup();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        let text = rb.export(&output, ExportFormat::TextTable).unwrap();
        assert!(text.contains("Test Report"));
        assert!(text.contains("Name"));
        assert!(text.contains("Alice"));
    }

    #[test]
    fn test_export_csv() {
        let rb = setup();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        let csv = rb.export(&output, ExportFormat::Csv).unwrap();
        assert!(csv.starts_with("Name,Department,Salary\n"));
        assert!(csv.contains("Alice"));
    }

    #[test]
    fn test_export_json() {
        let rb = setup();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        let json = rb.export(&output, ExportFormat::Json).unwrap();
        assert!(json.contains("\"title\""));
        assert!(json.contains("Test Report"));
    }

    #[test]
    fn test_column_formatting_decimal() {
        let col = ColumnDef {
            field: "salary".into(),
            label: "Salary".into(),
            width: None,
            alignment: Alignment::Right,
            format: Some(ColumnFormat::Decimal(2)),
            visible: true,
        };
        assert_eq!(format_column_value("90000", &col), "90000.00");
    }

    #[test]
    fn test_column_formatting_prefix_suffix() {
        let col_prefix = ColumnDef {
            field: "price".into(),
            label: "Price".into(),
            width: None,
            alignment: Alignment::Left,
            format: Some(ColumnFormat::Prefix("$".into())),
            visible: true,
        };
        assert_eq!(format_column_value("100", &col_prefix), "$100");

        let col_suffix = ColumnDef {
            field: "pct".into(),
            label: "%".into(),
            width: None,
            alignment: Alignment::Left,
            format: Some(ColumnFormat::Suffix("%".into())),
            visible: true,
        };
        assert_eq!(format_column_value("95", &col_suffix), "95%");
    }

    #[test]
    fn test_parameterized_report() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.title = "Report for {{dept}}".into();
        report.parameters = vec![ReportParam {
            name: "dept".into(),
            label: "Department".into(),
            default_value: None,
            required: true,
        }];
        report.filters = vec![FilterSpec {
            column: "dept".into(),
            op: FilterOp::Equals("{{dept}}".into()),
        }];
        rb.register(report).unwrap();

        let mut params = HashMap::new();
        params.insert("dept".into(), "Engineering".into());
        let output = rb.generate("r1", &sample_data(), &params).unwrap();
        assert_eq!(output.title, "Report for Engineering");
        assert_eq!(output.total_rows, 2);
    }

    #[test]
    fn test_required_parameter_missing() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.parameters = vec![ReportParam {
            name: "x".into(),
            label: "X".into(),
            default_value: None,
            required: true,
        }];
        rb.register(report).unwrap();
        let err = rb
            .generate("r1", &sample_data(), &HashMap::new())
            .unwrap_err();
        assert!(matches!(err, ReportError::InvalidParameter { .. }));
    }

    #[test]
    fn test_header_footer() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.header = Some("Confidential".into());
        report.footer = Some("Page 1".into());
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.header.as_deref(), Some("Confidential"));
        assert_eq!(output.footer.as_deref(), Some("Page 1"));
    }

    #[test]
    fn test_page_calculation() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.page_size = Some(2);
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.total_pages, 2);
    }

    #[test]
    fn test_hidden_column_excluded() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.columns[1].visible = false;
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.columns.len(), 2);
        assert!(!output.columns.iter().any(|c| c.field == "dept"));
    }

    #[test]
    fn test_filter_is_empty() {
        let data = vec![
            DataRow::new().set("name", "Alice").set("dept", ""),
            DataRow::new().set("name", "Bob").set("dept", "Sales"),
        ];
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.filters = vec![FilterSpec {
            column: "dept".into(),
            op: FilterOp::IsEmpty,
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &data, &HashMap::new()).unwrap();
        assert_eq!(output.total_rows, 1);
    }

    #[test]
    fn test_column_uppercase_format() {
        let col = ColumnDef {
            field: "name".into(),
            label: "Name".into(),
            width: None,
            alignment: Alignment::Left,
            format: Some(ColumnFormat::Uppercase),
            visible: true,
        };
        assert_eq!(format_column_value("alice", &col), "ALICE");
    }

    #[test]
    fn test_pad_str_center() {
        assert_eq!(pad_str("hi", 6, Alignment::Center), "  hi  ");
    }

    #[test]
    fn test_csv_escape_with_quotes() {
        assert_eq!(csv_escape("a \"b\" c"), "\"a \"\"b\"\" c\"");
    }

    #[test]
    fn test_default_parameter_value() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.parameters = vec![ReportParam {
            name: "dept".into(),
            label: "Department".into(),
            default_value: Some("All".into()),
            required: true,
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(output.parameters_used.get("dept"), Some(&"All".to_string()));
    }

    #[test]
    fn test_data_row_builder() {
        let row = DataRow::new().set("a", "1").set("b", "2");
        assert_eq!(row.get("a"), "1");
        assert_eq!(row.get("b"), "2");
        assert_eq!(row.get("c"), "");
    }

    #[test]
    fn test_summary_with_groups() {
        let mut rb = ReportBuilder::new();
        let mut report = make_report("r1");
        report.show_summary = true;
        report.groups = vec![GroupSpec {
            column: "dept".into(),
            aggregates: vec![AggregateSpec {
                column: "salary".into(),
                function: AggregateFunction::Sum,
                label: Some("Grand Total".into()),
            }],
        }];
        rb.register(report).unwrap();
        let output = rb.generate("r1", &sample_data(), &HashMap::new()).unwrap();
        assert_eq!(
            output.summary.get("Grand Total"),
            Some(&"340000".to_string())
        );
    }
}
