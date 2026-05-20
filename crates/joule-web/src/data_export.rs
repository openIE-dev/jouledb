//! Export data grid contents to CSV, JSON, HTML, and TSV.
//!
//! Replaces SheetJS / papaparse with pure Rust export logic.
//! Stream-friendly: rows can be generated one at a time via the iterator API.

use std::collections::HashMap;

// ── ExportFormat ────────────────────────────────────────────────

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Tsv,
    Json,
    Html,
}

// ── ExportConfig ────────────────────────────────────────────────

/// Controls what and how to export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    /// Columns to include (in order).  Empty = all columns.
    pub columns: Vec<String>,
    /// Whether to include a header row (CSV/TSV/HTML).
    pub include_headers: bool,
    /// Delimiter for CSV (default `','`).
    pub delimiter: char,
    /// Header mapping: field name → display name.
    pub header_map: HashMap<String, String>,
    /// Date format pattern (placeholder — applied by caller).
    pub date_format: String,
    /// Number format: decimal places.
    pub number_decimals: Option<usize>,
    /// If set, only export these row ids.
    pub selected_rows: Option<Vec<String>>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Csv,
            columns: Vec::new(),
            include_headers: true,
            delimiter: ',',
            header_map: HashMap::new(),
            date_format: "%Y-%m-%d".into(),
            number_decimals: None,
            selected_rows: None,
        }
    }
}

// ── ExportRow ───────────────────────────────────────────────────

/// A flat row of string values for export.
#[derive(Debug, Clone)]
pub struct ExportRow {
    pub id: String,
    pub values: HashMap<String, String>,
}

// ── Exporter ────────────────────────────────────────────────────

/// Generates export output from rows.
pub struct Exporter<'a> {
    config: &'a ExportConfig,
    columns: Vec<String>,
}

impl<'a> Exporter<'a> {
    pub fn new(config: &'a ExportConfig, all_columns: &[String]) -> Self {
        let columns = if config.columns.is_empty() {
            all_columns.to_vec()
        } else {
            config.columns.clone()
        };
        Self { config, columns }
    }

    /// Display name for a column field.
    fn display_name(&self, field: &str) -> String {
        self.config
            .header_map
            .get(field)
            .cloned()
            .unwrap_or_else(|| field.to_string())
    }

    /// Format a cell value according to config.
    fn format_value(&self, raw: &str) -> String {
        if let Some(dec) = self.config.number_decimals {
            if let Ok(n) = raw.parse::<f64>() {
                return format!("{:.prec$}", n, prec = dec);
            }
        }
        raw.to_string()
    }

    // ── CSV / TSV ───────────────────────────────────────────────

    fn escape_csv(value: &str, delim: char) -> String {
        if value.contains(delim) || value.contains('"') || value.contains('\n') {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_string()
        }
    }

    fn delimited_header(&self, delim: char) -> String {
        self.columns
            .iter()
            .map(|c| Self::escape_csv(&self.display_name(c), delim))
            .collect::<Vec<_>>()
            .join(&delim.to_string())
    }

    fn delimited_row(&self, row: &ExportRow, delim: char) -> String {
        self.columns
            .iter()
            .map(|c| {
                let raw = row.values.get(c).map(|s| s.as_str()).unwrap_or("");
                Self::escape_csv(&self.format_value(raw), delim)
            })
            .collect::<Vec<_>>()
            .join(&delim.to_string())
    }

    /// Generate a single CSV/TSV row string (no trailing newline).
    pub fn row_to_delimited(&self, row: &ExportRow) -> String {
        let delim = match self.config.format {
            ExportFormat::Tsv => '\t',
            _ => self.config.delimiter,
        };
        self.delimited_row(row, delim)
    }

    // ── Full export ─────────────────────────────────────────────

    /// Export all rows to a string in the configured format.
    pub fn export(&self, rows: &[ExportRow]) -> String {
        let filtered = self.filter_rows(rows);
        match self.config.format {
            ExportFormat::Csv => self.export_delimited(&filtered, self.config.delimiter),
            ExportFormat::Tsv => self.export_delimited(&filtered, '\t'),
            ExportFormat::Json => self.export_json(&filtered),
            ExportFormat::Html => self.export_html(&filtered),
        }
    }

    fn filter_rows<'b>(&self, rows: &'b [ExportRow]) -> Vec<&'b ExportRow> {
        match &self.config.selected_rows {
            Some(ids) => rows.iter().filter(|r| ids.contains(&r.id)).collect(),
            Option::None => rows.iter().collect(),
        }
    }

    fn export_delimited(&self, rows: &[&ExportRow], delim: char) -> String {
        let mut out = String::new();
        if self.config.include_headers {
            out.push_str(&self.delimited_header(delim));
            out.push('\n');
        }
        for row in rows {
            out.push_str(&self.delimited_row(row, delim));
            out.push('\n');
        }
        out
    }

    fn export_json(&self, rows: &[&ExportRow]) -> String {
        let mut out = String::from("[\n");
        for (i, row) in rows.iter().enumerate() {
            out.push_str("  {");
            for (j, col) in self.columns.iter().enumerate() {
                let key = self.display_name(col);
                let val = row.values.get(col).map(|s| s.as_str()).unwrap_or("");
                let formatted = self.format_value(val);
                // Escape JSON string.
                let escaped = formatted
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n");
                out.push_str(&format!("\"{key}\": \"{escaped}\""));
                if j + 1 < self.columns.len() {
                    out.push_str(", ");
                }
            }
            out.push('}');
            if i + 1 < rows.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push(']');
        out
    }

    fn export_html(&self, rows: &[&ExportRow]) -> String {
        let mut out = String::from("<table>\n");
        if self.config.include_headers {
            out.push_str("  <thead><tr>");
            for col in &self.columns {
                let name = self.display_name(col);
                out.push_str(&format!("<th>{}</th>", html_escape(&name)));
            }
            out.push_str("</tr></thead>\n");
        }
        out.push_str("  <tbody>\n");
        for row in rows {
            out.push_str("    <tr>");
            for col in &self.columns {
                let val = row.values.get(col).map(|s| s.as_str()).unwrap_or("");
                let formatted = self.format_value(val);
                out.push_str(&format!("<td>{}</td>", html_escape(&formatted)));
            }
            out.push_str("</tr>\n");
        }
        out.push_str("  </tbody>\n</table>");
        out
    }
}

/// Minimal HTML escaping.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Convenience ─────────────────────────────────────────────────

/// Quick export with default config.
pub fn export_csv(columns: &[String], rows: &[ExportRow]) -> String {
    let config = ExportConfig::default();
    let exporter = Exporter::new(&config, columns);
    exporter.export(rows)
}

pub fn export_json(columns: &[String], rows: &[ExportRow]) -> String {
    let config = ExportConfig { format: ExportFormat::Json, ..Default::default() };
    let exporter = Exporter::new(&config, columns);
    exporter.export(rows)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cols() -> Vec<String> {
        vec!["name".into(), "age".into()]
    }

    fn rows() -> Vec<ExportRow> {
        vec![
            ExportRow {
                id: "r1".into(),
                values: HashMap::from([
                    ("name".into(), "Alice".into()),
                    ("age".into(), "30".into()),
                ]),
            },
            ExportRow {
                id: "r2".into(),
                values: HashMap::from([
                    ("name".into(), "Bob".into()),
                    ("age".into(), "25".into()),
                ]),
            },
        ]
    }

    #[test]
    fn csv_with_headers() {
        let csv = export_csv(&cols(), &rows());
        assert!(csv.starts_with("name,age\n"));
        assert!(csv.contains("Alice,30"));
    }

    #[test]
    fn csv_without_headers() {
        let config = ExportConfig {
            include_headers: false,
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let csv = exporter.export(&rows());
        assert!(csv.starts_with("Alice,30"));
    }

    #[test]
    fn tsv_export() {
        let config = ExportConfig {
            format: ExportFormat::Tsv,
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let tsv = exporter.export(&rows());
        assert!(tsv.contains("name\tage"));
        assert!(tsv.contains("Alice\t30"));
    }

    #[test]
    fn json_export() {
        let json = export_json(&cols(), &rows());
        assert!(json.starts_with('['));
        assert!(json.contains("\"name\": \"Alice\""));
    }

    #[test]
    fn html_export() {
        let config = ExportConfig {
            format: ExportFormat::Html,
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let html = exporter.export(&rows());
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>name</th>"));
        assert!(html.contains("<td>Alice</td>"));
    }

    #[test]
    fn header_mapping() {
        let config = ExportConfig {
            header_map: HashMap::from([("name".into(), "Full Name".into())]),
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let csv = exporter.export(&rows());
        assert!(csv.starts_with("Full Name,age\n"));
    }

    #[test]
    fn selected_rows_only() {
        let config = ExportConfig {
            selected_rows: Some(vec!["r2".into()]),
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let csv = exporter.export(&rows());
        assert!(csv.contains("Bob"));
        assert!(!csv.contains("Alice"));
    }

    #[test]
    fn number_formatting() {
        let config = ExportConfig {
            number_decimals: Some(2),
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let csv = exporter.export(&rows());
        assert!(csv.contains("30.00"));
    }

    #[test]
    fn csv_escapes_commas() {
        let rows = vec![ExportRow {
            id: "r1".into(),
            values: HashMap::from([
                ("name".into(), "Doe, Jane".into()),
                ("age".into(), "28".into()),
            ]),
        }];
        let csv = export_csv(&cols(), &rows);
        assert!(csv.contains("\"Doe, Jane\""));
    }

    #[test]
    fn html_escapes_entities() {
        let rows = vec![ExportRow {
            id: "r1".into(),
            values: HashMap::from([
                ("name".into(), "<script>".into()),
                ("age".into(), "0".into()),
            ]),
        }];
        let config = ExportConfig {
            format: ExportFormat::Html,
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let html = exporter.export(&rows);
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn column_subset() {
        let config = ExportConfig {
            columns: vec!["age".into()],
            ..Default::default()
        };
        let exporter = Exporter::new(&config, &cols());
        let csv = exporter.export(&rows());
        assert!(!csv.contains("name"));
        assert!(csv.contains("age"));
    }

    #[test]
    fn stream_row_by_row() {
        let config = ExportConfig::default();
        let exporter = Exporter::new(&config, &cols());
        let row_str = exporter.row_to_delimited(&rows()[0]);
        assert_eq!(row_str, "Alice,30");
    }
}
