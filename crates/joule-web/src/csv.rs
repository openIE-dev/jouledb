//! CSV parsing and generation.
//!
//! Replaces PapaParse and csv-parse with a pure Rust implementation.
//! Handles quoted fields, escaped quotes, multiline values, custom
//! delimiters, and header-based access.

// ── Config ──────────────────────────────────────────────────────

/// Configuration for CSV parsing and generation.
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Field delimiter (default: ',').
    pub delimiter: char,
    /// Quote character (default: '"').
    pub quote: char,
    /// First row is headers (default: true).
    pub has_header: bool,
    /// Trim whitespace from unquoted fields (default: false).
    pub trim_whitespace: bool,
    /// Skip empty rows (default: true).
    pub skip_empty_rows: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote: '"',
            has_header: true,
            trim_whitespace: false,
            skip_empty_rows: true,
        }
    }
}

// ── Types ───────────────────────────────────────────────────────

/// A single row of CSV data.
pub type CsvRow = Vec<String>;

/// A parsed CSV table with optional headers.
#[derive(Debug, Clone)]
pub struct CsvTable {
    pub headers: Option<Vec<String>>,
    pub rows: Vec<CsvRow>,
}

// ── Parsing ─────────────────────────────────────────────────────

/// Parse CSV with default configuration.
pub fn parse_csv(input: &str) -> CsvTable {
    parse_csv_with(input, &CsvConfig::default())
}

/// Parse CSV with custom configuration.
pub fn parse_csv_with(input: &str, config: &CsvConfig) -> CsvTable {
    let all_rows = parse_rows(input, config);

    let (headers, data_rows) = if config.has_header && !all_rows.is_empty() {
        (Some(all_rows[0].clone()), all_rows[1..].to_vec())
    } else {
        (None, all_rows)
    };

    CsvTable {
        headers,
        rows: data_rows,
    }
}

fn parse_rows(input: &str, config: &CsvConfig) -> Vec<CsvRow> {
    let mut rows: Vec<CsvRow> = Vec::new();
    let mut current_row: CsvRow = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        if in_quotes {
            if c == config.quote {
                // Check for escaped quote ("")
                if i + 1 < len && chars[i + 1] == config.quote {
                    current_field.push(config.quote);
                    i += 2;
                    continue;
                }
                // End of quoted field
                in_quotes = false;
                i += 1;
                continue;
            }
            current_field.push(c);
            i += 1;
            continue;
        }

        // Not in quotes
        if c == config.quote && current_field.is_empty() {
            in_quotes = true;
            i += 1;
            continue;
        }

        if c == config.delimiter {
            push_field(&mut current_row, &current_field, config);
            current_field = String::new();
            i += 1;
            continue;
        }

        if c == '\n' || (c == '\r' && i + 1 < len && chars[i + 1] == '\n') {
            push_field(&mut current_row, &current_field, config);
            current_field = String::new();

            if !config.skip_empty_rows || !is_empty_row(&current_row) {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }

            if c == '\r' {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if c == '\r' {
            // Bare \r as line ending
            push_field(&mut current_row, &current_field, config);
            current_field = String::new();

            if !config.skip_empty_rows || !is_empty_row(&current_row) {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }
            i += 1;
            continue;
        }

        current_field.push(c);
        i += 1;
    }

    // Handle last field/row
    if !current_field.is_empty() || !current_row.is_empty() {
        push_field(&mut current_row, &current_field, config);
        if !config.skip_empty_rows || !is_empty_row(&current_row) {
            rows.push(current_row);
        }
    }

    rows
}

fn push_field(row: &mut CsvRow, field: &str, config: &CsvConfig) {
    let value = if config.trim_whitespace {
        field.trim().to_string()
    } else {
        field.to_string()
    };
    row.push(value);
}

fn is_empty_row(row: &[String]) -> bool {
    row.iter().all(|f| f.is_empty())
}

// ── CsvTable methods ────────────────────────────────────────────

impl CsvTable {
    /// Number of data rows (excludes header).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Maximum number of columns across all rows.
    pub fn column_count(&self) -> usize {
        let header_cols = self.headers.as_ref().map_or(0, |h| h.len());
        let max_row_cols = self.rows.iter().map(|r| r.len()).max().unwrap_or(0);
        header_cols.max(max_row_cols)
    }

    /// Get a cell by row and column index.
    pub fn get(&self, row: usize, col: usize) -> Option<&str> {
        self.rows
            .get(row)
            .and_then(|r| r.get(col))
            .map(|s| s.as_str())
    }

    /// Get a cell by row index and header name.
    pub fn get_by_header(&self, row: usize, header: &str) -> Option<&str> {
        let col = self.header_index(header)?;
        self.get(row, col)
    }

    /// Get all values in a column by index.
    pub fn column(&self, col: usize) -> Vec<Option<&str>> {
        self.rows
            .iter()
            .map(|r| r.get(col).map(|s| s.as_str()))
            .collect()
    }

    /// Get all values in a column by header name.
    pub fn column_by_name(&self, name: &str) -> Vec<Option<&str>> {
        match self.header_index(name) {
            Some(col) => self.column(col),
            None => Vec::new(),
        }
    }

    /// Iterate over data rows.
    pub fn iter_rows(&self) -> impl Iterator<Item = &CsvRow> {
        self.rows.iter()
    }

    /// Filter rows where a column (by header name) equals a value.
    pub fn filter_rows(&self, column: &str, value: &str) -> Vec<&CsvRow> {
        let col = match self.header_index(column) {
            Some(c) => c,
            None => return Vec::new(),
        };
        self.rows
            .iter()
            .filter(|row| row.get(col).is_some_and(|v| v == value))
            .collect()
    }

    /// Sort rows by a column (by header name). Returns a sorted copy.
    pub fn sort_by(&self, column: &str, ascending: bool) -> Vec<CsvRow> {
        let col = match self.header_index(column) {
            Some(c) => c,
            None => return self.rows.clone(),
        };
        let mut sorted = self.rows.clone();
        sorted.sort_by(|a, b| {
            let va = a.get(col).map(|s| s.as_str()).unwrap_or("");
            let vb = b.get(col).map(|s| s.as_str()).unwrap_or("");
            if ascending {
                va.cmp(vb)
            } else {
                vb.cmp(va)
            }
        });
        sorted
    }

    fn header_index(&self, name: &str) -> Option<usize> {
        self.headers.as_ref()?.iter().position(|h| h == name)
    }
}

// ── Generation ──────────────────────────────────────────────────

/// CSV writer with fluent API.
pub struct CsvWriter {
    config: CsvConfig,
    buffer: String,
    has_content: bool,
}

impl CsvWriter {
    /// Create a writer with default configuration.
    pub fn new() -> Self {
        Self {
            config: CsvConfig::default(),
            buffer: String::new(),
            has_content: false,
        }
    }

    /// Create a writer with custom configuration.
    pub fn with_config(config: CsvConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            has_content: false,
        }
    }

    /// Write header row.
    pub fn headers(&mut self, headers: &[&str]) -> &mut Self {
        self.write_row(headers);
        self
    }

    /// Write a data row.
    pub fn row(&mut self, values: &[&str]) -> &mut Self {
        self.write_row(values);
        self
    }

    /// Get the final CSV string.
    pub fn finish(&self) -> String {
        self.buffer.clone()
    }

    fn write_row(&mut self, values: &[&str]) {
        if self.has_content {
            self.buffer.push('\n');
        }
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                self.buffer.push(self.config.delimiter);
            }
            self.write_field(value);
        }
        self.has_content = true;
    }

    fn write_field(&mut self, value: &str) {
        let needs_quoting = value.contains(self.config.delimiter)
            || value.contains(self.config.quote)
            || value.contains('\n')
            || value.contains('\r');

        if needs_quoting {
            self.buffer.push(self.config.quote);
            for c in value.chars() {
                if c == self.config.quote {
                    self.buffer.push(self.config.quote);
                }
                self.buffer.push(c);
            }
            self.buffer.push(self.config.quote);
        } else {
            self.buffer.push_str(value);
        }
    }
}

impl Default for CsvWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate CSV from a table using default configuration.
pub fn to_csv(table: &CsvTable) -> String {
    to_csv_with(
        table,
        &CsvConfig {
            has_header: table.headers.is_some(),
            ..CsvConfig::default()
        },
    )
}

/// Generate CSV from a table using custom configuration.
pub fn to_csv_with(table: &CsvTable, config: &CsvConfig) -> String {
    let mut writer = CsvWriter::with_config(config.clone());
    if let Some(ref headers) = table.headers {
        let h: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
        writer.headers(&h);
    }
    for row in &table.rows {
        let r: Vec<&str> = row.iter().map(|s| s.as_str()).collect();
        writer.row(&r);
    }
    writer.finish()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_csv() {
        let input = "a,b,c\n1,2,3\n4,5,6";
        let table = parse_csv(input);
        assert_eq!(table.headers.as_ref().unwrap(), &["a", "b", "c"]);
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.get(0, 0), Some("1"));
        assert_eq!(table.get(1, 2), Some("6"));
    }

    #[test]
    fn parse_with_headers() {
        let input = "name,age\nAlice,30\nBob,25";
        let table = parse_csv(input);
        assert_eq!(table.get_by_header(0, "name"), Some("Alice"));
        assert_eq!(table.get_by_header(1, "age"), Some("25"));
    }

    #[test]
    fn parse_quoted_fields() {
        let input = "name,desc\nAlice,\"hello, world\"\nBob,\"simple\"";
        let table = parse_csv(input);
        assert_eq!(table.get_by_header(0, "desc"), Some("hello, world"));
    }

    #[test]
    fn parse_escaped_quotes() {
        let input = "val\n\"he said \"\"hello\"\"\"";
        let table = parse_csv(input);
        assert_eq!(table.get(0, 0), Some("he said \"hello\""));
    }

    #[test]
    fn parse_multiline_in_quotes() {
        let input = "a,b\n\"line1\nline2\",val";
        let table = parse_csv(input);
        assert_eq!(table.row_count(), 1);
        assert_eq!(table.get(0, 0), Some("line1\nline2"));
        assert_eq!(table.get(0, 1), Some("val"));
    }

    #[test]
    fn get_by_header_works() {
        let input = "x,y,z\n1,2,3";
        let table = parse_csv(input);
        assert_eq!(table.get_by_header(0, "y"), Some("2"));
        assert_eq!(table.get_by_header(0, "missing"), None);
    }

    #[test]
    fn column_by_name_test() {
        let input = "a,b\n1,X\n2,Y\n3,Z";
        let table = parse_csv(input);
        let col = table.column_by_name("b");
        assert_eq!(col, vec![Some("X"), Some("Y"), Some("Z")]);
    }

    #[test]
    fn filter_rows_works() {
        let input = "color,size\nred,S\nblue,M\nred,L";
        let table = parse_csv(input);
        let reds = table.filter_rows("color", "red");
        assert_eq!(reds.len(), 2);
    }

    #[test]
    fn sort_by_works() {
        let input = "name,score\nCharlie,80\nAlice,95\nBob,70";
        let table = parse_csv(input);
        let sorted = table.sort_by("name", true);
        assert_eq!(sorted[0][0], "Alice");
        assert_eq!(sorted[1][0], "Bob");
        assert_eq!(sorted[2][0], "Charlie");
    }

    #[test]
    fn generate_simple() {
        let mut w = CsvWriter::new();
        w.headers(&["a", "b"]).row(&["1", "2"]).row(&["3", "4"]);
        assert_eq!(w.finish(), "a,b\n1,2\n3,4");
    }

    #[test]
    fn generate_with_quoting() {
        let mut w = CsvWriter::new();
        w.headers(&["val"])
            .row(&["hello, world"])
            .row(&["say \"hi\""]);
        let output = w.finish();
        assert!(output.contains("\"hello, world\""));
        assert!(output.contains("\"say \"\"hi\"\"\""));
    }

    #[test]
    fn roundtrip_parse_generate_parse() {
        let input = "name,city\nAlice,\"New York\"\nBob,\"San Francisco\"";
        let table1 = parse_csv(input);
        let generated = to_csv(&table1);
        let table2 = parse_csv(&generated);
        assert_eq!(table1.row_count(), table2.row_count());
        assert_eq!(table1.get(0, 0), table2.get(0, 0));
        assert_eq!(table1.get(0, 1), table2.get(0, 1));
        assert_eq!(table1.get(1, 0), table2.get(1, 0));
        assert_eq!(table1.get(1, 1), table2.get(1, 1));
    }

    #[test]
    fn tsv_with_tab_delimiter() {
        let config = CsvConfig {
            delimiter: '\t',
            ..CsvConfig::default()
        };
        let input = "a\tb\n1\t2";
        let table = parse_csv_with(input, &config);
        assert_eq!(table.get(0, 0), Some("1"));
        assert_eq!(table.get(0, 1), Some("2"));
    }

    #[test]
    fn empty_input() {
        let table = parse_csv("");
        assert_eq!(table.row_count(), 0);
        assert!(table.headers.is_none());
    }

    #[test]
    fn skip_empty_rows_test() {
        let input = "a,b\n1,2\n\n3,4";
        let table = parse_csv(input);
        assert_eq!(table.row_count(), 2);
    }

    #[test]
    fn trim_whitespace_test() {
        let config = CsvConfig {
            trim_whitespace: true,
            has_header: false,
            ..CsvConfig::default()
        };
        let input = " hello , world \nfoo, bar ";
        let table = parse_csv_with(input, &config);
        assert_eq!(table.get(0, 0), Some("hello"));
        assert_eq!(table.get(0, 1), Some("world"));
        assert_eq!(table.get(1, 1), Some("bar"));
    }
}
