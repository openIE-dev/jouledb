// SPDX-License-Identifier: MIT
//! CSV processing engine — RFC 4180 compliant.
//!
//! Features: parsing (quoted fields, embedded delimiters/newlines), custom
//! delimiters, typed column access, row iteration, CSV writing, dialect
//! detection, header management.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CsvError {
    UnterminatedQuote { line: usize },
    ColumnNotFound(String),
    IndexOutOfBounds { index: usize, len: usize },
    ParseError { row: usize, col: usize, message: String },
    InvalidDialect(String),
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnterminatedQuote { line } => write!(f, "unterminated quote at line {line}"),
            Self::ColumnNotFound(name) => write!(f, "column not found: {name}"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "column index {index} out of bounds (len {len})")
            }
            Self::ParseError { row, col, message } => {
                write!(f, "parse error at row {row}, col {col}: {message}")
            }
            Self::InvalidDialect(msg) => write!(f, "invalid dialect: {msg}"),
        }
    }
}

// ── Dialect ─────────────────────────────────────────────────────────────────

/// CSV dialect configuration.
#[derive(Debug, Clone)]
pub struct CsvDialect {
    pub delimiter: char,
    pub quote_char: char,
    pub escape_char: Option<char>,
    pub has_header: bool,
    pub line_terminator: String,
    pub double_quote: bool, // doubled quote inside quoted field
}

impl Default for CsvDialect {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            escape_char: None,
            has_header: true,
            line_terminator: "\r\n".into(),
            double_quote: true,
        }
    }
}

impl CsvDialect {
    pub fn tsv() -> Self {
        Self { delimiter: '\t', ..Default::default() }
    }

    pub fn pipe() -> Self {
        Self { delimiter: '|', ..Default::default() }
    }

    pub fn semicolon() -> Self {
        Self { delimiter: ';', ..Default::default() }
    }
}

/// Detect the most likely dialect from a sample of text.
pub fn detect_dialect(sample: &str) -> CsvDialect {
    let candidates = [',', '\t', ';', '|'];
    let first_line = sample.lines().next().unwrap_or("");

    let mut best_delim = ',';
    let mut best_count = 0;
    for &d in &candidates {
        let count = first_line.chars().filter(|c| *c == d).count();
        if count > best_count {
            best_count = count;
            best_delim = d;
        }
    }

    let has_header = {
        let lines: Vec<&str> = sample.lines().take(3).collect();
        if lines.len() >= 2 {
            // Heuristic: if first row has no digits but subsequent rows do
            let first_numeric = lines[0]
                .split(best_delim)
                .filter(|f| f.trim().parse::<f64>().is_ok())
                .count();
            let second_numeric = lines[1]
                .split(best_delim)
                .filter(|f| f.trim().parse::<f64>().is_ok())
                .count();
            first_numeric < second_numeric
        } else {
            false
        }
    };

    CsvDialect {
        delimiter: best_delim,
        has_header,
        ..Default::default()
    }
}

// ── Parsed Table ────────────────────────────────────────────────────────────

/// A fully parsed CSV table.
#[derive(Debug, Clone)]
pub struct CsvTable {
    pub headers: Option<Vec<String>>,
    pub rows: Vec<Vec<String>>,
    pub dialect: CsvDialect,
}

impl CsvTable {
    /// Number of data rows (excluding header).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns (from header or first row).
    pub fn col_count(&self) -> usize {
        self.headers
            .as_ref()
            .map(|h| h.len())
            .or_else(|| self.rows.first().map(|r| r.len()))
            .unwrap_or(0)
    }

    /// Get a cell by row and column index.
    pub fn get(&self, row: usize, col: usize) -> Result<&str, CsvError> {
        let r = self.rows.get(row).ok_or(CsvError::IndexOutOfBounds {
            index: row,
            len: self.rows.len(),
        })?;
        r.get(col)
            .map(|s| s.as_str())
            .ok_or(CsvError::IndexOutOfBounds {
                index: col,
                len: r.len(),
            })
    }

    /// Get a cell by row index and column name.
    pub fn get_by_name(&self, row: usize, col_name: &str) -> Result<&str, CsvError> {
        let headers = self.headers.as_ref().ok_or_else(|| {
            CsvError::ColumnNotFound(col_name.into())
        })?;
        let col_idx = headers
            .iter()
            .position(|h| h == col_name)
            .ok_or_else(|| CsvError::ColumnNotFound(col_name.into()))?;
        self.get(row, col_idx)
    }

    /// Get a column as a vector of string references.
    pub fn column(&self, col: usize) -> Result<Vec<&str>, CsvError> {
        if col >= self.col_count() {
            return Err(CsvError::IndexOutOfBounds {
                index: col,
                len: self.col_count(),
            });
        }
        Ok(self.rows.iter().filter_map(|r| r.get(col).map(|s| s.as_str())).collect())
    }

    /// Get a column by name.
    pub fn column_by_name(&self, name: &str) -> Result<Vec<&str>, CsvError> {
        let idx = self
            .headers
            .as_ref()
            .and_then(|h| h.iter().position(|x| x == name))
            .ok_or_else(|| CsvError::ColumnNotFound(name.into()))?;
        self.column(idx)
    }

    /// Parse a column as a specific type.
    pub fn column_typed<T: std::str::FromStr>(
        &self,
        col: usize,
    ) -> Result<Vec<Result<T, CsvError>>, CsvError> {
        let col_data = self.column(col)?;
        Ok(col_data
            .into_iter()
            .enumerate()
            .map(|(row, s)| {
                s.parse::<T>().map_err(|_| CsvError::ParseError {
                    row,
                    col,
                    message: format!("cannot parse '{s}'"),
                })
            })
            .collect())
    }

    /// Iterate rows as slices of fields.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[String]> {
        self.rows.iter().map(|r| r.as_slice())
    }
}

// ── Parser ──────────────────────────────────────────────────────────────────

/// Parse CSV text with the given dialect.
pub fn parse(input: &str, dialect: &CsvDialect) -> Result<CsvTable, CsvError> {
    let rows = parse_rows(input, dialect)?;
    let (headers, data_rows) = if dialect.has_header && !rows.is_empty() {
        (Some(rows[0].clone()), rows[1..].to_vec())
    } else {
        (None, rows)
    };
    Ok(CsvTable {
        headers,
        rows: data_rows,
        dialect: dialect.clone(),
    })
}

/// Parse CSV text with auto-detected dialect.
pub fn parse_auto(input: &str) -> Result<CsvTable, CsvError> {
    let dialect = detect_dialect(input);
    parse(input, &dialect)
}

fn parse_rows(input: &str, dialect: &CsvDialect) -> Result<Vec<Vec<String>>, CsvError> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quote = false;
    let mut line_num = 1;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if in_quote {
            if ch == dialect.quote_char {
                // Check for doubled quote
                if dialect.double_quote && i + 1 < chars.len() && chars[i + 1] == dialect.quote_char
                {
                    field.push(dialect.quote_char);
                    i += 2;
                    continue;
                }
                // Check for escape char
                if let Some(esc) = dialect.escape_char {
                    if ch == esc && i + 1 < chars.len() && chars[i + 1] == dialect.quote_char {
                        field.push(dialect.quote_char);
                        i += 2;
                        continue;
                    }
                }
                in_quote = false;
                i += 1;
                continue;
            }
            if ch == '\n' {
                line_num += 1;
            }
            field.push(ch);
            i += 1;
            continue;
        }

        // Not in quote
        if ch == dialect.quote_char && field.is_empty() {
            in_quote = true;
            i += 1;
            continue;
        }

        if ch == dialect.delimiter {
            current_row.push(std::mem::take(&mut field));
            i += 1;
            continue;
        }

        if ch == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
            current_row.push(std::mem::take(&mut field));
            if !current_row.iter().all(|f| f.is_empty()) || !current_row.is_empty() {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }
            line_num += 1;
            i += 2;
            continue;
        }

        if ch == '\n' {
            current_row.push(std::mem::take(&mut field));
            if !current_row.iter().all(|f| f.is_empty()) || !current_row.is_empty() {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }
            line_num += 1;
            i += 1;
            continue;
        }

        field.push(ch);
        i += 1;
    }

    if in_quote {
        return Err(CsvError::UnterminatedQuote { line: line_num });
    }

    // Push remaining field/row
    if !field.is_empty() || !current_row.is_empty() {
        current_row.push(field);
        rows.push(current_row);
    }

    Ok(rows)
}

// ── Writer ──────────────────────────────────────────────────────────────────

/// Write a CsvTable back to CSV text.
pub fn write(table: &CsvTable) -> String {
    write_with_dialect(table, &table.dialect)
}

/// Write a CsvTable with a specific dialect.
pub fn write_with_dialect(table: &CsvTable, dialect: &CsvDialect) -> String {
    let mut out = String::new();
    if let Some(headers) = &table.headers {
        write_row(&mut out, headers, dialect);
    }
    for row in &table.rows {
        write_row(&mut out, row, dialect);
    }
    out
}

fn write_row(out: &mut String, fields: &[String], dialect: &CsvDialect) {
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.push(dialect.delimiter);
        }
        let needs_quote = field.contains(dialect.delimiter)
            || field.contains(dialect.quote_char)
            || field.contains('\n')
            || field.contains('\r');
        if needs_quote {
            out.push(dialect.quote_char);
            for ch in field.chars() {
                if ch == dialect.quote_char && dialect.double_quote {
                    out.push(dialect.quote_char);
                    out.push(dialect.quote_char);
                } else {
                    out.push(ch);
                }
            }
            out.push(dialect.quote_char);
        } else {
            out.push_str(field);
        }
    }
    out.push_str(&dialect.line_terminator);
}

/// Build a CsvTable programmatically.
pub fn build_table(headers: Vec<&str>, rows: Vec<Vec<&str>>) -> CsvTable {
    CsvTable {
        headers: Some(headers.into_iter().map(String::from).collect()),
        rows: rows
            .into_iter()
            .map(|r| r.into_iter().map(String::from).collect())
            .collect(),
        dialect: CsvDialect::default(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_csv() {
        let input = "name,age\nAlice,30\nBob,25\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.headers.as_ref().unwrap(), &["name", "age"]);
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.get(0, 0).unwrap(), "Alice");
        assert_eq!(table.get(1, 1).unwrap(), "25");
    }

    #[test]
    fn quoted_fields() {
        let input = "a,b\n\"hello, world\",\"line1\nline2\"\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "hello, world");
        assert_eq!(table.get(0, 1).unwrap(), "line1\nline2");
    }

    #[test]
    fn doubled_quotes() {
        let input = "a\n\"He said \"\"hi\"\"\"\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "He said \"hi\"");
    }

    #[test]
    fn crlf_line_endings() {
        let input = "a,b\r\n1,2\r\n3,4\r\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.get(0, 0).unwrap(), "1");
        assert_eq!(table.get(1, 1).unwrap(), "4");
    }

    #[test]
    fn no_trailing_newline() {
        let input = "a,b\n1,2";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.row_count(), 1);
    }

    #[test]
    fn no_header() {
        let dialect = CsvDialect { has_header: false, ..Default::default() };
        let input = "1,2\n3,4\n";
        let table = parse(input, &dialect).unwrap();
        assert!(table.headers.is_none());
        assert_eq!(table.row_count(), 2);
    }

    #[test]
    fn tsv() {
        let input = "name\tage\nAlice\t30\n";
        let table = parse(input, &CsvDialect::tsv()).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "Alice");
    }

    #[test]
    fn pipe_delimiter() {
        let input = "a|b\n1|2\n";
        let table = parse(input, &CsvDialect::pipe()).unwrap();
        assert_eq!(table.get(0, 1).unwrap(), "2");
    }

    #[test]
    fn semicolon_delimiter() {
        let input = "a;b\n1;2\n";
        let table = parse(input, &CsvDialect::semicolon()).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "1");
    }

    #[test]
    fn get_by_name() {
        let input = "name,age,city\nAlice,30,NYC\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.get_by_name(0, "city").unwrap(), "NYC");
    }

    #[test]
    fn get_by_name_missing() {
        let input = "name\nAlice\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert!(matches!(
            table.get_by_name(0, "nope"),
            Err(CsvError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn column_access() {
        let input = "a,b\n1,2\n3,4\n5,6\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let col = table.column(1).unwrap();
        assert_eq!(col, vec!["2", "4", "6"]);
    }

    #[test]
    fn column_by_name_access() {
        let input = "x,y\n10,20\n30,40\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let col = table.column_by_name("y").unwrap();
        assert_eq!(col, vec!["20", "40"]);
    }

    #[test]
    fn typed_column() {
        let input = "val\n1\n2\n3\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let vals: Vec<i32> = table
            .column_typed::<i32>(0)
            .unwrap()
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn typed_column_error() {
        let input = "val\nabc\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let results = table.column_typed::<i32>(0).unwrap();
        assert!(results[0].is_err());
    }

    #[test]
    fn iter_rows() {
        let input = "a,b\n1,2\n3,4\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let rows: Vec<&[String]> = table.iter_rows().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], "1");
    }

    #[test]
    fn col_count() {
        let input = "a,b,c\n1,2,3\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.col_count(), 3);
    }

    #[test]
    fn write_roundtrip() {
        let input = "name,age\r\nAlice,30\r\nBob,25\r\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        let output = write(&table);
        assert_eq!(output, input);
    }

    #[test]
    fn write_quotes_when_needed() {
        let table = build_table(vec!["a"], vec![vec!["hello, world"]]);
        let output = write(&table);
        assert!(output.contains("\"hello, world\""));
    }

    #[test]
    fn write_escapes_quotes() {
        let table = build_table(vec!["a"], vec![vec!["say \"hi\""]]);
        let output = write(&table);
        assert!(output.contains("\"say \"\"hi\"\"\""));
    }

    #[test]
    fn detect_comma() {
        let d = detect_dialect("a,b,c\n1,2,3\n");
        assert_eq!(d.delimiter, ',');
    }

    #[test]
    fn detect_tab() {
        let d = detect_dialect("a\tb\tc\n1\t2\t3\n");
        assert_eq!(d.delimiter, '\t');
    }

    #[test]
    fn detect_semicolon() {
        let d = detect_dialect("a;b;c\n1;2;3\n");
        assert_eq!(d.delimiter, ';');
    }

    #[test]
    fn detect_pipe() {
        let d = detect_dialect("a|b|c\n1|2|3\n");
        assert_eq!(d.delimiter, '|');
    }

    #[test]
    fn detect_header_heuristic() {
        let d = detect_dialect("name,age\nAlice,30\nBob,25\n");
        assert!(d.has_header);
    }

    #[test]
    fn unterminated_quote() {
        let input = "a\n\"unclosed\n";
        let r = parse(input, &CsvDialect::default());
        assert!(matches!(r, Err(CsvError::UnterminatedQuote { .. })));
    }

    #[test]
    fn index_out_of_bounds() {
        let input = "a\n1\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert!(matches!(table.get(5, 0), Err(CsvError::IndexOutOfBounds { .. })));
        assert!(matches!(table.get(0, 5), Err(CsvError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn empty_fields() {
        let input = "a,b,c\n,,\n";
        let table = parse(input, &CsvDialect::default()).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "");
        assert_eq!(table.get(0, 1).unwrap(), "");
        assert_eq!(table.get(0, 2).unwrap(), "");
    }

    #[test]
    fn build_table_helper() {
        let table = build_table(
            vec!["x", "y"],
            vec![vec!["1", "2"], vec!["3", "4"]],
        );
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.col_count(), 2);
    }

    #[test]
    fn error_display() {
        assert_eq!(
            format!("{}", CsvError::UnterminatedQuote { line: 5 }),
            "unterminated quote at line 5"
        );
        assert_eq!(
            format!("{}", CsvError::ColumnNotFound("x".into())),
            "column not found: x"
        );
    }

    #[test]
    fn auto_parse() {
        let input = "name\tage\nAlice\t30\nBob\t25\n";
        let table = parse_auto(input).unwrap();
        assert_eq!(table.get(0, 0).unwrap(), "Alice");
    }
}
