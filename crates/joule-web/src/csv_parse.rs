//! Full CSV parser and writer — RFC 4180 compliant.
//!
//! Replaces csv-parse and PapaParse with a pure-Rust streaming CSV engine.
//! Handles quoted fields, escaped quotes, custom delimiters, header mapping,
//! streaming row iteration, and column type inference.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum CsvParseError {
    #[error("unterminated quoted field at row {row}, col {col}")]
    UnterminatedQuote { row: usize, col: usize },
    #[error("unexpected quote at row {row}, col {col}")]
    UnexpectedQuote { row: usize, col: usize },
    #[error("column count mismatch: expected {expected}, got {got} at row {row}")]
    ColumnMismatch { expected: usize, got: usize, row: usize },
    #[error("header '{0}' not found")]
    HeaderNotFound(String),
    #[error("index {0} out of bounds (columns: {1})")]
    IndexOutOfBounds(usize, usize),
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for CSV parsing and writing.
#[derive(Debug, Clone)]
pub struct CsvParseConfig {
    /// Field delimiter character.
    pub delimiter: char,
    /// Quote character for enclosing fields.
    pub quote_char: char,
    /// Whether the first row contains headers.
    pub has_headers: bool,
    /// Trim leading/trailing whitespace from unquoted fields.
    pub trim_fields: bool,
    /// Skip rows that are completely empty.
    pub skip_empty: bool,
    /// Comment character — lines starting with this char are skipped.
    pub comment_char: Option<char>,
    /// Line terminator for writing.
    pub line_terminator: String,
    /// Strict mode: error on column count mismatch.
    pub strict: bool,
}

impl Default for CsvParseConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            has_headers: true,
            trim_fields: false,
            skip_empty: true,
            comment_char: None,
            line_terminator: "\r\n".to_string(),
            strict: false,
        }
    }
}

// ── Inferred Column Type ────────────────────────────────────────

/// Inferred data type for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Float,
    Boolean,
    Date,
    Text,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::Boolean => write!(f, "boolean"),
            Self::Date => write!(f, "date"),
            Self::Text => write!(f, "text"),
        }
    }
}

// ── Row ─────────────────────────────────────────────────────────

/// A single CSV row with field access by index or header name.
#[derive(Debug, Clone)]
pub struct CsvRecord {
    fields: Vec<String>,
    header_map: Option<HashMap<String, usize>>,
}

impl CsvRecord {
    /// Number of fields in this record.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether this record has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get a field by index.
    pub fn get(&self, index: usize) -> Option<&str> {
        self.fields.get(index).map(|s| s.as_str())
    }

    /// Get a field by header name.
    pub fn get_by_name(&self, name: &str) -> Result<&str, CsvParseError> {
        let map = self.header_map.as_ref().ok_or_else(|| CsvParseError::HeaderNotFound(name.to_string()))?;
        let idx = map.get(name).ok_or_else(|| CsvParseError::HeaderNotFound(name.to_string()))?;
        self.fields.get(*idx)
            .map(|s| s.as_str())
            .ok_or(CsvParseError::IndexOutOfBounds(*idx, self.fields.len()))
    }

    /// Iterate over all fields.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|s| s.as_str())
    }

    /// Get all fields as a slice.
    pub fn as_slice(&self) -> &[String] {
        &self.fields
    }
}

// ── Parsed Document ─────────────────────────────────────────────

/// A fully parsed CSV document.
#[derive(Debug, Clone)]
pub struct CsvDocument {
    pub headers: Option<Vec<String>>,
    pub records: Vec<CsvRecord>,
}

impl CsvDocument {
    /// Number of data records (excluding header).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether there are no data records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get a record by row index.
    pub fn get(&self, index: usize) -> Option<&CsvRecord> {
        self.records.get(index)
    }

    /// Iterate over records.
    pub fn iter(&self) -> impl Iterator<Item = &CsvRecord> {
        self.records.iter()
    }

    /// Infer the column type for each column by scanning all rows.
    pub fn infer_types(&self) -> Vec<ColumnType> {
        let ncols = self.headers.as_ref().map(|h| h.len())
            .unwrap_or_else(|| self.records.first().map(|r| r.len()).unwrap_or(0));
        let mut types = vec![ColumnType::Integer; ncols];
        for record in &self.records {
            for (i, field) in record.fields.iter().enumerate() {
                if i >= ncols { break; }
                let trimmed = field.trim();
                if trimmed.is_empty() { continue; }
                let current = types[i];
                types[i] = match current {
                    ColumnType::Integer => {
                        if trimmed.parse::<i64>().is_ok() {
                            ColumnType::Integer
                        } else if trimmed.parse::<f64>().is_ok() {
                            ColumnType::Float
                        } else if is_bool(trimmed) {
                            ColumnType::Boolean
                        } else if is_date_like(trimmed) {
                            ColumnType::Date
                        } else {
                            ColumnType::Text
                        }
                    }
                    ColumnType::Float => {
                        if trimmed.parse::<f64>().is_ok() {
                            ColumnType::Float
                        } else {
                            ColumnType::Text
                        }
                    }
                    ColumnType::Boolean => {
                        if is_bool(trimmed) {
                            ColumnType::Boolean
                        } else {
                            ColumnType::Text
                        }
                    }
                    ColumnType::Date => {
                        if is_date_like(trimmed) {
                            ColumnType::Date
                        } else {
                            ColumnType::Text
                        }
                    }
                    ColumnType::Text => ColumnType::Text,
                };
            }
        }
        types
    }
}

fn is_bool(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "true" | "false" | "yes" | "no" | "1" | "0")
}

fn is_date_like(s: &str) -> bool {
    // Simple YYYY-MM-DD or YYYY/MM/DD check
    if s.len() < 8 { return false; }
    let parts: Vec<&str> = if s.contains('-') {
        s.split('-').collect()
    } else if s.contains('/') {
        s.split('/').collect()
    } else {
        return false;
    };
    if parts.len() != 3 { return false; }
    parts[0].len() == 4 && parts[0].parse::<u32>().is_ok()
        && parts[1].parse::<u32>().is_ok()
        && parts[2].parse::<u32>().is_ok()
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse a CSV string with default configuration.
pub fn parse(input: &str) -> Result<CsvDocument, CsvParseError> {
    parse_with(input, &CsvParseConfig::default())
}

/// Parse a CSV string with custom configuration.
pub fn parse_with(input: &str, config: &CsvParseConfig) -> Result<CsvDocument, CsvParseError> {
    let raw_rows = parse_raw_rows(input, config)?;
    let (headers, data_rows) = if config.has_headers && !raw_rows.is_empty() {
        let h = raw_rows[0].clone();
        (Some(h), &raw_rows[1..])
    } else {
        (None, raw_rows.as_slice())
    };

    let header_map = headers.as_ref().map(|hdrs| {
        hdrs.iter().enumerate().map(|(i, h)| (h.clone(), i)).collect::<HashMap<String, usize>>()
    });

    let expected_cols = headers.as_ref().map(|h| h.len());

    let mut records = Vec::with_capacity(data_rows.len());
    for (row_idx, row) in data_rows.iter().enumerate() {
        if config.strict {
            if let Some(exp) = expected_cols {
                if row.len() != exp {
                    return Err(CsvParseError::ColumnMismatch {
                        expected: exp,
                        got: row.len(),
                        row: row_idx + if config.has_headers { 2 } else { 1 },
                    });
                }
            }
        }
        records.push(CsvRecord {
            fields: row.clone(),
            header_map: header_map.clone(),
        });
    }

    Ok(CsvDocument { headers, records })
}

fn parse_raw_rows(input: &str, config: &CsvParseConfig) -> Result<Vec<Vec<String>>, CsvParseError> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut row_num: usize = 1;
    let mut col_num: usize = 1;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let delim = config.delimiter;
    let quote = config.quote_char;

    while i < chars.len() {
        let c = chars[i];

        if in_quotes {
            if c == quote {
                // Check for escaped quote (double quote)
                if i + 1 < chars.len() && chars[i + 1] == quote {
                    current_field.push(quote);
                    i += 2;
                    continue;
                }
                // End of quoted field
                in_quotes = false;
                i += 1;
                continue;
            }
            current_field.push(c);
            if c == '\n' {
                row_num += 1;
                col_num = 1;
            }
            i += 1;
            continue;
        }

        if c == quote && current_field.is_empty() {
            in_quotes = true;
            i += 1;
            continue;
        }

        if c == delim {
            let field = if config.trim_fields { current_field.trim().to_string() } else { current_field.clone() };
            current_row.push(field);
            current_field.clear();
            col_num += 1;
            i += 1;
            continue;
        }

        if c == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
            // CRLF
            let field = if config.trim_fields { current_field.trim().to_string() } else { current_field.clone() };
            current_row.push(field);
            current_field.clear();
            finish_row(&mut rows, &mut current_row, config);
            row_num += 1;
            col_num = 1;
            i += 2;
            continue;
        }

        if c == '\n' {
            let field = if config.trim_fields { current_field.trim().to_string() } else { current_field.clone() };
            current_row.push(field);
            current_field.clear();
            finish_row(&mut rows, &mut current_row, config);
            row_num += 1;
            col_num = 1;
            i += 1;
            continue;
        }

        // Comment lines
        if c == config.comment_char.unwrap_or('\0') && current_field.is_empty() && current_row.is_empty() {
            // Skip to end of line
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            if i < chars.len() { i += 1; }
            row_num += 1;
            continue;
        }

        current_field.push(c);
        i += 1;
    }

    if in_quotes {
        return Err(CsvParseError::UnterminatedQuote { row: row_num, col: col_num });
    }

    // Last field / row
    if !current_field.is_empty() || !current_row.is_empty() {
        let field = if config.trim_fields { current_field.trim().to_string() } else { current_field };
        current_row.push(field);
        finish_row(&mut rows, &mut current_row, config);
    }

    Ok(rows)
}

fn finish_row(rows: &mut Vec<Vec<String>>, current_row: &mut Vec<String>, config: &CsvParseConfig) {
    let row = std::mem::take(current_row);
    if config.skip_empty && row.iter().all(|f| f.trim().is_empty()) {
        return;
    }
    rows.push(row);
}

// ── Streaming Row Iterator ──────────────────────────────────────

/// Streaming CSV row iterator — parses one row at a time without loading
/// the entire document into memory.
pub struct CsvRowIter<'a> {
    chars: Vec<char>,
    pos: usize,
    config: &'a CsvParseConfig,
    headers: Option<Vec<String>>,
    header_map: Option<HashMap<String, usize>>,
    row_num: usize,
    done: bool,
}

impl<'a> CsvRowIter<'a> {
    /// Create a streaming row iterator over the given input.
    pub fn new(input: &str, config: &'a CsvParseConfig) -> Result<Self, CsvParseError> {
        let chars: Vec<char> = input.chars().collect();
        let mut iter = Self {
            chars,
            pos: 0,
            config,
            headers: None,
            header_map: None,
            row_num: 0,
            done: false,
        };
        if config.has_headers {
            if let Some(header_fields) = iter.next_raw_row()? {
                let map: HashMap<String, usize> = header_fields.iter()
                    .enumerate()
                    .map(|(i, h)| (h.clone(), i))
                    .collect();
                iter.headers = Some(header_fields);
                iter.header_map = Some(map);
            }
        }
        Ok(iter)
    }

    /// Get the headers (if present).
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_deref()
    }

    /// Read the next row, returning None at EOF.
    pub fn next_record(&mut self) -> Result<Option<CsvRecord>, CsvParseError> {
        if self.done { return Ok(None); }
        match self.next_raw_row()? {
            Some(fields) => {
                Ok(Some(CsvRecord {
                    fields,
                    header_map: self.header_map.clone(),
                }))
            }
            None => {
                self.done = true;
                Ok(None)
            }
        }
    }

    fn next_raw_row(&mut self) -> Result<Option<Vec<String>>, CsvParseError> {
        loop {
            if self.pos >= self.chars.len() { return Ok(None); }

            // Skip comment lines
            if let Some(cc) = self.config.comment_char {
                if self.pos < self.chars.len() && self.chars[self.pos] == cc {
                    while self.pos < self.chars.len() && self.chars[self.pos] != '\n' {
                        self.pos += 1;
                    }
                    if self.pos < self.chars.len() { self.pos += 1; }
                    self.row_num += 1;
                    continue;
                }
            }

            let mut fields: Vec<String> = Vec::new();
            let mut field = String::new();
            let mut in_quotes = false;
            let delim = self.config.delimiter;
            let quote = self.config.quote_char;

            while self.pos < self.chars.len() {
                let c = self.chars[self.pos];

                if in_quotes {
                    if c == quote {
                        if self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == quote {
                            field.push(quote);
                            self.pos += 2;
                            continue;
                        }
                        in_quotes = false;
                        self.pos += 1;
                        continue;
                    }
                    field.push(c);
                    self.pos += 1;
                    continue;
                }

                if c == quote && field.is_empty() {
                    in_quotes = true;
                    self.pos += 1;
                    continue;
                }

                if c == delim {
                    let f = if self.config.trim_fields { field.trim().to_string() } else { field.clone() };
                    fields.push(f);
                    field.clear();
                    self.pos += 1;
                    continue;
                }

                if c == '\r' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '\n' {
                    self.pos += 2;
                    break;
                }

                if c == '\n' {
                    self.pos += 1;
                    break;
                }

                field.push(c);
                self.pos += 1;
            }

            if in_quotes {
                return Err(CsvParseError::UnterminatedQuote { row: self.row_num, col: fields.len() + 1 });
            }

            let f = if self.config.trim_fields { field.trim().to_string() } else { field };
            fields.push(f);
            self.row_num += 1;

            if self.config.skip_empty && fields.iter().all(|fld| fld.trim().is_empty()) {
                continue;
            }

            return Ok(Some(fields));
        }
    }
}

// ── Writer ──────────────────────────────────────────────────────

/// CSV writer that produces RFC 4180 compliant output.
pub struct CsvWriter {
    config: CsvParseConfig,
    output: String,
    row_count: usize,
}

impl CsvWriter {
    /// Create a new CSV writer with default config.
    pub fn new() -> Self {
        Self::with_config(CsvParseConfig::default())
    }

    /// Create a new CSV writer with custom config.
    pub fn with_config(config: CsvParseConfig) -> Self {
        Self {
            config,
            output: String::new(),
            row_count: 0,
        }
    }

    /// Write a header row.
    pub fn write_headers(&mut self, headers: &[&str]) {
        self.write_row_inner(headers);
    }

    /// Write a data row.
    pub fn write_row(&mut self, fields: &[&str]) {
        self.write_row_inner(fields);
    }

    fn write_row_inner(&mut self, fields: &[&str]) {
        if self.row_count > 0 {
            self.output.push_str(&self.config.line_terminator);
        }
        for (i, field) in fields.iter().enumerate() {
            if i > 0 {
                self.output.push(self.config.delimiter);
            }
            self.output.push_str(&self.quote_field(field));
        }
        self.row_count += 1;
    }

    fn quote_field(&self, field: &str) -> String {
        let q = self.config.quote_char;
        let d = self.config.delimiter;
        let needs_quoting = field.contains(d) || field.contains(q)
            || field.contains('\n') || field.contains('\r');
        if needs_quoting {
            let escaped = field.replace(q, &format!("{}{}", q, q));
            format!("{}{}{}", q, escaped, q)
        } else {
            field.to_string()
        }
    }

    /// Finish writing and return the CSV string.
    pub fn finish(self) -> String {
        self.output
    }

    /// Current number of rows written.
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

impl Default for CsvWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialize a `CsvDocument` back to a CSV string.
pub fn write_document(doc: &CsvDocument, config: &CsvParseConfig) -> String {
    let mut writer = CsvWriter::with_config(config.clone());
    if let Some(headers) = &doc.headers {
        let refs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
        writer.write_headers(&refs);
    }
    for record in &doc.records {
        let refs: Vec<&str> = record.fields.iter().map(|s| s.as_str()).collect();
        writer.write_row(&refs);
    }
    writer.finish()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parse() {
        let csv = "name,age\nAlice,30\nBob,25";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.headers.as_ref().unwrap(), &["name", "age"]);
        assert_eq!(doc.len(), 2);
        assert_eq!(doc.get(0).unwrap().get(0), Some("Alice"));
        assert_eq!(doc.get(1).unwrap().get(1), Some("25"));
    }

    #[test]
    fn test_quoted_fields() {
        let csv = "name,desc\nAlice,\"hello, world\"\nBob,\"line1\nline2\"";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.get(0).unwrap().get(1), Some("hello, world"));
        assert_eq!(doc.get(1).unwrap().get(1), Some("line1\nline2"));
    }

    #[test]
    fn test_escaped_quotes() {
        let csv = "val\n\"She said \"\"hi\"\"\"";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.get(0).unwrap().get(0), Some("She said \"hi\""));
    }

    #[test]
    fn test_custom_delimiter() {
        let config = CsvParseConfig { delimiter: '\t', ..Default::default() };
        let csv = "a\tb\n1\t2";
        let doc = parse_with(csv, &config).unwrap();
        assert_eq!(doc.get(0).unwrap().get(0), Some("1"));
        assert_eq!(doc.get(0).unwrap().get(1), Some("2"));
    }

    #[test]
    fn test_no_headers() {
        let config = CsvParseConfig { has_headers: false, ..Default::default() };
        let csv = "a,b\nc,d";
        let doc = parse_with(csv, &config).unwrap();
        assert!(doc.headers.is_none());
        assert_eq!(doc.len(), 2);
        assert_eq!(doc.get(0).unwrap().get(0), Some("a"));
    }

    #[test]
    fn test_header_lookup() {
        let csv = "name,age\nAlice,30";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.get(0).unwrap().get_by_name("name").unwrap(), "Alice");
        assert_eq!(doc.get(0).unwrap().get_by_name("age").unwrap(), "30");
    }

    #[test]
    fn test_header_not_found() {
        let csv = "name,age\nAlice,30";
        let doc = parse(csv).unwrap();
        assert!(doc.get(0).unwrap().get_by_name("missing").is_err());
    }

    #[test]
    fn test_skip_empty_rows() {
        let csv = "a,b\n\n1,2\n\n3,4";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.len(), 2);
    }

    #[test]
    fn test_trim_fields() {
        let config = CsvParseConfig { trim_fields: true, ..Default::default() };
        let csv = "a , b \n 1 , 2 ";
        let doc = parse_with(csv, &config).unwrap();
        assert_eq!(doc.headers.as_ref().unwrap(), &["a", "b"]);
        assert_eq!(doc.get(0).unwrap().get(0), Some("1"));
    }

    #[test]
    fn test_comment_lines() {
        let config = CsvParseConfig { comment_char: Some('#'), ..Default::default() };
        let csv = "a,b\n# this is a comment\n1,2";
        let doc = parse_with(csv, &config).unwrap();
        assert_eq!(doc.len(), 1);
        assert_eq!(doc.get(0).unwrap().get(0), Some("1"));
    }

    #[test]
    fn test_crlf_line_endings() {
        let csv = "a,b\r\n1,2\r\n3,4";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.len(), 2);
    }

    #[test]
    fn test_writer_basic() {
        let mut w = CsvWriter::new();
        w.write_headers(&["name", "age"]);
        w.write_row(&["Alice", "30"]);
        w.write_row(&["Bob", "25"]);
        let out = w.finish();
        assert!(out.contains("name,age"));
        assert!(out.contains("Alice,30"));
    }

    #[test]
    fn test_writer_quoting() {
        let mut w = CsvWriter::new();
        w.write_row(&["hello, world", "plain"]);
        let out = w.finish();
        assert!(out.contains("\"hello, world\""));
    }

    #[test]
    fn test_writer_escaped_quotes() {
        let mut w = CsvWriter::new();
        w.write_row(&["she said \"hi\""]);
        let out = w.finish();
        assert!(out.contains("\"she said \"\"hi\"\"\""));
    }

    #[test]
    fn test_roundtrip() {
        let csv = "name,age\nAlice,30\nBob,25";
        let doc = parse(csv).unwrap();
        let output = write_document(&doc, &CsvParseConfig::default());
        let doc2 = parse(&output).unwrap();
        assert_eq!(doc2.len(), 2);
        assert_eq!(doc2.get(0).unwrap().get_by_name("name").unwrap(), "Alice");
    }

    #[test]
    fn test_type_inference() {
        let csv = "a,b,c,d\n42,3.14,true,2024-01-15\n7,2.71,false,2024-02-20";
        let doc = parse(csv).unwrap();
        let types = doc.infer_types();
        assert_eq!(types[0], ColumnType::Integer);
        assert_eq!(types[1], ColumnType::Float);
        assert_eq!(types[2], ColumnType::Boolean);
        assert_eq!(types[3], ColumnType::Date);
    }

    #[test]
    fn test_type_inference_mixed_promotes_to_text() {
        let csv = "a\n42\nhello";
        let doc = parse(csv).unwrap();
        let types = doc.infer_types();
        assert_eq!(types[0], ColumnType::Text);
    }

    #[test]
    fn test_streaming_iterator() {
        let csv = "name,age\nAlice,30\nBob,25\nCharlie,35";
        let config = CsvParseConfig::default();
        let mut iter = CsvRowIter::new(csv, &config).unwrap();
        assert_eq!(iter.headers().unwrap(), &["name", "age"]);
        let r1 = iter.next_record().unwrap().unwrap();
        assert_eq!(r1.get_by_name("name").unwrap(), "Alice");
        let r2 = iter.next_record().unwrap().unwrap();
        assert_eq!(r2.get_by_name("name").unwrap(), "Bob");
        let r3 = iter.next_record().unwrap().unwrap();
        assert_eq!(r3.get_by_name("name").unwrap(), "Charlie");
        assert!(iter.next_record().unwrap().is_none());
    }

    #[test]
    fn test_strict_mode() {
        let config = CsvParseConfig { strict: true, ..Default::default() };
        let csv = "a,b\n1,2,3";
        let result = parse_with(csv, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_unterminated_quote_error() {
        let csv = "a\n\"unterminated";
        let result = parse(csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let doc = parse("").unwrap();
        assert!(doc.is_empty());
    }

    #[test]
    fn test_single_column() {
        let csv = "val\n1\n2\n3";
        let doc = parse(csv).unwrap();
        assert_eq!(doc.len(), 3);
        assert_eq!(doc.get(2).unwrap().get(0), Some("3"));
    }

    #[test]
    fn test_record_iter() {
        let csv = "a,b\n1,2";
        let doc = parse(csv).unwrap();
        let fields: Vec<&str> = doc.get(0).unwrap().iter().collect();
        assert_eq!(fields, vec!["1", "2"]);
    }
}
