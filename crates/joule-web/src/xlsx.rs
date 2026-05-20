//! Spreadsheet model — Workbook, Sheet, Cell grid.
//!
//! Replaces SheetJS (xlsx) with a pure Rust spreadsheet engine.
//! Supports cell addressing (A1 notation), basic formula evaluation
//! (SUM, AVERAGE, COUNT, MIN, MAX), and CSV export.

use std::collections::HashMap;
use std::fmt;

// ── Cell Addressing ────────────────────────────────────────────

/// Parse A1-style cell reference to zero-based (col, row).
/// "A1" -> (0, 0), "B3" -> (1, 2), "AA1" -> (26, 0).
pub fn parse_cell_ref(reference: &str) -> Option<(usize, usize)> {
    let reference = reference.trim();
    if reference.is_empty() {
        return None;
    }
    let mut col = 0usize;
    let mut chars = reference.chars().peekable();
    let mut found_alpha = false;

    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_alphabetic() {
            found_alpha = true;
            col = col * 26 + (ch.to_ascii_uppercase() as usize - b'A' as usize + 1);
            chars.next();
        } else {
            break;
        }
    }

    if !found_alpha || col == 0 {
        return None;
    }
    col -= 1; // zero-based

    let row_str: String = chars.collect();
    let row: usize = row_str.parse().ok()?;
    if row == 0 {
        return None;
    }
    Some((col, row - 1))
}

/// Convert zero-based (col, row) to A1 notation.
pub fn col_row_to_ref(col: usize, row: usize) -> String {
    let mut col_str = String::new();
    let mut c = col + 1;
    while c > 0 {
        c -= 1;
        col_str.insert(0, (b'A' + (c % 26) as u8) as char);
        c /= 26;
    }
    format!("{}{}", col_str, row + 1)
}

/// Parse a range like "A1:C3" into ((col_start, row_start), (col_end, row_end)).
pub fn parse_range(range: &str) -> Option<((usize, usize), (usize, usize))> {
    let parts: Vec<&str> = range.split(':').collect();
    if parts.len() != 2 {
        return None;
    }
    let start = parse_cell_ref(parts[0])?;
    let end = parse_cell_ref(parts[1])?;
    Some((start, end))
}

// ── Cell Value ─────────────────────────────────────────────────

/// Value stored in a spreadsheet cell.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Number(f64),
    Text(String),
    Bool(bool),
    Formula(String),
    Empty,
}

impl fmt::Display for CellValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => {
                if *n == (*n as i64) as f64 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
            Self::Text(s) => write!(f, "{s}"),
            Self::Bool(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Self::Formula(s) => write!(f, "={s}"),
            Self::Empty => Ok(()),
        }
    }
}

// ── Cell ───────────────────────────────────────────────────────

/// A cell in the spreadsheet.
#[derive(Debug, Clone)]
pub struct Cell {
    pub value: CellValue,
    pub format: Option<String>,
}

impl Cell {
    pub fn new(value: CellValue) -> Self {
        Self { value, format: None }
    }

    pub fn empty() -> Self {
        Self::new(CellValue::Empty)
    }

    pub fn number(n: f64) -> Self {
        Self::new(CellValue::Number(n))
    }

    pub fn text(s: impl Into<String>) -> Self {
        Self::new(CellValue::Text(s.into()))
    }

    pub fn boolean(b: bool) -> Self {
        Self::new(CellValue::Bool(b))
    }

    pub fn formula(f: impl Into<String>) -> Self {
        Self::new(CellValue::Formula(f.into()))
    }
}

// ── Sheet ──────────────────────────────────────────────────────

/// A single spreadsheet sheet (tab).
#[derive(Debug, Clone)]
pub struct Sheet {
    pub name: String,
    cells: HashMap<(usize, usize), Cell>,
    col_count: usize,
    row_count: usize,
}

impl Sheet {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cells: HashMap::new(),
            col_count: 0,
            row_count: 0,
        }
    }

    /// Set a cell by zero-based (col, row).
    pub fn set_cell(&mut self, col: usize, row: usize, cell: Cell) {
        if col + 1 > self.col_count {
            self.col_count = col + 1;
        }
        if row + 1 > self.row_count {
            self.row_count = row + 1;
        }
        self.cells.insert((col, row), cell);
    }

    /// Set a cell using A1 notation.
    pub fn set(&mut self, reference: &str, cell: Cell) -> bool {
        if let Some((col, row)) = parse_cell_ref(reference) {
            self.set_cell(col, row, cell);
            true
        } else {
            false
        }
    }

    /// Get a cell by zero-based (col, row).
    pub fn get_cell(&self, col: usize, row: usize) -> Option<&Cell> {
        self.cells.get(&(col, row))
    }

    /// Get a cell using A1 notation.
    pub fn get(&self, reference: &str) -> Option<&Cell> {
        let (col, row) = parse_cell_ref(reference)?;
        self.get_cell(col, row)
    }

    /// Get the numeric value of a cell (for formula evaluation).
    pub fn numeric_value(&self, col: usize, row: usize) -> Option<f64> {
        match self.get_cell(col, row).map(|c| &c.value) {
            Some(CellValue::Number(n)) => Some(*n),
            Some(CellValue::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Collect numeric values from a range.
    fn collect_range_numbers(&self, range: &str) -> Vec<f64> {
        let Some(((c1, r1), (c2, r2))) = parse_range(range) else {
            return Vec::new();
        };
        let col_lo = c1.min(c2);
        let col_hi = c1.max(c2);
        let row_lo = r1.min(r2);
        let row_hi = r1.max(r2);

        let mut nums = Vec::new();
        for r in row_lo..=row_hi {
            for c in col_lo..=col_hi {
                if let Some(n) = self.numeric_value(c, r) {
                    nums.push(n);
                }
            }
        }
        nums
    }

    /// Count non-empty cells in a range.
    fn count_range(&self, range: &str) -> usize {
        let Some(((c1, r1), (c2, r2))) = parse_range(range) else {
            return 0;
        };
        let col_lo = c1.min(c2);
        let col_hi = c1.max(c2);
        let row_lo = r1.min(r2);
        let row_hi = r1.max(r2);

        let mut count = 0;
        for r in row_lo..=row_hi {
            for c in col_lo..=col_hi {
                if let Some(cell) = self.get_cell(c, r) {
                    if !matches!(cell.value, CellValue::Empty) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Evaluate a formula string. Supports SUM, AVERAGE, COUNT, MIN, MAX.
    pub fn eval_formula(&self, formula: &str) -> Option<f64> {
        let formula = formula.trim();
        // Strip leading '=' if present.
        let formula = formula.strip_prefix('=').unwrap_or(formula);
        let formula = formula.trim();

        // Match FUNC(RANGE)
        let open = formula.find('(')?;
        let close = formula.rfind(')')?;
        if close <= open {
            return None;
        }
        let func = formula[..open].trim().to_ascii_uppercase();
        let arg = formula[open + 1..close].trim();

        match func.as_str() {
            "SUM" => {
                let nums = self.collect_range_numbers(arg);
                Some(nums.iter().sum())
            }
            "AVERAGE" | "AVG" => {
                let nums = self.collect_range_numbers(arg);
                if nums.is_empty() {
                    None
                } else {
                    Some(nums.iter().sum::<f64>() / nums.len() as f64)
                }
            }
            "COUNT" => Some(self.count_range(arg) as f64),
            "MIN" => {
                let nums = self.collect_range_numbers(arg);
                nums.iter().copied().reduce(f64::min)
            }
            "MAX" => {
                let nums = self.collect_range_numbers(arg);
                nums.iter().copied().reduce(f64::max)
            }
            _ => None,
        }
    }

    /// Insert a row at the given index, shifting cells down.
    pub fn insert_row(&mut self, at_row: usize) {
        let keys: Vec<(usize, usize)> = self.cells.keys().copied().collect();
        let mut shifted = HashMap::new();
        for (c, r) in keys {
            let cell = self.cells.remove(&(c, r)).unwrap();
            if r >= at_row {
                shifted.insert((c, r + 1), cell);
            } else {
                shifted.insert((c, r), cell);
            }
        }
        self.cells = shifted;
        self.row_count += 1;
    }

    /// Insert a column at the given index, shifting cells right.
    pub fn insert_col(&mut self, at_col: usize) {
        let keys: Vec<(usize, usize)> = self.cells.keys().copied().collect();
        let mut shifted = HashMap::new();
        for (c, r) in keys {
            let cell = self.cells.remove(&(c, r)).unwrap();
            if c >= at_col {
                shifted.insert((c + 1, r), cell);
            } else {
                shifted.insert((c, r), cell);
            }
        }
        self.cells = shifted;
        self.col_count += 1;
    }

    /// Delete a row, shifting cells up.
    pub fn delete_row(&mut self, del_row: usize) {
        let keys: Vec<(usize, usize)> = self.cells.keys().copied().collect();
        let mut shifted = HashMap::new();
        for (c, r) in keys {
            let cell = self.cells.remove(&(c, r)).unwrap();
            if r == del_row {
                continue;
            }
            if r > del_row {
                shifted.insert((c, r - 1), cell);
            } else {
                shifted.insert((c, r), cell);
            }
        }
        self.cells = shifted;
        if self.row_count > 0 {
            self.row_count -= 1;
        }
    }

    /// Delete a column, shifting cells left.
    pub fn delete_col(&mut self, del_col: usize) {
        let keys: Vec<(usize, usize)> = self.cells.keys().copied().collect();
        let mut shifted = HashMap::new();
        for (c, r) in keys {
            let cell = self.cells.remove(&(c, r)).unwrap();
            if c == del_col {
                continue;
            }
            if c > del_col {
                shifted.insert((c - 1, r), cell);
            } else {
                shifted.insert((c, r), cell);
            }
        }
        self.cells = shifted;
        if self.col_count > 0 {
            self.col_count -= 1;
        }
    }

    /// Number of columns (based on max populated column).
    pub fn col_count(&self) -> usize {
        self.col_count
    }

    /// Number of rows (based on max populated row).
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Export the sheet as CSV.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        for r in 0..self.row_count {
            for c in 0..self.col_count {
                if c > 0 {
                    out.push(',');
                }
                if let Some(cell) = self.get_cell(c, r) {
                    let text = cell.value.to_string();
                    if text.contains(',') || text.contains('"') || text.contains('\n') {
                        out.push('"');
                        out.push_str(&text.replace('"', "\"\""));
                        out.push('"');
                    } else {
                        out.push_str(&text);
                    }
                }
            }
            out.push('\n');
        }
        out
    }
}

// ── Workbook ───────────────────────────────────────────────────

/// A workbook containing multiple named sheets.
#[derive(Debug, Clone)]
pub struct Workbook {
    pub sheets: Vec<Sheet>,
}

impl Workbook {
    pub fn new() -> Self {
        Self { sheets: Vec::new() }
    }

    pub fn add_sheet(&mut self, sheet: Sheet) {
        self.sheets.push(sheet);
    }

    pub fn sheet(&self, name: &str) -> Option<&Sheet> {
        self.sheets.iter().find(|s| s.name == name)
    }

    pub fn sheet_mut(&mut self, name: &str) -> Option<&mut Sheet> {
        self.sheets.iter_mut().find(|s| s.name == name)
    }

    pub fn sheet_count(&self) -> usize {
        self.sheets.len()
    }

    /// Remove a sheet by name, returning it if found.
    pub fn remove_sheet(&mut self, name: &str) -> Option<Sheet> {
        let idx = self.sheets.iter().position(|s| s.name == name)?;
        Some(self.sheets.remove(idx))
    }
}

impl Default for Workbook {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_a1_basic() {
        assert_eq!(parse_cell_ref("A1"), Some((0, 0)));
        assert_eq!(parse_cell_ref("B3"), Some((1, 2)));
        assert_eq!(parse_cell_ref("Z1"), Some((25, 0)));
    }

    #[test]
    fn parse_a1_multi_letter() {
        assert_eq!(parse_cell_ref("AA1"), Some((26, 0)));
        assert_eq!(parse_cell_ref("AB2"), Some((27, 1)));
    }

    #[test]
    fn parse_a1_invalid() {
        assert_eq!(parse_cell_ref(""), None);
        assert_eq!(parse_cell_ref("1A"), None);
        assert_eq!(parse_cell_ref("A0"), None);
    }

    #[test]
    fn col_row_roundtrip() {
        for col in 0..30 {
            for row in 0..5 {
                let r = col_row_to_ref(col, row);
                assert_eq!(parse_cell_ref(&r), Some((col, row)));
            }
        }
    }

    #[test]
    fn sheet_set_get() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::number(42.0));
        sheet.set("B2", Cell::text("hello"));
        assert_eq!(
            sheet.get("A1").map(|c| &c.value),
            Some(&CellValue::Number(42.0))
        );
        assert_eq!(
            sheet.get("B2").map(|c| &c.value),
            Some(&CellValue::Text("hello".into()))
        );
        assert!(sheet.get("C3").is_none());
    }

    #[test]
    fn formula_sum() {
        let mut sheet = Sheet::new("Test");
        for i in 1..=5 {
            sheet.set(&format!("A{i}"), Cell::number(i as f64 * 10.0));
        }
        assert_eq!(sheet.eval_formula("=SUM(A1:A5)"), Some(150.0));
    }

    #[test]
    fn formula_average() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::number(10.0));
        sheet.set("A2", Cell::number(20.0));
        sheet.set("A3", Cell::number(30.0));
        assert_eq!(sheet.eval_formula("=AVERAGE(A1:A3)"), Some(20.0));
    }

    #[test]
    fn formula_min_max() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::number(5.0));
        sheet.set("A2", Cell::number(2.0));
        sheet.set("A3", Cell::number(8.0));
        assert_eq!(sheet.eval_formula("=MIN(A1:A3)"), Some(2.0));
        assert_eq!(sheet.eval_formula("=MAX(A1:A3)"), Some(8.0));
    }

    #[test]
    fn formula_count() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::number(1.0));
        sheet.set("A2", Cell::text("x"));
        sheet.set("A3", Cell::empty());
        sheet.set("A4", Cell::boolean(true));
        assert_eq!(sheet.eval_formula("=COUNT(A1:A4)"), Some(3.0));
    }

    #[test]
    fn csv_export() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::text("Name"));
        sheet.set("B1", Cell::text("Value"));
        sheet.set("A2", Cell::text("alpha"));
        sheet.set("B2", Cell::number(100.0));
        let csv = sheet.to_csv();
        assert!(csv.contains("Name,Value"));
        assert!(csv.contains("alpha,100"));
    }

    #[test]
    fn insert_delete_row() {
        let mut sheet = Sheet::new("Test");
        sheet.set("A1", Cell::number(1.0));
        sheet.set("A2", Cell::number(2.0));
        sheet.set("A3", Cell::number(3.0));
        sheet.insert_row(1);
        // Row 0 unchanged, row 1 empty, row 2 has old row 1's data
        assert_eq!(sheet.numeric_value(0, 0), Some(1.0));
        assert_eq!(sheet.numeric_value(0, 1), None);
        assert_eq!(sheet.numeric_value(0, 2), Some(2.0));
        assert_eq!(sheet.numeric_value(0, 3), Some(3.0));
    }

    #[test]
    fn workbook_operations() {
        let mut wb = Workbook::new();
        wb.add_sheet(Sheet::new("Sheet1"));
        wb.add_sheet(Sheet::new("Sheet2"));
        assert_eq!(wb.sheet_count(), 2);
        assert!(wb.sheet("Sheet1").is_some());
        let removed = wb.remove_sheet("Sheet1");
        assert!(removed.is_some());
        assert_eq!(wb.sheet_count(), 1);
    }

    #[test]
    fn cell_value_display() {
        assert_eq!(CellValue::Number(3.14).to_string(), "3.14");
        assert_eq!(CellValue::Bool(true).to_string(), "TRUE");
        assert_eq!(CellValue::Empty.to_string(), "");
        assert_eq!(CellValue::Formula("SUM(A1:A3)".into()).to_string(), "=SUM(A1:A3)");
    }
}
