//! Database cursor abstraction — forward/backward cursor, seekable, buffered
//! fetch (batch size), cursor state (open/exhausted/closed), column access
//! by name/index, result set metadata.
//!
//! Replaces ad-hoc iteration patterns over result sets with a proper
//! database cursor that supports navigation, buffering, and metadata.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by cursor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CursorError {
    /// Cursor is closed and cannot be used.
    CursorClosed,
    /// Cursor is exhausted (past the last row).
    CursorExhausted,
    /// Column not found by name.
    ColumnNotFound(String),
    /// Column index out of bounds.
    ColumnOutOfBounds(usize),
    /// Cursor not positioned on a valid row.
    NotOnRow,
    /// Backward navigation not supported.
    BackwardNotSupported,
    /// Invalid batch size.
    InvalidBatchSize,
}

impl fmt::Display for CursorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CursorClosed => write!(f, "cursor is closed"),
            Self::CursorExhausted => write!(f, "cursor exhausted"),
            Self::ColumnNotFound(name) => write!(f, "column not found: {name}"),
            Self::ColumnOutOfBounds(idx) => write!(f, "column index {idx} out of bounds"),
            Self::NotOnRow => write!(f, "cursor not positioned on a row"),
            Self::BackwardNotSupported => write!(f, "backward navigation not supported"),
            Self::InvalidBatchSize => write!(f, "invalid batch size"),
        }
    }
}

impl std::error::Error for CursorError {}

// ── Value type ───────────────────────────────────────────────────

/// A cell value in a result set row.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

/// A row is a vector of values.
pub type Row = Vec<Value>;

// ── Cursor state ─────────────────────────────────────────────────

/// The state of a cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorState {
    /// Cursor is open and ready to navigate.
    Open,
    /// Cursor has passed the last row.
    Exhausted,
    /// Cursor is closed and cannot be used.
    Closed,
}

impl fmt::Display for CursorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "OPEN"),
            Self::Exhausted => write!(f, "EXHAUSTED"),
            Self::Closed => write!(f, "CLOSED"),
        }
    }
}

// ── Column metadata ──────────────────────────────────────────────

/// Data type hint for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Float,
    Text,
    Boolean,
    Unknown,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer => write!(f, "INTEGER"),
            Self::Float => write!(f, "FLOAT"),
            Self::Text => write!(f, "TEXT"),
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Metadata for a single column.
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub column_type: ColumnType,
    pub nullable: bool,
    pub position: usize,
}

/// Metadata for the entire result set.
#[derive(Debug, Clone)]
pub struct ResultSetMeta {
    pub columns: Vec<ColumnMeta>,
    pub total_rows: usize,
    /// Column name -> position index.
    name_map: HashMap<String, usize>,
}

impl ResultSetMeta {
    /// Create metadata from column definitions.
    pub fn new(columns: Vec<ColumnMeta>, total_rows: usize) -> Self {
        let name_map: HashMap<String, usize> = columns
            .iter()
            .map(|c| (c.name.clone(), c.position))
            .collect();
        Self {
            columns,
            total_rows,
            name_map,
        }
    }

    /// Look up column position by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.name_map.get(name).copied()
    }

    /// Number of columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

// ── Navigation direction ─────────────────────────────────────────

/// Cursor navigation capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorDirection {
    /// Forward-only cursor.
    ForwardOnly,
    /// Scrollable (forward + backward).
    Scrollable,
}

// ── Cursor ───────────────────────────────────────────────────────

/// A database cursor over a result set.
pub struct Cursor {
    /// All rows in the result set.
    rows: Vec<Row>,
    /// Column metadata.
    meta: ResultSetMeta,
    /// Current position (-1 = before first, rows.len() = after last).
    position: isize,
    /// Cursor state.
    state: CursorState,
    /// Navigation direction.
    direction: CursorDirection,
    /// Batch size for fetch_batch.
    batch_size: usize,
    /// Number of fetch calls.
    fetch_count: u64,
}

impl Cursor {
    /// Create a new cursor over a result set.
    pub fn new(
        rows: Vec<Row>,
        meta: ResultSetMeta,
        direction: CursorDirection,
        batch_size: usize,
    ) -> Self {
        Self {
            state: if rows.is_empty() {
                CursorState::Exhausted
            } else {
                CursorState::Open
            },
            rows,
            meta,
            position: -1, // before first row
            direction,
            batch_size: if batch_size == 0 { 1 } else { batch_size },
            fetch_count: 0,
        }
    }

    /// Create a forward-only cursor with default batch size.
    pub fn forward(rows: Vec<Row>, meta: ResultSetMeta) -> Self {
        Self::new(rows, meta, CursorDirection::ForwardOnly, 1)
    }

    /// Create a scrollable cursor.
    pub fn scrollable(rows: Vec<Row>, meta: ResultSetMeta) -> Self {
        Self::new(rows, meta, CursorDirection::Scrollable, 1)
    }

    /// Get the cursor state.
    pub fn state(&self) -> CursorState {
        self.state
    }

    /// Get the result set metadata.
    pub fn metadata(&self) -> &ResultSetMeta {
        &self.meta
    }

    /// Get the current row position (0-based). Returns None if not on a row.
    pub fn position(&self) -> Option<usize> {
        if self.position >= 0 && (self.position as usize) < self.rows.len() {
            Some(self.position as usize)
        } else {
            None
        }
    }

    /// Total number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of fetch operations performed.
    pub fn fetch_count(&self) -> u64 {
        self.fetch_count
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set the batch size.
    pub fn set_batch_size(&mut self, size: usize) -> Result<(), CursorError> {
        if size == 0 {
            return Err(CursorError::InvalidBatchSize);
        }
        self.batch_size = size;
        Ok(())
    }

    // ── Navigation ──

    /// Move to the next row. Returns true if positioned on a valid row.
    pub fn next(&mut self) -> Result<bool, CursorError> {
        self.check_not_closed()?;
        self.position += 1;
        self.fetch_count += 1;
        if self.position as usize >= self.rows.len() {
            self.state = CursorState::Exhausted;
            Ok(false)
        } else {
            self.state = CursorState::Open;
            Ok(true)
        }
    }

    /// Move to the previous row. Returns true if positioned on a valid row.
    pub fn previous(&mut self) -> Result<bool, CursorError> {
        self.check_not_closed()?;
        if self.direction == CursorDirection::ForwardOnly {
            return Err(CursorError::BackwardNotSupported);
        }
        self.position -= 1;
        self.fetch_count += 1;
        if self.position < 0 {
            Ok(false)
        } else {
            self.state = CursorState::Open;
            Ok(true)
        }
    }

    /// Seek to an absolute row position (0-based).
    pub fn seek(&mut self, pos: usize) -> Result<bool, CursorError> {
        self.check_not_closed()?;
        if self.direction == CursorDirection::ForwardOnly && (pos as isize) < self.position {
            return Err(CursorError::BackwardNotSupported);
        }
        self.position = pos as isize;
        if pos < self.rows.len() {
            self.state = CursorState::Open;
            Ok(true)
        } else {
            self.state = CursorState::Exhausted;
            Ok(false)
        }
    }

    /// Move to the first row.
    pub fn first(&mut self) -> Result<bool, CursorError> {
        self.check_not_closed()?;
        if self.direction == CursorDirection::ForwardOnly && self.position > 0 {
            return Err(CursorError::BackwardNotSupported);
        }
        self.position = 0;
        if self.rows.is_empty() {
            self.state = CursorState::Exhausted;
            Ok(false)
        } else {
            self.state = CursorState::Open;
            Ok(true)
        }
    }

    /// Move past the last row.
    pub fn last(&mut self) -> Result<bool, CursorError> {
        self.check_not_closed()?;
        if self.rows.is_empty() {
            self.state = CursorState::Exhausted;
            return Ok(false);
        }
        self.position = (self.rows.len() - 1) as isize;
        self.state = CursorState::Open;
        Ok(true)
    }

    // ── Data access ──

    /// Get the current row.
    pub fn current_row(&self) -> Result<&Row, CursorError> {
        self.check_not_closed()?;
        self.check_on_row()?;
        Ok(&self.rows[self.position as usize])
    }

    /// Get a column value by index.
    pub fn get_by_index(&self, col: usize) -> Result<&Value, CursorError> {
        let row = self.current_row()?;
        if col >= row.len() {
            return Err(CursorError::ColumnOutOfBounds(col));
        }
        Ok(&row[col])
    }

    /// Get a column value by name.
    pub fn get_by_name(&self, name: &str) -> Result<&Value, CursorError> {
        let idx = self
            .meta
            .column_index(name)
            .ok_or_else(|| CursorError::ColumnNotFound(name.to_string()))?;
        self.get_by_index(idx)
    }

    /// Fetch a batch of rows starting from the current position.
    /// Advances the cursor past the fetched rows.
    pub fn fetch_batch(&mut self) -> Result<Vec<Row>, CursorError> {
        self.check_not_closed()?;
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.next()? {
                true => batch.push(self.rows[self.position as usize].clone()),
                false => break,
            }
        }
        Ok(batch)
    }

    /// Fetch all remaining rows.
    pub fn fetch_all(&mut self) -> Result<Vec<Row>, CursorError> {
        self.check_not_closed()?;
        let mut all = Vec::new();
        while self.next()? {
            all.push(self.rows[self.position as usize].clone());
        }
        Ok(all)
    }

    /// Close the cursor.
    pub fn close(&mut self) {
        self.state = CursorState::Closed;
    }

    /// Whether the cursor is on a valid row.
    pub fn is_on_row(&self) -> bool {
        self.position >= 0 && (self.position as usize) < self.rows.len()
    }

    // ── Helpers ──

    fn check_not_closed(&self) -> Result<(), CursorError> {
        if self.state == CursorState::Closed {
            Err(CursorError::CursorClosed)
        } else {
            Ok(())
        }
    }

    fn check_on_row(&self) -> Result<(), CursorError> {
        if !self.is_on_row() {
            Err(CursorError::NotOnRow)
        } else {
            Ok(())
        }
    }
}

// ── Builder helper ───────────────────────────────────────────────

/// Build result set metadata from column name/type pairs.
pub fn build_meta(
    columns: &[(&str, ColumnType)],
    total_rows: usize,
) -> ResultSetMeta {
    let cols: Vec<ColumnMeta> = columns
        .iter()
        .enumerate()
        .map(|(i, (name, ct))| ColumnMeta {
            name: name.to_string(),
            column_type: *ct,
            nullable: true,
            position: i,
        })
        .collect();
    ResultSetMeta::new(cols, total_rows)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> ResultSetMeta {
        build_meta(
            &[
                ("id", ColumnType::Integer),
                ("name", ColumnType::Text),
                ("score", ColumnType::Float),
            ],
            3,
        )
    }

    fn sample_rows() -> Vec<Row> {
        vec![
            vec![Value::Int(1), Value::Text("Alice".into()), Value::Float(95.5)],
            vec![Value::Int(2), Value::Text("Bob".into()), Value::Float(88.0)],
            vec![Value::Int(3), Value::Text("Carol".into()), Value::Float(92.3)],
        ]
    }

    #[test]
    fn new_cursor_state() {
        let cursor = Cursor::forward(sample_rows(), sample_meta());
        assert_eq!(cursor.state(), CursorState::Open);
        assert_eq!(cursor.row_count(), 3);
        assert!(!cursor.is_on_row());
    }

    #[test]
    fn empty_cursor_is_exhausted() {
        let cursor = Cursor::forward(vec![], sample_meta());
        assert_eq!(cursor.state(), CursorState::Exhausted);
    }

    #[test]
    fn next_navigates_forward() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        assert!(cursor.next().unwrap());
        assert_eq!(cursor.position(), Some(0));
        assert!(cursor.next().unwrap());
        assert_eq!(cursor.position(), Some(1));
        assert!(cursor.next().unwrap());
        assert_eq!(cursor.position(), Some(2));
        assert!(!cursor.next().unwrap());
        assert_eq!(cursor.state(), CursorState::Exhausted);
    }

    #[test]
    fn get_by_index() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        assert_eq!(cursor.get_by_index(0).unwrap(), &Value::Int(1));
        assert_eq!(
            cursor.get_by_index(1).unwrap(),
            &Value::Text("Alice".into())
        );
    }

    #[test]
    fn get_by_name() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        assert_eq!(
            cursor.get_by_name("name").unwrap(),
            &Value::Text("Alice".into())
        );
    }

    #[test]
    fn get_by_name_not_found() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        let err = cursor.get_by_name("nonexistent").unwrap_err();
        assert_eq!(
            err,
            CursorError::ColumnNotFound("nonexistent".into())
        );
    }

    #[test]
    fn get_by_index_out_of_bounds() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        let err = cursor.get_by_index(99).unwrap_err();
        assert_eq!(err, CursorError::ColumnOutOfBounds(99));
    }

    #[test]
    fn not_on_row_error() {
        let cursor = Cursor::forward(sample_rows(), sample_meta());
        let err = cursor.current_row().unwrap_err();
        assert_eq!(err, CursorError::NotOnRow);
    }

    #[test]
    fn close_and_access_fails() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.close();
        let err = cursor.next().unwrap_err();
        assert_eq!(err, CursorError::CursorClosed);
    }

    #[test]
    fn scrollable_previous() {
        let mut cursor = Cursor::scrollable(sample_rows(), sample_meta());
        cursor.next().unwrap();
        cursor.next().unwrap();
        assert_eq!(cursor.position(), Some(1));
        cursor.previous().unwrap();
        assert_eq!(cursor.position(), Some(0));
    }

    #[test]
    fn forward_only_rejects_previous() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        let err = cursor.previous().unwrap_err();
        assert_eq!(err, CursorError::BackwardNotSupported);
    }

    #[test]
    fn seek_to_position() {
        let mut cursor = Cursor::scrollable(sample_rows(), sample_meta());
        assert!(cursor.seek(2).unwrap());
        assert_eq!(cursor.position(), Some(2));
        assert_eq!(cursor.get_by_index(0).unwrap(), &Value::Int(3));
    }

    #[test]
    fn seek_past_end() {
        let mut cursor = Cursor::scrollable(sample_rows(), sample_meta());
        assert!(!cursor.seek(100).unwrap());
        assert_eq!(cursor.state(), CursorState::Exhausted);
    }

    #[test]
    fn first_and_last() {
        let mut cursor = Cursor::scrollable(sample_rows(), sample_meta());
        assert!(cursor.last().unwrap());
        assert_eq!(cursor.position(), Some(2));
        assert!(cursor.first().unwrap());
        assert_eq!(cursor.position(), Some(0));
    }

    #[test]
    fn fetch_batch() {
        let mut cursor = Cursor::new(
            sample_rows(),
            sample_meta(),
            CursorDirection::ForwardOnly,
            2,
        );
        let batch = cursor.fetch_batch().unwrap();
        assert_eq!(batch.len(), 2);
        let batch2 = cursor.fetch_batch().unwrap();
        assert_eq!(batch2.len(), 1);
        let batch3 = cursor.fetch_batch().unwrap();
        assert!(batch3.is_empty());
    }

    #[test]
    fn fetch_all() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        let all = cursor.fetch_all().unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(cursor.state(), CursorState::Exhausted);
    }

    #[test]
    fn set_batch_size() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.set_batch_size(5).unwrap();
        assert_eq!(cursor.batch_size(), 5);
    }

    #[test]
    fn set_batch_size_zero_errors() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        let err = cursor.set_batch_size(0).unwrap_err();
        assert_eq!(err, CursorError::InvalidBatchSize);
    }

    #[test]
    fn metadata_column_count() {
        let meta = sample_meta();
        assert_eq!(meta.column_count(), 3);
    }

    #[test]
    fn metadata_column_index() {
        let meta = sample_meta();
        assert_eq!(meta.column_index("name"), Some(1));
        assert_eq!(meta.column_index("missing"), None);
    }

    #[test]
    fn current_row_returns_full_row() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        cursor.next().unwrap();
        let row = cursor.current_row().unwrap();
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn fetch_count_tracked() {
        let mut cursor = Cursor::forward(sample_rows(), sample_meta());
        assert_eq!(cursor.fetch_count(), 0);
        cursor.next().unwrap();
        cursor.next().unwrap();
        assert_eq!(cursor.fetch_count(), 2);
    }

    #[test]
    fn scrollable_seek_backward() {
        let mut cursor = Cursor::scrollable(sample_rows(), sample_meta());
        cursor.seek(2).unwrap();
        cursor.seek(0).unwrap();
        assert_eq!(cursor.position(), Some(0));
    }
}
