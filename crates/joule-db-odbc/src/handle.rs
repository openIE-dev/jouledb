//! ODBC handle management — Environment, Connection, Statement.
//!
//! Every ODBC handle is an opaque pointer. Internally we store the real state
//! in a global `HandleStore` (behind `OnceLock<Mutex<...>>`). The pointer
//! value handed back to the ODBC Driver Manager is just the map key cast to
//! `*mut c_void`.

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

use crate::types::*;

// ============================================================================
// Diagnostic record
// ============================================================================

/// A single ODBC diagnostic record (SQLSTATE + native error + message).
#[derive(Debug, Clone)]
pub struct DiagRecord {
    /// 5-character SQLSTATE code (e.g. "HY000", "08001").
    pub state: String,
    /// Driver-specific native error code.
    pub native_error: SQLINTEGER,
    /// Human-readable message.
    pub message: String,
}

impl DiagRecord {
    pub fn new(state: &str, native_error: SQLINTEGER, message: impl Into<String>) -> Self {
        Self {
            state: state.to_string(),
            native_error,
            message: message.into(),
        }
    }

    /// Convenience: general error with SQLSTATE HY000.
    pub fn general(message: impl Into<String>) -> Self {
        Self::new("HY000", 0, message)
    }

    /// Connection failure — SQLSTATE 08001.
    pub fn connection_failure(message: impl Into<String>) -> Self {
        Self::new("08001", 0, message)
    }

    /// Invalid handle — SQLSTATE HY092.
    pub fn invalid_attr(message: impl Into<String>) -> Self {
        Self::new("HY092", 0, message)
    }

    /// Function sequence error — SQLSTATE HY010.
    pub fn sequence_error(message: impl Into<String>) -> Self {
        Self::new("HY010", 0, message)
    }

    /// Invalid cursor state — SQLSTATE 24000.
    pub fn invalid_cursor(message: impl Into<String>) -> Self {
        Self::new("24000", 0, message)
    }

    /// Invalid descriptor index — SQLSTATE 07009.
    pub fn invalid_column(message: impl Into<String>) -> Self {
        Self::new("07009", 0, message)
    }
}

// ============================================================================
// Column metadata
// ============================================================================

/// Metadata for a single result column.
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    /// Column name as returned by the server.
    pub name: String,
    /// ODBC SQL data type constant.
    pub sql_type: SQLSMALLINT,
    /// Column display/storage size.
    pub column_size: SQLULEN,
    /// Number of decimal digits (0 for non-numeric).
    pub decimal_digits: SQLSMALLINT,
    /// Whether the column is nullable.
    pub nullable: SQLSMALLINT,
}

// ============================================================================
// Column binding (for SQLBindCol)
// ============================================================================

/// A binding registered via `SQLBindCol`.
#[derive(Debug, Clone)]
pub struct ColumnBinding {
    /// 1-based column number.
    pub column_number: SQLUSMALLINT,
    /// C target type.
    pub target_type: SQLSMALLINT,
    /// Pointer to the application buffer.
    pub target_value: SQLPOINTER,
    /// Size of the application buffer.
    pub buffer_length: SQLLEN,
    /// Pointer to the length/indicator value.
    pub str_len_or_ind: *mut SQLLEN,
}

// SAFETY: The raw pointers in ColumnBinding point to application-owned memory
// that the ODBC contract guarantees will remain valid for the lifetime of the
// binding. We need Send+Sync so we can store bindings in the global handle map.
unsafe impl Send for ColumnBinding {}
unsafe impl Sync for ColumnBinding {}

// ============================================================================
// Environment
// ============================================================================

/// ODBC environment state.
#[derive(Debug)]
pub struct Environment {
    /// ODBC version set by the application (SQL_OV_ODBC3 etc.).
    pub odbc_version: SQLINTEGER,
    /// Diagnostic records.
    pub diagnostics: Vec<DiagRecord>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            odbc_version: 0,
            diagnostics: Vec::new(),
        }
    }
}

// ============================================================================
// Connection
// ============================================================================

/// ODBC connection state — wraps a TCP socket to JouleDB.
#[derive(Debug)]
pub struct Connection {
    /// Parent environment handle key.
    pub env_handle: usize,
    /// Host name / IP address.
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Database name.
    pub database: String,
    /// Username for authentication.
    pub username: String,
    /// Password for authentication.
    pub password: String,
    /// Active TCP connection (None until SQLConnect / SQLDriverConnect).
    pub stream: Option<TcpStream>,
    /// Whether we are currently connected.
    pub connected: bool,
    /// Monotonic request-ID counter for the wire protocol.
    pub next_request_id: u32,
    /// Diagnostic records.
    pub diagnostics: Vec<DiagRecord>,
}

impl Connection {
    pub fn new(env_handle: usize) -> Self {
        Self {
            env_handle,
            host: String::new(),
            port: 9000,
            database: "default".to_string(),
            username: String::new(),
            password: String::new(),
            stream: None,
            connected: false,
            next_request_id: 1,
            diagnostics: Vec::new(),
        }
    }

    /// Allocate the next request ID.
    pub fn alloc_request_id(&mut self) -> u32 {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        id
    }
}

// ============================================================================
// Statement
// ============================================================================

/// ODBC statement state.
#[derive(Debug)]
pub struct Statement {
    /// Parent connection handle key.
    pub conn_handle: usize,
    /// The SQL text most recently executed.
    pub sql: Option<String>,
    /// Column metadata for the current result set.
    pub columns: Vec<ColumnMeta>,
    /// Row data — outer vec is rows, inner vec is cell values (as strings).
    /// `None` means SQL NULL.
    pub rows: Vec<Vec<Option<String>>>,
    /// Current row cursor position (0-based; `usize::MAX` means "before first row").
    pub cursor: usize,
    /// Number of rows affected by the last DML statement (INSERT/UPDATE/DELETE).
    pub affected_rows: Option<usize>,
    /// Column bindings set by SQLBindCol.
    pub bindings: Vec<ColumnBinding>,
    /// Whether a result set is open (i.e., cursor is active).
    pub has_result_set: bool,
    /// Diagnostic records.
    pub diagnostics: Vec<DiagRecord>,
}

impl Statement {
    pub fn new(conn_handle: usize) -> Self {
        Self {
            conn_handle,
            sql: None,
            columns: Vec::new(),
            rows: Vec::new(),
            cursor: usize::MAX,
            affected_rows: None,
            bindings: Vec::new(),
            has_result_set: false,
            diagnostics: Vec::new(),
        }
    }

    /// Reset the result set state for a new execution.
    pub fn reset_results(&mut self) {
        self.columns.clear();
        self.rows.clear();
        self.cursor = usize::MAX;
        self.affected_rows = None;
        self.has_result_set = false;
    }

    /// Close the cursor without clearing the result set metadata.
    pub fn close_cursor(&mut self) {
        self.cursor = usize::MAX;
        self.has_result_set = false;
    }
}

// ============================================================================
// Handle kind enum
// ============================================================================

/// Tagged union for the three handle kinds.
#[derive(Debug)]
pub enum Handle {
    Env(Environment),
    Dbc(Connection),
    Stmt(Statement),
}

// ============================================================================
// Global handle store
// ============================================================================

/// Thread-safe global storage for all ODBC handles.
///
/// Keys are `usize` values that get cast to/from `SQLHANDLE` (`*mut c_void`).
/// We start from 1 so that 0 (null pointer) is never a valid handle.
pub struct HandleStore {
    handles: HashMap<usize, Handle>,
    next_id: usize,
}

impl HandleStore {
    fn new() -> Self {
        Self {
            handles: HashMap::new(),
            next_id: 1,
        }
    }

    /// Insert a handle and return its key.
    pub fn insert(&mut self, handle: Handle) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(id, handle);
        id
    }

    /// Remove a handle by key. Returns the removed handle, if any.
    pub fn remove(&mut self, key: usize) -> Option<Handle> {
        self.handles.remove(&key)
    }

    /// Get a shared reference to a handle.
    pub fn get(&self, key: usize) -> Option<&Handle> {
        self.handles.get(&key)
    }

    /// Get an exclusive reference to a handle.
    pub fn get_mut(&mut self, key: usize) -> Option<&mut Handle> {
        self.handles.get_mut(&key)
    }

    /// Remove all statement handles that belong to a given connection.
    pub fn remove_stmts_for_connection(&mut self, conn_key: usize) -> Vec<usize> {
        let keys: Vec<usize> = self
            .handles
            .iter()
            .filter_map(|(&k, h)| match h {
                Handle::Stmt(s) if s.conn_handle == conn_key => Some(k),
                _ => None,
            })
            .collect();
        for &k in &keys {
            self.handles.remove(&k);
        }
        keys
    }

    /// Remove all connection handles (and their statements) that belong to a
    /// given environment.
    pub fn remove_connections_for_env(&mut self, env_key: usize) -> Vec<usize> {
        let conn_keys: Vec<usize> = self
            .handles
            .iter()
            .filter_map(|(&k, h)| match h {
                Handle::Dbc(c) if c.env_handle == env_key => Some(k),
                _ => None,
            })
            .collect();
        for &ck in &conn_keys {
            self.remove_stmts_for_connection(ck);
            self.handles.remove(&ck);
        }
        conn_keys
    }
}

/// Global singleton.
fn global_store() -> &'static Mutex<HandleStore> {
    static STORE: OnceLock<Mutex<HandleStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HandleStore::new()))
}

/// Lock the global handle store. Panics on poisoned mutex (unrecoverable).
pub fn lock_store() -> std::sync::MutexGuard<'static, HandleStore> {
    global_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ============================================================================
// Key <-> SQLHANDLE conversion helpers
// ============================================================================

/// Convert a `usize` key to an `SQLHANDLE`.
pub fn key_to_handle(key: usize) -> SQLHANDLE {
    key as *mut std::ffi::c_void
}

/// Convert an `SQLHANDLE` back to a `usize` key.
pub fn handle_to_key(handle: SQLHANDLE) -> usize {
    handle as usize
}

// ============================================================================
// Diagnostic helpers
// ============================================================================

/// Push a diagnostic record onto the appropriate handle.
pub fn push_diag(store: &mut HandleStore, key: usize, rec: DiagRecord) {
    if let Some(handle) = store.get_mut(key) {
        match handle {
            Handle::Env(e) => e.diagnostics.push(rec),
            Handle::Dbc(c) => c.diagnostics.push(rec),
            Handle::Stmt(s) => s.diagnostics.push(rec),
        }
    }
}

/// Clear all diagnostic records on a handle.
pub fn clear_diag(store: &mut HandleStore, key: usize) {
    if let Some(handle) = store.get_mut(key) {
        match handle {
            Handle::Env(e) => e.diagnostics.clear(),
            Handle::Dbc(c) => c.diagnostics.clear(),
            Handle::Stmt(s) => s.diagnostics.clear(),
        }
    }
}

/// Read a diagnostic record by 1-based index. Returns `None` if out of range.
pub fn get_diag(store: &HandleStore, key: usize, rec_number: usize) -> Option<DiagRecord> {
    store.get(key).and_then(|handle| {
        let diags = match handle {
            Handle::Env(e) => &e.diagnostics,
            Handle::Dbc(c) => &c.diagnostics,
            Handle::Stmt(s) => &s.diagnostics,
        };
        if rec_number == 0 || rec_number > diags.len() {
            None
        } else {
            Some(diags[rec_number - 1].clone())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_store_insert_and_get() {
        let mut store = HandleStore::new();
        let key = store.insert(Handle::Env(Environment::new()));
        assert!(store.get(key).is_some());
        assert!(matches!(store.get(key), Some(Handle::Env(_))));
    }

    #[test]
    fn handle_store_remove() {
        let mut store = HandleStore::new();
        let key = store.insert(Handle::Env(Environment::new()));
        assert!(store.remove(key).is_some());
        assert!(store.get(key).is_none());
    }

    #[test]
    fn key_handle_roundtrip() {
        let key: usize = 42;
        let handle = key_to_handle(key);
        assert_eq!(handle_to_key(handle), key);
    }

    #[test]
    fn diag_push_and_get() {
        let mut store = HandleStore::new();
        let key = store.insert(Handle::Env(Environment::new()));
        push_diag(&mut store, key, DiagRecord::general("test error"));
        let rec = get_diag(&store, key, 1);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert_eq!(rec.state, "HY000");
        assert_eq!(rec.message, "test error");
    }

    #[test]
    fn diag_clear() {
        let mut store = HandleStore::new();
        let key = store.insert(Handle::Env(Environment::new()));
        push_diag(&mut store, key, DiagRecord::general("err"));
        clear_diag(&mut store, key);
        assert!(get_diag(&store, key, 1).is_none());
    }

    #[test]
    fn remove_stmts_for_connection() {
        let mut store = HandleStore::new();
        let env_key = store.insert(Handle::Env(Environment::new()));
        let conn_key = store.insert(Handle::Dbc(Connection::new(env_key)));
        let _s1 = store.insert(Handle::Stmt(Statement::new(conn_key)));
        let _s2 = store.insert(Handle::Stmt(Statement::new(conn_key)));
        let removed = store.remove_stmts_for_connection(conn_key);
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn connection_alloc_request_id() {
        let mut conn = Connection::new(1);
        assert_eq!(conn.alloc_request_id(), 1);
        assert_eq!(conn.alloc_request_id(), 2);
        assert_eq!(conn.alloc_request_id(), 3);
    }

    #[test]
    fn statement_reset_results() {
        let mut stmt = Statement::new(1);
        stmt.columns.push(ColumnMeta {
            name: "id".to_string(),
            sql_type: SQL_INTEGER,
            column_size: 10,
            decimal_digits: 0,
            nullable: SQL_NULLABLE,
        });
        stmt.rows.push(vec![Some("1".to_string())]);
        stmt.has_result_set = true;
        stmt.cursor = 0;

        stmt.reset_results();
        assert!(stmt.columns.is_empty());
        assert!(stmt.rows.is_empty());
        assert_eq!(stmt.cursor, usize::MAX);
        assert!(!stmt.has_result_set);
    }
}
