#![cfg_attr(not(test), deny(clippy::unwrap_used))]
//! JouleDB C Foreign Function Interface
//!
//! Provides a SQLite-style C API for embedding JouleDB in any language.
//!
//! ## API Style
//!
//! Follows the SQLite prepare/bind/step/column pattern:
//!
//! ```c
//! JouleDB *db = joule_db_open("my.db");
//! joule_db_exec(db, "CREATE TABLE users (id INT, name TEXT)", NULL);
//!
//! JouleDBStmt *stmt = joule_db_prepare(db, "SELECT * FROM users WHERE id = $1");
//! joule_db_bind_int(stmt, 1, 42);
//! while (joule_db_step(stmt) == JOULE_ROW) {
//!     int id = joule_db_column_int(stmt, 0);
//!     const char *name = joule_db_column_text(stmt, 1);
//! }
//! joule_db_finalize(stmt);
//! joule_db_close(db);
//! ```

use joule_db_local::Database;
use joule_db_query::ast::Value as AstValue;
use joule_db_query::{execution::QueryContext, storage_executor::StorageExecutor};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use std::sync::{Arc, Mutex};

/// Result codes
pub const JOULE_OK: c_int = 0;
pub const JOULE_ERROR: c_int = 1;
pub const JOULE_ROW: c_int = 100;
pub const JOULE_DONE: c_int = 101;

// ============================================================================
// Database Handle
// ============================================================================

/// Opaque database handle
pub struct JouleDBHandle {
    db: Arc<Database>,
    executor: Mutex<StorageExecutor>,
    last_error: Mutex<Option<String>>,
}

/// Open a database connection
///
/// Returns a pointer to the database handle, or NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_open(path: *const c_char) -> *mut JouleDBHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match Database::open(path_str) {
        Ok(db) => {
            let db_arc = Arc::new(db);
            let engine = db_arc.engine();
            let executor = StorageExecutor::new(engine);

            let handle = Box::new(JouleDBHandle {
                db: db_arc,
                executor: Mutex::new(executor),
                last_error: Mutex::new(None),
            });
            Box::into_raw(handle)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Close a database connection and free all resources
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_close(handle: *mut JouleDBHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// Execute a SQL statement directly (no result set)
///
/// Returns JOULE_OK (0) on success, non-zero on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_exec(
    handle: *mut JouleDBHandle,
    sql: *const c_char,
    err_msg: *mut *mut c_char,
) -> c_int {
    if handle.is_null() || sql.is_null() {
        return JOULE_ERROR;
    }

    let c_sql = unsafe { CStr::from_ptr(sql) };
    let sql_str = match c_sql.to_str() {
        Ok(s) => s,
        Err(_) => return JOULE_ERROR,
    };

    let db_ref = unsafe { &*handle };
    let mut executor = match db_ref.executor.lock() {
        Ok(guard) => guard,
        Err(_) => return JOULE_ERROR,
    };

    let context = QueryContext::default();
    match executor.execute_sql(sql_str, &context) {
        Ok(_result) => JOULE_OK,
        Err(e) => {
            let msg = e.to_string();
            if let Ok(mut last_err) = db_ref.last_error.lock() {
                *last_err = Some(msg.clone());
            }
            if !err_msg.is_null() {
                let s = CString::new(msg).unwrap_or_default();
                unsafe {
                    *err_msg = s.into_raw();
                }
            }
            JOULE_ERROR
        }
    }
}

/// Get the last error message
///
/// Returns NULL if no error. The returned string is valid until the next API call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_errmsg(handle: *mut JouleDBHandle) -> *const c_char {
    if handle.is_null() {
        return std::ptr::null();
    }

    let db_ref = unsafe { &*handle };
    match db_ref.last_error.lock() {
        Ok(err) => match &*err {
            Some(msg) => match CString::new(msg.clone()) {
                Ok(s) => s.into_raw() as *const c_char,
                Err(_) => std::ptr::null(),
            },
            None => std::ptr::null(),
        },
        Err(_) => std::ptr::null(),
    }
}

// ============================================================================
// Prepared Statement Handle
// ============================================================================

/// Opaque prepared statement handle
pub struct JouleDBStmt {
    /// SQL text
    sql: String,
    /// Bound parameters (index 0 = $1)
    params: Vec<AstValue>,
    /// Result columns (after step)
    columns: Vec<String>,
    /// Result rows (after step)
    rows: Vec<Vec<AstValue>>,
    /// Current row cursor (-1 = before first row)
    cursor: i64,
    /// Number of affected rows (for INSERT/UPDATE/DELETE)
    affected_rows: usize,
    /// Reference to parent database
    db: *mut JouleDBHandle,
}

/// Prepare a SQL statement for execution
///
/// Returns a statement handle, or NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_prepare(
    handle: *mut JouleDBHandle,
    sql: *const c_char,
) -> *mut JouleDBStmt {
    if handle.is_null() || sql.is_null() {
        return std::ptr::null_mut();
    }

    let c_sql = unsafe { CStr::from_ptr(sql) };
    let sql_str = match c_sql.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let param_count = count_params(sql_str);

    let stmt = Box::new(JouleDBStmt {
        sql: sql_str.to_string(),
        params: vec![AstValue::Null; param_count],
        columns: Vec::new(),
        rows: Vec::new(),
        cursor: -1,
        affected_rows: 0,
        db: handle,
    });
    Box::into_raw(stmt)
}

/// Bind an integer value to parameter (1-based index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_bind_int(
    stmt: *mut JouleDBStmt,
    index: c_int,
    value: i64,
) -> c_int {
    if stmt.is_null() || index < 1 {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };
    let idx = (index - 1) as usize;
    if idx >= stmt_ref.params.len() {
        return JOULE_ERROR;
    }

    stmt_ref.params[idx] = AstValue::Int(value);
    JOULE_OK
}

/// Bind a double value to parameter (1-based index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_bind_double(
    stmt: *mut JouleDBStmt,
    index: c_int,
    value: c_double,
) -> c_int {
    if stmt.is_null() || index < 1 {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };
    let idx = (index - 1) as usize;
    if idx >= stmt_ref.params.len() {
        return JOULE_ERROR;
    }

    stmt_ref.params[idx] = AstValue::Float(value);
    JOULE_OK
}

/// Bind a text value to parameter (1-based index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_bind_text(
    stmt: *mut JouleDBStmt,
    index: c_int,
    value: *const c_char,
) -> c_int {
    if stmt.is_null() || index < 1 || value.is_null() {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };
    let idx = (index - 1) as usize;
    if idx >= stmt_ref.params.len() {
        return JOULE_ERROR;
    }

    let c_str = unsafe { CStr::from_ptr(value) };
    match c_str.to_str() {
        Ok(s) => {
            stmt_ref.params[idx] = AstValue::String(s.to_string());
            JOULE_OK
        }
        Err(_) => JOULE_ERROR,
    }
}

/// Bind NULL to parameter (1-based index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_bind_null(stmt: *mut JouleDBStmt, index: c_int) -> c_int {
    if stmt.is_null() || index < 1 {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };
    let idx = (index - 1) as usize;
    if idx >= stmt_ref.params.len() {
        return JOULE_ERROR;
    }

    stmt_ref.params[idx] = AstValue::Null;
    JOULE_OK
}

/// Step through query results
///
/// Returns:
/// - JOULE_ROW (100) if a row is available
/// - JOULE_DONE (101) if no more rows
/// - JOULE_ERROR (1) on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_step(stmt: *mut JouleDBStmt) -> c_int {
    if stmt.is_null() {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };

    // First call executes the query
    if stmt_ref.cursor == -1 {
        let db_ref = unsafe { &*stmt_ref.db };
        let mut executor = match db_ref.executor.lock() {
            Ok(guard) => guard,
            Err(_) => return JOULE_ERROR,
        };

        let context = QueryContext::default();

        // Substitute parameters into SQL (simple $N replacement)
        let mut sql = stmt_ref.sql.clone();
        for (i, param) in stmt_ref.params.iter().enumerate().rev() {
            let placeholder = format!("${}", i + 1);
            let replacement = match param {
                AstValue::Null => "NULL".to_string(),
                AstValue::Bool(b) => b.to_string(),
                AstValue::Int(n) => n.to_string(),
                AstValue::Float(f) => f.to_string(),
                AstValue::String(s) => format!("'{}'", s.replace('\'', "''")),
                AstValue::Bytes(b) => format!(
                    "X'{}'",
                    b.iter()
                        .map(|byte| format!("{:02x}", byte))
                        .collect::<String>()
                ),
                AstValue::Timestamp(ts) => ts.to_string(),
                AstValue::Uuid(u) => format!("'{}'", u),
                AstValue::Array(_) | AstValue::Object(_) | AstValue::Vector(_) => {
                    "NULL".to_string()
                }
            };
            sql = sql.replace(&placeholder, &replacement);
        }

        match executor.execute_sql(&sql, &context) {
            Ok(result) => {
                stmt_ref.columns = result.result_set.columns;
                stmt_ref.rows = result
                    .result_set
                    .rows
                    .into_iter()
                    .map(|r| r.values)
                    .collect();
                stmt_ref.affected_rows = result.affected_rows;
                stmt_ref.cursor = 0;
            }
            Err(e) => {
                if let Ok(mut last_err) = db_ref.last_error.lock() {
                    *last_err = Some(e.to_string());
                }
                return JOULE_ERROR;
            }
        }
    } else {
        stmt_ref.cursor += 1;
    }

    if (stmt_ref.cursor as usize) < stmt_ref.rows.len() {
        JOULE_ROW
    } else {
        JOULE_DONE
    }
}

/// Get column count for the current result set
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_count(stmt: *mut JouleDBStmt) -> c_int {
    if stmt.is_null() {
        return 0;
    }
    let stmt_ref = unsafe { &*stmt };
    stmt_ref.columns.len() as c_int
}

/// Get column name by index (0-based)
///
/// Returns NULL if index is out of bounds. String is valid until statement is finalized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_name(stmt: *mut JouleDBStmt, col: c_int) -> *const c_char {
    if stmt.is_null() || col < 0 {
        return std::ptr::null();
    }

    let stmt_ref = unsafe { &*stmt };
    let idx = col as usize;
    if idx >= stmt_ref.columns.len() {
        return std::ptr::null();
    }

    match CString::new(stmt_ref.columns[idx].clone()) {
        Ok(s) => s.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

/// Get integer value from current row (0-based column index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_int(stmt: *mut JouleDBStmt, col: c_int) -> i64 {
    if stmt.is_null() || col < 0 {
        return 0;
    }

    let stmt_ref = unsafe { &*stmt };
    let row_idx = stmt_ref.cursor as usize;
    let col_idx = col as usize;

    if row_idx >= stmt_ref.rows.len() || col_idx >= stmt_ref.rows[row_idx].len() {
        return 0;
    }

    match &stmt_ref.rows[row_idx][col_idx] {
        AstValue::Int(n) => *n,
        AstValue::Float(f) => *f as i64,
        AstValue::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        AstValue::String(s) => s.parse().unwrap_or(0),
        AstValue::Timestamp(ts) => *ts,
        _ => 0,
    }
}

/// Get double value from current row (0-based column index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_double(stmt: *mut JouleDBStmt, col: c_int) -> c_double {
    if stmt.is_null() || col < 0 {
        return 0.0;
    }

    let stmt_ref = unsafe { &*stmt };
    let row_idx = stmt_ref.cursor as usize;
    let col_idx = col as usize;

    if row_idx >= stmt_ref.rows.len() || col_idx >= stmt_ref.rows[row_idx].len() {
        return 0.0;
    }

    match &stmt_ref.rows[row_idx][col_idx] {
        AstValue::Float(f) => *f,
        AstValue::Int(n) => *n as f64,
        AstValue::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        AstValue::String(s) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Get text value from current row (0-based column index)
///
/// Returns NULL for NULL values. String is valid until next step() or finalize().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_text(stmt: *mut JouleDBStmt, col: c_int) -> *const c_char {
    if stmt.is_null() || col < 0 {
        return std::ptr::null();
    }

    let stmt_ref = unsafe { &*stmt };
    let row_idx = stmt_ref.cursor as usize;
    let col_idx = col as usize;

    if row_idx >= stmt_ref.rows.len() || col_idx >= stmt_ref.rows[row_idx].len() {
        return std::ptr::null();
    }

    let text = match &stmt_ref.rows[row_idx][col_idx] {
        AstValue::Null => return std::ptr::null(),
        AstValue::String(s) => s.clone(),
        AstValue::Int(n) => n.to_string(),
        AstValue::Float(f) => f.to_string(),
        AstValue::Bool(b) => b.to_string(),
        AstValue::Uuid(u) => u.clone(),
        AstValue::Timestamp(ts) => ts.to_string(),
        AstValue::Bytes(b) => base64_encode(b),
        AstValue::Array(_) | AstValue::Object(_) | AstValue::Vector(_) => {
            format!("{:?}", stmt_ref.rows[row_idx][col_idx])
        }
    };

    match CString::new(text) {
        Ok(s) => s.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

/// Check if a column is NULL (0-based column index)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_column_is_null(stmt: *mut JouleDBStmt, col: c_int) -> c_int {
    if stmt.is_null() || col < 0 {
        return 1;
    }

    let stmt_ref = unsafe { &*stmt };
    let row_idx = stmt_ref.cursor as usize;
    let col_idx = col as usize;

    if row_idx >= stmt_ref.rows.len() || col_idx >= stmt_ref.rows[row_idx].len() {
        return 1;
    }

    if matches!(stmt_ref.rows[row_idx][col_idx], AstValue::Null) {
        1
    } else {
        0
    }
}

/// Reset a prepared statement for re-execution with new parameters
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_reset(stmt: *mut JouleDBStmt) -> c_int {
    if stmt.is_null() {
        return JOULE_ERROR;
    }

    let stmt_ref = unsafe { &mut *stmt };
    stmt_ref.cursor = -1;
    stmt_ref.columns.clear();
    stmt_ref.rows.clear();
    stmt_ref.affected_rows = 0;
    JOULE_OK
}

/// Finalize (destroy) a prepared statement
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_finalize(stmt: *mut JouleDBStmt) -> c_int {
    if !stmt.is_null() {
        unsafe {
            let _ = Box::from_raw(stmt);
        }
    }
    JOULE_OK
}

/// Free a string returned by the API (error messages, column names, text values)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn joule_db_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Get the JouleDB version string
#[unsafe(no_mangle)]
pub extern "C" fn joule_db_version() -> *const c_char {
    b"0.1.0\0".as_ptr() as *const c_char
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Simple base64 encoding for bytes display
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Count positional parameters ($1, $2, ...) in SQL
fn count_params(sql: &str) -> usize {
    let mut max_param = 0usize;
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            i += 1;
            let mut num = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                num = num * 10 + (bytes[i] - b'0') as usize;
                i += 1;
            }
            if num > max_param {
                max_param = num;
            }
        } else {
            i += 1;
        }
    }
    max_param
}
