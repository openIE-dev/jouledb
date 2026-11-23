//! ODBC 3.x API entry points.
//!
//! Every function in this module is `#[unsafe(no_mangle)] pub unsafe extern "C"`
//! so that the Driver Manager can load them from the shared library by name.
//!
//! Internal logic is safe Rust as much as possible; `unsafe` is confined to
//! the FFI boundary (dereferencing raw pointers passed in by the DM).

use std::ffi::CStr;

use crate::handle::*;
use crate::protocol;
use crate::types::*;

// ============================================================================
// Helpers
// ============================================================================

/// Read a C string from a `*const SQLCHAR` with an explicit length.
/// If `len` is `SQL_NTS`, the string is null-terminated.
/// Returns an empty string if the pointer is null.
fn read_sql_str(ptr: *const SQLCHAR, len: SQLSMALLINT) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe {
        if len == SQL_NTS {
            // Null-terminated string.
            CStr::from_ptr(ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        } else if len > 0 {
            let slice = std::slice::from_raw_parts(ptr, len as usize);
            String::from_utf8_lossy(slice).into_owned()
        } else {
            String::new()
        }
    }
}

/// Write a Rust string into a caller-provided `SQLCHAR` buffer.
/// Returns the number of characters written (excluding the null terminator).
fn write_sql_str(
    value: &str,
    out_buf: *mut SQLCHAR,
    buf_len: SQLSMALLINT,
    out_len: *mut SQLSMALLINT,
) -> SQLRETURN {
    let bytes = value.as_bytes();
    let needed = bytes.len() as SQLSMALLINT;

    if !out_len.is_null() {
        unsafe { *out_len = needed };
    }

    if out_buf.is_null() || buf_len <= 0 {
        // Caller only wants the length.
        return SQL_SUCCESS_WITH_INFO;
    }

    let copy_len = std::cmp::min(bytes.len(), (buf_len - 1) as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, copy_len);
        *out_buf.add(copy_len) = 0; // null-terminate
    }

    if copy_len < bytes.len() {
        SQL_SUCCESS_WITH_INFO // truncated
    } else {
        SQL_SUCCESS
    }
}

/// Write a string value into a generic `SQLPOINTER` buffer (used by SQLGetInfo).
fn write_info_str(
    value: &str,
    out_buf: SQLPOINTER,
    buf_len: SQLSMALLINT,
    out_len: *mut SQLSMALLINT,
) -> SQLRETURN {
    write_sql_str(value, out_buf as *mut SQLCHAR, buf_len, out_len)
}

/// Write a `u16` value into an `SQLPOINTER` (used by SQLGetInfo for numeric info).
fn write_info_u16(value: u16, out_buf: SQLPOINTER, out_len: *mut SQLSMALLINT) -> SQLRETURN {
    if !out_buf.is_null() {
        unsafe { *(out_buf as *mut u16) = value };
    }
    if !out_len.is_null() {
        unsafe { *out_len = std::mem::size_of::<u16>() as SQLSMALLINT };
    }
    SQL_SUCCESS
}

/// Write a `u32` value into an `SQLPOINTER` (used by SQLGetInfo for numeric info).
fn write_info_u32(value: u32, out_buf: SQLPOINTER, out_len: *mut SQLSMALLINT) -> SQLRETURN {
    if !out_buf.is_null() {
        unsafe { *(out_buf as *mut u32) = value };
    }
    if !out_len.is_null() {
        unsafe { *out_len = std::mem::size_of::<u32>() as SQLSMALLINT };
    }
    SQL_SUCCESS
}

/// Parse a connection string like:
/// `Driver={JouleDB};Server=host;Port=9000;Database=default;UID=user;PWD=pass`
fn parse_connection_string(conn_str: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for part in conn_str.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(eq_pos) = part.find('=') {
            let key = part[..eq_pos].trim().to_uppercase();
            let mut value = part[eq_pos + 1..].trim().to_string();
            // Strip braces from values like {JouleDB}.
            if value.starts_with('{') && value.ends_with('}') {
                value = value[1..value.len() - 1].to_string();
            }
            map.insert(key, value);
        }
    }
    map
}

/// Convert a `serde_json::Value` to an `Option<String>` for storage in the
/// statement result set.
fn json_value_to_string(val: &serde_json::Value) -> Option<String> {
    match val {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(if *b { "1".to_string() } else { "0".to_string() }),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// Fill bound column buffers for the current row of a statement.
fn fill_bound_columns(stmt: &Statement) {
    if stmt.cursor >= stmt.rows.len() {
        return;
    }
    let row = &stmt.rows[stmt.cursor];

    for binding in &stmt.bindings {
        let col_idx = binding.column_number as usize;
        if col_idx == 0 || col_idx > row.len() {
            // Out-of-range column — write NULL indicator.
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = SQL_NULL_DATA };
            }
            continue;
        }

        let cell = &row[col_idx - 1];
        match cell {
            None => {
                // SQL NULL.
                if !binding.str_len_or_ind.is_null() {
                    unsafe { *binding.str_len_or_ind = SQL_NULL_DATA };
                }
            }
            Some(value) => {
                write_cell_to_binding(binding, value);
            }
        }
    }
}

/// Write a single cell value into a column binding buffer.
fn write_cell_to_binding(binding: &ColumnBinding, value: &str) {
    if binding.target_value.is_null() {
        if !binding.str_len_or_ind.is_null() {
            unsafe { *binding.str_len_or_ind = value.len() as SQLLEN };
        }
        return;
    }

    match binding.target_type {
        SQL_C_CHAR | SQL_C_DEFAULT => {
            // Copy as C string.
            let bytes = value.as_bytes();
            let buf_len = binding.buffer_length as usize;
            let copy_len = std::cmp::min(bytes.len(), buf_len.saturating_sub(1));
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    binding.target_value as *mut u8,
                    copy_len,
                );
                *(binding.target_value as *mut u8).add(copy_len) = 0;
            }
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = bytes.len() as SQLLEN };
            }
        }
        SQL_C_SLONG => {
            let parsed: i32 = value.parse().unwrap_or(0);
            unsafe { *(binding.target_value as *mut i32) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = std::mem::size_of::<i32>() as SQLLEN };
            }
        }
        SQL_C_SBIGINT => {
            let parsed: i64 = value.parse().unwrap_or(0);
            unsafe { *(binding.target_value as *mut i64) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = std::mem::size_of::<i64>() as SQLLEN };
            }
        }
        SQL_C_DOUBLE => {
            let parsed: f64 = value.parse().unwrap_or(0.0);
            unsafe { *(binding.target_value as *mut f64) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = std::mem::size_of::<f64>() as SQLLEN };
            }
        }
        SQL_C_FLOAT => {
            let parsed: f32 = value.parse().unwrap_or(0.0);
            unsafe { *(binding.target_value as *mut f32) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = std::mem::size_of::<f32>() as SQLLEN };
            }
        }
        SQL_C_BIT => {
            let parsed: u8 = match value {
                "0" | "false" | "FALSE" | "f" | "F" => 0,
                _ => 1,
            };
            unsafe { *(binding.target_value as *mut u8) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = 1 };
            }
        }
        SQL_C_SSHORT => {
            let parsed: i16 = value.parse().unwrap_or(0);
            unsafe { *(binding.target_value as *mut i16) = parsed };
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = std::mem::size_of::<i16>() as SQLLEN };
            }
        }
        _ => {
            // Fall back to string copy for unrecognized types.
            let bytes = value.as_bytes();
            let buf_len = binding.buffer_length as usize;
            let copy_len = std::cmp::min(bytes.len(), buf_len.saturating_sub(1));
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    binding.target_value as *mut u8,
                    copy_len,
                );
                *(binding.target_value as *mut u8).add(copy_len) = 0;
            }
            if !binding.str_len_or_ind.is_null() {
                unsafe { *binding.str_len_or_ind = bytes.len() as SQLLEN };
            }
        }
    }
}

// ============================================================================
// SQLAllocHandle
// ============================================================================

/// Allocate an ODBC handle (environment, connection, or statement).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLAllocHandle(
    handle_type: SQLSMALLINT,
    input_handle: SQLHANDLE,
    output_handle: *mut SQLHANDLE,
) -> SQLRETURN {
    if output_handle.is_null() {
        return SQL_ERROR;
    }

    let mut store = lock_store();

    match handle_type {
        SQL_HANDLE_ENV => {
            let key = store.insert(Handle::Env(Environment::new()));
            unsafe { *output_handle = key_to_handle(key) };
            SQL_SUCCESS
        }
        SQL_HANDLE_DBC => {
            if input_handle.is_null() {
                return SQL_INVALID_HANDLE;
            }
            let env_key = handle_to_key(input_handle);
            // Verify the input is actually an environment handle.
            match store.get(env_key) {
                Some(Handle::Env(_)) => {}
                _ => return SQL_INVALID_HANDLE,
            }
            let key = store.insert(Handle::Dbc(Connection::new(env_key)));
            unsafe { *output_handle = key_to_handle(key) };
            SQL_SUCCESS
        }
        SQL_HANDLE_STMT => {
            if input_handle.is_null() {
                return SQL_INVALID_HANDLE;
            }
            let conn_key = handle_to_key(input_handle);
            match store.get(conn_key) {
                Some(Handle::Dbc(_)) => {}
                _ => return SQL_INVALID_HANDLE,
            }
            let key = store.insert(Handle::Stmt(Statement::new(conn_key)));
            unsafe { *output_handle = key_to_handle(key) };
            SQL_SUCCESS
        }
        _ => SQL_ERROR,
    }
}

// ============================================================================
// SQLFreeHandle
// ============================================================================

/// Free an ODBC handle and all its child handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLFreeHandle(handle_type: SQLSMALLINT, handle: SQLHANDLE) -> SQLRETURN {
    if handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(handle);
    let mut store = lock_store();

    match handle_type {
        SQL_HANDLE_ENV => {
            // Remove all child connections (and their statements).
            store.remove_connections_for_env(key);
            match store.remove(key) {
                Some(Handle::Env(_)) => SQL_SUCCESS,
                _ => SQL_INVALID_HANDLE,
            }
        }
        SQL_HANDLE_DBC => {
            // Remove all child statements.
            store.remove_stmts_for_connection(key);
            match store.remove(key) {
                Some(Handle::Dbc(_)) => SQL_SUCCESS,
                _ => SQL_INVALID_HANDLE,
            }
        }
        SQL_HANDLE_STMT => match store.remove(key) {
            Some(Handle::Stmt(_)) => SQL_SUCCESS,
            _ => SQL_INVALID_HANDLE,
        },
        _ => SQL_ERROR,
    }
}

// ============================================================================
// SQLSetEnvAttr
// ============================================================================

/// Set an environment attribute (typically SQL_ATTR_ODBC_VERSION).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLSetEnvAttr(
    environment_handle: SQLHENV,
    attribute: SQLINTEGER,
    value: SQLPOINTER,
    _string_length: SQLINTEGER,
) -> SQLRETURN {
    if environment_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(environment_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let env = match store.get_mut(key) {
        Some(Handle::Env(e)) => e,
        _ => return SQL_INVALID_HANDLE,
    };

    match attribute {
        SQL_ATTR_ODBC_VERSION => {
            let version = value as SQLINTEGER;
            match version {
                SQL_OV_ODBC3 | SQL_OV_ODBC3_80 => {
                    env.odbc_version = version;
                    SQL_SUCCESS
                }
                _ => {
                    env.diagnostics.push(DiagRecord::invalid_attr(format!(
                        "unsupported ODBC version: {}",
                        version
                    )));
                    SQL_ERROR
                }
            }
        }
        _ => {
            env.diagnostics.push(DiagRecord::invalid_attr(format!(
                "unsupported environment attribute: {}",
                attribute
            )));
            SQL_ERROR
        }
    }
}

// ============================================================================
// SQLConnect
// ============================================================================

/// Connect to JouleDB using server name, username, and password.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLConnect(
    connection_handle: SQLHDBC,
    server_name: *const SQLCHAR,
    name_length1: SQLSMALLINT,
    user_name: *const SQLCHAR,
    name_length2: SQLSMALLINT,
    authentication: *const SQLCHAR,
    name_length3: SQLSMALLINT,
) -> SQLRETURN {
    if connection_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(connection_handle);
    let server = read_sql_str(server_name, name_length1);
    let user = read_sql_str(user_name, name_length2);
    let pass = read_sql_str(authentication, name_length3);

    let mut store = lock_store();
    clear_diag(&mut store, key);

    // Parse host:port from server name.
    let (host, port) = parse_host_port(&server);

    let conn = match store.get_mut(key) {
        Some(Handle::Dbc(c)) => c,
        _ => return SQL_INVALID_HANDLE,
    };

    conn.host = host;
    conn.port = port;
    conn.username = user;
    conn.password = pass;

    // Establish TCP connection.
    match protocol::connect(&conn.host, conn.port) {
        Ok(stream) => {
            conn.stream = Some(stream);
            conn.connected = true;
            SQL_SUCCESS
        }
        Err(e) => {
            conn.diagnostics
                .push(DiagRecord::connection_failure(format!(
                    "failed to connect to {}:{}: {}",
                    conn.host, conn.port, e
                )));
            SQL_ERROR
        }
    }
}

/// Parse "host:port" or "host" from a server name string.
fn parse_host_port(server: &str) -> (String, u16) {
    if let Some(colon) = server.rfind(':') {
        let host = &server[..colon];
        let port_str = &server[colon + 1..];
        if let Ok(port) = port_str.parse::<u16>() {
            return (host.to_string(), port);
        }
    }
    (server.to_string(), protocol::DEFAULT_PORT)
}

// ============================================================================
// SQLDriverConnect
// ============================================================================

/// Connect to JouleDB using a connection string.
///
/// Format: `Driver={JouleDB};Server=host;Port=9000;Database=default;UID=user;PWD=pass`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLDriverConnect(
    connection_handle: SQLHDBC,
    _window_handle: SQLHWND,
    in_connection_string: *const SQLCHAR,
    string_length1: SQLSMALLINT,
    out_connection_string: *mut SQLCHAR,
    buffer_length: SQLSMALLINT,
    string_length2_ptr: *mut SQLSMALLINT,
    _driver_completion: SQLUSMALLINT,
) -> SQLRETURN {
    if connection_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(connection_handle);
    let conn_str = read_sql_str(in_connection_string, string_length1);

    let mut store = lock_store();
    clear_diag(&mut store, key);

    let params = parse_connection_string(&conn_str);

    let host = params
        .get("SERVER")
        .or_else(|| params.get("HOST"))
        .cloned()
        .unwrap_or_else(|| "127.0.0.1".to_string());

    let port: u16 = params
        .get("PORT")
        .and_then(|p| p.parse().ok())
        .unwrap_or(protocol::DEFAULT_PORT);

    let database = params
        .get("DATABASE")
        .or_else(|| params.get("DB"))
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let username = params
        .get("UID")
        .or_else(|| params.get("USER"))
        .cloned()
        .unwrap_or_default();

    let password = params
        .get("PWD")
        .or_else(|| params.get("PASSWORD"))
        .cloned()
        .unwrap_or_default();

    let conn = match store.get_mut(key) {
        Some(Handle::Dbc(c)) => c,
        _ => return SQL_INVALID_HANDLE,
    };

    conn.host = host;
    conn.port = port;
    conn.database = database;
    conn.username = username;
    conn.password = password;

    // Establish TCP connection.
    match protocol::connect(&conn.host, conn.port) {
        Ok(stream) => {
            conn.stream = Some(stream);
            conn.connected = true;
        }
        Err(e) => {
            conn.diagnostics
                .push(DiagRecord::connection_failure(format!(
                    "failed to connect to {}:{}: {}",
                    conn.host, conn.port, e
                )));
            return SQL_ERROR;
        }
    }

    // Build the output connection string.
    let out_str = format!(
        "Driver={{JouleDB}};Server={};Port={};Database={};UID={}",
        conn.host, conn.port, conn.database, conn.username
    );
    write_sql_str(
        &out_str,
        out_connection_string,
        buffer_length,
        string_length2_ptr,
    );

    SQL_SUCCESS
}

// ============================================================================
// SQLDisconnect
// ============================================================================

/// Disconnect from JouleDB.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLDisconnect(connection_handle: SQLHDBC) -> SQLRETURN {
    if connection_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(connection_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let conn = match store.get_mut(key) {
        Some(Handle::Dbc(c)) => c,
        _ => return SQL_INVALID_HANDLE,
    };

    // Drop the TCP stream (closes the socket).
    conn.stream = None;
    conn.connected = false;

    SQL_SUCCESS
}

// ============================================================================
// SQLExecDirect
// ============================================================================

/// Execute a SQL statement directly (no preparation step).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLExecDirect(
    statement_handle: SQLHSTMT,
    statement_text: *const SQLCHAR,
    text_length: SQLINTEGER,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let sql = read_sql_str(statement_text, text_length as SQLSMALLINT);

    if sql.is_empty() {
        let mut store = lock_store();
        push_diag(&mut store, key, DiagRecord::general("empty SQL statement"));
        return SQL_ERROR;
    }

    let mut store = lock_store();
    clear_diag(&mut store, key);

    // Get the parent connection handle key.
    let conn_key = match store.get(key) {
        Some(Handle::Stmt(s)) => s.conn_handle,
        _ => return SQL_INVALID_HANDLE,
    };

    // Borrow the connection to get a mutable stream reference.
    // We need to do pointer gymnastics because we can't hold two mutable
    // borrows from the store simultaneously.
    let (stream_ptr, request_id) = match store.get_mut(conn_key) {
        Some(Handle::Dbc(conn)) => {
            if !conn.connected || conn.stream.is_none() {
                // Connection not open — store diag on connection handle,
                // then return error.
                conn.diagnostics.push(DiagRecord::sequence_error(
                    "not connected — call SQLConnect first",
                ));
                return SQL_ERROR;
            }
            let rid = conn.alloc_request_id();
            // SAFETY: We take a raw pointer to the stream. The stream lives
            // inside the global store which is locked for the duration of
            // this function. We will not move or drop the stream.
            let ptr = conn.stream.as_mut().unwrap() as *mut std::net::TcpStream;
            (ptr, rid)
        }
        _ => return SQL_INVALID_HANDLE,
    };

    // Send the query over the wire.
    // SAFETY: `stream_ptr` is valid for the duration of the lock.
    let stream = unsafe { &mut *stream_ptr };
    let result = protocol::send_query(stream, &sql, request_id as u64);

    // Now update the statement with results.
    let stmt = match store.get_mut(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };
    stmt.reset_results();
    stmt.sql = Some(sql);

    match result {
        Ok(response) => {
            // Build column metadata.
            for col in &response.columns {
                let sql_type = protocol::jouledb_type_to_sql(&col.col_type);
                stmt.columns.push(ColumnMeta {
                    name: col.name.clone(),
                    sql_type,
                    column_size: protocol::default_column_size(sql_type),
                    decimal_digits: 0,
                    nullable: SQL_NULLABLE_UNKNOWN,
                });
            }

            // Build row data.
            for row in &response.rows {
                let cells: Vec<Option<String>> =
                    row.iter().map(|v| json_value_to_string(v)).collect();
                stmt.rows.push(cells);
            }

            stmt.affected_rows = response.affected_rows;
            stmt.has_result_set = !stmt.columns.is_empty();
            stmt.cursor = usize::MAX; // before first row

            SQL_SUCCESS
        }
        Err(e) => {
            stmt.diagnostics
                .push(DiagRecord::general(format!("query failed: {}", e)));
            SQL_ERROR
        }
    }
}

// ============================================================================
// SQLFetch
// ============================================================================

/// Fetch the next row of a result set. Advances the cursor by one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLFetch(statement_handle: SQLHSTMT) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let stmt = match store.get_mut(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    if !stmt.has_result_set {
        stmt.diagnostics
            .push(DiagRecord::invalid_cursor("no open cursor"));
        return SQL_ERROR;
    }

    // Advance cursor.
    let next = if stmt.cursor == usize::MAX {
        0
    } else {
        stmt.cursor + 1
    };

    if next >= stmt.rows.len() {
        // Past end of result set.
        return SQL_NO_DATA;
    }

    stmt.cursor = next;

    // Fill bound column buffers.
    fill_bound_columns(stmt);

    SQL_SUCCESS
}

// ============================================================================
// SQLGetData
// ============================================================================

/// Retrieve data for a single column in the current row.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLGetData(
    statement_handle: SQLHSTMT,
    column_number: SQLUSMALLINT,
    target_type: SQLSMALLINT,
    target_value: SQLPOINTER,
    buffer_length: SQLLEN,
    str_len_or_ind: *mut SQLLEN,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let stmt = match store.get_mut(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    if stmt.cursor == usize::MAX || stmt.cursor >= stmt.rows.len() {
        stmt.diagnostics
            .push(DiagRecord::invalid_cursor("cursor not positioned on a row"));
        return SQL_ERROR;
    }

    let col_idx = column_number as usize;
    if col_idx == 0 || col_idx > stmt.columns.len() {
        stmt.diagnostics.push(DiagRecord::invalid_column(format!(
            "column {} out of range (1..{})",
            column_number,
            stmt.columns.len()
        )));
        return SQL_ERROR;
    }

    let row = &stmt.rows[stmt.cursor];
    let cell = if col_idx <= row.len() {
        &row[col_idx - 1]
    } else {
        &None
    };

    match cell {
        None => {
            // SQL NULL.
            if !str_len_or_ind.is_null() {
                unsafe { *str_len_or_ind = SQL_NULL_DATA };
            }
            SQL_SUCCESS
        }
        Some(value) => {
            // Create a temporary binding and use the shared writer.
            let binding = ColumnBinding {
                column_number,
                target_type,
                target_value,
                buffer_length,
                str_len_or_ind,
            };
            write_cell_to_binding(&binding, value);
            SQL_SUCCESS
        }
    }
}

// ============================================================================
// SQLNumResultCols
// ============================================================================

/// Return the number of columns in the result set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLNumResultCols(
    statement_handle: SQLHSTMT,
    column_count_ptr: *mut SQLSMALLINT,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }
    if column_count_ptr.is_null() {
        return SQL_ERROR;
    }

    let key = handle_to_key(statement_handle);
    let store = lock_store();

    let stmt = match store.get(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    unsafe { *column_count_ptr = stmt.columns.len() as SQLSMALLINT };
    SQL_SUCCESS
}

// ============================================================================
// SQLDescribeCol
// ============================================================================

/// Describe a result set column by number (1-based).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLDescribeCol(
    statement_handle: SQLHSTMT,
    column_number: SQLUSMALLINT,
    column_name: *mut SQLCHAR,
    buffer_length: SQLSMALLINT,
    name_length_ptr: *mut SQLSMALLINT,
    data_type_ptr: *mut SQLSMALLINT,
    column_size_ptr: *mut SQLULEN,
    decimal_digits_ptr: *mut SQLSMALLINT,
    nullable_ptr: *mut SQLSMALLINT,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let stmt = match store.get(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    let col_idx = column_number as usize;
    let num_cols = stmt.columns.len();
    if col_idx == 0 || col_idx > num_cols {
        // Store diag and return error. We need a mutable borrow for diag,
        // so release the immutable reference first.
        let _ = stmt;
        push_diag(
            &mut store,
            key,
            DiagRecord::invalid_column(format!(
                "column {} out of range (1..{})",
                column_number, num_cols
            )),
        );
        return SQL_ERROR;
    }

    let col = &stmt.columns[col_idx - 1];

    // Write column name.
    let ret = write_sql_str(&col.name, column_name, buffer_length, name_length_ptr);

    // Write data type.
    if !data_type_ptr.is_null() {
        unsafe { *data_type_ptr = col.sql_type };
    }

    // Write column size.
    if !column_size_ptr.is_null() {
        unsafe { *column_size_ptr = col.column_size };
    }

    // Write decimal digits.
    if !decimal_digits_ptr.is_null() {
        unsafe { *decimal_digits_ptr = col.decimal_digits };
    }

    // Write nullable.
    if !nullable_ptr.is_null() {
        unsafe { *nullable_ptr = col.nullable };
    }

    ret
}

// ============================================================================
// SQLRowCount
// ============================================================================

/// Return the number of rows affected by the last INSERT/UPDATE/DELETE.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLRowCount(
    statement_handle: SQLHSTMT,
    row_count_ptr: *mut SQLLEN,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }
    if row_count_ptr.is_null() {
        return SQL_ERROR;
    }

    let key = handle_to_key(statement_handle);
    let store = lock_store();

    let stmt = match store.get(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    let count = stmt.affected_rows.unwrap_or(0) as SQLLEN;
    unsafe { *row_count_ptr = count };
    SQL_SUCCESS
}

// ============================================================================
// SQLCloseCursor
// ============================================================================

/// Close the cursor associated with a statement handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLCloseCursor(statement_handle: SQLHSTMT) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let stmt = match store.get_mut(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    if !stmt.has_result_set {
        stmt.diagnostics
            .push(DiagRecord::invalid_cursor("no open cursor to close"));
        return SQL_ERROR;
    }

    stmt.close_cursor();
    SQL_SUCCESS
}

// ============================================================================
// SQLBindCol
// ============================================================================

/// Bind a column to an application buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLBindCol(
    statement_handle: SQLHSTMT,
    column_number: SQLUSMALLINT,
    target_type: SQLSMALLINT,
    target_value: SQLPOINTER,
    buffer_length: SQLLEN,
    str_len_or_ind: *mut SQLLEN,
) -> SQLRETURN {
    if statement_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(statement_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    let stmt = match store.get_mut(key) {
        Some(Handle::Stmt(s)) => s,
        _ => return SQL_INVALID_HANDLE,
    };

    if column_number == 0 {
        stmt.diagnostics.push(DiagRecord::invalid_column(
            "column number 0 (bookmark) not supported",
        ));
        return SQL_ERROR;
    }

    // Remove any existing binding for this column.
    stmt.bindings.retain(|b| b.column_number != column_number);

    // If target_value is null, the column is unbound.
    if target_value.is_null() && str_len_or_ind.is_null() {
        return SQL_SUCCESS;
    }

    stmt.bindings.push(ColumnBinding {
        column_number,
        target_type,
        target_value,
        buffer_length,
        str_len_or_ind,
    });

    SQL_SUCCESS
}

// ============================================================================
// SQLGetDiagRec
// ============================================================================

/// Retrieve a diagnostic record for a handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLGetDiagRec(
    handle_type: SQLSMALLINT,
    handle: SQLHANDLE,
    rec_number: SQLSMALLINT,
    sqlstate: *mut SQLCHAR,
    native_error: *mut SQLINTEGER,
    message_text: *mut SQLCHAR,
    buffer_length: SQLSMALLINT,
    text_length: *mut SQLSMALLINT,
) -> SQLRETURN {
    if handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    // Validate handle type.
    match handle_type {
        SQL_HANDLE_ENV | SQL_HANDLE_DBC | SQL_HANDLE_STMT => {}
        _ => return SQL_INVALID_HANDLE,
    }

    if rec_number < 1 {
        return SQL_ERROR;
    }

    let key = handle_to_key(handle);
    let store = lock_store();

    let rec = match get_diag(&store, key, rec_number as usize) {
        Some(r) => r,
        None => return SQL_NO_DATA,
    };

    // Write SQLSTATE (always 5 chars + null).
    if !sqlstate.is_null() {
        let state_bytes = rec.state.as_bytes();
        let copy_len = std::cmp::min(state_bytes.len(), 5);
        unsafe {
            std::ptr::copy_nonoverlapping(state_bytes.as_ptr(), sqlstate, copy_len);
            // Pad with zeros if state is shorter than 5 chars.
            for i in copy_len..5 {
                *sqlstate.add(i) = b'0';
            }
            *sqlstate.add(5) = 0; // null-terminate
        }
    }

    // Write native error code.
    if !native_error.is_null() {
        unsafe { *native_error = rec.native_error };
    }

    // Write message text.
    let prefix = format!("[JouleDB ODBC][{}] ", rec.state);
    let full_msg = format!("{}{}", prefix, rec.message);
    write_sql_str(&full_msg, message_text, buffer_length, text_length);

    SQL_SUCCESS
}

// ============================================================================
// SQLGetInfo
// ============================================================================

/// Return information about the driver and data source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SQLGetInfo(
    connection_handle: SQLHDBC,
    info_type: SQLUSMALLINT,
    info_value: SQLPOINTER,
    buffer_length: SQLSMALLINT,
    string_length: *mut SQLSMALLINT,
) -> SQLRETURN {
    if connection_handle.is_null() {
        return SQL_INVALID_HANDLE;
    }

    let key = handle_to_key(connection_handle);
    let mut store = lock_store();
    clear_diag(&mut store, key);

    // Verify handle is a connection.
    match store.get(key) {
        Some(Handle::Dbc(_)) => {}
        _ => return SQL_INVALID_HANDLE,
    }

    match info_type {
        SQL_DRIVER_NAME => {
            write_info_str("joule-db-odbc", info_value, buffer_length, string_length)
        }
        SQL_DRIVER_VER => write_info_str("00.01.0000", info_value, buffer_length, string_length),
        SQL_DBMS_NAME => write_info_str("JouleDB", info_value, buffer_length, string_length),
        SQL_DBMS_VER => write_info_str("00.01.0000", info_value, buffer_length, string_length),
        SQL_SERVER_NAME => {
            let server_name = match store.get(key) {
                Some(Handle::Dbc(c)) => format!("{}:{}", c.host, c.port),
                _ => String::new(),
            };
            write_info_str(&server_name, info_value, buffer_length, string_length)
        }
        SQL_DATABASE_NAME => {
            let db_name = match store.get(key) {
                Some(Handle::Dbc(c)) => c.database.clone(),
                _ => String::new(),
            };
            write_info_str(&db_name, info_value, buffer_length, string_length)
        }
        SQL_DATA_SOURCE_NAME => write_info_str("JouleDB", info_value, buffer_length, string_length),
        SQL_IDENTIFIER_QUOTE_CHAR => write_info_str("\"", info_value, buffer_length, string_length),
        SQL_SEARCH_PATTERN_ESCAPE => write_info_str("\\", info_value, buffer_length, string_length),
        SQL_CATALOG_NAME_SEPARATOR => write_info_str(".", info_value, buffer_length, string_length),
        SQL_CATALOG_TERM => write_info_str("database", info_value, buffer_length, string_length),
        SQL_SCHEMA_TERM => write_info_str("schema", info_value, buffer_length, string_length),
        SQL_TABLE_TERM => write_info_str("table", info_value, buffer_length, string_length),
        SQL_MAX_COLUMNS_IN_SELECT => write_info_u16(0, info_value, string_length), // 0 = no limit
        SQL_MAX_COLUMNS_IN_TABLE => write_info_u16(0, info_value, string_length),
        SQL_MAX_TABLES_IN_SELECT => write_info_u16(0, info_value, string_length),
        SQL_MAX_IDENTIFIER_LEN => write_info_u16(128, info_value, string_length),
        _ => {
            push_diag(
                &mut store,
                key,
                DiagRecord::invalid_attr(format!("unsupported SQLGetInfo type: {}", info_type)),
            );
            SQL_ERROR
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    // --- Helper: allocate env + dbc for tests --------------------------------

    fn alloc_env() -> SQLHENV {
        let mut env: SQLHENV = ptr::null_mut();
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &mut env) };
        assert_eq!(ret, SQL_SUCCESS);
        assert!(!env.is_null());
        env
    }

    fn alloc_dbc(env: SQLHENV) -> SQLHDBC {
        let mut dbc: SQLHDBC = ptr::null_mut();
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_DBC, env, &mut dbc) };
        assert_eq!(ret, SQL_SUCCESS);
        assert!(!dbc.is_null());
        dbc
    }

    fn alloc_stmt(dbc: SQLHDBC) -> SQLHSTMT {
        let mut stmt: SQLHSTMT = ptr::null_mut();
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_STMT, dbc, &mut stmt) };
        assert_eq!(ret, SQL_SUCCESS);
        assert!(!stmt.is_null());
        stmt
    }

    // --- SQLAllocHandle / SQLFreeHandle --------------------------------------

    #[test]
    fn alloc_and_free_env() {
        let env = alloc_env();
        let ret = unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
        assert_eq!(ret, SQL_SUCCESS);
    }

    #[test]
    fn alloc_and_free_dbc() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let ret = unsafe { SQLFreeHandle(SQL_HANDLE_DBC, dbc) };
        assert_eq!(ret, SQL_SUCCESS);
        unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
    }

    #[test]
    fn alloc_and_free_stmt() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);
        let ret = unsafe { SQLFreeHandle(SQL_HANDLE_STMT, stmt) };
        assert_eq!(ret, SQL_SUCCESS);
        unsafe {
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn alloc_null_output_returns_error() {
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, ptr::null_mut()) };
        assert_eq!(ret, SQL_ERROR);
    }

    #[test]
    fn alloc_dbc_with_null_env_returns_invalid() {
        let mut dbc: SQLHDBC = ptr::null_mut();
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_DBC, SQL_NULL_HANDLE, &mut dbc) };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    #[test]
    fn alloc_stmt_with_null_dbc_returns_invalid() {
        let mut stmt: SQLHSTMT = ptr::null_mut();
        let ret = unsafe { SQLAllocHandle(SQL_HANDLE_STMT, SQL_NULL_HANDLE, &mut stmt) };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    #[test]
    fn free_null_handle_returns_invalid() {
        let ret = unsafe { SQLFreeHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE) };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    #[test]
    fn free_env_cascades_to_children() {
        let env = alloc_env();
        let _dbc = alloc_dbc(env);
        let _dbc2 = alloc_dbc(env);
        // Freeing env should remove both dbcs.
        let ret = unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
        assert_eq!(ret, SQL_SUCCESS);
    }

    // --- SQLSetEnvAttr -------------------------------------------------------

    #[test]
    fn set_env_attr_odbc_version() {
        let env = alloc_env();
        let ret =
            unsafe { SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, SQL_OV_ODBC3 as SQLPOINTER, 0) };
        assert_eq!(ret, SQL_SUCCESS);
        unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
    }

    #[test]
    fn set_env_attr_null_handle() {
        let ret = unsafe {
            SQLSetEnvAttr(
                SQL_NULL_HANDLE,
                SQL_ATTR_ODBC_VERSION,
                SQL_OV_ODBC3 as SQLPOINTER,
                0,
            )
        };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    #[test]
    fn set_env_attr_unsupported() {
        let env = alloc_env();
        let ret = unsafe { SQLSetEnvAttr(env, 9999, ptr::null_mut(), 0) };
        assert_eq!(ret, SQL_ERROR);
        unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
    }

    // --- SQLDisconnect (without connection) ----------------------------------

    #[test]
    fn disconnect_not_connected() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        // Should succeed even if not connected (no-op).
        let ret = unsafe { SQLDisconnect(dbc) };
        assert_eq!(ret, SQL_SUCCESS);
        unsafe {
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn disconnect_null_handle() {
        let ret = unsafe { SQLDisconnect(SQL_NULL_HANDLE) };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    // --- SQLNumResultCols / SQLRowCount (no results) -------------------------

    #[test]
    fn num_result_cols_no_results() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);
        let mut count: SQLSMALLINT = -1;
        let ret = unsafe { SQLNumResultCols(stmt, &mut count) };
        assert_eq!(ret, SQL_SUCCESS);
        assert_eq!(count, 0);
        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn row_count_no_results() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);
        let mut count: SQLLEN = -1;
        let ret = unsafe { SQLRowCount(stmt, &mut count) };
        assert_eq!(ret, SQL_SUCCESS);
        assert_eq!(count, 0);
        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLCloseCursor (no open cursor) -------------------------------------

    #[test]
    fn close_cursor_no_result_set() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);
        let ret = unsafe { SQLCloseCursor(stmt) };
        assert_eq!(ret, SQL_ERROR); // no open cursor
        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLGetDiagRec -------------------------------------------------------

    #[test]
    fn get_diag_rec_no_records() {
        let env = alloc_env();
        let mut state = [0u8; 6];
        let mut native: SQLINTEGER = 0;
        let mut msg = [0u8; 256];
        let mut msg_len: SQLSMALLINT = 0;

        let ret = unsafe {
            SQLGetDiagRec(
                SQL_HANDLE_ENV,
                env,
                1,
                state.as_mut_ptr(),
                &mut native,
                msg.as_mut_ptr(),
                256,
                &mut msg_len,
            )
        };
        assert_eq!(ret, SQL_NO_DATA);
        unsafe { SQLFreeHandle(SQL_HANDLE_ENV, env) };
    }

    #[test]
    fn get_diag_rec_null_handle() {
        let mut state = [0u8; 6];
        let mut native: SQLINTEGER = 0;
        let mut msg = [0u8; 256];
        let mut msg_len: SQLSMALLINT = 0;

        let ret = unsafe {
            SQLGetDiagRec(
                SQL_HANDLE_ENV,
                SQL_NULL_HANDLE,
                1,
                state.as_mut_ptr(),
                &mut native,
                msg.as_mut_ptr(),
                256,
                &mut msg_len,
            )
        };
        assert_eq!(ret, SQL_INVALID_HANDLE);
    }

    // --- SQLGetInfo ----------------------------------------------------------

    #[test]
    fn get_info_driver_name() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let mut buf = [0u8; 64];
        let mut len: SQLSMALLINT = 0;
        let ret = unsafe {
            SQLGetInfo(
                dbc,
                SQL_DRIVER_NAME,
                buf.as_mut_ptr() as SQLPOINTER,
                64,
                &mut len,
            )
        };
        assert_eq!(ret, SQL_SUCCESS);
        let name = std::str::from_utf8(&buf[..len as usize]).unwrap();
        assert_eq!(name, "joule-db-odbc");
        unsafe {
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn get_info_dbms_name() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let mut buf = [0u8; 64];
        let mut len: SQLSMALLINT = 0;
        let ret = unsafe {
            SQLGetInfo(
                dbc,
                SQL_DBMS_NAME,
                buf.as_mut_ptr() as SQLPOINTER,
                64,
                &mut len,
            )
        };
        assert_eq!(ret, SQL_SUCCESS);
        let name = std::str::from_utf8(&buf[..len as usize]).unwrap();
        assert_eq!(name, "JouleDB");
        unsafe {
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn get_info_unsupported_type() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let mut buf = [0u8; 64];
        let mut len: SQLSMALLINT = 0;
        let ret = unsafe { SQLGetInfo(dbc, 9999, buf.as_mut_ptr() as SQLPOINTER, 64, &mut len) };
        assert_eq!(ret, SQL_ERROR);
        unsafe {
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- Connection string parsing -------------------------------------------

    #[test]
    fn parse_connection_string_full() {
        let params = parse_connection_string(
            "Driver={JouleDB};Server=10.10.10.1;Port=9000;Database=mydb;UID=admin;PWD=secret",
        );
        assert_eq!(params.get("DRIVER").unwrap(), "JouleDB");
        assert_eq!(params.get("SERVER").unwrap(), "10.10.10.1");
        assert_eq!(params.get("PORT").unwrap(), "9000");
        assert_eq!(params.get("DATABASE").unwrap(), "mydb");
        assert_eq!(params.get("UID").unwrap(), "admin");
        assert_eq!(params.get("PWD").unwrap(), "secret");
    }

    #[test]
    fn parse_connection_string_minimal() {
        let params = parse_connection_string("Server=localhost");
        assert_eq!(params.get("SERVER").unwrap(), "localhost");
        assert!(params.get("PORT").is_none());
    }

    #[test]
    fn parse_connection_string_empty() {
        let params = parse_connection_string("");
        assert!(params.is_empty());
    }

    #[test]
    fn parse_host_port_with_port() {
        let (host, port) = parse_host_port("10.10.10.1:9000");
        assert_eq!(host, "10.10.10.1");
        assert_eq!(port, 9000);
    }

    #[test]
    fn parse_host_port_without_port() {
        let (host, port) = parse_host_port("localhost");
        assert_eq!(host, "localhost");
        assert_eq!(port, protocol::DEFAULT_PORT);
    }

    #[test]
    fn parse_host_port_invalid_port() {
        let (host, port) = parse_host_port("host:notaport");
        assert_eq!(host, "host:notaport");
        assert_eq!(port, protocol::DEFAULT_PORT);
    }

    // --- Helper function tests -----------------------------------------------

    #[test]
    fn read_sql_str_null() {
        let result = read_sql_str(ptr::null(), 0);
        assert_eq!(result, "");
    }

    #[test]
    fn read_sql_str_with_length() {
        let data = b"hello world";
        let result = read_sql_str(data.as_ptr(), 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn read_sql_str_null_terminated() {
        let data = b"hello\0";
        let result = read_sql_str(data.as_ptr(), SQL_NTS);
        assert_eq!(result, "hello");
    }

    #[test]
    fn json_value_to_string_null() {
        assert_eq!(json_value_to_string(&serde_json::Value::Null), None);
    }

    #[test]
    fn json_value_to_string_bool() {
        assert_eq!(
            json_value_to_string(&serde_json::Value::Bool(true)),
            Some("1".to_string())
        );
        assert_eq!(
            json_value_to_string(&serde_json::Value::Bool(false)),
            Some("0".to_string())
        );
    }

    #[test]
    fn json_value_to_string_number() {
        assert_eq!(
            json_value_to_string(&serde_json::json!(42)),
            Some("42".to_string())
        );
        assert_eq!(
            json_value_to_string(&serde_json::json!(3.14)),
            Some("3.14".to_string())
        );
    }

    #[test]
    fn json_value_to_string_string() {
        assert_eq!(
            json_value_to_string(&serde_json::json!("hello")),
            Some("hello".to_string())
        );
    }

    // --- SQLBindCol ----------------------------------------------------------

    #[test]
    fn bind_and_unbind_column() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let mut buf = [0u8; 64];
        let mut ind: SQLLEN = 0;

        // Bind column 1.
        let ret = unsafe {
            SQLBindCol(
                stmt,
                1,
                SQL_C_CHAR,
                buf.as_mut_ptr() as SQLPOINTER,
                64,
                &mut ind,
            )
        };
        assert_eq!(ret, SQL_SUCCESS);

        // Unbind column 1 (pass null pointers).
        let ret = unsafe { SQLBindCol(stmt, 1, SQL_C_CHAR, ptr::null_mut(), 0, ptr::null_mut()) };
        assert_eq!(ret, SQL_SUCCESS);

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn bind_column_zero_returns_error() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let mut buf = [0u8; 64];
        let mut ind: SQLLEN = 0;

        let ret = unsafe {
            SQLBindCol(
                stmt,
                0, // bookmark column — not supported
                SQL_C_CHAR,
                buf.as_mut_ptr() as SQLPOINTER,
                64,
                &mut ind,
            )
        };
        assert_eq!(ret, SQL_ERROR);

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLExecDirect (without server — should fail) ------------------------

    #[test]
    fn exec_direct_not_connected() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let sql = b"SELECT 1\0";
        let ret = unsafe { SQLExecDirect(stmt, sql.as_ptr(), SQL_NTS as SQLINTEGER) };
        assert_eq!(ret, SQL_ERROR);

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    #[test]
    fn exec_direct_empty_sql() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let ret = unsafe { SQLExecDirect(stmt, ptr::null(), 0) };
        assert_eq!(ret, SQL_ERROR);

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLFetch (no result set) --------------------------------------------

    #[test]
    fn fetch_no_result_set() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let ret = unsafe { SQLFetch(stmt) };
        assert_eq!(ret, SQL_ERROR); // no cursor

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLGetData (no result set) ------------------------------------------

    #[test]
    fn get_data_no_cursor() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let mut buf = [0u8; 64];
        let mut ind: SQLLEN = 0;
        let ret = unsafe {
            SQLGetData(
                stmt,
                1,
                SQL_C_CHAR,
                buf.as_mut_ptr() as SQLPOINTER,
                64,
                &mut ind,
            )
        };
        assert_eq!(ret, SQL_ERROR);

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- SQLDescribeCol (no columns) -----------------------------------------

    #[test]
    fn describe_col_no_columns() {
        let env = alloc_env();
        let dbc = alloc_dbc(env);
        let stmt = alloc_stmt(dbc);

        let mut name = [0u8; 64];
        let mut name_len: SQLSMALLINT = 0;
        let mut data_type: SQLSMALLINT = 0;
        let mut col_size: SQLULEN = 0;
        let mut dec_digits: SQLSMALLINT = 0;
        let mut nullable: SQLSMALLINT = 0;

        let ret = unsafe {
            SQLDescribeCol(
                stmt,
                1,
                name.as_mut_ptr(),
                64,
                &mut name_len,
                &mut data_type,
                &mut col_size,
                &mut dec_digits,
                &mut nullable,
            )
        };
        assert_eq!(ret, SQL_ERROR); // no columns

        unsafe {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
    }

    // --- Write helpers -------------------------------------------------------

    #[test]
    fn write_sql_str_fits() {
        let mut buf = [0u8; 16];
        let mut len: SQLSMALLINT = 0;
        let ret = write_sql_str("hello", buf.as_mut_ptr(), 16, &mut len);
        assert_eq!(ret, SQL_SUCCESS);
        assert_eq!(len, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(buf[5], 0); // null-terminated
    }

    #[test]
    fn write_sql_str_truncated() {
        let mut buf = [0u8; 4]; // only room for 3 chars + null
        let mut len: SQLSMALLINT = 0;
        let ret = write_sql_str("hello", buf.as_mut_ptr(), 4, &mut len);
        assert_eq!(ret, SQL_SUCCESS_WITH_INFO);
        assert_eq!(len, 5); // full length reported
        assert_eq!(&buf[..3], b"hel");
        assert_eq!(buf[3], 0); // null-terminated
    }

    #[test]
    fn write_sql_str_null_buffer() {
        let mut len: SQLSMALLINT = 0;
        let ret = write_sql_str("hello", ptr::null_mut(), 0, &mut len);
        assert_eq!(ret, SQL_SUCCESS_WITH_INFO);
        assert_eq!(len, 5);
    }
}
