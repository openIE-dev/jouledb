//! ODBC 3.x C type definitions for FFI.
//!
//! These type aliases and constants mirror the ODBC C header (`sql.h`, `sqlext.h`,
//! `sqltypes.h`) so that the exported `extern "C"` functions have the correct ABI.

// ============================================================================
// Fundamental ODBC scalar types
// ============================================================================

/// Signed 16-bit integer (SWORD).
pub type SQLSMALLINT = i16;

/// Unsigned 16-bit integer (UWORD).
pub type SQLUSMALLINT = u16;

/// Signed 32-bit integer (SDWORD).
pub type SQLINTEGER = i32;

/// Unsigned 32-bit integer (UDWORD).
pub type SQLUINTEGER = u32;

/// Signed pointer-width integer (used for lengths/indicators).
pub type SQLLEN = isize;

/// Unsigned pointer-width integer.
pub type SQLULEN = usize;

/// 8-bit character (UCHAR).
pub type SQLCHAR = u8;

/// Wide character (UTF-16 on Windows, UCS-2).
pub type SQLWCHAR = u16;

/// Generic pointer (void*).
pub type SQLPOINTER = *mut std::ffi::c_void;

/// Return code from ODBC functions.
pub type SQLRETURN = SQLSMALLINT;

// ============================================================================
// Handle types
// ============================================================================

/// Opaque handle — pointer-sized integer.
pub type SQLHANDLE = *mut std::ffi::c_void;

/// Environment handle.
pub type SQLHENV = SQLHANDLE;

/// Connection handle.
pub type SQLHDBC = SQLHANDLE;

/// Statement handle.
pub type SQLHSTMT = SQLHANDLE;

/// Window handle (unused on non-Windows, but needed for SQLDriverConnect).
pub type SQLHWND = SQLHANDLE;

// ============================================================================
// Return codes
// ============================================================================

pub const SQL_SUCCESS: SQLRETURN = 0;
pub const SQL_SUCCESS_WITH_INFO: SQLRETURN = 1;
pub const SQL_ERROR: SQLRETURN = -1;
pub const SQL_INVALID_HANDLE: SQLRETURN = -2;
pub const SQL_NO_DATA: SQLRETURN = 100;
pub const SQL_STILL_EXECUTING: SQLRETURN = 2;
pub const SQL_NEED_DATA: SQLRETURN = 99;

// ============================================================================
// Handle type identifiers (for SQLAllocHandle / SQLFreeHandle)
// ============================================================================

pub const SQL_HANDLE_ENV: SQLSMALLINT = 1;
pub const SQL_HANDLE_DBC: SQLSMALLINT = 2;
pub const SQL_HANDLE_STMT: SQLSMALLINT = 3;
pub const SQL_HANDLE_DESC: SQLSMALLINT = 4;

/// Null handle sentinel.
pub const SQL_NULL_HANDLE: SQLHANDLE = std::ptr::null_mut();

// ============================================================================
// Environment attributes
// ============================================================================

pub const SQL_ATTR_ODBC_VERSION: SQLINTEGER = 200;
pub const SQL_OV_ODBC3: SQLINTEGER = 3;
pub const SQL_OV_ODBC3_80: SQLINTEGER = 380;

// ============================================================================
// Connection attributes
// ============================================================================

pub const SQL_ATTR_AUTOCOMMIT: SQLINTEGER = 102;
pub const SQL_ATTR_CONNECTION_TIMEOUT: SQLINTEGER = 113;
pub const SQL_ATTR_LOGIN_TIMEOUT: SQLINTEGER = 103;
pub const SQL_ATTR_CURRENT_CATALOG: SQLINTEGER = 109;

pub const SQL_AUTOCOMMIT_ON: SQLUINTEGER = 1;
pub const SQL_AUTOCOMMIT_OFF: SQLUINTEGER = 0;

// ============================================================================
// Statement attributes
// ============================================================================

pub const SQL_ATTR_ROW_ARRAY_SIZE: SQLINTEGER = 27;
pub const SQL_ATTR_ROW_BIND_TYPE: SQLINTEGER = 5;

// ============================================================================
// Null-terminated string sentinel
// ============================================================================

pub const SQL_NTS: SQLSMALLINT = -3;

// ============================================================================
// SQL data type constants
// ============================================================================

pub const SQL_UNKNOWN_TYPE: SQLSMALLINT = 0;
pub const SQL_CHAR: SQLSMALLINT = 1;
pub const SQL_NUMERIC: SQLSMALLINT = 2;
pub const SQL_DECIMAL: SQLSMALLINT = 3;
pub const SQL_INTEGER: SQLSMALLINT = 4;
pub const SQL_SMALLINT: SQLSMALLINT = 5;
pub const SQL_FLOAT: SQLSMALLINT = 6;
pub const SQL_REAL: SQLSMALLINT = 7;
pub const SQL_DOUBLE: SQLSMALLINT = 8;
pub const SQL_DATETIME: SQLSMALLINT = 9;
pub const SQL_VARCHAR: SQLSMALLINT = 12;
pub const SQL_TYPE_DATE: SQLSMALLINT = 91;
pub const SQL_TYPE_TIME: SQLSMALLINT = 92;
pub const SQL_TYPE_TIMESTAMP: SQLSMALLINT = 93;
pub const SQL_BIGINT: SQLSMALLINT = -5;
pub const SQL_TINYINT: SQLSMALLINT = -6;
pub const SQL_BIT: SQLSMALLINT = -7;
pub const SQL_WCHAR: SQLSMALLINT = -8;
pub const SQL_WVARCHAR: SQLSMALLINT = -9;
pub const SQL_WLONGVARCHAR: SQLSMALLINT = -10;
pub const SQL_LONGVARCHAR: SQLSMALLINT = -1;
pub const SQL_BINARY: SQLSMALLINT = -2;
pub const SQL_VARBINARY: SQLSMALLINT = -3;
pub const SQL_LONGVARBINARY: SQLSMALLINT = -4;

// ============================================================================
// C data type constants (for SQLBindCol / SQLGetData TargetType)
// ============================================================================

pub const SQL_C_CHAR: SQLSMALLINT = SQL_CHAR;
pub const SQL_C_WCHAR: SQLSMALLINT = SQL_WCHAR;
pub const SQL_C_SLONG: SQLSMALLINT = -16;
pub const SQL_C_ULONG: SQLSMALLINT = -18;
pub const SQL_C_SSHORT: SQLSMALLINT = -15;
pub const SQL_C_USHORT: SQLSMALLINT = -17;
pub const SQL_C_FLOAT: SQLSMALLINT = SQL_REAL;
pub const SQL_C_DOUBLE: SQLSMALLINT = SQL_DOUBLE;
pub const SQL_C_BIT: SQLSMALLINT = SQL_BIT;
pub const SQL_C_SBIGINT: SQLSMALLINT = -25;
pub const SQL_C_UBIGINT: SQLSMALLINT = -27;
pub const SQL_C_DEFAULT: SQLSMALLINT = 99;

// ============================================================================
// Nullable column constants
// ============================================================================

pub const SQL_NO_NULLS: SQLSMALLINT = 0;
pub const SQL_NULLABLE: SQLSMALLINT = 1;
pub const SQL_NULLABLE_UNKNOWN: SQLSMALLINT = 2;

// ============================================================================
// Null indicator
// ============================================================================

pub const SQL_NULL_DATA: SQLLEN = -1;
pub const SQL_DATA_AT_EXEC: SQLLEN = -2;

// ============================================================================
// SQLDriverConnect completion flags
// ============================================================================

pub const SQL_DRIVER_NOPROMPT: SQLUSMALLINT = 0;
pub const SQL_DRIVER_COMPLETE: SQLUSMALLINT = 1;
pub const SQL_DRIVER_PROMPT: SQLUSMALLINT = 2;
pub const SQL_DRIVER_COMPLETE_REQUIRED: SQLUSMALLINT = 3;

// ============================================================================
// SQLGetInfo information types
// ============================================================================

pub const SQL_DRIVER_NAME: SQLUSMALLINT = 6;
pub const SQL_DRIVER_VER: SQLUSMALLINT = 7;
pub const SQL_DBMS_NAME: SQLUSMALLINT = 17;
pub const SQL_DBMS_VER: SQLUSMALLINT = 18;
pub const SQL_SERVER_NAME: SQLUSMALLINT = 13;
pub const SQL_DATABASE_NAME: SQLUSMALLINT = 16;
pub const SQL_DATA_SOURCE_NAME: SQLUSMALLINT = 2;
pub const SQL_MAX_COLUMNS_IN_SELECT: SQLUSMALLINT = 100;
pub const SQL_MAX_COLUMNS_IN_TABLE: SQLUSMALLINT = 101;
pub const SQL_MAX_TABLES_IN_SELECT: SQLUSMALLINT = 106;
pub const SQL_MAX_IDENTIFIER_LEN: SQLUSMALLINT = 10005;
pub const SQL_IDENTIFIER_QUOTE_CHAR: SQLUSMALLINT = 29;
pub const SQL_SEARCH_PATTERN_ESCAPE: SQLUSMALLINT = 14;
pub const SQL_CATALOG_NAME_SEPARATOR: SQLUSMALLINT = 41;
pub const SQL_CATALOG_TERM: SQLUSMALLINT = 42;
pub const SQL_SCHEMA_TERM: SQLUSMALLINT = 39;
pub const SQL_TABLE_TERM: SQLUSMALLINT = 45;

// ============================================================================
// SQLFreeStmt options
// ============================================================================

pub const SQL_CLOSE: SQLUSMALLINT = 0;
pub const SQL_DROP: SQLUSMALLINT = 1;
pub const SQL_UNBIND: SQLUSMALLINT = 2;
pub const SQL_RESET_PARAMS: SQLUSMALLINT = 3;

// ============================================================================
// Fetch orientation
// ============================================================================

pub const SQL_FETCH_NEXT: SQLSMALLINT = 1;
pub const SQL_FETCH_FIRST: SQLSMALLINT = 2;
pub const SQL_FETCH_LAST: SQLSMALLINT = 3;
pub const SQL_FETCH_PRIOR: SQLSMALLINT = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_code_values() {
        assert_eq!(SQL_SUCCESS, 0);
        assert_eq!(SQL_SUCCESS_WITH_INFO, 1);
        assert_eq!(SQL_ERROR, -1);
        assert_eq!(SQL_INVALID_HANDLE, -2);
        assert_eq!(SQL_NO_DATA, 100);
    }

    #[test]
    fn handle_type_values() {
        assert_eq!(SQL_HANDLE_ENV, 1);
        assert_eq!(SQL_HANDLE_DBC, 2);
        assert_eq!(SQL_HANDLE_STMT, 3);
    }

    #[test]
    fn null_handle_is_null_ptr() {
        assert!(SQL_NULL_HANDLE.is_null());
    }

    #[test]
    fn data_type_constants() {
        assert_eq!(SQL_CHAR, 1);
        assert_eq!(SQL_VARCHAR, 12);
        assert_eq!(SQL_INTEGER, 4);
        assert_eq!(SQL_BIGINT, -5);
        assert_eq!(SQL_FLOAT, 6);
        assert_eq!(SQL_DOUBLE, 8);
        assert_eq!(SQL_BIT, -7);
    }
}
