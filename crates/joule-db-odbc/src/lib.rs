//! # joule-db-odbc
//!
//! ODBC 3.x driver for [JouleDB](https://joule-db.dev).
//!
//! This crate compiles to a shared library (`.dylib` on macOS, `.so` on Linux,
//! `.dll` on Windows) that implements the standard ODBC C API.  Applications
//! such as Excel, Tableau, Power BI, DBeaver, and any other ODBC-compatible
//! tool can use this driver to connect to a JouleDB server.
//!
//! ## Wire Protocol
//!
//! The driver communicates with JouleDB over TCP using a binary wire protocol:
//!
//! - **Magic bytes**: `GDB\0` (0x47, 0x44, 0x42, 0x00)
//! - **Protocol version**: 1
//! - **Header size**: 19 bytes (4 magic + 1 version + 1 flags + 1 msg type +
//!   4 payload length + 8 request ID)
//! - **Payload**: JSON-encoded
//! - **Default port**: 9000
//!
//! ## Connection String
//!
//! ```text
//! Driver={JouleDB};Server=host;Port=9000;Database=default;UID=user;PWD=pass
//! ```
//!
//! ## Exported ODBC Functions
//!
//! - `SQLAllocHandle` / `SQLFreeHandle`
//! - `SQLSetEnvAttr`
//! - `SQLConnect` / `SQLDriverConnect` / `SQLDisconnect`
//! - `SQLExecDirect`
//! - `SQLFetch` / `SQLGetData` / `SQLBindCol`
//! - `SQLNumResultCols` / `SQLDescribeCol` / `SQLRowCount`
//! - `SQLCloseCursor`
//! - `SQLGetDiagRec`
//! - `SQLGetInfo`

pub mod api;
pub mod handle;
pub mod protocol;
pub mod types;

// Re-export every ODBC entry point at the crate root so that the linker
// sees them as top-level symbols in the cdylib.
pub use api::*;
