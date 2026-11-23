//! JouleDB binary wire protocol client.
//!
//! Implements the JouleDB TCP binary protocol for sending queries, executions,
//! and pings over a `std::net::TcpStream`.
//!
//! ## Wire Format
//!
//! ```text
//! +--------+---------+-------+----------+-------------+-----------+
//! | Magic  | Version | Flags | Msg Type | Payload Len | Req ID    |
//! | 4 bytes| 1 byte  |1 byte | 1 byte   | 4 bytes     | 8 bytes   |
//! +--------+---------+-------+----------+-------------+-----------+
//! |                     Payload (JSON)                             |
//! +----------------------------------------------------------------+
//! ```
//!
//! Total header: 19 bytes.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ============================================================================
// Protocol constants
// ============================================================================

/// Magic bytes: `GDB\0` (0x47, 0x44, 0x42, 0x00).
pub const MAGIC: [u8; 4] = [0x47, 0x44, 0x42, 0x00];

/// Protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 19;

/// Maximum payload size (16 MiB).
pub const MAX_PAYLOAD_SIZE: u32 = 16 * 1024 * 1024;

/// Default JouleDB TCP port.
pub const DEFAULT_PORT: u16 = 9000;

// ============================================================================
// Message type constants
// ============================================================================

/// Request: SQL query (expects a result set).
pub const MSG_QUERY: u8 = 0x10;

/// Request: execute statement (DDL/DML, no result set).
pub const MSG_EXECUTE: u8 = 0x12;

/// Request: ping/keepalive.
pub const MSG_PING: u8 = 0xF0;

/// Response: query result set.
pub const MSG_QUERY_RESPONSE: u8 = 0x90;

/// Response: execution result (affected rows).
pub const MSG_EXECUTE_RESPONSE: u8 = 0x92;

/// Response: error.
pub const MSG_ERROR: u8 = 0xFF;

/// Response: pong (ping reply).
pub const MSG_PONG: u8 = 0xF1;

// ============================================================================
// Error type
// ============================================================================

/// Protocol-level errors.
#[derive(Debug)]
pub enum ProtocolError {
    /// I/O error from the TCP socket.
    Io(io::Error),
    /// Invalid data received from the server.
    Protocol(String),
    /// Server returned an error message.
    Server(String),
    /// JSON serialization/deserialization failure.
    Json(serde_json::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Io(e) => write!(f, "I/O error: {}", e),
            ProtocolError::Protocol(msg) => write!(f, "protocol error: {}", msg),
            ProtocolError::Server(msg) => write!(f, "server error: {}", msg),
            ProtocolError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<io::Error> for ProtocolError {
    fn from(e: io::Error) -> Self {
        ProtocolError::Io(e)
    }
}

impl From<serde_json::Error> for ProtocolError {
    fn from(e: serde_json::Error) -> Self {
        ProtocolError::Json(e)
    }
}

// ============================================================================
// Request/response payload types (JSON)
// ============================================================================

/// JSON payload for a query request.
#[derive(Debug, Serialize)]
pub struct QueryRequest {
    pub sql: String,
}

/// JSON payload for an execute request.
#[derive(Debug, Serialize)]
pub struct ExecuteRequest {
    pub sql: String,
}

/// A single column descriptor in a query response.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseColumn {
    pub name: String,
    #[serde(default = "default_col_type")]
    pub col_type: String,
}

fn default_col_type() -> String {
    "VARCHAR".to_string()
}

/// JSON payload for a query response.
#[derive(Debug, Deserialize)]
pub struct QueryResponse {
    /// Column metadata.
    #[serde(default)]
    pub columns: Vec<ResponseColumn>,
    /// Row data — each row is a vector of JSON values.
    #[serde(default)]
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Number of rows affected (for DML).
    #[serde(default)]
    pub affected_rows: Option<usize>,
    /// Server-side execution time in milliseconds.
    #[serde(default)]
    pub execution_time_ms: Option<f64>,
    /// Whether the result set was truncated.
    #[serde(default)]
    pub truncated: Option<bool>,
    /// Warning messages from the server.
    #[serde(default)]
    pub warnings: Option<Vec<String>>,
}

/// JSON payload for an execute response.
#[derive(Debug, Deserialize)]
pub struct ExecuteResponse {
    /// Number of rows affected.
    #[serde(default)]
    pub affected_rows: i64,
}

/// JSON payload for a server error.
#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    /// Error message text.
    #[serde(default)]
    pub error: String,
    /// Optional error code.
    #[serde(default)]
    pub code: Option<String>,
}

// ============================================================================
// Connection
// ============================================================================

/// Establish a TCP connection to JouleDB.
pub fn connect(host: &str, port: u16) -> Result<TcpStream, ProtocolError> {
    let addr = format!("{}:{}", host, port);
    let stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| ProtocolError::Protocol(format!("invalid address '{}': {}", addr, e)))?,
        Duration::from_secs(10),
    )?;
    // Set read/write timeouts for the ongoing connection.
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    stream.set_nodelay(true)?;
    Ok(stream)
}

// ============================================================================
// Message encoding
// ============================================================================

/// Encode a full wire-protocol message (header + payload).
fn encode_message(msg_type: u8, request_id: u64, payload: &[u8]) -> Vec<u8> {
    let payload_len = payload.len() as u32;
    let mut buf = Vec::with_capacity(HEADER_SIZE + payload.len());

    // Header (19 bytes)
    buf.extend_from_slice(&MAGIC); // 4 bytes: magic
    buf.push(PROTOCOL_VERSION); // 1 byte:  version
    buf.push(0x00); // 1 byte:  flags (reserved)
    buf.push(msg_type); // 1 byte:  message type
    buf.extend_from_slice(&payload_len.to_le_bytes()); // 4 bytes: payload length
    buf.extend_from_slice(&request_id.to_le_bytes()); // 8 bytes: request ID

    // Payload
    buf.extend_from_slice(payload);

    buf
}

/// Read exactly `n` bytes from the stream.
fn read_exact(stream: &mut TcpStream, n: usize) -> Result<Vec<u8>, ProtocolError> {
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read a complete response message from the stream.
/// Returns `(msg_type, request_id, payload_bytes)`.
fn read_response(stream: &mut TcpStream) -> Result<(u8, u64, Vec<u8>), ProtocolError> {
    let header = read_exact(stream, HEADER_SIZE)?;

    // Validate magic bytes.
    if header[0..4] != MAGIC {
        return Err(ProtocolError::Protocol(format!(
            "invalid magic bytes: {:02X} {:02X} {:02X} {:02X}",
            header[0], header[1], header[2], header[3]
        )));
    }

    // Validate version.
    let version = header[4];
    if version != PROTOCOL_VERSION {
        return Err(ProtocolError::Protocol(format!(
            "unsupported protocol version: {}",
            version
        )));
    }

    // Parse header fields.
    let _flags = header[5];
    let msg_type = header[6];
    let payload_len = u32::from_le_bytes([header[7], header[8], header[9], header[10]]);
    let request_id = u64::from_le_bytes([
        header[11], header[12], header[13], header[14], header[15], header[16], header[17],
        header[18],
    ]);

    if payload_len > MAX_PAYLOAD_SIZE {
        return Err(ProtocolError::Protocol(format!(
            "payload too large: {} bytes (max {})",
            payload_len, MAX_PAYLOAD_SIZE
        )));
    }

    let payload = if payload_len > 0 {
        read_exact(stream, payload_len as usize)?
    } else {
        Vec::new()
    };

    Ok((msg_type, request_id, payload))
}

// ============================================================================
// High-level operations
// ============================================================================

/// Send a SQL query and receive the result set.
pub fn send_query(
    stream: &mut TcpStream,
    sql: &str,
    request_id: u64,
) -> Result<QueryResponse, ProtocolError> {
    let req = QueryRequest {
        sql: sql.to_string(),
    };
    let payload = serde_json::to_vec(&req)?;
    let msg = encode_message(MSG_QUERY, request_id, &payload);
    stream.write_all(&msg)?;
    stream.flush()?;

    let (resp_type, _resp_id, resp_payload) = read_response(stream)?;

    match resp_type {
        MSG_QUERY_RESPONSE => {
            let response: QueryResponse = serde_json::from_slice(&resp_payload)?;
            Ok(response)
        }
        MSG_ERROR => {
            let err: ErrorResponse =
                serde_json::from_slice(&resp_payload).unwrap_or_else(|_| ErrorResponse {
                    error: String::from_utf8_lossy(&resp_payload).to_string(),
                    code: None,
                });
            Err(ProtocolError::Server(err.error))
        }
        other => Err(ProtocolError::Protocol(format!(
            "unexpected response type: 0x{:02X}",
            other
        ))),
    }
}

/// Send an execute statement (DDL/DML) and return the number of affected rows.
pub fn send_execute(
    stream: &mut TcpStream,
    sql: &str,
    request_id: u64,
) -> Result<i64, ProtocolError> {
    let req = ExecuteRequest {
        sql: sql.to_string(),
    };
    let payload = serde_json::to_vec(&req)?;
    let msg = encode_message(MSG_EXECUTE, request_id, &payload);
    stream.write_all(&msg)?;
    stream.flush()?;

    let (resp_type, _resp_id, resp_payload) = read_response(stream)?;

    match resp_type {
        MSG_EXECUTE_RESPONSE => {
            let response: ExecuteResponse = serde_json::from_slice(&resp_payload)?;
            Ok(response.affected_rows)
        }
        // Some servers return a QueryResponse for all SQL — handle gracefully.
        MSG_QUERY_RESPONSE => {
            let response: QueryResponse = serde_json::from_slice(&resp_payload)?;
            Ok(response.affected_rows.unwrap_or(0) as i64)
        }
        MSG_ERROR => {
            let err: ErrorResponse =
                serde_json::from_slice(&resp_payload).unwrap_or_else(|_| ErrorResponse {
                    error: String::from_utf8_lossy(&resp_payload).to_string(),
                    code: None,
                });
            Err(ProtocolError::Server(err.error))
        }
        other => Err(ProtocolError::Protocol(format!(
            "unexpected response type: 0x{:02X}",
            other
        ))),
    }
}

/// Send a ping and wait for a pong response.
pub fn ping(stream: &mut TcpStream, request_id: u64) -> Result<(), ProtocolError> {
    let msg = encode_message(MSG_PING, request_id, &[]);
    stream.write_all(&msg)?;
    stream.flush()?;

    let (resp_type, _resp_id, _resp_payload) = read_response(stream)?;

    match resp_type {
        MSG_PONG => Ok(()),
        MSG_ERROR => {
            let err: ErrorResponse =
                serde_json::from_slice(&_resp_payload).unwrap_or_else(|_| ErrorResponse {
                    error: "ping failed".to_string(),
                    code: None,
                });
            Err(ProtocolError::Server(err.error))
        }
        other => Err(ProtocolError::Protocol(format!(
            "expected PONG (0x{:02X}), got 0x{:02X}",
            MSG_PONG, other
        ))),
    }
}

// ============================================================================
// Utility: map JouleDB column type string to ODBC SQL type
// ============================================================================

/// Map a JouleDB type name (from the QueryResponse column metadata) to an ODBC
/// SQL data type constant.
pub fn jouledb_type_to_sql(type_name: &str) -> i16 {
    match type_name.to_uppercase().as_str() {
        "INTEGER" | "INT" | "INT4" => crate::types::SQL_INTEGER,
        "BIGINT" | "INT8" => crate::types::SQL_BIGINT,
        "SMALLINT" | "INT2" => crate::types::SQL_SMALLINT,
        "TINYINT" | "INT1" => crate::types::SQL_TINYINT,
        "FLOAT" | "FLOAT4" | "REAL" => crate::types::SQL_REAL,
        "DOUBLE" | "FLOAT8" | "DOUBLE PRECISION" => crate::types::SQL_DOUBLE,
        "NUMERIC" | "DECIMAL" => crate::types::SQL_NUMERIC,
        "BOOLEAN" | "BOOL" | "BIT" => crate::types::SQL_BIT,
        "DATE" => crate::types::SQL_TYPE_DATE,
        "TIME" => crate::types::SQL_TYPE_TIME,
        "TIMESTAMP" | "DATETIME" => crate::types::SQL_TYPE_TIMESTAMP,
        "CHAR" => crate::types::SQL_CHAR,
        "BLOB" | "BYTEA" | "BINARY" | "VARBINARY" => crate::types::SQL_VARBINARY,
        "TEXT" | "VARCHAR" | "STRING" | _ => crate::types::SQL_VARCHAR,
    }
}

/// Return a default column display size for a given ODBC SQL type.
pub fn default_column_size(sql_type: i16) -> usize {
    match sql_type {
        crate::types::SQL_BIT => 1,
        crate::types::SQL_TINYINT => 3,
        crate::types::SQL_SMALLINT => 5,
        crate::types::SQL_INTEGER => 10,
        crate::types::SQL_BIGINT => 19,
        crate::types::SQL_REAL => 7,
        crate::types::SQL_DOUBLE => 15,
        crate::types::SQL_NUMERIC => 38,
        crate::types::SQL_TYPE_DATE => 10,
        crate::types::SQL_TYPE_TIME => 8,
        crate::types::SQL_TYPE_TIMESTAMP => 26,
        crate::types::SQL_CHAR => 1,
        crate::types::SQL_VARBINARY => 65535,
        _ => 255, // VARCHAR and others
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_message_header_layout() {
        let payload = b"hello";
        let msg = encode_message(MSG_QUERY, 42, payload);

        assert_eq!(msg.len(), HEADER_SIZE + payload.len());

        // Magic
        assert_eq!(&msg[0..4], &MAGIC);
        // Version
        assert_eq!(msg[4], PROTOCOL_VERSION);
        // Flags
        assert_eq!(msg[5], 0x00);
        // Message type
        assert_eq!(msg[6], MSG_QUERY);
        // Payload length (LE)
        let plen = u32::from_le_bytes([msg[7], msg[8], msg[9], msg[10]]);
        assert_eq!(plen, 5);
        // Request ID (LE)
        let rid = u64::from_le_bytes([
            msg[11], msg[12], msg[13], msg[14], msg[15], msg[16], msg[17], msg[18],
        ]);
        assert_eq!(rid, 42);
        // Payload
        assert_eq!(&msg[19..], b"hello");
    }

    #[test]
    fn encode_message_empty_payload() {
        let msg = encode_message(MSG_PING, 1, &[]);
        assert_eq!(msg.len(), HEADER_SIZE);
        let plen = u32::from_le_bytes([msg[7], msg[8], msg[9], msg[10]]);
        assert_eq!(plen, 0);
    }

    #[test]
    fn type_mapping_varchar() {
        assert_eq!(jouledb_type_to_sql("VARCHAR"), crate::types::SQL_VARCHAR);
        assert_eq!(jouledb_type_to_sql("TEXT"), crate::types::SQL_VARCHAR);
        assert_eq!(jouledb_type_to_sql("STRING"), crate::types::SQL_VARCHAR);
    }

    #[test]
    fn type_mapping_integer() {
        assert_eq!(jouledb_type_to_sql("INTEGER"), crate::types::SQL_INTEGER);
        assert_eq!(jouledb_type_to_sql("INT"), crate::types::SQL_INTEGER);
        assert_eq!(jouledb_type_to_sql("BIGINT"), crate::types::SQL_BIGINT);
    }

    #[test]
    fn type_mapping_float() {
        assert_eq!(jouledb_type_to_sql("FLOAT"), crate::types::SQL_REAL);
        assert_eq!(jouledb_type_to_sql("DOUBLE"), crate::types::SQL_DOUBLE);
    }

    #[test]
    fn type_mapping_bool() {
        assert_eq!(jouledb_type_to_sql("BOOLEAN"), crate::types::SQL_BIT);
        assert_eq!(jouledb_type_to_sql("BIT"), crate::types::SQL_BIT);
    }

    #[test]
    fn type_mapping_datetime() {
        assert_eq!(jouledb_type_to_sql("DATE"), crate::types::SQL_TYPE_DATE);
        assert_eq!(
            jouledb_type_to_sql("TIMESTAMP"),
            crate::types::SQL_TYPE_TIMESTAMP
        );
    }

    #[test]
    fn type_mapping_case_insensitive() {
        assert_eq!(jouledb_type_to_sql("integer"), crate::types::SQL_INTEGER);
        assert_eq!(jouledb_type_to_sql("Integer"), crate::types::SQL_INTEGER);
        assert_eq!(jouledb_type_to_sql("varchar"), crate::types::SQL_VARCHAR);
    }

    #[test]
    fn type_mapping_unknown_defaults_to_varchar() {
        assert_eq!(
            jouledb_type_to_sql("UNKNOWN_TYPE"),
            crate::types::SQL_VARCHAR
        );
    }

    #[test]
    fn default_column_sizes() {
        assert_eq!(default_column_size(crate::types::SQL_INTEGER), 10);
        assert_eq!(default_column_size(crate::types::SQL_BIGINT), 19);
        assert_eq!(default_column_size(crate::types::SQL_VARCHAR), 255);
        assert_eq!(default_column_size(crate::types::SQL_BIT), 1);
    }

    #[test]
    fn query_response_deserialize() {
        let json = r#"{
            "columns": [
                {"name": "id", "col_type": "INTEGER"},
                {"name": "name", "col_type": "VARCHAR"}
            ],
            "rows": [[1, "Alice"], [2, "Bob"]],
            "affected_rows": null,
            "execution_time_ms": 1.5
        }"#;
        let resp: QueryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.columns.len(), 2);
        assert_eq!(resp.columns[0].name, "id");
        assert_eq!(resp.columns[0].col_type, "INTEGER");
        assert_eq!(resp.rows.len(), 2);
    }

    #[test]
    fn query_response_deserialize_minimal() {
        // Server may return minimal fields.
        let json = r#"{"columns":[],"rows":[]}"#;
        let resp: QueryResponse = serde_json::from_str(json).unwrap();
        assert!(resp.columns.is_empty());
        assert!(resp.rows.is_empty());
        assert!(resp.affected_rows.is_none());
    }

    #[test]
    fn execute_response_deserialize() {
        let json = r#"{"affected_rows": 42}"#;
        let resp: ExecuteResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.affected_rows, 42);
    }

    #[test]
    fn error_response_deserialize() {
        let json = r#"{"error": "table not found", "code": "42S02"}"#;
        let resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error, "table not found");
        assert_eq!(resp.code.as_deref(), Some("42S02"));
    }

    #[test]
    fn message_type_constants() {
        assert_eq!(MSG_QUERY, 0x10);
        assert_eq!(MSG_EXECUTE, 0x12);
        assert_eq!(MSG_PING, 0xF0);
        assert_eq!(MSG_QUERY_RESPONSE, 0x90);
        assert_eq!(MSG_EXECUTE_RESPONSE, 0x92);
        assert_eq!(MSG_ERROR, 0xFF);
    }

    #[test]
    fn header_size_is_19() {
        assert_eq!(HEADER_SIZE, 19);
    }

    #[test]
    fn magic_bytes() {
        assert_eq!(MAGIC, [0x47, 0x44, 0x42, 0x00]);
        // 'G' 'D' 'B' '\0'
        assert_eq!(MAGIC[0], b'G');
        assert_eq!(MAGIC[1], b'D');
        assert_eq!(MAGIC[2], b'B');
        assert_eq!(MAGIC[3], 0);
    }
}
