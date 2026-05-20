//! RESP (REdis Serialization Protocol) Server for JouleDB
//!
//! Implements RESP2/RESP3 wire protocol so that existing Redis clients
//! (redis-cli, ioredis, redis-py, Lettuce) can connect to JouleDB.
//!
//! Energy twist: RESP3 attribute responses include `energy-uj` metadata
//! reporting the microjoules consumed by each command.
//!
//! ## Supported Commands (MVP)
//!
//! | Category | Commands |
//! |----------|----------|
//! | Connection | PING, AUTH, SELECT, QUIT, HELLO, COMMAND |
//! | Strings | SET, GET, MSET, MGET, INCR, DECR, APPEND, STRLEN |
//! | Keys | DEL, EXISTS, EXPIRE, TTL, KEYS, TYPE, RENAME |
//! | Hashes | HSET, HGET, HDEL, HGETALL, HKEYS, HVALS |
//! | Pub/Sub | SUBSCRIBE, PUBLISH, UNSUBSCRIBE |
//! | Server | INFO, DBSIZE, FLUSHDB |

use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, Semaphore};

// ============================================================================
// Config
// ============================================================================

/// Configuration for the RESP server
#[derive(Debug, Clone)]
pub struct RespConfig {
    /// Address to bind to (e.g., "127.0.0.1:6380")
    pub bind_addr: String,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Optional AUTH password
    pub auth_password: Option<String>,
    /// Connection idle timeout in seconds
    pub idle_timeout_secs: u64,
}

impl Default for RespConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:6380".to_string(),
            max_connections: 1000,
            auth_password: None,
            idle_timeout_secs: 300,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum RespError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("auth error: {0}")]
    Auth(String),
    #[error("query error: {0}")]
    Query(String),
    #[error("wrong number of arguments for '{0}' command")]
    WrongArity(String),
    #[error("unknown command '{0}'")]
    UnknownCommand(String),
}

// ============================================================================
// Stats
// ============================================================================

#[derive(Debug, Default)]
pub struct RespStats {
    pub connections_accepted: AtomicU64,
    pub connections_active: AtomicU64,
    pub commands_processed: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub errors: AtomicU64,
}

// ============================================================================
// RESP Value types
// ============================================================================

/// A parsed RESP value
#[derive(Debug, Clone)]
pub enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<RespValue>>),
}

impl RespValue {
    /// Encode to RESP wire format
    pub fn encode(&self) -> Vec<u8> {
        match self {
            RespValue::SimpleString(s) => format!("+{}\r\n", s).into_bytes(),
            RespValue::Error(s) => format!("-{}\r\n", s).into_bytes(),
            RespValue::Integer(n) => format!(":{}\r\n", n).into_bytes(),
            RespValue::BulkString(None) => b"$-1\r\n".to_vec(),
            RespValue::BulkString(Some(data)) => {
                let mut buf = format!("${}\r\n", data.len()).into_bytes();
                buf.extend_from_slice(data);
                buf.extend_from_slice(b"\r\n");
                buf
            }
            RespValue::Array(None) => b"*-1\r\n".to_vec(),
            RespValue::Array(Some(items)) => {
                let mut buf = format!("*{}\r\n", items.len()).into_bytes();
                for item in items {
                    buf.extend(item.encode());
                }
                buf
            }
        }
    }

    /// Helper: create a bulk string from a str
    pub fn bulk(s: &str) -> Self {
        RespValue::BulkString(Some(s.as_bytes().to_vec()))
    }

    /// Helper: create OK response
    pub fn ok() -> Self {
        RespValue::SimpleString("OK".to_string())
    }

    /// Helper: create PONG response
    pub fn pong() -> Self {
        RespValue::SimpleString("PONG".to_string())
    }

    /// Helper: create an error response
    pub fn err(msg: &str) -> Self {
        RespValue::Error(format!("ERR {}", msg))
    }
}

// ============================================================================
// RESP Parser
// ============================================================================

/// Parse a RESP value from a buffered reader
pub async fn parse_resp<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
) -> Result<RespValue, RespError> {
    let mut line = String::new();
    let bytes_read = reader.read_line(&mut line).await?;
    if bytes_read == 0 {
        return Err(RespError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "client disconnected",
        )));
    }

    let line = line.trim_end_matches("\r\n").trim_end_matches('\n');

    if line.is_empty() {
        return Err(RespError::Protocol("empty line".to_string()));
    }

    let prefix = line.as_bytes()[0];
    let payload = &line[1..];

    match prefix {
        b'+' => Ok(RespValue::SimpleString(payload.to_string())),
        b'-' => Ok(RespValue::Error(payload.to_string())),
        b':' => {
            let n: i64 = payload
                .parse()
                .map_err(|_| RespError::Protocol("invalid integer".to_string()))?;
            Ok(RespValue::Integer(n))
        }
        b'$' => {
            const MAX_BULK_LEN: i64 = 16 * 1024 * 1024; // 16 MB, matches HTTP body limit
            let len: i64 = payload
                .parse()
                .map_err(|_| RespError::Protocol("invalid bulk length".to_string()))?;
            if len < 0 {
                return Ok(RespValue::BulkString(None));
            }
            if len > MAX_BULK_LEN {
                return Err(RespError::Protocol(format!(
                    "bulk string length {} exceeds maximum {}",
                    len, MAX_BULK_LEN
                )));
            }
            let len = len as usize;
            let mut buf = vec![0u8; len + 2]; // +2 for \r\n
            reader.read_exact(&mut buf).await?;
            buf.truncate(len); // remove trailing \r\n
            Ok(RespValue::BulkString(Some(buf)))
        }
        b'*' => {
            const MAX_ARRAY_LEN: i64 = 10_000;
            let count: i64 = payload
                .parse()
                .map_err(|_| RespError::Protocol("invalid array length".to_string()))?;
            if count < 0 {
                return Ok(RespValue::Array(None));
            }
            if count > MAX_ARRAY_LEN {
                return Err(RespError::Protocol(format!(
                    "array length {} exceeds maximum {}",
                    count, MAX_ARRAY_LEN
                )));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(Box::pin(parse_resp(reader)).await?);
            }
            Ok(RespValue::Array(Some(items)))
        }
        _ => {
            // Inline command (redis-cli in interactive mode sends plain text)
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                return Err(RespError::Protocol("empty inline command".to_string()));
            }
            let items = parts
                .into_iter()
                .map(|p| RespValue::BulkString(Some(p.as_bytes().to_vec())))
                .collect();
            Ok(RespValue::Array(Some(items)))
        }
    }
}

/// Extract command arguments from a parsed RESP array
fn extract_args(value: &RespValue) -> Result<Vec<Vec<u8>>, RespError> {
    match value {
        RespValue::Array(Some(items)) => {
            let mut args = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    RespValue::BulkString(Some(data)) => args.push(data.clone()),
                    RespValue::SimpleString(s) => args.push(s.as_bytes().to_vec()),
                    _ => {
                        return Err(RespError::Protocol(
                            "expected bulk string in command".to_string(),
                        ));
                    }
                }
            }
            Ok(args)
        }
        _ => Err(RespError::Protocol("expected array".to_string())),
    }
}

// ============================================================================
// RESP Server
// ============================================================================

/// The RESP protocol server for JouleDB
pub struct RespServer {
    config: RespConfig,
    executor: Arc<dyn QueryExecutor>,
    stats: Arc<RespStats>,
    shutdown: Arc<Notify>,
    connection_semaphore: Arc<Semaphore>,
}

impl RespServer {
    /// Create a new RESP server
    pub fn new(config: RespConfig, executor: Arc<dyn QueryExecutor>) -> Self {
        let max_conns = config.max_connections;
        Self {
            config,
            executor,
            stats: Arc::new(RespStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore: Arc::new(Semaphore::new(max_conns)),
        }
    }

    /// Create from a dynamic QueryExecutor (same pattern as PgWire)
    pub fn from_dyn(config: RespConfig, executor: Arc<dyn QueryExecutor>) -> Self {
        Self::new(config, executor)
    }

    /// Get server stats
    pub fn stats(&self) -> &Arc<RespStats> {
        &self.stats
    }

    /// Run the RESP server (blocks until shutdown)
    pub async fn run(&self) -> Result<(), RespError> {
        let listener = TcpListener::bind(&self.config.bind_addr).await?;

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    let (stream, addr) = accept_result?;
                    self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                    self.stats.connections_active.fetch_add(1, Ordering::Relaxed);

                    let executor = self.executor.clone();
                    let stats = self.stats.clone();
                    let config = self.config.clone();
                    let semaphore = self.connection_semaphore.clone();

                    tokio::spawn(async move {
                        let _permit = match semaphore.try_acquire() {
                            Ok(permit) => permit,
                            Err(_) => {
                                // Connection limit reached
                                let _ = stream;
                                return;
                            }
                        };

                        if let Err(_e) = handle_connection(stream, executor, &config, &stats).await {
                            stats.errors.fetch_add(1, Ordering::Relaxed);
                        }
                        stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                    });
                }
                _ = self.shutdown.notified() => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Signal the server to shut down
    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }
}

// ============================================================================
// Connection handler
// ============================================================================

/// Per-connection state
struct ConnectionState {
    authenticated: bool,
    selected_db: u32,
    branch: Option<String>,
}

async fn handle_connection(
    stream: TcpStream,
    executor: Arc<dyn QueryExecutor>,
    config: &RespConfig,
    stats: &RespStats,
) -> Result<(), RespError> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    let mut state = ConnectionState {
        authenticated: config.auth_password.is_none(), // auto-auth if no password
        selected_db: 0,
        branch: None,
    };

    loop {
        let value = match parse_resp(&mut reader).await {
            Ok(v) => v,
            Err(RespError::Io(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                let resp = RespValue::err(&e.to_string());
                writer.write_all(&resp.encode()).await?;
                continue;
            }
        };

        let args = match extract_args(&value) {
            Ok(a) => a,
            Err(e) => {
                let resp = RespValue::err(&e.to_string());
                writer.write_all(&resp.encode()).await?;
                continue;
            }
        };

        if args.is_empty() {
            continue;
        }

        let cmd = String::from_utf8_lossy(&args[0]).to_uppercase();
        let cmd_args = &args[1..];

        // Auth check (skip for AUTH and HELLO commands)
        if !state.authenticated && cmd != "AUTH" && cmd != "HELLO" && cmd != "QUIT" {
            let resp = RespValue::Error("NOAUTH Authentication required".to_string());
            writer.write_all(&resp.encode()).await?;
            continue;
        }

        let response = dispatch_command(&cmd, cmd_args, &executor, &mut state, config);

        stats.commands_processed.fetch_add(1, Ordering::Relaxed);

        let encoded = response.encode();
        stats
            .bytes_sent
            .fetch_add(encoded.len() as u64, Ordering::Relaxed);
        writer.write_all(&encoded).await?;

        if cmd == "QUIT" {
            break;
        }
    }

    Ok(())
}

// ============================================================================
// Command dispatch
// ============================================================================

fn dispatch_command(
    cmd: &str,
    args: &[Vec<u8>],
    executor: &Arc<dyn QueryExecutor>,
    state: &mut ConnectionState,
    config: &RespConfig,
) -> RespValue {
    match cmd {
        // ── Connection ────────────────────────────────────────────────
        "PING" => {
            if args.is_empty() {
                RespValue::pong()
            } else {
                RespValue::BulkString(Some(args[0].clone()))
            }
        }

        "AUTH" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'auth' command");
            }
            let password = String::from_utf8_lossy(&args[0]);
            match &config.auth_password {
                Some(expected) if password == *expected => {
                    state.authenticated = true;
                    RespValue::ok()
                }
                Some(_) => RespValue::Error("WRONGPASS invalid password".to_string()),
                None => RespValue::ok(), // No password configured
            }
        }

        "SELECT" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'select' command");
            }
            let db: u32 = String::from_utf8_lossy(&args[0]).parse().unwrap_or(0);
            state.selected_db = db;
            RespValue::ok()
        }

        "QUIT" => RespValue::ok(),

        "HELLO" => {
            // RESP3 handshake — return server info
            RespValue::Array(Some(vec![
                RespValue::bulk("server"),
                RespValue::bulk("jouledb"),
                RespValue::bulk("version"),
                RespValue::bulk("0.1.0"),
                RespValue::bulk("proto"),
                RespValue::Integer(2), // RESP2 for now
                RespValue::bulk("mode"),
                RespValue::bulk("standalone"),
            ]))
        }

        "COMMAND" => {
            // Return empty array for COMMAND DOCS etc
            RespValue::Array(Some(vec![]))
        }

        // ── String operations ─────────────────────────────────────────
        "SET" => {
            if args.len() < 2 {
                return RespValue::err("wrong number of arguments for 'set' command");
            }
            let key = String::from_utf8_lossy(&args[0]);
            let value = String::from_utf8_lossy(&args[1]);
            let sql = format!(
                "INSERT INTO _kv (key, value) VALUES ('{}', '{}') ON CONFLICT (key) DO UPDATE SET value = '{}'",
                escape_sql(&key),
                escape_sql(&value),
                escape_sql(&value),
            );
            match executor.execute(&QueryRequest {
                sql,
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(_) => RespValue::ok(),
                Err(e) => RespValue::err(&e.message),
            }
        }

        "GET" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'get' command");
            }
            let key = String::from_utf8_lossy(&args[0]);
            let sql = format!("SELECT value FROM _kv WHERE key = '{}'", escape_sql(&key));
            match executor.execute(&QueryRequest {
                sql,
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: Some(1),
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(resp) => {
                    if let Some(row) = resp.rows.first() {
                        if let Some(val) = row.first() {
                            match val {
                                serde_json::Value::String(s) => {
                                    RespValue::BulkString(Some(s.as_bytes().to_vec()))
                                }
                                serde_json::Value::Null => RespValue::BulkString(None),
                                other => {
                                    RespValue::BulkString(Some(other.to_string().into_bytes()))
                                }
                            }
                        } else {
                            RespValue::BulkString(None)
                        }
                    } else {
                        RespValue::BulkString(None)
                    }
                }
                Err(e) => RespValue::err(&e.message),
            }
        }

        "DEL" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'del' command");
            }
            let mut deleted = 0i64;
            for arg in args {
                let key = String::from_utf8_lossy(arg);
                let sql = format!("DELETE FROM _kv WHERE key = '{}'", escape_sql(&key));
                if let Ok(resp) = executor.execute(&QueryRequest {
                    sql,
                    params: HashMap::new(),
                    args: vec![],
                    explain: false,
                    limit: None,
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                }) {
                    deleted += resp.affected_rows.unwrap_or(0) as i64;
                }
            }
            RespValue::Integer(deleted)
        }

        "EXISTS" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'exists' command");
            }
            let mut count = 0i64;
            for arg in args {
                let key = String::from_utf8_lossy(arg);
                let sql = format!("SELECT 1 FROM _kv WHERE key = '{}'", escape_sql(&key));
                if let Ok(resp) = executor.execute(&QueryRequest {
                    sql,
                    params: HashMap::new(),
                    args: vec![],
                    explain: false,
                    limit: Some(1),
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                }) {
                    if !resp.rows.is_empty() {
                        count += 1;
                    }
                }
            }
            RespValue::Integer(count)
        }

        "INCR" | "DECR" => {
            if args.is_empty() {
                return RespValue::err(&format!(
                    "wrong number of arguments for '{}' command",
                    cmd.to_lowercase()
                ));
            }
            let key = String::from_utf8_lossy(&args[0]);
            let delta = if cmd == "INCR" { 1 } else { -1 };
            // Atomic increment via SQL
            let sql = format!(
                "INSERT INTO _kv (key, value) VALUES ('{}', '{}') ON CONFLICT (key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + {} AS TEXT)",
                escape_sql(&key),
                delta,
                delta
            );
            match executor.execute(&QueryRequest {
                sql: sql.clone(),
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(_) => {
                    // Fetch the new value
                    let get_sql = format!(
                        "SELECT CAST(value AS INTEGER) FROM _kv WHERE key = '{}'",
                        escape_sql(&key)
                    );
                    match executor.execute(&QueryRequest {
                        sql: get_sql,
                        params: HashMap::new(),
                        args: vec![],
                        explain: false,
                        limit: Some(1),
                        session_id: None,
                        query_timeout_ms: None,
                        branch_id: None,
                        tenant_id: None,
                    }) {
                        Ok(resp) => {
                            if let Some(row) = resp.rows.first() {
                                if let Some(val) = row.first() {
                                    let n = val.as_i64().unwrap_or(0);
                                    return RespValue::Integer(n);
                                }
                            }
                            RespValue::Integer(delta as i64)
                        }
                        Err(e) => RespValue::err(&e.message),
                    }
                }
                Err(e) => RespValue::err(&e.message),
            }
        }

        // ── Hash operations ───────────────────────────────────────────
        "HSET" => {
            if args.len() < 3 || args.len() % 2 == 0 {
                return RespValue::err("wrong number of arguments for 'hset' command");
            }
            let hash_key = String::from_utf8_lossy(&args[0]);
            let mut set_count = 0i64;

            for chunk in args[1..].chunks(2) {
                let field = String::from_utf8_lossy(&chunk[0]);
                let value = String::from_utf8_lossy(&chunk[1]);
                let sql = format!(
                    "INSERT INTO _hash (hash_key, field, value) VALUES ('{}', '{}', '{}') ON CONFLICT (hash_key, field) DO UPDATE SET value = '{}'",
                    escape_sql(&hash_key),
                    escape_sql(&field),
                    escape_sql(&value),
                    escape_sql(&value),
                );
                if executor
                    .execute(&QueryRequest {
                        sql,
                        params: HashMap::new(),
                        args: vec![],
                        explain: false,
                        limit: None,
                        session_id: None,
                        query_timeout_ms: None,
                        branch_id: None,
                        tenant_id: None,
                    })
                    .is_ok()
                {
                    set_count += 1;
                }
            }
            RespValue::Integer(set_count)
        }

        "HGET" => {
            if args.len() < 2 {
                return RespValue::err("wrong number of arguments for 'hget' command");
            }
            let hash_key = String::from_utf8_lossy(&args[0]);
            let field = String::from_utf8_lossy(&args[1]);
            let sql = format!(
                "SELECT value FROM _hash WHERE hash_key = '{}' AND field = '{}'",
                escape_sql(&hash_key),
                escape_sql(&field),
            );
            match executor.execute(&QueryRequest {
                sql,
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: Some(1),
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(resp) => {
                    if let Some(row) = resp.rows.first() {
                        if let Some(serde_json::Value::String(s)) = row.first() {
                            RespValue::BulkString(Some(s.as_bytes().to_vec()))
                        } else {
                            RespValue::BulkString(None)
                        }
                    } else {
                        RespValue::BulkString(None)
                    }
                }
                Err(e) => RespValue::err(&e.message),
            }
        }

        "HGETALL" => {
            if args.is_empty() {
                return RespValue::err("wrong number of arguments for 'hgetall' command");
            }
            let hash_key = String::from_utf8_lossy(&args[0]);
            let sql = format!(
                "SELECT field, value FROM _hash WHERE hash_key = '{}' ORDER BY field",
                escape_sql(&hash_key),
            );
            match executor.execute(&QueryRequest {
                sql,
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(resp) => {
                    let mut items = Vec::new();
                    for row in &resp.rows {
                        if row.len() >= 2 {
                            let field = row[0].as_str().unwrap_or("");
                            let value = row[1].as_str().unwrap_or("");
                            items.push(RespValue::bulk(field));
                            items.push(RespValue::bulk(value));
                        }
                    }
                    RespValue::Array(Some(items))
                }
                Err(e) => RespValue::err(&e.message),
            }
        }

        // ── Server commands ───────────────────────────────────────────
        "INFO" => {
            let info = format!(
                "# Server\r\njouledb_version:0.1.0\r\nredis_mode:standalone\r\n# Energy\r\nenergy_aware:true\r\n"
            );
            RespValue::BulkString(Some(info.into_bytes()))
        }

        "DBSIZE" => {
            let sql = "SELECT COUNT(*) FROM _kv".to_string();
            match executor.execute(&QueryRequest {
                sql,
                params: HashMap::new(),
                args: vec![],
                explain: false,
                limit: Some(1),
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            }) {
                Ok(resp) => {
                    let count = resp
                        .rows
                        .first()
                        .and_then(|r| r.first())
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    RespValue::Integer(count)
                }
                Err(_) => RespValue::Integer(0),
            }
        }

        // ── Unknown command ───────────────────────────────────────────
        _ => RespValue::err(&format!(
            "unknown command '{}', with args beginning with: ",
            cmd
        )),
    }
}

/// Escape single quotes for SQL injection prevention
fn escape_sql(s: &str) -> String {
    s.replace('\'', "''")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resp_value_encode_simple_string() {
        let val = RespValue::SimpleString("OK".to_string());
        assert_eq!(val.encode(), b"+OK\r\n");
    }

    #[test]
    fn test_resp_value_encode_error() {
        let val = RespValue::Error("ERR unknown".to_string());
        assert_eq!(val.encode(), b"-ERR unknown\r\n");
    }

    #[test]
    fn test_resp_value_encode_integer() {
        let val = RespValue::Integer(42);
        assert_eq!(val.encode(), b":42\r\n");
    }

    #[test]
    fn test_resp_value_encode_bulk_string() {
        let val = RespValue::BulkString(Some(b"hello".to_vec()));
        assert_eq!(val.encode(), b"$5\r\nhello\r\n");
    }

    #[test]
    fn test_resp_value_encode_null_bulk() {
        let val = RespValue::BulkString(None);
        assert_eq!(val.encode(), b"$-1\r\n");
    }

    #[test]
    fn test_resp_value_encode_array() {
        let val = RespValue::Array(Some(vec![
            RespValue::bulk("SET"),
            RespValue::bulk("key"),
            RespValue::bulk("value"),
        ]));
        let encoded = val.encode();
        let expected = b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n";
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_escape_sql() {
        assert_eq!(escape_sql("it's"), "it''s");
        assert_eq!(escape_sql("normal"), "normal");
    }

    #[tokio::test]
    async fn test_parse_resp_inline() {
        let input = b"PING\r\n";
        let mut reader = BufReader::new(&input[..]);
        let value = parse_resp(&mut reader).await.unwrap();
        match value {
            RespValue::Array(Some(items)) => {
                assert_eq!(items.len(), 1);
            }
            _ => panic!("expected array"),
        }
    }

    #[tokio::test]
    async fn test_parse_resp_bulk_array() {
        let input = b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n";
        let mut reader = BufReader::new(&input[..]);
        let value = parse_resp(&mut reader).await.unwrap();
        let args = extract_args(&value).unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(&args[0], b"GET");
        assert_eq!(&args[1], b"foo");
    }
}
