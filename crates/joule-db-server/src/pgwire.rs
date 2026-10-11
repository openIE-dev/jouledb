//! PostgreSQL Wire Protocol v3 Implementation
//!
//! Enables JouleDB to accept connections from any PostgreSQL-compatible client
//! (psql, pgAdmin, JDBC, libpq, SQLAlchemy, Prisma, etc.).
//!
//! ## Protocol Overview
//!
//! PostgreSQL wire protocol v3 uses length-prefixed messages:
//!
//! ```text
//! Frontend (client) → Backend (server):
//!   Startup: [Length(4)] [Version(4)] [key=val\0 pairs] [\0]
//!   Query:   ['Q'] [Length(4)] [sql\0]
//!   Parse:   ['P'] [Length(4)] [name\0] [sql\0] [param_count(2)] [oid...]
//!   Bind:    ['B'] [Length(4)] [portal\0] [stmt\0] [formats] [values] [result_formats]
//!   Execute: ['E'] [Length(4)] [portal\0] [max_rows(4)]
//!   Sync:    ['S'] [Length(4=4)]
//!
//! Backend (server) → Frontend (client):
//!   AuthOk:          ['R'] [Length(4)] [0(4)]
//!   ParameterStatus: ['S'] [Length(4)] [key\0] [value\0]
//!   ReadyForQuery:   ['Z'] [Length(4)] [status(1)]  // 'I'=idle, 'T'=in_tx, 'E'=error
//!   RowDescription:  ['T'] [Length(4)] [num_fields(2)] [field...]
//!   DataRow:         ['D'] [Length(4)] [num_cols(2)] [col_len(4) col_data...]
//!   CommandComplete: ['C'] [Length(4)] [tag\0]
//!   ErrorResponse:   ['E'] [Length(4)] [field_type(1) value\0]... [\0]
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, Semaphore};

use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};

// Protocol version 3.0
const PROTOCOL_VERSION_3: i32 = 196608; // (3 << 16) | 0
// Maximum PgWire message body size (256 MB) — prevents OOM from malicious clients
const MAX_PGWIRE_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
// Idle connection timeout — drop connections with no activity (5 minutes)
const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
// SSL request magic number
const SSL_REQUEST: i32 = 80877103;
// Cancel request magic number
const CANCEL_REQUEST: i32 = 80877102;

/// PostgreSQL wire protocol server configuration
#[derive(Clone)]
pub struct PgWireConfig {
    /// Bind address (e.g., "0.0.0.0:5433")
    pub bind_addr: String,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Server version string reported to clients
    pub server_version: String,
    /// Whether authentication is required
    pub auth_enabled: bool,
    /// Password for cleartext authentication (when auth_enabled, fallback)
    pub auth_password: Option<String>,
    /// RBAC manager for per-user SCRAM-SHA-256 authentication
    pub rbac_manager: Option<std::sync::Arc<crate::rbac::RBACManager>>,
    /// Query timeout in milliseconds (0 = no timeout)
    pub query_timeout_ms: u64,
    /// TLS acceptor for SSL connections (behind `tls` feature flag)
    #[cfg(feature = "tls")]
    pub tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    /// Require TLS — reject plaintext connections
    #[cfg(feature = "tls")]
    pub require_tls: bool,
}

impl std::fmt::Debug for PgWireConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PgWireConfig")
            .field("bind_addr", &self.bind_addr)
            .field("max_connections", &self.max_connections)
            .field("auth_enabled", &self.auth_enabled)
            .field("auth_password", &self.auth_password.as_ref().map(|_| "***"))
            .finish()
    }
}

impl Default for PgWireConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:5433".to_string(),
            max_connections: 256,
            connection_timeout_secs: 300,
            server_version: "16.0.0 JouleDB 0.1.0".to_string(),
            auth_enabled: false,
            auth_password: None,
            rbac_manager: None,
            query_timeout_ms: 0,
            #[cfg(feature = "tls")]
            tls_acceptor: None,
            #[cfg(feature = "tls")]
            require_tls: false,
        }
    }
}

/// Statistics for the pgwire server
#[derive(Debug, Default)]
pub struct PgWireStats {
    pub connections_accepted: AtomicU64,
    pub connections_active: AtomicU64,
    pub queries_executed: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub errors: AtomicU64,
}

impl PgWireStats {
    pub fn snapshot(&self) -> PgWireStatsSnapshot {
        PgWireStatsSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            connections_active: self.connections_active.load(Ordering::Relaxed),
            queries_executed: self.queries_executed.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PgWireStatsSnapshot {
    pub connections_accepted: u64,
    pub connections_active: u64,
    pub queries_executed: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub errors: u64,
}

/// PostgreSQL wire protocol server
pub struct PgWireServer {
    config: PgWireConfig,
    executor: Arc<dyn QueryExecutor>,
    stats: Arc<PgWireStats>,
    shutdown: Arc<Notify>,
    connection_semaphore: Arc<Semaphore>,
}

impl PgWireServer {
    pub fn new<E: QueryExecutor + 'static>(config: PgWireConfig, executor: Arc<E>) -> Self {
        let max_connections = config.max_connections;
        Self {
            config,
            executor: executor as Arc<dyn QueryExecutor>,
            stats: Arc::new(PgWireStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore: Arc::new(Semaphore::new(max_connections)),
        }
    }

    pub fn from_dyn(config: PgWireConfig, executor: Arc<dyn QueryExecutor>) -> Self {
        let max_connections = config.max_connections;
        Self {
            config,
            executor,
            stats: Arc::new(PgWireStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore: Arc::new(Semaphore::new(max_connections)),
        }
    }

    pub fn stats(&self) -> &Arc<PgWireStats> {
        &self.stats
    }

    /// Signal the server to shut down
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Run the pgwire server
    pub async fn run(&self) -> Result<(), PgWireError> {
        let listener = TcpListener::bind(&self.config.bind_addr)
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;

        tracing::info!("PgWire server listening on {}", self.config.bind_addr);

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, addr)) => {
                            self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);

                            let permit = match self.connection_semaphore.clone().try_acquire_owned() {
                                Ok(permit) => permit,
                                Err(_) => {
                                    tracing::warn!("Max connections reached, rejecting {}", addr);
                                    continue;
                                }
                            };

                            self.stats.connections_active.fetch_add(1, Ordering::Relaxed);
                            let executor = self.executor.clone();
                            let stats = self.stats.clone();
                            let config = self.config.clone();

                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, executor, &config, &stats).await {
                                    tracing::debug!("Connection error from {}: {}", addr, e);
                                    stats.errors.fetch_add(1, Ordering::Relaxed);
                                }
                                stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                                drop(permit);
                            });
                        }
                        Err(e) => {
                            tracing::error!("Accept error: {}", e);
                        }
                    }
                }
                _ = self.shutdown.notified() => {
                    tracing::info!("PgWire server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Handle a single client connection through the full protocol lifecycle
async fn handle_connection(
    mut stream: TcpStream,
    executor: Arc<dyn QueryExecutor>,
    config: &PgWireConfig,
    stats: &PgWireStats,
) -> Result<(), PgWireError> {
    stream.set_nodelay(true).ok();

    // Phase 0: Check for SSL request before splitting the stream
    // Read first 8 bytes: length(4) + version/code(4)
    let mut initial = [0u8; 8];
    stream
        .read_exact(&mut initial)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    let _length = i32::from_be_bytes([initial[0], initial[1], initial[2], initial[3]]);
    let version = i32::from_be_bytes([initial[4], initial[5], initial[6], initial[7]]);

    if version == SSL_REQUEST {
        #[cfg(feature = "tls")]
        if let Some(ref tls_acceptor) = config.tls_acceptor {
            // Accept SSL: send 'S', then do TLS handshake
            stream
                .write_all(&[b'S'])
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            stream
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;

            let tls_stream = tls_acceptor
                .accept(stream)
                .await
                .map_err(|e| PgWireError::Io(format!("TLS handshake failed: {}", e)))?;

            let (tls_reader, tls_writer) = tokio::io::split(tls_stream);
            return handle_connection_inner(
                BufReader::new(tls_reader),
                BufWriter::new(tls_writer),
                executor,
                config,
                stats,
                None,
            )
            .await;
        }

        // No TLS configured: decline SSL, read real startup
        stream
            .write_all(&[b'N'])
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        stream
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;

        // Client will send a new startup message
        let (reader, writer) = stream.into_split();
        return handle_connection_inner(
            BufReader::new(reader),
            BufWriter::new(writer),
            executor,
            config,
            stats,
            None,
        )
        .await;
    }

    // Not an SSL request — it's a regular startup message
    // If TLS is required, reject plaintext connections
    #[cfg(feature = "tls")]
    if config.require_tls {
        tracing::warn!("PgWire: Rejecting plaintext connection (require_tls=true)");
        // Send an error response and close
        let (_, mut writer) = stream.into_split();
        let error_msg = "SSL required: plaintext connections are not allowed";
        if let Err(e) = write_error_response(&mut writer, "FATAL", error_msg).await {
            tracing::error!("PgWire: Failed to send TLS-required error: {}", e);
        }
        return Err(PgWireError::Protocol(
            "TLS required but client connected without SSL".into(),
        ));
    }

    // Pass the already-read 8 bytes as prefix
    let (reader, writer) = stream.into_split();
    handle_connection_inner(
        BufReader::new(reader),
        BufWriter::new(writer),
        executor,
        config,
        stats,
        Some(initial),
    )
    .await
}

/// Inner connection handler, generic over stream type
async fn handle_connection_inner<R, W>(
    mut reader: R,
    mut writer: W,
    executor: Arc<dyn QueryExecutor>,
    config: &PgWireConfig,
    stats: &PgWireStats,
    startup_prefix: Option<[u8; 8]>,
) -> Result<(), PgWireError>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // Phase 1: Startup handshake
    let params =
        startup_handshake_with_prefix(&mut reader, &mut writer, config, startup_prefix).await?;

    // Phase 2: Main query loop with connection state
    let user_id = params
        .get("user")
        .cloned()
        .unwrap_or_else(|| "anonymous".to_string());
    let mut conn_state = ConnectionState::with_user(user_id.clone(), config.auth_enabled);

    // Load actual roles from RBAC if available
    if let Some(ref rbac) = config.rbac_manager {
        if let Ok(user) = rbac.get_user(&user_id) {
            conn_state.roles = user.roles.iter().cloned().collect();
        }
    }

    loop {
        // Read message type byte (with idle timeout to prevent connection hoarding)
        let msg_type =
            match tokio::time::timeout(IDLE_TIMEOUT, AsyncReadExt::read_u8(&mut reader)).await {
                Ok(Ok(b)) => b,
                Ok(Err(_)) => break, // Client disconnected
                Err(_) => break,     // Idle timeout — reclaim connection slot
            };

        // Read message length (includes self but not type byte)
        let length = AsyncReadExt::read_i32(&mut reader)
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))? as usize;

        if length < 4 {
            return Err(PgWireError::Protocol("Invalid message length".into()));
        }

        let body_len = length - 4;

        // Reject oversized messages to prevent OOM from malicious clients
        if body_len > MAX_PGWIRE_MESSAGE_SIZE {
            write_error_response(
                &mut writer,
                "08P01",
                &format!(
                    "Message too large: {} bytes (max {})",
                    body_len, MAX_PGWIRE_MESSAGE_SIZE
                ),
            )
            .await?;
            return Err(PgWireError::Protocol(format!(
                "Message size {} exceeds maximum {}",
                body_len, MAX_PGWIRE_MESSAGE_SIZE
            )));
        }

        match msg_type {
            b'Q' => {
                // Simple Query
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                // Strip null terminator
                let sql =
                    String::from_utf8_lossy(&body[..body.len().saturating_sub(1)]).to_string();
                let sql = sql.trim().to_string();

                if sql.is_empty() {
                    write_empty_query_response(&mut writer).await?;
                } else if let Some(copy) = parse_copy(&sql) {
                    let timeout = (config.query_timeout_ms > 0).then_some(config.query_timeout_ms);
                    handle_copy(
                        copy,
                        &mut reader,
                        &mut writer,
                        &executor,
                        &mut conn_state,
                        timeout,
                    )
                    .await?;
                } else {
                    let timeout = if config.query_timeout_ms > 0 {
                        Some(config.query_timeout_ms)
                    } else {
                        None
                    };
                    execute_and_write(
                        &sql,
                        &executor,
                        &mut writer,
                        stats,
                        &mut conn_state,
                        timeout,
                    )
                    .await?;
                }

                // ReadyForQuery with current transaction status
                write_ready_for_query(&mut writer, conn_state.transaction_status).await?;
                writer
                    .flush()
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
            }

            b'P' => {
                // Parse (Extended Query) — extract statement name and SQL
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                let (stmt_name, offset) = extract_cstring(&body, 0);
                let (sql, offset2) = extract_cstring(&body, offset);
                let param_count = if offset2 + 2 <= body.len() {
                    i16::from_be_bytes([body[offset2], body[offset2 + 1]]) as usize
                } else {
                    0
                };
                let mut oids = Vec::with_capacity(param_count);
                for k in 0..param_count {
                    let o = offset2 + 2 + k * 4;
                    if o + 4 > body.len() {
                        break;
                    }
                    oids.push(i32::from_be_bytes([
                        body[o],
                        body[o + 1],
                        body[o + 2],
                        body[o + 3],
                    ]));
                }
                let oid_key = stmt_name.clone();
                conn_state.param_oids.insert(oid_key, oids);

                // Store in prepared statement cache
                let cache_name = if stmt_name.is_empty() {
                    "".to_string()
                } else {
                    stmt_name
                };
                conn_state
                    .prepared_statements
                    .insert(cache_name, sql, param_count);

                // Send ParseComplete
                write_message(&mut writer, b'1', &[]).await?;
            }

            b'B' => {
                // Bind — extract portal name, statement name, and parameters
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                let (portal_name, offset) = extract_cstring(&body, 0);
                let (stmt_name, mut offset) = extract_cstring(&body, offset);

                // Parameter format codes (0 text, 1 binary; one code applies to all)
                let mut param_formats: Vec<i16> = Vec::new();
                if offset + 2 <= body.len() {
                    let n = i16::from_be_bytes([body[offset], body[offset + 1]]) as usize;
                    offset += 2;
                    let skip = n.saturating_mul(2);
                    if offset + skip > body.len() {
                        write_error_response(
                            &mut writer,
                            "08P01",
                            "Malformed Bind message: format codes exceed message length",
                        )
                        .await?;
                        continue;
                    }
                    for k in 0..n {
                        param_formats.push(i16::from_be_bytes([
                            body[offset + 2 * k],
                            body[offset + 2 * k + 1],
                        ]));
                    }
                    offset += skip;
                }
                let stmt_oids = conn_state
                    .param_oids
                    .get(&stmt_name)
                    .cloned()
                    .unwrap_or_default();

                // Parse parameter values
                let mut params = Vec::new();
                let num_params = if offset + 2 <= body.len() {
                    let n = i16::from_be_bytes([body[offset], body[offset + 1]]) as usize;
                    offset += 2;
                    n
                } else {
                    0
                };

                for _ in 0..num_params {
                    if offset + 4 > body.len() {
                        break;
                    }
                    let param_len = i32::from_be_bytes([
                        body[offset],
                        body[offset + 1],
                        body[offset + 2],
                        body[offset + 3],
                    ]);
                    offset += 4;
                    if param_len == -1 {
                        params.push(serde_json::Value::Null);
                    } else if param_len < 0 {
                        // Negative lengths other than -1 (NULL) are invalid per PG protocol
                        break;
                    } else {
                        let param_len = param_len as usize;
                        if offset + param_len <= body.len() {
                            let k = params.len();
                            let fmt = match param_formats.len() {
                                0 => 0,
                                1 => param_formats[0],
                                _ => param_formats.get(k).copied().unwrap_or(0),
                            };
                            let oid = stmt_oids.get(k).copied().unwrap_or(0);
                            params.push(decode_param(&body[offset..offset + param_len], fmt, oid));
                            offset += param_len;
                        }
                    }
                }

                // Result-column format codes
                let mut result_formats: Vec<i16> = Vec::new();
                if offset + 2 <= body.len() {
                    let n = i16::from_be_bytes([body[offset], body[offset + 1]]) as usize;
                    offset += 2;
                    for k in 0..n {
                        if offset + 2 * k + 2 > body.len() {
                            break;
                        }
                        result_formats.push(i16::from_be_bytes([
                            body[offset + 2 * k],
                            body[offset + 2 * k + 1],
                        ]));
                    }
                }

                // Look up the prepared statement and create a portal
                let lookup_name = if stmt_name.is_empty() {
                    "".to_string()
                } else {
                    stmt_name.clone()
                };
                if let Some(stmt) = conn_state.prepared_statements.get(&lookup_name) {
                    let sql = stmt.sql.clone();
                    conn_state.portals.insert(
                        portal_name,
                        Portal {
                            statement_name: lookup_name,
                            sql,
                            params,
                            result_formats,
                            described: None,
                        },
                    );
                    write_message(&mut writer, b'2', &[]).await?; // BindComplete
                } else {
                    write_error_response(
                        &mut writer,
                        "26000",
                        &format!("prepared statement \"{}\" does not exist", stmt_name),
                    )
                    .await?;
                }
            }

            b'D' => {
                // Describe
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                if !body.is_empty() {
                    let describe_type = body[0]; // 'S' = statement, 'P' = portal
                    let (name, _) = extract_cstring(&body, 1);

                    if describe_type == b'S' {
                        // Describe Statement — send ParameterDescription + RowDescription
                        let param_count = conn_state
                            .prepared_statements
                            .get(&name)
                            .map(|s| s.param_count.max(count_placeholders(&s.sql)))
                            .unwrap_or(0);
                        let declared = conn_state
                            .param_oids
                            .get(&name)
                            .cloned()
                            .unwrap_or_default();
                        // ParameterDescription
                        let mut pd = Vec::new();
                        pd.extend_from_slice(&(param_count as i16).to_be_bytes());
                        for k in 0..param_count {
                            let oid = declared.get(k).copied().filter(|o| *o != 0).unwrap_or(25);
                            pd.extend_from_slice(&oid.to_be_bytes());
                        }
                        write_message(&mut writer, b't', &pd).await?;
                        // Try to extract column names from SQL
                        let cols = conn_state
                            .prepared_statements
                            .get(&name)
                            .and_then(|s| extract_select_columns(&s.sql));
                        if let Some(columns) = cols {
                            write_row_description(&mut writer, &columns).await?;
                        } else {
                            write_message(&mut writer, b'n', &[]).await?;
                        }
                    } else {
                        // Describe Portal: a row-returning statement is executed here and
                        // its result cached, so the RowDescription carries the real columns
                        // and types and Execute sends only DataRows (the protocol forbids a
                        // second RowDescription there).
                        let target = conn_state
                            .portals
                            .get(&name)
                            .map(|p| (bound_sql(p), p.result_formats.clone()));
                        match target {
                            Some((sql, formats)) if returns_rows(&sql) => {
                                let timeout = (config.query_timeout_ms > 0)
                                    .then_some(config.query_timeout_ms);
                                let res = run_statement(&sql, &executor, &mut conn_state, timeout);
                                match &res {
                                    Ok(r) => {
                                        let oids = column_oids(&r.rows, r.columns.len(), &formats);
                                        write_row_description_fmt(
                                            &mut writer,
                                            &r.columns,
                                            &oids,
                                            &formats,
                                        )
                                        .await?;
                                    }
                                    Err(_) => write_message(&mut writer, b'n', &[]).await?,
                                }
                                if let Some(p) = conn_state.portals.get_mut(&name) {
                                    p.described = Some(res);
                                }
                            }
                            _ => write_message(&mut writer, b'n', &[]).await?,
                        }
                    }
                } else {
                    write_message(&mut writer, b'n', &[]).await?;
                }
            }

            b'E' => {
                // Execute — look up portal and execute query
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                let (portal_name, _offset) = extract_cstring(&body, 0);

                if let Some(portal) = conn_state.portals.get_mut(&portal_name) {
                    let sql = bound_sql(portal);
                    let formats = portal.result_formats.clone();
                    let cached = portal.described.take();
                    let timeout = if config.query_timeout_ms > 0 {
                        Some(config.query_timeout_ms)
                    } else {
                        None
                    };
                    if let Some(copy) = parse_copy(&sql) {
                        handle_copy(
                            copy,
                            &mut reader,
                            &mut writer,
                            &executor,
                            &mut conn_state,
                            timeout,
                        )
                        .await?;
                        continue;
                    }
                    stats.queries_executed.fetch_add(1, Ordering::Relaxed);
                    let res = match cached {
                        Some(r) => r,
                        None => run_statement(&sql, &executor, &mut conn_state, timeout),
                    };
                    write_statement_result(
                        &sql,
                        res,
                        &mut writer,
                        stats,
                        &mut conn_state,
                        false,
                        &formats,
                    )
                    .await?;
                } else {
                    write_error_response(
                        &mut writer,
                        "34000",
                        &format!("portal \"{}\" does not exist", portal_name),
                    )
                    .await?;
                }
            }

            b'S' => {
                // Sync — marks end of extended query protocol message group
                if body_len > 0 {
                    let mut body = vec![0u8; body_len];
                    reader
                        .read_exact(&mut body)
                        .await
                        .map_err(|e| PgWireError::Io(e.to_string()))?;
                }
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);

                // Sync in error state: if we were in an explicit transaction (T→E),
                // stay in error state until ROLLBACK. If error was outside a transaction
                // (I→E), reset to idle.
                if conn_state.transaction_status == b'E' && !conn_state.in_explicit_transaction {
                    conn_state.transaction_status = b'I';
                }
                write_ready_for_query(&mut writer, conn_state.transaction_status).await?;
                writer
                    .flush()
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
            }

            b'C' => {
                // Close — destroy a prepared statement or portal
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);
                // Send CloseComplete ('3')
                writer
                    .write_u8(b'3')
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                writer
                    .write_all(&4i32.to_be_bytes())
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
            }

            b'H' => {
                // Flush — send all pending output immediately
                let mut body = vec![0u8; body_len];
                reader
                    .read_exact(&mut body)
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                stats
                    .bytes_received
                    .fetch_add(body_len as u64 + 5, Ordering::Relaxed);
                writer
                    .flush()
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
            }

            b'X' => {
                // Terminate
                break;
            }

            other => {
                // Skip unknown messages
                if body_len > 0 {
                    let mut body = vec![0u8; body_len];
                    reader
                        .read_exact(&mut body)
                        .await
                        .map_err(|e| PgWireError::Io(e.to_string()))?;
                }
                tracing::debug!("Unknown message type: {}", other as char);
            }
        }
    }

    Ok(())
}

/// Perform the startup handshake
async fn startup_handshake<R, W>(
    reader: &mut R,
    writer: &mut W,
    config: &PgWireConfig,
) -> Result<HashMap<String, String>, PgWireError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // Read startup message length
    let length = reader
        .read_i32()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))? as usize;

    if length < 8 || length > 10000 {
        return Err(PgWireError::Protocol(
            "Invalid startup message length".into(),
        ));
    }

    // Read version
    let version = reader
        .read_i32()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Handle SSL request
    if version == SSL_REQUEST {
        // Decline SSL (send 'N')
        writer
            .write_u8(b'N')
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;

        // Now read the real startup message
        return Box::pin(startup_handshake(reader, writer, config)).await;
    }

    // Handle cancel request — per PostgreSQL protocol, cancel requests arrive on
    // a separate temporary connection that should be silently closed after processing.
    if version == CANCEL_REQUEST {
        let mut _body = vec![0u8; length - 8];
        reader
            .read_exact(&mut _body)
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        // Silently close the connection (cancel requests don't get responses)
        return Ok(HashMap::new());
    }

    if version != PROTOCOL_VERSION_3 {
        return Err(PgWireError::Protocol(format!(
            "Unsupported protocol version: {}.{}",
            version >> 16,
            version & 0xFFFF
        )));
    }

    // Read startup parameters (key=val\0 pairs, terminated by \0)
    let remaining = length - 8;
    let mut body = vec![0u8; remaining];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    let mut params = HashMap::new();
    let mut i = 0;
    while i < body.len() {
        if body[i] == 0 {
            break;
        }
        // Read key
        let key_start = i;
        while i < body.len() && body[i] != 0 {
            i += 1;
        }
        let key = String::from_utf8_lossy(&body[key_start..i]).to_string();
        i += 1; // skip null

        // Read value
        let val_start = i;
        while i < body.len() && body[i] != 0 {
            i += 1;
        }
        let val = String::from_utf8_lossy(&body[val_start..i]).to_string();
        i += 1; // skip null

        params.insert(key, val);
    }

    // Authentication phase
    if config.auth_enabled {
        let username = params.get("user").cloned().unwrap_or_default();

        // Try SCRAM-SHA-256 first if RBAC manager has credentials for this user
        let scram_creds = config
            .rbac_manager
            .as_ref()
            .and_then(|rbac| get_user_scram_credentials(rbac, &username));

        if let Some(ref creds) = scram_creds {
            // SCRAM-SHA-256 authentication
            do_scram_auth(reader, writer, creds).await?;
        } else if let Some(ref expected_password) = config.auth_password {
            // Fallback: cleartext password authentication
            // Send AuthenticationCleartextPassword (R message, type=3)
            let mut auth_cleartext = Vec::with_capacity(8);
            auth_cleartext.extend_from_slice(&8i32.to_be_bytes()); // length
            auth_cleartext.extend_from_slice(&3i32.to_be_bytes()); // auth type 3 = cleartext
            writer
                .write_u8(b'R')
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            writer
                .write_all(&auth_cleartext)
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            writer
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;

            // Read PasswordMessage: 'p' + length(4) + password\0
            let msg_type = reader
                .read_u8()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            if msg_type != b'p' {
                write_error_response(writer, "28P01", "Expected password message").await?;
                return Err(PgWireError::Auth("Expected password message".into()));
            }
            let pw_length = reader
                .read_i32()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))? as usize;
            if pw_length < 5 || pw_length > 1024 {
                write_error_response(writer, "28P01", "Invalid password message length").await?;
                return Err(PgWireError::Auth("Invalid password message length".into()));
            }
            let mut pw_body = vec![0u8; pw_length - 4];
            reader
                .read_exact(&mut pw_body)
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            // Strip null terminator
            let password =
                String::from_utf8_lossy(&pw_body[..pw_body.len().saturating_sub(1)]).to_string();

            if password != *expected_password || expected_password.is_empty() {
                write_error_response(writer, "28P01", "password authentication failed").await?;
                writer
                    .flush()
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                return Err(PgWireError::Auth("password authentication failed".into()));
            }
        } else {
            // Auth enabled but no password configured and no SCRAM credentials — reject
            write_error_response(
                writer,
                "28P01",
                "authentication required but no password configured",
            )
            .await?;
            writer
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            return Err(PgWireError::Auth("no password configured".into()));
        }
    }

    // Send AuthenticationOk
    let mut auth_ok = Vec::with_capacity(8);
    auth_ok.extend_from_slice(&8i32.to_be_bytes()); // length
    auth_ok.extend_from_slice(&0i32.to_be_bytes()); // auth type 0 = ok
    writer
        .write_u8(b'R')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&auth_ok)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Send ParameterStatus messages
    let status_params = [
        ("server_version", config.server_version.as_str()),
        ("server_encoding", "UTF8"),
        ("client_encoding", "UTF8"),
        ("DateStyle", "ISO, MDY"),
        ("integer_datetimes", "on"),
        ("standard_conforming_strings", "on"),
    ];

    for (key, value) in &status_params {
        write_parameter_status(writer, key, value).await?;
    }

    // Send BackendKeyData (process_id, secret_key)
    let mut key_data = Vec::with_capacity(12);
    key_data.extend_from_slice(&12i32.to_be_bytes()); // length
    key_data.extend_from_slice(&std::process::id().to_be_bytes()); // process_id
    key_data.extend_from_slice(&0i32.to_be_bytes()); // secret_key
    writer
        .write_u8(b'K')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&key_data)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Send ReadyForQuery (Idle)
    write_ready_for_query(writer, b'I').await?;
    writer
        .flush()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    Ok(params)
}

/// Perform startup handshake with optional pre-read prefix bytes.
///
/// When calling from `handle_connection`, the first 8 bytes (length + version)
/// may have already been read to detect SSL requests. Those bytes are passed
/// as `prefix` so we can reconstruct the startup message.
async fn startup_handshake_with_prefix<R, W>(
    reader: &mut R,
    writer: &mut W,
    config: &PgWireConfig,
    prefix: Option<[u8; 8]>,
) -> Result<HashMap<String, String>, PgWireError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    if let Some(initial) = prefix {
        // We already have the first 8 bytes — reconstruct length and version
        let length = i32::from_be_bytes([initial[0], initial[1], initial[2], initial[3]]) as usize;
        let version = i32::from_be_bytes([initial[4], initial[5], initial[6], initial[7]]);

        if length < 8 || length > 10000 {
            return Err(PgWireError::Protocol(
                "Invalid startup message length".into(),
            ));
        }

        if version == CANCEL_REQUEST {
            let mut _body = vec![0u8; length - 8];
            reader
                .read_exact(&mut _body)
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            // Silently close — cancel requests don't get responses
            return Ok(HashMap::new());
        }

        if version != PROTOCOL_VERSION_3 {
            return Err(PgWireError::Protocol(format!(
                "Unsupported protocol version: {}.{}",
                version >> 16,
                version & 0xFFFF
            )));
        }

        // Read remaining startup body after the 8 bytes we already have
        let remaining = length - 8;
        let mut body = vec![0u8; remaining];
        reader
            .read_exact(&mut body)
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;

        // Parse params from body
        let mut params = HashMap::new();
        let mut i = 0;
        while i < body.len() {
            if body[i] == 0 {
                break;
            }
            let key_start = i;
            while i < body.len() && body[i] != 0 {
                i += 1;
            }
            let key = String::from_utf8_lossy(&body[key_start..i]).to_string();
            i += 1;
            let val_start = i;
            while i < body.len() && body[i] != 0 {
                i += 1;
            }
            let val = String::from_utf8_lossy(&body[val_start..i]).to_string();
            i += 1;
            params.insert(key, val);
        }

        // Run auth + send AuthOk + ParameterStatus + ReadyForQuery
        do_auth_and_ready(reader, writer, config, params).await
    } else {
        // No prefix — read from scratch (TLS path already read SSL request)
        startup_handshake(reader, writer, config).await
    }
}

/// Shared auth + parameter status + ready-for-query logic
async fn do_auth_and_ready<R, W>(
    reader: &mut R,
    writer: &mut W,
    config: &PgWireConfig,
    params: HashMap<String, String>,
) -> Result<HashMap<String, String>, PgWireError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // Authentication phase
    if config.auth_enabled {
        let username = params.get("user").cloned().unwrap_or_default();

        // Try SCRAM-SHA-256 first if RBAC manager has credentials for this user
        let scram_creds = config
            .rbac_manager
            .as_ref()
            .and_then(|rbac| get_user_scram_credentials(rbac, &username));

        if let Some(ref creds) = scram_creds {
            // SCRAM-SHA-256 authentication
            do_scram_auth(reader, writer, creds).await?;
        } else if let Some(ref expected_password) = config.auth_password {
            // Fallback: cleartext password authentication
            let mut auth_cleartext = Vec::with_capacity(8);
            auth_cleartext.extend_from_slice(&8i32.to_be_bytes());
            auth_cleartext.extend_from_slice(&3i32.to_be_bytes());
            writer
                .write_u8(b'R')
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            writer
                .write_all(&auth_cleartext)
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            writer
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;

            let msg_type = reader
                .read_u8()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            if msg_type != b'p' {
                write_error_response(writer, "28P01", "Expected password message").await?;
                return Err(PgWireError::Auth("Expected password message".into()));
            }
            let pw_length = reader
                .read_i32()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))? as usize;
            if pw_length < 5 || pw_length > 1024 {
                write_error_response(writer, "28P01", "Invalid password message length").await?;
                return Err(PgWireError::Auth("Invalid password message length".into()));
            }
            let mut pw_body = vec![0u8; pw_length - 4];
            reader
                .read_exact(&mut pw_body)
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            let password =
                String::from_utf8_lossy(&pw_body[..pw_body.len().saturating_sub(1)]).to_string();

            if password != *expected_password || expected_password.is_empty() {
                write_error_response(writer, "28P01", "password authentication failed").await?;
                writer
                    .flush()
                    .await
                    .map_err(|e| PgWireError::Io(e.to_string()))?;
                return Err(PgWireError::Auth("password authentication failed".into()));
            }
        } else {
            write_error_response(
                writer,
                "28P01",
                "authentication required but no password configured",
            )
            .await?;
            writer
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            return Err(PgWireError::Auth("no password configured".into()));
        }
    }

    // Send AuthenticationOk
    let mut auth_ok = Vec::with_capacity(8);
    auth_ok.extend_from_slice(&8i32.to_be_bytes());
    auth_ok.extend_from_slice(&0i32.to_be_bytes());
    writer
        .write_u8(b'R')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&auth_ok)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Send ParameterStatus messages
    let status_params = [
        ("server_version", config.server_version.as_str()),
        ("server_encoding", "UTF8"),
        ("client_encoding", "UTF8"),
        ("DateStyle", "ISO, MDY"),
        ("integer_datetimes", "on"),
        ("standard_conforming_strings", "on"),
    ];
    for (key, value) in &status_params {
        write_parameter_status(writer, key, value).await?;
    }

    // Send BackendKeyData
    let mut key_data = Vec::with_capacity(12);
    key_data.extend_from_slice(&12i32.to_be_bytes());
    key_data.extend_from_slice(&std::process::id().to_be_bytes());
    key_data.extend_from_slice(&0i32.to_be_bytes());
    writer
        .write_u8(b'K')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&key_data)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Send ReadyForQuery (Idle)
    write_ready_for_query(writer, b'I').await?;
    writer
        .flush()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    Ok(params)
}

fn is_tx_begin(sql: &str) -> bool {
    let u = sql.trim_start().to_ascii_uppercase();
    u.starts_with("BEGIN") || u.starts_with("START TRANSACTION")
}

fn is_tx_end(sql: &str) -> bool {
    let u = sql.trim_start().to_ascii_uppercase();
    u.starts_with("COMMIT")
        || u.starts_with("ROLLBACK")
        || u.starts_with("END")
        || u.starts_with("ABORT")
}

/// Infer PostgreSQL type OIDs from the first row of data
fn infer_type_oids(columns: &[String], rows: &[Vec<serde_json::Value>]) -> Vec<i32> {
    if let Some(first_row) = rows.first() {
        first_row.iter().map(|v| value_to_pg_oid(v)).collect()
    } else {
        vec![25; columns.len()] // default to text for empty results
    }
}

/// Run one statement for this connection, carrying its MVCC session.
fn run_statement(
    sql: &str,
    executor: &Arc<dyn QueryExecutor>,
    conn_state: &mut ConnectionState,
    query_timeout_ms: Option<u64>,
) -> Result<QueryResponse, QueryErrorResponse> {
    if let Some(r) = show_setting(sql).or_else(|| type_info_lookup(sql)) {
        return Ok(r);
    }
    {
        // Session commands drivers issue that the SQL engine has no use for.
        let t = sql.trim().trim_end_matches(';').trim();
        let u = t.to_ascii_uppercase();
        let mut w = u.split_whitespace();
        let first = w.next().unwrap_or("");
        if first == "DEALLOCATE" || u == "DISCARD ALL" {
            let name = t
                .split_whitespace()
                .last()
                .unwrap_or("")
                .trim_matches('"')
                .to_string();
            if u == "DISCARD ALL" || name.eq_ignore_ascii_case("ALL") {
                conn_state.prepared_statements = PreparedStatementCache::new(256);
                conn_state.param_oids.clear();
            } else {
                conn_state.prepared_statements.remove(&name);
                conn_state.param_oids.remove(&name);
            }
            return Ok(empty_response());
        }
        if first == "SET" || first == "RESET" {
            // Session GUCs (extra_float_digits, application_name, DateStyle,
            // TimeZone, ...) are accepted and have no effect on results.
            return Ok(empty_response());
        }
    }
    // RBAC: check write permission before executing
    crate::query::check_write_permission(&conn_state.user_id, &conn_state.roles, sql)?;
    let request = QueryRequest {
        sql: strip_pg_catalog(sql),
        params: HashMap::new(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: conn_state.session_id.clone(),
        query_timeout_ms,
        branch_id: None,
        tenant_id: None,
    };
    let result = executor.execute(&request);
    if is_tx_end(sql) {
        // COMMIT/ROLLBACK end the session whether or not they succeeded.
        conn_state.session_id = None;
    }
    if let Ok(response) = &result {
        conn_state.update_transaction_status(sql);
        if is_tx_begin(sql) && response.session_id.is_some() {
            conn_state.session_id = response.session_id.clone();
        }
    }
    result
}

/// Write a statement's result. `describe` sends the RowDescription (simple
/// protocol); the extended protocol already sent it from Describe.
async fn write_statement_result<W: AsyncWriteExt + Unpin>(
    sql: &str,
    result: Result<QueryResponse, QueryErrorResponse>,
    writer: &mut W,
    stats: &PgWireStats,
    conn_state: &mut ConnectionState,
    describe: bool,
    formats: &[i16],
) -> Result<(), PgWireError> {
    match result {
        Ok(response) => {
            let oids = column_oids(&response.rows, response.columns.len(), formats);
            if describe && (!response.columns.is_empty() || returns_rows(sql)) {
                write_row_description_fmt(writer, &response.columns, &oids, formats).await?;
            }
            for row in &response.rows {
                write_data_row_fmt(writer, row, &oids, formats).await?;
            }
            write_command_complete(writer, &command_tag(sql, &response)).await?;
            let bytes_sent = response.rows.len() * response.columns.len() * 8; // estimate
            stats
                .bytes_sent
                .fetch_add(bytes_sent as u64, Ordering::Relaxed);
        }
        Err(err) => {
            conn_state.set_error_in_transaction();
            // Map internal error codes to PostgreSQL SQLSTATE codes
            let sqlstate = match err.code.as_str() {
                "QUERY_TIMEOUT" => "57014", // query_canceled
                other => other,
            };
            write_error_response(writer, sqlstate, &err.message).await?;
            stats.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}

/// Execute a SQL query and write results to the PgWire stream (simple protocol)
async fn execute_and_write<W: AsyncWriteExt + Unpin>(
    sql: &str,
    executor: &Arc<dyn QueryExecutor>,
    writer: &mut W,
    stats: &PgWireStats,
    conn_state: &mut ConnectionState,
    query_timeout_ms: Option<u64>,
) -> Result<(), PgWireError> {
    stats.queries_executed.fetch_add(1, Ordering::Relaxed);
    let res = run_statement(sql, executor, conn_state, query_timeout_ms);
    write_statement_result(sql, res, writer, stats, conn_state, true, &[]).await
}

/// `SHOW <setting>` for the session settings drivers read on connect
/// (SQLAlchemy, JDBC, BI tools). `SHOW TABLES`/`SHOW COLUMNS` go to the executor.
fn show_setting(sql: &str) -> Option<QueryResponse> {
    let t = sql.trim().trim_end_matches(';').trim();
    let mut w = t.split_whitespace();
    if !w.next()?.eq_ignore_ascii_case("SHOW") {
        return None;
    }
    let name = w.collect::<Vec<_>>().join(" ").to_ascii_lowercase();
    let value = match name.as_str() {
        "standard_conforming_strings" => "on",
        "transaction isolation level"
        | "transaction_isolation"
        | "default_transaction_isolation" => "repeatable read",
        "server_version" => "16.0",
        "server_version_num" => "160000",
        "server_encoding" | "client_encoding" => "UTF8",
        "datestyle" => "ISO, MDY",
        "timezone" => "UTC",
        "integer_datetimes" => "on",
        "search_path" => "\"$user\", public",
        "transaction_read_only" | "default_transaction_read_only" => "off",
        "max_identifier_length" => "63",
        "is_superuser" => "on",
        _ => return None,
    };
    Some(QueryResponse {
        columns: vec![name.replace(' ', "_")],
        rows: vec![vec![serde_json::json!(value)]],
        affected_rows: None,
        execution_time_ms: 0,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

/// psycopg's `TypeInfo.fetch` (SQLAlchemy runs it for hstore on connect) looks
/// up extension types by name with `'name'::regtype`. JouleDB has no extension
/// types, so it answers with zero rows, which psycopg reads as "not installed".
fn type_info_lookup(sql: &str) -> Option<QueryResponse> {
    let l = sql.to_ascii_lowercase();
    if !(l.contains("from pg_type t") && l.contains("::regtype")) {
        return None;
    }
    Some(QueryResponse {
        columns: ["name", "oid", "array_oid", "regtype", "delimiter"]
            .map(String::from)
            .to_vec(),
        rows: Vec::new(),
        affected_rows: None,
        execution_time_ms: 0,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

/// `pg_catalog.pg_class` -> `pg_class` (the catalog tables live unqualified),
/// skipping string literals.
fn strip_pg_catalog(sql: &str) -> String {
    const P: &str = "pg_catalog.";
    let mut out = String::with_capacity(sql.len());
    let mut in_str = false;
    let mut i = 0;
    let b = sql.as_bytes();
    while i < b.len() {
        if b[i] == b'\'' {
            in_str = !in_str;
        }
        if !in_str
            && b.len() - i >= P.len()
            && b[i..i + P.len()].eq_ignore_ascii_case(P.as_bytes())
            && (i == 0
                || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'))
        {
            i += P.len();
            continue;
        }
        let ch = sql[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn empty_response() -> QueryResponse {
    QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: None,
        execution_time_ms: 0,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    }
}

fn command_tag(sql: &str, response: &QueryResponse) -> String {
    let u = sql.trim_start().to_ascii_uppercase();
    let word = u.split_whitespace().next().unwrap_or("");
    let n = response.affected_rows.unwrap_or(0);
    match word {
        "INSERT" => format!("INSERT 0 {n}"),
        "UPDATE" => format!("UPDATE {n}"),
        "DELETE" => format!("DELETE {n}"),
        "BEGIN" | "START" => "BEGIN".into(),
        "COMMIT" | "END" => "COMMIT".into(),
        "ROLLBACK" | "ABORT" => "ROLLBACK".into(),
        "CREATE" | "DROP" | "ALTER" => {
            let obj = u.split_whitespace().nth(1).unwrap_or("TABLE");
            format!("{word} {obj}")
        }
        "SET" => "SET".into(),
        "SHOW" => "SHOW".into(),
        "RESET" => "RESET".into(),
        "DEALLOCATE" => "DEALLOCATE".into(),
        "DISCARD" => "DISCARD ALL".into(),
        _ => format!("SELECT {}", response.rows.len()),
    }
}

// ============================================================================
// COPY (text and CSV, FROM STDIN and TO STDOUT)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
struct CopySpec {
    /// Table name, or the inner query for `COPY (query) TO`
    target: String,
    is_query: bool,
    columns: Vec<String>,
    from_stdin: bool,
    csv: bool,
    header: bool,
    delimiter: char,
    null: String,
}

/// Parse `COPY t [(a, b)] FROM STDIN | COPY t|(query) TO STDOUT [WITH] [(options)] | [CSV] [HEADER]`.
fn parse_copy(sql: &str) -> Option<CopySpec> {
    let t = sql.trim().trim_end_matches(';').trim();
    if !t.get(..5)?.eq_ignore_ascii_case("COPY ") {
        return None;
    }
    let mut rest = t[5..].trim_start();
    let (target, is_query) = if rest.starts_with('(') {
        let mut depth = 0i32;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let e = end?;
        let q = rest[1..e].trim().to_string();
        rest = rest[e + 1..].trim_start();
        (q, true)
    } else {
        let e = rest
            .find(|c: char| c.is_whitespace() || c == '(')
            .unwrap_or(rest.len());
        let name = rest[..e].trim_matches('"').to_string();
        rest = rest[e..].trim_start();
        (name, false)
    };
    let mut columns = Vec::new();
    if !is_query && rest.starts_with('(') {
        let e = rest.find(')')?;
        columns = rest[1..e]
            .split(',')
            .map(|c| c.trim().trim_matches('"').to_string())
            .filter(|c| !c.is_empty())
            .collect();
        rest = rest[e + 1..].trim_start();
    }
    let upper = rest.to_ascii_uppercase();
    let from_stdin = if upper.starts_with("FROM STDIN") {
        true
    } else if upper.starts_with("TO STDOUT") {
        false
    } else {
        return None; // server-side files are not supported over the wire
    };
    let opts = rest[if from_stdin { 10 } else { 9 }..].to_string();
    let ou = opts.to_ascii_uppercase();
    let csv = ou.contains("CSV");
    let header = ou.contains("HEADER") && !ou.contains("HEADER FALSE") && !ou.contains("HEADER 0");
    let quoted = |key: &str| -> Option<String> {
        let i = ou.find(key)?;
        let after = &opts[i + key.len()..];
        let a = after.find('\'')?;
        let b = after[a + 1..].find('\'')?;
        Some(after[a + 1..a + 1 + b].to_string())
    };
    let delimiter = quoted("DELIMITER")
        .and_then(|d| {
            if d == "\\t" {
                Some('\t')
            } else {
                d.chars().next()
            }
        })
        .unwrap_or(if csv { ',' } else { '\t' });
    let null = quoted("NULL").unwrap_or_else(|| if csv { String::new() } else { "\\N".into() });
    Some(CopySpec {
        target,
        is_query,
        columns,
        from_stdin,
        csv,
        header,
        delimiter,
        null,
    })
}

/// Split one line of COPY text format (backslash escapes) into fields.
fn split_copy_text(line: &str, delim: char, null: &str) -> Vec<Option<String>> {
    line.split(delim)
        .map(|f| {
            if f == null {
                return None;
            }
            let mut out = String::with_capacity(f.len());
            let mut it = f.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('t') => out.push('\t'),
                        Some('n') => out.push('\n'),
                        Some('r') => out.push('\r'),
                        Some('\\') => out.push('\\'),
                        Some(o) => out.push(o),
                        None => {}
                    }
                } else {
                    out.push(c);
                }
            }
            Some(out)
        })
        .collect()
}

/// Parse CSV records (quotes, doubled quotes, embedded newlines).
fn split_csv(data: &str, delim: char, null: &str) -> Vec<Vec<Option<String>>> {
    // Without quotes every record is one line: split large inputs at line
    // boundaries and parse the pieces in parallel (same rows, same order).
    const PAR_MIN: usize = 1 << 20;
    if data.len() >= PAR_MIN && delim != '\n' && !data.as_bytes().contains(&b'"') {
        use rayon::prelude::*;
        let pieces = rayon::current_num_threads().max(1) * 4;
        let step = data.len() / pieces + 1;
        let mut cuts = vec![0usize];
        let mut at = 0;
        while at + step < data.len() {
            match data.as_bytes()[at + step..]
                .iter()
                .position(|&b| b == b'\n')
            {
                Some(p) => {
                    at = at + step + p + 1;
                    cuts.push(at);
                }
                None => break,
            }
        }
        cuts.push(data.len());
        cuts.dedup();
        return cuts
            .par_windows(2)
            .map(|w| split_csv_seq(&data[w[0]..w[1]], delim, null))
            .flatten_iter()
            .collect();
    }
    split_csv_seq(data, delim, null)
}

fn split_csv_seq(data: &str, delim: char, null: &str) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut row: Vec<Option<String>> = Vec::new();
    let mut field = String::new();
    let (mut in_q, mut was_q) = (false, false);
    let mut chars = data.chars().peekable();
    let finish = |field: &mut String, was_q: &mut bool, row: &mut Vec<Option<String>>| {
        let v = if !*was_q && field == null {
            None
        } else {
            Some(std::mem::take(field))
        };
        field.clear();
        *was_q = false;
        row.push(v);
    };
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_q = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' {
            in_q = true;
            was_q = true;
        } else if c == delim {
            finish(&mut field, &mut was_q, &mut row);
        } else if c == '\n' || c == '\r' {
            if c == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            finish(&mut field, &mut was_q, &mut row);
            if !(row.len() == 1 && row[0].as_deref() == Some("")) || was_q {
                rows.push(std::mem::take(&mut row));
            }
            row.clear();
        } else {
            field.push(c);
        }
    }
    if !field.is_empty() || !row.is_empty() || was_q {
        finish(&mut field, &mut was_q, &mut row);
        rows.push(row);
    }
    rows
}

fn sql_literal(v: &Option<String>) -> String {
    match v {
        None => "NULL".into(),
        Some(s) => {
            let t = s.trim();
            let numeric = !t.is_empty()
                && t.parse::<f64>().map(|f| f.is_finite()).unwrap_or(false)
                && !t.starts_with('+')
                && !(t.len() > 1 && t.starts_with('0') && !t.starts_with("0."));
            if numeric {
                t.to_string()
            } else {
                format!("'{}'", s.replace('\'', "''"))
            }
        }
    }
}

fn csv_field(s: &str, delim: char) -> String {
    if s.contains(delim) || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn copy_text_field(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

async fn read_copy_message<R: AsyncReadExt + Unpin>(
    reader: &mut R,
) -> Result<(u8, Vec<u8>), PgWireError> {
    let t = reader
        .read_u8()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    let len = reader
        .read_i32()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    let n = (len as usize).saturating_sub(4);
    if n > MAX_PGWIRE_MESSAGE_SIZE {
        return Err(PgWireError::Protocol(format!(
            "CopyData message too large ({n} bytes)"
        )));
    }
    let mut body = vec![0u8; n];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    Ok((t, body))
}

/// Rows per INSERT statement issued for COPY FROM.
const COPY_BATCH_ROWS: usize = 500;

async fn handle_copy<R, W>(
    spec: CopySpec,
    reader: &mut R,
    writer: &mut W,
    executor: &Arc<dyn QueryExecutor>,
    conn_state: &mut ConnectionState,
    timeout: Option<u64>,
) -> Result<(), PgWireError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    if spec.from_stdin {
        if spec.is_query {
            write_error_response(writer, "42601", "COPY (query) FROM STDIN is not valid").await?;
            return Ok(());
        }
        if let Err(e) = crate::query::check_write_permission(
            &conn_state.user_id,
            &conn_state.roles,
            &format!("INSERT INTO {} VALUES (1)", spec.target),
        ) {
            write_error_response(writer, &e.code, &e.message).await?;
            return Ok(());
        }
        // CopyInResponse: overall text format, per-column text
        let n = spec.columns.len() as i16;
        let mut b = vec![0u8];
        b.extend_from_slice(&n.to_be_bytes());
        for _ in 0..n {
            b.extend_from_slice(&0i16.to_be_bytes());
        }
        write_message(writer, b'G', &b).await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        let mut data = Vec::new();
        loop {
            let (t, body) = read_copy_message(reader).await?;
            match t {
                b'd' => data.extend_from_slice(&body),
                b'c' => break,
                b'f' => {
                    let msg = String::from_utf8_lossy(&body)
                        .trim_end_matches('\0')
                        .to_string();
                    conn_state.set_error_in_transaction();
                    write_error_response(
                        writer,
                        "57014",
                        &format!("COPY from stdin failed: {msg}"),
                    )
                    .await?;
                    return Ok(());
                }
                b'H' | b'S' => {} // Flush/Sync may arrive mid-copy in extended mode
                other => {
                    return Err(PgWireError::Protocol(format!(
                        "unexpected message '{}' during COPY",
                        other as char
                    )));
                }
            }
        }
        let text = String::from_utf8_lossy(&data);
        let mut rows: Vec<Vec<Option<String>>> = if spec.csv {
            split_csv(&text, spec.delimiter, &spec.null)
        } else {
            text.lines()
                .filter(|l| *l != "\\.")
                .map(|l| split_copy_text(l, spec.delimiter, &spec.null))
                .collect()
        };
        if spec.header && !rows.is_empty() {
            rows.remove(0);
        }
        let cols = if spec.columns.is_empty() {
            String::new()
        } else {
            format!(" ({})", spec.columns.join(", "))
        };
        if conn_state.session_id.is_none() {
            let t0 = std::time::Instant::now();
            if let Some(r) = executor.bulk_copy(&spec.target, &spec.columns, &rows) {
                match r {
                    Ok(n) => {
                        if std::env::var_os("JOULEDB_INGEST_TRACE").is_some() {
                            eprintln!("copy direct rows={n} {:?}", t0.elapsed());
                        }
                        write_command_complete(writer, &format!("COPY {n}")).await?;
                    }
                    Err(e) => {
                        conn_state.set_error_in_transaction();
                        write_error_response(writer, &e.code, &e.message).await?;
                    }
                }
                return Ok(());
            }
        }
        let mut total = 0usize;
        let trace = std::env::var_os("JOULEDB_INGEST_TRACE").is_some();
        let (mut t_fmt, mut t_run) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for chunk in rows.chunks(COPY_BATCH_ROWS) {
            let t0 = std::time::Instant::now();
            let values: Vec<String> = chunk
                .iter()
                .map(|r| {
                    format!(
                        "({})",
                        r.iter().map(sql_literal).collect::<Vec<_>>().join(", ")
                    )
                })
                .collect();
            let sql = format!(
                "INSERT INTO {}{} VALUES {}",
                spec.target,
                cols,
                values.join(", ")
            );
            let t1 = std::time::Instant::now();
            t_fmt += t1 - t0;
            let r = run_statement(&sql, executor, conn_state, timeout);
            t_run += t1.elapsed();
            if let Err(e) = r {
                conn_state.set_error_in_transaction();
                write_error_response(
                    writer,
                    &e.code,
                    &format!("COPY row {}: {}", total + 1, e.message),
                )
                .await?;
                return Ok(());
            }
            total += chunk.len();
        }
        if trace {
            eprintln!("copy rows={total} format={t_fmt:?} insert={t_run:?}");
        }
        write_command_complete(writer, &format!("COPY {total}")).await?;
        return Ok(());
    }
    // COPY ... TO STDOUT
    let query = if spec.is_query {
        spec.target.clone()
    } else if spec.columns.is_empty() {
        format!("SELECT * FROM {}", spec.target)
    } else {
        format!("SELECT {} FROM {}", spec.columns.join(", "), spec.target)
    };
    let resp = match run_statement(&query, executor, conn_state, timeout) {
        Ok(r) => r,
        Err(e) => {
            conn_state.set_error_in_transaction();
            write_error_response(writer, &e.code, &e.message).await?;
            return Ok(());
        }
    };
    let n = resp.columns.len() as i16;
    let mut b = vec![0u8];
    b.extend_from_slice(&n.to_be_bytes());
    for _ in 0..n {
        b.extend_from_slice(&0i16.to_be_bytes());
    }
    write_message(writer, b'H', &b).await?;
    let d = spec.delimiter.to_string();
    let line = |vals: Vec<String>| format!("{}\n", vals.join(&d));
    if spec.header {
        let h = resp.columns.iter().map(|c| {
            if spec.csv {
                csv_field(c, spec.delimiter)
            } else {
                copy_text_field(c)
            }
        });
        write_message(writer, b'd', line(h.collect()).as_bytes()).await?;
    }
    for r in &resp.rows {
        let vals = r
            .iter()
            .map(|v| match v {
                serde_json::Value::Null => spec.null.clone(),
                other => {
                    let t = text_value(other);
                    if spec.csv {
                        csv_field(&t, spec.delimiter)
                    } else {
                        copy_text_field(&t)
                    }
                }
            })
            .collect();
        write_message(writer, b'd', line(vals).as_bytes()).await?;
    }
    write_message(writer, b'c', &[]).await?;
    write_command_complete(writer, &format!("COPY {}", resp.rows.len())).await?;
    Ok(())
}

fn bound_sql(p: &Portal) -> String {
    if p.params.is_empty() {
        p.sql.clone()
    } else {
        substitute_params(&p.sql, &p.params)
    }
}

fn count_placeholders(sql: &str) -> usize {
    let b = sql.as_bytes();
    let mut max = 0usize;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'$' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if let Ok(n) = sql[i + 1..j].parse::<usize>() {
                max = max.max(n);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    max
}

/// Statements whose result has columns, so Describe should report them.
fn returns_rows(sql: &str) -> bool {
    let u = sql
        .trim_start()
        .trim_start_matches('(')
        .to_ascii_uppercase();
    let w = u.split_whitespace().next().unwrap_or("");
    matches!(
        w,
        "SELECT" | "WITH" | "VALUES" | "SHOW" | "TABLE" | "EXPLAIN"
    ) || u.contains(" RETURNING ")
}

/// Decode one Bind parameter using its format code and declared type OID.
fn decode_param(raw: &[u8], fmt: i16, oid: i32) -> serde_json::Value {
    use serde_json::Value;
    if fmt == 1 {
        let int = |b: &[u8]| -> Option<i64> {
            Some(match b.len() {
                2 => i16::from_be_bytes(b.try_into().ok()?) as i64,
                4 => i32::from_be_bytes(b.try_into().ok()?) as i64,
                8 => i64::from_be_bytes(b.try_into().ok()?),
                _ => return None,
            })
        };
        let v = match oid {
            16 => raw.first().map(|b| Value::Bool(*b != 0)),
            20 | 21 | 23 => int(raw).map(Value::from),
            700 => raw
                .try_into()
                .ok()
                .map(|a| Value::from(f32::from_be_bytes(a) as f64)),
            701 => raw
                .try_into()
                .ok()
                .map(|a| Value::from(f64::from_be_bytes(a))),
            _ => None,
        };
        return v.unwrap_or_else(|| Value::String(String::from_utf8_lossy(raw).into_owned()));
    }
    let text = String::from_utf8_lossy(raw).into_owned();
    match oid {
        20 | 21 | 23 | 26 => text
            .trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or(Value::String(text)),
        700 | 701 | 1700 => match text.trim().parse::<f64>() {
            Ok(f) if f.is_finite() => Value::from(f),
            _ => Value::String(text),
        },
        16 => match text.trim() {
            "t" | "true" | "TRUE" | "1" | "on" | "yes" => Value::Bool(true),
            "f" | "false" | "FALSE" | "0" | "off" | "no" => Value::Bool(false),
            _ => Value::String(text),
        },
        _ => Value::String(text),
    }
}

fn result_format(formats: &[i16], i: usize) -> i16 {
    match formats.len() {
        0 => 0,
        1 => formats[0],
        _ => formats.get(i).copied().unwrap_or(0),
    }
}

/// One type OID per column, consistent across every row (binary encoding
/// needs that). Integers widen to int8, mixed numbers to float8; a binary
/// column that is neither number nor bool is sent as text.
fn column_oids(rows: &[Vec<serde_json::Value>], ncols: usize, formats: &[i16]) -> Vec<i32> {
    (0..ncols)
        .map(|c| {
            let (mut ints, mut nums, mut bools, mut other, mut wide) = (0, 0, 0, 0, false);
            let mut first_other: Option<&serde_json::Value> = None;
            for r in rows {
                match r.get(c) {
                    None | Some(serde_json::Value::Null) => {}
                    Some(serde_json::Value::Bool(_)) => bools += 1,
                    Some(serde_json::Value::Number(n)) => {
                        if let Some(i) = n.as_i64() {
                            ints += 1;
                            wide |= i32::try_from(i).is_err();
                        } else {
                            nums += 1;
                        }
                    }
                    Some(v) => {
                        other += 1;
                        first_other.get_or_insert(v);
                    }
                }
            }
            let binary = result_format(formats, c) == 1;
            if other == 0 && bools == 0 && nums == 0 && ints > 0 {
                if wide { 20 } else { 23 }
            } else if other == 0 && bools == 0 && nums > 0 {
                701
            } else if other == 0 && bools > 0 && ints + nums == 0 {
                16
            } else if other > 0 && ints + nums + bools == 0 && !binary {
                first_other.map(value_to_pg_oid).unwrap_or(25)
            } else {
                25
            }
        })
        .collect()
}

async fn write_row_description_fmt<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    columns: &[String],
    oids: &[i32],
    formats: &[i16],
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    body.extend_from_slice(&(columns.len() as i16).to_be_bytes());
    for (i, col) in columns.iter().enumerate() {
        body.extend_from_slice(col.as_bytes());
        body.push(0);
        body.extend_from_slice(&0i32.to_be_bytes());
        body.extend_from_slice(&0i16.to_be_bytes());
        let oid = oids.get(i).copied().unwrap_or(25);
        body.extend_from_slice(&oid.to_be_bytes());
        let size: i16 = match oid {
            16 => 1,
            23 => 4,
            20 | 701 => 8,
            _ => -1,
        };
        body.extend_from_slice(&size.to_be_bytes());
        body.extend_from_slice(&(-1i32).to_be_bytes());
        body.extend_from_slice(&result_format(formats, i).to_be_bytes());
    }
    write_message(writer, b'T', &body).await
}

fn text_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => if *b { "t" } else { "f" }.to_string(),
        other => other.to_string(),
    }
}

async fn write_data_row_fmt<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    row: &[serde_json::Value],
    oids: &[i32],
    formats: &[i16],
) -> Result<(), PgWireError> {
    if formats.iter().all(|f| *f == 0) {
        return write_data_row(writer, row).await;
    }
    let mut body = Vec::new();
    body.extend_from_slice(&(row.len() as i16).to_be_bytes());
    for (i, v) in row.iter().enumerate() {
        if v.is_null() {
            body.extend_from_slice(&(-1i32).to_be_bytes());
            continue;
        }
        let bytes: Vec<u8> = if result_format(formats, i) == 1 {
            match (oids.get(i).copied().unwrap_or(25), v) {
                (16, serde_json::Value::Bool(b)) => vec![*b as u8],
                (23, serde_json::Value::Number(n)) => {
                    (n.as_i64().unwrap_or(0) as i32).to_be_bytes().to_vec()
                }
                (20, serde_json::Value::Number(n)) => {
                    n.as_i64().unwrap_or(0).to_be_bytes().to_vec()
                }
                (701, serde_json::Value::Number(n)) => {
                    n.as_f64().unwrap_or(0.0).to_be_bytes().to_vec()
                }
                _ => text_value(v).into_bytes(),
            }
        } else {
            text_value(v).into_bytes()
        };
        body.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
        body.extend_from_slice(&bytes);
    }
    write_message(writer, b'D', &body).await
}

/// Handle a simple query (Q message) — backward-compatible wrapper
async fn handle_simple_query<W: AsyncWriteExt + Unpin>(
    sql: &str,
    executor: &Arc<dyn QueryExecutor>,
    writer: &mut W,
    stats: &PgWireStats,
) -> Result<(), PgWireError> {
    let mut conn_state = ConnectionState::new();
    execute_and_write(sql, executor, writer, stats, &mut conn_state, None).await
}

// ============================================================================
// Message Writing Helpers
// ============================================================================

async fn write_message<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    tag: u8,
    body: &[u8],
) -> Result<(), PgWireError> {
    writer
        .write_u8(tag)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    let length = (body.len() as i32) + 4;
    writer
        .write_all(&length.to_be_bytes())
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    if !body.is_empty() {
        writer
            .write_all(body)
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
    }
    Ok(())
}

async fn write_parameter_status<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    key: &str,
    value: &str,
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    body.extend_from_slice(key.as_bytes());
    body.push(0);
    body.extend_from_slice(value.as_bytes());
    body.push(0);
    write_message(writer, b'S', &body).await
}

async fn write_ready_for_query<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    status: u8,
) -> Result<(), PgWireError> {
    write_message(writer, b'Z', &[status]).await
}

async fn write_row_description<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    columns: &[String],
) -> Result<(), PgWireError> {
    write_row_description_typed(writer, columns, None).await
}

async fn write_row_description_typed<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    columns: &[String],
    type_oids: Option<&[i32]>,
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    // Number of fields
    body.extend_from_slice(&(columns.len() as i16).to_be_bytes());

    for (i, col) in columns.iter().enumerate() {
        // Field name (null-terminated)
        body.extend_from_slice(col.as_bytes());
        body.push(0);
        // Table OID (0 = not a table column)
        body.extend_from_slice(&0i32.to_be_bytes());
        // Column attribute number (0)
        body.extend_from_slice(&0i16.to_be_bytes());
        // Data type OID
        let oid = type_oids
            .and_then(|oids| oids.get(i).copied())
            .unwrap_or(25);
        body.extend_from_slice(&oid.to_be_bytes());
        // Data type size (-1 for variable length)
        body.extend_from_slice(&(-1i16).to_be_bytes());
        // Type modifier (-1)
        body.extend_from_slice(&(-1i32).to_be_bytes());
        // Format code (0 = text)
        body.extend_from_slice(&0i16.to_be_bytes());
    }

    write_message(writer, b'T', &body).await
}

async fn write_data_row<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    row: &[serde_json::Value],
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    // Number of columns
    body.extend_from_slice(&(row.len() as i16).to_be_bytes());

    for value in row {
        match value {
            serde_json::Value::Null => {
                // NULL: length = -1
                body.extend_from_slice(&(-1i32).to_be_bytes());
            }
            _ => {
                let text = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => {
                        if *b {
                            "t".to_string()
                        } else {
                            "f".to_string()
                        }
                    }
                    other => other.to_string(),
                };
                let bytes = text.as_bytes();
                body.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
                body.extend_from_slice(bytes);
            }
        }
    }

    write_message(writer, b'D', &body).await
}

async fn write_command_complete<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    tag: &str,
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    body.extend_from_slice(tag.as_bytes());
    body.push(0);
    write_message(writer, b'C', &body).await
}

async fn write_empty_query_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
) -> Result<(), PgWireError> {
    write_message(writer, b'I', &[]).await
}

async fn write_error_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    code: &str,
    message: &str,
) -> Result<(), PgWireError> {
    let mut body = Vec::new();
    // Severity (localized)
    body.push(b'S');
    body.extend_from_slice(b"ERROR");
    body.push(0);
    // Severity (non-localized, required by protocol v3)
    body.push(b'V');
    body.extend_from_slice(b"ERROR");
    body.push(0);
    // SQLSTATE code
    body.push(b'C');
    let sqlstate = match code {
        "SYNTAX_ERROR" => "42601",
        "TABLE_NOT_FOUND" | "RELATION_NOT_FOUND" => "42P01",
        "COLUMN_NOT_FOUND" | "UNDEFINED_COLUMN" => "42703",
        "DUPLICATE_TABLE" | "TABLE_ALREADY_EXISTS" => "42P07",
        "UNIQUE_VIOLATION" | "DUPLICATE_KEY" => "23505",
        "NOT_NULL_VIOLATION" => "23502",
        "CHECK_VIOLATION" | "CHECK_CONSTRAINT" => "23514",
        "FOREIGN_KEY_VIOLATION" | "FK_VIOLATION" => "23503",
        "DIVISION_BY_ZERO" => "22012",
        "INVALID_TEXT_REPRESENTATION" => "22P02",
        "PERMISSION_DENIED" | "INSUFFICIENT_PRIVILEGE" => "42501",
        "UNDEFINED_FUNCTION" => "42883",
        "TRANSACTION_ERROR" | "IN_FAILED_SQL_TRANSACTION" => "25P02",
        "DATA_EXCEPTION" => "22000",
        "EXECUTION_ERROR" => "P0001", // PL/pgSQL raise_exception
        // If the caller passes a 5-char SQLSTATE code directly, use it as-is
        c if c.len() == 5 => c,
        _ => "XX000",
    };
    body.extend_from_slice(sqlstate.as_bytes());
    body.push(0);
    // Message
    body.push(b'M');
    body.extend_from_slice(message.as_bytes());
    body.push(0);
    // Terminator
    body.push(0);

    write_message(writer, b'E', &body).await
}

// ============================================================================
// SCRAM-SHA-256 Authentication
// ============================================================================

/// Perform SCRAM-SHA-256 authentication over PgWire.
///
/// Protocol flow:
///   Server → AuthenticationSASL (type 10) with mechanism list
///   Client → SASLInitialResponse ('p' message)
///   Server → AuthenticationSASLContinue (type 11)
///   Client → SASLResponse ('p' message)
///   Server → AuthenticationSASLFinal (type 12)
///   Server → AuthenticationOk (type 0)
async fn do_scram_auth<R, W>(
    reader: &mut R,
    writer: &mut W,
    credentials: &crate::scram::ScramCredentials,
) -> Result<(), PgWireError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // Step 1: Send AuthenticationSASL (type 10) with SCRAM-SHA-256 mechanism
    let mechanism = b"SCRAM-SHA-256\0";
    let msg_len = 4 + 4 + mechanism.len() + 1; // length + type + mechanism + final \0
    let mut auth_sasl = Vec::with_capacity(msg_len);
    auth_sasl.extend_from_slice(&(msg_len as i32).to_be_bytes());
    auth_sasl.extend_from_slice(&10i32.to_be_bytes()); // auth type 10 = SASL
    auth_sasl.extend_from_slice(mechanism);
    auth_sasl.push(0); // end of mechanism list
    writer
        .write_u8(b'R')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&auth_sasl)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .flush()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Step 2: Read SASLInitialResponse ('p' message)
    let msg_type = reader
        .read_u8()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    if msg_type != b'p' {
        write_error_response(writer, "28P01", "Expected SASL initial response").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth("Expected SASL initial response".into()));
    }
    let msg_len = reader
        .read_i32()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))? as usize;
    if msg_len < 4 || msg_len > 4096 {
        write_error_response(writer, "28P01", "Invalid SASL message length").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth("Invalid SASL message length".into()));
    }
    let mut body = vec![0u8; msg_len - 4];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Parse: mechanism_name\0 + client_first_length(4) + client_first_data
    let null_pos = body
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| PgWireError::Auth("malformed SASL initial response".into()))?;
    let _mechanism_name = &body[..null_pos]; // Should be "SCRAM-SHA-256"
    let rest = &body[null_pos + 1..];

    // Per PostgreSQL protocol: response_length(int32, -1 = none) + response_data
    if rest.len() < 4 {
        write_error_response(writer, "28P01", "SASL initial response too short").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth("SASL initial response too short".into()));
    }
    let client_first_len_raw = i32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let client_first_data = if client_first_len_raw == -1 || client_first_len_raw == 0 {
        // -1 = no initial response, 0 = empty response
        &rest[4..4]
    } else if client_first_len_raw > 0 {
        let len = client_first_len_raw as usize;
        if rest.len() >= 4 + len {
            &rest[4..4 + len]
        } else {
            // Length exceeds available data — use remainder (client may omit length prefix)
            &rest[4..]
        }
    } else {
        // Other negative values are protocol errors
        write_error_response(writer, "28P01", "Invalid SASL initial response length").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth(
            "Invalid SASL initial response length".into(),
        ));
    };

    let client_first = String::from_utf8_lossy(client_first_data).to_string();

    // Step 3: Process client-first, send AuthenticationSASLContinue (type 11)
    let mut scram_server = crate::scram::ScramServer::new(credentials);
    let server_first = scram_server
        .handle_client_first(&client_first)
        .map_err(|e| PgWireError::Auth(format!("SCRAM client-first: {}", e)))?;

    let server_first_bytes = server_first.as_bytes();
    let cont_len = 4 + 4 + server_first_bytes.len();
    let mut auth_cont = Vec::with_capacity(cont_len);
    auth_cont.extend_from_slice(&(cont_len as i32).to_be_bytes());
    auth_cont.extend_from_slice(&11i32.to_be_bytes()); // type 11 = SASLContinue
    auth_cont.extend_from_slice(server_first_bytes);
    writer
        .write_u8(b'R')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&auth_cont)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .flush()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    // Step 4: Read SASLResponse ('p' message)
    let msg_type = reader
        .read_u8()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    if msg_type != b'p' {
        write_error_response(writer, "28P01", "Expected SASL response").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth("Expected SASL response".into()));
    }
    let msg_len = reader
        .read_i32()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))? as usize;
    if msg_len < 4 || msg_len > 4096 {
        write_error_response(writer, "28P01", "Invalid SASL response length").await?;
        writer
            .flush()
            .await
            .map_err(|e| PgWireError::Io(e.to_string()))?;
        return Err(PgWireError::Auth("Invalid SASL response length".into()));
    }
    let mut client_final_body = vec![0u8; msg_len - 4];
    reader
        .read_exact(&mut client_final_body)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    let client_final = String::from_utf8_lossy(&client_final_body).to_string();

    // Step 5: Verify proof, send AuthenticationSASLFinal (type 12)
    let server_final = match scram_server.handle_client_final(&client_final) {
        Ok(sf) => sf,
        Err(_) => {
            write_error_response(writer, "28P01", "password authentication failed").await?;
            writer
                .flush()
                .await
                .map_err(|e| PgWireError::Io(e.to_string()))?;
            return Err(PgWireError::Auth("SCRAM authentication failed".into()));
        }
    };

    let server_final_bytes = server_final.as_bytes();
    let final_len = 4 + 4 + server_final_bytes.len();
    let mut auth_final = Vec::with_capacity(final_len);
    auth_final.extend_from_slice(&(final_len as i32).to_be_bytes());
    auth_final.extend_from_slice(&12i32.to_be_bytes()); // type 12 = SASLFinal
    auth_final.extend_from_slice(server_final_bytes);
    writer
        .write_u8(b'R')
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .write_all(&auth_final)
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;
    writer
        .flush()
        .await
        .map_err(|e| PgWireError::Io(e.to_string()))?;

    Ok(())
}

/// Try to look up SCRAM credentials for a user from the RBAC manager.
fn get_user_scram_credentials(
    rbac: &crate::rbac::RBACManager,
    username: &str,
) -> Option<crate::scram::ScramCredentials> {
    let user = rbac.get_user(username).ok()?;
    let password_hash = user.metadata.get("password_hash")?;
    crate::scram::credentials_from_json(password_hash)
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone)]
pub enum PgWireError {
    Io(String),
    Protocol(String),
    Query(String),
    Auth(String),
}

impl std::fmt::Display for PgWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Protocol(e) => write!(f, "Protocol error: {}", e),
            Self::Query(e) => write!(f, "Query error: {}", e),
            Self::Auth(e) => write!(f, "Authentication error: {}", e),
        }
    }
}

impl std::error::Error for PgWireError {}

// ============================================================================
// Prepared Statement Cache
// ============================================================================

/// A cached prepared statement
#[derive(Debug, Clone)]
pub struct CachedStatement {
    pub name: String,
    pub sql: String,
    pub param_count: usize,
    pub created_at: Instant,
}

/// LRU-based prepared statement cache
pub struct PreparedStatementCache {
    statements: HashMap<String, CachedStatement>,
    max_size: usize,
    access_order: VecDeque<String>,
}

impl PreparedStatementCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            statements: HashMap::new(),
            max_size: max_size.max(1),
            access_order: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, name: String, sql: String, param_count: usize) {
        // If key already exists, remove from access_order first
        if self.statements.contains_key(&name) {
            self.access_order.retain(|k| k != &name);
        } else {
            // Evict oldest if at capacity
            while self.statements.len() >= self.max_size {
                if let Some(oldest) = self.access_order.pop_front() {
                    self.statements.remove(&oldest);
                } else {
                    break;
                }
            }
        }

        self.statements.insert(
            name.clone(),
            CachedStatement {
                name: name.clone(),
                sql,
                param_count,
                created_at: Instant::now(),
            },
        );
        self.access_order.push_back(name);
    }

    pub fn get(&mut self, name: &str) -> Option<&CachedStatement> {
        if self.statements.contains_key(name) {
            // Move to back of access_order (most recently used)
            self.access_order.retain(|k| k != name);
            self.access_order.push_back(name.to_string());
            self.statements.get(name)
        } else {
            None
        }
    }

    pub fn remove(&mut self, name: &str) -> bool {
        if self.statements.remove(name).is_some() {
            self.access_order.retain(|k| k != name);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.statements.len()
    }
}

// ============================================================================
// Connection State (Extended Query Protocol)
// ============================================================================

/// Per-connection state for the extended query protocol
struct ConnectionState {
    prepared_statements: PreparedStatementCache,
    portals: HashMap<String, Portal>,
    transaction_status: u8, // b'I' = idle, b'T' = in transaction, b'E' = error
    /// Whether the client issued an explicit BEGIN (tracks T→E transitions)
    in_explicit_transaction: bool,
    /// User identity from startup params (for RBAC)
    user_id: String,
    /// User roles for write permission checks
    roles: Vec<String>,
    /// MVCC session from BEGIN, sent with every statement until COMMIT/ROLLBACK
    session_id: Option<String>,
    /// Parameter type OIDs declared in Parse, by statement name
    param_oids: HashMap<String, Vec<i32>>,
}

/// A prepared statement parsed from a Parse message
#[derive(Debug, Clone)]
struct PreparedStatementEntry {
    sql: String,
    param_count: usize,
}

/// A portal created by a Bind message
#[derive(Debug, Clone)]
struct Portal {
    statement_name: String,
    sql: String,
    params: Vec<serde_json::Value>,
    /// Result-column format codes from Bind (0 text, 1 binary)
    result_formats: Vec<i16>,
    /// Result computed by Describe so Execute does not run it twice
    described: Option<Result<QueryResponse, QueryErrorResponse>>,
}

/// Substitute `$1`, `$2`, ... placeholders with bound parameter values.
///
/// Parameters are SQL-escaped: strings are single-quoted with internal quotes
/// doubled, NULLs become the literal `NULL`, and numbers/booleans pass through.
/// This is safe because the values come from the Bind message (already parsed),
/// not from concatenated user text.
fn substitute_params(sql: &str, params: &[serde_json::Value]) -> String {
    if params.is_empty() {
        return sql.to_string();
    }

    let mut result = String::with_capacity(sql.len() + params.len() * 8);
    let bytes = sql.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Check for $N placeholder (not inside a string literal)
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            // Parse the parameter number
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if let Ok(idx) = sql[start..end].parse::<usize>() {
                if idx >= 1 && idx <= params.len() {
                    let val = &params[idx - 1];
                    match val {
                        serde_json::Value::Null => result.push_str("NULL"),
                        serde_json::Value::Bool(b) => {
                            result.push_str(if *b { "TRUE" } else { "FALSE" });
                        }
                        serde_json::Value::Number(n) => {
                            result.push_str(&n.to_string());
                        }
                        serde_json::Value::String(s) => {
                            // SQL-escape: single-quote with doubled internal quotes
                            result.push('\'');
                            for ch in s.chars() {
                                if ch == '\'' {
                                    result.push_str("''");
                                } else {
                                    result.push(ch);
                                }
                            }
                            result.push('\'');
                        }
                        _ => {
                            // Arrays/objects: serialize as JSON string
                            let json_str = val.to_string();
                            result.push('\'');
                            for ch in json_str.chars() {
                                if ch == '\'' {
                                    result.push_str("''");
                                } else {
                                    result.push(ch);
                                }
                            }
                            result.push('\'');
                        }
                    }
                    i = end;
                    continue;
                }
            }
        }

        // Skip over dollar-quoted strings ($tag$...$tag$ or $$...$$)
        if bytes[i] == b'$' && (i + 1 >= bytes.len() || !bytes[i + 1].is_ascii_digit()) {
            // Try to parse a dollar-quote tag: $[a-zA-Z0-9_]*$
            let tag_start = i;
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'$' {
                // Found opening tag like $$ or $tag$
                let tag = &sql[tag_start..=j];
                // Push the opening tag
                result.push_str(tag);
                i = j + 1;
                // Find the closing tag
                while i < bytes.len() {
                    if bytes[i] == b'$' {
                        if sql[i..].starts_with(tag) {
                            result.push_str(tag);
                            i += tag.len();
                            break;
                        }
                    }
                    result.push(bytes[i] as char);
                    i += 1;
                }
                continue;
            }
        }

        // Skip over string literals to avoid replacing $N inside them
        if bytes[i] == b'\'' {
            result.push('\'');
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\'' {
                    result.push('\'');
                    i += 1;
                    // Doubled quote = escaped quote, continue
                    if i < bytes.len() && bytes[i] == b'\'' {
                        result.push('\'');
                        i += 1;
                        continue;
                    }
                    break;
                }
                result.push(bytes[i] as char);
                i += 1;
            }
            continue;
        }

        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

impl ConnectionState {
    fn new() -> Self {
        Self {
            prepared_statements: PreparedStatementCache::new(256),
            portals: HashMap::new(),
            transaction_status: b'I',
            in_explicit_transaction: false,
            session_id: None,
            param_oids: HashMap::new(),
            user_id: "anonymous".to_string(),
            roles: vec!["superuser".to_string()],
        }
    }

    fn with_user(user_id: String, auth_enabled: bool) -> Self {
        Self {
            prepared_statements: PreparedStatementCache::new(256),
            portals: HashMap::new(),
            transaction_status: b'I',
            in_explicit_transaction: false,
            session_id: None,
            param_oids: HashMap::new(),
            user_id,
            // When auth is enabled, default to readonly; when disabled, grant superuser
            roles: if auth_enabled {
                vec!["writer".to_string()]
            } else {
                vec!["superuser".to_string()]
            },
        }
    }

    fn update_transaction_status(&mut self, sql: &str) {
        let upper = sql.trim().to_uppercase();
        if upper.starts_with("BEGIN") || upper.starts_with("START TRANSACTION") {
            self.transaction_status = b'T';
            self.in_explicit_transaction = true;
        } else if upper.starts_with("COMMIT") || upper.starts_with("END") {
            self.transaction_status = b'I';
            self.in_explicit_transaction = false;
        } else if upper.starts_with("ROLLBACK") {
            self.transaction_status = b'I';
            self.in_explicit_transaction = false;
        }
    }

    fn set_error_in_transaction(&mut self) {
        if self.transaction_status == b'T' {
            self.transaction_status = b'E';
        }
    }
}

// ============================================================================
// Type OID Mapping
// ============================================================================

/// Map a serde_json::Value to a PostgreSQL type OID
pub fn value_to_pg_oid(value: &serde_json::Value) -> i32 {
    match value {
        serde_json::Value::Bool(_) => 16, // bool
        serde_json::Value::Number(n) => {
            // Detect floats by checking for fractional part or non-integer representation.
            // serde_json's is_i64()/is_f64() overlap for whole-number floats like 3.0.
            if let Some(f) = n.as_f64() {
                if f.fract() != 0.0 || (!n.is_i64() && !n.is_u64()) {
                    701 // float8
                } else {
                    23 // int4
                }
            } else {
                23 // int4
            }
        }
        serde_json::Value::String(s) => {
            // Try to detect common types from string format
            if s.len() == 36 && s.chars().filter(|c| *c == '-').count() == 4 {
                2950 // uuid
            } else if s.starts_with('{') || s.starts_with('[') {
                3802 // jsonb
            } else {
                25 // text
            }
        }
        serde_json::Value::Null => 25,       // text (default for NULL)
        serde_json::Value::Array(_) => 3802, // jsonb for arrays
        serde_json::Value::Object(_) => 3802, // jsonb for objects
    }
}

/// Map a SQL type name to a PostgreSQL type OID (for RowDescription)
pub fn type_name_to_pg_oid(name: &str) -> i32 {
    match name.to_lowercase().as_str() {
        "bool" | "boolean" => 16,
        "bytea" | "blob" | "binary" => 17,
        "char" => 18,
        "int8" | "bigint" => 20,
        "int2" | "smallint" => 21,
        "int4" | "integer" | "int" => 23,
        "text" | "string" => 25,
        "oid" => 26,
        "json" => 114,
        "float4" | "real" => 700,
        "float8" | "double" | "double precision" | "float" => 701,
        "varchar" | "character varying" => 1043,
        "date" => 1082,
        "time" => 1083,
        "timestamp" | "datetime" => 1114,
        "timestamptz" | "timestamp with time zone" => 1184,
        "numeric" | "decimal" => 1700,
        "uuid" => 2950,
        "jsonb" => 3802,
        "vector" => 16385,
        _ => 25, // default to text
    }
}

/// Extract column names from a SELECT SQL statement for Describe responses.
/// Returns None for non-SELECT or unparseable SQL.
fn extract_select_columns(sql: &str) -> Option<Vec<String>> {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();
    if !upper.starts_with("SELECT") {
        return None;
    }
    // Find text between SELECT and FROM (or end of string)
    let after_select = &trimmed[6..]; // skip "SELECT"
    let from_pos = upper[6..].find(" FROM ").unwrap_or(after_select.len());
    let select_list = after_select[..from_pos].trim();

    if select_list == "*" {
        return None; // can't determine columns without schema
    }

    // Split by comma, extract column names/aliases
    let mut cols = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let bytes = select_list.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b',' if depth == 0 => {
                cols.push(extract_column_alias(&select_list[start..i]));
                start = i + 1;
            }
            _ => {}
        }
    }
    cols.push(extract_column_alias(&select_list[start..]));

    if cols.is_empty() { None } else { Some(cols) }
}

fn extract_column_alias(expr: &str) -> String {
    let trimmed = expr.trim();
    // Check for "expr AS alias" pattern (case-insensitive)
    let upper = trimmed.to_uppercase();
    if let Some(pos) = upper.rfind(" AS ") {
        return trimmed[pos + 4..].trim().trim_matches('"').to_string();
    }
    // Check for "table.column" — use the column part
    if let Some(pos) = trimmed.rfind('.') {
        return trimmed[pos + 1..].trim().trim_matches('"').to_string();
    }
    // Use the expression itself as the column name
    trimmed.trim_matches('"').to_string()
}

/// Extract null-terminated string from buffer starting at offset
fn extract_cstring(body: &[u8], offset: usize) -> (String, usize) {
    if offset >= body.len() {
        return (String::new(), offset);
    }
    let mut end = offset;
    while end < body.len() && body[end] != 0 {
        end += 1;
    }
    let s = String::from_utf8_lossy(&body[offset..end]).to_string();
    if end < body.len() {
        (s, end + 1) // +1 to skip the null terminator
    } else {
        // No null terminator found — return end of buffer
        (s, body.len())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgwire_config_default() {
        let config = PgWireConfig::default();
        assert_eq!(config.bind_addr, "0.0.0.0:5433");
        assert_eq!(config.max_connections, 256);
        assert!(config.server_version.contains("JouleDB"));
    }

    #[test]
    fn test_pgwire_stats() {
        let stats = PgWireStats::default();
        stats.connections_accepted.fetch_add(5, Ordering::Relaxed);
        stats.queries_executed.fetch_add(10, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.connections_accepted, 5);
        assert_eq!(snap.queries_executed, 10);
    }

    #[tokio::test]
    async fn test_write_message() {
        let mut buf = Vec::new();
        write_message(&mut buf, b'Z', &[b'I']).await.unwrap();
        // tag + 4-byte length + body
        assert_eq!(buf[0], b'Z');
        assert_eq!(&buf[1..5], &5i32.to_be_bytes()); // length = 4 + 1
        assert_eq!(buf[5], b'I');
    }

    #[tokio::test]
    async fn test_write_row_description() {
        let mut buf = Vec::new();
        let cols = vec!["id".to_string(), "name".to_string()];
        write_row_description(&mut buf, &cols).await.unwrap();
        assert_eq!(buf[0], b'T');
        // After tag and 4-byte length, first 2 bytes are field count
        let field_count = i16::from_be_bytes([buf[5], buf[6]]);
        assert_eq!(field_count, 2);
    }

    #[tokio::test]
    async fn test_write_data_row() {
        let mut buf = Vec::new();
        let row = vec![
            serde_json::Value::Number(serde_json::Number::from(42)),
            serde_json::Value::String("hello".to_string()),
            serde_json::Value::Null,
        ];
        write_data_row(&mut buf, &row).await.unwrap();
        assert_eq!(buf[0], b'D');
        let col_count = i16::from_be_bytes([buf[5], buf[6]]);
        assert_eq!(col_count, 3);
    }

    #[tokio::test]
    async fn test_write_error_response() {
        let mut buf = Vec::new();
        write_error_response(&mut buf, "SYNTAX_ERROR", "bad query")
            .await
            .unwrap();
        assert_eq!(buf[0], b'E');
        // Body should contain 'S', 'C', 'M' fields
        let body = &buf[5..];
        assert!(body.contains(&b'S'));
        assert!(body.contains(&b'M'));
    }

    #[tokio::test]
    async fn test_write_command_complete() {
        let mut buf = Vec::new();
        write_command_complete(&mut buf, "SELECT 3").await.unwrap();
        assert_eq!(buf[0], b'C');
        let body_str = String::from_utf8_lossy(&buf[5..]);
        assert!(body_str.contains("SELECT 3"));
    }

    struct MockExecutor;
    impl QueryExecutor for MockExecutor {
        fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
            Ok(QueryResponse {
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![vec![serde_json::json!(1), serde_json::json!("alice")]],
                affected_rows: None,
                execution_time_ms: 1,
                truncated: false,
                warnings: vec![],
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: None,
                session_id: None,
                viz_hint: None,
                energy_receipt: None,
            })
        }
    }

    #[tokio::test]
    async fn test_handle_simple_query() {
        let executor: Arc<dyn QueryExecutor> = Arc::new(MockExecutor);
        let stats = PgWireStats::default();
        let mut buf = Vec::new();

        handle_simple_query("SELECT 1", &executor, &mut buf, &stats)
            .await
            .unwrap();

        // Should contain RowDescription ('T'), DataRow ('D'), CommandComplete ('C')
        let tags: Vec<u8> = buf
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                if *i == 0 {
                    return true;
                }
                // Find message boundaries by scanning
                false
            })
            .map(|(_, b)| *b)
            .collect();

        assert_eq!(buf[0], b'T'); // First message is RowDescription
        assert_eq!(stats.queries_executed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_pgwire_server_creation() {
        let config = PgWireConfig::default();
        let executor = Arc::new(MockExecutor);
        let server = PgWireServer::new(config, executor);
        assert_eq!(server.stats().snapshot().connections_accepted, 0);
    }

    // ── Prepared Statement Cache tests (Group 4) ───────────────────────

    #[test]
    fn test_stmt_cache_insert_get() {
        let mut cache = PreparedStatementCache::new(10);
        cache.insert("s1".into(), "SELECT 1".into(), 0);
        let stmt = cache.get("s1");
        assert!(stmt.is_some());
        assert_eq!(stmt.unwrap().sql, "SELECT 1");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_stmt_cache_eviction() {
        let mut cache = PreparedStatementCache::new(2);
        cache.insert("s1".into(), "SELECT 1".into(), 0);
        cache.insert("s2".into(), "SELECT 2".into(), 0);
        cache.insert("s3".into(), "SELECT 3".into(), 0);
        // s1 should be evicted (oldest)
        assert!(cache.get("s1").is_none());
        assert!(cache.get("s2").is_some());
        assert!(cache.get("s3").is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_stmt_cache_lru_update() {
        let mut cache = PreparedStatementCache::new(2);
        cache.insert("s1".into(), "SELECT 1".into(), 0);
        cache.insert("s2".into(), "SELECT 2".into(), 0);
        // Access s1 to make it most recently used
        cache.get("s1");
        // Insert s3 — should evict s2 (now least recently used)
        cache.insert("s3".into(), "SELECT 3".into(), 0);
        assert!(cache.get("s1").is_some());
        assert!(cache.get("s2").is_none());
        assert!(cache.get("s3").is_some());
    }

    #[test]
    fn test_stmt_cache_remove() {
        let mut cache = PreparedStatementCache::new(10);
        cache.insert("s1".into(), "SELECT 1".into(), 0);
        assert!(cache.remove("s1"));
        assert_eq!(cache.len(), 0);
        assert!(!cache.remove("s1")); // already removed
    }

    #[test]
    fn test_stmt_cache_overwrite() {
        let mut cache = PreparedStatementCache::new(10);
        cache.insert("s1".into(), "SELECT 1".into(), 0);
        cache.insert("s1".into(), "SELECT 2".into(), 1);
        assert_eq!(cache.len(), 1);
        let stmt = cache.get("s1").unwrap();
        assert_eq!(stmt.sql, "SELECT 2");
        assert_eq!(stmt.param_count, 1);
    }

    // ── PgWire Extended Protocol tests (Group 1) ───────────────────────

    #[test]
    fn test_pgwire_server_config() {
        let config = crate::ServerConfig::default();
        assert!(config.enable_pgwire);
        assert_eq!(config.pgwire_addr, "127.0.0.1:5433");
    }

    #[test]
    fn test_pgwire_prepared_statement_cache() {
        let mut state = ConnectionState::new();
        state
            .prepared_statements
            .insert("stmt1".into(), "SELECT id FROM users".into(), 0);
        state
            .prepared_statements
            .insert("stmt2".into(), "INSERT INTO t VALUES ($1)".into(), 1);

        let s = state.prepared_statements.get("stmt1").unwrap();
        assert_eq!(s.sql, "SELECT id FROM users");

        // Overwrite
        state
            .prepared_statements
            .insert("stmt1".into(), "SELECT * FROM users".into(), 0);
        let s = state.prepared_statements.get("stmt1").unwrap();
        assert_eq!(s.sql, "SELECT * FROM users");
    }

    #[test]
    fn test_pgwire_portal_binding() {
        let mut state = ConnectionState::new();
        state
            .prepared_statements
            .insert("s1".into(), "SELECT * FROM t WHERE id = $1".into(), 1);

        // Create a portal
        let stmt = state.prepared_statements.get("s1").unwrap();
        let sql = stmt.sql.clone();
        state.portals.insert(
            "p1".into(),
            Portal {
                statement_name: "s1".into(),
                sql,
                params: vec![serde_json::json!("42")],
                result_formats: Vec::new(),
                described: None,
            },
        );

        let portal = state.portals.get("p1").unwrap();
        assert_eq!(portal.statement_name, "s1");
        assert_eq!(portal.params.len(), 1);
        assert_eq!(portal.sql, "SELECT * FROM t WHERE id = $1");
    }

    #[test]
    fn test_pgwire_type_oid_mapping() {
        assert_eq!(value_to_pg_oid(&serde_json::Value::Bool(true)), 16);
        assert_eq!(value_to_pg_oid(&serde_json::json!(42)), 23);
        assert_eq!(value_to_pg_oid(&serde_json::json!(3.14)), 701);
        assert_eq!(value_to_pg_oid(&serde_json::json!("hello")), 25);
        assert_eq!(value_to_pg_oid(&serde_json::Value::Null), 25);
    }

    #[test]
    fn test_pgwire_transaction_state_tracking() {
        let mut state = ConnectionState::new();
        assert_eq!(state.transaction_status, b'I');

        state.update_transaction_status("BEGIN");
        assert_eq!(state.transaction_status, b'T');

        state.update_transaction_status("SELECT 1");
        assert_eq!(state.transaction_status, b'T'); // stays in TX

        state.set_error_in_transaction();
        assert_eq!(state.transaction_status, b'E');

        // ROLLBACK clears error state
        state.update_transaction_status("ROLLBACK");
        assert_eq!(state.transaction_status, b'I');

        // Test COMMIT path
        state.update_transaction_status("BEGIN");
        assert_eq!(state.transaction_status, b'T');
        state.update_transaction_status("COMMIT");
        assert_eq!(state.transaction_status, b'I');
    }

    #[test]
    fn test_pgwire_parse_message_extraction() {
        // Construct a Parse message body: stmt_name\0 + sql\0 + param_count(2)
        let mut body = Vec::new();
        body.extend_from_slice(b"my_stmt\0");
        body.extend_from_slice(b"SELECT $1::int\0");
        body.extend_from_slice(&1i16.to_be_bytes()); // 1 parameter

        let (name, offset) = extract_cstring(&body, 0);
        assert_eq!(name, "my_stmt");
        let (sql, offset2) = extract_cstring(&body, offset);
        assert_eq!(sql, "SELECT $1::int");
        let param_count = i16::from_be_bytes([body[offset2], body[offset2 + 1]]);
        assert_eq!(param_count, 1);
    }

    #[test]
    fn test_pgwire_bind_message_extraction() {
        // Construct a Bind message body:
        // portal\0 + stmt\0 + num_format_codes(2) + num_params(2) + param_len(4) + param_data
        let mut body = Vec::new();
        body.extend_from_slice(b"\0"); // unnamed portal
        body.extend_from_slice(b"my_stmt\0"); // statement name
        body.extend_from_slice(&0i16.to_be_bytes()); // 0 format codes
        body.extend_from_slice(&1i16.to_be_bytes()); // 1 parameter
        let param = b"hello";
        body.extend_from_slice(&(param.len() as i32).to_be_bytes());
        body.extend_from_slice(param);

        let (portal, offset) = extract_cstring(&body, 0);
        assert_eq!(portal, "");
        let (stmt, mut offset) = extract_cstring(&body, offset);
        assert_eq!(stmt, "my_stmt");

        // Skip format codes
        let num_fc = i16::from_be_bytes([body[offset], body[offset + 1]]) as usize;
        offset += 2 + num_fc * 2;

        // Read params
        let num_params = i16::from_be_bytes([body[offset], body[offset + 1]]) as usize;
        offset += 2;
        assert_eq!(num_params, 1);

        let param_len = i32::from_be_bytes([
            body[offset],
            body[offset + 1],
            body[offset + 2],
            body[offset + 3],
        ]) as usize;
        offset += 4;
        let val = String::from_utf8_lossy(&body[offset..offset + param_len]).to_string();
        assert_eq!(val, "hello");
    }

    #[test]
    fn test_pgwire_execute_nonexistent_portal() {
        let state = ConnectionState::new();
        assert!(state.portals.get("nonexistent").is_none());
    }

    // --- Authentication Tests ---

    /// Helper: build a PgWire v3 startup message
    fn build_startup_message(user: &str, database: &str) -> Vec<u8> {
        let mut params = Vec::new();
        params.extend_from_slice(b"user\0");
        params.extend_from_slice(user.as_bytes());
        params.push(0);
        params.extend_from_slice(b"database\0");
        params.extend_from_slice(database.as_bytes());
        params.push(0);
        params.push(0); // params terminator

        let length = 4 + 4 + params.len(); // length + version + params
        let mut msg = Vec::new();
        msg.extend_from_slice(&(length as i32).to_be_bytes());
        msg.extend_from_slice(&PROTOCOL_VERSION_3.to_be_bytes());
        msg.extend_from_slice(&params);
        msg
    }

    /// Helper: build a PasswordMessage ('p' + length + password\0)
    fn build_password_message(password: &str) -> Vec<u8> {
        let pw_bytes = password.as_bytes();
        let length = 4 + pw_bytes.len() + 1; // length field + password + null
        let mut msg = Vec::new();
        msg.push(b'p');
        msg.extend_from_slice(&(length as i32).to_be_bytes());
        msg.extend_from_slice(pw_bytes);
        msg.push(0);
        msg
    }

    #[tokio::test]
    async fn test_pgwire_auth_disabled_allows_access() {
        let config = PgWireConfig {
            auth_enabled: false,
            ..Default::default()
        };
        let startup = build_startup_message("test_user", "testdb");
        let mut input = std::io::Cursor::new(startup);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.get("user").map(|s| s.as_str()), Some("test_user"));
        // Output should contain AuthenticationOk (R + 8-byte-len + 0)
        assert!(output.contains(&b'R'));
    }

    #[tokio::test]
    async fn test_pgwire_auth_sends_cleartext_challenge() {
        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("secret123".to_string()),
            ..Default::default()
        };
        let mut input_data = build_startup_message("test_user", "testdb");
        input_data.extend_from_slice(&build_password_message("secret123"));
        let mut input = std::io::Cursor::new(input_data);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(result.is_ok());
        // Output should contain AuthenticationCleartextPassword (R + type=3)
        // followed by AuthenticationOk (R + type=0)
        // Find the first R message
        let r_positions: Vec<usize> = output
            .iter()
            .enumerate()
            .filter(|&(_, b)| *b == b'R')
            .map(|(i, _)| i)
            .collect();
        assert!(
            r_positions.len() >= 2,
            "Should have at least 2 R messages (cleartext + ok)"
        );
        // First R message should have auth type 3 (cleartext)
        let first_r = r_positions[0];
        let auth_type = i32::from_be_bytes([
            output[first_r + 5],
            output[first_r + 6],
            output[first_r + 7],
            output[first_r + 8],
        ]);
        assert_eq!(auth_type, 3, "First R should be CleartextPassword (3)");
    }

    #[tokio::test]
    async fn test_pgwire_auth_correct_password_succeeds() {
        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("my_password".to_string()),
            ..Default::default()
        };
        let mut input_data = build_startup_message("alice", "mydb");
        input_data.extend_from_slice(&build_password_message("my_password"));
        let mut input = std::io::Cursor::new(input_data);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.get("user").map(|s| s.as_str()), Some("alice"));
    }

    #[tokio::test]
    async fn test_pgwire_auth_wrong_password_fails() {
        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("correct_password".to_string()),
            ..Default::default()
        };
        let mut input_data = build_startup_message("alice", "mydb");
        input_data.extend_from_slice(&build_password_message("wrong_password"));
        let mut input = std::io::Cursor::new(input_data);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            PgWireError::Auth(msg) => assert!(msg.contains("password authentication failed")),
            other => panic!("Expected Auth error, got: {}", other),
        }
    }

    #[tokio::test]
    async fn test_pgwire_auth_empty_password_fails() {
        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("".to_string()),
            ..Default::default()
        };
        let mut input_data = build_startup_message("alice", "mydb");
        input_data.extend_from_slice(&build_password_message(""));
        let mut input = std::io::Cursor::new(input_data);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(result.is_err(), "Empty password should be rejected");
    }

    #[test]
    fn test_pgwire_auth_config_wiring() {
        // Verify PgWireConfig defaults
        let config = PgWireConfig::default();
        assert!(!config.auth_enabled);
        assert!(config.auth_password.is_none());

        // Verify custom auth config
        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("test_pass".to_string()),
            ..Default::default()
        };
        assert!(config.auth_enabled);
        assert_eq!(config.auth_password.as_deref(), Some("test_pass"));
    }

    // --- SCRAM-SHA-256 Tests ---

    /// Build a SASLInitialResponse message ('p' + length + mechanism\0 + client-first-len(4) + client-first)
    fn build_sasl_initial_response(mechanism: &str, client_first: &str) -> Vec<u8> {
        let mech_bytes = mechanism.as_bytes();
        let cf_bytes = client_first.as_bytes();
        // Body: mechanism\0 + client_first_length(4) + client_first_data
        let body_len = mech_bytes.len() + 1 + 4 + cf_bytes.len();
        let msg_len = 4 + body_len; // length includes self
        let mut msg = Vec::new();
        msg.push(b'p');
        msg.extend_from_slice(&(msg_len as i32).to_be_bytes());
        msg.extend_from_slice(mech_bytes);
        msg.push(0); // null terminator for mechanism
        msg.extend_from_slice(&(cf_bytes.len() as i32).to_be_bytes());
        msg.extend_from_slice(cf_bytes);
        msg
    }

    /// Build a SASLResponse message ('p' + length + client-final)
    fn build_sasl_response(client_final: &str) -> Vec<u8> {
        let cf_bytes = client_final.as_bytes();
        let msg_len = 4 + cf_bytes.len();
        let mut msg = Vec::new();
        msg.push(b'p');
        msg.extend_from_slice(&(msg_len as i32).to_be_bytes());
        msg.extend_from_slice(cf_bytes);
        msg
    }

    /// Parse an R message from output buffer at given offset.
    /// Returns (auth_type, data_after_type, next_offset).
    fn parse_r_message(buf: &[u8], offset: usize) -> (i32, Vec<u8>, usize) {
        assert_eq!(
            buf[offset], b'R',
            "Expected 'R' message at offset {}",
            offset
        );
        let len = i32::from_be_bytes([
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
        ]) as usize;
        let auth_type = i32::from_be_bytes([
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
            buf[offset + 8],
        ]);
        let data = buf[offset + 9..offset + 1 + len].to_vec();
        (auth_type, data, offset + 1 + len)
    }

    #[tokio::test]
    async fn test_pgwire_scram_auth_success() {
        // Set up RBAC with a user that has SCRAM credentials
        let rbac = std::sync::Arc::new(crate::rbac::RBACManager::new());
        let mut user = crate::rbac::User::new("alice", "alice");
        let scram_creds = crate::scram::generate_credentials("secret123");
        let scram_json = crate::scram::credentials_to_json(&scram_creds);
        user.metadata
            .insert("password_hash".to_string(), scram_json);
        user.roles.insert("writer".to_string());
        rbac.create_user(user).unwrap();

        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: None,
            rbac_manager: Some(rbac),
            ..Default::default()
        };

        // Build input: startup + SASL handshake
        // We simulate this by using a ScramClient from our scram module
        let mut scram_client = crate::scram::ScramClient::new("alice", "secret123");
        let client_first = scram_client.client_first();

        // Build startup + SASLInitialResponse
        let mut input_data = build_startup_message("alice", "testdb");
        input_data.extend_from_slice(&build_sasl_initial_response("SCRAM-SHA-256", &client_first));

        // We need a two-phase approach: first send startup + initial response,
        // then read server-first to compute client-final.
        // Since we're using Cursor, we need to pre-compute everything.
        // But we can't pre-compute client_final without server_first.
        // So we use a pipe (tokio duplex).
        let (client_stream, server_stream) = tokio::io::duplex(8192);
        let (mut client_reader, mut client_writer) = tokio::io::split(client_stream);

        let config_clone = config.clone();
        let server_handle = tokio::spawn(async move {
            let (server_reader, server_writer) = tokio::io::split(server_stream);
            let mut reader = BufReader::new(server_reader);
            let mut writer = BufWriter::new(server_writer);
            startup_handshake(&mut reader, &mut writer, &config_clone).await
        });

        // Client side: send startup
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let startup = build_startup_message("alice", "testdb");
        client_writer.write_all(&startup).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read AuthenticationSASL (type 10)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 10, "Should get AuthenticationSASL");
        // Read remaining SASL data (mechanism list)
        let remaining = len - 8; // length includes self(4) + auth_type(4)
        let mut sasl_data = vec![0u8; remaining];
        client_reader.read_exact(&mut sasl_data).await.unwrap();
        // Should contain "SCRAM-SHA-256\0\0"
        let mechanism = String::from_utf8_lossy(&sasl_data);
        assert!(
            mechanism.contains("SCRAM-SHA-256"),
            "Should offer SCRAM-SHA-256"
        );

        // Send SASLInitialResponse
        let sasl_initial = build_sasl_initial_response("SCRAM-SHA-256", &client_first);
        client_writer.write_all(&sasl_initial).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read AuthenticationSASLContinue (type 11) with server-first
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 11, "Should get AuthenticationSASLContinue");
        let remaining = len - 8;
        let mut server_first_bytes = vec![0u8; remaining];
        client_reader
            .read_exact(&mut server_first_bytes)
            .await
            .unwrap();
        let server_first = String::from_utf8_lossy(&server_first_bytes).to_string();

        // Compute client-final
        let client_final = scram_client.client_final(&server_first).unwrap();

        // Send SASLResponse
        let sasl_response = build_sasl_response(&client_final);
        client_writer.write_all(&sasl_response).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read AuthenticationSASLFinal (type 12)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 12, "Should get AuthenticationSASLFinal");
        let remaining = len - 8;
        let mut server_final_bytes = vec![0u8; remaining];
        client_reader
            .read_exact(&mut server_final_bytes)
            .await
            .unwrap();
        let server_final = String::from_utf8_lossy(&server_final_bytes).to_string();
        assert!(
            server_final.starts_with("v="),
            "Server final should start with v="
        );

        // Read AuthenticationOk (type 0)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let _len = client_reader.read_i32().await.unwrap();
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 0, "Should get AuthenticationOk");

        // Server should complete successfully
        let result = server_handle.await.unwrap();
        assert!(result.is_ok(), "SCRAM auth should succeed: {:?}", result);
    }

    #[tokio::test]
    async fn test_pgwire_scram_wrong_password_fails() {
        let rbac = std::sync::Arc::new(crate::rbac::RBACManager::new());
        let mut user = crate::rbac::User::new("bob", "bob");
        let scram_creds = crate::scram::generate_credentials("correct_password");
        let scram_json = crate::scram::credentials_to_json(&scram_creds);
        user.metadata
            .insert("password_hash".to_string(), scram_json);
        rbac.create_user(user).unwrap();

        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: None,
            rbac_manager: Some(rbac),
            ..Default::default()
        };

        let (client_stream, server_stream) = tokio::io::duplex(8192);
        let (mut client_reader, mut client_writer) = tokio::io::split(client_stream);

        let config_clone = config.clone();
        let server_handle = tokio::spawn(async move {
            let (server_reader, server_writer) = tokio::io::split(server_stream);
            let mut reader = BufReader::new(server_reader);
            let mut writer = BufWriter::new(server_writer);
            startup_handshake(&mut reader, &mut writer, &config_clone).await
        });

        // Client side with WRONG password
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut scram_client = crate::scram::ScramClient::new("bob", "wrong_password");
        let client_first = scram_client.client_first();

        let startup = build_startup_message("bob", "testdb");
        client_writer.write_all(&startup).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read AuthenticationSASL (type 10)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut sasl_data = vec![0u8; remaining];
        client_reader.read_exact(&mut sasl_data).await.unwrap();

        // Send SASLInitialResponse
        let sasl_initial = build_sasl_initial_response("SCRAM-SHA-256", &client_first);
        client_writer.write_all(&sasl_initial).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read AuthenticationSASLContinue (type 11)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut server_first_bytes = vec![0u8; remaining];
        client_reader
            .read_exact(&mut server_first_bytes)
            .await
            .unwrap();
        let server_first = String::from_utf8_lossy(&server_first_bytes).to_string();

        // Compute client-final with wrong password
        let client_final = scram_client.client_final(&server_first).unwrap();
        let sasl_response = build_sasl_response(&client_final);
        client_writer.write_all(&sasl_response).await.unwrap();
        client_writer.flush().await.unwrap();

        // Server should reject — read ErrorResponse ('E') or detect server error
        let result = server_handle.await.unwrap();
        assert!(result.is_err(), "Wrong password should fail SCRAM auth");
    }

    #[tokio::test]
    async fn test_pgwire_scram_fallback_to_cleartext() {
        // User not in RBAC but cleartext password is configured → falls back to cleartext
        let rbac = std::sync::Arc::new(crate::rbac::RBACManager::new());
        // Don't add any users — unknown_user won't have SCRAM creds

        let config = PgWireConfig {
            auth_enabled: true,
            auth_password: Some("fallback_pass".to_string()),
            rbac_manager: Some(rbac),
            ..Default::default()
        };

        // Build input with cleartext password flow
        let mut input_data = build_startup_message("unknown_user", "testdb");
        input_data.extend_from_slice(&build_password_message("fallback_pass"));
        let mut input = std::io::Cursor::new(input_data);
        let mut output = Vec::new();

        let result = startup_handshake(&mut input, &mut output, &config).await;
        assert!(
            result.is_ok(),
            "Should fall back to cleartext auth: {:?}",
            result
        );

        // First R message should be type 3 (cleartext), not type 10 (SASL)
        let (auth_type, _, _) = parse_r_message(&output, 0);
        assert_eq!(auth_type, 3, "Should use cleartext auth for unknown user");
    }

    #[tokio::test]
    async fn test_pgwire_scram_credentials_lookup() {
        let rbac = std::sync::Arc::new(crate::rbac::RBACManager::new());

        // User with SCRAM credentials
        let mut user = crate::rbac::User::new("scram_user", "scram_user");
        let scram_creds = crate::scram::generate_credentials("test_pass");
        let scram_json = crate::scram::credentials_to_json(&scram_creds);
        user.metadata
            .insert("password_hash".to_string(), scram_json);
        rbac.create_user(user).unwrap();

        // User with old-format SHA-256 hash (no SCRAM)
        let mut user2 = crate::rbac::User::new("legacy_user", "legacy_user");
        user2
            .metadata
            .insert("password_hash".to_string(), "abc123def456".to_string());
        rbac.create_user(user2).unwrap();

        // SCRAM user → should get credentials
        let creds = get_user_scram_credentials(&rbac, "scram_user");
        assert!(creds.is_some(), "SCRAM user should have credentials");

        // Legacy user → should get None (falls back to cleartext)
        let creds = get_user_scram_credentials(&rbac, "legacy_user");
        assert!(
            creds.is_none(),
            "Legacy user should not have SCRAM credentials"
        );

        // Unknown user → should get None
        let creds = get_user_scram_credentials(&rbac, "nonexistent");
        assert!(
            creds.is_none(),
            "Unknown user should not have SCRAM credentials"
        );
    }

    #[tokio::test]
    async fn test_pgwire_scram_do_auth_roundtrip() {
        // Test do_scram_auth directly with pipe
        let creds = crate::scram::generate_credentials("mypassword");
        let (client_stream, server_stream) = tokio::io::duplex(8192);
        let (mut client_reader, mut client_writer) = tokio::io::split(client_stream);

        let creds_clone = creds.clone();
        let server_handle = tokio::spawn(async move {
            let (server_reader, server_writer) = tokio::io::split(server_stream);
            let mut reader = BufReader::new(server_reader);
            let mut writer = BufWriter::new(server_writer);
            do_scram_auth(&mut reader, &mut writer, &creds_clone).await
        });

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Read AuthenticationSASL (type 10)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 10);
        let remaining = len - 8;
        let mut sasl_data = vec![0u8; remaining];
        client_reader.read_exact(&mut sasl_data).await.unwrap();

        // Send SASLInitialResponse
        let mut scram_client = crate::scram::ScramClient::new("user", "mypassword");
        let client_first = scram_client.client_first();
        let sasl_initial = build_sasl_initial_response("SCRAM-SHA-256", &client_first);
        client_writer.write_all(&sasl_initial).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read SASLContinue (type 11)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 11);
        let remaining = len - 8;
        let mut sf_bytes = vec![0u8; remaining];
        client_reader.read_exact(&mut sf_bytes).await.unwrap();
        let server_first = String::from_utf8_lossy(&sf_bytes).to_string();

        // Send SASLResponse (client-final)
        let client_final = scram_client.client_final(&server_first).unwrap();
        let sasl_response = build_sasl_response(&client_final);
        client_writer.write_all(&sasl_response).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read SASLFinal (type 12)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let auth_type = client_reader.read_i32().await.unwrap();
        assert_eq!(auth_type, 12);
        let remaining = len - 8;
        let mut final_bytes = vec![0u8; remaining];
        client_reader.read_exact(&mut final_bytes).await.unwrap();
        let server_final = String::from_utf8_lossy(&final_bytes).to_string();
        assert!(server_final.starts_with("v="));

        let result = server_handle.await.unwrap();
        assert!(result.is_ok(), "do_scram_auth should succeed: {:?}", result);
    }

    /// Build a SASLInitialResponse matching real libpq format exactly:
    /// 'p' + msg_length(4) + mechanism\0 + response_length(4) + response_data
    fn build_sasl_initial_response_libpq(mechanism: &str, client_first: &str) -> Vec<u8> {
        let mech_bytes = mechanism.as_bytes();
        let cf_bytes = client_first.as_bytes();
        let body_len = mech_bytes.len() + 1 + 4 + cf_bytes.len();
        let msg_len = 4 + body_len;
        let mut msg = Vec::new();
        msg.push(b'p');
        msg.extend_from_slice(&(msg_len as i32).to_be_bytes());
        msg.extend_from_slice(mech_bytes);
        msg.push(0);
        msg.extend_from_slice(&(cf_bytes.len() as i32).to_be_bytes());
        msg.extend_from_slice(cf_bytes);
        msg
    }

    /// Build a SASLInitialResponse with no initial data (length = -1)
    fn build_sasl_initial_response_no_data(mechanism: &str) -> Vec<u8> {
        let mech_bytes = mechanism.as_bytes();
        let body_len = mech_bytes.len() + 1 + 4; // mechanism\0 + length(-1)
        let msg_len = 4 + body_len;
        let mut msg = Vec::new();
        msg.push(b'p');
        msg.extend_from_slice(&(msg_len as i32).to_be_bytes());
        msg.extend_from_slice(mech_bytes);
        msg.push(0);
        msg.extend_from_slice(&(-1i32).to_be_bytes()); // -1 = no initial data
        msg
    }

    #[tokio::test]
    async fn test_pgwire_scram_libpq_format_compatibility() {
        // Test with exact libpq wire format: mechanism\0 + length(4) + data
        let creds = crate::scram::generate_credentials("testpass");
        let (client_stream, server_stream) = tokio::io::duplex(8192);
        let (mut client_reader, mut client_writer) = tokio::io::split(client_stream);

        let creds_clone = creds.clone();
        let server_handle = tokio::spawn(async move {
            let (server_reader, server_writer) = tokio::io::split(server_stream);
            let mut reader = BufReader::new(server_reader);
            let mut writer = BufWriter::new(server_writer);
            do_scram_auth(&mut reader, &mut writer, &creds_clone).await
        });

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Read AuthenticationSASL (type 10)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut sasl_data = vec![0u8; remaining];
        client_reader.read_exact(&mut sasl_data).await.unwrap();

        // Send SASLInitialResponse in exact libpq format (with 4-byte length prefix)
        let mut scram_client = crate::scram::ScramClient::new("user", "testpass");
        let client_first = scram_client.client_first();
        let sasl_initial = build_sasl_initial_response_libpq("SCRAM-SHA-256", &client_first);
        client_writer.write_all(&sasl_initial).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read SASLContinue (type 11)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut sf_bytes = vec![0u8; remaining];
        client_reader.read_exact(&mut sf_bytes).await.unwrap();
        let server_first = String::from_utf8_lossy(&sf_bytes).to_string();

        // Send SASLResponse (client-final)
        let client_final = scram_client.client_final(&server_first).unwrap();
        let sasl_response = build_sasl_response(&client_final);
        client_writer.write_all(&sasl_response).await.unwrap();
        client_writer.flush().await.unwrap();

        // Read SASLFinal (type 12)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut final_bytes = vec![0u8; remaining];
        client_reader.read_exact(&mut final_bytes).await.unwrap();
        let server_final = String::from_utf8_lossy(&final_bytes).to_string();
        assert!(
            server_final.starts_with("v="),
            "Server final should be v=signature"
        );

        let result = server_handle.await.unwrap();
        assert!(
            result.is_ok(),
            "libpq-format SCRAM should succeed: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_pgwire_scram_no_initial_response() {
        // Test that length=-1 (no initial response) is handled gracefully
        let creds = crate::scram::generate_credentials("testpass");
        let (client_stream, server_stream) = tokio::io::duplex(8192);
        let (mut client_reader, mut client_writer) = tokio::io::split(client_stream);

        let creds_clone = creds.clone();
        let server_handle = tokio::spawn(async move {
            let (server_reader, server_writer) = tokio::io::split(server_stream);
            let mut reader = BufReader::new(server_reader);
            let mut writer = BufWriter::new(server_writer);
            do_scram_auth(&mut reader, &mut writer, &creds_clone).await
        });

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Read AuthenticationSASL (type 10)
        let tag = client_reader.read_u8().await.unwrap();
        assert_eq!(tag, b'R');
        let len = client_reader.read_i32().await.unwrap() as usize;
        let _auth_type = client_reader.read_i32().await.unwrap();
        let remaining = len - 8;
        let mut sasl_data = vec![0u8; remaining];
        client_reader.read_exact(&mut sasl_data).await.unwrap();

        // Send SASLInitialResponse with length=-1 (no data)
        let sasl_initial = build_sasl_initial_response_no_data("SCRAM-SHA-256");
        client_writer.write_all(&sasl_initial).await.unwrap();
        client_writer.flush().await.unwrap();

        // Server should return an error (empty client-first is invalid for SCRAM)
        let result = server_handle.await.unwrap();
        assert!(
            result.is_err(),
            "No initial response should fail SCRAM: {:?}",
            result
        );
    }

    // --- TLS Tests ---

    #[cfg(feature = "tls")]
    mod tls_tests {
        use super::*;

        fn generate_self_signed_cert() -> (String, String) {
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
            let cert_pem = cert.cert.pem();
            let key_pem = cert.key_pair.serialize_pem();
            (cert_pem, key_pem)
        }

        fn create_tls_acceptor_from_pem(
            cert_pem: &str,
            key_pem: &str,
        ) -> tokio_rustls::TlsAcceptor {
            let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
                rustls_pemfile::certs(&mut cert_pem.as_bytes())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
            let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())
                .unwrap()
                .unwrap();
            let config = rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)
                .unwrap();
            tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config))
        }

        #[test]
        fn test_pgwire_tls_config_wiring() {
            let (cert_pem, key_pem) = generate_self_signed_cert();
            let acceptor = create_tls_acceptor_from_pem(&cert_pem, &key_pem);
            let config = PgWireConfig {
                tls_acceptor: Some(acceptor),
                ..Default::default()
            };
            assert!(config.tls_acceptor.is_some());

            let config_no_tls = PgWireConfig::default();
            assert!(config_no_tls.tls_acceptor.is_none());
        }

        #[tokio::test]
        async fn test_pgwire_tls_ssl_request_accepted() {
            // Simulate: client sends SSL_REQUEST, server has TLS → responds with 'S'
            let (cert_pem, key_pem) = generate_self_signed_cert();
            let acceptor = create_tls_acceptor_from_pem(&cert_pem, &key_pem);

            // Start a listener, connect a client
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            let server_handle = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                stream.set_nodelay(true).ok();

                // Read 8 bytes (SSL request)
                let mut buf = [0u8; 8];
                let mut stream = stream;
                use tokio::io::AsyncReadExt;
                stream.read_exact(&mut buf).await.unwrap();

                let version = i32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
                assert_eq!(version, SSL_REQUEST);

                // Server would respond 'S' when TLS is configured
                use tokio::io::AsyncWriteExt;
                stream.write_all(&[b'S']).await.unwrap();
                stream.flush().await.unwrap();

                // TLS handshake would happen here, but we just verify the 'S' response
                true
            });

            // Client sends SSL request
            let client = tokio::net::TcpStream::connect(addr).await.unwrap();
            let (_, mut client_writer) = tokio::io::split(client);
            let mut ssl_request = Vec::new();
            ssl_request.extend_from_slice(&8i32.to_be_bytes());
            ssl_request.extend_from_slice(&SSL_REQUEST.to_be_bytes());
            use tokio::io::AsyncWriteExt;
            client_writer.write_all(&ssl_request).await.unwrap();

            assert!(server_handle.await.unwrap());
        }

        #[tokio::test]
        async fn test_pgwire_tls_ssl_request_declined_no_config() {
            // No TLS configured → server responds 'N'
            let config = PgWireConfig::default();
            assert!(config.tls_acceptor.is_none());

            // Build SSL_REQUEST as input
            let mut input_data = Vec::new();
            input_data.extend_from_slice(&8i32.to_be_bytes());
            input_data.extend_from_slice(&SSL_REQUEST.to_be_bytes());
            // After 'N' response, client sends a real startup:
            let startup = build_startup_message("user", "db");
            input_data.extend_from_slice(&startup);

            // In startup_handshake, SSL_REQUEST triggers 'N' + recursive call
            let mut input = std::io::Cursor::new(input_data);
            let mut output = Vec::new();

            let result = startup_handshake(&mut input, &mut output, &config).await;
            assert!(result.is_ok());
            // First byte of output should be 'N' (SSL declined)
            assert_eq!(output[0], b'N');
        }

        #[tokio::test]
        async fn test_pgwire_tls_handshake_succeeds() {
            // Full end-to-end TLS test: client→SSL_REQUEST→'S'→TLS handshake→startup→query
            let (cert_pem, key_pem) = generate_self_signed_cert();
            let acceptor = create_tls_acceptor_from_pem(&cert_pem, &key_pem);

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            // Create a mock executor
            let dir = tempfile::tempdir().unwrap();
            let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic =
                std::sync::Arc::new(crate::amorphic_adapter::AmorphicTableStorage::new(store));
            let executor: std::sync::Arc<dyn crate::query::QueryExecutor> =
                std::sync::Arc::new(crate::query::SimpleQueryExecutor::with_amorphic(amorphic));

            let config = PgWireConfig {
                tls_acceptor: Some(acceptor),
                ..Default::default()
            };

            let server_executor = executor.clone();
            let server_config = config.clone();
            let server_handle = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                handle_connection(
                    stream,
                    server_executor,
                    &server_config,
                    &PgWireStats::default(),
                )
                .await
            });

            // Client: send SSL request, do TLS handshake, then send startup + query
            let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            let mut stream = stream;

            // Send SSL request
            let mut ssl_request = Vec::new();
            ssl_request.extend_from_slice(&8i32.to_be_bytes());
            ssl_request.extend_from_slice(&SSL_REQUEST.to_be_bytes());
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            stream.write_all(&ssl_request).await.unwrap();
            stream.flush().await.unwrap();

            // Read response
            let response = stream.read_u8().await.unwrap();
            assert_eq!(response, b'S', "Server should accept SSL");

            // Now do TLS handshake as client
            let mut root_store = rustls::RootCertStore::empty();
            let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
                rustls_pemfile::certs(&mut cert_pem.as_bytes())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
            for cert in &certs {
                root_store.add(cert.clone()).unwrap();
            }
            let client_config = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(client_config));
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            let mut tls_stream = connector.connect(server_name, stream).await.unwrap();

            // Send startup message over TLS
            let startup = build_startup_message("test_user", "testdb");
            tls_stream.write_all(&startup).await.unwrap();
            tls_stream.flush().await.unwrap();

            // Read AuthenticationOk (R + 8 bytes)
            let tag = tls_stream.read_u8().await.unwrap();
            assert_eq!(tag, b'R');
            let len = tls_stream.read_i32().await.unwrap();
            assert_eq!(len, 8);
            let auth_type = tls_stream.read_i32().await.unwrap();
            assert_eq!(auth_type, 0, "Should get AuthenticationOk");

            // Send Terminate to close cleanly
            tls_stream.write_all(&[b'X']).await.unwrap();
            tls_stream.write_all(&4i32.to_be_bytes()).await.unwrap();
            tls_stream.flush().await.unwrap();

            // Server should complete without error
            let result = server_handle.await.unwrap();
            assert!(
                result.is_ok(),
                "Server should handle TLS connection: {:?}",
                result
            );
        }

        #[tokio::test]
        async fn test_pgwire_tls_query_over_tls() {
            let (cert_pem, key_pem) = generate_self_signed_cert();
            let acceptor = create_tls_acceptor_from_pem(&cert_pem, &key_pem);

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            let dir = tempfile::tempdir().unwrap();
            let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic =
                std::sync::Arc::new(crate::amorphic_adapter::AmorphicTableStorage::new(store));
            let executor: std::sync::Arc<dyn crate::query::QueryExecutor> =
                std::sync::Arc::new(crate::query::SimpleQueryExecutor::with_amorphic(amorphic));

            let config = PgWireConfig {
                tls_acceptor: Some(acceptor),
                ..Default::default()
            };

            let server_executor = executor.clone();
            let server_config = config.clone();
            let server_handle = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                handle_connection(
                    stream,
                    server_executor,
                    &server_config,
                    &PgWireStats::default(),
                )
                .await
            });

            // Client setup
            let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            let mut stream = stream;

            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            // SSL request
            let mut ssl_request = Vec::new();
            ssl_request.extend_from_slice(&8i32.to_be_bytes());
            ssl_request.extend_from_slice(&SSL_REQUEST.to_be_bytes());
            stream.write_all(&ssl_request).await.unwrap();
            let response = stream.read_u8().await.unwrap();
            assert_eq!(response, b'S');

            // TLS handshake
            let mut root_store = rustls::RootCertStore::empty();
            let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
                rustls_pemfile::certs(&mut cert_pem.as_bytes())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
            for cert in &certs {
                root_store.add(cert.clone()).unwrap();
            }
            let client_config = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(client_config));
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            let mut tls_stream = connector.connect(server_name, stream).await.unwrap();

            // Startup
            let startup = build_startup_message("test_user", "testdb");
            tls_stream.write_all(&startup).await.unwrap();
            tls_stream.flush().await.unwrap();

            // Consume startup response (AuthOk + ParameterStatus + BackendKeyData + ReadyForQuery)
            let mut response_buf = vec![0u8; 4096];
            let n = tls_stream.read(&mut response_buf).await.unwrap();
            assert!(n > 0);

            // Send a simple query: SELECT 1 AS value
            let sql = b"SELECT 1 AS value\0";
            let mut query_msg = Vec::new();
            query_msg.push(b'Q');
            query_msg.extend_from_slice(&((sql.len() as i32) + 4).to_be_bytes());
            query_msg.extend_from_slice(sql);
            tls_stream.write_all(&query_msg).await.unwrap();
            tls_stream.flush().await.unwrap();

            // Read response — should contain row data
            let mut resp = vec![0u8; 4096];
            let n = tls_stream.read(&mut resp).await.unwrap();
            assert!(n > 0, "Should receive query response over TLS");

            // Find 'T' (RowDescription) or 'D' (DataRow) in response
            let has_row_data = resp[..n].iter().any(|&b| b == b'T' || b == b'D');
            assert!(
                has_row_data,
                "Response should contain row description or data"
            );

            // Terminate
            tls_stream.write_all(&[b'X']).await.unwrap();
            tls_stream.write_all(&4i32.to_be_bytes()).await.unwrap();
            tls_stream.flush().await.unwrap();

            let result = server_handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    // ── Parameter substitution tests ─────────────────────────────────

    #[test]
    fn test_substitute_params_basic() {
        let sql = "SELECT * FROM users WHERE id = $1 AND name = $2";
        let params = vec![serde_json::json!(42), serde_json::json!("alice")];
        let result = substitute_params(sql, &params);
        assert_eq!(
            result,
            "SELECT * FROM users WHERE id = 42 AND name = 'alice'"
        );
    }

    #[test]
    fn test_substitute_params_null_and_bool() {
        let sql = "INSERT INTO t VALUES ($1, $2, $3)";
        let params = vec![
            serde_json::Value::Null,
            serde_json::json!(true),
            serde_json::json!(false),
        ];
        let result = substitute_params(sql, &params);
        assert_eq!(result, "INSERT INTO t VALUES (NULL, TRUE, FALSE)");
    }

    #[test]
    fn test_substitute_params_sql_injection_escaped() {
        let sql = "SELECT * FROM users WHERE name = $1";
        let params = vec![serde_json::json!("O'Brien")];
        let result = substitute_params(sql, &params);
        assert_eq!(result, "SELECT * FROM users WHERE name = 'O''Brien'");
    }

    #[test]
    fn test_substitute_params_inside_string_literal_untouched() {
        let sql = "SELECT '$1' AS literal, $1 AS param";
        let params = vec![serde_json::json!(99)];
        let result = substitute_params(sql, &params);
        assert_eq!(result, "SELECT '$1' AS literal, 99 AS param");
    }

    #[test]
    fn test_substitute_params_empty() {
        let sql = "SELECT 1";
        let result = substitute_params(sql, &[]);
        assert_eq!(result, "SELECT 1");
    }

    #[test]
    fn test_substitute_params_multi_digit() {
        let sql = "SELECT $1, $10, $2";
        let params: Vec<serde_json::Value> = (1..=10).map(|i| serde_json::json!(i)).collect();
        let result = substitute_params(sql, &params);
        assert_eq!(result, "SELECT 1, 10, 2");
    }

    #[test]
    fn split_csv_parallel_matches_sequential() {
        let mut d = String::new();
        for i in 0..60_000 {
            d.push_str(&format!("{i},,u{},{}.5\r\n", i % 13, i));
            if i % 977 == 0 {
                d.push('\n');
            }
        }
        d.push_str("last,1");
        assert!(d.len() >= 1 << 20);
        assert_eq!(split_csv(&d, ',', ""), split_csv_seq(&d, ',', ""));
    }

    #[test]
    fn copy_parse_and_fields() {
        let c = parse_copy("COPY t (a, b) FROM STDIN WITH (FORMAT csv, HEADER true)").unwrap();
        assert!(c.from_stdin && c.csv && c.header && c.columns == vec!["a", "b"]);
        let o = parse_copy("COPY (SELECT 1) TO STDOUT").unwrap();
        assert!(o.is_query && !o.from_stdin && o.delimiter == '\t');
        assert!(parse_copy("COPY t FROM '/etc/passwd'").is_none());
        assert_eq!(
            split_copy_text("1\t\\N\ta\\tb", '\t', "\\N"),
            vec![Some("1".into()), None, Some("a\tb".into())]
        );
        let rows = split_csv("1,\"x,y\",\n2,\"q\"\"\",z\n", ',', "");
        assert_eq!(rows[0], vec![Some("1".into()), Some("x,y".into()), None]);
        assert_eq!(
            rows[1],
            vec![Some("2".into()), Some("q\"".into()), Some("z".into())]
        );
        assert_eq!(sql_literal(&Some("007".into())), "'007'");
        assert_eq!(sql_literal(&Some("2.5".into())), "2.5");
        assert_eq!(sql_literal(&Some("o'k".into())), "'o''k'");
    }

    #[test]
    fn params_decode_by_oid_and_format() {
        assert_eq!(decode_param(b"42", 0, 23), serde_json::json!(42));
        assert_eq!(decode_param(b"1.5", 0, 701), serde_json::json!(1.5));
        assert_eq!(decode_param(b"42", 0, 0), serde_json::json!("42"));
        assert_eq!(
            decode_param(&7i32.to_be_bytes(), 1, 23),
            serde_json::json!(7)
        );
        assert_eq!(
            decode_param(&2.5f64.to_be_bytes(), 1, 701),
            serde_json::json!(2.5)
        );
        assert_eq!(decode_param(b"t", 0, 16), serde_json::json!(true));
    }

    #[test]
    fn column_oids_are_consistent() {
        let rows = vec![
            vec![serde_json::json!(1), serde_json::json!(1.5)],
            vec![serde_json::json!(5_000_000_000i64), serde_json::json!(2)],
        ];
        assert_eq!(column_oids(&rows, 2, &[]), vec![20, 701]);
        assert_eq!(
            strip_pg_catalog(
                "select pg_catalog.version(), 'pg_catalog.x' from pg_catalog.pg_class"
            ),
            "select version(), 'pg_catalog.x' from pg_class"
        );
        // multi-byte text must not split a char (stress_pgwire unicode regression)
        assert_eq!(
            strip_pg_catalog("SELECT 'héllo 世界 🎉' FROM t"),
            "SELECT 'héllo 世界 🎉' FROM t"
        );
        assert_eq!(
            strip_pg_catalog("SELECT 世界, pg_catalog.version()"),
            "SELECT 世界, version()"
        );
        assert!(
            returns_rows("  WITH a AS (SELECT 1) SELECT * FROM a")
                && !returns_rows("INSERT INTO t VALUES (1)")
        );
        assert_eq!(count_placeholders("SELECT $1, $3"), 3);
    }
}
