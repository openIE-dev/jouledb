//! TCP server implementation using the JouleDB binary protocol
//!
//! This module provides a high-performance TCP server for JouleDB that
//! implements the binary protocol defined in `joule-db-core::persistence::network`.
//!
//! ## Features
//!
//! - Async I/O using tokio
//! - Connection pooling
//! - Request pipelining
//! - Graceful shutdown
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_local::{Database, server::TcpServer};
//!
//! let db = Database::open("./mydb")?;
//! let server = TcpServer::new(db);
//! server.bind("127.0.0.1:6379").await?;
//! server.run().await?;
//! ```

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, Semaphore};

use joule_db_core::StorageError;
use joule_db_core::persistence::network::{
    ErrorCode, HEADER_SIZE, Message, OpCode, PROTOCOL_MAGIC, decode_key_value, encode_key_value,
};

use crate::Database;

/// TCP server configuration
#[derive(Debug, Clone)]
pub struct TcpServerConfig {
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Read buffer size per connection
    pub read_buffer_size: usize,
    /// Write buffer size per connection
    pub write_buffer_size: usize,
    /// Connection timeout in seconds (0 = no timeout)
    pub connection_timeout_secs: u64,
    /// Idle timeout in seconds (0 = no timeout)
    pub idle_timeout_secs: u64,
    /// Enable TCP keepalive
    pub tcp_keepalive: bool,
    /// Enable TCP nodelay
    pub tcp_nodelay: bool,
}

impl Default for TcpServerConfig {
    fn default() -> Self {
        Self {
            max_connections: 10000,
            read_buffer_size: 64 * 1024,  // 64KB
            write_buffer_size: 64 * 1024, // 64KB
            connection_timeout_secs: 30,
            idle_timeout_secs: 300, // 5 minutes
            tcp_keepalive: true,
            tcp_nodelay: true,
        }
    }
}

/// Server statistics
#[derive(Debug, Default)]
pub struct ServerStats {
    /// Total connections accepted
    pub connections_accepted: AtomicU64,
    /// Currently active connections
    pub active_connections: AtomicU32,
    /// Total requests processed
    pub requests_processed: AtomicU64,
    /// Total bytes received
    pub bytes_received: AtomicU64,
    /// Total bytes sent
    pub bytes_sent: AtomicU64,
    /// Total errors
    pub errors: AtomicU64,
}

impl ServerStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of current stats
    pub fn snapshot(&self) -> ServerStatsSnapshot {
        ServerStatsSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of server statistics
#[derive(Debug, Clone)]
pub struct ServerStatsSnapshot {
    /// Total connections accepted
    pub connections_accepted: u64,
    /// Currently active connections
    pub active_connections: u32,
    /// Total requests processed
    pub requests_processed: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total errors
    pub errors: u64,
}

/// Connection state
struct Connection {
    /// Peer address
    peer_addr: SocketAddr,
    /// Request ID counter
    next_request_id: AtomicU32,
    /// Authenticated flag
    authenticated: bool,
    /// Active transaction ID (0 = no active transaction)
    active_tx_id: u64,
}

impl Connection {
    fn new(peer_addr: SocketAddr) -> Self {
        Self {
            peer_addr,
            next_request_id: AtomicU32::new(1),
            authenticated: false,
            active_tx_id: 0,
        }
    }

    /// Get the peer address
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Get the next request ID
    pub fn next_request_id(&self) -> u32 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Check if connection is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    /// Set authentication status
    pub fn set_authenticated(&mut self, authenticated: bool) {
        self.authenticated = authenticated;
    }

    /// Get active transaction ID (0 means no active transaction)
    pub fn active_tx_id(&self) -> u64 {
        self.active_tx_id
    }

    /// Set active transaction ID
    pub fn set_active_tx_id(&mut self, tx_id: u64) {
        self.active_tx_id = tx_id;
    }
}

/// TCP server for JouleDB
pub struct TcpServer {
    /// Database instance
    db: Arc<RwLock<Database>>,
    /// Server configuration
    config: TcpServerConfig,
    /// Server statistics
    stats: Arc<ServerStats>,
    /// Connection semaphore for limiting concurrent connections
    connection_semaphore: Arc<Semaphore>,
    /// Shutdown flag
    shutdown: Arc<tokio::sync::Notify>,
}

impl TcpServer {
    /// Create a new TCP server
    pub fn new(db: Database) -> Self {
        Self::with_config(db, TcpServerConfig::default())
    }

    /// Create a new TCP server with configuration
    pub fn with_config(db: Database, config: TcpServerConfig) -> Self {
        let connection_semaphore = Arc::new(Semaphore::new(config.max_connections));
        Self {
            db: Arc::new(RwLock::new(db)),
            config,
            stats: Arc::new(ServerStats::new()),
            connection_semaphore,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Get server statistics
    pub fn stats(&self) -> ServerStatsSnapshot {
        self.stats.snapshot()
    }

    /// Trigger graceful shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Run the server on the given address
    pub async fn run(&self, addr: &str) -> Result<(), StorageError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| StorageError::Backend(format!("Failed to bind: {}", e)))?;

        log::info!("JouleDB TCP server listening on {}", addr);

        loop {
            tokio::select! {
                // Accept new connections
                result = listener.accept() => {
                    match result {
                        Ok((stream, peer_addr)) => {
                            // Try to acquire connection permit
                            let permit = match self.connection_semaphore.clone().try_acquire_owned() {
                                Ok(permit) => permit,
                                Err(_) => {
                                    log::warn!("Max connections reached, rejecting {}", peer_addr);
                                    continue;
                                }
                            };

                            self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                            self.stats.active_connections.fetch_add(1, Ordering::Relaxed);

                            // Spawn connection handler
                            let db = self.db.clone();
                            let stats = self.stats.clone();
                            let config = self.config.clone();
                            let shutdown = self.shutdown.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(
                                    stream,
                                    peer_addr,
                                    db,
                                    stats.clone(),
                                    config,
                                    shutdown,
                                ).await {
                                    log::debug!("Connection {} error: {}", peer_addr, e);
                                    stats.errors.fetch_add(1, Ordering::Relaxed);
                                }
                                stats.active_connections.fetch_sub(1, Ordering::Relaxed);
                                drop(permit);
                            });
                        }
                        Err(e) => {
                            log::error!("Accept error: {}", e);
                            self.stats.errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                // Shutdown signal
                _ = self.shutdown.notified() => {
                    log::info!("Shutdown signal received");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a single connection
    async fn handle_connection(
        mut stream: TcpStream,
        peer_addr: SocketAddr,
        db: Arc<RwLock<Database>>,
        stats: Arc<ServerStats>,
        config: TcpServerConfig,
        shutdown: Arc<tokio::sync::Notify>,
    ) -> Result<(), StorageError> {
        log::debug!("New connection from {}", peer_addr);

        // Configure socket
        if config.tcp_nodelay {
            let _ = stream.set_nodelay(true);
        }

        let mut conn = Connection::new(peer_addr);
        let mut read_buf = vec![0u8; config.read_buffer_size];

        loop {
            tokio::select! {
                // Read from socket
                result = stream.read(&mut read_buf) => {
                    match result {
                        Ok(0) => {
                            // Connection closed
                            log::debug!("Connection closed by {}", peer_addr);
                            break;
                        }
                        Ok(n) => {
                            stats.bytes_received.fetch_add(n as u64, Ordering::Relaxed);

                            // Parse and handle message
                            match Self::parse_message(&read_buf[..n]) {
                                Ok(message) => {
                                    let response = Self::handle_message(
                                        &message,
                                        &mut conn,
                                        &db,
                                    ).await;

                                    let response_bytes = response.encode();
                                    stats.bytes_sent.fetch_add(response_bytes.len() as u64, Ordering::Relaxed);

                                    if let Err(e) = stream.write_all(&response_bytes).await {
                                        log::debug!("Write error to {}: {}", peer_addr, e);
                                        break;
                                    }

                                    stats.requests_processed.fetch_add(1, Ordering::Relaxed);

                                    // Check for close
                                    if message.opcode == OpCode::Close {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    log::debug!("Parse error from {}: {}", peer_addr, e);
                                    let error_response = Message::error(0, ErrorCode::InvalidRequest as u16, &e);
                                    let _ = stream.write_all(&error_response.encode()).await;
                                    stats.errors.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        Err(e) => {
                            log::debug!("Read error from {}: {}", peer_addr, e);
                            break;
                        }
                    }
                }

                // Shutdown signal
                _ = shutdown.notified() => {
                    log::debug!("Shutdown, closing connection to {}", peer_addr);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Parse a message from bytes
    fn parse_message(buf: &[u8]) -> Result<Message, String> {
        if buf.len() < HEADER_SIZE {
            return Err("Message too short".to_string());
        }

        // Verify magic
        if &buf[0..2] != &PROTOCOL_MAGIC {
            return Err("Invalid protocol magic".to_string());
        }

        Message::decode(buf).map_err(|e| e.to_string())
    }

    /// Handle a message and produce a response
    async fn handle_message(
        message: &Message,
        conn: &mut Connection,
        db: &Arc<RwLock<Database>>,
    ) -> Message {
        match message.opcode {
            OpCode::Ping => Message::response(message.request_id, OpCode::Pong, vec![]),

            OpCode::Get => {
                let key = &message.payload;
                let db_guard = db.read().await;
                match db_guard.get(key) {
                    Ok(Some(value)) => Message::response(message.request_id, OpCode::Get, value),
                    Ok(None) => Message::error(
                        message.request_id,
                        ErrorCode::NotFound as u16,
                        "Key not found",
                    ),
                    Err(e) => Message::error(
                        message.request_id,
                        ErrorCode::Internal as u16,
                        &e.to_string(),
                    ),
                }
            }

            OpCode::Put => {
                match decode_key_value(&message.payload) {
                    Ok((key, value)) => {
                        let db_guard = db.write().await;
                        match db_guard.put(key, value) {
                            Ok(()) => {
                                Message::response(message.request_id, OpCode::Put, vec![1]) // Success
                            }
                            Err(e) => Message::error(
                                message.request_id,
                                ErrorCode::Internal as u16,
                                &e.to_string(),
                            ),
                        }
                    }
                    Err(e) => Message::error(
                        message.request_id,
                        ErrorCode::InvalidRequest as u16,
                        &e.to_string(),
                    ),
                }
            }

            OpCode::Delete => {
                let key = &message.payload;
                let db_guard = db.write().await;
                match db_guard.delete(key) {
                    Ok(deleted) => {
                        Message::response(message.request_id, OpCode::Delete, vec![deleted as u8])
                    }
                    Err(e) => Message::error(
                        message.request_id,
                        ErrorCode::Internal as u16,
                        &e.to_string(),
                    ),
                }
            }

            OpCode::Exists => {
                let key = &message.payload;
                let db_guard = db.read().await;
                match db_guard.get(key) {
                    Ok(Some(_)) => Message::response(message.request_id, OpCode::Exists, vec![1]),
                    Ok(None) => Message::response(message.request_id, OpCode::Exists, vec![0]),
                    Err(e) => Message::error(
                        message.request_id,
                        ErrorCode::Internal as u16,
                        &e.to_string(),
                    ),
                }
            }

            OpCode::MGet => Self::handle_mget(message, db).await,

            OpCode::MPut => Self::handle_mput(message, db).await,

            OpCode::Info => Self::handle_info(message, db).await,

            OpCode::Flush => {
                let db_guard = db.write().await;
                match db_guard.sync() {
                    Ok(()) => Message::response(message.request_id, OpCode::Flush, vec![1]),
                    Err(e) => Message::error(
                        message.request_id,
                        ErrorCode::Internal as u16,
                        &e.to_string(),
                    ),
                }
            }

            OpCode::Close => Message::response(message.request_id, OpCode::Close, vec![]),

            _ => Message::error(
                message.request_id,
                ErrorCode::NotImplemented as u16,
                &format!("Opcode {:?} not implemented", message.opcode),
            ),
        }
    }

    /// Handle MGET (multi-get) operation
    async fn handle_mget(message: &Message, db: &Arc<RwLock<Database>>) -> Message {
        let payload = &message.payload;

        if payload.len() < 4 {
            return Message::error(
                message.request_id,
                ErrorCode::InvalidRequest as u16,
                "Invalid MGET payload",
            );
        }

        let count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let mut offset = 4;
        let mut results = Vec::new();

        let db_guard = db.read().await;

        for _ in 0..count {
            if offset + 4 > payload.len() {
                return Message::error(
                    message.request_id,
                    ErrorCode::InvalidRequest as u16,
                    "Truncated MGET payload",
                );
            }

            let key_len = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + key_len > payload.len() {
                return Message::error(
                    message.request_id,
                    ErrorCode::InvalidRequest as u16,
                    "Truncated key",
                );
            }

            let key = &payload[offset..offset + key_len];
            offset += key_len;

            match db_guard.get(key) {
                Ok(Some(value)) => {
                    results.push(1u8); // Found
                    results.extend_from_slice(&(value.len() as u32).to_le_bytes());
                    results.extend_from_slice(&value);
                }
                Ok(None) => {
                    results.push(0u8); // Not found
                }
                Err(_) => {
                    results.push(2u8); // Error
                }
            }
        }

        Message::response(message.request_id, OpCode::MGet, results)
    }

    /// Handle MPUT (multi-put) operation
    async fn handle_mput(message: &Message, db: &Arc<RwLock<Database>>) -> Message {
        let payload = &message.payload;

        if payload.len() < 4 {
            return Message::error(
                message.request_id,
                ErrorCode::InvalidRequest as u16,
                "Invalid MPUT payload",
            );
        }

        let count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let mut offset = 4;
        let mut results = Vec::with_capacity(count);

        let db_guard = db.write().await;

        for _ in 0..count {
            if offset + 8 > payload.len() {
                return Message::error(
                    message.request_id,
                    ErrorCode::InvalidRequest as u16,
                    "Truncated MPUT payload",
                );
            }

            let key_len = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + key_len > payload.len() {
                return Message::error(
                    message.request_id,
                    ErrorCode::InvalidRequest as u16,
                    "Truncated key",
                );
            }

            let key = &payload[offset..offset + key_len];
            offset += key_len;

            let value_len = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + value_len > payload.len() {
                return Message::error(
                    message.request_id,
                    ErrorCode::InvalidRequest as u16,
                    "Truncated value",
                );
            }

            let value = &payload[offset..offset + value_len];
            offset += value_len;

            match db_guard.put(key, value) {
                Ok(()) => results.push(1u8),
                Err(_) => results.push(0u8),
            }
        }

        Message::response(message.request_id, OpCode::MPut, results)
    }

    /// Handle INFO operation
    async fn handle_info(message: &Message, db: &Arc<RwLock<Database>>) -> Message {
        let db_guard = db.read().await;

        // Build info string
        let info = format!(
            "joule_db_version:0.1.0\n\
             db_path:{}\n\
             uptime_secs:0\n",
            db_guard.path()
        );

        Message::response(message.request_id, OpCode::Info, info.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db() -> (TempDir, Database) {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path()).unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn test_message_encoding() {
        let msg = Message::request(OpCode::Ping, vec![]);
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(decoded.opcode, OpCode::Ping);
    }

    #[tokio::test]
    async fn test_handle_ping() {
        let (_dir, db) = create_test_db();
        let db = Arc::new(RwLock::new(db));
        let mut conn = Connection::new("127.0.0.1:12345".parse().unwrap());

        let msg = Message::request(OpCode::Ping, vec![]);
        let response = TcpServer::handle_message(&msg, &mut conn, &db).await;

        assert_eq!(response.opcode, OpCode::Pong);
    }

    #[tokio::test]
    async fn test_handle_put_get() {
        let (_dir, db) = create_test_db();
        let db = Arc::new(RwLock::new(db));
        let mut conn = Connection::new("127.0.0.1:12345".parse().unwrap());

        // PUT
        let put_payload = encode_key_value(b"test_key", b"test_value");
        let put_msg = Message::request(OpCode::Put, put_payload);
        let put_response = TcpServer::handle_message(&put_msg, &mut conn, &db).await;
        assert!(!put_response.flags.is_error());

        // GET
        let get_msg = Message::request(OpCode::Get, b"test_key".to_vec());
        let get_response = TcpServer::handle_message(&get_msg, &mut conn, &db).await;
        assert!(!get_response.flags.is_error());
        assert_eq!(get_response.payload, b"test_value");
    }

    #[tokio::test]
    async fn test_handle_delete() {
        let (_dir, db) = create_test_db();
        let db = Arc::new(RwLock::new(db));
        let mut conn = Connection::new("127.0.0.1:12345".parse().unwrap());

        // PUT first
        let put_payload = encode_key_value(b"to_delete", b"value");
        let put_msg = Message::request(OpCode::Put, put_payload);
        TcpServer::handle_message(&put_msg, &mut conn, &db).await;

        // DELETE
        let del_msg = Message::request(OpCode::Delete, b"to_delete".to_vec());
        let del_response = TcpServer::handle_message(&del_msg, &mut conn, &db).await;
        assert!(!del_response.flags.is_error());
        assert_eq!(del_response.payload, vec![1]); // Deleted

        // GET should fail
        let get_msg = Message::request(OpCode::Get, b"to_delete".to_vec());
        let get_response = TcpServer::handle_message(&get_msg, &mut conn, &db).await;
        assert!(get_response.flags.is_error());
    }

    #[tokio::test]
    async fn test_handle_exists() {
        let (_dir, db) = create_test_db();
        let db = Arc::new(RwLock::new(db));
        let mut conn = Connection::new("127.0.0.1:12345".parse().unwrap());

        // Check non-existent
        let exists_msg = Message::request(OpCode::Exists, b"nonexistent".to_vec());
        let response = TcpServer::handle_message(&exists_msg, &mut conn, &db).await;
        assert_eq!(response.payload, vec![0]);

        // PUT
        let put_payload = encode_key_value(b"exists_key", b"value");
        let put_msg = Message::request(OpCode::Put, put_payload);
        TcpServer::handle_message(&put_msg, &mut conn, &db).await;

        // Check exists
        let exists_msg = Message::request(OpCode::Exists, b"exists_key".to_vec());
        let response = TcpServer::handle_message(&exists_msg, &mut conn, &db).await;
        assert_eq!(response.payload, vec![1]);
    }

    #[tokio::test]
    async fn test_server_stats() {
        let stats = ServerStats::new();
        stats.connections_accepted.fetch_add(5, Ordering::Relaxed);
        stats.requests_processed.fetch_add(100, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.connections_accepted, 5);
        assert_eq!(snapshot.requests_processed, 100);
    }
}
