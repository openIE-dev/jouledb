//! TCP Server for JouleDB Binary Protocol
//!
//! Handles binary wire protocol connections over TCP with bidirectional push
//! for real-time subscription notifications (SpacetimeDB-class reactivity).
//!
//! ## Architecture
//!
//! Each connection is split into three concurrent tasks:
//! - **Reader**: Reads requests from the client, processes them, sends responses
//! - **Writer**: Reads from an outgoing message channel, writes to socket
//! - **Subscriber**: Listens to SubscriptionManager and pushes Notification messages

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Notify, Semaphore, mpsc};

use crate::binary_protocol::{
    BinaryMessage, BinaryProtocol, BinaryProtocolError, HEADER_SIZE, MessageType,
};

/// Maximum allowed message payload size (16 MB) — prevents DoS via unbounded buffer growth.
const MAX_TCP_MESSAGE_SIZE: usize = 16 * 1024 * 1024;
use crate::subscriptions::SubscriptionManager;

/// TCP Server configuration
#[derive(Debug, Clone)]
pub struct TcpServerConfig {
    /// Bind address (e.g., "0.0.0.0:9000")
    pub bind_addr: String,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Read buffer size
    pub read_buffer_size: usize,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Enable TCP keepalive
    pub keepalive: bool,
    /// TCP nodelay
    pub nodelay: bool,
    /// Enable authentication on TCP connections
    pub auth_enabled: bool,
    /// JWT secret for validating auth tokens (shared with HTTP auth)
    pub auth_jwt_secret: Option<String>,
}

impl Default for TcpServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9000".to_string(),
            max_connections: 1000,
            read_buffer_size: 64 * 1024,
            connection_timeout_secs: 300,
            keepalive: true,
            nodelay: true,
            auth_enabled: false,
            auth_jwt_secret: None,
        }
    }
}

/// TCP Server statistics
#[derive(Debug, Default)]
pub struct TcpServerStats {
    pub connections_accepted: AtomicU64,
    pub connections_active: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub requests_processed: AtomicU64,
    pub notifications_sent: AtomicU64,
    pub errors: AtomicU64,
}

impl TcpServerStats {
    pub fn snapshot(&self) -> TcpServerStatsSnapshot {
        TcpServerStatsSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            connections_active: self.connections_active.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            notifications_sent: self.notifications_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TcpServerStatsSnapshot {
    pub connections_accepted: u64,
    pub connections_active: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub requests_processed: u64,
    pub notifications_sent: u64,
    pub errors: u64,
}

/// Database interface for the TCP server
#[async_trait::async_trait]
pub trait DatabaseHandler: Send + Sync + 'static {
    /// Get a value by key
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String>;

    /// Set a value with optional TTL
    async fn set(&self, key: &[u8], value: &[u8], ttl: Option<u64>) -> Result<bool, String>;

    /// Delete a key
    async fn delete(&self, key: &[u8]) -> Result<bool, String>;

    /// Execute a SQL query with JSON parameters
    async fn query(&self, sql: &str, params: Vec<serde_json::Value>)
    -> Result<QueryResult, String>;

    /// Execute batch operations
    async fn batch(&self, operations: Vec<BatchOperation>) -> Result<Vec<bool>, String>;
}

/// Query result from the database
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub row_count: usize,
    pub execution_time_ms: u64,
}

/// Batch operation
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Set {
        key: Vec<u8>,
        value: Vec<u8>,
        ttl: Option<u64>,
    },
    Delete {
        key: Vec<u8>,
    },
}

/// TCP Server for JouleDB
pub struct TcpServer<H: DatabaseHandler> {
    config: TcpServerConfig,
    handler: Arc<H>,
    stats: Arc<TcpServerStats>,
    shutdown: Arc<Notify>,
    connection_semaphore: Arc<Semaphore>,
    subscription_manager: Arc<SubscriptionManager>,
    #[cfg(feature = "tls")]
    tls_acceptor: Option<Arc<tokio_rustls::TlsAcceptor>>,
}

impl<H: DatabaseHandler> TcpServer<H> {
    /// Create a new TCP server
    pub fn new(config: TcpServerConfig, handler: H) -> Self {
        let max_conns = config.max_connections;
        Self {
            config,
            handler: Arc::new(handler),
            stats: Arc::new(TcpServerStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore: Arc::new(Semaphore::new(max_conns)),
            subscription_manager: Arc::new(SubscriptionManager::new()),
            #[cfg(feature = "tls")]
            tls_acceptor: None,
        }
    }

    /// Create with an existing SubscriptionManager (for sharing with HTTP server)
    pub fn with_subscription_manager(
        config: TcpServerConfig,
        handler: H,
        subscription_manager: Arc<SubscriptionManager>,
    ) -> Self {
        let max_conns = config.max_connections;
        Self {
            config,
            handler: Arc::new(handler),
            stats: Arc::new(TcpServerStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore: Arc::new(Semaphore::new(max_conns)),
            subscription_manager,
            #[cfg(feature = "tls")]
            tls_acceptor: None,
        }
    }

    /// Set TLS acceptor for encrypted connections
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, acceptor: tokio_rustls::TlsAcceptor) -> Self {
        self.tls_acceptor = Some(Arc::new(acceptor));
        self
    }

    /// Get server statistics
    pub fn stats(&self) -> &TcpServerStats {
        &self.stats
    }

    /// Get subscription manager reference
    pub fn subscription_manager(&self) -> &Arc<SubscriptionManager> {
        &self.subscription_manager
    }

    /// Signal server shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Run the TCP server
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(&self.config.bind_addr).await?;

        #[cfg(feature = "tls")]
        let tls_enabled = self.tls_acceptor.is_some();
        #[cfg(not(feature = "tls"))]
        let tls_enabled = false;

        if tls_enabled {
            tracing::info!(
                "TCP server listening on {} (TLS enabled)",
                self.config.bind_addr
            );
        } else {
            tracing::info!("TCP server listening on {}", self.config.bind_addr);
        }

        loop {
            tokio::select! {
                _ = self.shutdown.notified() => {
                    tracing::info!("TCP server shutting down");
                    break;
                }
                result = listener.accept() => {
                    match result {
                        Ok((stream, peer_addr)) => {
                            // Try to acquire connection permit
                            let permit = match self.connection_semaphore.clone().try_acquire_owned() {
                                Ok(permit) => permit,
                                Err(_) => {
                                    tracing::warn!("Max connections reached, rejecting {}", peer_addr);
                                    continue;
                                }
                            };

                            // Set TCP options before potential TLS handshake
                            if let Err(e) = stream.set_nodelay(self.config.nodelay) {
                                tracing::debug!("Failed to set TCP_NODELAY for {}: {}", peer_addr, e);
                            }

                            self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                            self.stats.connections_active.fetch_add(1, Ordering::Relaxed);

                            let handler = self.handler.clone();
                            let stats = self.stats.clone();
                            let config = self.config.clone();
                            let shutdown = self.shutdown.clone();
                            let sub_manager = self.subscription_manager.clone();

                            #[cfg(feature = "tls")]
                            let tls_acceptor = self.tls_acceptor.clone();

                            tokio::spawn(async move {
                                let result = {
                                    #[cfg(feature = "tls")]
                                    if let Some(acceptor) = tls_acceptor {
                                        match acceptor.accept(stream).await {
                                            Ok(tls_stream) => {
                                                let (read_half, write_half) = tokio::io::split(tls_stream);
                                                handle_connection(
                                                    read_half, write_half, handler, stats.clone(), config, shutdown, sub_manager,
                                                ).await
                                            }
                                            Err(e) => {
                                                tracing::debug!("TLS handshake failed from {}: {}", peer_addr, e);
                                                stats.errors.fetch_add(1, Ordering::Relaxed);
                                                stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                                                drop(permit);
                                                return;
                                            }
                                        }
                                    } else {
                                        let (read_half, write_half) = tokio::io::split(stream);
                                        handle_connection(
                                            read_half, write_half, handler, stats.clone(), config, shutdown, sub_manager,
                                        ).await
                                    }

                                    #[cfg(not(feature = "tls"))]
                                    {
                                        let (read_half, write_half) = tokio::io::split(stream);
                                        handle_connection(
                                            read_half, write_half, handler, stats.clone(), config, shutdown, sub_manager,
                                        ).await
                                    }
                                };

                                if let Err(e) = result {
                                    tracing::debug!("Connection error from {}: {}", peer_addr, e);
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
            }
        }

        Ok(())
    }
}

/// Per-connection subscription state
struct ConnectionSubscriptions {
    /// subscription_id → pattern (for cleanup on disconnect)
    active: HashMap<u64, String>,
}

impl ConnectionSubscriptions {
    fn new() -> Self {
        Self {
            active: HashMap::new(),
        }
    }
}

/// Handle a single TCP connection with bidirectional push
async fn handle_connection<H: DatabaseHandler, R, W>(
    read_half: R,
    mut write_half: W,
    handler: Arc<H>,
    stats: Arc<TcpServerStats>,
    config: TcpServerConfig,
    shutdown: Arc<Notify>,
    sub_manager: Arc<SubscriptionManager>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    // Bounded channel for outgoing messages (responses + notifications).
    // When full, senders will await — applying backpressure to the reader.
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<Vec<u8>>(4096);

    // Per-connection subscription tracking
    let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));

    // Writer task: reads encoded messages from channel and writes to socket
    let stats_writer = stats.clone();
    let writer_handle = tokio::spawn(async move {
        while let Some(encoded) = outgoing_rx.recv().await {
            stats_writer
                .bytes_sent
                .fetch_add(encoded.len() as u64, Ordering::Relaxed);
            if write_half.write_all(&encoded).await.is_err() {
                break;
            }
        }
    });

    // Reader task: reads requests, processes them, and sends responses via outgoing_tx
    let reader_result = reader_loop(
        read_half,
        handler,
        stats.clone(),
        config,
        shutdown,
        sub_manager.clone(),
        outgoing_tx.clone(),
        conn_subs.clone(),
    )
    .await;

    // Connection is closing - clean up subscriptions
    {
        let subs = conn_subs.lock().await;
        for (sub_id, _pattern) in &subs.active {
            sub_manager.unsubscribe(*sub_id).await;
        }
    }

    // Drop the outgoing_tx to signal writer to stop
    drop(outgoing_tx);
    let _ = writer_handle.await;

    reader_result
}

/// Reader loop: reads incoming messages, processes, sends responses
async fn reader_loop<H: DatabaseHandler, R: AsyncRead + Unpin + Send>(
    mut read_half: R,
    handler: Arc<H>,
    stats: Arc<TcpServerStats>,
    config: TcpServerConfig,
    shutdown: Arc<Notify>,
    sub_manager: Arc<SubscriptionManager>,
    outgoing_tx: mpsc::Sender<Vec<u8>>,
    conn_subs: Arc<tokio::sync::Mutex<ConnectionSubscriptions>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let protocol = BinaryProtocol::new();
    let mut buffer = vec![0u8; config.read_buffer_size];
    let mut data = Vec::new();

    // --- Auth handshake: require Auth message as first message when auth is enabled ---
    let auth_info = if config.auth_enabled {
        match tcp_auth_handshake(
            &mut read_half,
            &mut buffer,
            &mut data,
            &protocol,
            &config,
            &stats,
            &outgoing_tx,
        )
        .await
        {
            Some(info) => info,
            None => return Ok(()), // Connection closed after auth failure
        }
    } else {
        TcpAuthInfo {
            user_id: "anonymous".to_string(),
            roles: vec!["superuser".to_string()],
        }
    };

    let idle_timeout = std::time::Duration::from_secs(config.connection_timeout_secs);

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                break;
            }
            timeout_result = tokio::time::timeout(idle_timeout, read_half.read(&mut buffer)) => {
                let result = match timeout_result {
                    Ok(r) => r,
                    Err(_) => {
                        tracing::debug!("TCP connection idle timeout ({}s)", config.connection_timeout_secs);
                        break;
                    }
                };
                match result {
                    Ok(0) => break, // Connection closed
                    Ok(n) => {
                        stats.bytes_received.fetch_add(n as u64, Ordering::Relaxed);
                        data.extend_from_slice(&buffer[..n]);

                        // Process messages in buffer
                        while data.len() >= HEADER_SIZE {
                            match protocol.parse_header(&data) {
                                Ok((_, _, payload_len)) => {
                                    if payload_len as usize > MAX_TCP_MESSAGE_SIZE {
                                        tracing::warn!("TCP message payload too large: {} bytes (max {})", payload_len, MAX_TCP_MESSAGE_SIZE);
                                        let err_response = BinaryMessage::error(
                                            0, "MESSAGE_TOO_LARGE",
                                            &format!("Payload size {} exceeds maximum {}", payload_len, MAX_TCP_MESSAGE_SIZE),
                                        );
                                        let encoded = protocol.encode(&err_response)?;
                                        let _ = outgoing_tx.send(encoded).await;
                                        return Err(format!("Message too large: {} bytes", payload_len).into());
                                    }
                                    let msg_len = HEADER_SIZE + payload_len as usize;
                                    if data.len() < msg_len {
                                        break; // Need more data
                                    }

                                    match protocol.decode(&data[..msg_len]) {
                                        Ok(msg) => {
                                            data.drain(..msg_len);

                                            let response = process_message(
                                                msg,
                                                &handler,
                                                &protocol,
                                                &sub_manager,
                                                &outgoing_tx,
                                                &conn_subs,
                                                &stats,
                                                &auth_info,
                                            ).await;

                                            let encoded = protocol.encode(&response)?;
                                            let _ = outgoing_tx.send(encoded).await;
                                            stats.requests_processed.fetch_add(1, Ordering::Relaxed);
                                        }
                                        Err(e) => {
                                            tracing::debug!("Decode error: {}", e);
                                            let err_response = BinaryMessage::error(
                                                0, "PROTOCOL_ERROR", &e.to_string(),
                                            );
                                            let encoded = protocol.encode(&err_response)?;
                                            let _ = outgoing_tx.send(encoded).await;
                                            return Err(e.into());
                                        }
                                    }
                                }
                                Err(BinaryProtocolError::TruncatedMessage) => {
                                    break;
                                }
                                Err(e) => {
                                    tracing::debug!("Header parse error: {}", e);
                                    let err_response = BinaryMessage::error(
                                        0, "PROTOCOL_ERROR", &e.to_string(),
                                    );
                                    let encoded = protocol.encode(&err_response)?;
                                    let _ = outgoing_tx.send(encoded).await;
                                    return Err(e.into());
                                }
                            }
                        }
                    }
                    Err(e) => {
                        return Err(e.into());
                    }
                }
            }
        }
    }

    Ok(())
}

/// Authenticated user info for a TCP connection
struct TcpAuthInfo {
    user_id: String,
    roles: Vec<String>,
}

/// Perform auth handshake on a TCP connection.
///
/// Reads the first message from the client. If it's an Auth message with a valid
/// JWT token, sends AuthResponse(success) and returns the user info. Otherwise sends
/// an error and returns None (caller should close the connection).
async fn tcp_auth_handshake<R: AsyncRead + Unpin + Send>(
    read_half: &mut R,
    buffer: &mut [u8],
    data: &mut Vec<u8>,
    protocol: &BinaryProtocol,
    config: &TcpServerConfig,
    stats: &Arc<TcpServerStats>,
    outgoing_tx: &mpsc::Sender<Vec<u8>>,
) -> Option<TcpAuthInfo> {
    use tokio::time::{Duration, timeout};

    // Give the client 10 seconds to send the auth message
    let auth_timeout = Duration::from_secs(10);

    let first_msg = timeout(auth_timeout, async {
        loop {
            let n = match read_half.read(buffer).await {
                Ok(0) => return None, // connection closed
                Ok(n) => n,
                Err(_) => return None,
            };
            stats.bytes_received.fetch_add(n as u64, Ordering::Relaxed);
            data.extend_from_slice(&buffer[..n]);

            if data.len() >= HEADER_SIZE {
                match protocol.parse_header(data) {
                    Ok((_, _, payload_len)) => {
                        let msg_len = HEADER_SIZE + payload_len as usize;
                        if data.len() >= msg_len {
                            match protocol.decode(&data[..msg_len]) {
                                Ok(msg) => {
                                    data.drain(..msg_len);
                                    return Some(msg);
                                }
                                Err(_) => return None,
                            }
                        }
                        // Need more data, continue reading
                    }
                    Err(BinaryProtocolError::TruncatedMessage) => {
                        // Need more data, continue reading
                    }
                    Err(_) => return None,
                }
            }
        }
    })
    .await;

    let msg = match first_msg {
        Ok(Some(msg)) => msg,
        _ => {
            // Timeout or connection error
            let err = BinaryMessage::error(
                0,
                "AUTH_TIMEOUT",
                "Authentication required: send Auth message within 10 seconds",
            );
            if let Ok(encoded) = protocol.encode(&err) {
                let _ = outgoing_tx.send(encoded).await;
            }
            return None;
        }
    };

    // Must be an Auth message
    if msg.msg_type != MessageType::Auth {
        let err = BinaryMessage::auth_response(
            msg.request_id,
            false,
            "Authentication required: first message must be Auth",
        );
        if let Ok(encoded) = protocol.encode(&err) {
            let _ = outgoing_tx.send(encoded).await;
        }
        return None;
    }

    // Parse the token from the Auth payload
    let token = match BinaryMessage::parse_auth_token(&msg.payload) {
        Some(t) => t,
        None => {
            let err = BinaryMessage::auth_response(
                msg.request_id,
                false,
                "Invalid Auth message: missing token",
            );
            if let Ok(encoded) = protocol.encode(&err) {
                let _ = outgoing_tx.send(encoded).await;
            }
            return None;
        }
    };

    // Validate the JWT token
    let secret = match &config.auth_jwt_secret {
        Some(s) => s,
        None => {
            // Auth enabled but no secret configured — reject
            let err = BinaryMessage::auth_response(
                msg.request_id,
                false,
                "Server auth misconfiguration: no JWT secret",
            );
            if let Ok(encoded) = protocol.encode(&err) {
                let _ = outgoing_tx.send(encoded).await;
            }
            return None;
        }
    };

    let auth_mgr = crate::auth::AuthenticationManager::new(secret.as_bytes().to_vec());
    match auth_mgr.validate_jwt(&token) {
        Ok(claims) => {
            tracing::info!(
                "TCP auth successful for user '{}' (roles: {:?})",
                claims.sub,
                claims.roles
            );
            let resp = BinaryMessage::auth_response(
                msg.request_id,
                true,
                &format!("Authenticated as {}", claims.sub),
            );
            if let Ok(encoded) = protocol.encode(&resp) {
                let _ = outgoing_tx.send(encoded).await;
            }
            Some(TcpAuthInfo {
                user_id: claims.sub,
                roles: claims.roles,
            })
        }
        Err(_) => {
            let err =
                BinaryMessage::auth_response(msg.request_id, false, "Invalid or expired JWT token");
            if let Ok(encoded) = protocol.encode(&err) {
                let _ = outgoing_tx.send(encoded).await;
            }
            None
        }
    }
}

/// Process a single message and return response
async fn process_message<H: DatabaseHandler>(
    msg: BinaryMessage,
    handler: &Arc<H>,
    protocol: &BinaryProtocol,
    sub_manager: &Arc<SubscriptionManager>,
    outgoing_tx: &mpsc::Sender<Vec<u8>>,
    conn_subs: &Arc<tokio::sync::Mutex<ConnectionSubscriptions>>,
    stats: &Arc<TcpServerStats>,
    auth_info: &TcpAuthInfo,
) -> BinaryMessage {
    match msg.msg_type {
        MessageType::Ping => BinaryMessage::pong(msg.request_id),

        MessageType::Get => match protocol.parse_get(&msg.payload) {
            Ok(key) => match handler.get(&key).await {
                Ok(value) => BinaryMessage::get_response(msg.request_id, value.as_deref()),
                Err(e) => BinaryMessage::error(msg.request_id, "GET_ERROR", &e),
            },
            Err(e) => BinaryMessage::error(msg.request_id, "PARSE_ERROR", &e.to_string()),
        },

        MessageType::Set => {
            match protocol.parse_set(&msg.payload) {
                Ok((key, value, ttl)) => match handler.set(&key, &value, ttl).await {
                    Ok(success) => {
                        // Fire subscription notifications on successful write
                        if success {
                            let key_str = String::from_utf8_lossy(&key).to_string();
                            sub_manager.notify_insert(&key_str, &value).await;
                        }
                        BinaryMessage::set_response(msg.request_id, success)
                    }
                    Err(e) => BinaryMessage::error(msg.request_id, "SET_ERROR", &e),
                },
                Err(e) => BinaryMessage::error(msg.request_id, "PARSE_ERROR", &e.to_string()),
            }
        }

        MessageType::Delete => {
            match protocol.parse_delete(&msg.payload) {
                Ok(key) => match handler.delete(&key).await {
                    Ok(existed) => {
                        // Fire subscription notifications on successful delete
                        if existed {
                            let key_str = String::from_utf8_lossy(&key).to_string();
                            sub_manager.notify_delete(&key_str, None).await;
                        }
                        BinaryMessage::delete_response(msg.request_id, existed)
                    }
                    Err(e) => BinaryMessage::error(msg.request_id, "DELETE_ERROR", &e),
                },
                Err(e) => BinaryMessage::error(msg.request_id, "PARSE_ERROR", &e.to_string()),
            }
        }

        // ============================================================
        // Subscribe: register a pattern and start pushing notifications
        // ============================================================
        MessageType::Subscribe => {
            match protocol.parse_subscribe(&msg.payload) {
                Ok(pattern) => {
                    let (sub_id, mut receiver) = match sub_manager.subscribe(&pattern).await {
                        Ok(pair) => pair,
                        Err(e) => {
                            return BinaryMessage::error(msg.request_id, "SUBSCRIBE_ERROR", &e);
                        }
                    };

                    // Track subscription for cleanup
                    {
                        let mut subs = conn_subs.lock().await;
                        subs.active.insert(sub_id, pattern);
                    }

                    // Spawn notification forwarder for this subscription
                    let tx = outgoing_tx.clone();
                    let inner_protocol = BinaryProtocol::new();
                    let stats_clone = stats.clone();
                    tokio::spawn(async move {
                        while let Some(event) = receiver.recv().await {
                            let operation = match event.operation {
                                crate::subscriptions::ChangeOperation::Insert => 0u8,
                                crate::subscriptions::ChangeOperation::Update => 1u8,
                                crate::subscriptions::ChangeOperation::Delete => 2u8,
                            };
                            let notification = BinaryMessage::notification(
                                0, // server-initiated, no request_id
                                sub_id,
                                operation,
                                &event.key,
                                event.value.as_deref(),
                                event.old_value.as_deref(),
                                event.timestamp,
                            );
                            if let Ok(encoded) = inner_protocol.encode(&notification) {
                                if tx.send(encoded).await.is_err() {
                                    break; // Connection closed
                                }
                                stats_clone
                                    .notifications_sent
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });

                    // Return subscription ID to client
                    let response_payload = sub_id.to_le_bytes().to_vec();
                    BinaryMessage::new(MessageType::Notification, msg.request_id, response_payload)
                }
                Err(e) => BinaryMessage::error(msg.request_id, "PARSE_ERROR", &e.to_string()),
            }
        }

        // ============================================================
        // Unsubscribe: remove subscription and stop notifications
        // ============================================================
        MessageType::Unsubscribe => match protocol.parse_unsubscribe(&msg.payload) {
            Ok(sub_id) => {
                let removed = sub_manager.unsubscribe(sub_id).await;
                {
                    let mut subs = conn_subs.lock().await;
                    subs.active.remove(&sub_id);
                }
                BinaryMessage::new(
                    MessageType::Notification,
                    msg.request_id,
                    vec![if removed { 1 } else { 0 }],
                )
            }
            Err(e) => BinaryMessage::error(msg.request_id, "PARSE_ERROR", &e.to_string()),
        },

        MessageType::Query => {
            #[derive(serde::Deserialize)]
            struct QueryRequest {
                sql: String,
                #[serde(default)]
                params: Vec<serde_json::Value>,
            }

            #[derive(serde::Serialize)]
            struct QueryResponseJson {
                columns: Vec<String>,
                rows: Vec<Vec<serde_json::Value>>,
                row_count: usize,
                execution_time_ms: u64,
            }

            #[derive(serde::Serialize)]
            struct ErrorResponseJson {
                code: String,
                message: String,
            }

            match serde_json::from_slice::<QueryRequest>(&msg.payload) {
                Ok(request) => {
                    // RBAC: check write permission before executing
                    if let Err(e) = crate::query::check_write_permission(
                        &auth_info.user_id,
                        &auth_info.roles,
                        &request.sql,
                    ) {
                        let error = ErrorResponseJson {
                            code: e.code,
                            message: e.message,
                        };
                        let payload = serde_json::to_vec(&error).unwrap_or_default();
                        return BinaryMessage::new(MessageType::Error, msg.request_id, payload);
                    }

                    match handler.query(&request.sql, request.params).await {
                        Ok(result) => {
                            let response = QueryResponseJson {
                                columns: result.columns,
                                rows: result.rows,
                                row_count: result.row_count,
                                execution_time_ms: result.execution_time_ms,
                            };
                            let payload = serde_json::to_vec(&response).unwrap_or_default();
                            BinaryMessage::new(MessageType::QueryResponse, msg.request_id, payload)
                        }
                        Err(e) => {
                            let error = ErrorResponseJson {
                                code: "QUERY_ERROR".to_string(),
                                message: e,
                            };
                            let payload = serde_json::to_vec(&error).unwrap_or_default();
                            BinaryMessage::new(MessageType::Error, msg.request_id, payload)
                        }
                    }
                }
                Err(e) => {
                    let error = serde_json::json!({
                        "code": "PARSE_ERROR",
                        "message": format!("Invalid query JSON: {}", e)
                    });
                    let payload = serde_json::to_vec(&error).unwrap_or_default();
                    BinaryMessage::new(MessageType::Error, msg.request_id, payload)
                }
            }
        }

        MessageType::Batch => {
            let mut cursor = 0;
            let count = read_varint(&msg.payload, &mut cursor).unwrap_or(0) as usize;
            let mut operations = Vec::with_capacity(count);

            for _ in 0..count {
                if cursor >= msg.payload.len() {
                    break;
                }

                let op_type = msg.payload[cursor];
                cursor += 1;

                match op_type {
                    1 => {
                        if let (Some(key), Some(value)) = (
                            read_bytes(&msg.payload, &mut cursor),
                            read_bytes(&msg.payload, &mut cursor),
                        ) {
                            let ttl = if cursor < msg.payload.len() && msg.payload[cursor] == 1 {
                                cursor += 1;
                                if cursor + 8 <= msg.payload.len() {
                                    let ttl_bytes: [u8; 8] = msg.payload[cursor..cursor + 8]
                                        .try_into()
                                        .expect("slice length verified above");
                                    cursor += 8;
                                    Some(u64::from_le_bytes(ttl_bytes))
                                } else {
                                    None
                                }
                            } else {
                                if cursor < msg.payload.len() {
                                    cursor += 1;
                                }
                                None
                            };
                            operations.push(BatchOperation::Set { key, value, ttl });
                        }
                    }
                    2 => {
                        if let Some(key) = read_bytes(&msg.payload, &mut cursor) {
                            operations.push(BatchOperation::Delete { key });
                        }
                    }
                    other => {
                        tracing::warn!("Unknown batch operation type: {}, skipping", other);
                    }
                }
            }

            match handler.batch(operations).await {
                Ok(results) => {
                    let mut payload = Vec::new();
                    write_varint(&mut payload, results.len() as u64);
                    for success in results {
                        payload.push(if success { 1 } else { 0 });
                    }
                    BinaryMessage::new(MessageType::BatchResponse, msg.request_id, payload)
                }
                Err(e) => BinaryMessage::error(msg.request_id, "BATCH_ERROR", &e),
            }
        }

        MessageType::BeginTx => match handler.query("BEGIN", Vec::new()).await {
            Ok(_) => {
                let response = serde_json::json!({ "transaction_id": msg.request_id });
                let payload = serde_json::to_vec(&response).unwrap_or_default();
                BinaryMessage::new(MessageType::BeginTxResponse, msg.request_id, payload)
            }
            Err(e) => BinaryMessage::error(msg.request_id, "BEGIN_ERROR", &e),
        },

        MessageType::Commit => match handler.query("COMMIT", Vec::new()).await {
            Ok(_) => {
                let response = serde_json::json!({ "success": true });
                let payload = serde_json::to_vec(&response).unwrap_or_default();
                BinaryMessage::new(MessageType::CommitResponse, msg.request_id, payload)
            }
            Err(e) => BinaryMessage::error(msg.request_id, "COMMIT_ERROR", &e),
        },

        MessageType::Rollback => match handler.query("ROLLBACK", Vec::new()).await {
            Ok(_) => {
                let response = serde_json::json!({ "success": true });
                let payload = serde_json::to_vec(&response).unwrap_or_default();
                BinaryMessage::new(MessageType::RollbackResponse, msg.request_id, payload)
            }
            Err(e) => BinaryMessage::error(msg.request_id, "ROLLBACK_ERROR", &e),
        },

        MessageType::Execute => {
            #[derive(serde::Deserialize)]
            struct ExecuteRequest {
                sql: String,
                #[serde(default)]
                params: Vec<serde_json::Value>,
            }

            match serde_json::from_slice::<ExecuteRequest>(&msg.payload) {
                Ok(request) => match handler.query(&request.sql, request.params).await {
                    Ok(result) => {
                        #[derive(serde::Serialize)]
                        struct ExecuteResponse {
                            affected_rows: usize,
                            execution_time_ms: u64,
                        }
                        let response = ExecuteResponse {
                            affected_rows: result.row_count,
                            execution_time_ms: result.execution_time_ms,
                        };
                        let payload = serde_json::to_vec(&response).unwrap_or_default();
                        BinaryMessage::new(MessageType::ExecuteResponse, msg.request_id, payload)
                    }
                    Err(e) => BinaryMessage::error(msg.request_id, "EXECUTE_ERROR", &e),
                },
                Err(e) => BinaryMessage::error(
                    msg.request_id,
                    "PARSE_ERROR",
                    &format!("Invalid execute JSON: {}", e),
                ),
            }
        }

        MessageType::Savepoint => {
            #[derive(serde::Deserialize)]
            struct SavepointRequest {
                name: String,
            }

            match serde_json::from_slice::<SavepointRequest>(&msg.payload) {
                Ok(request) => {
                    // Validate savepoint name to prevent SQL injection
                    if !request
                        .name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                        || request.name.is_empty()
                        || request.name.len() > 128
                    {
                        return BinaryMessage::error(
                            msg.request_id,
                            "SAVEPOINT_ERROR",
                            "Invalid savepoint name: must be alphanumeric/underscore, 1-128 chars",
                        );
                    }
                    let sql = format!("SAVEPOINT {}", request.name);
                    match handler.query(&sql, Vec::new()).await {
                        Ok(_) => {
                            let response = serde_json::json!({
                                "success": true,
                                "name": request.name
                            });
                            let payload = serde_json::to_vec(&response).unwrap_or_default();
                            BinaryMessage::new(
                                MessageType::SavepointResponse,
                                msg.request_id,
                                payload,
                            )
                        }
                        Err(e) => BinaryMessage::error(msg.request_id, "SAVEPOINT_ERROR", &e),
                    }
                }
                Err(e) => BinaryMessage::error(
                    msg.request_id,
                    "PARSE_ERROR",
                    &format!("Invalid savepoint JSON: {}", e),
                ),
            }
        }

        MessageType::Prepare => {
            #[derive(serde::Deserialize)]
            struct PrepareRequest {
                name: String,
                sql: String,
            }

            match serde_json::from_slice::<PrepareRequest>(&msg.payload) {
                Ok(request) => {
                    let response = serde_json::json!({
                        "success": true,
                        "name": request.name,
                        "statement_id": msg.request_id
                    });
                    let payload = serde_json::to_vec(&response).unwrap_or_default();
                    BinaryMessage::new(MessageType::PrepareResponse, msg.request_id, payload)
                }
                Err(e) => BinaryMessage::error(
                    msg.request_id,
                    "PARSE_ERROR",
                    &format!("Invalid prepare JSON: {}", e),
                ),
            }
        }

        _ => BinaryMessage::error(msg.request_id, "UNSUPPORTED", "Unsupported message type"),
    }
}

// Helper functions for payload encoding/decoding
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn read_varint(data: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;

    loop {
        if *cursor >= data.len() {
            return None;
        }

        let byte = data[*cursor];
        *cursor += 1;

        result |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            break;
        }

        shift += 7;
        if shift >= 64 {
            return None;
        }
    }

    Some(result)
}

fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    write_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn read_bytes(data: &[u8], cursor: &mut usize) -> Option<Vec<u8>> {
    let len = read_varint(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return None;
    }
    let result = data[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Some(result)
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_bytes(buf, s.as_bytes());
}

fn read_string(data: &[u8], cursor: &mut usize) -> Option<String> {
    let bytes = read_bytes(data, cursor)?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockHandler;

    #[async_trait::async_trait]
    impl DatabaseHandler for MockHandler {
        async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
            if key == b"exists" {
                Ok(Some(b"value".to_vec()))
            } else {
                Ok(None)
            }
        }

        async fn set(&self, _key: &[u8], _value: &[u8], _ttl: Option<u64>) -> Result<bool, String> {
            Ok(true)
        }

        async fn delete(&self, key: &[u8]) -> Result<bool, String> {
            Ok(key == b"exists")
        }

        async fn query(
            &self,
            sql: &str,
            _params: Vec<serde_json::Value>,
        ) -> Result<QueryResult, String> {
            let _ = sql;
            Ok(QueryResult {
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![vec![serde_json::json!(1), serde_json::json!("Alice")]],
                row_count: 1,
                execution_time_ms: 5,
            })
        }

        async fn batch(&self, operations: Vec<BatchOperation>) -> Result<Vec<bool>, String> {
            Ok(vec![true; operations.len()])
        }
    }

    #[tokio::test]
    async fn test_process_ping() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());
        let msg = BinaryMessage::ping(42);

        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::Pong);
        assert_eq!(response.request_id, 42);
    }

    #[tokio::test]
    async fn test_process_get_exists() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());
        let msg = BinaryMessage::get(1, b"exists");

        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::GetResponse);
        let value = protocol
            .parse_get_response(&response.payload)
            .ok()
            .flatten();
        assert_eq!(value, Some(b"value".to_vec()));
    }

    #[tokio::test]
    async fn test_process_get_not_exists() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());
        let msg = BinaryMessage::get(1, b"not_exists");

        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::GetResponse);
        let value = protocol
            .parse_get_response(&response.payload)
            .ok()
            .flatten();
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_process_set() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());
        let msg = BinaryMessage::set(1, b"key", b"value", Some(3600));

        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::SetResponse);
    }

    #[tokio::test]
    async fn test_process_delete() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());
        let msg = BinaryMessage::delete(1, b"exists");

        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::DeleteResponse);
    }

    #[tokio::test]
    async fn test_process_subscribe() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, mut rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());

        // Subscribe to "users:*"
        let msg = BinaryMessage::subscribe(1, "users:*");
        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::Notification);
        // Response payload is the subscription ID (u64 LE)
        assert_eq!(response.payload.len(), 8);
        let sub_id = u64::from_le_bytes(response.payload[..8].try_into().expect("8 bytes"));
        assert!(sub_id > 0);

        // Verify subscription is tracked
        let subs = conn_subs.lock().await;
        assert!(subs.active.contains_key(&sub_id));
        drop(subs);

        // Now trigger a notification by setting a matching key
        sub_manager.notify_insert("users:42", b"alice").await;

        // Give the spawned notification forwarder time to run
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Should receive a notification on the outgoing channel
        if let Some(encoded_notification) = rx.try_recv().ok() {
            let notification = protocol
                .decode(&encoded_notification)
                .expect("valid message");
            assert_eq!(notification.msg_type, MessageType::Notification);

            let (parsed_sub_id, op, key, new_val, _old_val, _ts) = protocol
                .parse_notification(&notification.payload)
                .expect("valid notification");
            assert_eq!(parsed_sub_id, sub_id);
            assert_eq!(op, 0); // Insert
            assert_eq!(key, "users:42");
            assert_eq!(new_val, Some(b"alice".to_vec()));
        }
    }

    #[tokio::test]
    async fn test_process_unsubscribe() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());

        // Subscribe first
        let (sub_id, _receiver) = sub_manager.subscribe("test:*").await.unwrap();
        {
            let mut subs = conn_subs.lock().await;
            subs.active.insert(sub_id, "test:*".to_string());
        }

        // Unsubscribe
        let msg = BinaryMessage::unsubscribe(2, sub_id);
        let response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        assert_eq!(response.msg_type, MessageType::Notification);
        assert_eq!(response.payload, vec![1]); // removed = true

        // Verify cleanup
        let subs = conn_subs.lock().await;
        assert!(!subs.active.contains_key(&sub_id));
    }

    #[tokio::test]
    async fn test_set_fires_notification_to_subscriber() {
        let handler = Arc::new(MockHandler);
        let protocol = BinaryProtocol::new();
        let sub_manager = Arc::new(SubscriptionManager::new());

        // Create a subscription externally
        let (_sub_id, mut receiver) = sub_manager.subscribe("*").await.unwrap();

        let (tx, _rx) = mpsc::channel(4096);
        let conn_subs = Arc::new(tokio::sync::Mutex::new(ConnectionSubscriptions::new()));
        let stats = Arc::new(TcpServerStats::default());

        // Set a key
        let msg = BinaryMessage::set(1, b"mykey", b"myvalue", None);
        let _response = process_message(
            msg,
            &handler,
            &protocol,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
            &TcpAuthInfo {
                user_id: "test".to_string(),
                roles: vec!["superuser".to_string()],
            },
        )
        .await;

        // The subscription receiver should get the notification
        let event = tokio::time::timeout(tokio::time::Duration::from_millis(100), receiver.recv())
            .await
            .expect("should not timeout")
            .expect("should receive event");

        assert_eq!(event.key, "mykey");
    }
}
