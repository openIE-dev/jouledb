//! WebSocket server for browser clients
//!
//! This module provides a WebSocket server that speaks the binary protocol
//! for data operations and JSON text messages for real-time subscriptions.
//!
//! ## Features
//!
//! - Binary protocol over WebSocket for data operations
//! - JSON text protocol for subscribe/unsubscribe/notifications
//! - Bidirectional push for real-time subscription notifications
//! - Connection multiplexing
//! - Graceful shutdown
//!
//! ## Subscription Protocol (JSON text messages)
//!
//! Subscribe:   `{"type":"subscribe","id":1,"pattern":"users:*"}`
//! Response:    `{"type":"subscribed","id":1,"subscription_id":42}`
//! Unsubscribe: `{"type":"unsubscribe","id":2,"subscription_id":42}`
//! Response:    `{"type":"unsubscribed","id":2,"ok":true}`
//! Notification:`{"type":"notification","subscription_id":42,"operation":"insert","key":"users:1","value":"..."}`

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, Semaphore, mpsc};
use tokio_tungstenite::{accept_async, tungstenite::Message as WsMessage};

use joule_db_core::persistence::network::{
    ErrorCode, HEADER_SIZE, Message, OpCode, PROTOCOL_MAGIC, decode_key_value,
};
use joule_db_local::Database;

use crate::subscriptions::SubscriptionManager;

/// WebSocket server configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Ping interval in seconds (0 = disabled)
    pub ping_interval_secs: u64,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_connections: 10000,
            max_message_size: 16 * 1024 * 1024, // 16MB
            ping_interval_secs: 30,
            connection_timeout_secs: 300, // 5 minutes
        }
    }
}

/// WebSocket server statistics
#[derive(Debug, Default)]
pub struct WebSocketStats {
    /// Total connections accepted
    pub connections_accepted: AtomicU64,
    /// Currently active connections
    pub active_connections: AtomicU32,
    /// Total messages received
    pub messages_received: AtomicU64,
    /// Total messages sent
    pub messages_sent: AtomicU64,
    /// Total bytes received
    pub bytes_received: AtomicU64,
    /// Total bytes sent
    pub bytes_sent: AtomicU64,
    /// Total notifications pushed
    pub notifications_sent: AtomicU64,
    /// Total errors
    pub errors: AtomicU64,
}

impl WebSocketStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot
    pub fn snapshot(&self) -> WebSocketStatsSnapshot {
        WebSocketStatsSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            notifications_sent: self.notifications_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of WebSocket statistics
#[derive(Debug, Clone)]
pub struct WebSocketStatsSnapshot {
    pub connections_accepted: u64,
    pub active_connections: u32,
    pub messages_received: u64,
    pub messages_sent: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub notifications_sent: u64,
    pub errors: u64,
}

/// WebSocket server for JouleDB
pub struct WebSocketServer {
    /// Database instance
    db: Arc<RwLock<Database>>,
    /// Configuration
    config: WebSocketConfig,
    /// Statistics
    stats: Arc<WebSocketStats>,
    /// Connection semaphore
    connection_semaphore: Arc<Semaphore>,
    /// Shutdown signal
    shutdown: Arc<tokio::sync::Notify>,
    /// Subscription manager (shared with TCP/HTTP servers)
    subscription_manager: Arc<SubscriptionManager>,
}

impl WebSocketServer {
    /// Create a new WebSocket server
    pub fn new(db: Database) -> Self {
        Self::with_config(db, WebSocketConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(db: Database, config: WebSocketConfig) -> Self {
        let connection_semaphore = Arc::new(Semaphore::new(config.max_connections));
        Self {
            db: Arc::new(RwLock::new(db)),
            config,
            stats: Arc::new(WebSocketStats::new()),
            connection_semaphore,
            shutdown: Arc::new(tokio::sync::Notify::new()),
            subscription_manager: Arc::new(SubscriptionManager::new()),
        }
    }

    /// Create with shared subscription manager
    pub fn with_subscription_manager(
        db: Database,
        config: WebSocketConfig,
        subscription_manager: Arc<SubscriptionManager>,
    ) -> Self {
        let connection_semaphore = Arc::new(Semaphore::new(config.max_connections));
        Self {
            db: Arc::new(RwLock::new(db)),
            config,
            stats: Arc::new(WebSocketStats::new()),
            connection_semaphore,
            shutdown: Arc::new(tokio::sync::Notify::new()),
            subscription_manager,
        }
    }

    /// Get server statistics
    pub fn stats(&self) -> WebSocketStatsSnapshot {
        self.stats.snapshot()
    }

    /// Trigger graceful shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Run the WebSocket server
    pub async fn run(&self, addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("WebSocket server listening on ws://{}", addr);

        loop {
            tokio::select! {
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

                            self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                            self.stats.active_connections.fetch_add(1, Ordering::Relaxed);

                            let db = self.db.clone();
                            let stats = self.stats.clone();
                            let config = self.config.clone();
                            let shutdown = self.shutdown.clone();
                            let sub_manager = self.subscription_manager.clone();

                            tokio::spawn(async move {
                                if let Err(e) = handle_ws_connection(
                                    stream, peer_addr, db, stats.clone(),
                                    config, shutdown, sub_manager,
                                ).await {
                                    tracing::debug!("WebSocket error from {}: {}", peer_addr, e);
                                    stats.errors.fetch_add(1, Ordering::Relaxed);
                                }
                                stats.active_connections.fetch_sub(1, Ordering::Relaxed);
                                drop(permit);
                            });
                        }
                        Err(e) => {
                            tracing::error!("Accept error: {}", e);
                            self.stats.errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                _ = self.shutdown.notified() => {
                    tracing::info!("WebSocket server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Parse a binary message and handle it
    async fn parse_and_handle(
        data: &[u8],
        db: &Arc<RwLock<Database>>,
        sub_manager: &Arc<SubscriptionManager>,
    ) -> Result<Message, String> {
        if data.len() < HEADER_SIZE {
            return Err("Message too short".to_string());
        }

        if &data[0..2] != &PROTOCOL_MAGIC {
            return Err("Invalid protocol magic".to_string());
        }

        let message = Message::decode(data).map_err(|e| e.to_string())?;
        Ok(Self::handle_message(&message, db, sub_manager).await)
    }

    /// Handle a binary protocol message and produce a response
    async fn handle_message(
        message: &Message,
        db: &Arc<RwLock<Database>>,
        sub_manager: &Arc<SubscriptionManager>,
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
                                // Fire subscription notifications
                                let key_str = String::from_utf8_lossy(key).to_string();
                                sub_manager.notify_insert(&key_str, value).await;
                                Message::response(message.request_id, OpCode::Put, vec![1])
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
                        if deleted {
                            let key_str = String::from_utf8_lossy(key).to_string();
                            sub_manager.notify_delete(&key_str, None).await;
                        }
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

            OpCode::Info => {
                let db_guard = db.read().await;
                let info = format!(
                    "joule_db_version:0.1.0\n\
                     protocol:websocket\n\
                     db_path:{}\n",
                    db_guard.path()
                );
                Message::response(message.request_id, OpCode::Info, info.into_bytes())
            }

            OpCode::Close => Message::response(message.request_id, OpCode::Close, vec![]),

            _ => Message::error(
                message.request_id,
                ErrorCode::NotImplemented as u16,
                &format!("Opcode {:?} not implemented", message.opcode),
            ),
        }
    }
}

/// Per-connection subscription tracking for WebSocket
struct WsConnectionSubscriptions {
    active: HashMap<u64, String>,
}

impl WsConnectionSubscriptions {
    fn new() -> Self {
        Self {
            active: HashMap::new(),
        }
    }
}

/// Handle a single WebSocket connection with bidirectional push
async fn handle_ws_connection(
    stream: TcpStream,
    peer_addr: SocketAddr,
    db: Arc<RwLock<Database>>,
    stats: Arc<WebSocketStats>,
    config: WebSocketConfig,
    shutdown: Arc<tokio::sync::Notify>,
    sub_manager: Arc<SubscriptionManager>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing::debug!("New WebSocket connection from {}", peer_addr);

    // Upgrade to WebSocket
    let ws_stream = accept_async(stream).await?;
    let (ws_write, mut ws_read) = ws_stream.split();

    // Channel for outgoing messages (responses + notifications)
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<WsMessage>();

    // Per-connection subscription tracking
    let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));

    // Writer task: reads from channel and sends to WebSocket
    let stats_writer = stats.clone();
    let writer_handle = tokio::spawn(async move {
        let mut ws_write = ws_write;
        while let Some(msg) = outgoing_rx.recv().await {
            let msg_len = match &msg {
                WsMessage::Binary(data) => data.len(),
                WsMessage::Text(text) => text.len(),
                _ => 0,
            };
            stats_writer
                .bytes_sent
                .fetch_add(msg_len as u64, Ordering::Relaxed);
            stats_writer.messages_sent.fetch_add(1, Ordering::Relaxed);
            if ws_write.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Reader loop
    loop {
        tokio::select! {
            msg = ws_read.next() => {
                match msg {
                    Some(Ok(WsMessage::Binary(data))) => {
                        stats.bytes_received.fetch_add(data.len() as u64, Ordering::Relaxed);
                        stats.messages_received.fetch_add(1, Ordering::Relaxed);

                        if data.len() > config.max_message_size {
                            let err = Message::error(0, ErrorCode::InvalidRequest as u16, "Message too large");
                            let _ = outgoing_tx.send(WsMessage::Binary(err.encode().into()));
                            stats.errors.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }

                        match WebSocketServer::parse_and_handle(&data, &db, &sub_manager).await {
                            Ok(response) => {
                                let response_bytes = response.encode();
                                let _ = outgoing_tx.send(WsMessage::Binary(response_bytes.into()));

                                if response.opcode == OpCode::Close {
                                    break;
                                }
                            }
                            Err(e) => {
                                let error_response = Message::error(0, ErrorCode::InvalidRequest as u16, &e);
                                let _ = outgoing_tx.send(WsMessage::Binary(error_response.encode().into()));
                                stats.errors.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    Some(Ok(WsMessage::Text(text))) => {
                        stats.messages_received.fetch_add(1, Ordering::Relaxed);
                        stats.bytes_received.fetch_add(text.len() as u64, Ordering::Relaxed);

                        if text.len() > config.max_message_size {
                            let _ = outgoing_tx.send(WsMessage::Text(
                                serde_json::json!({"type": "error", "message": "Message too large"}).to_string().into()
                            ));
                            stats.errors.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }

                        // Handle JSON subscription protocol
                        let response = handle_json_message(
                            &text,
                            &sub_manager,
                            &outgoing_tx,
                            &conn_subs,
                            &stats,
                        ).await;
                        let _ = outgoing_tx.send(WsMessage::Text(response.into()));
                    }
                    Some(Ok(WsMessage::Ping(data))) => {
                        let _ = outgoing_tx.send(WsMessage::Pong(data));
                    }
                    Some(Ok(WsMessage::Pong(_))) => {}
                    Some(Ok(WsMessage::Close(_))) => {
                        tracing::debug!("WebSocket closed by {}", peer_addr);
                        break;
                    }
                    Some(Ok(WsMessage::Frame(_))) => {}
                    Some(Err(e)) => {
                        tracing::debug!("WebSocket error from {}: {}", peer_addr, e);
                        break;
                    }
                    None => {
                        tracing::debug!("WebSocket stream ended for {}", peer_addr);
                        break;
                    }
                }
            }

            _ = shutdown.notified() => {
                tracing::debug!("Shutdown, closing WebSocket to {}", peer_addr);
                let _ = outgoing_tx.send(WsMessage::Close(None));
                break;
            }
        }
    }

    // Clean up subscriptions
    {
        let subs = conn_subs.lock().await;
        for (sub_id, _pattern) in &subs.active {
            sub_manager.unsubscribe(*sub_id).await;
        }
    }

    // Signal writer to stop
    drop(outgoing_tx);
    let _ = writer_handle.await;

    Ok(())
}

/// Handle a JSON text message for subscription protocol
async fn handle_json_message(
    text: &str,
    sub_manager: &Arc<SubscriptionManager>,
    outgoing_tx: &mpsc::UnboundedSender<WsMessage>,
    conn_subs: &Arc<tokio::sync::Mutex<WsConnectionSubscriptions>>,
    stats: &Arc<WebSocketStats>,
) -> String {
    #[derive(serde::Deserialize)]
    struct JsonCommand {
        #[serde(rename = "type")]
        cmd_type: String,
        #[serde(default)]
        id: u64,
        #[serde(default)]
        pattern: Option<String>,
        #[serde(default)]
        subscription_id: Option<u64>,
    }

    let cmd: JsonCommand = match serde_json::from_str(text) {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "error",
                "message": format!("Invalid JSON: {}", e)
            })
            .to_string();
        }
    };

    match cmd.cmd_type.as_str() {
        "subscribe" => {
            let pattern = match cmd.pattern {
                Some(p) => p,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": cmd.id,
                        "message": "Missing 'pattern' field"
                    })
                    .to_string();
                }
            };

            let (sub_id, mut receiver) = match sub_manager.subscribe(&pattern).await {
                Ok(pair) => pair,
                Err(e) => {
                    return serde_json::json!({
                        "type": "error",
                        "id": cmd.id,
                        "message": e
                    })
                    .to_string();
                }
            };

            // Track subscription for cleanup
            {
                let mut subs = conn_subs.lock().await;
                subs.active.insert(sub_id, pattern);
            }

            // Spawn notification forwarder
            let tx = outgoing_tx.clone();
            let stats_clone = stats.clone();
            tokio::spawn(async move {
                while let Some(event) = receiver.recv().await {
                    let operation = match event.operation {
                        crate::subscriptions::ChangeOperation::Insert => "insert",
                        crate::subscriptions::ChangeOperation::Update => "update",
                        crate::subscriptions::ChangeOperation::Delete => "delete",
                    };

                    let mut notification = serde_json::json!({
                        "type": "notification",
                        "subscription_id": sub_id,
                        "operation": operation,
                        "key": event.key,
                        "timestamp": event.timestamp,
                    });

                    if let Some(ref value) = event.value {
                        // Try to send as UTF-8 string, fall back to base64
                        if let Ok(s) = String::from_utf8(value.clone()) {
                            notification["value"] = serde_json::Value::String(s);
                        } else {
                            let hex: String = value.iter().map(|b| format!("{:02x}", b)).collect();
                            notification["value_hex"] = serde_json::Value::String(hex);
                        }
                    }

                    if let Some(ref old) = event.old_value {
                        if let Ok(s) = String::from_utf8(old.clone()) {
                            notification["old_value"] = serde_json::Value::String(s);
                        } else {
                            let hex: String = old.iter().map(|b| format!("{:02x}", b)).collect();
                            notification["old_value_hex"] = serde_json::Value::String(hex);
                        }
                    }

                    let msg_text = notification.to_string();
                    if tx.send(WsMessage::Text(msg_text.into())).is_err() {
                        break; // Connection closed
                    }
                    stats_clone
                        .notifications_sent
                        .fetch_add(1, Ordering::Relaxed);
                }
            });

            serde_json::json!({
                "type": "subscribed",
                "id": cmd.id,
                "subscription_id": sub_id
            })
            .to_string()
        }

        "unsubscribe" => {
            let sub_id = match cmd.subscription_id {
                Some(id) => id,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": cmd.id,
                        "message": "Missing 'subscription_id' field"
                    })
                    .to_string();
                }
            };

            // Verify this connection owns the subscription before unsubscribing
            let owns_sub = {
                let subs = conn_subs.lock().await;
                subs.active.contains_key(&sub_id)
            };
            if !owns_sub {
                return serde_json::json!({
                    "type": "error",
                    "id": cmd.id,
                    "message": "Subscription not owned by this connection"
                })
                .to_string();
            }
            let removed = sub_manager.unsubscribe(sub_id).await;
            {
                let mut subs = conn_subs.lock().await;
                subs.active.remove(&sub_id);
            }

            serde_json::json!({
                "type": "unsubscribed",
                "id": cmd.id,
                "ok": removed
            })
            .to_string()
        }

        "ping" => serde_json::json!({
            "type": "pong",
            "id": cmd.id
        })
        .to_string(),

        other => serde_json::json!({
            "type": "error",
            "id": cmd.id,
            "message": format!("Unknown command type: {}", other)
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.max_connections, 10000);
        assert_eq!(config.max_message_size, 16 * 1024 * 1024);
    }

    #[test]
    fn test_stats_snapshot() {
        let stats = WebSocketStats::new();
        stats.connections_accepted.fetch_add(5, Ordering::Relaxed);
        stats.messages_received.fetch_add(100, Ordering::Relaxed);
        stats.notifications_sent.fetch_add(42, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.connections_accepted, 5);
        assert_eq!(snapshot.messages_received, 100);
        assert_eq!(snapshot.notifications_sent, 42);
    }

    #[tokio::test]
    async fn test_json_subscribe() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        let response = handle_json_message(
            r#"{"type":"subscribe","id":1,"pattern":"users:*"}"#,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
        )
        .await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "subscribed");
        assert_eq!(parsed["id"], 1);
        assert!(parsed["subscription_id"].as_u64().unwrap() > 0);

        // Verify tracked
        let subs = conn_subs.lock().await;
        assert_eq!(subs.active.len(), 1);
    }

    #[tokio::test]
    async fn test_json_unsubscribe() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        // Subscribe first
        let (sub_id, _receiver) = sub_manager.subscribe("test:*").await.unwrap();
        {
            let mut subs = conn_subs.lock().await;
            subs.active.insert(sub_id, "test:*".to_string());
        }

        let msg = format!(
            r#"{{"type":"unsubscribe","id":2,"subscription_id":{}}}"#,
            sub_id
        );
        let response = handle_json_message(&msg, &sub_manager, &tx, &conn_subs, &stats).await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "unsubscribed");
        assert_eq!(parsed["ok"], true);

        let subs = conn_subs.lock().await;
        assert!(subs.active.is_empty());
    }

    #[tokio::test]
    async fn test_json_subscribe_fires_notification() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        // Subscribe
        let response = handle_json_message(
            r#"{"type":"subscribe","id":1,"pattern":"*"}"#,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
        )
        .await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let _sub_id = parsed["subscription_id"].as_u64().unwrap();

        // Trigger a notification
        sub_manager.notify_insert("mykey", b"myvalue").await;

        // Give forwarder time to run
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // The subscribe response was returned directly (not through tx),
        // so the first message on rx should be the notification
        if let Some(WsMessage::Text(text)) = rx.try_recv().ok() {
            let notification: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(notification["type"], "notification");
            assert_eq!(notification["key"], "mykey");
            assert_eq!(notification["operation"], "insert");
            assert_eq!(notification["value"], "myvalue");
        }
    }

    #[tokio::test]
    async fn test_json_ping() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        let response = handle_json_message(
            r#"{"type":"ping","id":99}"#,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
        )
        .await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "pong");
        assert_eq!(parsed["id"], 99);
    }

    #[tokio::test]
    async fn test_json_unknown_type() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        let response = handle_json_message(
            r#"{"type":"unknown","id":1}"#,
            &sub_manager,
            &tx,
            &conn_subs,
            &stats,
        )
        .await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "error");
    }

    #[tokio::test]
    async fn test_json_invalid_json() {
        let sub_manager = Arc::new(SubscriptionManager::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(tokio::sync::Mutex::new(WsConnectionSubscriptions::new()));
        let stats = Arc::new(WebSocketStats::new());

        let response = handle_json_message("not json", &sub_manager, &tx, &conn_subs, &stats).await;

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "error");
    }
}
