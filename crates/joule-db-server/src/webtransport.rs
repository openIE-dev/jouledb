//! WebTransport server for ultra-low-latency real-time communication
//!
//! Provides an HTTP/3 + QUIC based WebTransport server that supports:
//! - Bidirectional streams for reliable JSON subscription protocol
//! - Datagrams for fire-and-forget low-latency notifications
//! - Same JSON protocol as WebSocket (`subscribe`, `unsubscribe`, `notification`)
//!
//! ## Why WebTransport?
//!
//! - No head-of-line blocking (QUIC multiplexed streams)
//! - Lower latency than WebSocket (0-RTT connection establishment)
//! - Unreliable datagrams for real-time game state (no retransmission delay)
//! - Native browser support (Chrome, Edge)
//!
//! ## Subscription Protocol (same as WebSocket)
//!
//! Subscribe:   `{"type":"subscribe","id":1,"pattern":"users:*"}`
//! Response:    `{"type":"subscribed","id":1,"subscription_id":42}`
//! Notification:`{"type":"notification","subscription_id":42,"operation":"insert","key":"users:1","value":"..."}`

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tokio::sync::{RwLock, mpsc};
use wtransport::{Connection, Endpoint, Identity, ServerConfig};

use joule_db_local::Database;

use crate::subscriptions::{ChangeOperation, SubscriptionManager};

/// WebTransport server configuration
#[derive(Debug, Clone)]
pub struct WebTransportConfig {
    /// Bind address (default: 0.0.0.0:4433)
    pub bind_addr: String,
    /// Bind port (extracted from bind_addr)
    pub bind_port: u16,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// TLS certificate PEM file path (None = self-signed for dev)
    pub cert_path: Option<String>,
    /// TLS private key PEM file path
    pub key_path: Option<String>,
    /// Enable datagram-based notifications (unreliable but lower latency)
    pub enable_datagrams: bool,
}

impl Default for WebTransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:4433".to_string(),
            bind_port: 4433,
            max_connections: 10000,
            cert_path: None,
            key_path: None,
            enable_datagrams: true,
        }
    }
}

/// WebTransport server statistics
#[derive(Debug, Default)]
pub struct WebTransportStats {
    pub connections_accepted: AtomicU64,
    pub active_connections: AtomicU32,
    pub messages_received: AtomicU64,
    pub messages_sent: AtomicU64,
    pub datagrams_sent: AtomicU64,
    pub notifications_sent: AtomicU64,
    pub errors: AtomicU64,
}

impl WebTransportStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> WebTransportStatsSnapshot {
        WebTransportStatsSnapshot {
            connections_accepted: self.connections_accepted.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            datagrams_sent: self.datagrams_sent.load(Ordering::Relaxed),
            notifications_sent: self.notifications_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of WebTransport stats
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebTransportStatsSnapshot {
    pub connections_accepted: u64,
    pub active_connections: u32,
    pub messages_received: u64,
    pub messages_sent: u64,
    pub datagrams_sent: u64,
    pub notifications_sent: u64,
    pub errors: u64,
}

/// WebTransport server
pub struct WebTransportServer {
    db: Arc<RwLock<Database>>,
    config: WebTransportConfig,
    stats: Arc<WebTransportStats>,
    subscription_manager: Arc<SubscriptionManager>,
}

impl WebTransportServer {
    /// Create a new WebTransport server
    pub fn new(
        db: Arc<RwLock<Database>>,
        config: WebTransportConfig,
        subscription_manager: Arc<SubscriptionManager>,
    ) -> Self {
        Self {
            db,
            config,
            stats: Arc::new(WebTransportStats::new()),
            subscription_manager,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &Arc<WebTransportStats> {
        &self.stats
    }

    /// Get the certificate hash for browser clients (development mode)
    ///
    /// When using self-signed certificates, browsers need the certificate hash
    /// passed via `serverCertificateHashes` in the WebTransport constructor.
    pub fn cert_hash(&self) -> Option<Vec<u8>> {
        // Will be set during run() if using self-signed certs
        None
    }

    /// Run the WebTransport server
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let identity =
            if let (Some(cert), Some(key)) = (&self.config.cert_path, &self.config.key_path) {
                Identity::load_pemfiles(cert, key).await?
            } else {
                // Self-signed certificate for development
                Identity::self_signed(["localhost", "127.0.0.1", "::1"])
                    .map_err(|e| format!("Failed to generate self-signed cert: {}", e))?
            };

        let server_config = ServerConfig::builder()
            .with_bind_default(self.config.bind_port)
            .with_identity(identity)
            .build();

        let endpoint = Endpoint::server(server_config)?;
        let local_addr = endpoint.local_addr()?;

        tracing::info!("WebTransport server listening on {}", local_addr);

        // Accept connections in a loop
        loop {
            let incoming = endpoint.accept().await;

            let db = self.db.clone();
            let stats = self.stats.clone();
            let sub_mgr = self.subscription_manager.clone();
            let enable_datagrams = self.config.enable_datagrams;

            tokio::spawn(async move {
                // Accept the session request
                let request = match incoming.await {
                    Ok(req) => req,
                    Err(e) => {
                        tracing::debug!("WebTransport session error: {}", e);
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                };

                // Accept the WebTransport session
                let connection = match request.accept().await {
                    Ok(conn) => conn,
                    Err(e) => {
                        tracing::debug!("WebTransport accept error: {}", e);
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                };

                stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                stats.active_connections.fetch_add(1, Ordering::Relaxed);

                handle_wt_connection(connection, db, sub_mgr, stats.clone(), enable_datagrams)
                    .await;

                stats.active_connections.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }
}

/// Handle a single WebTransport connection
async fn handle_wt_connection(
    connection: Connection,
    _db: Arc<RwLock<Database>>,
    sub_mgr: Arc<SubscriptionManager>,
    stats: Arc<WebTransportStats>,
    enable_datagrams: bool,
) {
    // Channel for pushing messages to the client (notifications + responses)
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<OutgoingMessage>();

    // Track active subscriptions for this connection
    let conn_subs: Arc<RwLock<HashMap<u64, tokio::task::JoinHandle<()>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    let conn_subs_cleanup = conn_subs.clone();
    let sub_mgr_cleanup = sub_mgr.clone();
    let conn_ref = Arc::new(connection);
    let conn_writer = conn_ref.clone();
    let conn_reader = conn_ref.clone();

    // Writer task: sends outgoing messages via bidirectional streams or datagrams
    let stats_writer = stats.clone();
    let writer = tokio::spawn(async move {
        // We'll open a server-initiated bidirectional stream for pushing notifications
        // and responses. The client reads from this stream.
        while let Some(msg) = outgoing_rx.recv().await {
            let json_bytes = msg.data.as_bytes();

            if msg.use_datagram && enable_datagrams {
                // Try datagram (unreliable, low latency)
                if conn_writer.send_datagram(json_bytes).is_ok() {
                    stats_writer.datagrams_sent.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Send via a new unidirectional stream (reliable)
                match conn_writer.open_uni().await {
                    Ok(opening) => {
                        match opening.await {
                            Ok(mut send) => {
                                // Write length-prefixed JSON
                                let len = json_bytes.len() as u32;
                                if send.write_all(&len.to_le_bytes()).await.is_ok()
                                    && send.write_all(json_bytes).await.is_ok()
                                {
                                    let _ = send.finish().await;
                                    stats_writer.messages_sent.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    Err(_) => break, // Connection closed
                }
            }
        }
    });

    // Reader loop: accept bidirectional streams from the client
    loop {
        match conn_reader.accept_bi().await {
            Ok((send_stream, recv_stream)) => {
                let outgoing_tx = outgoing_tx.clone();
                let sub_mgr = sub_mgr.clone();
                let conn_subs = conn_subs.clone();
                let stats = stats.clone();

                // Handle each bidirectional stream in its own task
                tokio::spawn(async move {
                    handle_wt_stream(
                        send_stream,
                        recv_stream,
                        outgoing_tx,
                        sub_mgr,
                        conn_subs,
                        stats,
                    )
                    .await;
                });
            }
            Err(_) => {
                // Connection closed
                break;
            }
        }
    }

    // Cleanup: unsubscribe all active subscriptions
    {
        let mut subs = conn_subs_cleanup.write().await;
        for (sub_id, handle) in subs.drain() {
            handle.abort();
            sub_mgr_cleanup.unsubscribe(sub_id).await;
        }
    }

    writer.abort();
}

/// Outgoing message wrapper
struct OutgoingMessage {
    /// JSON payload
    data: String,
    /// Whether to send as datagram (unreliable) instead of stream
    use_datagram: bool,
}

/// Handle a single bidirectional stream (one request-response exchange)
async fn handle_wt_stream(
    send_stream: wtransport::SendStream,
    recv_stream: wtransport::RecvStream,
    outgoing_tx: mpsc::UnboundedSender<OutgoingMessage>,
    sub_mgr: Arc<SubscriptionManager>,
    conn_subs: Arc<RwLock<HashMap<u64, tokio::task::JoinHandle<()>>>>,
    stats: Arc<WebTransportStats>,
) {
    use tokio::io::AsyncReadExt;

    let mut recv = recv_stream;

    // Read length-prefixed JSON message
    let mut len_buf = [0u8; 4];
    if recv.read_exact(&mut len_buf).await.is_err() {
        return;
    }
    let msg_len = u32::from_le_bytes(len_buf) as usize;

    if msg_len > 16 * 1024 * 1024 {
        // Message too large
        return;
    }

    let mut msg_buf = vec![0u8; msg_len];
    if recv.read_exact(&mut msg_buf).await.is_err() {
        return;
    }

    stats.messages_received.fetch_add(1, Ordering::Relaxed);

    let json_str = match String::from_utf8(msg_buf) {
        Ok(s) => s,
        Err(_) => return,
    };

    let json: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => {
            let error_resp =
                serde_json::json!({"type":"error","message":"Invalid JSON"}).to_string();
            send_response(send_stream, &error_resp).await;
            return;
        }
    };

    // Process the JSON message (same protocol as WebSocket)
    let response = handle_wt_json_message(&json, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

    // Send response on the same bidirectional stream
    send_response(send_stream, &response).await;
}

/// Send a length-prefixed response on a send stream
async fn send_response(send_stream: wtransport::SendStream, response: &str) {
    let mut send = send_stream;
    let bytes = response.as_bytes();
    let len = bytes.len() as u32;
    let _ = send.write_all(&len.to_le_bytes()).await;
    let _ = send.write_all(bytes).await;
    let _ = send.finish().await;
}

/// Handle a JSON subscription message (same protocol as WebSocket)
async fn handle_wt_json_message(
    json: &serde_json::Value,
    sub_mgr: &Arc<SubscriptionManager>,
    outgoing_tx: &mpsc::UnboundedSender<OutgoingMessage>,
    conn_subs: &Arc<RwLock<HashMap<u64, tokio::task::JoinHandle<()>>>>,
    stats: &Arc<WebTransportStats>,
) -> String {
    let msg_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let request_id = json.get("id").and_then(|v| v.as_u64()).unwrap_or(0);

    match msg_type {
        "subscribe" => {
            let pattern = match json.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": request_id,
                        "message": "Missing 'pattern' field"
                    })
                    .to_string();
                }
            };

            let (sub_id, mut receiver) = match sub_mgr.subscribe(pattern).await {
                Ok(pair) => pair,
                Err(e) => {
                    return serde_json::json!({
                        "type": "error",
                        "message": e
                    })
                    .to_string();
                }
            };

            // Spawn notification forwarder
            let tx = outgoing_tx.clone();
            let stats_fwd = stats.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(event) = receiver.recv().await {
                    let value_field =
                        event
                            .value
                            .as_ref()
                            .map(|v| match String::from_utf8(v.clone()) {
                                Ok(s) => serde_json::json!(s),
                                Err(_) => serde_json::json!(
                                    v.iter().map(|b| format!("{:02x}", b)).collect::<String>()
                                ),
                            });

                    let notification = serde_json::json!({
                        "type": "notification",
                        "subscription_id": sub_id,
                        "operation": match event.operation {
                            ChangeOperation::Insert => "insert",
                            ChangeOperation::Update => "update",
                            ChangeOperation::Delete => "delete",
                        },
                        "key": event.key,
                        "value": value_field,
                        "timestamp": event.timestamp,
                    });

                    let msg = OutgoingMessage {
                        data: notification.to_string(),
                        use_datagram: true, // Use datagrams for notifications (fast path)
                    };

                    if tx.send(msg).is_err() {
                        break;
                    }
                    stats_fwd.notifications_sent.fetch_add(1, Ordering::Relaxed);
                }
            });

            conn_subs.write().await.insert(sub_id, forwarder);

            serde_json::json!({
                "type": "subscribed",
                "id": request_id,
                "subscription_id": sub_id,
            })
            .to_string()
        }

        "unsubscribe" => {
            let sub_id = match json.get("subscription_id").and_then(|v| v.as_u64()) {
                Some(id) => id,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": request_id,
                        "message": "Missing 'subscription_id' field"
                    })
                    .to_string();
                }
            };

            // Verify this connection owns the subscription before unsubscribing
            let owns_sub = conn_subs.read().await.contains_key(&sub_id);
            if !owns_sub {
                return serde_json::json!({
                    "type": "error",
                    "id": request_id,
                    "message": "Subscription not owned by this connection"
                })
                .to_string();
            }
            if let Some(handle) = conn_subs.write().await.remove(&sub_id) {
                handle.abort();
            }
            let ok = sub_mgr.unsubscribe(sub_id).await;

            serde_json::json!({
                "type": "unsubscribed",
                "id": request_id,
                "ok": ok,
            })
            .to_string()
        }

        "ping" => serde_json::json!({
            "type": "pong",
            "id": request_id,
        })
        .to_string(),

        _ => serde_json::json!({
            "type": "error",
            "id": request_id,
            "message": format!("Unknown message type: {}", msg_type),
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = WebTransportConfig::default();
        assert_eq!(config.bind_port, 4433);
        assert_eq!(config.max_connections, 10000);
        assert!(config.enable_datagrams);
        assert!(config.cert_path.is_none());
    }

    #[test]
    fn test_stats_snapshot() {
        let stats = WebTransportStats::new();
        stats.connections_accepted.fetch_add(5, Ordering::Relaxed);
        stats.active_connections.fetch_add(3, Ordering::Relaxed);
        stats.messages_received.fetch_add(100, Ordering::Relaxed);
        stats.datagrams_sent.fetch_add(50, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.connections_accepted, 5);
        assert_eq!(snapshot.active_connections, 3);
        assert_eq!(snapshot.messages_received, 100);
        assert_eq!(snapshot.datagrams_sent, 50);
    }

    #[tokio::test]
    async fn test_json_subscribe_message() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        let msg = serde_json::json!({
            "type": "subscribe",
            "id": 1,
            "pattern": "users:*"
        });

        let response =
            handle_wt_json_message(&msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(resp["type"], "subscribed");
        assert_eq!(resp["id"], 1);
        assert!(resp["subscription_id"].as_u64().is_some());

        // Should have created a subscription
        assert_eq!(sub_mgr.active_count().await, 1);
        assert_eq!(conn_subs.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_json_unsubscribe_message() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        // Subscribe first
        let sub_msg = serde_json::json!({
            "type": "subscribe",
            "id": 1,
            "pattern": "test:*"
        });
        let sub_resp =
            handle_wt_json_message(&sub_msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;
        let sub_resp: serde_json::Value = serde_json::from_str(&sub_resp).unwrap();
        let sub_id = sub_resp["subscription_id"].as_u64().unwrap();

        // Unsubscribe
        let unsub_msg = serde_json::json!({
            "type": "unsubscribe",
            "id": 2,
            "subscription_id": sub_id
        });
        let response =
            handle_wt_json_message(&unsub_msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(resp["type"], "unsubscribed");
        assert_eq!(resp["ok"], true);

        assert_eq!(sub_mgr.active_count().await, 0);
        assert_eq!(conn_subs.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_json_ping_message() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        let msg = serde_json::json!({
            "type": "ping",
            "id": 42
        });

        let response =
            handle_wt_json_message(&msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(resp["type"], "pong");
        assert_eq!(resp["id"], 42);
    }

    #[tokio::test]
    async fn test_json_unknown_type() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        let msg = serde_json::json!({
            "type": "bogus",
            "id": 1
        });

        let response =
            handle_wt_json_message(&msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(resp["type"], "error");
        assert!(resp["message"].as_str().unwrap().contains("Unknown"));
    }

    #[tokio::test]
    async fn test_subscribe_missing_pattern() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        let msg = serde_json::json!({
            "type": "subscribe",
            "id": 1
        });

        let response =
            handle_wt_json_message(&msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(resp["type"], "error");
        assert!(resp["message"].as_str().unwrap().contains("pattern"));
    }

    #[tokio::test]
    async fn test_notification_forwarding() {
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel();
        let conn_subs = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(WebTransportStats::new());

        // Subscribe
        let msg = serde_json::json!({
            "type": "subscribe",
            "id": 1,
            "pattern": "game:*"
        });
        let _ = handle_wt_json_message(&msg, &sub_mgr, &outgoing_tx, &conn_subs, &stats).await;

        // Fire a notification
        sub_mgr
            .notify_insert("game:player1", b"position:10,20")
            .await;

        // Should receive the notification via outgoing channel
        let notification =
            tokio::time::timeout(std::time::Duration::from_secs(2), outgoing_rx.recv())
                .await
                .unwrap()
                .unwrap();

        let notif: serde_json::Value = serde_json::from_str(&notification.data).unwrap();
        assert_eq!(notif["type"], "notification");
        assert_eq!(notif["key"], "game:player1");
        assert_eq!(notif["operation"], "insert");
        assert!(notification.use_datagram); // Should prefer datagram for notifications
    }
}
