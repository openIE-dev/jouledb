//! WebSocket client — protocol-level state machine and message handling.
//!
//! Replaces `socket.io-client` and `ws` with a pure-Rust protocol state
//! machine.  No actual I/O is performed here; the caller is responsible for
//! wiring the connection to a real transport.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::Serialize;

// ── Close codes (RFC 6455 §7.4.1) ──────────────────────────────────────────

/// Standard WebSocket close codes.
pub struct WsCloseCode;

impl WsCloseCode {
    pub const NORMAL: u16 = 1000;
    pub const GOING_AWAY: u16 = 1001;
    pub const PROTOCOL_ERROR: u16 = 1002;
    pub const UNSUPPORTED: u16 = 1003;
    pub const ABNORMAL: u16 = 1006;
    pub const INVALID_PAYLOAD: u16 = 1007;
    pub const POLICY_VIOLATION: u16 = 1008;
    pub const TOO_LARGE: u16 = 1009;
    pub const SERVER_ERROR: u16 = 1011;
}

// ── State & messages ────────────────────────────────────────────────────────

/// Connection lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsState {
    Connecting,
    Open,
    Closing,
    Closed,
}

/// A WebSocket message (data or control frame).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close { code: u16, reason: String },
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a WebSocket connection.
#[derive(Debug, Clone)]
pub struct WsConfig {
    pub url: String,
    pub protocols: Vec<String>,
    pub reconnect: bool,
    pub max_reconnect_attempts: u32,
    pub reconnect_delay_ms: u64,
    pub reconnect_backoff_multiplier: f64,
    pub heartbeat_interval_ms: Option<u64>,
    pub max_message_size: Option<usize>,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            protocols: Vec::new(),
            reconnect: true,
            max_reconnect_attempts: 5,
            reconnect_delay_ms: 1000,
            reconnect_backoff_multiplier: 2.0,
            heartbeat_interval_ms: Some(30_000),
            max_message_size: Some(64 * 1024),
        }
    }
}

// ── Connection ──────────────────────────────────────────────────────────────

/// Protocol-level WebSocket connection state machine.
pub struct WsConnection {
    pub config: WsConfig,
    pub state: WsState,
    pub reconnect_attempts: u32,
    pub send_queue: VecDeque<WsMessage>,
    pub received: VecDeque<WsMessage>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub last_pong: Option<DateTime<Utc>>,
    pub connected_at: Option<DateTime<Utc>>,
}

impl WsConnection {
    pub fn new(config: WsConfig) -> Self {
        Self {
            config,
            state: WsState::Closed,
            reconnect_attempts: 0,
            send_queue: VecDeque::new(),
            received: VecDeque::new(),
            last_heartbeat: None,
            last_pong: None,
            connected_at: None,
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// Begin the connection handshake.
    pub fn connect(&mut self) {
        self.state = WsState::Connecting;
    }

    /// Called when the transport confirms the connection is open.
    pub fn on_open(&mut self) {
        self.state = WsState::Open;
        self.reconnect_attempts = 0;
        self.connected_at = Some(Utc::now());
    }

    /// Called when a message arrives from the remote end.
    pub fn on_message(&mut self, msg: WsMessage) {
        // Auto-respond to Ping with Pong.
        if let WsMessage::Ping(ref payload) = msg {
            self.send_queue.push_back(WsMessage::Pong(payload.clone()));
        }
        if let WsMessage::Pong(_) = msg {
            self.last_pong = Some(Utc::now());
        }
        self.received.push_back(msg);
    }

    /// Called when the connection is closed.
    pub fn on_close(&mut self, code: u16, reason: &str) {
        self.state = WsState::Closed;
        self.connected_at = None;
        self.received.push_back(WsMessage::Close {
            code,
            reason: reason.to_string(),
        });
    }

    /// Called on transport error.  If reconnect is enabled and we haven't
    /// exhausted attempts, transition to Connecting.
    pub fn on_error(&mut self, _error: &str) {
        if self.should_reconnect() {
            self.attempt_reconnect();
        } else {
            self.state = WsState::Closed;
        }
    }

    // ── Sending ─────────────────────────────────────────────────────────

    /// Queue a message for sending.  Returns `false` if not connected.
    pub fn send(&mut self, msg: WsMessage) -> bool {
        if self.state != WsState::Open {
            return false;
        }
        self.send_queue.push_back(msg);
        true
    }

    /// Convenience: send a text message.
    pub fn send_text(&mut self, text: &str) -> bool {
        self.send(WsMessage::Text(text.to_string()))
    }

    /// Convenience: serialize `data` as JSON and send as text.
    pub fn send_json<T: Serialize>(&mut self, data: &T) -> bool {
        match serde_json::to_string(data) {
            Ok(json) => self.send(WsMessage::Text(json)),
            Err(_) => false,
        }
    }

    /// Drain and return all queued outgoing messages.
    pub fn drain_send_queue(&mut self) -> Vec<WsMessage> {
        self.send_queue.drain(..).collect()
    }

    // ── Receiving ───────────────────────────────────────────────────────

    /// Pop the next received message.
    pub fn recv(&mut self) -> Option<WsMessage> {
        self.received.pop_front()
    }

    /// Drain all received messages.
    pub fn recv_all(&mut self) -> Vec<WsMessage> {
        self.received.drain(..).collect()
    }

    // ── Reconnect logic ─────────────────────────────────────────────────

    /// Whether we should attempt a reconnect.
    pub fn should_reconnect(&self) -> bool {
        self.config.reconnect && self.reconnect_attempts < self.config.max_reconnect_attempts
    }

    /// Exponential-backoff delay for the next reconnect.
    pub fn next_reconnect_delay_ms(&self) -> u64 {
        let multiplier = self.config.reconnect_backoff_multiplier
            .powi(self.reconnect_attempts as i32);
        (self.config.reconnect_delay_ms as f64 * multiplier) as u64
    }

    /// Increment attempt counter and transition to `Connecting`.
    pub fn attempt_reconnect(&mut self) {
        self.reconnect_attempts += 1;
        self.state = WsState::Connecting;
    }

    // ── Heartbeat ───────────────────────────────────────────────────────

    /// Whether enough time has elapsed to send a heartbeat.
    pub fn should_heartbeat(&self, now: &DateTime<Utc>) -> bool {
        if self.state != WsState::Open {
            return false;
        }
        let Some(interval_ms) = self.config.heartbeat_interval_ms else {
            return false;
        };
        match self.last_heartbeat {
            None => true,
            Some(last) => {
                let elapsed = (*now - last).num_milliseconds();
                elapsed >= interval_ms as i64
            }
        }
    }

    /// Queue a Ping frame and record the heartbeat time.
    pub fn send_heartbeat(&mut self) {
        self.send_queue.push_back(WsMessage::Ping(Vec::new()));
        self.last_heartbeat = Some(Utc::now());
    }

    /// Whether the connection is currently open.
    pub fn is_connected(&self) -> bool {
        self.state == WsState::Open
    }

    /// How long the connection has been open.
    pub fn uptime(&self, now: &DateTime<Utc>) -> Option<chrono::Duration> {
        self.connected_at.map(|at| *now - at)
    }
}

// ── Channel-based pub/sub ───────────────────────────────────────────────────

/// A named channel that maps event names to sets of handler IDs.
#[derive(Debug, Clone)]
pub struct WsChannel {
    pub name: String,
    pub subscriptions: HashMap<String, Vec<u64>>,
}

impl WsChannel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            subscriptions: HashMap::new(),
        }
    }
}

/// Manages multiple named channels.
pub struct WsChannelManager {
    pub channels: HashMap<String, WsChannel>,
}

impl WsChannelManager {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Subscribe a handler to an event on a channel (auto-creates channel).
    pub fn subscribe(&mut self, channel: &str, event: &str, handler_id: u64) {
        let ch = self
            .channels
            .entry(channel.to_string())
            .or_insert_with(|| WsChannel::new(channel));
        let handlers = ch
            .subscriptions
            .entry(event.to_string())
            .or_default();
        if !handlers.contains(&handler_id) {
            handlers.push(handler_id);
        }
    }

    /// Unsubscribe a handler from an event.
    pub fn unsubscribe(&mut self, channel: &str, event: &str, handler_id: u64) {
        if let Some(ch) = self.channels.get_mut(channel) {
            if let Some(handlers) = ch.subscriptions.get_mut(event) {
                handlers.retain(|id| *id != handler_id);
            }
        }
    }

    /// Return handler IDs subscribed to the event on the given channel.
    pub fn dispatch(&self, channel: &str, event: &str) -> Vec<u64> {
        self.channels
            .get(channel)
            .and_then(|ch| ch.subscriptions.get(event))
            .cloned()
            .unwrap_or_default()
    }

    /// List all channel names.
    pub fn channel_names(&self) -> Vec<&str> {
        self.channels.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for WsChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> WsConfig {
        WsConfig {
            url: "ws://localhost:8080".into(),
            reconnect_delay_ms: 100,
            ..WsConfig::default()
        }
    }

    #[test]
    fn connect_open_close_lifecycle() {
        let mut conn = WsConnection::new(test_config());
        assert_eq!(conn.state, WsState::Closed);

        conn.connect();
        assert_eq!(conn.state, WsState::Connecting);

        conn.on_open();
        assert_eq!(conn.state, WsState::Open);
        assert!(conn.connected_at.is_some());

        conn.on_close(WsCloseCode::NORMAL, "bye");
        assert_eq!(conn.state, WsState::Closed);
    }

    #[test]
    fn send_queues_when_open() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();

        assert!(conn.send_text("hello"));
        assert_eq!(conn.send_queue.len(), 1);
    }

    #[test]
    fn send_fails_when_closed() {
        let mut conn = WsConnection::new(test_config());
        assert!(!conn.send_text("nope"));
        assert!(conn.send_queue.is_empty());
    }

    #[test]
    fn recv_returns_messages() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();
        conn.on_message(WsMessage::Text("hi".into()));

        let msg = conn.recv().unwrap();
        assert_eq!(msg, WsMessage::Text("hi".into()));
        assert!(conn.recv().is_none());
    }

    #[test]
    fn auto_pong_on_ping() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();

        conn.on_message(WsMessage::Ping(vec![1, 2, 3]));

        // The send queue should contain a Pong with the same payload.
        let queued = conn.drain_send_queue();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0], WsMessage::Pong(vec![1, 2, 3]));
    }

    #[test]
    fn reconnect_with_backoff() {
        let mut conn = WsConnection::new(test_config());
        assert_eq!(conn.next_reconnect_delay_ms(), 100); // base

        conn.reconnect_attempts = 1;
        assert_eq!(conn.next_reconnect_delay_ms(), 200); // 100 * 2^1

        conn.reconnect_attempts = 2;
        assert_eq!(conn.next_reconnect_delay_ms(), 400); // 100 * 2^2
    }

    #[test]
    fn max_reconnect_attempts() {
        let mut conn = WsConnection::new(WsConfig {
            max_reconnect_attempts: 2,
            ..test_config()
        });
        assert!(conn.should_reconnect());

        conn.reconnect_attempts = 2;
        assert!(!conn.should_reconnect());
    }

    #[test]
    fn heartbeat_timing() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();

        let now = Utc::now();
        assert!(conn.should_heartbeat(&now)); // never sent before

        conn.send_heartbeat();
        // Immediately after sending, should not need another heartbeat.
        let now2 = Utc::now();
        assert!(!conn.should_heartbeat(&now2));
    }

    #[test]
    fn send_json_serializes() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();

        #[derive(Serialize)]
        struct Payload {
            value: i32,
        }
        assert!(conn.send_json(&Payload { value: 42 }));

        let queued = conn.drain_send_queue();
        assert_eq!(queued.len(), 1);
        if let WsMessage::Text(ref text) = queued[0] {
            assert!(text.contains("42"));
        } else {
            panic!("expected text message");
        }
    }

    #[test]
    fn channel_subscribe_dispatch() {
        let mut mgr = WsChannelManager::new();
        mgr.subscribe("chat", "message", 1);
        mgr.subscribe("chat", "message", 2);

        let ids = mgr.dispatch("chat", "message");
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn channel_unsubscribe() {
        let mut mgr = WsChannelManager::new();
        mgr.subscribe("chat", "message", 1);
        mgr.subscribe("chat", "message", 2);
        mgr.unsubscribe("chat", "message", 1);

        let ids = mgr.dispatch("chat", "message");
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn drain_send_queue_empties() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();
        conn.send_text("a");
        conn.send_text("b");

        let drained = conn.drain_send_queue();
        assert_eq!(drained.len(), 2);
        assert!(conn.send_queue.is_empty());
    }

    #[test]
    fn channel_names() {
        let mut mgr = WsChannelManager::new();
        mgr.subscribe("alpha", "ev", 1);
        mgr.subscribe("beta", "ev", 2);

        let mut names = mgr.channel_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn uptime_returns_duration_when_connected() {
        let mut conn = WsConnection::new(test_config());
        conn.connect();
        conn.on_open();

        let now = Utc::now();
        let up = conn.uptime(&now);
        assert!(up.is_some());
    }
}
