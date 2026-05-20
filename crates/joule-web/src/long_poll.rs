//! Long polling — request holding, timeout management, event queue per client,
//! reconnection tracking, message ordering, and comet-style patterns.
//!
//! Pure-Rust long-polling server model with no real timers or I/O. Time is
//! injected by the caller; the module produces response payloads.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ── Message ────────────────────────────────────────────────────────

/// A message in the long-polling event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LpMessage {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Channel/topic this message belongs to.
    pub channel: String,
    /// Event type (e.g., "message", "update", "delete").
    pub event_type: String,
    /// Payload data (opaque string, typically JSON).
    pub data: String,
    /// Timestamp in milliseconds when the message was created.
    pub created_ms: u64,
}

impl LpMessage {
    pub fn new(
        seq: u64,
        channel: impl Into<String>,
        event_type: impl Into<String>,
        data: impl Into<String>,
        created_ms: u64,
    ) -> Self {
        Self {
            seq,
            channel: channel.into(),
            event_type: event_type.into(),
            data: data.into(),
            created_ms,
        }
    }
}

// ── Poll result ────────────────────────────────────────────────────

/// Result of a long-poll request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollResult {
    /// Messages are available to return immediately.
    Messages(Vec<LpMessage>),
    /// No messages; the request is being held (pending).
    Pending { request_id: u64 },
    /// The poll timed out with no new messages.
    Timeout { last_seq: u64 },
    /// The client is not registered.
    NotRegistered,
}

// ── Client state ───────────────────────────────────────────────────

/// Per-client state in the long-polling server.
#[derive(Debug)]
struct ClientState {
    /// Last sequence number delivered to this client.
    last_delivered_seq: u64,
    /// Channels this client is subscribed to.
    subscriptions: Vec<String>,
    /// Whether this client has a pending (held) request.
    pending_request: Option<PendingRequest>,
    /// Timestamp of last poll from this client.
    last_poll_ms: u64,
    /// Number of reconnections.
    reconnect_count: u32,
}

/// A held (pending) poll request.
#[derive(Debug)]
struct PendingRequest {
    request_id: u64,
    started_ms: u64,
    timeout_ms: u64,
}

impl PendingRequest {
    fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.started_ms) >= self.timeout_ms
    }
}

// ── Server configuration ───────────────────────────────────────────

/// Configuration for the long-polling server.
#[derive(Debug, Clone)]
pub struct LongPollConfig {
    /// Default timeout for held requests in milliseconds.
    pub default_timeout_ms: u64,
    /// Maximum number of messages to return in a single response.
    pub max_batch_size: usize,
    /// Maximum number of messages to retain in the server buffer.
    pub max_buffer_size: usize,
    /// How long to consider a client alive after its last poll (ms).
    pub client_timeout_ms: u64,
}

impl Default for LongPollConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 30_000,
            max_batch_size: 100,
            max_buffer_size: 10_000,
            client_timeout_ms: 120_000,
        }
    }
}

// ── Long poll server ───────────────────────────────────────────────

/// A long-polling server managing multiple clients and message channels.
#[derive(Debug)]
pub struct LongPollServer {
    config: LongPollConfig,
    /// Global message buffer, ordered by sequence number.
    messages: VecDeque<LpMessage>,
    /// Per-client state.
    clients: HashMap<String, ClientState>,
    /// Next sequence number to assign.
    next_seq: u64,
    /// Next request ID to assign.
    next_request_id: u64,
}

impl LongPollServer {
    pub fn new(config: LongPollConfig) -> Self {
        Self {
            config,
            messages: VecDeque::new(),
            clients: HashMap::new(),
            next_seq: 1,
            next_request_id: 1,
        }
    }

    /// Register a new client. Returns false if already registered.
    pub fn register_client(&mut self, client_id: impl Into<String>, now_ms: u64) -> bool {
        let cid = client_id.into();
        if self.clients.contains_key(&cid) {
            return false;
        }
        self.clients.insert(cid, ClientState {
            last_delivered_seq: 0,
            subscriptions: Vec::new(),
            pending_request: None,
            last_poll_ms: now_ms,
            reconnect_count: 0,
        });
        true
    }

    /// Unregister a client.
    pub fn unregister_client(&mut self, client_id: &str) -> bool {
        self.clients.remove(client_id).is_some()
    }

    /// Subscribe a client to a channel.
    pub fn subscribe(&mut self, client_id: &str, channel: impl Into<String>) -> bool {
        if let Some(state) = self.clients.get_mut(client_id) {
            let ch = channel.into();
            if !state.subscriptions.contains(&ch) {
                state.subscriptions.push(ch);
            }
            true
        } else {
            false
        }
    }

    /// Unsubscribe a client from a channel.
    pub fn unsubscribe(&mut self, client_id: &str, channel: &str) -> bool {
        if let Some(state) = self.clients.get_mut(client_id) {
            let before = state.subscriptions.len();
            state.subscriptions.retain(|c| c != channel);
            state.subscriptions.len() < before
        } else {
            false
        }
    }

    /// Publish a message to a channel. Returns the assigned sequence number.
    pub fn publish(
        &mut self,
        channel: impl Into<String>,
        event_type: impl Into<String>,
        data: impl Into<String>,
        now_ms: u64,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let msg = LpMessage::new(seq, channel, event_type, data, now_ms);
        self.messages.push_back(msg);

        // Trim buffer if too large
        while self.messages.len() > self.config.max_buffer_size {
            self.messages.pop_front();
        }

        seq
    }

    /// A client polls for messages. Returns available messages or holds the request.
    pub fn poll(
        &mut self,
        client_id: &str,
        since_seq: Option<u64>,
        now_ms: u64,
    ) -> PollResult {
        let Some(state) = self.clients.get_mut(client_id) else {
            return PollResult::NotRegistered;
        };

        state.last_poll_ms = now_ms;
        if state.pending_request.is_some() {
            state.reconnect_count += 1;
        }

        let base_seq = since_seq.unwrap_or(state.last_delivered_seq);

        // Collect messages for this client
        let relevant: Vec<LpMessage> = self.messages.iter()
            .filter(|m| m.seq > base_seq && state.subscriptions.contains(&m.channel))
            .take(self.config.max_batch_size)
            .cloned()
            .collect();

        if !relevant.is_empty() {
            let max_seq = relevant.iter().map(|m| m.seq).max().unwrap();
            // Need to re-borrow state mutably
            let state = self.clients.get_mut(client_id).unwrap();
            state.last_delivered_seq = max_seq;
            state.pending_request = None;
            PollResult::Messages(relevant)
        } else {
            let request_id = self.next_request_id;
            self.next_request_id += 1;
            let state = self.clients.get_mut(client_id).unwrap();
            state.pending_request = Some(PendingRequest {
                request_id,
                started_ms: now_ms,
                timeout_ms: self.config.default_timeout_ms,
            });
            PollResult::Pending { request_id }
        }
    }

    /// Tick the server: check for timed-out pending requests and stale clients.
    /// Returns a list of (client_id, PollResult) for timed-out requests.
    pub fn tick(&mut self, now_ms: u64) -> Vec<(String, PollResult)> {
        let mut results = Vec::new();

        // Check pending requests for timeout
        let client_ids: Vec<String> = self.clients.keys().cloned().collect();
        for cid in &client_ids {
            let state = self.clients.get_mut(cid).unwrap();
            if let Some(ref pending) = state.pending_request {
                if pending.is_expired(now_ms) {
                    let last_seq = state.last_delivered_seq;
                    state.pending_request = None;
                    results.push((cid.clone(), PollResult::Timeout { last_seq }));
                }
            }
        }

        // Evict stale clients
        let timeout = self.config.client_timeout_ms;
        let stale: Vec<String> = self.clients.iter()
            .filter(|(_, state)| now_ms.saturating_sub(state.last_poll_ms) > timeout)
            .map(|(cid, _)| cid.clone())
            .collect();
        for cid in stale {
            self.clients.remove(&cid);
        }

        results
    }

    /// Try to deliver newly published messages to pending requests.
    /// Returns a list of (client_id, messages) that should be sent.
    pub fn flush_pending(&mut self) -> Vec<(String, Vec<LpMessage>)> {
        let mut deliveries = Vec::new();
        let client_ids: Vec<String> = self.clients.keys().cloned().collect();

        for cid in client_ids {
            let state = self.clients.get(&cid).unwrap();
            if state.pending_request.is_none() {
                continue;
            }
            let base_seq = state.last_delivered_seq;
            let subs = &state.subscriptions;
            let batch_size = self.config.max_batch_size;

            let msgs: Vec<LpMessage> = self.messages.iter()
                .filter(|m| m.seq > base_seq && subs.contains(&m.channel))
                .take(batch_size)
                .cloned()
                .collect();

            if !msgs.is_empty() {
                let max_seq = msgs.iter().map(|m| m.seq).max().unwrap();
                let state = self.clients.get_mut(&cid).unwrap();
                state.last_delivered_seq = max_seq;
                state.pending_request = None;
                deliveries.push((cid, msgs));
            }
        }

        deliveries
    }

    /// Number of registered clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Number of clients with pending (held) requests.
    pub fn pending_count(&self) -> usize {
        self.clients.values().filter(|s| s.pending_request.is_some()).count()
    }

    /// Number of messages in the buffer.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get the reconnect count for a client.
    pub fn reconnect_count(&self, client_id: &str) -> Option<u32> {
        self.clients.get(client_id).map(|s| s.reconnect_count)
    }

    /// Get the last delivered sequence number for a client.
    pub fn last_delivered_seq(&self, client_id: &str) -> Option<u64> {
        self.clients.get(client_id).map(|s| s.last_delivered_seq)
    }
}

impl Default for LongPollServer {
    fn default() -> Self {
        Self::new(LongPollConfig::default())
    }
}

impl fmt::Display for LpMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}:{} {}", self.seq, self.channel, self.event_type, self.data)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> LongPollServer {
        LongPollServer::new(LongPollConfig {
            default_timeout_ms: 1000,
            max_batch_size: 10,
            max_buffer_size: 100,
            client_timeout_ms: 5000,
        })
    }

    // ── Registration ───────────────────────────────────────────────

    #[test]
    fn register_client() {
        let mut srv = make_server();
        assert!(srv.register_client("c1", 0));
        assert!(!srv.register_client("c1", 0)); // duplicate
        assert_eq!(srv.client_count(), 1);
    }

    #[test]
    fn unregister_client() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        assert!(srv.unregister_client("c1"));
        assert!(!srv.unregister_client("c1")); // already gone
        assert_eq!(srv.client_count(), 0);
    }

    // ── Subscriptions ──────────────────────────────────────────────

    #[test]
    fn subscribe_unsubscribe() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        assert!(srv.subscribe("c1", "chat"));
        assert!(srv.subscribe("c1", "alerts"));
        assert!(srv.unsubscribe("c1", "chat"));
        assert!(!srv.unsubscribe("c1", "chat")); // already removed
    }

    #[test]
    fn subscribe_unknown_client() {
        let mut srv = make_server();
        assert!(!srv.subscribe("ghost", "chat"));
    }

    #[test]
    fn subscribe_idempotent() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "chat");
        srv.subscribe("c1", "chat"); // no duplicate
        // Should still get only one copy of messages
    }

    // ── Publish ────────────────────────────────────────────────────

    #[test]
    fn publish_increments_seq() {
        let mut srv = make_server();
        let s1 = srv.publish("chat", "message", "hello", 0);
        let s2 = srv.publish("chat", "message", "world", 1);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(srv.message_count(), 2);
    }

    #[test]
    fn publish_trims_buffer() {
        let mut srv = LongPollServer::new(LongPollConfig {
            max_buffer_size: 3,
            ..Default::default()
        });
        for i in 0..5 {
            srv.publish("ch", "evt", format!("{}", i), 0);
        }
        assert_eq!(srv.message_count(), 3);
    }

    // ── Polling ────────────────────────────────────────────────────

    #[test]
    fn poll_immediate_messages() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "chat");
        srv.publish("chat", "msg", "hello", 0);
        srv.publish("chat", "msg", "world", 1);

        match srv.poll("c1", None, 2) {
            PollResult::Messages(msgs) => {
                assert_eq!(msgs.len(), 2);
                assert_eq!(msgs[0].data, "hello");
                assert_eq!(msgs[1].data, "world");
            }
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    #[test]
    fn poll_with_since_seq() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "chat");
        srv.publish("chat", "msg", "a", 0);
        srv.publish("chat", "msg", "b", 1);
        srv.publish("chat", "msg", "c", 2);

        match srv.poll("c1", Some(2), 3) {
            PollResult::Messages(msgs) => {
                assert_eq!(msgs.len(), 1);
                assert_eq!(msgs[0].data, "c");
            }
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    #[test]
    fn poll_pending_when_no_messages() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "chat");

        match srv.poll("c1", None, 0) {
            PollResult::Pending { request_id } => {
                assert!(request_id > 0);
            }
            other => panic!("expected Pending, got {:?}", other),
        }
        assert_eq!(srv.pending_count(), 1);
    }

    #[test]
    fn poll_not_registered() {
        let mut srv = make_server();
        assert!(matches!(srv.poll("ghost", None, 0), PollResult::NotRegistered));
    }

    #[test]
    fn poll_only_subscribed_channels() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "chat");
        srv.publish("alerts", "alert", "fire", 0); // not subscribed
        srv.publish("chat", "msg", "hello", 1);

        match srv.poll("c1", None, 2) {
            PollResult::Messages(msgs) => {
                assert_eq!(msgs.len(), 1);
                assert_eq!(msgs[0].channel, "chat");
            }
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    #[test]
    fn poll_updates_last_delivered() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        srv.publish("ch", "e", "a", 0);
        srv.publish("ch", "e", "b", 1);
        srv.poll("c1", None, 2);
        assert_eq!(srv.last_delivered_seq("c1"), Some(2));
    }

    // ── Tick: timeouts ─────────────────────────────────────────────

    #[test]
    fn tick_timeout_pending() {
        let mut srv = make_server(); // timeout = 1000ms
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        srv.poll("c1", None, 0); // pending since t=0

        let results = srv.tick(500); // not expired yet
        assert!(results.is_empty());

        let results = srv.tick(1001); // expired
        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0].1, PollResult::Timeout { .. }));
    }

    #[test]
    fn tick_evict_stale_client() {
        let mut srv = make_server(); // client_timeout = 5000ms
        srv.register_client("c1", 0);
        assert_eq!(srv.client_count(), 1);

        srv.tick(6000);
        assert_eq!(srv.client_count(), 0);
    }

    #[test]
    fn tick_active_client_not_evicted() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        srv.poll("c1", None, 4000); // recent activity
        srv.tick(5001); // 5001 - 4000 = 1001 < 5000
        assert_eq!(srv.client_count(), 1);
    }

    // ── Flush pending ──────────────────────────────────────────────

    #[test]
    fn flush_delivers_to_pending() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        srv.poll("c1", None, 0); // hold request

        srv.publish("ch", "e", "hello", 1);
        let deliveries = srv.flush_pending();
        assert_eq!(deliveries.len(), 1);
        assert_eq!(deliveries[0].0, "c1");
        assert_eq!(deliveries[0].1.len(), 1);
        assert_eq!(deliveries[0].1[0].data, "hello");
        assert_eq!(srv.pending_count(), 0);
    }

    #[test]
    fn flush_no_pending() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        // No pending request
        srv.publish("ch", "e", "hello", 1);
        let deliveries = srv.flush_pending();
        assert!(deliveries.is_empty());
    }

    // ── Reconnection tracking ──────────────────────────────────────

    #[test]
    fn reconnect_count_tracked() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        srv.poll("c1", None, 0); // pending
        srv.poll("c1", None, 100); // new poll while pending = reconnect
        assert_eq!(srv.reconnect_count("c1"), Some(1));
    }

    #[test]
    fn reconnect_count_unknown() {
        let srv = make_server();
        assert_eq!(srv.reconnect_count("ghost"), None);
    }

    // ── Message ordering ───────────────────────────────────────────

    #[test]
    fn messages_ordered_by_seq() {
        let mut srv = make_server();
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        for i in 0..5 {
            srv.publish("ch", "e", format!("msg-{}", i), i as u64);
        }
        match srv.poll("c1", None, 5) {
            PollResult::Messages(msgs) => {
                for window in msgs.windows(2) {
                    assert!(window[0].seq < window[1].seq);
                }
            }
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    // ── Batch size limit ───────────────────────────────────────────

    #[test]
    fn batch_size_respected() {
        let mut srv = LongPollServer::new(LongPollConfig {
            max_batch_size: 3,
            ..Default::default()
        });
        srv.register_client("c1", 0);
        srv.subscribe("c1", "ch");
        for i in 0..10 {
            srv.publish("ch", "e", format!("{}", i), 0);
        }
        match srv.poll("c1", None, 0) {
            PollResult::Messages(msgs) => {
                assert_eq!(msgs.len(), 3);
            }
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    // ── Display ────────────────────────────────────────────────────

    #[test]
    fn message_display() {
        let msg = LpMessage::new(42, "chat", "message", "hello", 1000);
        let s = msg.to_string();
        assert!(s.contains("[42]"));
        assert!(s.contains("chat:message"));
        assert!(s.contains("hello"));
    }

    // ── Default ────────────────────────────────────────────────────

    #[test]
    fn default_server() {
        let srv = LongPollServer::default();
        assert_eq!(srv.client_count(), 0);
        assert_eq!(srv.message_count(), 0);
    }
}
