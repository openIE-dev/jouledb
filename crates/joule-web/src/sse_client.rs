//! Server-Sent Events (SSE) client — event parsing, reconnection, dispatch.
//!
//! Replaces `eventsource`, `EventSource`, and `sse.js` with pure Rust.
//! Parses SSE streams (data, event, id, retry fields), tracks last-event-id,
//! reconnection with exponential backoff, event buffering, named event dispatch.

use std::collections::HashMap;
use std::fmt;

// ── SSE event ──────────────────────────────────────────────────

/// A parsed Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Event type (defaults to "message" if not specified).
    pub event_type: String,
    /// Event data (may be multi-line).
    pub data: String,
    /// Event ID (last-event-id).
    pub id: Option<String>,
    /// Retry interval in milliseconds (if server sends `retry:` field).
    pub retry_ms: Option<u64>,
}

impl SseEvent {
    pub fn new(data: &str) -> Self {
        Self {
            event_type: "message".to_string(),
            data: data.to_string(),
            id: None,
            retry_ms: None,
        }
    }

    pub fn with_type(mut self, event_type: &str) -> Self {
        self.event_type = event_type.to_string();
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_retry(mut self, ms: u64) -> Self {
        self.retry_ms = Some(ms);
        self
    }

    /// Serialize to SSE wire format.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        if self.event_type != "message" {
            out.push_str("event: ");
            out.push_str(&self.event_type);
            out.push('\n');
        }
        if let Some(id) = &self.id {
            out.push_str("id: ");
            out.push_str(id);
            out.push('\n');
        }
        if let Some(retry) = self.retry_ms {
            out.push_str("retry: ");
            out.push_str(&retry.to_string());
            out.push('\n');
        }
        for line in self.data.lines() {
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n'); // blank line terminates event
        out
    }
}

impl fmt::Display for SseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SSE[type={}, data_len={}]", self.event_type, self.data.len())
    }
}

// ── SSE parser ─────────────────────────────────────────────────

/// Incremental SSE stream parser.
#[derive(Debug)]
pub struct SseParser {
    buffer: String,
    // Fields being accumulated for the current event.
    current_event_type: Option<String>,
    current_data: Vec<String>,
    current_id: Option<String>,
    current_retry: Option<u64>,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event_type: None,
            current_data: Vec::new(),
            current_id: None,
            current_retry: None,
        }
    }

    /// Feed raw SSE data into the parser, returning any complete events.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            // Find the next line boundary.
            let line_end = if let Some(pos) = self.buffer.find('\n') {
                pos
            } else {
                break;
            };

            let line = self.buffer[..line_end].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[line_end + 1..].to_string();

            if line.is_empty() {
                // Blank line = dispatch event.
                if !self.current_data.is_empty() {
                    let event = SseEvent {
                        event_type: self
                            .current_event_type
                            .take()
                            .unwrap_or_else(|| "message".to_string()),
                        data: self.current_data.join("\n"),
                        id: self.current_id.take(),
                        retry_ms: self.current_retry.take(),
                    };
                    events.push(event);
                    self.current_data.clear();
                } else {
                    // Reset fields even if no data.
                    self.current_event_type = None;
                    self.current_id = None;
                    self.current_retry = None;
                }
                continue;
            }

            // Skip comments.
            if line.starts_with(':') {
                continue;
            }

            // Parse field: value.
            let (field, value) = if let Some(colon) = line.find(':') {
                let f = &line[..colon];
                let v = line[colon + 1..].strip_prefix(' ').unwrap_or(&line[colon + 1..]);
                (f.to_string(), v.to_string())
            } else {
                // Field with no value.
                (line, String::new())
            };

            match field.as_str() {
                "event" => self.current_event_type = Some(value),
                "data" => self.current_data.push(value),
                "id" => self.current_id = Some(value),
                "retry" => {
                    if let Ok(ms) = value.parse::<u64>() {
                        self.current_retry = Some(ms);
                    }
                }
                _ => {} // Unknown fields are ignored per spec.
            }
        }

        events
    }

    /// Reset parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_event_type = None;
        self.current_data.clear();
        self.current_id = None;
        self.current_retry = None;
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Reconnection strategy ──────────────────────────────────────

/// Reconnection backoff strategy.
#[derive(Debug, Clone)]
pub struct ReconnectStrategy {
    pub initial_ms: u64,
    pub max_ms: u64,
    pub multiplier: f64,
    pub current_ms: u64,
    pub attempt: u32,
}

impl ReconnectStrategy {
    pub fn new(initial_ms: u64, max_ms: u64) -> Self {
        Self {
            initial_ms,
            max_ms,
            multiplier: 2.0,
            current_ms: initial_ms,
            attempt: 0,
        }
    }

    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }

    /// Get the next delay and advance the backoff.
    pub fn next_delay(&mut self) -> u64 {
        let delay = self.current_ms;
        self.attempt += 1;
        let next = (self.current_ms as f64 * self.multiplier) as u64;
        self.current_ms = next.min(self.max_ms);
        delay
    }

    /// Reset backoff (e.g. after successful connection).
    pub fn reset(&mut self) {
        self.current_ms = self.initial_ms;
        self.attempt = 0;
    }

    /// Update from a server-sent `retry:` field.
    pub fn set_from_server(&mut self, retry_ms: u64) {
        self.initial_ms = retry_ms;
        self.current_ms = retry_ms;
    }
}

impl Default for ReconnectStrategy {
    fn default() -> Self {
        Self::new(1000, 30_000)
    }
}

// ── Connection state ───────────────────────────────────────────

/// SSE connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Open,
    Closed,
}

// ── Event dispatcher ───────────────────────────────────────────

/// Tracks event handlers by event type and dispatches events.
/// Since we can't store closures without Box<dyn>, we store handler IDs
/// and let the caller match on them.
#[derive(Debug, Default)]
pub struct EventDispatcher {
    /// Map from event type to list of handler names.
    handlers: HashMap<String, Vec<String>>,
    /// Buffered events (for replay or offline handling).
    event_buffer: Vec<SseEvent>,
    pub buffer_limit: usize,
}

impl EventDispatcher {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            event_buffer: Vec::new(),
            buffer_limit: 1000,
        }
    }

    /// Register a named handler for an event type.
    pub fn on(&mut self, event_type: &str, handler_name: &str) {
        self.handlers
            .entry(event_type.to_string())
            .or_default()
            .push(handler_name.to_string());
    }

    /// Remove a handler.
    pub fn off(&mut self, event_type: &str, handler_name: &str) {
        if let Some(handlers) = self.handlers.get_mut(event_type) {
            handlers.retain(|h| h != handler_name);
        }
    }

    /// Get handler names that should handle this event.
    pub fn dispatch(&self, event: &SseEvent) -> Vec<&str> {
        self.handlers
            .get(&event.event_type)
            .map(|hs| hs.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Buffer an event.
    pub fn buffer_event(&mut self, event: SseEvent) {
        if self.event_buffer.len() >= self.buffer_limit {
            self.event_buffer.remove(0);
        }
        self.event_buffer.push(event);
    }

    pub fn buffered_events(&self) -> &[SseEvent] {
        &self.event_buffer
    }

    pub fn clear_buffer(&mut self) {
        self.event_buffer.clear();
    }

    pub fn handler_count(&self, event_type: &str) -> usize {
        self.handlers.get(event_type).map_or(0, |h| h.len())
    }
}

// ── SSE client state ───────────────────────────────────────────

/// SSE client state machine (no actual I/O).
#[derive(Debug)]
pub struct SseClient {
    pub url: String,
    pub state: ConnectionState,
    pub last_event_id: Option<String>,
    pub parser: SseParser,
    pub reconnect: ReconnectStrategy,
    pub dispatcher: EventDispatcher,
}

impl SseClient {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            state: ConnectionState::Connecting,
            last_event_id: None,
            parser: SseParser::new(),
            reconnect: ReconnectStrategy::default(),
            dispatcher: EventDispatcher::new(),
        }
    }

    /// Process incoming SSE data chunk. Returns parsed events.
    pub fn on_data(&mut self, chunk: &str) -> Vec<SseEvent> {
        if self.state != ConnectionState::Open {
            self.state = ConnectionState::Open;
            self.reconnect.reset();
        }

        let events = self.parser.feed(chunk);

        for event in &events {
            // Track last-event-id.
            if let Some(id) = &event.id {
                self.last_event_id = Some(id.clone());
            }
            // Update retry from server.
            if let Some(retry_ms) = event.retry_ms {
                self.reconnect.set_from_server(retry_ms);
            }
            // Buffer event.
            self.dispatcher.buffer_event(event.clone());
        }

        events
    }

    /// Signal that the connection was lost.
    pub fn on_disconnect(&mut self) {
        self.state = ConnectionState::Closed;
        self.parser.reset();
    }

    /// Get the reconnect delay for the next attempt.
    pub fn reconnect_delay(&mut self) -> u64 {
        self.state = ConnectionState::Connecting;
        self.reconnect.next_delay()
    }

    /// Get headers to send on reconnect (includes Last-Event-ID).
    pub fn reconnect_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        headers.push(("Accept".to_string(), "text/event-stream".to_string()));
        headers.push(("Cache-Control".to_string(), "no-cache".to_string()));
        if let Some(id) = &self.last_event_id {
            headers.push(("Last-Event-ID".to_string(), id.clone()));
        }
        headers
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn parse_named_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: update\ndata: {\"key\":\"val\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "update");
    }

    #[test]
    fn parse_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn parse_event_with_id() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 42\ndata: test\n\n");
        assert_eq!(events[0].id.as_deref(), Some("42"));
    }

    #[test]
    fn parse_retry_field() {
        let mut parser = SseParser::new();
        let events = parser.feed("retry: 5000\ndata: x\n\n");
        assert_eq!(events[0].retry_ms, Some(5000));
    }

    #[test]
    fn parse_comments_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: real data\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "real data");
    }

    #[test]
    fn parse_incremental_feed() {
        let mut parser = SseParser::new();
        // Feed partial data.
        let e1 = parser.feed("data: hel");
        assert!(e1.is_empty());
        let e2 = parser.feed("lo\n\n");
        assert_eq!(e2.len(), 1);
        assert_eq!(e2[0].data, "hello");
    }

    #[test]
    fn parse_multiple_events() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn event_serialization() {
        let event = SseEvent::new("hello world")
            .with_type("chat")
            .with_id("1");
        let serialized = event.serialize();
        assert!(serialized.contains("event: chat\n"));
        assert!(serialized.contains("id: 1\n"));
        assert!(serialized.contains("data: hello world\n"));
        assert!(serialized.ends_with("\n\n"));

        // Round-trip through parser.
        let mut parser = SseParser::new();
        let parsed = parser.feed(&serialized);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].event_type, "chat");
        assert_eq!(parsed[0].data, "hello world");
    }

    #[test]
    fn reconnect_backoff() {
        let mut strategy = ReconnectStrategy::new(100, 5000);
        assert_eq!(strategy.next_delay(), 100);
        assert_eq!(strategy.next_delay(), 200);
        assert_eq!(strategy.next_delay(), 400);
        assert_eq!(strategy.next_delay(), 800);
        assert_eq!(strategy.next_delay(), 1600);
        assert_eq!(strategy.next_delay(), 3200);
        assert_eq!(strategy.next_delay(), 5000); // capped
        assert_eq!(strategy.next_delay(), 5000); // stays capped
    }

    #[test]
    fn reconnect_reset() {
        let mut strategy = ReconnectStrategy::new(100, 5000);
        strategy.next_delay();
        strategy.next_delay();
        strategy.reset();
        assert_eq!(strategy.next_delay(), 100);
    }

    #[test]
    fn reconnect_server_override() {
        let mut strategy = ReconnectStrategy::new(100, 5000);
        strategy.next_delay(); // 100
        strategy.next_delay(); // 200
        strategy.set_from_server(500);
        assert_eq!(strategy.next_delay(), 500);
    }

    #[test]
    fn event_dispatcher_basic() {
        let mut disp = EventDispatcher::new();
        disp.on("message", "logger");
        disp.on("message", "counter");
        disp.on("error", "error_handler");

        let event = SseEvent::new("test");
        let handlers = disp.dispatch(&event);
        assert_eq!(handlers.len(), 2);
        assert!(handlers.contains(&"logger"));

        disp.off("message", "logger");
        assert_eq!(disp.handler_count("message"), 1);
    }

    #[test]
    fn event_buffer_limit() {
        let mut disp = EventDispatcher::new();
        disp.buffer_limit = 3;
        for i in 0..5 {
            disp.buffer_event(SseEvent::new(&format!("event {i}")));
        }
        assert_eq!(disp.buffered_events().len(), 3);
        assert_eq!(disp.buffered_events()[0].data, "event 2");
    }

    #[test]
    fn sse_client_on_data() {
        let mut client = SseClient::new("https://api.example.com/events");
        let events = client.on_data("id: 1\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(client.state, ConnectionState::Open);
        assert_eq!(client.last_event_id.as_deref(), Some("1"));
    }

    #[test]
    fn sse_client_reconnect_headers() {
        let mut client = SseClient::new("https://example.com/sse");
        client.last_event_id = Some("42".to_string());
        let headers = client.reconnect_headers();
        let last_id = headers.iter().find(|(k, _)| k == "Last-Event-ID");
        assert_eq!(last_id.unwrap().1, "42");
    }

    #[test]
    fn sse_client_disconnect_reconnect() {
        let mut client = SseClient::new("https://example.com/sse");
        client.on_data("data: x\n\n");
        assert_eq!(client.state, ConnectionState::Open);
        client.on_disconnect();
        assert_eq!(client.state, ConnectionState::Closed);
        let delay = client.reconnect_delay();
        assert_eq!(delay, 1000); // default initial
        assert_eq!(client.state, ConnectionState::Connecting);
    }
}
