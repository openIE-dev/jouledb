//! Server-Sent Events (SSE) — event formatting, event IDs, retry fields,
//! named events, multi-line data, event stream builder, reconnection modeling.
//!
//! Pure-Rust replacement for eventsource-parser, sse.js, etc.

use std::fmt;
use std::collections::VecDeque;

// ── Event ─────────────────────────────────────────────────────────

/// A single SSE event.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    /// Optional event type (maps to the `event:` field).
    pub event_type: Option<String>,
    /// The data payload (may contain newlines).
    pub data: String,
    /// Optional event ID (maps to the `id:` field).
    pub id: Option<String>,
    /// Optional retry interval in milliseconds (maps to `retry:` field).
    pub retry: Option<u64>,
    /// Optional comment lines (prefixed with `:` in the wire format).
    pub comments: Vec<String>,
}

impl Default for SseEvent {
    fn default() -> Self {
        Self {
            event_type: None,
            data: String::new(),
            id: None,
            retry: None,
            comments: Vec::new(),
        }
    }
}

impl SseEvent {
    pub fn new(data: &str) -> Self {
        Self { data: data.into(), ..Default::default() }
    }

    pub fn with_type(mut self, event_type: &str) -> Self {
        self.event_type = Some(event_type.into());
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_retry(mut self, ms: u64) -> Self {
        self.retry = Some(ms);
        self
    }

    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comments.push(comment.into());
        self
    }

    /// Format this event to the SSE wire format.
    pub fn to_wire(&self) -> String {
        let mut out = String::new();
        for c in &self.comments {
            for line in c.lines() {
                out.push_str(": ");
                out.push_str(line);
                out.push('\n');
            }
        }
        if let Some(ref et) = self.event_type {
            out.push_str("event: ");
            out.push_str(et);
            out.push('\n');
        }
        if let Some(ref id) = self.id {
            out.push_str("id: ");
            out.push_str(id);
            out.push('\n');
        }
        if let Some(retry) = self.retry {
            out.push_str("retry: ");
            out.push_str(&retry.to_string());
            out.push('\n');
        }
        for line in self.data.lines() {
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }
        if self.data.is_empty() {
            out.push_str("data: \n");
        }
        out.push('\n');
        out
    }
}

impl fmt::Display for SseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_wire())
    }
}

// ── Parser ────────────────────────────────────────────────────────

/// Parser state for an incoming SSE stream.
#[derive(Debug, Clone)]
pub struct SseParser {
    buffer: String,
    current_event_type: Option<String>,
    current_data: Vec<String>,
    current_id: Option<String>,
    current_retry: Option<u64>,
    current_comments: Vec<String>,
    last_event_id: Option<String>,
    reconnection_time: u64,
}

impl Default for SseParser {
    fn default() -> Self {
        Self {
            buffer: String::new(),
            current_event_type: None,
            current_data: Vec::new(),
            current_id: None,
            current_retry: None,
            current_comments: Vec::new(),
            last_event_id: None,
            reconnection_time: 3000,
        }
    }
}

impl SseParser {
    pub fn new() -> Self { Self::default() }

    /// Feed a chunk of bytes into the parser, returning any complete events.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            let newline_pos = match self.buffer.find('\n') {
                Some(p) => p,
                None => break,
            };
            let line = self.buffer[..newline_pos].to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            let line = if line.ends_with('\r') {
                &line[..line.len() - 1]
            } else {
                &line
            };

            if line.is_empty() {
                if let Some(event) = self.dispatch() {
                    events.push(event);
                }
            } else {
                self.process_line(line);
            }
        }
        events
    }

    fn process_line(&mut self, line: &str) {
        if let Some(rest) = line.strip_prefix(':') {
            let comment = rest.strip_prefix(' ').unwrap_or(rest);
            self.current_comments.push(comment.to_string());
        } else if let Some(idx) = line.find(':') {
            let field = &line[..idx];
            let value = line[idx + 1..].strip_prefix(' ').unwrap_or(&line[idx + 1..]);
            self.process_field(field, value);
        } else {
            self.process_field(line, "");
        }
    }

    fn process_field(&mut self, field: &str, value: &str) {
        match field {
            "event" => self.current_event_type = Some(value.to_string()),
            "data" => self.current_data.push(value.to_string()),
            "id" => {
                if !value.contains('\0') {
                    self.current_id = Some(value.to_string());
                }
            }
            "retry" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.current_retry = Some(ms);
                    self.reconnection_time = ms;
                }
            }
            _ => {}
        }
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if self.current_data.is_empty() && self.current_event_type.is_none()
            && self.current_id.is_none() && self.current_retry.is_none()
            && self.current_comments.is_empty()
        {
            return None;
        }

        let data = self.current_data.join("\n");
        let event = SseEvent {
            event_type: self.current_event_type.take(),
            data,
            id: self.current_id.take(),
            retry: self.current_retry.take(),
            comments: std::mem::take(&mut self.current_comments),
        };

        if let Some(ref id) = event.id {
            self.last_event_id = Some(id.clone());
        }
        self.current_data.clear();
        Some(event)
    }

    /// The last event ID seen (for `Last-Event-ID` header on reconnect).
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// The current reconnection time in milliseconds.
    pub fn reconnection_time(&self) -> u64 {
        self.reconnection_time
    }
}

// ── Stream builder ────────────────────────────────────────────────

/// Builds an SSE stream from a sequence of events.
#[derive(Debug, Clone)]
pub struct SseStream {
    events: VecDeque<SseEvent>,
    next_id: u64,
    auto_id: bool,
}

impl Default for SseStream {
    fn default() -> Self {
        Self { events: VecDeque::new(), next_id: 1, auto_id: false }
    }
}

impl SseStream {
    pub fn new() -> Self { Self::default() }

    /// Enable auto-generated sequential IDs.
    pub fn with_auto_id(mut self) -> Self {
        self.auto_id = true;
        self
    }

    /// Push an event to the stream.
    pub fn push(&mut self, mut event: SseEvent) {
        if self.auto_id && event.id.is_none() {
            event.id = Some(self.next_id.to_string());
            self.next_id += 1;
        }
        self.events.push_back(event);
    }

    /// Pop the next event.
    pub fn pop(&mut self) -> Option<SseEvent> {
        self.events.pop_front()
    }

    /// Number of pending events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the stream is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Serialize all pending events to the SSE wire format.
    pub fn to_wire(&self) -> String {
        let mut out = String::new();
        for event in &self.events {
            out.push_str(&event.to_wire());
        }
        out
    }

    /// Drain all events, returning the wire-format string.
    pub fn drain_to_wire(&mut self) -> String {
        let mut out = String::new();
        while let Some(event) = self.events.pop_front() {
            out.push_str(&event.to_wire());
        }
        out
    }
}

// ── Reconnection model ───────────────────────────────────────────

/// Models exponential backoff for SSE reconnection.
#[derive(Debug, Clone)]
pub struct ReconnectionPolicy {
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub multiplier: f64,
    pub attempts: u32,
    current_delay_ms: u64,
}

impl ReconnectionPolicy {
    pub fn new(base_delay_ms: u64, max_delay_ms: u64, multiplier: f64) -> Self {
        Self {
            base_delay_ms,
            max_delay_ms,
            multiplier,
            attempts: 0,
            current_delay_ms: base_delay_ms,
        }
    }

    pub fn default_policy() -> Self {
        Self::new(1000, 30000, 2.0)
    }

    /// Record a failed attempt and return the delay before next retry.
    pub fn next_delay(&mut self) -> u64 {
        let delay = self.current_delay_ms;
        self.attempts += 1;
        let next = (self.current_delay_ms as f64 * self.multiplier) as u64;
        self.current_delay_ms = next.min(self.max_delay_ms);
        delay
    }

    /// Reset after a successful connection.
    pub fn reset(&mut self) {
        self.attempts = 0;
        self.current_delay_ms = self.base_delay_ms;
    }

    /// Whether more retries should be attempted (always true for SSE).
    pub fn should_retry(&self) -> bool {
        true
    }
}

// ── Keep-alive ────────────────────────────────────────────────────

/// Generates SSE comment-based keep-alive messages.
pub fn keepalive_comment() -> String {
    ": keepalive\n\n".to_string()
}

/// Generate a comment with a custom message.
pub fn comment_line(msg: &str) -> String {
    let mut out = String::new();
    for line in msg.lines() {
        out.push_str(": ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_event() {
        let e = SseEvent::new("hello");
        let wire = e.to_wire();
        assert_eq!(wire, "data: hello\n\n");
    }

    #[test]
    fn event_with_type() {
        let e = SseEvent::new("payload").with_type("update");
        let wire = e.to_wire();
        assert!(wire.contains("event: update\n"));
        assert!(wire.contains("data: payload\n"));
    }

    #[test]
    fn event_with_id() {
        let e = SseEvent::new("data").with_id("42");
        let wire = e.to_wire();
        assert!(wire.contains("id: 42\n"));
    }

    #[test]
    fn event_with_retry() {
        let e = SseEvent::new("data").with_retry(5000);
        let wire = e.to_wire();
        assert!(wire.contains("retry: 5000\n"));
    }

    #[test]
    fn event_with_comment() {
        let e = SseEvent::new("data").with_comment("test comment");
        let wire = e.to_wire();
        assert!(wire.starts_with(": test comment\n"));
    }

    #[test]
    fn multiline_data() {
        let e = SseEvent::new("line1\nline2\nline3");
        let wire = e.to_wire();
        assert!(wire.contains("data: line1\n"));
        assert!(wire.contains("data: line2\n"));
        assert!(wire.contains("data: line3\n"));
    }

    #[test]
    fn empty_data_event() {
        let e = SseEvent::new("");
        let wire = e.to_wire();
        assert!(wire.contains("data: \n"));
    }

    #[test]
    fn full_event_wire() {
        let e = SseEvent::new("json data")
            .with_type("message")
            .with_id("1")
            .with_retry(3000)
            .with_comment("ping");
        let wire = e.to_wire();
        let expected = ": ping\nevent: message\nid: 1\nretry: 3000\ndata: json data\n\n";
        assert_eq!(wire, expected);
    }

    #[test]
    fn display_trait() {
        let e = SseEvent::new("test");
        assert_eq!(format!("{e}"), e.to_wire());
    }

    #[test]
    fn parse_simple_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn parse_typed_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: update\ndata: payload\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type.as_deref(), Some("update"));
        assert_eq!(events[0].data, "payload");
    }

    #[test]
    fn parse_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn parse_event_id() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 42\ndata: x\n\n");
        assert_eq!(events[0].id.as_deref(), Some("42"));
        assert_eq!(parser.last_event_id(), Some("42"));
    }

    #[test]
    fn parse_retry() {
        let mut parser = SseParser::new();
        let events = parser.feed("retry: 5000\ndata: x\n\n");
        assert_eq!(events[0].retry, Some(5000));
        assert_eq!(parser.reconnection_time(), 5000);
    }

    #[test]
    fn parse_invalid_retry_ignored() {
        let mut parser = SseParser::new();
        parser.feed("retry: abc\ndata: x\n\n");
        assert_eq!(parser.reconnection_time(), 3000);
    }

    #[test]
    fn parse_comment() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: x\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].comments, vec!["this is a comment"]);
    }

    #[test]
    fn parse_chunked_input() {
        let mut parser = SseParser::new();
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
    fn parse_id_with_null_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: bad\0id\ndata: x\n\n");
        assert!(events[0].id.is_none());
    }

    #[test]
    fn parse_unknown_field_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed("unknown: value\ndata: x\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn parse_field_no_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed("data\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn parse_crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn stream_builder_basic() {
        let mut stream = SseStream::new();
        stream.push(SseEvent::new("one"));
        stream.push(SseEvent::new("two"));
        assert_eq!(stream.len(), 2);
        assert!(!stream.is_empty());
        let wire = stream.to_wire();
        assert!(wire.contains("data: one\n"));
        assert!(wire.contains("data: two\n"));
    }

    #[test]
    fn stream_auto_id() {
        let mut stream = SseStream::new().with_auto_id();
        stream.push(SseEvent::new("a"));
        stream.push(SseEvent::new("b"));
        let e1 = stream.pop().unwrap();
        let e2 = stream.pop().unwrap();
        assert_eq!(e1.id.as_deref(), Some("1"));
        assert_eq!(e2.id.as_deref(), Some("2"));
    }

    #[test]
    fn stream_auto_id_preserves_explicit() {
        let mut stream = SseStream::new().with_auto_id();
        stream.push(SseEvent::new("a").with_id("custom"));
        let e = stream.pop().unwrap();
        assert_eq!(e.id.as_deref(), Some("custom"));
    }

    #[test]
    fn stream_drain() {
        let mut stream = SseStream::new();
        stream.push(SseEvent::new("x"));
        stream.push(SseEvent::new("y"));
        let wire = stream.drain_to_wire();
        assert!(stream.is_empty());
        assert!(wire.contains("data: x\n"));
        assert!(wire.contains("data: y\n"));
    }

    #[test]
    fn reconnection_backoff() {
        let mut policy = ReconnectionPolicy::new(100, 1000, 2.0);
        assert_eq!(policy.next_delay(), 100);
        assert_eq!(policy.attempts, 1);
        assert_eq!(policy.next_delay(), 200);
        assert_eq!(policy.next_delay(), 400);
        assert_eq!(policy.next_delay(), 800);
        assert_eq!(policy.next_delay(), 1000);
        assert_eq!(policy.next_delay(), 1000);
    }

    #[test]
    fn reconnection_reset() {
        let mut policy = ReconnectionPolicy::new(100, 1000, 2.0);
        policy.next_delay();
        policy.next_delay();
        policy.reset();
        assert_eq!(policy.attempts, 0);
        assert_eq!(policy.next_delay(), 100);
    }

    #[test]
    fn reconnection_default() {
        let policy = ReconnectionPolicy::default_policy();
        assert_eq!(policy.base_delay_ms, 1000);
        assert_eq!(policy.max_delay_ms, 30000);
        assert!(policy.should_retry());
    }

    #[test]
    fn keepalive() {
        let ka = keepalive_comment();
        assert_eq!(ka, ": keepalive\n\n");
    }

    #[test]
    fn custom_comment_line() {
        let c = comment_line("health check");
        assert_eq!(c, ": health check\n\n");
    }

    #[test]
    fn multiline_comment() {
        let c = comment_line("line1\nline2");
        assert_eq!(c, ": line1\n: line2\n\n");
    }

    #[test]
    fn roundtrip_event() {
        let original = SseEvent::new("test data")
            .with_type("msg")
            .with_id("7");
        let wire = original.to_wire();
        let mut parser = SseParser::new();
        let events = parser.feed(&wire);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "test data");
        assert_eq!(events[0].event_type.as_deref(), Some("msg"));
        assert_eq!(events[0].id.as_deref(), Some("7"));
    }

    #[test]
    fn roundtrip_multiline() {
        let original = SseEvent::new("a\nb\nc");
        let wire = original.to_wire();
        let mut parser = SseParser::new();
        let events = parser.feed(&wire);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb\nc");
    }

    #[test]
    fn last_event_id_tracks_across_events() {
        let mut parser = SseParser::new();
        parser.feed("id: 1\ndata: a\n\n");
        assert_eq!(parser.last_event_id(), Some("1"));
        parser.feed("data: b\n\n");
        assert_eq!(parser.last_event_id(), Some("1"));
        parser.feed("id: 5\ndata: c\n\n");
        assert_eq!(parser.last_event_id(), Some("5"));
    }

    #[test]
    fn empty_blank_lines_no_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("\n\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn json_data_in_event() {
        let json_str = r#"{"type":"update","value":42}"#;
        let e = SseEvent::new(json_str).with_type("json");
        let wire = e.to_wire();
        let mut parser = SseParser::new();
        let events = parser.feed(&wire);
        assert_eq!(events[0].data, json_str);
        let parsed: serde_json::Value = serde_json::from_str(&events[0].data).unwrap();
        assert_eq!(parsed["value"], 42);
    }
}
