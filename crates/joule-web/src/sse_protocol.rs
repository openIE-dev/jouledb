//! Server-Sent Events protocol — event formatting (data/event/id/retry fields),
//! event stream builder, last-event-id tracking, reconnection logic, event
//! filtering, multi-line data, and event parsing.
//!
//! Replaces `eventsource`, `event-source-polyfill`, and similar JS SSE libraries
//! with a pure-Rust SSE protocol implementation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// SSE protocol error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseError {
    /// Invalid event data (e.g., contains null byte).
    InvalidData(String),
    /// Empty event type.
    EmptyEventType,
    /// Invalid retry value.
    InvalidRetry(String),
    /// Parse error in event stream.
    ParseError(String),
    /// Stream ended unexpectedly.
    StreamEnded,
    /// Maximum reconnection attempts exceeded.
    MaxReconnectExceeded { attempts: u32, max: u32 },
}

impl fmt::Display for SseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(msg) => write!(f, "invalid SSE data: {msg}"),
            Self::EmptyEventType => write!(f, "empty event type"),
            Self::InvalidRetry(v) => write!(f, "invalid retry value: {v}"),
            Self::ParseError(msg) => write!(f, "SSE parse error: {msg}"),
            Self::StreamEnded => write!(f, "SSE stream ended"),
            Self::MaxReconnectExceeded { attempts, max } => {
                write!(f, "max reconnect exceeded: {attempts}/{max}")
            }
        }
    }
}

impl std::error::Error for SseError {}

// ── Event ────────────────────────────────────────────────────────

/// A single Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SseEvent {
    /// Event ID (optional).
    pub id: Option<String>,
    /// Event type (defaults to "message" if not set).
    pub event_type: Option<String>,
    /// Event data (can be multi-line).
    pub data: String,
    /// Retry interval in milliseconds (optional).
    pub retry: Option<u64>,
    /// Comment (prefixed with ":" in the wire format).
    pub comment: Option<String>,
}

impl SseEvent {
    /// Create a new event with data only.
    pub fn new(data: &str) -> Self {
        Self {
            id: None,
            event_type: None,
            data: data.to_string(),
            retry: None,
            comment: None,
        }
    }

    /// Set the event ID.
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Set the event type.
    pub fn with_event_type(mut self, event_type: &str) -> Self {
        self.event_type = Some(event_type.to_string());
        self
    }

    /// Set the retry interval.
    pub fn with_retry(mut self, retry_ms: u64) -> Self {
        self.retry = Some(retry_ms);
        self
    }

    /// Set a comment.
    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comment = Some(comment.to_string());
        self
    }

    /// Validate the event.
    pub fn validate(&self) -> Result<(), SseError> {
        if self.data.contains('\0') {
            return Err(SseError::InvalidData("data contains null byte".to_string()));
        }
        if let Some(et) = &self.event_type {
            if et.is_empty() {
                return Err(SseError::EmptyEventType);
            }
            if et.contains('\n') || et.contains('\r') {
                return Err(SseError::InvalidData(
                    "event type contains newline".to_string(),
                ));
            }
        }
        if let Some(id) = &self.id {
            if id.contains('\0') {
                return Err(SseError::InvalidData(
                    "id contains null byte".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Format this event as an SSE wire-format string.
    pub fn format(&self) -> Result<String, SseError> {
        self.validate()?;
        let mut out = String::new();

        if let Some(comment) = &self.comment {
            for line in comment.lines() {
                out.push_str(&format!(": {line}\n"));
            }
        }

        if let Some(id) = &self.id {
            out.push_str(&format!("id: {id}\n"));
        }

        if let Some(event_type) = &self.event_type {
            out.push_str(&format!("event: {event_type}\n"));
        }

        if let Some(retry) = self.retry {
            out.push_str(&format!("retry: {retry}\n"));
        }

        // Multi-line data: each line gets its own "data:" field.
        for line in self.data.split('\n') {
            out.push_str(&format!("data: {line}\n"));
        }

        // Blank line terminates the event.
        out.push('\n');
        Ok(out)
    }

    /// The effective event type (defaults to "message").
    pub fn effective_type(&self) -> &str {
        self.event_type.as_deref().unwrap_or("message")
    }
}

impl fmt::Display for SseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.format() {
            Ok(s) => f.write_str(&s),
            Err(e) => write!(f, "<invalid event: {e}>"),
        }
    }
}

// ── Event Parser ─────────────────────────────────────────────────

/// Parse an SSE event stream into individual events.
pub fn parse_event_stream(input: &str) -> Result<Vec<SseEvent>, SseError> {
    let mut events = Vec::new();
    let mut current_data = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_event_type: Option<String> = None;
    let mut current_retry: Option<u64> = None;
    let mut current_comment: Option<String> = None;

    for line in input.lines() {
        if line.is_empty() {
            // Blank line = dispatch event.
            if !current_data.is_empty()
                || current_id.is_some()
                || current_event_type.is_some()
                || current_comment.is_some()
            {
                let data = current_data.join("\n");
                events.push(SseEvent {
                    id: current_id.take(),
                    event_type: current_event_type.take(),
                    data,
                    retry: current_retry.take(),
                    comment: current_comment.take(),
                });
                current_data.clear();
            }
            continue;
        }

        if let Some(comment_text) = line.strip_prefix(": ") {
            let existing = current_comment.get_or_insert_with(String::new);
            if !existing.is_empty() {
                existing.push('\n');
            }
            existing.push_str(comment_text);
        } else if line.starts_with(':') {
            // Comment with no space after colon
            let text = &line[1..];
            let trimmed = text.strip_prefix(' ').unwrap_or(text);
            let existing = current_comment.get_or_insert_with(String::new);
            if !existing.is_empty() {
                existing.push('\n');
            }
            existing.push_str(trimmed);
        } else if let Some(value) = line.strip_prefix("data: ") {
            current_data.push(value.to_string());
        } else if line == "data:" || line == "data" {
            current_data.push(String::new());
        } else if let Some(value) = line.strip_prefix("id: ") {
            current_id = Some(value.to_string());
        } else if line == "id:" || line == "id" {
            current_id = Some(String::new());
        } else if let Some(value) = line.strip_prefix("event: ") {
            current_event_type = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("retry: ") {
            match value.parse::<u64>() {
                Ok(ms) => current_retry = Some(ms),
                Err(_) => {
                    return Err(SseError::InvalidRetry(value.to_string()));
                }
            }
        }
        // Unknown field names are ignored per spec.
    }

    // If there's data remaining without a trailing blank line, still dispatch it.
    if !current_data.is_empty()
        || current_id.is_some()
        || current_event_type.is_some()
    {
        let data = current_data.join("\n");
        events.push(SseEvent {
            id: current_id,
            event_type: current_event_type,
            data,
            retry: current_retry,
            comment: current_comment,
        });
    }

    Ok(events)
}

// ── Event Stream Builder ─────────────────────────────────────────

/// Builder for creating an SSE event stream.
#[derive(Debug, Clone)]
pub struct EventStreamBuilder {
    events: Vec<SseEvent>,
}

impl EventStreamBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event.
    pub fn event(mut self, event: SseEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Add a data-only event.
    pub fn data(self, data: &str) -> Self {
        self.event(SseEvent::new(data))
    }

    /// Add a comment-only line (keep-alive).
    pub fn comment(self, comment: &str) -> Self {
        self.event(SseEvent {
            id: None,
            event_type: None,
            data: String::new(),
            retry: None,
            comment: Some(comment.to_string()),
        })
    }

    /// Build the full stream as a string.
    pub fn build(&self) -> Result<String, SseError> {
        let mut out = String::new();
        for event in &self.events {
            // Comments without data should still format correctly
            if event.data.is_empty() && event.comment.is_some() {
                if let Some(comment) = &event.comment {
                    for line in comment.lines() {
                        out.push_str(&format!(": {line}\n"));
                    }
                    out.push('\n');
                }
            } else {
                out.push_str(&event.format()?);
            }
        }
        Ok(out)
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the builder has no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for EventStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Reconnection Tracker ─────────────────────────────────────────

/// Tracks reconnection state for an SSE client.
#[derive(Debug, Clone)]
pub struct ReconnectionTracker {
    /// Current retry interval in milliseconds.
    pub retry_ms: u64,
    /// Default retry interval.
    pub default_retry_ms: u64,
    /// Maximum retry interval (backoff cap).
    pub max_retry_ms: u64,
    /// Maximum reconnection attempts (0 = unlimited).
    pub max_attempts: u32,
    /// Current attempt count.
    pub attempts: u32,
    /// Last event ID received.
    pub last_event_id: Option<String>,
    /// Backoff multiplier (e.g. 2.0 for exponential backoff).
    pub backoff_factor: f64,
}

impl ReconnectionTracker {
    /// Create a tracker with default settings.
    pub fn new(default_retry_ms: u64) -> Self {
        Self {
            retry_ms: default_retry_ms,
            default_retry_ms,
            max_retry_ms: 30_000,
            max_attempts: 0,
            attempts: 0,
            last_event_id: None,
            backoff_factor: 2.0,
        }
    }

    /// Set the max attempts (0 = unlimited).
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Set the max retry interval.
    pub fn with_max_retry(mut self, ms: u64) -> Self {
        self.max_retry_ms = ms;
        self
    }

    /// Set the backoff factor.
    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Record a successful connection (reset attempts).
    pub fn on_connected(&mut self) {
        self.attempts = 0;
        self.retry_ms = self.default_retry_ms;
    }

    /// Record a received event.
    pub fn on_event(&mut self, event: &SseEvent) {
        if let Some(id) = &event.id {
            self.last_event_id = Some(id.clone());
        }
        if let Some(retry) = event.retry {
            self.retry_ms = retry;
        }
    }

    /// Record a disconnection. Returns the delay before reconnecting.
    pub fn on_disconnect(&mut self) -> Result<u64, SseError> {
        self.attempts += 1;
        if self.max_attempts > 0 && self.attempts > self.max_attempts {
            return Err(SseError::MaxReconnectExceeded {
                attempts: self.attempts,
                max: self.max_attempts,
            });
        }
        let delay = self.retry_ms;
        let next = (self.retry_ms as f64 * self.backoff_factor) as u64;
        self.retry_ms = next.min(self.max_retry_ms);
        Ok(delay)
    }

    /// Get the Last-Event-ID header value for reconnection.
    pub fn last_event_id_header(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Whether we should reconnect.
    pub fn should_reconnect(&self) -> bool {
        self.max_attempts == 0 || self.attempts < self.max_attempts
    }
}

// ── Event Filter ─────────────────────────────────────────────────

/// Filter events by type.
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Allowed event types. Empty means allow all.
    allowed_types: Vec<String>,
    /// Denied event types.
    denied_types: Vec<String>,
}

impl EventFilter {
    /// Create a filter that allows all events.
    pub fn allow_all() -> Self {
        Self {
            allowed_types: Vec::new(),
            denied_types: Vec::new(),
        }
    }

    /// Create a filter that only allows specific types.
    pub fn allow_only(types: &[&str]) -> Self {
        Self {
            allowed_types: types.iter().map(|s| s.to_string()).collect(),
            denied_types: Vec::new(),
        }
    }

    /// Create a filter that denies specific types.
    pub fn deny(types: &[&str]) -> Self {
        Self {
            allowed_types: Vec::new(),
            denied_types: types.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Check if an event passes the filter.
    pub fn accepts(&self, event: &SseEvent) -> bool {
        let et = event.effective_type().to_string();
        if !self.denied_types.is_empty() && self.denied_types.contains(&et) {
            return false;
        }
        if !self.allowed_types.is_empty() && !self.allowed_types.contains(&et) {
            return false;
        }
        true
    }

    /// Filter a list of events.
    pub fn filter_events(&self, events: &[SseEvent]) -> Vec<SseEvent> {
        events.iter().filter(|e| self.accepts(e)).cloned().collect()
    }
}

// ── Event History ────────────────────────────────────────────────

/// Tracks event history for replay on reconnection.
#[derive(Debug, Clone)]
pub struct EventHistory {
    events: Vec<SseEvent>,
    max_size: usize,
    id_index: HashMap<String, usize>,
}

impl EventHistory {
    /// Create a history buffer with max size.
    pub fn new(max_size: usize) -> Self {
        Self {
            events: Vec::new(),
            max_size,
            id_index: HashMap::new(),
        }
    }

    /// Record an event.
    pub fn push(&mut self, event: SseEvent) {
        if let Some(id) = &event.id {
            self.id_index.insert(id.clone(), self.events.len());
        }
        self.events.push(event);
        // Trim if over capacity.
        while self.events.len() > self.max_size {
            self.events.remove(0);
            // Rebuild index after removal.
            self.id_index.clear();
            for (i, ev) in self.events.iter().enumerate() {
                if let Some(id) = &ev.id {
                    self.id_index.insert(id.clone(), i);
                }
            }
        }
    }

    /// Get events since the given last-event-id.
    pub fn since(&self, last_event_id: &str) -> Vec<SseEvent> {
        if let Some(&idx) = self.id_index.get(last_event_id) {
            if idx + 1 < self.events.len() {
                return self.events[idx + 1..].to_vec();
            }
            return Vec::new();
        }
        // ID not found — return all events.
        self.events.clone()
    }

    /// Number of events in history.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.events.clear();
        self.id_index.clear();
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_event_format() {
        let event = SseEvent::new("hello world");
        let formatted = event.format().unwrap();
        assert!(formatted.contains("data: hello world\n"));
        assert!(formatted.ends_with("\n\n"));
    }

    #[test]
    fn test_event_with_all_fields() {
        let event = SseEvent::new("payload")
            .with_id("42")
            .with_event_type("update")
            .with_retry(5000);
        let formatted = event.format().unwrap();
        assert!(formatted.contains("id: 42\n"));
        assert!(formatted.contains("event: update\n"));
        assert!(formatted.contains("retry: 5000\n"));
        assert!(formatted.contains("data: payload\n"));
    }

    #[test]
    fn test_multiline_data() {
        let event = SseEvent::new("line1\nline2\nline3");
        let formatted = event.format().unwrap();
        assert!(formatted.contains("data: line1\n"));
        assert!(formatted.contains("data: line2\n"));
        assert!(formatted.contains("data: line3\n"));
    }

    #[test]
    fn test_event_with_comment() {
        let event = SseEvent::new("data").with_comment("keep-alive");
        let formatted = event.format().unwrap();
        assert!(formatted.contains(": keep-alive\n"));
    }

    #[test]
    fn test_validate_null_byte() {
        let event = SseEvent::new("bad\0data");
        assert!(matches!(event.validate(), Err(SseError::InvalidData(_))));
    }

    #[test]
    fn test_validate_empty_event_type() {
        let event = SseEvent::new("data").with_event_type("");
        assert!(matches!(event.validate(), Err(SseError::EmptyEventType)));
    }

    #[test]
    fn test_effective_type() {
        let default = SseEvent::new("data");
        assert_eq!(default.effective_type(), "message");

        let custom = SseEvent::new("data").with_event_type("ping");
        assert_eq!(custom.effective_type(), "ping");
    }

    #[test]
    fn test_parse_simple() {
        let stream = "data: hello\n\n";
        let events = parse_event_stream(stream).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_parse_multiple_events() {
        let stream = "data: first\n\ndata: second\n\n";
        let events = parse_event_stream(stream).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn test_parse_with_id_and_type() {
        let stream = "id: 5\nevent: update\ndata: payload\n\n";
        let events = parse_event_stream(stream).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_deref(), Some("5"));
        assert_eq!(events[0].event_type.as_deref(), Some("update"));
        assert_eq!(events[0].data, "payload");
    }

    #[test]
    fn test_parse_multiline_data() {
        let stream = "data: line1\ndata: line2\n\n";
        let events = parse_event_stream(stream).unwrap();
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn test_parse_with_retry() {
        let stream = "retry: 3000\ndata: test\n\n";
        let events = parse_event_stream(stream).unwrap();
        assert_eq!(events[0].retry, Some(3000));
    }

    #[test]
    fn test_roundtrip() {
        let original = SseEvent::new("hello")
            .with_id("1")
            .with_event_type("greeting");
        let formatted = original.format().unwrap();
        let parsed = parse_event_stream(&formatted).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].data, "hello");
        assert_eq!(parsed[0].id, Some("1".to_string()));
        assert_eq!(parsed[0].event_type, Some("greeting".to_string()));
    }

    #[test]
    fn test_stream_builder() {
        let stream = EventStreamBuilder::new()
            .data("msg1")
            .event(SseEvent::new("msg2").with_event_type("update"))
            .build()
            .unwrap();
        let events = parse_event_stream(&stream).unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_reconnection_tracker() {
        let mut tracker = ReconnectionTracker::new(1000)
            .with_max_attempts(3)
            .with_backoff_factor(2.0);

        let delay1 = tracker.on_disconnect().unwrap();
        assert_eq!(delay1, 1000);

        let delay2 = tracker.on_disconnect().unwrap();
        assert_eq!(delay2, 2000);

        let delay3 = tracker.on_disconnect().unwrap();
        assert_eq!(delay3, 4000);

        // Fourth attempt should fail.
        let err = tracker.on_disconnect().unwrap_err();
        assert!(matches!(err, SseError::MaxReconnectExceeded { .. }));
    }

    #[test]
    fn test_reconnection_on_connected() {
        let mut tracker = ReconnectionTracker::new(1000);
        tracker.on_disconnect().unwrap();
        tracker.on_disconnect().unwrap();
        tracker.on_connected();
        assert_eq!(tracker.attempts, 0);
        assert_eq!(tracker.retry_ms, 1000);
    }

    #[test]
    fn test_reconnection_server_retry() {
        let mut tracker = ReconnectionTracker::new(1000);
        let event = SseEvent::new("data").with_id("10").with_retry(5000);
        tracker.on_event(&event);
        assert_eq!(tracker.last_event_id, Some("10".to_string()));
        assert_eq!(tracker.retry_ms, 5000);
    }

    #[test]
    fn test_event_filter_allow_only() {
        let filter = EventFilter::allow_only(&["update", "create"]);
        assert!(filter.accepts(&SseEvent::new("d").with_event_type("update")));
        assert!(!filter.accepts(&SseEvent::new("d").with_event_type("delete")));
    }

    #[test]
    fn test_event_filter_deny() {
        let filter = EventFilter::deny(&["debug"]);
        assert!(filter.accepts(&SseEvent::new("d").with_event_type("update")));
        assert!(!filter.accepts(&SseEvent::new("d").with_event_type("debug")));
    }

    #[test]
    fn test_event_filter_all() {
        let filter = EventFilter::allow_all();
        assert!(filter.accepts(&SseEvent::new("anything")));
    }

    #[test]
    fn test_event_history() {
        let mut history = EventHistory::new(5);
        for i in 0..3 {
            history.push(SseEvent::new(&format!("event{i}")).with_id(&format!("{i}")));
        }
        assert_eq!(history.len(), 3);

        let since = history.since("0");
        assert_eq!(since.len(), 2);
        assert_eq!(since[0].data, "event1");
    }

    #[test]
    fn test_event_history_overflow() {
        let mut history = EventHistory::new(3);
        for i in 0..5 {
            history.push(SseEvent::new(&format!("ev{i}")).with_id(&format!("{i}")));
        }
        assert_eq!(history.len(), 3);
        assert_eq!(history.events[0].data, "ev2");
    }

    #[test]
    fn test_event_history_clear() {
        let mut history = EventHistory::new(10);
        history.push(SseEvent::new("data").with_id("1"));
        assert!(!history.is_empty());
        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn test_stream_builder_comment() {
        let stream = EventStreamBuilder::new()
            .comment("ping")
            .data("msg")
            .build()
            .unwrap();
        assert!(stream.contains(": ping\n"));
    }

    #[test]
    fn test_max_retry_cap() {
        let mut tracker = ReconnectionTracker::new(1000)
            .with_max_retry(5000)
            .with_backoff_factor(10.0);

        tracker.on_disconnect().unwrap(); // 1000
        let delay = tracker.on_disconnect().unwrap(); // would be 10000, capped to 5000
        assert_eq!(delay, 5000);
    }
}
