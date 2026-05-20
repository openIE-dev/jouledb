//! Distributed tracing: W3C Trace Context compatible.
//!
//! Trace/span IDs, parent-child relationships, baggage propagation,
//! span timing, sampling decisions, and span export. Pure Rust —
//! no I/O, no external tracing backends.

use std::collections::HashMap;
use std::fmt;

// ── Trace and span IDs ────────────────────────────────────────────

/// 16-byte trace ID rendered as 32-char lowercase hex.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceId(pub [u8; 16]);

impl TraceId {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Parse from a 32-char hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for i in 0..16 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self(bytes))
    }

    /// Render as 32-char lowercase hex.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Check if this is the zero (invalid) trace ID.
    pub fn is_valid(&self) -> bool {
        self.0.iter().any(|b| *b != 0)
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// 8-byte span ID rendered as 16-char lowercase hex.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanId(pub [u8; 8]);

impl SpanId {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Parse from a 16-char hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 16 {
            return None;
        }
        let mut bytes = [0u8; 8];
        for i in 0..8 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self(bytes))
    }

    /// Render as 16-char lowercase hex.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Check if this is the zero (invalid) span ID.
    pub fn is_valid(&self) -> bool {
        self.0.iter().any(|b| *b != 0)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ── Trace flags ───────────────────────────────────────────────────

/// W3C trace flags (1 byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceFlags(pub u8);

impl TraceFlags {
    /// The sampled flag (bit 0).
    pub const SAMPLED: u8 = 0x01;

    /// Create with specific flags.
    pub fn new(flags: u8) -> Self {
        Self(flags)
    }

    /// Check if the sampled flag is set.
    pub fn is_sampled(&self) -> bool {
        self.0 & Self::SAMPLED != 0
    }

    /// Render as 2-char hex.
    pub fn to_hex(&self) -> String {
        format!("{:02x}", self.0)
    }

    /// Parse from 2-char hex.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 2 {
            return None;
        }
        u8::from_str_radix(hex, 16).ok().map(Self)
    }
}

// ── W3C traceparent ───────────────────────────────────────────────

/// W3C Trace Context `traceparent` header value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceParent {
    pub version: u8,
    pub trace_id: TraceId,
    pub parent_id: SpanId,
    pub flags: TraceFlags,
}

impl TraceParent {
    /// Create a new traceparent.
    pub fn new(trace_id: TraceId, parent_id: SpanId, flags: TraceFlags) -> Self {
        Self {
            version: 0,
            trace_id,
            parent_id,
            flags,
        }
    }

    /// Parse from the `traceparent` header format: `VV-TTTTTTTT-SSSSSSSS-FF`.
    pub fn parse(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }
        let version = u8::from_str_radix(parts[0], 16).ok()?;
        let trace_id = TraceId::from_hex(parts[1])?;
        let parent_id = SpanId::from_hex(parts[2])?;
        let flags = TraceFlags::from_hex(parts[3])?;
        Some(Self {
            version,
            trace_id,
            parent_id,
            flags,
        })
    }

    /// Render as a `traceparent` header value.
    pub fn to_header(&self) -> String {
        format!(
            "{:02x}-{}-{}-{}",
            self.version,
            self.trace_id.to_hex(),
            self.parent_id.to_hex(),
            self.flags.to_hex()
        )
    }
}

impl fmt::Display for TraceParent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_header())
    }
}

// ── Baggage ───────────────────────────────────────────────────────

/// W3C Baggage: key-value pairs propagated across service boundaries.
#[derive(Debug, Clone, Default)]
pub struct Baggage {
    items: HashMap<String, String>,
}

impl Baggage {
    /// Create empty baggage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a baggage item.
    pub fn set(&mut self, key: &str, value: &str) {
        self.items.insert(key.to_string(), value.to_string());
    }

    /// Get a baggage item.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.items.get(key).map(|s| s.as_str())
    }

    /// Remove a baggage item.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.items.remove(key)
    }

    /// Parse from the `baggage` header format: `key=value,key2=value2`.
    pub fn parse(header: &str) -> Self {
        let mut baggage = Self::new();
        for pair in header.split(',') {
            let pair = pair.trim();
            if let Some((k, v)) = pair.split_once('=') {
                baggage.set(k.trim(), v.trim());
            }
        }
        baggage
    }

    /// Render as a `baggage` header value (sorted for determinism).
    pub fn to_header(&self) -> String {
        let mut pairs: Vec<String> = self
            .items
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        pairs.sort();
        pairs.join(",")
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── Span kind ─────────────────────────────────────────────────────

/// The role of a span in the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// Internal operation.
    Internal,
    /// Incoming request (server side).
    Server,
    /// Outgoing request (client side).
    Client,
    /// Asynchronous producer.
    Producer,
    /// Asynchronous consumer.
    Consumer,
}

impl fmt::Display for SpanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal => write!(f, "internal"),
            Self::Server => write!(f, "server"),
            Self::Client => write!(f, "client"),
            Self::Producer => write!(f, "producer"),
            Self::Consumer => write!(f, "consumer"),
        }
    }
}

// ── Span status ───────────────────────────────────────────────────

/// Status of a completed span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// Unset (default).
    Unset,
    /// Operation completed successfully.
    Ok,
    /// Operation encountered an error.
    Error(String),
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unset => write!(f, "unset"),
            Self::Ok => write!(f, "ok"),
            Self::Error(msg) => write!(f, "error: {msg}"),
        }
    }
}

// ── Span event ────────────────────────────────────────────────────

/// An event recorded during a span's lifetime.
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// Timestamp as microseconds since some epoch (caller-supplied).
    pub timestamp_us: u64,
    /// Attributes attached to the event.
    pub attributes: HashMap<String, String>,
}

// ── Span ──────────────────────────────────────────────────────────

/// A single span in a distributed trace.
#[derive(Debug, Clone)]
pub struct Span {
    /// Name/operation of the span.
    pub name: String,
    /// Trace this span belongs to.
    pub trace_id: TraceId,
    /// This span's unique ID.
    pub span_id: SpanId,
    /// Parent span ID (None for root spans).
    pub parent_span_id: Option<SpanId>,
    /// Kind of span.
    pub kind: SpanKind,
    /// Start timestamp in microseconds.
    pub start_us: u64,
    /// End timestamp in microseconds (0 if not ended).
    pub end_us: u64,
    /// Span status.
    pub status: SpanStatus,
    /// Attributes.
    pub attributes: HashMap<String, String>,
    /// Events recorded during this span.
    pub events: Vec<SpanEvent>,
    /// Whether this span was sampled.
    pub sampled: bool,
}

impl Span {
    /// Create a new span.
    pub fn new(
        name: &str,
        trace_id: TraceId,
        span_id: SpanId,
        kind: SpanKind,
        start_us: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            trace_id,
            span_id,
            parent_span_id: None,
            kind,
            start_us,
            end_us: 0,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            sampled: true,
        }
    }

    /// Set parent span.
    pub fn with_parent(mut self, parent: SpanId) -> Self {
        self.parent_span_id = Some(parent);
        self
    }

    /// Set an attribute.
    pub fn set_attribute(&mut self, key: &str, value: &str) {
        self.attributes.insert(key.to_string(), value.to_string());
    }

    /// Record an event.
    pub fn add_event(&mut self, name: &str, timestamp_us: u64) {
        self.events.push(SpanEvent {
            name: name.to_string(),
            timestamp_us,
            attributes: HashMap::new(),
        });
    }

    /// End the span with a timestamp.
    pub fn end(&mut self, end_us: u64) {
        self.end_us = end_us;
    }

    /// Set the span status.
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// Duration in microseconds (0 if not ended).
    pub fn duration_us(&self) -> u64 {
        if self.end_us > self.start_us {
            self.end_us - self.start_us
        } else {
            0
        }
    }

    /// Is this a root span?
    pub fn is_root(&self) -> bool {
        self.parent_span_id.is_none()
    }

    /// Build a traceparent header for a child span of this one.
    pub fn child_traceparent(&self, child_span_id: SpanId) -> TraceParent {
        let flags = if self.sampled {
            TraceFlags::new(TraceFlags::SAMPLED)
        } else {
            TraceFlags::new(0)
        };
        TraceParent::new(self.trace_id.clone(), child_span_id, flags)
    }
}

// ── Sampling decision ─────────────────────────────────────────────

/// Sampling strategy for traces.
#[derive(Debug, Clone)]
pub enum SamplingDecision {
    /// Always sample.
    AlwaysOn,
    /// Never sample.
    AlwaysOff,
    /// Sample 1 in N traces.
    RateBased(u64),
}

impl SamplingDecision {
    /// Decide whether to sample given a counter.
    pub fn should_sample(&self, counter: u64) -> bool {
        match self {
            Self::AlwaysOn => true,
            Self::AlwaysOff => false,
            Self::RateBased(n) => {
                if *n == 0 {
                    return false;
                }
                counter % n == 0
            }
        }
    }
}

// ── Span collector ────────────────────────────────────────────────

/// In-memory span collector for export.
#[derive(Debug, Clone)]
pub struct SpanCollector {
    spans: Vec<Span>,
    sampling: SamplingDecision,
    counter: u64,
}

impl SpanCollector {
    /// Create a new collector.
    pub fn new() -> Self {
        Self {
            spans: Vec::new(),
            sampling: SamplingDecision::AlwaysOn,
            counter: 0,
        }
    }

    /// Set the sampling strategy.
    pub fn set_sampling(&mut self, decision: SamplingDecision) {
        self.sampling = decision;
    }

    /// Record a completed span.
    pub fn record(&mut self, span: Span) {
        if span.sampled {
            self.counter = self.counter.wrapping_add(1);
            if self.sampling.should_sample(self.counter) {
                self.spans.push(span);
            }
        }
    }

    /// Get all recorded spans.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Find spans belonging to a specific trace.
    pub fn spans_for_trace(&self, trace_id: &TraceId) -> Vec<&Span> {
        self.spans
            .iter()
            .filter(|s| &s.trace_id == trace_id)
            .collect()
    }

    /// Find root spans (no parent).
    pub fn root_spans(&self) -> Vec<&Span> {
        self.spans.iter().filter(|s| s.is_root()).collect()
    }

    /// Count total spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Clear all spans.
    pub fn clear(&mut self) {
        self.spans.clear();
    }
}

impl Default for SpanCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace_id() -> TraceId {
        TraceId::from_bytes([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ])
    }

    fn make_span_id() -> SpanId {
        SpanId::from_bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
    }

    #[test]
    fn trace_id_hex_roundtrip() {
        let tid = make_trace_id();
        let hex = tid.to_hex();
        assert_eq!(hex, "00112233445566778899aabbccddeeff");
        let parsed = TraceId::from_hex(&hex).unwrap();
        assert_eq!(parsed, tid);
    }

    #[test]
    fn trace_id_invalid_hex() {
        assert!(TraceId::from_hex("short").is_none());
        assert!(TraceId::from_hex("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz").is_none());
    }

    #[test]
    fn trace_id_validity() {
        assert!(make_trace_id().is_valid());
        assert!(!TraceId::from_bytes([0; 16]).is_valid());
    }

    #[test]
    fn span_id_hex_roundtrip() {
        let sid = make_span_id();
        let hex = sid.to_hex();
        assert_eq!(hex, "0102030405060708");
        let parsed = SpanId::from_hex(&hex).unwrap();
        assert_eq!(parsed, sid);
    }

    #[test]
    fn span_id_invalid_hex() {
        assert!(SpanId::from_hex("short").is_none());
    }

    #[test]
    fn span_id_validity() {
        assert!(make_span_id().is_valid());
        assert!(!SpanId::from_bytes([0; 8]).is_valid());
    }

    #[test]
    fn trace_flags_sampled() {
        let flags = TraceFlags::new(TraceFlags::SAMPLED);
        assert!(flags.is_sampled());
        assert_eq!(flags.to_hex(), "01");
    }

    #[test]
    fn trace_flags_not_sampled() {
        let flags = TraceFlags::new(0);
        assert!(!flags.is_sampled());
        assert_eq!(flags.to_hex(), "00");
    }

    #[test]
    fn trace_flags_hex_roundtrip() {
        let flags = TraceFlags::new(0x01);
        let hex = flags.to_hex();
        let parsed = TraceFlags::from_hex(&hex).unwrap();
        assert_eq!(parsed, flags);
    }

    #[test]
    fn traceparent_parse_and_render() {
        let header = "00-00112233445566778899aabbccddeeff-0102030405060708-01";
        let tp = TraceParent::parse(header).unwrap();
        assert_eq!(tp.version, 0);
        assert_eq!(tp.trace_id, make_trace_id());
        assert_eq!(tp.parent_id, make_span_id());
        assert!(tp.flags.is_sampled());
        assert_eq!(tp.to_header(), header);
    }

    #[test]
    fn traceparent_display() {
        let tp = TraceParent::new(make_trace_id(), make_span_id(), TraceFlags::new(0x01));
        let s = tp.to_string();
        assert!(s.starts_with("00-"));
        assert!(s.ends_with("-01"));
    }

    #[test]
    fn traceparent_parse_invalid() {
        assert!(TraceParent::parse("invalid").is_none());
        assert!(TraceParent::parse("00-short-short-00").is_none());
    }

    #[test]
    fn baggage_set_and_get() {
        let mut b = Baggage::new();
        b.set("tenant", "acme");
        assert_eq!(b.get("tenant"), Some("acme"));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn baggage_remove() {
        let mut b = Baggage::new();
        b.set("k", "v");
        assert_eq!(b.remove("k"), Some("v".into()));
        assert!(b.is_empty());
    }

    #[test]
    fn baggage_parse_and_render() {
        let header = "key1=val1,key2=val2";
        let b = Baggage::parse(header);
        assert_eq!(b.len(), 2);
        assert_eq!(b.get("key1"), Some("val1"));
        assert_eq!(b.get("key2"), Some("val2"));
        // to_header sorts.
        assert_eq!(b.to_header(), "key1=val1,key2=val2");
    }

    #[test]
    fn baggage_parse_whitespace() {
        let b = Baggage::parse("a = 1 , b = 2");
        assert_eq!(b.get("a"), Some("1"));
        assert_eq!(b.get("b"), Some("2"));
    }

    #[test]
    fn span_kind_display() {
        assert_eq!(SpanKind::Internal.to_string(), "internal");
        assert_eq!(SpanKind::Server.to_string(), "server");
        assert_eq!(SpanKind::Client.to_string(), "client");
        assert_eq!(SpanKind::Producer.to_string(), "producer");
        assert_eq!(SpanKind::Consumer.to_string(), "consumer");
    }

    #[test]
    fn span_status_display() {
        assert_eq!(SpanStatus::Unset.to_string(), "unset");
        assert_eq!(SpanStatus::Ok.to_string(), "ok");
        assert_eq!(
            SpanStatus::Error("timeout".into()).to_string(),
            "error: timeout"
        );
    }

    #[test]
    fn span_creation_and_timing() {
        let mut span = Span::new("GET /api", make_trace_id(), make_span_id(), SpanKind::Server, 1000);
        assert!(span.is_root());
        assert_eq!(span.duration_us(), 0);

        span.end(5000);
        assert_eq!(span.duration_us(), 4000);
    }

    #[test]
    fn span_parent_child() {
        let parent_id = SpanId::from_bytes([0xAA; 8]);
        let span = Span::new("child", make_trace_id(), make_span_id(), SpanKind::Internal, 100)
            .with_parent(parent_id.clone());
        assert!(!span.is_root());
        assert_eq!(span.parent_span_id, Some(parent_id));
    }

    #[test]
    fn span_attributes_and_events() {
        let mut span = Span::new("op", make_trace_id(), make_span_id(), SpanKind::Internal, 0);
        span.set_attribute("http.method", "POST");
        span.add_event("request_received", 100);
        span.add_event("response_sent", 500);

        assert_eq!(span.attributes.get("http.method").unwrap(), "POST");
        assert_eq!(span.events.len(), 2);
        assert_eq!(span.events[0].name, "request_received");
    }

    #[test]
    fn span_status_set() {
        let mut span = Span::new("op", make_trace_id(), make_span_id(), SpanKind::Internal, 0);
        assert_eq!(span.status, SpanStatus::Unset);
        span.set_status(SpanStatus::Ok);
        assert_eq!(span.status, SpanStatus::Ok);
        span.set_status(SpanStatus::Error("fail".into()));
        assert_eq!(span.status, SpanStatus::Error("fail".into()));
    }

    #[test]
    fn span_child_traceparent() {
        let span = Span::new("root", make_trace_id(), make_span_id(), SpanKind::Server, 0);
        let child_sid = SpanId::from_bytes([0xBB; 8]);
        let tp = span.child_traceparent(child_sid.clone());
        assert_eq!(tp.trace_id, make_trace_id());
        assert_eq!(tp.parent_id, child_sid);
        assert!(tp.flags.is_sampled());
    }

    #[test]
    fn span_child_traceparent_not_sampled() {
        let mut span = Span::new("root", make_trace_id(), make_span_id(), SpanKind::Server, 0);
        span.sampled = false;
        let tp = span.child_traceparent(SpanId::from_bytes([0xCC; 8]));
        assert!(!tp.flags.is_sampled());
    }

    #[test]
    fn sampling_always_on() {
        let s = SamplingDecision::AlwaysOn;
        for i in 0..10 {
            assert!(s.should_sample(i));
        }
    }

    #[test]
    fn sampling_always_off() {
        let s = SamplingDecision::AlwaysOff;
        for i in 0..10 {
            assert!(!s.should_sample(i));
        }
    }

    #[test]
    fn sampling_rate_based() {
        let s = SamplingDecision::RateBased(5);
        let results: Vec<bool> = (0..10).map(|i| s.should_sample(i)).collect();
        assert_eq!(
            results,
            vec![true, false, false, false, false, true, false, false, false, false]
        );
    }

    #[test]
    fn sampling_rate_zero() {
        let s = SamplingDecision::RateBased(0);
        assert!(!s.should_sample(0));
    }

    #[test]
    fn collector_record_and_query() {
        let mut collector = SpanCollector::new();

        let tid = make_trace_id();
        let mut s1 = Span::new("root", tid.clone(), make_span_id(), SpanKind::Server, 0);
        s1.end(1000);

        let child_sid = SpanId::from_bytes([0x11; 8]);
        let mut s2 = Span::new("child", tid.clone(), child_sid, SpanKind::Internal, 100)
            .with_parent(make_span_id());
        s2.end(900);

        collector.record(s1);
        collector.record(s2);

        assert_eq!(collector.len(), 2);
        assert_eq!(collector.spans_for_trace(&tid).len(), 2);
        assert_eq!(collector.root_spans().len(), 1);
    }

    #[test]
    fn collector_sampling() {
        let mut collector = SpanCollector::new();
        collector.set_sampling(SamplingDecision::RateBased(2));

        for i in 0u8..4 {
            let span = Span::new(
                "op",
                make_trace_id(),
                SpanId::from_bytes([i; 8]),
                SpanKind::Internal,
                0,
            );
            collector.record(span);
        }
        // counter: 1,2,3,4 → sampled when %2==0 → 2,4 → 2 spans
        assert_eq!(collector.len(), 2);
    }

    #[test]
    fn collector_skips_unsampled_spans() {
        let mut collector = SpanCollector::new();
        let mut span = Span::new("op", make_trace_id(), make_span_id(), SpanKind::Internal, 0);
        span.sampled = false;
        collector.record(span);
        assert!(collector.is_empty());
    }

    #[test]
    fn collector_clear() {
        let mut collector = SpanCollector::new();
        let span = Span::new("op", make_trace_id(), make_span_id(), SpanKind::Internal, 0);
        collector.record(span);
        assert!(!collector.is_empty());
        collector.clear();
        assert!(collector.is_empty());
    }

    #[test]
    fn trace_id_display() {
        let tid = make_trace_id();
        assert_eq!(format!("{tid}"), "00112233445566778899aabbccddeeff");
    }

    #[test]
    fn span_id_display() {
        let sid = make_span_id();
        assert_eq!(format!("{sid}"), "0102030405060708");
    }

    #[test]
    fn span_duration_not_ended() {
        let span = Span::new("op", make_trace_id(), make_span_id(), SpanKind::Internal, 500);
        assert_eq!(span.duration_us(), 0);
    }

    #[test]
    fn default_collector() {
        let c = SpanCollector::default();
        assert!(c.is_empty());
    }

    #[test]
    fn baggage_empty() {
        let b = Baggage::new();
        assert!(b.is_empty());
        assert_eq!(b.to_header(), "");
    }

    #[test]
    fn span_event_attributes() {
        let event = SpanEvent {
            name: "exception".into(),
            timestamp_us: 999,
            attributes: {
                let mut m = HashMap::new();
                m.insert("exception.type".into(), "NullPointer".into());
                m
            },
        };
        assert_eq!(event.attributes["exception.type"], "NullPointer");
    }
}
