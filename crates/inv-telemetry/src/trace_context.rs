use std::fmt;
use std::time::Instant;

use rand::RngExt;
use serde::{Deserialize, Serialize};

/// A 128-bit trace identifier (W3C Trace Context).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId([u8; 16]);

/// A 64-bit span identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId([u8; 8]);

/// Trace flags (1 byte). Bit 0 = sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceFlags(u8);

/// The core span context propagated across service boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub flags: TraceFlags,
}

/// A trace context that tracks timing and hierarchy.
///
/// Implements W3C Trace Context (`traceparent` header format):
/// `{version}-{trace_id}-{span_id}-{flags}`
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub span: SpanContext,
    pub name: String,
    pub start: Instant,
    pub end: Option<Instant>,
}

impl TraceId {
    /// Generate a new random trace ID.
    pub fn generate() -> Self {
        let mut rng = rand::rng();
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);
        Self(bytes)
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Parse from a 32-character hex string.
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

    /// Render as a 32-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl SpanId {
    /// Generate a new random span ID.
    pub fn generate() -> Self {
        let mut rng = rand::rng();
        let mut bytes = [0u8; 8];
        rng.fill(&mut bytes);
        Self(bytes)
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Parse from a 16-character hex string.
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

    /// Render as a 16-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl TraceFlags {
    /// Default sampled flags.
    pub fn sampled() -> Self {
        Self(0x01)
    }

    /// No flags set.
    pub fn empty() -> Self {
        Self(0x00)
    }

    /// Whether the sampled bit is set.
    pub fn is_sampled(&self) -> bool {
        self.0 & 0x01 != 0
    }

    /// The raw byte value.
    pub fn as_byte(&self) -> u8 {
        self.0
    }

    /// Parse from a 2-character hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        u8::from_str_radix(hex, 16).ok().map(Self)
    }

    /// Render as a 2-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        format!("{:02x}", self.0)
    }
}

impl TraceContext {
    /// Create a new root trace context (no parent).
    pub fn new_root(name: &str) -> Self {
        Self {
            span: SpanContext {
                trace_id: TraceId::generate(),
                span_id: SpanId::generate(),
                parent_span_id: None,
                flags: TraceFlags::sampled(),
            },
            name: name.to_string(),
            start: Instant::now(),
            end: None,
        }
    }

    /// Create a child span within the same trace.
    pub fn child(&self, name: &str) -> Self {
        Self {
            span: SpanContext {
                trace_id: self.span.trace_id,
                span_id: SpanId::generate(),
                parent_span_id: Some(self.span.span_id),
                flags: self.span.flags,
            },
            name: name.to_string(),
            start: Instant::now(),
            end: None,
        }
    }

    /// Mark the span as finished.
    pub fn finish(&mut self) {
        self.end = Some(Instant::now());
    }

    /// Duration of the span in seconds (returns None if not finished).
    pub fn duration_secs(&self) -> Option<f64> {
        self.end
            .map(|end| end.duration_since(self.start).as_secs_f64())
    }

    /// Whether this span has been finished.
    pub fn is_finished(&self) -> bool {
        self.end.is_some()
    }

    /// Encode as a W3C `traceparent` header value.
    ///
    /// Format: `00-{trace_id}-{span_id}-{flags}`
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{}",
            self.span.trace_id.to_hex(),
            self.span.span_id.to_hex(),
            self.span.flags.to_hex(),
        )
    }

    /// Parse a W3C `traceparent` header value into a `SpanContext`.
    ///
    /// Returns `None` if the format is invalid.
    pub fn from_traceparent(header: &str) -> Option<SpanContext> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let _version = parts[0]; // "00"
        let trace_id = TraceId::from_hex(parts[1])?;
        let span_id = SpanId::from_hex(parts[2])?;
        let flags = TraceFlags::from_hex(parts[3])?;

        Some(SpanContext {
            trace_id,
            span_id,
            parent_span_id: None,
            flags,
        })
    }

    /// Create a TraceContext from an incoming traceparent header,
    /// starting a new child span.
    pub fn from_incoming(header: &str, name: &str) -> Option<Self> {
        let parent = Self::from_traceparent(header)?;
        Some(Self {
            span: SpanContext {
                trace_id: parent.trace_id,
                span_id: SpanId::generate(),
                parent_span_id: Some(parent.span_id),
                flags: parent.flags,
            },
            name: name.to_string(),
            start: Instant::now(),
            end: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_hex_roundtrip() {
        let id = TraceId::generate();
        let hex = id.to_hex();
        assert_eq!(hex.len(), 32);
        let parsed = TraceId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn span_id_hex_roundtrip() {
        let id = SpanId::generate();
        let hex = id.to_hex();
        assert_eq!(hex.len(), 16);
        let parsed = SpanId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn trace_flags_sampled() {
        let flags = TraceFlags::sampled();
        assert!(flags.is_sampled());
        assert_eq!(flags.to_hex(), "01");
    }

    #[test]
    fn trace_flags_empty() {
        let flags = TraceFlags::empty();
        assert!(!flags.is_sampled());
        assert_eq!(flags.to_hex(), "00");
    }

    #[test]
    fn new_root_creates_unique_ids() {
        let t1 = TraceContext::new_root("op1");
        let t2 = TraceContext::new_root("op2");
        assert_ne!(t1.span.trace_id, t2.span.trace_id);
        assert_ne!(t1.span.span_id, t2.span.span_id);
        assert!(t1.span.parent_span_id.is_none());
    }

    #[test]
    fn child_inherits_trace_id() {
        let parent = TraceContext::new_root("parent");
        let child = parent.child("child");
        assert_eq!(child.span.trace_id, parent.span.trace_id);
        assert_ne!(child.span.span_id, parent.span.span_id);
        assert_eq!(child.span.parent_span_id, Some(parent.span.span_id));
    }

    #[test]
    fn finish_records_duration() {
        let mut ctx = TraceContext::new_root("op");
        assert!(!ctx.is_finished());
        assert!(ctx.duration_secs().is_none());
        std::thread::sleep(std::time::Duration::from_millis(10));
        ctx.finish();
        assert!(ctx.is_finished());
        let dur = ctx.duration_secs().unwrap();
        assert!(dur >= 0.01, "Duration should be at least 10ms: {dur}");
    }

    #[test]
    fn traceparent_roundtrip() {
        let ctx = TraceContext::new_root("test");
        let header = ctx.to_traceparent();

        // Format: 00-{32 hex}-{16 hex}-{2 hex}
        let parts: Vec<&str> = header.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[2].len(), 16);
        assert_eq!(parts[3].len(), 2);

        let parsed = TraceContext::from_traceparent(&header).unwrap();
        assert_eq!(parsed.trace_id, ctx.span.trace_id);
        assert_eq!(parsed.span_id, ctx.span.span_id);
        assert_eq!(parsed.flags, ctx.span.flags);
    }

    #[test]
    fn from_incoming_creates_child() {
        let parent = TraceContext::new_root("parent");
        let header = parent.to_traceparent();

        let child = TraceContext::from_incoming(&header, "child").unwrap();
        assert_eq!(child.span.trace_id, parent.span.trace_id);
        assert_ne!(child.span.span_id, parent.span.span_id);
        assert_eq!(child.span.parent_span_id, Some(parent.span.span_id));
        assert_eq!(child.name, "child");
    }

    #[test]
    fn invalid_traceparent_returns_none() {
        assert!(TraceContext::from_traceparent("").is_none());
        assert!(TraceContext::from_traceparent("not-a-valid-header").is_none());
        assert!(TraceContext::from_traceparent("00-short-short-00").is_none());
    }

    #[test]
    fn trace_id_from_bytes() {
        let bytes = [1u8; 16];
        let id = TraceId::from_bytes(bytes);
        assert_eq!(*id.as_bytes(), bytes);
        assert_eq!(id.to_hex(), "01010101010101010101010101010101");
    }

    #[test]
    fn display_impls() {
        let tid = TraceId::from_bytes([0xab; 16]);
        assert_eq!(format!("{tid}"), "abababababababababababababababab");

        let sid = SpanId::from_bytes([0xcd; 8]);
        assert_eq!(format!("{sid}"), "cdcdcdcdcdcdcdcd");
    }
}
