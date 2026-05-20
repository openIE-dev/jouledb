//! Request/Response Tracing and Correlation
//!
//! Provides distributed tracing support for request/response correlation
//! across services and operations.

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Trace ID for request correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub Uuid);

impl TraceId {
    /// Generate a new trace ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from string
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Convert to string
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Span ID for operation tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u64);

impl SpanId {
    /// Generate a new span ID
    pub fn new() -> Self {
        Self(rand::random::<u64>())
    }

    /// Create from u64
    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

    /// Get as u64
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

/// Trace context for request correlation
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: TraceId,
    /// Parent span ID
    pub parent_span_id: Option<SpanId>,
    /// Current span ID
    pub span_id: SpanId,
    /// Additional baggage (key-value pairs)
    pub baggage: HashMap<String, String>,
    /// Flags
    pub flags: u8,
}

impl TraceContext {
    /// Create a new trace context
    pub fn new() -> Self {
        Self {
            trace_id: TraceId::new(),
            parent_span_id: None,
            span_id: SpanId::new(),
            baggage: HashMap::new(),
            flags: 0,
        }
    }

    /// Create a child span
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            parent_span_id: Some(self.span_id),
            span_id: SpanId::new(),
            baggage: self.baggage.clone(),
            flags: self.flags,
        }
    }

    /// Add baggage
    pub fn add_baggage(&mut self, key: String, value: String) {
        self.baggage.insert(key, value);
    }

    /// Get baggage
    pub fn get_baggage(&self, key: &str) -> Option<&String> {
        self.baggage.get(key)
    }

    /// Serialize to HTTP headers
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("X-Trace-Id".to_string(), self.trace_id.to_string());
        headers.insert("X-Span-Id".to_string(), self.span_id.as_u64().to_string());

        if let Some(parent) = self.parent_span_id {
            headers.insert("X-Parent-Span-Id".to_string(), parent.as_u64().to_string());
        }

        for (key, value) in &self.baggage {
            headers.insert(format!("X-Baggage-{}", key), value.clone());
        }

        headers
    }

    /// Parse from HTTP headers
    pub fn from_headers(headers: &HashMap<String, String>) -> Option<Self> {
        let trace_id = headers
            .get("X-Trace-Id")
            .and_then(|s| TraceId::from_str(s).ok())?;

        let span_id = headers
            .get("X-Span-Id")
            .and_then(|s| s.parse::<u64>().ok())
            .map(SpanId::from_u64)
            .unwrap_or_else(SpanId::new);

        let parent_span_id = headers
            .get("X-Parent-Span-Id")
            .and_then(|s| s.parse::<u64>().ok())
            .map(SpanId::from_u64);

        let mut baggage = HashMap::new();
        for (key, value) in headers {
            if let Some(baggage_key) = key.strip_prefix("X-Baggage-") {
                baggage.insert(baggage_key.to_string(), value.clone());
            }
        }

        Some(Self {
            trace_id,
            parent_span_id,
            span_id,
            baggage,
            flags: 0,
        })
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Trace span for operation tracking
pub struct TraceSpan {
    /// Context
    context: TraceContext,
    /// Operation name
    operation: String,
    /// Start time
    start_time: std::time::Instant,
    /// Tags
    tags: HashMap<String, String>,
    /// Logs
    logs: Vec<(std::time::Instant, String)>,
}

impl TraceSpan {
    /// Create a new span
    pub fn new(context: TraceContext, operation: String) -> Self {
        Self {
            context,
            operation,
            start_time: std::time::Instant::now(),
            tags: HashMap::new(),
            logs: Vec::new(),
        }
    }

    /// Add a tag
    pub fn tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    /// Log an event
    pub fn log(&mut self, message: String) {
        self.logs.push((std::time::Instant::now(), message));
    }

    /// Get context
    pub fn context(&self) -> &TraceContext {
        &self.context
    }

    /// Get duration
    pub fn duration(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Finish the span
    pub fn finish(self) -> FinishedSpan {
        FinishedSpan {
            context: self.context,
            operation: self.operation,
            duration: self.start_time.elapsed(),
            tags: self.tags,
            logs: self.logs,
        }
    }
}

/// Finished span for reporting
#[derive(Debug, Clone)]
pub struct FinishedSpan {
    /// Context
    pub context: TraceContext,
    /// Operation name
    pub operation: String,
    /// Duration
    pub duration: std::time::Duration,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Logs
    pub logs: Vec<(std::time::Instant, String)>,
}

/// Trace collector trait
pub trait TraceCollector: Send + Sync {
    /// Collect a finished span
    fn collect(&self, span: FinishedSpan);
}

/// In-memory trace collector (for testing)
pub struct InMemoryTraceCollector {
    spans: Arc<std::sync::Mutex<Vec<FinishedSpan>>>,
}

impl InMemoryTraceCollector {
    /// Create a new in-memory collector
    pub fn new() -> Self {
        Self {
            spans: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Get collected spans
    pub fn spans(&self) -> Vec<FinishedSpan> {
        crate::lock_util::mutex_lock(&self.spans).clone()
    }

    /// Clear collected spans
    pub fn clear(&self) {
        crate::lock_util::mutex_lock(&self.spans).clear();
    }
}

impl TraceCollector for InMemoryTraceCollector {
    fn collect(&self, span: FinishedSpan) {
        crate::lock_util::mutex_lock(&self.spans).push(span);
    }
}

/// No-op trace collector
pub struct NoOpTraceCollector;

impl TraceCollector for NoOpTraceCollector {
    fn collect(&self, _span: FinishedSpan) {
        // No-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context() {
        let ctx = TraceContext::new();
        assert!(ctx.parent_span_id.is_none());

        let child = ctx.child();
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_eq!(child.parent_span_id, Some(ctx.span_id));
    }

    #[test]
    fn test_trace_context_headers() {
        let mut ctx = TraceContext::new();
        ctx.add_baggage("user".to_string(), "alice".to_string());

        let headers = ctx.to_headers();
        assert!(headers.contains_key("X-Trace-Id"));
        assert!(headers.contains_key("X-Span-Id"));
        assert_eq!(headers.get("X-Baggage-user"), Some(&"alice".to_string()));

        let parsed = TraceContext::from_headers(&headers).unwrap();
        assert_eq!(parsed.trace_id.to_string(), ctx.trace_id.to_string());
        assert_eq!(parsed.get_baggage("user"), Some(&"alice".to_string()));
    }

    #[test]
    fn test_trace_span() {
        let ctx = TraceContext::new();
        let mut span = TraceSpan::new(ctx, "test_operation".to_string());

        span.tag("key".to_string(), "value".to_string());
        span.log("test log".to_string());

        let finished = span.finish();
        assert_eq!(finished.operation, "test_operation");
        assert_eq!(finished.tags.get("key"), Some(&"value".to_string()));
        assert!(!finished.logs.is_empty());
    }
}
