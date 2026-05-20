//! Distributed tracing context: trace ID, span ID, parent span, baggage items,
//! context propagation (inject/extract from headers), span timing, span tags,
//! and trace tree reconstruction.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// Status of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error,
}

impl SpanStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpanStatus::Unset => "UNSET",
            SpanStatus::Ok => "OK",
            SpanStatus::Error => "ERROR",
        }
    }
}

/// Kind of span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Internal,
    Client,
    Server,
    Producer,
    Consumer,
}

impl SpanKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpanKind::Internal => "INTERNAL",
            SpanKind::Client => "CLIENT",
            SpanKind::Server => "SERVER",
            SpanKind::Producer => "PRODUCER",
            SpanKind::Consumer => "CONSUMER",
        }
    }
}

/// Generate a 16-char hex trace ID.
fn generate_trace_id() -> String {
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate an 8-char hex span ID.
fn generate_span_id() -> String {
    let uuid = Uuid::new_v4();
    let bytes = &uuid.as_bytes()[..8];
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// A single span event (annotation).
#[derive(Debug, Clone)]
pub struct SpanEvent {
    pub name: String,
    pub timestamp: DateTime<Utc>,
    pub attributes: HashMap<String, String>,
}

impl SpanEvent {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            timestamp: Utc::now(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }
}

/// A single span in a distributed trace.
#[derive(Debug, Clone)]
pub struct Span {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    pub status: SpanStatus,
    pub status_message: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub tags: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
    pub service_name: Option<String>,
}

impl Span {
    pub fn new(trace_id: &str, name: &str) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            span_id: generate_span_id(),
            parent_span_id: None,
            name: name.to_string(),
            kind: SpanKind::Internal,
            status: SpanStatus::Unset,
            status_message: None,
            start_time: Utc::now(),
            end_time: None,
            tags: HashMap::new(),
            events: Vec::new(),
            service_name: None,
        }
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_span_id = Some(parent_id.to_string());
        self
    }

    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_service(mut self, service: &str) -> Self {
        self.service_name = Some(service.to_string());
        self
    }

    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    pub fn set_status(&mut self, status: SpanStatus, message: Option<&str>) {
        self.status = status;
        self.status_message = message.map(|m| m.to_string());
    }

    pub fn add_event(&mut self, event: SpanEvent) {
        self.events.push(event);
    }

    /// Finish the span, recording the end time.
    pub fn finish(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Finish with a specific end time.
    pub fn finish_at(&mut self, time: DateTime<Utc>) {
        self.end_time = Some(time);
    }

    /// Duration of the span in milliseconds.
    pub fn duration_ms(&self) -> Option<f64> {
        self.end_time.map(|end| {
            let dur = end - self.start_time;
            dur.num_microseconds().unwrap_or(0) as f64 / 1000.0
        })
    }

    pub fn is_root(&self) -> bool {
        self.parent_span_id.is_none()
    }

    pub fn is_finished(&self) -> bool {
        self.end_time.is_some()
    }
}

// ── Baggage ──

/// Baggage — key-value pairs propagated across service boundaries.
#[derive(Debug, Clone, Default)]
pub struct Baggage {
    items: HashMap<String, String>,
}

impl Baggage {
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.items.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.items.get(key).map(|s| s.as_str())
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.items.remove(key)
    }

    pub fn items(&self) -> &HashMap<String, String> {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── Trace Context (W3C-style) ──

/// Trace context for propagation — models W3C traceparent/tracestate.
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub trace_flags: u8,
    pub baggage: Baggage,
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            trace_flags: 0x01, // sampled
            baggage: Baggage::new(),
        }
    }

    pub fn from_span(span: &Span) -> Self {
        Self {
            trace_id: span.trace_id.clone(),
            span_id: span.span_id.clone(),
            trace_flags: 0x01,
            baggage: Baggage::new(),
        }
    }

    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    pub fn set_sampled(&mut self, sampled: bool) {
        if sampled {
            self.trace_flags |= 0x01;
        } else {
            self.trace_flags &= !0x01;
        }
    }

    /// Inject context into a header map (W3C traceparent format).
    pub fn inject(&self, headers: &mut HashMap<String, String>) {
        let traceparent = format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        );
        headers.insert("traceparent".to_string(), traceparent);

        // Inject baggage
        if !self.baggage.is_empty() {
            let baggage_str: String = self
                .baggage
                .items()
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            headers.insert("baggage".to_string(), baggage_str);
        }
    }

    /// Extract context from a header map.
    pub fn extract(headers: &HashMap<String, String>) -> Option<Self> {
        let traceparent = headers.get("traceparent")?;
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let _version = parts[0]; // "00"
        let trace_id = parts[1].to_string();
        let span_id = parts[2].to_string();
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        let mut baggage = Baggage::new();
        if let Some(baggage_str) = headers.get("baggage") {
            for item in baggage_str.split(',') {
                let item = item.trim();
                if let Some(eq_pos) = item.find('=') {
                    let key = &item[..eq_pos];
                    let value = &item[eq_pos + 1..];
                    baggage.set(key.trim(), value.trim());
                }
            }
        }

        Some(Self {
            trace_id,
            span_id,
            trace_flags: flags,
            baggage,
        })
    }

    /// Create a child context with a new span ID.
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: generate_span_id(),
            trace_flags: self.trace_flags,
            baggage: self.baggage.clone(),
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Trace Tree ──

/// A node in a reconstructed trace tree.
#[derive(Debug, Clone)]
pub struct TraceTreeNode {
    pub span: Span,
    pub children: Vec<TraceTreeNode>,
    pub depth: usize,
}

impl TraceTreeNode {
    pub fn new(span: Span, depth: usize) -> Self {
        Self {
            span,
            children: Vec::new(),
            depth,
        }
    }

    /// Total duration including all children (the root span's duration).
    pub fn total_duration_ms(&self) -> Option<f64> {
        self.span.duration_ms()
    }

    /// Count total spans in this subtree.
    pub fn span_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.span_count()).sum::<usize>()
    }

    /// Max depth of this subtree.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children.iter().map(|c| c.max_depth()).max().unwrap_or(self.depth)
        }
    }

    /// Collect all spans in pre-order traversal.
    pub fn flatten(&self) -> Vec<&Span> {
        let mut result = vec![&self.span];
        for child in &self.children {
            result.extend(child.flatten());
        }
        result
    }
}

/// Reconstruct a trace tree from a flat list of spans.
pub fn reconstruct_trace_tree(spans: Vec<Span>) -> Vec<TraceTreeNode> {
    if spans.is_empty() {
        return Vec::new();
    }

    // Find roots (no parent)
    let root_ids: Vec<String> = spans
        .iter()
        .filter(|s| s.parent_span_id.is_none())
        .map(|s| s.span_id.clone())
        .collect();

    // Build children map: parent_id -> [spans]
    let mut children_map: HashMap<String, Vec<&Span>> = HashMap::new();
    for span in &spans {
        if let Some(parent_id) = &span.parent_span_id {
            children_map
                .entry(parent_id.clone())
                .or_default()
                .push(span);
        }
    }

    fn build_tree<'a>(
        span: &'a Span,
        children_map: &HashMap<String, Vec<&'a Span>>,
        depth: usize,
    ) -> TraceTreeNode {
        let mut node = TraceTreeNode::new(span.clone(), depth);
        if let Some(children) = children_map.get(&span.span_id) {
            for child_span in children {
                let child_node = build_tree(child_span, children_map, depth + 1);
                node.children.push(child_node);
            }
        }
        node
    }

    spans
        .iter()
        .filter(|s| root_ids.contains(&s.span_id))
        .map(|root| build_tree(root, &children_map, 0))
        .collect()
}

// ── Trace Collector ──

/// Collects spans and provides trace-level operations.
#[derive(Debug, Default)]
pub struct TraceCollector {
    spans: Vec<Span>,
}

impl TraceCollector {
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Start a new trace with a root span.
    pub fn start_trace(&mut self, name: &str) -> (String, String) {
        let trace_id = generate_trace_id();
        let span = Span::new(&trace_id, name);
        let span_id = span.span_id.clone();
        self.spans.push(span);
        (trace_id, span_id)
    }

    /// Start a new child span.
    pub fn start_span(
        &mut self,
        trace_id: &str,
        parent_span_id: &str,
        name: &str,
    ) -> String {
        let span = Span::new(trace_id, name).with_parent(parent_span_id);
        let span_id = span.span_id.clone();
        self.spans.push(span);
        span_id
    }

    /// Finish a span by ID.
    pub fn finish_span(&mut self, span_id: &str) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.span_id == span_id) {
            span.finish();
        }
    }

    /// Set status on a span.
    pub fn set_span_status(&mut self, span_id: &str, status: SpanStatus, message: Option<&str>) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.span_id == span_id) {
            span.set_status(status, message);
        }
    }

    /// Add a tag to a span.
    pub fn set_span_tag(&mut self, span_id: &str, key: &str, value: &str) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.span_id == span_id) {
            span.set_tag(key, value);
        }
    }

    /// Add an event to a span.
    pub fn add_span_event(&mut self, span_id: &str, event: SpanEvent) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.span_id == span_id) {
            span.add_event(event);
        }
    }

    /// Get all spans for a trace.
    pub fn get_trace(&self, trace_id: &str) -> Vec<&Span> {
        self.spans
            .iter()
            .filter(|s| s.trace_id == trace_id)
            .collect()
    }

    /// Get a specific span.
    pub fn get_span(&self, span_id: &str) -> Option<&Span> {
        self.spans.iter().find(|s| s.span_id == span_id)
    }

    /// Reconstruct a trace as a tree.
    pub fn trace_tree(&self, trace_id: &str) -> Vec<TraceTreeNode> {
        let trace_spans: Vec<Span> = self
            .spans
            .iter()
            .filter(|s| s.trace_id == trace_id)
            .cloned()
            .collect();
        reconstruct_trace_tree(trace_spans)
    }

    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    /// List all unique trace IDs.
    pub fn trace_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .spans
            .iter()
            .map(|s| s.trace_id.clone())
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_creation() {
        let span = Span::new("trace-1", "GET /api/users");
        assert_eq!(span.trace_id, "trace-1");
        assert!(span.is_root());
        assert!(!span.is_finished());
        assert_eq!(span.status, SpanStatus::Unset);
    }

    #[test]
    fn test_span_with_parent() {
        let span = Span::new("trace-1", "db.query").with_parent("parent-span-1");
        assert!(!span.is_root());
        assert_eq!(span.parent_span_id.as_deref(), Some("parent-span-1"));
    }

    #[test]
    fn test_span_finish() {
        let mut span = Span::new("trace-1", "operation");
        span.finish();
        assert!(span.is_finished());
        assert!(span.duration_ms().is_some());
        assert!(span.duration_ms().unwrap() >= 0.0);
    }

    #[test]
    fn test_span_tags() {
        let mut span = Span::new("trace-1", "op");
        span.set_tag("http.method", "GET");
        span.set_tag("http.status_code", "200");
        assert_eq!(span.tags.get("http.method").unwrap(), "GET");
        assert_eq!(span.tags.get("http.status_code").unwrap(), "200");
    }

    #[test]
    fn test_span_events() {
        let mut span = Span::new("trace-1", "op");
        span.add_event(SpanEvent::new("cache_hit").with_attribute("key", "user:42"));
        assert_eq!(span.events.len(), 1);
        assert_eq!(span.events[0].name, "cache_hit");
    }

    #[test]
    fn test_span_status() {
        let mut span = Span::new("trace-1", "op");
        span.set_status(SpanStatus::Error, Some("timeout"));
        assert_eq!(span.status, SpanStatus::Error);
        assert_eq!(span.status_message.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_baggage() {
        let mut baggage = Baggage::new();
        baggage.set("user_id", "42");
        baggage.set("tenant", "acme");
        assert_eq!(baggage.get("user_id"), Some("42"));
        assert_eq!(baggage.len(), 2);
        baggage.remove("user_id");
        assert_eq!(baggage.len(), 1);
    }

    #[test]
    fn test_trace_context_new() {
        let ctx = TraceContext::new();
        assert!(!ctx.trace_id.is_empty());
        assert!(!ctx.span_id.is_empty());
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_trace_context_sampled() {
        let mut ctx = TraceContext::new();
        assert!(ctx.is_sampled());
        ctx.set_sampled(false);
        assert!(!ctx.is_sampled());
        ctx.set_sampled(true);
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_trace_context_inject_extract() {
        let mut ctx = TraceContext::new();
        ctx.baggage.set("user_id", "42");
        let mut headers = HashMap::new();
        ctx.inject(&mut headers);
        assert!(headers.contains_key("traceparent"));
        assert!(headers.contains_key("baggage"));

        let extracted = TraceContext::extract(&headers).unwrap();
        assert_eq!(extracted.trace_id, ctx.trace_id);
        assert_eq!(extracted.span_id, ctx.span_id);
        assert_eq!(extracted.baggage.get("user_id"), Some("42"));
    }

    #[test]
    fn test_trace_context_extract_invalid() {
        let mut headers = HashMap::new();
        headers.insert("traceparent".to_string(), "invalid".to_string());
        assert!(TraceContext::extract(&headers).is_none());
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::new();
        let child = parent.child();
        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
    }

    #[test]
    fn test_trace_context_from_span() {
        let span = Span::new("trace-abc", "root");
        let ctx = TraceContext::from_span(&span);
        assert_eq!(ctx.trace_id, "trace-abc");
        assert_eq!(ctx.span_id, span.span_id);
    }

    #[test]
    fn test_trace_collector_basic() {
        let mut collector = TraceCollector::new();
        let (trace_id, root_id) = collector.start_trace("root");
        let child_id = collector.start_span(&trace_id, &root_id, "child");
        collector.finish_span(&child_id);
        collector.finish_span(&root_id);
        assert_eq!(collector.span_count(), 2);
    }

    #[test]
    fn test_trace_collector_get_trace() {
        let mut collector = TraceCollector::new();
        let (trace_id, _root_id) = collector.start_trace("root");
        let spans = collector.get_trace(&trace_id);
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn test_trace_tree_reconstruction() {
        let mut collector = TraceCollector::new();
        let (trace_id, root_id) = collector.start_trace("root");
        let child_a = collector.start_span(&trace_id, &root_id, "child-a");
        let child_b = collector.start_span(&trace_id, &root_id, "child-b");
        let _grandchild = collector.start_span(&trace_id, &child_a, "grandchild");

        let trees = collector.trace_tree(&trace_id);
        assert_eq!(trees.len(), 1); // One root
        let root = &trees[0];
        assert_eq!(root.children.len(), 2); // Two children
        assert_eq!(root.span_count(), 4); // Total 4 spans
    }

    #[test]
    fn test_trace_tree_max_depth() {
        let mut collector = TraceCollector::new();
        let (trace_id, root_id) = collector.start_trace("root");
        let child = collector.start_span(&trace_id, &root_id, "child");
        let _grandchild = collector.start_span(&trace_id, &child, "grandchild");

        let trees = collector.trace_tree(&trace_id);
        assert_eq!(trees[0].max_depth(), 2);
    }

    #[test]
    fn test_trace_tree_flatten() {
        let mut collector = TraceCollector::new();
        let (trace_id, root_id) = collector.start_trace("root");
        let _child = collector.start_span(&trace_id, &root_id, "child");

        let trees = collector.trace_tree(&trace_id);
        let flat = trees[0].flatten();
        assert_eq!(flat.len(), 2);
    }

    #[test]
    fn test_trace_ids() {
        let mut collector = TraceCollector::new();
        let (t1, _) = collector.start_trace("trace-a");
        let (t2, _) = collector.start_trace("trace-b");
        let ids = collector.trace_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&t1));
        assert!(ids.contains(&t2));
    }

    #[test]
    fn test_span_kind_as_str() {
        assert_eq!(SpanKind::Internal.as_str(), "INTERNAL");
        assert_eq!(SpanKind::Client.as_str(), "CLIENT");
        assert_eq!(SpanKind::Server.as_str(), "SERVER");
        assert_eq!(SpanKind::Producer.as_str(), "PRODUCER");
        assert_eq!(SpanKind::Consumer.as_str(), "CONSUMER");
    }

    #[test]
    fn test_span_with_service() {
        let span = Span::new("trace-1", "op").with_service("auth-service");
        assert_eq!(span.service_name.as_deref(), Some("auth-service"));
    }

    #[test]
    fn test_collector_set_span_tag() {
        let mut collector = TraceCollector::new();
        let (_, root_id) = collector.start_trace("root");
        collector.set_span_tag(&root_id, "env", "prod");
        let span = collector.get_span(&root_id).unwrap();
        assert_eq!(span.tags.get("env").unwrap(), "prod");
    }
}
