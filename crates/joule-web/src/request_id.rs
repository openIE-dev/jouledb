//! Request ID propagation.
//!
//! Generation, extraction from headers, propagation through context,
//! correlation chains, and request tree tracking. Pure Rust — no
//! external UUID library beyond the workspace `uuid` crate.

use std::collections::HashMap;
use std::fmt;

// ── Request ID ──────────────────────────────────────────────────

/// A unique identifier for a request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);

impl RequestId {
    /// Create from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Generate a new request ID from a counter and a prefix.
    pub fn generate(prefix: &str, counter: u64) -> Self {
        Self(format!("{prefix}-{counter:016x}"))
    }

    /// Generate from raw bytes (hex-encoded).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        Self(hex)
    }

    pub fn as_str(&self) -> &str { &self.0 }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── ID Generator ────────────────────────────────────────────────

/// Sequential request ID generator with a configurable prefix.
#[derive(Debug, Clone)]
pub struct IdGenerator {
    prefix: String,
    counter: u64,
}

impl IdGenerator {
    pub fn new(prefix: &str) -> Self {
        Self { prefix: prefix.to_string(), counter: 0 }
    }

    pub fn with_start(prefix: &str, start: u64) -> Self {
        Self { prefix: prefix.to_string(), counter: start }
    }

    /// Generate the next request ID.
    pub fn next(&mut self) -> RequestId {
        let id = RequestId::generate(&self.prefix, self.counter);
        self.counter += 1;
        id
    }

    pub fn counter(&self) -> u64 { self.counter }
    pub fn prefix(&self) -> &str { &self.prefix }
}

// ── Header extraction ───────────────────────────────────────────

/// Well-known header names for request IDs.
pub const HEADER_X_REQUEST_ID: &str = "x-request-id";
pub const HEADER_X_CORRELATION_ID: &str = "x-correlation-id";
pub const HEADER_X_TRACE_ID: &str = "x-trace-id";
pub const HEADER_TRACEPARENT: &str = "traceparent";

/// Extract a request ID from headers, trying multiple header names.
/// Returns the first match.
pub fn extract_request_id(headers: &HashMap<String, String>) -> Option<RequestId> {
    let candidates = [
        HEADER_X_REQUEST_ID,
        HEADER_X_CORRELATION_ID,
        HEADER_X_TRACE_ID,
    ];
    for name in &candidates {
        // Case-insensitive lookup.
        for (key, value) in headers {
            if key.to_lowercase() == *name && !value.is_empty() {
                return Some(RequestId::from_string(value.clone()));
            }
        }
    }
    None
}

/// Extract a trace-id from a W3C traceparent header.
/// Format: `version-trace_id-parent_id-flags` e.g. `00-abc123-def456-01`.
pub fn extract_traceparent_trace_id(value: &str) -> Option<RequestId> {
    let parts: Vec<&str> = value.split('-').collect();
    if parts.len() >= 2 && !parts[1].is_empty() {
        Some(RequestId::from_string(parts[1]))
    } else {
        None
    }
}

/// Build response headers for propagation.
pub fn propagation_headers(id: &RequestId) -> HashMap<String, String> {
    let mut h = HashMap::new();
    h.insert(HEADER_X_REQUEST_ID.to_string(), id.to_string());
    h
}

// ── Request context ─────────────────────────────────────────────

/// Context that carries the request ID and correlation chain.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// The primary request ID.
    pub request_id: RequestId,
    /// Correlation ID (may differ from request_id in chained calls).
    pub correlation_id: RequestId,
    /// Parent request ID, if this is a child request.
    pub parent_id: Option<RequestId>,
    /// Arbitrary baggage propagated through the chain.
    pub baggage: HashMap<String, String>,
}

impl RequestContext {
    /// Create a root context (no parent).
    pub fn root(id: RequestId) -> Self {
        let corr = id.clone();
        Self {
            request_id: id,
            correlation_id: corr,
            parent_id: None,
            baggage: HashMap::new(),
        }
    }

    /// Create a child context inheriting the correlation ID.
    pub fn child(&self, child_id: RequestId) -> Self {
        Self {
            request_id: child_id,
            correlation_id: self.correlation_id.clone(),
            parent_id: Some(self.request_id.clone()),
            baggage: self.baggage.clone(),
        }
    }

    /// Set a baggage value.
    pub fn set_baggage(&mut self, key: &str, value: &str) {
        self.baggage.insert(key.to_string(), value.to_string());
    }

    /// Get a baggage value.
    pub fn get_baggage(&self, key: &str) -> Option<&str> {
        self.baggage.get(key).map(|s| s.as_str())
    }

    /// Build headers for outgoing requests.
    pub fn outgoing_headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert(HEADER_X_REQUEST_ID.to_string(), self.request_id.to_string());
        h.insert(HEADER_X_CORRELATION_ID.to_string(), self.correlation_id.to_string());
        h
    }

    /// Is this a root context (no parent)?
    pub fn is_root(&self) -> bool { self.parent_id.is_none() }

    /// Depth in the call chain (root = 0).
    pub fn depth(&self) -> usize {
        if self.parent_id.is_some() { 1 } else { 0 }
    }
}

// ── Request tree ────────────────────────────────────────────────

/// A node in a request tree for tracking parent-child relationships.
#[derive(Debug, Clone)]
pub struct RequestNode {
    pub id: RequestId,
    pub parent_id: Option<RequestId>,
    pub service: String,
    pub started_ms: u64,
    pub completed_ms: Option<u64>,
}

impl RequestNode {
    pub fn duration_ms(&self) -> Option<u64> {
        self.completed_ms.map(|end| end.saturating_sub(self.started_ms))
    }

    pub fn is_complete(&self) -> bool {
        self.completed_ms.is_some()
    }
}

/// Tracks a tree of requests for distributed tracing.
#[derive(Debug, Clone)]
pub struct RequestTree {
    nodes: HashMap<String, RequestNode>,
    root_id: Option<RequestId>,
}

impl RequestTree {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), root_id: None }
    }

    /// Add a root request.
    pub fn add_root(&mut self, id: RequestId, service: &str, started_ms: u64) {
        self.root_id = Some(id.clone());
        self.nodes.insert(id.0.clone(), RequestNode {
            id,
            parent_id: None,
            service: service.to_string(),
            started_ms,
            completed_ms: None,
        });
    }

    /// Add a child request.
    pub fn add_child(&mut self, id: RequestId, parent_id: RequestId, service: &str, started_ms: u64) {
        self.nodes.insert(id.0.clone(), RequestNode {
            id,
            parent_id: Some(parent_id),
            service: service.to_string(),
            started_ms,
            completed_ms: None,
        });
    }

    /// Mark a request as completed.
    pub fn complete(&mut self, id: &RequestId, completed_ms: u64) -> bool {
        if let Some(node) = self.nodes.get_mut(&id.0) {
            node.completed_ms = Some(completed_ms);
            true
        } else {
            false
        }
    }

    /// Get a node by ID.
    pub fn get(&self, id: &RequestId) -> Option<&RequestNode> {
        self.nodes.get(&id.0)
    }

    /// Get the root ID.
    pub fn root_id(&self) -> Option<&RequestId> {
        self.root_id.as_ref()
    }

    /// Get children of a node.
    pub fn children_of(&self, id: &RequestId) -> Vec<&RequestNode> {
        let mut children: Vec<&RequestNode> = self.nodes.values()
            .filter(|n| n.parent_id.as_ref() == Some(id))
            .collect();
        children.sort_by_key(|n| n.started_ms);
        children
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of completed nodes.
    pub fn completed_count(&self) -> usize {
        self.nodes.values().filter(|n| n.is_complete()).count()
    }

    /// All leaf nodes (no children).
    pub fn leaves(&self) -> Vec<&RequestNode> {
        let parent_ids: std::collections::HashSet<&str> = self.nodes.values()
            .filter_map(|n| n.parent_id.as_ref().map(|p| p.0.as_str()))
            .collect();
        let mut leaves: Vec<&RequestNode> = self.nodes.values()
            .filter(|n| !parent_ids.contains(n.id.0.as_str()))
            .collect();
        leaves.sort_by_key(|n| &n.id.0);
        leaves
    }

    /// Maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        let mut max_d = 0;
        for node in self.nodes.values() {
            let d = self.depth_of(node);
            if d > max_d {
                max_d = d;
            }
        }
        max_d
    }

    fn depth_of(&self, node: &RequestNode) -> usize {
        let mut depth = 0;
        let mut current = node;
        while let Some(pid) = &current.parent_id {
            depth += 1;
            if depth > self.nodes.len() {
                break; // cycle guard
            }
            if let Some(parent) = self.nodes.get(&pid.0) {
                current = parent;
            } else {
                break;
            }
        }
        depth
    }

    /// Collect the full chain from a node to the root.
    pub fn chain_to_root(&self, id: &RequestId) -> Vec<&RequestNode> {
        let mut chain = Vec::new();
        let mut current_id = Some(id);
        let limit = self.nodes.len() + 1;
        let mut steps = 0;
        while let Some(cid) = current_id {
            if steps >= limit { break; }
            steps += 1;
            if let Some(node) = self.nodes.get(&cid.0) {
                chain.push(node);
                current_id = node.parent_id.as_ref();
            } else {
                break;
            }
        }
        chain
    }
}

impl Default for RequestTree {
    fn default() -> Self { Self::new() }
}

// ── Correlation chain builder ───────────────────────────────────

/// Builds a correlation chain string from request IDs.
/// Format: `id1 -> id2 -> id3`
pub fn format_correlation_chain(ids: &[&RequestId]) -> String {
    ids.iter().map(|id| id.0.as_str()).collect::<Vec<_>>().join(" -> ")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generate() {
        let id = RequestId::generate("req", 42);
        assert!(id.as_str().starts_with("req-"));
        assert!(id.as_str().contains("2a"));
    }

    #[test]
    fn test_request_id_from_string() {
        let id = RequestId::from_string("abc-123");
        assert_eq!(id.as_str(), "abc-123");
        assert_eq!(id.to_string(), "abc-123");
    }

    #[test]
    fn test_request_id_from_bytes() {
        let id = RequestId::from_bytes(&[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(id.as_str(), "deadbeef");
    }

    #[test]
    fn test_id_generator_sequential() {
        let mut id_gen = IdGenerator::new("svc");
        let id0 = id_gen.next();
        let id1 = id_gen.next();
        let id2 = id_gen.next();

        assert_ne!(id0, id1);
        assert_ne!(id1, id2);
        assert_eq!(id_gen.counter(), 3);
        assert_eq!(id_gen.prefix(), "svc");
    }

    #[test]
    fn test_id_generator_with_start() {
        let mut id_gen = IdGenerator::with_start("tx", 100);
        let id = id_gen.next();
        assert!(id.as_str().contains("64")); // 100 = 0x64
        assert_eq!(id_gen.counter(), 101);
    }

    #[test]
    fn test_extract_request_id_x_request_id() {
        let mut headers = HashMap::new();
        headers.insert("X-Request-Id".to_string(), "abc-123".to_string());
        let id = extract_request_id(&headers).unwrap();
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn test_extract_request_id_correlation() {
        let mut headers = HashMap::new();
        headers.insert("x-correlation-id".to_string(), "corr-456".to_string());
        let id = extract_request_id(&headers).unwrap();
        assert_eq!(id.as_str(), "corr-456");
    }

    #[test]
    fn test_extract_request_id_trace() {
        let mut headers = HashMap::new();
        headers.insert("x-trace-id".to_string(), "trace-789".to_string());
        let id = extract_request_id(&headers).unwrap();
        assert_eq!(id.as_str(), "trace-789");
    }

    #[test]
    fn test_extract_request_id_empty_value() {
        let mut headers = HashMap::new();
        headers.insert("x-request-id".to_string(), "".to_string());
        assert!(extract_request_id(&headers).is_none());
    }

    #[test]
    fn test_extract_request_id_none() {
        let headers = HashMap::new();
        assert!(extract_request_id(&headers).is_none());
    }

    #[test]
    fn test_extract_traceparent() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let id = extract_traceparent_trace_id(tp).unwrap();
        assert_eq!(id.as_str(), "4bf92f3577b34da6a3ce929d0e0e4736");
    }

    #[test]
    fn test_extract_traceparent_invalid() {
        assert!(extract_traceparent_trace_id("").is_none());
        assert!(extract_traceparent_trace_id("00-").is_none());
    }

    #[test]
    fn test_propagation_headers() {
        let id = RequestId::from_string("req-1");
        let h = propagation_headers(&id);
        assert_eq!(h.get(HEADER_X_REQUEST_ID).unwrap(), "req-1");
    }

    #[test]
    fn test_request_context_root() {
        let id = RequestId::from_string("root-1");
        let ctx = RequestContext::root(id.clone());
        assert!(ctx.is_root());
        assert_eq!(ctx.depth(), 0);
        assert_eq!(ctx.request_id, id);
        assert_eq!(ctx.correlation_id, id);
    }

    #[test]
    fn test_request_context_child() {
        let root_id = RequestId::from_string("root-1");
        let child_id = RequestId::from_string("child-1");

        let root_ctx = RequestContext::root(root_id.clone());
        let child_ctx = root_ctx.child(child_id.clone());

        assert!(!child_ctx.is_root());
        assert_eq!(child_ctx.request_id, child_id);
        assert_eq!(child_ctx.correlation_id, root_id);
        assert_eq!(child_ctx.parent_id, Some(root_id));
    }

    #[test]
    fn test_request_context_baggage() {
        let id = RequestId::from_string("req");
        let mut ctx = RequestContext::root(id);
        ctx.set_baggage("user-id", "u-123");
        ctx.set_baggage("tenant", "acme");

        assert_eq!(ctx.get_baggage("user-id"), Some("u-123"));
        assert_eq!(ctx.get_baggage("tenant"), Some("acme"));
        assert_eq!(ctx.get_baggage("missing"), None);
    }

    #[test]
    fn test_request_context_baggage_propagation() {
        let root = RequestContext::root(RequestId::from_string("r1"));
        let mut root_with_baggage = root;
        root_with_baggage.set_baggage("env", "prod");

        let child = root_with_baggage.child(RequestId::from_string("c1"));
        assert_eq!(child.get_baggage("env"), Some("prod"));
    }

    #[test]
    fn test_request_context_outgoing_headers() {
        let id = RequestId::from_string("req-42");
        let ctx = RequestContext::root(id);
        let h = ctx.outgoing_headers();
        assert_eq!(h.get(HEADER_X_REQUEST_ID).unwrap(), "req-42");
        assert_eq!(h.get(HEADER_X_CORRELATION_ID).unwrap(), "req-42");
    }

    #[test]
    fn test_request_tree_basic() {
        let mut tree = RequestTree::new();
        let root = RequestId::from_string("r1");
        let c1 = RequestId::from_string("c1");
        let c2 = RequestId::from_string("c2");

        tree.add_root(root.clone(), "gateway", 100);
        tree.add_child(c1.clone(), root.clone(), "auth", 110);
        tree.add_child(c2.clone(), root.clone(), "data", 120);

        assert_eq!(tree.node_count(), 3);
        assert_eq!(tree.root_id(), Some(&root));
        assert_eq!(tree.children_of(&root).len(), 2);
    }

    #[test]
    fn test_request_tree_complete() {
        let mut tree = RequestTree::new();
        let root = RequestId::from_string("r1");
        tree.add_root(root.clone(), "svc", 100);

        assert!(!tree.get(&root).unwrap().is_complete());
        assert!(tree.complete(&root, 200));
        assert!(tree.get(&root).unwrap().is_complete());
        assert_eq!(tree.get(&root).unwrap().duration_ms(), Some(100));
    }

    #[test]
    fn test_request_tree_complete_unknown() {
        let mut tree = RequestTree::new();
        assert!(!tree.complete(&RequestId::from_string("ghost"), 100));
    }

    #[test]
    fn test_request_tree_leaves() {
        let mut tree = RequestTree::new();
        let r = RequestId::from_string("r");
        let c1 = RequestId::from_string("c1");
        let c2 = RequestId::from_string("c2");
        let gc = RequestId::from_string("gc");

        tree.add_root(r.clone(), "gw", 0);
        tree.add_child(c1.clone(), r.clone(), "a", 10);
        tree.add_child(c2.clone(), r.clone(), "b", 20);
        tree.add_child(gc.clone(), c1.clone(), "c", 30);

        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 2);
        // c2 and gc are leaves
        let leaf_ids: Vec<&str> = leaves.iter().map(|n| n.id.as_str()).collect();
        assert!(leaf_ids.contains(&"c2"));
        assert!(leaf_ids.contains(&"gc"));
    }

    #[test]
    fn test_request_tree_max_depth() {
        let mut tree = RequestTree::new();
        let r = RequestId::from_string("r");
        let c1 = RequestId::from_string("c1");
        let gc = RequestId::from_string("gc");
        let ggc = RequestId::from_string("ggc");

        tree.add_root(r.clone(), "svc", 0);
        tree.add_child(c1.clone(), r.clone(), "svc", 10);
        tree.add_child(gc.clone(), c1.clone(), "svc", 20);
        tree.add_child(ggc.clone(), gc.clone(), "svc", 30);

        assert_eq!(tree.max_depth(), 3);
    }

    #[test]
    fn test_request_tree_chain_to_root() {
        let mut tree = RequestTree::new();
        let r = RequestId::from_string("r");
        let c = RequestId::from_string("c");
        let gc = RequestId::from_string("gc");

        tree.add_root(r.clone(), "gw", 0);
        tree.add_child(c.clone(), r.clone(), "a", 10);
        tree.add_child(gc.clone(), c.clone(), "b", 20);

        let chain = tree.chain_to_root(&gc);
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].id, gc);
        assert_eq!(chain[1].id, c);
        assert_eq!(chain[2].id, r);
    }

    #[test]
    fn test_request_tree_completed_count() {
        let mut tree = RequestTree::new();
        let r = RequestId::from_string("r");
        let c = RequestId::from_string("c");

        tree.add_root(r.clone(), "svc", 0);
        tree.add_child(c.clone(), r.clone(), "svc", 10);

        assert_eq!(tree.completed_count(), 0);
        tree.complete(&c, 50);
        assert_eq!(tree.completed_count(), 1);
        tree.complete(&r, 60);
        assert_eq!(tree.completed_count(), 2);
    }

    #[test]
    fn test_format_correlation_chain() {
        let ids = vec![
            RequestId::from_string("a"),
            RequestId::from_string("b"),
            RequestId::from_string("c"),
        ];
        let refs: Vec<&RequestId> = ids.iter().collect();
        assert_eq!(format_correlation_chain(&refs), "a -> b -> c");
    }

    #[test]
    fn test_format_correlation_chain_empty() {
        assert_eq!(format_correlation_chain(&[]), "");
    }

    #[test]
    fn test_request_node_duration() {
        let node = RequestNode {
            id: RequestId::from_string("n"),
            parent_id: None,
            service: "svc".into(),
            started_ms: 100,
            completed_ms: Some(250),
        };
        assert_eq!(node.duration_ms(), Some(150));
        assert!(node.is_complete());
    }

    #[test]
    fn test_request_node_incomplete() {
        let node = RequestNode {
            id: RequestId::from_string("n"),
            parent_id: None,
            service: "svc".into(),
            started_ms: 100,
            completed_ms: None,
        };
        assert_eq!(node.duration_ms(), None);
        assert!(!node.is_complete());
    }
}
