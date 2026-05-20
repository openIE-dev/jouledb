//! Chain of responsibility — sequential handler pipeline with short-circuit.
//!
//! Provides a `Handler` trait, a `HandlerChain` that processes requests through
//! an ordered sequence of handlers, dynamic chain modification, fallback
//! handler support, and handler metadata.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Request / Response ─────────────────────────────────────────────

/// A request flowing through the chain.
#[derive(Debug, Clone)]
pub struct Request {
    pub kind: String,
    pub payload: Value,
    pub metadata: HashMap<String, String>,
}

impl Request {
    pub fn new(kind: impl Into<String>, payload: Value) -> Self {
        Self {
            kind: kind.into(),
            payload,
            metadata: HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Result of processing a request.
#[derive(Debug, Clone)]
pub enum HandleResult {
    /// Handled — stop the chain.
    Handled(Value),
    /// Not handled — pass to the next handler.
    Skip,
    /// Error — stop the chain with an error message.
    Error(String),
}

// ── Handler trait ──────────────────────────────────────────────────

/// Metadata about a handler.
#[derive(Debug, Clone)]
pub struct HandlerMeta {
    pub name: String,
    pub description: String,
    pub priority: i32,
}

impl HandlerMeta {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            priority: 0,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// A handler in the chain.
pub trait Handler {
    /// Metadata for this handler.
    fn meta(&self) -> HandlerMeta;

    /// Process a request. Returns `Handled`, `Skip`, or `Error`.
    fn handle(&self, request: &Request) -> HandleResult;

    /// Whether this handler can potentially process the given request kind.
    fn can_handle(&self, kind: &str) -> bool;
}

// ── Closure-based handler ──────────────────────────────────────────

/// A handler built from closures for quick prototyping.
pub struct FnHandler {
    meta: HandlerMeta,
    kinds: Vec<String>,
    handler_fn: Box<dyn Fn(&Request) -> HandleResult>,
}

impl FnHandler {
    pub fn new(
        meta: HandlerMeta,
        kinds: Vec<String>,
        f: impl Fn(&Request) -> HandleResult + 'static,
    ) -> Self {
        Self {
            meta,
            kinds,
            handler_fn: Box::new(f),
        }
    }
}

impl fmt::Debug for FnHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnHandler")
            .field("meta", &self.meta)
            .field("kinds", &self.kinds)
            .finish()
    }
}

impl Handler for FnHandler {
    fn meta(&self) -> HandlerMeta {
        self.meta.clone()
    }

    fn handle(&self, request: &Request) -> HandleResult {
        (self.handler_fn)(request)
    }

    fn can_handle(&self, kind: &str) -> bool {
        self.kinds.is_empty() || self.kinds.iter().any(|k| k == kind)
    }
}

// ── Chain processing result ────────────────────────────────────────

/// The outcome of running a request through the chain.
#[derive(Debug, Clone)]
pub struct ChainResult {
    /// Which handler handled the request (if any).
    pub handled_by: Option<String>,
    /// The result value.
    pub result: HandleResult,
    /// Number of handlers inspected before resolution.
    pub handlers_checked: usize,
}

// ── HandlerChain ───────────────────────────────────────────────────

/// An ordered chain of handlers with optional fallback.
pub struct HandlerChain {
    handlers: Vec<Box<dyn Handler>>,
    fallback: Option<Box<dyn Handler>>,
    process_count: u64,
}

impl HandlerChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            fallback: None,
            process_count: 0,
        }
    }

    /// Append a handler to the end of the chain.
    pub fn add_handler(&mut self, handler: Box<dyn Handler>) {
        self.handlers.push(handler);
    }

    /// Insert a handler at a specific position.
    pub fn insert_handler(&mut self, index: usize, handler: Box<dyn Handler>) {
        let idx = index.min(self.handlers.len());
        self.handlers.insert(idx, handler);
    }

    /// Remove the handler at the given index.
    pub fn remove_handler(&mut self, index: usize) -> bool {
        if index < self.handlers.len() {
            self.handlers.remove(index);
            true
        } else {
            false
        }
    }

    /// Remove the first handler whose name matches.
    pub fn remove_by_name(&mut self, name: &str) -> bool {
        if let Some(idx) = self.handlers.iter().position(|h| h.meta().name == name) {
            self.handlers.remove(idx);
            true
        } else {
            false
        }
    }

    /// Set the fallback handler (used when no handler in the chain handles the request).
    pub fn set_fallback(&mut self, handler: Box<dyn Handler>) {
        self.fallback = Some(handler);
    }

    /// Number of handlers in the chain (excluding fallback).
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the chain has no handlers.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Sort handlers by priority (highest first).
    pub fn sort_by_priority(&mut self) {
        self.handlers.sort_by(|a, b| b.meta().priority.cmp(&a.meta().priority));
    }

    /// Metadata for all handlers, in chain order.
    pub fn handler_metadata(&self) -> Vec<HandlerMeta> {
        self.handlers.iter().map(|h| h.meta()).collect()
    }

    /// How many times `process` has been called.
    pub fn process_count(&self) -> u64 {
        self.process_count
    }

    /// Process a request through the chain.
    ///
    /// Handlers are tried in order. The first handler that returns
    /// `Handled` or `Error` stops the chain. If all handlers return
    /// `Skip`, the fallback is tried.
    pub fn process(&mut self, request: &Request) -> ChainResult {
        self.process_count += 1;
        let mut checked = 0;

        for handler in &self.handlers {
            checked += 1;
            if !handler.can_handle(&request.kind) {
                continue;
            }
            match handler.handle(request) {
                HandleResult::Handled(val) => {
                    return ChainResult {
                        handled_by: Some(handler.meta().name),
                        result: HandleResult::Handled(val),
                        handlers_checked: checked,
                    };
                }
                HandleResult::Error(msg) => {
                    return ChainResult {
                        handled_by: Some(handler.meta().name),
                        result: HandleResult::Error(msg),
                        handlers_checked: checked,
                    };
                }
                HandleResult::Skip => {}
            }
        }

        // Try fallback.
        if let Some(fb) = &self.fallback {
            checked += 1;
            let result = fb.handle(request);
            let handled_by = match &result {
                HandleResult::Skip => None,
                _ => Some(fb.meta().name),
            };
            return ChainResult {
                handled_by,
                result,
                handlers_checked: checked,
            };
        }

        ChainResult {
            handled_by: None,
            result: HandleResult::Skip,
            handlers_checked: checked,
        }
    }
}

impl Default for HandlerChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn log_handler() -> FnHandler {
        FnHandler::new(
            HandlerMeta::new("logger").with_priority(10),
            vec!["log".to_string()],
            |req| {
                HandleResult::Handled(Value::String(format!("logged: {}", req.payload)))
            },
        )
    }

    fn auth_handler() -> FnHandler {
        FnHandler::new(
            HandlerMeta::new("auth").with_priority(20),
            vec!["auth".to_string()],
            |req| {
                if req.metadata.get("token").is_some() {
                    HandleResult::Handled(Value::String("authenticated".to_string()))
                } else {
                    HandleResult::Error("no token".to_string())
                }
            },
        )
    }

    fn skip_handler() -> FnHandler {
        FnHandler::new(
            HandlerMeta::new("skipper").with_priority(5),
            vec![],
            |_| HandleResult::Skip,
        )
    }

    #[test]
    fn basic_handling() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));

        let req = Request::new("log", Value::String("hello".to_string()));
        let result = chain.process(&req);
        assert!(result.handled_by.is_some());
        assert_eq!(result.handled_by.unwrap(), "logger");
    }

    #[test]
    fn short_circuit_on_handled() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        chain.add_handler(Box::new(auth_handler()));

        let req = Request::new("log", Value::String("test".to_string()));
        let result = chain.process(&req);
        assert_eq!(result.handled_by.unwrap(), "logger");
        // The auth handler should not be reached for a "log" request handled by logger.
        assert_eq!(result.handlers_checked, 1);
    }

    #[test]
    fn skip_to_next_handler() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(skip_handler()));
        chain.add_handler(Box::new(log_handler()));

        let req = Request::new("log", Value::String("data".to_string()));
        let result = chain.process(&req);
        assert_eq!(result.handled_by.unwrap(), "logger");
        assert_eq!(result.handlers_checked, 2);
    }

    #[test]
    fn error_stops_chain() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(auth_handler()));
        chain.add_handler(Box::new(log_handler()));

        let req = Request::new("auth", Value::Null);
        let result = chain.process(&req);
        assert_eq!(result.handled_by.unwrap(), "auth");
        match result.result {
            HandleResult::Error(msg) => assert_eq!(msg, "no token"),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn successful_auth() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(auth_handler()));

        let req = Request::new("auth", Value::Null).with_meta("token", "abc123");
        let result = chain.process(&req);
        match result.result {
            HandleResult::Handled(v) => assert_eq!(v, Value::String("authenticated".to_string())),
            _ => panic!("expected handled"),
        }
    }

    #[test]
    fn fallback_handler() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        chain.set_fallback(Box::new(FnHandler::new(
            HandlerMeta::new("fallback"),
            vec![],
            |_| HandleResult::Handled(Value::String("default".to_string())),
        )));

        let req = Request::new("unknown", Value::Null);
        let result = chain.process(&req);
        assert_eq!(result.handled_by.unwrap(), "fallback");
    }

    #[test]
    fn no_handler_matches() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));

        let req = Request::new("unknown", Value::Null);
        let result = chain.process(&req);
        assert!(result.handled_by.is_none());
        match result.result {
            HandleResult::Skip => {}
            _ => panic!("expected skip"),
        }
    }

    #[test]
    fn insert_handler_at_position() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(skip_handler()));
        chain.add_handler(Box::new(log_handler()));

        // Insert auth at position 1 (between skip and log).
        chain.insert_handler(1, Box::new(auth_handler()));
        let meta = chain.handler_metadata();
        assert_eq!(meta.len(), 3);
        assert_eq!(meta[0].name, "skipper");
        assert_eq!(meta[1].name, "auth");
        assert_eq!(meta[2].name, "logger");
    }

    #[test]
    fn remove_handler_by_index() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        chain.add_handler(Box::new(auth_handler()));
        assert!(chain.remove_handler(0));
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.handler_metadata()[0].name, "auth");
    }

    #[test]
    fn remove_handler_by_name() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        chain.add_handler(Box::new(auth_handler()));
        assert!(chain.remove_by_name("logger"));
        assert_eq!(chain.len(), 1);
        assert!(!chain.remove_by_name("nonexistent"));
    }

    #[test]
    fn remove_handler_out_of_bounds() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        assert!(!chain.remove_handler(5));
    }

    #[test]
    fn sort_by_priority() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(skip_handler()));   // priority 5
        chain.add_handler(Box::new(auth_handler()));    // priority 20
        chain.add_handler(Box::new(log_handler()));     // priority 10

        chain.sort_by_priority();
        let meta = chain.handler_metadata();
        assert_eq!(meta[0].name, "auth");
        assert_eq!(meta[1].name, "logger");
        assert_eq!(meta[2].name, "skipper");
    }

    #[test]
    fn process_count_tracking() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        assert_eq!(chain.process_count(), 0);

        let req = Request::new("log", Value::Null);
        chain.process(&req);
        chain.process(&req);
        assert_eq!(chain.process_count(), 2);
    }

    #[test]
    fn can_handle_empty_kinds_matches_all() {
        let h = skip_handler();
        assert!(h.can_handle("anything"));
        assert!(h.can_handle("other"));
    }

    #[test]
    fn can_handle_specific_kinds() {
        let h = log_handler();
        assert!(h.can_handle("log"));
        assert!(!h.can_handle("auth"));
    }

    #[test]
    fn request_with_metadata() {
        let req = Request::new("test", Value::Null)
            .with_meta("key1", "val1")
            .with_meta("key2", "val2");
        assert_eq!(req.metadata.get("key1").unwrap(), "val1");
        assert_eq!(req.metadata.get("key2").unwrap(), "val2");
    }

    #[test]
    fn empty_chain_returns_skip() {
        let mut chain = HandlerChain::new();
        let req = Request::new("test", Value::Null);
        let result = chain.process(&req);
        assert!(result.handled_by.is_none());
        assert_eq!(result.handlers_checked, 0);
    }

    #[test]
    fn handler_meta_description() {
        let meta = HandlerMeta::new("test")
            .with_description("A test handler")
            .with_priority(42);
        assert_eq!(meta.name, "test");
        assert_eq!(meta.description, "A test handler");
        assert_eq!(meta.priority, 42);
    }

    #[test]
    fn insert_beyond_length_appends() {
        let mut chain = HandlerChain::new();
        chain.add_handler(Box::new(log_handler()));
        chain.insert_handler(100, Box::new(auth_handler()));
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.handler_metadata()[1].name, "auth");
    }
}
