//! Error chaining — structured errors with source chains, key-value context,
//! chain display, downcasting, kind categorization, serialization, and iteration.
//!
//! Pure Rust error chain infrastructure for rich diagnostic errors.
//! Each error carries an optional source chain, contextual key-value pairs,
//! and a categorized kind for programmatic handling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Error Kind ──────────────────────────────────────────────────

/// Categorization of an error for programmatic handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorKind {
    /// Network or I/O failure.
    Network,
    /// Authentication or authorization failure.
    Auth,
    /// Input validation failure.
    Validation,
    /// Resource not found.
    NotFound,
    /// Conflict with existing state.
    Conflict,
    /// Timeout exceeded.
    Timeout,
    /// Rate limit exceeded.
    RateLimited,
    /// Internal / unexpected error.
    Internal,
    /// Upstream service failure.
    Upstream,
    /// Configuration error.
    Configuration,
    /// Cancelled by caller.
    Cancelled,
    /// Unknown category.
    Unknown,
}

impl ErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Auth => "auth",
            Self::Validation => "validation",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Timeout => "timeout",
            Self::RateLimited => "rate_limited",
            Self::Internal => "internal",
            Self::Upstream => "upstream",
            Self::Configuration => "configuration",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this kind of error is retryable by default.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network | Self::Timeout | Self::RateLimited | Self::Upstream
        )
    }

    /// Whether this kind of error is a client fault.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::Auth | Self::Validation | Self::NotFound | Self::Conflict | Self::Cancelled
        )
    }
}

// ── Error Context ───────────────────────────────────────────────

/// Key-value context attached to an error for diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorContext {
    pairs: Vec<(String, String)>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        // Replace existing key.
        if let Some(pos) = self.pairs.iter().position(|(k, _)| *k == key) {
            self.pairs[pos].1 = value.into();
        } else {
            self.pairs.push((key, value.into()));
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.pairs.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        if let Some(pos) = self.pairs.iter().position(|(k, _)| k == key) {
            Some(self.pairs.remove(pos).1)
        } else {
            None
        }
    }

    /// Merge another context into this one (other wins on conflict).
    pub fn merge(&mut self, other: &ErrorContext) {
        for (k, v) in &other.pairs {
            self.insert(k.clone(), v.clone());
        }
    }
}

// ── Chained Error ───────────────────────────────────────────────

/// An error with an optional source chain, context, and kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainedError {
    /// Human-readable message for this error.
    message: String,
    /// Categorized kind.
    kind: ErrorKind,
    /// Contextual key-value pairs.
    context: ErrorContext,
    /// Source error (boxed to allow recursion).
    source: Option<Box<ChainedError>>,
}

impl ChainedError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind,
            context: ErrorContext::new(),
            source: None,
        }
    }

    /// Attach a source error.
    pub fn with_source(mut self, source: ChainedError) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Attach a context key-value pair.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Add context in-place.
    pub fn add_context(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.context.insert(key, value);
    }

    /// Set the source in-place.
    pub fn set_source(&mut self, source: ChainedError) {
        self.source = Some(Box::new(source));
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn context(&self) -> &ErrorContext {
        &self.context
    }

    pub fn source(&self) -> Option<&ChainedError> {
        self.source.as_deref()
    }

    /// Get the root cause (deepest source in chain).
    pub fn root_cause(&self) -> &ChainedError {
        let mut current = self;
        while let Some(src) = current.source.as_deref() {
            current = src;
        }
        current
    }

    /// Depth of the error chain (1 = no source).
    pub fn chain_depth(&self) -> usize {
        let mut depth = 1;
        let mut current = self;
        while let Some(src) = current.source.as_deref() {
            depth += 1;
            current = src;
        }
        depth
    }

    /// Iterate over the chain starting from self.
    pub fn chain_iter(&self) -> ChainIter<'_> {
        ChainIter {
            current: Some(self),
        }
    }

    /// Find the first error in the chain with the given kind.
    pub fn find_kind(&self, kind: ErrorKind) -> Option<&ChainedError> {
        self.chain_iter().find(|e| e.kind == kind)
    }

    /// Whether any error in the chain has the given kind.
    pub fn has_kind(&self, kind: ErrorKind) -> bool {
        self.find_kind(kind).is_some()
    }

    /// Collect all error kinds in the chain.
    pub fn kind_chain(&self) -> Vec<ErrorKind> {
        self.chain_iter().map(|e| e.kind).collect()
    }

    /// Format the full chain as a multi-line display.
    pub fn display_chain(&self) -> String {
        let mut lines = Vec::new();
        for (i, err) in self.chain_iter().enumerate() {
            if i == 0 {
                lines.push(format!("error: {}", err.message));
            } else {
                lines.push(format!("caused by: {}", err.message));
            }
            for (k, v) in err.context.iter() {
                lines.push(format!("  {k} = {v}"));
            }
        }
        lines.join("\n")
    }

    /// Format as a single-line chain (messages joined by `: `).
    pub fn display_oneline(&self) -> String {
        let messages: Vec<&str> = self.chain_iter().map(|e| e.message.as_str()).collect();
        messages.join(": ")
    }

    /// Serialize the chain to JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Deserialize a chain from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Convert to a HashMap representation for logging / telemetry.
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("message".to_string(), self.message.clone());
        map.insert("kind".to_string(), self.kind.as_str().to_string());
        map.insert("chain_depth".to_string(), self.chain_depth().to_string());
        for (k, v) in self.context.iter() {
            map.insert(format!("ctx.{k}"), v.to_string());
        }
        if let Some(src) = &self.source {
            map.insert("source".to_string(), src.message.clone());
            map.insert("root_cause".to_string(), self.root_cause().message.clone());
        }
        map
    }

    /// Wrap another error as a source of a new higher-level error.
    pub fn wrap(kind: ErrorKind, message: impl Into<String>, source: ChainedError) -> Self {
        Self::new(kind, message).with_source(source)
    }

    /// Whether this error (or any in chain) is retryable.
    pub fn is_retryable(&self) -> bool {
        self.chain_iter().any(|e| e.kind.is_retryable())
    }

    /// Whether this error is a client fault.
    pub fn is_client_error(&self) -> bool {
        self.kind.is_client_error()
    }
}

impl std::fmt::Display for ChainedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.kind.as_str(), self.message)?;
        if let Some(src) = &self.source {
            write!(f, ": {src}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ChainedError {}

// ── Chain Iterator ──────────────────────────────────────────────

/// Iterator over the error chain.
pub struct ChainIter<'a> {
    current: Option<&'a ChainedError>,
}

impl<'a> Iterator for ChainIter<'a> {
    type Item = &'a ChainedError;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.current?;
        self.current = item.source.as_deref();
        Some(item)
    }
}

// ── Error Builder ───────────────────────────────────────────────

/// Fluent builder for constructing chained errors.
pub struct ErrorBuilder {
    message: String,
    kind: ErrorKind,
    context: ErrorContext,
    source: Option<ChainedError>,
}

impl ErrorBuilder {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind,
            context: ErrorContext::new(),
            source: None,
        }
    }

    pub fn context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key, value);
        self
    }

    pub fn source(mut self, source: ChainedError) -> Self {
        self.source = Some(source);
        self
    }

    pub fn build(self) -> ChainedError {
        ChainedError {
            message: self.message,
            kind: self.kind,
            context: self.context,
            source: self.source.map(Box::new),
        }
    }
}

// ── Convenience Constructors ────────────────────────────────────

/// Create a network error.
pub fn network_error(msg: impl Into<String>) -> ChainedError {
    ChainedError::new(ErrorKind::Network, msg)
}

/// Create a validation error.
pub fn validation_error(msg: impl Into<String>) -> ChainedError {
    ChainedError::new(ErrorKind::Validation, msg)
}

/// Create a not-found error.
pub fn not_found_error(msg: impl Into<String>) -> ChainedError {
    ChainedError::new(ErrorKind::NotFound, msg)
}

/// Create a timeout error.
pub fn timeout_error(msg: impl Into<String>) -> ChainedError {
    ChainedError::new(ErrorKind::Timeout, msg)
}

/// Create an internal error.
pub fn internal_error(msg: impl Into<String>) -> ChainedError {
    ChainedError::new(ErrorKind::Internal, msg)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_error() {
        let err = ChainedError::new(ErrorKind::Network, "connection refused");
        assert_eq!(err.message(), "connection refused");
        assert_eq!(err.kind(), ErrorKind::Network);
        assert!(err.source().is_none());
        assert_eq!(err.chain_depth(), 1);
    }

    #[test]
    fn test_error_with_context() {
        let err = ChainedError::new(ErrorKind::Network, "connection failed")
            .with_context("host", "example.com")
            .with_context("port", "443");
        assert_eq!(err.context().get("host"), Some("example.com"));
        assert_eq!(err.context().get("port"), Some("443"));
        assert_eq!(err.context().len(), 2);
    }

    #[test]
    fn test_error_chain() {
        let root = ChainedError::new(ErrorKind::Network, "socket timeout");
        let mid = ChainedError::new(ErrorKind::Upstream, "service unavailable").with_source(root);
        let top = ChainedError::new(ErrorKind::Internal, "request failed").with_source(mid);
        assert_eq!(top.chain_depth(), 3);
        assert_eq!(top.root_cause().message(), "socket timeout");
        assert_eq!(top.root_cause().kind(), ErrorKind::Network);
    }

    #[test]
    fn test_chain_iter() {
        let root = ChainedError::new(ErrorKind::Network, "dns failed");
        let mid = ChainedError::wrap(ErrorKind::Upstream, "backend down", root);
        let top = ChainedError::wrap(ErrorKind::Internal, "page load failed", mid);
        let messages: Vec<&str> = top.chain_iter().map(|e| e.message()).collect();
        assert_eq!(
            messages,
            vec!["page load failed", "backend down", "dns failed"]
        );
    }

    #[test]
    fn test_find_kind() {
        let root = ChainedError::new(ErrorKind::Timeout, "timed out");
        let top = ChainedError::wrap(ErrorKind::Internal, "handler failed", root);
        assert!(top.find_kind(ErrorKind::Timeout).is_some());
        assert!(top.find_kind(ErrorKind::Auth).is_none());
    }

    #[test]
    fn test_has_kind() {
        let root = ChainedError::new(ErrorKind::RateLimited, "429");
        let top = ChainedError::wrap(ErrorKind::Upstream, "call failed", root);
        assert!(top.has_kind(ErrorKind::RateLimited));
        assert!(!top.has_kind(ErrorKind::Validation));
    }

    #[test]
    fn test_kind_chain() {
        let root = ChainedError::new(ErrorKind::Network, "net err");
        let mid = ChainedError::wrap(ErrorKind::Upstream, "svc err", root);
        let top = ChainedError::wrap(ErrorKind::Internal, "top err", mid);
        assert_eq!(
            top.kind_chain(),
            vec![ErrorKind::Internal, ErrorKind::Upstream, ErrorKind::Network]
        );
    }

    #[test]
    fn test_display_chain() {
        let root = ChainedError::new(ErrorKind::Network, "socket closed");
        let top = ChainedError::wrap(ErrorKind::Internal, "request failed", root)
            .with_context("url", "/api/data");
        let display = top.display_chain();
        assert!(display.contains("error: request failed"));
        assert!(display.contains("caused by: socket closed"));
        assert!(display.contains("url = /api/data"));
    }

    #[test]
    fn test_display_oneline() {
        let root = ChainedError::new(ErrorKind::Network, "refused");
        let top = ChainedError::wrap(ErrorKind::Internal, "failed", root);
        assert_eq!(top.display_oneline(), "failed: refused");
    }

    #[test]
    fn test_display_trait() {
        let err = ChainedError::new(ErrorKind::Auth, "forbidden");
        let s = format!("{err}");
        assert!(s.contains("auth"));
        assert!(s.contains("forbidden"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let err = ChainedError::new(ErrorKind::Validation, "bad input")
            .with_context("field", "email")
            .with_source(ChainedError::new(ErrorKind::Internal, "parse error"));
        let json = err.to_json().unwrap();
        let restored = ChainedError::from_json(&json).unwrap();
        assert_eq!(restored.message(), "bad input");
        assert_eq!(restored.kind(), ErrorKind::Validation);
        assert_eq!(restored.context().get("field"), Some("email"));
        assert_eq!(restored.source().unwrap().message(), "parse error");
    }

    #[test]
    fn test_to_map() {
        let err = ChainedError::new(ErrorKind::NotFound, "page not found")
            .with_context("path", "/foo");
        let map = err.to_map();
        assert_eq!(map.get("message").unwrap(), "page not found");
        assert_eq!(map.get("kind").unwrap(), "not_found");
        assert_eq!(map.get("ctx.path").unwrap(), "/foo");
    }

    #[test]
    fn test_error_kind_retryable() {
        assert!(ErrorKind::Network.is_retryable());
        assert!(ErrorKind::Timeout.is_retryable());
        assert!(ErrorKind::RateLimited.is_retryable());
        assert!(ErrorKind::Upstream.is_retryable());
        assert!(!ErrorKind::Auth.is_retryable());
        assert!(!ErrorKind::Validation.is_retryable());
    }

    #[test]
    fn test_error_kind_client() {
        assert!(ErrorKind::Auth.is_client_error());
        assert!(ErrorKind::Validation.is_client_error());
        assert!(ErrorKind::NotFound.is_client_error());
        assert!(!ErrorKind::Network.is_client_error());
        assert!(!ErrorKind::Internal.is_client_error());
    }

    #[test]
    fn test_is_retryable_chain() {
        let root = ChainedError::new(ErrorKind::Timeout, "timed out");
        let top = ChainedError::wrap(ErrorKind::Internal, "internal", root);
        assert!(top.is_retryable()); // Timeout in chain makes it retryable.

        let non = ChainedError::new(ErrorKind::Auth, "denied");
        assert!(!non.is_retryable());
    }

    #[test]
    fn test_builder() {
        let err = ErrorBuilder::new(ErrorKind::Conflict, "duplicate key")
            .context("table", "users")
            .context("key", "email")
            .source(ChainedError::new(ErrorKind::Internal, "db error"))
            .build();
        assert_eq!(err.kind(), ErrorKind::Conflict);
        assert_eq!(err.context().get("table"), Some("users"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_convenience_constructors() {
        let ne = network_error("net fail");
        assert_eq!(ne.kind(), ErrorKind::Network);

        let ve = validation_error("bad data");
        assert_eq!(ve.kind(), ErrorKind::Validation);

        let nf = not_found_error("missing");
        assert_eq!(nf.kind(), ErrorKind::NotFound);

        let te = timeout_error("slow");
        assert_eq!(te.kind(), ErrorKind::Timeout);

        let ie = internal_error("oops");
        assert_eq!(ie.kind(), ErrorKind::Internal);
    }

    #[test]
    fn test_context_replace() {
        let mut ctx = ErrorContext::new();
        ctx.insert("k", "v1");
        ctx.insert("k", "v2");
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx.get("k"), Some("v2"));
    }

    #[test]
    fn test_context_remove() {
        let mut ctx = ErrorContext::new();
        ctx.insert("a", "1");
        ctx.insert("b", "2");
        assert_eq!(ctx.remove("a"), Some("1".to_string()));
        assert_eq!(ctx.len(), 1);
        assert!(ctx.remove("a").is_none());
    }

    #[test]
    fn test_context_merge() {
        let mut a = ErrorContext::new();
        a.insert("x", "1");
        a.insert("y", "2");
        let mut b = ErrorContext::new();
        b.insert("y", "99");
        b.insert("z", "3");
        a.merge(&b);
        assert_eq!(a.get("x"), Some("1"));
        assert_eq!(a.get("y"), Some("99")); // Overwritten.
        assert_eq!(a.get("z"), Some("3"));
    }

    #[test]
    fn test_add_context_in_place() {
        let mut err = ChainedError::new(ErrorKind::Internal, "oops");
        err.add_context("op", "save");
        assert_eq!(err.context().get("op"), Some("save"));
    }

    #[test]
    fn test_set_source_in_place() {
        let mut err = ChainedError::new(ErrorKind::Internal, "top");
        err.set_source(ChainedError::new(ErrorKind::Network, "root"));
        assert_eq!(err.chain_depth(), 2);
    }

    #[test]
    fn test_deep_chain() {
        let mut err = ChainedError::new(ErrorKind::Network, "layer_0");
        for i in 1..10 {
            err = ChainedError::wrap(ErrorKind::Internal, format!("layer_{i}"), err);
        }
        assert_eq!(err.chain_depth(), 10);
        assert_eq!(err.root_cause().message(), "layer_0");
    }

    #[test]
    fn test_error_kind_as_str() {
        assert_eq!(ErrorKind::Network.as_str(), "network");
        assert_eq!(ErrorKind::Unknown.as_str(), "unknown");
        assert_eq!(ErrorKind::Configuration.as_str(), "configuration");
        assert_eq!(ErrorKind::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn test_context_iter() {
        let mut ctx = ErrorContext::new();
        ctx.insert("a", "1");
        ctx.insert("b", "2");
        let pairs: Vec<(&str, &str)> = ctx.iter().collect();
        assert_eq!(pairs.len(), 2);
        // Insertion order is preserved in Vec.
        assert_eq!(pairs[0], ("a", "1"));
        assert_eq!(pairs[1], ("b", "2"));
    }
}
