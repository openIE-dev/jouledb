//! RPC framework — service definition, method registration, request/response.
//!
//! Pure-Rust RPC framework. Supports service definition with method registration,
//! request/response serialization, unary/server-streaming/client-streaming/bidi
//! concepts, RPC context with metadata and deadlines, standard error codes
//! (OK, Cancelled, InvalidArgument, etc.), and request routing.

use std::collections::HashMap;
use std::fmt;

// ── Error Codes ──────────────────────────────────────────────

/// Standard RPC error codes (modeled after gRPC status codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RpcCode {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    AlreadyExists = 6,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    FailedPrecondition = 9,
    Aborted = 10,
    OutOfRange = 11,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    DataLoss = 15,
    Unauthenticated = 16,
}

impl RpcCode {
    /// Numeric code.
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Parse from numeric code.
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::Cancelled,
            2 => Self::Unknown,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            _ => Self::Unknown,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Cancelled => "CANCELLED",
            Self::Unknown => "UNKNOWN",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::NotFound => "NOT_FOUND",
            Self::AlreadyExists => "ALREADY_EXISTS",
            Self::PermissionDenied => "PERMISSION_DENIED",
            Self::ResourceExhausted => "RESOURCE_EXHAUSTED",
            Self::FailedPrecondition => "FAILED_PRECONDITION",
            Self::Aborted => "ABORTED",
            Self::OutOfRange => "OUT_OF_RANGE",
            Self::Unimplemented => "UNIMPLEMENTED",
            Self::Internal => "INTERNAL",
            Self::Unavailable => "UNAVAILABLE",
            Self::DataLoss => "DATA_LOSS",
            Self::Unauthenticated => "UNAUTHENTICATED",
        }
    }

    /// Whether this code indicates success.
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Whether this code is retryable by default.
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Unavailable | Self::ResourceExhausted | Self::Aborted)
    }
}

impl fmt::Display for RpcCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── RPC Error ────────────────────────────────────────────────

/// An RPC error with code, message, and optional details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: RpcCode,
    pub message: String,
    pub details: Vec<ErrorDetail>,
}

/// Structured error detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorDetail {
    pub type_url: String,
    pub value: Vec<u8>,
}

impl RpcError {
    pub fn new(code: RpcCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: Vec::new(),
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(RpcCode::Cancelled, message)
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(RpcCode::InvalidArgument, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(RpcCode::NotFound, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(RpcCode::Internal, message)
    }

    pub fn unimplemented(message: impl Into<String>) -> Self {
        Self::new(RpcCode::Unimplemented, message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(RpcCode::Unavailable, message)
    }

    pub fn deadline_exceeded(message: impl Into<String>) -> Self {
        Self::new(RpcCode::DeadlineExceeded, message)
    }

    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(RpcCode::PermissionDenied, message)
    }

    pub fn unauthenticated(message: impl Into<String>) -> Self {
        Self::new(RpcCode::Unauthenticated, message)
    }

    /// Add an error detail.
    pub fn with_detail(mut self, type_url: impl Into<String>, value: Vec<u8>) -> Self {
        self.details.push(ErrorDetail { type_url: type_url.into(), value });
        self
    }

    /// Whether this error is retryable.
    pub fn is_retryable(&self) -> bool {
        self.code.is_retryable()
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

// ── Metadata ─────────────────────────────────────────────────

/// RPC metadata (key-value headers). Keys are case-insensitive ASCII.
/// Binary values use keys ending with "-bin".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    entries: Vec<(String, Vec<u8>)>,
}

impl Metadata {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Insert a text value.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into().to_ascii_lowercase();
        let value_bytes = value.into().into_bytes();
        self.entries.push((key, value_bytes));
    }

    /// Insert a binary value.
    pub fn insert_bin(&mut self, key: impl Into<String>, value: Vec<u8>) {
        let mut key = key.into().to_ascii_lowercase();
        if !key.ends_with("-bin") {
            key.push_str("-bin");
        }
        self.entries.push((key, value));
    }

    /// Get the first text value for a key.
    pub fn get(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_ascii_lowercase();
        for (k, v) in &self.entries {
            if *k == key_lower {
                return std::str::from_utf8(v).ok();
            }
        }
        None
    }

    /// Get the first binary value for a key.
    pub fn get_bin(&self, key: &str) -> Option<&[u8]> {
        let key_lower = key.to_ascii_lowercase();
        for (k, v) in &self.entries {
            if *k == key_lower {
                return Some(v);
            }
        }
        None
    }

    /// Get all values for a key (text).
    pub fn get_all(&self, key: &str) -> Vec<&str> {
        let key_lower = key.to_ascii_lowercase();
        self.entries.iter()
            .filter(|(k, _)| *k == key_lower)
            .filter_map(|(_, v)| std::str::from_utf8(v).ok())
            .collect()
    }

    /// Remove all entries with the given key.
    pub fn remove(&mut self, key: &str) {
        let key_lower = key.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key_lower);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All keys (sorted, deduplicated).
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.iter().map(|(k, _)| k.clone()).collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        let key_lower = key.to_ascii_lowercase();
        self.entries.iter().any(|(k, _)| *k == key_lower)
    }

    /// Merge another metadata set (appending).
    pub fn merge(&mut self, other: &Metadata) {
        for (k, v) in &other.entries {
            self.entries.push((k.clone(), v.clone()));
        }
    }
}

// ── Streaming Mode ───────────────────────────────────────────

/// RPC method streaming configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    Unary,
    ServerStreaming,
    ClientStreaming,
    BidiStreaming,
}

impl fmt::Display for StreamingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unary => f.write_str("unary"),
            Self::ServerStreaming => f.write_str("server_streaming"),
            Self::ClientStreaming => f.write_str("client_streaming"),
            Self::BidiStreaming => f.write_str("bidi_streaming"),
        }
    }
}

// ── Method Definition ────────────────────────────────────────

/// Handler function type — takes serialized request bytes and context,
/// returns serialized response bytes or an error.
pub type HandlerFn = fn(&[u8], &RpcContext) -> Result<Vec<u8>, RpcError>;

/// An RPC method definition.
#[derive(Clone)]
pub struct MethodDef {
    /// Method name.
    pub name: String,
    /// Full path (service/method).
    pub full_path: String,
    /// Request type name.
    pub request_type: String,
    /// Response type name.
    pub response_type: String,
    /// Streaming mode.
    pub streaming: StreamingMode,
    /// Handler function.
    handler: Option<HandlerFn>,
}

impl fmt::Debug for MethodDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MethodDef")
            .field("name", &self.name)
            .field("full_path", &self.full_path)
            .field("request_type", &self.request_type)
            .field("response_type", &self.response_type)
            .field("streaming", &self.streaming)
            .field("has_handler", &self.handler.is_some())
            .finish()
    }
}

impl MethodDef {
    pub fn new(
        name: impl Into<String>,
        request_type: impl Into<String>,
        response_type: impl Into<String>,
    ) -> Self {
        let name = name.into();
        Self {
            full_path: name.clone(),
            name,
            request_type: request_type.into(),
            response_type: response_type.into(),
            streaming: StreamingMode::Unary,
            handler: None,
        }
    }

    /// Set the streaming mode.
    pub fn with_streaming(mut self, mode: StreamingMode) -> Self {
        self.streaming = mode;
        self
    }

    /// Set the handler.
    pub fn with_handler(mut self, handler: HandlerFn) -> Self {
        self.handler = Some(handler);
        self
    }

    /// Invoke the handler.
    pub fn invoke(&self, request: &[u8], ctx: &RpcContext) -> Result<Vec<u8>, RpcError> {
        match self.handler {
            Some(h) => h(request, ctx),
            None => Err(RpcError::unimplemented(format!("method {} has no handler", self.name))),
        }
    }

    /// Whether a handler is registered.
    pub fn has_handler(&self) -> bool {
        self.handler.is_some()
    }
}

// ── RPC Context ──────────────────────────────────────────────

/// Context passed to RPC handlers.
#[derive(Debug, Clone)]
pub struct RpcContext {
    /// Request metadata.
    pub metadata: Metadata,
    /// Deadline in milliseconds since epoch (0 = no deadline).
    pub deadline_ms: u64,
    /// Peer address.
    pub peer: String,
    /// Authority (host).
    pub authority: String,
    /// Request-scoped values.
    pub values: HashMap<String, String>,
    /// Whether the call has been cancelled.
    pub cancelled: bool,
}

impl RpcContext {
    pub fn new() -> Self {
        Self {
            metadata: Metadata::new(),
            deadline_ms: 0,
            peer: String::new(),
            authority: String::new(),
            values: HashMap::new(),
            cancelled: false,
        }
    }

    /// Set a deadline.
    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = deadline_ms;
        self
    }

    /// Set peer address.
    pub fn with_peer(mut self, peer: impl Into<String>) -> Self {
        self.peer = peer.into();
        self
    }

    /// Set authority.
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authority = authority.into();
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set a request-scoped value.
    pub fn with_value(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    /// Get a request-scoped value.
    pub fn get_value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Check if the deadline has passed (given current_ms).
    pub fn is_deadline_exceeded(&self, current_ms: u64) -> bool {
        self.deadline_ms > 0 && current_ms >= self.deadline_ms
    }

    /// Remaining time until deadline in ms (None if no deadline).
    pub fn remaining_ms(&self, current_ms: u64) -> Option<u64> {
        if self.deadline_ms == 0 {
            return None;
        }
        if current_ms >= self.deadline_ms {
            Some(0)
        } else {
            Some(self.deadline_ms - current_ms)
        }
    }

    /// Cancel the call.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}

impl Default for RpcContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Service Definition ───────────────────────────────────────

/// An RPC service containing methods.
#[derive(Debug, Clone)]
pub struct ServiceDef {
    /// Service name.
    pub name: String,
    /// Fully-qualified package + service name.
    pub full_name: String,
    /// Methods, keyed by name.
    methods: Vec<MethodDef>,
}

impl ServiceDef {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            full_name: name.clone(),
            name,
            methods: Vec::new(),
        }
    }

    /// Set the package prefix.
    pub fn with_package(mut self, package: impl Into<String>) -> Self {
        let pkg = package.into();
        self.full_name = format!("{}.{}", pkg, self.name);
        self
    }

    /// Register a method.
    pub fn add_method(&mut self, mut method: MethodDef) {
        method.full_path = format!("/{}/{}", self.full_name, method.name);
        self.methods.push(method);
    }

    /// Find a method by name.
    pub fn method(&self, name: &str) -> Option<&MethodDef> {
        self.methods.iter().find(|m| m.name == name)
    }

    /// Find a method by full path.
    pub fn method_by_path(&self, path: &str) -> Option<&MethodDef> {
        self.methods.iter().find(|m| m.full_path == path)
    }

    /// All method names.
    pub fn method_names(&self) -> Vec<&str> {
        self.methods.iter().map(|m| m.name.as_str()).collect()
    }

    /// Number of methods.
    pub fn method_count(&self) -> usize {
        self.methods.len()
    }
}

// ── Service Router ───────────────────────────────────────────

/// Routes incoming RPC calls to the appropriate service and method.
#[derive(Debug, Default)]
pub struct ServiceRouter {
    services: Vec<ServiceDef>,
}

impl ServiceRouter {
    pub fn new() -> Self {
        Self { services: Vec::new() }
    }

    /// Register a service.
    pub fn register(&mut self, service: ServiceDef) {
        self.services.push(service);
    }

    /// Route a call by full path (e.g., "/package.Service/Method").
    pub fn route(&self, path: &str) -> Result<(&ServiceDef, &MethodDef), RpcError> {
        for svc in &self.services {
            if let Some(method) = svc.method_by_path(path) {
                return Ok((svc, method));
            }
        }
        Err(RpcError::unimplemented(format!("method not found: {path}")))
    }

    /// Handle a call: route + invoke handler.
    pub fn handle(&self, path: &str, request: &[u8], ctx: &RpcContext) -> Result<Vec<u8>, RpcError> {
        if ctx.cancelled {
            return Err(RpcError::cancelled("call cancelled"));
        }
        let (_, method) = self.route(path)?;
        method.invoke(request, ctx)
    }

    /// All registered service names.
    pub fn service_names(&self) -> Vec<&str> {
        self.services.iter().map(|s| s.name.as_str()).collect()
    }

    /// Total method count across all services.
    pub fn total_methods(&self) -> usize {
        self.services.iter().map(|s| s.method_count()).sum()
    }

    /// List all method paths.
    pub fn all_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        for svc in &self.services {
            for method in &svc.methods {
                paths.push(method.full_path.clone());
            }
        }
        paths
    }
}

// ── Request/Response ─────────────────────────────────────────

/// A serialized RPC request.
#[derive(Debug, Clone)]
pub struct RpcRequest {
    /// Full method path.
    pub path: String,
    /// Serialized request body.
    pub body: Vec<u8>,
    /// Metadata.
    pub metadata: Metadata,
    /// Deadline (ms since epoch, 0 = none).
    pub deadline_ms: u64,
}

impl RpcRequest {
    pub fn new(path: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            path: path.into(),
            body,
            metadata: Metadata::new(),
            deadline_ms: 0,
        }
    }

    /// Set metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set deadline.
    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = deadline_ms;
        self
    }

    /// Build an RpcContext from this request.
    pub fn to_context(&self) -> RpcContext {
        RpcContext {
            metadata: self.metadata.clone(),
            deadline_ms: self.deadline_ms,
            peer: String::new(),
            authority: String::new(),
            values: HashMap::new(),
            cancelled: false,
        }
    }
}

/// A serialized RPC response.
#[derive(Debug, Clone)]
pub struct RpcResponse {
    /// Serialized response body.
    pub body: Vec<u8>,
    /// Response metadata (trailing).
    pub metadata: Metadata,
    /// Status code.
    pub code: RpcCode,
    /// Status message.
    pub message: String,
}

impl RpcResponse {
    /// Successful response.
    pub fn ok(body: Vec<u8>) -> Self {
        Self {
            body,
            metadata: Metadata::new(),
            code: RpcCode::Ok,
            message: String::new(),
        }
    }

    /// Error response.
    pub fn error(err: &RpcError) -> Self {
        Self {
            body: Vec::new(),
            metadata: Metadata::new(),
            code: err.code,
            message: err.message.clone(),
        }
    }

    /// Whether this is a success response.
    pub fn is_ok(&self) -> bool {
        self.code.is_ok()
    }

    /// Add trailing metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_handler(req: &[u8], _ctx: &RpcContext) -> Result<Vec<u8>, RpcError> {
        Ok(req.to_vec())
    }

    fn error_handler(_req: &[u8], _ctx: &RpcContext) -> Result<Vec<u8>, RpcError> {
        Err(RpcError::internal("test error"))
    }

    #[test]
    fn rpc_code_roundtrip() {
        for code_val in 0..=16 {
            let code = RpcCode::from_code(code_val);
            assert_eq!(code.code(), code_val);
        }
    }

    #[test]
    fn rpc_code_unknown_maps() {
        assert_eq!(RpcCode::from_code(99), RpcCode::Unknown);
    }

    #[test]
    fn rpc_code_properties() {
        assert!(RpcCode::Ok.is_ok());
        assert!(!RpcCode::Internal.is_ok());
        assert!(RpcCode::Unavailable.is_retryable());
        assert!(!RpcCode::NotFound.is_retryable());
    }

    #[test]
    fn rpc_code_display() {
        assert_eq!(RpcCode::Ok.to_string(), "OK");
        assert_eq!(RpcCode::DeadlineExceeded.to_string(), "DEADLINE_EXCEEDED");
    }

    #[test]
    fn rpc_error_constructors() {
        let e = RpcError::not_found("resource");
        assert_eq!(e.code, RpcCode::NotFound);
        assert_eq!(e.message, "resource");
        assert!(e.details.is_empty());
    }

    #[test]
    fn rpc_error_with_detail() {
        let e = RpcError::internal("fail")
            .with_detail("type.googleapis.com/Error", vec![1, 2, 3]);
        assert_eq!(e.details.len(), 1);
        assert_eq!(e.details[0].type_url, "type.googleapis.com/Error");
    }

    #[test]
    fn rpc_error_display() {
        let e = RpcError::cancelled("user cancelled");
        assert_eq!(e.to_string(), "CANCELLED: user cancelled");
    }

    #[test]
    fn rpc_error_retryable() {
        assert!(RpcError::unavailable("down").is_retryable());
        assert!(!RpcError::not_found("nope").is_retryable());
    }

    #[test]
    fn metadata_basic() {
        let mut md = Metadata::new();
        md.insert("Authorization", "Bearer token");
        md.insert("x-request-id", "abc");
        assert_eq!(md.get("authorization"), Some("Bearer token"));
        assert_eq!(md.get("X-Request-Id"), Some("abc"));
        assert_eq!(md.len(), 2);
    }

    #[test]
    fn metadata_binary() {
        let mut md = Metadata::new();
        md.insert_bin("data-bin", vec![0xFF, 0x00]);
        assert_eq!(md.get_bin("data-bin"), Some([0xFF, 0x00].as_slice()));
    }

    #[test]
    fn metadata_multi_value() {
        let mut md = Metadata::new();
        md.insert("x-key", "v1");
        md.insert("x-key", "v2");
        let vals = md.get_all("x-key");
        assert_eq!(vals, vec!["v1", "v2"]);
    }

    #[test]
    fn metadata_remove() {
        let mut md = Metadata::new();
        md.insert("x-key", "val");
        md.remove("X-Key");
        assert!(md.is_empty());
    }

    #[test]
    fn metadata_keys() {
        let mut md = Metadata::new();
        md.insert("b-key", "1");
        md.insert("a-key", "2");
        let keys = md.keys();
        assert_eq!(keys, vec!["a-key", "b-key"]);
    }

    #[test]
    fn metadata_merge() {
        let mut md1 = Metadata::new();
        md1.insert("k1", "v1");
        let mut md2 = Metadata::new();
        md2.insert("k2", "v2");
        md1.merge(&md2);
        assert_eq!(md1.len(), 2);
        assert!(md1.contains_key("k2"));
    }

    #[test]
    fn method_def_basic() {
        let method = MethodDef::new("SayHello", "HelloRequest", "HelloReply")
            .with_handler(echo_handler);
        assert!(method.has_handler());
        assert_eq!(method.streaming, StreamingMode::Unary);
    }

    #[test]
    fn method_def_streaming() {
        let method = MethodDef::new("Chat", "Msg", "Msg")
            .with_streaming(StreamingMode::BidiStreaming);
        assert_eq!(method.streaming, StreamingMode::BidiStreaming);
    }

    #[test]
    fn method_invoke_success() {
        let method = MethodDef::new("Echo", "Req", "Resp").with_handler(echo_handler);
        let ctx = RpcContext::new();
        let result = method.invoke(b"hello", &ctx).unwrap();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn method_invoke_no_handler() {
        let method = MethodDef::new("Missing", "Req", "Resp");
        let ctx = RpcContext::new();
        let err = method.invoke(b"", &ctx).unwrap_err();
        assert_eq!(err.code, RpcCode::Unimplemented);
    }

    #[test]
    fn method_invoke_error() {
        let method = MethodDef::new("Fail", "Req", "Resp").with_handler(error_handler);
        let ctx = RpcContext::new();
        let err = method.invoke(b"", &ctx).unwrap_err();
        assert_eq!(err.code, RpcCode::Internal);
    }

    #[test]
    fn rpc_context_deadline() {
        let ctx = RpcContext::new().with_deadline(1000);
        assert!(!ctx.is_deadline_exceeded(500));
        assert!(ctx.is_deadline_exceeded(1000));
        assert!(ctx.is_deadline_exceeded(1500));
        assert_eq!(ctx.remaining_ms(500), Some(500));
        assert_eq!(ctx.remaining_ms(1000), Some(0));
    }

    #[test]
    fn rpc_context_no_deadline() {
        let ctx = RpcContext::new();
        assert!(!ctx.is_deadline_exceeded(u64::MAX));
        assert_eq!(ctx.remaining_ms(100), None);
    }

    #[test]
    fn rpc_context_values() {
        let ctx = RpcContext::new()
            .with_value("user_id", "123")
            .with_peer("127.0.0.1:8080");
        assert_eq!(ctx.get_value("user_id"), Some("123"));
        assert_eq!(ctx.peer, "127.0.0.1:8080");
    }

    #[test]
    fn rpc_context_cancel() {
        let mut ctx = RpcContext::new();
        assert!(!ctx.cancelled);
        ctx.cancel();
        assert!(ctx.cancelled);
    }

    #[test]
    fn service_def_basic() {
        let mut svc = ServiceDef::new("Greeter").with_package("example.v1");
        svc.add_method(MethodDef::new("SayHello", "HelloRequest", "HelloReply").with_handler(echo_handler));
        assert_eq!(svc.method_count(), 1);
        assert!(svc.method("SayHello").is_some());
        assert_eq!(svc.method("SayHello").unwrap().full_path, "/example.v1.Greeter/SayHello");
    }

    #[test]
    fn service_router_route() {
        let mut svc = ServiceDef::new("Greeter").with_package("test");
        svc.add_method(MethodDef::new("Hello", "Req", "Resp").with_handler(echo_handler));

        let mut router = ServiceRouter::new();
        router.register(svc);

        let (_, method) = router.route("/test.Greeter/Hello").unwrap();
        assert_eq!(method.name, "Hello");
    }

    #[test]
    fn service_router_not_found() {
        let router = ServiceRouter::new();
        let err = router.route("/missing/Method").unwrap_err();
        assert_eq!(err.code, RpcCode::Unimplemented);
    }

    #[test]
    fn service_router_handle() {
        let mut svc = ServiceDef::new("Echo");
        svc.add_method(MethodDef::new("Do", "Req", "Resp").with_handler(echo_handler));

        let mut router = ServiceRouter::new();
        router.register(svc);

        let ctx = RpcContext::new();
        let result = router.handle("/Echo/Do", b"data", &ctx).unwrap();
        assert_eq!(result, b"data");
    }

    #[test]
    fn service_router_cancelled() {
        let mut ctx = RpcContext::new();
        ctx.cancel();
        let router = ServiceRouter::new();
        let err = router.handle("/any/path", b"", &ctx).unwrap_err();
        assert_eq!(err.code, RpcCode::Cancelled);
    }

    #[test]
    fn service_router_all_paths() {
        let mut svc = ServiceDef::new("S");
        svc.add_method(MethodDef::new("A", "R", "R"));
        svc.add_method(MethodDef::new("B", "R", "R"));
        let mut router = ServiceRouter::new();
        router.register(svc);
        let paths = router.all_paths();
        assert_eq!(paths.len(), 2);
        assert_eq!(router.total_methods(), 2);
    }

    #[test]
    fn rpc_request_to_context() {
        let req = RpcRequest::new("/svc/method", vec![1, 2, 3])
            .with_metadata("auth", "token")
            .with_deadline(5000);
        let ctx = req.to_context();
        assert_eq!(ctx.metadata.get("auth"), Some("token"));
        assert_eq!(ctx.deadline_ms, 5000);
    }

    #[test]
    fn rpc_response_ok() {
        let resp = RpcResponse::ok(vec![1, 2, 3]);
        assert!(resp.is_ok());
        assert_eq!(resp.body, vec![1, 2, 3]);
    }

    #[test]
    fn rpc_response_error() {
        let err = RpcError::internal("bad");
        let resp = RpcResponse::error(&err);
        assert!(!resp.is_ok());
        assert_eq!(resp.code, RpcCode::Internal);
    }

    #[test]
    fn streaming_mode_display() {
        assert_eq!(StreamingMode::Unary.to_string(), "unary");
        assert_eq!(StreamingMode::BidiStreaming.to_string(), "bidi_streaming");
    }

    #[test]
    fn metadata_contains_key() {
        let mut md = Metadata::new();
        md.insert("x-key", "val");
        assert!(md.contains_key("X-Key"));
        assert!(!md.contains_key("missing"));
    }

    #[test]
    fn rpc_error_all_constructors() {
        let _e = RpcError::deadline_exceeded("timeout");
        let _e = RpcError::permission_denied("no access");
        let _e = RpcError::unauthenticated("login required");
        let _e = RpcError::invalid_argument("bad param");
    }

    #[test]
    fn metadata_insert_bin_auto_suffix() {
        let mut md = Metadata::new();
        md.insert_bin("my-data", vec![0xAB]);
        assert!(md.contains_key("my-data-bin"));
    }

    #[test]
    fn service_router_service_names() {
        let mut router = ServiceRouter::new();
        router.register(ServiceDef::new("A"));
        router.register(ServiceDef::new("B"));
        let names = router.service_names();
        assert_eq!(names.len(), 2);
    }
}
