//! HTTP middleware framework — middleware trait, chain composition,
//! request/response transformation, short-circuit (early response), logging
//! middleware, timing middleware, auth middleware, compression middleware.
//!
//! Replaces Express/Koa middleware, `connect`, and similar JS middleware
//! frameworks with a pure-Rust, composable middleware chain.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Middleware error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiddlewareError {
    /// Short-circuit: middleware wants to return early.
    ShortCircuit(Response),
    /// Authentication failure.
    Unauthorized(String),
    /// Forbidden.
    Forbidden(String),
    /// Generic middleware error.
    Internal(String),
    /// Middleware not found by name.
    NotFound(String),
    /// Chain already executed.
    AlreadyExecuted,
}

impl fmt::Display for MiddlewareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShortCircuit(resp) => write!(f, "short-circuit: status {}", resp.status),
            Self::Unauthorized(msg) => write!(f, "unauthorized: {msg}"),
            Self::Forbidden(msg) => write!(f, "forbidden: {msg}"),
            Self::Internal(msg) => write!(f, "middleware error: {msg}"),
            Self::NotFound(name) => write!(f, "middleware not found: {name}"),
            Self::AlreadyExecuted => write!(f, "chain already executed"),
        }
    }
}

impl std::error::Error for MiddlewareError {}

// ── Request / Response ───────────────────────────────────────────

/// HTTP request representation for middleware processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Headers (lowercased keys).
    pub headers: HashMap<String, String>,
    /// Body content.
    pub body: String,
    /// Query parameters.
    pub query: HashMap<String, String>,
    /// Metadata added by middleware.
    pub extensions: HashMap<String, String>,
}

impl Request {
    /// Create a new request.
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            path: path.to_string(),
            headers: HashMap::new(),
            body: String::new(),
            query: HashMap::new(),
            extensions: HashMap::new(),
        }
    }

    /// Set a header.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_lowercase(), value.to_string());
        self
    }

    /// Set the body.
    pub fn with_body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }

    /// Get a header value (case-insensitive).
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Get an extension value set by middleware.
    pub fn extension(&self, key: &str) -> Option<&str> {
        self.extensions.get(key).map(|s| s.as_str())
    }

    /// Content-Type header shortcut.
    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }
}

/// HTTP response representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Response body.
    pub body: String,
}

impl Response {
    /// Create a new response with status 200.
    pub fn ok(body: &str) -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: body.to_string(),
        }
    }

    /// Create a response with a given status.
    pub fn with_status(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: HashMap::new(),
            body: body.to_string(),
        }
    }

    /// Set a header on the response.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_lowercase(), value.to_string());
        self
    }

    /// Get a response header.
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Check if this is a success status.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

// ── Middleware trait ──────────────────────────────────────────────

/// Outcome of processing a request through middleware.
#[derive(Debug, Clone)]
pub enum MiddlewareOutcome {
    /// Continue to the next middleware (possibly with modified request).
    Continue(Request),
    /// Short-circuit: return this response immediately.
    ShortCircuit(Response),
}

/// Result of post-processing a response through middleware.
#[derive(Debug, Clone)]
pub struct PostProcessResult {
    /// The (possibly modified) response.
    pub response: Response,
    /// Log entries generated by this middleware.
    pub log_entries: Vec<String>,
}

/// A middleware unit that can inspect/modify requests and responses.
#[derive(Debug, Clone)]
pub struct MiddlewareUnit {
    /// Name for identification.
    pub name: String,
    /// The kind of middleware — determines behavior.
    pub kind: MiddlewareKind,
}

/// Built-in middleware behaviors.
#[derive(Debug, Clone)]
pub enum MiddlewareKind {
    /// Logging: records method + path.
    Logger,
    /// Timing: adds X-Response-Time header.
    Timing,
    /// Auth: checks for Authorization header with a required token.
    Auth { token: String },
    /// CORS: adds CORS headers.
    Cors { allowed_origins: Vec<String> },
    /// Compression indicator: adds Content-Encoding header.
    Compression,
    /// Request ID: adds X-Request-Id extension.
    RequestId,
    /// Custom header injection.
    CustomHeader { key: String, value: String },
    /// Security headers.
    Security,
}

impl MiddlewareUnit {
    /// Create a logger middleware.
    pub fn logger() -> Self {
        Self { name: "logger".to_string(), kind: MiddlewareKind::Logger }
    }

    /// Create a timing middleware.
    pub fn timing() -> Self {
        Self { name: "timing".to_string(), kind: MiddlewareKind::Timing }
    }

    /// Create an auth middleware.
    pub fn auth(token: &str) -> Self {
        Self {
            name: "auth".to_string(),
            kind: MiddlewareKind::Auth { token: token.to_string() },
        }
    }

    /// Create a CORS middleware.
    pub fn cors(origins: &[&str]) -> Self {
        Self {
            name: "cors".to_string(),
            kind: MiddlewareKind::Cors {
                allowed_origins: origins.iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    /// Create a compression middleware.
    pub fn compression() -> Self {
        Self { name: "compression".to_string(), kind: MiddlewareKind::Compression }
    }

    /// Create a request ID middleware.
    pub fn request_id() -> Self {
        Self { name: "request-id".to_string(), kind: MiddlewareKind::RequestId }
    }

    /// Create a custom header injection middleware.
    pub fn custom_header(key: &str, value: &str) -> Self {
        Self {
            name: format!("custom-header-{key}"),
            kind: MiddlewareKind::CustomHeader {
                key: key.to_string(),
                value: value.to_string(),
            },
        }
    }

    /// Create security headers middleware.
    pub fn security() -> Self {
        Self { name: "security".to_string(), kind: MiddlewareKind::Security }
    }

    /// Process a request (pre-handler).
    pub fn process_request(&self, mut request: Request) -> MiddlewareOutcome {
        match &self.kind {
            MiddlewareKind::Logger => {
                request.extensions.insert(
                    "log_entry".to_string(),
                    format!("{} {}", request.method, request.path),
                );
                MiddlewareOutcome::Continue(request)
            }
            MiddlewareKind::Timing => {
                // Record a "start" marker; real timing would use an actual clock.
                request.extensions.insert("timing_started".to_string(), "true".to_string());
                MiddlewareOutcome::Continue(request)
            }
            MiddlewareKind::Auth { token } => {
                match request.header("authorization") {
                    Some(auth_val) => {
                        let expected = format!("Bearer {token}");
                        if auth_val == expected {
                            request.extensions.insert(
                                "authenticated".to_string(),
                                "true".to_string(),
                            );
                            MiddlewareOutcome::Continue(request)
                        } else {
                            MiddlewareOutcome::ShortCircuit(
                                Response::with_status(401, "Invalid token"),
                            )
                        }
                    }
                    None => {
                        MiddlewareOutcome::ShortCircuit(
                            Response::with_status(401, "Authorization required"),
                        )
                    }
                }
            }
            MiddlewareKind::Cors { .. } => {
                // CORS preflight check
                if request.method == "OPTIONS" {
                    MiddlewareOutcome::ShortCircuit(Response::with_status(204, ""))
                } else {
                    MiddlewareOutcome::Continue(request)
                }
            }
            MiddlewareKind::Compression => {
                // Check if client accepts compression
                let accepts = request
                    .header("accept-encoding")
                    .unwrap_or("")
                    .contains("gzip");
                request.extensions.insert(
                    "accepts_gzip".to_string(),
                    accepts.to_string(),
                );
                MiddlewareOutcome::Continue(request)
            }
            MiddlewareKind::RequestId => {
                let id = format!("req-{:08x}", simple_hash(request.path.as_bytes()));
                request.extensions.insert("request_id".to_string(), id);
                MiddlewareOutcome::Continue(request)
            }
            MiddlewareKind::CustomHeader { .. } => {
                MiddlewareOutcome::Continue(request)
            }
            MiddlewareKind::Security => {
                MiddlewareOutcome::Continue(request)
            }
        }
    }

    /// Process a response (post-handler).
    pub fn process_response(
        &self,
        request: &Request,
        mut response: Response,
    ) -> PostProcessResult {
        let mut log_entries = Vec::new();

        match &self.kind {
            MiddlewareKind::Logger => {
                if let Some(entry) = request.extensions.get("log_entry") {
                    log_entries.push(format!("{entry} -> {}", response.status));
                }
            }
            MiddlewareKind::Timing => {
                if request.extensions.get("timing_started").is_some() {
                    // Simulated timing — in production you'd measure real elapsed.
                    response.headers.insert(
                        "x-response-time".to_string(),
                        "0ms".to_string(),
                    );
                }
            }
            MiddlewareKind::Auth { .. } => {}
            MiddlewareKind::Cors { allowed_origins } => {
                let origin = request.header("origin").unwrap_or("*");
                let allowed = if allowed_origins.contains(&"*".to_string())
                    || allowed_origins.contains(&origin.to_string())
                {
                    origin.to_string()
                } else if !allowed_origins.is_empty() {
                    allowed_origins[0].clone()
                } else {
                    "*".to_string()
                };
                response.headers.insert(
                    "access-control-allow-origin".to_string(),
                    allowed,
                );
                response.headers.insert(
                    "access-control-allow-methods".to_string(),
                    "GET, POST, PUT, DELETE, OPTIONS".to_string(),
                );
            }
            MiddlewareKind::Compression => {
                let accepts_gzip = request
                    .extensions
                    .get("accepts_gzip")
                    .map(|v| v == "true")
                    .unwrap_or(false);
                if accepts_gzip && response.body.len() > 128 {
                    response.headers.insert(
                        "content-encoding".to_string(),
                        "gzip".to_string(),
                    );
                }
            }
            MiddlewareKind::RequestId => {
                if let Some(id) = request.extensions.get("request_id") {
                    response.headers.insert(
                        "x-request-id".to_string(),
                        id.clone(),
                    );
                }
            }
            MiddlewareKind::CustomHeader { key, value } => {
                response.headers.insert(key.to_lowercase(), value.clone());
            }
            MiddlewareKind::Security => {
                response.headers.insert(
                    "x-content-type-options".to_string(),
                    "nosniff".to_string(),
                );
                response.headers.insert(
                    "x-frame-options".to_string(),
                    "DENY".to_string(),
                );
                response.headers.insert(
                    "x-xss-protection".to_string(),
                    "1; mode=block".to_string(),
                );
            }
        }

        PostProcessResult { response, log_entries }
    }
}

// ── Middleware Chain ──────────────────────────────────────────────

/// A composable chain of middleware units.
#[derive(Debug, Clone)]
pub struct MiddlewareChain {
    units: Vec<MiddlewareUnit>,
}

impl MiddlewareChain {
    /// Create a new empty chain.
    pub fn new() -> Self {
        Self { units: Vec::new() }
    }

    /// Add a middleware unit.
    pub fn add(&mut self, unit: MiddlewareUnit) {
        self.units.push(unit);
    }

    /// Builder pattern: add and return self.
    pub fn with(mut self, unit: MiddlewareUnit) -> Self {
        self.units.push(unit);
        self
    }

    /// Process a request through the chain, returning the (possibly modified)
    /// request, or a short-circuit response.
    pub fn process_request(
        &self,
        mut request: Request,
    ) -> Result<Request, Response> {
        for unit in &self.units {
            match unit.process_request(request) {
                MiddlewareOutcome::Continue(req) => {
                    request = req;
                }
                MiddlewareOutcome::ShortCircuit(resp) => {
                    return Err(resp);
                }
            }
        }
        Ok(request)
    }

    /// Process a response through the chain in reverse order.
    pub fn process_response(
        &self,
        request: &Request,
        mut response: Response,
    ) -> (Response, Vec<String>) {
        let mut all_logs = Vec::new();
        for unit in self.units.iter().rev() {
            let result = unit.process_response(request, response);
            response = result.response;
            all_logs.extend(result.log_entries);
        }
        (response, all_logs)
    }

    /// Run the full pipeline: process request, invoke handler, process response.
    /// The handler is a simple function that takes a request and returns a response.
    pub fn execute<F>(&self, request: Request, handler: F) -> (Response, Vec<String>)
    where
        F: FnOnce(&Request) -> Response,
    {
        match self.process_request(request) {
            Err(short_circuit) => (short_circuit, Vec::new()),
            Ok(req) => {
                let resp = handler(&req);
                self.process_response(&req, resp)
            }
        }
    }

    /// Number of middleware units.
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Names of all middleware in order.
    pub fn names(&self) -> Vec<&str> {
        self.units.iter().map(|u| u.name.as_str()).collect()
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple non-cryptographic hash for request ID generation.
fn simple_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for byte in data {
        h ^= *byte as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request() -> Request {
        Request::new("GET", "/api/users")
    }

    #[test]
    fn test_request_creation() {
        let req = make_request();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/api/users");
    }

    #[test]
    fn test_request_headers() {
        let req = make_request().with_header("Content-Type", "application/json");
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_response_ok() {
        let resp = Response::ok("hello");
        assert_eq!(resp.status, 200);
        assert!(resp.is_success());
        assert_eq!(resp.body, "hello");
    }

    #[test]
    fn test_response_with_status() {
        let resp = Response::with_status(404, "not found");
        assert_eq!(resp.status, 404);
        assert!(!resp.is_success());
    }

    #[test]
    fn test_response_headers() {
        let resp = Response::ok("ok").with_header("X-Custom", "value");
        assert_eq!(resp.header("x-custom"), Some("value"));
    }

    #[test]
    fn test_logger_middleware() {
        let logger = MiddlewareUnit::logger();
        let req = make_request();
        match logger.process_request(req) {
            MiddlewareOutcome::Continue(req) => {
                assert!(req.extensions.contains_key("log_entry"));
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn test_timing_middleware() {
        let timing = MiddlewareUnit::timing();
        let req = make_request();
        let outcome = timing.process_request(req);
        match outcome {
            MiddlewareOutcome::Continue(req) => {
                assert_eq!(req.extensions.get("timing_started").unwrap(), "true");
                let resp = Response::ok("ok");
                let result = timing.process_response(&req, resp);
                assert!(result.response.headers.contains_key("x-response-time"));
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn test_auth_middleware_no_header() {
        let auth = MiddlewareUnit::auth("secret123");
        let req = make_request();
        match auth.process_request(req) {
            MiddlewareOutcome::ShortCircuit(resp) => {
                assert_eq!(resp.status, 401);
            }
            _ => panic!("expected ShortCircuit"),
        }
    }

    #[test]
    fn test_auth_middleware_valid_token() {
        let auth = MiddlewareUnit::auth("secret123");
        let req = make_request().with_header("Authorization", "Bearer secret123");
        match auth.process_request(req) {
            MiddlewareOutcome::Continue(req) => {
                assert_eq!(req.extensions.get("authenticated").unwrap(), "true");
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn test_auth_middleware_invalid_token() {
        let auth = MiddlewareUnit::auth("secret123");
        let req = make_request().with_header("Authorization", "Bearer wrong");
        match auth.process_request(req) {
            MiddlewareOutcome::ShortCircuit(resp) => {
                assert_eq!(resp.status, 401);
            }
            _ => panic!("expected ShortCircuit"),
        }
    }

    #[test]
    fn test_cors_preflight() {
        let cors = MiddlewareUnit::cors(&["https://example.com"]);
        let req = Request::new("OPTIONS", "/api/data");
        match cors.process_request(req) {
            MiddlewareOutcome::ShortCircuit(resp) => {
                assert_eq!(resp.status, 204);
            }
            _ => panic!("expected ShortCircuit for OPTIONS"),
        }
    }

    #[test]
    fn test_cors_response_headers() {
        let cors = MiddlewareUnit::cors(&["https://example.com"]);
        let req = make_request().with_header("Origin", "https://example.com");
        let resp = Response::ok("data");
        let result = cors.process_response(&req, resp);
        assert_eq!(
            result.response.headers.get("access-control-allow-origin").unwrap(),
            "https://example.com"
        );
    }

    #[test]
    fn test_compression_middleware() {
        let comp = MiddlewareUnit::compression();
        let req = make_request().with_header("Accept-Encoding", "gzip, deflate");
        match comp.process_request(req) {
            MiddlewareOutcome::Continue(req) => {
                assert_eq!(req.extensions.get("accepts_gzip").unwrap(), "true");
                // Body larger than 128 bytes to trigger compression header
                let long_body = "x".repeat(256);
                let resp = Response::ok(&long_body);
                let result = comp.process_response(&req, resp);
                assert_eq!(
                    result.response.headers.get("content-encoding").unwrap(),
                    "gzip"
                );
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn test_request_id_middleware() {
        let rid = MiddlewareUnit::request_id();
        let req = make_request();
        match rid.process_request(req) {
            MiddlewareOutcome::Continue(req) => {
                assert!(req.extensions.contains_key("request_id"));
                let resp = Response::ok("ok");
                let result = rid.process_response(&req, resp);
                assert!(result.response.headers.contains_key("x-request-id"));
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn test_security_headers() {
        let sec = MiddlewareUnit::security();
        let req = make_request();
        let resp = Response::ok("ok");
        let result = sec.process_response(&req, resp);
        assert_eq!(
            result.response.headers.get("x-content-type-options").unwrap(),
            "nosniff"
        );
        assert_eq!(
            result.response.headers.get("x-frame-options").unwrap(),
            "DENY"
        );
    }

    #[test]
    fn test_chain_composition() {
        let chain = MiddlewareChain::new()
            .with(MiddlewareUnit::logger())
            .with(MiddlewareUnit::timing());

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.names(), vec!["logger", "timing"]);
    }

    #[test]
    fn test_chain_execute_success() {
        let chain = MiddlewareChain::new()
            .with(MiddlewareUnit::logger())
            .with(MiddlewareUnit::timing());

        let req = make_request();
        let (resp, _logs) = chain.execute(req, |_req| Response::ok("handled"));
        assert_eq!(resp.status, 200);
        assert!(resp.headers.contains_key("x-response-time"));
    }

    #[test]
    fn test_chain_short_circuit() {
        let chain = MiddlewareChain::new()
            .with(MiddlewareUnit::auth("secret"))
            .with(MiddlewareUnit::logger());

        let req = make_request(); // no auth header
        let (resp, _) = chain.execute(req, |_| Response::ok("should not reach"));
        assert_eq!(resp.status, 401);
    }

    #[test]
    fn test_chain_with_auth_pass() {
        let chain = MiddlewareChain::new()
            .with(MiddlewareUnit::auth("tok123"))
            .with(MiddlewareUnit::logger());

        let req = make_request().with_header("Authorization", "Bearer tok123");
        let (resp, _) = chain.execute(req, |_| Response::ok("authorized"));
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "authorized");
    }

    #[test]
    fn test_custom_header_middleware() {
        let mw = MiddlewareUnit::custom_header("X-Powered-By", "JouleWeb");
        let req = make_request();
        let resp = Response::ok("ok");
        let result = mw.process_response(&req, resp);
        assert_eq!(
            result.response.headers.get("x-powered-by").unwrap(),
            "JouleWeb"
        );
    }

    #[test]
    fn test_chain_empty() {
        let chain = MiddlewareChain::new();
        assert!(chain.is_empty());

        let req = make_request();
        let (resp, _) = chain.execute(req, |_| Response::ok("passthrough"));
        assert_eq!(resp.body, "passthrough");
    }

    #[test]
    fn test_request_content_type() {
        let req = make_request().with_header("Content-Type", "text/html");
        assert_eq!(req.content_type(), Some("text/html"));
    }

    #[test]
    fn test_error_display() {
        let err = MiddlewareError::Unauthorized("bad token".to_string());
        assert!(err.to_string().contains("bad token"));
    }
}
