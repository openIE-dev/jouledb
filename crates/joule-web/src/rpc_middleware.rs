//! RPC middleware — interceptor chain, logging, auth, retry, timeout, metrics.
//!
//! Pure-Rust RPC middleware framework. Supports an interceptor chain pattern
//! where each interceptor can modify the request/response or short-circuit
//! processing. Includes logging, auth, retry, timeout, metrics, and error
//! mapping interceptors with composable middleware stacks.

use std::collections::HashMap;
use std::fmt;

// ── RPC Types (self-contained) ───────────────────────────────

/// Standard RPC error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RpcCode {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    Unauthenticated = 16,
}

impl fmt::Display for RpcCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Ok => "OK",
            Self::Cancelled => "CANCELLED",
            Self::Unknown => "UNKNOWN",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::NotFound => "NOT_FOUND",
            Self::PermissionDenied => "PERMISSION_DENIED",
            Self::ResourceExhausted => "RESOURCE_EXHAUSTED",
            Self::Unimplemented => "UNIMPLEMENTED",
            Self::Internal => "INTERNAL",
            Self::Unavailable => "UNAVAILABLE",
            Self::Unauthenticated => "UNAUTHENTICATED",
        };
        f.write_str(s)
    }
}

/// RPC error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: RpcCode,
    pub message: String,
}

impl RpcError {
    pub fn new(code: RpcCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

/// RPC metadata (key-value pairs).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, String)>,
}

impl Metadata {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.push((key.into().to_ascii_lowercase(), value.into()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        let k = key.to_ascii_lowercase();
        self.entries.iter().find(|(ek, _)| *ek == k).map(|(_, v)| v.as_str())
    }

    pub fn remove(&mut self, key: &str) {
        let k = key.to_ascii_lowercase();
        self.entries.retain(|(ek, _)| *ek != k);
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let k = key.to_ascii_lowercase();
        self.entries.iter().any(|(ek, _)| *ek == k)
    }
}

/// RPC request.
#[derive(Debug, Clone)]
pub struct RpcRequest {
    pub path: String,
    pub body: Vec<u8>,
    pub metadata: Metadata,
    pub deadline_ms: u64,
}

impl RpcRequest {
    pub fn new(path: impl Into<String>, body: Vec<u8>) -> Self {
        Self { path: path.into(), body, metadata: Metadata::new(), deadline_ms: 0 }
    }
}

/// RPC response.
#[derive(Debug, Clone)]
pub struct RpcResponse {
    pub body: Vec<u8>,
    pub code: RpcCode,
    pub message: String,
    pub metadata: Metadata,
}

impl RpcResponse {
    pub fn ok(body: Vec<u8>) -> Self {
        Self { body, code: RpcCode::Ok, message: String::new(), metadata: Metadata::new() }
    }

    pub fn error(err: &RpcError) -> Self {
        Self { body: Vec::new(), code: err.code, message: err.message.clone(), metadata: Metadata::new() }
    }

    pub fn is_ok(&self) -> bool { matches!(self.code, RpcCode::Ok) }
}

// ── Interceptor Trait ────────────────────────────────────────

/// Handler type — takes a request and produces a response.
pub type HandlerFn = fn(&RpcRequest) -> Result<RpcResponse, RpcError>;

/// An interceptor that can wrap RPC calls.
pub trait Interceptor: fmt::Debug {
    /// Process a request. Call `next` to continue the chain, or return early.
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError>;

    /// Interceptor name for logging.
    fn name(&self) -> &str;
}

// ── Interceptor Chain ────────────────────────────────────────

/// A composed chain of interceptors with a terminal handler.
pub struct InterceptorChain {
    interceptors: Vec<Box<dyn Interceptor>>,
    handler: HandlerFn,
}

impl fmt::Debug for InterceptorChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.interceptors.iter().map(|i| i.name()).collect();
        f.debug_struct("InterceptorChain")
            .field("interceptors", &names)
            .finish()
    }
}

impl InterceptorChain {
    pub fn new(handler: HandlerFn) -> Self {
        Self { interceptors: Vec::new(), handler }
    }

    /// Add an interceptor to the front of the chain.
    pub fn add(&mut self, interceptor: Box<dyn Interceptor>) {
        self.interceptors.push(interceptor);
    }

    /// Number of interceptors.
    pub fn len(&self) -> usize {
        self.interceptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interceptors.is_empty()
    }

    /// Execute the chain.
    pub fn execute(&self, request: &mut RpcRequest) -> Result<RpcResponse, RpcError> {
        self.execute_from(0, request)
    }

    fn execute_from(&self, index: usize, request: &mut RpcRequest) -> Result<RpcResponse, RpcError> {
        if index >= self.interceptors.len() {
            return (self.handler)(request);
        }
        let interceptor = &self.interceptors[index];
        let next_index = index + 1;

        // Build the "next" closure that calls the rest of the chain.
        // We need to be careful about the borrow checker here.
        let next_fn = |req: &RpcRequest| -> Result<RpcResponse, RpcError> {
            // Since we cannot recurse with &mut through a Fn closure,
            // we clone the request and proceed. This is a simulation model.
            let mut req_clone = req.clone();
            // We use an iterative approach for the remaining interceptors.
            let remaining = &self.interceptors[next_index..];
            let handler = self.handler;
            execute_remaining(remaining, &mut req_clone, handler)
        };

        interceptor.intercept(request, &next_fn)
    }

    /// Names of all interceptors in order.
    pub fn interceptor_names(&self) -> Vec<&str> {
        self.interceptors.iter().map(|i| i.name()).collect()
    }
}

fn execute_remaining(
    interceptors: &[Box<dyn Interceptor>],
    request: &mut RpcRequest,
    handler: HandlerFn,
) -> Result<RpcResponse, RpcError> {
    if interceptors.is_empty() {
        return handler(request);
    }
    let (first, rest) = interceptors.split_first().unwrap();
    let rest_ref = rest;
    let next_fn = move |req: &RpcRequest| -> Result<RpcResponse, RpcError> {
        let mut req_clone = req.clone();
        execute_remaining(rest_ref, &mut req_clone, handler)
    };
    first.intercept(request, &next_fn)
}

// ── Logging Interceptor ──────────────────────────────────────

/// Logs RPC call path, duration (in calls), and result code.
#[derive(Debug)]
pub struct LoggingInterceptor {
    /// Collected log entries (path, code).
    log: std::cell::RefCell<Vec<LogEntry>>,
}

/// A log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    pub path: String,
    pub code: RpcCode,
    pub error_message: Option<String>,
}

impl LoggingInterceptor {
    pub fn new() -> Self {
        Self { log: std::cell::RefCell::new(Vec::new()) }
    }

    /// Get logged entries.
    pub fn entries(&self) -> Vec<LogEntry> {
        self.log.borrow().clone()
    }

    /// Number of logged calls.
    pub fn call_count(&self) -> usize {
        self.log.borrow().len()
    }
}

impl Default for LoggingInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for LoggingInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        let path = request.path.clone();
        match next(request) {
            Ok(resp) => {
                self.log.borrow_mut().push(LogEntry {
                    path,
                    code: resp.code,
                    error_message: None,
                });
                Ok(resp)
            }
            Err(err) => {
                self.log.borrow_mut().push(LogEntry {
                    path,
                    code: err.code,
                    error_message: Some(err.message.clone()),
                });
                Err(err)
            }
        }
    }

    fn name(&self) -> &str { "logging" }
}

// ── Auth Interceptor ─────────────────────────────────────────

/// Checks for an authorization metadata key and validates the token.
#[derive(Debug)]
pub struct AuthInterceptor {
    /// Valid tokens.
    valid_tokens: Vec<String>,
    /// Metadata key to check.
    header_key: String,
}

impl AuthInterceptor {
    pub fn new(valid_tokens: Vec<String>) -> Self {
        Self { valid_tokens, header_key: "authorization".to_string() }
    }

    /// Set a custom header key.
    pub fn with_header(mut self, key: impl Into<String>) -> Self {
        self.header_key = key.into();
        self
    }

    /// Add a valid token.
    pub fn add_token(&mut self, token: impl Into<String>) {
        self.valid_tokens.push(token.into());
    }
}

impl Interceptor for AuthInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        let token = request.metadata.get(&self.header_key);
        match token {
            None => Err(RpcError::new(RpcCode::Unauthenticated, "missing auth token")),
            Some(t) => {
                if self.valid_tokens.iter().any(|vt| vt == t) {
                    next(request)
                } else {
                    Err(RpcError::new(RpcCode::PermissionDenied, "invalid auth token"))
                }
            }
        }
    }

    fn name(&self) -> &str { "auth" }
}

// ── Retry Interceptor ────────────────────────────────────────

/// Retries failed calls for retryable error codes.
#[derive(Debug)]
pub struct RetryInterceptor {
    /// Maximum number of attempts (including initial).
    pub max_attempts: u32,
    /// Retryable codes.
    pub retryable_codes: Vec<RpcCode>,
    /// Total attempts made across all calls.
    attempts: std::cell::RefCell<u64>,
}

impl RetryInterceptor {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            retryable_codes: vec![RpcCode::Unavailable, RpcCode::ResourceExhausted],
            attempts: std::cell::RefCell::new(0),
        }
    }

    /// Set retryable codes.
    pub fn with_retryable_codes(mut self, codes: Vec<RpcCode>) -> Self {
        self.retryable_codes = codes;
        self
    }

    /// Total attempts made.
    pub fn total_attempts(&self) -> u64 {
        *self.attempts.borrow()
    }
}

impl Interceptor for RetryInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        let mut last_err = None;
        for _ in 0..self.max_attempts {
            *self.attempts.borrow_mut() += 1;
            match next(request) {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    if !self.retryable_codes.contains(&err.code) {
                        return Err(err);
                    }
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| RpcError::new(RpcCode::Internal, "retry exhausted")))
    }

    fn name(&self) -> &str { "retry" }
}

// ── Timeout Interceptor ──────────────────────────────────────

/// Enforces a deadline on RPC calls by checking the request deadline.
#[derive(Debug)]
pub struct TimeoutInterceptor {
    /// Default timeout in ms (applied if no deadline set).
    pub default_timeout_ms: u64,
    /// Simulated current time in ms for testing.
    pub current_time_ms: u64,
}

impl TimeoutInterceptor {
    pub fn new(default_timeout_ms: u64) -> Self {
        Self { default_timeout_ms, current_time_ms: 0 }
    }

    /// Set the simulated current time.
    pub fn with_current_time(mut self, ms: u64) -> Self {
        self.current_time_ms = ms;
        self
    }
}

impl Interceptor for TimeoutInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        // Apply default deadline if none set.
        if request.deadline_ms == 0 {
            request.deadline_ms = self.current_time_ms + self.default_timeout_ms;
        }

        // Check if deadline already exceeded.
        if self.current_time_ms >= request.deadline_ms {
            return Err(RpcError::new(RpcCode::DeadlineExceeded, "deadline exceeded before call"));
        }

        next(request)
    }

    fn name(&self) -> &str { "timeout" }
}

// ── Metrics Interceptor ──────────────────────────────────────

/// Collects RPC call metrics.
#[derive(Debug)]
pub struct MetricsInterceptor {
    metrics: std::cell::RefCell<MetricsData>,
}

#[derive(Debug, Clone, Default)]
pub struct MetricsData {
    /// Total calls.
    pub total_calls: u64,
    /// Successful calls.
    pub success_count: u64,
    /// Failed calls.
    pub error_count: u64,
    /// Calls by method path.
    pub calls_by_path: HashMap<String, u64>,
    /// Errors by code.
    pub errors_by_code: HashMap<u32, u64>,
}

impl MetricsInterceptor {
    pub fn new() -> Self {
        Self { metrics: std::cell::RefCell::new(MetricsData::default()) }
    }

    /// Get a snapshot of the metrics.
    pub fn snapshot(&self) -> MetricsData {
        self.metrics.borrow().clone()
    }

    /// Reset metrics.
    pub fn reset(&self) {
        *self.metrics.borrow_mut() = MetricsData::default();
    }
}

impl Default for MetricsInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for MetricsInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        let path = request.path.clone();
        let mut m = self.metrics.borrow_mut();
        m.total_calls += 1;
        *m.calls_by_path.entry(path).or_insert(0) += 1;
        drop(m);

        match next(request) {
            Ok(resp) => {
                self.metrics.borrow_mut().success_count += 1;
                Ok(resp)
            }
            Err(err) => {
                let mut m = self.metrics.borrow_mut();
                m.error_count += 1;
                *m.errors_by_code.entry(err.code as u32).or_insert(0) += 1;
                drop(m);
                Err(err)
            }
        }
    }

    fn name(&self) -> &str { "metrics" }
}

// ── Error Mapping Interceptor ────────────────────────────────

/// Maps error codes to different codes.
#[derive(Debug)]
pub struct ErrorMappingInterceptor {
    mappings: HashMap<u32, RpcCode>,
}

impl ErrorMappingInterceptor {
    pub fn new() -> Self {
        Self { mappings: HashMap::new() }
    }

    /// Add a mapping from one code to another.
    pub fn map_code(mut self, from: RpcCode, to: RpcCode) -> Self {
        self.mappings.insert(from as u32, to);
        self
    }
}

impl Default for ErrorMappingInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for ErrorMappingInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        match next(request) {
            Ok(resp) => Ok(resp),
            Err(mut err) => {
                if let Some(mapped) = self.mappings.get(&(err.code as u32)) {
                    err.code = *mapped;
                }
                Err(err)
            }
        }
    }

    fn name(&self) -> &str { "error_mapping" }
}

// ── Metadata Interceptor ─────────────────────────────────────

/// Injects metadata into every request.
#[derive(Debug)]
pub struct MetadataInterceptor {
    entries: Vec<(String, String)>,
}

impl MetadataInterceptor {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a metadata entry to inject.
    pub fn add(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.entries.push((key.into(), value.into()));
        self
    }
}

impl Default for MetadataInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for MetadataInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        for (k, v) in &self.entries {
            if !request.metadata.contains_key(k) {
                request.metadata.insert(k.clone(), v.clone());
            }
        }
        next(request)
    }

    fn name(&self) -> &str { "metadata" }
}

// ── Rate Limit Interceptor ───────────────────────────────────

/// Simple rate limiter that rejects calls beyond a threshold.
#[derive(Debug)]
pub struct RateLimitInterceptor {
    max_calls: u64,
    call_count: std::cell::RefCell<u64>,
}

impl RateLimitInterceptor {
    pub fn new(max_calls: u64) -> Self {
        Self { max_calls, call_count: std::cell::RefCell::new(0) }
    }

    /// Current call count.
    pub fn current_count(&self) -> u64 {
        *self.call_count.borrow()
    }

    /// Reset the counter.
    pub fn reset(&self) {
        *self.call_count.borrow_mut() = 0;
    }
}

impl Interceptor for RateLimitInterceptor {
    fn intercept(
        &self,
        request: &mut RpcRequest,
        next: &dyn Fn(&RpcRequest) -> Result<RpcResponse, RpcError>,
    ) -> Result<RpcResponse, RpcError> {
        let current = *self.call_count.borrow();
        if current >= self.max_calls {
            return Err(RpcError::new(RpcCode::ResourceExhausted, "rate limit exceeded"));
        }
        *self.call_count.borrow_mut() += 1;
        next(request)
    }

    fn name(&self) -> &str { "rate_limit" }
}

// ── Middleware Builder ────────────────────────────────────────

/// Fluent builder for constructing middleware chains.
pub struct MiddlewareBuilder {
    interceptors: Vec<Box<dyn Interceptor>>,
}

impl MiddlewareBuilder {
    pub fn new() -> Self {
        Self { interceptors: Vec::new() }
    }

    /// Add an interceptor.
    pub fn with(mut self, interceptor: Box<dyn Interceptor>) -> Self {
        self.interceptors.push(interceptor);
        self
    }

    /// Add logging.
    pub fn with_logging(self) -> Self {
        self.with(Box::new(LoggingInterceptor::new()))
    }

    /// Add auth.
    pub fn with_auth(self, tokens: Vec<String>) -> Self {
        self.with(Box::new(AuthInterceptor::new(tokens)))
    }

    /// Add retry.
    pub fn with_retry(self, max_attempts: u32) -> Self {
        self.with(Box::new(RetryInterceptor::new(max_attempts)))
    }

    /// Add timeout.
    pub fn with_timeout(self, default_ms: u64) -> Self {
        self.with(Box::new(TimeoutInterceptor::new(default_ms)))
    }

    /// Add metrics.
    pub fn with_metrics(self) -> Self {
        self.with(Box::new(MetricsInterceptor::new()))
    }

    /// Build the chain with the given terminal handler.
    pub fn build(self, handler: HandlerFn) -> InterceptorChain {
        let mut chain = InterceptorChain::new(handler);
        for i in self.interceptors {
            chain.add(i);
        }
        chain
    }

    /// Number of interceptors added.
    pub fn len(&self) -> usize {
        self.interceptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interceptors.is_empty()
    }
}

impl Default for MiddlewareBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_handler(req: &RpcRequest) -> Result<RpcResponse, RpcError> {
        Ok(RpcResponse::ok(req.body.clone()))
    }

    fn error_handler(_req: &RpcRequest) -> Result<RpcResponse, RpcError> {
        Err(RpcError::new(RpcCode::Internal, "handler error"))
    }

    fn unavailable_handler(_req: &RpcRequest) -> Result<RpcResponse, RpcError> {
        Err(RpcError::new(RpcCode::Unavailable, "service down"))
    }

    #[test]
    fn empty_chain_passes_through() {
        let chain = InterceptorChain::new(ok_handler);
        let mut req = RpcRequest::new("/svc/method", vec![1, 2, 3]);
        let resp = chain.execute(&mut req).unwrap();
        assert!(resp.is_ok());
        assert_eq!(resp.body, vec![1, 2, 3]);
    }

    #[test]
    fn logging_interceptor_records() {
        let logger = LoggingInterceptor::new();
        let mut req = RpcRequest::new("/svc/echo", vec![]);
        logger.intercept(&mut req, &|r| ok_handler(r)).unwrap();

        let entries = logger.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "/svc/echo");
        assert_eq!(entries[0].code, RpcCode::Ok);
    }

    #[test]
    fn logging_interceptor_records_errors() {
        let logger = LoggingInterceptor::new();
        let mut req = RpcRequest::new("/svc/fail", vec![]);
        let _ = logger.intercept(&mut req, &|r| error_handler(r));

        let entries = logger.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].code, RpcCode::Internal);
        assert!(entries[0].error_message.is_some());
    }

    #[test]
    fn auth_interceptor_no_token() {
        let auth = AuthInterceptor::new(vec!["valid".to_string()]);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = auth.intercept(&mut req, &|r| ok_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Unauthenticated);
    }

    #[test]
    fn auth_interceptor_invalid_token() {
        let auth = AuthInterceptor::new(vec!["valid".to_string()]);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.metadata.insert("authorization", "invalid");
        let err = auth.intercept(&mut req, &|r| ok_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::PermissionDenied);
    }

    #[test]
    fn auth_interceptor_valid_token() {
        let auth = AuthInterceptor::new(vec!["my-token".to_string()]);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.metadata.insert("authorization", "my-token");
        let resp = auth.intercept(&mut req, &|r| ok_handler(r)).unwrap();
        assert!(resp.is_ok());
    }

    #[test]
    fn auth_interceptor_custom_header() {
        let auth = AuthInterceptor::new(vec!["token123".to_string()])
            .with_header("x-api-key");
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.metadata.insert("x-api-key", "token123");
        let resp = auth.intercept(&mut req, &|r| ok_handler(r)).unwrap();
        assert!(resp.is_ok());
    }

    #[test]
    fn retry_interceptor_no_retry_on_success() {
        let retry = RetryInterceptor::new(3);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let resp = retry.intercept(&mut req, &|r| ok_handler(r)).unwrap();
        assert!(resp.is_ok());
        assert_eq!(retry.total_attempts(), 1);
    }

    #[test]
    fn retry_interceptor_retries_unavailable() {
        let retry = RetryInterceptor::new(3);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = retry.intercept(&mut req, &|r| unavailable_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Unavailable);
        assert_eq!(retry.total_attempts(), 3);
    }

    #[test]
    fn retry_interceptor_no_retry_on_non_retryable() {
        let retry = RetryInterceptor::new(3);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = retry.intercept(&mut req, &|r| error_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Internal);
        assert_eq!(retry.total_attempts(), 1); // No retry for Internal
    }

    #[test]
    fn timeout_interceptor_sets_default() {
        let timeout = TimeoutInterceptor::new(5000).with_current_time(1000);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let _ = timeout.intercept(&mut req, &|r| ok_handler(r));
        assert_eq!(req.deadline_ms, 6000);
    }

    #[test]
    fn timeout_interceptor_deadline_exceeded() {
        let timeout = TimeoutInterceptor::new(5000).with_current_time(10000);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.deadline_ms = 5000; // Already past
        let err = timeout.intercept(&mut req, &|r| ok_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::DeadlineExceeded);
    }

    #[test]
    fn metrics_interceptor_counts() {
        let metrics = MetricsInterceptor::new();
        let mut req = RpcRequest::new("/svc/a", vec![]);
        let _ = metrics.intercept(&mut req, &|r| ok_handler(r));
        let _ = metrics.intercept(&mut req, &|r| ok_handler(r));
        let mut req2 = RpcRequest::new("/svc/b", vec![]);
        let _ = metrics.intercept(&mut req2, &|r| error_handler(r));

        let snap = metrics.snapshot();
        assert_eq!(snap.total_calls, 3);
        assert_eq!(snap.success_count, 2);
        assert_eq!(snap.error_count, 1);
    }

    #[test]
    fn metrics_interceptor_reset() {
        let metrics = MetricsInterceptor::new();
        let mut req = RpcRequest::new("/svc/a", vec![]);
        let _ = metrics.intercept(&mut req, &|r| ok_handler(r));
        metrics.reset();
        let snap = metrics.snapshot();
        assert_eq!(snap.total_calls, 0);
    }

    #[test]
    fn error_mapping_interceptor() {
        let mapper = ErrorMappingInterceptor::new()
            .map_code(RpcCode::Internal, RpcCode::Unknown);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = mapper.intercept(&mut req, &|r| error_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Unknown);
    }

    #[test]
    fn error_mapping_no_match() {
        let mapper = ErrorMappingInterceptor::new()
            .map_code(RpcCode::NotFound, RpcCode::Unknown);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = mapper.intercept(&mut req, &|r| error_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Internal); // unchanged
    }

    #[test]
    fn metadata_interceptor_injects() {
        let mi = MetadataInterceptor::new()
            .add("x-request-id", "abc-123")
            .add("x-source", "test");
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let _ = mi.intercept(&mut req, &|r| {
            assert_eq!(r.metadata.get("x-request-id"), Some("abc-123"));
            assert_eq!(r.metadata.get("x-source"), Some("test"));
            ok_handler(r)
        });
    }

    #[test]
    fn metadata_interceptor_no_overwrite() {
        let mi = MetadataInterceptor::new().add("x-key", "default");
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.metadata.insert("x-key", "custom");
        let _ = mi.intercept(&mut req, &|r| {
            assert_eq!(r.metadata.get("x-key"), Some("custom"));
            ok_handler(r)
        });
    }

    #[test]
    fn rate_limit_interceptor() {
        let rl = RateLimitInterceptor::new(2);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        assert!(rl.intercept(&mut req, &|r| ok_handler(r)).is_ok());
        assert!(rl.intercept(&mut req, &|r| ok_handler(r)).is_ok());
        let err = rl.intercept(&mut req, &|r| ok_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::ResourceExhausted);
        assert_eq!(rl.current_count(), 2);
    }

    #[test]
    fn rate_limit_interceptor_reset() {
        let rl = RateLimitInterceptor::new(1);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let _ = rl.intercept(&mut req, &|r| ok_handler(r));
        rl.reset();
        let resp = rl.intercept(&mut req, &|r| ok_handler(r)).unwrap();
        assert!(resp.is_ok());
    }

    #[test]
    fn chain_with_multiple_interceptors() {
        let chain = MiddlewareBuilder::new()
            .with_logging()
            .with_metrics()
            .build(ok_handler);
        assert_eq!(chain.len(), 2);
        let names = chain.interceptor_names();
        assert_eq!(names, vec!["logging", "metrics"]);
    }

    #[test]
    fn chain_execute_with_interceptors() {
        let chain = MiddlewareBuilder::new()
            .with_logging()
            .build(ok_handler);
        let mut req = RpcRequest::new("/svc/echo", vec![42]);
        let resp = chain.execute(&mut req).unwrap();
        assert_eq!(resp.body, vec![42]);
    }

    #[test]
    fn middleware_builder_empty() {
        let builder = MiddlewareBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn rpc_error_display() {
        let err = RpcError::new(RpcCode::NotFound, "resource missing");
        assert_eq!(err.to_string(), "NOT_FOUND: resource missing");
    }

    #[test]
    fn interceptor_names() {
        assert_eq!(LoggingInterceptor::new().name(), "logging");
        assert_eq!(AuthInterceptor::new(vec![]).name(), "auth");
        assert_eq!(RetryInterceptor::new(1).name(), "retry");
        assert_eq!(TimeoutInterceptor::new(1000).name(), "timeout");
        assert_eq!(MetricsInterceptor::new().name(), "metrics");
        assert_eq!(ErrorMappingInterceptor::new().name(), "error_mapping");
        assert_eq!(MetadataInterceptor::new().name(), "metadata");
        assert_eq!(RateLimitInterceptor::new(1).name(), "rate_limit");
    }

    #[test]
    fn auth_add_token() {
        let mut auth = AuthInterceptor::new(vec![]);
        auth.add_token("new-token");
        let mut req = RpcRequest::new("/svc/m", vec![]);
        req.metadata.insert("authorization", "new-token");
        let resp = auth.intercept(&mut req, &|r| ok_handler(r)).unwrap();
        assert!(resp.is_ok());
    }

    #[test]
    fn retry_with_custom_codes() {
        let retry = RetryInterceptor::new(2)
            .with_retryable_codes(vec![RpcCode::Internal]);
        let mut req = RpcRequest::new("/svc/m", vec![]);
        let err = retry.intercept(&mut req, &|r| error_handler(r)).unwrap_err();
        assert_eq!(err.code, RpcCode::Internal);
        assert_eq!(retry.total_attempts(), 2);
    }

    #[test]
    fn middleware_builder_all() {
        let chain = MiddlewareBuilder::new()
            .with_logging()
            .with_auth(vec!["tok".to_string()])
            .with_retry(3)
            .with_timeout(5000)
            .with_metrics()
            .build(ok_handler);
        assert_eq!(chain.len(), 5);
    }
}
