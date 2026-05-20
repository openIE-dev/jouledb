//! RPC testing utilities — mock service, request capture, response stubs.
//!
//! Pure-Rust RPC testing framework. Supports mock services with response stubs,
//! request capture and verification, streaming test helpers, deadline testing,
//! error injection, and a fluent test client builder.

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

/// RPC metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

// ── Captured Request ─────────────────────────────────────────

/// A captured RPC request for verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedRequest {
    /// Method path.
    pub path: String,
    /// Request body.
    pub body: Vec<u8>,
    /// Metadata.
    pub metadata: Metadata,
    /// Deadline (ms since epoch, 0 = none).
    pub deadline_ms: u64,
    /// Timestamp of capture (sequential counter).
    pub sequence: u64,
}

impl CapturedRequest {
    /// Check if the body matches.
    pub fn body_eq(&self, expected: &[u8]) -> bool {
        self.body == expected
    }

    /// Check if metadata has a key.
    pub fn has_metadata(&self, key: &str) -> bool {
        self.metadata.get(key).is_some()
    }

    /// Check if metadata matches.
    pub fn metadata_eq(&self, key: &str, value: &str) -> bool {
        self.metadata.get(key) == Some(value)
    }

    /// Check if deadline is set.
    pub fn has_deadline(&self) -> bool {
        self.deadline_ms > 0
    }
}

// ── Response Stub ────────────────────────────────────────────

/// What a mock should respond with.
#[derive(Debug, Clone)]
pub enum StubResponse {
    /// Return a successful response.
    Ok(Vec<u8>),
    /// Return an error.
    Error(RpcCode, String),
    /// Return a sequence of responses (for streaming).
    Stream(Vec<Vec<u8>>),
    /// Simulate a deadline exceeded error.
    DeadlineExceeded,
    /// Return Ok after a specific number of calls (for retry testing).
    FailThenOk {
        fail_count: u32,
        fail_code: RpcCode,
        fail_message: String,
        ok_body: Vec<u8>,
    },
}

// ── Stub Rule ────────────────────────────────────────────────

/// A matching rule for stubbing responses.
#[derive(Debug, Clone)]
pub struct StubRule {
    /// Method path pattern (exact or "*" for any).
    pub path_pattern: String,
    /// Optional body matcher.
    pub body_matcher: Option<Vec<u8>>,
    /// Required metadata.
    pub required_metadata: HashMap<String, String>,
    /// Response to return.
    pub response: StubResponse,
    /// How many times this rule can fire (0 = unlimited).
    pub max_uses: u32,
    /// How many times this rule has fired.
    pub use_count: u32,
    /// Priority (lower = matched first).
    pub priority: u32,
}

impl StubRule {
    pub fn new(path: impl Into<String>, response: StubResponse) -> Self {
        Self {
            path_pattern: path.into(),
            body_matcher: None,
            required_metadata: HashMap::new(),
            response,
            max_uses: 0,
            use_count: 0,
            priority: 100,
        }
    }

    /// Require a specific body.
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body_matcher = Some(body);
        self
    }

    /// Require metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.required_metadata.insert(key.into(), value.into());
        self
    }

    /// Set max uses.
    pub fn times(mut self, n: u32) -> Self {
        self.max_uses = n;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    /// Whether this rule matches a request.
    fn matches(&self, path: &str, body: &[u8], metadata: &Metadata) -> bool {
        // Check path.
        if self.path_pattern != "*" && self.path_pattern != path {
            return false;
        }
        // Check body.
        if let Some(expected) = &self.body_matcher {
            if body != expected.as_slice() {
                return false;
            }
        }
        // Check metadata.
        for (k, v) in &self.required_metadata {
            if metadata.get(k) != Some(v.as_str()) {
                return false;
            }
        }
        // Check uses.
        if self.max_uses > 0 && self.use_count >= self.max_uses {
            return false;
        }
        true
    }

    /// Whether this rule is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.max_uses > 0 && self.use_count >= self.max_uses
    }

    /// Remaining uses (None if unlimited).
    pub fn remaining(&self) -> Option<u32> {
        if self.max_uses == 0 {
            None
        } else {
            Some(self.max_uses.saturating_sub(self.use_count))
        }
    }
}

// ── Mock Service ─────────────────────────────────────────────

/// A mock RPC service for testing.
#[derive(Debug)]
pub struct MockService {
    /// Service name.
    pub name: String,
    /// Stub rules.
    rules: Vec<StubRule>,
    /// Captured requests.
    captures: Vec<CapturedRequest>,
    /// Sequence counter.
    sequence: u64,
    /// Default response for unmatched calls.
    default_response: StubResponse,
}

impl MockService {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: Vec::new(),
            captures: Vec::new(),
            sequence: 0,
            default_response: StubResponse::Error(RpcCode::NotFound, "no stub matched".into()),
        }
    }

    /// Set the default response for unmatched calls.
    pub fn set_default(&mut self, response: StubResponse) {
        self.default_response = response;
    }

    /// Add a stub rule.
    pub fn add_rule(&mut self, rule: StubRule) {
        self.rules.push(rule);
        // Sort by priority.
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Stub a method to return Ok.
    pub fn stub_ok(&mut self, path: impl Into<String>, body: Vec<u8>) {
        self.add_rule(StubRule::new(path, StubResponse::Ok(body)));
    }

    /// Stub a method to return an error.
    pub fn stub_error(&mut self, path: impl Into<String>, code: RpcCode, message: impl Into<String>) {
        self.add_rule(StubRule::new(path, StubResponse::Error(code, message.into())));
    }

    /// Stub a streaming response.
    pub fn stub_stream(&mut self, path: impl Into<String>, responses: Vec<Vec<u8>>) {
        self.add_rule(StubRule::new(path, StubResponse::Stream(responses)));
    }

    /// Call the mock service.
    pub fn call(
        &mut self,
        path: &str,
        body: &[u8],
        metadata: &Metadata,
        deadline_ms: u64,
    ) -> Result<Vec<u8>, RpcError> {
        // Capture the request.
        self.sequence += 1;
        self.captures.push(CapturedRequest {
            path: path.to_string(),
            body: body.to_vec(),
            metadata: metadata.clone(),
            deadline_ms,
            sequence: self.sequence,
        });

        // Find matching rule.
        let mut matched_idx = None;
        for (i, rule) in self.rules.iter().enumerate() {
            if rule.matches(path, body, metadata) {
                matched_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = matched_idx {
            self.rules[idx].use_count += 1;
            let response = self.rules[idx].response.clone();
            Self::execute_response(&response, self.rules[idx].use_count)
        } else {
            Self::execute_response(&self.default_response, 1)
        }
    }

    fn execute_response(response: &StubResponse, call_count: u32) -> Result<Vec<u8>, RpcError> {
        match response {
            StubResponse::Ok(body) => Ok(body.clone()),
            StubResponse::Error(code, msg) => Err(RpcError::new(*code, msg.clone())),
            StubResponse::Stream(responses) => {
                // For a simple call, return the first response.
                Ok(responses.first().cloned().unwrap_or_default())
            }
            StubResponse::DeadlineExceeded => {
                Err(RpcError::new(RpcCode::DeadlineExceeded, "deadline exceeded"))
            }
            StubResponse::FailThenOk { fail_count, fail_code, fail_message, ok_body } => {
                if call_count <= *fail_count {
                    Err(RpcError::new(*fail_code, fail_message.clone()))
                } else {
                    Ok(ok_body.clone())
                }
            }
        }
    }

    /// Get all captured requests.
    pub fn captures(&self) -> &[CapturedRequest] {
        &self.captures
    }

    /// Get captures for a specific path.
    pub fn captures_for(&self, path: &str) -> Vec<&CapturedRequest> {
        self.captures.iter().filter(|c| c.path == path).collect()
    }

    /// Number of total calls.
    pub fn call_count(&self) -> usize {
        self.captures.len()
    }

    /// Number of calls to a specific path.
    pub fn call_count_for(&self, path: &str) -> usize {
        self.captures.iter().filter(|c| c.path == path).count()
    }

    /// Clear all captured requests.
    pub fn clear_captures(&mut self) {
        self.captures.clear();
        self.sequence = 0;
    }

    /// Clear all rules.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Reset everything.
    pub fn reset(&mut self) {
        self.clear_captures();
        self.clear_rules();
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Verify that a path was called at least N times.
    pub fn verify_called(&self, path: &str, min_times: usize) -> Result<(), String> {
        let count = self.call_count_for(path);
        if count >= min_times {
            Ok(())
        } else {
            Err(format!(
                "expected {path} to be called at least {min_times} times, got {count}"
            ))
        }
    }

    /// Verify that a path was called exactly N times.
    pub fn verify_called_exactly(&self, path: &str, times: usize) -> Result<(), String> {
        let count = self.call_count_for(path);
        if count == times {
            Ok(())
        } else {
            Err(format!(
                "expected {path} to be called exactly {times} times, got {count}"
            ))
        }
    }

    /// Verify that a path was never called.
    pub fn verify_not_called(&self, path: &str) -> Result<(), String> {
        self.verify_called_exactly(path, 0)
    }

    /// Verify call order: paths should appear in this order.
    pub fn verify_order(&self, paths: &[&str]) -> Result<(), String> {
        let mut path_iter = paths.iter();
        let mut expected = path_iter.next();

        for cap in &self.captures {
            if let Some(exp) = expected {
                if cap.path == **exp {
                    expected = path_iter.next();
                }
            }
        }

        if expected.is_some() {
            Err(format!("expected call order {:?} not satisfied", paths))
        } else {
            Ok(())
        }
    }
}

// ── Streaming Test Helper ────────────────────────────────────

/// Helper for testing streaming RPC scenarios.
#[derive(Debug)]
pub struct StreamCollector {
    items: Vec<Vec<u8>>,
    errors: Vec<RpcError>,
    completed: bool,
}

impl StreamCollector {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            errors: Vec::new(),
            completed: false,
        }
    }

    /// Push a message.
    pub fn push(&mut self, data: Vec<u8>) {
        self.items.push(data);
    }

    /// Push an error.
    pub fn push_error(&mut self, err: RpcError) {
        self.errors.push(err);
    }

    /// Mark as completed.
    pub fn complete(&mut self) {
        self.completed = true;
    }

    /// Number of items received.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Number of errors received.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Whether the stream completed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Get item at index.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.items.get(index).map(|v| v.as_slice())
    }

    /// Get all items.
    pub fn items(&self) -> &[Vec<u8>] {
        &self.items
    }

    /// Get all errors.
    pub fn errors(&self) -> &[RpcError] {
        &self.errors
    }

    /// Total bytes received.
    pub fn total_bytes(&self) -> usize {
        self.items.iter().map(|i| i.len()).sum()
    }
}

impl Default for StreamCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Error Injector ───────────────────────────────────────────

/// Injects errors based on configurable rules.
#[derive(Debug)]
pub struct ErrorInjector {
    /// Error to inject.
    error: RpcError,
    /// Probability (0-100).
    probability: u32,
    /// Counter for deterministic injection.
    call_count: u32,
    /// Inject every Nth call (0 = disabled).
    every_n: u32,
    /// Maximum injections (0 = unlimited).
    max_injections: u32,
    /// Current injection count.
    injection_count: u32,
}

impl ErrorInjector {
    pub fn new(code: RpcCode, message: impl Into<String>) -> Self {
        Self {
            error: RpcError::new(code, message),
            probability: 100,
            call_count: 0,
            every_n: 0,
            max_injections: 0,
            injection_count: 0,
        }
    }

    /// Set probability (0-100).
    pub fn with_probability(mut self, pct: u32) -> Self {
        self.probability = pct.min(100);
        self
    }

    /// Inject every Nth call.
    pub fn every(mut self, n: u32) -> Self {
        self.every_n = n;
        self
    }

    /// Limit total injections.
    pub fn max_injections(mut self, n: u32) -> Self {
        self.max_injections = n;
        self
    }

    /// Check if an error should be injected on this call.
    pub fn should_inject(&mut self) -> bool {
        self.call_count += 1;
        if self.max_injections > 0 && self.injection_count >= self.max_injections {
            return false;
        }
        if self.every_n > 0 {
            if self.call_count % self.every_n == 0 {
                self.injection_count += 1;
                return true;
            }
            return false;
        }
        // Use probability with a simple deterministic approach.
        if self.probability >= 100 {
            self.injection_count += 1;
            return true;
        }
        if self.probability == 0 {
            return false;
        }
        // Simple modular check for testability.
        let threshold = self.call_count * 100 / self.call_count.max(1);
        if threshold <= self.probability {
            self.injection_count += 1;
            true
        } else {
            false
        }
    }

    /// Get the error to inject.
    pub fn error(&self) -> &RpcError {
        &self.error
    }

    /// Total calls seen.
    pub fn total_calls(&self) -> u32 {
        self.call_count
    }

    /// Total injections.
    pub fn total_injections(&self) -> u32 {
        self.injection_count
    }

    /// Reset counters.
    pub fn reset(&mut self) {
        self.call_count = 0;
        self.injection_count = 0;
    }
}

// ── Test Client Builder ──────────────────────────────────────

/// Fluent builder for a test RPC client configuration.
#[derive(Debug)]
pub struct TestClientBuilder {
    /// Target service name.
    service: String,
    /// Default metadata to send.
    metadata: Metadata,
    /// Default deadline (ms).
    deadline_ms: u64,
    /// Error injector.
    error_injector: Option<ErrorInjector>,
    /// Request interceptor (captures).
    capture_requests: bool,
    /// Captured requests.
    captured: Vec<CapturedRequest>,
    /// Sequence counter.
    sequence: u64,
}

impl TestClientBuilder {
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            metadata: Metadata::new(),
            deadline_ms: 0,
            error_injector: None,
            capture_requests: false,
            captured: Vec::new(),
            sequence: 0,
        }
    }

    /// Add default metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set default deadline.
    pub fn with_deadline(mut self, ms: u64) -> Self {
        self.deadline_ms = ms;
        self
    }

    /// Enable request capture.
    pub fn capture(mut self) -> Self {
        self.capture_requests = true;
        self
    }

    /// Set an error injector.
    pub fn with_error_injector(mut self, injector: ErrorInjector) -> Self {
        self.error_injector = Some(injector);
        self
    }

    /// Build a test call and send to a mock.
    pub fn call(
        &mut self,
        mock: &mut MockService,
        path: &str,
        body: &[u8],
    ) -> Result<Vec<u8>, RpcError> {
        // Check error injector.
        if let Some(injector) = &mut self.error_injector {
            if injector.should_inject() {
                let err = injector.error().clone();
                return Err(err);
            }
        }

        // Build metadata.
        let metadata = self.metadata.clone();

        // Capture.
        if self.capture_requests {
            self.sequence += 1;
            self.captured.push(CapturedRequest {
                path: path.to_string(),
                body: body.to_vec(),
                metadata: metadata.clone(),
                deadline_ms: self.deadline_ms,
                sequence: self.sequence,
            });
        }

        mock.call(path, body, &metadata, self.deadline_ms)
    }

    /// Get captured requests.
    pub fn captures(&self) -> &[CapturedRequest] {
        &self.captured
    }

    /// Service name.
    pub fn service(&self) -> &str {
        &self.service
    }

    /// Clear captures.
    pub fn clear_captures(&mut self) {
        self.captured.clear();
        self.sequence = 0;
    }
}

// ── Deadline Test Helper ─────────────────────────────────────

/// Helper for testing deadline behavior.
#[derive(Debug)]
pub struct DeadlineTest {
    /// Simulated time (ms).
    current_time_ms: u64,
}

impl DeadlineTest {
    pub fn new(start_ms: u64) -> Self {
        Self { current_time_ms: start_ms }
    }

    /// Advance time.
    pub fn advance(&mut self, ms: u64) {
        self.current_time_ms += ms;
    }

    /// Current time.
    pub fn now(&self) -> u64 {
        self.current_time_ms
    }

    /// Create a deadline from now + duration.
    pub fn deadline_from_now(&self, duration_ms: u64) -> u64 {
        self.current_time_ms + duration_ms
    }

    /// Check if a deadline has expired.
    pub fn is_expired(&self, deadline_ms: u64) -> bool {
        deadline_ms > 0 && self.current_time_ms >= deadline_ms
    }

    /// Remaining time until deadline.
    pub fn remaining(&self, deadline_ms: u64) -> u64 {
        if deadline_ms == 0 || self.current_time_ms >= deadline_ms {
            0
        } else {
            deadline_ms - self.current_time_ms
        }
    }
}

// ── Call Verifier ────────────────────────────────────────────

/// Fluent assertion builder for verifying captured calls.
pub struct CallVerifier<'a> {
    captures: &'a [CapturedRequest],
    path_filter: Option<String>,
}

impl<'a> CallVerifier<'a> {
    pub fn new(captures: &'a [CapturedRequest]) -> Self {
        Self { captures, path_filter: None }
    }

    /// Filter to a specific path.
    pub fn for_path(mut self, path: &str) -> Self {
        self.path_filter = Some(path.to_string());
        self
    }

    /// Filtered captures.
    fn filtered(&self) -> Vec<&CapturedRequest> {
        match &self.path_filter {
            Some(p) => self.captures.iter().filter(|c| c.path == *p).collect(),
            None => self.captures.iter().collect(),
        }
    }

    /// Assert call count.
    pub fn assert_count(&self, expected: usize) -> Result<(), String> {
        let actual = self.filtered().len();
        if actual == expected {
            Ok(())
        } else {
            Err(format!("expected {expected} calls, got {actual}"))
        }
    }

    /// Assert at least N calls.
    pub fn assert_at_least(&self, min: usize) -> Result<(), String> {
        let actual = self.filtered().len();
        if actual >= min {
            Ok(())
        } else {
            Err(format!("expected at least {min} calls, got {actual}"))
        }
    }

    /// Assert all calls have a specific metadata key.
    pub fn assert_all_have_metadata(&self, key: &str) -> Result<(), String> {
        for cap in self.filtered() {
            if !cap.has_metadata(key) {
                return Err(format!(
                    "call to {} (seq {}) missing metadata '{key}'",
                    cap.path, cap.sequence
                ));
            }
        }
        Ok(())
    }

    /// Assert all calls have a deadline.
    pub fn assert_all_have_deadline(&self) -> Result<(), String> {
        for cap in self.filtered() {
            if !cap.has_deadline() {
                return Err(format!(
                    "call to {} (seq {}) has no deadline",
                    cap.path, cap.sequence
                ));
            }
        }
        Ok(())
    }

    /// Assert no calls were made.
    pub fn assert_none(&self) -> Result<(), String> {
        self.assert_count(0)
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_service_stub_ok() {
        let mut mock = MockService::new("greeter");
        mock.stub_ok("/greeter/Hello", b"world".to_vec());
        let md = Metadata::new();
        let result = mock.call("/greeter/Hello", b"", &md, 0).unwrap();
        assert_eq!(result, b"world");
    }

    #[test]
    fn mock_service_stub_error() {
        let mut mock = MockService::new("greeter");
        mock.stub_error("/greeter/Fail", RpcCode::Internal, "oops");
        let md = Metadata::new();
        let err = mock.call("/greeter/Fail", b"", &md, 0).unwrap_err();
        assert_eq!(err.code, RpcCode::Internal);
        assert_eq!(err.message, "oops");
    }

    #[test]
    fn mock_service_default_not_found() {
        let mut mock = MockService::new("greeter");
        let md = Metadata::new();
        let err = mock.call("/unknown", b"", &md, 0).unwrap_err();
        assert_eq!(err.code, RpcCode::NotFound);
    }

    #[test]
    fn mock_service_captures() {
        let mut mock = MockService::new("greeter");
        mock.stub_ok("/greeter/Hello", vec![]);
        let md = Metadata::new();
        mock.call("/greeter/Hello", b"req1", &md, 0).unwrap();
        mock.call("/greeter/Hello", b"req2", &md, 0).unwrap();

        assert_eq!(mock.call_count(), 2);
        assert_eq!(mock.call_count_for("/greeter/Hello"), 2);
        assert!(mock.captures()[0].body_eq(b"req1"));
        assert!(mock.captures()[1].body_eq(b"req2"));
    }

    #[test]
    fn mock_service_verify_called() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();

        assert!(mock.verify_called("/svc/A", 1).is_ok());
        assert!(mock.verify_called("/svc/A", 2).is_err());
        assert!(mock.verify_not_called("/svc/B").is_ok());
    }

    #[test]
    fn mock_service_verify_exactly() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        mock.call("/svc/A", b"", &md, 0).unwrap();

        assert!(mock.verify_called_exactly("/svc/A", 2).is_ok());
        assert!(mock.verify_called_exactly("/svc/A", 1).is_err());
    }

    #[test]
    fn mock_service_verify_order() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("*", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        mock.call("/svc/B", b"", &md, 0).unwrap();
        mock.call("/svc/C", b"", &md, 0).unwrap();

        assert!(mock.verify_order(&["/svc/A", "/svc/B", "/svc/C"]).is_ok());
        assert!(mock.verify_order(&["/svc/A", "/svc/C"]).is_ok());
        assert!(mock.verify_order(&["/svc/C", "/svc/A"]).is_err());
    }

    #[test]
    fn mock_service_metadata_matching() {
        let mut mock = MockService::new("svc");
        mock.add_rule(
            StubRule::new("/svc/A", StubResponse::Ok(b"matched".to_vec()))
                .with_metadata("x-key", "special"),
        );
        mock.stub_ok("/svc/A", b"default".to_vec());

        let mut md = Metadata::new();
        md.insert("x-key", "special");
        let result = mock.call("/svc/A", b"", &md, 0).unwrap();
        assert_eq!(result, b"matched");

        let md2 = Metadata::new();
        let result2 = mock.call("/svc/A", b"", &md2, 0).unwrap();
        assert_eq!(result2, b"default");
    }

    #[test]
    fn stub_rule_max_uses() {
        let mut mock = MockService::new("svc");
        mock.add_rule(
            StubRule::new("/svc/A", StubResponse::Ok(b"limited".to_vec())).times(2),
        );
        mock.stub_ok("/svc/A", b"fallback".to_vec());

        let md = Metadata::new();
        assert_eq!(mock.call("/svc/A", b"", &md, 0).unwrap(), b"limited");
        assert_eq!(mock.call("/svc/A", b"", &md, 0).unwrap(), b"limited");
        assert_eq!(mock.call("/svc/A", b"", &md, 0).unwrap(), b"fallback");
    }

    #[test]
    fn stub_rule_exhaustion() {
        let mut rule = StubRule::new("/svc/A", StubResponse::Ok(vec![])).times(1);
        assert!(!rule.is_exhausted());
        assert_eq!(rule.remaining(), Some(1));
        rule.use_count = 1;
        assert!(rule.is_exhausted());
        assert_eq!(rule.remaining(), Some(0));
    }

    #[test]
    fn fail_then_ok_stub() {
        let mut mock = MockService::new("svc");
        mock.add_rule(StubRule::new("/svc/retry", StubResponse::FailThenOk {
            fail_count: 2,
            fail_code: RpcCode::Unavailable,
            fail_message: "down".to_string(),
            ok_body: b"recovered".to_vec(),
        }));
        let md = Metadata::new();
        assert!(mock.call("/svc/retry", b"", &md, 0).is_err());
        assert!(mock.call("/svc/retry", b"", &md, 0).is_err());
        assert_eq!(mock.call("/svc/retry", b"", &md, 0).unwrap(), b"recovered");
    }

    #[test]
    fn stream_collector() {
        let mut sc = StreamCollector::new();
        sc.push(b"msg1".to_vec());
        sc.push(b"msg2".to_vec());
        sc.push_error(RpcError::new(RpcCode::Internal, "oops"));
        sc.complete();

        assert_eq!(sc.item_count(), 2);
        assert_eq!(sc.error_count(), 1);
        assert!(sc.is_completed());
        assert_eq!(sc.get(0), Some(b"msg1".as_slice()));
        assert_eq!(sc.total_bytes(), 8);
    }

    #[test]
    fn stream_collector_empty() {
        let sc = StreamCollector::new();
        assert_eq!(sc.item_count(), 0);
        assert!(!sc.is_completed());
        assert_eq!(sc.total_bytes(), 0);
    }

    #[test]
    fn error_injector_always() {
        let mut inj = ErrorInjector::new(RpcCode::Internal, "injected");
        assert!(inj.should_inject());
        assert!(inj.should_inject());
        assert_eq!(inj.total_calls(), 2);
        assert_eq!(inj.total_injections(), 2);
    }

    #[test]
    fn error_injector_every_n() {
        let mut inj = ErrorInjector::new(RpcCode::Unavailable, "down").every(3);
        assert!(!inj.should_inject()); // call 1
        assert!(!inj.should_inject()); // call 2
        assert!(inj.should_inject());  // call 3
        assert!(!inj.should_inject()); // call 4
        assert!(!inj.should_inject()); // call 5
        assert!(inj.should_inject());  // call 6
    }

    #[test]
    fn error_injector_max_injections() {
        let mut inj = ErrorInjector::new(RpcCode::Internal, "err")
            .every(1)
            .max_injections(2);
        assert!(inj.should_inject());
        assert!(inj.should_inject());
        assert!(!inj.should_inject()); // max reached
    }

    #[test]
    fn error_injector_reset() {
        let mut inj = ErrorInjector::new(RpcCode::Internal, "err");
        inj.should_inject();
        inj.reset();
        assert_eq!(inj.total_calls(), 0);
        assert_eq!(inj.total_injections(), 0);
    }

    #[test]
    fn deadline_test_helper() {
        let mut dt = DeadlineTest::new(1000);
        let deadline = dt.deadline_from_now(5000);
        assert_eq!(deadline, 6000);
        assert!(!dt.is_expired(deadline));
        assert_eq!(dt.remaining(deadline), 5000);

        dt.advance(5000);
        assert!(dt.is_expired(deadline));
        assert_eq!(dt.remaining(deadline), 0);
    }

    #[test]
    fn deadline_test_no_deadline() {
        let dt = DeadlineTest::new(1000);
        assert!(!dt.is_expired(0));
        assert_eq!(dt.remaining(0), 0);
    }

    #[test]
    fn test_client_builder() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/Echo", b"echo".to_vec());

        let mut client = TestClientBuilder::new("svc")
            .with_metadata("auth", "token")
            .with_deadline(5000)
            .capture();

        let result = client.call(&mut mock, "/svc/Echo", b"hi").unwrap();
        assert_eq!(result, b"echo");
        assert_eq!(client.captures().len(), 1);
        assert!(client.captures()[0].metadata_eq("auth", "token"));
        assert_eq!(client.service(), "svc");
    }

    #[test]
    fn test_client_error_injection() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);

        let injector = ErrorInjector::new(RpcCode::Unavailable, "injected");
        let mut client = TestClientBuilder::new("svc")
            .with_error_injector(injector);

        let err = client.call(&mut mock, "/svc/A", b"").unwrap_err();
        assert_eq!(err.code, RpcCode::Unavailable);
    }

    #[test]
    fn call_verifier_count() {
        let caps = vec![
            CapturedRequest {
                path: "/svc/A".to_string(),
                body: vec![],
                metadata: Metadata::new(),
                deadline_ms: 0,
                sequence: 1,
            },
            CapturedRequest {
                path: "/svc/B".to_string(),
                body: vec![],
                metadata: Metadata::new(),
                deadline_ms: 0,
                sequence: 2,
            },
        ];

        let v = CallVerifier::new(&caps);
        assert!(v.assert_count(2).is_ok());

        let v2 = CallVerifier::new(&caps).for_path("/svc/A");
        assert!(v2.assert_count(1).is_ok());
    }

    #[test]
    fn call_verifier_metadata() {
        let mut md = Metadata::new();
        md.insert("auth", "tok");
        let caps = vec![CapturedRequest {
            path: "/svc/A".to_string(),
            body: vec![],
            metadata: md,
            deadline_ms: 1000,
            sequence: 1,
        }];
        let v = CallVerifier::new(&caps);
        assert!(v.assert_all_have_metadata("auth").is_ok());
        assert!(v.assert_all_have_metadata("missing").is_err());
        assert!(v.assert_all_have_deadline().is_ok());
    }

    #[test]
    fn call_verifier_none() {
        let caps: Vec<CapturedRequest> = vec![];
        let v = CallVerifier::new(&caps);
        assert!(v.assert_none().is_ok());
    }

    #[test]
    fn mock_service_reset() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        mock.reset();
        assert_eq!(mock.call_count(), 0);
        assert_eq!(mock.rule_count(), 0);
    }

    #[test]
    fn mock_service_clear_captures() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        mock.clear_captures();
        assert_eq!(mock.call_count(), 0);
        assert_eq!(mock.rule_count(), 1);
    }

    #[test]
    fn mock_service_set_default() {
        let mut mock = MockService::new("svc");
        mock.set_default(StubResponse::Ok(b"default".to_vec()));
        let md = Metadata::new();
        let result = mock.call("/anything", b"", &md, 0).unwrap();
        assert_eq!(result, b"default");
    }

    #[test]
    fn captured_request_helpers() {
        let mut md = Metadata::new();
        md.insert("key", "val");
        let cap = CapturedRequest {
            path: "/svc/A".to_string(),
            body: b"test".to_vec(),
            metadata: md,
            deadline_ms: 5000,
            sequence: 1,
        };
        assert!(cap.body_eq(b"test"));
        assert!(!cap.body_eq(b"other"));
        assert!(cap.has_metadata("key"));
        assert!(cap.metadata_eq("key", "val"));
        assert!(cap.has_deadline());
    }

    #[test]
    fn stub_body_matcher() {
        let mut mock = MockService::new("svc");
        mock.add_rule(
            StubRule::new("/svc/A", StubResponse::Ok(b"body_matched".to_vec()))
                .with_body(b"specific".to_vec()),
        );
        mock.stub_ok("/svc/A", b"generic".to_vec());

        let md = Metadata::new();
        assert_eq!(mock.call("/svc/A", b"specific", &md, 0).unwrap(), b"body_matched");
        assert_eq!(mock.call("/svc/A", b"other", &md, 0).unwrap(), b"generic");
    }

    #[test]
    fn stream_stub_returns_first() {
        let mut mock = MockService::new("svc");
        mock.stub_stream("/svc/S", vec![b"a".to_vec(), b"b".to_vec()]);
        let md = Metadata::new();
        let result = mock.call("/svc/S", b"", &md, 0).unwrap();
        assert_eq!(result, b"a");
    }

    #[test]
    fn deadline_exceeded_stub() {
        let mut mock = MockService::new("svc");
        mock.add_rule(StubRule::new("/svc/slow", StubResponse::DeadlineExceeded));
        let md = Metadata::new();
        let err = mock.call("/svc/slow", b"", &md, 0).unwrap_err();
        assert_eq!(err.code, RpcCode::DeadlineExceeded);
    }

    #[test]
    fn test_client_clear_captures() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("/svc/A", vec![]);
        let mut client = TestClientBuilder::new("svc").capture();
        client.call(&mut mock, "/svc/A", b"").unwrap();
        assert_eq!(client.captures().len(), 1);
        client.clear_captures();
        assert_eq!(client.captures().len(), 0);
    }

    #[test]
    fn wildcard_stub() {
        let mut mock = MockService::new("svc");
        mock.add_rule(StubRule::new("*", StubResponse::Ok(b"any".to_vec())));
        let md = Metadata::new();
        assert_eq!(mock.call("/any/path", b"", &md, 0).unwrap(), b"any");
        assert_eq!(mock.call("/other", b"", &md, 0).unwrap(), b"any");
    }

    #[test]
    fn captures_for_path() {
        let mut mock = MockService::new("svc");
        mock.stub_ok("*", vec![]);
        let md = Metadata::new();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        mock.call("/svc/B", b"", &md, 0).unwrap();
        mock.call("/svc/A", b"", &md, 0).unwrap();
        assert_eq!(mock.captures_for("/svc/A").len(), 2);
        assert_eq!(mock.captures_for("/svc/B").len(), 1);
    }
}
