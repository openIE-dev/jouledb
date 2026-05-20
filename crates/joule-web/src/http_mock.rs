//! HTTP mocking — request matchers, response builders, expectation counts,
//! request recording, mock server state, and verification.
//!
//! Replaces JS HTTP mocking libraries (nock, msw, wiremock) with a pure-Rust
//! mock HTTP layer for testing request/response flows without real networking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// HTTP mock errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockError {
    /// No mock matched the request.
    NoMatchFound(String),
    /// Mock expectation not met.
    ExpectationNotMet { mock_id: String, expected: u32, actual: u32 },
    /// Duplicate mock ID.
    DuplicateMockId(String),
    /// Mock server not started.
    NotStarted,
    /// Verification failed — multiple unmet expectations.
    VerificationFailed(Vec<String>),
}

impl fmt::Display for MockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMatchFound(url) => write!(f, "no mock matched request: {url}"),
            Self::ExpectationNotMet { mock_id, expected, actual } => {
                write!(f, "mock '{mock_id}': expected {expected} calls, got {actual}")
            }
            Self::DuplicateMockId(id) => write!(f, "duplicate mock ID: {id}"),
            Self::NotStarted => write!(f, "mock server not started"),
            Self::VerificationFailed(msgs) => {
                write!(f, "verification failed:\n{}", msgs.join("\n"))
            }
        }
    }
}

impl std::error::Error for MockError {}

// ── HTTP Method ────────────────────────────────────────────────

/// HTTP method.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Patch => write!(f, "PATCH"),
            Self::Delete => write!(f, "DELETE"),
            Self::Head => write!(f, "HEAD"),
            Self::Options => write!(f, "OPTIONS"),
        }
    }
}

// ── Request / Response ─────────────────────────────────────────

/// A recorded HTTP request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MockRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub query_params: HashMap<String, String>,
}

impl MockRequest {
    /// Create a new GET request.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Get,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            query_params: HashMap::new(),
        }
    }

    /// Create a new POST request.
    pub fn post(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            query_params: HashMap::new(),
        }
    }

    /// Add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add a body.
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Add a query parameter.
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    /// Create a request with a specific method.
    pub fn with_method(method: HttpMethod, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            query_params: HashMap::new(),
        }
    }
}

/// A mock HTTP response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MockResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

impl MockResponse {
    /// Create a 200 OK response.
    pub fn ok() -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Create a response with a status code.
    pub fn status(code: u16) -> Self {
        Self {
            status: code,
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Add a response body.
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Add a JSON response body.
    pub fn with_json(mut self, value: &serde_json::Value) -> Self {
        self.body = Some(serde_json::to_string(value).unwrap_or_default());
        self.headers.insert("content-type".to_string(), "application/json".to_string());
        self
    }

    /// Add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

// ── Request Matcher ────────────────────────────────────────────

/// How to match an incoming request against a mock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestMatcher {
    /// Exact URL match.
    ExactUrl(String),
    /// URL prefix match.
    UrlPrefix(String),
    /// URL contains substring.
    UrlContains(String),
    /// Match a specific header key-value.
    Header { key: String, value: String },
    /// Match a specific query parameter.
    QueryParam { key: String, value: String },
    /// Match body contains substring.
    BodyContains(String),
    /// Match HTTP method.
    Method(HttpMethod),
    /// All matchers must match (AND).
    All(Vec<RequestMatcher>),
    /// At least one matcher must match (OR).
    Any(Vec<RequestMatcher>),
}

impl RequestMatcher {
    /// Check if a request matches this matcher.
    pub fn matches(&self, req: &MockRequest) -> bool {
        match self {
            Self::ExactUrl(url) => req.url == *url,
            Self::UrlPrefix(prefix) => req.url.starts_with(prefix),
            Self::UrlContains(sub) => req.url.contains(sub),
            Self::Header { key, value } => {
                req.headers.get(key).map_or(false, |v| v == value)
            }
            Self::QueryParam { key, value } => {
                req.query_params.get(key).map_or(false, |v| v == value)
            }
            Self::BodyContains(sub) => {
                req.body.as_ref().map_or(false, |b| b.contains(sub))
            }
            Self::Method(method) => req.method == *method,
            Self::All(matchers) => matchers.iter().all(|m| m.matches(req)),
            Self::Any(matchers) => matchers.iter().any(|m| m.matches(req)),
        }
    }
}

// ── Expectation ────────────────────────────────────────────────

/// How many times a mock should be called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expectation {
    /// Exactly N times.
    Exactly(u32),
    /// At least N times.
    AtLeast(u32),
    /// At most N times.
    AtMost(u32),
    /// Between min and max (inclusive).
    Between(u32, u32),
    /// Any number of times (no check).
    Any,
}

impl Expectation {
    /// Check if actual count satisfies expectation.
    pub fn is_satisfied(&self, actual: u32) -> bool {
        match self {
            Self::Exactly(n) => actual == *n,
            Self::AtLeast(n) => actual >= *n,
            Self::AtMost(n) => actual <= *n,
            Self::Between(min, max) => actual >= *min && actual <= *max,
            Self::Any => true,
        }
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        match self {
            Self::Exactly(n) => format!("exactly {n}"),
            Self::AtLeast(n) => format!("at least {n}"),
            Self::AtMost(n) => format!("at most {n}"),
            Self::Between(min, max) => format!("between {min} and {max}"),
            Self::Any => "any number of".to_string(),
        }
    }
}

// ── Mock Definition ────────────────────────────────────────────

/// A single mock definition with matcher, response, and expectation.
#[derive(Debug, Clone)]
pub struct MockDefinition {
    pub id: String,
    pub matcher: RequestMatcher,
    pub response: MockResponse,
    pub expectation: Expectation,
    pub call_count: u32,
    pub priority: u32,
}

impl MockDefinition {
    /// Create a new mock.
    pub fn new(id: impl Into<String>, matcher: RequestMatcher, response: MockResponse) -> Self {
        Self {
            id: id.into(),
            matcher,
            response,
            expectation: Expectation::Any,
            call_count: 0,
            priority: 0,
        }
    }

    /// Set expected call count.
    pub fn with_expectation(mut self, expectation: Expectation) -> Self {
        self.expectation = expectation;
        self
    }

    /// Set priority (higher = checked first).
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if the expectation is satisfied.
    pub fn is_satisfied(&self) -> bool {
        self.expectation.is_satisfied(self.call_count)
    }
}

// ── Mock Server ────────────────────────────────────────────────

/// A mock HTTP server that records requests and returns configured responses.
#[derive(Debug, Clone)]
pub struct MockServer {
    mocks: Vec<MockDefinition>,
    recorded_requests: Vec<MockRequest>,
    started: bool,
    unmatched_requests: Vec<MockRequest>,
}

impl MockServer {
    /// Create a new mock server.
    pub fn new() -> Self {
        Self {
            mocks: Vec::new(),
            recorded_requests: Vec::new(),
            started: false,
            unmatched_requests: Vec::new(),
        }
    }

    /// Start the mock server.
    pub fn start(&mut self) {
        self.started = true;
    }

    /// Stop the mock server.
    pub fn stop(&mut self) {
        self.started = false;
    }

    /// Check if server is started.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Register a mock definition.
    pub fn register(&mut self, mock: MockDefinition) -> Result<(), MockError> {
        if self.mocks.iter().any(|m| m.id == mock.id) {
            return Err(MockError::DuplicateMockId(mock.id));
        }
        self.mocks.push(mock);
        // Sort by priority descending so higher priority checked first
        self.mocks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    /// Remove a mock by ID.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.mocks.len();
        self.mocks.retain(|m| m.id != id);
        self.mocks.len() < before
    }

    /// Handle a request — find matching mock and return response.
    pub fn handle(&mut self, request: MockRequest) -> Result<MockResponse, MockError> {
        if !self.started {
            return Err(MockError::NotStarted);
        }

        self.recorded_requests.push(request.clone());

        for mock in &mut self.mocks {
            if mock.matcher.matches(&request) {
                mock.call_count += 1;
                return Ok(mock.response.clone());
            }
        }

        self.unmatched_requests.push(request.clone());
        Err(MockError::NoMatchFound(request.url))
    }

    /// Get all recorded requests.
    pub fn recorded_requests(&self) -> &[MockRequest] {
        &self.recorded_requests
    }

    /// Get unmatched requests.
    pub fn unmatched_requests(&self) -> &[MockRequest] {
        &self.unmatched_requests
    }

    /// Get recorded requests matching a specific URL.
    pub fn requests_for(&self, url: &str) -> Vec<&MockRequest> {
        self.recorded_requests
            .iter()
            .filter(|r| r.url == url)
            .collect()
    }

    /// Get total number of recorded requests.
    pub fn request_count(&self) -> usize {
        self.recorded_requests.len()
    }

    /// Verify all mock expectations.
    pub fn verify(&self) -> Result<(), MockError> {
        let mut failures = Vec::new();
        for mock in &self.mocks {
            if !mock.is_satisfied() {
                failures.push(format!(
                    "mock '{}': expected {} calls, got {}",
                    mock.id,
                    mock.expectation.describe(),
                    mock.call_count,
                ));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(MockError::VerificationFailed(failures))
        }
    }

    /// Reset all mocks and recorded requests.
    pub fn reset(&mut self) {
        self.mocks.clear();
        self.recorded_requests.clear();
        self.unmatched_requests.clear();
    }

    /// Number of registered mocks.
    pub fn mock_count(&self) -> usize {
        self.mocks.len()
    }
}

impl Default for MockServer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn started_server() -> MockServer {
        let mut s = MockServer::new();
        s.start();
        s
    }

    #[test]
    fn test_mock_request_get() {
        let req = MockRequest::get("/api/users");
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.url, "/api/users");
    }

    #[test]
    fn test_mock_request_post_with_body() {
        let req = MockRequest::post("/api/users")
            .with_body(r#"{"name":"alice"}"#)
            .with_header("content-type", "application/json");
        assert_eq!(req.method, HttpMethod::Post);
        assert!(req.body.as_ref().unwrap().contains("alice"));
        assert_eq!(req.headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_mock_request_with_query() {
        let req = MockRequest::get("/search")
            .with_query("q", "rust")
            .with_query("page", "1");
        assert_eq!(req.query_params.get("q").unwrap(), "rust");
        assert_eq!(req.query_params.get("page").unwrap(), "1");
    }

    #[test]
    fn test_mock_response_ok() {
        let resp = MockResponse::ok();
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn test_mock_response_json() {
        let resp = MockResponse::ok().with_json(&json!({"status": "ok"}));
        assert!(resp.body.unwrap().contains("ok"));
        assert_eq!(resp.headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_mock_response_status() {
        let resp = MockResponse::status(404).with_body("not found");
        assert_eq!(resp.status, 404);
        assert_eq!(resp.body.unwrap(), "not found");
    }

    #[test]
    fn test_matcher_exact_url() {
        let matcher = RequestMatcher::ExactUrl("/api/v1".to_string());
        assert!(matcher.matches(&MockRequest::get("/api/v1")));
        assert!(!matcher.matches(&MockRequest::get("/api/v2")));
    }

    #[test]
    fn test_matcher_url_prefix() {
        let matcher = RequestMatcher::UrlPrefix("/api".to_string());
        assert!(matcher.matches(&MockRequest::get("/api/v1")));
        assert!(matcher.matches(&MockRequest::get("/api/v2")));
        assert!(!matcher.matches(&MockRequest::get("/other")));
    }

    #[test]
    fn test_matcher_url_contains() {
        let matcher = RequestMatcher::UrlContains("users".to_string());
        assert!(matcher.matches(&MockRequest::get("/api/users/1")));
        assert!(!matcher.matches(&MockRequest::get("/api/posts")));
    }

    #[test]
    fn test_matcher_header() {
        let matcher = RequestMatcher::Header {
            key: "auth".to_string(),
            value: "bearer abc".to_string(),
        };
        let req = MockRequest::get("/").with_header("auth", "bearer abc");
        assert!(matcher.matches(&req));
        let req2 = MockRequest::get("/");
        assert!(!matcher.matches(&req2));
    }

    #[test]
    fn test_matcher_query_param() {
        let matcher = RequestMatcher::QueryParam {
            key: "format".to_string(),
            value: "json".to_string(),
        };
        let req = MockRequest::get("/").with_query("format", "json");
        assert!(matcher.matches(&req));
    }

    #[test]
    fn test_matcher_body_contains() {
        let matcher = RequestMatcher::BodyContains("alice".to_string());
        let req = MockRequest::post("/").with_body("user: alice");
        assert!(matcher.matches(&req));
        let req2 = MockRequest::post("/").with_body("user: bob");
        assert!(!matcher.matches(&req2));
    }

    #[test]
    fn test_matcher_method() {
        let matcher = RequestMatcher::Method(HttpMethod::Post);
        assert!(matcher.matches(&MockRequest::post("/")));
        assert!(!matcher.matches(&MockRequest::get("/")));
    }

    #[test]
    fn test_matcher_all() {
        let matcher = RequestMatcher::All(vec![
            RequestMatcher::Method(HttpMethod::Get),
            RequestMatcher::UrlPrefix("/api".to_string()),
        ]);
        assert!(matcher.matches(&MockRequest::get("/api/x")));
        assert!(!matcher.matches(&MockRequest::post("/api/x")));
        assert!(!matcher.matches(&MockRequest::get("/other")));
    }

    #[test]
    fn test_matcher_any() {
        let matcher = RequestMatcher::Any(vec![
            RequestMatcher::ExactUrl("/a".to_string()),
            RequestMatcher::ExactUrl("/b".to_string()),
        ]);
        assert!(matcher.matches(&MockRequest::get("/a")));
        assert!(matcher.matches(&MockRequest::get("/b")));
        assert!(!matcher.matches(&MockRequest::get("/c")));
    }

    #[test]
    fn test_expectation_exactly() {
        assert!(Expectation::Exactly(3).is_satisfied(3));
        assert!(!Expectation::Exactly(3).is_satisfied(2));
    }

    #[test]
    fn test_expectation_at_least() {
        assert!(Expectation::AtLeast(2).is_satisfied(3));
        assert!(Expectation::AtLeast(2).is_satisfied(2));
        assert!(!Expectation::AtLeast(2).is_satisfied(1));
    }

    #[test]
    fn test_expectation_at_most() {
        assert!(Expectation::AtMost(3).is_satisfied(2));
        assert!(Expectation::AtMost(3).is_satisfied(3));
        assert!(!Expectation::AtMost(3).is_satisfied(4));
    }

    #[test]
    fn test_expectation_between() {
        assert!(Expectation::Between(2, 5).is_satisfied(3));
        assert!(Expectation::Between(2, 5).is_satisfied(2));
        assert!(Expectation::Between(2, 5).is_satisfied(5));
        assert!(!Expectation::Between(2, 5).is_satisfied(1));
        assert!(!Expectation::Between(2, 5).is_satisfied(6));
    }

    #[test]
    fn test_expectation_any() {
        assert!(Expectation::Any.is_satisfied(0));
        assert!(Expectation::Any.is_satisfied(100));
    }

    #[test]
    fn test_expectation_describe() {
        assert_eq!(Expectation::Exactly(3).describe(), "exactly 3");
        assert_eq!(Expectation::AtLeast(1).describe(), "at least 1");
        assert_eq!(Expectation::AtMost(5).describe(), "at most 5");
        assert!(Expectation::Between(2, 4).describe().contains("between"));
    }

    #[test]
    fn test_server_not_started() {
        let mut server = MockServer::new();
        let result = server.handle(MockRequest::get("/"));
        assert!(matches!(result, Err(MockError::NotStarted)));
    }

    #[test]
    fn test_server_basic_mock() {
        let mut server = started_server();
        let mock = MockDefinition::new(
            "get_users",
            RequestMatcher::ExactUrl("/api/users".to_string()),
            MockResponse::ok().with_body("[]"),
        );
        server.register(mock).unwrap();
        let resp = server.handle(MockRequest::get("/api/users")).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body.unwrap(), "[]");
    }

    #[test]
    fn test_server_no_match() {
        let mut server = started_server();
        let result = server.handle(MockRequest::get("/unknown"));
        assert!(matches!(result, Err(MockError::NoMatchFound(_))));
        assert_eq!(server.unmatched_requests().len(), 1);
    }

    #[test]
    fn test_server_duplicate_mock() {
        let mut server = started_server();
        let mock1 = MockDefinition::new("m1", RequestMatcher::ExactUrl("/a".to_string()), MockResponse::ok());
        let mock2 = MockDefinition::new("m1", RequestMatcher::ExactUrl("/b".to_string()), MockResponse::ok());
        server.register(mock1).unwrap();
        let result = server.register(mock2);
        assert!(matches!(result, Err(MockError::DuplicateMockId(_))));
    }

    #[test]
    fn test_server_recording() {
        let mut server = started_server();
        let mock = MockDefinition::new(
            "any",
            RequestMatcher::UrlPrefix("/".to_string()),
            MockResponse::ok(),
        );
        server.register(mock).unwrap();

        server.handle(MockRequest::get("/a")).unwrap();
        server.handle(MockRequest::get("/b")).unwrap();
        server.handle(MockRequest::post("/c")).unwrap();

        assert_eq!(server.request_count(), 3);
        assert_eq!(server.requests_for("/a").len(), 1);
        assert_eq!(server.requests_for("/b").len(), 1);
    }

    #[test]
    fn test_server_verify_success() {
        let mut server = started_server();
        let mock = MockDefinition::new(
            "once",
            RequestMatcher::ExactUrl("/x".to_string()),
            MockResponse::ok(),
        )
        .with_expectation(Expectation::Exactly(1));
        server.register(mock).unwrap();
        server.handle(MockRequest::get("/x")).unwrap();
        assert!(server.verify().is_ok());
    }

    #[test]
    fn test_server_verify_failure() {
        let mut server = started_server();
        let mock = MockDefinition::new(
            "twice",
            RequestMatcher::ExactUrl("/x".to_string()),
            MockResponse::ok(),
        )
        .with_expectation(Expectation::Exactly(2));
        server.register(mock).unwrap();
        server.handle(MockRequest::get("/x")).unwrap();
        let result = server.verify();
        assert!(matches!(result, Err(MockError::VerificationFailed(_))));
    }

    #[test]
    fn test_server_priority() {
        let mut server = started_server();
        let low = MockDefinition::new(
            "low",
            RequestMatcher::UrlPrefix("/api".to_string()),
            MockResponse::status(200),
        )
        .with_priority(1);
        let high = MockDefinition::new(
            "high",
            RequestMatcher::UrlPrefix("/api".to_string()),
            MockResponse::status(201),
        )
        .with_priority(10);
        server.register(low).unwrap();
        server.register(high).unwrap();
        let resp = server.handle(MockRequest::get("/api/data")).unwrap();
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn test_server_remove_mock() {
        let mut server = started_server();
        let mock = MockDefinition::new("rm", RequestMatcher::ExactUrl("/x".to_string()), MockResponse::ok());
        server.register(mock).unwrap();
        assert_eq!(server.mock_count(), 1);
        assert!(server.remove("rm"));
        assert_eq!(server.mock_count(), 0);
        assert!(!server.remove("rm"));
    }

    #[test]
    fn test_server_reset() {
        let mut server = started_server();
        let mock = MockDefinition::new("m", RequestMatcher::UrlPrefix("/".to_string()), MockResponse::ok());
        server.register(mock).unwrap();
        server.handle(MockRequest::get("/")).unwrap();
        server.reset();
        assert_eq!(server.mock_count(), 0);
        assert_eq!(server.request_count(), 0);
    }

    #[test]
    fn test_server_start_stop() {
        let mut server = MockServer::new();
        assert!(!server.is_started());
        server.start();
        assert!(server.is_started());
        server.stop();
        assert!(!server.is_started());
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(format!("{}", HttpMethod::Get), "GET");
        assert_eq!(format!("{}", HttpMethod::Post), "POST");
        assert_eq!(format!("{}", HttpMethod::Delete), "DELETE");
    }

    #[test]
    fn test_error_display() {
        let err = MockError::NoMatchFound("/test".to_string());
        assert!(format!("{err}").contains("/test"));
    }

    #[test]
    fn test_mock_with_method() {
        let req = MockRequest::with_method(HttpMethod::Put, "/api/item/1");
        assert_eq!(req.method, HttpMethod::Put);
        assert_eq!(req.url, "/api/item/1");
    }

    #[test]
    fn test_response_with_header() {
        let resp = MockResponse::ok().with_header("x-custom", "value123");
        assert_eq!(resp.headers.get("x-custom").unwrap(), "value123");
    }
}
