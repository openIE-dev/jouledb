//! HTTP request mocking.
//!
//! Replaces `nock`, `msw`, and `mockttp` with a pure-Rust mock server model.
//! Supports route matching (exact, glob, regex), request recording, response
//! sequencing, latency simulation, and assertion helpers.

use std::collections::HashMap;

// ── HTTP types ──────────────────────────────────────────────────

/// HTTP method.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Method {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl Method {
    pub fn as_str(&self) -> &str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Patch => "PATCH",
            Method::Delete => "DELETE",
            Method::Head => "HEAD",
            Method::Options => "OPTIONS",
        }
    }
}

/// A recorded HTTP request.
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: Method,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

/// A mock HTTP response.
#[derive(Debug, Clone)]
pub struct MockResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub latency_ms: u64,
}

impl MockResponse {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: HashMap::new(),
            body: String::new(),
            latency_ms: 0,
        }
    }

    pub fn status(mut self, code: u16) -> Self {
        self.status = code;
        self
    }

    pub fn body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }

    pub fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn json(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self.headers
            .insert("content-type".to_string(), "application/json".to_string());
        self
    }
}

// ── Path matching ───────────────────────────────────────────────

/// Strategy for matching request paths.
#[derive(Debug, Clone)]
pub enum PathPattern {
    /// Exact string match.
    Exact(String),
    /// Glob pattern (* and **).
    Glob(String),
    /// Regex pattern.
    Regex(String),
}

impl PathPattern {
    /// Check if a path matches this pattern.
    pub fn matches(&self, path: &str) -> bool {
        match self {
            PathPattern::Exact(expected) => path == expected,
            PathPattern::Glob(pattern) => glob_match(pattern, path),
            PathPattern::Regex(pattern) => regex_match(pattern, path),
        }
    }
}

/// Simple glob matching (supports * for single segment, ** for any).
fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    glob_match_parts(&pattern_parts, &path_parts)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() && path.is_empty() {
        return true;
    }
    if pattern.is_empty() {
        return false;
    }

    if pattern[0] == "**" {
        // ** matches zero or more segments
        for i in 0..=path.len() {
            if glob_match_parts(&pattern[1..], &path[i..]) {
                return true;
            }
        }
        return false;
    }

    if path.is_empty() {
        return false;
    }

    if pattern[0] == "*" || pattern[0] == path[0] {
        glob_match_parts(&pattern[1..], &path[1..])
    } else {
        false
    }
}

/// Simple regex-like matching. Supports: `.` (any char), `.*` (any sequence),
/// `\d` (digit), literal characters. Anchored (full match).
fn regex_match(pattern: &str, text: &str) -> bool {
    let mut pi = 0;
    let mut ti = 0;
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();

    regex_match_inner(&pat, pi, &txt, ti)
}

fn regex_match_inner(pat: &[char], mut pi: usize, txt: &[char], mut ti: usize) -> bool {
    while pi < pat.len() {
        if pi + 1 < pat.len() && pat[pi] == '.' && pat[pi + 1] == '*' {
            // .* matches zero or more of anything
            for i in (ti..=txt.len()).rev() {
                if regex_match_inner(pat, pi + 2, txt, i) {
                    return true;
                }
            }
            return false;
        }

        if ti >= txt.len() {
            return false;
        }

        if pat[pi] == '.' {
            // . matches any single char
            pi += 1;
            ti += 1;
        } else if pat[pi] == '\\' && pi + 1 < pat.len() && pat[pi + 1] == 'd' {
            if !txt[ti].is_ascii_digit() {
                return false;
            }
            pi += 2;
            ti += 1;
        } else if pat[pi] == txt[ti] {
            pi += 1;
            ti += 1;
        } else {
            return false;
        }
    }

    ti == txt.len()
}

// ── MockRoute ───────────────────────────────────────────────────

/// A configured mock route with one or more responses.
#[derive(Debug, Clone)]
pub struct MockRoute {
    pub method: Method,
    pub pattern: PathPattern,
    /// Responses served in order. When exhausted, the last response repeats.
    pub responses: Vec<MockResponse>,
    call_count: usize,
}

impl MockRoute {
    pub fn new(method: Method, pattern: PathPattern, response: MockResponse) -> Self {
        Self {
            method,
            pattern,
            responses: vec![response],
            call_count: 0,
        }
    }

    /// Add a response to the sequence.
    pub fn then(mut self, response: MockResponse) -> Self {
        self.responses.push(response);
        self
    }

    /// How many times this route has been matched.
    pub fn call_count(&self) -> usize {
        self.call_count
    }

    /// Get the next response (advancing the sequence).
    fn next_response(&mut self) -> &MockResponse {
        let idx = self.call_count.min(self.responses.len().saturating_sub(1));
        self.call_count += 1;
        &self.responses[idx]
    }

    /// Check if a request matches this route.
    fn matches(&self, method: &Method, path: &str) -> bool {
        self.method == *method && self.pattern.matches(path)
    }
}

// ── MockServer ──────────────────────────────────────────────────

/// A mock HTTP server that matches requests against configured routes.
#[derive(Debug, Clone, Default)]
pub struct MockServer {
    routes: Vec<MockRoute>,
    recorded: Vec<RecordedRequest>,
}

impl MockServer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a route to the server.
    pub fn add_route(&mut self, route: MockRoute) {
        self.routes.push(route);
    }

    /// Convenience: add a GET route with an exact path.
    pub fn on_get(&mut self, path: &str, response: MockResponse) {
        self.add_route(MockRoute::new(
            Method::Get,
            PathPattern::Exact(path.to_string()),
            response,
        ));
    }

    /// Convenience: add a POST route with an exact path.
    pub fn on_post(&mut self, path: &str, response: MockResponse) {
        self.add_route(MockRoute::new(
            Method::Post,
            PathPattern::Exact(path.to_string()),
            response,
        ));
    }

    /// Dispatch a request and get a response. Records all requests.
    pub fn dispatch(
        &mut self,
        method: Method,
        path: &str,
        headers: HashMap<String, String>,
        body: Option<String>,
    ) -> MockResponse {
        self.recorded.push(RecordedRequest {
            method: method.clone(),
            path: path.to_string(),
            headers: headers.clone(),
            body: body.clone(),
        });

        for route in &mut self.routes {
            if route.matches(&method, path) {
                return route.next_response().clone();
            }
        }

        // No match: 404
        MockResponse::new(404).body("Not Found")
    }

    /// Get all recorded requests.
    pub fn recorded_requests(&self) -> &[RecordedRequest] {
        &self.recorded
    }

    /// Count total requests received.
    pub fn request_count(&self) -> usize {
        self.recorded.len()
    }

    /// Count requests to a specific path.
    pub fn request_count_for(&self, path: &str) -> usize {
        self.recorded.iter().filter(|r| r.path == path).count()
    }

    /// Count requests with a specific method.
    pub fn request_count_for_method(&self, method: &Method) -> usize {
        self.recorded.iter().filter(|r| r.method == *method).count()
    }

    /// Assert that exactly `n` requests were made to `path`.
    pub fn assert_request_count(&self, path: &str, expected: usize) {
        let actual = self.request_count_for(path);
        assert_eq!(
            actual, expected,
            "Expected {expected} requests to \"{path}\", got {actual}"
        );
    }

    /// Assert that a request to `path` had the given header.
    pub fn assert_request_header(&self, path: &str, header_key: &str, header_value: &str) {
        let matching = self
            .recorded
            .iter()
            .filter(|r| r.path == path)
            .any(|r| r.headers.get(header_key).map(|v| v.as_str()) == Some(header_value));
        assert!(
            matching,
            "Expected a request to \"{path}\" with header {header_key}: {header_value}"
        );
    }

    /// Assert that a request to `path` had the given body content.
    pub fn assert_request_body_contains(&self, path: &str, needle: &str) {
        let matching = self
            .recorded
            .iter()
            .filter(|r| r.path == path)
            .any(|r| r.body.as_ref().map(|b| b.contains(needle)).unwrap_or(false));
        assert!(
            matching,
            "Expected a request to \"{path}\" with body containing \"{needle}\""
        );
    }

    /// Reset all recorded requests.
    pub fn reset_recordings(&mut self) {
        self.recorded.clear();
    }

    /// Reset everything (routes and recordings).
    pub fn reset(&mut self) {
        self.routes.clear();
        self.recorded.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_headers() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn exact_route_matching() {
        let mut server = MockServer::new();
        server.on_get("/api/users", MockResponse::new(200).json(r#"[{"id":1}]"#));

        let resp = server.dispatch(Method::Get, "/api/users", empty_headers(), None);
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("id"));
    }

    #[test]
    fn unmatched_returns_404() {
        let mut server = MockServer::new();
        let resp = server.dispatch(Method::Get, "/nonexistent", empty_headers(), None);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn method_must_match() {
        let mut server = MockServer::new();
        server.on_get("/api/data", MockResponse::new(200));

        let resp = server.dispatch(Method::Post, "/api/data", empty_headers(), None);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn glob_pattern_matching() {
        let mut server = MockServer::new();
        server.add_route(MockRoute::new(
            Method::Get,
            PathPattern::Glob("/api/*/items".to_string()),
            MockResponse::new(200).body("ok"),
        ));

        let resp = server.dispatch(Method::Get, "/api/users/items", empty_headers(), None);
        assert_eq!(resp.status, 200);

        let resp = server.dispatch(Method::Get, "/api/orders/items", empty_headers(), None);
        assert_eq!(resp.status, 200);

        let resp = server.dispatch(Method::Get, "/api/items", empty_headers(), None);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn regex_pattern_matching() {
        let mut server = MockServer::new();
        server.add_route(MockRoute::new(
            Method::Get,
            PathPattern::Regex("/users/\\d\\d\\d".to_string()),
            MockResponse::new(200),
        ));

        let resp = server.dispatch(Method::Get, "/users/123", empty_headers(), None);
        assert_eq!(resp.status, 200);

        let resp = server.dispatch(Method::Get, "/users/abc", empty_headers(), None);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn request_recording() {
        let mut server = MockServer::new();
        server.on_get("/api/ping", MockResponse::new(200));

        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token123".to_string());
        server.dispatch(Method::Get, "/api/ping", headers, None);

        assert_eq!(server.request_count(), 1);
        let req = &server.recorded_requests()[0];
        assert_eq!(req.path, "/api/ping");
        assert_eq!(req.headers.get("Authorization").map(|s| s.as_str()), Some("Bearer token123"));
    }

    #[test]
    fn assert_request_count_passes() {
        let mut server = MockServer::new();
        server.on_get("/api/x", MockResponse::new(200));
        server.dispatch(Method::Get, "/api/x", empty_headers(), None);
        server.dispatch(Method::Get, "/api/x", empty_headers(), None);
        server.assert_request_count("/api/x", 2);
    }

    #[test]
    fn assert_request_header_passes() {
        let mut server = MockServer::new();
        server.on_post("/api/submit", MockResponse::new(201));
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        server.dispatch(Method::Post, "/api/submit", headers, Some("{}".to_string()));
        server.assert_request_header("/api/submit", "Content-Type", "application/json");
    }

    #[test]
    fn assert_request_body_contains_passes() {
        let mut server = MockServer::new();
        server.on_post("/api/data", MockResponse::new(200));
        server.dispatch(
            Method::Post,
            "/api/data",
            empty_headers(),
            Some(r#"{"name":"Alice"}"#.to_string()),
        );
        server.assert_request_body_contains("/api/data", "Alice");
    }

    #[test]
    fn response_sequencing() {
        let mut server = MockServer::new();
        let route = MockRoute::new(
            Method::Get,
            PathPattern::Exact("/api/token".to_string()),
            MockResponse::new(200).body("token_1"),
        )
        .then(MockResponse::new(200).body("token_2"))
        .then(MockResponse::new(429).body("rate limited"));

        server.add_route(route);

        let r1 = server.dispatch(Method::Get, "/api/token", empty_headers(), None);
        assert_eq!(r1.body, "token_1");
        let r2 = server.dispatch(Method::Get, "/api/token", empty_headers(), None);
        assert_eq!(r2.body, "token_2");
        let r3 = server.dispatch(Method::Get, "/api/token", empty_headers(), None);
        assert_eq!(r3.body, "rate limited");
        // After exhausted, repeats last
        let r4 = server.dispatch(Method::Get, "/api/token", empty_headers(), None);
        assert_eq!(r4.body, "rate limited");
    }

    #[test]
    fn latency_simulation() {
        let resp = MockResponse::new(200).latency(500);
        assert_eq!(resp.latency_ms, 500);
    }

    #[test]
    fn reset_clears_everything() {
        let mut server = MockServer::new();
        server.on_get("/x", MockResponse::new(200));
        server.dispatch(Method::Get, "/x", empty_headers(), None);
        server.reset();
        assert_eq!(server.request_count(), 0);
        let resp = server.dispatch(Method::Get, "/x", empty_headers(), None);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn glob_double_star() {
        let mut server = MockServer::new();
        server.add_route(MockRoute::new(
            Method::Get,
            PathPattern::Glob("/api/**".to_string()),
            MockResponse::new(200),
        ));

        let resp = server.dispatch(Method::Get, "/api/a/b/c", empty_headers(), None);
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn request_count_by_method() {
        let mut server = MockServer::new();
        server.on_get("/x", MockResponse::new(200));
        server.on_post("/x", MockResponse::new(201));
        server.dispatch(Method::Get, "/x", empty_headers(), None);
        server.dispatch(Method::Get, "/x", empty_headers(), None);
        server.dispatch(Method::Post, "/x", empty_headers(), None);
        assert_eq!(server.request_count_for_method(&Method::Get), 2);
        assert_eq!(server.request_count_for_method(&Method::Post), 1);
    }
}
