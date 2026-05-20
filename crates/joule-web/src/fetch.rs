//! Platform-agnostic HTTP client abstraction with interceptors and retry policies.
//!
//! Replaces Axios/fetch with a pure-Rust request/response builder that prepares
//! requests for platform-specific execution (browser fetch API or native HTTP).

use serde::Serialize;
use serde::de::DeserializeOwned;

// ── Method ──

/// HTTP request method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::Get => write!(f, "GET"),
            Method::Post => write!(f, "POST"),
            Method::Put => write!(f, "PUT"),
            Method::Patch => write!(f, "PATCH"),
            Method::Delete => write!(f, "DELETE"),
            Method::Head => write!(f, "HEAD"),
            Method::Options => write!(f, "OPTIONS"),
        }
    }
}

// ── Headers ──

/// HTTP header collection.
#[derive(Debug, Clone, Default)]
pub struct Headers {
    entries: Vec<(String, String)>,
}

impl Headers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a header value. Replaces any existing header with the same name (case-insensitive).
    pub fn set(&mut self, name: &str, value: &str) {
        let lower = name.to_lowercase();
        self.entries.retain(|(k, _)| k.to_lowercase() != lower);
        self.entries.push((name.to_string(), value.to_string()));
    }

    /// Get the first value for a header name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Remove all entries for a header name (case-insensitive).
    pub fn remove(&mut self, name: &str) {
        let lower = name.to_lowercase();
        self.entries.retain(|(k, _)| k.to_lowercase() != lower);
    }

    /// Iterate over all header entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Set Content-Type to application/json.
    pub fn json(&mut self) {
        self.set("Content-Type", "application/json");
    }

    /// Set Content-Type to application/x-www-form-urlencoded.
    pub fn form(&mut self) {
        self.set("Content-Type", "application/x-www-form-urlencoded");
    }

    /// Set Content-Type to text/plain.
    pub fn text(&mut self) {
        self.set("Content-Type", "text/plain");
    }
}

// ── Request ──

/// An HTTP request that can be built fluently.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: Headers,
    pub body: Option<Vec<u8>>,
    pub timeout_ms: Option<u64>,
    pub query_params: Vec<(String, String)>,
}

impl Request {
    fn new(method: Method, url: &str) -> Self {
        Self {
            method,
            url: url.to_string(),
            headers: Headers::new(),
            body: None,
            timeout_ms: None,
            query_params: Vec::new(),
        }
    }

    pub fn get(url: &str) -> Self {
        Self::new(Method::Get, url)
    }

    pub fn post(url: &str) -> Self {
        Self::new(Method::Post, url)
    }

    pub fn put(url: &str) -> Self {
        Self::new(Method::Put, url)
    }

    pub fn patch(url: &str) -> Self {
        Self::new(Method::Patch, url)
    }

    pub fn delete(url: &str) -> Self {
        Self::new(Method::Delete, url)
    }

    pub fn head(url: &str) -> Self {
        Self::new(Method::Head, url)
    }

    pub fn options(url: &str) -> Self {
        Self::new(Method::Options, url)
    }

    /// Set a header.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.set(name, value);
        self
    }

    /// Serialize a body as JSON and set the Content-Type header.
    pub fn json<T: Serialize>(mut self, body: &T) -> Self {
        match serde_json::to_vec(body) {
            Ok(bytes) => {
                self.body = Some(bytes);
                self.headers.json();
            }
            Err(_) => {
                // In a production system this would return a Result.
                // For now, leave body empty on serialization failure.
            }
        }
        self
    }

    /// Set form-encoded body from key-value pairs.
    pub fn form(mut self, pairs: &[(&str, &str)]) -> Self {
        let encoded: Vec<String> = pairs
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect();
        self.body = Some(encoded.join("&").into_bytes());
        self.headers.form();
        self
    }

    /// Add a query parameter.
    pub fn query(mut self, key: &str, val: &str) -> Self {
        self.query_params.push((key.to_string(), val.to_string()));
        self
    }

    /// Set a request timeout in milliseconds.
    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Set an Authorization: Bearer token.
    pub fn bearer_token(mut self, token: &str) -> Self {
        self.headers
            .set("Authorization", &format!("Bearer {token}"));
        self
    }

    /// Build the final URL with query parameters appended.
    pub fn build(mut self) -> Self {
        if !self.query_params.is_empty() {
            let qs: Vec<String> = self
                .query_params
                .iter()
                .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
                .collect();
            let sep = if self.url.contains('?') { "&" } else { "?" };
            self.url = format!("{}{}{}", self.url, sep, qs.join("&"));
            self.query_params.clear();
        }
        self
    }
}

/// Minimal URL encoding for query parameters.
fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(ch),
            ' ' => result.push('+'),
            _ => {
                for byte in ch.to_string().as_bytes() {
                    result.push_str(&format!("%{byte:02X}"));
                }
            }
        }
    }
    result
}

// ── Response ──

/// An HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub status_text: String,
    pub headers: Headers,
    pub body: Vec<u8>,
}

/// Errors that can occur when processing responses.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Response {
    /// Create a new response (typically from platform-specific code).
    pub fn new(status: u16, status_text: &str, headers: Headers, body: Vec<u8>) -> Self {
        Self {
            status,
            status_text: status_text.to_string(),
            headers,
            body,
        }
    }

    /// Returns `true` if the status is in the 200-299 range.
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Decode the body as UTF-8 text.
    pub fn text(&self) -> Result<String, FetchError> {
        Ok(String::from_utf8(self.body.clone())?)
    }

    /// Deserialize the body as JSON.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T, FetchError> {
        Ok(serde_json::from_slice(&self.body)?)
    }
}

// ── Interceptor ──

/// Trait for request/response interceptors (e.g., auth, logging, caching).
pub trait Interceptor {
    /// Called before a request is sent. Can modify the request.
    fn on_request(&self, req: &mut Request);
    /// Called after a response is received. Can modify the response.
    fn on_response(&self, resp: &mut Response);
}

// ── Fetch Client ──

/// Platform-agnostic HTTP client that prepares requests with defaults and interceptors.
///
/// Does NOT actually execute HTTP — actual execution is platform-specific.
/// The `prepare()` method is the testable surface.
pub struct FetchClient {
    pub base_url: Option<String>,
    pub default_headers: Headers,
    interceptors: Vec<Box<dyn Interceptor>>,
    pub timeout_ms: u64,
}

impl FetchClient {
    pub fn new() -> Self {
        Self {
            base_url: None,
            default_headers: Headers::new(),
            interceptors: Vec::new(),
            timeout_ms: 30_000,
        }
    }

    pub fn with_base_url(url: &str) -> Self {
        Self {
            base_url: Some(url.trim_end_matches('/').to_string()),
            ..Self::new()
        }
    }

    /// Add an interceptor.
    pub fn add_interceptor(&mut self, i: impl Interceptor + 'static) {
        self.interceptors.push(Box::new(i));
    }

    /// Prepare a request: apply base URL, default headers, build query params,
    /// and run request interceptors.
    pub fn prepare(&self, mut req: Request) -> Request {
        // Apply base URL.
        if let Some(base) = &self.base_url {
            if !req.url.starts_with("http://") && !req.url.starts_with("https://") {
                let sep = if req.url.starts_with('/') { "" } else { "/" };
                req.url = format!("{base}{sep}{}", req.url);
            }
        }

        // Apply default headers (don't overwrite existing).
        for (name, value) in self.default_headers.iter() {
            if req.headers.get(name).is_none() {
                req.headers.set(name, value);
            }
        }

        // Apply default timeout if not set.
        if req.timeout_ms.is_none() {
            req.timeout_ms = Some(self.timeout_ms);
        }

        // Build query params into URL.
        req = req.build();

        // Run interceptors.
        for interceptor in &self.interceptors {
            interceptor.on_request(&mut req);
        }

        req
    }
}

impl Default for FetchClient {
    fn default() -> Self {
        Self::new()
    }
}

// ── Retry Policy ──

/// Configurable retry policy with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff_ms: Vec<u64>,
    pub retry_on: Vec<u16>,
}

impl RetryPolicy {
    /// Default: 3 retries, exponential backoff (1s, 2s, 4s), retry on common server errors.
    pub fn default_policy() -> Self {
        Self {
            max_retries: 3,
            backoff_ms: vec![1000, 2000, 4000],
            retry_on: vec![429, 500, 502, 503, 504],
        }
    }

    /// Determine whether to retry and the backoff duration.
    /// Returns `Some(backoff_ms)` if should retry, `None` otherwise.
    pub fn should_retry(&self, attempt: u32, status: u16) -> Option<u64> {
        if attempt >= self.max_retries {
            return None;
        }
        if !self.retry_on.contains(&status) {
            return None;
        }
        let backoff = if (attempt as usize) < self.backoff_ms.len() {
            self.backoff_ms[attempt as usize]
        } else {
            // Extrapolate: double the last value.
            self.backoff_ms
                .last()
                .copied()
                .unwrap_or(1000)
                .saturating_mul(2)
        };
        Some(backoff)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder() {
        let req = Request::get("https://api.example.com/users")
            .header("Accept", "application/json")
            .timeout(5000)
            .build();
        assert_eq!(req.method, Method::Get);
        assert_eq!(req.url, "https://api.example.com/users");
        assert_eq!(req.headers.get("Accept").unwrap(), "application/json");
        assert_eq!(req.timeout_ms, Some(5000));
    }

    #[test]
    fn test_json_body_serialized() {
        #[derive(Serialize)]
        struct Payload {
            name: String,
        }
        let req = Request::post("https://api.example.com/users")
            .json(&Payload {
                name: "Alice".into(),
            })
            .build();
        assert!(req.body.is_some());
        let body_str = String::from_utf8(req.body.unwrap()).unwrap();
        assert!(body_str.contains("Alice"));
        assert_eq!(
            req.headers.get("Content-Type").unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_query_params_appended() {
        let req = Request::get("https://api.example.com/search")
            .query("q", "rust")
            .query("page", "2")
            .build();
        assert!(req.url.contains("q=rust"));
        assert!(req.url.contains("page=2"));
        assert!(req.url.contains('?'));
    }

    #[test]
    fn test_headers_set_get_remove() {
        let mut h = Headers::new();
        h.set("X-Custom", "value1");
        assert_eq!(h.get("x-custom").unwrap(), "value1");
        h.set("X-Custom", "value2");
        assert_eq!(h.get("X-Custom").unwrap(), "value2");
        h.remove("x-custom");
        assert!(h.get("X-Custom").is_none());
    }

    #[test]
    fn test_response_ok_200() {
        let resp = Response::new(200, "OK", Headers::new(), vec![]);
        assert!(resp.ok());
    }

    #[test]
    fn test_response_ok_404() {
        let resp = Response::new(404, "Not Found", Headers::new(), vec![]);
        assert!(!resp.ok());
    }

    #[test]
    fn test_response_json_deserialization() {
        use serde::Deserialize;
        #[derive(Deserialize, Debug, PartialEq)]
        struct User {
            name: String,
        }
        let body = br#"{"name":"Bob"}"#.to_vec();
        let resp = Response::new(200, "OK", Headers::new(), body);
        let user: User = resp.json().unwrap();
        assert_eq!(user.name, "Bob");
    }

    #[test]
    fn test_interceptor_modifies_request() {
        struct AuthInterceptor;
        impl Interceptor for AuthInterceptor {
            fn on_request(&self, req: &mut Request) {
                req.headers.set("X-Auth", "secret");
            }
            fn on_response(&self, _resp: &mut Response) {}
        }

        let mut client = FetchClient::new();
        client.add_interceptor(AuthInterceptor);
        let req = client.prepare(Request::get("https://api.example.com/data"));
        assert_eq!(req.headers.get("X-Auth").unwrap(), "secret");
    }

    #[test]
    fn test_base_url_prepends() {
        let client = FetchClient::with_base_url("https://api.example.com");
        let req = client.prepare(Request::get("/users"));
        assert_eq!(req.url, "https://api.example.com/users");
    }

    #[test]
    fn test_base_url_does_not_prepend_absolute() {
        let client = FetchClient::with_base_url("https://api.example.com");
        let req = client.prepare(Request::get("https://other.com/data"));
        assert_eq!(req.url, "https://other.com/data");
    }

    #[test]
    fn test_bearer_token() {
        let req = Request::get("https://api.example.com")
            .bearer_token("my-token")
            .build();
        assert_eq!(
            req.headers.get("Authorization").unwrap(),
            "Bearer my-token"
        );
    }

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy::default_policy();
        // Attempt 0 on 500 -> retry with 1000ms backoff.
        assert_eq!(policy.should_retry(0, 500), Some(1000));
        // Attempt 1 on 500 -> retry with 2000ms backoff.
        assert_eq!(policy.should_retry(1, 500), Some(2000));
        // Attempt 2 on 500 -> retry with 4000ms backoff.
        assert_eq!(policy.should_retry(2, 500), Some(4000));
    }

    #[test]
    fn test_retry_policy_respects_max() {
        let policy = RetryPolicy::default_policy();
        // Attempt 3 (past max_retries of 3) -> no retry.
        assert_eq!(policy.should_retry(3, 500), None);
    }

    #[test]
    fn test_retry_policy_no_retry_on_success() {
        let policy = RetryPolicy::default_policy();
        assert_eq!(policy.should_retry(0, 200), None);
    }

    #[test]
    fn test_form_encoded_body() {
        let req = Request::post("https://api.example.com/login")
            .form(&[("username", "alice"), ("password", "secret")])
            .build();
        let body = String::from_utf8(req.body.unwrap()).unwrap();
        assert!(body.contains("username=alice"));
        assert!(body.contains("password=secret"));
        assert_eq!(
            req.headers.get("Content-Type").unwrap(),
            "application/x-www-form-urlencoded"
        );
    }

    #[test]
    fn test_default_timeout_applied() {
        let client = FetchClient::new();
        let req = client.prepare(Request::get("https://api.example.com"));
        assert_eq!(req.timeout_ms, Some(30_000));
    }
}
