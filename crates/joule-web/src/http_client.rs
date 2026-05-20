//! HTTP client abstraction — builder layer with no actual I/O.
//!
//! Replaces `reqwest`, `axios`, `node-fetch`, and `got` builder patterns with
//! pure Rust.  Request/Response builders, method types, header map, query string
//! builder, timeout config, redirect policy, cookie forwarding, request
//! interceptors.

use std::collections::HashMap;
use std::fmt;

// ── Method ─────────────────────────────────────────────────────

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
    Trace,
    Connect,
}

impl Method {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Trace => "TRACE",
            Self::Connect => "CONNECT",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "PATCH" => Some(Self::Patch),
            "DELETE" => Some(Self::Delete),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            "TRACE" => Some(Self::Trace),
            "CONNECT" => Some(Self::Connect),
            _ => None,
        }
    }

    pub fn is_safe(&self) -> bool {
        matches!(self, Self::Get | Self::Head | Self::Options | Self::Trace)
    }

    pub fn is_idempotent(&self) -> bool {
        matches!(
            self,
            Self::Get | Self::Head | Self::Options | Self::Trace | Self::Put | Self::Delete
        )
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Header map ─────────────────────────────────────────────────

/// Case-insensitive header map.
#[derive(Debug, Clone, Default)]
pub struct HeaderMap {
    inner: Vec<(String, String)>,
}

impl HeaderMap {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn insert(&mut self, name: &str, value: &str) {
        let lower = name.to_ascii_lowercase();
        // Replace first existing occurrence, or append.
        for entry in &mut self.inner {
            if entry.0 == lower {
                entry.1 = value.to_string();
                return;
            }
        }
        self.inner.push((lower, value.to_string()));
    }

    pub fn append(&mut self, name: &str, value: &str) {
        self.inner
            .push((name.to_ascii_lowercase(), value.to_string()));
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        let lower = name.to_ascii_lowercase();
        self.inner
            .iter()
            .find(|(k, _)| *k == lower)
            .map(|(_, v)| v.as_str())
    }

    pub fn get_all(&self, name: &str) -> Vec<&str> {
        let lower = name.to_ascii_lowercase();
        self.inner
            .iter()
            .filter(|(k, _)| *k == lower)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    pub fn remove(&mut self, name: &str) {
        let lower = name.to_ascii_lowercase();
        self.inner.retain(|(k, _)| *k != lower);
    }

    pub fn contains(&self, name: &str) -> bool {
        let lower = name.to_ascii_lowercase();
        self.inner.iter().any(|(k, _)| *k == lower)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

// ── Query string builder ───────────────────────────────────────

/// URL query string builder with percent-encoding.
#[derive(Debug, Clone, Default)]
pub struct QueryString {
    params: Vec<(String, String)>,
}

impl QueryString {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn param(mut self, key: &str, value: &str) -> Self {
        self.params.push((key.to_string(), value.to_string()));
        self
    }

    pub fn encode(&self) -> String {
        self.params
            .iter()
            .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&")
    }

    pub fn parse(qs: &str) -> Self {
        let s = qs.strip_prefix('?').unwrap_or(qs);
        let params: Vec<(String, String)> = s
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let key = percent_decode(parts.next().unwrap_or(""));
                let val = percent_decode(parts.next().unwrap_or(""));
                (key, val)
            })
            .collect();
        Self { params }
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push(char::from(HEX[(b >> 4) as usize]));
                out.push(char::from(HEX[(b & 0x0f) as usize]));
            }
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (from_hex(bytes[i + 1]), from_hex(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

const HEX: [u8; 16] = *b"0123456789ABCDEF";

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'A'..=b'F' => Some(b - b'A' + 10),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

// ── Redirect policy ────────────────────────────────────────────

/// Redirect following policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedirectPolicy {
    /// Never follow redirects.
    None,
    /// Follow up to N redirects.
    Limited(u32),
    /// Follow same-origin redirects only.
    SameOrigin,
}

impl Default for RedirectPolicy {
    fn default() -> Self {
        Self::Limited(10)
    }
}

impl RedirectPolicy {
    /// Check whether a redirect should be followed given current count and origin info.
    pub fn should_follow(&self, count: u32, same_origin: bool) -> bool {
        match self {
            Self::None => false,
            Self::Limited(max) => count < *max,
            Self::SameOrigin => same_origin,
        }
    }
}

// ── Timeout config ─────────────────────────────────────────────

/// Timeout configuration.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub connect_ms: u64,
    pub read_ms: u64,
    pub write_ms: u64,
    pub total_ms: u64,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect_ms: 10_000,
            read_ms: 30_000,
            write_ms: 30_000,
            total_ms: 60_000,
        }
    }
}

// ── Request interceptor ────────────────────────────────────────

/// A request interceptor that can modify a request before it is sent.
#[derive(Clone)]
pub struct Interceptor {
    pub name: String,
    pub headers: HashMap<String, String>,
}

impl Interceptor {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            headers: HashMap::new(),
        }
    }

    pub fn add_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn apply(&self, req: &mut Request) {
        for (k, v) in &self.headers {
            req.headers.insert(k, v);
        }
    }
}

impl fmt::Debug for Interceptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interceptor")
            .field("name", &self.name)
            .field("headers", &self.headers)
            .finish()
    }
}

// ── Request ────────────────────────────────────────────────────

/// An HTTP request (builder pattern).
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub query: QueryString,
    pub body: Option<Vec<u8>>,
    pub timeout: TimeoutConfig,
    pub redirect: RedirectPolicy,
    pub cookies: Vec<String>,
}

impl Request {
    pub fn new(method: Method, url: &str) -> Self {
        Self {
            method,
            url: url.to_string(),
            headers: HeaderMap::new(),
            query: QueryString::new(),
            body: None,
            timeout: TimeoutConfig::default(),
            redirect: RedirectPolicy::default(),
            cookies: Vec::new(),
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
    pub fn delete(url: &str) -> Self {
        Self::new(Method::Delete, url)
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn query(mut self, key: &str, value: &str) -> Self {
        self.query = self.query.param(key, value);
        self
    }

    pub fn body_bytes(mut self, data: Vec<u8>) -> Self {
        self.body = Some(data);
        self
    }

    pub fn body_text(mut self, text: &str) -> Self {
        self.body = Some(text.as_bytes().to_vec());
        self
    }

    pub fn json_body(mut self, json: &str) -> Self {
        self.headers.insert("content-type", "application/json");
        self.body = Some(json.as_bytes().to_vec());
        self
    }

    pub fn timeout_config(mut self, t: TimeoutConfig) -> Self {
        self.timeout = t;
        self
    }

    pub fn redirect_policy(mut self, p: RedirectPolicy) -> Self {
        self.redirect = p;
        self
    }

    pub fn cookie(mut self, c: &str) -> Self {
        self.cookies.push(c.to_string());
        self
    }

    /// Build the full URL including query string.
    pub fn full_url(&self) -> String {
        let qs = self.query.encode();
        if qs.is_empty() {
            self.url.clone()
        } else if self.url.contains('?') {
            format!("{}&{}", self.url, qs)
        } else {
            format!("{}?{}", self.url, qs)
        }
    }
}

// ── Response ───────────────────────────────────────────────────

/// An HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: HeaderMap::new(),
            body: Vec::new(),
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn body(mut self, data: Vec<u8>) -> Self {
        self.body = data;
        self
    }

    pub fn body_text(mut self, text: &str) -> Self {
        self.body = text.as_bytes().to_vec();
        self
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn is_redirect(&self) -> bool {
        (300..400).contains(&self.status)
    }

    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.status)
    }

    pub fn is_server_error(&self) -> bool {
        (500..600).contains(&self.status)
    }

    pub fn content_length(&self) -> usize {
        self.body.len()
    }
}

// ── Client builder ─────────────────────────────────────────────

/// HTTP client builder: accumulates interceptors, defaults, base URL.
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    pub base_url: Option<String>,
    pub default_headers: HeaderMap,
    pub timeout: TimeoutConfig,
    pub redirect: RedirectPolicy,
    pub interceptors: Vec<Interceptor>,
    pub max_retries: u32,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            base_url: None,
            default_headers: HeaderMap::new(),
            timeout: TimeoutConfig::default(),
            redirect: RedirectPolicy::default(),
            interceptors: Vec::new(),
            max_retries: 0,
        }
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = Some(url.trim_end_matches('/').to_string());
        self
    }

    pub fn default_header(mut self, name: &str, value: &str) -> Self {
        self.default_headers.insert(name, value);
        self
    }

    pub fn timeout(mut self, t: TimeoutConfig) -> Self {
        self.timeout = t;
        self
    }

    pub fn redirect(mut self, p: RedirectPolicy) -> Self {
        self.redirect = p;
        self
    }

    pub fn interceptor(mut self, i: Interceptor) -> Self {
        self.interceptors.push(i);
        self
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Prepare a request: resolve base URL, apply interceptors and defaults.
    pub fn prepare(&self, mut req: Request) -> Request {
        // Resolve base URL.
        if let Some(base) = &self.base_url {
            if req.url.starts_with('/') {
                req.url = format!("{}{}", base, req.url);
            }
        }
        // Apply default headers (don't override explicit ones).
        for (k, v) in self.default_headers.iter() {
            if !req.headers.contains(k) {
                req.headers.insert(k, v);
            }
        }
        // Apply interceptors.
        for interceptor in &self.interceptors {
            interceptor.apply(&mut req);
        }
        req
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_roundtrip() {
        for m in [
            Method::Get,
            Method::Post,
            Method::Put,
            Method::Patch,
            Method::Delete,
            Method::Head,
            Method::Options,
        ] {
            let s = m.as_str();
            assert_eq!(Method::from_str(s), Some(m));
        }
    }

    #[test]
    fn method_safety() {
        assert!(Method::Get.is_safe());
        assert!(!Method::Post.is_safe());
        assert!(Method::Put.is_idempotent());
        assert!(!Method::Post.is_idempotent());
    }

    #[test]
    fn header_map_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert("Content-Type", "text/plain");
        assert_eq!(h.get("content-type"), Some("text/plain"));
        assert_eq!(h.get("CONTENT-TYPE"), Some("text/plain"));
        h.insert("content-type", "application/json");
        assert_eq!(h.get("Content-Type"), Some("application/json"));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn header_map_append_multi() {
        let mut h = HeaderMap::new();
        h.append("Set-Cookie", "a=1");
        h.append("Set-Cookie", "b=2");
        let vals = h.get_all("set-cookie");
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], "a=1");
    }

    #[test]
    fn query_string_build() {
        let qs = QueryString::new()
            .param("q", "hello world")
            .param("page", "1");
        assert_eq!(qs.encode(), "q=hello%20world&page=1");
    }

    #[test]
    fn query_string_parse() {
        let qs = QueryString::parse("?foo=bar&baz=qux%20x");
        assert_eq!(qs.len(), 2);
        let encoded = qs.encode();
        assert!(encoded.contains("foo=bar"));
    }

    #[test]
    fn redirect_policy_limited() {
        let p = RedirectPolicy::Limited(3);
        assert!(p.should_follow(0, true));
        assert!(p.should_follow(2, false));
        assert!(!p.should_follow(3, true));
    }

    #[test]
    fn redirect_policy_same_origin() {
        let p = RedirectPolicy::SameOrigin;
        assert!(p.should_follow(100, true));
        assert!(!p.should_follow(0, false));
    }

    #[test]
    fn request_builder_full_url() {
        let req = Request::get("https://api.example.com/v1/items")
            .query("page", "2")
            .query("limit", "10");
        assert_eq!(
            req.full_url(),
            "https://api.example.com/v1/items?page=2&limit=10"
        );
    }

    #[test]
    fn request_json_body() {
        let req = Request::post("https://api.example.com/data")
            .json_body(r#"{"key":"value"}"#);
        assert_eq!(
            req.headers.get("content-type"),
            Some("application/json")
        );
        assert!(req.body.is_some());
    }

    #[test]
    fn response_status_classification() {
        assert!(Response::new(200).is_success());
        assert!(Response::new(301).is_redirect());
        assert!(Response::new(404).is_client_error());
        assert!(Response::new(503).is_server_error());
    }

    #[test]
    fn client_builder_prepare() {
        let client = ClientBuilder::new()
            .base_url("https://api.example.com")
            .default_header("Authorization", "Bearer token")
            .interceptor(
                Interceptor::new("trace")
                    .add_header("X-Trace-Id", "abc123"),
            );

        let req = client.prepare(Request::get("/users"));
        assert_eq!(req.url, "https://api.example.com/users");
        assert_eq!(req.headers.get("authorization"), Some("Bearer token"));
        assert_eq!(req.headers.get("x-trace-id"), Some("abc123"));
    }

    #[test]
    fn client_builder_no_override_explicit_headers() {
        let client = ClientBuilder::new()
            .default_header("Accept", "application/xml");
        let req = client.prepare(
            Request::get("https://example.com").header("Accept", "text/html"),
        );
        assert_eq!(req.headers.get("accept"), Some("text/html"));
    }

    #[test]
    fn percent_encode_special_chars() {
        let encoded = percent_encode("a b&c=d");
        assert_eq!(encoded, "a%20b%26c%3Dd");
        let decoded = percent_decode(&encoded);
        assert_eq!(decoded, "a b&c=d");
    }

    #[test]
    fn cookie_on_request() {
        let req = Request::get("https://example.com")
            .cookie("session=abc123")
            .cookie("theme=dark");
        assert_eq!(req.cookies.len(), 2);
    }
}
