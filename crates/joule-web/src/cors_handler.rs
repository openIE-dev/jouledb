//! CORS policy engine — origin allowlist/denylist, method/header whitelisting,
//! preflight response generation, max-age configuration, credentials support,
//! wildcard vs specific origin handling.
//!
//! Replaces `cors`, `koa-cors`, `express-cors` with a comprehensive pure-Rust
//! CORS policy evaluator and preflight response builder.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ── Errors ─────────────────────────────────────────────────────

/// CORS handler errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorsHandlerError {
    /// Origin not allowed.
    OriginDenied(String),
    /// Method not allowed.
    MethodNotAllowed(String),
    /// Header not allowed.
    HeaderNotAllowed(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for CorsHandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OriginDenied(o) => write!(f, "origin denied: {o}"),
            Self::MethodNotAllowed(m) => write!(f, "method not allowed: {m}"),
            Self::HeaderNotAllowed(h) => write!(f, "header not allowed: {h}"),
            Self::InvalidConfig(s) => write!(f, "invalid CORS config: {s}"),
        }
    }
}

impl std::error::Error for CorsHandlerError {}

// ── Types ──────────────────────────────────────────────────────

/// HTTP methods for CORS.
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

impl HttpMethod {
    /// Convert to uppercase string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str_ci(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "PATCH" => Some(Self::Patch),
            "DELETE" => Some(Self::Delete),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            _ => None,
        }
    }
}

/// Origin matching mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginPolicy {
    /// Allow any origin (Access-Control-Allow-Origin: *).
    Any,
    /// Allow specific origins only.
    Allowlist(Vec<String>),
    /// Deny specific origins, allow all others.
    Denylist(Vec<String>),
    /// Allow origins matching a prefix pattern (e.g., "https://*.example.com").
    Pattern(Vec<String>),
}

// ── CORS Policy ────────────────────────────────────────────────

/// CORS policy configuration.
#[derive(Debug, Clone)]
pub struct CorsPolicy {
    /// Origin matching policy.
    pub origin_policy: OriginPolicy,
    /// Allowed HTTP methods.
    pub allowed_methods: HashSet<HttpMethod>,
    /// Allowed request headers (lowercase).
    pub allowed_headers: HashSet<String>,
    /// Headers exposed to the browser.
    pub expose_headers: Vec<String>,
    /// Preflight cache max-age in seconds.
    pub max_age_secs: Option<u32>,
    /// Whether to allow credentials (cookies, auth headers).
    pub allow_credentials: bool,
}

impl Default for CorsPolicy {
    fn default() -> Self {
        let mut methods = HashSet::new();
        methods.insert(HttpMethod::Get);
        methods.insert(HttpMethod::Post);
        methods.insert(HttpMethod::Options);

        let mut headers = HashSet::new();
        headers.insert("content-type".to_string());
        headers.insert("accept".to_string());
        headers.insert("authorization".to_string());

        Self {
            origin_policy: OriginPolicy::Any,
            allowed_methods: methods,
            allowed_headers: headers,
            expose_headers: Vec::new(),
            max_age_secs: Some(86400),
            allow_credentials: false,
        }
    }
}

impl CorsPolicy {
    /// Create a restrictive policy (no origins allowed by default).
    pub fn restrictive() -> Self {
        Self {
            origin_policy: OriginPolicy::Allowlist(Vec::new()),
            allowed_methods: HashSet::new(),
            allowed_headers: HashSet::new(),
            expose_headers: Vec::new(),
            max_age_secs: None,
            allow_credentials: false,
        }
    }

    /// Set origin policy.
    pub fn with_origins(mut self, policy: OriginPolicy) -> Self {
        self.origin_policy = policy;
        self
    }

    /// Add an allowed method.
    pub fn allow_method(mut self, method: HttpMethod) -> Self {
        self.allowed_methods.insert(method);
        self
    }

    /// Add an allowed header (stored lowercase).
    pub fn allow_header(mut self, header: &str) -> Self {
        self.allowed_headers.insert(header.to_lowercase());
        self
    }

    /// Set exposed headers.
    pub fn expose(mut self, headers: Vec<String>) -> Self {
        self.expose_headers = headers;
        self
    }

    /// Set preflight max-age.
    pub fn with_max_age(mut self, secs: u32) -> Self {
        self.max_age_secs = Some(secs);
        self
    }

    /// Enable credentials.
    pub fn with_credentials(mut self) -> Self {
        self.allow_credentials = true;
        self
    }
}

// ── CORS Response Headers ──────────────────────────────────────

/// CORS response headers to include in HTTP responses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorsHeaders {
    pub headers: Vec<(String, String)>,
}

impl CorsHeaders {
    fn new() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    fn add(&mut self, key: &str, value: &str) {
        self.headers
            .push((key.to_string(), value.to_string()));
    }

    /// Get a header value by name.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Check if a header is present.
    pub fn has(&self, key: &str) -> bool {
        self.headers.iter().any(|(k, _)| k == key)
    }

    /// Number of headers.
    pub fn len(&self) -> usize {
        self.headers.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }
}

// ── CORS Handler ───────────────────────────────────────────────

/// CORS policy evaluator and response builder.
pub struct CorsHandler {
    policy: CorsPolicy,
}

impl CorsHandler {
    /// Create a handler with the given policy.
    pub fn new(policy: CorsPolicy) -> Self {
        Self { policy }
    }

    /// Create a handler with the default (permissive) policy.
    pub fn permissive() -> Self {
        Self {
            policy: CorsPolicy::default(),
        }
    }

    /// Check whether an origin is allowed.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        match &self.policy.origin_policy {
            OriginPolicy::Any => true,
            OriginPolicy::Allowlist(list) => list.iter().any(|o| o == origin),
            OriginPolicy::Denylist(list) => !list.iter().any(|o| o == origin),
            OriginPolicy::Pattern(patterns) => {
                patterns.iter().any(|p| origin_matches_pattern(origin, p))
            }
        }
    }

    /// Check whether a method is allowed.
    pub fn is_method_allowed(&self, method: &str) -> bool {
        if let Some(m) = HttpMethod::from_str_ci(method) {
            self.policy.allowed_methods.contains(&m)
        } else {
            false
        }
    }

    /// Check whether a request header is allowed.
    pub fn is_header_allowed(&self, header: &str) -> bool {
        self.policy.allowed_headers.contains(&header.to_lowercase())
    }

    /// Evaluate a simple (non-preflight) CORS request.
    pub fn evaluate_simple(
        &self,
        origin: &str,
    ) -> Result<CorsHeaders, CorsHandlerError> {
        if !self.is_origin_allowed(origin) {
            return Err(CorsHandlerError::OriginDenied(origin.to_string()));
        }

        let mut headers = CorsHeaders::new();
        self.add_origin_header(&mut headers, origin);
        self.add_credentials_header(&mut headers);
        self.add_expose_headers(&mut headers);
        Ok(headers)
    }

    /// Evaluate a preflight (OPTIONS) request.
    pub fn evaluate_preflight(
        &self,
        origin: &str,
        request_method: &str,
        request_headers: &[&str],
    ) -> Result<CorsHeaders, CorsHandlerError> {
        if !self.is_origin_allowed(origin) {
            return Err(CorsHandlerError::OriginDenied(origin.to_string()));
        }

        if !self.is_method_allowed(request_method) {
            return Err(CorsHandlerError::MethodNotAllowed(
                request_method.to_string(),
            ));
        }

        for header in request_headers {
            if !self.is_header_allowed(header) {
                return Err(CorsHandlerError::HeaderNotAllowed(header.to_string()));
            }
        }

        let mut headers = CorsHeaders::new();
        self.add_origin_header(&mut headers, origin);
        self.add_credentials_header(&mut headers);

        // Allowed methods.
        let methods: Vec<&str> = self
            .policy
            .allowed_methods
            .iter()
            .map(|m| m.as_str())
            .collect();
        let mut methods_sorted = methods;
        methods_sorted.sort();
        headers.add(
            "Access-Control-Allow-Methods",
            &methods_sorted.join(", "),
        );

        // Allowed headers.
        let mut hdrs: Vec<&str> = self
            .policy
            .allowed_headers
            .iter()
            .map(|h| h.as_str())
            .collect();
        hdrs.sort();
        headers.add("Access-Control-Allow-Headers", &hdrs.join(", "));

        // Max-Age.
        if let Some(max_age) = self.policy.max_age_secs {
            headers.add("Access-Control-Max-Age", &max_age.to_string());
        }

        Ok(headers)
    }

    /// Validate a full CORS request (origin + method + headers).
    pub fn validate_request(
        &self,
        origin: &str,
        method: &str,
        headers_list: &[&str],
    ) -> Result<(), CorsHandlerError> {
        if !self.is_origin_allowed(origin) {
            return Err(CorsHandlerError::OriginDenied(origin.to_string()));
        }
        if !self.is_method_allowed(method) {
            return Err(CorsHandlerError::MethodNotAllowed(method.to_string()));
        }
        for h in headers_list {
            if !self.is_header_allowed(h) {
                return Err(CorsHandlerError::HeaderNotAllowed(h.to_string()));
            }
        }
        Ok(())
    }

    /// Get a reference to the policy.
    pub fn policy(&self) -> &CorsPolicy {
        &self.policy
    }

    // ── Internal helpers ──

    fn add_origin_header(&self, headers: &mut CorsHeaders, origin: &str) {
        match &self.policy.origin_policy {
            OriginPolicy::Any if !self.policy.allow_credentials => {
                headers.add("Access-Control-Allow-Origin", "*");
            }
            _ => {
                headers.add("Access-Control-Allow-Origin", origin);
                headers.add("Vary", "Origin");
            }
        }
    }

    fn add_credentials_header(&self, headers: &mut CorsHeaders) {
        if self.policy.allow_credentials {
            headers.add("Access-Control-Allow-Credentials", "true");
        }
    }

    fn add_expose_headers(&self, headers: &mut CorsHeaders) {
        if !self.policy.expose_headers.is_empty() {
            let mut sorted = self.policy.expose_headers.clone();
            sorted.sort();
            headers.add(
                "Access-Control-Expose-Headers",
                &sorted.join(", "),
            );
        }
    }
}

/// Match an origin against a wildcard pattern.
///
/// Supports `*` as a single-segment wildcard in the hostname portion.
/// E.g., "https://*.example.com" matches "https://sub.example.com".
fn origin_matches_pattern(origin: &str, pattern: &str) -> bool {
    if !pattern.contains('*') {
        return origin == pattern;
    }

    // Split into scheme+host portions.
    let pattern_parts: Vec<&str> = pattern.splitn(2, "://").collect();
    let origin_parts: Vec<&str> = origin.splitn(2, "://").collect();

    if pattern_parts.len() != 2 || origin_parts.len() != 2 {
        return false;
    }

    // Schemes must match.
    if pattern_parts[0] != origin_parts[0] {
        return false;
    }

    let pattern_host = pattern_parts[1];
    let origin_host = origin_parts[1];

    // Simple wildcard: "*.example.com" matches "sub.example.com".
    if let Some(suffix) = pattern_host.strip_prefix("*") {
        return origin_host.ends_with(suffix) && origin_host.len() > suffix.len();
    }

    false
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn handler_with_allowlist(origins: Vec<&str>) -> CorsHandler {
        let policy = CorsPolicy::default().with_origins(OriginPolicy::Allowlist(
            origins.into_iter().map(|s| s.to_string()).collect(),
        ));
        CorsHandler::new(policy)
    }

    #[test]
    fn test_any_origin_allowed() {
        let h = CorsHandler::permissive();
        assert!(h.is_origin_allowed("https://example.com"));
        assert!(h.is_origin_allowed("http://evil.com"));
    }

    #[test]
    fn test_allowlist_origin() {
        let h = handler_with_allowlist(vec!["https://example.com"]);
        assert!(h.is_origin_allowed("https://example.com"));
        assert!(!h.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_denylist_origin() {
        let policy = CorsPolicy::default().with_origins(OriginPolicy::Denylist(vec![
            "https://evil.com".to_string(),
        ]));
        let h = CorsHandler::new(policy);
        assert!(h.is_origin_allowed("https://example.com"));
        assert!(!h.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_pattern_origin() {
        let policy = CorsPolicy::default().with_origins(OriginPolicy::Pattern(vec![
            "https://*.example.com".to_string(),
        ]));
        let h = CorsHandler::new(policy);
        assert!(h.is_origin_allowed("https://sub.example.com"));
        assert!(h.is_origin_allowed("https://api.example.com"));
        assert!(!h.is_origin_allowed("https://example.com"));
        assert!(!h.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_method_allowed() {
        let h = CorsHandler::permissive();
        assert!(h.is_method_allowed("GET"));
        assert!(h.is_method_allowed("get"));
        assert!(h.is_method_allowed("POST"));
        assert!(!h.is_method_allowed("DELETE")); // not in default
    }

    #[test]
    fn test_header_allowed() {
        let h = CorsHandler::permissive();
        assert!(h.is_header_allowed("Content-Type"));
        assert!(h.is_header_allowed("authorization"));
        assert!(!h.is_header_allowed("X-Custom"));
    }

    #[test]
    fn test_simple_request_allowed() {
        let h = CorsHandler::permissive();
        let result = h.evaluate_simple("https://example.com");
        assert!(result.is_ok());
        let headers = result.unwrap();
        assert_eq!(
            headers.get("Access-Control-Allow-Origin"),
            Some("*")
        );
    }

    #[test]
    fn test_simple_request_denied() {
        let h = handler_with_allowlist(vec!["https://ok.com"]);
        let result = h.evaluate_simple("https://evil.com");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_request_specific_origin() {
        let h = handler_with_allowlist(vec!["https://example.com"]);
        let headers = h.evaluate_simple("https://example.com").unwrap();
        assert_eq!(
            headers.get("Access-Control-Allow-Origin"),
            Some("https://example.com")
        );
        assert_eq!(headers.get("Vary"), Some("Origin"));
    }

    #[test]
    fn test_preflight_allowed() {
        let policy = CorsPolicy::default()
            .allow_method(HttpMethod::Put)
            .allow_header("x-custom");
        let h = CorsHandler::new(policy);
        let result = h.evaluate_preflight(
            "https://example.com",
            "PUT",
            &["x-custom"],
        );
        assert!(result.is_ok());
        let headers = result.unwrap();
        assert!(headers.has("Access-Control-Allow-Methods"));
        assert!(headers.has("Access-Control-Allow-Headers"));
    }

    #[test]
    fn test_preflight_method_denied() {
        let h = CorsHandler::permissive();
        let result =
            h.evaluate_preflight("https://example.com", "DELETE", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            CorsHandlerError::MethodNotAllowed(m) => assert_eq!(m, "DELETE"),
            _ => panic!("expected MethodNotAllowed"),
        }
    }

    #[test]
    fn test_preflight_header_denied() {
        let h = CorsHandler::permissive();
        let result = h.evaluate_preflight(
            "https://example.com",
            "POST",
            &["x-forbidden"],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_preflight_max_age() {
        let policy = CorsPolicy::default().with_max_age(600);
        let h = CorsHandler::new(policy);
        let headers = h
            .evaluate_preflight("https://example.com", "GET", &[])
            .unwrap();
        assert_eq!(headers.get("Access-Control-Max-Age"), Some("600"));
    }

    #[test]
    fn test_credentials() {
        let policy = CorsPolicy::default().with_credentials();
        let h = CorsHandler::new(policy);
        let headers = h.evaluate_simple("https://example.com").unwrap();
        assert_eq!(
            headers.get("Access-Control-Allow-Credentials"),
            Some("true")
        );
        // With credentials, origin must be specific, not *.
        assert_ne!(
            headers.get("Access-Control-Allow-Origin"),
            Some("*")
        );
    }

    #[test]
    fn test_expose_headers() {
        let policy =
            CorsPolicy::default().expose(vec!["X-Request-Id".to_string()]);
        let h = CorsHandler::new(policy);
        let headers = h.evaluate_simple("https://example.com").unwrap();
        assert!(headers.has("Access-Control-Expose-Headers"));
    }

    #[test]
    fn test_validate_request_full() {
        let policy = CorsPolicy::default()
            .with_origins(OriginPolicy::Allowlist(vec![
                "https://ok.com".to_string(),
            ]))
            .allow_method(HttpMethod::Put);
        let h = CorsHandler::new(policy);
        assert!(h
            .validate_request("https://ok.com", "PUT", &["content-type"])
            .is_ok());
        assert!(h
            .validate_request("https://evil.com", "GET", &[])
            .is_err());
    }

    #[test]
    fn test_restrictive_policy() {
        let h = CorsHandler::new(CorsPolicy::restrictive());
        assert!(!h.is_origin_allowed("https://example.com"));
        assert!(!h.is_method_allowed("GET"));
        assert!(!h.is_header_allowed("content-type"));
    }

    #[test]
    fn test_http_method_roundtrip() {
        for method in &[
            HttpMethod::Get,
            HttpMethod::Post,
            HttpMethod::Put,
            HttpMethod::Patch,
            HttpMethod::Delete,
            HttpMethod::Head,
            HttpMethod::Options,
        ] {
            let s = method.as_str();
            let parsed = HttpMethod::from_str_ci(s).unwrap();
            assert_eq!(&parsed, method);
        }
        assert!(HttpMethod::from_str_ci("TRACE").is_none());
    }

    #[test]
    fn test_origin_pattern_scheme_mismatch() {
        assert!(!origin_matches_pattern(
            "http://sub.example.com",
            "https://*.example.com"
        ));
    }

    #[test]
    fn test_cors_headers_empty() {
        let h = CorsHeaders::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert!(h.get("X-Missing").is_none());
    }

    #[test]
    fn test_error_display() {
        let e = CorsHandlerError::OriginDenied("https://evil.com".to_string());
        assert!(e.to_string().contains("evil.com"));
    }
}
