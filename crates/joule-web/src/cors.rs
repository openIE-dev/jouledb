//! CORS policy engine — evaluate cross-origin requests against configurable rules.
//!
//! Replaces `cors`, `koa-cors`, and `express-cors` with a pure-Rust policy
//! evaluator.  Supports origin matching (exact, wildcard, regex patterns),
//! allowed methods/headers, preflight cache, credential rules, expose headers,
//! and max-age.

use std::collections::HashMap;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────

/// HTTP method for CORS matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CorsMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl CorsMethod {
    pub fn as_str(&self) -> &str {
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

    pub fn from_str(s: &str) -> Option<Self> {
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

impl fmt::Display for CorsMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Origin pattern ─────────────────────────────────────────────

/// Origin matching pattern.
#[derive(Debug, Clone)]
pub enum OriginPattern {
    /// Match any origin.
    Any,
    /// Exact match.
    Exact(String),
    /// Suffix wildcard: `*.example.com` matches `sub.example.com`.
    Wildcard(String),
    /// Simple regex-like pattern — supports `*` as wildcard.
    Pattern(String),
}

impl OriginPattern {
    pub fn matches(&self, origin: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(s) => s == origin,
            Self::Wildcard(suffix) => {
                // suffix is like ".example.com"
                origin.ends_with(suffix.as_str())
            }
            Self::Pattern(pat) => simple_pattern_match(pat, origin),
        }
    }
}

/// Simple glob-style matching: `*` matches any sequence of chars.
fn simple_pattern_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if let Some(idx) = text[pos..].find(part) {
            if i == 0 && idx != 0 {
                return false; // Must match from start if no leading *.
            }
            pos += idx + part.len();
        } else {
            return false;
        }
    }
    // If pattern doesn't end with *, text must end exactly.
    if !pattern.ends_with('*') {
        return pos == text.len();
    }
    true
}

// ── Preflight cache ────────────────────────────────────────────

/// Cached preflight result.
#[derive(Debug, Clone)]
pub struct PreflightEntry {
    pub origin: String,
    pub method: String,
    pub headers: Vec<String>,
    pub max_age: u64,
    pub created_at_epoch_s: u64,
}

impl PreflightEntry {
    pub fn is_valid(&self, now_epoch_s: u64) -> bool {
        now_epoch_s < self.created_at_epoch_s + self.max_age
    }
}

/// Preflight cache keyed by (origin, method, sorted headers).
#[derive(Debug, Default)]
pub struct PreflightCache {
    entries: HashMap<String, PreflightEntry>,
}

impl PreflightCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn cache_key(origin: &str, method: &str, headers: &[String]) -> String {
        let mut sorted = headers.to_vec();
        sorted.sort();
        format!("{}|{}|{}", origin, method, sorted.join(","))
    }

    pub fn insert(&mut self, entry: PreflightEntry) {
        let key = Self::cache_key(&entry.origin, &entry.method, &entry.headers);
        self.entries.insert(key, entry);
    }

    pub fn lookup(
        &self,
        origin: &str,
        method: &str,
        headers: &[String],
        now_epoch_s: u64,
    ) -> Option<&PreflightEntry> {
        let key = Self::cache_key(origin, method, headers);
        self.entries
            .get(&key)
            .filter(|e| e.is_valid(now_epoch_s))
    }

    pub fn evict_expired(&mut self, now_epoch_s: u64) {
        self.entries.retain(|_, e| e.is_valid(now_epoch_s));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── CORS policy ────────────────────────────────────────────────

/// A CORS policy configuration.
#[derive(Debug, Clone)]
pub struct CorsPolicy {
    pub allowed_origins: Vec<OriginPattern>,
    pub allowed_methods: Vec<CorsMethod>,
    pub allowed_headers: Vec<String>,
    pub expose_headers: Vec<String>,
    pub max_age_seconds: u64,
    pub allow_credentials: bool,
}

impl Default for CorsPolicy {
    fn default() -> Self {
        Self {
            allowed_origins: vec![OriginPattern::Any],
            allowed_methods: vec![CorsMethod::Get, CorsMethod::Post, CorsMethod::Options],
            allowed_headers: vec![
                "content-type".to_string(),
                "authorization".to_string(),
            ],
            expose_headers: Vec::new(),
            max_age_seconds: 86400,
            allow_credentials: false,
        }
    }
}

/// The result of evaluating a CORS request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorsResult {
    /// Request is allowed.
    Allowed,
    /// Preflight response should be returned.
    Preflight,
    /// Origin not allowed.
    OriginDenied,
    /// Method not allowed.
    MethodDenied,
    /// Header not allowed.
    HeaderDenied(String),
    /// Credentials requested but not allowed.
    CredentialsDenied,
}

/// Represents an incoming cross-origin request.
#[derive(Debug, Clone)]
pub struct CorsRequest {
    pub origin: String,
    pub method: String,
    pub is_preflight: bool,
    pub request_headers: Vec<String>,
    pub with_credentials: bool,
}

impl CorsPolicy {
    pub fn allow_origin(mut self, pattern: OriginPattern) -> Self {
        self.allowed_origins.push(pattern);
        self
    }

    pub fn allow_method(mut self, method: CorsMethod) -> Self {
        self.allowed_methods.push(method);
        self
    }

    pub fn allow_header(mut self, header: &str) -> Self {
        self.allowed_headers
            .push(header.to_ascii_lowercase());
        self
    }

    pub fn expose_header(mut self, header: &str) -> Self {
        self.expose_headers.push(header.to_string());
        self
    }

    pub fn credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }

    pub fn max_age(mut self, seconds: u64) -> Self {
        self.max_age_seconds = seconds;
        self
    }

    /// Evaluate whether a request passes the CORS policy.
    pub fn evaluate(&self, req: &CorsRequest) -> CorsResult {
        // 1. Check origin.
        let origin_ok = self
            .allowed_origins
            .iter()
            .any(|p| p.matches(&req.origin));
        if !origin_ok {
            return CorsResult::OriginDenied;
        }

        // 2. Credentials check.
        if req.with_credentials && !self.allow_credentials {
            return CorsResult::CredentialsDenied;
        }

        // 3. Method check.
        let method_ok = self.allowed_methods.iter().any(|m| {
            m.as_str().eq_ignore_ascii_case(&req.method)
        });
        if !method_ok {
            return CorsResult::MethodDenied;
        }

        // 4. Header check (only for preflight).
        if req.is_preflight {
            for header in &req.request_headers {
                let lower = header.to_ascii_lowercase();
                let ok = self
                    .allowed_headers
                    .iter()
                    .any(|h| h.eq_ignore_ascii_case(&lower));
                if !ok {
                    return CorsResult::HeaderDenied(header.clone());
                }
            }
            return CorsResult::Preflight;
        }

        CorsResult::Allowed
    }

    /// Build response headers for an allowed request.
    pub fn response_headers(&self, origin: &str) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        // Use specific origin if credentials are allowed (wildcard not allowed with credentials).
        if self.allow_credentials {
            headers.push((
                "Access-Control-Allow-Origin".to_string(),
                origin.to_string(),
            ));
            headers.push((
                "Access-Control-Allow-Credentials".to_string(),
                "true".to_string(),
            ));
        } else {
            let has_any = self
                .allowed_origins
                .iter()
                .any(|p| matches!(p, OriginPattern::Any));
            if has_any {
                headers.push((
                    "Access-Control-Allow-Origin".to_string(),
                    "*".to_string(),
                ));
            } else {
                headers.push((
                    "Access-Control-Allow-Origin".to_string(),
                    origin.to_string(),
                ));
            }
        }

        let methods: Vec<&str> = self.allowed_methods.iter().map(|m| m.as_str()).collect();
        headers.push((
            "Access-Control-Allow-Methods".to_string(),
            methods.join(", "),
        ));

        if !self.allowed_headers.is_empty() {
            headers.push((
                "Access-Control-Allow-Headers".to_string(),
                self.allowed_headers.join(", "),
            ));
        }

        if !self.expose_headers.is_empty() {
            headers.push((
                "Access-Control-Expose-Headers".to_string(),
                self.expose_headers.join(", "),
            ));
        }

        headers.push((
            "Access-Control-Max-Age".to_string(),
            self.max_age_seconds.to_string(),
        ));

        headers
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_origin_match() {
        let policy = CorsPolicy {
            allowed_origins: vec![OriginPattern::Exact(
                "https://example.com".to_string(),
            )],
            ..Default::default()
        };
        let req = CorsRequest {
            origin: "https://example.com".to_string(),
            method: "GET".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: false,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::Allowed);
    }

    #[test]
    fn origin_denied() {
        let policy = CorsPolicy {
            allowed_origins: vec![OriginPattern::Exact("https://a.com".to_string())],
            ..Default::default()
        };
        let req = CorsRequest {
            origin: "https://evil.com".to_string(),
            method: "GET".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: false,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::OriginDenied);
    }

    #[test]
    fn wildcard_origin() {
        let policy = CorsPolicy {
            allowed_origins: vec![OriginPattern::Wildcard(".example.com".to_string())],
            ..Default::default()
        };
        let req = CorsRequest {
            origin: "https://sub.example.com".to_string(),
            method: "GET".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: false,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::Allowed);
    }

    #[test]
    fn pattern_origin() {
        let pat = OriginPattern::Pattern("https://*.example.*".to_string());
        assert!(pat.matches("https://api.example.com"));
        assert!(pat.matches("https://web.example.org"));
        assert!(!pat.matches("http://api.example.com"));
    }

    #[test]
    fn method_denied() {
        let policy = CorsPolicy {
            allowed_methods: vec![CorsMethod::Get],
            ..Default::default()
        };
        let req = CorsRequest {
            origin: "https://any.com".to_string(),
            method: "DELETE".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: false,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::MethodDenied);
    }

    #[test]
    fn preflight_header_denied() {
        let policy = CorsPolicy::default();
        let req = CorsRequest {
            origin: "https://any.com".to_string(),
            method: "POST".to_string(),
            is_preflight: true,
            request_headers: vec!["X-Custom-Header".to_string()],
            with_credentials: false,
        };
        assert!(matches!(
            policy.evaluate(&req),
            CorsResult::HeaderDenied(_)
        ));
    }

    #[test]
    fn preflight_allowed() {
        let policy = CorsPolicy::default().allow_header("x-custom");
        let req = CorsRequest {
            origin: "https://any.com".to_string(),
            method: "POST".to_string(),
            is_preflight: true,
            request_headers: vec!["X-Custom".to_string()],
            with_credentials: false,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::Preflight);
    }

    #[test]
    fn credentials_denied() {
        let policy = CorsPolicy::default(); // allow_credentials = false
        let req = CorsRequest {
            origin: "https://any.com".to_string(),
            method: "GET".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: true,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::CredentialsDenied);
    }

    #[test]
    fn credentials_allowed() {
        let policy = CorsPolicy::default().credentials(true);
        let req = CorsRequest {
            origin: "https://any.com".to_string(),
            method: "GET".to_string(),
            is_preflight: false,
            request_headers: vec![],
            with_credentials: true,
        };
        assert_eq!(policy.evaluate(&req), CorsResult::Allowed);
    }

    #[test]
    fn response_headers_with_credentials() {
        let policy = CorsPolicy::default()
            .credentials(true)
            .expose_header("X-Request-Id");
        let headers = policy.response_headers("https://app.example.com");
        let origin_hdr = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Origin");
        assert_eq!(origin_hdr.unwrap().1, "https://app.example.com");
        let cred_hdr = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Credentials");
        assert_eq!(cred_hdr.unwrap().1, "true");
    }

    #[test]
    fn response_headers_wildcard() {
        let policy = CorsPolicy::default(); // OriginPattern::Any
        let headers = policy.response_headers("https://whatever.com");
        let origin_hdr = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Origin");
        assert_eq!(origin_hdr.unwrap().1, "*");
    }

    #[test]
    fn preflight_cache_basic() {
        let mut cache = PreflightCache::new();
        let entry = PreflightEntry {
            origin: "https://a.com".to_string(),
            method: "POST".to_string(),
            headers: vec!["content-type".to_string()],
            max_age: 3600,
            created_at_epoch_s: 1000,
        };
        cache.insert(entry);
        assert_eq!(cache.len(), 1);
        let found = cache.lookup(
            "https://a.com",
            "POST",
            &["content-type".to_string()],
            1500,
        );
        assert!(found.is_some());
        // Expired
        let found = cache.lookup(
            "https://a.com",
            "POST",
            &["content-type".to_string()],
            5000,
        );
        assert!(found.is_none());
    }

    #[test]
    fn preflight_cache_evict() {
        let mut cache = PreflightCache::new();
        cache.insert(PreflightEntry {
            origin: "a".to_string(),
            method: "GET".to_string(),
            headers: vec![],
            max_age: 10,
            created_at_epoch_s: 100,
        });
        cache.insert(PreflightEntry {
            origin: "b".to_string(),
            method: "GET".to_string(),
            headers: vec![],
            max_age: 1000,
            created_at_epoch_s: 100,
        });
        cache.evict_expired(200);
        assert_eq!(cache.len(), 1);
    }
}
