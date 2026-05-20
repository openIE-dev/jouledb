//! CSRF protection — token generation, double-submit cookie pattern, origin/referer
//! validation, token rotation, per-session tokens, SameSite cookie configuration.
//!
//! Replaces csurf (Node.js) and Django CSRF middleware with a comprehensive
//! pure-Rust CSRF protection engine.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ─────────────────────────────────────────────────────

/// CSRF protection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsrfProtectError {
    /// Token missing from request.
    TokenMissing,
    /// Token mismatch between cookie and form/header.
    TokenMismatch,
    /// Token has expired.
    TokenExpired,
    /// Invalid token format.
    InvalidFormat(String),
    /// Session not found.
    SessionNotFound(String),
    /// Origin validation failed.
    OriginMismatch { expected: String, got: String },
    /// Referer validation failed.
    RefererMismatch { expected: String, got: String },
}

impl std::fmt::Display for CsrfProtectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenMissing => write!(f, "CSRF token missing"),
            Self::TokenMismatch => write!(f, "CSRF token mismatch"),
            Self::TokenExpired => write!(f, "CSRF token expired"),
            Self::InvalidFormat(s) => write!(f, "invalid CSRF token format: {s}"),
            Self::SessionNotFound(id) => write!(f, "session not found: {id}"),
            Self::OriginMismatch { expected, got } => {
                write!(f, "origin mismatch: expected {expected}, got {got}")
            }
            Self::RefererMismatch { expected, got } => {
                write!(f, "referer mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for CsrfProtectError {}

// ── Random Bytes ───────────────────────────────────────────────

/// Generate pseudo-random bytes from system entropy sources.
fn random_bytes(len: usize) -> Vec<u8> {
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stack_val: usize = 0;
    let stack_addr = &stack_val as *const _ as usize;

    let mut seed = Vec::new();
    seed.extend_from_slice(&ts.to_le_bytes());
    seed.extend_from_slice(&counter.to_le_bytes());
    seed.extend_from_slice(&stack_addr.to_le_bytes());
    seed.extend_from_slice(&(std::process::id() as u64).to_le_bytes());

    let mut result = Vec::with_capacity(len);
    let mut idx = 0u64;
    while result.len() < len {
        let mut block_seed = seed.clone();
        block_seed.extend_from_slice(&idx.to_le_bytes());
        let hash = fnv_expand(&block_seed, 32);
        let take = (len - result.len()).min(hash.len());
        result.extend_from_slice(&hash[..take]);
        idx += 1;
    }
    result
}

/// FNV-1a-based expansion to arbitrary length.
fn fnv_expand(data: &[u8], len: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(len);
    for round in 0..len {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= round as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        result.push((hash & 0xFF) as u8);
    }
    result
}

/// Convert bytes to hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Timing-safe byte comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ── SameSite ───────────────────────────────────────────────────

/// SameSite cookie attribute for CSRF protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SameSitePolicy {
    /// Cookie sent only with same-site requests.
    Strict,
    /// Cookie sent with same-site + top-level navigations.
    Lax,
    /// Cookie sent with all requests (requires Secure).
    None,
}

impl std::fmt::Display for SameSitePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strict => write!(f, "Strict"),
            Self::Lax => write!(f, "Lax"),
            Self::None => write!(f, "None"),
        }
    }
}

// ── Cookie Config ──────────────────────────────────────────────

/// CSRF cookie configuration for the double-submit pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrfCookieConfig {
    pub name: String,
    pub path: String,
    pub http_only: bool,
    pub secure: bool,
    pub same_site: SameSitePolicy,
    pub max_age_secs: u64,
    pub domain: Option<String>,
}

impl Default for CsrfCookieConfig {
    fn default() -> Self {
        Self {
            name: "__csrf".to_string(),
            path: "/".to_string(),
            http_only: false, // Must be readable by JS for double-submit.
            secure: true,
            same_site: SameSitePolicy::Lax,
            max_age_secs: 3600,
            domain: None,
        }
    }
}

impl CsrfCookieConfig {
    /// Build a Set-Cookie header value.
    pub fn to_set_cookie(&self, token: &str) -> String {
        let mut parts = vec![format!("{}={}", self.name, token)];
        parts.push(format!("Path={}", self.path));
        parts.push(format!("Max-Age={}", self.max_age_secs));
        parts.push(format!("SameSite={}", self.same_site));
        if self.http_only {
            parts.push("HttpOnly".to_string());
        }
        if self.secure {
            parts.push("Secure".to_string());
        }
        if let Some(domain) = &self.domain {
            parts.push(format!("Domain={domain}"));
        }
        parts.join("; ")
    }
}

// ── Token ──────────────────────────────────────────────────────

/// A CSRF token with metadata.
#[derive(Debug, Clone)]
pub struct CsrfToken {
    /// The token value (hex string).
    pub value: String,
    /// When the token was created.
    pub created_at: DateTime<Utc>,
    /// When the token expires.
    pub expires_at: DateTime<Utc>,
}

impl CsrfToken {
    /// Generate a new CSRF token with the given TTL.
    pub fn generate(ttl_secs: i64) -> Self {
        let bytes = random_bytes(32);
        let now = Utc::now();
        Self {
            value: bytes_to_hex(&bytes),
            created_at: now,
            expires_at: now + Duration::seconds(ttl_secs),
        }
    }

    /// Check whether this token has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Timing-safe comparison with a candidate string.
    pub fn matches(&self, candidate: &str) -> bool {
        constant_time_eq(self.value.as_bytes(), candidate.as_bytes())
    }
}

// ── Session Token Store ────────────────────────────────────────

/// Per-session CSRF token store supporting rotation.
pub struct SessionTokenStore {
    /// Map from session_id to list of valid tokens (for rotation grace period).
    sessions: HashMap<String, Vec<CsrfToken>>,
    /// Default token TTL in seconds.
    pub token_ttl_secs: i64,
    /// Maximum tokens per session (for rotation grace).
    pub max_tokens_per_session: usize,
}

impl SessionTokenStore {
    /// Create a new session token store.
    pub fn new(token_ttl_secs: i64) -> Self {
        Self {
            sessions: HashMap::new(),
            token_ttl_secs,
            max_tokens_per_session: 5,
        }
    }

    /// Generate a new token for a session, rotating old ones.
    pub fn generate_for_session(&mut self, session_id: &str) -> String {
        let token = CsrfToken::generate(self.token_ttl_secs);
        let value = token.value.clone();

        let tokens = self.sessions.entry(session_id.to_string()).or_default();

        // Remove expired tokens.
        tokens.retain(|t| !t.is_expired());

        tokens.push(token);

        // Enforce max tokens per session.
        while tokens.len() > self.max_tokens_per_session {
            tokens.remove(0);
        }

        value
    }

    /// Validate a token for a session.
    pub fn validate(
        &self,
        session_id: &str,
        candidate: &str,
    ) -> Result<(), CsrfProtectError> {
        let tokens = self
            .sessions
            .get(session_id)
            .ok_or_else(|| CsrfProtectError::SessionNotFound(session_id.to_string()))?;

        for token in tokens {
            if token.is_expired() {
                continue;
            }
            if token.matches(candidate) {
                return Ok(());
            }
        }

        // Check if any expired token matches (for better error message).
        for token in tokens {
            if token.matches(candidate) {
                return Err(CsrfProtectError::TokenExpired);
            }
        }

        Err(CsrfProtectError::TokenMismatch)
    }

    /// Invalidate all tokens for a session (e.g., on logout).
    pub fn invalidate_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Purge all expired tokens across all sessions.
    pub fn purge_expired(&mut self) {
        self.sessions.retain(|_, tokens| {
            tokens.retain(|t| !t.is_expired());
            !tokens.is_empty()
        });
    }
}

// ── Double Submit Cookie ───────────────────────────────────────

/// Double-submit cookie CSRF protection.
///
/// The server sets a CSRF token in both a cookie and a hidden form field.
/// On submission, the server compares the cookie value against the form value.
pub struct DoubleSubmitProtection {
    pub cookie_config: CsrfCookieConfig,
}

impl DoubleSubmitProtection {
    /// Create with default cookie config.
    pub fn new() -> Self {
        Self {
            cookie_config: CsrfCookieConfig::default(),
        }
    }

    /// Create with custom cookie config.
    pub fn with_config(config: CsrfCookieConfig) -> Self {
        Self {
            cookie_config: config,
        }
    }

    /// Generate a token and the corresponding Set-Cookie header.
    pub fn generate(&self) -> (String, String) {
        let token = bytes_to_hex(&random_bytes(32));
        let cookie = self.cookie_config.to_set_cookie(&token);
        (token, cookie)
    }

    /// Validate that the cookie token matches the submitted token.
    pub fn validate(
        &self,
        cookie_token: &str,
        submitted_token: &str,
    ) -> Result<(), CsrfProtectError> {
        if cookie_token.is_empty() || submitted_token.is_empty() {
            return Err(CsrfProtectError::TokenMissing);
        }
        if !constant_time_eq(cookie_token.as_bytes(), submitted_token.as_bytes()) {
            return Err(CsrfProtectError::TokenMismatch);
        }
        Ok(())
    }
}

// ── Origin / Referer Validation ────────────────────────────────

/// Validate request origin against allowed origins.
pub struct OriginValidator {
    /// Allowed origins (e.g., "https://example.com").
    allowed_origins: Vec<String>,
}

impl OriginValidator {
    /// Create a new origin validator.
    pub fn new(allowed_origins: Vec<String>) -> Self {
        Self { allowed_origins }
    }

    /// Validate the Origin header.
    pub fn validate_origin(&self, origin: &str) -> Result<(), CsrfProtectError> {
        if self.allowed_origins.iter().any(|o| o == origin) {
            Ok(())
        } else {
            Err(CsrfProtectError::OriginMismatch {
                expected: self.allowed_origins.join(", "),
                got: origin.to_string(),
            })
        }
    }

    /// Validate the Referer header by checking its origin portion.
    pub fn validate_referer(&self, referer: &str) -> Result<(), CsrfProtectError> {
        let origin = extract_origin(referer);
        if self.allowed_origins.iter().any(|o| o == &origin) {
            Ok(())
        } else {
            Err(CsrfProtectError::RefererMismatch {
                expected: self.allowed_origins.join(", "),
                got: referer.to_string(),
            })
        }
    }

    /// Add an allowed origin.
    pub fn add_origin(&mut self, origin: String) {
        if !self.allowed_origins.contains(&origin) {
            self.allowed_origins.push(origin);
        }
    }

    /// Get allowed origins.
    pub fn allowed_origins(&self) -> &[String] {
        &self.allowed_origins
    }
}

/// Extract the origin (scheme + host + port) from a URL.
fn extract_origin(url: &str) -> String {
    // Find scheme.
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = &url[scheme_end + 3..];
        // Find path start.
        if let Some(path_start) = after_scheme.find('/') {
            url[..scheme_end + 3 + path_start].to_string()
        } else {
            url.to_string()
        }
    } else {
        url.to_string()
    }
}

// ── Full CSRF Guard ────────────────────────────────────────────

/// Comprehensive CSRF guard combining multiple protection strategies.
pub struct CsrfGuard {
    /// Session token store.
    pub token_store: SessionTokenStore,
    /// Double submit protection.
    pub double_submit: DoubleSubmitProtection,
    /// Origin validator (optional).
    pub origin_validator: Option<OriginValidator>,
    /// Whether to check Origin header.
    pub check_origin: bool,
    /// Whether to check Referer header.
    pub check_referer: bool,
    /// HTTP methods that are exempt from CSRF checks.
    pub safe_methods: Vec<String>,
}

impl CsrfGuard {
    /// Create a new CSRF guard with default settings.
    pub fn new(token_ttl_secs: i64) -> Self {
        Self {
            token_store: SessionTokenStore::new(token_ttl_secs),
            double_submit: DoubleSubmitProtection::new(),
            origin_validator: None,
            check_origin: false,
            check_referer: false,
            safe_methods: vec![
                "GET".to_string(),
                "HEAD".to_string(),
                "OPTIONS".to_string(),
            ],
        }
    }

    /// Enable origin checking with allowed origins.
    pub fn with_origin_check(mut self, allowed_origins: Vec<String>) -> Self {
        self.origin_validator = Some(OriginValidator::new(allowed_origins));
        self.check_origin = true;
        self
    }

    /// Enable referer checking.
    pub fn with_referer_check(mut self) -> Self {
        self.check_referer = true;
        self
    }

    /// Check whether an HTTP method is safe (exempt from CSRF).
    pub fn is_safe_method(&self, method: &str) -> bool {
        self.safe_methods.iter().any(|m| m.eq_ignore_ascii_case(method))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_token() {
        let token = CsrfToken::generate(3600);
        assert_eq!(token.value.len(), 64); // 32 bytes = 64 hex chars
        assert!(!token.is_expired());
    }

    #[test]
    fn test_token_expiry() {
        let mut token = CsrfToken::generate(3600);
        assert!(!token.is_expired());
        token.expires_at = Utc::now() - Duration::seconds(1);
        assert!(token.is_expired());
    }

    #[test]
    fn test_token_matches() {
        let token = CsrfToken::generate(3600);
        assert!(token.matches(&token.value));
        assert!(!token.matches("wrong_token"));
    }

    #[test]
    fn test_timing_safe_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hi", b"hello"));
    }

    #[test]
    fn test_double_submit_generate() {
        let ds = DoubleSubmitProtection::new();
        let (token, cookie) = ds.generate();
        assert_eq!(token.len(), 64);
        assert!(cookie.contains("__csrf="));
        assert!(cookie.contains("Secure"));
    }

    #[test]
    fn test_double_submit_validate_ok() {
        let ds = DoubleSubmitProtection::new();
        let (token, _) = ds.generate();
        assert!(ds.validate(&token, &token).is_ok());
    }

    #[test]
    fn test_double_submit_validate_mismatch() {
        let ds = DoubleSubmitProtection::new();
        let (token, _) = ds.generate();
        let err = ds.validate(&token, "wrong").unwrap_err();
        assert_eq!(err, CsrfProtectError::TokenMismatch);
    }

    #[test]
    fn test_double_submit_validate_empty() {
        let ds = DoubleSubmitProtection::new();
        let err = ds.validate("", "something").unwrap_err();
        assert_eq!(err, CsrfProtectError::TokenMissing);
    }

    #[test]
    fn test_session_store_generate_and_validate() {
        let mut store = SessionTokenStore::new(3600);
        let token = store.generate_for_session("sess1");
        assert!(store.validate("sess1", &token).is_ok());
    }

    #[test]
    fn test_session_store_wrong_token() {
        let mut store = SessionTokenStore::new(3600);
        store.generate_for_session("sess1");
        let err = store.validate("sess1", "wrong").unwrap_err();
        assert_eq!(err, CsrfProtectError::TokenMismatch);
    }

    #[test]
    fn test_session_store_not_found() {
        let store = SessionTokenStore::new(3600);
        let err = store.validate("missing", "token").unwrap_err();
        match err {
            CsrfProtectError::SessionNotFound(id) => assert_eq!(id, "missing"),
            _ => panic!("expected SessionNotFound"),
        }
    }

    #[test]
    fn test_session_store_rotation() {
        let mut store = SessionTokenStore::new(3600);
        let token1 = store.generate_for_session("s");
        let token2 = store.generate_for_session("s");
        // Both should still be valid during grace period.
        assert!(store.validate("s", &token1).is_ok());
        assert!(store.validate("s", &token2).is_ok());
    }

    #[test]
    fn test_session_store_max_tokens() {
        let mut store = SessionTokenStore::new(3600);
        store.max_tokens_per_session = 2;
        let _t1 = store.generate_for_session("s");
        let t2 = store.generate_for_session("s");
        let t3 = store.generate_for_session("s");
        // t1 should have been evicted.
        assert!(store.validate("s", &t2).is_ok());
        assert!(store.validate("s", &t3).is_ok());
    }

    #[test]
    fn test_session_store_invalidate() {
        let mut store = SessionTokenStore::new(3600);
        let token = store.generate_for_session("s");
        store.invalidate_session("s");
        assert!(store.validate("s", &token).is_err());
    }

    #[test]
    fn test_origin_validator() {
        let v = OriginValidator::new(vec!["https://example.com".to_string()]);
        assert!(v.validate_origin("https://example.com").is_ok());
        assert!(v.validate_origin("https://evil.com").is_err());
    }

    #[test]
    fn test_referer_validator() {
        let v = OriginValidator::new(vec!["https://example.com".to_string()]);
        assert!(v.validate_referer("https://example.com/path?q=1").is_ok());
        assert!(v.validate_referer("https://evil.com/path").is_err());
    }

    #[test]
    fn test_extract_origin() {
        assert_eq!(
            extract_origin("https://example.com/path?q=1"),
            "https://example.com"
        );
        assert_eq!(
            extract_origin("http://localhost:3000/api"),
            "http://localhost:3000"
        );
        assert_eq!(extract_origin("https://example.com"), "https://example.com");
    }

    #[test]
    fn test_cookie_config_default() {
        let config = CsrfCookieConfig::default();
        assert_eq!(config.name, "__csrf");
        assert!(!config.http_only); // Must be JS-readable for double-submit
        assert!(config.secure);
        assert_eq!(config.same_site, SameSitePolicy::Lax);
    }

    #[test]
    fn test_cookie_config_with_domain() {
        let mut config = CsrfCookieConfig::default();
        config.domain = Some("example.com".to_string());
        let cookie = config.to_set_cookie("abc123");
        assert!(cookie.contains("Domain=example.com"));
    }

    #[test]
    fn test_csrf_guard_safe_methods() {
        let guard = CsrfGuard::new(3600);
        assert!(guard.is_safe_method("GET"));
        assert!(guard.is_safe_method("get"));
        assert!(guard.is_safe_method("HEAD"));
        assert!(guard.is_safe_method("OPTIONS"));
        assert!(!guard.is_safe_method("POST"));
        assert!(!guard.is_safe_method("PUT"));
    }

    #[test]
    fn test_session_store_purge_expired() {
        let mut store = SessionTokenStore::new(3600);
        let token = store.generate_for_session("s");
        assert_eq!(store.session_count(), 1);
        // Manually expire.
        if let Some(tokens) = store.sessions.get_mut("s") {
            for t in tokens.iter_mut() {
                t.expires_at = Utc::now() - Duration::seconds(1);
            }
        }
        store.purge_expired();
        assert_eq!(store.session_count(), 0);
        assert!(store.validate("s", &token).is_err());
    }

    #[test]
    fn test_same_site_display() {
        assert_eq!(SameSitePolicy::Strict.to_string(), "Strict");
        assert_eq!(SameSitePolicy::Lax.to_string(), "Lax");
        assert_eq!(SameSitePolicy::None.to_string(), "None");
    }

    #[test]
    fn test_error_display() {
        let e = CsrfProtectError::TokenMissing;
        assert_eq!(e.to_string(), "CSRF token missing");
        let e2 = CsrfProtectError::OriginMismatch {
            expected: "a".to_string(),
            got: "b".to_string(),
        };
        assert!(e2.to_string().contains("origin mismatch"));
    }

    #[test]
    fn test_origin_validator_add() {
        let mut v = OriginValidator::new(vec![]);
        assert!(v.validate_origin("https://example.com").is_err());
        v.add_origin("https://example.com".to_string());
        assert!(v.validate_origin("https://example.com").is_ok());
        // Duplicate add is idempotent.
        v.add_origin("https://example.com".to_string());
        assert_eq!(v.allowed_origins().len(), 1);
    }
}
