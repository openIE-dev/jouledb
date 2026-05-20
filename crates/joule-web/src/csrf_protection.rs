//! CSRF protection — double-submit cookie, synchronizer token, per-request tokens,
//! token validation, SameSite cookie attributes, and origin checking.
//!
//! Replaces `csurf`, `csrf-csrf`, and `tiny-csrf` with a pure-Rust CSRF defense
//! engine supporting multiple protection strategies and cookie configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// CSRF protection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsrfError {
    /// CSRF token missing from request.
    TokenMissing,
    /// CSRF token does not match expected value.
    TokenMismatch { expected_prefix: String },
    /// Token has expired.
    TokenExpired { token_id: String, expired_at_ms: u64 },
    /// Token already used (one-time tokens).
    TokenAlreadyUsed(String),
    /// Origin header mismatch.
    OriginMismatch { expected: String, actual: String },
    /// Referer header mismatch.
    RefererMismatch { expected: String, actual: String },
    /// Invalid cookie configuration.
    InvalidConfig(String),
    /// Session not found for synchronizer token.
    SessionNotFound(String),
}

impl fmt::Display for CsrfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenMissing => write!(f, "CSRF token missing"),
            Self::TokenMismatch { expected_prefix } => {
                write!(f, "CSRF token mismatch (expected prefix: {expected_prefix})")
            }
            Self::TokenExpired { token_id, expired_at_ms } => {
                write!(f, "CSRF token {token_id} expired at {expired_at_ms}")
            }
            Self::TokenAlreadyUsed(id) => write!(f, "CSRF token already used: {id}"),
            Self::OriginMismatch { expected, actual } => {
                write!(f, "origin mismatch: expected {expected}, got {actual}")
            }
            Self::RefererMismatch { expected, actual } => {
                write!(f, "referer mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid CSRF config: {msg}"),
            Self::SessionNotFound(id) => write!(f, "session not found: {id}"),
        }
    }
}

impl std::error::Error for CsrfError {}

// ── SameSite Cookie Attribute ──────────────────────────────────

/// SameSite cookie attribute values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SameSite {
    /// Cookie sent only for same-site requests.
    Strict,
    /// Cookie sent for same-site and top-level cross-site GET.
    Lax,
    /// Cookie always sent (requires Secure).
    None,
}

impl SameSite {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Strict => "Strict",
            Self::Lax => "Lax",
            Self::None => "None",
        }
    }
}

impl fmt::Display for SameSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Cookie Configuration ───────────────────────────────────────

/// CSRF cookie configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrfCookieConfig {
    /// Cookie name.
    pub name: String,
    /// Cookie path.
    pub path: String,
    /// Cookie domain (optional).
    pub domain: Option<String>,
    /// Secure flag (HTTPS only).
    pub secure: bool,
    /// HttpOnly flag.
    pub http_only: bool,
    /// SameSite attribute.
    pub same_site: SameSite,
    /// Max age in seconds.
    pub max_age_seconds: Option<u64>,
}

impl Default for CsrfCookieConfig {
    fn default() -> Self {
        Self {
            name: "__csrf".to_string(),
            path: "/".to_string(),
            domain: None,
            secure: true,
            http_only: false, // Must be readable by JS for double-submit
            same_site: SameSite::Lax,
            max_age_seconds: Some(3600),
        }
    }
}

impl CsrfCookieConfig {
    /// Validate the cookie configuration.
    pub fn validate(&self) -> Result<(), CsrfError> {
        if self.name.is_empty() {
            return Err(CsrfError::InvalidConfig("cookie name is empty".into()));
        }
        // SameSite=None requires Secure
        if self.same_site == SameSite::None && !self.secure {
            return Err(CsrfError::InvalidConfig(
                "SameSite=None requires Secure flag".into(),
            ));
        }
        Ok(())
    }

    /// Build a Set-Cookie header value for this token.
    pub fn to_set_cookie_header(&self, token: &str) -> String {
        let mut parts = vec![format!("{}={}", self.name, token)];
        parts.push(format!("Path={}", self.path));
        if let Some(domain) = &self.domain {
            parts.push(format!("Domain={domain}"));
        }
        if self.secure {
            parts.push("Secure".to_string());
        }
        if self.http_only {
            parts.push("HttpOnly".to_string());
        }
        parts.push(format!("SameSite={}", self.same_site));
        if let Some(max_age) = self.max_age_seconds {
            parts.push(format!("Max-Age={max_age}"));
        }
        parts.join("; ")
    }
}

// ── CSRF Protection Strategy ───────────────────────────────────

/// Protection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CsrfStrategy {
    /// Double-submit cookie: token in cookie must match token in header/body.
    DoubleSubmitCookie,
    /// Synchronizer token: server stores token per session, validates on submit.
    SynchronizerToken,
    /// Origin/Referer checking only (no token).
    OriginCheck,
}

// ── Token Generation ───────────────────────────────────────────

/// Generate a deterministic CSRF token from a seed value.
/// Uses a mixing function to produce a URL-safe token string.
fn generate_token_from_seed(seed: u64) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut state = seed;
    let mut token = String::with_capacity(32);
    for _ in 0..32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state >> 33) as usize % CHARS.len();
        token.push(CHARS[idx] as char);
    }
    token
}

/// Constant-time comparison to prevent timing attacks.
fn timing_safe_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ── Token Store (Synchronizer) ─────────────────────────────────

/// A CSRF token with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrfToken {
    /// Token ID.
    pub token_id: String,
    /// The token value.
    pub token_value: String,
    /// Associated session ID.
    pub session_id: String,
    /// When the token was issued (ms since epoch).
    pub issued_at_ms: u64,
    /// When the token expires (ms since epoch).
    pub expires_at_ms: u64,
    /// Whether this token has been used (for one-time tokens).
    pub used: bool,
}

/// In-memory CSRF token store for synchronizer token pattern.
pub struct CsrfTokenStore {
    tokens: HashMap<String, CsrfToken>,
    by_session: HashMap<String, Vec<String>>,
    next_id: u64,
    token_ttl_ms: u64,
    one_time_tokens: bool,
}

impl CsrfTokenStore {
    pub fn new(token_ttl_ms: u64, one_time_tokens: bool) -> Self {
        Self {
            tokens: HashMap::new(),
            by_session: HashMap::new(),
            next_id: 1,
            token_ttl_ms,
            one_time_tokens,
        }
    }

    /// Generate a new CSRF token for a session.
    pub fn generate(&mut self, session_id: &str, now_ms: u64) -> CsrfToken {
        let token_id = format!("csrf_{:08x}", self.next_id);
        let token_value = generate_token_from_seed(self.next_id.wrapping_mul(now_ms.wrapping_add(42)));
        self.next_id += 1;

        let token = CsrfToken {
            token_id: token_id.clone(),
            token_value: token_value.clone(),
            session_id: session_id.to_string(),
            issued_at_ms: now_ms,
            expires_at_ms: now_ms + self.token_ttl_ms,
            used: false,
        };

        self.tokens.insert(token_id.clone(), token.clone());
        self.by_session
            .entry(session_id.to_string())
            .or_default()
            .push(token_id);

        token
    }

    /// Validate a submitted CSRF token.
    pub fn validate(
        &mut self,
        session_id: &str,
        submitted_token: &str,
        now_ms: u64,
    ) -> Result<(), CsrfError> {
        // Find token by value in the session's tokens
        let session_tokens = self
            .by_session
            .get(session_id)
            .ok_or(CsrfError::SessionNotFound(session_id.to_string()))?;

        let matching_id = session_tokens
            .iter()
            .find(|tid| {
                self.tokens
                    .get(*tid)
                    .map(|t| timing_safe_eq(t.token_value.as_bytes(), submitted_token.as_bytes()))
                    .unwrap_or(false)
            })
            .cloned();

        let token_id = matching_id.ok_or(CsrfError::TokenMismatch {
            expected_prefix: format!("session={session_id}"),
        })?;

        let token = self.tokens.get(&token_id).unwrap();

        // Check expiry
        if token.expires_at_ms <= now_ms {
            return Err(CsrfError::TokenExpired {
                token_id: token.token_id.clone(),
                expired_at_ms: token.expires_at_ms,
            });
        }

        // Check one-time use
        if self.one_time_tokens && token.used {
            return Err(CsrfError::TokenAlreadyUsed(token.token_id.clone()));
        }

        // Mark as used
        if let Some(t) = self.tokens.get_mut(&token_id) {
            t.used = true;
        }

        Ok(())
    }

    /// Remove expired tokens.
    pub fn cleanup(&mut self, now_ms: u64) -> usize {
        let expired: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, t)| t.expires_at_ms <= now_ms)
            .map(|(id, _)| id.clone())
            .collect();
        let count = expired.len();
        for id in &expired {
            if let Some(token) = self.tokens.remove(id) {
                if let Some(session_tokens) = self.by_session.get_mut(&token.session_id) {
                    session_tokens.retain(|tid| tid != id);
                }
            }
        }
        count
    }

    /// Get all tokens for a session.
    pub fn tokens_for_session(&self, session_id: &str) -> Vec<&CsrfToken> {
        self.by_session
            .get(session_id)
            .map(|ids| ids.iter().filter_map(|id| self.tokens.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

// ── Double-Submit Cookie Validator ─────────────────────────────

/// Double-submit cookie CSRF validator.
/// The cookie value and a header/form field value must match.
pub struct DoubleSubmitValidator {
    pub cookie_config: CsrfCookieConfig,
}

impl DoubleSubmitValidator {
    pub fn new(cookie_config: CsrfCookieConfig) -> Self {
        Self { cookie_config }
    }

    /// Generate a token for a new request (to set in cookie + inject in page).
    pub fn generate_token(&self, seed: u64) -> String {
        generate_token_from_seed(seed)
    }

    /// Validate that the cookie value matches the submitted value.
    pub fn validate(&self, cookie_value: &str, submitted_value: &str) -> Result<(), CsrfError> {
        if cookie_value.is_empty() || submitted_value.is_empty() {
            return Err(CsrfError::TokenMissing);
        }
        if timing_safe_eq(cookie_value.as_bytes(), submitted_value.as_bytes()) {
            Ok(())
        } else {
            Err(CsrfError::TokenMismatch {
                expected_prefix: cookie_value[..cookie_value.len().min(8)].to_string(),
            })
        }
    }
}

// ── Origin / Referer Checker ───────────────────────────────────

/// Origin and Referer header validator.
pub struct OriginChecker {
    /// Allowed origins (exact match).
    pub allowed_origins: Vec<String>,
}

impl OriginChecker {
    pub fn new(allowed_origins: Vec<String>) -> Self {
        Self { allowed_origins }
    }

    /// Check the Origin header.
    pub fn check_origin(&self, origin: Option<&str>) -> Result<(), CsrfError> {
        match origin {
            None => Ok(()), // No Origin header = same-origin (browsers don't send it for same-origin)
            Some(o) => {
                if self.allowed_origins.iter().any(|a| a == o) {
                    Ok(())
                } else {
                    Err(CsrfError::OriginMismatch {
                        expected: self.allowed_origins.join(", "),
                        actual: o.to_string(),
                    })
                }
            }
        }
    }

    /// Check the Referer header (extract origin part).
    pub fn check_referer(&self, referer: Option<&str>) -> Result<(), CsrfError> {
        match referer {
            None => Ok(()), // No Referer is acceptable (privacy settings)
            Some(r) => {
                let origin = extract_origin_from_url(r);
                if self.allowed_origins.iter().any(|a| a == &origin) {
                    Ok(())
                } else {
                    Err(CsrfError::RefererMismatch {
                        expected: self.allowed_origins.join(", "),
                        actual: origin,
                    })
                }
            }
        }
    }
}

/// Extract the origin (scheme + host + port) from a URL.
fn extract_origin_from_url(url: &str) -> String {
    // Find scheme
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = &url[scheme_end + 3..];
        // Find the end of the host+port (first / or end of string)
        let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        format!("{}{}", &url[..scheme_end + 3], &after_scheme[..host_end])
    } else {
        url.to_string()
    }
}

// ── Combined CSRF Middleware ───────────────────────────────────

/// HTTP methods that should be CSRF-protected (state-changing).
pub fn is_state_changing_method(method: &str) -> bool {
    matches!(
        method.to_uppercase().as_str(),
        "POST" | "PUT" | "PATCH" | "DELETE"
    )
}

/// HTTP methods that are safe (no CSRF protection needed).
pub fn is_safe_method(method: &str) -> bool {
    matches!(method.to_uppercase().as_str(), "GET" | "HEAD" | "OPTIONS")
}

/// A CSRF request to validate.
#[derive(Debug, Clone)]
pub struct CsrfRequest {
    pub method: String,
    pub origin: Option<String>,
    pub referer: Option<String>,
    pub csrf_cookie: Option<String>,
    pub csrf_header: Option<String>,
    pub csrf_body_field: Option<String>,
    pub session_id: Option<String>,
}

/// Combined CSRF protection engine.
pub struct CsrfProtection {
    pub strategy: CsrfStrategy,
    pub origin_checker: Option<OriginChecker>,
    pub double_submit: Option<DoubleSubmitValidator>,
    pub token_store: Option<CsrfTokenStore>,
    pub skip_safe_methods: bool,
}

impl CsrfProtection {
    /// Create protection with double-submit cookie strategy.
    pub fn double_submit(cookie_config: CsrfCookieConfig) -> Self {
        Self {
            strategy: CsrfStrategy::DoubleSubmitCookie,
            origin_checker: None,
            double_submit: Some(DoubleSubmitValidator::new(cookie_config)),
            token_store: None,
            skip_safe_methods: true,
        }
    }

    /// Create protection with synchronizer token strategy.
    pub fn synchronizer(token_ttl_ms: u64, one_time: bool) -> Self {
        Self {
            strategy: CsrfStrategy::SynchronizerToken,
            origin_checker: None,
            double_submit: None,
            token_store: Some(CsrfTokenStore::new(token_ttl_ms, one_time)),
            skip_safe_methods: true,
        }
    }

    /// Create protection with origin checking only.
    pub fn origin_only(allowed_origins: Vec<String>) -> Self {
        Self {
            strategy: CsrfStrategy::OriginCheck,
            origin_checker: Some(OriginChecker::new(allowed_origins)),
            double_submit: None,
            token_store: None,
            skip_safe_methods: true,
        }
    }

    /// Add origin checking to any strategy.
    pub fn with_origin_check(mut self, allowed_origins: Vec<String>) -> Self {
        self.origin_checker = Some(OriginChecker::new(allowed_origins));
        self
    }

    /// Validate an incoming request.
    pub fn validate(&mut self, request: &CsrfRequest, now_ms: u64) -> Result<(), CsrfError> {
        // Skip safe methods if configured
        if self.skip_safe_methods && is_safe_method(&request.method) {
            return Ok(());
        }

        // Origin check (if configured)
        if let Some(checker) = &self.origin_checker {
            checker.check_origin(request.origin.as_deref())?;
        }

        // Strategy-specific validation
        match self.strategy {
            CsrfStrategy::DoubleSubmitCookie => {
                let validator = self.double_submit.as_ref().ok_or_else(|| {
                    CsrfError::InvalidConfig("double submit validator not configured".into())
                })?;
                let cookie = request
                    .csrf_cookie
                    .as_deref()
                    .ok_or(CsrfError::TokenMissing)?;
                let submitted = request
                    .csrf_header
                    .as_deref()
                    .or(request.csrf_body_field.as_deref())
                    .ok_or(CsrfError::TokenMissing)?;
                validator.validate(cookie, submitted)?;
            }
            CsrfStrategy::SynchronizerToken => {
                let store = self.token_store.as_mut().ok_or_else(|| {
                    CsrfError::InvalidConfig("token store not configured".into())
                })?;
                let session_id = request
                    .session_id
                    .as_deref()
                    .ok_or(CsrfError::SessionNotFound("no session".into()))?;
                let submitted = request
                    .csrf_header
                    .as_deref()
                    .or(request.csrf_body_field.as_deref())
                    .ok_or(CsrfError::TokenMissing)?;
                store.validate(session_id, submitted, now_ms)?;
            }
            CsrfStrategy::OriginCheck => {
                // Already checked above if origin_checker is set
                if self.origin_checker.is_none() {
                    return Err(CsrfError::InvalidConfig(
                        "origin checker not configured".into(),
                    ));
                }
            }
        }

        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_site_display() {
        assert_eq!(SameSite::Strict.as_str(), "Strict");
        assert_eq!(SameSite::Lax.as_str(), "Lax");
        assert_eq!(SameSite::None.as_str(), "None");
        assert_eq!(format!("{}", SameSite::Strict), "Strict");
    }

    #[test]
    fn test_cookie_config_default() {
        let cfg = CsrfCookieConfig::default();
        assert_eq!(cfg.name, "__csrf");
        assert!(cfg.secure);
        assert!(!cfg.http_only);
        assert_eq!(cfg.same_site, SameSite::Lax);
    }

    #[test]
    fn test_cookie_config_validate_ok() {
        let cfg = CsrfCookieConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_cookie_config_validate_empty_name() {
        let mut cfg = CsrfCookieConfig::default();
        cfg.name = "".into();
        assert!(matches!(cfg.validate(), Err(CsrfError::InvalidConfig(_))));
    }

    #[test]
    fn test_cookie_config_validate_samesite_none_no_secure() {
        let mut cfg = CsrfCookieConfig::default();
        cfg.same_site = SameSite::None;
        cfg.secure = false;
        assert!(matches!(cfg.validate(), Err(CsrfError::InvalidConfig(_))));
    }

    #[test]
    fn test_cookie_config_samesite_none_with_secure() {
        let mut cfg = CsrfCookieConfig::default();
        cfg.same_site = SameSite::None;
        cfg.secure = true;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_set_cookie_header() {
        let cfg = CsrfCookieConfig {
            name: "__csrf".into(),
            path: "/".into(),
            domain: Some("example.com".into()),
            secure: true,
            http_only: false,
            same_site: SameSite::Strict,
            max_age_seconds: Some(3600),
        };
        let header = cfg.to_set_cookie_header("token123");
        assert!(header.contains("__csrf=token123"));
        assert!(header.contains("Path=/"));
        assert!(header.contains("Domain=example.com"));
        assert!(header.contains("Secure"));
        assert!(header.contains("SameSite=Strict"));
        assert!(header.contains("Max-Age=3600"));
        assert!(!header.contains("HttpOnly"));
    }

    #[test]
    fn test_token_generation_deterministic() {
        let t1 = generate_token_from_seed(42);
        let t2 = generate_token_from_seed(42);
        assert_eq!(t1, t2);
        assert_eq!(t1.len(), 32);
    }

    #[test]
    fn test_token_generation_unique() {
        let t1 = generate_token_from_seed(1);
        let t2 = generate_token_from_seed(2);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_timing_safe_eq() {
        assert!(timing_safe_eq(b"hello", b"hello"));
        assert!(!timing_safe_eq(b"hello", b"world"));
        assert!(!timing_safe_eq(b"short", b"longer"));
    }

    #[test]
    fn test_double_submit_valid() {
        let validator = DoubleSubmitValidator::new(CsrfCookieConfig::default());
        let token = validator.generate_token(12345);
        assert!(validator.validate(&token, &token).is_ok());
    }

    #[test]
    fn test_double_submit_mismatch() {
        let validator = DoubleSubmitValidator::new(CsrfCookieConfig::default());
        let token = validator.generate_token(12345);
        assert!(matches!(
            validator.validate(&token, "wrong_token"),
            Err(CsrfError::TokenMismatch { .. })
        ));
    }

    #[test]
    fn test_double_submit_empty() {
        let validator = DoubleSubmitValidator::new(CsrfCookieConfig::default());
        assert!(matches!(
            validator.validate("", "value"),
            Err(CsrfError::TokenMissing)
        ));
        assert!(matches!(
            validator.validate("value", ""),
            Err(CsrfError::TokenMissing)
        ));
    }

    #[test]
    fn test_synchronizer_token_generate_validate() {
        let mut store = CsrfTokenStore::new(60_000, false);
        let token = store.generate("session_1", 1000);
        assert!(!token.token_value.is_empty());
        assert_eq!(store.token_count(), 1);

        // Validate
        assert!(store.validate("session_1", &token.token_value, 2000).is_ok());
    }

    #[test]
    fn test_synchronizer_token_expired() {
        let mut store = CsrfTokenStore::new(60_000, false);
        let token = store.generate("session_1", 1000);
        assert!(matches!(
            store.validate("session_1", &token.token_value, 100_000),
            Err(CsrfError::TokenExpired { .. })
        ));
    }

    #[test]
    fn test_synchronizer_token_wrong_session() {
        let mut store = CsrfTokenStore::new(60_000, false);
        let token = store.generate("session_1", 1000);
        assert!(matches!(
            store.validate("session_2", &token.token_value, 2000),
            Err(CsrfError::SessionNotFound(_))
        ));
    }

    #[test]
    fn test_synchronizer_token_wrong_value() {
        let mut store = CsrfTokenStore::new(60_000, false);
        store.generate("session_1", 1000);
        assert!(matches!(
            store.validate("session_1", "wrong_token", 2000),
            Err(CsrfError::TokenMismatch { .. })
        ));
    }

    #[test]
    fn test_synchronizer_one_time_token() {
        let mut store = CsrfTokenStore::new(60_000, true);
        let token = store.generate("session_1", 1000);
        // First use succeeds
        assert!(store.validate("session_1", &token.token_value, 2000).is_ok());
        // Second use fails
        assert!(matches!(
            store.validate("session_1", &token.token_value, 3000),
            Err(CsrfError::TokenAlreadyUsed(_))
        ));
    }

    #[test]
    fn test_synchronizer_reusable_token() {
        let mut store = CsrfTokenStore::new(60_000, false);
        let token = store.generate("session_1", 1000);
        // Both uses succeed
        assert!(store.validate("session_1", &token.token_value, 2000).is_ok());
        assert!(store.validate("session_1", &token.token_value, 3000).is_ok());
    }

    #[test]
    fn test_synchronizer_cleanup() {
        let mut store = CsrfTokenStore::new(10_000, false);
        store.generate("s1", 1000);
        store.generate("s1", 2000);
        store.generate("s2", 50_000); // Not expired at now=20_000
        let removed = store.cleanup(20_000);
        assert_eq!(removed, 2);
        assert_eq!(store.token_count(), 1);
    }

    #[test]
    fn test_synchronizer_tokens_for_session() {
        let mut store = CsrfTokenStore::new(60_000, false);
        store.generate("s1", 1000);
        store.generate("s1", 2000);
        store.generate("s2", 3000);
        assert_eq!(store.tokens_for_session("s1").len(), 2);
        assert_eq!(store.tokens_for_session("s2").len(), 1);
        assert_eq!(store.tokens_for_session("s3").len(), 0);
    }

    #[test]
    fn test_origin_checker_allowed() {
        let checker = OriginChecker::new(vec![
            "https://example.com".into(),
            "https://app.example.com".into(),
        ]);
        assert!(checker.check_origin(Some("https://example.com")).is_ok());
        assert!(checker.check_origin(Some("https://app.example.com")).is_ok());
    }

    #[test]
    fn test_origin_checker_denied() {
        let checker = OriginChecker::new(vec!["https://example.com".into()]);
        assert!(matches!(
            checker.check_origin(Some("https://evil.com")),
            Err(CsrfError::OriginMismatch { .. })
        ));
    }

    #[test]
    fn test_origin_checker_no_header() {
        let checker = OriginChecker::new(vec!["https://example.com".into()]);
        assert!(checker.check_origin(None).is_ok());
    }

    #[test]
    fn test_referer_checker_allowed() {
        let checker = OriginChecker::new(vec!["https://example.com".into()]);
        assert!(checker
            .check_referer(Some("https://example.com/page?q=1"))
            .is_ok());
    }

    #[test]
    fn test_referer_checker_denied() {
        let checker = OriginChecker::new(vec!["https://example.com".into()]);
        assert!(matches!(
            checker.check_referer(Some("https://evil.com/page")),
            Err(CsrfError::RefererMismatch { .. })
        ));
    }

    #[test]
    fn test_referer_checker_no_header() {
        let checker = OriginChecker::new(vec!["https://example.com".into()]);
        assert!(checker.check_referer(None).is_ok());
    }

    #[test]
    fn test_extract_origin_from_url() {
        assert_eq!(
            extract_origin_from_url("https://example.com/path?q=1"),
            "https://example.com"
        );
        assert_eq!(
            extract_origin_from_url("https://app.example.com:8443/api"),
            "https://app.example.com:8443"
        );
        assert_eq!(
            extract_origin_from_url("http://localhost:3000/"),
            "http://localhost:3000"
        );
    }

    #[test]
    fn test_is_state_changing_method() {
        assert!(is_state_changing_method("POST"));
        assert!(is_state_changing_method("PUT"));
        assert!(is_state_changing_method("PATCH"));
        assert!(is_state_changing_method("DELETE"));
        assert!(is_state_changing_method("post")); // Case insensitive
        assert!(!is_state_changing_method("GET"));
        assert!(!is_state_changing_method("HEAD"));
    }

    #[test]
    fn test_is_safe_method() {
        assert!(is_safe_method("GET"));
        assert!(is_safe_method("HEAD"));
        assert!(is_safe_method("OPTIONS"));
        assert!(is_safe_method("get"));
        assert!(!is_safe_method("POST"));
    }

    #[test]
    fn test_protection_double_submit_pass() {
        let mut csrf = CsrfProtection::double_submit(CsrfCookieConfig::default());
        let token = generate_token_from_seed(42);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: None,
            referer: None,
            csrf_cookie: Some(token.clone()),
            csrf_header: Some(token),
            csrf_body_field: None,
            session_id: None,
        };
        assert!(csrf.validate(&request, 1000).is_ok());
    }

    #[test]
    fn test_protection_double_submit_fail() {
        let mut csrf = CsrfProtection::double_submit(CsrfCookieConfig::default());
        let request = CsrfRequest {
            method: "POST".into(),
            origin: None,
            referer: None,
            csrf_cookie: Some("cookie_token".into()),
            csrf_header: Some("different_token".into()),
            csrf_body_field: None,
            session_id: None,
        };
        assert!(matches!(
            csrf.validate(&request, 1000),
            Err(CsrfError::TokenMismatch { .. })
        ));
    }

    #[test]
    fn test_protection_skip_safe_methods() {
        let mut csrf = CsrfProtection::double_submit(CsrfCookieConfig::default());
        let request = CsrfRequest {
            method: "GET".into(),
            origin: None,
            referer: None,
            csrf_cookie: None,
            csrf_header: None,
            csrf_body_field: None,
            session_id: None,
        };
        assert!(csrf.validate(&request, 1000).is_ok());
    }

    #[test]
    fn test_protection_synchronizer_pass() {
        let mut csrf = CsrfProtection::synchronizer(60_000, false);
        let token = csrf
            .token_store
            .as_mut()
            .unwrap()
            .generate("sess_1", 1000);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: None,
            referer: None,
            csrf_cookie: None,
            csrf_header: Some(token.token_value),
            csrf_body_field: None,
            session_id: Some("sess_1".into()),
        };
        assert!(csrf.validate(&request, 2000).is_ok());
    }

    #[test]
    fn test_protection_origin_only() {
        let mut csrf = CsrfProtection::origin_only(vec!["https://example.com".into()]);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: Some("https://example.com".into()),
            referer: None,
            csrf_cookie: None,
            csrf_header: None,
            csrf_body_field: None,
            session_id: None,
        };
        assert!(csrf.validate(&request, 1000).is_ok());
    }

    #[test]
    fn test_protection_origin_only_denied() {
        let mut csrf = CsrfProtection::origin_only(vec!["https://example.com".into()]);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: Some("https://evil.com".into()),
            referer: None,
            csrf_cookie: None,
            csrf_header: None,
            csrf_body_field: None,
            session_id: None,
        };
        assert!(matches!(
            csrf.validate(&request, 1000),
            Err(CsrfError::OriginMismatch { .. })
        ));
    }

    #[test]
    fn test_protection_with_origin_check() {
        let mut csrf = CsrfProtection::double_submit(CsrfCookieConfig::default())
            .with_origin_check(vec!["https://example.com".into()]);
        let token = generate_token_from_seed(42);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: Some("https://evil.com".into()),
            referer: None,
            csrf_cookie: Some(token.clone()),
            csrf_header: Some(token),
            csrf_body_field: None,
            session_id: None,
        };
        // Fails on origin check even though token matches
        assert!(matches!(
            csrf.validate(&request, 1000),
            Err(CsrfError::OriginMismatch { .. })
        ));
    }

    #[test]
    fn test_protection_body_field_fallback() {
        let mut csrf = CsrfProtection::double_submit(CsrfCookieConfig::default());
        let token = generate_token_from_seed(99);
        let request = CsrfRequest {
            method: "POST".into(),
            origin: None,
            referer: None,
            csrf_cookie: Some(token.clone()),
            csrf_header: None,
            csrf_body_field: Some(token),
            session_id: None,
        };
        assert!(csrf.validate(&request, 1000).is_ok());
    }

    #[test]
    fn test_error_display() {
        let e = CsrfError::TokenMissing;
        assert_eq!(e.to_string(), "CSRF token missing");
        let e = CsrfError::OriginMismatch {
            expected: "https://a.com".into(),
            actual: "https://b.com".into(),
        };
        assert!(e.to_string().contains("https://a.com"));
    }
}
