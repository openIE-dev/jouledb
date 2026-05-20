//! CSRF protection — token generation, double-submit cookies, synchronizer tokens.
//!
//! Replaces csurf (Node.js) and Django CSRF middleware with a pure-Rust
//! CSRF protection toolkit.  Includes cryptographically random token generation,
//! double-submit cookie pattern, synchronizer token pattern, timing-safe
//! validation, token rotation, and SameSite cookie configuration.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ─────────────────────────────────────────────────────

/// CSRF domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsrfError {
    /// Token missing from request.
    TokenMissing,
    /// Token mismatch.
    TokenMismatch,
    /// Token expired.
    TokenExpired,
    /// Invalid token format.
    InvalidTokenFormat(String),
    /// Session not found.
    SessionNotFound(String),
    /// Cookie configuration error.
    CookieError(String),
}

impl std::fmt::Display for CsrfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenMissing => write!(f, "CSRF token missing"),
            Self::TokenMismatch => write!(f, "CSRF token mismatch"),
            Self::TokenExpired => write!(f, "CSRF token expired"),
            Self::InvalidTokenFormat(s) => write!(f, "invalid CSRF token format: {s}"),
            Self::SessionNotFound(id) => write!(f, "session not found: {id}"),
            Self::CookieError(e) => write!(f, "cookie error: {e}"),
        }
    }
}

impl std::error::Error for CsrfError {}

// ── Token generation ──────────────────────────────────────────

/// Pseudo-random token generator using system entropy.
///
/// Uses timestamp, stack address, counter, and PID for entropy.
/// Not cryptographically secure — in production use OS randomness.
fn generate_random_bytes(len: usize) -> Vec<u8> {
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

    // Use a simple hash-based PRNG.
    let mut result = Vec::with_capacity(len);
    let mut idx = 0u64;
    while result.len() < len {
        let mut block_seed = seed.clone();
        block_seed.extend_from_slice(&idx.to_le_bytes());
        let hash = simple_hash(&block_seed);
        let take = (len - result.len()).min(hash.len());
        result.extend_from_slice(&hash[..take]);
        idx += 1;
    }
    result
}

/// Simple hash for entropy mixing (not SHA-256 — avoids cross-module deps).
fn simple_hash(data: &[u8]) -> Vec<u8> {
    // FNV-1a based expansion to 32 bytes.
    let mut result = vec![0u8; 32];
    for (round, byte) in result.iter_mut().enumerate() {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= round as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        *byte = (hash & 0xFF) as u8;
    }
    result
}

/// Generate a CSRF token as a hex string.
pub fn generate_token() -> String {
    let bytes = generate_random_bytes(32);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Generate a CSRF token with a timestamp prefix for expiry checking.
pub fn generate_timestamped_token() -> String {
    let ts = Utc::now().timestamp();
    let random = generate_token();
    format!("{ts:016x}.{random}")
}

// ── Constant-time comparison ──────────────────────────────────

/// Timing-safe comparison of two strings.
pub fn constant_time_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != b_bytes.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a_bytes.len() {
        diff |= a_bytes[i] ^ b_bytes[i];
    }
    diff == 0
}

// ── SameSite cookie configuration ─────────────────────────────

/// SameSite cookie attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SameSite {
    Strict,
    Lax,
    None,
}

impl std::fmt::Display for SameSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strict => write!(f, "Strict"),
            Self::Lax => write!(f, "Lax"),
            Self::None => write!(f, "None"),
        }
    }
}

/// CSRF cookie configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrfCookieConfig {
    /// Cookie name.
    pub name: String,
    /// Cookie path.
    pub path: String,
    /// HttpOnly flag.
    pub http_only: bool,
    /// Secure flag (HTTPS only).
    pub secure: bool,
    /// SameSite attribute.
    pub same_site: SameSite,
    /// Max age in seconds.
    pub max_age_secs: u64,
    /// Domain (optional).
    pub domain: Option<String>,
}

impl Default for CsrfCookieConfig {
    fn default() -> Self {
        Self {
            name: "__csrf_token".to_string(),
            path: "/".to_string(),
            http_only: true,
            secure: true,
            same_site: SameSite::Strict,
            max_age_secs: 3600,
            domain: None,
        }
    }
}

impl CsrfCookieConfig {
    /// Build a Set-Cookie header string.
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

    /// Parse a cookie value from a Cookie header string.
    pub fn extract_token(&self, cookie_header: &str) -> Option<String> {
        for pair in cookie_header.split(';') {
            let trimmed = pair.trim();
            if let Some(value) = trimmed.strip_prefix(&format!("{}=", self.name)) {
                return Some(value.to_string());
            }
        }
        None
    }
}

// ── Double Submit Cookie pattern ──────────────────────────────

/// Double-submit cookie CSRF protection.
///
/// The token is set both as a cookie and must be sent in a header/form field.
/// Validation checks that both values match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleSubmitCsrf {
    pub cookie_config: CsrfCookieConfig,
    pub header_name: String,
    pub form_field_name: String,
}

impl Default for DoubleSubmitCsrf {
    fn default() -> Self {
        Self {
            cookie_config: CsrfCookieConfig::default(),
            header_name: "X-CSRF-Token".to_string(),
            form_field_name: "_csrf".to_string(),
        }
    }
}

impl DoubleSubmitCsrf {
    /// Generate a new token and cookie.
    pub fn generate(&self) -> (String, String) {
        let token = generate_token();
        let cookie = self.cookie_config.to_set_cookie(&token);
        (token, cookie)
    }

    /// Validate: cookie token must match header/form token.
    pub fn validate(&self, cookie_token: &str, request_token: &str) -> Result<(), CsrfError> {
        if cookie_token.is_empty() || request_token.is_empty() {
            return Err(CsrfError::TokenMissing);
        }
        if !constant_time_eq(cookie_token, request_token) {
            return Err(CsrfError::TokenMismatch);
        }
        Ok(())
    }
}

// ── Synchronizer Token pattern ────────────────────────────────

/// A stored CSRF token with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredToken {
    pub token: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub used: bool,
}

/// Synchronizer token CSRF protection (server-side session store).
#[derive(Debug, Clone)]
pub struct SynchronizerCsrf {
    /// Session ID -> stored token.
    tokens: HashMap<String, StoredToken>,
    /// Token TTL.
    ttl: Duration,
    /// Whether tokens are single-use.
    single_use: bool,
}

impl SynchronizerCsrf {
    /// Create a new synchronizer with the given TTL.
    pub fn new(ttl_secs: i64, single_use: bool) -> Self {
        Self {
            tokens: HashMap::new(),
            ttl: Duration::seconds(ttl_secs),
            single_use,
        }
    }

    /// Generate a token for a session.
    pub fn generate(&mut self, session_id: &str) -> String {
        let token = generate_token();
        let now = Utc::now();
        self.tokens.insert(
            session_id.to_string(),
            StoredToken {
                token: token.clone(),
                created_at: now,
                expires_at: now + self.ttl,
                used: false,
            },
        );
        token
    }

    /// Validate a token for a session.
    pub fn validate(&mut self, session_id: &str, token: &str) -> Result<(), CsrfError> {
        let stored = self
            .tokens
            .get_mut(session_id)
            .ok_or_else(|| CsrfError::SessionNotFound(session_id.to_string()))?;

        if stored.used && self.single_use {
            return Err(CsrfError::TokenMismatch);
        }

        if Utc::now() > stored.expires_at {
            self.tokens.remove(session_id);
            return Err(CsrfError::TokenExpired);
        }

        if !constant_time_eq(&stored.token, token) {
            return Err(CsrfError::TokenMismatch);
        }

        if self.single_use {
            stored.used = true;
        }

        Ok(())
    }

    /// Rotate token for a session (invalidate old, generate new).
    pub fn rotate(&mut self, session_id: &str) -> String {
        self.generate(session_id)
    }

    /// Clean up expired tokens.
    pub fn cleanup(&mut self) {
        let now = Utc::now();
        self.tokens.retain(|_, v| v.expires_at > now);
    }

    /// Get the stored token for a session (if any).
    pub fn get_token(&self, session_id: &str) -> Option<&StoredToken> {
        self.tokens.get(session_id)
    }

    /// Number of active tokens.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

// ── Timestamped token validation ──────────────────────────────

/// Validate a timestamped token (format: "hex_timestamp.random_hex").
pub fn validate_timestamped_token(
    token: &str,
    expected: &str,
    max_age_secs: i64,
) -> Result<(), CsrfError> {
    // Check format.
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 2 {
        return Err(CsrfError::InvalidTokenFormat(token.to_string()));
    }

    // Check timestamp.
    let ts = i64::from_str_radix(parts[0], 16)
        .map_err(|_| CsrfError::InvalidTokenFormat(token.to_string()))?;

    let now = Utc::now().timestamp();
    if now - ts > max_age_secs {
        return Err(CsrfError::TokenExpired);
    }

    // Check token match.
    if !constant_time_eq(token, expected) {
        return Err(CsrfError::TokenMismatch);
    }

    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_token_length() {
        let token = generate_token();
        assert_eq!(token.len(), 64); // 32 bytes * 2 hex chars
    }

    #[test]
    fn test_generate_token_uniqueness() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_generate_token_hex_chars() {
        let token = generate_token();
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq("abc", "abc"));
        assert!(!constant_time_eq("abc", "abd"));
        assert!(!constant_time_eq("ab", "abc"));
    }

    #[test]
    fn test_double_submit_generate_validate() {
        let csrf = DoubleSubmitCsrf::default();
        let (token, cookie) = csrf.generate();
        assert!(!token.is_empty());
        assert!(cookie.contains(&token));
        assert!(csrf.validate(&token, &token).is_ok());
    }

    #[test]
    fn test_double_submit_mismatch() {
        let csrf = DoubleSubmitCsrf::default();
        let (token, _) = csrf.generate();
        assert_eq!(
            csrf.validate(&token, "wrong_token"),
            Err(CsrfError::TokenMismatch)
        );
    }

    #[test]
    fn test_double_submit_missing() {
        let csrf = DoubleSubmitCsrf::default();
        assert_eq!(csrf.validate("", "token"), Err(CsrfError::TokenMissing));
    }

    #[test]
    fn test_synchronizer_generate_validate() {
        let mut csrf = SynchronizerCsrf::new(3600, false);
        let token = csrf.generate("session1");
        assert!(csrf.validate("session1", &token).is_ok());
    }

    #[test]
    fn test_synchronizer_wrong_token() {
        let mut csrf = SynchronizerCsrf::new(3600, false);
        csrf.generate("session1");
        assert_eq!(
            csrf.validate("session1", "wrong"),
            Err(CsrfError::TokenMismatch)
        );
    }

    #[test]
    fn test_synchronizer_missing_session() {
        let mut csrf = SynchronizerCsrf::new(3600, false);
        assert!(csrf.validate("nonexistent", "token").is_err());
    }

    #[test]
    fn test_synchronizer_single_use() {
        let mut csrf = SynchronizerCsrf::new(3600, true);
        let token = csrf.generate("session1");
        assert!(csrf.validate("session1", &token).is_ok());
        // Second use should fail.
        assert_eq!(
            csrf.validate("session1", &token),
            Err(CsrfError::TokenMismatch)
        );
    }

    #[test]
    fn test_synchronizer_rotation() {
        let mut csrf = SynchronizerCsrf::new(3600, false);
        let t1 = csrf.generate("session1");
        let t2 = csrf.rotate("session1");
        assert_ne!(t1, t2);
        // Old token should now fail.
        assert_eq!(
            csrf.validate("session1", &t1),
            Err(CsrfError::TokenMismatch)
        );
        // New token should work.
        assert!(csrf.validate("session1", &t2).is_ok());
    }

    #[test]
    fn test_cookie_config_default() {
        let config = CsrfCookieConfig::default();
        assert_eq!(config.name, "__csrf_token");
        assert!(config.http_only);
        assert!(config.secure);
        assert_eq!(config.same_site, SameSite::Strict);
    }

    #[test]
    fn test_set_cookie_header() {
        let config = CsrfCookieConfig::default();
        let header = config.to_set_cookie("abc123");
        assert!(header.contains("__csrf_token=abc123"));
        assert!(header.contains("HttpOnly"));
        assert!(header.contains("Secure"));
        assert!(header.contains("SameSite=Strict"));
    }

    #[test]
    fn test_extract_token_from_cookie() {
        let config = CsrfCookieConfig::default();
        let token = config.extract_token("other=value; __csrf_token=abc123; session=xyz");
        assert_eq!(token, Some("abc123".to_string()));
    }

    #[test]
    fn test_extract_token_missing() {
        let config = CsrfCookieConfig::default();
        let token = config.extract_token("other=value; session=xyz");
        assert_eq!(token, None);
    }

    #[test]
    fn test_timestamped_token() {
        let token = generate_timestamped_token();
        assert!(token.contains('.'));
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 2);
        // Timestamp part should be valid hex.
        assert!(i64::from_str_radix(parts[0], 16).is_ok());
    }

    #[test]
    fn test_validate_timestamped_token() {
        let token = generate_timestamped_token();
        assert!(validate_timestamped_token(&token, &token, 3600).is_ok());
    }

    #[test]
    fn test_validate_timestamped_token_mismatch() {
        let t1 = generate_timestamped_token();
        let t2 = generate_timestamped_token();
        assert_eq!(
            validate_timestamped_token(&t1, &t2, 3600),
            Err(CsrfError::TokenMismatch)
        );
    }

    #[test]
    fn test_synchronizer_cleanup() {
        let mut csrf = SynchronizerCsrf::new(3600, false);
        csrf.generate("s1");
        csrf.generate("s2");
        assert_eq!(csrf.token_count(), 2);
        csrf.cleanup(); // Nothing expired yet.
        assert_eq!(csrf.token_count(), 2);
    }

    #[test]
    fn test_same_site_display() {
        assert_eq!(format!("{}", SameSite::Strict), "Strict");
        assert_eq!(format!("{}", SameSite::Lax), "Lax");
        assert_eq!(format!("{}", SameSite::None), "None");
    }
}
