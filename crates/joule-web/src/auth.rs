//! Authentication flows — OAuth2, JWT decode, and session management.
//!
//! Replaces Auth0, NextAuth, and jose with protocol-level logic in pure Rust.
//! No HTTP calls — only constructs requests and parses tokens.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Authentication errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// JWT has wrong number of segments.
    MalformedToken,
    /// Base64 decoding failed.
    Base64DecodeError,
    /// JSON parsing failed.
    JsonParseError(String),
    /// Token is expired.
    TokenExpired,
    /// Invalid state parameter (CSRF mismatch).
    InvalidState,
    /// Session not found.
    SessionNotFound,
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedToken => write!(f, "malformed JWT token"),
            Self::Base64DecodeError => write!(f, "base64url decode error"),
            Self::JsonParseError(msg) => write!(f, "JSON parse error: {msg}"),
            Self::TokenExpired => write!(f, "token expired"),
            Self::InvalidState => write!(f, "invalid OAuth2 state"),
            Self::SessionNotFound => write!(f, "session not found"),
        }
    }
}

impl std::error::Error for AuthError {}

// ── Base64URL ───────────────────────────────────────────────────

/// Decode a base64url string (no padding) to bytes.
pub fn base64url_decode(input: &str) -> Result<Vec<u8>, AuthError> {
    // Replace URL-safe chars with standard base64
    let standard: String = input
        .chars()
        .map(|c| match c {
            '-' => '+',
            '_' => '/',
            other => other,
        })
        .collect();

    // Add padding
    let padded = match standard.len() % 4 {
        2 => format!("{standard}=="),
        3 => format!("{standard}="),
        0 => standard,
        _ => return Err(AuthError::Base64DecodeError),
    };

    // Decode
    base64_decode_standard(&padded).map_err(|_| AuthError::Base64DecodeError)
}

/// Encode bytes as base64url (no padding).
pub fn base64url_encode(input: &[u8]) -> String {
    let standard = base64_encode_standard(input);
    standard
        .trim_end_matches('=')
        .chars()
        .map(|c| match c {
            '+' => '-',
            '/' => '_',
            other => other,
        })
        .collect()
}

// Simple base64 implementation (standard alphabet)
fn base64_encode_standard(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i + 2 < input.len() {
        let n =
            ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push(CHARS[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let remaining = input.len() - i;
    if remaining == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    } else if remaining == 1 {
        let n = (input[i] as u32) << 16;
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    }
    out
}

fn base64_decode_standard(input: &str) -> Result<Vec<u8>, ()> {
    let input = input.trim_end_matches('=');
    let mut out = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for c in input.chars() {
        let val = match c {
            'A'..='Z' => c as u8 - b'A',
            'a'..='z' => c as u8 - b'a' + 26,
            '0'..='9' => c as u8 - b'0' + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(()),
        };
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

// ── OAuth2 ──────────────────────────────────────────────────────

/// OAuth2 client configuration.
#[derive(Debug, Clone)]
pub struct OAuth2Config {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
    pub code_challenge_method: Option<String>,
}

/// OAuth2 state for CSRF protection and PKCE.
#[derive(Debug, Clone)]
pub struct OAuth2State {
    pub state: String,
    pub code_verifier: Option<String>,
    pub nonce: Option<String>,
}

/// Generate a PKCE code verifier (43-128 char URL-safe string).
pub fn generate_code_verifier() -> String {
    // Use a deterministic but unique approach: counter-based
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0x5a3c_9e71_f284_6db0);
    let seed = CTR.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);

    let url_safe = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut state = seed;
    let mut verifier = String::with_capacity(64);
    for _ in 0..64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        verifier.push(url_safe[(state as usize) % url_safe.len()] as char);
    }
    verifier
}

/// Generate a code challenge from a verifier.
/// Uses a simple hash (not real SHA-256) for the challenge since we have
/// our crypto module available.
pub fn generate_code_challenge(verifier: &str) -> String {
    // Use our SHA-256 and base64url
    let hash = crate::crypto::sha256(verifier.as_bytes());
    base64url_encode(&hash)
}

/// Token exchange request (ready for HTTP POST).
#[derive(Debug, Clone)]
pub struct TokenRequest {
    pub url: String,
    pub body: HashMap<String, String>,
    pub method: String,
}

/// Token response from the authorization server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
    pub id_token: Option<String>,
}

/// OAuth2 authorization flow helpers.
pub struct OAuth2Flow;

impl OAuth2Flow {
    /// Build the authorization URL and generate CSRF state.
    pub fn build_auth_url(config: &OAuth2Config) -> (String, OAuth2State) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static STATE_CTR: AtomicU64 = AtomicU64::new(0xa1b2_c3d4_e5f6_0718);
        let seed = STATE_CTR.fetch_add(0x1234_5678_9abc_def0, Ordering::Relaxed);

        // Generate state token
        let state_token = format!("{seed:016x}");

        let mut url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&state={state_token}",
            config.auth_url,
            url_encode(&config.client_id),
            url_encode(&config.redirect_uri),
        );

        if !config.scopes.is_empty() {
            url.push_str(&format!("&scope={}", url_encode(&config.scopes.join(" "))));
        }

        let (code_verifier, nonce) =
            if config.code_challenge_method.as_deref() == Some("S256") {
                let verifier = generate_code_verifier();
                let challenge = generate_code_challenge(&verifier);
                url.push_str(&format!("&code_challenge={challenge}&code_challenge_method=S256"));
                (Some(verifier), None)
            } else {
                (None, None)
            };

        let oauth_state = OAuth2State {
            state: state_token,
            code_verifier,
            nonce,
        };

        (url, oauth_state)
    }

    /// Build the token exchange request from an authorization code.
    pub fn build_token_request(
        config: &OAuth2Config,
        code: &str,
        state: &OAuth2State,
    ) -> TokenRequest {
        let mut body = HashMap::new();
        body.insert("grant_type".to_string(), "authorization_code".to_string());
        body.insert("code".to_string(), code.to_string());
        body.insert("redirect_uri".to_string(), config.redirect_uri.clone());
        body.insert("client_id".to_string(), config.client_id.clone());

        if let Some(ref verifier) = state.code_verifier {
            body.insert("code_verifier".to_string(), verifier.clone());
        }

        TokenRequest {
            url: config.token_url.clone(),
            body,
            method: "POST".to_string(),
        }
    }
}

/// Minimal URL encoding for query parameters.
fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push(crate::crypto::hex_encode(&[b]).chars().next().unwrap());
                out.push(crate::crypto::hex_encode(&[b]).chars().nth(1).unwrap());
            }
        }
    }
    out
}

// ── JWT ─────────────────────────────────────────────────────────

/// JWT header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtHeader {
    pub alg: String,
    #[serde(default = "default_typ")]
    pub typ: String,
    pub kid: Option<String>,
}

fn default_typ() -> String {
    "JWT".to_string()
}

/// JWT payload (claims).
#[derive(Debug, Clone)]
pub struct JwtPayload {
    pub claims: HashMap<String, serde_json::Value>,
}

impl JwtPayload {
    /// Get the `sub` claim.
    pub fn sub(&self) -> Option<&str> {
        self.claims.get("sub").and_then(|v| v.as_str())
    }
    /// Get the `iss` claim.
    pub fn iss(&self) -> Option<&str> {
        self.claims.get("iss").and_then(|v| v.as_str())
    }
    /// Get the `aud` claim.
    pub fn aud(&self) -> Option<&str> {
        self.claims.get("aud").and_then(|v| v.as_str())
    }
    /// Get the `exp` claim.
    pub fn exp(&self) -> Option<i64> {
        self.claims.get("exp").and_then(|v| v.as_i64())
    }
    /// Get the `iat` claim.
    pub fn iat(&self) -> Option<i64> {
        self.claims.get("iat").and_then(|v| v.as_i64())
    }
    /// Get an arbitrary claim.
    pub fn claim(&self, name: &str) -> Option<&serde_json::Value> {
        self.claims.get(name)
    }
    /// Check if the token is expired relative to `now` (unix timestamp).
    pub fn is_expired(&self, now: i64) -> bool {
        match self.exp() {
            Some(exp) => now >= exp,
            None => false,
        }
    }
}

/// Decode a JWT token (header + payload). Does NOT verify the signature.
pub fn decode_jwt(token: &str) -> Result<(JwtHeader, JwtPayload), AuthError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(AuthError::MalformedToken);
    }

    let header_bytes = base64url_decode(parts[0])?;
    let header_json =
        String::from_utf8(header_bytes).map_err(|e| AuthError::JsonParseError(e.to_string()))?;
    let header: JwtHeader =
        serde_json::from_str(&header_json).map_err(|e| AuthError::JsonParseError(e.to_string()))?;

    let payload_bytes = base64url_decode(parts[1])?;
    let payload_json =
        String::from_utf8(payload_bytes).map_err(|e| AuthError::JsonParseError(e.to_string()))?;
    let claims: HashMap<String, serde_json::Value> =
        serde_json::from_str(&payload_json).map_err(|e| AuthError::JsonParseError(e.to_string()))?;

    Ok((header, JwtPayload { claims }))
}

// ── Session Management ──────────────────────────────────────────

/// A user session.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub data: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// In-memory session manager.
pub struct SessionManager {
    sessions: HashMap<String, Session>,
    token_counter: u64,
    max_age_seconds: u64,
}

impl SessionManager {
    /// Create a new session manager with the given max session age in seconds.
    pub fn new(max_age: u64) -> Self {
        Self {
            sessions: HashMap::new(),
            token_counter: 0,
            max_age_seconds: max_age,
        }
    }

    /// Create a new session for a user. Returns a reference to the created session.
    pub fn create(&mut self, user_id: &str) -> &Session {
        self.token_counter += 1;
        let id = format!("sess_{:016x}", self.token_counter);
        let now = Utc::now();
        let expires_at = now + Duration::seconds(self.max_age_seconds as i64);
        let session = Session {
            id: id.clone(),
            user_id: user_id.to_string(),
            data: HashMap::new(),
            created_at: now,
            last_accessed: now,
            expires_at,
        };
        self.sessions.insert(id.clone(), session);
        self.sessions.get(&id).unwrap()
    }

    /// Get a session by ID.
    pub fn get(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }

    /// Get a mutable reference to a session.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(id)
    }

    /// Update last_accessed and extend expires_at.
    pub fn touch(&mut self, id: &str) {
        if let Some(session) = self.sessions.get_mut(id) {
            let now = Utc::now();
            session.last_accessed = now;
            session.expires_at = now + Duration::seconds(self.max_age_seconds as i64);
        }
    }

    /// Destroy a session. Returns true if it existed.
    pub fn destroy(&mut self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    /// Remove all expired sessions. Returns the count removed.
    pub fn cleanup_expired(&mut self, now: &DateTime<Utc>) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|_, s| s.expires_at > *now);
        before - self.sessions.len()
    }

    /// Set a data key on a session. Returns false if session not found.
    pub fn set_data(&mut self, session_id: &str, key: &str, value: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.data.insert(key.to_string(), value.to_string());
            true
        } else {
            false
        }
    }

    /// Count of active sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_oauth2_config() -> OAuth2Config {
        OAuth2Config {
            client_id: "my-app".to_string(),
            auth_url: "https://auth.example.com/authorize".to_string(),
            token_url: "https://auth.example.com/token".to_string(),
            redirect_uri: "https://app.example.com/callback".to_string(),
            scopes: vec!["openid".to_string(), "profile".to_string()],
            code_challenge_method: None,
        }
    }

    #[test]
    fn build_auth_url_includes_params() {
        let config = test_oauth2_config();
        let (url, state) = OAuth2Flow::build_auth_url(&config);
        assert!(url.contains("client_id=my-app"));
        assert!(url.contains("redirect_uri="));
        assert!(url.contains("scope=openid"));
        assert!(url.contains(&format!("state={}", state.state)));
    }

    #[test]
    fn pkce_verifier_correct_length() {
        let verifier = generate_code_verifier();
        assert!(
            verifier.len() >= 43 && verifier.len() <= 128,
            "verifier len = {}",
            verifier.len()
        );
        // All chars should be URL-safe
        for c in verifier.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' || c == '~',
                "non-URL-safe char: {c}"
            );
        }
    }

    #[test]
    fn token_request_includes_code() {
        let config = test_oauth2_config();
        let (_, state) = OAuth2Flow::build_auth_url(&config);
        let req = OAuth2Flow::build_token_request(&config, "auth_code_123", &state);
        assert_eq!(req.method, "POST");
        assert_eq!(req.body["code"], "auth_code_123");
        assert_eq!(req.body["grant_type"], "authorization_code");
        assert_eq!(req.body["client_id"], "my-app");
    }

    #[test]
    fn decode_jwt_valid() {
        // Build a minimal JWT: {"alg":"HS256","typ":"JWT"}.{"sub":"1234","iss":"test","exp":9999999999}.<sig>
        let header = base64url_encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload =
            base64url_encode(br#"{"sub":"1234","iss":"test","exp":9999999999,"iat":1000000}"#);
        let token = format!("{header}.{payload}.fakesig");

        let (hdr, claims) = decode_jwt(&token).unwrap();
        assert_eq!(hdr.alg, "HS256");
        assert_eq!(hdr.typ, "JWT");
        assert_eq!(claims.sub(), Some("1234"));
        assert_eq!(claims.iss(), Some("test"));
        assert!(!claims.is_expired(1_000_000));
    }

    #[test]
    fn decode_jwt_expired() {
        let header = base64url_encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = base64url_encode(br#"{"sub":"user","exp":1000}"#);
        let token = format!("{header}.{payload}.sig");

        let (_, claims) = decode_jwt(&token).unwrap();
        assert!(claims.is_expired(2000));
        assert!(!claims.is_expired(500));
    }

    #[test]
    fn base64url_roundtrip() {
        let original = b"hello world! special chars: +/=";
        let encoded = base64url_encode(original);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn session_create_get() {
        let mut mgr = SessionManager::new(3600);
        let id = mgr.create("user1").id.clone();
        let session = mgr.get(&id).unwrap();
        assert_eq!(session.user_id, "user1");
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn session_touch_extends() {
        let mut mgr = SessionManager::new(3600);
        let id = mgr.create("user1").id.clone();
        let original_expires = mgr.get(&id).unwrap().expires_at;
        // Touch should update last_accessed and potentially extend expires_at
        mgr.touch(&id);
        let updated = mgr.get(&id).unwrap();
        assert!(updated.expires_at >= original_expires);
    }

    #[test]
    fn cleanup_expired_removes_old() {
        let mut mgr = SessionManager::new(1); // 1 second max age
        let _id = mgr.create("user1").id.clone();
        assert_eq!(mgr.active_count(), 1);
        // Simulate far future
        let future = Utc::now() + Duration::seconds(3600);
        let removed = mgr.cleanup_expired(&future);
        assert_eq!(removed, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn session_set_data() {
        let mut mgr = SessionManager::new(3600);
        let id = mgr.create("user1").id.clone();
        assert!(mgr.set_data(&id, "role", "admin"));
        assert_eq!(mgr.get(&id).unwrap().data["role"], "admin");
        assert!(!mgr.set_data("nonexistent", "k", "v"));
    }

    #[test]
    fn oauth2_state_in_url() {
        let config = test_oauth2_config();
        let (url, state) = OAuth2Flow::build_auth_url(&config);
        assert!(
            url.contains(&format!("state={}", state.state)),
            "URL must include state param"
        );
    }

    #[test]
    fn decode_jwt_malformed_returns_error() {
        assert!(decode_jwt("not.a.valid.jwt.token").is_err());
        assert!(decode_jwt("onlyone").is_err());
        assert!(decode_jwt("").is_err());
    }
}
