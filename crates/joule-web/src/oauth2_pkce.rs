//! OAuth2 Authorization Code flow with PKCE (RFC 7636).
//!
//! Replaces `oidc-client` / `oauth4webapi` with a pure Rust implementation.
//! Handles code verifier/challenge generation, authorization requests,
//! token exchange, refresh flow, and CSRF state parameter protection.

use crate::crypto::{sha256, base64_encode};
use std::fmt;
use std::time::Duration;

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthError {
    InvalidVerifierLength,
    InvalidState,
    TokenExpired,
    MissingField(String),
    InvalidGrant(String),
}

impl fmt::Display for OAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVerifierLength => write!(f, "verifier must be 43-128 characters"),
            Self::InvalidState => write!(f, "state parameter mismatch (possible CSRF)"),
            Self::TokenExpired => write!(f, "token has expired"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::InvalidGrant(msg) => write!(f, "invalid grant: {msg}"),
        }
    }
}

impl std::error::Error for OAuthError {}

// ── Base64url ──────────────────────────────────────────────────

/// Base64url encode (RFC 4648 section 5, no padding).
fn base64url_encode(input: &[u8]) -> String {
    base64_encode(input)
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_string()
}

// ── Code Verifier / Challenge ──────────────────────────────────

/// Unreserved characters allowed in a PKCE code verifier (RFC 7636 4.1).
const UNRESERVED: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

/// A PKCE code verifier (43-128 characters from the unreserved set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeVerifier(String);

impl CodeVerifier {
    /// Create a verifier from an existing string (validates charset and length).
    pub fn from_string(s: impl Into<String>) -> Result<Self, OAuthError> {
        let s = s.into();
        if s.len() < 43 || s.len() > 128 {
            return Err(OAuthError::InvalidVerifierLength);
        }
        if !s.bytes().all(|b| UNRESERVED.contains(&b)) {
            return Err(OAuthError::InvalidVerifierLength);
        }
        Ok(Self(s))
    }

    /// Generate a verifier from seed bytes (deterministic, for testing).
    /// Takes arbitrary bytes and base64url-encodes them, then truncates to desired length.
    pub fn from_bytes(seed: &[u8]) -> Result<Self, OAuthError> {
        let encoded = base64url_encode(seed);
        // Ensure we have at least 43 chars; pad by hashing if needed.
        let value = if encoded.len() < 43 {
            let hash = sha256(seed);
            let extra = base64url_encode(&hash);
            let combined = format!("{encoded}{extra}");
            combined[..43.min(combined.len())].to_string()
        } else if encoded.len() > 128 {
            encoded[..128].to_string()
        } else {
            encoded
        };

        if value.len() < 43 {
            return Err(OAuthError::InvalidVerifierLength);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Compute the S256 code challenge: base64url(sha256(verifier)).
    pub fn challenge_s256(&self) -> CodeChallenge {
        let hash = sha256(self.0.as_bytes());
        CodeChallenge {
            value: base64url_encode(&hash),
            method: ChallengeMethod::S256,
        }
    }

    /// Plain challenge (not recommended, but part of the spec).
    pub fn challenge_plain(&self) -> CodeChallenge {
        CodeChallenge {
            value: self.0.clone(),
            method: ChallengeMethod::Plain,
        }
    }
}

/// PKCE challenge method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeMethod {
    Plain,
    S256,
}

impl fmt::Display for ChallengeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain => write!(f, "plain"),
            Self::S256 => write!(f, "S256"),
        }
    }
}

/// A code challenge derived from a verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeChallenge {
    pub value: String,
    pub method: ChallengeMethod,
}

// ── Authorization Request ──────────────────────────────────────

/// An OAuth2 authorization request.
#[derive(Debug, Clone)]
pub struct AuthorizationRequest {
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub state: String,
    pub code_challenge: CodeChallenge,
    pub response_type: String,
    pub extra_params: Vec<(String, String)>,
}

impl AuthorizationRequest {
    pub fn new(
        client_id: impl Into<String>,
        redirect_uri: impl Into<String>,
        scope: impl Into<String>,
        state: impl Into<String>,
        code_challenge: CodeChallenge,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            redirect_uri: redirect_uri.into(),
            scope: scope.into(),
            state: state.into(),
            code_challenge,
            response_type: "code".into(),
            extra_params: Vec::new(),
        }
    }

    pub fn extra_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_params.push((key.into(), value.into()));
        self
    }

    /// Build the authorization URL.
    pub fn to_url(&self, authorize_endpoint: &str) -> String {
        let mut params = vec![
            ("response_type", self.response_type.as_str()),
            ("client_id", self.client_id.as_str()),
            ("redirect_uri", self.redirect_uri.as_str()),
            ("scope", self.scope.as_str()),
            ("state", self.state.as_str()),
            ("code_challenge", self.code_challenge.value.as_str()),
            ("code_challenge_method", match self.code_challenge.method {
                ChallengeMethod::Plain => "plain",
                ChallengeMethod::S256 => "S256",
            }),
        ];

        // Collect references to extra params.
        let extra_refs: Vec<(&str, &str)> = self
            .extra_params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        params.extend(extra_refs);

        let query = params
            .iter()
            .map(|(k, v)| format!("{k}={}", url_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let sep = if authorize_endpoint.contains('?') { "&" } else { "?" };
        format!("{authorize_endpoint}{sep}{query}")
    }
}

/// Minimal URL encoding.
fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

// ── Token Request ──────────────────────────────────────────────

/// A token exchange request.
#[derive(Debug, Clone)]
pub struct TokenRequest {
    pub grant_type: String,
    pub code: String,
    pub redirect_uri: String,
    pub client_id: String,
    pub code_verifier: String,
}

impl TokenRequest {
    pub fn authorization_code(
        code: impl Into<String>,
        redirect_uri: impl Into<String>,
        client_id: impl Into<String>,
        verifier: &CodeVerifier,
    ) -> Self {
        Self {
            grant_type: "authorization_code".into(),
            code: code.into(),
            redirect_uri: redirect_uri.into(),
            client_id: client_id.into(),
            code_verifier: verifier.as_str().to_string(),
        }
    }

    /// Encode as form parameters (application/x-www-form-urlencoded).
    pub fn to_form_body(&self) -> String {
        format!(
            "grant_type={}&code={}&redirect_uri={}&client_id={}&code_verifier={}",
            url_encode(&self.grant_type),
            url_encode(&self.code),
            url_encode(&self.redirect_uri),
            url_encode(&self.client_id),
            url_encode(&self.code_verifier),
        )
    }
}

// ── Token Response ─────────────────────────────────────────────

/// A token response from the authorization server.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
}

impl TokenResponse {
    /// Check if the token is expired given the time since it was issued.
    pub fn is_expired(&self, elapsed: Duration) -> bool {
        match self.expires_in {
            Some(exp) => elapsed.as_secs() >= exp,
            None => false, // No expiration.
        }
    }
}

// ── Token Refresh ──────────────────────────────────────────────

/// A refresh token request.
#[derive(Debug, Clone)]
pub struct RefreshTokenRequest {
    pub grant_type: String,
    pub refresh_token: String,
    pub client_id: String,
    pub scope: Option<String>,
}

impl RefreshTokenRequest {
    pub fn new(
        refresh_token: impl Into<String>,
        client_id: impl Into<String>,
    ) -> Self {
        Self {
            grant_type: "refresh_token".into(),
            refresh_token: refresh_token.into(),
            client_id: client_id.into(),
            scope: None,
        }
    }

    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    pub fn to_form_body(&self) -> String {
        let mut body = format!(
            "grant_type={}&refresh_token={}&client_id={}",
            url_encode(&self.grant_type),
            url_encode(&self.refresh_token),
            url_encode(&self.client_id),
        );
        if let Some(scope) = &self.scope {
            body.push_str(&format!("&scope={}", url_encode(scope)));
        }
        body
    }
}

// ── State CSRF Protection ──────────────────────────────────────

/// Validate the state parameter returned from the authorization server.
pub fn validate_state(expected: &str, received: &str) -> Result<(), OAuthError> {
    if expected != received {
        Err(OAuthError::InvalidState)
    } else {
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_verifier() -> CodeVerifier {
        // 43+ chars from unreserved set.
        CodeVerifier::from_string(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstu"
        ).unwrap()
    }

    #[test]
    fn verifier_valid_length() {
        let v = make_verifier();
        assert!(v.as_str().len() >= 43);
    }

    #[test]
    fn verifier_too_short() {
        let result = CodeVerifier::from_string("short");
        assert_eq!(result, Err(OAuthError::InvalidVerifierLength));
    }

    #[test]
    fn verifier_too_long() {
        let long = "A".repeat(129);
        let result = CodeVerifier::from_string(long);
        assert_eq!(result, Err(OAuthError::InvalidVerifierLength));
    }

    #[test]
    fn verifier_invalid_chars() {
        let with_space = format!("ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqr");
        let result = CodeVerifier::from_string(with_space);
        assert!(result.is_err());
    }

    #[test]
    fn challenge_s256() {
        let verifier = make_verifier();
        let challenge = verifier.challenge_s256();
        assert_eq!(challenge.method, ChallengeMethod::S256);
        // S256 challenge should be base64url(sha256(verifier)), which is 43 chars.
        assert_eq!(challenge.value.len(), 43);
        // No padding.
        assert!(!challenge.value.contains('='));
        // No standard base64 chars.
        assert!(!challenge.value.contains('+'));
        assert!(!challenge.value.contains('/'));
    }

    #[test]
    fn challenge_plain() {
        let verifier = make_verifier();
        let challenge = verifier.challenge_plain();
        assert_eq!(challenge.method, ChallengeMethod::Plain);
        assert_eq!(challenge.value, verifier.as_str());
    }

    #[test]
    fn authorization_url() {
        let verifier = make_verifier();
        let challenge = verifier.challenge_s256();
        let req = AuthorizationRequest::new(
            "my-client",
            "https://app.example.com/callback",
            "openid profile",
            "random-state-123",
            challenge,
        );
        let url = req.to_url("https://auth.example.com/authorize");
        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("client_id=my-client"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=random-state-123"));
        assert!(url.contains("redirect_uri="));
    }

    #[test]
    fn token_request_form() {
        let verifier = make_verifier();
        let req = TokenRequest::authorization_code(
            "auth-code-xyz",
            "https://app.example.com/callback",
            "my-client",
            &verifier,
        );
        let body = req.to_form_body();
        assert!(body.contains("grant_type=authorization_code"));
        assert!(body.contains("code=auth-code-xyz"));
        assert!(body.contains("code_verifier="));
    }

    #[test]
    fn token_response_expiry() {
        let token = TokenResponse {
            access_token: "access-abc".into(),
            token_type: "Bearer".into(),
            expires_in: Some(3600),
            refresh_token: Some("refresh-xyz".into()),
            scope: Some("openid".into()),
        };
        assert!(!token.is_expired(Duration::from_secs(100)));
        assert!(token.is_expired(Duration::from_secs(3600)));
        assert!(token.is_expired(Duration::from_secs(7200)));
    }

    #[test]
    fn refresh_token_request() {
        let req = RefreshTokenRequest::new("refresh-xyz", "my-client")
            .scope("openid profile");
        let body = req.to_form_body();
        assert!(body.contains("grant_type=refresh_token"));
        assert!(body.contains("refresh_token=refresh-xyz"));
        assert!(body.contains("scope=openid%20profile"));
    }

    #[test]
    fn state_validation() {
        assert!(validate_state("abc", "abc").is_ok());
        assert_eq!(validate_state("abc", "xyz"), Err(OAuthError::InvalidState));
    }

    #[test]
    fn verifier_from_bytes() {
        let seed = b"some-random-seed-bytes-that-are-long-enough-for-testing-purposes!!";
        let verifier = CodeVerifier::from_bytes(seed).unwrap();
        assert!(verifier.as_str().len() >= 43);
        assert!(verifier.as_str().len() <= 128);
    }

    #[test]
    fn url_encoding() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a+b=c"), "a%2Bb%3Dc");
        assert_eq!(url_encode("simple"), "simple");
    }
}
