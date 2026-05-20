//! OAuth 2.0 flow simulation — authorization code flow, PKCE (code
//! challenge/verifier), token request/response, refresh token, scope
//! management, state parameter CSRF protection, and token introspection.
//!
//! Replaces `passport`, `oidc-client`, `simple-oauth2`, and similar JS OAuth
//! libraries with a pure-Rust OAuth 2.0 implementation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// OAuth flow error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthError {
    /// Invalid client credentials.
    InvalidClient(String),
    /// Invalid authorization code.
    InvalidCode(String),
    /// Invalid redirect URI.
    InvalidRedirectUri(String),
    /// Invalid or expired refresh token.
    InvalidRefreshToken(String),
    /// CSRF state mismatch.
    StateMismatch { expected: String, got: String },
    /// PKCE code verifier mismatch.
    PkceFailure(String),
    /// Invalid scope requested.
    InvalidScope(String),
    /// Token expired.
    TokenExpired { token_id: String },
    /// Token revoked.
    TokenRevoked { token_id: String },
    /// Unsupported grant type.
    UnsupportedGrantType(String),
    /// Authorization request not found.
    AuthRequestNotFound(String),
}

impl fmt::Display for OAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidClient(id) => write!(f, "invalid client: {id}"),
            Self::InvalidCode(code) => write!(f, "invalid authorization code: {code}"),
            Self::InvalidRedirectUri(uri) => write!(f, "invalid redirect URI: {uri}"),
            Self::InvalidRefreshToken(tok) => write!(f, "invalid refresh token: {tok}"),
            Self::StateMismatch { expected, got } => {
                write!(f, "state mismatch: expected {expected}, got {got}")
            }
            Self::PkceFailure(msg) => write!(f, "PKCE failure: {msg}"),
            Self::InvalidScope(scope) => write!(f, "invalid scope: {scope}"),
            Self::TokenExpired { token_id } => write!(f, "token expired: {token_id}"),
            Self::TokenRevoked { token_id } => write!(f, "token revoked: {token_id}"),
            Self::UnsupportedGrantType(gt) => write!(f, "unsupported grant type: {gt}"),
            Self::AuthRequestNotFound(id) => write!(f, "auth request not found: {id}"),
        }
    }
}

impl std::error::Error for OAuthError {}

// ── Types ────────────────────────────────────────────────────────

/// OAuth grant type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantType {
    AuthorizationCode,
    RefreshToken,
    ClientCredentials,
}

impl GrantType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::AuthorizationCode => "authorization_code",
            Self::RefreshToken => "refresh_token",
            Self::ClientCredentials => "client_credentials",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "authorization_code" => Some(Self::AuthorizationCode),
            "refresh_token" => Some(Self::RefreshToken),
            "client_credentials" => Some(Self::ClientCredentials),
            _ => None,
        }
    }
}

impl fmt::Display for GrantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Registered OAuth client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthClient {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uris: Vec<String>,
    pub allowed_scopes: Vec<String>,
    pub allowed_grant_types: Vec<GrantType>,
}

impl OAuthClient {
    /// Create a new client.
    pub fn new(
        client_id: &str,
        client_secret: &str,
        redirect_uris: &[&str],
        scopes: &[&str],
    ) -> Self {
        Self {
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            redirect_uris: redirect_uris.iter().map(|s| s.to_string()).collect(),
            allowed_scopes: scopes.iter().map(|s| s.to_string()).collect(),
            allowed_grant_types: vec![
                GrantType::AuthorizationCode,
                GrantType::RefreshToken,
            ],
        }
    }

    /// Check if a redirect URI is valid for this client.
    pub fn validate_redirect_uri(&self, uri: &str) -> bool {
        self.redirect_uris.contains(&uri.to_string())
    }

    /// Check if a scope is allowed.
    pub fn validate_scopes(&self, scopes: &[String]) -> Result<(), OAuthError> {
        for scope in scopes {
            if !self.allowed_scopes.contains(scope) {
                return Err(OAuthError::InvalidScope(scope.clone()));
            }
        }
        Ok(())
    }
}

/// Scope set for OAuth tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeSet {
    scopes: Vec<String>,
}

impl ScopeSet {
    /// Create from a space-delimited scope string.
    pub fn from_str(s: &str) -> Self {
        let scopes: Vec<String> = s
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        Self { scopes }
    }

    /// Create from a list.
    pub fn from_list(scopes: &[&str]) -> Self {
        Self {
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// To space-delimited string.
    pub fn to_string_repr(&self) -> String {
        self.scopes.join(" ")
    }

    /// Check if this set contains a given scope.
    pub fn contains(&self, scope: &str) -> bool {
        self.scopes.contains(&scope.to_string())
    }

    /// Check if this set is a subset of another.
    pub fn is_subset_of(&self, other: &ScopeSet) -> bool {
        self.scopes.iter().all(|s| other.contains(s))
    }

    /// Number of scopes.
    pub fn len(&self) -> usize {
        self.scopes.len()
    }

    /// Whether the scope set is empty.
    pub fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    /// Get the scopes as a slice.
    pub fn as_slice(&self) -> &[String] {
        &self.scopes
    }
}

impl fmt::Display for ScopeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_repr())
    }
}

/// PKCE code challenge method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeChallengeMethod {
    /// Plain: challenge = verifier.
    Plain,
    /// S256: challenge = BASE64URL(SHA256(verifier)).
    S256,
}

/// PKCE pair: code verifier and code challenge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
    pub method: CodeChallengeMethod,
}

impl PkcePair {
    /// Generate a PKCE pair from a verifier string.
    pub fn generate(verifier: &str, method: CodeChallengeMethod) -> Self {
        let challenge = match method {
            CodeChallengeMethod::Plain => verifier.to_string(),
            CodeChallengeMethod::S256 => {
                let hash = sha256_bytes(verifier.as_bytes());
                base64url_encode(&hash)
            }
        };
        Self {
            verifier: verifier.to_string(),
            challenge,
            method,
        }
    }

    /// Verify a code verifier against a stored challenge.
    pub fn verify(verifier: &str, challenge: &str, method: CodeChallengeMethod) -> bool {
        match method {
            CodeChallengeMethod::Plain => verifier == challenge,
            CodeChallengeMethod::S256 => {
                let hash = sha256_bytes(verifier.as_bytes());
                let computed = base64url_encode(&hash);
                computed == challenge
            }
        }
    }
}

/// Authorization request (before the user authorizes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    pub client_id: String,
    pub redirect_uri: String,
    pub response_type: String,
    pub scope: ScopeSet,
    pub state: String,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<CodeChallengeMethod>,
}

/// Authorization code issued after user consent.
#[derive(Debug, Clone)]
pub struct AuthorizationCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: ScopeSet,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<CodeChallengeMethod>,
    pub issued_at: u64,
    pub expires_at: u64,
    pub used: bool,
}

/// Access token response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_token: Option<String>,
    pub scope: String,
}

/// Stored token for introspection.
#[derive(Debug, Clone)]
pub struct StoredToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub client_id: String,
    pub scope: ScopeSet,
    pub issued_at: u64,
    pub expires_at: u64,
    pub revoked: bool,
}

impl StoredToken {
    /// Check if the token is active.
    pub fn is_active(&self, now: u64) -> bool {
        !self.revoked && now < self.expires_at
    }
}

/// Token introspection response (RFC 7662).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionResponse {
    pub active: bool,
    pub scope: Option<String>,
    pub client_id: Option<String>,
    pub token_type: Option<String>,
    pub exp: Option<u64>,
    pub iat: Option<u64>,
}

// ── OAuth Server ─────────────────────────────────────────────────

/// OAuth 2.0 authorization server simulation.
#[derive(Debug, Clone)]
pub struct OAuthServer {
    clients: HashMap<String, OAuthClient>,
    auth_codes: HashMap<String, AuthorizationCode>,
    tokens: Vec<StoredToken>,
    pending_auth: HashMap<String, AuthorizationRequest>,
    token_counter: u64,
    code_counter: u64,
    access_token_ttl: u64,
    auth_code_ttl: u64,
}

impl OAuthServer {
    /// Create a new OAuth server.
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            auth_codes: HashMap::new(),
            tokens: Vec::new(),
            pending_auth: HashMap::new(),
            token_counter: 0,
            code_counter: 0,
            access_token_ttl: 3600,
            auth_code_ttl: 600,
        }
    }

    /// Set access token TTL in seconds.
    pub fn with_token_ttl(mut self, ttl: u64) -> Self {
        self.access_token_ttl = ttl;
        self
    }

    /// Register a client.
    pub fn register_client(&mut self, client: OAuthClient) {
        self.clients.insert(client.client_id.clone(), client);
    }

    /// Start an authorization request (step 1 of auth code flow).
    pub fn authorize(
        &mut self,
        request: AuthorizationRequest,
    ) -> Result<String, OAuthError> {
        let client = self
            .clients
            .get(&request.client_id)
            .ok_or_else(|| OAuthError::InvalidClient(request.client_id.clone()))?;

        if !client.validate_redirect_uri(&request.redirect_uri) {
            return Err(OAuthError::InvalidRedirectUri(request.redirect_uri.clone()));
        }

        client.validate_scopes(request.scope.as_slice())?;

        let request_id = format!("authreq_{}", self.code_counter);
        self.code_counter += 1;
        self.pending_auth.insert(request_id.clone(), request);
        Ok(request_id)
    }

    /// Approve an authorization request, issuing an authorization code.
    pub fn approve_authorization(
        &mut self,
        request_id: &str,
        now: u64,
    ) -> Result<(String, String), OAuthError> {
        let request = self
            .pending_auth
            .remove(request_id)
            .ok_or_else(|| OAuthError::AuthRequestNotFound(request_id.to_string()))?;

        self.code_counter += 1;
        let code = format!("code_{}", self.code_counter);

        let auth_code = AuthorizationCode {
            code: code.clone(),
            client_id: request.client_id,
            redirect_uri: request.redirect_uri.clone(),
            scope: request.scope,
            code_challenge: request.code_challenge,
            code_challenge_method: request.code_challenge_method,
            issued_at: now,
            expires_at: now + self.auth_code_ttl,
            used: false,
        };

        self.auth_codes.insert(code.clone(), auth_code);
        Ok((code, request.state))
    }

    /// Exchange an authorization code for tokens.
    pub fn exchange_code(
        &mut self,
        client_id: &str,
        client_secret: &str,
        code: &str,
        redirect_uri: &str,
        code_verifier: Option<&str>,
        now: u64,
    ) -> Result<TokenResponse, OAuthError> {
        // Validate client.
        let client = self
            .clients
            .get(client_id)
            .ok_or_else(|| OAuthError::InvalidClient(client_id.to_string()))?;
        if client.client_secret != client_secret {
            return Err(OAuthError::InvalidClient(client_id.to_string()));
        }

        // Validate code.
        let auth_code = self
            .auth_codes
            .get_mut(code)
            .ok_or_else(|| OAuthError::InvalidCode(code.to_string()))?;

        if auth_code.used {
            return Err(OAuthError::InvalidCode("code already used".to_string()));
        }
        if now > auth_code.expires_at {
            return Err(OAuthError::InvalidCode("code expired".to_string()));
        }
        if auth_code.client_id != client_id {
            return Err(OAuthError::InvalidClient(client_id.to_string()));
        }
        if auth_code.redirect_uri != redirect_uri {
            return Err(OAuthError::InvalidRedirectUri(redirect_uri.to_string()));
        }

        // PKCE verification.
        if let Some(challenge) = &auth_code.code_challenge {
            let method = auth_code
                .code_challenge_method
                .unwrap_or(CodeChallengeMethod::Plain);
            let verifier = code_verifier
                .ok_or_else(|| OAuthError::PkceFailure("missing code_verifier".to_string()))?;
            if !PkcePair::verify(verifier, challenge, method) {
                return Err(OAuthError::PkceFailure("challenge mismatch".to_string()));
            }
        }

        auth_code.used = true;
        let scope = auth_code.scope.clone();

        // Issue tokens.
        self.token_counter += 1;
        let access_token = format!("at_{}", self.token_counter);
        self.token_counter += 1;
        let refresh_token = format!("rt_{}", self.token_counter);

        let stored = StoredToken {
            access_token: access_token.clone(),
            refresh_token: Some(refresh_token.clone()),
            client_id: client_id.to_string(),
            scope: scope.clone(),
            issued_at: now,
            expires_at: now + self.access_token_ttl,
            revoked: false,
        };
        self.tokens.push(stored);

        Ok(TokenResponse {
            access_token,
            token_type: "Bearer".to_string(),
            expires_in: self.access_token_ttl,
            refresh_token: Some(refresh_token),
            scope: scope.to_string_repr(),
        })
    }

    /// Refresh an access token.
    pub fn refresh_token(
        &mut self,
        client_id: &str,
        client_secret: &str,
        refresh_token: &str,
        now: u64,
    ) -> Result<TokenResponse, OAuthError> {
        let client = self
            .clients
            .get(client_id)
            .ok_or_else(|| OAuthError::InvalidClient(client_id.to_string()))?;
        if client.client_secret != client_secret {
            return Err(OAuthError::InvalidClient(client_id.to_string()));
        }

        let old_token = self
            .tokens
            .iter_mut()
            .find(|t| {
                t.refresh_token.as_deref() == Some(refresh_token)
                    && t.client_id == client_id
                    && !t.revoked
            })
            .ok_or_else(|| {
                OAuthError::InvalidRefreshToken(refresh_token.to_string())
            })?;

        // Revoke old token.
        old_token.revoked = true;
        let scope = old_token.scope.clone();

        // Issue new tokens.
        self.token_counter += 1;
        let new_access = format!("at_{}", self.token_counter);
        self.token_counter += 1;
        let new_refresh = format!("rt_{}", self.token_counter);

        let stored = StoredToken {
            access_token: new_access.clone(),
            refresh_token: Some(new_refresh.clone()),
            client_id: client_id.to_string(),
            scope: scope.clone(),
            issued_at: now,
            expires_at: now + self.access_token_ttl,
            revoked: false,
        };
        self.tokens.push(stored);

        Ok(TokenResponse {
            access_token: new_access,
            token_type: "Bearer".to_string(),
            expires_in: self.access_token_ttl,
            refresh_token: Some(new_refresh),
            scope: scope.to_string_repr(),
        })
    }

    /// Introspect a token (RFC 7662).
    pub fn introspect(&self, token: &str, now: u64) -> IntrospectionResponse {
        if let Some(stored) = self.tokens.iter().find(|t| t.access_token == token) {
            if stored.is_active(now) {
                return IntrospectionResponse {
                    active: true,
                    scope: Some(stored.scope.to_string_repr()),
                    client_id: Some(stored.client_id.clone()),
                    token_type: Some("Bearer".to_string()),
                    exp: Some(stored.expires_at),
                    iat: Some(stored.issued_at),
                };
            }
        }
        IntrospectionResponse {
            active: false,
            scope: None,
            client_id: None,
            token_type: None,
            exp: None,
            iat: None,
        }
    }

    /// Revoke a token.
    pub fn revoke_token(&mut self, token: &str) -> bool {
        if let Some(stored) = self
            .tokens
            .iter_mut()
            .find(|t| t.access_token == token || t.refresh_token.as_deref() == Some(token))
        {
            stored.revoked = true;
            return true;
        }
        false
    }

    /// Validate a state parameter for CSRF protection.
    pub fn validate_state(
        expected: &str,
        got: &str,
    ) -> Result<(), OAuthError> {
        if expected != got {
            return Err(OAuthError::StateMismatch {
                expected: expected.to_string(),
                got: got.to_string(),
            });
        }
        Ok(())
    }
}

impl Default for OAuthServer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Minimal SHA-256 for PKCE S256.
fn sha256_bytes(data: &[u8]) -> Vec<u8> {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut state = H0;
    let total_len = data.len() as u64;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0x00);
    }
    buf.extend_from_slice(&(total_len * 8).to_be_bytes());

    for chunk in buf.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4], chunk[i * 4 + 1],
                chunk[i * 4 + 2], chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g; g = f; f = e;
            e = d.wrapping_add(t1);
            d = c; c = b; b = a;
            a = t1.wrapping_add(t2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = Vec::with_capacity(32);
    for word in &state {
        out.extend_from_slice(&word.to_be_bytes());
    }
    out
}

/// Base64url encoding (no padding) per RFC 4648.
fn base64url_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if i + 1 < data.len() {
            out.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        }
        if i + 2 < data.len() {
            out.push(CHARS[(triple & 0x3F) as usize] as char);
        }
        i += 3;
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_server() -> OAuthServer {
        let mut server = OAuthServer::new().with_token_ttl(3600);
        server.register_client(OAuthClient::new(
            "client1",
            "secret1",
            &["https://example.com/callback"],
            &["read", "write", "admin"],
        ));
        server
    }

    fn make_auth_request() -> AuthorizationRequest {
        AuthorizationRequest {
            client_id: "client1".to_string(),
            redirect_uri: "https://example.com/callback".to_string(),
            response_type: "code".to_string(),
            scope: ScopeSet::from_list(&["read", "write"]),
            state: "csrf_state_123".to_string(),
            code_challenge: None,
            code_challenge_method: None,
        }
    }

    #[test]
    fn test_scope_set_from_str() {
        let s = ScopeSet::from_str("read write admin");
        assert_eq!(s.len(), 3);
        assert!(s.contains("read"));
        assert!(s.contains("admin"));
        assert!(!s.contains("delete"));
    }

    #[test]
    fn test_scope_set_subset() {
        let full = ScopeSet::from_list(&["read", "write", "admin"]);
        let partial = ScopeSet::from_list(&["read", "write"]);
        assert!(partial.is_subset_of(&full));
        assert!(!full.is_subset_of(&partial));
    }

    #[test]
    fn test_scope_set_display() {
        let s = ScopeSet::from_list(&["read", "write"]);
        assert_eq!(s.to_string_repr(), "read write");
    }

    #[test]
    fn test_pkce_plain() {
        let pair = PkcePair::generate("verifier123", CodeChallengeMethod::Plain);
        assert_eq!(pair.challenge, "verifier123");
        assert!(PkcePair::verify("verifier123", &pair.challenge, CodeChallengeMethod::Plain));
    }

    #[test]
    fn test_pkce_s256() {
        let pair = PkcePair::generate("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk", CodeChallengeMethod::S256);
        assert_ne!(pair.challenge, pair.verifier);
        assert!(PkcePair::verify(
            &pair.verifier,
            &pair.challenge,
            CodeChallengeMethod::S256,
        ));
        assert!(!PkcePair::verify(
            "wrong_verifier",
            &pair.challenge,
            CodeChallengeMethod::S256,
        ));
    }

    #[test]
    fn test_authorize_invalid_client() {
        let mut server = setup_server();
        let mut req = make_auth_request();
        req.client_id = "nonexistent".to_string();
        assert!(matches!(server.authorize(req), Err(OAuthError::InvalidClient(_))));
    }

    #[test]
    fn test_authorize_invalid_redirect() {
        let mut server = setup_server();
        let mut req = make_auth_request();
        req.redirect_uri = "https://evil.com/callback".to_string();
        assert!(matches!(server.authorize(req), Err(OAuthError::InvalidRedirectUri(_))));
    }

    #[test]
    fn test_authorize_invalid_scope() {
        let mut server = setup_server();
        let mut req = make_auth_request();
        req.scope = ScopeSet::from_list(&["read", "superadmin"]);
        assert!(matches!(server.authorize(req), Err(OAuthError::InvalidScope(_))));
    }

    #[test]
    fn test_full_auth_code_flow() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, state) = server.approve_authorization(&request_id, 1000).unwrap();
        assert_eq!(state, "csrf_state_123");

        let token_resp = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1001)
            .unwrap();
        assert!(token_resp.access_token.starts_with("at_"));
        assert!(token_resp.refresh_token.is_some());
        assert_eq!(token_resp.token_type, "Bearer");
        assert_eq!(token_resp.expires_in, 3600);
    }

    #[test]
    fn test_code_reuse_rejected() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        // First use succeeds.
        server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1001)
            .unwrap();

        // Second use fails.
        let err = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1002)
            .unwrap_err();
        assert!(matches!(err, OAuthError::InvalidCode(_)));
    }

    #[test]
    fn test_code_expired() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        // Try to exchange after expiration (default 600s).
        let err = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 2000)
            .unwrap_err();
        assert!(matches!(err, OAuthError::InvalidCode(_)));
    }

    #[test]
    fn test_auth_code_flow_with_pkce() {
        let mut server = setup_server();
        let pkce = PkcePair::generate("my_verifier_string", CodeChallengeMethod::S256);

        let req = AuthorizationRequest {
            client_id: "client1".to_string(),
            redirect_uri: "https://example.com/callback".to_string(),
            response_type: "code".to_string(),
            scope: ScopeSet::from_list(&["read"]),
            state: "state_xyz".to_string(),
            code_challenge: Some(pkce.challenge.clone()),
            code_challenge_method: Some(CodeChallengeMethod::S256),
        };

        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        // Exchange with correct verifier.
        let resp = server
            .exchange_code(
                "client1", "secret1", &code,
                "https://example.com/callback",
                Some("my_verifier_string"),
                1001,
            )
            .unwrap();
        assert!(!resp.access_token.is_empty());
    }

    #[test]
    fn test_pkce_wrong_verifier() {
        let mut server = setup_server();
        let pkce = PkcePair::generate("correct_verifier", CodeChallengeMethod::S256);

        let req = AuthorizationRequest {
            client_id: "client1".to_string(),
            redirect_uri: "https://example.com/callback".to_string(),
            response_type: "code".to_string(),
            scope: ScopeSet::from_list(&["read"]),
            state: "s".to_string(),
            code_challenge: Some(pkce.challenge.clone()),
            code_challenge_method: Some(CodeChallengeMethod::S256),
        };

        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        let err = server
            .exchange_code(
                "client1", "secret1", &code,
                "https://example.com/callback",
                Some("wrong_verifier"),
                1001,
            )
            .unwrap_err();
        assert!(matches!(err, OAuthError::PkceFailure(_)));
    }

    #[test]
    fn test_refresh_token_flow() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        let initial = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1001)
            .unwrap();

        let rt = initial.refresh_token.unwrap();
        let refreshed = server.refresh_token("client1", "secret1", &rt, 2000).unwrap();
        assert_ne!(refreshed.access_token, initial.access_token);
        assert!(refreshed.refresh_token.is_some());

        // Old refresh token should no longer work.
        let err = server.refresh_token("client1", "secret1", &rt, 2001).unwrap_err();
        assert!(matches!(err, OAuthError::InvalidRefreshToken(_)));
    }

    #[test]
    fn test_token_introspection() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        let token_resp = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1001)
            .unwrap();

        // Active token.
        let intro = server.introspect(&token_resp.access_token, 2000);
        assert!(intro.active);
        assert_eq!(intro.client_id.as_deref(), Some("client1"));

        // Expired token.
        let intro_expired = server.introspect(&token_resp.access_token, 10000);
        assert!(!intro_expired.active);

        // Unknown token.
        let intro_unknown = server.introspect("nonexistent_token", 2000);
        assert!(!intro_unknown.active);
    }

    #[test]
    fn test_revoke_token() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        let token_resp = server
            .exchange_code("client1", "secret1", &code, "https://example.com/callback", None, 1001)
            .unwrap();

        assert!(server.revoke_token(&token_resp.access_token));
        let intro = server.introspect(&token_resp.access_token, 2000);
        assert!(!intro.active);
    }

    #[test]
    fn test_state_validation() {
        assert!(OAuthServer::validate_state("abc", "abc").is_ok());
        assert!(matches!(
            OAuthServer::validate_state("abc", "xyz"),
            Err(OAuthError::StateMismatch { .. })
        ));
    }

    #[test]
    fn test_grant_type_display() {
        assert_eq!(GrantType::AuthorizationCode.as_str(), "authorization_code");
        assert_eq!(GrantType::RefreshToken.as_str(), "refresh_token");
        assert_eq!(
            GrantType::from_str("client_credentials"),
            Some(GrantType::ClientCredentials),
        );
    }

    #[test]
    fn test_wrong_client_secret() {
        let mut server = setup_server();
        let req = make_auth_request();
        let request_id = server.authorize(req).unwrap();
        let (code, _) = server.approve_authorization(&request_id, 1000).unwrap();

        let err = server
            .exchange_code("client1", "wrong_secret", &code, "https://example.com/callback", None, 1001)
            .unwrap_err();
        assert!(matches!(err, OAuthError::InvalidClient(_)));
    }

    #[test]
    fn test_error_display() {
        let err = OAuthError::TokenExpired { token_id: "t1".to_string() };
        assert!(err.to_string().contains("t1"));
    }
}
