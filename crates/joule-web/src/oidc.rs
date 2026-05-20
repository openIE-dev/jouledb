//! OpenID Connect — ID token parsing/validation, claims extraction, discovery
//! document modeling, userinfo endpoint modeling, nonce verification, and session
//! management.
//!
//! Replaces `openid-client`, `oidc-client-js`, and `passport-openidconnect` with
//! a pure-Rust OIDC implementation for token validation and claim extraction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// OIDC errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OidcError {
    /// Invalid ID token format.
    InvalidToken(String),
    /// Token signature verification failed.
    SignatureInvalid(String),
    /// Token has expired.
    TokenExpired { sub: String, exp: u64, now: u64 },
    /// Nonce mismatch.
    NonceMismatch { expected: String, actual: String },
    /// Audience mismatch.
    AudienceMismatch { expected: String, actual: String },
    /// Issuer mismatch.
    IssuerMismatch { expected: String, actual: String },
    /// Required claim missing.
    MissingClaim(String),
    /// Discovery document error.
    DiscoveryError(String),
    /// Session error.
    SessionError(String),
}

impl fmt::Display for OidcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken(msg) => write!(f, "invalid ID token: {msg}"),
            Self::SignatureInvalid(msg) => write!(f, "signature invalid: {msg}"),
            Self::TokenExpired { sub, exp, now } => {
                write!(f, "token for {sub} expired: exp={exp}, now={now}")
            }
            Self::NonceMismatch { expected, actual } => {
                write!(f, "nonce mismatch: expected {expected}, got {actual}")
            }
            Self::AudienceMismatch { expected, actual } => {
                write!(f, "audience mismatch: expected {expected}, got {actual}")
            }
            Self::IssuerMismatch { expected, actual } => {
                write!(f, "issuer mismatch: expected {expected}, got {actual}")
            }
            Self::MissingClaim(name) => write!(f, "missing required claim: {name}"),
            Self::DiscoveryError(msg) => write!(f, "discovery error: {msg}"),
            Self::SessionError(msg) => write!(f, "session error: {msg}"),
        }
    }
}

impl std::error::Error for OidcError {}

// ── Standard Claims ────────────────────────────────────────────

/// Standard OIDC claims from an ID token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdTokenClaims {
    /// Issuer identifier.
    pub iss: String,
    /// Subject identifier.
    pub sub: String,
    /// Audience (client ID).
    pub aud: Vec<String>,
    /// Expiration time (unix seconds).
    pub exp: u64,
    /// Issued at (unix seconds).
    pub iat: u64,
    /// Auth time (unix seconds, optional).
    pub auth_time: Option<u64>,
    /// Nonce.
    pub nonce: Option<String>,
    /// Authentication context class reference.
    pub acr: Option<String>,
    /// Authentication methods references.
    pub amr: Option<Vec<String>>,
    /// Authorized party.
    pub azp: Option<String>,
    /// Additional claims.
    pub extra: HashMap<String, serde_json::Value>,
}

impl IdTokenClaims {
    /// Check if a specific audience is present.
    pub fn has_audience(&self, client_id: &str) -> bool {
        self.aud.iter().any(|a| a == client_id)
    }

    /// Get the primary audience (first entry).
    pub fn primary_audience(&self) -> Option<&str> {
        self.aud.first().map(|s| s.as_str())
    }

    /// Get an extra claim as string.
    pub fn extra_string(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(|v| v.as_str())
    }
}

/// Userinfo response claims.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserInfoClaims {
    pub sub: String,
    pub name: Option<String>,
    pub given_name: Option<String>,
    pub family_name: Option<String>,
    pub middle_name: Option<String>,
    pub nickname: Option<String>,
    pub preferred_username: Option<String>,
    pub profile: Option<String>,
    pub picture: Option<String>,
    pub website: Option<String>,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub gender: Option<String>,
    pub birthdate: Option<String>,
    pub zoneinfo: Option<String>,
    pub locale: Option<String>,
    pub phone_number: Option<String>,
    pub phone_number_verified: Option<bool>,
    pub updated_at: Option<u64>,
    pub extra: HashMap<String, serde_json::Value>,
}

// ── ID Token ───────────────────────────────────────────────────

/// A parsed (but not necessarily verified) ID token.
#[derive(Debug, Clone)]
pub struct IdToken {
    pub header_json: String,
    pub claims: IdTokenClaims,
    pub signature_bytes: Vec<u8>,
    pub raw: String,
}

/// Simple base64url decode (no padding).
fn base64url_decode(input: &str) -> Result<Vec<u8>, OidcError> {
    const TABLE: [u8; 128] = {
        let mut t = [255u8; 128];
        let mut i = 0u8;
        while i < 26 {
            t[(b'A' + i) as usize] = i;
            t[(b'a' + i) as usize] = i + 26;
            i += 1;
        }
        let mut d = 0u8;
        while d < 10 {
            t[(b'0' + d) as usize] = d + 52;
            d += 1;
        }
        t[b'-' as usize] = 62;
        t[b'_' as usize] = 63;
        t
    };

    let mut out = Vec::new();
    let bytes: Vec<u8> = input.bytes().filter(|b| *b != b'=').collect();
    let chunks = bytes.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let (a, b, c, d) = (
            TABLE.get(bytes[idx] as usize).copied().unwrap_or(255),
            TABLE.get(bytes[idx + 1] as usize).copied().unwrap_or(255),
            TABLE.get(bytes[idx + 2] as usize).copied().unwrap_or(255),
            TABLE.get(bytes[idx + 3] as usize).copied().unwrap_or(255),
        );
        if a == 255 || b == 255 || c == 255 || d == 255 {
            return Err(OidcError::InvalidToken("invalid base64url character".into()));
        }
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    }
    let rem = bytes.len() % 4;
    if rem >= 2 {
        let idx = chunks * 4;
        let a = TABLE.get(bytes[idx] as usize).copied().unwrap_or(255);
        let b = TABLE.get(bytes[idx + 1] as usize).copied().unwrap_or(255);
        if a == 255 || b == 255 {
            return Err(OidcError::InvalidToken("invalid base64url character".into()));
        }
        let n = ((a as u32) << 18) | ((b as u32) << 12);
        out.push((n >> 16) as u8);
        if rem >= 3 {
            let c = TABLE.get(bytes[idx + 2] as usize).copied().unwrap_or(255);
            if c == 255 {
                return Err(OidcError::InvalidToken("invalid base64url character".into()));
            }
            let n2 = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6);
            // Replace last byte and add next
            let len = out.len();
            out[len - 1] = (n2 >> 16) as u8;
            out.push((n2 >> 8) as u8);
        }
    }
    Ok(out)
}

/// Parse a JWT-format ID token into its parts.
pub fn parse_id_token(raw: &str) -> Result<IdToken, OidcError> {
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() != 3 {
        return Err(OidcError::InvalidToken(format!(
            "expected 3 JWT parts, got {}",
            parts.len()
        )));
    }

    let header_bytes = base64url_decode(parts[0])?;
    let header_json =
        String::from_utf8(header_bytes).map_err(|e| OidcError::InvalidToken(e.to_string()))?;

    let claims_bytes = base64url_decode(parts[1])?;
    let claims_json =
        String::from_utf8(claims_bytes).map_err(|e| OidcError::InvalidToken(e.to_string()))?;
    let claims: IdTokenClaims =
        serde_json::from_str(&claims_json).map_err(|e| OidcError::InvalidToken(e.to_string()))?;

    let signature_bytes = base64url_decode(parts[2])?;

    Ok(IdToken {
        header_json,
        claims,
        signature_bytes,
        raw: raw.to_string(),
    })
}

// ── Validation ─────────────────────────────────────────────────

/// Configuration for ID token validation.
#[derive(Debug, Clone)]
pub struct TokenValidationConfig {
    pub expected_issuer: String,
    pub expected_audience: String,
    pub expected_nonce: Option<String>,
    pub now_seconds: u64,
    pub clock_skew_seconds: u64,
}

/// Validate ID token claims against expected values.
pub fn validate_id_token(
    claims: &IdTokenClaims,
    config: &TokenValidationConfig,
) -> Result<(), OidcError> {
    // Issuer
    if claims.iss != config.expected_issuer {
        return Err(OidcError::IssuerMismatch {
            expected: config.expected_issuer.clone(),
            actual: claims.iss.clone(),
        });
    }

    // Audience
    if !claims.has_audience(&config.expected_audience) {
        return Err(OidcError::AudienceMismatch {
            expected: config.expected_audience.clone(),
            actual: claims.aud.join(", "),
        });
    }

    // Expiration
    if claims.exp + config.clock_skew_seconds < config.now_seconds {
        return Err(OidcError::TokenExpired {
            sub: claims.sub.clone(),
            exp: claims.exp,
            now: config.now_seconds,
        });
    }

    // Nonce
    if let Some(expected_nonce) = &config.expected_nonce {
        match &claims.nonce {
            Some(actual_nonce) if actual_nonce == expected_nonce => {}
            Some(actual_nonce) => {
                return Err(OidcError::NonceMismatch {
                    expected: expected_nonce.clone(),
                    actual: actual_nonce.clone(),
                });
            }
            None => {
                return Err(OidcError::MissingClaim("nonce".into()));
            }
        }
    }

    Ok(())
}

// ── Discovery Document ─────────────────────────────────────────

/// OIDC Discovery document (/.well-known/openid-configuration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryDocument {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub userinfo_endpoint: Option<String>,
    pub jwks_uri: String,
    pub registration_endpoint: Option<String>,
    pub scopes_supported: Vec<String>,
    pub response_types_supported: Vec<String>,
    pub response_modes_supported: Option<Vec<String>>,
    pub grant_types_supported: Option<Vec<String>>,
    pub subject_types_supported: Vec<String>,
    pub id_token_signing_alg_values_supported: Vec<String>,
    pub claims_supported: Option<Vec<String>>,
    pub token_endpoint_auth_methods_supported: Option<Vec<String>>,
    pub code_challenge_methods_supported: Option<Vec<String>>,
    pub end_session_endpoint: Option<String>,
    pub revocation_endpoint: Option<String>,
    pub introspection_endpoint: Option<String>,
}

impl DiscoveryDocument {
    /// Validate that a discovery document has required fields.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.issuer.is_empty() {
            return Err(OidcError::DiscoveryError("issuer is empty".into()));
        }
        if self.authorization_endpoint.is_empty() {
            return Err(OidcError::DiscoveryError("authorization_endpoint is empty".into()));
        }
        if self.token_endpoint.is_empty() {
            return Err(OidcError::DiscoveryError("token_endpoint is empty".into()));
        }
        if self.jwks_uri.is_empty() {
            return Err(OidcError::DiscoveryError("jwks_uri is empty".into()));
        }
        if self.response_types_supported.is_empty() {
            return Err(OidcError::DiscoveryError("response_types_supported is empty".into()));
        }
        if self.subject_types_supported.is_empty() {
            return Err(OidcError::DiscoveryError("subject_types_supported is empty".into()));
        }
        if self.id_token_signing_alg_values_supported.is_empty() {
            return Err(OidcError::DiscoveryError(
                "id_token_signing_alg_values_supported is empty".into(),
            ));
        }
        // Issuer must be HTTPS (or http for localhost)
        if !self.issuer.starts_with("https://") && !self.issuer.starts_with("http://localhost") {
            return Err(OidcError::DiscoveryError("issuer must use HTTPS".into()));
        }
        Ok(())
    }

    /// Check if a scope is supported.
    pub fn supports_scope(&self, scope: &str) -> bool {
        self.scopes_supported.iter().any(|s| s == scope)
    }

    /// Check if a grant type is supported.
    pub fn supports_grant_type(&self, grant: &str) -> bool {
        self.grant_types_supported
            .as_ref()
            .map(|g| g.iter().any(|s| s == grant))
            .unwrap_or(false)
    }

    /// Check if PKCE is supported.
    pub fn supports_pkce(&self) -> bool {
        self.code_challenge_methods_supported
            .as_ref()
            .map(|m| m.iter().any(|s| s == "S256"))
            .unwrap_or(false)
    }
}

// ── Session Management ─────────────────────────────────────────

/// OIDC session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OidcSessionState {
    Active,
    Expired,
    LoggedOut,
}

/// An OIDC session tracking user authentication state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcSession {
    pub session_id: String,
    pub sub: String,
    pub issuer: String,
    pub id_token_exp: u64,
    pub access_token_exp: Option<u64>,
    pub nonce: String,
    pub state: OidcSessionState,
    pub created_at_s: u64,
    pub last_validated_at_s: u64,
}

/// OIDC session store.
pub struct OidcSessionStore {
    sessions: HashMap<String, OidcSession>,
    by_sub: HashMap<String, Vec<String>>,
    next_id: u64,
}

impl OidcSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            by_sub: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new session from validated ID token claims.
    pub fn create_session(
        &mut self,
        claims: &IdTokenClaims,
        nonce: &str,
        access_token_exp: Option<u64>,
        now_s: u64,
    ) -> String {
        let session_id = format!("oidc_sess_{:08x}", self.next_id);
        self.next_id += 1;

        let session = OidcSession {
            session_id: session_id.clone(),
            sub: claims.sub.clone(),
            issuer: claims.iss.clone(),
            id_token_exp: claims.exp,
            access_token_exp,
            nonce: nonce.to_string(),
            state: OidcSessionState::Active,
            created_at_s: now_s,
            last_validated_at_s: now_s,
        };

        self.sessions.insert(session_id.clone(), session);
        self.by_sub
            .entry(claims.sub.clone())
            .or_default()
            .push(session_id.clone());

        session_id
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Result<&OidcSession, OidcError> {
        self.sessions
            .get(session_id)
            .ok_or_else(|| OidcError::SessionError(format!("session not found: {session_id}")))
    }

    /// Check if a session is still valid at a given time.
    pub fn is_session_valid(&self, session_id: &str, now_s: u64) -> Result<bool, OidcError> {
        let session = self.get_session(session_id)?;
        Ok(session.state == OidcSessionState::Active && session.id_token_exp > now_s)
    }

    /// Mark a session as logged out.
    pub fn logout(&mut self, session_id: &str) -> Result<(), OidcError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| OidcError::SessionError(format!("session not found: {session_id}")))?;
        session.state = OidcSessionState::LoggedOut;
        Ok(())
    }

    /// Log out all sessions for a subject.
    pub fn logout_all(&mut self, sub: &str) -> usize {
        let ids: Vec<String> = self
            .by_sub
            .get(sub)
            .cloned()
            .unwrap_or_default();
        let count = ids.len();
        for id in ids {
            if let Some(s) = self.sessions.get_mut(&id) {
                s.state = OidcSessionState::LoggedOut;
            }
        }
        count
    }

    /// Get all session IDs for a subject.
    pub fn sessions_for_sub(&self, sub: &str) -> Vec<&OidcSession> {
        self.by_sub
            .get(sub)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.sessions.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove expired/logged-out sessions.
    pub fn cleanup(&mut self, now_s: u64) -> usize {
        let to_remove: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.state == OidcSessionState::LoggedOut || s.id_token_exp <= now_s)
            .map(|(id, _)| id.clone())
            .collect();
        let count = to_remove.len();
        for id in &to_remove {
            if let Some(s) = self.sessions.remove(id) {
                if let Some(ids) = self.by_sub.get_mut(&s.sub) {
                    ids.retain(|i| i != id);
                }
            }
        }
        count
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for OidcSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_claims() -> IdTokenClaims {
        IdTokenClaims {
            iss: "https://auth.example.com".into(),
            sub: "user_42".into(),
            aud: vec!["my_app".into()],
            exp: 1700000000,
            iat: 1699996400,
            auth_time: Some(1699996400),
            nonce: Some("nonce_abc".into()),
            acr: None,
            amr: Some(vec!["pwd".into()]),
            azp: Some("my_app".into()),
            extra: HashMap::new(),
        }
    }

    fn test_validation_config() -> TokenValidationConfig {
        TokenValidationConfig {
            expected_issuer: "https://auth.example.com".into(),
            expected_audience: "my_app".into(),
            expected_nonce: Some("nonce_abc".into()),
            now_seconds: 1699998000,
            clock_skew_seconds: 60,
        }
    }

    fn test_discovery() -> DiscoveryDocument {
        DiscoveryDocument {
            issuer: "https://auth.example.com".into(),
            authorization_endpoint: "https://auth.example.com/authorize".into(),
            token_endpoint: "https://auth.example.com/token".into(),
            userinfo_endpoint: Some("https://auth.example.com/userinfo".into()),
            jwks_uri: "https://auth.example.com/.well-known/jwks.json".into(),
            registration_endpoint: None,
            scopes_supported: vec!["openid".into(), "profile".into(), "email".into()],
            response_types_supported: vec!["code".into(), "id_token".into()],
            response_modes_supported: Some(vec!["query".into(), "fragment".into()]),
            grant_types_supported: Some(vec![
                "authorization_code".into(),
                "refresh_token".into(),
            ]),
            subject_types_supported: vec!["public".into()],
            id_token_signing_alg_values_supported: vec!["RS256".into()],
            claims_supported: Some(vec!["sub".into(), "name".into(), "email".into()]),
            token_endpoint_auth_methods_supported: Some(vec!["client_secret_basic".into()]),
            code_challenge_methods_supported: Some(vec!["S256".into()]),
            end_session_endpoint: Some("https://auth.example.com/logout".into()),
            revocation_endpoint: None,
            introspection_endpoint: None,
        }
    }

    #[test]
    fn test_validate_claims_ok() {
        let claims = test_claims();
        let config = test_validation_config();
        assert!(validate_id_token(&claims, &config).is_ok());
    }

    #[test]
    fn test_validate_issuer_mismatch() {
        let claims = test_claims();
        let mut config = test_validation_config();
        config.expected_issuer = "https://other.example.com".into();
        assert!(matches!(
            validate_id_token(&claims, &config),
            Err(OidcError::IssuerMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_audience_mismatch() {
        let claims = test_claims();
        let mut config = test_validation_config();
        config.expected_audience = "wrong_app".into();
        assert!(matches!(
            validate_id_token(&claims, &config),
            Err(OidcError::AudienceMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_expired() {
        let claims = test_claims();
        let mut config = test_validation_config();
        config.now_seconds = 1700100000; // Way past exp
        assert!(matches!(
            validate_id_token(&claims, &config),
            Err(OidcError::TokenExpired { .. })
        ));
    }

    #[test]
    fn test_validate_clock_skew() {
        let claims = test_claims();
        let mut config = test_validation_config();
        // Just barely expired but within skew
        config.now_seconds = claims.exp + 30;
        config.clock_skew_seconds = 60;
        assert!(validate_id_token(&claims, &config).is_ok());
    }

    #[test]
    fn test_validate_nonce_mismatch() {
        let claims = test_claims();
        let mut config = test_validation_config();
        config.expected_nonce = Some("wrong_nonce".into());
        assert!(matches!(
            validate_id_token(&claims, &config),
            Err(OidcError::NonceMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_nonce_missing() {
        let mut claims = test_claims();
        claims.nonce = None;
        let config = test_validation_config();
        assert!(matches!(
            validate_id_token(&claims, &config),
            Err(OidcError::MissingClaim(_))
        ));
    }

    #[test]
    fn test_validate_no_nonce_expected() {
        let mut claims = test_claims();
        claims.nonce = None;
        let mut config = test_validation_config();
        config.expected_nonce = None;
        assert!(validate_id_token(&claims, &config).is_ok());
    }

    #[test]
    fn test_claims_has_audience() {
        let claims = test_claims();
        assert!(claims.has_audience("my_app"));
        assert!(!claims.has_audience("other"));
    }

    #[test]
    fn test_claims_primary_audience() {
        let claims = test_claims();
        assert_eq!(claims.primary_audience(), Some("my_app"));
    }

    #[test]
    fn test_claims_extra() {
        let mut claims = test_claims();
        claims.extra.insert("role".into(), serde_json::Value::String("admin".into()));
        assert_eq!(claims.extra_string("role"), Some("admin"));
        assert_eq!(claims.extra_string("missing"), None);
    }

    #[test]
    fn test_discovery_validate_ok() {
        let disco = test_discovery();
        assert!(disco.validate().is_ok());
    }

    #[test]
    fn test_discovery_empty_issuer() {
        let mut disco = test_discovery();
        disco.issuer = "".into();
        assert!(matches!(disco.validate(), Err(OidcError::DiscoveryError(_))));
    }

    #[test]
    fn test_discovery_http_issuer() {
        let mut disco = test_discovery();
        disco.issuer = "http://auth.example.com".into();
        assert!(matches!(disco.validate(), Err(OidcError::DiscoveryError(_))));
    }

    #[test]
    fn test_discovery_http_localhost_ok() {
        let mut disco = test_discovery();
        disco.issuer = "http://localhost:8080".into();
        assert!(disco.validate().is_ok());
    }

    #[test]
    fn test_discovery_supports_scope() {
        let disco = test_discovery();
        assert!(disco.supports_scope("openid"));
        assert!(disco.supports_scope("profile"));
        assert!(!disco.supports_scope("admin"));
    }

    #[test]
    fn test_discovery_supports_grant() {
        let disco = test_discovery();
        assert!(disco.supports_grant_type("authorization_code"));
        assert!(!disco.supports_grant_type("client_credentials"));
    }

    #[test]
    fn test_discovery_supports_pkce() {
        let disco = test_discovery();
        assert!(disco.supports_pkce());
    }

    #[test]
    fn test_discovery_no_pkce() {
        let mut disco = test_discovery();
        disco.code_challenge_methods_supported = None;
        assert!(!disco.supports_pkce());
    }

    #[test]
    fn test_base64url_decode() {
        // "hello" in base64url = "aGVsbG8"
        let decoded = base64url_decode("aGVsbG8").unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn test_base64url_decode_with_padding() {
        let decoded = base64url_decode("YQ").unwrap();
        assert_eq!(decoded, b"a");
    }

    #[test]
    fn test_parse_id_token_bad_format() {
        assert!(matches!(
            parse_id_token("not.a.valid.jwt.too.many"),
            Err(OidcError::InvalidToken(_))
        ));
        assert!(matches!(
            parse_id_token("only_one_part"),
            Err(OidcError::InvalidToken(_))
        ));
    }

    #[test]
    fn test_session_store_create() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        let sid = store.create_session(&claims, "nonce_abc", Some(1700003600), 1699996400);
        assert!(sid.starts_with("oidc_sess_"));
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn test_session_store_get() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        let sid = store.create_session(&claims, "nonce_abc", None, 1699996400);
        let session = store.get_session(&sid).unwrap();
        assert_eq!(session.sub, "user_42");
        assert_eq!(session.state, OidcSessionState::Active);
    }

    #[test]
    fn test_session_store_not_found() {
        let store = OidcSessionStore::new();
        assert!(matches!(
            store.get_session("nonexistent"),
            Err(OidcError::SessionError(_))
        ));
    }

    #[test]
    fn test_session_validity() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        let sid = store.create_session(&claims, "nonce_abc", None, 1699996400);
        assert!(store.is_session_valid(&sid, 1699998000).unwrap());
        // After exp
        assert!(!store.is_session_valid(&sid, 1700100000).unwrap());
    }

    #[test]
    fn test_session_logout() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        let sid = store.create_session(&claims, "nonce_abc", None, 1699996400);
        store.logout(&sid).unwrap();
        assert!(!store.is_session_valid(&sid, 1699998000).unwrap());
    }

    #[test]
    fn test_session_logout_all() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        store.create_session(&claims, "n1", None, 1699996400);
        store.create_session(&claims, "n2", None, 1699996400);
        let count = store.logout_all("user_42");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_session_sessions_for_sub() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        store.create_session(&claims, "n1", None, 1699996400);
        store.create_session(&claims, "n2", None, 1699996400);
        let sessions = store.sessions_for_sub("user_42");
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn test_session_cleanup() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        let sid = store.create_session(&claims, "n1", None, 1699996400);
        store.logout(&sid).unwrap();
        store.create_session(&claims, "n2", None, 1699996400);
        let removed = store.cleanup(1699998000);
        assert_eq!(removed, 1); // Only logged out one
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn test_session_cleanup_expired() {
        let claims = test_claims();
        let mut store = OidcSessionStore::new();
        store.create_session(&claims, "n1", None, 1699996400);
        let removed = store.cleanup(1700100000); // After exp
        assert_eq!(removed, 1);
        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_userinfo_claims_default() {
        let ui = UserInfoClaims::default();
        assert!(ui.sub.is_empty());
        assert!(ui.name.is_none());
        assert!(ui.email.is_none());
    }

    #[test]
    fn test_error_display() {
        let e = OidcError::MissingClaim("nonce".into());
        assert_eq!(e.to_string(), "missing required claim: nonce");
        let e = OidcError::TokenExpired { sub: "u1".into(), exp: 100, now: 200 };
        assert!(e.to_string().contains("u1"));
    }

    #[test]
    fn test_discovery_empty_endpoints() {
        let mut disco = test_discovery();
        disco.authorization_endpoint = "".into();
        assert!(matches!(disco.validate(), Err(OidcError::DiscoveryError(_))));
    }

    #[test]
    fn test_discovery_empty_signing_alg() {
        let mut disco = test_discovery();
        disco.id_token_signing_alg_values_supported = vec![];
        assert!(matches!(disco.validate(), Err(OidcError::DiscoveryError(_))));
    }
}
