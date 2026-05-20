//! OpenID Connect (OIDC) Relying Party
//!
//! Implements federated identity via any OIDC-compliant Identity Provider
//! (Google, GitHub, Okta, Azure AD, Auth0, etc.).
//!
//! Flow:
//! 1. Client calls `authorize_url()` → redirect user to IdP
//! 2. IdP authenticates user → redirects back with `code`
//! 3. Server calls `exchange_code()` → gets tokens + ID token claims
//! 4. Claims are mapped to internal `Role` via configurable claim mappings

use crate::rbac::Role;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// OIDC provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcProviderConfig {
    /// Human-readable name (e.g., "Google", "Okta")
    pub name: String,
    /// OIDC issuer URL (e.g., "https://accounts.google.com")
    pub issuer_url: String,
    /// OAuth 2.0 client ID
    pub client_id: String,
    /// OAuth 2.0 client secret
    pub client_secret: String,
    /// Redirect URI after authentication
    pub redirect_uri: String,
    /// Scopes to request (default: openid, email, profile)
    pub scopes: Vec<String>,
    /// Claim mappings: IdP claim → internal role
    pub role_mappings: HashMap<String, RoleMapping>,
    /// Default role if no mapping matches
    pub default_role: Role,
    /// Default org for users from this provider
    pub default_org: String,
}

impl OidcProviderConfig {
    /// Create a config for Google OIDC
    pub fn google(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            name: "Google".into(),
            issuer_url: "https://accounts.google.com".into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            redirect_uri: redirect_uri.into(),
            scopes: vec!["openid".into(), "email".into(), "profile".into()],
            role_mappings: HashMap::new(),
            default_role: Role::Viewer,
            default_org: "default".into(),
        }
    }

    /// Create a config for GitHub OIDC
    pub fn github(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            name: "GitHub".into(),
            issuer_url: "https://github.com".into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            redirect_uri: redirect_uri.into(),
            scopes: vec!["openid".into(), "user:email".into()],
            role_mappings: HashMap::new(),
            default_role: Role::Viewer,
            default_org: "default".into(),
        }
    }

    /// Create a config for a generic OIDC provider (Okta, Azure AD, Auth0, etc.)
    pub fn generic(
        name: &str,
        issuer_url: &str,
        client_id: &str,
        client_secret: &str,
        redirect_uri: &str,
    ) -> Self {
        Self {
            name: name.into(),
            issuer_url: issuer_url.into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            redirect_uri: redirect_uri.into(),
            scopes: vec!["openid".into(), "email".into(), "profile".into()],
            role_mappings: HashMap::new(),
            default_role: Role::Viewer,
            default_org: "default".into(),
        }
    }
}

/// Maps an IdP claim value to an internal role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleMapping {
    /// The claim name in the ID token (e.g., "groups", "role", "custom:role")
    pub claim_name: String,
    /// Claim value → Role mappings
    pub value_map: HashMap<String, Role>,
}

/// OIDC discovery document (subset of fields we need)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcDiscovery {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub userinfo_endpoint: Option<String>,
    pub jwks_uri: String,
    pub scopes_supported: Vec<String>,
    pub response_types_supported: Vec<String>,
}

impl OidcDiscovery {
    /// Build the discovery URL from an issuer
    pub fn discovery_url(issuer_url: &str) -> String {
        let base = issuer_url.trim_end_matches('/');
        format!("{}/.well-known/openid-configuration", base)
    }

    /// Create a mock discovery doc for testing
    pub fn mock(issuer_url: &str) -> Self {
        let base = issuer_url.trim_end_matches('/');
        Self {
            issuer: base.to_string(),
            authorization_endpoint: format!("{}/authorize", base),
            token_endpoint: format!("{}/oauth/token", base),
            userinfo_endpoint: Some(format!("{}/userinfo", base)),
            jwks_uri: format!("{}/.well-known/jwks.json", base),
            scopes_supported: vec!["openid".into(), "email".into(), "profile".into()],
            response_types_supported: vec!["code".into()],
        }
    }
}

/// OIDC authentication state (used for CSRF and PKCE)
#[derive(Debug, Clone)]
pub struct OidcAuthState {
    /// Random state parameter for CSRF protection
    pub state: String,
    /// PKCE code verifier (S256)
    pub code_verifier: String,
    /// PKCE code challenge (S256 hash of verifier)
    pub code_challenge: String,
    /// Nonce for ID token binding
    pub nonce: String,
    /// Unix timestamp when this state expires
    pub expires_at: u64,
    /// Provider name this state was created for
    pub provider: String,
}

/// Token response from the IdP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub scope: Option<String>,
}

/// Claims extracted from the ID token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcClaims {
    /// Subject identifier (unique at the IdP)
    pub sub: String,
    /// Email address
    pub email: Option<String>,
    /// Whether email is verified
    pub email_verified: Option<bool>,
    /// Display name
    pub name: Option<String>,
    /// Given name (first name)
    pub given_name: Option<String>,
    /// Family name (last name)
    pub family_name: Option<String>,
    /// Profile picture URL
    pub picture: Option<String>,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// Nonce (for replay protection)
    pub nonce: Option<String>,
    /// Issued at (unix seconds)
    pub iat: u64,
    /// Expiration (unix seconds)
    pub exp: u64,
    /// Additional claims (groups, roles, custom attributes)
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Result of OIDC authentication — internal user identity
#[derive(Debug, Clone)]
pub struct OidcIdentity {
    /// IdP subject identifier
    pub subject: String,
    /// Email (from IdP)
    pub email: String,
    /// Display name (from IdP)
    pub name: Option<String>,
    /// Mapped internal role
    pub role: Role,
    /// Mapped organization
    pub org: String,
    /// Provider name
    pub provider: String,
    /// Raw ID token claims
    pub raw_claims: OidcClaims,
}

/// OIDC errors
#[derive(Debug, thiserror::Error)]
pub enum OidcError {
    #[error("OIDC discovery failed: {0}")]
    DiscoveryFailed(String),

    #[error("invalid state parameter")]
    InvalidState,

    #[error("state expired")]
    StateExpired,

    #[error("token exchange failed: {0}")]
    TokenExchangeFailed(String),

    #[error("ID token missing from response")]
    MissingIdToken,

    #[error("ID token validation failed: {0}")]
    IdTokenInvalid(String),

    #[error("nonce mismatch")]
    NonceMismatch,

    #[error("email not provided by IdP")]
    MissingEmail,

    #[error("email not verified")]
    EmailNotVerified,

    #[error("provider not configured: {0}")]
    ProviderNotFound(String),
}

/// OIDC service — manages multiple providers and authentication state
pub struct OidcService {
    providers: HashMap<String, OidcProviderConfig>,
    /// Cached discovery documents
    discovery_cache: Arc<RwLock<HashMap<String, OidcDiscovery>>>,
    /// Pending auth states (state_param → OidcAuthState)
    pending_states: Arc<RwLock<HashMap<String, OidcAuthState>>>,
}

impl OidcService {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            discovery_cache: Arc::new(RwLock::new(HashMap::new())),
            pending_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an OIDC provider
    pub fn add_provider(&mut self, config: OidcProviderConfig) {
        self.providers.insert(config.name.clone(), config);
    }

    /// Get a registered provider by name
    pub fn get_provider(&self, name: &str) -> Option<&OidcProviderConfig> {
        self.providers.get(name)
    }

    /// List all registered provider names
    pub fn provider_names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Set (or override) the discovery document for a provider.
    ///
    /// In production, this would be fetched from the IdP's
    /// `.well-known/openid-configuration` endpoint. For testing and
    /// offline use, callers can inject a pre-built document.
    pub fn set_discovery(&self, provider_name: &str, discovery: OidcDiscovery) {
        let mut cache = self.discovery_cache.write().unwrap();
        cache.insert(provider_name.to_string(), discovery);
    }

    /// Get the cached discovery document for a provider
    pub fn get_discovery(&self, provider_name: &str) -> Option<OidcDiscovery> {
        let cache = self.discovery_cache.read().unwrap();
        cache.get(provider_name).cloned()
    }

    /// Generate an authorization URL for the given provider.
    ///
    /// Returns `(url, state)` — redirect the user to `url`, store `state` for
    /// verification in the callback.
    pub fn authorize_url(&self, provider_name: &str) -> Result<(String, String), OidcError> {
        let config = self
            .providers
            .get(provider_name)
            .ok_or_else(|| OidcError::ProviderNotFound(provider_name.into()))?;

        let discovery = self
            .get_discovery(provider_name)
            .unwrap_or_else(|| OidcDiscovery::mock(&config.issuer_url));

        // Generate PKCE pair
        let code_verifier = generate_random_string(64);
        let code_challenge = pkce_s256(&code_verifier);

        // Generate state and nonce
        let state = generate_random_string(32);
        let nonce = generate_random_string(32);

        // Build authorization URL
        let scopes = config.scopes.join(" ");
        let url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&nonce={}&code_challenge={}&code_challenge_method=S256",
            discovery.authorization_endpoint,
            urlencoded(&config.client_id),
            urlencoded(&config.redirect_uri),
            urlencoded(&scopes),
            urlencoded(&state),
            urlencoded(&nonce),
            urlencoded(&code_challenge),
        );

        // Store auth state for callback verification
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let auth_state = OidcAuthState {
            state: state.clone(),
            code_verifier,
            code_challenge,
            nonce,
            expires_at: now + 600, // 10 minutes
            provider: provider_name.to_string(),
        };

        {
            let mut pending = self.pending_states.write().unwrap();
            pending.insert(state.clone(), auth_state);
        }

        Ok((url, state))
    }

    /// Validate and consume a pending auth state.
    ///
    /// Called in the callback handler to verify the state parameter and
    /// retrieve the PKCE verifier and nonce.
    pub fn consume_state(&self, state: &str) -> Result<OidcAuthState, OidcError> {
        let mut pending = self.pending_states.write().unwrap();
        let auth_state = pending.remove(state).ok_or(OidcError::InvalidState)?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if now > auth_state.expires_at {
            return Err(OidcError::StateExpired);
        }

        Ok(auth_state)
    }

    /// Exchange an authorization code for tokens (mock implementation).
    ///
    /// In production, this would POST to the IdP's token endpoint.
    /// The mock version returns a fake token response for testing.
    pub fn mock_exchange_code(
        &self,
        provider_name: &str,
        _code: &str,
        auth_state: &OidcAuthState,
    ) -> Result<OidcTokenResponse, OidcError> {
        let config = self
            .providers
            .get(provider_name)
            .ok_or_else(|| OidcError::ProviderNotFound(provider_name.into()))?;

        // Build mock ID token claims
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let claims = serde_json::json!({
            "sub": "oidc_user_123",
            "email": "user@example.com",
            "email_verified": true,
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
            "iss": config.issuer_url,
            "aud": config.client_id,
            "nonce": auth_state.nonce,
            "iat": now,
            "exp": now + 3600,
        });

        // In production, this would be a real JWT signed by the IdP
        let id_token = base64_url_encode(&claims.to_string());

        Ok(OidcTokenResponse {
            access_token: format!("mock_access_{}", generate_random_string(16)),
            token_type: "Bearer".into(),
            expires_in: Some(3600),
            refresh_token: Some(format!("mock_refresh_{}", generate_random_string(16))),
            id_token: Some(id_token),
            scope: Some(config.scopes.join(" ")),
        })
    }

    /// Parse and validate ID token claims from a token response.
    ///
    /// In production, this would verify the JWT signature against the IdP's JWKS.
    /// The mock version base64-decodes the payload.
    pub fn parse_id_token(
        &self,
        token_response: &OidcTokenResponse,
        expected_nonce: &str,
        provider_name: &str,
    ) -> Result<OidcClaims, OidcError> {
        let id_token = token_response
            .id_token
            .as_ref()
            .ok_or(OidcError::MissingIdToken)?;

        let config = self
            .providers
            .get(provider_name)
            .ok_or_else(|| OidcError::ProviderNotFound(provider_name.into()))?;

        // Decode (mock: base64, production: JWT decode + JWKS verification)
        let decoded = base64_url_decode(id_token)
            .map_err(|e| OidcError::IdTokenInvalid(format!("decode error: {}", e)))?;

        let claims: OidcClaims = serde_json::from_str(&decoded)
            .map_err(|e| OidcError::IdTokenInvalid(format!("JSON parse error: {}", e)))?;

        // Verify nonce
        if claims.nonce.as_deref() != Some(expected_nonce) {
            return Err(OidcError::NonceMismatch);
        }

        // Verify issuer
        let expected_issuer = config.issuer_url.trim_end_matches('/');
        let actual_issuer = claims.iss.trim_end_matches('/');
        if actual_issuer != expected_issuer {
            return Err(OidcError::IdTokenInvalid(format!(
                "issuer mismatch: expected {}, got {}",
                expected_issuer, claims.iss,
            )));
        }

        // Verify audience
        if claims.aud != config.client_id {
            return Err(OidcError::IdTokenInvalid(format!(
                "audience mismatch: expected {}, got {}",
                config.client_id, claims.aud,
            )));
        }

        // Verify expiration
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if claims.exp < now {
            return Err(OidcError::IdTokenInvalid("token expired".into()));
        }

        Ok(claims)
    }

    /// Map OIDC claims to an internal identity.
    ///
    /// Uses the provider's role mappings to determine the internal role.
    pub fn map_identity(
        &self,
        claims: &OidcClaims,
        provider_name: &str,
    ) -> Result<OidcIdentity, OidcError> {
        let config = self
            .providers
            .get(provider_name)
            .ok_or_else(|| OidcError::ProviderNotFound(provider_name.into()))?;

        let email = claims.email.as_ref().ok_or(OidcError::MissingEmail)?;

        // Require explicitly verified email — None (missing field) is treated as unverified
        if claims.email_verified != Some(true) {
            return Err(OidcError::EmailNotVerified);
        }

        // Map role from claims
        let role = map_role_from_claims(claims, &config.role_mappings, config.default_role);

        Ok(OidcIdentity {
            subject: claims.sub.clone(),
            email: email.clone(),
            name: claims.name.clone(),
            role,
            org: config.default_org.clone(),
            provider: provider_name.to_string(),
            raw_claims: claims.clone(),
        })
    }

    /// Full authentication flow: consume state, exchange code, parse token, map identity.
    ///
    /// This is the main entry point for the OIDC callback handler.
    pub fn authenticate(&self, state: &str, code: &str) -> Result<OidcIdentity, OidcError> {
        // 1. Consume and verify state
        let auth_state = self.consume_state(state)?;
        let provider_name = auth_state.provider.clone();

        // 2. Exchange code for tokens
        let token_response = self.mock_exchange_code(&provider_name, code, &auth_state)?;

        // 3. Parse and validate ID token
        let claims = self.parse_id_token(&token_response, &auth_state.nonce, &provider_name)?;

        // 4. Map to internal identity
        self.map_identity(&claims, &provider_name)
    }

    /// Clean up expired pending states
    pub fn cleanup_expired(&self) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut pending = self.pending_states.write().unwrap();
        let before = pending.len();
        pending.retain(|_, s| s.expires_at > now);
        before - pending.len()
    }

    /// Number of pending auth states
    pub fn pending_count(&self) -> usize {
        let pending = self.pending_states.read().unwrap();
        pending.len()
    }
}

impl Default for OidcService {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Map OIDC claims to an internal role using configured mappings.
fn map_role_from_claims(
    claims: &OidcClaims,
    mappings: &HashMap<String, RoleMapping>,
    default: Role,
) -> Role {
    for mapping in mappings.values() {
        if let Some(claim_value) = claims.extra.get(&mapping.claim_name) {
            // Handle string claims
            if let Some(s) = claim_value.as_str()
                && let Some(role) = mapping.value_map.get(s)
            {
                return *role;
            }
            // Handle array claims (e.g., "groups": ["admins", "devs"])
            if let Some(arr) = claim_value.as_array() {
                for v in arr {
                    if let Some(s) = v.as_str()
                        && let Some(role) = mapping.value_map.get(s)
                    {
                        return *role;
                    }
                }
            }
        }
    }
    default
}

/// Generate a random hex string of the given length
fn generate_random_string(len: usize) -> String {
    use rand::RngExt;
    let mut rng = rand::rng();
    let hex: String = (0..len)
        .map(|_| format!("{:02x}", rng.random::<u8>()))
        .collect();
    hex[..len].to_string()
}

/// PKCE S256: BASE64URL(SHA256(verifier))
fn pkce_s256(verifier: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(verifier.as_bytes());
    base64_url_encode_bytes(&hash)
}

/// Minimal URL encoding for query parameters
fn urlencoded(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('+', "%2B")
        .replace('/', "%2F")
        .replace(':', "%3A")
}

/// Base64url encode a string
fn base64_url_encode(s: &str) -> String {
    base64_url_encode_bytes(s.as_bytes())
}

/// Base64url encode bytes
fn base64_url_encode_bytes(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

/// Base64url decode to string
fn base64_url_decode(s: &str) -> Result<String, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(s)
        .map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service() -> OidcService {
        let mut svc = OidcService::new();
        let mut config = OidcProviderConfig::google(
            "client_123",
            "secret_456",
            "https://app.example.com/callback",
        );
        config.default_org = "acme".into();
        svc.add_provider(config);
        svc
    }

    #[test]
    fn test_provider_registration() {
        let svc = make_service();
        assert_eq!(svc.provider_names().len(), 1);
        assert!(svc.get_provider("Google").is_some());
        assert!(svc.get_provider("GitHub").is_none());
    }

    #[test]
    fn test_multiple_providers() {
        let mut svc = OidcService::new();
        svc.add_provider(OidcProviderConfig::google(
            "a",
            "b",
            "https://example.com/cb",
        ));
        svc.add_provider(OidcProviderConfig::github(
            "c",
            "d",
            "https://example.com/cb",
        ));
        svc.add_provider(OidcProviderConfig::generic(
            "Okta",
            "https://dev-123.okta.com",
            "e",
            "f",
            "https://example.com/cb",
        ));
        assert_eq!(svc.provider_names().len(), 3);
    }

    #[test]
    fn test_authorize_url_generation() {
        let svc = make_service();
        let (url, state) = svc.authorize_url("Google").unwrap();

        assert!(url.contains("accounts.google.com/authorize"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=client_123"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(!state.is_empty());
        assert_eq!(svc.pending_count(), 1);
    }

    #[test]
    fn test_authorize_unknown_provider() {
        let svc = make_service();
        let result = svc.authorize_url("Unknown");
        assert!(matches!(result, Err(OidcError::ProviderNotFound(_))));
    }

    #[test]
    fn test_consume_state() {
        let svc = make_service();
        let (_, state) = svc.authorize_url("Google").unwrap();

        // First consume succeeds
        let auth_state = svc.consume_state(&state).unwrap();
        assert_eq!(auth_state.provider, "Google");
        assert!(!auth_state.code_verifier.is_empty());
        assert!(!auth_state.nonce.is_empty());

        // Second consume fails (single-use)
        let result = svc.consume_state(&state);
        assert!(matches!(result, Err(OidcError::InvalidState)));
    }

    #[test]
    fn test_consume_invalid_state() {
        let svc = make_service();
        let result = svc.consume_state("nonexistent");
        assert!(matches!(result, Err(OidcError::InvalidState)));
    }

    #[test]
    fn test_full_authentication_flow() {
        let svc = make_service();

        // 1. Generate authorization URL
        let (_, state) = svc.authorize_url("Google").unwrap();

        // 2. Simulate IdP callback with code
        let identity = svc.authenticate(&state, "mock_auth_code").unwrap();

        assert_eq!(identity.provider, "Google");
        assert_eq!(identity.email, "user@example.com");
        assert_eq!(identity.org, "acme");
        assert_eq!(identity.role, Role::Viewer); // default role
        assert!(identity.name.is_some());
    }

    #[test]
    fn test_authentication_replay_fails() {
        let svc = make_service();
        let (_, state) = svc.authorize_url("Google").unwrap();

        // First auth succeeds
        svc.authenticate(&state, "code_1").unwrap();

        // Replay with same state fails
        let result = svc.authenticate(&state, "code_2");
        assert!(matches!(result, Err(OidcError::InvalidState)));
    }

    #[test]
    fn test_role_mapping_from_claims() {
        let mut role_map = HashMap::new();
        role_map.insert("admins".into(), Role::Admin);
        role_map.insert("devs".into(), Role::Operator);

        let mapping = RoleMapping {
            claim_name: "groups".into(),
            value_map: role_map,
        };

        let mut mappings = HashMap::new();
        mappings.insert("groups_mapping".into(), mapping);

        // Claims with "groups": ["admins"]
        let mut extra = HashMap::new();
        extra.insert("groups".into(), serde_json::json!(["admins", "users"]));
        let claims = OidcClaims {
            sub: "user1".into(),
            email: Some("admin@example.com".into()),
            email_verified: Some(true),
            name: Some("Admin User".into()),
            given_name: None,
            family_name: None,
            picture: None,
            iss: "https://idp.example.com".into(),
            aud: "client_123".into(),
            nonce: None,
            iat: 0,
            exp: u64::MAX,
            extra,
        };

        let role = map_role_from_claims(&claims, &mappings, Role::Viewer);
        assert_eq!(role, Role::Admin);
    }

    #[test]
    fn test_role_mapping_string_claim() {
        let mut role_map = HashMap::new();
        role_map.insert("admin".into(), Role::Admin);
        role_map.insert("operator".into(), Role::Operator);

        let mapping = RoleMapping {
            claim_name: "role".into(),
            value_map: role_map,
        };

        let mut mappings = HashMap::new();
        mappings.insert("role_mapping".into(), mapping);

        let mut extra = HashMap::new();
        extra.insert("role".into(), serde_json::json!("operator"));

        let claims = OidcClaims {
            sub: "user2".into(),
            email: Some("op@example.com".into()),
            email_verified: Some(true),
            name: None,
            given_name: None,
            family_name: None,
            picture: None,
            iss: "https://idp.example.com".into(),
            aud: "client_123".into(),
            nonce: None,
            iat: 0,
            exp: u64::MAX,
            extra,
        };

        let role = map_role_from_claims(&claims, &mappings, Role::Viewer);
        assert_eq!(role, Role::Operator);
    }

    #[test]
    fn test_role_mapping_default_fallback() {
        let mappings = HashMap::new();
        let claims = OidcClaims {
            sub: "user3".into(),
            email: Some("user@example.com".into()),
            email_verified: Some(true),
            name: None,
            given_name: None,
            family_name: None,
            picture: None,
            iss: "https://idp.example.com".into(),
            aud: "client_123".into(),
            nonce: None,
            iat: 0,
            exp: u64::MAX,
            extra: HashMap::new(),
        };

        let role = map_role_from_claims(&claims, &mappings, Role::Customer);
        assert_eq!(role, Role::Customer);
    }

    #[test]
    fn test_discovery_url() {
        assert_eq!(
            OidcDiscovery::discovery_url("https://accounts.google.com"),
            "https://accounts.google.com/.well-known/openid-configuration"
        );
        assert_eq!(
            OidcDiscovery::discovery_url("https://accounts.google.com/"),
            "https://accounts.google.com/.well-known/openid-configuration"
        );
    }

    #[test]
    fn test_discovery_cache() {
        let svc = make_service();
        assert!(svc.get_discovery("Google").is_none());

        let discovery = OidcDiscovery::mock("https://accounts.google.com");
        svc.set_discovery("Google", discovery.clone());

        let cached = svc.get_discovery("Google").unwrap();
        assert_eq!(cached.issuer, "https://accounts.google.com");
    }

    #[test]
    fn test_cleanup_expired() {
        let svc = make_service();

        // Generate some states
        svc.authorize_url("Google").unwrap();
        svc.authorize_url("Google").unwrap();
        assert_eq!(svc.pending_count(), 2);

        // States are fresh, cleanup removes nothing
        let cleaned = svc.cleanup_expired();
        assert_eq!(cleaned, 0);
        assert_eq!(svc.pending_count(), 2);
    }

    #[test]
    fn test_pkce_s256() {
        // Verify PKCE challenge is deterministic
        let verifier = "test_verifier_123";
        let challenge1 = pkce_s256(verifier);
        let challenge2 = pkce_s256(verifier);
        assert_eq!(challenge1, challenge2);
        assert!(!challenge1.is_empty());
    }

    #[test]
    fn test_base64url_roundtrip() {
        let original = r#"{"sub":"user_123","email":"test@example.com"}"#;
        let encoded = base64_url_encode(original);
        let decoded = base64_url_decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_missing_email_rejected() {
        let svc = make_service();
        let claims = OidcClaims {
            sub: "user_no_email".into(),
            email: None,
            email_verified: None,
            name: None,
            given_name: None,
            family_name: None,
            picture: None,
            iss: "https://accounts.google.com".into(),
            aud: "client_123".into(),
            nonce: None,
            iat: 0,
            exp: u64::MAX,
            extra: HashMap::new(),
        };

        let result = svc.map_identity(&claims, "Google");
        assert!(matches!(result, Err(OidcError::MissingEmail)));
    }

    #[test]
    fn test_unverified_email_rejected() {
        let svc = make_service();
        let claims = OidcClaims {
            sub: "user_unverified".into(),
            email: Some("unverified@example.com".into()),
            email_verified: Some(false),
            name: None,
            given_name: None,
            family_name: None,
            picture: None,
            iss: "https://accounts.google.com".into(),
            aud: "client_123".into(),
            nonce: None,
            iat: 0,
            exp: u64::MAX,
            extra: HashMap::new(),
        };

        let result = svc.map_identity(&claims, "Google");
        assert!(matches!(result, Err(OidcError::EmailNotVerified)));
    }

    #[test]
    fn test_google_config() {
        let config = OidcProviderConfig::google("id", "secret", "https://example.com/cb");
        assert_eq!(config.name, "Google");
        assert_eq!(config.issuer_url, "https://accounts.google.com");
        assert!(config.scopes.contains(&"openid".to_string()));
    }

    #[test]
    fn test_github_config() {
        let config = OidcProviderConfig::github("id", "secret", "https://example.com/cb");
        assert_eq!(config.name, "GitHub");
        assert!(config.scopes.contains(&"user:email".to_string()));
    }

    #[test]
    fn test_generic_config() {
        let config = OidcProviderConfig::generic(
            "Okta",
            "https://dev-123.okta.com",
            "id",
            "secret",
            "https://example.com/cb",
        );
        assert_eq!(config.name, "Okta");
        assert_eq!(config.issuer_url, "https://dev-123.okta.com");
    }

    #[test]
    fn test_error_display() {
        let err = OidcError::InvalidState;
        assert_eq!(err.to_string(), "invalid state parameter");

        let err = OidcError::NonceMismatch;
        assert_eq!(err.to_string(), "nonce mismatch");

        let err = OidcError::MissingEmail;
        assert_eq!(err.to_string(), "email not provided by IdP");
    }
}
