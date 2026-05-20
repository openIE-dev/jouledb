//! Authentication and Authorization for JouleDB Server
//!
//! Thin wrapper around `inv-auth` that preserves the existing public API.
//! JWT validation/revocation delegates to inv-auth's TokenService + RevocationService.
//! API key management delegates to inv-auth's ApiKeyService.
//! Session management delegates to inv-auth's SessionManager.
//! RBAC delegates to inv-auth's Role/Permission types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Types (preserved public API)
// ============================================================================

/// Authentication error
#[derive(Debug, Clone, PartialEq)]
pub enum AuthError {
    InvalidCredentials,
    TokenExpired,
    TokenRevoked,
    ApiKeyExpired,
    ApiKeyNotFound,
    SessionExpired,
    SessionNotFound,
    InsufficientPermissions,
    InvalidToken(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCredentials => write!(f, "Invalid credentials"),
            Self::TokenExpired => write!(f, "Token has expired"),
            Self::TokenRevoked => write!(f, "Token has been revoked"),
            Self::ApiKeyExpired => write!(f, "API key has expired"),
            Self::ApiKeyNotFound => write!(f, "API key not found"),
            Self::SessionExpired => write!(f, "Session has expired"),
            Self::SessionNotFound => write!(f, "Session not found"),
            Self::InsufficientPermissions => write!(f, "Insufficient permissions"),
            Self::InvalidToken(msg) => write!(f, "Invalid token: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

/// Convert inv-auth errors to our local AuthError
fn from_inv_auth_error(e: inv_auth::AuthError) -> AuthError {
    match e {
        inv_auth::AuthError::InvalidCredentials => AuthError::InvalidCredentials,
        inv_auth::AuthError::TokenExpired => AuthError::TokenExpired,
        inv_auth::AuthError::TokenRevoked => AuthError::TokenRevoked,
        inv_auth::AuthError::InvalidToken(msg) => AuthError::InvalidToken(msg),
        inv_auth::AuthError::ApiKeyNotFound => AuthError::ApiKeyNotFound,
        inv_auth::AuthError::ApiKeyExpired => AuthError::ApiKeyExpired,
        inv_auth::AuthError::ApiKeyRevoked => AuthError::ApiKeyNotFound,
        inv_auth::AuthError::SessionNotFound => AuthError::SessionNotFound,
        inv_auth::AuthError::InsufficientPermissions { .. } => AuthError::InsufficientPermissions,
        inv_auth::AuthError::MissingAuth => AuthError::InvalidCredentials,
        inv_auth::AuthError::OrgMismatch { .. } => {
            AuthError::InvalidToken("org mismatch".to_string())
        }
    }
}

/// API Key (preserved public shape)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique key ID
    pub key_id: String,
    /// Hashed key value (SHA-256)
    pub key_hash: [u8; 32],
    /// Associated client ID
    pub client_id: String,
    /// Assigned roles
    pub roles: Vec<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Expiration timestamp (None = never expires)
    pub expires_at: Option<u64>,
    /// Last used timestamp
    pub last_used: Option<u64>,
    /// Request count
    pub request_count: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Session (preserved public shape)
#[derive(Debug, Clone)]
pub struct Session {
    /// Session ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Roles
    pub roles: Vec<String>,
    /// Created timestamp
    pub created_at: u64,
    /// Expires timestamp
    pub expires_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

/// JWT Claims - standard JWT claims plus custom fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at (timestamp)
    pub iat: u64,
    /// Expiration (timestamp)
    pub exp: u64,
    /// Not before (timestamp)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbf: Option<u64>,
    /// Issuer
    pub iss: String,
    /// Audience
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// JWT ID (unique identifier)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
    /// Roles (custom claim)
    pub roles: Vec<String>,
    /// Additional claims
    #[serde(flatten)]
    pub additional: HashMap<String, serde_json::Value>,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// JWT expiration duration (seconds)
    pub jwt_expiration: u64,
    /// Session expiration duration (seconds)
    pub session_expiration: u64,
    /// Refresh token expiration (seconds)
    pub refresh_expiration: u64,
    /// Session inactivity timeout (seconds)
    pub session_timeout: u64,
    /// JWT issuer
    pub jwt_issuer: String,
    /// JWT audience
    pub jwt_audience: Option<String>,
    /// Enable refresh tokens
    pub enable_refresh: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_expiration: 3600,        // 1 hour
            session_expiration: 86400,   // 24 hours
            refresh_expiration: 2592000, // 30 days
            session_timeout: 1800,       // 30 minutes
            jwt_issuer: "joule_db".to_string(),
            jwt_audience: None,
            enable_refresh: true,
        }
    }
}

/// Authentication statistics
#[derive(Debug, Clone)]
pub struct AuthStats {
    pub active_sessions: usize,
    pub active_api_keys: usize,
    pub blacklisted_tokens: usize,
}

// ============================================================================
// Authentication Manager - delegates to inv-auth
// ============================================================================

/// Authentication Manager - delegates JWT, API keys, sessions, and revocation
/// to inv-auth while preserving the existing joule-db-server public API.
///
/// JWT encode/decode uses jsonwebtoken (via inv-auth's dependency) because
/// joule-db-server's JwtClaims shape (string roles, additional claims) differs
/// from inv-auth::Claims (typed Role enum, org-scoped). The token blacklist
/// delegates to inv-auth::RevocationService.
///
/// API key and session management delegate to inv-auth::ApiKeyService and
/// inv-auth::SessionManager respectively, with thin adapters to map between
/// the inv-auth types and the existing joule-db-server public types.
pub struct AuthenticationManager {
    /// JWT encoding key (from jsonwebtoken, shared dep with inv-auth)
    encoding_key: jsonwebtoken::EncodingKey,
    /// JWT decoding key
    decoding_key: jsonwebtoken::DecodingKey,
    /// inv-auth API key service
    api_key_service: inv_auth::ApiKeyService,
    /// inv-auth session manager
    session_manager: inv_auth::SessionManager,
    /// inv-auth revocation service for token blacklist
    revocation_service: inv_auth::RevocationService,
    /// Local tracking: key_id -> (inv_auth Uuid, client_id, roles, request_count)
    api_key_meta: Arc<RwLock<HashMap<String, ApiKeyMeta>>>,
    /// Local session tracking: session_id_string -> (inv_auth Uuid, expiry, timeout tracking)
    session_meta: Arc<RwLock<HashMap<String, SessionMeta>>>,
    /// Configuration
    config: AuthConfig,
}

#[derive(Debug, Clone)]
struct ApiKeyMeta {
    inv_id: uuid::Uuid,
    client_id: String,
    roles: Vec<String>,
    request_count: u64,
}

#[derive(Debug, Clone)]
struct SessionMeta {
    inv_id: uuid::Uuid,
    user_id: String,
    roles: Vec<String>,
    created_at: u64,
    expires_at: u64,
    last_activity: u64,
    ip_address: Option<String>,
    user_agent: Option<String>,
}

impl AuthenticationManager {
    /// Create new authentication manager with a secret key
    ///
    /// # Arguments
    /// * `jwt_secret` - Secret key for HMAC signing (should be at least 32 bytes)
    pub fn new(jwt_secret: Vec<u8>) -> Self {
        Self::with_config(jwt_secret, AuthConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(jwt_secret: Vec<u8>, config: AuthConfig) -> Self {
        // Ensure minimum key length for security
        let secret = if jwt_secret.len() < 32 {
            let mut padded = jwt_secret;
            padded.resize(32, 0);
            padded
        } else {
            jwt_secret
        };

        Self {
            encoding_key: jsonwebtoken::EncodingKey::from_secret(&secret),
            decoding_key: jsonwebtoken::DecodingKey::from_secret(&secret),
            api_key_service: inv_auth::ApiKeyService::new(),
            session_manager: inv_auth::SessionManager::new(),
            revocation_service: inv_auth::RevocationService::new(),
            api_key_meta: Arc::new(RwLock::new(HashMap::new())),
            session_meta: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn generate_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }

    // ========================================================================
    // JWT Token Management
    // ========================================================================

    /// Generate JWT token
    pub fn generate_jwt(&self, user_id: &str, roles: Vec<String>) -> Result<String, AuthError> {
        self.generate_jwt_with_claims(user_id, roles, HashMap::new())
    }

    /// Generate JWT token with additional claims
    pub fn generate_jwt_with_claims(
        &self,
        user_id: &str,
        roles: Vec<String>,
        additional: HashMap<String, serde_json::Value>,
    ) -> Result<String, AuthError> {
        let now = Self::current_timestamp();
        let jti = Self::generate_id();

        let claims = JwtClaims {
            sub: user_id.to_string(),
            iat: now,
            exp: now + self.config.jwt_expiration,
            nbf: Some(now),
            iss: self.config.jwt_issuer.clone(),
            aud: self.config.jwt_audience.clone(),
            jti: Some(jti),
            roles,
            additional,
        };

        let header = jsonwebtoken::Header::default();

        jsonwebtoken::encode(&header, &claims, &self.encoding_key)
            .map_err(|e| AuthError::InvalidToken(format!("Failed to encode JWT: {}", e)))
    }

    /// Validate JWT token
    pub fn validate_jwt(&self, token: &str) -> Result<JwtClaims, AuthError> {
        let mut validation = jsonwebtoken::Validation::default();
        validation.set_issuer(&[&self.config.jwt_issuer]);
        validation.leeway = 0;

        if let Some(ref aud) = self.config.jwt_audience {
            validation.set_audience(&[aud]);
        } else {
            validation.validate_aud = false;
        }

        let token_data = jsonwebtoken::decode::<JwtClaims>(token, &self.decoding_key, &validation)
            .map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                jsonwebtoken::errors::ErrorKind::InvalidToken => {
                    AuthError::InvalidToken("Invalid token format".to_string())
                }
                jsonwebtoken::errors::ErrorKind::InvalidSignature => {
                    AuthError::InvalidToken("Invalid signature".to_string())
                }
                jsonwebtoken::errors::ErrorKind::InvalidIssuer => {
                    AuthError::InvalidToken("Invalid issuer".to_string())
                }
                jsonwebtoken::errors::ErrorKind::InvalidAudience => {
                    AuthError::InvalidToken("Invalid audience".to_string())
                }
                _ => AuthError::InvalidToken(format!("JWT error: {}", e)),
            })?;

        let claims = token_data.claims;

        // Check revocation using inv-auth's RevocationService
        if let Some(ref jti) = claims.jti {
            if self.revocation_service.is_revoked(jti) {
                return Err(AuthError::TokenRevoked);
            }
        }

        Ok(claims)
    }

    /// Refresh JWT token
    pub fn refresh_token(&self, token: &str) -> Result<String, AuthError> {
        if !self.config.enable_refresh {
            return Err(AuthError::InvalidToken("Refresh disabled".to_string()));
        }

        let claims = self.validate_jwt(token)?;
        self.generate_jwt_with_claims(&claims.sub, claims.roles, claims.additional)
    }

    /// Revoke JWT token (delegates blacklist to inv-auth::RevocationService)
    pub fn revoke_token(&self, token: &str) -> Result<(), AuthError> {
        let claims = self.validate_jwt(token)?;

        if let Some(jti) = claims.jti {
            self.revocation_service.revoke_token(&jti, claims.exp);
        }

        Ok(())
    }

    // ========================================================================
    // API Key Management (delegates to inv-auth::ApiKeyService)
    // ========================================================================

    /// Create API key
    pub fn create_api_key(
        &self,
        client_id: &str,
        roles: Vec<String>,
        expires_in_secs: Option<u64>,
    ) -> Result<(String, String), AuthError> {
        let expires_at = expires_in_secs.map(|secs| {
            chrono::Utc::now() + chrono::TimeDelta::seconds(secs as i64)
        });

        // Delegate creation to inv-auth
        let result = self.api_key_service.create(
            "joule_db", // org
            client_id,  // name (we use client_id as the key name)
            inv_auth::Role::Admin, // placeholder role (actual authz uses string roles)
            vec![],     // no specific permissions (string roles used instead)
            expires_at,
        );

        let key_id = format!("key_{}", result.api_key.id);
        let raw_key = result.plaintext_key;

        // Track local metadata for the public API shape
        crate::lock_util::write_lock(&self.api_key_meta).insert(
            key_id.clone(),
            ApiKeyMeta {
                inv_id: result.api_key.id,
                client_id: client_id.to_string(),
                roles: roles.clone(),
                request_count: 0,
            },
        );

        Ok((key_id, raw_key))
    }

    /// Validate API key - delegates to inv-auth::ApiKeyService
    pub fn validate_api_key(&self, key: &str) -> Result<ApiKey, AuthError> {
        let inv_key = self.api_key_service.validate(key).map_err(from_inv_auth_error)?;

        let key_id = format!("key_{}", inv_key.id);

        // Update request count in local meta
        let mut meta_guard = crate::lock_util::write_lock(&self.api_key_meta);
        let meta = meta_guard.get_mut(&key_id);

        let (client_id, roles, request_count) = if let Some(m) = meta {
            m.request_count += 1;
            (m.client_id.clone(), m.roles.clone(), m.request_count)
        } else {
            // Key was created externally or meta was lost; use inv-auth data
            (inv_key.name.clone(), vec![], 1)
        };

        let now = Self::current_timestamp();
        Ok(ApiKey {
            key_id,
            key_hash: [0u8; 32], // Hash not exposed by inv-auth (stored internally)
            client_id,
            roles,
            created_at: inv_key.created_at.timestamp() as u64,
            expires_at: inv_key.expires_at.map(|t| t.timestamp() as u64),
            last_used: Some(now),
            request_count,
            metadata: HashMap::new(),
        })
    }

    /// Revoke API key by ID
    pub fn revoke_api_key(&self, key_id: &str) -> Result<(), AuthError> {
        let meta_guard = crate::lock_util::read_lock(&self.api_key_meta);
        let meta = meta_guard.get(key_id).ok_or(AuthError::ApiKeyNotFound)?;
        let inv_id = meta.inv_id;
        drop(meta_guard);

        self.api_key_service.revoke(inv_id).map_err(from_inv_auth_error)?;
        crate::lock_util::write_lock(&self.api_key_meta).remove(key_id);

        Ok(())
    }

    /// List all API keys (returns key_id and metadata, not the actual keys)
    pub fn list_api_keys(&self) -> Vec<(String, String, Vec<String>)> {
        crate::lock_util::read_lock(&self.api_key_meta)
            .values()
            .map(|m| (format!("key_{}", m.inv_id), m.client_id.clone(), m.roles.clone()))
            .collect()
    }

    // ========================================================================
    // Session Management (delegates to inv-auth::SessionManager)
    // ========================================================================

    /// Create session
    pub fn create_session(
        &self,
        user_id: &str,
        roles: Vec<String>,
        ip_address: Option<String>,
        user_agent: Option<String>,
    ) -> Session {
        let ip_addr = ip_address
            .as_deref()
            .and_then(|s| s.parse::<std::net::IpAddr>().ok());

        let inv_session = self.session_manager.create(
            "joule_db",
            user_id,
            inv_auth::Role::Admin, // placeholder (actual authz uses string roles)
            ip_addr,
            user_agent.clone(),
        );

        let now = Self::current_timestamp();
        let session_id = format!("sess_{}", inv_session.id);

        let meta = SessionMeta {
            inv_id: inv_session.id,
            user_id: user_id.to_string(),
            roles: roles.clone(),
            created_at: now,
            expires_at: now + self.config.session_expiration,
            last_activity: now,
            ip_address: ip_address.clone(),
            user_agent: user_agent.clone(),
        };

        crate::lock_util::write_lock(&self.session_meta)
            .insert(session_id.clone(), meta);

        Session {
            id: session_id,
            user_id: user_id.to_string(),
            roles,
            created_at: now,
            expires_at: now + self.config.session_expiration,
            last_activity: now,
            ip_address,
            user_agent,
        }
    }

    /// Validate session
    pub fn validate_session(&self, session_id: &str) -> Result<Session, AuthError> {
        let mut sessions = crate::lock_util::write_lock(&self.session_meta);
        let meta = sessions.get_mut(session_id).ok_or(AuthError::SessionNotFound)?;

        let now = Self::current_timestamp();

        // Check expiration
        if now > meta.expires_at {
            let inv_id = meta.inv_id;
            sessions.remove(session_id);
            let _ = self.session_manager.terminate(inv_id);
            return Err(AuthError::SessionExpired);
        }

        // Check inactivity timeout
        if now - meta.last_activity > self.config.session_timeout {
            let inv_id = meta.inv_id;
            sessions.remove(session_id);
            let _ = self.session_manager.terminate(inv_id);
            return Err(AuthError::SessionExpired);
        }

        // Update last activity (both locally and in inv-auth)
        meta.last_activity = now;
        let _ = self.session_manager.touch(meta.inv_id);

        Ok(Session {
            id: session_id.to_string(),
            user_id: meta.user_id.clone(),
            roles: meta.roles.clone(),
            created_at: meta.created_at,
            expires_at: meta.expires_at,
            last_activity: now,
            ip_address: meta.ip_address.clone(),
            user_agent: meta.user_agent.clone(),
        })
    }

    /// Destroy session
    pub fn destroy_session(&self, session_id: &str) -> Result<(), AuthError> {
        let meta = crate::lock_util::write_lock(&self.session_meta)
            .remove(session_id)
            .ok_or(AuthError::SessionNotFound)?;

        let _ = self.session_manager.terminate(meta.inv_id);
        Ok(())
    }

    /// Get all active sessions for a user
    pub fn get_user_sessions(&self, user_id: &str) -> Vec<Session> {
        crate::lock_util::read_lock(&self.session_meta)
            .iter()
            .filter(|(_, m)| m.user_id == user_id)
            .map(|(sid, m)| Session {
                id: sid.clone(),
                user_id: m.user_id.clone(),
                roles: m.roles.clone(),
                created_at: m.created_at,
                expires_at: m.expires_at,
                last_activity: m.last_activity,
                ip_address: m.ip_address.clone(),
                user_agent: m.user_agent.clone(),
            })
            .collect()
    }

    /// Destroy all sessions for a user
    pub fn destroy_user_sessions(&self, user_id: &str) -> usize {
        let mut sessions = crate::lock_util::write_lock(&self.session_meta);
        let to_remove: Vec<(String, uuid::Uuid)> = sessions
            .iter()
            .filter(|(_, m)| m.user_id == user_id)
            .map(|(sid, m)| (sid.clone(), m.inv_id))
            .collect();

        let count = to_remove.len();
        for (sid, inv_id) in to_remove {
            sessions.remove(&sid);
            let _ = self.session_manager.terminate(inv_id);
        }
        count
    }

    // ========================================================================
    // Maintenance (delegates to inv-auth cleanup)
    // ========================================================================

    /// Clean expired sessions and blacklisted tokens
    pub fn cleanup(&self) {
        let now = Self::current_timestamp();

        // Clean sessions
        let expired_sessions: Vec<(String, uuid::Uuid)> =
            crate::lock_util::read_lock(&self.session_meta)
                .iter()
                .filter(|(_, m)| m.expires_at <= now)
                .map(|(sid, m)| (sid.clone(), m.inv_id))
                .collect();

        if !expired_sessions.is_empty() {
            let mut sessions = crate::lock_util::write_lock(&self.session_meta);
            for (sid, inv_id) in expired_sessions {
                sessions.remove(&sid);
                let _ = self.session_manager.terminate(inv_id);
            }
        }

        // Clean revocation list via inv-auth
        self.revocation_service.cleanup_expired();
    }

    /// Get statistics
    pub fn stats(&self) -> AuthStats {
        let revocation_stats = self.revocation_service.stats();
        AuthStats {
            active_sessions: crate::lock_util::read_lock(&self.session_meta).len(),
            active_api_keys: crate::lock_util::read_lock(&self.api_key_meta).len(),
            blacklisted_tokens: revocation_stats.active_entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_manager_creation() {
        let manager = AuthenticationManager::new(b"secret_key_at_least_32_bytes_long".to_vec());
        assert_eq!(manager.stats().active_sessions, 0);
    }

    #[test]
    fn test_jwt_generation_and_validation() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let token = manager
            .generate_jwt("user123", vec!["admin".to_string()])
            .unwrap();

        let claims = manager.validate_jwt(&token).unwrap();
        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.roles, vec!["admin"]);
        assert_eq!(claims.iss, "joule_db");
    }

    #[test]
    fn test_jwt_expiration() {
        let config = AuthConfig {
            jwt_expiration: 1, // Expire in 1 second
            ..Default::default()
        };
        let manager = AuthenticationManager::with_config(
            b"test_secret_key_must_be_32_bytes".to_vec(),
            config,
        );

        let token = manager.generate_jwt("user", vec![]).unwrap();

        // Token should be expired after waiting
        std::thread::sleep(std::time::Duration::from_secs(2));
        let result = manager.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_jwt_revocation() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let token = manager.generate_jwt("user", vec![]).unwrap();

        // Validate before revocation
        assert!(manager.validate_jwt(&token).is_ok());

        // Revoke
        manager.revoke_token(&token).unwrap();

        // Validate after revocation
        let result = manager.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::TokenRevoked)));
    }

    #[test]
    fn test_jwt_invalid_signature() {
        let manager1 = AuthenticationManager::new(b"secret1_must_be_32_bytes_long!!".to_vec());
        let manager2 = AuthenticationManager::new(b"secret2_must_be_32_bytes_long!!".to_vec());

        let token = manager1.generate_jwt("user", vec![]).unwrap();

        // Different secret should fail
        let result = manager2.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::InvalidToken(_))));
    }

    #[test]
    fn test_api_key() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let (key_id, raw_key) = manager
            .create_api_key("client1", vec!["read".to_string()], None)
            .unwrap();

        assert!(key_id.starts_with("key_"));
        assert!(raw_key.starts_with("inv_key_"));

        let api_key = manager.validate_api_key(&raw_key).unwrap();
        assert_eq!(api_key.client_id, "client1");
        assert_eq!(api_key.request_count, 1);
    }

    #[test]
    fn test_api_key_not_found() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let result = manager.validate_api_key("invalid_key");
        assert!(matches!(result, Err(AuthError::ApiKeyNotFound)));
    }

    #[test]
    fn test_api_key_expiration() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let (_, raw_key) = manager
            .create_api_key("client1", vec![], Some(1)) // Expires in 1 second
            .unwrap();

        std::thread::sleep(std::time::Duration::from_secs(2));
        let result = manager.validate_api_key(&raw_key);
        assert!(matches!(result, Err(AuthError::ApiKeyExpired)));
    }

    #[test]
    fn test_api_key_revocation() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let (key_id, raw_key) = manager.create_api_key("client1", vec![], None).unwrap();

        // Revoke
        manager.revoke_api_key(&key_id).unwrap();

        // Should not be found
        let result = manager.validate_api_key(&raw_key);
        assert!(matches!(result, Err(AuthError::ApiKeyNotFound)));
    }

    #[test]
    fn test_session() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let session = manager.create_session(
            "user123",
            vec!["user".to_string()],
            Some("127.0.0.1".to_string()),
            None,
        );

        assert!(session.id.starts_with("sess_"));

        let validated = manager.validate_session(&session.id).unwrap();
        assert_eq!(validated.user_id, "user123");
    }

    #[test]
    fn test_session_not_found() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());
        let result = manager.validate_session("invalid_session");
        assert!(matches!(result, Err(AuthError::SessionNotFound)));
    }

    #[test]
    fn test_destroy_user_sessions() {
        let manager = AuthenticationManager::new(b"test_secret_key_must_be_32_bytes".to_vec());

        manager.create_session("user1", vec![], None, None);
        manager.create_session("user1", vec![], None, None);
        manager.create_session("user2", vec![], None, None);

        let destroyed = manager.destroy_user_sessions("user1");
        assert_eq!(destroyed, 2);

        let sessions = manager.get_user_sessions("user1");
        assert!(sessions.is_empty());
    }

    #[test]
    fn test_cleanup() {
        let config = AuthConfig {
            session_expiration: 0,
            ..Default::default()
        };
        let manager = AuthenticationManager::with_config(
            b"test_secret_key_must_be_32_bytes".to_vec(),
            config,
        );

        manager.create_session("user", vec![], None, None);

        std::thread::sleep(std::time::Duration::from_millis(100));
        manager.cleanup();

        assert_eq!(manager.stats().active_sessions, 0);
    }
}
