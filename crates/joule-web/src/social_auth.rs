//! Social authentication — provider abstraction, OAuth2 redirect flow,
//! profile normalization, account linking, and provider-specific claim mapping.
//!
//! Replaces passport.js, next-auth, Auth.js, and similar JS/TS social auth
//! libraries with a pure-Rust provider-agnostic authentication engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Social auth errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocialAuthError {
    /// Provider not registered.
    ProviderNotFound(String),
    /// Duplicate provider registration.
    DuplicateProvider(String),
    /// Invalid OAuth state parameter.
    InvalidState,
    /// Invalid authorization code.
    InvalidCode,
    /// Token exchange failed.
    TokenExchangeFailed(String),
    /// Profile field missing.
    ProfileFieldMissing(String),
    /// Account already linked to a different user.
    AccountAlreadyLinked { provider: String, external_id: String },
    /// User not found.
    UserNotFound(String),
    /// Link not found.
    LinkNotFound { user_id: String, provider: String },
    /// Invalid redirect URI.
    InvalidRedirectUri(String),
}

impl fmt::Display for SocialAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderNotFound(id) => write!(f, "provider not found: {id}"),
            Self::DuplicateProvider(id) => write!(f, "duplicate provider: {id}"),
            Self::InvalidState => write!(f, "invalid OAuth state parameter"),
            Self::InvalidCode => write!(f, "invalid authorization code"),
            Self::TokenExchangeFailed(msg) => write!(f, "token exchange failed: {msg}"),
            Self::ProfileFieldMissing(field) => write!(f, "profile field missing: {field}"),
            Self::AccountAlreadyLinked {
                provider,
                external_id,
            } => write!(f, "account {external_id} already linked to {provider}"),
            Self::UserNotFound(id) => write!(f, "user not found: {id}"),
            Self::LinkNotFound { user_id, provider } => {
                write!(f, "no link for user {user_id} to provider {provider}")
            }
            Self::InvalidRedirectUri(uri) => write!(f, "invalid redirect URI: {uri}"),
        }
    }
}

impl std::error::Error for SocialAuthError {}

// ── Types ──────────────────────────────────────────────────────

/// Known OAuth2 provider types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderType {
    Google,
    GitHub,
    Microsoft,
    Apple,
    Facebook,
    Twitter,
    LinkedIn,
    Custom(String),
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Google => write!(f, "google"),
            Self::GitHub => write!(f, "github"),
            Self::Microsoft => write!(f, "microsoft"),
            Self::Apple => write!(f, "apple"),
            Self::Facebook => write!(f, "facebook"),
            Self::Twitter => write!(f, "twitter"),
            Self::LinkedIn => write!(f, "linkedin"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Configuration for an OAuth2 provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub client_id: String,
    pub client_secret: String,
    pub authorize_url: String,
    pub token_url: String,
    pub userinfo_url: String,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
    /// Mapping from provider-specific claim names to normalized fields.
    pub claim_mapping: ClaimMapping,
}

/// Maps provider-specific JWT/profile claims to normalized profile fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimMapping {
    pub id_claim: String,
    pub email_claim: String,
    pub name_claim: String,
    pub avatar_claim: Option<String>,
    pub locale_claim: Option<String>,
}

impl ClaimMapping {
    /// Default mapping for Google.
    pub fn google() -> Self {
        Self {
            id_claim: "sub".to_string(),
            email_claim: "email".to_string(),
            name_claim: "name".to_string(),
            avatar_claim: Some("picture".to_string()),
            locale_claim: Some("locale".to_string()),
        }
    }

    /// Default mapping for GitHub.
    pub fn github() -> Self {
        Self {
            id_claim: "id".to_string(),
            email_claim: "email".to_string(),
            name_claim: "name".to_string(),
            avatar_claim: Some("avatar_url".to_string()),
            locale_claim: None,
        }
    }

    /// Default mapping for Microsoft.
    pub fn microsoft() -> Self {
        Self {
            id_claim: "sub".to_string(),
            email_claim: "email".to_string(),
            name_claim: "displayName".to_string(),
            avatar_claim: None,
            locale_claim: Some("preferred_language".to_string()),
        }
    }

    /// Generic mapping.
    pub fn generic() -> Self {
        Self {
            id_claim: "sub".to_string(),
            email_claim: "email".to_string(),
            name_claim: "name".to_string(),
            avatar_claim: None,
            locale_claim: None,
        }
    }
}

/// Normalized user profile from any provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedProfile {
    pub provider: String,
    pub external_id: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub avatar_url: Option<String>,
    pub locale: Option<String>,
    pub raw_claims: HashMap<String, String>,
}

/// An OAuth2 authorization request (state before redirect).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    pub provider_id: String,
    pub state: String,
    pub nonce: Option<String>,
    pub redirect_uri: String,
    pub authorize_url: String,
}

/// Result of exchanging the authorization code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in_secs: Option<u64>,
    pub token_type: String,
    pub id_token: Option<String>,
}

/// A link between a local user and an external provider account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountLink {
    pub user_id: String,
    pub provider: String,
    pub external_id: String,
    pub linked_at_secs: u64,
    pub profile: NormalizedProfile,
}

// ── Engine ─────────────────────────────────────────────────────

/// The social auth engine managing providers, flows, and account links.
#[derive(Debug, Clone)]
pub struct SocialAuthEngine {
    providers: HashMap<String, ProviderConfig>,
    /// Pending authorization states: state -> provider_id.
    pending_states: HashMap<String, String>,
    /// Account links: (provider, external_id) -> AccountLink.
    links: HashMap<(String, String), AccountLink>,
    /// User to links index: user_id -> Vec<(provider, external_id)>.
    user_links: HashMap<String, Vec<(String, String)>>,
}

impl SocialAuthEngine {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            pending_states: HashMap::new(),
            links: HashMap::new(),
            user_links: HashMap::new(),
        }
    }

    /// Register an OAuth2 provider.
    pub fn register_provider(&mut self, config: ProviderConfig) -> Result<(), SocialAuthError> {
        let key = config.provider_type.to_string();
        if self.providers.contains_key(&key) {
            return Err(SocialAuthError::DuplicateProvider(key));
        }
        self.providers.insert(key, config);
        Ok(())
    }

    /// Get a registered provider.
    pub fn get_provider(&self, provider_id: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider_id)
    }

    /// List registered provider IDs.
    pub fn list_providers(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.providers.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Build an authorization URL for the OAuth2 redirect.
    pub fn build_authorize_url(
        &mut self,
        provider_id: &str,
        state: &str,
        nonce: Option<&str>,
    ) -> Result<AuthorizationRequest, SocialAuthError> {
        let config = self
            .providers
            .get(provider_id)
            .ok_or_else(|| SocialAuthError::ProviderNotFound(provider_id.to_string()))?;

        if config.redirect_uri.is_empty() {
            return Err(SocialAuthError::InvalidRedirectUri(
                config.redirect_uri.clone(),
            ));
        }

        let scopes = config.scopes.join(" ");
        let mut url = format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}",
            config.authorize_url, config.client_id, config.redirect_uri, scopes, state
        );

        if let Some(n) = nonce {
            url.push_str(&format!("&nonce={n}"));
        }

        self.pending_states
            .insert(state.to_string(), provider_id.to_string());

        Ok(AuthorizationRequest {
            provider_id: provider_id.to_string(),
            state: state.to_string(),
            nonce: nonce.map(|s| s.to_string()),
            redirect_uri: config.redirect_uri.clone(),
            authorize_url: url,
        })
    }

    /// Validate the state parameter from the callback and return the provider ID.
    pub fn validate_state(&mut self, state: &str) -> Result<String, SocialAuthError> {
        self.pending_states
            .remove(state)
            .ok_or(SocialAuthError::InvalidState)
    }

    /// Normalize raw claims from a provider into a standard profile.
    pub fn normalize_profile(
        &self,
        provider_id: &str,
        raw_claims: HashMap<String, String>,
    ) -> Result<NormalizedProfile, SocialAuthError> {
        let config = self
            .providers
            .get(provider_id)
            .ok_or_else(|| SocialAuthError::ProviderNotFound(provider_id.to_string()))?;

        let mapping = &config.claim_mapping;

        let external_id = raw_claims
            .get(&mapping.id_claim)
            .ok_or_else(|| SocialAuthError::ProfileFieldMissing(mapping.id_claim.clone()))?
            .clone();

        let email = raw_claims.get(&mapping.email_claim).cloned();
        let name = raw_claims.get(&mapping.name_claim).cloned();
        let avatar_url = mapping
            .avatar_claim
            .as_ref()
            .and_then(|k| raw_claims.get(k))
            .cloned();
        let locale = mapping
            .locale_claim
            .as_ref()
            .and_then(|k| raw_claims.get(k))
            .cloned();

        Ok(NormalizedProfile {
            provider: provider_id.to_string(),
            external_id,
            email,
            name,
            avatar_url,
            locale,
            raw_claims,
        })
    }

    /// Link a provider account to a local user.
    pub fn link_account(
        &mut self,
        user_id: &str,
        profile: NormalizedProfile,
        linked_at_secs: u64,
    ) -> Result<(), SocialAuthError> {
        let key = (profile.provider.clone(), profile.external_id.clone());

        // Check if already linked to a different user.
        if let Some(existing) = self.links.get(&key) {
            if existing.user_id != user_id {
                return Err(SocialAuthError::AccountAlreadyLinked {
                    provider: profile.provider.clone(),
                    external_id: profile.external_id.clone(),
                });
            }
        }

        let link = AccountLink {
            user_id: user_id.to_string(),
            provider: profile.provider.clone(),
            external_id: profile.external_id.clone(),
            linked_at_secs,
            profile,
        };

        let key_clone = key.clone();
        self.links.insert(key_clone, link);
        self.user_links
            .entry(user_id.to_string())
            .or_default()
            .push(key);

        Ok(())
    }

    /// Unlink a provider account from a user.
    pub fn unlink_account(
        &mut self,
        user_id: &str,
        provider_id: &str,
    ) -> Result<(), SocialAuthError> {
        let user_link_list = self
            .user_links
            .get_mut(user_id)
            .ok_or_else(|| SocialAuthError::LinkNotFound {
                user_id: user_id.to_string(),
                provider: provider_id.to_string(),
            })?;

        let idx = user_link_list
            .iter()
            .position(|(p, _)| p == provider_id)
            .ok_or_else(|| SocialAuthError::LinkNotFound {
                user_id: user_id.to_string(),
                provider: provider_id.to_string(),
            })?;

        let key = user_link_list.remove(idx);
        self.links.remove(&key);

        if user_link_list.is_empty() {
            self.user_links.remove(user_id);
        }

        Ok(())
    }

    /// Find a user by provider and external ID.
    pub fn find_user_by_provider(
        &self,
        provider_id: &str,
        external_id: &str,
    ) -> Option<&AccountLink> {
        let key = (provider_id.to_string(), external_id.to_string());
        self.links.get(&key)
    }

    /// Get all linked providers for a user.
    pub fn user_providers(&self, user_id: &str) -> Vec<&AccountLink> {
        self.user_links
            .get(user_id)
            .map(|keys| {
                keys.iter()
                    .filter_map(|key| self.links.get(key))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Count of pending authorization states.
    pub fn pending_count(&self) -> usize {
        self.pending_states.len()
    }

    /// Count of linked accounts.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }
}

impl Default for SocialAuthEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn google_config() -> ProviderConfig {
        ProviderConfig {
            provider_type: ProviderType::Google,
            client_id: "google-client-id".to_string(),
            client_secret: "google-secret".to_string(),
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            userinfo_url: "https://openidconnect.googleapis.com/v1/userinfo".to_string(),
            redirect_uri: "https://app.example.com/callback".to_string(),
            scopes: vec!["openid".to_string(), "email".to_string(), "profile".to_string()],
            claim_mapping: ClaimMapping::google(),
        }
    }

    fn github_config() -> ProviderConfig {
        ProviderConfig {
            provider_type: ProviderType::GitHub,
            client_id: "gh-client-id".to_string(),
            client_secret: "gh-secret".to_string(),
            authorize_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            userinfo_url: "https://api.github.com/user".to_string(),
            redirect_uri: "https://app.example.com/callback/github".to_string(),
            scopes: vec!["user:email".to_string()],
            claim_mapping: ClaimMapping::github(),
        }
    }

    fn setup_engine() -> SocialAuthEngine {
        let mut engine = SocialAuthEngine::new();
        engine.register_provider(google_config()).unwrap();
        engine.register_provider(github_config()).unwrap();
        engine
    }

    #[test]
    fn test_register_provider() {
        let engine = setup_engine();
        assert!(engine.get_provider("google").is_some());
        assert!(engine.get_provider("github").is_some());
        assert!(engine.get_provider("twitter").is_none());
    }

    #[test]
    fn test_duplicate_provider() {
        let mut engine = setup_engine();
        let err = engine.register_provider(google_config()).unwrap_err();
        assert!(matches!(err, SocialAuthError::DuplicateProvider(_)));
    }

    #[test]
    fn test_list_providers() {
        let engine = setup_engine();
        let providers = engine.list_providers();
        assert_eq!(providers, vec!["github", "google"]);
    }

    #[test]
    fn test_build_authorize_url() {
        let mut engine = setup_engine();
        let req = engine
            .build_authorize_url("google", "random-state-123", Some("nonce-456"))
            .unwrap();

        assert!(req.authorize_url.contains("client_id=google-client-id"));
        assert!(req.authorize_url.contains("state=random-state-123"));
        assert!(req.authorize_url.contains("nonce=nonce-456"));
        assert!(req.authorize_url.contains("response_type=code"));
        assert_eq!(req.provider_id, "google");
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn test_build_authorize_url_unknown_provider() {
        let mut engine = setup_engine();
        let err = engine
            .build_authorize_url("twitter", "state", None)
            .unwrap_err();
        assert!(matches!(err, SocialAuthError::ProviderNotFound(_)));
    }

    #[test]
    fn test_validate_state() {
        let mut engine = setup_engine();
        engine
            .build_authorize_url("google", "my-state", None)
            .unwrap();

        let provider = engine.validate_state("my-state").unwrap();
        assert_eq!(provider, "google");
        // State is consumed
        assert!(engine.validate_state("my-state").is_err());
    }

    #[test]
    fn test_validate_invalid_state() {
        let mut engine = setup_engine();
        let err = engine.validate_state("nonexistent").unwrap_err();
        assert_eq!(err, SocialAuthError::InvalidState);
    }

    #[test]
    fn test_normalize_google_profile() {
        let engine = setup_engine();
        let mut claims = HashMap::new();
        claims.insert("sub".to_string(), "google-123".to_string());
        claims.insert("email".to_string(), "alice@gmail.com".to_string());
        claims.insert("name".to_string(), "Alice Smith".to_string());
        claims.insert(
            "picture".to_string(),
            "https://lh3.google.com/photo".to_string(),
        );
        claims.insert("locale".to_string(), "en-US".to_string());

        let profile = engine.normalize_profile("google", claims).unwrap();
        assert_eq!(profile.external_id, "google-123");
        assert_eq!(profile.email.as_deref(), Some("alice@gmail.com"));
        assert_eq!(profile.name.as_deref(), Some("Alice Smith"));
        assert!(profile.avatar_url.is_some());
        assert_eq!(profile.locale.as_deref(), Some("en-US"));
    }

    #[test]
    fn test_normalize_github_profile() {
        let engine = setup_engine();
        let mut claims = HashMap::new();
        claims.insert("id".to_string(), "gh-456".to_string());
        claims.insert("email".to_string(), "bob@github.com".to_string());
        claims.insert("name".to_string(), "Bob Jones".to_string());
        claims.insert(
            "avatar_url".to_string(),
            "https://avatars.github.com/u/456".to_string(),
        );

        let profile = engine.normalize_profile("github", claims).unwrap();
        assert_eq!(profile.external_id, "gh-456");
        assert_eq!(profile.provider, "github");
        // GitHub has no locale claim
        assert!(profile.locale.is_none());
    }

    #[test]
    fn test_normalize_missing_id() {
        let engine = setup_engine();
        let claims = HashMap::new();
        let err = engine.normalize_profile("google", claims).unwrap_err();
        assert!(matches!(err, SocialAuthError::ProfileFieldMissing(_)));
    }

    #[test]
    fn test_link_account() {
        let mut engine = setup_engine();
        let profile = NormalizedProfile {
            provider: "google".to_string(),
            external_id: "g-123".to_string(),
            email: Some("a@b.com".to_string()),
            name: Some("Alice".to_string()),
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };

        engine.link_account("user-1", profile, 1000).unwrap();
        assert_eq!(engine.link_count(), 1);

        let found = engine.find_user_by_provider("google", "g-123").unwrap();
        assert_eq!(found.user_id, "user-1");
    }

    #[test]
    fn test_link_already_linked_different_user() {
        let mut engine = setup_engine();
        let profile = NormalizedProfile {
            provider: "google".to_string(),
            external_id: "g-123".to_string(),
            email: None,
            name: None,
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };

        engine
            .link_account("user-1", profile.clone(), 1000)
            .unwrap();
        let err = engine.link_account("user-2", profile, 1001).unwrap_err();
        assert!(matches!(err, SocialAuthError::AccountAlreadyLinked { .. }));
    }

    #[test]
    fn test_link_same_user_idempotent() {
        let mut engine = setup_engine();
        let profile = NormalizedProfile {
            provider: "google".to_string(),
            external_id: "g-123".to_string(),
            email: None,
            name: None,
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };

        engine
            .link_account("user-1", profile.clone(), 1000)
            .unwrap();
        // Same user can re-link (update)
        engine.link_account("user-1", profile, 1001).unwrap();
        assert_eq!(engine.link_count(), 1);
    }

    #[test]
    fn test_unlink_account() {
        let mut engine = setup_engine();
        let profile = NormalizedProfile {
            provider: "google".to_string(),
            external_id: "g-123".to_string(),
            email: None,
            name: None,
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };

        engine.link_account("user-1", profile, 1000).unwrap();
        engine.unlink_account("user-1", "google").unwrap();
        assert_eq!(engine.link_count(), 0);
        assert!(engine.find_user_by_provider("google", "g-123").is_none());
    }

    #[test]
    fn test_unlink_not_found() {
        let mut engine = setup_engine();
        let err = engine.unlink_account("user-1", "google").unwrap_err();
        assert!(matches!(err, SocialAuthError::LinkNotFound { .. }));
    }

    #[test]
    fn test_user_providers() {
        let mut engine = setup_engine();

        let google_profile = NormalizedProfile {
            provider: "google".to_string(),
            external_id: "g-123".to_string(),
            email: None,
            name: None,
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };
        let github_profile = NormalizedProfile {
            provider: "github".to_string(),
            external_id: "gh-456".to_string(),
            email: None,
            name: None,
            avatar_url: None,
            locale: None,
            raw_claims: HashMap::new(),
        };

        engine
            .link_account("user-1", google_profile, 1000)
            .unwrap();
        engine
            .link_account("user-1", github_profile, 1001)
            .unwrap();

        let providers = engine.user_providers("user-1");
        assert_eq!(providers.len(), 2);
    }

    #[test]
    fn test_user_providers_empty() {
        let engine = setup_engine();
        let providers = engine.user_providers("nobody");
        assert!(providers.is_empty());
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::Google.to_string(), "google");
        assert_eq!(ProviderType::GitHub.to_string(), "github");
        assert_eq!(
            ProviderType::Custom("discord".to_string()).to_string(),
            "discord"
        );
    }

    #[test]
    fn test_error_display() {
        let e = SocialAuthError::InvalidState;
        assert_eq!(e.to_string(), "invalid OAuth state parameter");
    }

    #[test]
    fn test_default_engine() {
        let engine = SocialAuthEngine::default();
        assert_eq!(engine.link_count(), 0);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn test_claim_mapping_microsoft() {
        let m = ClaimMapping::microsoft();
        assert_eq!(m.id_claim, "sub");
        assert_eq!(m.name_claim, "displayName");
    }

    #[test]
    fn test_claim_mapping_generic() {
        let m = ClaimMapping::generic();
        assert_eq!(m.id_claim, "sub");
        assert!(m.avatar_claim.is_none());
    }

    #[test]
    fn test_multiple_states_pending() {
        let mut engine = setup_engine();
        engine
            .build_authorize_url("google", "state-1", None)
            .unwrap();
        engine
            .build_authorize_url("github", "state-2", None)
            .unwrap();
        assert_eq!(engine.pending_count(), 2);

        engine.validate_state("state-1").unwrap();
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn test_normalize_optional_fields_missing() {
        let engine = setup_engine();
        let mut claims = HashMap::new();
        claims.insert("sub".to_string(), "g-789".to_string());
        // No email, name, picture, locale

        let profile = engine.normalize_profile("google", claims).unwrap();
        assert_eq!(profile.external_id, "g-789");
        assert!(profile.email.is_none());
        assert!(profile.name.is_none());
        assert!(profile.avatar_url.is_none());
        assert!(profile.locale.is_none());
    }
}
