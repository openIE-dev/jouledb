//! Token management — access/refresh token pairs, token families for rotation
//! detection, token blacklisting, sliding refresh, and concurrent session limits.
//!
//! Replaces jsonwebtoken, jose, passport-jwt, and similar JS/TS token management
//! libraries with a pure-Rust session and token lifecycle engine.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Token engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    /// Token not found.
    TokenNotFound(String),
    /// Token has been revoked/blacklisted.
    TokenRevoked(String),
    /// Token has expired.
    TokenExpired(String),
    /// Token family compromised (rotation reuse detected).
    FamilyCompromised(String),
    /// Session limit exceeded.
    SessionLimitExceeded { user_id: String, limit: usize },
    /// Invalid token type for operation.
    InvalidTokenType { expected: String, got: String },
    /// User not found.
    UserNotFound(String),
}

impl fmt::Display for TokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenNotFound(id) => write!(f, "token not found: {id}"),
            Self::TokenRevoked(id) => write!(f, "token revoked: {id}"),
            Self::TokenExpired(id) => write!(f, "token expired: {id}"),
            Self::FamilyCompromised(fam) => write!(f, "token family compromised: {fam}"),
            Self::SessionLimitExceeded { user_id, limit } => {
                write!(f, "session limit {limit} exceeded for user {user_id}")
            }
            Self::InvalidTokenType { expected, got } => {
                write!(f, "expected token type {expected}, got {got}")
            }
            Self::UserNotFound(id) => write!(f, "user not found: {id}"),
        }
    }
}

impl std::error::Error for TokenError {}

// ── Types ──────────────────────────────────────────────────────

/// Token type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    Access,
    Refresh,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Access => write!(f, "access"),
            Self::Refresh => write!(f, "refresh"),
        }
    }
}

/// A token with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: String,
    pub user_id: String,
    pub token_type: TokenType,
    pub family_id: String,
    pub issued_at_secs: u64,
    pub expires_at_secs: u64,
    /// Generation within the family (incremented on rotation).
    pub generation: u64,
    /// Whether this token has been used for rotation (consumed).
    pub consumed: bool,
    /// Device/client identifier for session tracking.
    pub device_id: Option<String>,
}

impl Token {
    /// Check if token is expired at the given time.
    pub fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.expires_at_secs
    }

    /// Remaining lifetime in seconds, or 0 if expired.
    pub fn remaining_secs(&self, now_secs: u64) -> u64 {
        self.expires_at_secs.saturating_sub(now_secs)
    }
}

/// A token pair (access + refresh).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: Token,
    pub refresh_token: Token,
}

/// A token family tracks lineage for rotation-reuse detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFamily {
    pub id: String,
    pub user_id: String,
    pub created_at_secs: u64,
    pub current_generation: u64,
    pub compromised: bool,
    pub device_id: Option<String>,
}

/// Configuration for token lifetimes and limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub access_lifetime_secs: u64,
    pub refresh_lifetime_secs: u64,
    /// Max concurrent sessions per user (0 = unlimited).
    pub max_sessions: usize,
    /// Grace period (secs) for sliding refresh after access expires.
    pub sliding_window_secs: u64,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            access_lifetime_secs: 900,      // 15 min
            refresh_lifetime_secs: 604800,  // 7 days
            max_sessions: 5,
            sliding_window_secs: 60,
        }
    }
}

// ── Engine ─────────────────────────────────────────────────────

/// The token management engine.
#[derive(Debug, Clone)]
pub struct TokenEngine {
    config: TokenConfig,
    /// All tokens by ID.
    tokens: HashMap<String, Token>,
    /// Token families by family ID.
    families: HashMap<String, TokenFamily>,
    /// Blacklisted token IDs.
    blacklist: HashSet<String>,
    /// User ID -> set of active family IDs.
    user_sessions: HashMap<String, Vec<String>>,
    /// Counter for generating unique IDs.
    id_counter: u64,
}

impl TokenEngine {
    pub fn new(config: TokenConfig) -> Self {
        Self {
            config,
            tokens: HashMap::new(),
            families: HashMap::new(),
            blacklist: HashSet::new(),
            user_sessions: HashMap::new(),
            id_counter: 0,
        }
    }

    /// Generate a unique ID.
    fn next_id(&mut self, prefix: &str) -> String {
        self.id_counter += 1;
        format!("{prefix}_{}", self.id_counter)
    }

    /// Issue a new token pair, creating a new family.
    pub fn issue_tokens(
        &mut self,
        user_id: &str,
        now_secs: u64,
        device_id: Option<String>,
    ) -> Result<TokenPair, TokenError> {
        // Check session limit.
        if self.config.max_sessions > 0 {
            let sessions = self
                .user_sessions
                .entry(user_id.to_string())
                .or_default();

            // Clean up expired/compromised families.
            let active_families: Vec<String> = sessions
                .iter()
                .filter(|fid| {
                    self.families
                        .get(*fid)
                        .map(|f| !f.compromised)
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if active_families.len() >= self.config.max_sessions {
                return Err(TokenError::SessionLimitExceeded {
                    user_id: user_id.to_string(),
                    limit: self.config.max_sessions,
                });
            }
        }

        let family_id = self.next_id("fam");
        let family = TokenFamily {
            id: family_id.clone(),
            user_id: user_id.to_string(),
            created_at_secs: now_secs,
            current_generation: 0,
            compromised: false,
            device_id: device_id.clone(),
        };
        self.families.insert(family_id.clone(), family);

        self.user_sessions
            .entry(user_id.to_string())
            .or_default()
            .push(family_id.clone());

        let pair = self.create_pair(user_id, &family_id, 0, now_secs, device_id);
        Ok(pair)
    }

    /// Create an access+refresh pair.
    fn create_pair(
        &mut self,
        user_id: &str,
        family_id: &str,
        generation: u64,
        now_secs: u64,
        device_id: Option<String>,
    ) -> TokenPair {
        let access_id = self.next_id("at");
        let refresh_id = self.next_id("rt");

        let access = Token {
            id: access_id.clone(),
            user_id: user_id.to_string(),
            token_type: TokenType::Access,
            family_id: family_id.to_string(),
            issued_at_secs: now_secs,
            expires_at_secs: now_secs + self.config.access_lifetime_secs,
            generation,
            consumed: false,
            device_id: device_id.clone(),
        };

        let refresh = Token {
            id: refresh_id.clone(),
            user_id: user_id.to_string(),
            token_type: TokenType::Refresh,
            family_id: family_id.to_string(),
            issued_at_secs: now_secs,
            expires_at_secs: now_secs + self.config.refresh_lifetime_secs,
            generation,
            consumed: false,
            device_id,
        };

        self.tokens.insert(access_id, access.clone());
        self.tokens.insert(refresh_id, refresh.clone());

        TokenPair {
            access_token: access,
            refresh_token: refresh,
        }
    }

    /// Validate a token (check existence, blacklist, expiry).
    pub fn validate_token(
        &self,
        token_id: &str,
        now_secs: u64,
    ) -> Result<&Token, TokenError> {
        if self.blacklist.contains(token_id) {
            return Err(TokenError::TokenRevoked(token_id.to_string()));
        }

        let token = self
            .tokens
            .get(token_id)
            .ok_or_else(|| TokenError::TokenNotFound(token_id.to_string()))?;

        if token.is_expired(now_secs) {
            return Err(TokenError::TokenExpired(token_id.to_string()));
        }

        // Check if family is compromised.
        if let Some(family) = self.families.get(&token.family_id) {
            if family.compromised {
                return Err(TokenError::FamilyCompromised(token.family_id.clone()));
            }
        }

        Ok(token)
    }

    /// Rotate tokens using a refresh token. Issues a new pair and invalidates the old.
    /// Detects rotation reuse (token replay attack).
    pub fn rotate_tokens(
        &mut self,
        refresh_token_id: &str,
        now_secs: u64,
    ) -> Result<TokenPair, TokenError> {
        // Validate the refresh token first.
        if self.blacklist.contains(refresh_token_id) {
            return Err(TokenError::TokenRevoked(refresh_token_id.to_string()));
        }

        let token = self
            .tokens
            .get(refresh_token_id)
            .ok_or_else(|| TokenError::TokenNotFound(refresh_token_id.to_string()))?
            .clone();

        if token.token_type != TokenType::Refresh {
            return Err(TokenError::InvalidTokenType {
                expected: "refresh".to_string(),
                got: token.token_type.to_string(),
            });
        }

        if token.is_expired(now_secs) {
            return Err(TokenError::TokenExpired(refresh_token_id.to_string()));
        }

        // Check for rotation reuse: if this token was already consumed,
        // the family is compromised.
        if token.consumed {
            // Mark family as compromised and blacklist all its tokens.
            self.compromise_family(&token.family_id);
            return Err(TokenError::FamilyCompromised(token.family_id.clone()));
        }

        let family = self
            .families
            .get(&token.family_id)
            .ok_or_else(|| TokenError::TokenNotFound(token.family_id.clone()))?;

        if family.compromised {
            return Err(TokenError::FamilyCompromised(token.family_id.clone()));
        }

        let new_generation = token.generation + 1;
        let user_id = token.user_id.clone();
        let family_id = token.family_id.clone();
        let device_id = token.device_id.clone();

        // Mark old refresh token as consumed.
        self.tokens.get_mut(refresh_token_id).unwrap().consumed = true;

        // Update family generation.
        self.families.get_mut(&family_id).unwrap().current_generation = new_generation;

        let pair = self.create_pair(&user_id, &family_id, new_generation, now_secs, device_id);
        Ok(pair)
    }

    /// Mark a token family as compromised and blacklist all tokens in it.
    fn compromise_family(&mut self, family_id: &str) {
        if let Some(family) = self.families.get_mut(family_id) {
            family.compromised = true;
        }

        let to_blacklist: Vec<String> = self
            .tokens
            .values()
            .filter(|t| t.family_id == family_id)
            .map(|t| t.id.clone())
            .collect();

        for tid in to_blacklist {
            self.blacklist.insert(tid);
        }
    }

    /// Blacklist a specific token (revoke it).
    pub fn blacklist_token(&mut self, token_id: &str) -> Result<(), TokenError> {
        if !self.tokens.contains_key(token_id) {
            return Err(TokenError::TokenNotFound(token_id.to_string()));
        }
        self.blacklist.insert(token_id.to_string());
        Ok(())
    }

    /// Revoke all sessions for a user.
    pub fn revoke_all_sessions(&mut self, user_id: &str) {
        if let Some(family_ids) = self.user_sessions.remove(user_id) {
            for fid in family_ids {
                self.compromise_family(&fid);
            }
        }
    }

    /// Revoke a specific session (family).
    pub fn revoke_session(&mut self, family_id: &str) -> Result<(), TokenError> {
        if !self.families.contains_key(family_id) {
            return Err(TokenError::TokenNotFound(family_id.to_string()));
        }
        self.compromise_family(family_id);
        Ok(())
    }

    /// Check if a refresh can slide (within grace period after access expiry).
    pub fn can_slide_refresh(
        &self,
        refresh_token_id: &str,
        now_secs: u64,
    ) -> Result<bool, TokenError> {
        let token = self
            .tokens
            .get(refresh_token_id)
            .ok_or_else(|| TokenError::TokenNotFound(refresh_token_id.to_string()))?;

        if token.token_type != TokenType::Refresh {
            return Err(TokenError::InvalidTokenType {
                expected: "refresh".to_string(),
                got: token.token_type.to_string(),
            });
        }

        if token.is_expired(now_secs) {
            return Ok(false);
        }

        // Find the corresponding access token in the same family and generation.
        let access_expired = self
            .tokens
            .values()
            .any(|t| {
                t.family_id == token.family_id
                    && t.token_type == TokenType::Access
                    && t.generation == token.generation
                    && t.is_expired(now_secs)
                    && now_secs < t.expires_at_secs + self.config.sliding_window_secs
            });

        Ok(access_expired)
    }

    /// Count active sessions for a user.
    pub fn active_session_count(&self, user_id: &str) -> usize {
        self.user_sessions
            .get(user_id)
            .map(|fids| {
                fids.iter()
                    .filter(|fid| {
                        self.families
                            .get(*fid)
                            .map(|f| !f.compromised)
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    /// Total tokens in the store.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Total blacklisted tokens.
    pub fn blacklist_count(&self) -> usize {
        self.blacklist.len()
    }

    /// Get token by ID (without validation).
    pub fn get_token(&self, token_id: &str) -> Option<&Token> {
        self.tokens.get(token_id)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_engine() -> TokenEngine {
        TokenEngine::new(TokenConfig {
            access_lifetime_secs: 300,
            refresh_lifetime_secs: 3600,
            max_sessions: 3,
            sliding_window_secs: 60,
        })
    }

    #[test]
    fn test_issue_tokens() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        assert_eq!(pair.access_token.token_type, TokenType::Access);
        assert_eq!(pair.refresh_token.token_type, TokenType::Refresh);
        assert_eq!(pair.access_token.user_id, "user-1");
        assert_eq!(pair.access_token.family_id, pair.refresh_token.family_id);
        assert_eq!(pair.access_token.generation, 0);
    }

    #[test]
    fn test_validate_token() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        let token = engine
            .validate_token(&pair.access_token.id, 1100)
            .unwrap();
        assert_eq!(token.user_id, "user-1");
    }

    #[test]
    fn test_validate_expired_token() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        let err = engine
            .validate_token(&pair.access_token.id, 2000)
            .unwrap_err();
        assert!(matches!(err, TokenError::TokenExpired(_)));
    }

    #[test]
    fn test_validate_blacklisted_token() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        engine.blacklist_token(&pair.access_token.id).unwrap();
        let err = engine
            .validate_token(&pair.access_token.id, 1100)
            .unwrap_err();
        assert!(matches!(err, TokenError::TokenRevoked(_)));
    }

    #[test]
    fn test_rotate_tokens() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        let new_pair = engine
            .rotate_tokens(&pair.refresh_token.id, 1200)
            .unwrap();
        assert_eq!(new_pair.access_token.generation, 1);
        assert_eq!(new_pair.access_token.family_id, pair.access_token.family_id);
    }

    #[test]
    fn test_rotation_reuse_detection() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        // First rotation succeeds.
        let _new_pair = engine
            .rotate_tokens(&pair.refresh_token.id, 1200)
            .unwrap();

        // Reusing the old refresh token triggers compromise.
        let err = engine
            .rotate_tokens(&pair.refresh_token.id, 1400)
            .unwrap_err();
        assert!(matches!(err, TokenError::FamilyCompromised(_)));
    }

    #[test]
    fn test_rotate_access_token_rejected() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        let err = engine
            .rotate_tokens(&pair.access_token.id, 1200)
            .unwrap_err();
        assert!(matches!(err, TokenError::InvalidTokenType { .. }));
    }

    #[test]
    fn test_rotate_expired_refresh() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        let err = engine
            .rotate_tokens(&pair.refresh_token.id, 100000)
            .unwrap_err();
        assert!(matches!(err, TokenError::TokenExpired(_)));
    }

    #[test]
    fn test_session_limit() {
        let mut engine = default_engine();
        engine.issue_tokens("user-1", 1000, None).unwrap();
        engine.issue_tokens("user-1", 1001, None).unwrap();
        engine.issue_tokens("user-1", 1002, None).unwrap();

        let err = engine.issue_tokens("user-1", 1003, None).unwrap_err();
        assert!(matches!(err, TokenError::SessionLimitExceeded { .. }));
    }

    #[test]
    fn test_revoke_all_sessions() {
        let mut engine = default_engine();
        let pair1 = engine.issue_tokens("user-1", 1000, None).unwrap();
        let pair2 = engine.issue_tokens("user-1", 1001, None).unwrap();

        engine.revoke_all_sessions("user-1");

        assert!(engine
            .validate_token(&pair1.access_token.id, 1100)
            .is_err());
        assert!(engine
            .validate_token(&pair2.access_token.id, 1100)
            .is_err());
        assert_eq!(engine.active_session_count("user-1"), 0);
    }

    #[test]
    fn test_revoke_single_session() {
        let mut engine = default_engine();
        let pair1 = engine.issue_tokens("user-1", 1000, None).unwrap();
        let pair2 = engine.issue_tokens("user-1", 1001, None).unwrap();

        engine
            .revoke_session(&pair1.access_token.family_id)
            .unwrap();

        assert!(engine
            .validate_token(&pair1.access_token.id, 1100)
            .is_err());
        assert!(engine
            .validate_token(&pair2.access_token.id, 1100)
            .is_ok());
    }

    #[test]
    fn test_sliding_refresh() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        // Access expires at 1300 (1000 + 300). Check sliding at 1310 (within 60s window).
        let can_slide = engine
            .can_slide_refresh(&pair.refresh_token.id, 1310)
            .unwrap();
        assert!(can_slide);

        // Too late (past sliding window).
        let can_slide = engine
            .can_slide_refresh(&pair.refresh_token.id, 1400)
            .unwrap();
        assert!(!can_slide);
    }

    #[test]
    fn test_sliding_refresh_not_expired_yet() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();

        // Access not yet expired — sliding not needed.
        let can_slide = engine
            .can_slide_refresh(&pair.refresh_token.id, 1100)
            .unwrap();
        assert!(!can_slide);
    }

    #[test]
    fn test_token_remaining_secs() {
        let token = Token {
            id: "t1".to_string(),
            user_id: "u1".to_string(),
            token_type: TokenType::Access,
            family_id: "f1".to_string(),
            issued_at_secs: 1000,
            expires_at_secs: 1300,
            generation: 0,
            consumed: false,
            device_id: None,
        };

        assert_eq!(token.remaining_secs(1100), 200);
        assert_eq!(token.remaining_secs(1300), 0);
        assert_eq!(token.remaining_secs(1500), 0);
    }

    #[test]
    fn test_device_id_tracking() {
        let mut engine = default_engine();
        let pair = engine
            .issue_tokens("user-1", 1000, Some("iphone-12".to_string()))
            .unwrap();
        assert_eq!(
            pair.access_token.device_id.as_deref(),
            Some("iphone-12")
        );
        assert_eq!(
            pair.refresh_token.device_id.as_deref(),
            Some("iphone-12")
        );
    }

    #[test]
    fn test_active_session_count() {
        let mut engine = default_engine();
        engine.issue_tokens("user-1", 1000, None).unwrap();
        engine.issue_tokens("user-1", 1001, None).unwrap();
        assert_eq!(engine.active_session_count("user-1"), 2);
        assert_eq!(engine.active_session_count("user-2"), 0);
    }

    #[test]
    fn test_blacklist_nonexistent() {
        let mut engine = default_engine();
        let err = engine.blacklist_token("ghost").unwrap_err();
        assert!(matches!(err, TokenError::TokenNotFound(_)));
    }

    #[test]
    fn test_token_count() {
        let mut engine = default_engine();
        assert_eq!(engine.token_count(), 0);
        engine.issue_tokens("user-1", 1000, None).unwrap();
        assert_eq!(engine.token_count(), 2); // access + refresh
    }

    #[test]
    fn test_blacklist_count() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();
        assert_eq!(engine.blacklist_count(), 0);
        engine.blacklist_token(&pair.access_token.id).unwrap();
        assert_eq!(engine.blacklist_count(), 1);
    }

    #[test]
    fn test_get_token() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();
        assert!(engine.get_token(&pair.access_token.id).is_some());
        assert!(engine.get_token("ghost").is_none());
    }

    #[test]
    fn test_error_display() {
        let e = TokenError::TokenRevoked("abc".to_string());
        assert_eq!(e.to_string(), "token revoked: abc");
    }

    #[test]
    fn test_token_type_display() {
        assert_eq!(TokenType::Access.to_string(), "access");
        assert_eq!(TokenType::Refresh.to_string(), "refresh");
    }

    #[test]
    fn test_default_config() {
        let cfg = TokenConfig::default();
        assert_eq!(cfg.access_lifetime_secs, 900);
        assert_eq!(cfg.refresh_lifetime_secs, 604800);
        assert_eq!(cfg.max_sessions, 5);
    }

    #[test]
    fn test_multi_rotation_chain() {
        let mut engine = default_engine();
        let pair0 = engine.issue_tokens("user-1", 1000, None).unwrap();
        let pair1 = engine
            .rotate_tokens(&pair0.refresh_token.id, 1100)
            .unwrap();
        let pair2 = engine
            .rotate_tokens(&pair1.refresh_token.id, 1200)
            .unwrap();
        let pair3 = engine
            .rotate_tokens(&pair2.refresh_token.id, 1300)
            .unwrap();

        assert_eq!(pair3.access_token.generation, 3);
        assert!(engine
            .validate_token(&pair3.access_token.id, 1400)
            .is_ok());
    }

    #[test]
    fn test_revoke_session_not_found() {
        let mut engine = default_engine();
        let err = engine.revoke_session("ghost").unwrap_err();
        assert!(matches!(err, TokenError::TokenNotFound(_)));
    }

    #[test]
    fn test_compromised_family_blocks_validation() {
        let mut engine = default_engine();
        let pair = engine.issue_tokens("user-1", 1000, None).unwrap();
        let new_pair = engine
            .rotate_tokens(&pair.refresh_token.id, 1100)
            .unwrap();

        // Trigger compromise by reusing old refresh.
        let _ = engine.rotate_tokens(&pair.refresh_token.id, 1200);

        // Even the new pair's access token should be blocked.
        let err = engine
            .validate_token(&new_pair.access_token.id, 1150)
            .unwrap_err();
        assert!(matches!(err, TokenError::FamilyCompromised(_) | TokenError::TokenRevoked(_)));
    }
}
