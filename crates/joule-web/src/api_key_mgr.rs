//! API key management — key generation (random + prefix), key hashing for storage,
//! key validation, key scoping (permissions), key rotation, expiry, usage tracking,
//! and rate limits per key.
//!
//! Replaces `passport-headerapikey`, `express-api-key`, and custom key management
//! with a pure-Rust API key lifecycle manager.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// API key management errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyError {
    /// Key not found (by hash or prefix).
    KeyNotFound(String),
    /// Key has expired.
    KeyExpired { key_id: String, expired_at_ms: u64 },
    /// Key is revoked.
    KeyRevoked(String),
    /// Key lacks required permission.
    InsufficientScope { key_id: String, required: String },
    /// Rate limit exceeded for this key.
    RateLimited { key_id: String, limit: u64, window_ms: u64 },
    /// Duplicate key prefix.
    DuplicatePrefix(String),
    /// Invalid key format.
    InvalidFormat(String),
}

impl fmt::Display for ApiKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(id) => write!(f, "API key not found: {id}"),
            Self::KeyExpired { key_id, expired_at_ms } => {
                write!(f, "API key {key_id} expired at {expired_at_ms}")
            }
            Self::KeyRevoked(id) => write!(f, "API key revoked: {id}"),
            Self::InsufficientScope { key_id, required } => {
                write!(f, "key {key_id} lacks scope: {required}")
            }
            Self::RateLimited { key_id, limit, window_ms } => {
                write!(f, "key {key_id} rate limited: {limit} req/{window_ms}ms")
            }
            Self::DuplicatePrefix(p) => write!(f, "duplicate key prefix: {p}"),
            Self::InvalidFormat(msg) => write!(f, "invalid key format: {msg}"),
        }
    }
}

impl std::error::Error for ApiKeyError {}

// ── Types ──────────────────────────────────────────────────────

/// Status of an API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyStatus {
    Active,
    Revoked,
    Expired,
    Rotated,
}

impl KeyStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Revoked => "revoked",
            Self::Expired => "expired",
            Self::Rotated => "rotated",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

impl fmt::Display for KeyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A scope/permission attached to a key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyScope {
    /// Resource pattern (e.g., "documents:*", "users:read").
    pub resource: String,
    /// Actions allowed (e.g., "read", "write", "*").
    pub actions: Vec<String>,
}

impl KeyScope {
    pub fn new(resource: &str, actions: &[&str]) -> Self {
        Self {
            resource: resource.to_string(),
            actions: actions.iter().map(|a| a.to_string()).collect(),
        }
    }

    /// Check if this scope grants access to a resource/action pair.
    pub fn allows(&self, resource: &str, action: &str) -> bool {
        let resource_match = self.resource == "*"
            || self.resource == resource
            || (self.resource.ends_with(":*")
                && resource.starts_with(self.resource.trim_end_matches(":*")));

        let action_match = self.actions.iter().any(|a| a == "*" || a == action);

        resource_match && action_match
    }
}

/// Usage statistics for a key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyUsage {
    /// Total requests made with this key.
    pub total_requests: u64,
    /// Last used timestamp (epoch ms).
    pub last_used_ms: u64,
    /// First used timestamp (epoch ms).
    pub first_used_ms: u64,
    /// Request timestamps in the current rate window (for rate limiting).
    pub window_requests: Vec<u64>,
}

/// A stored API key record (the actual key is not stored, only its hash).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    /// Unique key ID (the prefix portion is visible).
    pub id: String,
    /// Hash of the full key (FNV-1a based).
    pub key_hash: String,
    /// Display prefix (e.g., "sk_live_abc").
    pub prefix: String,
    /// Owner/creator.
    pub owner: String,
    /// Human-readable name/label.
    pub name: String,
    /// Status.
    pub status: KeyStatus,
    /// Scopes/permissions.
    pub scopes: Vec<KeyScope>,
    /// Creation timestamp (epoch ms).
    pub created_at_ms: u64,
    /// Expiry timestamp (epoch ms). 0 = no expiry.
    pub expires_at_ms: u64,
    /// Revoked timestamp (epoch ms). 0 = not revoked.
    pub revoked_at_ms: u64,
    /// The ID of the key this was rotated from (if rotated).
    pub rotated_from: Option<String>,
    /// Rate limit: max requests per window. 0 = unlimited.
    pub rate_limit: u64,
    /// Rate limit window in milliseconds.
    pub rate_window_ms: u64,
    /// Usage statistics.
    pub usage: KeyUsage,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

/// The result of generating a new API key.
#[derive(Debug, Clone)]
pub struct GeneratedKey {
    /// The full key (only returned once — MUST be shown to the user).
    pub full_key: String,
    /// The key record (hash, not the key itself).
    pub record: ApiKeyRecord,
}

/// API Key Manager.
pub struct ApiKeyManager {
    /// Stored key records, keyed by key ID.
    keys: HashMap<String, ApiKeyRecord>,
    /// Index: key_hash -> key_id for fast lookup.
    hash_index: HashMap<String, String>,
    /// Counter for generating unique IDs.
    next_id: u64,
    /// Default rate limit for new keys (0 = unlimited).
    pub default_rate_limit: u64,
    /// Default rate window for new keys.
    pub default_rate_window_ms: u64,
}

impl ApiKeyManager {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            hash_index: HashMap::new(),
            next_id: 1,
            default_rate_limit: 0,
            default_rate_window_ms: 60_000,
        }
    }

    /// Generate a new API key with the given prefix and owner.
    pub fn generate_key(
        &mut self,
        prefix: &str,
        owner: &str,
        name: &str,
        seed: u64,
    ) -> GeneratedKey {
        let id = format!("key-{}", self.next_id);
        self.next_id += 1;

        // Generate pseudo-random key body from seed.
        let body = generate_key_body(seed, self.next_id);
        let full_key = format!("{prefix}_{body}");
        let key_hash = hash_key(&full_key);

        let record = ApiKeyRecord {
            id: id.clone(),
            key_hash: key_hash.clone(),
            prefix: prefix.to_string(),
            owner: owner.to_string(),
            name: name.to_string(),
            status: KeyStatus::Active,
            scopes: Vec::new(),
            created_at_ms: 0,
            expires_at_ms: 0,
            revoked_at_ms: 0,
            rotated_from: None,
            rate_limit: self.default_rate_limit,
            rate_window_ms: self.default_rate_window_ms,
            usage: KeyUsage::default(),
            metadata: HashMap::new(),
        };

        self.hash_index.insert(key_hash, id.clone());
        self.keys.insert(id, record.clone());

        GeneratedKey { full_key, record }
    }

    /// Set creation and expiry times on a key.
    pub fn set_key_times(
        &mut self,
        key_id: &str,
        created_at_ms: u64,
        expires_at_ms: u64,
    ) -> Result<(), ApiKeyError> {
        let rec = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?;
        rec.created_at_ms = created_at_ms;
        rec.expires_at_ms = expires_at_ms;
        Ok(())
    }

    /// Add a scope to a key.
    pub fn add_scope(&mut self, key_id: &str, scope: KeyScope) -> Result<(), ApiKeyError> {
        let rec = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?;
        rec.scopes.push(scope);
        Ok(())
    }

    /// Set rate limit for a key.
    pub fn set_rate_limit(
        &mut self,
        key_id: &str,
        limit: u64,
        window_ms: u64,
    ) -> Result<(), ApiKeyError> {
        let rec = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?;
        rec.rate_limit = limit;
        rec.rate_window_ms = window_ms;
        Ok(())
    }

    /// Validate a full API key: check hash, status, expiry, scopes, and rate limit.
    pub fn validate(
        &mut self,
        full_key: &str,
        resource: &str,
        action: &str,
        now_ms: u64,
    ) -> Result<&ApiKeyRecord, ApiKeyError> {
        let key_hash = hash_key(full_key);
        let key_id = self
            .hash_index
            .get(&key_hash)
            .ok_or_else(|| ApiKeyError::KeyNotFound("unknown".to_string()))?
            .clone();

        // Check status.
        let rec = self.keys.get(&key_id).unwrap();
        match rec.status {
            KeyStatus::Revoked => {
                return Err(ApiKeyError::KeyRevoked(key_id));
            }
            KeyStatus::Expired | KeyStatus::Rotated => {
                return Err(ApiKeyError::KeyExpired {
                    key_id,
                    expired_at_ms: rec.expires_at_ms,
                });
            }
            KeyStatus::Active => {}
        }

        // Check expiry.
        if rec.expires_at_ms > 0 && now_ms > rec.expires_at_ms {
            // Mark as expired.
            let rec_mut = self.keys.get_mut(&key_id).unwrap();
            rec_mut.status = KeyStatus::Expired;
            return Err(ApiKeyError::KeyExpired {
                key_id,
                expired_at_ms: rec_mut.expires_at_ms,
            });
        }

        // Check scopes.
        if !rec.scopes.is_empty() {
            let has_scope = rec.scopes.iter().any(|s| s.allows(resource, action));
            if !has_scope {
                return Err(ApiKeyError::InsufficientScope {
                    key_id,
                    required: format!("{resource}:{action}"),
                });
            }
        }

        // Check rate limit.
        let rate_limit = rec.rate_limit;
        let rate_window = rec.rate_window_ms;
        if rate_limit > 0 {
            let rec_mut = self.keys.get_mut(&key_id).unwrap();
            let cutoff = now_ms.saturating_sub(rate_window);
            rec_mut.usage.window_requests.retain(|ts| *ts > cutoff);
            if rec_mut.usage.window_requests.len() as u64 >= rate_limit {
                return Err(ApiKeyError::RateLimited {
                    key_id,
                    limit: rate_limit,
                    window_ms: rate_window,
                });
            }
            rec_mut.usage.window_requests.push(now_ms);
        }

        // Update usage.
        let rec_mut = self.keys.get_mut(&key_id).unwrap();
        rec_mut.usage.total_requests += 1;
        rec_mut.usage.last_used_ms = now_ms;
        if rec_mut.usage.first_used_ms == 0 {
            rec_mut.usage.first_used_ms = now_ms;
        }

        Ok(self.keys.get(&key_id).unwrap())
    }

    /// Revoke a key.
    pub fn revoke_key(&mut self, key_id: &str, now_ms: u64) -> Result<(), ApiKeyError> {
        let rec = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?;
        rec.status = KeyStatus::Revoked;
        rec.revoked_at_ms = now_ms;
        Ok(())
    }

    /// Rotate a key: revoke the old one and generate a new one with the same scopes.
    pub fn rotate_key(
        &mut self,
        key_id: &str,
        now_ms: u64,
        seed: u64,
    ) -> Result<GeneratedKey, ApiKeyError> {
        let old_rec = self
            .keys
            .get(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?
            .clone();

        // Revoke old key.
        let old_mut = self.keys.get_mut(key_id).unwrap();
        old_mut.status = KeyStatus::Rotated;
        old_mut.revoked_at_ms = now_ms;

        // Generate new key with same prefix, owner, scopes.
        let mut new_key = self.generate_key(&old_rec.prefix, &old_rec.owner, &old_rec.name, seed);
        new_key.record.scopes = old_rec.scopes;
        new_key.record.rotated_from = Some(key_id.to_string());
        new_key.record.created_at_ms = now_ms;
        new_key.record.rate_limit = old_rec.rate_limit;
        new_key.record.rate_window_ms = old_rec.rate_window_ms;

        // Update stored record.
        let new_id = new_key.record.id.clone();
        self.keys.insert(new_id, new_key.record.clone());

        Ok(new_key)
    }

    /// Get a key record by ID.
    pub fn get_key(&self, key_id: &str) -> Option<&ApiKeyRecord> {
        self.keys.get(key_id)
    }

    /// List all keys for an owner.
    pub fn keys_by_owner(&self, owner: &str) -> Vec<&ApiKeyRecord> {
        self.keys.values().filter(|k| k.owner == owner).collect()
    }

    /// Total number of keys.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// List active keys.
    pub fn active_keys(&self) -> Vec<&ApiKeyRecord> {
        self.keys.values().filter(|k| k.status.is_active()).collect()
    }

    /// Add metadata to a key.
    pub fn set_metadata(
        &mut self,
        key_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), ApiKeyError> {
        let rec = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::KeyNotFound(key_id.to_string()))?;
        rec.metadata.insert(key.to_string(), value.to_string());
        Ok(())
    }
}

impl Default for ApiKeyManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a pseudo-random key body (hex string).
fn generate_key_body(seed: u64, counter: u64) -> String {
    // FNV-1a based PRNG for deterministic test-friendly generation.
    let mut h = seed ^ 0xcbf29ce484222325;
    h = h.wrapping_mul(0x100000001b3);
    h ^= counter;
    h = h.wrapping_mul(0x100000001b3);
    let h2 = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
    format!("{h:016x}{h2:016x}")
}

/// Hash a key for storage (FNV-1a, hex-encoded).
fn hash_key(key: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_key() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk_live", "alice", "Production Key", 42);
        assert!(generated.full_key.starts_with("sk_live_"));
        assert_eq!(generated.record.owner, "alice");
        assert_eq!(generated.record.status, KeyStatus::Active);
    }

    #[test]
    fn test_validate_key() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let full_key = generated.full_key.clone();
        let result = mgr.validate(&full_key, "any", "read", 1000);
        assert!(result.is_ok());
        let rec = result.unwrap();
        assert_eq!(rec.usage.total_requests, 1);
    }

    #[test]
    fn test_validate_wrong_key() {
        let mut mgr = ApiKeyManager::new();
        mgr.generate_key("sk", "alice", "test", 42);
        let err = mgr.validate("sk_wrong_key_value", "any", "read", 1000).unwrap_err();
        match err {
            ApiKeyError::KeyNotFound(_) => {}
            other => panic!("expected KeyNotFound, got: {other}"),
        }
    }

    #[test]
    fn test_key_expiry() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.set_key_times(&key_id, 1000, 5000).unwrap();
        // Before expiry
        assert!(mgr.validate(&full_key, "any", "read", 3000).is_ok());
        // After expiry
        let err = mgr.validate(&full_key, "any", "read", 6000).unwrap_err();
        match err {
            ApiKeyError::KeyExpired { .. } => {}
            other => panic!("expected KeyExpired, got: {other}"),
        }
    }

    #[test]
    fn test_key_revocation() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.revoke_key(&key_id, 5000).unwrap();
        let err = mgr.validate(&full_key, "any", "read", 6000).unwrap_err();
        match err {
            ApiKeyError::KeyRevoked(_) => {}
            other => panic!("expected KeyRevoked, got: {other}"),
        }
    }

    #[test]
    fn test_key_scopes() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.add_scope(&key_id, KeyScope::new("documents", &["read"])).unwrap();

        // Allowed
        assert!(mgr.validate(&full_key, "documents", "read", 1000).is_ok());
        // Denied (wrong action)
        let err = mgr.validate(&full_key, "documents", "delete", 2000).unwrap_err();
        match err {
            ApiKeyError::InsufficientScope { .. } => {}
            other => panic!("expected InsufficientScope, got: {other}"),
        }
    }

    #[test]
    fn test_wildcard_scope() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.add_scope(&key_id, KeyScope::new("*", &["*"])).unwrap();
        assert!(mgr.validate(&full_key, "anything", "any_action", 1000).is_ok());
    }

    #[test]
    fn test_prefix_wildcard_scope() {
        let scope = KeyScope::new("documents:*", &["read", "write"]);
        assert!(scope.allows("documents:123", "read"));
        assert!(scope.allows("documents:abc", "write"));
        assert!(!scope.allows("users:1", "read"));
        assert!(!scope.allows("documents:123", "delete"));
    }

    #[test]
    fn test_rate_limiting() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.set_rate_limit(&key_id, 3, 10_000).unwrap();

        assert!(mgr.validate(&full_key, "any", "read", 1000).is_ok());
        assert!(mgr.validate(&full_key, "any", "read", 2000).is_ok());
        assert!(mgr.validate(&full_key, "any", "read", 3000).is_ok());
        let err = mgr.validate(&full_key, "any", "read", 4000).unwrap_err();
        match err {
            ApiKeyError::RateLimited { .. } => {}
            other => panic!("expected RateLimited, got: {other}"),
        }
    }

    #[test]
    fn test_rate_limit_window_reset() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        let full_key = generated.full_key.clone();
        mgr.set_rate_limit(&key_id, 2, 5000).unwrap();

        assert!(mgr.validate(&full_key, "any", "read", 1000).is_ok());
        assert!(mgr.validate(&full_key, "any", "read", 2000).is_ok());
        assert!(mgr.validate(&full_key, "any", "read", 3000).is_err());
        // After window resets
        assert!(mgr.validate(&full_key, "any", "read", 10_000).is_ok());
    }

    #[test]
    fn test_key_rotation() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let old_id = generated.record.id.clone();
        let old_key = generated.full_key.clone();
        mgr.add_scope(&old_id, KeyScope::new("docs", &["read"])).unwrap();

        let new_gen = mgr.rotate_key(&old_id, 5000, 99).unwrap();
        let new_key = new_gen.full_key;

        // Old key should no longer work
        assert!(mgr.validate(&old_key, "docs", "read", 6000).is_err());
        // New key should work with inherited scopes
        assert!(mgr.validate(&new_key, "docs", "read", 6000).is_ok());
        // New key should reference old key
        assert_eq!(
            mgr.get_key(&new_gen.record.id).unwrap().rotated_from,
            Some(old_id)
        );
    }

    #[test]
    fn test_keys_by_owner() {
        let mut mgr = ApiKeyManager::new();
        mgr.generate_key("sk", "alice", "key1", 1);
        mgr.generate_key("sk", "bob", "key2", 2);
        mgr.generate_key("sk", "alice", "key3", 3);

        let alice_keys = mgr.keys_by_owner("alice");
        assert_eq!(alice_keys.len(), 2);
    }

    #[test]
    fn test_active_keys() {
        let mut mgr = ApiKeyManager::new();
        let g1 = mgr.generate_key("sk", "alice", "k1", 1);
        mgr.generate_key("sk", "alice", "k2", 2);
        mgr.revoke_key(&g1.record.id, 1000).unwrap();

        let active = mgr.active_keys();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_usage_tracking() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let full_key = generated.full_key.clone();
        let key_id = generated.record.id.clone();

        mgr.validate(&full_key, "any", "read", 1000).unwrap();
        mgr.validate(&full_key, "any", "read", 2000).unwrap();
        mgr.validate(&full_key, "any", "read", 3000).unwrap();

        let rec = mgr.get_key(&key_id).unwrap();
        assert_eq!(rec.usage.total_requests, 3);
        assert_eq!(rec.usage.first_used_ms, 1000);
        assert_eq!(rec.usage.last_used_ms, 3000);
    }

    #[test]
    fn test_key_metadata() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let key_id = generated.record.id.clone();
        mgr.set_metadata(&key_id, "environment", "production").unwrap();
        let rec = mgr.get_key(&key_id).unwrap();
        assert_eq!(rec.metadata.get("environment"), Some(&"production".to_string()));
    }

    #[test]
    fn test_hash_deterministic() {
        assert_eq!(hash_key("test-key"), hash_key("test-key"));
        assert_ne!(hash_key("test-key-1"), hash_key("test-key-2"));
    }

    #[test]
    fn test_key_count() {
        let mut mgr = ApiKeyManager::new();
        assert_eq!(mgr.key_count(), 0);
        mgr.generate_key("sk", "alice", "k1", 1);
        mgr.generate_key("sk", "alice", "k2", 2);
        assert_eq!(mgr.key_count(), 2);
    }

    #[test]
    fn test_key_not_found_on_revoke() {
        let mut mgr = ApiKeyManager::new();
        let err = mgr.revoke_key("nonexistent", 1000).unwrap_err();
        assert_eq!(err, ApiKeyError::KeyNotFound("nonexistent".into()));
    }

    #[test]
    fn test_key_status_display() {
        assert_eq!(KeyStatus::Active.to_string(), "active");
        assert_eq!(KeyStatus::Revoked.to_string(), "revoked");
        assert!(KeyStatus::Active.is_active());
        assert!(!KeyStatus::Revoked.is_active());
    }

    #[test]
    fn test_no_scope_allows_everything() {
        let mut mgr = ApiKeyManager::new();
        let generated = mgr.generate_key("sk", "alice", "test", 42);
        let full_key = generated.full_key.clone();
        // Key with no scopes should allow any resource/action
        assert!(mgr.validate(&full_key, "anything", "any", 1000).is_ok());
    }
}
