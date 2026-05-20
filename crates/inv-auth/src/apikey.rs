use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;
use uuid::Uuid;

use crate::AuthError;
use crate::rbac::{Permission, Role};

/// An API key for programmatic access to Invisible Infrastructure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique identifier for this API key.
    pub id: Uuid,
    /// Organization this key belongs to.
    pub org: String,
    /// Human-readable name for the key (e.g., "ci-pipeline-key").
    pub name: String,
    /// SHA-256 hash of the plaintext key (the plaintext is never stored).
    pub key_hash: String,
    /// Role assigned to this key.
    pub role: Role,
    /// Specific permissions granted (if empty, uses role defaults).
    pub permissions: Vec<Permission>,
    /// When the key was created.
    pub created_at: DateTime<Utc>,
    /// When the key expires (None = no expiry).
    pub expires_at: Option<DateTime<Utc>>,
    /// When the key was last used for authentication.
    pub last_used: Option<DateTime<Utc>>,
    /// Whether the key is currently active.
    pub active: bool,
}

impl ApiKey {
    /// Check whether this key grants a specific permission.
    ///
    /// If the key has explicit permissions, those are checked.
    /// Otherwise, the role's default permissions are used.
    pub fn has_permission(&self, perm: Permission) -> bool {
        if self.permissions.is_empty() {
            self.role.has_permission(perm)
        } else {
            self.permissions.contains(&perm)
        }
    }

    /// Check whether this key has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() >= expires_at
        } else {
            false
        }
    }
}

/// Result of creating a new API key.
///
/// The plaintext key is returned exactly once at creation time and is never stored.
#[derive(Debug, Clone)]
pub struct ApiKeyCreateResult {
    /// The API key metadata (with hash, not plaintext).
    pub api_key: ApiKey,
    /// The plaintext key to give to the user. Shown only once.
    pub plaintext_key: String,
}

/// Service for managing API keys.
#[derive(Clone)]
pub struct ApiKeyService {
    /// Keys indexed by their ID.
    keys_by_id: Arc<RwLock<HashMap<Uuid, ApiKey>>>,
    /// Lookup from key hash to key ID for fast validation.
    hash_to_id: Arc<RwLock<HashMap<String, Uuid>>>,
}

impl ApiKeyService {
    /// Create a new API key service.
    pub fn new() -> Self {
        Self {
            keys_by_id: Arc::new(RwLock::new(HashMap::new())),
            hash_to_id: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new API key. Returns the key metadata and the plaintext key (shown once).
    pub fn create(
        &self,
        org: &str,
        name: &str,
        role: Role,
        permissions: Vec<Permission>,
        expires_at: Option<DateTime<Utc>>,
    ) -> ApiKeyCreateResult {
        let plaintext_key = generate_api_key();
        let key_hash = hash_key(&plaintext_key);
        let id = Uuid::new_v4();

        let api_key = ApiKey {
            id,
            org: org.to_string(),
            name: name.to_string(),
            key_hash: key_hash.clone(),
            role,
            permissions,
            created_at: Utc::now(),
            expires_at,
            last_used: None,
            active: true,
        };

        {
            let mut keys = self.keys_by_id.write().unwrap();
            keys.insert(id, api_key.clone());
        }
        {
            let mut hashes = self.hash_to_id.write().unwrap();
            hashes.insert(key_hash, id);
        }

        info!(key_id = %id, org = org, name = name, "API key created");

        ApiKeyCreateResult {
            api_key,
            plaintext_key,
        }
    }

    /// Validate a plaintext API key and return the associated key metadata.
    ///
    /// Also updates the `last_used` timestamp on success.
    pub fn validate(&self, plaintext_key: &str) -> Result<ApiKey, AuthError> {
        let key_hash = hash_key(plaintext_key);

        let id = {
            let hashes = self.hash_to_id.read().unwrap();
            hashes.get(&key_hash).copied()
        };

        let Some(id) = id else {
            return Err(AuthError::ApiKeyNotFound);
        };

        let mut keys = self.keys_by_id.write().unwrap();
        let Some(key) = keys.get_mut(&id) else {
            return Err(AuthError::ApiKeyNotFound);
        };

        if !key.active {
            return Err(AuthError::ApiKeyRevoked);
        }

        if key.is_expired() {
            return Err(AuthError::ApiKeyExpired);
        }

        key.last_used = Some(Utc::now());
        Ok(key.clone())
    }

    /// Revoke an API key by its ID.
    pub fn revoke(&self, key_id: Uuid) -> Result<(), AuthError> {
        let mut keys = self.keys_by_id.write().unwrap();
        let Some(key) = keys.get_mut(&key_id) else {
            return Err(AuthError::ApiKeyNotFound);
        };
        key.active = false;
        info!(key_id = %key_id, "API key revoked");
        Ok(())
    }

    /// List all API keys for an organization.
    pub fn list(&self, org: &str) -> Vec<ApiKey> {
        let keys = self.keys_by_id.read().unwrap();
        keys.values().filter(|k| k.org == org).cloned().collect()
    }

    /// Update the last-used timestamp for a key.
    pub fn update_last_used(&self, key_id: Uuid) -> Result<(), AuthError> {
        let mut keys = self.keys_by_id.write().unwrap();
        let Some(key) = keys.get_mut(&key_id) else {
            return Err(AuthError::ApiKeyNotFound);
        };
        key.last_used = Some(Utc::now());
        Ok(())
    }

    /// Get a key by its ID.
    pub fn get(&self, key_id: Uuid) -> Result<ApiKey, AuthError> {
        let keys = self.keys_by_id.read().unwrap();
        keys.get(&key_id).cloned().ok_or(AuthError::ApiKeyNotFound)
    }
}

impl Default for ApiKeyService {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a new API key with the `inv_key_` prefix.
fn generate_api_key() -> String {
    let mut rng = rand::rng();
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    format!("inv_key_{}", hex::encode(bytes))
}

/// Compute the SHA-256 hash of a plaintext API key, returned as a hex string.
fn hash_key(plaintext: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(plaintext.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_returns_plaintext_and_metadata() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Operator, vec![], None);

        assert!(result.plaintext_key.starts_with("inv_key_"));
        assert_eq!(result.api_key.org, "acme");
        assert_eq!(result.api_key.name, "test-key");
        assert_eq!(result.api_key.role, Role::Operator);
        assert!(result.api_key.active);
        assert!(result.api_key.last_used.is_none());
    }

    #[test]
    fn validate_succeeds_with_correct_key() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Operator, vec![], None);

        let key = svc.validate(&result.plaintext_key).unwrap();
        assert_eq!(key.id, result.api_key.id);
        assert_eq!(key.org, "acme");
        assert!(key.last_used.is_some());
    }

    #[test]
    fn validate_fails_with_wrong_key() {
        let svc = ApiKeyService::new();
        let _result = svc.create("acme", "test-key", Role::Operator, vec![], None);

        let err = svc.validate("inv_key_wrong").unwrap_err();
        assert!(matches!(err, AuthError::ApiKeyNotFound));
    }

    #[test]
    fn validate_fails_after_revocation() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Operator, vec![], None);

        svc.revoke(result.api_key.id).unwrap();

        let err = svc.validate(&result.plaintext_key).unwrap_err();
        assert!(matches!(err, AuthError::ApiKeyRevoked));
    }

    #[test]
    fn validate_fails_when_expired() {
        let svc = ApiKeyService::new();
        // Create a key that already expired
        let expired = Utc::now() - chrono::TimeDelta::hours(1);
        let result = svc.create("acme", "expired-key", Role::Operator, vec![], Some(expired));

        let err = svc.validate(&result.plaintext_key).unwrap_err();
        assert!(matches!(err, AuthError::ApiKeyExpired));
    }

    #[test]
    fn revoke_nonexistent_key_fails() {
        let svc = ApiKeyService::new();
        let err = svc.revoke(Uuid::new_v4()).unwrap_err();
        assert!(matches!(err, AuthError::ApiKeyNotFound));
    }

    #[test]
    fn list_filters_by_org() {
        let svc = ApiKeyService::new();
        svc.create("acme", "key-1", Role::Operator, vec![], None);
        svc.create("acme", "key-2", Role::Viewer, vec![], None);
        svc.create("other-org", "key-3", Role::Admin, vec![], None);

        let acme_keys = svc.list("acme");
        assert_eq!(acme_keys.len(), 2);
        assert!(acme_keys.iter().all(|k| k.org == "acme"));

        let other_keys = svc.list("other-org");
        assert_eq!(other_keys.len(), 1);
        assert_eq!(other_keys[0].name, "key-3");
    }

    #[test]
    fn update_last_used_sets_timestamp() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Operator, vec![], None);

        assert!(result.api_key.last_used.is_none());

        svc.update_last_used(result.api_key.id).unwrap();

        let key = svc.get(result.api_key.id).unwrap();
        assert!(key.last_used.is_some());
    }

    #[test]
    fn custom_permissions_override_role() {
        let svc = ApiKeyService::new();
        // Create an Operator key but only grant NodeRead
        let result = svc.create(
            "acme",
            "limited-key",
            Role::Operator,
            vec![Permission::NodeRead],
            None,
        );

        let key = svc.validate(&result.plaintext_key).unwrap();
        assert!(key.has_permission(Permission::NodeRead));
        // Operator normally has WorkloadDeploy, but custom permissions restrict it
        assert!(!key.has_permission(Permission::WorkloadDeploy));
    }

    #[test]
    fn api_key_format_is_correct() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Viewer, vec![], None);

        assert!(result.plaintext_key.starts_with("inv_key_"));
        // inv_key_ (8 chars) + 64 hex chars (32 bytes)
        assert_eq!(result.plaintext_key.len(), 8 + 64);
    }

    #[test]
    fn hash_is_deterministic() {
        let key = "inv_key_abc123";
        let hash1 = hash_key(key);
        let hash2 = hash_key(key);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn get_returns_key_by_id() {
        let svc = ApiKeyService::new();
        let result = svc.create("acme", "test-key", Role::Admin, vec![], None);

        let key = svc.get(result.api_key.id).unwrap();
        assert_eq!(key.name, "test-key");
        assert_eq!(key.org, "acme");
    }

    #[test]
    fn get_nonexistent_key_fails() {
        let svc = ApiKeyService::new();
        let err = svc.get(Uuid::new_v4()).unwrap_err();
        assert!(matches!(err, AuthError::ApiKeyNotFound));
    }
}
