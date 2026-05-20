//! Authentication & Authorization — token-based auth with row-level security.
//!
//! Every content provider needs:
//! - Token validation (API keys, JWTs)
//! - Role-based access control (admin, reader, writer)
//! - Row-level security (user can only see their own data)
//! - Audit trail of all authenticated operations

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Authentication token (opaque string, validated by the auth layer).
pub type Token = String;

/// User identity after authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub user_id: String,
    pub tenant_id: String,
    pub roles: Vec<Role>,
    /// Row-level filter: if set, queries are automatically filtered to only
    /// return records where this field matches this value.
    pub row_filter: Option<(String, String)>,
}

/// Access roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Full access — CRUD + admin operations
    Admin,
    /// Read + write data
    Writer,
    /// Read-only access
    Reader,
    /// Can only run moderation operations
    Moderator,
    /// Analytics queries only (aggregates, no raw records)
    Analyst,
}

/// Permission check result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Allowed,
    Denied,
    RequiresRowFilter,
}

/// Operation types for permission checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Read,
    Write,
    Delete,
    Query,
    Admin,
    Moderate,
}

/// The auth store: manages tokens, identities, and permissions.
pub struct AuthStore {
    /// Token → Identity mapping
    tokens: HashMap<Token, Identity>,
    /// Role → allowed operations
    role_permissions: HashMap<Role, HashSet<Operation>>,
}

impl AuthStore {
    pub fn new() -> Self {
        let mut role_permissions: HashMap<Role, HashSet<Operation>> = HashMap::new();

        role_permissions.insert(
            Role::Admin,
            [
                Operation::Read,
                Operation::Write,
                Operation::Delete,
                Operation::Query,
                Operation::Admin,
                Operation::Moderate,
            ]
            .into(),
        );
        role_permissions.insert(
            Role::Writer,
            [Operation::Read, Operation::Write, Operation::Query].into(),
        );
        role_permissions.insert(
            Role::Reader,
            [Operation::Read, Operation::Query].into(),
        );
        role_permissions.insert(
            Role::Moderator,
            [Operation::Read, Operation::Moderate].into(),
        );
        role_permissions.insert(
            Role::Analyst,
            [Operation::Query].into(),
        );

        Self {
            tokens: HashMap::new(),
            role_permissions,
        }
    }

    /// Register a token for a user identity.
    pub fn register_token(&mut self, token: Token, identity: Identity) {
        self.tokens.insert(token, identity);
    }

    /// Generate a simple API key (not cryptographically secure — use a proper
    /// token service in production).
    pub fn generate_api_key(user_id: &str, tenant_id: &str, roles: Vec<Role>) -> (Token, Identity) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        tenant_id.hash(&mut hasher);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        ts.hash(&mut hasher);
        let token = format!("jdb_{:016x}", hasher.finish());

        let identity = Identity {
            user_id: user_id.to_string(),
            tenant_id: tenant_id.to_string(),
            roles,
            row_filter: None,
        };

        (token, identity)
    }

    /// Authenticate a token. Returns the identity if valid.
    pub fn authenticate(&self, token: &str) -> Option<&Identity> {
        self.tokens.get(token)
    }

    /// Check if an identity has permission for an operation.
    pub fn check_permission(&self, identity: &Identity, operation: Operation) -> Permission {
        for role in &identity.roles {
            if let Some(perms) = self.role_permissions.get(role) {
                if perms.contains(&operation) {
                    return Permission::Allowed;
                }
            }
        }
        Permission::Denied
    }

    /// Revoke a token.
    pub fn revoke_token(&mut self, token: &str) -> bool {
        self.tokens.remove(token).is_some()
    }

    /// Number of active tokens.
    pub fn active_tokens(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for AuthStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_flow() {
        let mut auth = AuthStore::new();

        let (token, identity) = AuthStore::generate_api_key("user1", "netflix", vec![Role::Writer]);
        auth.register_token(token.clone(), identity);

        // Authenticate
        let id = auth.authenticate(&token).unwrap();
        assert_eq!(id.user_id, "user1");
        assert_eq!(id.tenant_id, "netflix");

        // Check permissions
        assert_eq!(
            auth.check_permission(id, Operation::Read),
            Permission::Allowed
        );
        assert_eq!(
            auth.check_permission(id, Operation::Write),
            Permission::Allowed
        );
        assert_eq!(
            auth.check_permission(id, Operation::Admin),
            Permission::Denied
        );
    }

    #[test]
    fn test_role_hierarchy() {
        let auth = AuthStore::new();

        let admin = Identity {
            user_id: "admin".into(),
            tenant_id: "t".into(),
            roles: vec![Role::Admin],
            row_filter: None,
        };
        let reader = Identity {
            user_id: "reader".into(),
            tenant_id: "t".into(),
            roles: vec![Role::Reader],
            row_filter: None,
        };

        // Admin can do everything
        assert_eq!(auth.check_permission(&admin, Operation::Delete), Permission::Allowed);
        assert_eq!(auth.check_permission(&admin, Operation::Admin), Permission::Allowed);

        // Reader can only read/query
        assert_eq!(auth.check_permission(&reader, Operation::Read), Permission::Allowed);
        assert_eq!(auth.check_permission(&reader, Operation::Write), Permission::Denied);
        assert_eq!(auth.check_permission(&reader, Operation::Delete), Permission::Denied);
    }

    #[test]
    fn test_token_revocation() {
        let mut auth = AuthStore::new();
        let (token, identity) = AuthStore::generate_api_key("u", "t", vec![Role::Reader]);
        auth.register_token(token.clone(), identity);

        assert!(auth.authenticate(&token).is_some());
        auth.revoke_token(&token);
        assert!(auth.authenticate(&token).is_none());
    }

    #[test]
    fn test_invalid_token() {
        let auth = AuthStore::new();
        assert!(auth.authenticate("invalid_token").is_none());
    }
}
