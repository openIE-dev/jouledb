//! Permission/authorization engine — role-based access control (RBAC),
//! permission inheritance, resource-level permissions, wildcard permissions,
//! policy evaluation, deny-overrides-allow.
//!
//! Replaces casl, casbin, and accesscontrol with a pure-Rust authorization
//! engine supporting hierarchical roles and fine-grained permissions.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Errors ─────────────────────────────────────────────────────

/// Permission engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionError {
    /// Access denied.
    AccessDenied {
        actor: String,
        action: String,
        resource: String,
    },
    /// Role not found.
    RoleNotFound(String),
    /// Circular role inheritance detected.
    CircularInheritance(String),
    /// Invalid permission format.
    InvalidFormat(String),
}

impl std::fmt::Display for PermissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccessDenied {
                actor,
                action,
                resource,
            } => write!(f, "access denied: {actor} cannot {action} on {resource}"),
            Self::RoleNotFound(r) => write!(f, "role not found: {r}"),
            Self::CircularInheritance(r) => write!(f, "circular inheritance detected: {r}"),
            Self::InvalidFormat(s) => write!(f, "invalid permission format: {s}"),
        }
    }
}

impl std::error::Error for PermissionError {}

// ── Permission ─────────────────────────────────────────────────

/// Effect of a permission rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    /// Explicitly allow.
    Allow,
    /// Explicitly deny (overrides Allow).
    Deny,
}

/// A single permission rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permission {
    /// The action (e.g., "read", "write", "delete", or "*" for any).
    pub action: String,
    /// The resource pattern (e.g., "document:*", "user:123", or "*").
    pub resource: String,
    /// Allow or deny.
    pub effect: Effect,
    /// Optional conditions (key-value constraints).
    pub conditions: HashMap<String, String>,
}

impl Permission {
    /// Create an allow permission.
    pub fn allow(action: &str, resource: &str) -> Self {
        Self {
            action: action.to_string(),
            resource: resource.to_string(),
            effect: Effect::Allow,
            conditions: HashMap::new(),
        }
    }

    /// Create a deny permission.
    pub fn deny(action: &str, resource: &str) -> Self {
        Self {
            action: action.to_string(),
            resource: resource.to_string(),
            effect: Effect::Deny,
            conditions: HashMap::new(),
        }
    }

    /// Add a condition.
    pub fn with_condition(mut self, key: &str, value: &str) -> Self {
        self.conditions.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if this permission matches the given action and resource.
    pub fn matches(&self, action: &str, resource: &str) -> bool {
        pattern_matches(&self.action, action) && pattern_matches(&self.resource, resource)
    }

    /// Check conditions against a context.
    pub fn conditions_met(&self, context: &HashMap<String, String>) -> bool {
        self.conditions
            .iter()
            .all(|(k, v)| context.get(k).map_or(false, |cv| cv == v))
    }
}

/// Match a pattern against a value.
///
/// Supports:
/// - `*` matches everything
/// - `prefix:*` matches anything starting with `prefix:`
/// - Exact match otherwise
fn pattern_matches(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix(":*") {
        return value.starts_with(prefix) && value.len() > prefix.len() && value.as_bytes()[prefix.len()] == b':';
    }
    if pattern.contains('*') {
        // General glob: split on '*' and check prefix/suffix.
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return value.starts_with(parts[0]) && value.ends_with(parts[1]);
        }
    }
    pattern == value
}

// ── Role ───────────────────────────────────────────────────────

/// A role with permissions and optional parent roles for inheritance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name.
    pub name: String,
    /// Direct permissions granted to this role.
    pub permissions: Vec<Permission>,
    /// Parent role names (inherit their permissions).
    pub parents: Vec<String>,
    /// Description.
    pub description: String,
}

impl Role {
    /// Create a new role.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            permissions: Vec::new(),
            parents: Vec::new(),
            description: String::new(),
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a permission.
    pub fn add_permission(&mut self, perm: Permission) {
        self.permissions.push(perm);
    }

    /// Add a parent role for inheritance.
    pub fn add_parent(&mut self, parent: &str) {
        if !self.parents.contains(&parent.to_string()) {
            self.parents.push(parent.to_string());
        }
    }
}

// ── Policy Decision ────────────────────────────────────────────

/// Result of a policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Access allowed.
    Allow,
    /// Access denied.
    Deny,
    /// No matching rule found (default deny).
    NoMatch,
}

impl Decision {
    /// Whether access is granted.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }
}

// ── Permission Engine ──────────────────────────────────────────

/// The authorization engine supporting RBAC with inheritance.
pub struct PermissionEngine {
    /// Defined roles.
    roles: HashMap<String, Role>,
    /// Actor-to-roles mapping.
    actor_roles: HashMap<String, Vec<String>>,
    /// Actor-level permissions (in addition to role permissions).
    actor_permissions: HashMap<String, Vec<Permission>>,
    /// Default decision when no rules match.
    pub default_decision: Decision,
}

impl PermissionEngine {
    /// Create a new engine with default-deny policy.
    pub fn new() -> Self {
        Self {
            roles: HashMap::new(),
            actor_roles: HashMap::new(),
            actor_permissions: HashMap::new(),
            default_decision: Decision::Deny,
        }
    }

    /// Add a role definition.
    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Assign a role to an actor.
    pub fn assign_role(&mut self, actor: &str, role: &str) {
        let roles = self.actor_roles.entry(actor.to_string()).or_default();
        if !roles.contains(&role.to_string()) {
            roles.push(role.to_string());
        }
    }

    /// Remove a role from an actor.
    pub fn revoke_role(&mut self, actor: &str, role: &str) {
        if let Some(roles) = self.actor_roles.get_mut(actor) {
            roles.retain(|r| r != role);
        }
    }

    /// Add a direct permission to an actor (bypassing roles).
    pub fn add_actor_permission(&mut self, actor: &str, perm: Permission) {
        self.actor_permissions
            .entry(actor.to_string())
            .or_default()
            .push(perm);
    }

    /// Get all roles assigned to an actor.
    pub fn actor_roles(&self, actor: &str) -> Vec<&str> {
        self.actor_roles
            .get(actor)
            .map(|rs| rs.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Collect all permissions for an actor (from roles + direct).
    fn collect_permissions(&self, actor: &str) -> Vec<&Permission> {
        let mut perms = Vec::new();
        let mut visited = HashSet::new();

        // Collect from assigned roles (with inheritance).
        if let Some(role_names) = self.actor_roles.get(actor) {
            for role_name in role_names {
                self.collect_role_permissions(role_name, &mut perms, &mut visited);
            }
        }

        // Collect direct actor permissions.
        if let Some(actor_perms) = self.actor_permissions.get(actor) {
            for perm in actor_perms {
                perms.push(perm);
            }
        }

        perms
    }

    /// Recursively collect permissions from a role and its parents.
    fn collect_role_permissions<'a>(
        &'a self,
        role_name: &str,
        perms: &mut Vec<&'a Permission>,
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(role_name.to_string()) {
            return; // Avoid circular inheritance.
        }
        if let Some(role) = self.roles.get(role_name) {
            for perm in &role.permissions {
                perms.push(perm);
            }
            for parent in &role.parents {
                self.collect_role_permissions(parent, perms, visited);
            }
        }
    }

    /// Evaluate whether an actor can perform an action on a resource.
    ///
    /// Uses deny-overrides-allow: if any deny matches, the result is Deny,
    /// regardless of allow rules.
    pub fn evaluate(
        &self,
        actor: &str,
        action: &str,
        resource: &str,
    ) -> Decision {
        self.evaluate_with_context(actor, action, resource, &HashMap::new())
    }

    /// Evaluate with additional context for conditional permissions.
    pub fn evaluate_with_context(
        &self,
        actor: &str,
        action: &str,
        resource: &str,
        context: &HashMap<String, String>,
    ) -> Decision {
        let perms = self.collect_permissions(actor);

        let mut has_allow = false;

        for perm in &perms {
            if !perm.matches(action, resource) {
                continue;
            }
            if !perm.conditions_met(context) {
                continue;
            }
            match perm.effect {
                Effect::Deny => return Decision::Deny, // Deny overrides.
                Effect::Allow => has_allow = true,
            }
        }

        if has_allow {
            Decision::Allow
        } else {
            self.default_decision.clone()
        }
    }

    /// Check access, returning an error if denied.
    pub fn check(
        &self,
        actor: &str,
        action: &str,
        resource: &str,
    ) -> Result<(), PermissionError> {
        if self.evaluate(actor, action, resource).is_allowed() {
            Ok(())
        } else {
            Err(PermissionError::AccessDenied {
                actor: actor.to_string(),
                action: action.to_string(),
                resource: resource.to_string(),
            })
        }
    }

    /// Get all defined role names.
    pub fn role_names(&self) -> Vec<&str> {
        self.roles.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a role exists.
    pub fn has_role(&self, name: &str) -> bool {
        self.roles.contains_key(name)
    }

    /// Detect circular inheritance for a role.
    pub fn has_circular_inheritance(&self, role_name: &str) -> bool {
        let mut visited = HashSet::new();
        self.detect_cycle(role_name, &mut visited)
    }

    fn detect_cycle(&self, role_name: &str, visited: &mut HashSet<String>) -> bool {
        if !visited.insert(role_name.to_string()) {
            return true;
        }
        if let Some(role) = self.roles.get(role_name) {
            for parent in &role.parents {
                if self.detect_cycle(parent, visited) {
                    return true;
                }
            }
        }
        visited.remove(role_name);
        false
    }

    /// Get number of actors tracked.
    pub fn actor_count(&self) -> usize {
        self.actor_roles.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_engine() -> PermissionEngine {
        let mut engine = PermissionEngine::new();

        let mut viewer = Role::new("viewer");
        viewer.add_permission(Permission::allow("read", "*"));

        let mut editor = Role::new("editor");
        editor.add_parent("viewer");
        editor.add_permission(Permission::allow("write", "document:*"));
        editor.add_permission(Permission::allow("update", "document:*"));

        let mut admin = Role::new("admin");
        admin.add_parent("editor");
        admin.add_permission(Permission::allow("*", "*"));

        engine.add_role(viewer);
        engine.add_role(editor);
        engine.add_role(admin);
        engine
    }

    #[test]
    fn test_direct_allow() {
        let mut engine = setup_engine();
        engine.assign_role("alice", "viewer");
        assert!(engine.evaluate("alice", "read", "document:1").is_allowed());
    }

    #[test]
    fn test_direct_deny() {
        let mut engine = setup_engine();
        engine.assign_role("bob", "viewer");
        assert!(!engine.evaluate("bob", "write", "document:1").is_allowed());
    }

    #[test]
    fn test_role_inheritance() {
        let mut engine = setup_engine();
        engine.assign_role("charlie", "editor");
        // Editor inherits viewer's read.
        assert!(engine.evaluate("charlie", "read", "document:1").is_allowed());
        // Editor has write on documents.
        assert!(engine.evaluate("charlie", "write", "document:1").is_allowed());
    }

    #[test]
    fn test_admin_wildcard() {
        let mut engine = setup_engine();
        engine.assign_role("dave", "admin");
        assert!(engine.evaluate("dave", "delete", "user:42").is_allowed());
        assert!(engine.evaluate("dave", "read", "anything").is_allowed());
    }

    #[test]
    fn test_deny_overrides_allow() {
        let mut engine = PermissionEngine::new();
        let mut role = Role::new("restricted");
        role.add_permission(Permission::allow("read", "*"));
        role.add_permission(Permission::deny("read", "secret:*"));
        engine.add_role(role);
        engine.assign_role("eve", "restricted");

        assert!(engine.evaluate("eve", "read", "document:1").is_allowed());
        assert!(!engine.evaluate("eve", "read", "secret:data").is_allowed());
    }

    #[test]
    fn test_actor_direct_permissions() {
        let mut engine = PermissionEngine::new();
        engine.add_actor_permission("frank", Permission::allow("read", "file:1"));
        assert!(engine.evaluate("frank", "read", "file:1").is_allowed());
        assert!(!engine.evaluate("frank", "write", "file:1").is_allowed());
    }

    #[test]
    fn test_wildcard_action() {
        assert!(pattern_matches("*", "anything"));
        assert!(pattern_matches("read", "read"));
        assert!(!pattern_matches("read", "write"));
    }

    #[test]
    fn test_wildcard_resource() {
        assert!(pattern_matches("document:*", "document:123"));
        assert!(pattern_matches("document:*", "document:abc"));
        assert!(!pattern_matches("document:*", "user:123"));
        assert!(!pattern_matches("document:*", "document")); // no colon after prefix
    }

    #[test]
    fn test_conditional_permission() {
        let mut engine = PermissionEngine::new();
        let mut role = Role::new("regional");
        role.add_permission(
            Permission::allow("read", "data:*").with_condition("region", "us-east"),
        );
        engine.add_role(role);
        engine.assign_role("geo-user", "regional");

        let mut ctx_east = HashMap::new();
        ctx_east.insert("region".to_string(), "us-east".to_string());
        assert!(engine
            .evaluate_with_context("geo-user", "read", "data:sales", &ctx_east)
            .is_allowed());

        let mut ctx_west = HashMap::new();
        ctx_west.insert("region".to_string(), "us-west".to_string());
        assert!(!engine
            .evaluate_with_context("geo-user", "read", "data:sales", &ctx_west)
            .is_allowed());
    }

    #[test]
    fn test_revoke_role() {
        let mut engine = setup_engine();
        engine.assign_role("alice", "admin");
        assert!(engine.evaluate("alice", "delete", "user:1").is_allowed());
        engine.revoke_role("alice", "admin");
        assert!(!engine.evaluate("alice", "delete", "user:1").is_allowed());
    }

    #[test]
    fn test_check_error() {
        let engine = PermissionEngine::new();
        let err = engine.check("nobody", "read", "file:1").unwrap_err();
        match err {
            PermissionError::AccessDenied { actor, .. } => assert_eq!(actor, "nobody"),
            _ => panic!("expected AccessDenied"),
        }
    }

    #[test]
    fn test_has_role() {
        let engine = setup_engine();
        assert!(engine.has_role("viewer"));
        assert!(engine.has_role("editor"));
        assert!(!engine.has_role("nonexistent"));
    }

    #[test]
    fn test_circular_inheritance_detection() {
        let mut engine = PermissionEngine::new();
        let mut a = Role::new("a");
        a.add_parent("b");
        let mut b = Role::new("b");
        b.add_parent("a");
        engine.add_role(a);
        engine.add_role(b);
        assert!(engine.has_circular_inheritance("a"));
    }

    #[test]
    fn test_no_circular_inheritance() {
        let engine = setup_engine();
        assert!(!engine.has_circular_inheritance("admin"));
        assert!(!engine.has_circular_inheritance("editor"));
        assert!(!engine.has_circular_inheritance("viewer"));
    }

    #[test]
    fn test_multiple_roles() {
        let mut engine = setup_engine();
        engine.assign_role("multi", "viewer");
        engine.assign_role("multi", "editor");
        let roles = engine.actor_roles("multi");
        assert_eq!(roles.len(), 2);
    }

    #[test]
    fn test_decision_is_allowed() {
        assert!(Decision::Allow.is_allowed());
        assert!(!Decision::Deny.is_allowed());
        assert!(!Decision::NoMatch.is_allowed());
    }

    #[test]
    fn test_effect_variants() {
        let allow = Permission::allow("r", "x");
        assert_eq!(allow.effect, Effect::Allow);
        let deny = Permission::deny("r", "x");
        assert_eq!(deny.effect, Effect::Deny);
    }

    #[test]
    fn test_permission_matches() {
        let perm = Permission::allow("read", "document:*");
        assert!(perm.matches("read", "document:123"));
        assert!(!perm.matches("write", "document:123"));
        assert!(!perm.matches("read", "user:123"));
    }

    #[test]
    fn test_error_display() {
        let e = PermissionError::AccessDenied {
            actor: "a".to_string(),
            action: "b".to_string(),
            resource: "c".to_string(),
        };
        assert!(e.to_string().contains("access denied"));
        let e2 = PermissionError::RoleNotFound("x".to_string());
        assert!(e2.to_string().contains("role not found"));
    }

    #[test]
    fn test_role_description() {
        let role = Role::new("admin").with_description("Full access");
        assert_eq!(role.description, "Full access");
    }

    #[test]
    fn test_actor_count() {
        let mut engine = setup_engine();
        assert_eq!(engine.actor_count(), 0);
        engine.assign_role("a", "viewer");
        engine.assign_role("b", "editor");
        assert_eq!(engine.actor_count(), 2);
    }

    #[test]
    fn test_unknown_actor_denied() {
        let engine = setup_engine();
        assert!(!engine.evaluate("unknown", "read", "file:1").is_allowed());
    }
}
