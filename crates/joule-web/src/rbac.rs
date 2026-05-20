//! Role-Based Access Control (RBAC) — roles, permissions, role hierarchy,
//! permission inheritance, role assignment, permission checking, wildcard permissions.
//!
//! Replaces casbin, casl, accesscontrol, and similar JS/TS RBAC libraries
//! with a pure-Rust engine supporting hierarchical roles and wildcard grants.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// RBAC engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RbacError {
    /// Role not found.
    RoleNotFound(String),
    /// Duplicate role ID.
    DuplicateRole(String),
    /// Permission not found.
    PermissionNotFound(String),
    /// Duplicate permission.
    DuplicatePermission(String),
    /// Would create a cycle in the role hierarchy.
    CycleDetected { parent: String, child: String },
    /// Subject not found.
    SubjectNotFound(String),
    /// Role already assigned to subject.
    RoleAlreadyAssigned { subject: String, role: String },
    /// Role not assigned to subject.
    RoleNotAssigned { subject: String, role: String },
    /// Invalid wildcard pattern.
    InvalidWildcard(String),
}

impl fmt::Display for RbacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoleNotFound(id) => write!(f, "role not found: {id}"),
            Self::DuplicateRole(id) => write!(f, "duplicate role: {id}"),
            Self::PermissionNotFound(id) => write!(f, "permission not found: {id}"),
            Self::DuplicatePermission(id) => write!(f, "duplicate permission: {id}"),
            Self::CycleDetected { parent, child } => {
                write!(f, "cycle detected: {parent} -> {child}")
            }
            Self::SubjectNotFound(id) => write!(f, "subject not found: {id}"),
            Self::RoleAlreadyAssigned { subject, role } => {
                write!(f, "role {role} already assigned to {subject}")
            }
            Self::RoleNotAssigned { subject, role } => {
                write!(f, "role {role} not assigned to {subject}")
            }
            Self::InvalidWildcard(pat) => write!(f, "invalid wildcard: {pat}"),
        }
    }
}

impl std::error::Error for RbacError {}

// ── Types ──────────────────────────────────────────────────────

/// A permission string, supporting wildcards (`*`).
/// Format: `resource:action` or `resource:*` or `*:*`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    pub resource: String,
    pub action: String,
}

impl Permission {
    pub fn new(resource: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            resource: resource.into(),
            action: action.into(),
        }
    }

    /// Parse from `resource:action` format.
    pub fn parse(s: &str) -> Result<Self, RbacError> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(RbacError::InvalidWildcard(s.to_string()));
        }
        Ok(Self::new(parts[0], parts[1]))
    }

    /// Check if this permission matches another (considering wildcards).
    /// A wildcard `*` in resource or action matches any value.
    pub fn matches(&self, other: &Permission) -> bool {
        let resource_ok = self.resource == "*" || self.resource == other.resource;
        let action_ok = self.action == "*" || self.action == other.action;
        resource_ok && action_ok
    }

    /// Returns canonical string form.
    pub fn as_str(&self) -> String {
        format!("{}:{}", self.resource, self.action)
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.resource, self.action)
    }
}

/// A role with a set of directly granted permissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: HashSet<String>,
    pub parent_roles: Vec<String>,
}

impl Role {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            permissions: HashSet::new(),
            parent_roles: Vec::new(),
        }
    }
}

/// A subject (user/service) with assigned roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    pub id: String,
    pub roles: Vec<String>,
}

impl Subject {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            roles: Vec::new(),
        }
    }
}

/// Result of a permission check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessDecision {
    pub allowed: bool,
    pub matching_permission: Option<String>,
    pub via_role: Option<String>,
    pub inherited: bool,
}

// ── Engine ─────────────────────────────────────────────────────

/// The RBAC engine holding all roles, permissions, and subject assignments.
#[derive(Debug, Clone)]
pub struct RbacEngine {
    roles: HashMap<String, Role>,
    permissions: HashMap<String, Permission>,
    subjects: HashMap<String, Subject>,
}

impl RbacEngine {
    pub fn new() -> Self {
        Self {
            roles: HashMap::new(),
            permissions: HashMap::new(),
            subjects: HashMap::new(),
        }
    }

    // ── Permission management ──

    /// Register a permission.
    pub fn add_permission(&mut self, perm: Permission) -> Result<(), RbacError> {
        let key = perm.as_str();
        if self.permissions.contains_key(&key) {
            return Err(RbacError::DuplicatePermission(key));
        }
        self.permissions.insert(key, perm);
        Ok(())
    }

    /// Get a permission by its canonical key.
    pub fn get_permission(&self, key: &str) -> Option<&Permission> {
        self.permissions.get(key)
    }

    // ── Role management ──

    /// Add a role.
    pub fn add_role(&mut self, role: Role) -> Result<(), RbacError> {
        if self.roles.contains_key(&role.id) {
            return Err(RbacError::DuplicateRole(role.id.clone()));
        }
        self.roles.insert(role.id.clone(), role);
        Ok(())
    }

    /// Get a role by ID.
    pub fn get_role(&self, id: &str) -> Option<&Role> {
        self.roles.get(id)
    }

    /// Grant a permission to a role.
    pub fn grant_permission_to_role(
        &mut self,
        role_id: &str,
        perm_key: &str,
    ) -> Result<(), RbacError> {
        let role = self
            .roles
            .get_mut(role_id)
            .ok_or_else(|| RbacError::RoleNotFound(role_id.to_string()))?;
        role.permissions.insert(perm_key.to_string());
        Ok(())
    }

    /// Revoke a permission from a role.
    pub fn revoke_permission_from_role(
        &mut self,
        role_id: &str,
        perm_key: &str,
    ) -> Result<(), RbacError> {
        let role = self
            .roles
            .get_mut(role_id)
            .ok_or_else(|| RbacError::RoleNotFound(role_id.to_string()))?;
        if !role.permissions.remove(perm_key) {
            return Err(RbacError::PermissionNotFound(perm_key.to_string()));
        }
        Ok(())
    }

    /// Set parent roles (inheritance). Checks for cycles.
    pub fn set_role_parents(
        &mut self,
        role_id: &str,
        parents: Vec<String>,
    ) -> Result<(), RbacError> {
        if !self.roles.contains_key(role_id) {
            return Err(RbacError::RoleNotFound(role_id.to_string()));
        }
        for parent in &parents {
            if !self.roles.contains_key(parent) {
                return Err(RbacError::RoleNotFound(parent.clone()));
            }
            // Temporarily set parents to check for cycles.
            if parent == role_id || self.would_create_cycle(role_id, parent) {
                return Err(RbacError::CycleDetected {
                    parent: parent.clone(),
                    child: role_id.to_string(),
                });
            }
        }
        self.roles.get_mut(role_id).unwrap().parent_roles = parents;
        Ok(())
    }

    /// Check if setting `child` as inheriting from `parent` would create a cycle.
    fn would_create_cycle(&self, child: &str, parent: &str) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![parent.to_string()];
        while let Some(current) = stack.pop() {
            if current == child {
                return true;
            }
            if visited.insert(current.clone()) {
                if let Some(role) = self.roles.get(&current) {
                    for p in &role.parent_roles {
                        stack.push(p.clone());
                    }
                }
            }
        }
        false
    }

    /// Collect all permissions for a role, including inherited ones.
    pub fn effective_permissions(&self, role_id: &str) -> Result<HashSet<String>, RbacError> {
        if !self.roles.contains_key(role_id) {
            return Err(RbacError::RoleNotFound(role_id.to_string()));
        }
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        self.collect_permissions(role_id, &mut result, &mut visited);
        Ok(result)
    }

    fn collect_permissions(
        &self,
        role_id: &str,
        result: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(role_id.to_string()) {
            return;
        }
        if let Some(role) = self.roles.get(role_id) {
            for perm in &role.permissions {
                result.insert(perm.clone());
            }
            for parent in &role.parent_roles {
                self.collect_permissions(parent, result, visited);
            }
        }
    }

    // ── Subject management ──

    /// Register a subject.
    pub fn add_subject(&mut self, subject: Subject) -> Result<(), RbacError> {
        if self.subjects.contains_key(&subject.id) {
            return Err(RbacError::SubjectNotFound(subject.id.clone()));
        }
        self.subjects.insert(subject.id.clone(), subject);
        Ok(())
    }

    /// Assign a role to a subject.
    pub fn assign_role(&mut self, subject_id: &str, role_id: &str) -> Result<(), RbacError> {
        if !self.roles.contains_key(role_id) {
            return Err(RbacError::RoleNotFound(role_id.to_string()));
        }
        let subject = self
            .subjects
            .get_mut(subject_id)
            .ok_or_else(|| RbacError::SubjectNotFound(subject_id.to_string()))?;
        if subject.roles.contains(&role_id.to_string()) {
            return Err(RbacError::RoleAlreadyAssigned {
                subject: subject_id.to_string(),
                role: role_id.to_string(),
            });
        }
        subject.roles.push(role_id.to_string());
        Ok(())
    }

    /// Remove a role from a subject.
    pub fn revoke_role(&mut self, subject_id: &str, role_id: &str) -> Result<(), RbacError> {
        let subject = self
            .subjects
            .get_mut(subject_id)
            .ok_or_else(|| RbacError::SubjectNotFound(subject_id.to_string()))?;
        let idx = subject
            .roles
            .iter()
            .position(|r| r == role_id)
            .ok_or_else(|| RbacError::RoleNotAssigned {
                subject: subject_id.to_string(),
                role: role_id.to_string(),
            })?;
        subject.roles.remove(idx);
        Ok(())
    }

    /// Get all roles assigned to a subject.
    pub fn subject_roles(&self, subject_id: &str) -> Result<&[String], RbacError> {
        let subject = self
            .subjects
            .get(subject_id)
            .ok_or_else(|| RbacError::SubjectNotFound(subject_id.to_string()))?;
        Ok(&subject.roles)
    }

    // ── Permission checking ──

    /// Check if a subject has a specific permission.
    pub fn check_permission(
        &self,
        subject_id: &str,
        resource: &str,
        action: &str,
    ) -> Result<AccessDecision, RbacError> {
        let subject = self
            .subjects
            .get(subject_id)
            .ok_or_else(|| RbacError::SubjectNotFound(subject_id.to_string()))?;

        let requested = Permission::new(resource, action);

        for role_id in &subject.roles {
            let effective = self.effective_permissions(role_id)?;
            for perm_key in &effective {
                if let Some(perm) = self.permissions.get(perm_key) {
                    if perm.matches(&requested) {
                        let role = self.roles.get(role_id).unwrap();
                        let inherited = !role.permissions.contains(perm_key);
                        return Ok(AccessDecision {
                            allowed: true,
                            matching_permission: Some(perm_key.clone()),
                            via_role: Some(role_id.clone()),
                            inherited,
                        });
                    }
                }
                // Also try parsing the perm_key as a wildcard pattern.
                if let Ok(parsed) = Permission::parse(perm_key) {
                    if parsed.matches(&requested) {
                        let role = self.roles.get(role_id).unwrap();
                        let inherited = !role.permissions.contains(perm_key);
                        return Ok(AccessDecision {
                            allowed: true,
                            matching_permission: Some(perm_key.clone()),
                            via_role: Some(role_id.clone()),
                            inherited,
                        });
                    }
                }
            }
        }

        Ok(AccessDecision {
            allowed: false,
            matching_permission: None,
            via_role: None,
            inherited: false,
        })
    }

    /// Bulk check multiple permissions for a subject.
    pub fn check_permissions(
        &self,
        subject_id: &str,
        checks: &[(String, String)],
    ) -> Result<Vec<AccessDecision>, RbacError> {
        let mut results = Vec::with_capacity(checks.len());
        for (resource, action) in checks {
            results.push(self.check_permission(subject_id, resource, action)?);
        }
        Ok(results)
    }

    /// List all roles in the engine.
    pub fn list_roles(&self) -> Vec<&Role> {
        self.roles.values().collect()
    }

    /// List all subjects.
    pub fn list_subjects(&self) -> Vec<&Subject> {
        self.subjects.values().collect()
    }

    /// Count of roles.
    pub fn role_count(&self) -> usize {
        self.roles.len()
    }

    /// Count of permissions.
    pub fn permission_count(&self) -> usize {
        self.permissions.len()
    }
}

impl Default for RbacEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_engine() -> RbacEngine {
        let mut engine = RbacEngine::new();

        // Permissions
        engine
            .add_permission(Permission::new("docs", "read"))
            .unwrap();
        engine
            .add_permission(Permission::new("docs", "write"))
            .unwrap();
        engine
            .add_permission(Permission::new("docs", "delete"))
            .unwrap();
        engine
            .add_permission(Permission::new("users", "read"))
            .unwrap();
        engine
            .add_permission(Permission::new("users", "manage"))
            .unwrap();
        engine
            .add_permission(Permission::new("*", "*"))
            .unwrap();

        // Roles
        let viewer = Role::new("viewer", "Viewer", "Read-only access");
        let editor = Role::new("editor", "Editor", "Can edit documents");
        let admin = Role::new("admin", "Admin", "Full access");

        engine.add_role(viewer).unwrap();
        engine.add_role(editor).unwrap();
        engine.add_role(admin).unwrap();

        // Grant permissions
        engine
            .grant_permission_to_role("viewer", "docs:read")
            .unwrap();
        engine
            .grant_permission_to_role("editor", "docs:read")
            .unwrap();
        engine
            .grant_permission_to_role("editor", "docs:write")
            .unwrap();
        engine
            .grant_permission_to_role("admin", "*:*")
            .unwrap();

        // Hierarchy: editor inherits from viewer (redundant here but tests inheritance)
        engine
            .set_role_parents("editor", vec!["viewer".to_string()])
            .unwrap();

        // Subjects
        engine.add_subject(Subject::new("alice")).unwrap();
        engine.add_subject(Subject::new("bob")).unwrap();
        engine.add_subject(Subject::new("charlie")).unwrap();

        engine.assign_role("alice", "viewer").unwrap();
        engine.assign_role("bob", "editor").unwrap();
        engine.assign_role("charlie", "admin").unwrap();

        engine
    }

    #[test]
    fn test_permission_parse() {
        let p = Permission::parse("docs:read").unwrap();
        assert_eq!(p.resource, "docs");
        assert_eq!(p.action, "read");
    }

    #[test]
    fn test_permission_parse_invalid() {
        assert!(Permission::parse("noaction").is_err());
        assert!(Permission::parse(":read").is_err());
        assert!(Permission::parse("docs:").is_err());
    }

    #[test]
    fn test_permission_wildcard_matching() {
        let star_all = Permission::new("*", "*");
        let star_action = Permission::new("docs", "*");
        let specific = Permission::new("docs", "read");

        assert!(star_all.matches(&Permission::new("anything", "anything")));
        assert!(star_action.matches(&Permission::new("docs", "write")));
        assert!(!star_action.matches(&Permission::new("users", "read")));
        assert!(specific.matches(&Permission::new("docs", "read")));
        assert!(!specific.matches(&Permission::new("docs", "write")));
    }

    #[test]
    fn test_duplicate_role() {
        let mut engine = RbacEngine::new();
        engine
            .add_role(Role::new("admin", "Admin", ""))
            .unwrap();
        let err = engine
            .add_role(Role::new("admin", "Admin2", ""))
            .unwrap_err();
        assert_eq!(err, RbacError::DuplicateRole("admin".to_string()));
    }

    #[test]
    fn test_duplicate_permission() {
        let mut engine = RbacEngine::new();
        engine
            .add_permission(Permission::new("docs", "read"))
            .unwrap();
        let err = engine
            .add_permission(Permission::new("docs", "read"))
            .unwrap_err();
        assert_eq!(err, RbacError::DuplicatePermission("docs:read".to_string()));
    }

    #[test]
    fn test_viewer_can_read() {
        let engine = setup_engine();
        let decision = engine.check_permission("alice", "docs", "read").unwrap();
        assert!(decision.allowed);
        assert_eq!(decision.matching_permission.as_deref(), Some("docs:read"));
        assert_eq!(decision.via_role.as_deref(), Some("viewer"));
        assert!(!decision.inherited);
    }

    #[test]
    fn test_viewer_cannot_write() {
        let engine = setup_engine();
        let decision = engine.check_permission("alice", "docs", "write").unwrap();
        assert!(!decision.allowed);
    }

    #[test]
    fn test_editor_inherits_viewer() {
        let engine = setup_engine();
        let effective = engine.effective_permissions("editor").unwrap();
        assert!(effective.contains("docs:read"));
        assert!(effective.contains("docs:write"));
    }

    #[test]
    fn test_admin_wildcard() {
        let engine = setup_engine();
        let decision = engine
            .check_permission("charlie", "anything", "anything")
            .unwrap();
        assert!(decision.allowed);
        assert_eq!(decision.matching_permission.as_deref(), Some("*:*"));
    }

    #[test]
    fn test_cycle_detection() {
        let mut engine = RbacEngine::new();
        engine.add_role(Role::new("a", "A", "")).unwrap();
        engine.add_role(Role::new("b", "B", "")).unwrap();
        engine.add_role(Role::new("c", "C", "")).unwrap();

        engine
            .set_role_parents("b", vec!["a".to_string()])
            .unwrap();
        engine
            .set_role_parents("c", vec!["b".to_string()])
            .unwrap();

        // a -> b -> c -> a would be a cycle
        let err = engine
            .set_role_parents("a", vec!["c".to_string()])
            .unwrap_err();
        assert!(matches!(err, RbacError::CycleDetected { .. }));
    }

    #[test]
    fn test_self_cycle_detection() {
        let mut engine = RbacEngine::new();
        engine.add_role(Role::new("a", "A", "")).unwrap();
        let err = engine
            .set_role_parents("a", vec!["a".to_string()])
            .unwrap_err();
        assert!(matches!(err, RbacError::CycleDetected { .. }));
    }

    #[test]
    fn test_assign_unknown_role() {
        let mut engine = RbacEngine::new();
        engine.add_subject(Subject::new("u1")).unwrap();
        let err = engine.assign_role("u1", "ghost").unwrap_err();
        assert_eq!(err, RbacError::RoleNotFound("ghost".to_string()));
    }

    #[test]
    fn test_revoke_role() {
        let mut engine = setup_engine();
        engine.revoke_role("alice", "viewer").unwrap();
        let decision = engine.check_permission("alice", "docs", "read").unwrap();
        assert!(!decision.allowed);
    }

    #[test]
    fn test_revoke_unassigned_role() {
        let mut engine = setup_engine();
        let err = engine.revoke_role("alice", "admin").unwrap_err();
        assert!(matches!(err, RbacError::RoleNotAssigned { .. }));
    }

    #[test]
    fn test_double_assign() {
        let mut engine = setup_engine();
        let err = engine.assign_role("alice", "viewer").unwrap_err();
        assert!(matches!(err, RbacError::RoleAlreadyAssigned { .. }));
    }

    #[test]
    fn test_bulk_permission_check() {
        let engine = setup_engine();
        let checks = vec![
            ("docs".to_string(), "read".to_string()),
            ("docs".to_string(), "write".to_string()),
            ("docs".to_string(), "delete".to_string()),
        ];
        let results = engine.check_permissions("bob", &checks).unwrap();
        assert!(results[0].allowed); // read
        assert!(results[1].allowed); // write
        assert!(!results[2].allowed); // delete
    }

    #[test]
    fn test_inherited_flag() {
        let engine = setup_engine();
        // Editor has docs:read directly AND via viewer. Direct takes priority.
        let decision = engine.check_permission("bob", "docs", "read").unwrap();
        assert!(decision.allowed);
    }

    #[test]
    fn test_revoke_permission_from_role() {
        let mut engine = setup_engine();
        engine
            .revoke_permission_from_role("viewer", "docs:read")
            .unwrap();
        // Alice (only viewer) should no longer have docs:read
        let decision = engine.check_permission("alice", "docs", "read").unwrap();
        assert!(!decision.allowed);
    }

    #[test]
    fn test_revoke_nonexistent_permission() {
        let mut engine = setup_engine();
        let err = engine
            .revoke_permission_from_role("viewer", "docs:delete")
            .unwrap_err();
        assert_eq!(
            err,
            RbacError::PermissionNotFound("docs:delete".to_string())
        );
    }

    #[test]
    fn test_permission_display() {
        let p = Permission::new("docs", "read");
        assert_eq!(p.to_string(), "docs:read");
    }

    #[test]
    fn test_error_display() {
        let e = RbacError::RoleNotFound("x".to_string());
        assert_eq!(e.to_string(), "role not found: x");
    }

    #[test]
    fn test_counts() {
        let engine = setup_engine();
        assert_eq!(engine.role_count(), 3);
        assert_eq!(engine.permission_count(), 6);
    }

    #[test]
    fn test_list_roles_and_subjects() {
        let engine = setup_engine();
        assert_eq!(engine.list_roles().len(), 3);
        assert_eq!(engine.list_subjects().len(), 3);
    }

    #[test]
    fn test_subject_roles() {
        let engine = setup_engine();
        let roles = engine.subject_roles("bob").unwrap();
        assert_eq!(roles, &["editor"]);
    }

    #[test]
    fn test_get_permission() {
        let engine = setup_engine();
        let perm = engine.get_permission("docs:read").unwrap();
        assert_eq!(perm.resource, "docs");
        assert_eq!(perm.action, "read");
        assert!(engine.get_permission("nonexistent:perm").is_none());
    }

    #[test]
    fn test_get_role() {
        let engine = setup_engine();
        let role = engine.get_role("admin").unwrap();
        assert_eq!(role.name, "Admin");
        assert!(engine.get_role("nonexistent").is_none());
    }

    #[test]
    fn test_deep_inheritance_chain() {
        let mut engine = RbacEngine::new();
        engine
            .add_permission(Permission::new("secret", "access"))
            .unwrap();

        engine.add_role(Role::new("l1", "L1", "")).unwrap();
        engine.add_role(Role::new("l2", "L2", "")).unwrap();
        engine.add_role(Role::new("l3", "L3", "")).unwrap();
        engine.add_role(Role::new("l4", "L4", "")).unwrap();

        engine
            .grant_permission_to_role("l1", "secret:access")
            .unwrap();
        engine
            .set_role_parents("l2", vec!["l1".to_string()])
            .unwrap();
        engine
            .set_role_parents("l3", vec!["l2".to_string()])
            .unwrap();
        engine
            .set_role_parents("l4", vec!["l3".to_string()])
            .unwrap();

        let effective = engine.effective_permissions("l4").unwrap();
        assert!(effective.contains("secret:access"));
    }

    #[test]
    fn test_diamond_inheritance() {
        let mut engine = RbacEngine::new();
        engine
            .add_permission(Permission::new("base", "perm"))
            .unwrap();
        engine
            .add_permission(Permission::new("left", "perm"))
            .unwrap();
        engine
            .add_permission(Permission::new("right", "perm"))
            .unwrap();

        engine.add_role(Role::new("base", "Base", "")).unwrap();
        engine.add_role(Role::new("left", "Left", "")).unwrap();
        engine.add_role(Role::new("right", "Right", "")).unwrap();
        engine.add_role(Role::new("top", "Top", "")).unwrap();

        engine
            .grant_permission_to_role("base", "base:perm")
            .unwrap();
        engine
            .grant_permission_to_role("left", "left:perm")
            .unwrap();
        engine
            .grant_permission_to_role("right", "right:perm")
            .unwrap();

        engine
            .set_role_parents("left", vec!["base".to_string()])
            .unwrap();
        engine
            .set_role_parents("right", vec!["base".to_string()])
            .unwrap();
        engine
            .set_role_parents("top", vec!["left".to_string(), "right".to_string()])
            .unwrap();

        let effective = engine.effective_permissions("top").unwrap();
        assert!(effective.contains("base:perm"));
        assert!(effective.contains("left:perm"));
        assert!(effective.contains("right:perm"));
    }

    #[test]
    fn test_default_engine() {
        let engine = RbacEngine::default();
        assert_eq!(engine.role_count(), 0);
        assert_eq!(engine.permission_count(), 0);
    }

    #[test]
    fn test_check_unknown_subject() {
        let engine = setup_engine();
        let err = engine
            .check_permission("ghost", "docs", "read")
            .unwrap_err();
        assert_eq!(err, RbacError::SubjectNotFound("ghost".to_string()));
    }

    #[test]
    fn test_set_parents_unknown_role() {
        let mut engine = RbacEngine::new();
        let err = engine
            .set_role_parents("ghost", vec![])
            .unwrap_err();
        assert_eq!(err, RbacError::RoleNotFound("ghost".to_string()));
    }

    #[test]
    fn test_set_parents_unknown_parent() {
        let mut engine = RbacEngine::new();
        engine.add_role(Role::new("a", "A", "")).unwrap();
        let err = engine
            .set_role_parents("a", vec!["ghost".to_string()])
            .unwrap_err();
        assert_eq!(err, RbacError::RoleNotFound("ghost".to_string()));
    }

    #[test]
    fn test_grant_to_unknown_role() {
        let mut engine = RbacEngine::new();
        let err = engine
            .grant_permission_to_role("ghost", "docs:read")
            .unwrap_err();
        assert_eq!(err, RbacError::RoleNotFound("ghost".to_string()));
    }
}
