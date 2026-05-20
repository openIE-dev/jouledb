//! Multi-tenant isolation — tenant registry, resource ownership, cross-tenant access
//! prevention, tenant context propagation, tenant-scoped queries, tenant quota
//! management, and tenant hierarchy (parent/child).
//!
//! Replaces multi-tenant middleware (express-tenants, Django multi-tenant) with a
//! pure-Rust isolation layer that enforces tenant boundaries at the data model level.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ── Errors ─────────────────────────────────────────────────────

/// Tenant isolation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantError {
    /// Tenant not found.
    TenantNotFound(String),
    /// Duplicate tenant ID.
    DuplicateTenant(String),
    /// Cross-tenant access denied.
    CrossTenantAccess {
        requesting_tenant: String,
        owning_tenant: String,
        resource_id: String,
    },
    /// Resource not found.
    ResourceNotFound(String),
    /// Duplicate resource ID.
    DuplicateResource(String),
    /// Quota exceeded.
    QuotaExceeded {
        tenant_id: String,
        quota_name: String,
        limit: u64,
        current: u64,
    },
    /// Invalid tenant hierarchy.
    InvalidHierarchy(String),
    /// Tenant is suspended.
    TenantSuspended(String),
}

impl fmt::Display for TenantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TenantNotFound(id) => write!(f, "tenant not found: {id}"),
            Self::DuplicateTenant(id) => write!(f, "duplicate tenant: {id}"),
            Self::CrossTenantAccess { requesting_tenant, owning_tenant, resource_id } => {
                write!(
                    f,
                    "cross-tenant access denied: tenant {requesting_tenant} cannot access resource {resource_id} owned by {owning_tenant}"
                )
            }
            Self::ResourceNotFound(id) => write!(f, "resource not found: {id}"),
            Self::DuplicateResource(id) => write!(f, "duplicate resource: {id}"),
            Self::QuotaExceeded { tenant_id, quota_name, limit, current } => {
                write!(f, "quota '{quota_name}' exceeded for tenant {tenant_id}: {current}/{limit}")
            }
            Self::InvalidHierarchy(msg) => write!(f, "invalid hierarchy: {msg}"),
            Self::TenantSuspended(id) => write!(f, "tenant suspended: {id}"),
        }
    }
}

impl std::error::Error for TenantError {}

// ── Types ──────────────────────────────────────────────────────

/// Status of a tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    Active,
    Suspended,
    Provisioning,
    Deactivated,
}

impl TenantStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Suspended => "suspended",
            Self::Provisioning => "provisioning",
            Self::Deactivated => "deactivated",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

impl fmt::Display for TenantStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A quota limit for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quota {
    /// Name of the quota (e.g., "max_resources", "max_storage_bytes").
    pub name: String,
    /// Maximum allowed value.
    pub limit: u64,
    /// Current usage.
    pub current: u64,
}

impl Quota {
    pub fn new(name: &str, limit: u64) -> Self {
        Self {
            name: name.to_string(),
            limit,
            current: 0,
        }
    }

    /// Check whether incrementing by `amount` would exceed the limit.
    pub fn would_exceed(&self, amount: u64) -> bool {
        self.current.saturating_add(amount) > self.limit
    }

    /// Increment usage. Returns an error if the quota would be exceeded.
    pub fn increment(&mut self, amount: u64) -> Result<(), ()> {
        if self.would_exceed(amount) {
            return Err(());
        }
        self.current = self.current.saturating_add(amount);
        Ok(())
    }

    /// Decrement usage (floor at zero).
    pub fn decrement(&mut self, amount: u64) {
        self.current = self.current.saturating_sub(amount);
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.current)
    }

    /// Usage fraction (0.0 to 1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.limit == 0 {
            return 0.0;
        }
        self.current as f64 / self.limit as f64
    }
}

/// A tenant in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Status.
    pub status: TenantStatus,
    /// Parent tenant ID (for hierarchy).
    pub parent_id: Option<String>,
    /// Quotas.
    pub quotas: HashMap<String, Quota>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (epoch millis).
    pub created_at_ms: u64,
}

impl Tenant {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            status: TenantStatus::Active,
            parent_id: None,
            quotas: HashMap::new(),
            metadata: HashMap::new(),
            created_at_ms: 0,
        }
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn with_quota(mut self, name: &str, limit: u64) -> Self {
        self.quotas.insert(name.to_string(), Quota::new(name, limit));
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_created_at(mut self, ms: u64) -> Self {
        self.created_at_ms = ms;
        self
    }
}

/// A resource owned by a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantResource {
    /// Unique resource ID.
    pub id: String,
    /// Owning tenant ID.
    pub tenant_id: String,
    /// Resource type (e.g., "document", "bucket").
    pub resource_type: String,
    /// Resource name/label.
    pub name: String,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (epoch millis).
    pub created_at_ms: u64,
}

impl TenantResource {
    pub fn new(id: &str, tenant_id: &str, resource_type: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            tenant_id: tenant_id.to_string(),
            resource_type: resource_type.to_string(),
            name: name.to_string(),
            metadata: HashMap::new(),
            created_at_ms: 0,
        }
    }
}

/// Propagated tenant context for request-scoped isolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantContext {
    /// Current tenant ID.
    pub tenant_id: String,
    /// User ID within the tenant.
    pub user_id: Option<String>,
    /// Roles within the tenant.
    pub roles: Vec<String>,
    /// Whether cross-tenant access is allowed (e.g., for admin).
    pub allow_cross_tenant: bool,
}

impl TenantContext {
    pub fn new(tenant_id: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            user_id: None,
            roles: Vec::new(),
            allow_cross_tenant: false,
        }
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_role(mut self, role: &str) -> Self {
        self.roles.push(role.to_string());
        self
    }

    pub fn with_cross_tenant(mut self, allow: bool) -> Self {
        self.allow_cross_tenant = allow;
        self
    }
}

/// The tenant isolation manager.
pub struct TenantManager {
    tenants: HashMap<String, Tenant>,
    resources: HashMap<String, TenantResource>,
    /// Cross-tenant access grants: (from_tenant, to_tenant, resource_type) -> allowed.
    cross_tenant_grants: Vec<CrossTenantGrant>,
}

/// A grant allowing one tenant to access another tenant's resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossTenantGrant {
    pub from_tenant: String,
    pub to_tenant: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
}

impl TenantManager {
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            resources: HashMap::new(),
            cross_tenant_grants: Vec::new(),
        }
    }

    // ── Tenant Registry ────────────────────────────────────

    /// Register a new tenant.
    pub fn register_tenant(&mut self, tenant: Tenant) -> Result<(), TenantError> {
        if self.tenants.contains_key(&tenant.id) {
            return Err(TenantError::DuplicateTenant(tenant.id));
        }
        // Validate parent exists if specified.
        if let Some(parent_id) = &tenant.parent_id {
            if !self.tenants.contains_key(parent_id) {
                return Err(TenantError::InvalidHierarchy(format!(
                    "parent tenant {parent_id} not found"
                )));
            }
        }
        self.tenants.insert(tenant.id.clone(), tenant);
        Ok(())
    }

    /// Get a tenant by ID.
    pub fn get_tenant(&self, id: &str) -> Option<&Tenant> {
        self.tenants.get(id)
    }

    /// Get a mutable reference.
    pub fn get_tenant_mut(&mut self, id: &str) -> Option<&mut Tenant> {
        self.tenants.get_mut(id)
    }

    /// Remove a tenant (and all its resources).
    pub fn remove_tenant(&mut self, id: &str) -> Result<Tenant, TenantError> {
        // Check no children
        let has_children = self.tenants.values().any(|t| t.parent_id.as_deref() == Some(id));
        if has_children {
            return Err(TenantError::InvalidHierarchy(format!(
                "tenant {id} has children, remove them first"
            )));
        }
        // Remove all resources
        self.resources.retain(|_, r| r.tenant_id != id);
        // Remove grants
        self.cross_tenant_grants
            .retain(|g| g.from_tenant != id && g.to_tenant != id);
        self.tenants
            .remove(id)
            .ok_or_else(|| TenantError::TenantNotFound(id.to_string()))
    }

    /// Update tenant status.
    pub fn set_status(&mut self, id: &str, status: TenantStatus) -> Result<(), TenantError> {
        let tenant = self
            .tenants
            .get_mut(id)
            .ok_or_else(|| TenantError::TenantNotFound(id.to_string()))?;
        tenant.status = status;
        Ok(())
    }

    /// Number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.tenants.len()
    }

    /// Get child tenants of a parent.
    pub fn child_tenants(&self, parent_id: &str) -> Vec<&Tenant> {
        self.tenants
            .values()
            .filter(|t| t.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// Get the full ancestor chain for a tenant (from immediate parent to root).
    pub fn ancestor_chain(&self, tenant_id: &str) -> Vec<&str> {
        let mut chain = Vec::new();
        let mut current = tenant_id;
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(current) {
                break; // circular reference guard
            }
            match self.tenants.get(current).and_then(|t| t.parent_id.as_deref()) {
                Some(pid) => {
                    chain.push(pid);
                    current = pid;
                }
                None => break,
            }
        }
        chain
    }

    // ── Resource Ownership ─────────────────────────────────

    /// Add a resource to a tenant. Checks quotas if "max_resources" is defined.
    pub fn add_resource(&mut self, resource: TenantResource) -> Result<(), TenantError> {
        let tid = resource.tenant_id.clone();
        let rid = resource.id.clone();

        // Tenant must exist
        if !self.tenants.contains_key(&tid) {
            return Err(TenantError::TenantNotFound(tid));
        }

        // Tenant must be active
        let status = self.tenants[&tid].status;
        if status == TenantStatus::Suspended {
            return Err(TenantError::TenantSuspended(tid));
        }

        // Check resource uniqueness
        if self.resources.contains_key(&rid) {
            return Err(TenantError::DuplicateResource(rid));
        }

        // Check quota
        let tenant = self.tenants.get_mut(&tid).unwrap();
        if let Some(quota) = tenant.quotas.get_mut("max_resources") {
            if quota.would_exceed(1) {
                return Err(TenantError::QuotaExceeded {
                    tenant_id: tid,
                    quota_name: "max_resources".into(),
                    limit: quota.limit,
                    current: quota.current,
                });
            }
            quota.current += 1;
        }

        self.resources.insert(rid, resource);
        Ok(())
    }

    /// Remove a resource.
    pub fn remove_resource(&mut self, resource_id: &str) -> Result<TenantResource, TenantError> {
        let resource = self
            .resources
            .remove(resource_id)
            .ok_or_else(|| TenantError::ResourceNotFound(resource_id.to_string()))?;
        // Decrement quota
        if let Some(tenant) = self.tenants.get_mut(&resource.tenant_id) {
            if let Some(quota) = tenant.quotas.get_mut("max_resources") {
                quota.decrement(1);
            }
        }
        Ok(resource)
    }

    /// Get a resource, enforcing tenant isolation.
    pub fn get_resource(
        &self,
        resource_id: &str,
        ctx: &TenantContext,
    ) -> Result<&TenantResource, TenantError> {
        let resource = self
            .resources
            .get(resource_id)
            .ok_or_else(|| TenantError::ResourceNotFound(resource_id.to_string()))?;

        self.check_access(ctx, resource)?;
        Ok(resource)
    }

    /// Check whether a context can access a resource.
    fn check_access(
        &self,
        ctx: &TenantContext,
        resource: &TenantResource,
    ) -> Result<(), TenantError> {
        if resource.tenant_id == ctx.tenant_id {
            return Ok(());
        }
        if ctx.allow_cross_tenant {
            return Ok(());
        }
        // Check cross-tenant grants
        let has_grant = self.cross_tenant_grants.iter().any(|g| {
            g.from_tenant == ctx.tenant_id
                && g.to_tenant == resource.tenant_id
                && (g.resource_type.is_none()
                    || g.resource_type.as_deref() == Some(&resource.resource_type))
                && (g.resource_id.is_none()
                    || g.resource_id.as_deref() == Some(&resource.id))
        });
        // Also check hierarchy: parent can access child resources
        let is_ancestor = self
            .ancestor_chain(&resource.tenant_id)
            .contains(&ctx.tenant_id.as_str());

        if has_grant || is_ancestor {
            Ok(())
        } else {
            Err(TenantError::CrossTenantAccess {
                requesting_tenant: ctx.tenant_id.clone(),
                owning_tenant: resource.tenant_id.clone(),
                resource_id: resource.id.clone(),
            })
        }
    }

    /// List resources for a tenant (scoped query).
    pub fn list_resources(&self, tenant_id: &str) -> Vec<&TenantResource> {
        self.resources
            .values()
            .filter(|r| r.tenant_id == tenant_id)
            .collect()
    }

    /// List resources of a specific type for a tenant.
    pub fn list_resources_by_type(
        &self,
        tenant_id: &str,
        resource_type: &str,
    ) -> Vec<&TenantResource> {
        self.resources
            .values()
            .filter(|r| r.tenant_id == tenant_id && r.resource_type == resource_type)
            .collect()
    }

    /// Total resource count for a tenant.
    pub fn resource_count(&self, tenant_id: &str) -> usize {
        self.resources.values().filter(|r| r.tenant_id == tenant_id).count()
    }

    // ── Cross-Tenant Grants ────────────────────────────────

    /// Grant cross-tenant access.
    pub fn grant_cross_tenant_access(&mut self, grant: CrossTenantGrant) {
        self.cross_tenant_grants.push(grant);
    }

    /// Revoke all grants from one tenant to another.
    pub fn revoke_cross_tenant_access(&mut self, from_tenant: &str, to_tenant: &str) {
        self.cross_tenant_grants
            .retain(|g| !(g.from_tenant == from_tenant && g.to_tenant == to_tenant));
    }

    // ── Quota Management ───────────────────────────────────

    /// Set or update a quota for a tenant.
    pub fn set_quota(
        &mut self,
        tenant_id: &str,
        name: &str,
        limit: u64,
    ) -> Result<(), TenantError> {
        let tenant = self
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| TenantError::TenantNotFound(tenant_id.to_string()))?;
        let quota = tenant.quotas.entry(name.to_string()).or_insert_with(|| Quota::new(name, 0));
        quota.limit = limit;
        Ok(())
    }

    /// Get the current usage for a named quota.
    pub fn get_quota(&self, tenant_id: &str, name: &str) -> Option<&Quota> {
        self.tenants.get(tenant_id)?.quotas.get(name)
    }

    /// Increment a named quota.
    pub fn increment_quota(
        &mut self,
        tenant_id: &str,
        name: &str,
        amount: u64,
    ) -> Result<(), TenantError> {
        let tenant = self
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| TenantError::TenantNotFound(tenant_id.to_string()))?;
        let quota = tenant
            .quotas
            .get_mut(name)
            .ok_or_else(|| TenantError::QuotaExceeded {
                tenant_id: tenant_id.to_string(),
                quota_name: name.to_string(),
                limit: 0,
                current: 0,
            })?;
        if quota.would_exceed(amount) {
            return Err(TenantError::QuotaExceeded {
                tenant_id: tenant_id.to_string(),
                quota_name: name.to_string(),
                limit: quota.limit,
                current: quota.current,
            });
        }
        quota.current += amount;
        Ok(())
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_manager() -> TenantManager {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(
            Tenant::new("t1", "Acme Corp").with_quota("max_resources", 10),
        )
        .unwrap();
        mgr.register_tenant(Tenant::new("t2", "Globex")).unwrap();
        mgr
    }

    #[test]
    fn test_register_tenant() {
        let mgr = setup_manager();
        assert_eq!(mgr.tenant_count(), 2);
        assert!(mgr.get_tenant("t1").is_some());
    }

    #[test]
    fn test_duplicate_tenant() {
        let mut mgr = setup_manager();
        let err = mgr.register_tenant(Tenant::new("t1", "Dup")).unwrap_err();
        assert_eq!(err, TenantError::DuplicateTenant("t1".into()));
    }

    #[test]
    fn test_remove_tenant() {
        let mut mgr = setup_manager();
        mgr.remove_tenant("t2").unwrap();
        assert_eq!(mgr.tenant_count(), 1);
    }

    #[test]
    fn test_add_resource_and_list() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "My Doc"))
            .unwrap();
        mgr.add_resource(TenantResource::new("r2", "t1", "doc", "My Doc 2"))
            .unwrap();
        assert_eq!(mgr.resource_count("t1"), 2);
        assert_eq!(mgr.resource_count("t2"), 0);
    }

    #[test]
    fn test_tenant_isolation_enforced() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "Secret"))
            .unwrap();

        let ctx_t1 = TenantContext::new("t1");
        assert!(mgr.get_resource("r1", &ctx_t1).is_ok());

        let ctx_t2 = TenantContext::new("t2");
        let err = mgr.get_resource("r1", &ctx_t2).unwrap_err();
        match err {
            TenantError::CrossTenantAccess { .. } => {}
            other => panic!("expected CrossTenantAccess, got: {other}"),
        }
    }

    #[test]
    fn test_cross_tenant_admin_bypass() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "Secret"))
            .unwrap();
        let ctx = TenantContext::new("t2").with_cross_tenant(true);
        assert!(mgr.get_resource("r1", &ctx).is_ok());
    }

    #[test]
    fn test_cross_tenant_grant() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "Shared"))
            .unwrap();
        mgr.grant_cross_tenant_access(CrossTenantGrant {
            from_tenant: "t2".into(),
            to_tenant: "t1".into(),
            resource_type: Some("doc".into()),
            resource_id: None,
        });
        let ctx = TenantContext::new("t2");
        assert!(mgr.get_resource("r1", &ctx).is_ok());
    }

    #[test]
    fn test_revoke_cross_tenant_grant() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "Shared"))
            .unwrap();
        mgr.grant_cross_tenant_access(CrossTenantGrant {
            from_tenant: "t2".into(),
            to_tenant: "t1".into(),
            resource_type: None,
            resource_id: None,
        });
        mgr.revoke_cross_tenant_access("t2", "t1");
        let ctx = TenantContext::new("t2");
        assert!(mgr.get_resource("r1", &ctx).is_err());
    }

    #[test]
    fn test_quota_enforcement() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("t1", "Small").with_quota("max_resources", 2))
            .unwrap();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "D1")).unwrap();
        mgr.add_resource(TenantResource::new("r2", "t1", "doc", "D2")).unwrap();
        let err = mgr.add_resource(TenantResource::new("r3", "t1", "doc", "D3")).unwrap_err();
        match err {
            TenantError::QuotaExceeded { .. } => {}
            other => panic!("expected QuotaExceeded, got: {other}"),
        }
    }

    #[test]
    fn test_quota_decrement_on_remove() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("t1", "T").with_quota("max_resources", 2))
            .unwrap();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "D1")).unwrap();
        mgr.add_resource(TenantResource::new("r2", "t1", "doc", "D2")).unwrap();
        mgr.remove_resource("r1").unwrap();
        // Should now succeed
        mgr.add_resource(TenantResource::new("r3", "t1", "doc", "D3")).unwrap();
    }

    #[test]
    fn test_tenant_hierarchy() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("parent", "Parent Corp")).unwrap();
        mgr.register_tenant(Tenant::new("child", "Child Inc").with_parent("parent"))
            .unwrap();
        let children = mgr.child_tenants("parent");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "child");
    }

    #[test]
    fn test_parent_access_to_child_resources() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("parent", "Parent")).unwrap();
        mgr.register_tenant(Tenant::new("child", "Child").with_parent("parent"))
            .unwrap();
        mgr.add_resource(TenantResource::new("r1", "child", "doc", "Child Doc"))
            .unwrap();
        // Parent should be able to access child's resource
        let ctx = TenantContext::new("parent");
        assert!(mgr.get_resource("r1", &ctx).is_ok());
    }

    #[test]
    fn test_ancestor_chain() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("root", "Root")).unwrap();
        mgr.register_tenant(Tenant::new("mid", "Mid").with_parent("root")).unwrap();
        mgr.register_tenant(Tenant::new("leaf", "Leaf").with_parent("mid")).unwrap();
        let chain = mgr.ancestor_chain("leaf");
        assert_eq!(chain, vec!["mid", "root"]);
    }

    #[test]
    fn test_invalid_parent() {
        let mut mgr = TenantManager::new();
        let err = mgr
            .register_tenant(Tenant::new("child", "C").with_parent("nonexistent"))
            .unwrap_err();
        match err {
            TenantError::InvalidHierarchy(_) => {}
            other => panic!("expected InvalidHierarchy, got: {other}"),
        }
    }

    #[test]
    fn test_suspended_tenant_cannot_add_resources() {
        let mut mgr = setup_manager();
        mgr.set_status("t1", TenantStatus::Suspended).unwrap();
        let err = mgr
            .add_resource(TenantResource::new("r1", "t1", "doc", "D"))
            .unwrap_err();
        assert_eq!(err, TenantError::TenantSuspended("t1".into()));
    }

    #[test]
    fn test_remove_tenant_with_children_fails() {
        let mut mgr = TenantManager::new();
        mgr.register_tenant(Tenant::new("parent", "P")).unwrap();
        mgr.register_tenant(Tenant::new("child", "C").with_parent("parent")).unwrap();
        let err = mgr.remove_tenant("parent").unwrap_err();
        match err {
            TenantError::InvalidHierarchy(_) => {}
            other => panic!("expected InvalidHierarchy, got: {other}"),
        }
    }

    #[test]
    fn test_list_resources_by_type() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "D1")).unwrap();
        mgr.add_resource(TenantResource::new("r2", "t1", "image", "I1")).unwrap();
        mgr.add_resource(TenantResource::new("r3", "t1", "doc", "D2")).unwrap();
        let docs = mgr.list_resources_by_type("t1", "doc");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_set_and_get_quota() {
        let mut mgr = setup_manager();
        mgr.set_quota("t1", "max_storage_bytes", 1_000_000).unwrap();
        let q = mgr.get_quota("t1", "max_storage_bytes").unwrap();
        assert_eq!(q.limit, 1_000_000);
        assert_eq!(q.current, 0);
    }

    #[test]
    fn test_increment_quota() {
        let mut mgr = setup_manager();
        mgr.set_quota("t1", "storage", 100).unwrap();
        mgr.increment_quota("t1", "storage", 50).unwrap();
        assert_eq!(mgr.get_quota("t1", "storage").unwrap().current, 50);
        let err = mgr.increment_quota("t1", "storage", 60).unwrap_err();
        match err {
            TenantError::QuotaExceeded { .. } => {}
            other => panic!("expected QuotaExceeded, got: {other}"),
        }
    }

    #[test]
    fn test_quota_usage_fraction() {
        let mut q = Quota::new("test", 200);
        q.current = 100;
        assert!((q.usage_fraction() - 0.5).abs() < f64::EPSILON);
        assert_eq!(q.remaining(), 100);
    }

    #[test]
    fn test_tenant_context_builder() {
        let ctx = TenantContext::new("t1")
            .with_user("user-42")
            .with_role("admin")
            .with_cross_tenant(false);
        assert_eq!(ctx.tenant_id, "t1");
        assert_eq!(ctx.user_id, Some("user-42".into()));
        assert_eq!(ctx.roles, vec!["admin"]);
        assert!(!ctx.allow_cross_tenant);
    }

    #[test]
    fn test_resource_not_found() {
        let mgr = setup_manager();
        let ctx = TenantContext::new("t1");
        let err = mgr.get_resource("nonexistent", &ctx).unwrap_err();
        assert_eq!(err, TenantError::ResourceNotFound("nonexistent".into()));
    }

    #[test]
    fn test_duplicate_resource() {
        let mut mgr = setup_manager();
        mgr.add_resource(TenantResource::new("r1", "t1", "doc", "D1")).unwrap();
        let err = mgr.add_resource(TenantResource::new("r1", "t1", "doc", "D2")).unwrap_err();
        assert_eq!(err, TenantError::DuplicateResource("r1".into()));
    }
}
