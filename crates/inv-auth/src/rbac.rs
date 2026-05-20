use std::fmt;

use serde::{Deserialize, Serialize};

/// Roles in the Invisible Infrastructure RBAC system.
///
/// Follows a hierarchical model: Admin > Operator > Viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Full control over the org: manage users, roles, billing, and all resources.
    Admin,
    /// Deploy, manage, and monitor workloads and infrastructure.
    Operator,
    /// Read-only access to all resources and telemetry.
    Viewer,
    /// Node agent — can heartbeat, report energy, but cannot manage workloads.
    Agent,
    /// Customer — can manage projects, billing, and API keys (no infrastructure access).
    Customer,
}

impl Role {
    /// Returns the permissions granted to this role.
    pub fn permissions(&self) -> &'static [Permission] {
        match self {
            Role::Admin => Permission::all(),
            Role::Operator => &[
                Permission::NodeRead,
                Permission::NodeRegister,
                Permission::NodeDeregister,
                Permission::WorkloadRead,
                Permission::WorkloadDeploy,
                Permission::WorkloadDestroy,
                Permission::EnergyRead,
                Permission::StorageRead,
                Permission::StorageWrite,
                Permission::SecretRead,
                Permission::AuditRead,
            ],
            Role::Viewer => &[
                Permission::NodeRead,
                Permission::WorkloadRead,
                Permission::EnergyRead,
                Permission::StorageRead,
                Permission::AuditRead,
            ],
            Role::Agent => &[
                Permission::NodeRead,
                Permission::NodeHeartbeat,
                Permission::EnergyReport,
                Permission::WorkloadRead,
            ],
            Role::Customer => &[
                Permission::ProjectRead,
                Permission::ProjectDeploy,
                Permission::ProjectDestroy,
                Permission::BillingRead,
                Permission::BillingManage,
                Permission::ApiKeyManage,
                Permission::WorkloadRead,
                Permission::EnergyRead,
                Permission::StorageRead,
            ],
        }
    }

    /// Check if this role has a specific permission.
    pub fn has_permission(&self, perm: Permission) -> bool {
        self.permissions().contains(&perm)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::Operator => write!(f, "operator"),
            Role::Viewer => write!(f, "viewer"),
            Role::Agent => write!(f, "agent"),
            Role::Customer => write!(f, "customer"),
        }
    }
}

/// Fine-grained permissions for API operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    // Node permissions
    NodeRead,
    NodeRegister,
    NodeDeregister,
    NodeHeartbeat,

    // Workload permissions
    WorkloadRead,
    WorkloadDeploy,
    WorkloadDestroy,

    // Energy permissions
    EnergyRead,
    EnergyReport,

    // Storage permissions
    StorageRead,
    StorageWrite,

    // Secret permissions
    SecretRead,
    SecretWrite,

    // Audit permissions
    AuditRead,

    // Rate limit permissions
    RateLimitManage,

    // Org management
    OrgManage,
    UserManage,

    // Project permissions (customer-facing)
    ProjectRead,
    ProjectDeploy,
    ProjectDestroy,

    // Billing permissions (customer-facing)
    BillingRead,
    BillingManage,

    // API key management
    ApiKeyManage,
}

impl Permission {
    /// All defined permissions.
    pub fn all() -> &'static [Permission] {
        &[
            Permission::NodeRead,
            Permission::NodeRegister,
            Permission::NodeDeregister,
            Permission::NodeHeartbeat,
            Permission::WorkloadRead,
            Permission::WorkloadDeploy,
            Permission::WorkloadDestroy,
            Permission::EnergyRead,
            Permission::EnergyReport,
            Permission::StorageRead,
            Permission::StorageWrite,
            Permission::SecretRead,
            Permission::SecretWrite,
            Permission::AuditRead,
            Permission::RateLimitManage,
            Permission::OrgManage,
            Permission::UserManage,
            Permission::ProjectRead,
            Permission::ProjectDeploy,
            Permission::ProjectDestroy,
            Permission::BillingRead,
            Permission::BillingManage,
            Permission::ApiKeyManage,
        ]
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Permission::NodeRead => write!(f, "node:read"),
            Permission::NodeRegister => write!(f, "node:register"),
            Permission::NodeDeregister => write!(f, "node:deregister"),
            Permission::NodeHeartbeat => write!(f, "node:heartbeat"),
            Permission::WorkloadRead => write!(f, "workload:read"),
            Permission::WorkloadDeploy => write!(f, "workload:deploy"),
            Permission::WorkloadDestroy => write!(f, "workload:destroy"),
            Permission::EnergyRead => write!(f, "energy:read"),
            Permission::EnergyReport => write!(f, "energy:report"),
            Permission::StorageRead => write!(f, "storage:read"),
            Permission::StorageWrite => write!(f, "storage:write"),
            Permission::SecretRead => write!(f, "secret:read"),
            Permission::SecretWrite => write!(f, "secret:write"),
            Permission::AuditRead => write!(f, "audit:read"),
            Permission::RateLimitManage => write!(f, "ratelimit:manage"),
            Permission::OrgManage => write!(f, "org:manage"),
            Permission::UserManage => write!(f, "user:manage"),
            Permission::ProjectRead => write!(f, "project:read"),
            Permission::ProjectDeploy => write!(f, "project:deploy"),
            Permission::ProjectDestroy => write!(f, "project:destroy"),
            Permission::BillingRead => write!(f, "billing:read"),
            Permission::BillingManage => write!(f, "billing:manage"),
            Permission::ApiKeyManage => write!(f, "apikey:manage"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_hierarchy() {
        // Admin has everything
        for perm in Permission::all() {
            assert!(
                Role::Admin.has_permission(*perm),
                "admin should have {perm}"
            );
        }

        // Operator has deploy but not org manage
        assert!(Role::Operator.has_permission(Permission::WorkloadDeploy));
        assert!(Role::Operator.has_permission(Permission::NodeRegister));
        assert!(Role::Operator.has_permission(Permission::SecretRead));
        assert!(Role::Operator.has_permission(Permission::AuditRead));
        assert!(!Role::Operator.has_permission(Permission::SecretWrite));
        assert!(!Role::Operator.has_permission(Permission::RateLimitManage));
        assert!(!Role::Operator.has_permission(Permission::OrgManage));
        assert!(!Role::Operator.has_permission(Permission::UserManage));

        // Viewer is read-only
        assert!(Role::Viewer.has_permission(Permission::NodeRead));
        assert!(Role::Viewer.has_permission(Permission::WorkloadRead));
        assert!(Role::Viewer.has_permission(Permission::AuditRead));
        assert!(!Role::Viewer.has_permission(Permission::SecretRead));
        assert!(!Role::Viewer.has_permission(Permission::WorkloadDeploy));
        assert!(!Role::Viewer.has_permission(Permission::NodeRegister));

        // Agent can heartbeat and report energy
        assert!(Role::Agent.has_permission(Permission::NodeHeartbeat));
        assert!(Role::Agent.has_permission(Permission::EnergyReport));
        assert!(!Role::Agent.has_permission(Permission::WorkloadDeploy));
        assert!(!Role::Agent.has_permission(Permission::SecretRead));
        assert!(!Role::Agent.has_permission(Permission::OrgManage));

        // Customer can manage projects and billing but not infrastructure
        assert!(Role::Customer.has_permission(Permission::ProjectRead));
        assert!(Role::Customer.has_permission(Permission::ProjectDeploy));
        assert!(Role::Customer.has_permission(Permission::ProjectDestroy));
        assert!(Role::Customer.has_permission(Permission::BillingRead));
        assert!(Role::Customer.has_permission(Permission::BillingManage));
        assert!(Role::Customer.has_permission(Permission::ApiKeyManage));
        assert!(Role::Customer.has_permission(Permission::WorkloadRead));
        assert!(Role::Customer.has_permission(Permission::EnergyRead));
        assert!(!Role::Customer.has_permission(Permission::NodeRegister));
        assert!(!Role::Customer.has_permission(Permission::OrgManage));
        assert!(!Role::Customer.has_permission(Permission::SecretRead));
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::Admin.to_string(), "admin");
        assert_eq!(Role::Operator.to_string(), "operator");
        assert_eq!(Role::Viewer.to_string(), "viewer");
        assert_eq!(Role::Agent.to_string(), "agent");
        assert_eq!(Role::Customer.to_string(), "customer");
    }

    #[test]
    fn permission_display() {
        assert_eq!(Permission::NodeRead.to_string(), "node:read");
        assert_eq!(Permission::WorkloadDeploy.to_string(), "workload:deploy");
        assert_eq!(Permission::OrgManage.to_string(), "org:manage");
    }

    #[test]
    fn role_serde_roundtrip() {
        let role = Role::Operator;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"operator\"");
        let parsed: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, role);
    }

    #[test]
    fn permission_serde_roundtrip() {
        let perm = Permission::WorkloadDeploy;
        let json = serde_json::to_string(&perm).unwrap();
        assert_eq!(json, "\"workload_deploy\"");
        let parsed: Permission = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, perm);
    }
}
