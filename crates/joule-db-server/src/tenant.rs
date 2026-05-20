//! Multi-Tenant Isolation for JouleDB
//!
//! Provides per-tenant namespace isolation with resource quotas and
//! energy budgets. Tenants are isolated via table name prefixing ---
//! tenant "acme" querying "users" actually accesses "acme::users".

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum TenantError {
    #[error("tenant not found: {0}")]
    NotFound(String),

    #[error("tenant already exists: {0}")]
    AlreadyExists(String),

    #[error("cannot delete the default tenant")]
    CannotDeleteDefault,

    #[error("energy budget exhausted: spent {spent_uj} \u{00b5}J of {budget_uj} \u{00b5}J budget")]
    EnergyBudgetExhausted { spent_uj: u64, budget_uj: u64 },

    #[error("tenant is suspended: {0}")]
    Suspended(String),

    #[error("storage error: {0}")]
    Storage(String),
}

// ============================================================================
// Tenant status
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TenantStatus {
    Active,
    Suspended,
    PendingDelete,
}

// ============================================================================
// Tenant quotas
// ============================================================================

/// Resource quotas applied to a tenant
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TenantQuotas {
    /// Maximum storage in bytes (None = unlimited)
    pub max_storage_bytes: Option<u64>,

    /// Maximum number of tables (None = unlimited)
    pub max_tables: Option<u32>,

    /// Energy budget in microjoules (None = unlimited)
    pub energy_budget_uj: Option<u64>,
}

// ============================================================================
// Tenant metadata
// ============================================================================

/// Metadata for a single tenant
#[derive(Debug, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique identifier (derived from name)
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Creation timestamp (Unix millis)
    pub created_at: u64,

    /// Current status
    pub status: TenantStatus,

    /// Resource quotas
    pub quotas: TenantQuotas,

    /// Energy consumed so far in microjoules
    #[serde(
        serialize_with = "serialize_atomic_u64",
        deserialize_with = "deserialize_atomic_u64"
    )]
    pub energy_spent_uj: Arc<AtomicU64>,

    /// Number of tables owned by this tenant
    #[serde(
        serialize_with = "serialize_atomic_u64_bare",
        deserialize_with = "deserialize_atomic_u64_bare"
    )]
    pub table_count: AtomicU64,
}

fn serialize_atomic_u64<S: serde::Serializer>(
    val: &Arc<AtomicU64>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_atomic_u64<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Arc<AtomicU64>, D::Error> {
    let val = u64::deserialize(deserializer)?;
    Ok(Arc::new(AtomicU64::new(val)))
}

fn serialize_atomic_u64_bare<S: serde::Serializer>(
    val: &AtomicU64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_atomic_u64_bare<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<AtomicU64, D::Error> {
    let val = u64::deserialize(deserializer)?;
    Ok(AtomicU64::new(val))
}

impl Tenant {
    /// Remaining energy budget in microjoules (None = unlimited)
    pub fn energy_remaining_uj(&self) -> Option<u64> {
        self.quotas.energy_budget_uj.map(|budget| {
            let spent = self.energy_spent_uj.load(Ordering::Relaxed);
            budget.saturating_sub(spent)
        })
    }

    /// Check whether the tenant can afford an operation with estimated cost
    pub fn can_afford(&self, estimated_uj: u64) -> bool {
        match self.energy_remaining_uj() {
            Some(remaining) => remaining >= estimated_uj,
            None => true,
        }
    }

    /// Record energy consumption. Returns Err if budget would be exhausted.
    pub fn record_energy(&self, consumed_uj: u64) -> Result<u64, TenantError> {
        if let Some(budget) = self.quotas.energy_budget_uj {
            loop {
                let current = self.energy_spent_uj.load(Ordering::Relaxed);
                let new_total = current + consumed_uj;
                if new_total > budget {
                    return Err(TenantError::EnergyBudgetExhausted {
                        spent_uj: current,
                        budget_uj: budget,
                    });
                }
                match self.energy_spent_uj.compare_exchange_weak(
                    current,
                    new_total,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Ok(new_total),
                    Err(_) => continue,
                }
            }
        } else {
            let prev = self
                .energy_spent_uj
                .fetch_add(consumed_uj, Ordering::Relaxed);
            Ok(prev + consumed_uj)
        }
    }
}

// ============================================================================
// Tenant info (serializable summary)
// ============================================================================

/// Summary info for listing tenants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantInfo {
    pub id: String,
    pub name: String,
    pub status: TenantStatus,
    pub created_at: u64,
    pub quotas: TenantQuotas,
    pub energy_spent_uj: u64,
    pub table_count: u64,
}

impl From<&Tenant> for TenantInfo {
    fn from(t: &Tenant) -> Self {
        Self {
            id: t.id.clone(),
            name: t.name.clone(),
            status: t.status,
            created_at: t.created_at,
            quotas: t.quotas.clone(),
            energy_spent_uj: t.energy_spent_uj.load(Ordering::Relaxed),
            table_count: t.table_count.load(Ordering::Relaxed),
        }
    }
}

// ============================================================================
// Create tenant request
// ============================================================================

/// Request to create a new tenant
#[derive(Debug, Clone, Deserialize)]
pub struct CreateTenantRequest {
    /// Name for the new tenant
    pub name: String,

    /// Optional resource quotas
    pub quotas: Option<TenantQuotas>,
}

// ============================================================================
// Tenant manager
// ============================================================================

/// The tenant manager tracks all tenants and coordinates isolation.
pub struct TenantManager {
    /// All tenants indexed by id
    tenants: RwLock<HashMap<String, Tenant>>,
}

impl TenantManager {
    /// Create a new tenant manager with an initial "default" tenant
    pub fn new() -> Self {
        let mut tenants = HashMap::new();
        tenants.insert(
            "default".to_string(),
            Tenant {
                id: "default".to_string(),
                name: "default".to_string(),
                created_at: now_millis(),
                status: TenantStatus::Active,
                quotas: TenantQuotas::default(),
                energy_spent_uj: Arc::new(AtomicU64::new(0)),
                table_count: AtomicU64::new(0),
            },
        );

        Self {
            tenants: RwLock::new(tenants),
        }
    }

    /// Create a new tenant
    pub fn create_tenant(&self, req: CreateTenantRequest) -> Result<TenantInfo, TenantError> {
        let mut tenants = self
            .tenants
            .write()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let id = tenant_id(&req.name)?;

        // Check id doesn't already exist
        if tenants.contains_key(&id) {
            return Err(TenantError::AlreadyExists(req.name));
        }

        let tenant = Tenant {
            id: id.clone(),
            name: req.name,
            created_at: now_millis(),
            status: TenantStatus::Active,
            quotas: req.quotas.unwrap_or_default(),
            energy_spent_uj: Arc::new(AtomicU64::new(0)),
            table_count: AtomicU64::new(0),
        };

        let info = TenantInfo::from(&tenant);
        tenants.insert(id, tenant);

        Ok(info)
    }

    /// Delete a tenant (cannot delete "default")
    pub fn delete_tenant(&self, id: &str) -> Result<TenantInfo, TenantError> {
        if id == "default" {
            return Err(TenantError::CannotDeleteDefault);
        }

        let mut tenants = self
            .tenants
            .write()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let tenant = tenants
            .remove(id)
            .ok_or_else(|| TenantError::NotFound(id.to_string()))?;

        Ok(TenantInfo::from(&tenant))
    }

    /// List all tenants
    pub fn list_tenants(&self) -> Result<Vec<TenantInfo>, TenantError> {
        let tenants = self
            .tenants
            .read()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let mut infos: Vec<TenantInfo> = tenants.values().map(TenantInfo::from).collect();
        infos.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(infos)
    }

    /// Get info about a specific tenant
    pub fn get_tenant(&self, id: &str) -> Result<TenantInfo, TenantError> {
        let tenants = self
            .tenants
            .read()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        tenants
            .get(id)
            .map(TenantInfo::from)
            .ok_or_else(|| TenantError::NotFound(id.to_string()))
    }

    /// Suspend a tenant, preventing further operations
    pub fn suspend_tenant(&self, id: &str) -> Result<TenantInfo, TenantError> {
        let mut tenants = self
            .tenants
            .write()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let tenant = tenants
            .get_mut(id)
            .ok_or_else(|| TenantError::NotFound(id.to_string()))?;

        tenant.status = TenantStatus::Suspended;

        Ok(TenantInfo::from(&*tenant))
    }

    /// Check if a tenant can afford an operation with the given estimated energy cost.
    ///
    /// Returns `false` if the budget would be exceeded, `true` otherwise.
    /// Returns `Err` if the tenant is not found or is suspended.
    pub fn check_energy_budget(&self, id: &str, estimated_uj: u64) -> Result<bool, TenantError> {
        let tenants = self
            .tenants
            .read()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let tenant = tenants
            .get(id)
            .ok_or_else(|| TenantError::NotFound(id.to_string()))?;

        if tenant.status == TenantStatus::Suspended {
            return Err(TenantError::Suspended(id.to_string()));
        }

        Ok(tenant.can_afford(estimated_uj))
    }

    /// Record energy consumption on a tenant
    pub fn record_energy(&self, id: &str, consumed_uj: u64) -> Result<u64, TenantError> {
        let tenants = self
            .tenants
            .read()
            .map_err(|e| TenantError::Storage(e.to_string()))?;

        let tenant = tenants
            .get(id)
            .ok_or_else(|| TenantError::NotFound(id.to_string()))?;

        if tenant.status == TenantStatus::Suspended {
            return Err(TenantError::Suspended(id.to_string()));
        }

        tenant.record_energy(consumed_uj)
    }

    /// Prefix a table name with the tenant namespace.
    ///
    /// Tenant "acme" querying "users" => "acme::users".
    pub fn prefix_table(tenant_id: &str, table: &str) -> String {
        format!("{tenant_id}::{table}")
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Derive a tenant id from the name (lowercase, spaces to hyphens).
///
/// Rejects names that are empty, longer than 128 chars, or contain the
/// namespace separator "::" (which would break tenant isolation).
fn tenant_id(name: &str) -> Result<String, TenantError> {
    let id = name.to_lowercase().replace(' ', "-");
    if id.is_empty() {
        return Err(TenantError::Storage("Tenant name cannot be empty".into()));
    }
    if id.len() > 128 {
        return Err(TenantError::Storage(
            "Tenant name too long (max 128 chars)".into(),
        ));
    }
    if id.contains("::") {
        return Err(TenantError::Storage(
            "Tenant name cannot contain '::' (reserved namespace separator)".into(),
        ));
    }
    Ok(id)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tenant() {
        let mgr = TenantManager::new();

        let info = mgr
            .create_tenant(CreateTenantRequest {
                name: "Acme Corp".to_string(),
                quotas: Some(TenantQuotas {
                    max_storage_bytes: Some(1_000_000),
                    max_tables: Some(50),
                    energy_budget_uj: Some(500_000),
                }),
            })
            .unwrap();

        assert_eq!(info.id, "acme-corp");
        assert_eq!(info.name, "Acme Corp");
        assert_eq!(info.status, TenantStatus::Active);
        assert_eq!(info.quotas.energy_budget_uj, Some(500_000));
        assert_eq!(info.energy_spent_uj, 0);
        assert_eq!(info.table_count, 0);
    }

    #[test]
    fn test_duplicate_name() {
        let mgr = TenantManager::new();

        mgr.create_tenant(CreateTenantRequest {
            name: "dup".to_string(),
            quotas: None,
        })
        .unwrap();

        let result = mgr.create_tenant(CreateTenantRequest {
            name: "dup".to_string(),
            quotas: None,
        });

        assert!(matches!(result, Err(TenantError::AlreadyExists(_))));
    }

    #[test]
    fn test_cannot_delete_default() {
        let mgr = TenantManager::new();
        assert!(matches!(
            mgr.delete_tenant("default"),
            Err(TenantError::CannotDeleteDefault)
        ));
    }

    #[test]
    fn test_delete_tenant() {
        let mgr = TenantManager::new();

        mgr.create_tenant(CreateTenantRequest {
            name: "temp".to_string(),
            quotas: None,
        })
        .unwrap();

        assert!(mgr.delete_tenant("temp").is_ok());
        assert!(mgr.get_tenant("temp").is_err());
    }

    #[test]
    fn test_list_tenants() {
        let mgr = TenantManager::new();

        mgr.create_tenant(CreateTenantRequest {
            name: "beta".to_string(),
            quotas: None,
        })
        .unwrap();

        mgr.create_tenant(CreateTenantRequest {
            name: "alpha".to_string(),
            quotas: None,
        })
        .unwrap();

        let list = mgr.list_tenants().unwrap();
        assert_eq!(list.len(), 3); // default + alpha + beta
        assert_eq!(list[0].id, "alpha"); // sorted by id
    }

    #[test]
    fn test_energy_budget() {
        let mgr = TenantManager::new();

        mgr.create_tenant(CreateTenantRequest {
            name: "metered".to_string(),
            quotas: Some(TenantQuotas {
                max_storage_bytes: None,
                max_tables: None,
                energy_budget_uj: Some(1000),
            }),
        })
        .unwrap();

        assert!(mgr.check_energy_budget("metered", 500).unwrap());
        mgr.record_energy("metered", 500).unwrap();

        assert!(mgr.check_energy_budget("metered", 500).unwrap());
        mgr.record_energy("metered", 500).unwrap();

        // Exhausted
        assert!(!mgr.check_energy_budget("metered", 1).unwrap());
        assert!(mgr.record_energy("metered", 1).is_err());
    }

    #[test]
    fn test_suspend_tenant() {
        let mgr = TenantManager::new();

        mgr.create_tenant(CreateTenantRequest {
            name: "suspendable".to_string(),
            quotas: None,
        })
        .unwrap();

        let info = mgr.suspend_tenant("suspendable").unwrap();
        assert_eq!(info.status, TenantStatus::Suspended);

        // Suspended tenants cannot record energy
        assert!(matches!(
            mgr.record_energy("suspendable", 100),
            Err(TenantError::Suspended(_))
        ));

        // Suspended tenants cannot check energy budget
        assert!(matches!(
            mgr.check_energy_budget("suspendable", 100),
            Err(TenantError::Suspended(_))
        ));
    }

    #[test]
    fn test_prefix_table() {
        assert_eq!(TenantManager::prefix_table("acme", "users"), "acme::users");
        assert_eq!(
            TenantManager::prefix_table("default", "orders"),
            "default::orders"
        );
    }
}
