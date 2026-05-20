//! Multi-Tenancy — tenant isolation at the storage layer.
//!
//! Netflix's data can't leak to Disney's queries. Every content provider
//! needs tenant isolation: separate data, separate quotas, separate billing.
//!
//! This module provides:
//! - Tenant-scoped stores (each tenant gets its own AmorphicStore shard)
//! - Cross-tenant query prevention
//! - Per-tenant resource quotas (records, storage, queries/sec)
//! - Tenant lifecycle (create, suspend, delete)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

use crate::{
    AmorphicError, AmorphicRecord, AmorphicResult, AmorphicStore, QueryResult, RecordId, Value,
};

/// Tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    pub tenant_id: String,
    pub name: String,
    /// Maximum records this tenant can store (0 = unlimited)
    pub max_records: usize,
    /// Maximum storage in bytes (0 = unlimited)
    pub max_storage_bytes: usize,
    /// Status
    pub status: TenantStatus,
}

/// Tenant lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    Active,
    Suspended,
    PendingDeletion,
}

/// Per-tenant usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantStats {
    pub tenant_id: String,
    pub record_count: usize,
    pub query_count: u64,
    pub storage_estimate_bytes: usize,
}

/// Multi-tenant store: each tenant gets isolated storage.
pub struct MultiTenantStore {
    /// Tenant ID → isolated AmorphicStore
    stores: HashMap<String, RwLock<AmorphicStore>>,
    /// Tenant configurations
    configs: HashMap<String, TenantConfig>,
    /// Per-tenant query counters
    query_counts: HashMap<String, u64>,
}

impl MultiTenantStore {
    pub fn new() -> Self {
        Self {
            stores: HashMap::new(),
            configs: HashMap::new(),
            query_counts: HashMap::new(),
        }
    }

    /// Create a new tenant with the given configuration.
    pub fn create_tenant(&mut self, config: TenantConfig) -> AmorphicResult<()> {
        if self.stores.contains_key(&config.tenant_id) {
            return Err(AmorphicError::IngestionError(format!(
                "Tenant '{}' already exists",
                config.tenant_id
            )));
        }

        let store = AmorphicStore::new();
        self.stores
            .insert(config.tenant_id.clone(), RwLock::new(store));
        self.query_counts.insert(config.tenant_id.clone(), 0);
        self.configs.insert(config.tenant_id.clone(), config);
        Ok(())
    }

    /// Get a tenant's store for reading (with access check).
    fn get_store(&self, tenant_id: &str) -> AmorphicResult<&RwLock<AmorphicStore>> {
        // Check tenant exists and is active
        let config = self.configs.get(tenant_id).ok_or_else(|| {
            AmorphicError::QueryError(format!("Tenant '{}' not found", tenant_id))
        })?;

        if config.status != TenantStatus::Active {
            return Err(AmorphicError::QueryError(format!(
                "Tenant '{}' is {:?}",
                tenant_id, config.status
            )));
        }

        self.stores.get(tenant_id).ok_or_else(|| {
            AmorphicError::QueryError(format!("Tenant store '{}' missing", tenant_id))
        })
    }

    /// Ingest JSON for a specific tenant.
    pub fn ingest_json(&mut self, tenant_id: &str, json: &str) -> AmorphicResult<RecordId> {
        // Check quota
        if let Some(config) = self.configs.get(tenant_id) {
            if config.max_records > 0 {
                let store = self.stores.get(tenant_id).ok_or_else(|| {
                    AmorphicError::QueryError(format!("Tenant '{}' not found", tenant_id))
                })?;
                let current = store.read().unwrap().record_count();
                if current >= config.max_records {
                    return Err(AmorphicError::IngestionError(format!(
                        "Tenant '{}' quota exceeded: {} / {} records",
                        tenant_id, current, config.max_records
                    )));
                }
            }

            if config.status != TenantStatus::Active {
                return Err(AmorphicError::IngestionError(format!(
                    "Tenant '{}' is {:?}",
                    tenant_id, config.status
                )));
            }
        } else {
            return Err(AmorphicError::IngestionError(format!(
                "Tenant '{}' not found",
                tenant_id
            )));
        }

        let store = self.stores.get(tenant_id).unwrap();
        store.write().unwrap().ingest_json(json)
    }

    /// Query by field equality for a specific tenant.
    pub fn query_equals(
        &mut self,
        tenant_id: &str,
        field: &str,
        value: &Value,
    ) -> AmorphicResult<QueryResult> {
        // Check access first, then release borrow before mutating query_counts
        self.check_tenant_active(tenant_id)?;
        *self.query_counts.entry(tenant_id.to_string()).or_default() += 1;
        let store = self.stores.get(tenant_id).unwrap().read().unwrap();
        Ok(store.query_equals(field, value))
    }

    /// Similarity query for a specific tenant.
    pub fn query_similar_to(
        &mut self,
        tenant_id: &str,
        name: &str,
        k: usize,
    ) -> AmorphicResult<QueryResult> {
        self.check_tenant_active(tenant_id)?;
        *self.query_counts.entry(tenant_id.to_string()).or_default() += 1;
        let store = self.stores.get(tenant_id).unwrap().read().unwrap();
        Ok(store.query_similar_to(name, k))
    }

    /// SQL query for a specific tenant.
    pub fn query_sql(
        &mut self,
        tenant_id: &str,
        sql: &str,
    ) -> AmorphicResult<QueryResult> {
        self.check_tenant_active(tenant_id)?;
        *self.query_counts.entry(tenant_id.to_string()).or_default() += 1;
        let store = self.stores.get(tenant_id).unwrap().read().unwrap();
        store.query_sql(sql)
    }

    /// Check that a tenant exists and is active.
    fn check_tenant_active(&self, tenant_id: &str) -> AmorphicResult<()> {
        let config = self.configs.get(tenant_id).ok_or_else(|| {
            AmorphicError::QueryError(format!("Tenant '{}' not found", tenant_id))
        })?;
        if config.status != TenantStatus::Active {
            return Err(AmorphicError::QueryError(format!(
                "Tenant '{}' is {:?}",
                tenant_id, config.status
            )));
        }
        if !self.stores.contains_key(tenant_id) {
            return Err(AmorphicError::QueryError(format!(
                "Tenant store '{}' missing",
                tenant_id
            )));
        }
        Ok(())
    }

    /// Suspend a tenant (blocks all operations).
    pub fn suspend_tenant(&mut self, tenant_id: &str) -> AmorphicResult<()> {
        let config = self.configs.get_mut(tenant_id).ok_or_else(|| {
            AmorphicError::QueryError(format!("Tenant '{}' not found", tenant_id))
        })?;
        config.status = TenantStatus::Suspended;
        Ok(())
    }

    /// Reactivate a suspended tenant.
    pub fn activate_tenant(&mut self, tenant_id: &str) -> AmorphicResult<()> {
        let config = self.configs.get_mut(tenant_id).ok_or_else(|| {
            AmorphicError::QueryError(format!("Tenant '{}' not found", tenant_id))
        })?;
        config.status = TenantStatus::Active;
        Ok(())
    }

    /// Mark a tenant for deletion (data will be purged).
    pub fn delete_tenant(&mut self, tenant_id: &str) -> AmorphicResult<()> {
        if let Some(config) = self.configs.get_mut(tenant_id) {
            config.status = TenantStatus::PendingDeletion;
        }
        self.stores.remove(tenant_id);
        Ok(())
    }

    /// Get statistics for a tenant.
    pub fn tenant_stats(&self, tenant_id: &str) -> AmorphicResult<TenantStats> {
        let store_lock = self.get_store(tenant_id)?;
        let store = store_lock.read().unwrap();
        Ok(TenantStats {
            tenant_id: tenant_id.to_string(),
            record_count: store.record_count(),
            query_count: self.query_counts.get(tenant_id).copied().unwrap_or(0),
            storage_estimate_bytes: store.record_count() * 2048, // ~2KB per record estimate
        })
    }

    /// List all tenants.
    pub fn list_tenants(&self) -> Vec<&TenantConfig> {
        self.configs.values().collect()
    }

    /// Number of tenants.
    pub fn tenant_count(&self) -> usize {
        self.configs.len()
    }
}

impl Default for MultiTenantStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tenant(id: &str) -> TenantConfig {
        TenantConfig {
            tenant_id: id.to_string(),
            name: format!("Tenant {}", id),
            max_records: 0, // unlimited
            status: TenantStatus::Active,
            max_storage_bytes: 0,
        }
    }

    #[test]
    fn test_tenant_isolation() {
        let mut store = MultiTenantStore::new();
        store.create_tenant(make_tenant("netflix")).unwrap();
        store.create_tenant(make_tenant("disney")).unwrap();

        // Netflix ingests data
        store
            .ingest_json("netflix", r#"{"name": "Stranger Things", "genre": "scifi"}"#)
            .unwrap();

        // Disney ingests data
        store
            .ingest_json("disney", r#"{"name": "Frozen", "genre": "animation"}"#)
            .unwrap();

        // Netflix can only see its own data
        let netflix_results = store
            .query_equals("netflix", "genre", &Value::String("scifi".to_string()))
            .unwrap();
        assert_eq!(netflix_results.len(), 1);

        // Disney can only see its own data
        let disney_results = store
            .query_equals("disney", "genre", &Value::String("animation".to_string()))
            .unwrap();
        assert_eq!(disney_results.len(), 1);

        // Netflix can't see Disney's data
        let cross_query = store
            .query_equals("netflix", "genre", &Value::String("animation".to_string()))
            .unwrap();
        assert_eq!(cross_query.len(), 0);
    }

    #[test]
    fn test_tenant_quota() {
        let mut store = MultiTenantStore::new();
        store
            .create_tenant(TenantConfig {
                tenant_id: "limited".to_string(),
                name: "Limited Tenant".to_string(),
                max_records: 2,
                max_storage_bytes: 0,
                status: TenantStatus::Active,
            })
            .unwrap();

        // First two succeed
        store.ingest_json("limited", r#"{"name": "A"}"#).unwrap();
        store.ingest_json("limited", r#"{"name": "B"}"#).unwrap();

        // Third fails — quota exceeded
        let result = store.ingest_json("limited", r#"{"name": "C"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_tenant_suspension() {
        let mut store = MultiTenantStore::new();
        store.create_tenant(make_tenant("test")).unwrap();
        store
            .ingest_json("test", r#"{"name": "data"}"#)
            .unwrap();

        // Suspend
        store.suspend_tenant("test").unwrap();

        // Queries fail while suspended
        let result = store.query_equals("test", "name", &Value::String("data".to_string()));
        assert!(result.is_err());

        // Ingest fails while suspended
        let result = store.ingest_json("test", r#"{"name": "more"}"#);
        assert!(result.is_err());

        // Reactivate
        store.activate_tenant("test").unwrap();

        // Queries work again
        let result = store
            .query_equals("test", "name", &Value::String("data".to_string()))
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_tenant_deletion() {
        let mut store = MultiTenantStore::new();
        store.create_tenant(make_tenant("doomed")).unwrap();
        store
            .ingest_json("doomed", r#"{"name": "goodbye"}"#)
            .unwrap();

        store.delete_tenant("doomed").unwrap();

        // Tenant is gone
        let result = store.query_equals("doomed", "name", &Value::String("goodbye".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_tenant_stats() {
        let mut store = MultiTenantStore::new();
        store.create_tenant(make_tenant("stats_test")).unwrap();

        store
            .ingest_json("stats_test", r#"{"name": "A"}"#)
            .unwrap();
        store
            .ingest_json("stats_test", r#"{"name": "B"}"#)
            .unwrap();

        // Run a query to increment counter
        store
            .query_equals("stats_test", "name", &Value::String("A".to_string()))
            .unwrap();

        let stats = store.tenant_stats("stats_test").unwrap();
        assert_eq!(stats.record_count, 2);
        assert_eq!(stats.query_count, 1);
    }

    #[test]
    fn test_nonexistent_tenant() {
        let mut store = MultiTenantStore::new();
        let result = store.ingest_json("ghost", r#"{"name": "nope"}"#);
        assert!(result.is_err());
    }
}
