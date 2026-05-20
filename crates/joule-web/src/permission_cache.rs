//! Permission caching — permission resolution cache with TTL, invalidation on
//! role change, hierarchical cache, bulk permission check, and cache warming.
//!
//! Replaces per-request permission lookups with an in-memory cache that
//! drastically reduces RBAC evaluation overhead for repeated checks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Permission cache errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Entry not found in cache.
    NotFound(String),
    /// Entry has expired.
    Expired(String),
    /// Invalid cache key format.
    InvalidKey(String),
    /// Cache is full and eviction failed.
    CacheFull { capacity: usize },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "cache entry not found: {key}"),
            Self::Expired(key) => write!(f, "cache entry expired: {key}"),
            Self::InvalidKey(key) => write!(f, "invalid cache key: {key}"),
            Self::CacheFull { capacity } => write!(f, "cache full, capacity={capacity}"),
        }
    }
}

impl std::error::Error for CacheError {}

// ── Types ──────────────────────────────────────────────────────

/// A cached permission decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedDecision {
    pub allowed: bool,
    pub via_role: Option<String>,
    pub matching_permission: Option<String>,
    pub inherited: bool,
}

/// An entry in the permission cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    decision: CachedDecision,
    created_at_secs: u64,
    ttl_secs: u64,
    access_count: u64,
    last_access_secs: u64,
}

impl CacheEntry {
    fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.created_at_secs + self.ttl_secs
    }
}

/// Cache key: (subject_id, resource, action).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    subject_id: String,
    resource: String,
    action: String,
}

impl CacheKey {
    fn new(subject_id: &str, resource: &str, action: &str) -> Self {
        Self {
            subject_id: subject_id.to_string(),
            resource: resource.to_string(),
            action: action.to_string(),
        }
    }

    fn as_string(&self) -> String {
        format!("{}:{}:{}", self.subject_id, self.resource, self.action)
    }
}

/// Statistics about cache performance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub entries: usize,
}

impl CacheStats {
    /// Hit rate as a ratio (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// Configuration for the permission cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionCacheConfig {
    /// Default TTL for cache entries.
    pub default_ttl_secs: u64,
    /// Maximum number of entries.
    pub max_entries: usize,
    /// Whether to track hierarchical invalidation.
    pub hierarchical: bool,
}

impl Default for PermissionCacheConfig {
    fn default() -> Self {
        Self {
            default_ttl_secs: 300,
            max_entries: 10000,
            hierarchical: true,
        }
    }
}

// ── Engine ─────────────────────────────────────────────────────

/// The permission cache.
#[derive(Debug, Clone)]
pub struct PermissionCache {
    config: PermissionCacheConfig,
    entries: HashMap<CacheKey, CacheEntry>,
    stats: CacheStats,
    /// Subject -> roles mapping for hierarchical invalidation.
    subject_roles: HashMap<String, Vec<String>>,
    /// Role -> set of subjects that have this role.
    role_subjects: HashMap<String, Vec<String>>,
}

impl PermissionCache {
    pub fn new(config: PermissionCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: CacheStats::default(),
            subject_roles: HashMap::new(),
            role_subjects: HashMap::new(),
        }
    }

    /// Register a subject's roles for hierarchical invalidation.
    pub fn register_subject_roles(&mut self, subject_id: &str, roles: Vec<String>) {
        // Remove old role->subject mappings.
        if let Some(old_roles) = self.subject_roles.get(subject_id) {
            for role in old_roles {
                if let Some(subjects) = self.role_subjects.get_mut(role) {
                    subjects.retain(|s| s != subject_id);
                }
            }
        }

        // Set new mappings.
        for role in &roles {
            self.role_subjects
                .entry(role.clone())
                .or_default()
                .push(subject_id.to_string());
        }
        self.subject_roles
            .insert(subject_id.to_string(), roles);
    }

    /// Look up a cached permission decision.
    pub fn get(
        &mut self,
        subject_id: &str,
        resource: &str,
        action: &str,
        now_secs: u64,
    ) -> Result<CachedDecision, CacheError> {
        let key = CacheKey::new(subject_id, resource, action);

        let entry = self
            .entries
            .get_mut(&key)
            .ok_or_else(|| {
                self.stats.misses += 1;
                CacheError::NotFound(key.as_string())
            })?;

        if entry.is_expired(now_secs) {
            let key_str = key.as_string();
            self.entries.remove(&key);
            self.stats.misses += 1;
            return Err(CacheError::Expired(key_str));
        }

        entry.access_count += 1;
        entry.last_access_secs = now_secs;
        self.stats.hits += 1;

        Ok(entry.decision.clone())
    }

    /// Store a permission decision in the cache.
    pub fn put(
        &mut self,
        subject_id: &str,
        resource: &str,
        action: &str,
        decision: CachedDecision,
        now_secs: u64,
    ) -> Result<(), CacheError> {
        self.put_with_ttl(subject_id, resource, action, decision, now_secs, self.config.default_ttl_secs)
    }

    /// Store with a custom TTL.
    pub fn put_with_ttl(
        &mut self,
        subject_id: &str,
        resource: &str,
        action: &str,
        decision: CachedDecision,
        now_secs: u64,
        ttl_secs: u64,
    ) -> Result<(), CacheError> {
        // Evict if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired(now_secs);
            if self.entries.len() >= self.config.max_entries {
                self.evict_lru();
                if self.entries.len() >= self.config.max_entries {
                    return Err(CacheError::CacheFull {
                        capacity: self.config.max_entries,
                    });
                }
            }
        }

        let key = CacheKey::new(subject_id, resource, action);
        let entry = CacheEntry {
            decision,
            created_at_secs: now_secs,
            ttl_secs,
            access_count: 0,
            last_access_secs: now_secs,
        };

        self.entries.insert(key, entry);
        self.stats.entries = self.entries.len();
        Ok(())
    }

    /// Evict all expired entries.
    fn evict_expired(&mut self, now_secs: u64) {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired(now_secs));
        let evicted = before - self.entries.len();
        self.stats.evictions += evicted as u64;
        self.stats.entries = self.entries.len();
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access_secs)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&lru_key);
            self.stats.evictions += 1;
            self.stats.entries = self.entries.len();
        }
    }

    /// Invalidate all cached entries for a subject.
    pub fn invalidate_subject(&mut self, subject_id: &str) {
        let before = self.entries.len();
        self.entries
            .retain(|key, _| key.subject_id != subject_id);
        let removed = before - self.entries.len();
        self.stats.invalidations += removed as u64;
        self.stats.entries = self.entries.len();
    }

    /// Invalidate all entries for subjects that have a given role.
    /// Used when a role's permissions change.
    pub fn invalidate_role(&mut self, role_id: &str) {
        if !self.config.hierarchical {
            return;
        }

        let affected_subjects: Vec<String> = self
            .role_subjects
            .get(role_id)
            .cloned()
            .unwrap_or_default();

        for subject_id in &affected_subjects {
            self.invalidate_subject(subject_id);
        }
    }

    /// Invalidate entries for a specific resource.
    pub fn invalidate_resource(&mut self, resource: &str) {
        let before = self.entries.len();
        self.entries.retain(|key, _| key.resource != resource);
        let removed = before - self.entries.len();
        self.stats.invalidations += removed as u64;
        self.stats.entries = self.entries.len();
    }

    /// Bulk permission check. Returns cached results for each (resource, action).
    /// Returns None for entries not in cache.
    pub fn bulk_get(
        &mut self,
        subject_id: &str,
        checks: &[(String, String)],
        now_secs: u64,
    ) -> Vec<Option<CachedDecision>> {
        checks
            .iter()
            .map(|(resource, action)| self.get(subject_id, resource, action, now_secs).ok())
            .collect()
    }

    /// Warm the cache with a set of pre-computed decisions.
    pub fn warm(
        &mut self,
        entries: Vec<(String, String, String, CachedDecision)>,
        now_secs: u64,
    ) -> usize {
        let mut count = 0;
        for (subject_id, resource, action, decision) in entries {
            if self
                .put(&subject_id, &resource, &action, decision, now_secs)
                .is_ok()
            {
                count += 1;
            }
        }
        count
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        let removed = self.entries.len();
        self.entries.clear();
        self.stats.invalidations += removed as u64;
        self.stats.entries = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether cache is at capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.config.max_entries
    }
}

impl Default for PermissionCache {
    fn default() -> Self {
        Self::new(PermissionCacheConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> PermissionCacheConfig {
        PermissionCacheConfig {
            default_ttl_secs: 60,
            max_entries: 5,
            hierarchical: true,
        }
    }

    fn decision(allowed: bool) -> CachedDecision {
        CachedDecision {
            allowed,
            via_role: Some("role-1".to_string()),
            matching_permission: Some("docs:read".to_string()),
            inherited: false,
        }
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        let result = cache.get("alice", "docs", "read", 1010).unwrap();
        assert!(result.allowed);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = PermissionCache::new(small_config());
        let err = cache.get("alice", "docs", "read", 1000).unwrap_err();
        assert!(matches!(err, CacheError::NotFound(_)));
    }

    #[test]
    fn test_cache_expiry() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        // After TTL
        let err = cache.get("alice", "docs", "read", 1061).unwrap_err();
        assert!(matches!(err, CacheError::Expired(_)));
    }

    #[test]
    fn test_custom_ttl() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put_with_ttl("alice", "docs", "read", decision(true), 1000, 10)
            .unwrap();

        assert!(cache.get("alice", "docs", "read", 1005).is_ok());
        assert!(cache.get("alice", "docs", "read", 1011).is_err());
    }

    #[test]
    fn test_invalidate_subject() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("alice", "users", "list", decision(true), 1000)
            .unwrap();
        cache
            .put("bob", "docs", "read", decision(true), 1000)
            .unwrap();

        cache.invalidate_subject("alice");
        assert!(cache.get("alice", "docs", "read", 1010).is_err());
        assert!(cache.get("alice", "users", "list", 1010).is_err());
        assert!(cache.get("bob", "docs", "read", 1010).is_ok());
    }

    #[test]
    fn test_invalidate_role() {
        let mut cache = PermissionCache::new(small_config());
        cache.register_subject_roles("alice", vec!["editor".to_string()]);
        cache.register_subject_roles("bob", vec!["editor".to_string()]);
        cache.register_subject_roles("charlie", vec!["viewer".to_string()]);

        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("bob", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("charlie", "docs", "read", decision(true), 1000)
            .unwrap();

        cache.invalidate_role("editor");

        // Alice and Bob (editors) should be invalidated.
        assert!(cache.get("alice", "docs", "read", 1010).is_err());
        assert!(cache.get("bob", "docs", "read", 1010).is_err());
        // Charlie (viewer) should still be cached.
        assert!(cache.get("charlie", "docs", "read", 1010).is_ok());
    }

    #[test]
    fn test_invalidate_resource() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("alice", "users", "list", decision(true), 1000)
            .unwrap();

        cache.invalidate_resource("docs");
        assert!(cache.get("alice", "docs", "read", 1010).is_err());
        assert!(cache.get("alice", "users", "list", 1010).is_ok());
    }

    #[test]
    fn test_eviction_on_full() {
        let mut cache = PermissionCache::new(small_config()); // max 5

        for i in 0..5 {
            let subj = format!("u{i}");
            cache
                .put(&subj, "docs", "read", decision(true), 1000 + i)
                .unwrap();
        }
        assert_eq!(cache.len(), 5);

        // 6th entry should trigger LRU eviction.
        cache
            .put("u99", "docs", "read", decision(true), 1010)
            .unwrap();
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_eviction_expired_first() {
        let mut cache = PermissionCache::new(small_config());

        // Fill with entries that will expire soon.
        for i in 0..5 {
            let subj = format!("u{i}");
            cache
                .put_with_ttl(&subj, "docs", "read", decision(true), 1000, 5)
                .unwrap();
        }

        // After TTL, new entry should succeed by evicting expired.
        cache
            .put("u99", "docs", "read", decision(true), 1010)
            .unwrap();
        assert!(cache.len() <= 5);
    }

    #[test]
    fn test_bulk_get() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("alice", "users", "list", decision(false), 1000)
            .unwrap();

        let checks = vec![
            ("docs".to_string(), "read".to_string()),
            ("users".to_string(), "list".to_string()),
            ("admin".to_string(), "manage".to_string()),
        ];
        let results = cache.bulk_get("alice", &checks, 1010);
        assert!(results[0].as_ref().unwrap().allowed);
        assert!(!results[1].as_ref().unwrap().allowed);
        assert!(results[2].is_none());
    }

    #[test]
    fn test_warm_cache() {
        let mut cache = PermissionCache::new(small_config());
        let entries = vec![
            ("alice".to_string(), "docs".to_string(), "read".to_string(), decision(true)),
            ("bob".to_string(), "docs".to_string(), "write".to_string(), decision(false)),
        ];

        let warmed = cache.warm(entries, 1000);
        assert_eq!(warmed, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_stats() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        // Hit
        let _ = cache.get("alice", "docs", "read", 1010);
        // Miss
        let _ = cache.get("alice", "docs", "write", 1010);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_zero_total() {
        let stats = CacheStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_full() {
        let mut cache = PermissionCache::new(small_config());
        assert!(!cache.is_full());

        for i in 0..5 {
            cache
                .put(&format!("u{i}"), "docs", "read", decision(true), 1000)
                .unwrap();
        }
        assert!(cache.is_full());
    }

    #[test]
    fn test_default_cache() {
        let cache = PermissionCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_register_subject_roles_update() {
        let mut cache = PermissionCache::new(small_config());
        cache.register_subject_roles("alice", vec!["editor".to_string()]);
        cache.register_subject_roles("alice", vec!["viewer".to_string()]);

        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        // Invalidating old role should not affect alice.
        cache.invalidate_role("editor");
        assert!(cache.get("alice", "docs", "read", 1010).is_ok());

        // Invalidating new role should affect alice.
        cache.invalidate_role("viewer");
        assert!(cache.get("alice", "docs", "read", 1010).is_err());
    }

    #[test]
    fn test_hierarchical_disabled() {
        let mut config = small_config();
        config.hierarchical = false;
        let mut cache = PermissionCache::new(config);

        cache.register_subject_roles("alice", vec!["editor".to_string()]);
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        // Role invalidation should be a no-op when hierarchical is off.
        cache.invalidate_role("editor");
        assert!(cache.get("alice", "docs", "read", 1010).is_ok());
    }

    #[test]
    fn test_access_count_increments() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();

        for t in 0..5 {
            let _ = cache.get("alice", "docs", "read", 1000 + t);
        }

        let key = CacheKey::new("alice", "docs", "read");
        let entry = cache.entries.get(&key).unwrap();
        assert_eq!(entry.access_count, 5);
    }

    #[test]
    fn test_error_display() {
        let e = CacheError::CacheFull { capacity: 100 };
        assert_eq!(e.to_string(), "cache full, capacity=100");
    }

    #[test]
    fn test_invalidations_stat() {
        let mut cache = PermissionCache::new(small_config());
        cache
            .put("alice", "docs", "read", decision(true), 1000)
            .unwrap();
        cache
            .put("alice", "docs", "write", decision(true), 1000)
            .unwrap();

        cache.invalidate_subject("alice");
        assert_eq!(cache.stats().invalidations, 2);
    }
}
