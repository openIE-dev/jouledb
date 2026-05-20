//! Key-value storage abstraction.
//!
//! Platform-agnostic trait with in-memory implementation that replaces
//! `localStorage`, `sessionStorage`, and `localForage`. Includes typed
//! serialization, key prefixing, TTL expiration, and change tracking.

use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("key not found")]
    NotFound,
    #[error("serialization error: {0}")]
    SerializationError(String),
    #[error("storage quota exceeded")]
    QuotaExceeded,
    #[error("storage backend unavailable")]
    Unavailable,
}

// ── StorageBackend trait ────────────────────────────────────────

/// Platform-agnostic key-value storage trait.
pub trait StorageBackend {
    /// Get a value by key. Returns `Ok(None)` if the key does not exist.
    fn get(&self, key: &str) -> Result<Option<String>, StorageError>;

    /// Set a key-value pair.
    fn set(&mut self, key: &str, value: &str) -> Result<(), StorageError>;

    /// Remove a key. Returns `Ok(true)` if the key existed.
    fn remove(&mut self, key: &str) -> Result<bool, StorageError>;

    /// Remove all entries.
    fn clear(&mut self) -> Result<(), StorageError>;

    /// List all keys.
    fn keys(&self) -> Result<Vec<String>, StorageError>;

    /// Number of entries.
    fn len(&self) -> Result<usize, StorageError>;

    /// Whether a key exists.
    fn contains_key(&self, key: &str) -> Result<bool, StorageError>;
}

// ── MemoryStorage ──────────────────────────────────────────────

/// In-memory storage backend backed by a `HashMap`.
pub struct MemoryStorage {
    data: HashMap<String, String>,
    max_size: Option<usize>,
}

impl MemoryStorage {
    /// Create a new unbounded in-memory store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            max_size: None,
        }
    }

    /// Create an in-memory store with a maximum entry count.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_size: Some(max_entries),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for MemoryStorage {
    fn get(&self, key: &str) -> Result<Option<String>, StorageError> {
        Ok(self.data.get(key).cloned())
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        if let Some(max) = self.max_size {
            // Only check quota if this is a new key
            if !self.data.contains_key(key) && self.data.len() >= max {
                return Err(StorageError::QuotaExceeded);
            }
        }
        self.data.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn remove(&mut self, key: &str) -> Result<bool, StorageError> {
        Ok(self.data.remove(key).is_some())
    }

    fn clear(&mut self) -> Result<(), StorageError> {
        self.data.clear();
        Ok(())
    }

    fn keys(&self) -> Result<Vec<String>, StorageError> {
        Ok(self.data.keys().cloned().collect())
    }

    fn len(&self) -> Result<usize, StorageError> {
        Ok(self.data.len())
    }

    fn contains_key(&self, key: &str) -> Result<bool, StorageError> {
        Ok(self.data.contains_key(key))
    }
}

// ── TypedStorage ───────────────────────────────────────────────

/// Wraps a `StorageBackend` with automatic JSON serialization/deserialization.
pub struct TypedStorage<B: StorageBackend> {
    backend: B,
}

impl<B: StorageBackend> TypedStorage<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Get a deserialized value by key.
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, StorageError> {
        match self.backend.get(key)? {
            Some(raw) => {
                let val = serde_json::from_str(&raw)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    /// Serialize and store a value.
    pub fn set<T: Serialize>(&mut self, key: &str, value: &T) -> Result<(), StorageError> {
        let raw = serde_json::to_string(value)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.backend.set(key, &raw)
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Result<bool, StorageError> {
        self.backend.remove(key)
    }

    /// Clear all entries.
    pub fn clear(&mut self) -> Result<(), StorageError> {
        self.backend.clear()
    }
}

// ── PrefixedStorage ────────────────────────────────────────────

/// Wraps a `StorageBackend` with automatic key prefixing for namespacing.
pub struct PrefixedStorage<B: StorageBackend> {
    backend: B,
    prefix: String,
}

impl<B: StorageBackend> PrefixedStorage<B> {
    pub fn new(backend: B, prefix: &str) -> Self {
        Self {
            backend,
            prefix: prefix.to_string(),
        }
    }

    fn prefixed_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }
}

impl<B: StorageBackend> StorageBackend for PrefixedStorage<B> {
    fn get(&self, key: &str) -> Result<Option<String>, StorageError> {
        self.backend.get(&self.prefixed_key(key))
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        self.backend.set(&self.prefixed_key(key), value)
    }

    fn remove(&mut self, key: &str) -> Result<bool, StorageError> {
        self.backend.remove(&self.prefixed_key(key))
    }

    fn clear(&mut self) -> Result<(), StorageError> {
        // Only clear keys with our prefix
        let to_remove: Vec<String> = self
            .backend
            .keys()?
            .into_iter()
            .filter(|k| k.starts_with(&self.prefix))
            .collect();
        for k in &to_remove {
            self.backend.remove(k)?;
        }
        Ok(())
    }

    fn keys(&self) -> Result<Vec<String>, StorageError> {
        Ok(self
            .backend
            .keys()?
            .into_iter()
            .filter_map(|k| k.strip_prefix(&self.prefix).map(String::from))
            .collect())
    }

    fn len(&self) -> Result<usize, StorageError> {
        Ok(self.keys()?.len())
    }

    fn contains_key(&self, key: &str) -> Result<bool, StorageError> {
        self.backend.contains_key(&self.prefixed_key(key))
    }
}

// ── ExpiringStorage ────────────────────────────────────────────

/// Internal envelope for values with expiration.
#[derive(serde::Serialize, serde::Deserialize)]
struct ExpiringEntry {
    value: String,
    expires_at: i64,
}

/// Wraps a `StorageBackend` with TTL-based expiration.
pub struct ExpiringStorage<B: StorageBackend> {
    backend: B,
}

impl<B: StorageBackend> ExpiringStorage<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Store a value with a time-to-live duration.
    pub fn set_with_ttl(
        &mut self,
        key: &str,
        value: &str,
        ttl: chrono::Duration,
    ) -> Result<(), StorageError> {
        let expires_at = Utc::now().timestamp_millis()
            + ttl.num_milliseconds();
        let entry = ExpiringEntry {
            value: value.to_string(),
            expires_at,
        };
        let raw = serde_json::to_string(&entry)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.backend.set(key, &raw)
    }

    /// Get a value, returning `None` if expired (and removing it).
    pub fn get(&mut self, key: &str) -> Result<Option<String>, StorageError> {
        match self.backend.get(key)? {
            Some(raw) => {
                let entry: ExpiringEntry = serde_json::from_str(&raw)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                if Utc::now().timestamp_millis() > entry.expires_at {
                    self.backend.remove(key)?;
                    Ok(None)
                } else {
                    Ok(Some(entry.value))
                }
            }
            None => Ok(None),
        }
    }

    /// Remove all expired entries. Returns the number removed.
    pub fn cleanup(&mut self) -> Result<usize, StorageError> {
        let now = Utc::now().timestamp_millis();
        let all_keys = self.backend.keys()?;
        let mut removed = 0;
        for key in all_keys {
            if let Some(raw) = self.backend.get(&key)? {
                if let Ok(entry) = serde_json::from_str::<ExpiringEntry>(&raw) {
                    if now > entry.expires_at {
                        self.backend.remove(&key)?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }
}

// ── StorageEvent ───────────────────────────────────────────────

/// Record of a mutation to a storage backend.
#[derive(Debug, Clone)]
pub struct StorageEvent {
    pub key: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub timestamp: DateTime<Utc>,
}

// ── ObservableStorage ──────────────────────────────────────────

/// Wraps a `StorageBackend` with change-event tracking.
pub struct ObservableStorage<B: StorageBackend> {
    backend: B,
    events: Vec<StorageEvent>,
}

impl<B: StorageBackend> ObservableStorage<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            events: Vec::new(),
        }
    }

    /// All recorded events since last clear.
    pub fn events(&self) -> &[StorageEvent] {
        &self.events
    }

    /// Discard all recorded events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }
}

impl<B: StorageBackend> StorageBackend for ObservableStorage<B> {
    fn get(&self, key: &str) -> Result<Option<String>, StorageError> {
        self.backend.get(key)
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        let old_value = self.backend.get(key)?;
        self.backend.set(key, value)?;
        self.events.push(StorageEvent {
            key: key.to_string(),
            old_value,
            new_value: Some(value.to_string()),
            timestamp: Utc::now(),
        });
        Ok(())
    }

    fn remove(&mut self, key: &str) -> Result<bool, StorageError> {
        let old_value = self.backend.get(key)?;
        let existed = self.backend.remove(key)?;
        if existed {
            self.events.push(StorageEvent {
                key: key.to_string(),
                old_value,
                new_value: None,
                timestamp: Utc::now(),
            });
        }
        Ok(existed)
    }

    fn clear(&mut self) -> Result<(), StorageError> {
        // Log an event for each key being cleared
        let all_keys = self.backend.keys()?;
        for key in &all_keys {
            let old_value = self.backend.get(key)?;
            self.events.push(StorageEvent {
                key: key.clone(),
                old_value,
                new_value: None,
                timestamp: Utc::now(),
            });
        }
        self.backend.clear()
    }

    fn keys(&self) -> Result<Vec<String>, StorageError> {
        self.backend.keys()
    }

    fn len(&self) -> Result<usize, StorageError> {
        self.backend.len()
    }

    fn contains_key(&self, key: &str) -> Result<bool, StorageError> {
        self.backend.contains_key(key)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryStorage CRUD ──

    #[test]
    fn memory_storage_set_and_get() {
        let mut store = MemoryStorage::new();
        store.set("key1", "value1").unwrap();
        assert_eq!(store.get("key1").unwrap(), Some("value1".to_string()));
    }

    #[test]
    fn memory_storage_get_missing_returns_none() {
        let store = MemoryStorage::new();
        assert_eq!(store.get("nonexistent").unwrap(), None);
    }

    #[test]
    fn memory_storage_remove() {
        let mut store = MemoryStorage::new();
        store.set("key1", "value1").unwrap();
        assert!(store.remove("key1").unwrap());
        assert_eq!(store.get("key1").unwrap(), None);
        // Removing again returns false
        assert!(!store.remove("key1").unwrap());
    }

    #[test]
    fn memory_storage_clear() {
        let mut store = MemoryStorage::new();
        store.set("a", "1").unwrap();
        store.set("b", "2").unwrap();
        store.clear().unwrap();
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn memory_storage_keys() {
        let mut store = MemoryStorage::new();
        store.set("x", "1").unwrap();
        store.set("y", "2").unwrap();
        let mut keys = store.keys().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn memory_storage_contains_key() {
        let mut store = MemoryStorage::new();
        store.set("present", "yes").unwrap();
        assert!(store.contains_key("present").unwrap());
        assert!(!store.contains_key("absent").unwrap());
    }

    #[test]
    fn memory_storage_len() {
        let mut store = MemoryStorage::new();
        assert_eq!(store.len().unwrap(), 0);
        store.set("a", "1").unwrap();
        assert_eq!(store.len().unwrap(), 1);
        store.set("b", "2").unwrap();
        assert_eq!(store.len().unwrap(), 2);
        store.remove("a").unwrap();
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn memory_storage_set_overwrites() {
        let mut store = MemoryStorage::new();
        store.set("k", "v1").unwrap();
        store.set("k", "v2").unwrap();
        assert_eq!(store.get("k").unwrap(), Some("v2".to_string()));
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn memory_storage_quota_exceeded() {
        let mut store = MemoryStorage::with_capacity(2);
        store.set("a", "1").unwrap();
        store.set("b", "2").unwrap();
        let result = store.set("c", "3");
        assert!(matches!(result, Err(StorageError::QuotaExceeded)));
        // Overwriting existing key should still work
        store.set("a", "updated").unwrap();
        assert_eq!(store.get("a").unwrap(), Some("updated".to_string()));
    }

    // ── TypedStorage ──

    #[test]
    fn typed_storage_serialize_deserialize() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct User {
            name: String,
            age: u32,
        }

        let mut store = TypedStorage::new(MemoryStorage::new());
        let user = User {
            name: "Alice".to_string(),
            age: 30,
        };
        store.set("user", &user).unwrap();
        let loaded: User = store.get("user").unwrap().unwrap();
        assert_eq!(loaded, user);
    }

    #[test]
    fn typed_storage_get_missing() {
        let store = TypedStorage::new(MemoryStorage::new());
        let result: Option<i32> = store.get("missing").unwrap();
        assert_eq!(result, None);
    }

    // ── PrefixedStorage ──

    #[test]
    fn prefixed_keys_isolated() {
        let mut backing = MemoryStorage::new();
        backing.set("other_key", "other").unwrap();

        let mut prefixed = PrefixedStorage::new(backing, "app:");
        prefixed.set("name", "value").unwrap();

        // The prefixed storage should see only its own key
        let keys = prefixed.keys().unwrap();
        assert_eq!(keys, vec!["name".to_string()]);

        // The underlying key is "app:name"
        assert!(prefixed.contains_key("name").unwrap());
        assert_eq!(
            prefixed.get("name").unwrap(),
            Some("value".to_string())
        );
    }

    #[test]
    fn prefixed_keys_returns_unprefixed() {
        let mut prefixed = PrefixedStorage::new(MemoryStorage::new(), "ns:");
        prefixed.set("alpha", "1").unwrap();
        prefixed.set("beta", "2").unwrap();
        let mut keys = prefixed.keys().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["alpha".to_string(), "beta".to_string()]);
    }

    // ── ExpiringStorage ──

    #[test]
    fn expiring_storage_returns_none_after_ttl() {
        let mut store = ExpiringStorage::new(MemoryStorage::new());
        // Set with a TTL of -1 second (already expired)
        let ttl = chrono::Duration::milliseconds(-1);
        store.set_with_ttl("temp", "data", ttl).unwrap();
        assert_eq!(store.get("temp").unwrap(), None);
    }

    #[test]
    fn expiring_storage_returns_value_before_ttl() {
        let mut store = ExpiringStorage::new(MemoryStorage::new());
        let ttl = chrono::Duration::hours(1);
        store.set_with_ttl("temp", "data", ttl).unwrap();
        assert_eq!(store.get("temp").unwrap(), Some("data".to_string()));
    }

    #[test]
    fn expiring_cleanup_removes_expired() {
        let mut store = ExpiringStorage::new(MemoryStorage::new());
        // One expired, one valid
        store
            .set_with_ttl("old", "stale", chrono::Duration::milliseconds(-1))
            .unwrap();
        store
            .set_with_ttl("fresh", "new", chrono::Duration::hours(1))
            .unwrap();
        let removed = store.cleanup().unwrap();
        assert_eq!(removed, 1);
        // "fresh" should still be accessible
        assert_eq!(store.get("fresh").unwrap(), Some("new".to_string()));
    }

    // ── ObservableStorage ──

    #[test]
    fn observable_logs_set_events() {
        let mut store = ObservableStorage::new(MemoryStorage::new());
        store.set("k", "v1").unwrap();
        assert_eq!(store.events().len(), 1);
        assert_eq!(store.events()[0].key, "k");
        assert_eq!(store.events()[0].old_value, None);
        assert_eq!(store.events()[0].new_value, Some("v1".to_string()));
    }

    #[test]
    fn observable_logs_set_and_remove() {
        let mut store = ObservableStorage::new(MemoryStorage::new());
        store.set("k", "v1").unwrap();
        store.remove("k").unwrap();
        assert_eq!(store.events().len(), 2);

        // First event: set
        assert_eq!(store.events()[0].new_value, Some("v1".to_string()));
        // Second event: remove
        assert_eq!(store.events()[1].key, "k");
        assert_eq!(store.events()[1].old_value, Some("v1".to_string()));
        assert_eq!(store.events()[1].new_value, None);
    }

    #[test]
    fn observable_clear_events() {
        let mut store = ObservableStorage::new(MemoryStorage::new());
        store.set("k", "v").unwrap();
        assert_eq!(store.events().len(), 1);
        store.clear_events();
        assert!(store.events().is_empty());
    }

    #[test]
    fn observable_set_overwrite_logs_old_value() {
        let mut store = ObservableStorage::new(MemoryStorage::new());
        store.set("k", "v1").unwrap();
        store.set("k", "v2").unwrap();
        assert_eq!(store.events().len(), 2);
        assert_eq!(store.events()[1].old_value, Some("v1".to_string()));
        assert_eq!(store.events()[1].new_value, Some("v2".to_string()));
    }
}
