//! State persistence with versioned migration, debounced saving,
//! allowlist/blocklist filtering, and encrypt/decrypt hooks.

use std::collections::{HashMap, HashSet};

// ── Storage Backend ──

/// Trait for a persistence storage backend.
pub trait StorageBackend {
    fn get(&self, key: &str) -> Option<String>;
    fn set(&mut self, key: &str, value: &str);
    fn remove(&mut self, key: &str);
}

/// In-memory storage backend for testing.
#[derive(Default, Clone)]
pub struct MemoryStorage {
    data: HashMap<String, String>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dump(&self) -> &HashMap<String, String> {
        &self.data
    }
}

impl StorageBackend for MemoryStorage {
    fn get(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }

    fn set(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    fn remove(&mut self, key: &str) {
        self.data.remove(key);
    }
}

// ── Persist Config ──

/// Configuration for state persistence.
pub struct PersistConfig {
    /// Storage key prefix.
    pub key: String,
    /// Current schema version.
    pub version: u32,
    /// Allowlist of state keys to persist. If empty, persist all.
    pub allowlist: HashSet<String>,
    /// Blocklist of state keys to exclude from persistence.
    pub blocklist: HashSet<String>,
    /// Encrypt hook: applied before writing to storage.
    pub encrypt: Option<Box<dyn Fn(&str) -> String>>,
    /// Decrypt hook: applied after reading from storage.
    pub decrypt: Option<Box<dyn Fn(&str) -> String>>,
    /// Migration function: given old version, old data, returns migrated data.
    pub migrate: Option<Box<dyn Fn(u32, &str) -> String>>,
}

impl PersistConfig {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            version: 1,
            allowlist: HashSet::new(),
            blocklist: HashSet::new(),
            encrypt: None,
            decrypt: None,
            migrate: None,
        }
    }

    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    pub fn with_allowlist(mut self, keys: Vec<String>) -> Self {
        self.allowlist = keys.into_iter().collect();
        self
    }

    pub fn with_blocklist(mut self, keys: Vec<String>) -> Self {
        self.blocklist = keys.into_iter().collect();
        self
    }

    pub fn with_encrypt(mut self, f: impl Fn(&str) -> String + 'static) -> Self {
        self.encrypt = Some(Box::new(f));
        self
    }

    pub fn with_decrypt(mut self, f: impl Fn(&str) -> String + 'static) -> Self {
        self.decrypt = Some(Box::new(f));
        self
    }

    pub fn with_migrate(mut self, f: impl Fn(u32, &str) -> String + 'static) -> Self {
        self.migrate = Some(Box::new(f));
        self
    }
}

// ── Persisted Envelope ──

/// Wrapper stored in the backend that includes version metadata.
#[derive(Debug, Clone)]
struct Envelope {
    version: u32,
    data: String,
}

impl Envelope {
    fn serialize(&self) -> String {
        format!("v{}:{}", self.version, self.data)
    }

    fn deserialize(raw: &str) -> Option<Self> {
        // Format: "vN:data"
        if !raw.starts_with('v') {
            return None;
        }
        let rest = &raw[1..];
        let colon = rest.find(':')?;
        let version: u32 = rest[..colon].parse().ok()?;
        let data = rest[colon + 1..].to_string();
        Some(Envelope { version, data })
    }
}

// ── PersistStore ──

/// Manages saving and restoring state with migration, filtering, and encryption.
pub struct PersistStore<B: StorageBackend> {
    config: PersistConfig,
    backend: B,
    /// Number of save calls since last actual write (for debounce).
    pending_saves: u32,
    /// Debounce threshold: only write after this many save calls.
    debounce_threshold: u32,
    /// Accumulated state to write on flush.
    pending_data: Option<String>,
    /// Total number of actual writes.
    write_count: u64,
}

impl<B: StorageBackend> PersistStore<B> {
    pub fn new(config: PersistConfig, backend: B) -> Self {
        Self {
            config,
            backend,
            pending_saves: 0,
            debounce_threshold: 1, // default: write every time
            pending_data: None,
            write_count: 0,
        }
    }

    /// Set the debounce threshold: state is only actually written to storage
    /// after `threshold` save calls (or on explicit flush).
    pub fn with_debounce(mut self, threshold: u32) -> Self {
        self.debounce_threshold = threshold;
        self
    }

    /// Filter a JSON state object through the allowlist/blocklist.
    fn filter_state(&self, data: &str) -> String {
        if self.config.allowlist.is_empty() && self.config.blocklist.is_empty() {
            return data.to_string();
        }

        let parsed: Result<serde_json::Map<String, serde_json::Value>, _> =
            serde_json::from_str(data);

        match parsed {
            Ok(map) => {
                let filtered: serde_json::Map<String, serde_json::Value> = map
                    .into_iter()
                    .filter(|(k, _)| {
                        let allowed = self.config.allowlist.is_empty()
                            || self.config.allowlist.contains(k);
                        let not_blocked = !self.config.blocklist.contains(k);
                        allowed && not_blocked
                    })
                    .collect();
                serde_json::to_string(&filtered).unwrap_or_else(|_| data.to_string())
            }
            Err(_) => data.to_string(),
        }
    }

    /// Save state. Respects debounce threshold.
    pub fn save(&mut self, data: &str) {
        let filtered = self.filter_state(data);
        self.pending_data = Some(filtered);
        self.pending_saves += 1;

        if self.pending_saves >= self.debounce_threshold {
            self.flush();
        }
    }

    /// Force write any pending data to storage.
    pub fn flush(&mut self) {
        if let Some(data) = self.pending_data.take() {
            let encrypted = match &self.config.encrypt {
                Some(enc) => enc(&data),
                None => data,
            };
            let envelope = Envelope {
                version: self.config.version,
                data: encrypted,
            };
            self.backend.set(&self.config.key, &envelope.serialize());
            self.pending_saves = 0;
            self.write_count += 1;
        }
    }

    /// Restore state from storage, running migration if needed.
    pub fn restore(&self) -> Option<String> {
        let raw = self.backend.get(&self.config.key)?;
        let envelope = Envelope::deserialize(&raw)?;

        // Decrypt
        let data = match &self.config.decrypt {
            Some(dec) => dec(&envelope.data),
            None => envelope.data,
        };

        // Migrate if version mismatch
        if envelope.version < self.config.version {
            if let Some(migrate) = &self.config.migrate {
                return Some(migrate(envelope.version, &data));
            }
        }

        Some(data)
    }

    /// Remove persisted state.
    pub fn clear(&mut self) {
        self.backend.remove(&self.config.key);
        self.pending_data = None;
        self.pending_saves = 0;
    }

    /// Total actual writes to storage.
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Whether there is unsaved pending data.
    pub fn has_pending(&self) -> bool {
        self.pending_data.is_some()
    }

    /// Get a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a mutable reference to the backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_save_and_restore() {
        let config = PersistConfig::new("app_state");
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save(r#"{"count":42}"#);
        let restored = store.restore().unwrap();
        assert_eq!(restored, r#"{"count":42}"#);
    }

    #[test]
    fn versioned_envelope() {
        let config = PersistConfig::new("app").with_version(2);
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save(r#"{"x":1}"#);

        let raw = store.backend().get("app").unwrap();
        assert!(raw.starts_with("v2:"));
    }

    #[test]
    fn migration_on_restore() {
        let mut backend = MemoryStorage::new();
        // Simulate data stored with version 1
        backend.set("app", "v1:{\"old_field\":1}");

        let config = PersistConfig::new("app")
            .with_version(2)
            .with_migrate(|old_ver, data| {
                assert_eq!(old_ver, 1);
                data.replace("old_field", "new_field")
            });
        let store = PersistStore::new(config, backend);
        let restored = store.restore().unwrap();
        assert!(restored.contains("new_field"));
    }

    #[test]
    fn no_migration_when_current() {
        let mut backend = MemoryStorage::new();
        backend.set("app", "v2:{\"x\":1}");

        let config = PersistConfig::new("app")
            .with_version(2)
            .with_migrate(|_, _| panic!("should not migrate"));
        let store = PersistStore::new(config, backend);
        let restored = store.restore().unwrap();
        assert_eq!(restored, r#"{"x":1}"#);
    }

    #[test]
    fn debounced_save() {
        let config = PersistConfig::new("app");
        let mut store = PersistStore::new(config, MemoryStorage::new())
            .with_debounce(3);

        store.save("a");
        assert!(store.has_pending());
        assert_eq!(store.write_count(), 0);

        store.save("b");
        assert_eq!(store.write_count(), 0);

        store.save("c"); // 3rd call triggers write
        assert_eq!(store.write_count(), 1);
        assert!(!store.has_pending());
    }

    #[test]
    fn manual_flush() {
        let config = PersistConfig::new("app");
        let mut store = PersistStore::new(config, MemoryStorage::new())
            .with_debounce(100);

        store.save("data");
        assert_eq!(store.write_count(), 0);

        store.flush();
        assert_eq!(store.write_count(), 1);
    }

    #[test]
    fn allowlist_filtering() {
        let config = PersistConfig::new("app")
            .with_allowlist(vec!["name".into(), "age".into()]);
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save(r#"{"name":"Alice","age":30,"secret":"hidden"}"#);
        let restored = store.restore().unwrap();
        assert!(restored.contains("name"));
        assert!(restored.contains("age"));
        assert!(!restored.contains("secret"));
    }

    #[test]
    fn blocklist_filtering() {
        let config = PersistConfig::new("app")
            .with_blocklist(vec!["password".into()]);
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save(r#"{"user":"bob","password":"123"}"#);
        let restored = store.restore().unwrap();
        assert!(restored.contains("user"));
        assert!(!restored.contains("password"));
    }

    #[test]
    fn encrypt_decrypt_hooks() {
        let config = PersistConfig::new("app")
            .with_encrypt(|data| {
                // Simple ROT13-ish: reverse the string
                data.chars().rev().collect()
            })
            .with_decrypt(|data| {
                data.chars().rev().collect()
            });
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save("hello");

        // Stored value should be encrypted (reversed)
        let raw = store.backend().get("app").unwrap();
        assert!(raw.contains("olleh")); // reversed "hello"

        // Restore should decrypt
        let restored = store.restore().unwrap();
        assert_eq!(restored, "hello");
    }

    #[test]
    fn clear_removes_data() {
        let config = PersistConfig::new("app");
        let mut store = PersistStore::new(config, MemoryStorage::new());
        store.save("data");
        assert!(store.restore().is_some());

        store.clear();
        assert!(store.restore().is_none());
    }

    #[test]
    fn restore_nonexistent_returns_none() {
        let config = PersistConfig::new("app");
        let store = PersistStore::new(config, MemoryStorage::new());
        assert!(store.restore().is_none());
    }

    #[test]
    fn restore_malformed_returns_none() {
        let mut backend = MemoryStorage::new();
        backend.set("app", "garbage data no version prefix");
        let config = PersistConfig::new("app");
        let store = PersistStore::new(config, backend);
        assert!(store.restore().is_none());
    }

    #[test]
    fn debounce_latest_value_wins() {
        let config = PersistConfig::new("app");
        let mut store = PersistStore::new(config, MemoryStorage::new())
            .with_debounce(3);

        store.save("first");
        store.save("second");
        store.save("third"); // triggers write

        let restored = store.restore().unwrap();
        assert_eq!(restored, "third");
    }
}
