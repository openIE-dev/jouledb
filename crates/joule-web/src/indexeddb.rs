//! Headless IndexedDB state model.
//!
//! Provides an in-memory abstraction of the IndexedDB API for testing and
//! server-side rendering without any browser dependency.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IndexedDbError {
    #[error("store not found: {0}")]
    StoreNotFound(String),
    #[error("key not found: {0}")]
    KeyNotFound(String),
    #[error("index not found: {0}")]
    IndexNotFound(String),
    #[error("unique constraint violated on index `{0}` for value `{1}`")]
    UniqueViolation(String, String),
    #[error("store already exists: {0}")]
    StoreAlreadyExists(String),
    #[error("transaction error: {0}")]
    TransactionError(String),
}

pub type Result<T> = std::result::Result<T, IndexedDbError>;

// ── Index definition ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDef {
    pub name: String,
    pub key_path: String,
    pub unique: bool,
    pub multi_entry: bool,
}

// ── Object store ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ObjectStore {
    pub name: String,
    pub key_path: String,
    pub auto_increment: bool,
    pub indexes: Vec<IndexDef>,
    pub records: BTreeMap<String, serde_json::Value>,
    next_key: u64,
}

impl ObjectStore {
    fn new(name: impl Into<String>, key_path: impl Into<String>, auto_increment: bool) -> Self {
        Self {
            name: name.into(),
            key_path: key_path.into(),
            auto_increment,
            indexes: Vec::new(),
            records: BTreeMap::new(),
            next_key: 1,
        }
    }

    /// Insert or update a record. If `auto_increment` is enabled and key is
    /// empty, a monotonically increasing key is assigned.
    pub fn put(&mut self, key: &str, value: serde_json::Value) -> Result<String> {
        let actual_key = if key.is_empty() && self.auto_increment {
            let k = self.next_key.to_string();
            self.next_key += 1;
            k
        } else {
            key.to_string()
        };

        // Check unique index constraints for new/updated value.
        for idx in &self.indexes {
            if !idx.unique {
                continue;
            }
            let new_val = value.get(&idx.key_path);
            for (existing_key, existing_val) in &self.records {
                if *existing_key == actual_key {
                    continue; // updating same record is fine
                }
                if existing_val.get(&idx.key_path) == new_val && new_val.is_some() {
                    return Err(IndexedDbError::UniqueViolation(
                        idx.name.clone(),
                        format!("{:?}", new_val),
                    ));
                }
            }
        }

        self.records.insert(actual_key.clone(), value);
        Ok(actual_key)
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.records.get(key)
    }

    pub fn delete(&mut self, key: &str) -> Result<()> {
        self.records
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| IndexedDbError::KeyNotFound(key.to_string()))
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    pub fn count(&self) -> usize {
        self.records.len()
    }

    pub fn get_all(&self) -> Vec<&serde_json::Value> {
        self.records.values().collect()
    }

    pub fn get_all_keys(&self) -> Vec<&String> {
        self.records.keys().collect()
    }

    pub fn create_index(&mut self, def: IndexDef) {
        self.indexes.push(def);
    }

    /// Return all records where `index_name`'s key_path matches `value`.
    pub fn get_by_index(
        &self,
        index_name: &str,
        value: &serde_json::Value,
    ) -> Result<Vec<&serde_json::Value>> {
        let idx = self
            .indexes
            .iter()
            .find(|i| i.name == index_name)
            .ok_or_else(|| IndexedDbError::IndexNotFound(index_name.to_string()))?;

        let results: Vec<_> = self
            .records
            .values()
            .filter(|rec| rec.get(&idx.key_path) == Some(value))
            .collect();

        Ok(results)
    }
}

// ── Transaction ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub enum TxOp {
    Put {
        store: String,
        key: String,
        value: serde_json::Value,
    },
    Delete {
        store: String,
        key: String,
    },
    Clear {
        store: String,
    },
}

#[derive(Debug, Clone)]
pub struct Transaction {
    pub stores: Vec<String>,
    pub mode: TransactionMode,
    pub operations: Vec<TxOp>,
}

// ── Database ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Database {
    pub name: String,
    pub version: u32,
    pub stores: HashMap<String, ObjectStore>,
}

impl Database {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
            stores: HashMap::new(),
        }
    }

    pub fn create_store(
        &mut self,
        name: impl Into<String>,
        key_path: impl Into<String>,
        auto_increment: bool,
    ) -> Result<&mut ObjectStore> {
        let name = name.into();
        if self.stores.contains_key(&name) {
            return Err(IndexedDbError::StoreAlreadyExists(name));
        }
        let store = ObjectStore::new(name.clone(), key_path, auto_increment);
        self.stores.insert(name.clone(), store);
        Ok(self.stores.get_mut(&name).unwrap())
    }

    pub fn delete_store(&mut self, name: &str) -> Result<()> {
        self.stores
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| IndexedDbError::StoreNotFound(name.to_string()))
    }

    /// Execute a transaction atomically. On any error the entire transaction
    /// is rolled back (we snapshot stores before applying).
    pub fn execute_transaction(&mut self, tx: Transaction) -> Result<()> {
        // Validate all referenced stores exist.
        for s in &tx.stores {
            if !self.stores.contains_key(s) {
                return Err(IndexedDbError::StoreNotFound(s.clone()));
            }
        }

        // Snapshot for rollback.
        let snapshot: HashMap<String, ObjectStore> = tx
            .stores
            .iter()
            .filter_map(|s| self.stores.get(s).map(|st| (s.clone(), st.clone())))
            .collect();

        for op in &tx.operations {
            let result = match op {
                TxOp::Put { store, key, value } => {
                    let st = self
                        .stores
                        .get_mut(store)
                        .ok_or_else(|| IndexedDbError::StoreNotFound(store.clone()))?;
                    st.put(key, value.clone()).map(|_| ())
                }
                TxOp::Delete { store, key } => {
                    let st = self
                        .stores
                        .get_mut(store)
                        .ok_or_else(|| IndexedDbError::StoreNotFound(store.clone()))?;
                    st.delete(key)
                }
                TxOp::Clear { store } => {
                    let st = self
                        .stores
                        .get_mut(store)
                        .ok_or_else(|| IndexedDbError::StoreNotFound(store.clone()))?;
                    st.clear();
                    Ok(())
                }
            };

            if let Err(e) = result {
                // Rollback.
                for (name, original) in snapshot {
                    self.stores.insert(name, original);
                }
                return Err(IndexedDbError::TransactionError(e.to_string()));
            }
        }

        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn crud_basic() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("users", "id", false).unwrap();
        store.put("1", json!({"id": "1", "name": "Alice"})).unwrap();
        assert_eq!(store.get("1").unwrap()["name"], "Alice");
        store.delete("1").unwrap();
        assert!(store.get("1").is_none());
    }

    #[test]
    fn auto_increment() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("items", "id", true).unwrap();
        let k1 = store.put("", json!({"val": "a"})).unwrap();
        let k2 = store.put("", json!({"val": "b"})).unwrap();
        assert_eq!(k1, "1");
        assert_eq!(k2, "2");
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn index_lookup() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("users", "id", false).unwrap();
        store.create_index(IndexDef {
            name: "by_age".into(),
            key_path: "age".into(),
            unique: false,
            multi_entry: false,
        });
        store.put("1", json!({"id": "1", "age": 30})).unwrap();
        store.put("2", json!({"id": "2", "age": 30})).unwrap();
        store.put("3", json!({"id": "3", "age": 25})).unwrap();
        let results = store.get_by_index("by_age", &json!(30)).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn unique_index_violation() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("emails", "id", false).unwrap();
        store.create_index(IndexDef {
            name: "unique_email".into(),
            key_path: "email".into(),
            unique: true,
            multi_entry: false,
        });
        store
            .put("1", json!({"id": "1", "email": "a@b.com"}))
            .unwrap();
        let err = store
            .put("2", json!({"id": "2", "email": "a@b.com"}))
            .unwrap_err();
        assert!(matches!(err, IndexedDbError::UniqueViolation(..)));
    }

    #[test]
    fn get_all_and_keys() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("s", "id", false).unwrap();
        store.put("a", json!(1)).unwrap();
        store.put("b", json!(2)).unwrap();
        store.put("c", json!(3)).unwrap();
        assert_eq!(store.get_all().len(), 3);
        assert_eq!(store.get_all_keys().len(), 3);
    }

    #[test]
    fn clear_store() {
        let mut db = Database::new("test", 1);
        let store = db.create_store("s", "id", false).unwrap();
        store.put("1", json!(1)).unwrap();
        store.put("2", json!(2)).unwrap();
        store.clear();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn transaction_atomic_success() {
        let mut db = Database::new("test", 1);
        db.create_store("s", "id", false).unwrap();
        let tx = Transaction {
            stores: vec!["s".into()],
            mode: TransactionMode::ReadWrite,
            operations: vec![
                TxOp::Put {
                    store: "s".into(),
                    key: "1".into(),
                    value: json!("hello"),
                },
                TxOp::Put {
                    store: "s".into(),
                    key: "2".into(),
                    value: json!("world"),
                },
            ],
        };
        db.execute_transaction(tx).unwrap();
        assert_eq!(db.stores["s"].count(), 2);
    }

    #[test]
    fn transaction_rollback_on_error() {
        let mut db = Database::new("test", 1);
        {
            let store = db.create_store("s", "id", false).unwrap();
            store.create_index(IndexDef {
                name: "uniq".into(),
                key_path: "v".into(),
                unique: true,
                multi_entry: false,
            });
            store
                .put("existing", json!({"v": "dup"}))
                .unwrap();
        }
        let tx = Transaction {
            stores: vec!["s".into()],
            mode: TransactionMode::ReadWrite,
            operations: vec![
                TxOp::Put {
                    store: "s".into(),
                    key: "new1".into(),
                    value: json!({"v": "ok"}),
                },
                // This will violate the unique index.
                TxOp::Put {
                    store: "s".into(),
                    key: "new2".into(),
                    value: json!({"v": "dup"}),
                },
            ],
        };
        assert!(db.execute_transaction(tx).is_err());
        // Rolled back — "new1" should not exist.
        assert_eq!(db.stores["s"].count(), 1);
        assert!(db.stores["s"].get("new1").is_none());
    }

    #[test]
    fn delete_store() {
        let mut db = Database::new("test", 1);
        db.create_store("s", "id", false).unwrap();
        db.delete_store("s").unwrap();
        assert!(db.stores.is_empty());
    }

    #[test]
    fn duplicate_store_name_errors() {
        let mut db = Database::new("test", 1);
        db.create_store("s", "id", false).unwrap();
        assert!(db.create_store("s", "id", false).is_err());
    }

    #[test]
    fn transaction_clear_op() {
        let mut db = Database::new("test", 1);
        {
            let store = db.create_store("s", "id", false).unwrap();
            store.put("1", json!(1)).unwrap();
            store.put("2", json!(2)).unwrap();
        }
        let tx = Transaction {
            stores: vec!["s".into()],
            mode: TransactionMode::ReadWrite,
            operations: vec![TxOp::Clear {
                store: "s".into(),
            }],
        };
        db.execute_transaction(tx).unwrap();
        assert_eq!(db.stores["s"].count(), 0);
    }
}
