//! Simple in-memory transaction implementation
//!
//! Provides basic transaction semantics for testing and simple use cases.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::traits::{
    IsolationLevel, ReadTransaction, Transaction, TransactionManager, TxId, TxState,
};
use crate::error::TransactionError;

/// Counter for generating unique transaction IDs
static TX_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique transaction ID
fn next_tx_id() -> TxId {
    TX_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Simple in-memory data store
type DataStore = Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>;

/// Simple transaction implementation
///
/// Uses a write buffer that is applied on commit.
pub struct SimpleTransaction {
    /// Transaction ID
    id: TxId,
    /// Isolation level
    isolation_level: IsolationLevel,
    /// Current state
    state: TxState,
    /// Reference to shared data store
    store: DataStore,
    /// Local write buffer (uncommitted changes)
    write_buffer: HashMap<Vec<u8>, Option<Vec<u8>>>,
    /// Read snapshot (for repeatable reads)
    read_snapshot: Option<HashMap<Vec<u8>, Vec<u8>>>,
}

impl SimpleTransaction {
    /// Create a new transaction
    pub fn new(store: DataStore, isolation_level: IsolationLevel) -> Self {
        let id = next_tx_id();

        // For repeatable read and above, take a snapshot
        let read_snapshot = match isolation_level {
            IsolationLevel::RepeatableRead
            | IsolationLevel::Serializable
            | IsolationLevel::Snapshot => Some(
                store
                    .read()
                    .expect("lock poisoned: data store read")
                    .clone(),
            ),
            _ => None,
        };

        Self {
            id,
            isolation_level,
            state: TxState::Active,
            store,
            write_buffer: HashMap::new(),
            read_snapshot,
        }
    }
}

impl Transaction for SimpleTransaction {
    fn id(&self) -> TxId {
        self.id
    }

    fn isolation_level(&self) -> IsolationLevel {
        self.isolation_level
    }

    fn state(&self) -> TxState {
        self.state
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // First check write buffer (uncommitted local changes)
        if let Some(value_opt) = self.write_buffer.get(key) {
            return Ok(value_opt.clone());
        }

        // Then check snapshot or live store
        if let Some(ref snapshot) = self.read_snapshot {
            Ok(snapshot.get(key).cloned())
        } else {
            let store = self
                .store
                .read()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;
            Ok(store.get(key).cloned())
        }
    }

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        self.write_buffer.insert(key.to_vec(), Some(value.to_vec()));
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check if key exists
        let existed = self.get(key)?.is_some();

        // Mark as deleted in write buffer
        self.write_buffer.insert(key.to_vec(), None);

        Ok(existed)
    }

    fn commit(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Apply write buffer to store
        let mut store = self
            .store
            .write()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        for (key, value_opt) in self.write_buffer.drain() {
            match value_opt {
                Some(value) => {
                    store.insert(key, value);
                }
                None => {
                    store.remove(&key);
                }
            }
        }

        self.state = TxState::Committed;
        Ok(())
    }

    fn rollback(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Simply discard write buffer
        self.write_buffer.clear();
        self.state = TxState::Aborted;
        Ok(())
    }
}

/// Read-only transaction implementation
pub struct SimpleReadTransaction {
    /// Transaction ID
    id: TxId,
    /// Snapshot of data at transaction start
    snapshot: HashMap<Vec<u8>, Vec<u8>>,
}

impl SimpleReadTransaction {
    /// Create a new read-only transaction
    pub fn new(store: &DataStore) -> Self {
        let id = next_tx_id();
        let snapshot = store
            .read()
            .expect("lock poisoned: data store read")
            .clone();
        Self { id, snapshot }
    }
}

impl ReadTransaction for SimpleReadTransaction {
    fn id(&self) -> TxId {
        self.id
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        Ok(self.snapshot.get(key).cloned())
    }
}

/// Simple transaction manager
pub struct SimpleTransactionManager {
    /// Shared data store
    store: DataStore,
}

impl SimpleTransactionManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with an existing store
    pub fn with_store(store: DataStore) -> Self {
        Self { store }
    }

    /// Get a reference to the underlying store
    pub fn store(&self) -> &DataStore {
        &self.store
    }

    /// Begin a read-only transaction
    pub fn begin_read(&self) -> SimpleReadTransaction {
        SimpleReadTransaction::new(&self.store)
    }
}

impl Default for SimpleTransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionManager for SimpleTransactionManager {
    type Tx = SimpleTransaction;

    fn begin(&self) -> Result<Self::Tx, TransactionError> {
        Ok(SimpleTransaction::new(
            self.store.clone(),
            IsolationLevel::default(),
        ))
    }

    fn begin_with_isolation(&self, level: IsolationLevel) -> Result<Self::Tx, TransactionError> {
        Ok(SimpleTransaction::new(self.store.clone(), level))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_transaction_basic() {
        let manager = SimpleTransactionManager::new();

        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.put(b"key2", b"value2").unwrap();
        tx.commit().unwrap();

        let tx = manager.begin().unwrap();
        assert_eq!(tx.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(tx.get(b"key2").unwrap(), Some(b"value2".to_vec()));
        assert_eq!(tx.get(b"key3").unwrap(), None);
    }

    #[test]
    fn test_transaction_rollback() {
        let manager = SimpleTransactionManager::new();

        // First transaction - commit
        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.commit().unwrap();

        // Second transaction - rollback
        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"changed").unwrap();
        tx.rollback().unwrap();

        // Value should be unchanged
        let tx = manager.begin().unwrap();
        assert_eq!(tx.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_transaction_delete() {
        let manager = SimpleTransactionManager::new();

        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.commit().unwrap();

        let mut tx = manager.begin().unwrap();
        let existed = tx.delete(b"key1").unwrap();
        assert!(existed);
        tx.commit().unwrap();

        let tx = manager.begin().unwrap();
        assert_eq!(tx.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_transaction_isolation() {
        let manager = SimpleTransactionManager::new();

        // Setup initial data
        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.commit().unwrap();

        // Start repeatable read transaction
        let tx1 = manager
            .begin_with_isolation(IsolationLevel::RepeatableRead)
            .unwrap();
        assert_eq!(tx1.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        // Commit change in another transaction
        let mut tx2 = manager.begin().unwrap();
        tx2.put(b"key1", b"changed").unwrap();
        tx2.commit().unwrap();

        // Repeatable read should still see old value
        assert_eq!(tx1.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_read_only_transaction() {
        let manager = SimpleTransactionManager::new();

        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.commit().unwrap();

        let read_tx = manager.begin_read();
        assert_eq!(read_tx.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_transaction_id_increments() {
        let manager = SimpleTransactionManager::new();

        let tx1 = manager.begin().unwrap();
        let tx2 = manager.begin().unwrap();

        assert!(tx2.id() > tx1.id());
    }

    #[test]
    fn test_uncommitted_writes_visible_within_tx() {
        let manager = SimpleTransactionManager::new();

        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();

        // Should see uncommitted write within same transaction
        assert_eq!(tx.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        // But not in another transaction
        let tx2 = manager.begin().unwrap();
        assert_eq!(tx2.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_exists_method() {
        let manager = SimpleTransactionManager::new();

        let mut tx = manager.begin().unwrap();
        tx.put(b"key1", b"value1").unwrap();
        tx.commit().unwrap();

        let tx = manager.begin().unwrap();
        assert!(tx.exists(b"key1").unwrap());
        assert!(!tx.exists(b"nonexistent").unwrap());
    }
}
