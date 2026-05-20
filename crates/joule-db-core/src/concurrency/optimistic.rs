//! Optimistic Concurrency Control (OCC) for JouleDB
//!
//! Implements timestamp-based optimistic concurrency control, which allows
//! transactions to proceed without locking, validating at commit time.
//!
//! ## Design
//!
//! - Each transaction gets a start timestamp (read timestamp)
//! - Writes are buffered locally until commit
//! - At commit, validate that no conflicting writes occurred
//! - If validation passes, assign commit timestamp and apply writes
//! - If validation fails, abort and retry
//!
//! ## Benefits
//!
//! - No locking overhead during transaction execution
//! - Excellent for read-heavy workloads
//! - Allows high concurrency
//! - Automatic conflict detection

use crate::error::TransactionError;
use crate::tx::{TxId, TxState};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Global timestamp counter
static TIMESTAMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Get next timestamp
fn next_timestamp() -> u64 {
    TIMESTAMP_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Versioned value with write timestamp
#[derive(Debug, Clone)]
pub(crate) struct VersionedValue {
    /// The actual value
    value: Vec<u8>,
    /// Timestamp when this version was written
    write_timestamp: u64,
}

/// OCC transaction state
#[derive(Debug)]
pub struct OccTransaction {
    /// Transaction ID
    id: TxId,
    /// Start timestamp (read timestamp)
    start_timestamp: u64,
    /// Commit timestamp (assigned on successful commit)
    commit_timestamp: Option<u64>,
    /// Current state
    state: TxState,
    /// Keys read during transaction
    read_set: HashSet<Vec<u8>>,
    /// Keys written during transaction
    write_set: HashMap<Vec<u8>, Vec<u8>>,
    /// Reference to versioned store
    store: Arc<RwLock<HashMap<Vec<u8>, VersionedValue>>>,
}

impl OccTransaction {
    /// Create a new OCC transaction
    pub fn new(store: Arc<RwLock<HashMap<Vec<u8>, VersionedValue>>>) -> Self {
        let id = next_timestamp();
        let start_timestamp = next_timestamp();

        Self {
            id,
            start_timestamp,
            commit_timestamp: None,
            state: TxState::Active,
            read_set: HashSet::new(),
            write_set: HashMap::new(),
            store,
        }
    }

    /// Read a value (adds to read set)
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check write set first (local writes)
        if let Some(value) = self.write_set.get(key) {
            self.read_set.insert(key.to_vec());
            return Ok(Some(value.clone()));
        }

        // Read from store
        let store = self
            .store
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        if let Some(versioned) = store.get(key) {
            // Check if version was written after our start timestamp
            if versioned.write_timestamp > self.start_timestamp {
                // Value was modified after we started - potential conflict
                // For now, we allow it (will be caught in validation)
            }

            self.read_set.insert(key.to_vec());
            Ok(Some(versioned.value.clone()))
        } else {
            self.read_set.insert(key.to_vec());
            Ok(None)
        }
    }

    /// Write a value (adds to write set)
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        self.write_set.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        let existed = self.get(key)?.is_some();
        // Mark as deleted by inserting empty vec (we'll handle None differently)
        self.write_set.insert(key.to_vec(), vec![]);
        Ok(existed)
    }

    /// Validate transaction (check for conflicts)
    fn validate(&self) -> Result<bool, TransactionError> {
        let store = self
            .store
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        // Check all read keys haven't been modified
        for key in &self.read_set {
            if let Some(versioned) = store.get(key) {
                // If key was written after our start timestamp, conflict!
                if versioned.write_timestamp > self.start_timestamp {
                    // Also check if we're writing to it (that's OK)
                    if !self.write_set.contains_key(key) {
                        return Ok(false); // Conflict detected
                    }
                }
            }
        }

        // Check write keys haven't been modified by others
        for key in self.write_set.keys() {
            if let Some(versioned) = store.get(key) {
                // If key was written after our start timestamp, conflict!
                if versioned.write_timestamp > self.start_timestamp {
                    return Ok(false); // Conflict detected
                }
            }
        }

        Ok(true) // No conflicts
    }

    /// Commit the transaction
    pub fn commit(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Validate transaction
        let valid = self.validate()?;
        if !valid {
            self.state = TxState::Aborted;
            return Err(TransactionError::SerializationFailure {
                reason: "validation failed - conflict detected".to_string(),
            });
        }

        // Assign commit timestamp
        let commit_ts = next_timestamp();
        self.commit_timestamp = Some(commit_ts);

        // Apply writes with commit timestamp
        let mut store = self
            .store
            .write()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        for (key, value) in self.write_set {
            if value.is_empty() {
                // Delete
                store.remove(&key);
            } else {
                // Insert/update with commit timestamp
                store.insert(
                    key,
                    VersionedValue {
                        value,
                        write_timestamp: commit_ts,
                    },
                );
            }
        }

        self.state = TxState::Committed;
        Ok(())
    }

    /// Rollback the transaction
    pub fn rollback(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Simply clear write set
        self.write_set.clear();
        self.read_set.clear();
        self.state = TxState::Aborted;
        Ok(())
    }

    /// Get transaction ID
    pub fn id(&self) -> TxId {
        self.id
    }

    /// Get start timestamp
    pub fn start_timestamp(&self) -> u64 {
        self.start_timestamp
    }

    /// Get commit timestamp
    pub fn commit_timestamp(&self) -> Option<u64> {
        self.commit_timestamp
    }

    /// Get state
    pub fn state(&self) -> TxState {
        self.state
    }
}

/// OCC transaction manager
#[derive(Clone)]
pub struct OccTransactionManager {
    store: Arc<RwLock<HashMap<Vec<u8>, VersionedValue>>>,
}

impl OccTransactionManager {
    /// Create a new OCC transaction manager
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Begin a new transaction
    pub fn begin(&self) -> OccTransaction {
        OccTransaction::new(self.store.clone())
    }

    /// Get a value (outside transaction, for testing)
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let store = self.store.read().ok()?;
        store.get(key).map(|v| v.value.clone())
    }
}

impl Default for OccTransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occ_basic() {
        let manager = OccTransactionManager::new();
        let mut tx = manager.begin();

        // Write
        tx.put(b"key1", b"value1").unwrap();
        tx.put(b"key2", b"value2").unwrap();

        // Read
        assert_eq!(tx.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        // Commit
        tx.commit().unwrap();

        // Verify
        assert_eq!(manager.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(manager.get(b"key2"), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_occ_conflict_detection() {
        let manager = OccTransactionManager::new();

        // First transaction
        let mut tx1 = manager.begin();
        tx1.put(b"key1", b"value1").unwrap();

        // Second transaction reads same key
        let mut tx2 = manager.begin();
        let _ = tx2.get(b"key1").unwrap();

        // First transaction commits
        tx1.commit().unwrap();

        // Second transaction should fail validation
        let result = tx2.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_occ_no_conflict() {
        let manager = OccTransactionManager::new();

        // First transaction
        let mut tx1 = manager.begin();
        tx1.put(b"key1", b"value1").unwrap();
        tx1.commit().unwrap();

        // Second transaction reads different key
        let mut tx2 = manager.begin();
        let _ = tx2.get(b"key1").unwrap(); // Read committed value
        tx2.put(b"key2", b"value2").unwrap();

        // Should commit successfully
        tx2.commit().unwrap();

        assert_eq!(manager.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(manager.get(b"key2"), Some(b"value2".to_vec()));
    }
}
