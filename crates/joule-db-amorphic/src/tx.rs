//! MVCC (Multi-Version Concurrency Control) Transactions for AmorphicStore
//!
//! This module provides transactional access to the AmorphicStore with support for
//! multiple isolation levels, including snapshot isolation.
//!
//! ## MVCC Design
//!
//! Each record has multiple versions, each tagged with a timestamp. Transactions:
//! 1. Read from a snapshot at their start timestamp
//! 2. Buffer writes locally until commit
//! 3. On commit, validate no write-write conflicts and apply changes
//!
//! ## Isolation Levels
//!
//! - **ReadCommitted**: See committed data, but reads within a transaction may change
//! - **RepeatableRead**: Reads are consistent within a transaction (locks)
//! - **Serializable**: Full isolation with conflict detection (SSI)
//! - **Snapshot**: MVCC-based, reads see a consistent snapshot

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use joule_db_core::error::TransactionError;
use joule_db_core::tx::{IsolationLevel, Transaction, TransactionManager, TxId, TxState};

use crate::{AmorphicStore, RecordId, Value};

/// Timestamp type for MVCC versioning
pub type Timestamp = u64;

/// A version of a record with its data and timestamps
#[derive(Debug, Clone)]
pub struct RecordVersion {
    /// The record data (serialized fields)
    pub data: HashMap<String, Value>,
    /// When this version was created (write timestamp)
    pub write_ts: Timestamp,
    /// When this version expires (if deleted, otherwise u64::MAX)
    pub delete_ts: Timestamp,
    /// Transaction that created this version (for uncommitted versions)
    pub created_by: Option<TxId>,
}

impl RecordVersion {
    /// Check if this version is visible at the given timestamp
    pub fn is_visible_at(&self, ts: Timestamp) -> bool {
        self.write_ts <= ts && ts < self.delete_ts
    }
}

/// Version store for MVCC - tracks all versions of all records
pub struct VersionStore {
    /// Versions indexed by RecordId
    versions: RwLock<HashMap<RecordId, Vec<RecordVersion>>>,
    /// Current global timestamp (monotonically increasing)
    current_ts: AtomicU64,
    /// Active transactions and their read timestamps
    active_transactions: RwLock<HashMap<TxId, Timestamp>>,
    /// Write locks (key -> tx holding lock)
    write_locks: RwLock<HashMap<RecordId, TxId>>,
    /// Predicate locks for SSI
    predicate_locks: PredicateLockManager,
}

impl VersionStore {
    /// Create a new empty version store
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            current_ts: AtomicU64::new(1),
            active_transactions: RwLock::new(HashMap::new()),
            write_locks: RwLock::new(HashMap::new()),
            predicate_locks: PredicateLockManager::new(),
        }
    }

    /// Get reference to predicate lock manager
    pub fn predicate_lock_manager(&self) -> &PredicateLockManager {
        &self.predicate_locks
    }

    /// Get the current timestamp
    pub fn current_timestamp(&self) -> Timestamp {
        self.current_ts.load(Ordering::SeqCst)
    }

    /// Advance the timestamp and return the new value
    pub fn advance_timestamp(&self) -> Timestamp {
        self.current_ts.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Register a new transaction with a read timestamp
    pub fn begin_transaction(&self, tx_id: TxId) -> Timestamp {
        let read_ts = self.current_timestamp();
        let mut active = self.active_transactions.write().unwrap();
        active.insert(tx_id, read_ts);
        read_ts
    }

    /// Get the read timestamp for a transaction
    pub fn get_read_timestamp(&self, tx_id: TxId) -> Option<Timestamp> {
        let active = self.active_transactions.read().unwrap();
        active.get(&tx_id).copied()
    }

    /// Get the version of a record visible at a given timestamp
    pub fn get_version_at(
        &self,
        record_id: RecordId,
        ts: Timestamp,
        tx_id: Option<TxId>,
    ) -> Option<RecordVersion> {
        let versions = self.versions.read().unwrap();
        if let Some(record_versions) = versions.get(&record_id) {
            // Find the latest version visible at this timestamp
            // Also include our own uncommitted writes
            record_versions
                .iter()
                .filter(|v| v.is_visible_at(ts) || (tx_id.is_some() && v.created_by == tx_id))
                .max_by_key(|v| v.write_ts)
                .cloned()
        } else {
            None
        }
    }

    /// Try to acquire a write lock on a record
    pub fn try_lock(&self, record_id: RecordId, tx_id: TxId) -> Result<(), TransactionError> {
        let mut locks = self.write_locks.write().unwrap();
        if let Some(&holder) = locks.get(&record_id) {
            if holder != tx_id {
                return Err(TransactionError::WriteConflict {
                    key: record_id.to_le_bytes().to_vec(),
                    holder_tx_id: holder,
                });
            }
        }
        locks.insert(record_id, tx_id);
        Ok(())
    }

    /// Release all write locks held by a transaction
    pub fn release_locks(&self, tx_id: TxId) {
        let mut locks = self.write_locks.write().unwrap();
        locks.retain(|_, &mut holder| holder != tx_id);
    }

    /// Add a new version (for commits)
    pub fn add_version(&self, record_id: RecordId, version: RecordVersion) {
        let mut versions = self.versions.write().unwrap();
        versions.entry(record_id).or_default().push(version);
    }

    /// Mark a record as deleted at a given timestamp
    pub fn mark_deleted(&self, record_id: RecordId, delete_ts: Timestamp) {
        let mut versions = self.versions.write().unwrap();
        if let Some(record_versions) = versions.get_mut(&record_id) {
            // Mark the latest version as deleted
            if let Some(latest) = record_versions.last_mut() {
                if latest.delete_ts == u64::MAX {
                    latest.delete_ts = delete_ts;
                }
            }
        }
    }

    /// Remove a transaction from active set
    pub fn end_transaction(&self, tx_id: TxId) {
        let mut active = self.active_transactions.write().unwrap();
        active.remove(&tx_id);
    }

    /// Get the minimum active read timestamp (for garbage collection)
    pub fn min_active_read_ts(&self) -> Option<Timestamp> {
        let active = self.active_transactions.read().unwrap();
        active.values().min().copied()
    }

    /// Garbage collect old versions that are no longer visible
    pub fn gc(&self) {
        let min_ts = match self.min_active_read_ts() {
            Some(ts) => ts,
            None => self.current_timestamp(), // No active transactions, can clean everything old
        };

        let mut versions = self.versions.write().unwrap();
        for record_versions in versions.values_mut() {
            // Keep versions that might still be visible
            // A version can be garbage collected if:
            // 1. It's been superseded by a newer version
            // 2. That newer version was committed before min_ts
            if record_versions.len() <= 1 {
                continue;
            }

            // Keep only the necessary versions
            let mut i = 0;
            while i < record_versions.len().saturating_sub(1) {
                let next_write_ts = record_versions[i + 1].write_ts;
                // Can GC if the next version is committed and visible to all active txns
                if next_write_ts < min_ts {
                    record_versions.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }
}

impl Default for VersionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Write operation buffered in a transaction
#[derive(Debug, Clone)]
pub enum WriteOp {
    /// Insert or update a record
    Put { data: HashMap<String, Value> },
    /// Delete a record
    Delete,
}

// =============================================================================
// PREDICATE LOCKS FOR SSI
// =============================================================================

/// Predicate lock type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateLockType {
    /// Range was read
    Read,
    /// Range was written
    Write,
}

/// A predicate lock for tracking range reads/writes
#[derive(Debug, Clone)]
pub struct PredicateLock {
    /// Table/column being locked
    pub table: String,
    /// Field being filtered on
    pub field: String,
    /// Minimum value (inclusive)
    pub min: f64,
    /// Maximum value (exclusive)
    pub max: f64,
    /// Transaction holding the lock
    pub tx_id: TxId,
    /// Type of lock
    pub lock_type: PredicateLockType,
    /// Timestamp when lock was acquired
    pub timestamp: Timestamp,
}

impl PredicateLock {
    /// Check if this lock overlaps with a range
    pub fn overlaps(&self, field: &str, min: f64, max: f64) -> bool {
        self.field == field && !(max <= self.min || min >= self.max)
    }

    /// Check if this lock conflicts with another
    pub fn conflicts_with(&self, other: &PredicateLock) -> bool {
        // Read-read is not a conflict
        if self.lock_type == PredicateLockType::Read && other.lock_type == PredicateLockType::Read {
            return false;
        }

        // Check if ranges overlap
        self.table == other.table && self.overlaps(&other.field, other.min, other.max)
    }
}

/// Manages predicate locks for SSI
pub struct PredicateLockManager {
    /// All active predicate locks
    locks: RwLock<Vec<PredicateLock>>,
}

impl PredicateLockManager {
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(Vec::new()),
        }
    }

    /// Record a predicate lock (range read or write)
    pub fn record_lock(&self, lock: PredicateLock) {
        let mut locks = self.locks.write().unwrap();
        locks.push(lock);
    }

    /// Record a range read
    pub fn record_range_read(
        &self,
        table: &str,
        field: &str,
        min: f64,
        max: f64,
        tx_id: TxId,
        timestamp: Timestamp,
    ) {
        self.record_lock(PredicateLock {
            table: table.to_string(),
            field: field.to_string(),
            min,
            max,
            tx_id,
            lock_type: PredicateLockType::Read,
            timestamp,
        });
    }

    /// Record a range write
    pub fn record_range_write(
        &self,
        table: &str,
        field: &str,
        min: f64,
        max: f64,
        tx_id: TxId,
        timestamp: Timestamp,
    ) {
        self.record_lock(PredicateLock {
            table: table.to_string(),
            field: field.to_string(),
            min,
            max,
            tx_id,
            lock_type: PredicateLockType::Write,
            timestamp,
        });
    }

    /// Check for conflicts between a transaction and others
    ///
    /// Returns list of conflicting transaction IDs
    pub fn check_conflicts(&self, tx_id: TxId) -> Vec<(TxId, String)> {
        let locks = self.locks.read().unwrap();

        let my_locks: Vec<&PredicateLock> = locks.iter().filter(|l| l.tx_id == tx_id).collect();

        let other_locks: Vec<&PredicateLock> = locks.iter().filter(|l| l.tx_id != tx_id).collect();

        let mut conflicts = Vec::new();

        for my_lock in &my_locks {
            for other_lock in &other_locks {
                if my_lock.conflicts_with(other_lock) {
                    let reason = format!(
                        "Conflict on {}.{} [{}, {}) vs [{}, {})",
                        my_lock.table,
                        my_lock.field,
                        my_lock.min,
                        my_lock.max,
                        other_lock.min,
                        other_lock.max
                    );
                    conflicts.push((other_lock.tx_id, reason));
                }
            }
        }

        conflicts
    }

    /// Remove all locks for a transaction
    pub fn release_locks(&self, tx_id: TxId) {
        let mut locks = self.locks.write().unwrap();
        locks.retain(|l| l.tx_id != tx_id);
    }

    /// Clean up old locks (older than min_active_ts can be removed)
    pub fn cleanup_old_locks(&self, min_active_ts: Timestamp) {
        let mut locks = self.locks.write().unwrap();
        locks.retain(|l| l.timestamp >= min_active_ts);
    }

    /// Get count of active locks
    pub fn lock_count(&self) -> usize {
        self.locks.read().unwrap().len()
    }
}

impl Default for PredicateLockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// MVCC Transaction for AmorphicStore
///
/// Provides transactional access with snapshot isolation semantics.
pub struct AmorphicTransaction {
    /// Transaction ID
    id: TxId,
    /// Isolation level
    isolation: IsolationLevel,
    /// Read timestamp (snapshot point)
    read_ts: Timestamp,
    /// Transaction state
    state: TxState,
    /// Buffered writes
    write_set: HashMap<RecordId, WriteOp>,
    /// Records we've read (for SSI validation)
    read_set: HashSet<RecordId>,
    /// Range reads for SSI predicate locking
    range_reads: Vec<(String, String, f64, f64)>, // (table, field, min, max)
    /// Reference to the version store
    version_store: Arc<VersionStore>,
    /// Reference to the underlying store (for future integration)
    #[allow(dead_code)]
    store: Arc<RwLock<AmorphicStore>>,
}

impl AmorphicTransaction {
    /// Create a new transaction
    fn new(
        id: TxId,
        isolation: IsolationLevel,
        version_store: Arc<VersionStore>,
        store: Arc<RwLock<AmorphicStore>>,
    ) -> Self {
        let read_ts = version_store.begin_transaction(id);
        Self {
            id,
            isolation,
            read_ts,
            state: TxState::Active,
            write_set: HashMap::new(),
            read_set: HashSet::new(),
            range_reads: Vec::new(),
            version_store,
            store,
        }
    }

    /// Record a range read for SSI predicate locking
    ///
    /// This should be called when a transaction reads a range of records
    /// (e.g., during a range query). The range is recorded for conflict
    /// detection during SSI validation.
    pub fn record_range_read(&mut self, table: &str, field: &str, min: f64, max: f64) {
        if self.isolation == IsolationLevel::Serializable {
            self.range_reads
                .push((table.to_string(), field.to_string(), min, max));

            // Also register with the version store's predicate lock manager
            self.version_store
                .predicate_lock_manager()
                .record_range_read(table, field, min, max, self.id, self.read_ts);
        }
    }

    /// Record a range write for SSI predicate locking
    pub fn record_range_write(&mut self, table: &str, field: &str, min: f64, max: f64) {
        if self.isolation == IsolationLevel::Serializable {
            self.version_store
                .predicate_lock_manager()
                .record_range_write(
                    table,
                    field,
                    min,
                    max,
                    self.id,
                    self.version_store.current_timestamp(),
                );
        }
    }

    /// Get a record by ID with MVCC visibility
    pub fn get_record(
        &self,
        record_id: RecordId,
    ) -> Result<Option<HashMap<String, Value>>, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check our write buffer first
        if let Some(op) = self.write_set.get(&record_id) {
            return Ok(match op {
                WriteOp::Put { data } => Some(data.clone()),
                WriteOp::Delete => None,
            });
        }

        // Read from version store
        let version = self
            .version_store
            .get_version_at(record_id, self.read_ts, Some(self.id));
        Ok(version.map(|v| v.data))
    }

    /// Put a record (insert or update)
    pub fn put_record(
        &mut self,
        record_id: RecordId,
        data: HashMap<String, Value>,
    ) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Acquire write lock
        self.version_store.try_lock(record_id, self.id)?;

        // Buffer the write
        self.write_set.insert(record_id, WriteOp::Put { data });
        Ok(())
    }

    /// Delete a record
    pub fn delete_record(&mut self, record_id: RecordId) -> Result<bool, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check if record exists
        let exists = self.get_record(record_id)?.is_some();
        if !exists {
            return Ok(false);
        }

        // Acquire write lock
        self.version_store.try_lock(record_id, self.id)?;

        // Buffer the delete
        self.write_set.insert(record_id, WriteOp::Delete);
        Ok(true)
    }

    /// Validate the transaction for serializable isolation
    fn validate_serializable(&self) -> Result<(), TransactionError> {
        // SSI: Check for write-write conflicts on our read set
        // A conflict exists if another committed transaction wrote to a record we read
        // after our read timestamp

        // 1. Check if any record we read was modified
        let current_ts = self.version_store.current_timestamp();

        for record_id in &self.read_set {
            if let Some(version) = self
                .version_store
                .get_version_at(*record_id, current_ts, None)
            {
                if version.write_ts > self.read_ts && version.created_by.is_none() {
                    // Another committed transaction modified this record
                    return Err(TransactionError::SerializationFailure {
                        reason: format!("Record {} was modified by another transaction", record_id),
                    });
                }
            }
        }

        // 2. Check predicate lock conflicts (for phantom prevention)
        // This detects read-write and write-read conflicts on ranges
        let conflicts = self
            .version_store
            .predicate_lock_manager()
            .check_conflicts(self.id);
        if !conflicts.is_empty() {
            let (conflicting_tx, reason) = &conflicts[0];
            return Err(TransactionError::SerializationFailure {
                reason: format!(
                    "Predicate lock conflict with transaction {}: {}",
                    conflicting_tx, reason
                ),
            });
        }

        Ok(())
    }

    /// Commit helper that validates and applies writes
    fn do_commit(&mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Validate for serializable isolation
        if self.isolation == IsolationLevel::Serializable {
            self.validate_serializable()?;
        }

        // Get commit timestamp
        let commit_ts = self.version_store.advance_timestamp();

        // Apply writes to version store
        for (record_id, op) in &self.write_set {
            match op {
                WriteOp::Put { data } => {
                    let version = RecordVersion {
                        data: data.clone(),
                        write_ts: commit_ts,
                        delete_ts: u64::MAX,
                        created_by: None, // Committed now
                    };
                    self.version_store.add_version(*record_id, version);
                }
                WriteOp::Delete => {
                    self.version_store.mark_deleted(*record_id, commit_ts);
                }
            }
        }

        // Release locks (both write locks and predicate locks) and end transaction
        self.version_store.release_locks(self.id);
        self.version_store
            .predicate_lock_manager()
            .release_locks(self.id);
        self.version_store.end_transaction(self.id);
        self.state = TxState::Committed;

        Ok(())
    }

    /// Rollback helper
    fn do_rollback(&mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Release locks (both write locks and predicate locks) and end transaction
        self.version_store.release_locks(self.id);
        self.version_store
            .predicate_lock_manager()
            .release_locks(self.id);
        self.version_store.end_transaction(self.id);
        self.state = TxState::Aborted;

        Ok(())
    }
}

impl Transaction for AmorphicTransaction {
    fn id(&self) -> TxId {
        self.id
    }

    fn isolation_level(&self) -> IsolationLevel {
        self.isolation
    }

    fn state(&self) -> TxState {
        self.state
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        // Convert key to RecordId (assuming it's a u64)
        if key.len() != 8 {
            return Ok(None);
        }
        let record_id = u64::from_le_bytes(key.try_into().unwrap());

        // Get the record data
        let data = self.get_record(record_id)?;

        // Serialize to JSON
        Ok(data.map(|d| serde_json::to_vec(&d).unwrap_or_default()))
    }

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if key.len() != 8 {
            return Err(TransactionError::SerializationFailure {
                reason: "Key must be 8 bytes (RecordId)".to_string(),
            });
        }
        let record_id = u64::from_le_bytes(key.try_into().unwrap());

        // Deserialize value as JSON fields
        let data: HashMap<String, Value> =
            serde_json::from_slice(value).map_err(|e| TransactionError::SerializationFailure {
                reason: format!("Invalid JSON value: {}", e),
            })?;

        self.put_record(record_id, data)
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        if key.len() != 8 {
            return Ok(false);
        }
        let record_id = u64::from_le_bytes(key.try_into().unwrap());
        self.delete_record(record_id)
    }

    fn commit(mut self) -> Result<(), TransactionError> {
        self.do_commit()
    }

    fn rollback(mut self) -> Result<(), TransactionError> {
        self.do_rollback()
    }
}

/// Transaction manager for AmorphicStore
///
/// Creates and manages MVCC transactions.
pub struct AmorphicTransactionManager {
    /// Next transaction ID
    next_tx_id: AtomicU64,
    /// Shared version store
    version_store: Arc<VersionStore>,
    /// Reference to the underlying store
    store: Arc<RwLock<AmorphicStore>>,
}

impl AmorphicTransactionManager {
    /// Create a new transaction manager for an AmorphicStore
    pub fn new(store: AmorphicStore) -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
            version_store: Arc::new(VersionStore::new()),
            store: Arc::new(RwLock::new(store)),
        }
    }

    /// Create a manager from an existing Arc<RwLock<AmorphicStore>>
    pub fn with_shared_store(store: Arc<RwLock<AmorphicStore>>) -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
            version_store: Arc::new(VersionStore::new()),
            store,
        }
    }

    /// Get the next transaction ID
    fn next_id(&self) -> TxId {
        self.next_tx_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the current global timestamp
    pub fn current_timestamp(&self) -> Timestamp {
        self.version_store.current_timestamp()
    }

    /// Trigger garbage collection of old versions
    pub fn gc(&self) {
        self.version_store.gc();
    }

    /// Get the number of active transactions
    pub fn active_transaction_count(&self) -> usize {
        let active = self.version_store.active_transactions.read().unwrap();
        active.len()
    }
}

impl TransactionManager for AmorphicTransactionManager {
    type Tx = AmorphicTransaction;

    fn begin(&self) -> Result<Self::Tx, TransactionError> {
        self.begin_with_isolation(IsolationLevel::default())
    }

    fn begin_with_isolation(&self, level: IsolationLevel) -> Result<Self::Tx, TransactionError> {
        let id = self.next_id();
        Ok(AmorphicTransaction::new(
            id,
            level,
            Arc::clone(&self.version_store),
            Arc::clone(&self.store),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_store_basic() {
        let store = VersionStore::new();

        // Begin a transaction
        let tx_id = 1;
        let read_ts = store.begin_transaction(tx_id);
        assert!(read_ts >= 1);

        // No versions yet
        assert!(store.get_version_at(1, read_ts, None).is_none());

        // Add a version
        store.add_version(
            1,
            RecordVersion {
                data: HashMap::from([("name".to_string(), Value::String("Alice".to_string()))]),
                write_ts: read_ts,
                delete_ts: u64::MAX,
                created_by: None,
            },
        );

        // Now we can see it
        let version = store.get_version_at(1, read_ts, None).unwrap();
        assert_eq!(
            version.data.get("name"),
            Some(&Value::String("Alice".to_string()))
        );

        store.end_transaction(tx_id);
    }

    #[test]
    fn test_transaction_read_write() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Begin transaction
        let mut tx = manager.begin().unwrap();
        assert_eq!(tx.state(), TxState::Active);

        // Write a record
        let data = HashMap::from([
            ("name".to_string(), Value::String("Bob".to_string())),
            ("age".to_string(), Value::Int(30)),
        ]);
        tx.put_record(1, data.clone()).unwrap();

        // Read it back (from write buffer)
        let read_data = tx.get_record(1).unwrap().unwrap();
        assert_eq!(read_data.get("name"), data.get("name"));

        // Commit
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_rollback() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Begin and write
        let mut tx = manager.begin().unwrap();
        let data = HashMap::from([("name".to_string(), Value::String("Charlie".to_string()))]);
        tx.put_record(1, data).unwrap();

        // Rollback
        tx.rollback().unwrap();

        // Start new transaction - should not see the rolled back data
        let tx2 = manager.begin().unwrap();
        assert!(tx2.get_record(1).unwrap().is_none());
    }

    #[test]
    fn test_write_conflict() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Transaction 1 acquires lock
        let mut tx1 = manager.begin().unwrap();
        let data1 = HashMap::from([("name".to_string(), Value::String("David".to_string()))]);
        tx1.put_record(1, data1).unwrap();

        // Transaction 2 tries to write same record
        let mut tx2 = manager.begin().unwrap();
        let data2 = HashMap::from([("name".to_string(), Value::String("Eve".to_string()))]);
        let result = tx2.put_record(1, data2);

        // Should get write conflict
        assert!(matches!(
            result,
            Err(TransactionError::WriteConflict { .. })
        ));

        // Cleanup
        tx1.rollback().unwrap();
        tx2.rollback().unwrap();
    }

    #[test]
    fn test_snapshot_isolation() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Write initial data
        {
            let mut tx = manager.begin().unwrap();
            let data = HashMap::from([("value".to_string(), Value::Int(100))]);
            tx.put_record(1, data).unwrap();
            tx.commit().unwrap();
        }

        // Start a reader transaction
        let reader = manager
            .begin_with_isolation(IsolationLevel::Snapshot)
            .unwrap();

        // Modify the data in another transaction
        {
            let mut writer = manager.begin().unwrap();
            let data = HashMap::from([("value".to_string(), Value::Int(200))]);
            writer.put_record(1, data).unwrap();
            writer.commit().unwrap();
        }

        // Reader should still see the old value (snapshot isolation)
        let data = reader.get_record(1).unwrap().unwrap();
        assert_eq!(data.get("value"), Some(&Value::Int(100)));
    }

    #[test]
    fn test_delete_record() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Create a record
        {
            let mut tx = manager.begin().unwrap();
            let data = HashMap::from([("name".to_string(), Value::String("Frank".to_string()))]);
            tx.put_record(1, data).unwrap();
            tx.commit().unwrap();
        }

        // Delete it
        {
            let mut tx = manager.begin().unwrap();
            let deleted = tx.delete_record(1).unwrap();
            assert!(deleted);

            // Should not see it anymore in this transaction
            assert!(tx.get_record(1).unwrap().is_none());

            tx.commit().unwrap();
        }

        // New transaction should not see it either
        {
            let tx = manager.begin().unwrap();
            assert!(tx.get_record(1).unwrap().is_none());
        }
    }

    #[test]
    fn test_transaction_trait_impl() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        let mut tx = manager.begin().unwrap();

        // Test trait methods
        let key = 1u64.to_le_bytes();
        let value = serde_json::to_vec(&HashMap::<String, Value>::from([(
            "test".to_string(),
            Value::String("value".to_string()),
        )]))
        .unwrap();

        tx.put(&key, &value).unwrap();

        let read = tx.get(&key).unwrap().unwrap();
        let read_data: HashMap<String, Value> = serde_json::from_slice(&read).unwrap();
        assert_eq!(
            read_data.get("test"),
            Some(&Value::String("value".to_string()))
        );

        assert!(tx.exists(&key).unwrap());

        tx.commit().unwrap();
    }

    #[test]
    fn test_concurrent_readers() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Write some data
        {
            let mut tx = manager.begin().unwrap();
            tx.put_record(1, HashMap::from([("x".to_string(), Value::Int(42))]))
                .unwrap();
            tx.commit().unwrap();
        }

        // Multiple concurrent readers
        let tx1 = manager.begin().unwrap();
        let tx2 = manager.begin().unwrap();
        let tx3 = manager.begin().unwrap();

        // All should see the same data
        assert_eq!(
            tx1.get_record(1).unwrap().unwrap().get("x"),
            Some(&Value::Int(42))
        );
        assert_eq!(
            tx2.get_record(1).unwrap().unwrap().get("x"),
            Some(&Value::Int(42))
        );
        assert_eq!(
            tx3.get_record(1).unwrap().unwrap().get("x"),
            Some(&Value::Int(42))
        );

        // Check active count
        assert_eq!(manager.active_transaction_count(), 3);
    }

    #[test]
    fn test_predicate_lock_basic() {
        let manager = PredicateLockManager::new();

        // Record a range read
        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);

        // Should have one lock
        assert_eq!(manager.lock_count(), 1);

        // Record a range write from another transaction
        manager.record_range_write("users", "age", 25.0, 35.0, 2, 101);

        // Should have two locks
        assert_eq!(manager.lock_count(), 2);

        // Check conflicts for tx 1 (read overlaps with tx 2's write)
        let conflicts = manager.check_conflicts(1);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, 2);

        // Check conflicts for tx 2 (write overlaps with tx 1's read)
        let conflicts = manager.check_conflicts(2);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, 1);
    }

    #[test]
    fn test_predicate_lock_no_conflict_non_overlapping() {
        let manager = PredicateLockManager::new();

        // Read range [0, 10)
        manager.record_range_read("users", "age", 0.0, 10.0, 1, 100);

        // Write range [20, 30) - no overlap
        manager.record_range_write("users", "age", 20.0, 30.0, 2, 101);

        // No conflicts
        let conflicts1 = manager.check_conflicts(1);
        assert!(conflicts1.is_empty());

        let conflicts2 = manager.check_conflicts(2);
        assert!(conflicts2.is_empty());
    }

    #[test]
    fn test_predicate_lock_no_conflict_read_read() {
        let manager = PredicateLockManager::new();

        // Both transactions read the same range - no conflict
        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);
        manager.record_range_read("users", "age", 25.0, 35.0, 2, 101);

        let conflicts1 = manager.check_conflicts(1);
        assert!(conflicts1.is_empty());

        let conflicts2 = manager.check_conflicts(2);
        assert!(conflicts2.is_empty());
    }

    #[test]
    fn test_predicate_lock_different_tables() {
        let manager = PredicateLockManager::new();

        // Read from "users" table
        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);

        // Write to "products" table - same range but different table
        manager.record_range_write("products", "age", 18.0, 30.0, 2, 101);

        // No conflicts because different tables
        let conflicts = manager.check_conflicts(1);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_predicate_lock_different_fields() {
        let manager = PredicateLockManager::new();

        // Read on "age" field
        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);

        // Write on "salary" field - same table but different field
        manager.record_range_write("users", "salary", 18.0, 30.0, 2, 101);

        // No conflicts because different fields
        let conflicts = manager.check_conflicts(1);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_predicate_lock_release() {
        let manager = PredicateLockManager::new();

        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);
        manager.record_range_write("users", "age", 25.0, 35.0, 2, 101);

        assert_eq!(manager.lock_count(), 2);

        // Release tx 1's locks
        manager.release_locks(1);

        assert_eq!(manager.lock_count(), 1);

        // No more conflicts for tx 2
        let conflicts = manager.check_conflicts(2);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_predicate_lock_cleanup() {
        let manager = PredicateLockManager::new();

        manager.record_range_read("users", "age", 18.0, 30.0, 1, 100);
        manager.record_range_write("users", "age", 25.0, 35.0, 2, 200);

        assert_eq!(manager.lock_count(), 2);

        // Cleanup locks older than timestamp 150
        manager.cleanup_old_locks(150);

        // Only the lock from tx 2 (timestamp 200) should remain
        assert_eq!(manager.lock_count(), 1);
    }

    #[test]
    fn test_transaction_record_range_read() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        let mut tx = manager
            .begin_with_isolation(IsolationLevel::Serializable)
            .unwrap();

        // Record range reads
        tx.record_range_read("users", "age", 18.0, 65.0);
        tx.record_range_read("users", "salary", 50000.0, 100000.0);

        // Should have 2 range reads recorded
        assert_eq!(tx.range_reads.len(), 2);
        assert_eq!(
            tx.range_reads[0],
            ("users".to_string(), "age".to_string(), 18.0, 65.0)
        );
        assert_eq!(
            tx.range_reads[1],
            ("users".to_string(), "salary".to_string(), 50000.0, 100000.0)
        );
    }

    #[test]
    fn test_ssi_predicate_lock_conflict_detection() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Transaction 1: Read a range
        let mut tx1 = manager
            .begin_with_isolation(IsolationLevel::Serializable)
            .unwrap();
        tx1.record_range_read("users", "age", 18.0, 30.0);

        // Transaction 2: Write to overlapping range
        let mut tx2 = manager
            .begin_with_isolation(IsolationLevel::Serializable)
            .unwrap();
        tx2.record_range_write("users", "age", 25.0, 35.0);

        // Both should detect conflicts when trying to commit
        // The first one to commit succeeds, the second fails
        // Since tx2 has a write that conflicts with tx1's read,
        // and tx1 has a read that conflicts with tx2's write,
        // both detect the conflict

        let result1 = tx1.commit();
        // Tx1 should fail because tx2's write conflicts with its read
        assert!(result1.is_err());
        if let Err(TransactionError::SerializationFailure { reason }) = result1 {
            assert!(reason.contains("Predicate lock conflict"));
        }
    }

    #[test]
    fn test_ssi_no_conflict_after_commit() {
        let amorphic = AmorphicStore::new();
        let manager = AmorphicTransactionManager::new(amorphic);

        // Transaction 1: Read and commit
        {
            let mut tx1 = manager
                .begin_with_isolation(IsolationLevel::Serializable)
                .unwrap();
            tx1.record_range_read("users", "age", 18.0, 30.0);
            tx1.commit().unwrap();
        }

        // Transaction 2: Write to same range - should succeed because tx1 is done
        {
            let mut tx2 = manager
                .begin_with_isolation(IsolationLevel::Serializable)
                .unwrap();
            tx2.record_range_write("users", "age", 25.0, 35.0);
            // No conflict because tx1's locks were released on commit
            tx2.commit().unwrap();
        }
    }

    #[test]
    fn test_predicate_lock_write_write_conflict() {
        let manager = PredicateLockManager::new();

        // Both transactions write to overlapping ranges
        manager.record_range_write("users", "age", 18.0, 30.0, 1, 100);
        manager.record_range_write("users", "age", 25.0, 35.0, 2, 101);

        // Should detect conflict
        let conflicts = manager.check_conflicts(1);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, 2);
    }
}
