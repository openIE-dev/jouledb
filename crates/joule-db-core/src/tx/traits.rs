//! Transaction traits and types

use crate::error::TransactionError;

/// Transaction ID
pub type TxId = u64;

/// Transaction isolation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Read uncommitted - dirty reads possible
    ReadUncommitted,
    /// Read committed - no dirty reads (default)
    #[default]
    ReadCommitted,
    /// Repeatable read - consistent reads within transaction
    RepeatableRead,
    /// Serializable - full isolation
    Serializable,
    /// Snapshot isolation - MVCC-based
    Snapshot,
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    /// Transaction is active and can perform operations
    Active,
    /// Transaction has been committed
    Committed,
    /// Transaction has been aborted/rolled back
    Aborted,
}

/// Transaction trait
///
/// Represents an active database transaction. Operations within a transaction
/// are isolated according to the isolation level and either all succeed
/// (commit) or all fail (rollback).
pub trait Transaction: Send {
    /// Get the transaction ID
    fn id(&self) -> TxId;

    /// Get the isolation level
    fn isolation_level(&self) -> IsolationLevel;

    /// Get current transaction state
    fn state(&self) -> TxState;

    /// Read a value by key
    ///
    /// Returns the value if found, or None if the key doesn't exist.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError>;

    /// Write a key-value pair
    ///
    /// Creates or updates the key with the given value.
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError>;

    /// Delete a key
    ///
    /// Returns true if the key existed and was deleted, false if it didn't exist.
    fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError>;

    /// Commit the transaction
    ///
    /// All changes become permanent and visible to other transactions.
    fn commit(self) -> Result<(), TransactionError>;

    /// Rollback/abort the transaction
    ///
    /// All changes are discarded.
    fn rollback(self) -> Result<(), TransactionError>;

    /// Check if key exists
    fn exists(&self, key: &[u8]) -> Result<bool, TransactionError> {
        Ok(self.get(key)?.is_some())
    }
}

/// Read-only transaction trait
///
/// A transaction that only allows read operations. Useful for ensuring
/// a consistent view of the data without any risk of modifications.
pub trait ReadTransaction: Send {
    /// Get the transaction ID
    fn id(&self) -> TxId;

    /// Read a value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError>;

    /// Check if key exists
    fn exists(&self, key: &[u8]) -> Result<bool, TransactionError> {
        Ok(self.get(key)?.is_some())
    }
}

/// Transaction manager trait
///
/// Creates and manages transactions.
pub trait TransactionManager: Send + Sync {
    /// The transaction type produced by this manager
    type Tx: Transaction;

    /// Begin a new transaction with default isolation level
    fn begin(&self) -> Result<Self::Tx, TransactionError>;

    /// Begin a transaction with specific isolation level
    fn begin_with_isolation(&self, level: IsolationLevel) -> Result<Self::Tx, TransactionError>;
}
