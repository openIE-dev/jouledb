//! Error types for JouleDB Core
//!
//! All errors are platform-agnostic and provide detailed context.
//!
//! # Error Categories
//!
//! - [`StorageError`]: Low-level storage operations (pages, blocks, I/O)
//! - [`TransactionError`]: Transaction lifecycle and concurrency
//! - [`IndexError`]: B-tree and learned index operations
//! - [`CodecError`]: Serialization and deserialization
//! - [`QueryError`]: SQL/Cypher parsing and execution
//! - [`ReplicationError`]: Distributed system operations
//! - [`EngineError`]: Core engine operations
//!
//! # Error Context
//!
//! All error types include rich context for debugging:
//! - Operation being performed
//! - Relevant identifiers (page IDs, transaction IDs, etc.)
//! - Expected vs actual values where applicable

use std::fmt;

/// Result type alias using JouleDB Error
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for JouleDB operations
#[derive(Debug)]
pub enum Error {
    /// Storage-related errors
    Storage(StorageError),
    /// Transaction errors
    Transaction(TransactionError),
    /// Index errors
    Index(IndexError),
    /// Encoding/decoding errors
    Codec(CodecError),
    /// Query errors
    Query(QueryError),
    /// Replication errors
    Replication(ReplicationError),
    /// Engine errors
    Engine(EngineError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Storage(e) => write!(f, "Storage error: {}", e),
            Error::Transaction(e) => write!(f, "Transaction error: {}", e),
            Error::Index(e) => write!(f, "Index error: {}", e),
            Error::Codec(e) => write!(f, "Codec error: {}", e),
            Error::Query(e) => write!(f, "Query error: {}", e),
            Error::Replication(e) => write!(f, "Replication error: {}", e),
            Error::Engine(e) => write!(f, "Engine error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Storage(e) => Some(e),
            Error::Transaction(e) => Some(e),
            Error::Index(e) => Some(e),
            Error::Codec(e) => Some(e),
            Error::Query(e) => Some(e),
            Error::Replication(e) => Some(e),
            Error::Engine(e) => Some(e),
        }
    }
}

// Conversions
impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        Error::Storage(e)
    }
}

impl From<TransactionError> for Error {
    fn from(e: TransactionError) -> Self {
        Error::Transaction(e)
    }
}

impl From<IndexError> for Error {
    fn from(e: IndexError) -> Self {
        Error::Index(e)
    }
}

impl From<CodecError> for Error {
    fn from(e: CodecError) -> Self {
        Error::Codec(e)
    }
}

impl From<EngineError> for Error {
    fn from(e: EngineError) -> Self {
        Error::Engine(e)
    }
}

impl From<QueryError> for Error {
    fn from(e: QueryError) -> Self {
        Error::Query(e)
    }
}

impl From<ReplicationError> for Error {
    fn from(e: ReplicationError) -> Self {
        Error::Replication(e)
    }
}

/// Storage-related errors
#[derive(Debug)]
pub enum StorageError {
    /// Page not found
    PageNotFound {
        /// The page ID that was not found
        page_id: u64,
    },
    /// Page data is corrupted
    Corrupted {
        /// The page ID
        page_id: u64,
        /// Description of corruption
        reason: String,
    },
    /// Checksum mismatch
    ChecksumMismatch {
        /// The page ID
        page_id: u64,
        /// Expected checksum
        expected: u32,
        /// Actual checksum
        actual: u32,
    },
    /// Storage is full
    OutOfSpace {
        /// Bytes requested
        requested: usize,
        /// Bytes available
        available: usize,
    },
    /// Page size exceeded
    PageSizeExceeded {
        /// Maximum page size
        max: usize,
        /// Actual size
        actual: usize,
    },
    /// Backend-specific error
    Backend(String),
    /// I/O error
    Io(String),
    /// Serialization error
    Serialization(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::PageNotFound { page_id } => {
                write!(f, "Page {} not found", page_id)
            }
            StorageError::Corrupted { page_id, reason } => {
                write!(f, "Page {} corrupted: {}", page_id, reason)
            }
            StorageError::ChecksumMismatch {
                page_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Page {} checksum mismatch: expected {}, got {}",
                    page_id, expected, actual
                )
            }
            StorageError::OutOfSpace {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of space: requested {} bytes, {} available",
                    requested, available
                )
            }
            StorageError::PageSizeExceeded { max, actual } => {
                write!(f, "Page size exceeded: max {}, actual {}", max, actual)
            }
            StorageError::Backend(msg) => write!(f, "Backend error: {}", msg),
            StorageError::Io(msg) => write!(f, "I/O error: {}", msg),
            StorageError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

/// Transaction-related errors
#[derive(Debug)]
pub enum TransactionError {
    /// Transaction not found
    NotFound {
        /// Transaction ID
        tx_id: u64,
    },
    /// Transaction already committed
    AlreadyCommitted {
        /// Transaction ID
        tx_id: u64,
    },
    /// Transaction already aborted
    AlreadyAborted {
        /// Transaction ID
        tx_id: u64,
    },
    /// Write-write conflict
    WriteConflict {
        /// The conflicting key
        key: Vec<u8>,
        /// Transaction that holds the lock
        holder_tx_id: u64,
    },
    /// Serialization failure (for serializable isolation)
    SerializationFailure {
        /// Reason for failure
        reason: String,
    },
    /// Transaction is read-only but attempted write
    ReadOnly,
    /// Transaction timed out
    Timeout,
    /// Deadlock detected
    Deadlock {
        /// Transactions involved
        transactions: Vec<u64>,
    },
}

impl fmt::Display for TransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionError::NotFound { tx_id } => {
                write!(f, "Transaction {} not found", tx_id)
            }
            TransactionError::AlreadyCommitted { tx_id } => {
                write!(f, "Transaction {} already committed", tx_id)
            }
            TransactionError::AlreadyAborted { tx_id } => {
                write!(f, "Transaction {} already aborted", tx_id)
            }
            TransactionError::WriteConflict { key, holder_tx_id } => {
                write!(
                    f,
                    "Write conflict on key {:?}, held by tx {}",
                    key, holder_tx_id
                )
            }
            TransactionError::SerializationFailure { reason } => {
                write!(f, "Serialization failure: {}", reason)
            }
            TransactionError::ReadOnly => write!(f, "Transaction is read-only"),
            TransactionError::Timeout => write!(f, "Transaction timed out"),
            TransactionError::Deadlock { transactions } => {
                write!(
                    f,
                    "Deadlock detected involving transactions: {:?}",
                    transactions
                )
            }
        }
    }
}

impl std::error::Error for TransactionError {}

/// Index-related errors
#[derive(Debug)]
pub enum IndexError {
    /// Key too large
    KeyTooLarge {
        /// Maximum key size
        max: usize,
        /// Actual key size
        actual: usize,
    },
    /// Value too large
    ValueTooLarge {
        /// Maximum value size
        max: usize,
        /// Actual value size
        actual: usize,
    },
    /// Index corrupted
    Corrupted {
        /// Description
        reason: String,
    },
    /// Duplicate key (for unique indexes)
    DuplicateKey {
        /// The duplicate key
        key: Vec<u8>,
    },
    /// Storage error
    Storage(StorageError),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::KeyTooLarge { max, actual } => {
                write!(f, "Key too large: max {} bytes, got {}", max, actual)
            }
            IndexError::ValueTooLarge { max, actual } => {
                write!(f, "Value too large: max {} bytes, got {}", max, actual)
            }
            IndexError::Corrupted { reason } => {
                write!(f, "Index corrupted: {}", reason)
            }
            IndexError::DuplicateKey { key } => {
                write!(f, "Duplicate key: {:?}", key)
            }
            IndexError::Storage(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for IndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IndexError::Storage(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StorageError> for IndexError {
    fn from(e: StorageError) -> Self {
        IndexError::Storage(e)
    }
}

/// Codec (encoding/decoding) errors
#[derive(Debug)]
pub enum CodecError {
    /// Unexpected end of data
    UnexpectedEof {
        /// Expected bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid data format
    InvalidFormat {
        /// Description
        reason: String,
    },
    /// Unknown type tag
    UnknownType {
        /// The unknown type tag
        tag: u8,
    },
    /// String is not valid UTF-8
    InvalidUtf8,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::UnexpectedEof { expected, actual } => {
                write!(
                    f,
                    "Unexpected EOF: expected {} bytes, got {}",
                    expected, actual
                )
            }
            CodecError::InvalidFormat { reason } => {
                write!(f, "Invalid format: {}", reason)
            }
            CodecError::UnknownType { tag } => {
                write!(f, "Unknown type tag: {}", tag)
            }
            CodecError::InvalidUtf8 => write!(f, "Invalid UTF-8 string"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Query-related errors (SQL, Cypher, etc.)
#[derive(Debug, Clone)]
pub enum QueryError {
    /// Query parsing failed
    ParseError {
        /// The query that failed to parse
        query: String,
        /// Position in query where error occurred
        position: usize,
        /// Description of the parse error
        message: String,
    },
    /// Table not found
    TableNotFound {
        /// Table name
        table: String,
    },
    /// Column not found
    ColumnNotFound {
        /// Column name
        column: String,
        /// Table name (if known)
        table: Option<String>,
    },
    /// Type mismatch in expression
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type
        actual: String,
        /// Expression that caused the mismatch
        expression: String,
    },
    /// Invalid expression
    InvalidExpression {
        /// Description
        reason: String,
    },
    /// Ambiguous column reference
    AmbiguousColumn {
        /// Column name
        column: String,
        /// Tables it could refer to
        tables: Vec<String>,
    },
    /// Subquery error
    SubqueryError {
        /// Description
        message: String,
    },
    /// Plan execution error
    ExecutionError {
        /// Description
        message: String,
    },
    /// Feature not supported
    Unsupported {
        /// Feature name
        feature: String,
    },
    /// Permission denied
    PermissionDenied {
        /// Operation attempted
        operation: String,
        /// Resource accessed
        resource: String,
    },
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::ParseError {
                query,
                position,
                message,
            } => {
                write!(
                    f,
                    "Parse error at position {}: {} (query: {})",
                    position,
                    message,
                    if query.len() > 50 {
                        &query[..50]
                    } else {
                        query
                    }
                )
            }
            QueryError::TableNotFound { table } => {
                write!(f, "Table '{}' not found", table)
            }
            QueryError::ColumnNotFound { column, table } => {
                if let Some(t) = table {
                    write!(f, "Column '{}' not found in table '{}'", column, t)
                } else {
                    write!(f, "Column '{}' not found", column)
                }
            }
            QueryError::TypeMismatch {
                expected,
                actual,
                expression,
            } => {
                write!(
                    f,
                    "Type mismatch: expected {}, got {} in expression '{}'",
                    expected, actual, expression
                )
            }
            QueryError::InvalidExpression { reason } => {
                write!(f, "Invalid expression: {}", reason)
            }
            QueryError::AmbiguousColumn { column, tables } => {
                write!(
                    f,
                    "Ambiguous column '{}' could refer to: {:?}",
                    column, tables
                )
            }
            QueryError::SubqueryError { message } => {
                write!(f, "Subquery error: {}", message)
            }
            QueryError::ExecutionError { message } => {
                write!(f, "Execution error: {}", message)
            }
            QueryError::Unsupported { feature } => {
                write!(f, "Feature not supported: {}", feature)
            }
            QueryError::PermissionDenied {
                operation,
                resource,
            } => {
                write!(f, "Permission denied: {} on {}", operation, resource)
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// Replication-related errors
#[derive(Debug, Clone)]
pub enum ReplicationError {
    /// Failed to connect to peer
    ConnectionFailed {
        /// Peer address
        peer: String,
        /// Reason
        reason: String,
    },
    /// Replication stream disconnected
    Disconnected {
        /// Peer ID
        peer_id: u64,
        /// Reason
        reason: String,
    },
    /// Log entry missing
    LogEntryMissing {
        /// The missing LSN
        lsn: u64,
    },
    /// Consensus failure
    ConsensusFailed {
        /// Term number
        term: u64,
        /// Reason
        reason: String,
    },
    /// Split brain detected
    SplitBrain {
        /// Partitions detected
        partitions: Vec<Vec<u64>>,
    },
    /// Leader election failed
    ElectionFailed {
        /// Term
        term: u64,
        /// Reason
        reason: String,
    },
    /// Quorum not reached
    QuorumNotReached {
        /// Votes received
        votes: u32,
        /// Quorum size
        quorum: u32,
    },
    /// Snapshot transfer failed
    SnapshotFailed {
        /// Reason
        reason: String,
    },
    /// Configuration change error
    ConfigurationError {
        /// Description
        message: String,
    },
    /// Protocol error
    ProtocolError {
        /// Description
        message: String,
    },
    /// I/O error
    IoError(String),
    /// Timeout
    Timeout {
        /// Operation that timed out
        operation: String,
    },
}

impl fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplicationError::ConnectionFailed { peer, reason } => {
                write!(f, "Failed to connect to {}: {}", peer, reason)
            }
            ReplicationError::Disconnected { peer_id, reason } => {
                write!(f, "Peer {} disconnected: {}", peer_id, reason)
            }
            ReplicationError::LogEntryMissing { lsn } => {
                write!(f, "Log entry at LSN {} not found", lsn)
            }
            ReplicationError::ConsensusFailed { term, reason } => {
                write!(f, "Consensus failed in term {}: {}", term, reason)
            }
            ReplicationError::SplitBrain { partitions } => {
                write!(f, "Split brain detected: {} partitions", partitions.len())
            }
            ReplicationError::ElectionFailed { term, reason } => {
                write!(f, "Leader election failed in term {}: {}", term, reason)
            }
            ReplicationError::QuorumNotReached { votes, quorum } => {
                write!(f, "Quorum not reached: {} votes, need {}", votes, quorum)
            }
            ReplicationError::SnapshotFailed { reason } => {
                write!(f, "Snapshot transfer failed: {}", reason)
            }
            ReplicationError::ConfigurationError { message } => {
                write!(f, "Configuration error: {}", message)
            }
            ReplicationError::ProtocolError { message } => {
                write!(f, "Protocol error: {}", message)
            }
            ReplicationError::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            ReplicationError::Timeout { operation } => {
                write!(f, "Operation timed out: {}", operation)
            }
        }
    }
}

impl std::error::Error for ReplicationError {}

/// Engine-level errors
#[derive(Debug)]
pub enum EngineError {
    /// Engine not initialized
    NotInitialized,
    /// Invalid operation
    InvalidOperation {
        /// Description
        reason: String,
    },
    /// Internal error
    Internal {
        /// Description
        reason: String,
    },
    /// Lock acquisition failed (poisoned lock)
    LockPoisoned {
        /// Description of which lock failed
        lock_name: String,
    },
}

// ============================================================================
// Safe Lock Utilities
// ============================================================================

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Extension trait for safe lock acquisition on RwLock
pub trait RwLockExt<T: ?Sized> {
    /// Acquire a read lock, returning an error instead of panicking on poison
    fn safe_read(&self, lock_name: &str) -> Result<RwLockReadGuard<'_, T>>;

    /// Acquire a write lock, returning an error instead of panicking on poison
    fn safe_write(&self, lock_name: &str) -> Result<RwLockWriteGuard<'_, T>>;
}

impl<T: ?Sized> RwLockExt<T> for RwLock<T> {
    fn safe_read(&self, lock_name: &str) -> Result<RwLockReadGuard<'_, T>> {
        self.read().map_err(|_| {
            Error::Engine(EngineError::LockPoisoned {
                lock_name: lock_name.to_string(),
            })
        })
    }

    fn safe_write(&self, lock_name: &str) -> Result<RwLockWriteGuard<'_, T>> {
        self.write().map_err(|_| {
            Error::Engine(EngineError::LockPoisoned {
                lock_name: lock_name.to_string(),
            })
        })
    }
}

/// Extension trait for safe lock acquisition on Mutex
pub trait MutexExt<T: ?Sized> {
    /// Acquire a mutex lock, returning an error instead of panicking on poison
    fn safe_lock(&self, lock_name: &str) -> Result<MutexGuard<'_, T>>;
}

impl<T: ?Sized> MutexExt<T> for Mutex<T> {
    fn safe_lock(&self, lock_name: &str) -> Result<MutexGuard<'_, T>> {
        self.lock().map_err(|_| {
            Error::Engine(EngineError::LockPoisoned {
                lock_name: lock_name.to_string(),
            })
        })
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::NotInitialized => write!(f, "Engine not initialized"),
            EngineError::InvalidOperation { reason } => {
                write!(f, "Invalid operation: {}", reason)
            }
            EngineError::Internal { reason } => {
                write!(f, "Internal error: {}", reason)
            }
            EngineError::LockPoisoned { lock_name } => {
                write!(f, "Lock '{}' is poisoned", lock_name)
            }
        }
    }
}

impl std::error::Error for EngineError {}

// ============================================================================
// Error Context and Builder
// ============================================================================

/// Context that can be attached to errors for better debugging
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// Operation being performed
    pub operation: Option<String>,
    /// Component where error occurred
    pub component: Option<String>,
    /// Additional key-value pairs
    pub metadata: Vec<(String, String)>,
}

impl ErrorContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add operation context
    pub fn with_operation(mut self, op: impl Into<String>) -> Self {
        self.operation = Some(op.into());
        self
    }

    /// Add component context
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(ref op) = self.operation {
            parts.push(format!("during {}", op));
        }
        if let Some(ref comp) = self.component {
            parts.push(format!("in {}", comp));
        }
        for (k, v) in &self.metadata {
            parts.push(format!("{}={}", k, v));
        }
        write!(f, "{}", parts.join(", "))
    }
}

/// Error with attached context
#[derive(Debug)]
pub struct ContextualError {
    /// The underlying error
    pub error: Error,
    /// Context information
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.context.operation.is_some()
            || self.context.component.is_some()
            || !self.context.metadata.is_empty()
        {
            write!(f, "{} ({})", self.error, self.context)
        } else {
            write!(f, "{}", self.error)
        }
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Extension trait for adding context to errors
pub trait ErrorExt {
    /// Add context to an error
    fn with_context(self, ctx: ErrorContext) -> ContextualError;

    /// Add operation context
    fn during(self, operation: impl Into<String>) -> ContextualError;

    /// Add component context
    fn in_component(self, component: impl Into<String>) -> ContextualError;
}

impl ErrorExt for Error {
    fn with_context(self, context: ErrorContext) -> ContextualError {
        ContextualError {
            error: self,
            context,
        }
    }

    fn during(self, operation: impl Into<String>) -> ContextualError {
        ContextualError {
            error: self,
            context: ErrorContext::new().with_operation(operation),
        }
    }

    fn in_component(self, component: impl Into<String>) -> ContextualError {
        ContextualError {
            error: self,
            context: ErrorContext::new().with_component(component),
        }
    }
}

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    /// Add context to an error result
    fn with_context(self, ctx: ErrorContext) -> std::result::Result<T, ContextualError>;

    /// Add operation context
    fn during(self, operation: impl Into<String>) -> std::result::Result<T, ContextualError>;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, ctx: ErrorContext) -> std::result::Result<T, ContextualError> {
        self.map_err(|e| e.with_context(ctx))
    }

    fn during(self, operation: impl Into<String>) -> std::result::Result<T, ContextualError> {
        self.map_err(|e| e.during(operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::Storage(StorageError::PageNotFound { page_id: 42 });
        assert!(err.to_string().contains("42"));

        let err = Error::Transaction(TransactionError::WriteConflict {
            key: b"test".to_vec(),
            holder_tx_id: 1,
        });
        assert!(err.to_string().contains("conflict"));
    }

    #[test]
    fn test_error_conversion() {
        let storage_err = StorageError::PageNotFound { page_id: 1 };
        let err: Error = storage_err.into();
        assert!(matches!(err, Error::Storage(_)));
    }
}
