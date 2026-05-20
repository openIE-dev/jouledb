//! # JouleDB Core
//!
//! Platform-agnostic database engine providing:
//! - B-tree storage engine with ACID transactions
//! - Pluggable storage backends
//! - Index abstractions (B-tree, Hash)
//! - Type system and serialization
//! - Persistence infrastructure (WAL, snapshots, recovery)
//! - Network protocol abstractions
//! - Compute backend abstractions (CPU/GPU/NPU)
//!
//! ## Design Principles
//!
//! 1. **Zero platform dependencies** - No wasm_bindgen, tokio, or platform-specific code
//! 2. **Trait-based abstractions** - Storage, transactions, indexes, persistence are all traits
//! 3. **Correctness first** - ACID guarantees over raw performance
//! 4. **Cross-platform** - Works on native, browser (WASM), and embedded (MCU)
//! 5. **GPU-native** - Designed from the ground up for hardware acceleration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      JouleDB Ecosystem                          │
//! ├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
//! │   Browser   │    MCU      │   Desktop   │   Server    │  Edge  │
//! │   (WASM)    │  (ARM/RISC) │ (Win/Mac/Ln)│  (Linux)    │  (IoT) │
//! ├─────────────┴─────────────┴─────────────┴─────────────┴────────┤
//! │                    Unified Query Layer                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │              GPU/NPU/TPU Acceleration Layer                     │
//! │         (WebGPU | CUDA | Metal | Vulkan | SIMD)                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                   Persistence Layer                             │
//! │              (WAL | Snapshots | Recovery)                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_core::{Database, MemoryBackend};
//!
//! // Create in-memory database
//! let backend = MemoryBackend::new();
//! let db = Database::open(backend)?;
//!
//! // Simple key-value operations
//! db.put(b"key", b"value")?;
//! let value = db.get(b"key")?;
//!
//! // Transactions
//! let tx = db.begin()?;
//! tx.put(b"key1", b"value1")?;
//! tx.put(b"key2", b"value2")?;
//! tx.commit()?;
//! ```

// deny(unsafe_code) instead of forbid — allows targeted #[allow(unsafe_code)]
// in mmap_extent.rs where unsafe is required for memory-mapped I/O.
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod allocator;
pub mod catalog;
pub mod concurrency;
pub mod encryption;
pub mod engine;
pub mod error;
pub mod index;
pub mod persistence;
pub mod query;
pub mod resilience;
pub mod snapshot;
pub mod storage;
pub mod temporal;
pub mod tx;
pub mod types;

#[cfg(test)]
mod proptest_verify;

#[cfg(kani)]
mod kani_proofs;

// Re-exports for convenience
pub use allocator::{FrameArena, PoolAllocator, TransactionArena};
pub use concurrency::{LatchManager, LatchManagerConfig};
pub use engine::Engine;
pub use engine::EngineConfig;
pub use engine::WriteTransaction;
pub use snapshot::Snapshot;
pub use error::{
    CodecError, ContextualError, EngineError, Error, ErrorContext, ErrorExt, IndexError, MutexExt,
    QueryError, ReplicationError, Result, ResultExt, RwLockExt, StorageError, TransactionError,
};
pub use query::{QueryHandle, TripleBufferedQueries, TripleBufferedQueryManager};
pub use storage::buffer::{BufferPool, BufferPoolConfig};
pub use storage::disk::DiskBackend;
pub use storage::memory::MemoryBackend;
pub use storage::{
    DEFAULT_PAGE_SIZE, NULL_PAGE_ID, Page, PageId, PageType, StorageBackend, StorageStats,
};
pub use storage::{Memory64Backend, Memory64Config, Memory64PageAllocator};
pub use tx::{IsolationLevel, Transaction};
pub use types::Value;

// Catalog re-exports
pub use catalog::{
    Catalog, ColumnDef, DataType as CatalogDataType, IndexDef, IndexType, TableSchema,
};

// Encryption re-exports
pub use encryption::{EncryptedBackend, EncryptionConfig, EncryptionStats, KeyId, KeyManager};

// Resilience re-exports
pub use resilience::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, RetryPolicy, RetryResult,
    RetryableError, TimeoutConfig, TimeoutError,
};

// Persistence re-exports
pub use persistence::{
    BufferHandle,
    BufferUsage,
    // Compute
    ComputeBackend,
    ComputeContext,
    ComputeOp,
    ComputeResult,
    ConnectionState,
    ConnectionStats,
    CpuComputeBackend,
    DeviceCapabilities,
    DeviceType,
    DurabilityPolicy,
    ErrorCode,
    // WAL types
    LSN,
    // Network protocol
    Message,
    MessageFlags,
    NetworkBackend,
    NetworkListener,
    OpCode,
    PersistenceConfig,
    PersistenceManager,
    RecoveryResult,
    RecoveryStrategy,
    SnapshotBackend,
    // Snapshot types
    SnapshotMetadata,
    TxId,
    WalBackend,
    WalEntry,
    WalEntryType,
};
