//! Browser Persistence Backends for JouleDB
//!
//! Complete persistence layer supporting multiple browser storage backends:
//! - IndexedDB: Browser-native key-value store with transaction support
//! - OPFS: Origin Private File System for high-performance file access
//! - Memory: In-memory storage for testing and transient data
//!
//! Features:
//! - Async operations with browser APIs
//! - Page-based storage with checksums for data integrity
//! - Optional compression and encryption
//! - WAL (Write-Ahead Logging) for crash recovery

mod config;
mod encryption;
mod error;
mod indexeddb;
mod memory;
mod migration;
mod opfs;
mod opfs_vbd;
mod oplog;
mod page;
mod wal;
mod zero_copy;

pub use config::{BackendSelection, BackendType, PersistenceConfig};
pub use encryption::OpfsEncryption;
pub use error::PersistenceError;
pub use indexeddb::IndexedDBBackend;
pub use memory::MemoryBackend;
pub use migration::{MigrationResult, migrate_to_vbd};
pub use opfs::OPFSBackend;
pub use opfs_vbd::OPFSVirtualBlockDevice;
pub use oplog::OpfsOplogTailer;
pub use page::{PAGE_DATA_SIZE, PAGE_HEADER_SIZE, PAGE_SIZE, PageFlags, SerializedPage};
pub use wal::{WalEntry, WalManager};
pub use zero_copy::{CacheStats, ZeroCopyBackend, ZeroCopyPageCache};

use joule_db_core::storage::PageId;

/// Storage statistics for monitoring and debugging
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total pages stored
    pub total_pages: u64,
    /// Total bytes used
    pub bytes_used: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Write operations
    pub write_ops: u64,
    /// Read operations
    pub read_ops: u64,
    /// Sync operations
    pub sync_ops: u64,
    /// WAL entries pending
    pub wal_pending: u64,
}

/// Dynamic backend type alias ensuring Send+Sync only on non-WASM
#[cfg(target_arch = "wasm32")]
pub type DynBackend = dyn AsyncPersistenceBackend;

#[cfg(not(target_arch = "wasm32"))]
pub type DynBackend = dyn AsyncPersistenceBackend + Send + Sync;

/// Async persistence backend trait for browser environments
#[cfg(target_arch = "wasm32")]
pub trait AsyncPersistenceBackend {
    /// Read a page asynchronously
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + 'static>,
    >;

    /// Write a page asynchronously
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>;

    /// Sync to storage
    fn sync(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Async persistence backend trait for non-WASM environments (requires Send + Sync)
#[cfg(not(target_arch = "wasm32"))]
pub trait AsyncPersistenceBackend: Send + Sync {
    /// Read a page asynchronously
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + Send + 'static>,
    >;

    /// Write a page asynchronously
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    >;

    /// Sync to storage
    fn sync(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    >;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Auto-select the best available backend
pub fn auto_select_backend() -> BackendSelection {
    #[cfg(target_arch = "wasm32")]
    {
        // Check for OPFS support first (better performance)
        // Prefer VBD (Virtual Block Device) for maximum performance
        if OPFSBackend::is_available() {
            // Prefer VBD if available (always use it for best performance)
            return BackendSelection {
                backend_type: BackendType::OPFSVBD,
                reason: "OPFS Virtual Block Device available for maximum performance".to_string(),
            };
        }

        // Fall back to IndexedDB
        if is_indexeddb_available() {
            return BackendSelection {
                backend_type: BackendType::IndexedDB,
                reason: "IndexedDB available".to_string(),
            };
        }
    }

    // Default to memory
    BackendSelection {
        backend_type: BackendType::Memory,
        reason: "Using in-memory storage".to_string(),
    }
}

#[cfg(target_arch = "wasm32")]
fn is_indexeddb_available() -> bool {
    web_sys::window()
        .and_then(|w| w.indexed_db().ok())
        .flatten()
        .is_some()
}

#[cfg(not(target_arch = "wasm32"))]
fn is_indexeddb_available() -> bool {
    false
}
