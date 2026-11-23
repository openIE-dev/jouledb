//! # JouleDB Browser
//!
//! Browser/WASM bindings for JouleDB with IndexedDB and OPFS storage backends.
//!
//! ## Features
//!
//! - `indexeddb` (default) - IndexedDB storage backend
//! - `opfs` - Origin Private File System storage backend
//!
//! ## Usage
//!
//! ```javascript
//! import { JouleDB } from 'joule-db';
//!
//! // Open with auto-selected backend (OPFS if available, else IndexedDB)
//! const db = await JouleDB.open('mydb');
//!
//! // Or explicitly choose backend
//! const db = await JouleDB.openWithBackend('mydb', 'opfs');
//!
//! await db.put('key', 'value');
//! const value = await db.get('key');
//! ```

pub mod apis;
#[path = "engine.rs"]
mod engine_impl;
pub mod ipc;
pub mod persistence;
pub mod storage;
pub mod engine {
    pub use super::engine_impl::*;
}
pub mod engine_enhanced;

// Single WASM entry point - all bindings consolidated here
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "deterministic")]
pub mod deterministic;
pub mod gpu;
pub mod htap;
#[cfg(feature = "learned")]
pub mod learned;
pub mod monitoring;
#[cfg(feature = "novel")]
pub mod novel;
pub mod resilience;
#[cfg(feature = "viz")]
pub mod viz;

// Re-export WASM bindings (only in WASM builds)
#[cfg(target_arch = "wasm32")]
pub use wasm::{
    EnhancedJouleDB, IsolationLevel, JouleDB, QueryResult, RangeBound, RangeEntry, RangeIterator,
    ScanDir,
};

// Feature-gated WASM bindings
#[cfg(target_arch = "wasm32")]
pub use wasm::HTAPStorage;

#[cfg(all(target_arch = "wasm32", feature = "learned"))]
pub use wasm::LearnedIndex;

#[cfg(all(target_arch = "wasm32", feature = "deterministic"))]
pub use wasm::{DeterministicTransactionManager, MultiTabCoordinator};

// Re-exports (internal types, no WASM bindings)
pub use persistence::{
    AsyncPersistenceBackend, BackendSelection, BackendType, CacheStats, IndexedDBBackend,
    MemoryBackend as BrowserMemoryBackend, MigrationResult, OPFSVirtualBlockDevice, OpfsEncryption,
    OpfsOplogTailer, PAGE_DATA_SIZE, PAGE_HEADER_SIZE, PAGE_SIZE, PageFlags, PersistenceConfig,
    PersistenceError, SerializedPage, StorageStats, WalEntry, WalManager, ZeroCopyBackend,
    ZeroCopyPageCache, auto_select_backend, migrate_to_vbd,
};

pub use storage::indexeddb::IndexedDbBackend;
pub use storage::{BatchConfig, BatchReader, BatchStats, IndexedDBBatcher};

// Browser API re-exports (internal types, no WASM bindings)
pub use apis::{
    BackgroundSyncManager, BeaconManager, BroadcastChannelManager, CacheManager,
    NavigatorStorageManager, PeriodicBackgroundSyncManager, PushManager, ServiceWorkerManager,
    WebLocksManager,
};

// IPC re-exports (internal types, no WASM bindings)
pub use ipc::{
    ArrowIpcBridge, SharedRingBuffer, SharedRingBufferWorker, WorkerBridge, get_worker_code,
};

// Re-export AsyncEngine (internal type, no WASM bindings)
pub use engine::AsyncEngine;

// Re-export enhanced features (internal types, no WASM bindings)
pub use engine_enhanced::EnhancedAsyncEngine;

// Re-export monitoring (internal types, no WASM bindings)
pub use monitoring::{DevToolsIntegration, Metrics, PerformanceMonitor};

// Re-export resilience (internal types, no WASM bindings)
pub use resilience::{CircuitBreaker, CircuitState, GracefulDegradation, RetryPolicy};

// Re-export HTAP (internal types, no WASM bindings)
pub use htap::{HTAPConfig, HTAPStorage as InternalHTAPStorage};

// Re-export learned indexes (internal types, no WASM bindings)
#[cfg(feature = "learned")]
pub use learned::{LearnedIndex as InternalLearnedIndex, LearnedIndexConfig};

// Re-export deterministic execution (internal types, no WASM bindings)
#[cfg(feature = "deterministic")]
pub use deterministic::{
    DeterministicTransactionManager as InternalDeterministicTransactionManager,
    MultiTabCoordinator as InternalMultiTabCoordinator,
};

// Re-export visualization (internal types, no WASM bindings)
