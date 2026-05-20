//! Concurrency primitives for JouleDB
//!
//! This module provides high-performance concurrency utilities:
//! - `BufferPool`: LRU cache with Arc-based sharing to reduce cloning
//! - `LatchManager`: Per-page latching for fine-grained concurrency
//! - `Optimistic`: Optimistic Concurrency Control (OCC) for high concurrency
//! - `MVCC`: Multi-Version Concurrency Control with version chains
//! - `LockFree`: Lock-free data structures for hot paths
//! - Sharded locking patterns for reduced contention

mod buffer_pool;
mod latch;
mod lockfree;
mod mvcc;
mod optimistic;

pub use buffer_pool::{BufferPool, BufferPoolConfig};
pub use latch::{LatchManager, LatchManagerConfig, LatchMode, LatchStats, MultiPageLatch};
pub use lockfree::{LockFreeCounter, LockFreeSize, LockFreeStats};
pub use mvcc::{
    MvccStats, MvccStore, MvccTransaction, MvccTransactionManager, Version, VersionChain,
    decode_record_id, encode_record_key, encode_table_prefix,
};
pub use optimistic::{OccTransaction, OccTransactionManager};
