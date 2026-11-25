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
pub mod elle_hermitage;
mod latch;
mod lockfree;
mod mvcc;
mod optimistic;

pub use buffer_pool::{BufferPool, BufferPoolConfig};
pub use elle_hermitage::{
    AdyaEdge, AdyaEdgeKind, AnomalyWitness, ElleCheckReport, ElleOp, ElleTxn, ElleTxnStatus,
    HermitageSuiteReceipt, IsolationAnomalyKind, analyze_elle_history, run_hermitage_ssi_suite,
    to_elle_edn,
};
pub use latch::{LatchManager, LatchManagerConfig, LatchMode, LatchStats, MultiPageLatch};
pub use lockfree::{LockFreeCounter, LockFreeSize, LockFreeStats};
pub use mvcc::{
    IsolationLevel, MvccStats, MvccStore, MvccTransaction, MvccTransactionManager, Version,
    VersionChain, advance_mvcc_timestamp, decode_record_id, encode_record_key, encode_table_prefix,
};
pub use optimistic::{OccTransaction, OccTransactionManager};
