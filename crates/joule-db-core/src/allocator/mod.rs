//! Memory allocators for transaction-scoped and frame-scoped allocations
//!
//! Provides arena allocators and pool allocators to eliminate GC pauses
//! and ensure predictable memory usage.

pub mod arena;

pub use arena::{FrameArena, PoolAllocator, TransactionArena};
