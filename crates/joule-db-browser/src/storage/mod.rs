//! Browser storage backends

pub mod batch;
pub mod indexeddb;

// Re-exports
pub use batch::{BatchConfig, BatchReader, BatchStats, IndexedDBBatcher};
