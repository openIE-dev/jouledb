//! B-tree storage engine
//!
//! The core storage engine for JouleDB, implementing a B-tree based
//! key-value store with ACID transactions.

mod btree;

pub use btree::BTreeRangeIterator;
pub use btree::Engine;
pub use btree::EngineConfig;
pub use btree::WriteTransaction;
