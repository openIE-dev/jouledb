//! GPU acceleration for joule-db-browser

pub mod btree_gpu;

pub use btree_gpu::{GpuBTreeOps, GpuBTreeRangeScan};
