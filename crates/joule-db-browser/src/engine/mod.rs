//! Engine module for browser backends
//!
//! Contains:
//! - AsyncEngine: Basic async key-value operations
//! - btree_async: Full B-tree implementation (future)

pub mod btree_async;

pub use crate::engine::AsyncEngine;

