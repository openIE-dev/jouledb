//! Storage abstractions for JouleDB
//!
//! This module provides the core storage traits and types that all
//! storage backends must implement.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │           Engine Layer              │
//! ├─────────────────────────────────────┤
//! │      StorageBackend Trait           │
//! ├──────────┬──────────┬───────────────┤
//! │ Memory   │  File    │  IndexedDB    │
//! │ Backend  │  Backend │  Backend      │
//! └──────────┴──────────┴───────────────┘
//! ```
//!
//! ## Backends
//!
//! - `MemoryBackend` - In-memory storage for testing
//! - File backends (in joule-db-local)
//! - Browser backends (in joule-db-browser)

pub mod buffer;
pub mod disk;
pub mod memory;
pub mod memory64;
/// Zero-copy mmap for extent-based large blob reads (LLM weights).
pub mod mmap_extent;
pub mod page;
mod traits;
pub mod vfs;

pub use memory64::{Memory64Backend, Memory64Config, Memory64PageAllocator, Memory64PageId};
pub use page::{
    DEFAULT_PAGE_SIZE, NULL_PAGE_ID, PAGE_HEADER_SIZE, Page, PageFlags, PageHeader, PageId,
    PageType,
};
pub use traits::{CommittedMeta, StorageBackend, StorageStats};
pub use vfs::{MemoryFile, MemoryVfs, StdFile, StdVfs, VirtualFile, VirtualFileSystem};
