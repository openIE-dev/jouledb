//! Storage backends for edge devices
//!
//! This module provides pluggable storage backends:
//! - `FlashBackend`: For NOR/NAND flash with wear leveling
//! - `FileBackend`: For filesystem-based storage (Yocto/Linux)
//!
//! # Abstraction
//! All backends implement the `EdgeStorageBackend` trait, enabling
//! the database to work seamlessly across different storage types.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "std")]
mod file;
pub mod flash;

#[cfg(feature = "std")]
pub use file::FileBackend;
pub use flash::FlashBackend;

use crate::storage::StorageError;

/// Edge storage backend trait
///
/// Provides a unified interface for different storage types:
/// - Flash memory (with wear leveling)
/// - File system (with mmap and WAL)
///
/// # Page-Based Design
/// Storage is organized into fixed-size pages for:
/// - Alignment with flash erase blocks
/// - Efficient I/O operations
/// - Power-loss safety
pub trait EdgeStorageBackend: Send + Sync {
    /// Read a page from storage
    ///
    /// # Arguments
    /// * `page_id` - The page identifier
    ///
    /// # Returns
    /// The page data or an error
    fn read_page(&self, page_id: u64) -> Result<Vec<u8>, StorageError>;

    /// Write a page to storage
    ///
    /// # Arguments
    /// * `page_id` - The page identifier
    /// * `data` - The data to write (must fit in a page)
    ///
    /// # Power Safety
    /// Implementations should ensure writes are atomic or use journaling
    fn write_page(&mut self, page_id: u64, data: &[u8]) -> Result<(), StorageError>;

    /// Sync all pending writes to persistent storage
    fn sync(&mut self) -> Result<(), StorageError>;

    /// Get the page size for this backend
    fn page_size(&self) -> usize;

    /// Get total storage capacity in bytes
    fn capacity(&self) -> u64;

    /// Get available space in bytes
    fn available(&self) -> u64;

    /// Delete a page (mark as free)
    fn delete_page(&mut self, page_id: u64) -> Result<(), StorageError>;

    /// Check if a page exists
    fn page_exists(&self, page_id: u64) -> bool;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Number of write operations
    pub write_ops: u64,
    /// Number of read operations
    pub read_ops: u64,
    /// Number of sync operations
    pub sync_ops: u64,
    /// Number of pages used
    pub pages_used: u64,
    /// Number of pages free
    pub pages_free: u64,
    /// Write amplification factor (for flash)
    pub write_amplification: f32,
}
