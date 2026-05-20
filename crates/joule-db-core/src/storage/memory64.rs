//! Memory64 Support for Large Databases
//!
//! Provides support for Wasm 3.0 Memory64, enabling databases >4GB.
//! Falls back to standard 32-bit addressing when Memory64 is unavailable.

use crate::error::StorageError;
use crate::storage::StorageBackend;
use crate::storage::page::DEFAULT_PAGE_SIZE;

/// Memory64-aware page identifier
///
/// Uses u64 for page IDs to support >4GB databases.
/// This is already the case in the standard PageId type,
/// but this module provides explicit Memory64 support.
pub type Memory64PageId = u64;

/// Memory64-aware storage backend trait
///
/// Extends StorageBackend with explicit 64-bit addressing support.
pub trait Memory64Backend: StorageBackend {
    /// Check if Memory64 is supported
    fn supports_memory64(&self) -> bool;

    /// Get maximum addressable size (in bytes)
    fn max_addressable_size(&self) -> u64;

    /// Allocate page with explicit 64-bit addressing
    fn allocate_page_64(&mut self) -> Result<Memory64PageId, StorageError> {
        // Default implementation uses standard allocate_page
        self.allocate_page()
    }
}

/// Memory64 configuration
#[derive(Debug, Clone, Copy)]
pub struct Memory64Config {
    /// Enable Memory64 (requires Wasm 3.0)
    pub enabled: bool,
    /// Maximum database size in bytes (0 = unlimited)
    pub max_size: u64,
}

impl Default for Memory64Config {
    fn default() -> Self {
        Self {
            enabled: cfg!(target_feature = "memory64"),
            max_size: 0, // Unlimited
        }
    }
}

impl Memory64Config {
    /// Create with Memory64 enabled
    pub fn with_memory64() -> Self {
        Self {
            enabled: true,
            max_size: 0,
        }
    }

    /// Create with size limit
    pub fn with_max_size(max_size: u64) -> Self {
        Self {
            enabled: true,
            max_size,
        }
    }

    /// Check if Memory64 is available in the current environment
    #[cfg(target_arch = "wasm32")]
    pub fn is_available() -> bool {
        // Check for Memory64 support
        // This would check for Wasm 3.0 features
        // For now, return false as Memory64 is not yet widely supported
        false
    }

    /// Check if standard 64-bit addressing is available
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_available() -> bool {
        // Native platforms always support 64-bit addressing
        true
    }
}

/// Memory64-aware page allocator
///
/// Manages page allocation with 64-bit addressing support.
pub struct Memory64PageAllocator {
    /// Next page ID
    next_page_id: u64,
    /// Free page list
    free_pages: Vec<u64>,
    /// Maximum page ID (0 = unlimited)
    max_page_id: u64,
    /// Configuration
    config: Memory64Config,
}

impl Memory64PageAllocator {
    /// Create a new Memory64 page allocator
    pub fn new(config: Memory64Config) -> Self {
        Self {
            next_page_id: 1, // Page 0 is reserved
            free_pages: Vec::new(),
            max_page_id: if config.max_size > 0 {
                config.max_size / DEFAULT_PAGE_SIZE as u64
            } else {
                0 // Unlimited
            },
            config,
        }
    }

    /// Allocate a new page ID
    pub fn allocate(&mut self) -> Result<u64, StorageError> {
        // Try to reuse a free page first
        if let Some(page_id) = self.free_pages.pop() {
            return Ok(page_id);
        }

        // Check size limit
        if self.max_page_id > 0 && self.next_page_id >= self.max_page_id {
            return Err(StorageError::Backend(
                "Maximum database size reached".to_string(),
            ));
        }

        // Allocate new page ID
        let page_id = self.next_page_id;
        self.next_page_id += 1;

        // Check for overflow (shouldn't happen with u64, but check anyway)
        if self.next_page_id == 0 {
            return Err(StorageError::Backend(
                "Page ID overflow (database too large)".to_string(),
            ));
        }

        Ok(page_id)
    }

    /// Free a page ID for reuse
    pub fn free(&mut self, page_id: u64) {
        if page_id > 0 && page_id < self.next_page_id {
            self.free_pages.push(page_id);
        }
    }

    /// Get current page count
    pub fn page_count(&self) -> u64 {
        self.next_page_id - 1 - self.free_pages.len() as u64
    }

    /// Get maximum addressable size
    pub fn max_size(&self) -> u64 {
        if self.max_page_id > 0 {
            self.max_page_id * DEFAULT_PAGE_SIZE as u64
        } else {
            // u64::MAX pages * 16KB = ~300 exabytes (effectively unlimited)
            u64::MAX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory64_allocator() {
        let mut allocator = Memory64PageAllocator::new(Memory64Config::default());

        let page1 = allocator.allocate().unwrap();
        assert_eq!(page1, 1);

        let page2 = allocator.allocate().unwrap();
        assert_eq!(page2, 2);

        allocator.free(page1);
        let page3 = allocator.allocate().unwrap();
        assert_eq!(page3, 1); // Reused
    }

    #[test]
    fn test_memory64_config() {
        let config = Memory64Config::with_max_size(1024 * 1024 * 1024); // 1GB
        assert!(config.enabled);
        assert_eq!(config.max_size, 1024 * 1024 * 1024);
    }
}
