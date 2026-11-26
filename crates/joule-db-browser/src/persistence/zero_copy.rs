//! Zero-Copy Page Transfer using SharedArrayBuffer
//!
//! Provides direct memory access to pages without copying,
//! using SharedArrayBuffer for efficient page cache.

use joule_db_core::storage::PageId;
use joule_db_core::storage::page::DEFAULT_PAGE_SIZE;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(target_arch = "wasm32")]
use js_sys::{SharedArrayBuffer, Uint8Array};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use super::error::PersistenceError;

/// Zero-copy page cache using SharedArrayBuffer
///
/// Stores pages directly in shared memory for zero-copy access.
/// Multiple workers can access the same pages without copying.
pub struct ZeroCopyPageCache {
    /// Shared buffer containing all pages
    #[cfg(target_arch = "wasm32")]
    shared_buffer: SharedArrayBuffer,
    /// Native: use RwLock to avoid full buffer clones on write
    #[cfg(not(target_arch = "wasm32"))]
    shared_buffer: RwLock<Vec<u8>>,
    /// Page table: PageID -> offset in shared buffer
    page_table: Arc<RwLock<std::collections::HashMap<PageId, usize>>>,
    /// Current write offset (atomic for thread-safe allocation)
    write_offset: Arc<AtomicUsize>,
    /// Capacity in bytes
    capacity: usize,
}

impl ZeroCopyPageCache {
    /// Create a new zero-copy page cache
    ///
    /// # Arguments
    /// * `capacity_mb` - Capacity in megabytes
    #[cfg(target_arch = "wasm32")]
    pub fn new(capacity_mb: usize) -> Result<Self, PersistenceError> {
        let capacity = capacity_mb * 1024 * 1024;
        let buffer = SharedArrayBuffer::new(capacity as u32);

        Ok(Self {
            shared_buffer: buffer,
            page_table: Arc::new(RwLock::new(std::collections::HashMap::new())),
            write_offset: Arc::new(AtomicUsize::new(0)),
            capacity,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(capacity_mb: usize) -> Result<Self, PersistenceError> {
        let capacity = capacity_mb * 1024 * 1024;
        Ok(Self {
            shared_buffer: RwLock::new(vec![0u8; capacity]),
            page_table: Arc::new(RwLock::new(std::collections::HashMap::new())),
            write_offset: Arc::new(AtomicUsize::new(0)),
            capacity,
        })
    }

    /// Write page directly to shared buffer (zero copy)
    ///
    /// Returns the offset where the page was written.
    #[cfg(target_arch = "wasm32")]
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> Result<usize, PersistenceError> {
        if data.len() > DEFAULT_PAGE_SIZE {
            return Err(PersistenceError::IoError("Page too large".to_string()));
        }

        // Atomic allocation
        let offset = self
            .write_offset
            .fetch_add(DEFAULT_PAGE_SIZE, Ordering::SeqCst);

        if offset + DEFAULT_PAGE_SIZE > self.capacity {
            return Err(PersistenceError::IoError("Cache exhausted".to_string()));
        }

        // Write directly to shared buffer
        let view = Uint8Array::new(&self.shared_buffer);
        let page_view = view.subarray(offset as u32, (offset + DEFAULT_PAGE_SIZE) as u32);

        // Copy data (this is the only copy)
        let data_array = Uint8Array::from(data);
        page_view.set(&data_array, 0);

        // Pad to page size if needed
        if data.len() < DEFAULT_PAGE_SIZE {
            let zero_array = Uint8Array::new_with_length((DEFAULT_PAGE_SIZE - data.len()) as u32);
            page_view.set(&zero_array, data.len() as u32);
        }

        // Update page table
        let mut table = self.page_table.write().unwrap_or_else(|e| e.into_inner());
        table.insert(page_id, offset);

        Ok(offset)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> Result<usize, PersistenceError> {
        if data.len() > DEFAULT_PAGE_SIZE {
            return Err(PersistenceError::IoError("Page too large".to_string()));
        }

        let offset = self
            .write_offset
            .fetch_add(DEFAULT_PAGE_SIZE, Ordering::SeqCst);

        if offset + DEFAULT_PAGE_SIZE > self.capacity {
            return Err(PersistenceError::IoError("Cache exhausted".to_string()));
        }

        // Use write lock instead of cloning the entire buffer
        let mut buffer = self
            .shared_buffer
            .write()
            .unwrap_or_else(|e| e.into_inner());
        buffer[offset..offset + data.len()].copy_from_slice(data);
        // Zero pad if needed
        if data.len() < DEFAULT_PAGE_SIZE {
            buffer[offset + data.len()..offset + DEFAULT_PAGE_SIZE].fill(0);
        }

        let mut table = self.page_table.write().unwrap_or_else(|e| e.into_inner());
        table.insert(page_id, offset);

        Ok(offset)
    }

    /// Get page as direct view into shared buffer (zero copy read)
    ///
    /// Returns a Uint8Array view into the shared buffer.
    #[cfg(target_arch = "wasm32")]
    pub fn get_page_view(&self, page_id: PageId) -> Option<Uint8Array> {
        let table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        let offset = *table.get(&page_id)?;

        let view = Uint8Array::new(&self.shared_buffer);
        Some(view.subarray(offset as u32, (offset + DEFAULT_PAGE_SIZE) as u32))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_page_view(&self, page_id: PageId) -> Option<Vec<u8>> {
        let table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        let offset = *table.get(&page_id)?;

        let buffer = self.shared_buffer.read().unwrap_or_else(|e| e.into_inner());
        Some(buffer[offset..offset + DEFAULT_PAGE_SIZE].to_vec())
    }

    /// Get the underlying SharedArrayBuffer (for passing to workers)
    #[cfg(target_arch = "wasm32")]
    pub fn buffer(&self) -> SharedArrayBuffer {
        self.shared_buffer.clone()
    }

    /// Check if page is cached
    pub fn has_page(&self, page_id: PageId) -> bool {
        let table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        table.contains_key(&page_id)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        let used = self.write_offset.load(Ordering::Relaxed);

        CacheStats {
            pages_cached: table.len(),
            bytes_used: used,
            capacity: self.capacity,
            utilization: (used as f64 / self.capacity as f64) * 100.0,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut table = self.page_table.write().unwrap_or_else(|e| e.into_inner());
        table.clear();
        self.write_offset.store(0, Ordering::Release);
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of pages cached
    pub pages_cached: usize,
    /// Bytes used
    pub bytes_used: usize,
    /// Total capacity
    pub capacity: usize,
    /// Utilization percentage
    pub utilization: f64,
}

/// Zero-copy page transfer wrapper
///
/// Combines a persistence backend with a zero-copy cache.
/// Pages are loaded into the cache on first access, then
/// accessed directly without copying.
pub struct ZeroCopyBackend {
    /// Underlying persistence backend
    backend: Arc<RwLock<dyn super::AsyncPersistenceBackend + Send + Sync>>,
    /// Zero-copy cache
    cache: Arc<ZeroCopyPageCache>,
}

impl ZeroCopyBackend {
    /// Create a new zero-copy backend
    pub fn new(
        backend: Arc<RwLock<dyn super::AsyncPersistenceBackend + Send + Sync>>,
        cache_capacity_mb: usize,
    ) -> Result<Self, PersistenceError> {
        let cache = ZeroCopyPageCache::new(cache_capacity_mb)?;

        Ok(Self {
            backend,
            cache: Arc::new(cache),
        })
    }

    /// Read page with zero-copy caching
    #[cfg(target_arch = "wasm32")]
    pub async fn read_page(&self, page_id: PageId) -> Result<Uint8Array, PersistenceError> {
        // Check cache first
        if let Some(view) = self.cache.get_page_view(page_id) {
            return Ok(view);
        }

        // Load from backend
        let backend = self.backend.read().unwrap_or_else(|e| e.into_inner());
        let data = backend.read_page(page_id).await?;

        // Write to cache (zero-copy)
        self.cache.write_page(page_id, &data)?;

        // Return view
        self.cache
            .get_page_view(page_id)
            .ok_or_else(|| PersistenceError::PageNotFound(page_id))
    }

    /// Write page with zero-copy caching
    pub async fn write_page(&self, page_id: PageId, data: &[u8]) -> Result<(), PersistenceError> {
        // Write to cache first (zero-copy)
        self.cache.write_page(page_id, data)?;

        // Write to backend
        let backend = self.backend.read().unwrap_or_else(|e| e.into_inner());
        backend.write_page(page_id, data).await?;

        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}
