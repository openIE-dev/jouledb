//! In-memory persistence backend

use joule_db_core::storage::PageId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::config::PersistenceConfig;
use super::error::PersistenceError;
use super::{AsyncPersistenceBackend, StorageStats};

/// In-memory persistence backend
///
/// Useful for testing and temporary storage.
/// Data is lost when the backend is dropped.
#[derive(Debug)]
pub struct MemoryBackend {
    /// Page data storage
    pages: Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    /// Storage statistics
    stats: Arc<RwLock<StorageStats>>,
    /// Configuration
    config: PersistenceConfig,
}

impl MemoryBackend {
    /// Create a new in-memory backend
    pub fn new() -> Self {
        Self::with_config(PersistenceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PersistenceConfig) -> Self {
        Self {
            pages: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            config,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PersistenceConfig {
        &self.config
    }

    /// Read a page synchronously
    pub fn read_page_sync(&self, page_id: PageId) -> Result<Vec<u8>, PersistenceError> {
        let pages = self.pages.read().unwrap();
        let mut stats = self.stats.write().unwrap();
        stats.read_ops += 1;

        pages
            .get(&page_id)
            .cloned()
            .ok_or_else(|| {
                stats.cache_misses += 1;
                PersistenceError::PageNotFound(page_id)
            })
            .map(|data| {
                stats.cache_hits += 1;
                data
            })
    }

    /// Write a page synchronously
    pub fn write_page_sync(&self, page_id: PageId, data: &[u8]) -> Result<(), PersistenceError> {
        let mut pages = self.pages.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let is_new = !pages.contains_key(&page_id);
        pages.insert(page_id, data.to_vec());

        stats.write_ops += 1;
        if is_new {
            stats.total_pages += 1;
        }
        stats.bytes_used = pages.values().map(|v| v.len() as u64).sum();

        Ok(())
    }

    /// Delete a page
    pub fn delete_page(&self, page_id: PageId) -> Result<bool, PersistenceError> {
        let mut pages = self.pages.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let existed = pages.remove(&page_id).is_some();
        if existed {
            stats.total_pages = stats.total_pages.saturating_sub(1);
            stats.bytes_used = pages.values().map(|v| v.len() as u64).sum();
        }

        Ok(existed)
    }

    /// Clear all pages
    pub fn clear(&self) {
        let mut pages = self.pages.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        pages.clear();
        stats.total_pages = 0;
        stats.bytes_used = 0;
    }

    /// Get all page IDs
    pub fn page_ids(&self) -> Vec<PageId> {
        self.pages.read().unwrap().keys().copied().collect()
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MemoryBackend {
    fn clone(&self) -> Self {
        Self {
            pages: Arc::clone(&self.pages),
            stats: Arc::clone(&self.stats),
            config: self.config.clone(),
        }
    }
}

impl AsyncPersistenceBackend for MemoryBackend {
    #[cfg(target_arch = "wasm32")]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + 'static>,
    > {
        let pages = Arc::clone(&self.pages);
        Box::pin(async move {
            let pages_guard = pages.read().unwrap();
            pages_guard
                .get(&page_id)
                .cloned()
                .ok_or(PersistenceError::PageNotFound(page_id))
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + Send + 'static>,
    > {
        let pages = Arc::clone(&self.pages);
        Box::pin(async move {
            let pages_guard = pages.read().unwrap();
            pages_guard
                .get(&page_id)
                .cloned()
                .ok_or(PersistenceError::PageNotFound(page_id))
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>
    {
        let data = data.to_vec();
        let pages = Arc::clone(&self.pages);
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            let mut pages_guard = pages.write().unwrap();
            pages_guard.insert(page_id, data);
            stats.write().unwrap().write_ops += 1;
            Ok(())
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    > {
        let data = data.to_vec();
        let pages = Arc::clone(&self.pages);
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            let mut pages_guard = pages.write().unwrap();
            pages_guard.insert(page_id, data);
            stats.write().unwrap().write_ops += 1;
            Ok(())
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn sync(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>
    {
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            stats.write().unwrap().sync_ops += 1;
            Ok(())
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sync(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    > {
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            stats.write().unwrap().sync_ops += 1;
            Ok(())
        })
    }

    fn stats(&self) -> StorageStats {
        self.stats.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_backend_basic() {
        let backend = MemoryBackend::new();

        // Write
        backend.write_page_sync(1, b"hello").unwrap();
        backend.write_page_sync(2, b"world").unwrap();

        // Read
        assert_eq!(backend.read_page_sync(1).unwrap(), b"hello");
        assert_eq!(backend.read_page_sync(2).unwrap(), b"world");

        // Stats
        let stats = backend.stats.read().unwrap();
        assert_eq!(stats.total_pages, 2);
        assert_eq!(stats.write_ops, 2);
    }

    #[test]
    fn test_memory_backend_not_found() {
        let backend = MemoryBackend::new();

        match backend.read_page_sync(999) {
            Err(PersistenceError::PageNotFound(id)) => assert_eq!(id, 999),
            _ => panic!("Expected PageNotFound error"),
        }
    }

    #[test]
    fn test_memory_backend_delete() {
        let backend = MemoryBackend::new();

        backend.write_page_sync(1, b"data").unwrap();
        assert!(backend.delete_page(1).unwrap());
        assert!(!backend.delete_page(1).unwrap());

        match backend.read_page_sync(1) {
            Err(PersistenceError::PageNotFound(_)) => {}
            _ => panic!("Expected PageNotFound error"),
        }
    }
}
