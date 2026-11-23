//! Async Engine wrapper for browser backends
//!
//! Provides async operations that work with async storage backends (OPFS/IndexedDB).
//! This wrapper provides the same interface as the sync Engine but with async methods.

use crate::persistence::{AsyncPersistenceBackend, DynBackend};
use joule_db_core::{
    Error as CoreError,
    RwLockExt, // Safe lock extension
    StorageError,
    storage::PageId,
};
use std::sync::{Arc, RwLock};

/// Async Engine wrapper that provides async operations
///
/// This provides async get/put/delete operations using async storage backends.
/// It maintains a simple key-value mapping on top of page storage.
///
/// Future enhancement: Full B-tree operations with async I/O
pub struct AsyncEngine {
    /// Async storage backend (OPFS or IndexedDB)
    backend: Arc<RwLock<DynBackend>>,
    /// In-memory index mapping keys to page IDs
    pub(crate) key_index: Arc<RwLock<std::collections::HashMap<Vec<u8>, PageId>>>,
    /// Next page ID to allocate
    next_page_id: Arc<RwLock<PageId>>,
    /// Metadata page ID (stores the key index)
    metadata_page_id: PageId,
}

impl AsyncEngine {
    /// Create a new async engine with the given backend
    pub fn new(backend: Arc<RwLock<DynBackend>>) -> Self {
        Self {
            backend,
            key_index: Arc::new(RwLock::new(std::collections::HashMap::new())),
            next_page_id: Arc::new(RwLock::new(2)), // Page 1 is metadata
            metadata_page_id: 1,
        }
    }

    /// Create a new async engine and initialize it
    pub async fn create(backend: Arc<RwLock<DynBackend>>) -> Result<Self, CoreError> {
        let engine = Self::new(backend);
        // Initialize metadata page
        engine.save_metadata().await?;
        Ok(engine)
    }

    /// Open an existing async engine
    pub async fn open(backend: Arc<RwLock<DynBackend>>) -> Result<Self, CoreError> {
        let engine = Self::new(backend);
        // Load metadata (key index)
        engine.load_metadata().await?;
        Ok(engine)
    }

    /// Save metadata (key index) to storage
    async fn save_metadata(&self) -> Result<(), CoreError> {
        let index = self.key_index.safe_read("key_index")?;
        let next_id = *self.next_page_id.safe_read("next_page_id")?;

        // Serialize key index
        let mut metadata = Vec::new();
        metadata.extend_from_slice(&next_id.to_le_bytes());
        metadata.extend_from_slice(&(index.len() as u64).to_le_bytes());

        for (key, page_id) in index.iter() {
            metadata.extend_from_slice(&(key.len() as u32).to_le_bytes());
            metadata.extend_from_slice(key);
            metadata.extend_from_slice(&page_id.to_le_bytes());
        }

        let backend = self.backend.safe_read("backend")?;
        backend
            .write_page(self.metadata_page_id, &metadata)
            .await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;

        Ok(())
    }

    /// Load metadata (key index) from storage
    async fn load_metadata(&self) -> Result<(), CoreError> {
        let backend = self.backend.safe_read("backend")?;
        let metadata = backend
            .read_page(self.metadata_page_id)
            .await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;

        if metadata.is_empty() {
            // New database, no metadata yet
            return Ok(());
        }

        // Deserialize key index
        let mut offset = 0;

        // Read next_page_id
        if offset + 8 > metadata.len() {
            return Err(CoreError::Storage(StorageError::Backend(
                "Corrupted metadata".to_string(),
            )));
        }
        let next_id = u64::from_le_bytes([
            metadata[offset],
            metadata[offset + 1],
            metadata[offset + 2],
            metadata[offset + 3],
            metadata[offset + 4],
            metadata[offset + 5],
            metadata[offset + 6],
            metadata[offset + 7],
        ]);
        offset += 8;
        *self.next_page_id.safe_write("next_page_id")? = next_id;

        // Read index size
        if offset + 8 > metadata.len() {
            return Err(CoreError::Storage(StorageError::Backend(
                "Corrupted metadata".to_string(),
            )));
        }
        let index_size = u64::from_le_bytes([
            metadata[offset],
            metadata[offset + 1],
            metadata[offset + 2],
            metadata[offset + 3],
            metadata[offset + 4],
            metadata[offset + 5],
            metadata[offset + 6],
            metadata[offset + 7],
        ]) as usize;
        offset += 8;

        // Read key-value pairs
        let mut index = self.key_index.safe_write("key_index")?;
        for _ in 0..index_size {
            // Read key length
            if offset + 4 > metadata.len() {
                return Err(CoreError::Storage(StorageError::Backend(
                    "Corrupted metadata".to_string(),
                )));
            }
            let key_len = u32::from_le_bytes([
                metadata[offset],
                metadata[offset + 1],
                metadata[offset + 2],
                metadata[offset + 3],
            ]) as usize;
            offset += 4;

            // Read key
            if offset + key_len > metadata.len() {
                return Err(CoreError::Storage(StorageError::Backend(
                    "Corrupted metadata".to_string(),
                )));
            }
            let key = metadata[offset..offset + key_len].to_vec();
            offset += key_len;

            // Read page ID
            if offset + 8 > metadata.len() {
                return Err(CoreError::Storage(StorageError::Backend(
                    "Corrupted metadata".to_string(),
                )));
            }
            let page_id = u64::from_le_bytes([
                metadata[offset],
                metadata[offset + 1],
                metadata[offset + 2],
                metadata[offset + 3],
                metadata[offset + 4],
                metadata[offset + 5],
                metadata[offset + 6],
                metadata[offset + 7],
            ]);
            offset += 8;

            index.insert(key, page_id);
        }

        Ok(())
    }

    /// Get a value by key
    pub async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, CoreError> {
        // Clone the internal Arcs so we can drop self reference before await
        let key_index = Arc::clone(&self.key_index);
        let backend = Arc::clone(&self.backend);

        // Use the static helper that works with cloned Arcs
        Self::get_with_arcs(key_index, backend, key).await
    }

    /// Get a value by key using cloned Arcs - allows calling without holding &self
    /// This is useful for AsyncRangeIterator which needs to call get without holding locks
    pub(crate) async fn get_with_arcs(
        key_index: Arc<RwLock<std::collections::HashMap<Vec<u8>, PageId>>>,
        backend: Arc<RwLock<DynBackend>>,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, CoreError> {
        // Look up page ID for this key (drop guard before await)
        let page_id = {
            let index = key_index.safe_read("key_index")?;
            index.get(key).copied()
        };

        if let Some(page_id) = page_id {
            // Read page from backend - get the future first, then drop guard, then await
            let read_future = {
                let backend_guard = backend.safe_read("backend")?;
                backend_guard.read_page(page_id)
            };
            let data = read_future
                .await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;

            // Page data is the value directly
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Get cloned internal Arcs for async operations without holding &self
    pub(crate) fn get_internal_arcs(
        &self,
    ) -> (
        Arc<RwLock<std::collections::HashMap<Vec<u8>, PageId>>>,
        Arc<RwLock<DynBackend>>,
    ) {
        (Arc::clone(&self.key_index), Arc::clone(&self.backend))
    }

    /// Put a key-value pair
    pub async fn put(&self, key: &[u8], value: &[u8]) -> Result<(), CoreError> {
        // Get or allocate page ID
        let (page_id, is_new) = {
            let mut index = self.key_index.safe_write("key_index")?;
            let mut next_id = self.next_page_id.safe_write("next_page_id")?;

            let is_new = !index.contains_key(key);
            let page_id = *index.entry(key.to_vec()).or_insert_with(|| {
                let id = *next_id;
                *next_id += 1;
                id
            });
            (page_id, is_new)
        };

        // Write page to backend
        let backend = self.backend.safe_read("backend")?;
        backend
            .write_page(page_id, value)
            .await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;

        // Update metadata if this is a new key
        if is_new {
            self.save_metadata().await?;
        }

        Ok(())
    }

    /// Delete a key
    pub async fn delete(&self, key: &[u8]) -> Result<bool, CoreError> {
        // Remove from index
        let removed = {
            let mut index = self.key_index.safe_write("key_index")?;
            index.remove(key).is_some()
        };

        if removed {
            // Update metadata
            self.save_metadata().await?;
            // Note: In a full implementation, we'd mark the page as free for reuse
            // For now, we just remove it from the index
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Sync to durable storage
    pub async fn sync(&self) -> Result<(), CoreError> {
        // Save metadata first
        self.save_metadata().await?;

        // Then sync backend
        let backend = self.backend.safe_read("backend")?;
        backend
            .sync()
            .await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
        Ok(())
    }
}
