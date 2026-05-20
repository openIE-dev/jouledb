//! In-memory storage backend
//!
//! A simple in-memory storage backend for testing and development.
//! All data is lost when the backend is dropped.

use std::collections::HashMap;
use std::sync::{
    RwLock,
    atomic::{AtomicU64, Ordering},
};

use super::page::{DEFAULT_PAGE_SIZE, Page, PageId};
use super::traits::{CommittedMeta, StorageBackend, StorageStats};
use crate::error::StorageError;

/// In-memory storage backend
///
/// Thread-safe implementation using RwLock for concurrent access.
/// Useful for testing and temporary databases.
///
/// # Example
///
/// ```rust
/// use joule_db_core::storage::memory::MemoryBackend;
/// use joule_db_core::storage::{StorageBackend, Page, PageId};
/// use joule_db_core::storage::page::PageType;
///
/// let mut backend = MemoryBackend::new();
///
/// // Allocate a page
/// let page_id = backend.allocate_page().unwrap();
///
/// // Write data
/// let page = Page::with_data(page_id, PageType::BTreeLeaf, b"hello".to_vec());
/// backend.write_page(page).unwrap();
///
/// // Read it back
/// let read = backend.read_page(page_id).unwrap().unwrap();
/// assert_eq!(read.data, b"hello");
/// ```
pub struct MemoryBackend {
    /// Page storage
    pages: RwLock<HashMap<PageId, Page>>,
    /// Next page ID to allocate
    next_page_id: AtomicU64,
    /// Free page IDs available for reuse
    free_pages: RwLock<Vec<PageId>>,
    /// Page size
    page_size: usize,
    /// Statistics
    stats: RwLock<MemoryStats>,
    /// Phase 2 of CoW MVCC: latest committed root + version, written
    /// atomically by `write_committed_meta`. Empty until the first
    /// commit. See `docs/joule-db/cow-mvcc-design.md`.
    committed_meta: RwLock<Option<CommittedMeta>>,
}

#[derive(Debug, Default)]
struct MemoryStats {
    pages_read: u64,
    pages_written: u64,
}

impl MemoryBackend {
    /// Create a new in-memory backend with default page size
    pub fn new() -> Self {
        Self::with_page_size(DEFAULT_PAGE_SIZE)
    }

    /// Create a new in-memory backend with custom page size
    pub fn with_page_size(page_size: usize) -> Self {
        Self {
            pages: RwLock::new(HashMap::new()),
            next_page_id: AtomicU64::new(1), // Page 0 is reserved
            free_pages: RwLock::new(Vec::new()),
            page_size,
            stats: RwLock::new(MemoryStats::default()),
            committed_meta: RwLock::new(None),
        }
    }

    /// Get the number of pages currently stored
    pub fn page_count(&self) -> usize {
        self.pages.read().expect("lock poisoned: pages read").len()
    }

    /// Clear all pages
    pub fn clear(&mut self) {
        self.pages
            .write()
            .expect("lock poisoned: pages write")
            .clear();
        self.free_pages
            .write()
            .expect("lock poisoned: free_pages write")
            .clear();
        self.next_page_id.store(1, Ordering::SeqCst);
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for MemoryBackend {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        let pages = self.pages.read().expect("lock poisoned: pages read");
        let result = pages.get(&page_id).cloned();

        // Update stats
        if result.is_some() {
            let mut stats = self.stats.write().expect("lock poisoned: stats write");
            stats.pages_read += 1;
        }

        Ok(result)
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        let mut pages = self.pages.write().expect("lock poisoned: pages write");
        pages.insert(page.id, page);

        // Update stats
        let mut stats = self.stats.write().expect("lock poisoned: stats write");
        stats.pages_written += 1;

        Ok(())
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        // First try to reuse a free page
        let mut free = self
            .free_pages
            .write()
            .expect("lock poisoned: free_pages write");
        if let Some(page_id) = free.pop() {
            return Ok(page_id);
        }
        drop(free);

        // Allocate new page ID
        let page_id = self.next_page_id.fetch_add(1, Ordering::SeqCst);
        Ok(page_id)
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        // Remove from storage
        let mut pages = self.pages.write().expect("lock poisoned: pages write");
        pages.remove(&page_id);
        drop(pages);

        // Add to free list
        let mut free = self
            .free_pages
            .write()
            .expect("lock poisoned: free_pages write");
        free.push(page_id);

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        // No-op for memory backend - data is always "synced"
        Ok(())
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn stats(&self) -> StorageStats {
        let pages = self.pages.read().expect("lock poisoned: pages read");
        let free = self
            .free_pages
            .read()
            .expect("lock poisoned: free_pages read");
        let stats = self.stats.read().expect("lock poisoned: stats read");

        StorageStats {
            total_pages: pages.len() as u64,
            free_pages: free.len() as u64,
            pages_read: stats.pages_read,
            pages_written: stats.pages_written,
            page_size: self.page_size,
        }
    }

    fn read_committed_meta(&self) -> Result<Option<CommittedMeta>, StorageError> {
        Ok(*self
            .committed_meta
            .read()
            .expect("lock poisoned: committed_meta read"))
    }

    fn write_committed_meta(&mut self, meta: &CommittedMeta) -> Result<(), StorageError> {
        // In-process atomicity is sufficient for the in-memory backend —
        // there is no concurrent reader process for it to torn-write
        // against. Real cross-process atomicity comes from FileBackend's
        // tmp + fsync + rename in the next phase.
        *self
            .committed_meta
            .write()
            .expect("lock poisoned: committed_meta write") = Some(*meta);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::page::PageType;

    #[test]
    fn test_basic_operations() {
        let mut backend = MemoryBackend::new();

        // Allocate page
        let page_id = backend.allocate_page().unwrap();
        assert!(page_id > 0);

        // Write page
        let page = Page::with_data(page_id, PageType::BTreeLeaf, b"test data".to_vec());
        backend.write_page(page).unwrap();

        // Read page
        let read = backend.read_page(page_id).unwrap().unwrap();
        assert_eq!(read.id, page_id);
        assert_eq!(read.data, b"test data");

        // Non-existent page
        assert!(backend.read_page(999).unwrap().is_none());
    }

    #[test]
    fn test_page_reuse() {
        let mut backend = MemoryBackend::new();

        // Allocate and free a page
        let page_id1 = backend.allocate_page().unwrap();
        backend.free_page(page_id1).unwrap();

        // Next allocation should reuse the freed page
        let page_id2 = backend.allocate_page().unwrap();
        assert_eq!(page_id1, page_id2);
    }

    #[test]
    fn test_stats() {
        let mut backend = MemoryBackend::new();

        let page_id = backend.allocate_page().unwrap();
        let page = Page::with_data(page_id, PageType::BTreeLeaf, b"data".to_vec());
        backend.write_page(page).unwrap();
        backend.read_page(page_id).unwrap();

        let stats = backend.stats();
        assert_eq!(stats.total_pages, 1);
        assert_eq!(stats.pages_written, 1);
        assert_eq!(stats.pages_read, 1);
    }

    #[test]
    fn test_clear() {
        let mut backend = MemoryBackend::new();

        // Add some pages
        for _ in 0..5 {
            let page_id = backend.allocate_page().unwrap();
            let page = Page::new(page_id, PageType::BTreeLeaf);
            backend.write_page(page).unwrap();
        }
        assert_eq!(backend.page_count(), 5);

        // Clear
        backend.clear();
        assert_eq!(backend.page_count(), 0);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(RwLock::new(MemoryBackend::new()));
        let mut handles = vec![];

        // Spawn multiple readers and writers
        for i in 0..10 {
            let backend = Arc::clone(&backend);
            handles.push(thread::spawn(move || {
                let mut backend = backend.write().unwrap();
                let page_id = backend.allocate_page().unwrap();
                let page = Page::with_data(
                    page_id,
                    PageType::BTreeLeaf,
                    format!("data{}", i).into_bytes(),
                );
                backend.write_page(page).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let backend = backend.read().unwrap();
        assert_eq!(backend.page_count(), 10);
    }
}
