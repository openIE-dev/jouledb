//! Storage backend traits
//!
//! These traits define the interface that all storage backends must implement.

use super::page::{DEFAULT_PAGE_SIZE, Page, PageId};
use crate::error::StorageError;

/// Atomically-committed root pointer + monotonic version.
///
/// Phase 2 of the CoW MVCC refactor (`docs/joule-db/cow-mvcc-design.md`):
/// the B-tree root pointer no longer lives in storage page 1
/// rewritten in-place. It lives in the backend's atomically-renamed
/// metadata file so a writer's commit appears atomic to readers
/// holding a snapshot.
///
/// `committed_version` increments on every successful commit.
/// Snapshot readers (Phase 3) capture this value to compare against
/// the writer's later commits when refreshing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommittedMeta {
    /// On-disk format version of the meta record. Currently 1.
    pub format_version: u32,
    /// Monotonic counter — incremented by every successful commit.
    pub committed_version: u64,
    /// Page id of the B-tree root visible to readers at this version.
    pub committed_root: PageId,
}

impl CommittedMeta {
    /// Initial meta for a freshly created database. Version starts at
    /// 1 so the first observable commit is `committed_version >= 1`.
    pub fn initial(root: PageId) -> Self {
        Self {
            format_version: 1,
            committed_version: 1,
            committed_root: root,
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of pages
    pub total_pages: u64,
    /// Number of free pages
    pub free_pages: u64,
    /// Number of pages read
    pub pages_read: u64,
    /// Number of pages written
    pub pages_written: u64,
    /// Page size in bytes
    pub page_size: usize,
}

/// Synchronous storage backend trait
///
/// This is the core abstraction for page-level storage. Implementations
/// can be in-memory, file-based, or use any other storage mechanism.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow use from multiple threads.
/// Internal synchronization is the responsibility of the implementation.
///
/// # Example
///
/// ```rust,ignore
/// use joule_db_core::storage::{StorageBackend, Page, PageId};
///
/// let mut backend = MemoryBackend::new();
///
/// // Allocate and write a page
/// let page_id = backend.allocate_page()?;
/// let page = Page::with_data(page_id, PageType::BTreeLeaf, b"data".to_vec());
/// backend.write_page(page)?;
///
/// // Read it back
/// let read_page = backend.read_page(page_id)?;
/// assert_eq!(read_page.unwrap().data, b"data");
/// ```
pub trait StorageBackend: Send + Sync {
    /// Read a page by ID
    ///
    /// Returns `Ok(None)` if the page doesn't exist.
    /// Returns `Err` on I/O or corruption errors.
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError>;

    /// Write a page
    ///
    /// Creates the page if it doesn't exist, updates if it does.
    /// The page ID is taken from `page.id`.
    fn write_page(&mut self, page: Page) -> Result<(), StorageError>;

    /// Allocate a new page
    ///
    /// Returns the ID of the newly allocated page. The page is not
    /// initialized - caller must write to it.
    fn allocate_page(&mut self) -> Result<PageId, StorageError>;

    /// Free a page
    ///
    /// Marks the page as free for reuse. The page data may or may not
    /// be immediately erased depending on the backend.
    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError>;

    /// Allocate N contiguous pages for extent storage.
    ///
    /// Returns the ID of the first page. Pages are guaranteed to have
    /// sequential IDs: first, first+1, first+2, ..., first+count-1.
    ///
    /// This is used for large blob storage (LLM weight tensors) where
    /// contiguous layout enables sequential I/O instead of random page hops.
    ///
    /// Default implementation allocates pages at the end of the file
    /// (bypasses free list to guarantee contiguity).
    fn allocate_contiguous(&mut self, count: usize) -> Result<PageId, StorageError> {
        // Default: allocate one at a time (no contiguity guarantee)
        // Backends should override for true contiguous allocation.
        let first = self.allocate_page()?;
        for _ in 1..count {
            self.allocate_page()?;
        }
        Ok(first)
    }

    /// Flush all pending writes to durable storage
    ///
    /// After this returns successfully, all previously written pages
    /// are guaranteed to be persisted.
    fn sync(&mut self) -> Result<(), StorageError>;

    /// Get the page size for this backend
    ///
    /// All pages must fit within this size (header + data).
    fn page_size(&self) -> usize {
        DEFAULT_PAGE_SIZE
    }

    /// Get storage statistics
    fn stats(&self) -> StorageStats {
        StorageStats {
            page_size: self.page_size(),
            ..Default::default()
        }
    }

    /// Check if a page exists
    fn page_exists(&self, page_id: PageId) -> Result<bool, StorageError> {
        Ok(self.read_page(page_id)?.is_some())
    }

    /// Read the most recently committed meta record, if the backend
    /// persists it. `Ok(None)` means the backend has no committed meta
    /// yet (fresh database, or a legacy backend that never stored one).
    ///
    /// **CoW MVCC (Phase 2).** Backends that persist this record (e.g.
    /// `FileBackend`) write it via an atomic tmp + fsync + rename, so
    /// readers see either the previous fully-committed state or the
    /// new one — never a torn record. See `docs/joule-db/cow-mvcc-design.md`.
    ///
    /// Default implementation returns `Ok(None)` for backends that do
    /// not yet persist meta records (memory, legacy disk, encrypted).
    fn read_committed_meta(&self) -> Result<Option<CommittedMeta>, StorageError> {
        Ok(None)
    }

    /// Atomically commit a new meta record. After this returns
    /// successfully, any subsequent `read_committed_meta` call must
    /// observe at least this record (no torn write may be exposed
    /// to a concurrent reader).
    ///
    /// Default implementation is a no-op so that backends opting out
    /// of MVCC (memory-only test backends, legacy disk) continue to
    /// compile. Engines using these backends fall back to in-memory
    /// root tracking.
    fn write_committed_meta(&mut self, _meta: &CommittedMeta) -> Result<(), StorageError> {
        Ok(())
    }

    /// **CoW MVCC Phase 4.** Returns true if any other process has a
    /// live snapshot against this database. Engines consult this
    /// before reclaiming pages — even when their own in-process
    /// snapshot counter is 0, a peer process's snapshot may still be
    /// reading pages we'd otherwise free.
    ///
    /// Default implementation returns `false` for backends that have
    /// no concept of cross-process sharing (e.g. `MemoryBackend`).
    /// `FileBackend` overrides to scan its `<db>/snapshots/` lockfile
    /// directory.
    fn any_external_snapshots_live(&self) -> bool {
        false
    }
}

/// Async storage backend trait
///
/// For backends that require async I/O (browser IndexedDB, async file I/O).
///
/// Note: This uses boxed futures to avoid depending on a specific async runtime.
#[cfg(feature = "async")]
pub trait AsyncStorageBackend: Send + Sync {
    /// Read a page asynchronously
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<Page>, StorageError>> + Send + '_>,
    >;

    /// Write a page asynchronously
    fn write_page(
        &self,
        page: Page,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Allocate a page asynchronously
    fn allocate_page(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PageId, StorageError>> + Send + '_>,
    >;

    /// Free a page asynchronously
    fn free_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Sync asynchronously
    fn sync(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), StorageError>> + Send + '_>>;

    /// Get page size
    fn page_size(&self) -> usize {
        DEFAULT_PAGE_SIZE
    }
}
