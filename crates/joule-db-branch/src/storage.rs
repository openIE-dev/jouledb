//! Copy-on-Write Branch Storage
//!
//! Wraps a parent `StorageBackend` with a page-level indirection map.
//! Reads check the branch-local overlay first; writes always go to the overlay.
//! Unmodified pages are served directly from the parent — zero copy cost.

use joule_db_core::error::StorageError;
use joule_db_core::storage::{Page, PageId, StorageBackend, StorageStats};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// A copy-on-write storage layer that overlays a parent backend.
///
/// Pages written on this branch are stored in `overlay`. Reads first check
/// the overlay, then fall through to the parent. This gives O(1) branch
/// creation and storage proportional only to the diff.
pub struct CowBranchStorage {
    /// Parent storage (shared, read-only from this branch's perspective)
    parent: Arc<dyn StorageBackend>,

    /// Branch-local page overlay (modified/new pages)
    overlay: RwLock<HashMap<PageId, Page>>,

    /// Pages that have been freed on this branch
    freed: RwLock<HashSet<PageId>>,

    /// Next page ID for allocations on this branch
    next_page_id: RwLock<PageId>,

    /// Branch name (for diagnostics)
    branch_name: String,
}

impl CowBranchStorage {
    /// Create a new CoW branch storage overlaying a parent
    pub fn new(parent: Arc<dyn StorageBackend>, branch_name: &str, next_page_id: PageId) -> Self {
        Self {
            parent,
            overlay: RwLock::new(HashMap::new()),
            freed: RwLock::new(HashSet::new()),
            next_page_id: RwLock::new(next_page_id),
            branch_name: branch_name.to_string(),
        }
    }

    /// Number of pages in the overlay (branch-local modifications)
    pub fn overlay_size(&self) -> usize {
        self.overlay.read().map(|o| o.len()).unwrap_or(0)
    }

    /// Get the set of modified page IDs
    pub fn modified_pages(&self) -> Vec<PageId> {
        self.overlay
            .read()
            .map(|o| o.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Get the set of freed page IDs
    pub fn freed_pages(&self) -> Vec<PageId> {
        self.freed
            .read()
            .map(|f| f.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Merge all overlay pages back into a target storage backend.
    /// Used during branch merge operations.
    pub fn drain_overlay(&self) -> Vec<(PageId, Page)> {
        let mut overlay = self.overlay.write().unwrap_or_else(|e| e.into_inner());
        overlay.drain().collect()
    }
}

impl StorageBackend for CowBranchStorage {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        // Check if freed on this branch
        if let Ok(freed) = self.freed.read() {
            if freed.contains(&page_id) {
                return Ok(None);
            }
        }

        // Check overlay first (branch-local writes)
        if let Ok(overlay) = self.overlay.read() {
            if let Some(page) = overlay.get(&page_id) {
                return Ok(Some(page.clone()));
            }
        }

        // Fall through to parent
        self.parent.read_page(page_id)
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        let mut overlay = self
            .overlay
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        overlay.insert(page.id, page);
        Ok(())
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        let mut next_id = self
            .next_page_id
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        // Remove from overlay if present
        if let Ok(mut overlay) = self.overlay.write() {
            overlay.remove(&page_id);
        }

        // Mark as freed so reads return None
        if let Ok(mut freed) = self.freed.write() {
            freed.insert(page_id);
        }

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        // Overlay is in-memory; sync is a no-op until merge
        Ok(())
    }

    fn page_size(&self) -> usize {
        self.parent.page_size()
    }

    fn stats(&self) -> StorageStats {
        let overlay_count = self.overlay_size() as u64;
        let parent_stats = self.parent.stats();
        StorageStats {
            total_pages: parent_stats.total_pages + overlay_count,
            free_pages: parent_stats.free_pages,
            pages_read: parent_stats.pages_read,
            pages_written: parent_stats.pages_written + overlay_count,
            page_size: parent_stats.page_size,
        }
    }

    fn page_exists(&self, page_id: PageId) -> Result<bool, StorageError> {
        // Check freed
        if let Ok(freed) = self.freed.read() {
            if freed.contains(&page_id) {
                return Ok(false);
            }
        }

        // Check overlay
        if let Ok(overlay) = self.overlay.read() {
            if overlay.contains_key(&page_id) {
                return Ok(true);
            }
        }

        // Fall through to parent
        self.parent.page_exists(page_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::storage::{DEFAULT_PAGE_SIZE, Page, PageType};
    use std::sync::Arc;

    /// Simple in-memory backend for testing
    struct MemBackend {
        pages: HashMap<PageId, Page>,
        next_id: PageId,
    }

    impl MemBackend {
        fn new() -> Self {
            Self {
                pages: HashMap::new(),
                next_id: 1,
            }
        }
    }

    impl StorageBackend for MemBackend {
        fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
            Ok(self.pages.get(&page_id).cloned())
        }
        fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
            self.pages.insert(page.id, page);
            Ok(())
        }
        fn allocate_page(&mut self) -> Result<PageId, StorageError> {
            let id = self.next_id;
            self.next_id += 1;
            Ok(id)
        }
        fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
            self.pages.remove(&page_id);
            Ok(())
        }
        fn sync(&mut self) -> Result<(), StorageError> {
            Ok(())
        }
        fn stats(&self) -> StorageStats {
            StorageStats {
                total_pages: self.pages.len() as u64,
                free_pages: 0,
                pages_read: 0,
                pages_written: 0,
                page_size: DEFAULT_PAGE_SIZE,
            }
        }
        fn page_exists(&self, page_id: PageId) -> Result<bool, StorageError> {
            Ok(self.pages.contains_key(&page_id))
        }
    }

    fn make_page(id: PageId, data: &[u8]) -> Page {
        Page::with_data(id, PageType::BTreeLeaf, data.to_vec())
    }

    #[test]
    fn test_cow_read_through() {
        let mut parent = MemBackend::new();
        parent.write_page(make_page(1, b"parent-data")).unwrap();
        let parent = Arc::new(parent) as Arc<dyn StorageBackend>;

        let cow = CowBranchStorage::new(parent, "test-branch", 100);

        // Read should fall through to parent
        let page = cow.read_page(1).unwrap().unwrap();
        assert_eq!(&page.data[..11], b"parent-data");
    }

    #[test]
    fn test_cow_write_overlay() {
        let mut parent = MemBackend::new();
        parent.write_page(make_page(1, b"parent-data")).unwrap();
        let parent = Arc::new(parent) as Arc<dyn StorageBackend>;

        let mut cow = CowBranchStorage::new(parent.clone(), "test-branch", 100);

        // Write to overlay
        cow.write_page(make_page(1, b"branch-data")).unwrap();

        // Read should return branch version
        let page = cow.read_page(1).unwrap().unwrap();
        assert_eq!(&page.data[..11], b"branch-data");

        // Parent should be unchanged
        let parent_page = parent.read_page(1).unwrap().unwrap();
        assert_eq!(&parent_page.data[..11], b"parent-data");
    }

    #[test]
    fn test_cow_free_page() {
        let mut parent = MemBackend::new();
        parent.write_page(make_page(1, b"parent-data")).unwrap();
        let parent = Arc::new(parent) as Arc<dyn StorageBackend>;

        let mut cow = CowBranchStorage::new(parent, "test-branch", 100);

        // Free page on branch
        cow.free_page(1).unwrap();

        // Should return None (freed on this branch)
        assert!(cow.read_page(1).unwrap().is_none());
        assert!(!cow.page_exists(1).unwrap());
    }

    #[test]
    fn test_cow_allocate() {
        let parent = Arc::new(MemBackend::new()) as Arc<dyn StorageBackend>;
        let mut cow = CowBranchStorage::new(parent, "test-branch", 100);

        assert_eq!(cow.allocate_page().unwrap(), 100);
        assert_eq!(cow.allocate_page().unwrap(), 101);
    }

    #[test]
    fn test_cow_overlay_tracking() {
        let parent = Arc::new(MemBackend::new()) as Arc<dyn StorageBackend>;
        let mut cow = CowBranchStorage::new(parent, "test-branch", 100);

        assert_eq!(cow.overlay_size(), 0);

        cow.write_page(make_page(1, b"data1")).unwrap();
        cow.write_page(make_page(2, b"data2")).unwrap();

        assert_eq!(cow.overlay_size(), 2);

        let modified = cow.modified_pages();
        assert!(modified.contains(&1));
        assert!(modified.contains(&2));
    }

    #[test]
    fn test_cow_drain_overlay() {
        let parent = Arc::new(MemBackend::new()) as Arc<dyn StorageBackend>;
        let mut cow = CowBranchStorage::new(parent, "test-branch", 100);

        cow.write_page(make_page(1, b"data1")).unwrap();
        cow.write_page(make_page(2, b"data2")).unwrap();

        let drained = cow.drain_overlay();
        assert_eq!(drained.len(), 2);
        assert_eq!(cow.overlay_size(), 0);
    }
}
