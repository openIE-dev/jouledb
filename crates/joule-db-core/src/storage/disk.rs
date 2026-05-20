//! Disk-based storage backend
//!
//! Provides durable storage using the filesystem.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

use super::page::{DEFAULT_PAGE_SIZE, Page, PageId, PageType};
use super::traits::{StorageBackend, StorageStats};
use crate::error::StorageError;

/// Disk-based storage backend
///
/// Stores pages in a single file. Thread-safe using internal mutex.
///
/// For zero-copy extent reads, use `MmapOverlay` from the `mmap_extent` module
/// alongside this backend. On Apple Silicon UMA, mmap'd extents are directly
/// GPU-accessible via Metal's StorageModeShared.
pub struct DiskBackend {
    file: Mutex<File>,
    path: std::path::PathBuf,
    page_size: usize,
    page_count: Mutex<u64>,
    free_list_head: Mutex<PageId>,
}

impl DiskBackend {
    /// Open or create a disk backend at the specified path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();

        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let metadata = file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let len = metadata.len();
        let page_size = DEFAULT_PAGE_SIZE;
        let page_count = len / page_size as u64;

        let mut backend = Self {
            file: Mutex::new(file),
            path,
            page_size,
            page_count: Mutex::new(page_count),
            free_list_head: Mutex::new(0),
        };

        // Initialize or load metadata
        backend.init_metadata(len)?;

        Ok(backend)
    }

    /// Open with custom page size (mostly for testing)
    pub fn open_with_page_size<P: AsRef<Path>>(
        path: P,
        page_size: usize,
    ) -> Result<Self, StorageError> {
        let mut backend = Self::open(path)?;
        backend.page_size = page_size;

        {
            let file = backend
                .file
                .lock()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            let len = file
                .metadata()
                .map_err(|e| StorageError::Io(e.to_string()))?
                .len();
            *backend
                .page_count
                .lock()
                .expect("lock poisoned: page_count") = len / page_size as u64;
        }

        // Re-init metadata with new page size if needed/implied
        // Note: Changing page size on existing DB invalidates data, assuming clean slate for tests

        Ok(backend)
    }

    fn init_metadata(&mut self, file_len: u64) -> Result<(), StorageError> {
        if file_len == 0 {
            // New database: Create metadata page at Page 1
            // We use Page 1 as metadata page (Offset 0)
            // Page ID 1 maps to Offset 0 in our scheme
            let mut head_page = Page::new(1, PageType::Metadata);
            // Write initial free list head (0 = empty)
            head_page.data.extend_from_slice(&0u64.to_le_bytes());

            // Write directly using internal helper query to avoid recursion or checks
            self.write_page_internal(head_page)?;

            *self.page_count.lock().expect("lock poisoned: page_count") = 1;
            *self
                .free_list_head
                .lock()
                .expect("lock poisoned: free_list_head") = 0;
        } else {
            // Existing database: Read Page 1
            match self.read_page_internal(1)? {
                Some(page) if page.page_type == PageType::Metadata => {
                    if page.data.len() >= 8 {
                        let head = u64::from_le_bytes(page.data[0..8].try_into().expect("8 bytes"));
                        *self
                            .free_list_head
                            .lock()
                            .expect("lock poisoned: free_list_head") = head;
                    }
                }
                Some(_) => {
                    // Page 1 exists but is not Metadata?
                    // This implies legacy DB or corrupted.
                    // For now, we defaulting free list to 0 (append only) to be safe.
                    // Ideally we should migration here.
                    *self
                        .free_list_head
                        .lock()
                        .expect("lock poisoned: free_list_head") = 0;
                }
                None => {
                    // Should not happen if len > 0
                }
            }
        }
        Ok(())
    }

    // internal helper to avoid locking self.file multiple times if we composed methods
    fn read_page_internal(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        // ... (Logic from read_page)
        // Duplicating logic here is messy without refactoring read_page to use this.
        // Let's just call self.read_page() since it's public and effectively stateless regarding 'self' mutation,
        // except keeping the file lock which we don't hold across calls in init_metadata
        self.read_page(page_id)
    }

    fn write_page_internal(&self, page: Page) -> Result<(), StorageError> {
        let mut file = self
            .file
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        let offset = (page.id - 1) as u64 * self.page_size as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let encoded = page.encode(self.page_size)?;
        file.write_all(&encoded)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(())
    }

    /// Get the file path for this backend.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl StorageBackend for DiskBackend {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        if page_id == 0 {
            return Err(StorageError::Backend(
                "Cannot read null page ID 0".to_string(),
            ));
        }

        let mut file = self
            .file
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        // Calculate offset
        // Page ID 1 is at offset 0 (0-indexed file, 1-indexed IDs usually, but let's assume 1-based IDs map to offset (id-1)*size)
        // If IDs are just raw indices, we can use id * size.
        // Let's assume PageId is an index starting at 0 for simplicity in mapping,
        // BUT traits.rs says NULL_PAGE_ID is 0. So valid pages start at 1.
        // Map PageId 1 -> Offset 0.
        let offset = (page_id - 1) as u64 * self.page_size as u64;

        let file_len = file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?
            .len();
        if offset >= file_len {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let mut buf = vec![0u8; self.page_size];
        file.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                StorageError::Backend("Unexpected EOF reading page".to_string())
            } else {
                StorageError::Io(e.to_string())
            }
        })?;

        // Decode the page
        // Page::decode expects the buffer to start with the header
        match Page::decode(&buf) {
            Ok(page) => {
                // Verify ID matches
                if page.id != page_id {
                    // specific case: free pages or zeroed pages might allow mismatch if we handle it?
                    // For now, strict check.
                    return Err(StorageError::Backend(format!(
                        "Page ID mismatch at offset {}: expected {}, got {}",
                        offset, page_id, page.id
                    )));
                }
                Ok(Some(page))
            }
            Err(e) => Err(StorageError::Serialization(e.to_string())),
        }
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        if page.id == 0 {
            return Err(StorageError::Backend(
                "Cannot write to null page ID 0".to_string(),
            ));
        }

        let mut file = self
            .file
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        let offset = (page.id - 1) as u64 * self.page_size as u64;

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let encoded = page.encode(self.page_size)?;
        file.write_all(&encoded)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        Ok(())
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        let mut head_lock = self
            .free_list_head
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        if *head_lock != 0 {
            // Re-use a free page
            let page_id = *head_lock;

            // Read the page to find the next free page
            // We use read_page instead of read_page_internal because we are in &mut self context so no conflict?
            // Actually read_page takes &self, allocate_page takes &mut self.
            // The issue is if read_page tries to lock something we already locked.
            // We locked head_lock (free_list_head). read_page locks `file`. Different locks. Safe.
            let page = self.read_page(page_id)?.ok_or_else(|| {
                StorageError::Backend(format!("Free list head page {} not found", page_id))
            })?;

            if page.data.len() < 8 {
                return Err(StorageError::Backend(format!(
                    "Corrupted free page {}",
                    page_id
                )));
            }

            let next_free = u64::from_le_bytes(page.data[0..8].try_into().expect("8 bytes"));

            // Update metadata page (Page 1)
            let mut meta_page = self
                .read_page(1)?
                .ok_or_else(|| StorageError::Backend("Metadata page missing".to_string()))?;
            // Ensure metadata page has enough space
            if meta_page.data.len() < 8 {
                meta_page.data.resize(8, 0);
            }
            meta_page.data[0..8].copy_from_slice(&next_free.to_le_bytes());

            // Sync to disk
            self.write_page_internal(meta_page)?;

            // Update memory
            *head_lock = next_free;

            return Ok(page_id);
        }

        let mut count = self
            .page_count
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        *count += 1;
        Ok(*count)
    }

    fn allocate_contiguous(&mut self, count: usize) -> Result<PageId, StorageError> {
        if count == 0 {
            return Err(StorageError::Backend("Cannot allocate 0 pages".to_string()));
        }
        // Always allocate at end of file to guarantee contiguity.
        // Bypasses free list — these pages are fresh, sequential.
        let mut page_count_lock = self
            .page_count
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        let first_id = *page_count_lock + 1;
        *page_count_lock += count as u64;
        Ok(first_id)
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        if page_id == 0 || page_id == 1 {
            return Err(StorageError::Backend(
                "Cannot free reserved page".to_string(),
            ));
        }

        let mut head_lock = self
            .free_list_head
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        // 1. Create free page pointing to current head
        let mut page = Page::new(page_id, PageType::Free);
        page.data.extend_from_slice(&head_lock.to_le_bytes());

        // 2. Write this page
        self.write_page_internal(page)?;

        // 3. Update metadata to point to this page
        let mut meta_page = self
            .read_page(1)?
            .ok_or_else(|| StorageError::Backend("Metadata page missing".to_string()))?;
        if meta_page.data.len() < 8 {
            meta_page.data.resize(8, 0);
        }
        meta_page.data[0..8].copy_from_slice(&page_id.to_le_bytes());
        self.write_page_internal(meta_page)?;

        // 4. Update memory
        *head_lock = page_id;

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        let file = self
            .file
            .lock()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        file.sync_all().map_err(|e| StorageError::Io(e.to_string()))
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn stats(&self) -> StorageStats {
        let count = *self.page_count.lock().unwrap_or_else(|e| e.into_inner()); // handle poison gracefully-ish
        StorageStats {
            total_pages: count,
            free_pages: 0, // Not tracking free pages yet
            pages_read: 0, // Not tracking I/O stats yet
            pages_written: 0,
            page_size: self.page_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::page::PageType;
    use tempfile::tempdir;

    #[test]
    fn test_disk_backend_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut backend = DiskBackend::open(&path).unwrap();

        // Allocate and write (Page 1 is reserved for metadata, so first data page is 2)
        let page_id = backend.allocate_page().unwrap();
        assert_eq!(page_id, 2);

        let data = b"test data".to_vec();
        let page = Page::with_data(page_id, PageType::BTreeLeaf, data.clone());
        backend.write_page(page).unwrap();
        backend.sync().unwrap();

        // Read back
        let read_page = backend.read_page(page_id).unwrap().unwrap();
        assert_eq!(read_page.id, page_id);
        assert_eq!(read_page.data, data);
    }

    #[test]
    fn test_reopen_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("persist.db");

        let page_id;
        {
            let mut backend = DiskBackend::open(&path).unwrap();
            page_id = backend.allocate_page().unwrap();
            let page = Page::with_data(page_id, PageType::BTreeLeaf, b"persistent".to_vec());
            backend.write_page(page).unwrap();
            backend.sync().unwrap(); // Ensure data is flushed to disk
        } // Close backend

        {
            // Reopen and read the same page we wrote
            let backend = DiskBackend::open(&path).unwrap();
            let read_page = backend.read_page(page_id).unwrap().unwrap();
            assert_eq!(read_page.data, b"persistent");
        }
    }
}
