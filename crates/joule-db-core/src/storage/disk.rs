//! Disk-based storage backend
//!
//! Provides durable storage using lock-free positional I/O (`pread`/`pwrite`
//! on Unix, `seek_read`/`seek_write` on Windows) so concurrent multi-core
//! readers and writers never contend on a shared file-cursor mutex.

use std::fs::File;
#[cfg(not(any(unix, windows)))]
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::page::{DEFAULT_PAGE_SIZE, Page, PageId, PageType};
use super::traits::{StorageBackend, StorageStats};
use crate::error::StorageError;

#[cfg(unix)]
fn pread_exact_file(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn pwrite_all_file(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn pread_exact_file(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading page",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn pwrite_all_file(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_write(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "WriteZero writing page",
                ));
            }
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Disk-based storage backend
///
/// Stores pages in a single file using lock-free positional `pread`/`pwrite`
/// syscalls on Unix/Windows. Multiple reader threads can execute concurrent
/// cache-miss reads across NVMe queues with zero lock contention.
///
/// For zero-copy extent reads, use `MmapOverlay` from the `mmap_extent` module
/// alongside this backend. On Apple Silicon UMA, mmap'd extents are directly
/// GPU-accessible via Metal's StorageModeShared.
pub struct DiskBackend {
    #[cfg(any(unix, windows))]
    file: File,
    #[cfg(not(any(unix, windows)))]
    file: Mutex<File>,
    path: std::path::PathBuf,
    page_size: usize,
    page_count: AtomicU64,
    free_list_head: Mutex<PageId>,
    free_pages_count: AtomicU64,
    pages_read: AtomicU64,
    pages_written: AtomicU64,
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
            #[cfg(any(unix, windows))]
            file,
            #[cfg(not(any(unix, windows)))]
            file: Mutex::new(file),
            path,
            page_size,
            page_count: AtomicU64::new(page_count),
            free_list_head: Mutex::new(0),
            free_pages_count: AtomicU64::new(0),
            pages_read: AtomicU64::new(0),
            pages_written: AtomicU64::new(0),
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

        let len = backend.file_len()?;
        backend
            .page_count
            .store(len / page_size as u64, Ordering::SeqCst);

        Ok(backend)
    }

    #[inline]
    fn file_len(&self) -> Result<u64, StorageError> {
        #[cfg(any(unix, windows))]
        {
            Ok(self
                .file
                .metadata()
                .map_err(|e| StorageError::Io(e.to_string()))?
                .len())
        }
        #[cfg(not(any(unix, windows)))]
        {
            let file = self
                .file
                .lock()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            Ok(file
                .metadata()
                .map_err(|e| StorageError::Io(e.to_string()))?
                .len())
        }
    }

    fn init_metadata(&mut self, file_len: u64) -> Result<(), StorageError> {
        if file_len == 0 {
            let mut head_page = Page::new(1, PageType::Metadata);
            head_page.data.extend_from_slice(&0u64.to_le_bytes());

            self.write_page_internal(head_page)?;

            self.page_count.store(1, Ordering::SeqCst);
            *self
                .free_list_head
                .lock()
                .expect("lock poisoned: free_list_head") = 0;
        } else {
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
                    *self
                        .free_list_head
                        .lock()
                        .expect("lock poisoned: free_list_head") = 0;
                }
                None => {}
            }
        }
        Ok(())
    }

    fn read_page_internal(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        self.read_page(page_id)
    }

    fn write_page_internal(&self, page: Page) -> Result<(), StorageError> {
        let offset = (page.id - 1) as u64 * self.page_size as u64;
        let encoded = page.encode(self.page_size)?;
        #[cfg(any(unix, windows))]
        {
            pwrite_all_file(&self.file, &encoded, offset)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut file = self
                .file
                .lock()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| StorageError::Io(e.to_string()))?;
            file.write_all(&encoded)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
        self.pages_written.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Lock-free positional page write (`pwrite`) usable from `&self` by concurrent callers
    /// writing disjoint page IDs.
    pub fn write_page_concurrent(&self, page: Page) -> Result<(), StorageError> {
        if page.id == 0 {
            return Err(StorageError::Backend(
                "Cannot write to null page ID 0".to_string(),
            ));
        }
        self.write_page_internal(page)
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

        let offset = (page_id - 1) as u64 * self.page_size as u64;
        let file_len = self.file_len()?;
        if offset >= file_len {
            return Ok(None);
        }

        let mut buf = vec![0u8; self.page_size];
        #[cfg(any(unix, windows))]
        {
            pread_exact_file(&self.file, &mut buf, offset).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    StorageError::Backend("Unexpected EOF reading page".to_string())
                } else {
                    StorageError::Io(e.to_string())
                }
            })?;
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut file = self
                .file
                .lock()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| StorageError::Io(e.to_string()))?;
            file.read_exact(&mut buf).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    StorageError::Backend("Unexpected EOF reading page".to_string())
                } else {
                    StorageError::Io(e.to_string())
                }
            })?;
        }

        self.pages_read.fetch_add(1, Ordering::Relaxed);

        match Page::decode(&buf) {
            Ok(page) => {
                if page.id != page_id {
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
        self.write_page_internal(page)
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
            self.free_pages_count
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                    Some(c.saturating_sub(1))
                })
                .ok();

            return Ok(page_id);
        }

        let new_id = self.page_count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(new_id)
    }

    fn allocate_contiguous(&mut self, count: usize) -> Result<PageId, StorageError> {
        if count == 0 {
            return Err(StorageError::Backend("Cannot allocate 0 pages".to_string()));
        }
        // Always allocate at end of file to guarantee contiguity.
        // Bypasses free list — these pages are fresh, sequential.
        let prev = self.page_count.fetch_add(count as u64, Ordering::SeqCst);
        Ok(prev + 1)
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
        self.free_pages_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        #[cfg(any(unix, windows))]
        {
            self.file
                .sync_all()
                .map_err(|e| StorageError::Io(e.to_string()))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let file = self
                .file
                .lock()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            file.sync_all().map_err(|e| StorageError::Io(e.to_string()))
        }
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn stats(&self) -> StorageStats {
        StorageStats {
            total_pages: self.page_count.load(Ordering::Relaxed),
            free_pages: self.free_pages_count.load(Ordering::Relaxed),
            pages_read: self.pages_read.load(Ordering::Relaxed),
            pages_written: self.pages_written.load(Ordering::Relaxed),
            page_size: self.page_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::page::PageType;
    use std::sync::Arc;
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

        let stats = backend.stats();
        assert!(stats.pages_read >= 1);
        assert!(stats.pages_written >= 2);
    }

    #[test]
    fn test_lockfree_concurrent_positional_io() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("concurrent_pread_pwrite.db");
        let mut backend = DiskBackend::open(&path).unwrap();

        let num_threads = 8usize;
        let pages_per_thread = 32usize;
        let first_page = backend
            .allocate_contiguous(num_threads * pages_per_thread)
            .unwrap();
        backend.sync().unwrap();

        let shared = Arc::new(backend);

        // Concurrent disjoint pwrite across 8 OS threads without any Mutex<File>
        let mut writers = Vec::new();
        for t in 0..num_threads {
            let db = Arc::clone(&shared);
            writers.push(std::thread::spawn(move || {
                for i in 0..pages_per_thread {
                    let pid = first_page + (t * pages_per_thread + i) as u64;
                    let payload = format!("thread-{t}-page-{i}-payload-{pid}").into_bytes();
                    let page = Page::with_data(pid, PageType::BTreeLeaf, payload);
                    db.write_page_concurrent(page).unwrap();
                }
            }));
        }
        for w in writers {
            w.join().unwrap();
        }

        // Concurrent lock-free pread verification across 8 OS threads
        let mut readers = Vec::new();
        for t in 0..num_threads {
            let db = Arc::clone(&shared);
            readers.push(std::thread::spawn(move || {
                for round in 0..4 {
                    for i in 0..pages_per_thread {
                        let pid = first_page
                            + (((t + round) % num_threads) * pages_per_thread + i) as u64;
                        let expected_t = (t + round) % num_threads;
                        let expected =
                            format!("thread-{expected_t}-page-{i}-payload-{pid}").into_bytes();
                        let got = db.read_page(pid).unwrap().unwrap();
                        assert_eq!(got.id, pid);
                        assert_eq!(got.data, expected);
                    }
                }
            }));
        }
        for r in readers {
            r.join().unwrap();
        }

        let stats = shared.stats();
        assert_eq!(
            stats.total_pages,
            1 + (num_threads * pages_per_thread) as u64
        );
        assert!(stats.pages_written >= (num_threads * pages_per_thread) as u64);
        assert!(stats.pages_read >= (num_threads * pages_per_thread * 4) as u64);
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
