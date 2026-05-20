//! Zero-copy mmap access for extent-based large blob storage.
//!
//! This module contains the unsafe mmap operations needed for zero-copy
//! reads of contiguous extents. On Apple Silicon UMA, the returned pointers
//! are directly GPU-accessible via Metal's StorageModeShared.
//!
//! # Safety
//!
//! The ExtentSlice type holds an Arc<Mmap> that keeps the mapping alive.
//! Callers must not outlive the Arc. The data is read-only.
#![allow(unsafe_code)]

use super::page::{PageId, PAGE_HEADER_SIZE};
use crate::error::StorageError;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Extent header size in data: page_count (u64) + total_bytes (u64)
const EXTENT_HEADER_SIZE: usize = 16;

/// Manages a memory-mapped view of a JouleDB database file.
///
/// Provides zero-copy access to extent data by returning pointers
/// directly into the mmap'd region. Thread-safe via internal mutex.
pub struct MmapOverlay {
    path: std::path::PathBuf,
    mmap: Mutex<Option<Arc<memmap2::Mmap>>>,
}

impl MmapOverlay {
    /// Create a new mmap overlay for the given database file.
    /// The mmap is created lazily on first access.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            mmap: Mutex::new(None),
        }
    }

    /// Get or create the mmap. Refreshes if file has grown.
    pub fn ensure(&self) -> Result<Arc<memmap2::Mmap>, StorageError> {
        let mut lock = self
            .mmap
            .lock()
            .map_err(|_| StorageError::Backend("mmap lock poisoned".to_string()))?;

        let file = std::fs::File::open(&self.path)
            .map_err(|e| StorageError::Io(format!("mmap open: {}", e)))?;

        let file_len = file
            .metadata()
            .map_err(|e| StorageError::Io(format!("mmap metadata: {}", e)))?
            .len() as usize;

        let needs_refresh = match &*lock {
            Some(existing) => existing.len() < file_len,
            None => true,
        };

        if needs_refresh {
            let new_mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| StorageError::Io(format!("mmap failed: {}", e)))?
            };
            *lock = Some(Arc::new(new_mmap));
        }

        Ok(lock.as_ref().unwrap().clone())
    }

    /// Invalidate the cached mmap (e.g., after writes that grew the file).
    pub fn invalidate(&self) {
        if let Ok(mut lock) = self.mmap.lock() {
            *lock = None;
        }
    }

    /// Create a zero-copy view of an extent.
    ///
    /// Returns an `ExtentSlice` that provides per-page data pointers
    /// directly into the mmap'd file region. No buffer pool, no locks,
    /// no copying.
    pub fn slice_extent(
        &self,
        first_page_id: PageId,
        page_count: usize,
        page_size: usize,
        total_bytes: usize,
    ) -> Result<ExtentSlice, StorageError> {
        let mmap = self.ensure()?;

        let first_offset = (first_page_id - 1) as usize * page_size;
        let extent_end = first_offset + page_count * page_size;

        if extent_end > mmap.len() {
            return Err(StorageError::Backend(format!(
                "Extent [{}, +{}] ({}B) exceeds mmap size {}",
                first_page_id, page_count, extent_end, mmap.len()
            )));
        }

        Ok(ExtentSlice {
            mmap,
            first_page_id,
            page_count,
            page_size,
            total_bytes,
        })
    }
}

/// Zero-copy view into a contiguous extent in the database file.
///
/// Holds an Arc to the mmap, keeping it alive. Provides access to
/// per-page data that skips page headers — suitable for direct GPU
/// access on Apple Silicon UMA.
pub struct ExtentSlice {
    mmap: Arc<memmap2::Mmap>,
    first_page_id: PageId,
    page_count: usize,
    page_size: usize,
    total_bytes: usize,
}

// Safety: ExtentSlice is read-only, mmap is Arc-protected.
unsafe impl Send for ExtentSlice {}
unsafe impl Sync for ExtentSlice {}

impl ExtentSlice {
    /// Total data bytes across all pages in the extent.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of pages in the extent.
    pub fn page_count(&self) -> usize {
        self.page_count
    }

    /// Get a slice of the data in page `i` of the extent (0-indexed).
    ///
    /// Skips the page header (32 bytes). For page 0 (ExtentHeader), also
    /// skips the 16-byte extent header, so the returned slice starts at
    /// actual tensor data.
    pub fn page_data(&self, page_idx: usize) -> Option<&[u8]> {
        if page_idx >= self.page_count {
            return None;
        }

        let page_id = self.first_page_id + page_idx as u64;
        let file_offset = (page_id - 1) as usize * self.page_size;

        if page_idx == 0 {
            let data_offset = file_offset + PAGE_HEADER_SIZE + EXTENT_HEADER_SIZE;
            let max_data = self.page_size - PAGE_HEADER_SIZE - EXTENT_HEADER_SIZE;
            let usable = max_data.min(self.total_bytes);
            Some(&self.mmap[data_offset..data_offset + usable])
        } else {
            let data_offset = file_offset + PAGE_HEADER_SIZE;
            let max_data = self.page_size - PAGE_HEADER_SIZE;
            let consumed = (self.page_size - PAGE_HEADER_SIZE - EXTENT_HEADER_SIZE)
                + (page_idx - 1) * (self.page_size - PAGE_HEADER_SIZE);
            let remaining = self.total_bytes.saturating_sub(consumed);
            let usable = max_data.min(remaining);
            if usable == 0 {
                return None;
            }
            Some(&self.mmap[data_offset..data_offset + usable])
        }
    }

    /// Copy the entire extent into a contiguous Vec.
    ///
    /// Faster than the buffer pool path — source is mmap'd, no file I/O,
    /// no locks, no page decoding overhead.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.total_bytes);
        for i in 0..self.page_count {
            if result.len() >= self.total_bytes {
                break;
            }
            if let Some(data) = self.page_data(i) {
                let remaining = self.total_bytes - result.len();
                let to_copy = data.len().min(remaining);
                result.extend_from_slice(&data[..to_copy]);
            }
        }
        result
    }

    /// Get a raw pointer to page data for GPU buffer creation.
    ///
    /// On Apple Silicon UMA, this pointer can be passed directly to
    /// `Metal::new_buffer_with_bytes_no_copy()` for zero-copy GPU access.
    ///
    /// Returns `(pointer, length)` for the data portion of the given page.
    pub fn page_ptr(&self, page_idx: usize) -> Option<(*const u8, usize)> {
        let data = self.page_data(page_idx)?;
        Some((data.as_ptr(), data.len()))
    }

    /// Get the backing mmap Arc (keeps the mapping alive).
    pub fn mmap_arc(&self) -> &Arc<memmap2::Mmap> {
        &self.mmap
    }
}
