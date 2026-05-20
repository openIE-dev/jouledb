//! Per-page latch management for fine-grained concurrency control
//!
//! This module provides page-level locking (latching) to replace global storage
//! locks, enabling much higher concurrency for B-tree operations.
//!
//! ## Latch vs Lock
//!
//! In database terminology:
//! - **Lock**: Logical lock held for transaction duration (user-facing)
//! - **Latch**: Physical lock held briefly during page access (internal)
//!
//! Latches are lightweight, short-duration locks used to protect page contents
//! during read/modify operations.
//!
//! ## Design
//!
//! Since joule-db-core forbids unsafe code, we use a closure-based API
//! that ensures latches are properly released. This avoids the need for
//! self-referential structs or transmute.
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_core::concurrency::LatchManager;
//!
//! let manager = LatchManager::new();
//!
//! // Read with latch
//! let value = manager.with_read(page_id, || {
//!     // ... read page ...
//!     42
//! });
//!
//! // Write with latch
//! manager.with_write(page_id, || {
//!     // ... modify page ...
//! });
//! ```

use crate::storage::PageId;
use dashmap::DashMap;
use std::sync::{Arc, RwLock};

/// Latch acquisition mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatchMode {
    /// Shared read access - multiple readers allowed
    Read,
    /// Exclusive write access - no other readers or writers
    Write,
}

/// Configuration for the latch manager
#[derive(Debug, Clone)]
pub struct LatchManagerConfig {
    /// Number of shards for reduced contention
    pub num_shards: usize,
    /// Initial capacity per shard
    pub initial_capacity: usize,
}

impl Default for LatchManagerConfig {
    fn default() -> Self {
        Self {
            num_shards: 64,
            initial_capacity: 256,
        }
    }
}

/// A single page latch (RwLock wrapper)
struct PageLatch {
    lock: RwLock<()>,
}

impl std::fmt::Debug for PageLatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageLatch").finish()
    }
}

impl Default for PageLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl PageLatch {
    fn new() -> Self {
        Self {
            lock: RwLock::new(()),
        }
    }

    /// Execute a closure while holding a read latch
    fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = self.lock.read().expect("lock poisoned: page latch read");
        f()
    }

    /// Execute a closure while holding a write latch
    fn with_write<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = self.lock.write().expect("lock poisoned: page latch write");
        f()
    }

    /// Try to execute with read latch, returns None if unavailable
    fn try_with_read<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        let _guard = self.lock.try_read().ok()?;
        Some(f())
    }

    /// Try to execute with write latch, returns None if unavailable
    fn try_with_write<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        let _guard = self.lock.try_write().ok()?;
        Some(f())
    }
}

/// Manager for per-page latches
///
/// Uses DashMap for per-shard concurrent latch storage, eliminating the
/// RwLock<HashMap> bottleneck from the previous sharded implementation.
///
/// This implementation uses a closure-based API to ensure latches are
/// always properly released, avoiding the need for unsafe code.
pub struct LatchManager {
    latches: DashMap<PageId, Arc<PageLatch>>,
}

impl LatchManager {
    /// Create a new latch manager with default configuration
    pub fn new() -> Self {
        Self {
            latches: DashMap::new(),
        }
    }

    /// Create a latch manager with custom configuration
    pub fn with_config(_config: LatchManagerConfig) -> Self {
        // DashMap manages its own sharding internally
        Self::new()
    }

    /// Get or create a latch for a page
    #[inline]
    fn get_latch(&self, page_id: PageId) -> Arc<PageLatch> {
        self.latches
            .entry(page_id)
            .or_insert_with(|| Arc::new(PageLatch::new()))
            .clone()
    }

    /// Execute a closure while holding a read latch on a page
    ///
    /// Multiple readers can hold the latch simultaneously.
    pub fn with_read<F, R>(&self, page_id: PageId, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let latch = self.get_latch(page_id);
        latch.with_read(f)
    }

    /// Execute a closure while holding a write latch on a page
    ///
    /// Exclusive access - no other readers or writers.
    pub fn with_write<F, R>(&self, page_id: PageId, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let latch = self.get_latch(page_id);
        latch.with_write(f)
    }

    /// Try to execute with read latch, returns None if unavailable
    pub fn try_with_read<F, R>(&self, page_id: PageId, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        let latch = self.get_latch(page_id);
        latch.try_with_read(f)
    }

    /// Try to execute with write latch, returns None if unavailable
    pub fn try_with_write<F, R>(&self, page_id: PageId, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        let latch = self.get_latch(page_id);
        latch.try_with_write(f)
    }

    /// Release latch for a freed page
    pub fn release_page(&self, page_id: PageId) {
        self.latches.remove(&page_id);
    }

    /// Get total number of active latches
    pub fn active_latches(&self) -> usize {
        self.latches.len()
    }
}

impl Default for LatchManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about latch usage
#[derive(Debug, Clone, Default)]
pub struct LatchStats {
    /// Number of read latch acquisitions
    pub read_acquisitions: u64,
    /// Number of write latch acquisitions
    pub write_acquisitions: u64,
    /// Number of failed try_acquire attempts
    pub contention_events: u64,
    /// Current number of active latches
    pub active_latches: usize,
}

/// Helper for multiple page operations with proper latch ordering
///
/// For operations that need to access multiple pages, this helper
/// ensures latches are acquired in a consistent order to prevent deadlocks.
pub struct MultiPageLatch<'a> {
    manager: &'a LatchManager,
    pages: Vec<PageId>,
}

impl<'a> MultiPageLatch<'a> {
    /// Create a new multi-page latch helper
    pub fn new(manager: &'a LatchManager) -> Self {
        Self {
            manager,
            pages: Vec::new(),
        }
    }

    /// Add pages that will be accessed (call before execute)
    pub fn add_pages(&mut self, pages: impl IntoIterator<Item = PageId>) {
        self.pages.extend(pages);
        // Sort to ensure consistent ordering
        self.pages.sort();
        self.pages.dedup();
    }

    /// Execute with read latches on all pages
    pub fn with_read<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.with_read_recursive(0, f)
    }

    fn with_read_recursive<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if idx >= self.pages.len() {
            f()
        } else {
            self.manager
                .with_read(self.pages[idx], || self.with_read_recursive(idx + 1, f))
        }
    }

    /// Execute with write latches on all pages
    pub fn with_write<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.with_write_recursive(0, f)
    }

    fn with_write_recursive<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if idx >= self.pages.len() {
            f()
        } else {
            self.manager
                .with_write(self.pages[idx], || self.with_write_recursive(idx + 1, f))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    #[test]
    fn test_basic_latch_acquire_release() {
        let manager = LatchManager::new();

        let result = manager.with_read(1, || 42);
        assert_eq!(result, 42);

        let result = manager.with_write(1, || "hello");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_multiple_readers() {
        let manager = Arc::new(LatchManager::new());
        let counter = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];

        for _ in 0..4 {
            let manager = Arc::clone(&manager);
            let counter = Arc::clone(&counter);

            handles.push(thread::spawn(move || {
                manager.with_read(1, || {
                    counter.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(std::time::Duration::from_millis(10));
                    counter.fetch_sub(1, Ordering::SeqCst);
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_writer_exclusive() {
        let manager = Arc::new(LatchManager::new());
        let value = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];

        for _ in 0..4 {
            let manager = Arc::clone(&manager);
            let value = Arc::clone(&value);

            handles.push(thread::spawn(move || {
                manager.with_write(1, || {
                    // Should have exclusive access
                    let old = value.load(Ordering::SeqCst);
                    thread::sleep(std::time::Duration::from_millis(1));
                    value.store(old + 1, Ordering::SeqCst);
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All increments should have happened atomically
        assert_eq!(value.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_try_acquire() {
        let manager = Arc::new(LatchManager::new());
        let manager2 = Arc::clone(&manager);

        // Hold a write latch in another thread
        let handle = thread::spawn(move || {
            manager2.with_write(1, || {
                thread::sleep(std::time::Duration::from_millis(100));
            });
        });

        // Give the other thread time to acquire
        thread::sleep(std::time::Duration::from_millis(10));

        // Should fail - write latch already held
        let result = manager.try_with_write(1, || 42);
        assert!(result.is_none());

        // Different page should succeed
        let result = manager.try_with_write(2, || 42);
        assert_eq!(result, Some(42));

        handle.join().unwrap();
    }

    #[test]
    fn test_different_pages_no_contention() {
        let manager = Arc::new(LatchManager::new());
        let mut handles = vec![];

        for i in 0..8 {
            let manager = Arc::clone(&manager);

            handles.push(thread::spawn(move || {
                // Each thread works on a different page
                manager.with_write(i, || {
                    thread::sleep(std::time::Duration::from_millis(10));
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_active_latches_count() {
        let manager = LatchManager::new();

        assert_eq!(manager.active_latches(), 0);

        manager.with_read(1, || {});
        manager.with_read(2, || {});
        manager.with_write(3, || {});

        // Latches are created and persist
        assert_eq!(manager.active_latches(), 3);

        // Release page 2's latch tracking
        manager.release_page(2);
        assert_eq!(manager.active_latches(), 2);
    }

    #[test]
    fn test_sharding() {
        let config = LatchManagerConfig {
            num_shards: 4,
            initial_capacity: 16,
        };
        let manager = LatchManager::with_config(config);

        // Access many pages - should distribute across shards
        for i in 0..100 {
            manager.with_read(i, || {});
        }

        // All 100 latches created
        assert_eq!(manager.active_latches(), 100);
    }

    #[test]
    fn test_nested_different_pages() {
        let manager = LatchManager::new();

        let result = manager.with_read(1, || manager.with_read(2, || manager.with_write(3, || 42)));

        assert_eq!(result, 42);
    }

    #[test]
    fn test_multi_page_latch() {
        let manager = LatchManager::new();

        let mut multi = MultiPageLatch::new(&manager);
        multi.add_pages([3, 1, 2]); // Will be sorted to [1, 2, 3]

        let result = multi.with_read(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_multi_page_prevents_deadlock() {
        let manager = Arc::new(LatchManager::new());
        let mut handles = vec![];

        // Multiple threads acquiring same pages in different order
        // MultiPageLatch sorts them to prevent deadlock
        for _ in 0..4 {
            let manager = Arc::clone(&manager);
            handles.push(thread::spawn(move || {
                let mut multi = MultiPageLatch::new(&manager);
                multi.add_pages([2, 1, 3]); // Different order each time
                multi.with_write(|| {
                    thread::sleep(std::time::Duration::from_millis(1));
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
