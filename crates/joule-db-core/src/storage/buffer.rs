//! Buffer pool manager
//!
//! Manages in-memory cache of pages from the storage backend.
//! Uses sharded frame storage to reduce lock contention under concurrent access.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::page::{Page, PageId};
use super::traits::{CommittedMeta, StorageBackend};
use crate::error::StorageError;

/// Default number of shards for the buffer pool
const DEFAULT_NUM_SHARDS: usize = 16;

/// Minimum-valid empty B-tree leaf body (matches `engine::btree::BTreeNode::serialize`).
///
/// Layout: `is_leaf=1 (u8) | num_keys=0 (u32 LE)` = 5 bytes.
///
/// # Why this exists
///
/// Newly-allocated pages (via [`BufferPool::new_page`]) start life as dirty
/// residents of the frame cache before the caller has written their real
/// content. If the shard fills and [`evict_lru`](BufferPool::evict_lru)
/// picks such a page as victim, its pre-content state is flushed to disk
/// verbatim. Prior to this constant, that flushed state was
/// `type=Free, data_len=0` — indistinguishable from a never-written page.
/// A subsequent catalog or parent-node pointer to this page would then
/// walk into an empty body and raise `Storage::Corrupted { reason: "Empty node data" }`,
/// making the whole database unopenable.
///
/// Initialising `new_page` with this valid empty-leaf body means the worst
/// case on eviction is "a correctly-formatted but empty leaf node on disk"
/// — callers overwriting with real content still works; orphaned pages
/// degrade to "missing some rows" instead of "DB won't open". Observed
/// 2026-04-21 after ~2-3 GB of scholar.askdavidc.ai ingest repeatedly
/// reproduced this corruption.
const EMPTY_LEAF_BODY: &[u8] = &[
    1, // is_leaf = true
    0, 0, 0, 0, // num_keys = 0
];

/// Buffer pool configuration
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum number of pages in memory
    pub capacity: usize,
    /// Number of shards to reduce lock contention (must be power of 2)
    pub num_shards: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            capacity: 1000, // ~16MB with default page size
            num_shards: DEFAULT_NUM_SHARDS,
        }
    }
}

/// Per-frame metadata for eviction decisions.
struct FrameMeta {
    page: Arc<RwLock<Page>>,
    /// Monotonic access counter — higher = more recently accessed.
    last_access: u64,
    /// If true, this page will not be evicted.
    pinned: bool,
}

/// A single shard of the frame cache
struct FrameShard {
    frames: RwLock<HashMap<PageId, FrameMeta>>,
    capacity: usize,
}

impl FrameShard {
    fn new(capacity: usize) -> Self {
        Self {
            frames: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.frames.read().unwrap_or_else(|p| p.into_inner()).len()
    }
}

/// Buffer pool manager
///
/// Thread-safe manager for caching pages. Uses sharded frame storage
/// to reduce lock contention when multiple threads access different pages.
///
/// Supports tensor-aware features:
/// - **LRU eviction**: Evicts least-recently-accessed page (not arbitrary)
/// - **Pinning**: Pages can be pinned to prevent eviction (embeddings, active layer)
/// - **Bulk evict**: Evict all pages in a range (completed layer cleanup)
/// - **Bulk prefetch**: Load a range of pages (next layer read-ahead)
pub struct BufferPool {
    backend: Arc<RwLock<Box<dyn StorageBackend>>>,
    config: BufferPoolConfig,
    shards: Vec<FrameShard>,
    shard_mask: usize,
    /// Monotonic counter for LRU tracking
    access_counter: AtomicU64,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(backend: Arc<RwLock<Box<dyn StorageBackend>>>, config: BufferPoolConfig) -> Self {
        let num_shards = config.num_shards.next_power_of_two();
        let shard_mask = num_shards - 1;
        let per_shard_capacity = config.capacity / num_shards + 1;
        let shards = (0..num_shards)
            .map(|_| FrameShard::new(per_shard_capacity))
            .collect();
        Self {
            backend,
            config,
            shards,
            shard_mask,
            access_counter: AtomicU64::new(0),
        }
    }

    /// Get the shard for a given page ID
    #[inline]
    fn shard_for(&self, page_id: PageId) -> &FrameShard {
        &self.shards[page_id as usize & self.shard_mask]
    }

    /// Get a page from the pool, loading it if necessary.
    /// Updates LRU access counter on every access.
    pub fn get_page(&self, page_id: PageId) -> Result<Arc<RwLock<Page>>, StorageError> {
        let shard = self.shard_for(page_id);
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);

        // Fast path: check existence with read lock on shard
        {
            let mut frames = shard
                .frames
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            if let Some(meta) = frames.get_mut(&page_id) {
                meta.last_access = tick;
                return Ok(meta.page.clone());
            }
        }

        // Slow path: load from backend with write lock on shard
        let mut frames = shard
            .frames
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        // Double check in case another thread loaded it
        if let Some(meta) = frames.get_mut(&page_id) {
            meta.last_access = tick;
            return Ok(meta.page.clone());
        }

        // Evict if shard is full
        if frames.len() >= shard.capacity {
            self.evict_lru(&mut frames)?;
        }

        // Load page from backend
        let backend = self
            .backend
            .read()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        let page = backend
            .read_page(page_id)?
            .ok_or_else(|| StorageError::Backend(format!("Page {} not found", page_id)))?;

        let page_ref = Arc::new(RwLock::new(page));
        frames.insert(
            page_id,
            FrameMeta {
                page: page_ref.clone(),
                last_access: tick,
                pinned: false,
            },
        );

        Ok(page_ref)
    }

    /// Create a new page
    pub fn new_page(&self) -> Result<Arc<RwLock<Page>>, StorageError> {
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);

        // Allocate page ID from backend (global lock required)
        let page_id = {
            let mut backend = self
                .backend
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            backend.allocate_page()?
        };

        // Insert into the appropriate shard
        let shard = self.shard_for(page_id);
        let mut frames = shard
            .frames
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        if frames.len() >= shard.capacity {
            self.evict_lru(&mut frames)?;
        }

        // Seed the page with a valid empty-leaf body rather than leaving it
        // as PageType::Free / empty data. If eviction flushes this frame
        // before the caller writes real content, the disk state is still
        // a readable B-tree node (0 keys) — the parent pointer stays
        // navigable instead of trapping `Empty node data` on the next open.
        // See `EMPTY_LEAF_BODY` for the full write-up.
        let mut page = Page::new(page_id, crate::storage::page::PageType::BTreeLeaf);
        page.data = EMPTY_LEAF_BODY.to_vec();
        page.mark_dirty();
        let page_ref = Arc::new(RwLock::new(page));
        frames.insert(
            page_id,
            FrameMeta {
                page: page_ref.clone(),
                last_access: tick,
                pinned: false,
            },
        );

        Ok(page_ref)
    }

    /// Allocate N contiguous pages for extent storage.
    ///
    /// Returns the first page ID. All N pages are guaranteed sequential.
    /// Pages are inserted into the buffer pool as dirty (will be written on flush).
    pub fn allocate_contiguous(&self, count: usize) -> Result<PageId, StorageError> {
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);

        let first_id = {
            let mut backend = self
                .backend
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            backend.allocate_contiguous(count)?
        };

        // Create page entries in the buffer pool
        for i in 0..count {
            let page_id = first_id + i as u64;
            let shard = self.shard_for(page_id);
            let mut frames = shard
                .frames
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

            if frames.len() >= shard.capacity {
                self.evict_lru(&mut frames)?;
            }

            let mut page = Page::new(page_id, super::page::PageType::Free);
            page.mark_dirty();
            let page_ref = Arc::new(RwLock::new(page));
            frames.insert(
                page_id,
                FrameMeta {
                    page: page_ref,
                    last_access: tick,
                    pinned: false,
                },
            );
        }

        Ok(first_id)
    }

    /// Flush a specific page to disk
    pub fn flush_page(&self, page: &mut Page) -> Result<(), StorageError> {
        if page.is_dirty() {
            let mut backend = self
                .backend
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            backend.write_page(page.clone())?;
            page.clear_dirty();
        }
        Ok(())
    }

    /// Flush all dirty pages
    pub fn flush_all(&self) -> Result<(), StorageError> {
        let mut backend = self
            .backend
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        for shard in &self.shards {
            let frames = shard
                .frames
                .read()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

            for meta in frames.values() {
                let mut page = meta
                    .page
                    .write()
                    .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
                if page.is_dirty() {
                    backend.write_page(page.clone())?;
                    page.clear_dirty();
                }
            }
        }

        backend.sync()?;
        Ok(())
    }

    /// Free a page both from cache and backend
    pub fn free_page(&self, page_id: PageId) -> Result<(), StorageError> {
        {
            let mut backend = self
                .backend
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            backend.free_page(page_id)?;
        }

        {
            let shard = self.shard_for(page_id);
            let mut frames = shard
                .frames
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            frames.remove(&page_id);
        }

        Ok(())
    }

    // ── Tensor-aware buffer pool operations ─────────────────────────

    /// Pin a page so it won't be evicted.
    ///
    /// Use for weights that must stay resident: embeddings, LM head,
    /// shared MoE experts, current layer during forward pass.
    pub fn pin_page(&self, page_id: PageId) -> Result<(), StorageError> {
        let shard = self.shard_for(page_id);
        let mut frames = shard
            .frames
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        if let Some(meta) = frames.get_mut(&page_id) {
            meta.pinned = true;
        }
        Ok(())
    }

    /// Unpin a page, allowing it to be evicted.
    pub fn unpin_page(&self, page_id: PageId) -> Result<(), StorageError> {
        let shard = self.shard_for(page_id);
        let mut frames = shard
            .frames
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        if let Some(meta) = frames.get_mut(&page_id) {
            meta.pinned = false;
        }
        Ok(())
    }

    /// Pin a contiguous range of pages (e.g., an entire extent).
    pub fn pin_range(&self, first_page_id: PageId, count: usize) -> Result<(), StorageError> {
        for i in 0..count {
            self.pin_page(first_page_id + i as u64)?;
        }
        Ok(())
    }

    /// Unpin a contiguous range of pages.
    pub fn unpin_range(&self, first_page_id: PageId, count: usize) -> Result<(), StorageError> {
        for i in 0..count {
            self.unpin_page(first_page_id + i as u64)?;
        }
        Ok(())
    }

    /// Evict a contiguous range of pages from the buffer pool.
    ///
    /// Use after a layer's forward pass completes — the pages are no longer
    /// needed and should free budget for the next layer's pages.
    /// Dirty pages are flushed before eviction. Pinned pages are skipped.
    pub fn evict_range(&self, first_page_id: PageId, count: usize) -> Result<(), StorageError> {
        let mut backend = self
            .backend
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        for i in 0..count {
            let page_id = first_page_id + i as u64;
            let shard = self.shard_for(page_id);
            let mut frames = shard
                .frames
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

            if let Some(meta) = frames.get(&page_id) {
                if meta.pinned {
                    continue; // Don't evict pinned pages
                }
                // Flush if dirty
                let mut page = meta
                    .page
                    .write()
                    .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
                if page.is_dirty() {
                    backend.write_page(page.clone())?;
                    page.clear_dirty();
                }
                drop(page);
            }
            frames.remove(&page_id);
        }
        Ok(())
    }

    /// Prefetch a contiguous range of pages into the buffer pool.
    ///
    /// Use to pre-load the next layer's extent before the current layer completes.
    /// Pages that are already in the pool are not re-read.
    pub fn prefetch_range(&self, first_page_id: PageId, count: usize) -> Result<(), StorageError> {
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed);

        let backend = self
            .backend
            .read()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

        for i in 0..count {
            let page_id = first_page_id + i as u64;
            let shard = self.shard_for(page_id);

            // Skip if already in pool
            {
                let frames = shard
                    .frames
                    .read()
                    .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
                if frames.contains_key(&page_id) {
                    continue;
                }
            }

            // Load from backend
            let mut frames = shard
                .frames
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;

            // Double check
            if frames.contains_key(&page_id) {
                continue;
            }

            if frames.len() >= shard.capacity {
                self.evict_lru(&mut frames)?;
            }

            if let Some(page) = backend.read_page(page_id)? {
                let page_ref = Arc::new(RwLock::new(page));
                frames.insert(
                    page_id,
                    FrameMeta {
                        page: page_ref,
                        last_access: tick,
                        pinned: false,
                    },
                );
            }
        }
        Ok(())
    }

    /// Get the page size from the underlying storage backend
    pub fn page_size(&self) -> usize {
        self.backend
            .read()
            .map(|b| b.page_size())
            .unwrap_or(super::page::DEFAULT_PAGE_SIZE)
    }

    /// **CoW MVCC Phase 2.** Forward to the backend's committed-meta
    /// reader. `Ok(None)` means the backend has no committed_meta yet
    /// (fresh DB or legacy backend without persistence).
    pub fn read_committed_meta(&self) -> Result<Option<CommittedMeta>, StorageError> {
        let backend = self
            .backend
            .read()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        backend.read_committed_meta()
    }

    /// **CoW MVCC Phase 2.** Forward to the backend's atomic
    /// committed-meta writer. The backend's contract: after this
    /// returns, any subsequent `read_committed_meta` observes at least
    /// this record.
    pub fn write_committed_meta(&self, meta: &CommittedMeta) -> Result<(), StorageError> {
        let mut backend = self
            .backend
            .write()
            .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
        backend.write_committed_meta(meta)
    }

    /// **CoW MVCC Phase 4.** Forward to the backend's cross-process
    /// snapshot detector. Returns true if any peer process has a
    /// snapshot live against this database.
    pub fn any_external_snapshots_live(&self) -> bool {
        match self.backend.read() {
            Ok(b) => b.any_external_snapshots_live(),
            // Lock poisoned — be conservative: assume snapshots may
            // be live so the writer defers frees rather than reusing
            // pages a peer might still read.
            Err(_) => true,
        }
    }

    /// Get the current number of pages in the cache
    pub fn cache_size(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Clear all pages from the buffer pool
    pub fn clear(&self) {
        for shard in &self.shards {
            let mut frames = shard.frames.write().unwrap_or_else(|p| p.into_inner());
            frames.clear();
        }
    }

    /// Evict the least-recently-used **evictable** page from a shard.
    ///
    /// A page is evictable only if it is unpinned AND not currently locked. We
    /// probe candidates in LRU order with a non-blocking `try_write`: a page
    /// that is read- or write-locked *right now* is in active use, and blocking
    /// on its lock here would **self-deadlock** the very thread driving the
    /// eviction. The concrete case: a reader holds a leaf node's page read-guard
    /// while reassembling that node's extent-stored value (`Engine::read_extent`
    /// → `get_page` over the extent's data pages). With a small pool, reading
    /// enough data pages makes the leaf page itself the LRU victim; a blocking
    /// `write()` on it then waits forever for a lock the same thread already
    /// holds for read. So we skip any page we can't immediately lock and try the
    /// next-oldest. If nothing is evictable this instant, we proceed without
    /// evicting (the pool transiently exceeds `capacity`) rather than deadlock.
    fn evict_lru(&self, frames: &mut HashMap<PageId, FrameMeta>) -> Result<(), StorageError> {
        // Fast path: the single least-recently-used unpinned page, in ONE O(n)
        // scan with NO allocation and NO sort. This is the steady-state case
        // (the LRU page is not in active use), so eviction stays O(n) per call —
        // a streaming read that evicts ~once per page-fault must not pay
        // O(n log n) (sort the whole shard) each time.
        let lru = frames
            .iter()
            .filter(|(_, meta)| !meta.pinned)
            .min_by_key(|(_, meta)| meta.last_access)
            .map(|(&id, _)| id);
        let Some(lru_id) = lru else { return Ok(()) }; // nothing unpinned to evict
        if self.try_evict_page(frames, lru_id)? {
            return Ok(());
        }

        // Slow path (rare): the LRU page is read/write-locked by an active reader
        // — e.g. a leaf node whose extent value is mid-`read_extent`. Blocking on
        // its lock here would self-deadlock the very thread driving the eviction,
        // so probe the remaining unpinned pages oldest-first and evict the first
        // we can `try_write`-lock.
        let mut candidates: Vec<(PageId, u64)> = frames
            .iter()
            .filter(|&(&id, meta)| !meta.pinned && id != lru_id)
            .map(|(&id, meta)| (id, meta.last_access))
            .collect();
        candidates.sort_unstable_by_key(|&(_, access)| access);
        for (victim_id, _) in candidates {
            if self.try_evict_page(frames, victim_id)? {
                return Ok(());
            }
        }

        // Nothing evictable right now (all pinned or actively locked). Don't
        // panic and don't block — the caller inserts anyway and the pool
        // temporarily holds more than `capacity` frames.
        Ok(())
    }

    /// Evict page `id`: flush it if dirty, then drop it from the frame map.
    /// Returns `false` (skip, don't evict) if the page is currently locked — a
    /// non-blocking `try_write` keeps eviction from ever deadlocking on a page in
    /// active use.
    fn try_evict_page(
        &self,
        frames: &mut HashMap<PageId, FrameMeta>,
        id: PageId,
    ) -> Result<bool, StorageError> {
        let page_arc = match frames.get(&id) {
            Some(meta) => meta.page.clone(),
            None => return Ok(false),
        };
        let mut page = match page_arc.try_write() {
            Ok(guard) => guard,
            Err(_) => return Ok(false),
        };
        if page.is_dirty() {
            let mut backend = self
                .backend
                .write()
                .map_err(|_| StorageError::Backend("Lock poisoned".to_string()))?;
            backend.write_page(page.clone())?;
            page.clear_dirty();
        }
        drop(page);
        frames.remove(&id);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the 2026-04-21 "Empty node data" corruption.
    ///
    /// The fix seeds every newly-allocated page with a valid empty-leaf
    /// body (is_leaf=1, num_keys=0). This test locks that contract in:
    /// any B-tree consumer that walks into a freshly-flushed page
    /// without subsequent content updates must still read a well-formed
    /// node with zero keys, not an empty body that raises a
    /// `Storage::Corrupted` error.
    #[test]
    fn empty_leaf_body_is_valid_empty_leaf() {
        assert_eq!(EMPTY_LEAF_BODY.len(), 5, "header = is_leaf + num_keys");
        assert_eq!(EMPTY_LEAF_BODY[0], 1, "is_leaf = true");
        assert_eq!(
            u32::from_le_bytes([
                EMPTY_LEAF_BODY[1],
                EMPTY_LEAF_BODY[2],
                EMPTY_LEAF_BODY[3],
                EMPTY_LEAF_BODY[4],
            ]),
            0,
            "num_keys = 0"
        );
    }

    /// Regression for the 2026-06-26 extent-read self-deadlock.
    ///
    /// A reader that holds a page's read-guard (e.g. a leaf node mid
    /// `Engine::read_extent`, reassembling that node's extent-stored value)
    /// could make that very page the LRU eviction victim once the read paged in
    /// enough of the extent. The old eviction took a **blocking** `write()` on
    /// the victim, so it waited forever for a lock the same thread already held
    /// for read — a 0%-CPU hang (only with a pool smaller than the value, hence
    /// it hid behind the 256 MB default). Eviction must now skip any page it
    /// cannot immediately lock. This test reproduces the exact shape: hold a
    /// read-guard on the LRU page, then force an eviction.
    #[test]
    fn evict_skips_a_locked_lru_page_instead_of_deadlocking() {
        use crate::storage::memory::MemoryBackend;

        let backend: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(MemoryBackend::new())));
        // capacity 1, single shard → per-shard capacity = 1/1 + 1 = 2 frames.
        let pool = BufferPool::new(
            backend,
            BufferPoolConfig {
                capacity: 1,
                num_shards: 1,
            },
        );

        let a = pool.new_page().expect("alloc A");
        let a_id = a.read().expect("read A id").id;
        let _b = pool.new_page().expect("alloc B"); // shard now holds 2 = capacity

        // Hold a READ guard on A (the oldest, untouched page → the LRU victim).
        let guard_a = a.read().expect("hold read guard on A");

        // Allocating C overflows the shard and triggers eviction. The oldest
        // candidate is A — read-locked right here. Pre-fix this blocks forever;
        // post-fix it skips A, evicts B, and returns.
        let c = pool.new_page().expect("alloc C must not deadlock");
        assert_ne!(c.read().unwrap().id, a_id);

        // A was skipped, not evicted: still the same cached frame.
        drop(guard_a);
        let a_again = pool.get_page(a_id).expect("A still resident");
        assert!(
            Arc::ptr_eq(&a, &a_again),
            "the locked LRU page must survive eviction (skipped, not evicted)"
        );
    }
}
