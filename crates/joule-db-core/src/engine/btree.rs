//! B-tree engine implementation
//!
//! A B-tree based storage engine with:
//! - ACID transactions with rollback support
//! - Large value support
//! - Proper empty value handling
//! - Page-based storage
//! - LRU buffer pool with Arc-based node sharing for reduced cloning
//! - Sharded locking for improved concurrency

use std::collections::HashMap;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use dashmap::DashMap;

use crate::concurrency::{LatchManager, LatchManagerConfig};
use crate::error::{Error, IndexError, StorageError, TransactionError};
use crate::index::{Bound, Index, IndexEntry, IndexIterator, ScanDirection};

use crate::storage::buffer::{BufferPool, BufferPoolConfig};
use crate::storage::{PageId, PageType, StorageBackend};
use crate::tx::{IsolationLevel, Transaction, TxId, TxState};

/// Maximum keys per B-tree node.
/// B-tree performance scales with log_{MAX_KEYS}(N), so higher values
/// reduce tree height. Overflow pages handle large values transparently.
const MAX_KEYS: usize = 256;

/// Minimum keys per node (for rebalancing)
/// Note: Root can have fewer keys than this
const MIN_KEYS: usize = MAX_KEYS / 4;

/// Value markers for distinguishing None from empty
const VALUE_MARKER_NONE: u8 = 0;
const VALUE_MARKER_EMPTY: u8 = 1;
const VALUE_MARKER_DATA: u8 = 2;
/// Overflow marker: value stored in external overflow page chain
const VALUE_MARKER_OVERFLOW: u8 = 3;
/// Extent marker: value stored in contiguous extent pages
const VALUE_MARKER_EXTENT: u8 = 4;

/// Overhead per overflow page: next_page_id (8 bytes)
const OVERFLOW_HEADER_SIZE: usize = 8;

/// Extent header size: page_count (u64) + total_bytes (u64) = 16 bytes
const EXTENT_HEADER_SIZE: usize = 16;

/// Internal result for optimistic put/delete attempts.
/// Signals whether the operation completed or needs a pessimistic retry.
enum OptimisticResult {
    /// Operation completed successfully without structural changes
    Done,
    /// Leaf was full (put) or would underflow (delete) — needs exclusive lock
    NeedsPessimistic,
}

/// Cache entry combining a node with its dirty flag.
/// Stored in DashMap for per-shard concurrent access.
struct CacheEntry {
    node: Arc<BTreeNode>,
    dirty: bool,
}

/// Snapshot of the engine's in-memory uncommitted-write state, taken
/// at the start of a write operation so it can be restored on failure.
/// See [`Engine::abort_uncommitted`].
#[derive(Clone, Copy)]
struct WriteStateMark {
    root: PageId,
    tree_height: u32,
    pending_free_len: usize,
}

/// B-tree node
#[derive(Clone, Debug)]
struct BTreeNode {
    keys: Vec<Vec<u8>>,
    values: Vec<Option<Vec<u8>>>,
    children: Vec<PageId>,
    is_leaf: bool,
    page_id: PageId,
}

impl BTreeNode {
    fn new_leaf(page_id: PageId) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
            page_id,
        }
    }

    fn new_internal(page_id: PageId) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
            page_id,
        }
    }

    /// Find key index using binary search
    fn find_key_index(&self, key: &[u8]) -> usize {
        self.keys
            .binary_search_by(|k| k.as_slice().cmp(key))
            .unwrap_or_else(|i| i)
    }

    fn is_full(&self) -> bool {
        self.keys.len() >= MAX_KEYS
    }

    /// Check if the node is full by byte size — serialized data would exceed page capacity.
    /// Uses 50% fill factor to leave room for the incoming insert before the next split check.
    fn is_full_by_size(&self, max_page_data: usize) -> bool {
        self.keys.len() >= MAX_KEYS || self.serialized_size() > max_page_data / 2
    }

    /// Check if this node has enough room to absorb `other`'s contents
    /// during a B-tree merge without exceeding page capacity.
    ///
    /// Predicts the post-merge serialized size as `self_size + other_size
    /// + separator_overhead - both nodes' fixed prefix overhead`. Uses a
    /// 90% fill factor to leave room for subsequent (small) growth before
    /// the next split — guards against merging into a node so full it
    /// can't accept any inserts without immediately splitting again.
    ///
    /// **Why this exists**: count-based `MIN_KEYS` underflow triggers
    /// merges aggressively when values are large (e.g. 2KB), but
    /// merging two byte-fat half-leaves can produce a node larger
    /// than `max_page_data`. Engines that don't check this end up
    /// writing oversized pages that fail at the storage encode step.
    /// See `docs/joule-db/cow-mvcc-design.md` known limitations.
    fn can_merge_with(
        &self,
        other: &BTreeNode,
        separator: &[u8],
        max_page_data: usize,
    ) -> bool {
        // Each node's serialized_size includes the 5-byte fixed
        // prefix (is_leaf + num_keys). The merged node has only one
        // such prefix, so subtract one of them. For internal nodes,
        // the merged result also gains one separator key (size = 4
        // + separator.len()) and shares the children-count u32.
        let mut merged = self.serialized_size() + other.serialized_size();
        merged = merged.saturating_sub(1 + 4);
        if !self.is_leaf {
            merged += 4 + separator.len();
            merged = merged.saturating_sub(4); // shared children-count u32
        }
        // 90% fill factor — leaves a little headroom for the next
        // insert before another split is triggered.
        merged <= (max_page_data * 9) / 10
    }

    /// Byte-size-based "underflow" criterion: a node is considered
    /// underfull when its serialized representation drops below
    /// 25% of `max_page_data`. Pairs naturally with the byte-size
    /// `is_full_by_size` check (50% triggers split). Decoupled from
    /// the count-based `MIN_KEYS = MAX_KEYS / 4` because, with
    /// large values, count and byte-size diverge sharply.
    fn is_underflowing_by_size(&self, max_page_data: usize) -> bool {
        self.serialized_size() < max_page_data / 4
    }

    /// Check if adding a key-value pair would exceed the page data capacity.
    /// Used by the Engine to decide whether to store a value inline or in overflow pages.
    fn would_exceed_page(
        &self,
        new_key_len: usize,
        new_value_len: usize,
        max_data_size: usize,
    ) -> bool {
        // Estimate new serialized size with the additional entry
        let additional = 4 + new_key_len + 1 + 4 + new_value_len; // key_len + key + marker + val_len + val
        self.serialized_size() + additional > max_data_size
    }

    /// Estimate serialized size for pre-allocation
    fn serialized_size(&self) -> usize {
        let mut size = 1 + 4; // is_leaf + num_keys
        for key in &self.keys {
            size += 4 + key.len();
        }
        if self.is_leaf {
            for value in &self.values {
                size += 1 + match value {
                    Some(v) => 4 + v.len(),
                    None => 0,
                };
            }
        } else {
            size += 4 + self.children.len() * 8;
        }
        size
    }

    /// Serialize node to bytes
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.serialized_size());

        // Format: is_leaf (1) + num_keys (4) + keys + values + children
        buf.push(if self.is_leaf { 1 } else { 0 });
        buf.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());

        // Keys
        for key in &self.keys {
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key);
        }

        // Values (only for leaf nodes)
        if self.is_leaf {
            for value in &self.values {
                match value {
                    None => buf.push(VALUE_MARKER_NONE),
                    Some(v) if v.is_empty() => buf.push(VALUE_MARKER_EMPTY),
                    Some(v) => {
                        buf.push(VALUE_MARKER_DATA);
                        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        buf.extend_from_slice(v);
                    }
                }
            }
        }

        // Children (only for internal nodes)
        if !self.is_leaf {
            buf.extend_from_slice(&(self.children.len() as u32).to_le_bytes());
            for child in &self.children {
                buf.extend_from_slice(&child.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize node from bytes
    fn deserialize(page_id: PageId, data: &[u8]) -> Result<Self, StorageError> {
        if data.is_empty() {
            return Err(StorageError::Corrupted {
                page_id,
                reason: "Empty node data".to_string(),
            });
        }

        let mut cursor = 0;

        // is_leaf
        let is_leaf = data[cursor] == 1;
        cursor += 1;

        // num_keys
        if cursor + 4 > data.len() {
            return Err(StorageError::Corrupted {
                page_id,
                reason: "Missing key count".to_string(),
            });
        }
        let num_keys = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Keys
        let mut keys = Vec::with_capacity(num_keys);
        for _ in 0..num_keys {
            if cursor + 4 > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Missing key length".to_string(),
                });
            }
            let key_len = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;
            cursor += 4;

            if cursor + key_len > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Key data truncated".to_string(),
                });
            }
            keys.push(data[cursor..cursor + key_len].to_vec());
            cursor += key_len;
        }

        // Values (leaf only)
        let mut values = Vec::with_capacity(if is_leaf { num_keys } else { 0 });
        if is_leaf {
            for _ in 0..num_keys {
                if cursor >= data.len() {
                    return Err(StorageError::Corrupted {
                        page_id,
                        reason: "Missing value marker".to_string(),
                    });
                }
                let marker = data[cursor];
                cursor += 1;

                match marker {
                    VALUE_MARKER_NONE => values.push(None),
                    VALUE_MARKER_EMPTY => values.push(Some(Vec::new())),
                    VALUE_MARKER_DATA => {
                        if cursor + 4 > data.len() {
                            return Err(StorageError::Corrupted {
                                page_id,
                                reason: "Missing value length".to_string(),
                            });
                        }
                        let val_len = u32::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                        ]) as usize;
                        cursor += 4;

                        if cursor + val_len > data.len() {
                            return Err(StorageError::Corrupted {
                                page_id,
                                reason: "Value data truncated".to_string(),
                            });
                        }
                        values.push(Some(data[cursor..cursor + val_len].to_vec()));
                        cursor += val_len;
                    }
                    _ => {
                        return Err(StorageError::Corrupted {
                            page_id,
                            reason: format!("Invalid value marker: {}", marker),
                        });
                    }
                }
            }
        }

        // Children (internal only)
        let mut children = Vec::new();
        if !is_leaf {
            if cursor + 4 > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Missing children count".to_string(),
                });
            }
            let num_children = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;
            cursor += 4;
            children.reserve(num_children);

            for _ in 0..num_children {
                if cursor + 8 > data.len() {
                    return Err(StorageError::Corrupted {
                        page_id,
                        reason: "Child pointer truncated".to_string(),
                    });
                }
                let child_id = u64::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                    data[cursor + 4],
                    data[cursor + 5],
                    data[cursor + 6],
                    data[cursor + 7],
                ]);
                cursor += 8;
                children.push(child_id);
            }
        }

        Ok(Self {
            keys,
            values,
            children,
            is_leaf,
            page_id,
        })
    }

    /// Split node, returns (new_node, median_key)
    fn split(&mut self, new_page_id: PageId) -> (Self, Vec<u8>) {
        let mid = self.keys.len() / 2;
        let median_key = self.keys[mid].clone();

        let mut right = if self.is_leaf {
            Self::new_leaf(new_page_id)
        } else {
            Self::new_internal(new_page_id)
        };

        if self.is_leaf {
            right.keys = self.keys.split_off(mid);
            right.values = self.values.split_off(mid);
        } else {
            right.keys = self.keys.split_off(mid + 1);
            right.children = self.children.split_off(mid + 1);
            self.keys.pop(); // Remove median from left
        }

        (right, median_key)
    }
}

/// Modification record for transaction rollback
#[derive(Clone, Debug)]
struct ModificationRecord {
    key: Vec<u8>,
    old_value: Option<Vec<u8>>,
}

impl ModificationRecord {
    /// Get the key that was modified
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get the old value before modification (None if key didn't exist)
    pub fn old_value(&self) -> Option<&[u8]> {
        self.old_value.as_deref()
    }
}

/// Result of a search operation (used to avoid holding latches across recursion)
enum SearchResult {
    Found(Option<Vec<u8>>),
    NotFound,
    Continue(PageId),
}

/// B-tree engine transaction
pub struct EngineTransaction<'a> {
    engine: &'a Engine,
    id: TxId,
    isolation: IsolationLevel,
    state: TxState,
    write_set: HashMap<Vec<u8>, Option<Vec<u8>>>,
    modifications: Vec<ModificationRecord>,
    /// Arena allocator for transaction-scoped allocations (optional)
    arena: Option<crate::allocator::TransactionArena>,
}

impl<'a> Transaction for EngineTransaction<'a> {
    fn id(&self) -> TxId {
        self.id
    }

    fn isolation_level(&self) -> IsolationLevel {
        self.isolation
    }

    fn state(&self) -> TxState {
        self.state
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        // Check write set first
        if let Some(value) = self.write_set.get(key) {
            return Ok(value.clone());
        }

        // Read from engine
        self.engine
            .get_internal(key)
            .map_err(|e| TransactionError::SerializationFailure {
                reason: e.to_string(),
            })
    }

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Record old value for rollback
        let old_value = self.get(key)?;
        self.modifications.push(ModificationRecord {
            key: key.to_vec(),
            old_value,
        });

        // Add to write set
        self.write_set.insert(key.to_vec(), Some(value.to_vec()));
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        let old_value = self.get(key)?;
        let existed = old_value.is_some();

        // Record for rollback
        self.modifications.push(ModificationRecord {
            key: key.to_vec(),
            old_value,
        });

        // Mark as deleted
        self.write_set.insert(key.to_vec(), None);
        Ok(existed)
    }

    fn commit(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Acquire GLOBAL WRITE LOCK for atomic commit
        let _guard =
            self.engine
                .write_lock
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "Failed to acquire lock".into(),
                })?;

        // Apply all writes to engine. CoW rollback rail — see
        // `Engine::abort_uncommitted`: if any entry's apply fails
        // partway, the half-applied CoW cascade (and its deferred
        // frees of pages still live in the committed tree) must be
        // discarded, or a later drain reuses a live page id.
        let mark = self.engine.write_state_mark();
        for (key, value) in &self.write_set {
            let res = match value {
                Some(v) => self.engine.put_locked(key, v),
                None => self.engine.delete_locked(key).map(|_| ()),
            };
            if let Err(e) = res {
                self.engine.abort_uncommitted(mark);
                self.state = TxState::Aborted;
                return Err(TransactionError::SerializationFailure {
                    reason: e.to_string(),
                });
            }
        }

        self.engine.clear_pending_allocations();
        self.state = TxState::Committed;
        Ok(())
    }

    fn rollback(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Discard write set - nothing was applied to storage
        self.write_set.clear();
        self.modifications.clear();

        // Reset arena (zero-cost cleanup)
        if let Some(arena) = &mut self.arena {
            arena.reset();
        }

        self.state = TxState::Aborted;
        Ok(())
    }
}

/// B-tree storage engine
///
/// Uses a sharded buffer pool for caching nodes to reduce lock contention
/// and Arc-based sharing to minimize cloning overhead.
///
/// ## Concurrency Model
///
/// The engine uses a multi-level locking strategy:
/// - **Global storage lock**: For page allocations and sync operations
/// - **Write serialization lock**: Serializes all write operations to ensure B-tree consistency
/// - **Per-page latches**: For individual page reads via `LatchManager`
///
/// Reads can proceed concurrently, but writes are serialized to maintain
/// Transaction handle for exclusive write access
pub struct WriteTransaction<'a> {
    engine: &'a Engine,
}

impl<'a> WriteTransaction<'a> {
    /// Put a value (atomic)
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        self.engine.put_locked(key, value)
    }

    /// Delete a value (atomic)
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, Error> {
        self.engine.delete_locked(key)
    }

    /// Get a value
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        // Reads are safe under write lock (exclusive)
        self.engine.get(key)
    }
}

/// B-tree invariants during splits and merges.
pub struct Engine {
    // Storage is managed by buffer_pool
    root_page_id: RwLock<PageId>,
    /// **CoW MVCC Phase 2.** Monotonic counter for committed root
    /// updates. Incremented atomically by `update_metadata` whenever
    /// the on-disk meta record is rewritten. Snapshot readers
    /// (Phase 3) capture this value at open and refresh against it.
    /// See `docs/joule-db/cow-mvcc-design.md`.
    committed_version: AtomicU64,
    /// **CoW MVCC Phase 5.** Last-synced root page id — distinct from
    /// `root_page_id` which is the *in-memory* working root that
    /// rotates on every put. Snapshot readers must observe a state
    /// that is *atomically committed*; reading the in-memory root
    /// during a multi-put transaction (between the puts but before
    /// the sync) would return a torn view. This field is published
    /// atomically together with `committed_version` at sync time.
    committed_root: AtomicU64,
    /// **CoW MVCC Phase 3.** Live snapshot count. While > 0, calls to
    /// `defer_free_page` queue freed page ids in `pending_free_pages`
    /// instead of releasing them immediately. Pages are released to
    /// the buffer pool only after the last snapshot drops, ensuring
    /// snapshot reads can never observe a freed-then-reused page.
    /// Phase 4 will replace this with a version-gated free list keyed
    /// by min-live-snapshot-version.
    live_snapshots: std::sync::atomic::AtomicUsize,
    /// Pages whose immediate `free_page` was deferred because a
    /// snapshot was active when the writer freed them. Drained on the
    /// last snapshot drop. Phase 3 stop-gap; Phase 4 replaces with
    /// `Vec<(PageId, version_freed)>` and version-gated reclamation.
    pending_free_pages: std::sync::Mutex<Vec<PageId>>,
    /// Pages allocated by the *currently in-flight* write operation
    /// (everything `save_node_cow` and `write_overflow_chain` got from
    /// `buffer_pool.new_page()` since the write_lock was last
    /// acquired). The write path is serialised on `write_lock`, so
    /// this is effectively per-transaction; cleared on every successful
    /// commit (those pages are now reachable from the new root) and
    /// freed on abort.
    ///
    /// **Why this exists.** Under CoW every put allocates new pages
    /// (leaf + ancestor chain + maybe overflow). When a write
    /// transaction fails partway and `abort_uncommitted` reverts
    /// `root_page_id`, those new pages become unreachable. Without
    /// tracking them, they sat forever in `data.wdb`'s allocated
    /// extent — auto-committed at the FileBackend level by
    /// buffer-pool evictions (so the bytes were on disk), with no
    /// later free. In production 2026-05-12 the scholar binary-split
    /// fired ~10× and `data.wdb` grew 220 GB in 5 h until the volume
    /// hit ENOSPC. With this queue, `abort_uncommitted` walks it and
    /// calls `buffer_pool.free_page` on each — the page ids return
    /// to the backend's free list and the next `new_page()` reuses
    /// them instead of extending the file.
    pending_allocations: std::sync::Mutex<Vec<PageId>>,
    next_tx_id: AtomicU64,
    /// Buffer pool for page persistence
    buffer_pool: BufferPool,
    /// Per-page latch manager for fine-grained read concurrency
    page_latches: LatchManager,
    /// Serializes write operations (put, delete) to maintain B-tree consistency
    write_lock: RwLock<()>,
    /// Concurrent cache of deserialized B-tree nodes with dirty tracking.
    /// Uses DashMap for per-shard locking — eliminates global RwLock contention.
    node_cache: DashMap<PageId, CacheEntry>,
    /// Number of internal levels in the tree (0 = root is leaf).
    /// Used by find_leaf to avoid loading leaf nodes during traversal,
    /// which prevents a race with take_node_for_write in the optimistic path.
    tree_height: AtomicU32,
    /// Monotonically increasing counter for structural changes (splits/merges).
    /// Readers snapshot this before latch-free traversal and verify after.
    structure_version: AtomicU64,
    /// Maximum value size that can be stored inline in a B-tree node.
    /// Values larger than this are stored in overflow page chains.
    /// Computed as (page_size - PAGE_HEADER_SIZE) / 4 to ensure nodes fit in pages.
    max_inline_value_size: usize,
    /// Maximum data that can fit in a single page (page_size - PAGE_HEADER_SIZE).
    max_page_data: usize,
}

/// Engine configuration combining buffer pool and latch settings
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Buffer pool configuration
    pub buffer_pool: BufferPoolConfig,
    /// Latch manager configuration
    pub latch_manager: LatchManagerConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            buffer_pool: BufferPoolConfig::default(),
            latch_manager: LatchManagerConfig::default(),
        }
    }
}

/// Metadata page ID (always page 1)
const METADATA_PAGE_ID: PageId = 1;

/// Metadata page magic bytes
const METADATA_MAGIC: [u8; 4] = [0x57, 0x44, 0x42, 0x4D]; // "WDBM"

/// Result of `Engine::prefix_count`. `count` is the number of
/// reachable keys matching the requested prefix; `errors` lists
/// `(page_id, reason)` pairs for any subtree that failed to walk
/// — typically a corrupt internal page chain. The count is
/// partial when `errors` is non-empty.
#[derive(Debug, Clone, Default)]
pub struct PrefixCount {
    /// Reachable keys matching the prefix. Excludes tombstones.
    pub count: u64,
    /// Page ids of subtree roots that errored during the walk.
    /// Each entry pairs the failed page id with a reason.
    pub errors: Vec<(PageId, String)>,
}

/// Helper for `Engine::prefix_count` subtree pruning. Given an
/// internal node's child range — exclusive lower bound (`lower`,
/// `None` if child is leftmost) and exclusive upper bound (`upper`,
/// `None` if child is rightmost) — return whether any key with the
/// requested prefix could appear in the child's subtree.
///
/// A key K starts with prefix P iff `prefix <= K < prefix_succ`,
/// where `prefix_succ` is the smallest byte string strictly greater
/// than every string with prefix P (computed by incrementing the
/// last non-0xFF byte; if every byte is 0xFF then no upper bound).
///
/// Child range covers keys in `[lower, upper)` (`lower` exclusive
/// for non-leftmost, but treated inclusive on the left edge of an
/// internal node; for our purposes inclusive vs exclusive at the
/// boundary is irrelevant — overlap is the same).
fn child_range_overlaps_prefix(
    lower: Option<&Vec<u8>>,
    upper: Option<&Vec<u8>>,
    prefix: &[u8],
) -> bool {
    // No prefix means "all keys" — every child overlaps.
    if prefix.is_empty() {
        return true;
    }
    // If child's upper bound <= prefix, no key in this child can
    // start with prefix (every key is strictly less than `prefix`).
    if let Some(upper_key) = upper {
        if upper_key.as_slice() <= prefix {
            return false;
        }
    }
    // If child's lower bound is past the prefix range entirely,
    // skip. Compute the prefix's exclusive upper bound: strip
    // trailing 0xFFs and increment the last byte.
    if let Some(lower_key) = lower {
        let mut prefix_succ = prefix.to_vec();
        while let Some(&0xFF) = prefix_succ.last() {
            prefix_succ.pop();
        }
        if let Some(last) = prefix_succ.last_mut() {
            *last += 1;
            if lower_key.as_slice() >= &prefix_succ[..] {
                return false;
            }
        }
        // else: prefix is all-0xFF — no upper bound, so any lower
        // bound is fine.
    }
    true
}

impl Engine {
    /// Create a new engine with the given storage backend
    ///
    /// This will create a fresh B-tree. For opening an existing database,
    /// use `Engine::open`.
    pub fn new(storage: impl StorageBackend + 'static) -> Result<Self, Error> {
        Self::create_with_config(storage, EngineConfig::default())
    }

    /// Create a new engine with custom buffer pool configuration (legacy API)
    pub fn with_buffer_config(
        storage: impl StorageBackend + 'static,
        cache_config: BufferPoolConfig,
    ) -> Result<Self, Error> {
        Self::create_with_config(
            storage,
            EngineConfig {
                buffer_pool: cache_config,
                latch_manager: LatchManagerConfig::default(),
            },
        )
    }

    /// Create a new engine with custom configuration
    pub fn with_config(
        storage: impl StorageBackend + 'static,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        Self::create_with_config(storage, config)
    }

    /// Create a new database (initializes fresh B-tree)
    pub fn create_with_config(
        storage: impl StorageBackend + 'static,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        // Initialize backend and pool first
        let backend: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(storage)));
        let buffer_pool = BufferPool::new(backend, config.buffer_pool);

        // Allocate metadata page first (page 1)
        let meta_page_arc = buffer_pool.new_page()?;
        let meta_page_id = meta_page_arc
            .read()
            .expect("lock poisoned: meta page read")
            .id;
        debug_assert_eq!(meta_page_id, METADATA_PAGE_ID);

        // Allocate root page (page 2+)
        let root_page_arc = buffer_pool.new_page()?;
        let root_id = root_page_arc
            .read()
            .expect("lock poisoned: root page read")
            .id;
        let root = BTreeNode::new_leaf(root_id);

        // Write root node
        {
            let mut page = root_page_arc
                .write()
                .expect("lock poisoned: root page write");
            page.page_type = PageType::BTreeLeaf;
            page.data = root.serialize();
            page.mark_dirty();
        }

        // Write metadata page
        {
            let mut meta_page = meta_page_arc
                .write()
                .expect("lock poisoned: meta page write");
            meta_page.page_type = PageType::BTreeInternal;
            meta_page.data = Self::encode_metadata(root_id);
            meta_page.mark_dirty();
        }

        let page_latches = LatchManager::with_config(config.latch_manager);

        let ps = buffer_pool.page_size();
        let max_page_data = ps - crate::storage::page::PAGE_HEADER_SIZE;
        let max_inline_value_size = max_page_data / 4;

        Ok(Self {
            root_page_id: RwLock::new(root_id),
            committed_version: AtomicU64::new(0),
            committed_root: AtomicU64::new(root_id),
            live_snapshots: std::sync::atomic::AtomicUsize::new(0),
            pending_free_pages: std::sync::Mutex::new(Vec::new()),
            pending_allocations: std::sync::Mutex::new(Vec::new()),
            next_tx_id: AtomicU64::new(1),
            buffer_pool,
            page_latches,
            write_lock: RwLock::new(()),
            node_cache: DashMap::new(),
            tree_height: AtomicU32::new(0), // root is a leaf
            structure_version: AtomicU64::new(0),
            max_inline_value_size,
            max_page_data,
        })
    }

    /// Execute a closure under a write lock (Exclusive Tx)
    pub fn write_transaction<F, T>(&self, f: F) -> Result<T, Error>
    where
        F: FnOnce(&mut WriteTransaction<'_>) -> Result<T, Error>,
    {
        let _guard = self.write_lock.write().map_err(|_| {
            Error::Storage(StorageError::Backend("Failed to acquire write lock".into()))
        })?;

        // CoW rollback rail: if the closure mutates the tree and then
        // returns Err (e.g. a corruption guard fires mid-cascade), the
        // half-applied CoW state — rotated `root_page_id`, deferred
        // frees of pages still live in the committed tree — must be
        // discarded, or a later `try_drain_pending_frees` reuses a
        // live page id and structurally corrupts the DB. See
        // `abort_uncommitted`.
        let mark = self.write_state_mark();
        let mut tx = WriteTransaction { engine: self };
        match f(&mut tx) {
            Ok(v) => {
                // Allocations made during this tx are now reachable
                // from the new `root_page_id` — they must NOT be freed.
                self.clear_pending_allocations();
                Ok(v)
            }
            Err(e) => {
                self.abort_uncommitted(mark);
                Err(e)
            }
        }
    }

    /// Open an existing database from storage
    ///
    /// This reads the metadata page to find the root and loads the existing B-tree.
    /// Returns an error if the database doesn't exist or is corrupted.
    pub fn open(storage: impl StorageBackend + 'static) -> Result<Self, Error> {
        Self::open_with_config(storage, EngineConfig::default())
    }

    /// Open an existing database with custom configuration
    /// Open an existing database with custom configuration
    pub fn open_with_config(
        storage: impl StorageBackend + 'static,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        let backend: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(storage)));
        let buffer_pool = BufferPool::new(backend, config.buffer_pool);

        // **CoW MVCC Phase 2**: prefer the backend's atomically-
        // committed `(root, version)` over the legacy page-1 record.
        // The backend's record is what FileBackend re-renamed
        // atomically at the last successful sync, so it cannot be
        // torn. Fall back to page 1 when the backend has no record
        // (legacy v0 database, or a backend that doesn't persist meta).
        let backend_meta = buffer_pool
            .read_committed_meta()
            .map_err(Error::Storage)?;
        let (root_id, committed_version) = match backend_meta {
            Some(meta) => (meta.committed_root, meta.committed_version),
            None => {
                let meta_page_arc = buffer_pool.get_page(METADATA_PAGE_ID)?;
                let meta_page =
                    meta_page_arc.read().expect("lock poisoned: meta page read");
                (Self::decode_metadata(&meta_page.data)?, 0)
            }
        };

        // Verify root page exists
        buffer_pool.get_page(root_id)?; // Just try to load it

        let page_latches = LatchManager::with_config(config.latch_manager);

        let ps = buffer_pool.page_size();
        let max_page_data = ps - crate::storage::page::PAGE_HEADER_SIZE;
        let max_inline_value_size = max_page_data / 4;

        let engine = Self {
            root_page_id: RwLock::new(root_id),
            committed_version: AtomicU64::new(committed_version),
            committed_root: AtomicU64::new(root_id),
            live_snapshots: std::sync::atomic::AtomicUsize::new(0),
            pending_free_pages: std::sync::Mutex::new(Vec::new()),
            pending_allocations: std::sync::Mutex::new(Vec::new()),
            next_tx_id: AtomicU64::new(1),
            buffer_pool,
            page_latches,
            write_lock: RwLock::new(()),
            node_cache: DashMap::new(),
            tree_height: AtomicU32::new(0),
            structure_version: AtomicU64::new(0),
            max_inline_value_size,
            max_page_data,
        };
        let height = engine.compute_tree_height()?;
        engine.tree_height.store(height, Ordering::Release);
        Ok(engine)
    }

    /// Open or create a database
    ///
    /// Opens an existing database if metadata exists, otherwise creates a new one.
    pub fn open_or_create(storage: impl StorageBackend + 'static) -> Result<Self, Error> {
        Self::open_or_create_with_config(storage, EngineConfig::default())
    }

    /// Open or create a database with custom configuration
    pub fn open_or_create_with_config(
        storage: impl StorageBackend + 'static,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        let backend_arc: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(storage)));
        let buffer_pool = BufferPool::new(backend_arc.clone(), config.buffer_pool.clone());

        // Try reading metadata directly via pool
        let result = buffer_pool.get_page(METADATA_PAGE_ID);

        match result {
            Ok(meta_page_arc) => {
                // Database exists
                let meta_page = meta_page_arc.read().expect("lock poisoned: meta page read");
                if let Ok(root_id_legacy) = Self::decode_metadata(&meta_page.data) {
                    drop(meta_page);

                    // CoW MVCC Phase 2: prefer backend committed_meta
                    // over legacy page-1 root, same as `open_with_config`.
                    let backend_meta = buffer_pool
                        .read_committed_meta()
                        .map_err(Error::Storage)?;
                    let (root_id, committed_version) = match backend_meta {
                        Some(meta) => (meta.committed_root, meta.committed_version),
                        None => (root_id_legacy, 0),
                    };

                    buffer_pool.get_page(root_id)?;

                    let page_latches = LatchManager::with_config(config.latch_manager);
                    let ps = buffer_pool.page_size();
                    let max_page_data = ps - crate::storage::page::PAGE_HEADER_SIZE;
                    let max_inline_value_size = max_page_data / 4;
                    let engine = Self {
                        root_page_id: RwLock::new(root_id),
                        committed_version: AtomicU64::new(committed_version),
                        committed_root: AtomicU64::new(root_id),
                        live_snapshots: std::sync::atomic::AtomicUsize::new(0),
                        pending_free_pages: std::sync::Mutex::new(Vec::new()),
            pending_allocations: std::sync::Mutex::new(Vec::new()),
                        next_tx_id: AtomicU64::new(1),
                        buffer_pool,
                        page_latches,
                        write_lock: RwLock::new(()),
                        node_cache: DashMap::new(),
                        tree_height: AtomicU32::new(0),
                        structure_version: AtomicU64::new(0),
                        max_inline_value_size,
                        max_page_data,
                    };
                    let height = engine.compute_tree_height()?;
                    engine.tree_height.store(height, Ordering::Release);
                    Ok(engine)
                } else {
                    // Metadata page exists but doesn't have Engine's WDBM format.
                    // This happens when DiskBackend::init_metadata wrote its own
                    // free-list format before Engine had a chance to initialize.
                    // Fall through to the create path — overwrite with proper format.
                    drop(meta_page);
                    Self::init_fresh_db(buffer_pool, config)
                }
            }
            Err(StorageError::PageNotFound { .. }) | Err(StorageError::Backend(_)) => {
                // No existing database — create fresh.
                Self::init_fresh_db(buffer_pool, config)
            }
            Err(e) => Err(Error::Storage(e)),
        }
    }

    /// Initialize a fresh database with proper Engine metadata format (WDBM magic).
    /// Used by both the create-new and the "DiskBackend wrote incompatible metadata" paths.
    fn init_fresh_db(buffer_pool: BufferPool, config: EngineConfig) -> Result<Self, Error> {
        // The metadata page MUST land at METADATA_PAGE_ID = 1 so that
        // `open()` can find it again. DiskBackend pre-allocates page 1 as
        // its own metadata stub; here we reuse that slot, overwriting it
        // with the engine's WDBM-format metadata. Allocating via
        // `new_page()` would give us page 2 instead — which is why the
        // original code silently lost root pointers across process restarts.
        let meta_page_arc = buffer_pool.get_page(METADATA_PAGE_ID).or_else(|_| {
            // If page 1 isn't present (backends that don't pre-allocate it),
            // allocate fresh. The first `new_page()` is guaranteed to return
            // id 1 in that case.
            buffer_pool.new_page()
        })?;

        let root_page_arc = buffer_pool.new_page()?;
        let root_id = root_page_arc
            .read()
            .expect("lock poisoned: root page read")
            .id;
        let root = BTreeNode::new_leaf(root_id);

        {
            let mut page = root_page_arc
                .write()
                .expect("lock poisoned: root page write");
            page.page_type = PageType::BTreeLeaf;
            page.data = root.serialize();
            page.mark_dirty();
        }

        {
            let mut meta_page = meta_page_arc
                .write()
                .expect("lock poisoned: meta page write");
            meta_page.page_type = PageType::BTreeInternal;
            meta_page.data = Self::encode_metadata(root_id);
            meta_page.mark_dirty();
        }

        let page_latches = LatchManager::with_config(config.latch_manager);

        let ps = buffer_pool.page_size();
        let max_page_data = ps - crate::storage::page::PAGE_HEADER_SIZE;
        let max_inline_value_size = max_page_data / 4;

        Ok(Self {
            root_page_id: RwLock::new(root_id),
            committed_version: AtomicU64::new(0),
            committed_root: AtomicU64::new(root_id),
            live_snapshots: std::sync::atomic::AtomicUsize::new(0),
            pending_free_pages: std::sync::Mutex::new(Vec::new()),
            pending_allocations: std::sync::Mutex::new(Vec::new()),
            next_tx_id: AtomicU64::new(1),
            buffer_pool,
            page_latches,
            write_lock: RwLock::new(()),
            node_cache: DashMap::new(),
            tree_height: AtomicU32::new(0),
            structure_version: AtomicU64::new(0),
            max_inline_value_size,
            max_page_data,
        })
    }

    /// Encode metadata to bytes
    fn encode_metadata(root_page_id: PageId) -> Vec<u8> {
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&METADATA_MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes()); // Version
        data.extend_from_slice(&root_page_id.to_le_bytes());
        data
    }

    /// Decode metadata from bytes
    fn decode_metadata(data: &[u8]) -> Result<PageId, Error> {
        if data.len() < 16 {
            return Err(Error::Storage(StorageError::Backend(
                "Metadata page too short".to_string(),
            )));
        }

        if &data[0..4] != &METADATA_MAGIC {
            return Err(Error::Storage(StorageError::Backend(
                "Invalid metadata magic".to_string(),
            )));
        }

        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let root_page_id = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);

        Ok(root_page_id)
    }

    /// Update metadata (call after root changes, e.g., split).
    ///
    /// **Cheap operation**: only refreshes the in-memory legacy page-1
    /// cache record. The atomic `(root, version)` commit to the
    /// backend's `meta.wdb` happens at `Engine::sync()` — the natural
    /// MVCC commit boundary. `committed_version` increments per sync,
    /// not per put, so a snapshot reader sees changes batched at sync
    /// granularity.
    ///
    /// See `docs/joule-db/cow-mvcc-design.md`.
    fn update_metadata(&self) -> Result<(), Error> {
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");

        // Legacy page-1 cache update. Still maintained because
        // Engine::open may fall back to reading page 1 if the backend
        // hasn't yet stored a committed_meta (v0 databases on first
        // open). The actual commit-on-disk happens at sync.
        let metadata = Self::encode_metadata(root_id);
        let page_arc = self.buffer_pool.get_page(METADATA_PAGE_ID)?;
        let mut page = page_arc
            .write()
            .expect("lock poisoned: metadata page write");
        page.data = metadata;
        page.mark_dirty();

        Ok(())
    }

    /// Snapshot of the in-memory uncommitted-write state taken at the
    /// start of a write transaction (or autocommit `put`/`delete`), so
    /// it can be restored if the operation fails partway. See
    /// [`Engine::abort_uncommitted`].
    fn write_state_mark(&self) -> WriteStateMark {
        WriteStateMark {
            root: *self
                .root_page_id
                .read()
                .expect("lock poisoned: root_page_id read (mark)"),
            tree_height: self.tree_height.load(Ordering::Acquire),
            pending_free_len: self.pending_free_count(),
        }
    }

    /// Roll back the engine's in-memory uncommitted CoW state to a
    /// pre-operation [`WriteStateMark`]. Call this whenever a
    /// `write_transaction` closure — or an autocommit `put`/`delete`,
    /// or `EngineTransaction::commit`'s apply loop — returns `Err`
    /// after having already mutated the tree.
    ///
    /// **Why this is mandatory.** Under copy-on-write every `put`
    /// rotates `root_page_id` to a fresh CoW root and may push the
    /// retired pages into `pending_free_pages` (deferred, because a
    /// reader snapshot — e.g. scholar-server — is live). If the
    /// operation fails partway:
    ///  - `root_page_id` now points at a CoW root that will never be
    ///    committed, and subsequent reads/writes from it are
    ///    operating on a half-built tree;
    ///  - the entries queued into `pending_free_pages` are for pages
    ///    still reachable from the *committed* root. A later
    ///    `try_drain_pending_frees` would release those live pages
    ///    for reuse, and the next allocation would overwrite a page
    ///    the committed tree still points at — silent structural
    ///    corruption (this is the class of bug behind the scholar
    ///    "cycle in internal child pointers" incidents).
    ///
    /// So the abort: (1) reverts `root_page_id` and `tree_height` to
    /// the mark, (2) truncates `pending_free_pages` back to its
    /// pre-op length — discarding this op's deferred frees, which
    /// means the *new* CoW pages it allocated leak as unreferenced
    /// dead space. Leaking dead pages is the safe direction; reusing
    /// a still-live page id is not. `next_page_id` deliberately does
    /// **not** regress. The orphaned CoW pages sitting dirty in the
    /// buffer pool get written out at the next `sync()` as dead
    /// space (the same way CoW already leaves retired pages on disk
    /// while any reader snapshot is live) — wasteful, not corrupting.
    /// The legacy page-1 metadata mirror is re-synced to the reverted
    /// root on a best-effort basis.
    fn abort_uncommitted(&self, mark: WriteStateMark) {
        *self
            .root_page_id
            .write()
            .expect("lock poisoned: root_page_id write (abort)") = mark.root;
        self.tree_height.store(mark.tree_height, Ordering::Release);
        {
            let mut pf = self
                .pending_free_pages
                .lock()
                .expect("lock poisoned: pending_free_pages (abort)");
            if pf.len() > mark.pending_free_len {
                pf.truncate(mark.pending_free_len);
            }
        }
        // Free the page ids the failed write allocated since this
        // transaction started — they were just `new_page`'d (via
        // `save_node_cow` / `write_overflow_chain`) but are now
        // unreachable from the reverted `root_page_id`. Without this,
        // every binary-split rollback leaks the allocated pages and
        // `data.wdb` grows monotonically until ENOSPC (prod 2026-05-12
        // observation: 220 GB in 5 h). `buffer_pool.free_page` returns
        // the id to the backend's free list — the next `new_page()`
        // reuses it instead of extending the file.
        let drained: Vec<PageId> = {
            let mut pa = self
                .pending_allocations
                .lock()
                .expect("lock poisoned: pending_allocations (abort)");
            std::mem::take(&mut *pa)
        };
        for pid in drained {
            // Best-effort: a single backend failure shouldn't abort the
            // others. The drop is harmless if it fails — the page id
            // just becomes a permanently-leaked orphan.
            let _ = self.buffer_pool.free_page(pid);
        }
        let _ = self.update_metadata();
    }

    /// Get the root page ID (for debugging/recovery)
    pub fn root_page_id(&self) -> PageId {
        *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read")
    }

    /// Begin a new transaction
    pub fn begin(&self) -> EngineTransaction<'_> {
        self.begin_with_arena(None)
    }

    /// Begin a transaction with specific isolation level
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> EngineTransaction<'_> {
        self.begin_with_isolation_and_arena(isolation, None)
    }

    /// Begin a transaction with arena allocator
    pub fn begin_with_arena(
        &self,
        arena: Option<crate::allocator::TransactionArena>,
    ) -> EngineTransaction<'_> {
        let id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        EngineTransaction {
            engine: self,
            id,
            isolation: IsolationLevel::default(),
            state: TxState::Active,
            write_set: HashMap::new(),
            modifications: Vec::new(),
            arena,
        }
    }

    /// Begin a transaction with isolation level and arena allocator
    pub fn begin_with_isolation_and_arena(
        &self,
        isolation: IsolationLevel,
        arena: Option<crate::allocator::TransactionArena>,
    ) -> EngineTransaction<'_> {
        let id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        EngineTransaction {
            engine: self,
            id,
            isolation,
            state: TxState::Active,
            write_set: HashMap::new(),
            modifications: Vec::new(),
            arena,
        }
    }

    /// Simple get (non-transactional)
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.get_internal(key)
    }

    /// Simple put (non-transactional)
    ///
    /// Uses optimistic latch crabbing: acquires write_lock.read() (shared barrier)
    /// and only the leaf's page latch. Multiple concurrent writers to different
    /// leaves proceed in parallel. Falls back to exclusive write_lock.write()
    /// only when a split is needed.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        // **CoW MVCC (Phase 1):** the leaf-only optimistic path is
        // disabled. Under copy-on-write semantics, modifying a leaf
        // produces a fresh page id, so the parent's `children[idx]`
        // pointer must be rewritten — cascading all the way up to the
        // root. There is no "touch only the leaf" path anymore.
        //
        // Single-writer is the new concurrency model (matches LMDB).
        // The page-latch parallelism that justified the optimistic
        // path is moot once writes serialize on `write_lock.write()`.
        // Concurrent *readers* gain through Phase 3's snapshot API,
        // not through writer parallelism.
        //
        // See `docs/joule-db/cow-mvcc-design.md` §4.
        let _guard = self.write_lock.write().map_err(|_| {
            Error::Storage(StorageError::Backend("Failed to acquire write lock".into()))
        })?;
        let mark = self.write_state_mark();
        let r = self.put_locked(key, value);
        if r.is_err() {
            self.abort_uncommitted(mark);
        } else {
            self.clear_pending_allocations();
        }
        r
    }

    /// Simple delete (non-transactional).
    ///
    /// **CoW MVCC (Phase 1):** the leaf-only optimistic path is
    /// disabled for the same reason `put`'s is — every leaf write
    /// allocates a fresh page id and must cascade to the root. See
    /// `pub fn put` for the full rationale and `docs/joule-db/cow-mvcc-design.md` §4.
    pub fn delete(&self, key: &[u8]) -> Result<bool, Error> {
        let _guard = self.write_lock.write().map_err(|_| {
            Error::Storage(StorageError::Backend("Failed to acquire write lock".into()))
        })?;
        let mark = self.write_state_mark();
        let r = self.delete_locked(key);
        if r.is_err() {
            self.abort_uncommitted(mark);
        } else {
            self.clear_pending_allocations();
        }
        r
    }

    /// Sync to durable storage.
    ///
    /// **CoW MVCC Phase 2 — the commit boundary.** This is the only
    /// place where a snapshot reader's view advances. Sequence:
    ///
    /// 1. Flush dirty nodes from the engine's node cache into the
    ///    buffer pool's pages.
    /// 2. Flush dirty pages from the buffer pool to the backend (and
    ///    the backend's data file, in turn).
    /// 3. Atomically commit the new `(committed_root, committed_version)`
    ///    via `backend.write_committed_meta` — the FileBackend
    ///    fsyncs the data file before the meta-rename, so the new root
    ///    chain is durable before any reader can observe it.
    ///
    /// `committed_version` increments by 1 per successful sync. A
    /// no-op sync (no dirty work) still bumps the version — that is
    /// fine and keeps the version monotonic without depending on
    /// "is anything dirty?" introspection.
    pub fn sync(&self) -> Result<(), Error> {
        // Phase 4: opportunistic GC of deferred frees. If no in-process
        // snapshot is live AND no peer process holds a snapshot, the
        // pending queue can drain into the buffer pool's free list and
        // those pages become reusable on the next allocation. Cheap
        // when there's nothing to drain.
        self.try_drain_pending_frees();

        self.flush_dirty_nodes()?;
        self.buffer_pool.flush_all()?;

        // Atomic root + version commit. The buffer pool's flush_all
        // already left the data file synced (FileBackend.flush_all
        // calls backend.sync). Backends that genuinely persist meta
        // (FileBackend) will additionally fsync inside
        // write_committed_meta; backends with the default no-op impl
        // (legacy disk, encrypted) skip the work entirely without
        // breaking correctness.
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let new_version = self.committed_version.fetch_add(1, Ordering::SeqCst) + 1;
        // **Phase 5 atomic publish**: publish the new committed_root
        // BEFORE writing the on-disk meta. Snapshot readers calling
        // `current_committed_meta` race against this; the worst case
        // is that a snapshot opened after this store but before the
        // write_committed_meta returns sees the new (root, version)
        // already — durable on disk because flush_all already synced
        // the data file. Either way the snapshot is consistent.
        self.committed_root.store(root_id, Ordering::SeqCst);
        let meta = crate::storage::CommittedMeta {
            format_version: 1,
            committed_version: new_version,
            committed_root: root_id,
        };
        self.buffer_pool
            .write_committed_meta(&meta)
            .map_err(Error::Storage)?;

        Ok(())
    }

    /// Most-recently-committed `(committed_root, committed_version)`
    /// for snapshot readers. Reads `committed_root` (the field
    /// published atomically at the last `sync()`) rather than the
    /// in-memory `root_page_id` (which rotates on every put). A
    /// snapshot opened during a multi-put transaction would see a
    /// torn root if it captured `root_page_id` directly — this
    /// getter is the safe one.
    pub fn current_committed_meta(&self) -> crate::storage::CommittedMeta {
        crate::storage::CommittedMeta {
            format_version: 1,
            committed_version: self.committed_version.load(Ordering::SeqCst),
            committed_root: self.committed_root.load(Ordering::SeqCst),
        }
    }

    /// **CoW MVCC Phase 7 — read-only-process helper.** Rotate the
    /// engine's in-memory `root_page_id` to match the most recently
    /// observed `committed_root`. Existing read paths (`get`,
    /// `range`, `scan`) traverse from `root_page_id`, so this makes
    /// them automatically observe peer commits after a
    /// `refresh_from_backend` call.
    ///
    /// **Do NOT call this in a process that performs writes.** A
    /// writer's `root_page_id` is mid-cascade between `put` and the
    /// next `sync`; overwriting it loses uncommitted state.
    /// Read-only processes (e.g. scholar-server reading
    /// scholar-ingestd's commits) can use this safely because they
    /// never have a half-committed in-memory tree.
    pub fn promote_committed_to_root_readonly(&self) {
        let committed = self.committed_root.load(Ordering::SeqCst);
        if let Ok(mut r) = self.root_page_id.write() {
            *r = committed;
        }
    }

    /// **CoW MVCC Phase 6.** Re-read the backend's atomically-committed
    /// meta record from disk. If the on-disk record is newer than this
    /// engine's in-memory mirror, update the in-memory atomics so
    /// subsequent `current_committed_meta` calls reflect the latest
    /// state.
    ///
    /// **Why this exists:** scholar-server and scholar-ingestd are
    /// separate processes that each open their own `Engine`. When
    /// ingestd commits a new root, scholar-server's engine has no way
    /// to learn about it through in-memory state. This method bridges
    /// the gap — call it periodically (or before opening a snapshot
    /// that should reflect the latest peer commits) so the reader
    /// process sees writers' progress.
    ///
    /// In single-process workloads this is a no-op fast path: the
    /// in-memory atomics are already in sync with the backend's
    /// in-memory mirror after `sync` returns.
    pub fn refresh_from_backend(&self) -> Result<crate::storage::CommittedMeta, Error> {
        let backend_meta = self
            .buffer_pool
            .read_committed_meta()
            .map_err(Error::Storage)?;
        if let Some(meta) = backend_meta {
            // Both atomics are SeqCst so the transition is observable.
            // We don't need a CAS loop because monotonicity is enforced
            // by the FileBackend's write_committed_meta refusing to
            // retire to an older version — the only writes to this
            // engine's atomics come from sync (in-process) or this
            // method (out-of-process), both of which only ever push
            // forward.
            let our_version = self.committed_version.load(Ordering::SeqCst);
            if meta.committed_version > our_version {
                self.committed_root
                    .store(meta.committed_root, Ordering::SeqCst);
                self.committed_version
                    .store(meta.committed_version, Ordering::SeqCst);
            }
            Ok(meta)
        } else {
            // Backend has no committed_meta yet (fresh DB, or a
            // backend that doesn't persist meta). Return our current
            // in-memory view.
            Ok(self.current_committed_meta())
        }
    }

    /// **CoW MVCC Phase 3.** Increment the live-snapshot counter.
    /// While > 0, `defer_free_page` queues frees instead of releasing
    /// them, so a snapshot reader cannot observe a freed-then-reused
    /// page. Called by `Database::open_snapshot`.
    pub fn acquire_snapshot(&self) {
        self.live_snapshots
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Decrement the live-snapshot counter. When it reaches 0 **and**
    /// no peer process holds a snapshot, the pending-free queue is
    /// drained — releasing the deferred page ids back to the buffer
    /// pool's free list. Called from `Snapshot::drop`.
    ///
    /// Phase 4 adds the cross-process check: even when our last
    /// in-process snapshot drops, a peer process's snapshot may
    /// still be reading pages we'd otherwise release. We check
    /// `buffer_pool.any_external_snapshots_live()` (which scans
    /// `<db>/snapshots/` for peer lockfiles) before draining. If a
    /// peer is live, the pending queue stays — the next drain
    /// attempt happens on the next `try_drain_pending_frees` call
    /// (e.g. at sync time, or in a future scheduled GC).
    pub fn release_snapshot(&self) {
        let prev = self
            .live_snapshots
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        if prev == 1 {
            self.try_drain_pending_frees();
        }
    }

    /// Attempt to drain the pending-free queue. Releases the queued
    /// pages to the buffer pool only if no in-process snapshot is
    /// active **and** no peer process holds a snapshot. A no-op when
    /// either condition fails.
    ///
    /// Called from `release_snapshot` (after the in-process counter
    /// drops to 0) and from `sync` (so peer-process snapshot
    /// transitions don't go unnoticed between explicit drop events).
    fn try_drain_pending_frees(&self) {
        if self
            .live_snapshots
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
        {
            return;
        }
        if self.buffer_pool.any_external_snapshots_live() {
            return;
        }
        let mut pending = self
            .pending_free_pages
            .lock()
            .expect("lock poisoned: pending_free_pages");
        let drained: Vec<PageId> = pending.drain(..).collect();
        drop(pending);
        for page_id in drained {
            // Best-effort — a single backend failure shouldn't abort
            // the others. In practice a healthy backend's free_page
            // doesn't fail.
            let _ = self.buffer_pool.free_page(page_id);
        }
    }

    /// **CoW MVCC Phase 4 — diagnostic.** Number of page ids
    /// currently sitting in the pending-free queue. Used by tests
    /// (and operators) to verify deferred frees are or aren't
    /// being released.
    pub fn pending_free_count(&self) -> usize {
        self.pending_free_pages
            .lock()
            .expect("lock poisoned: pending_free_pages")
            .len()
    }

    /// Record that `page_id` was just freshly allocated by the
    /// currently in-flight write (via `buffer_pool.new_page()` in
    /// `save_node_cow` or `write_overflow_chain`). Cleared on
    /// successful commit, drained-and-freed on `abort_uncommitted`.
    /// See the `pending_allocations` field doc for the leak this
    /// prevents.
    fn track_new_page(&self, page_id: PageId) {
        self.pending_allocations
            .lock()
            .expect("lock poisoned: pending_allocations (track)")
            .push(page_id);
    }

    /// Clear the in-flight allocation tracking. Called on a
    /// successful commit — the new pages are now reachable from the
    /// committed root and must NOT be freed.
    fn clear_pending_allocations(&self) {
        self.pending_allocations
            .lock()
            .expect("lock poisoned: pending_allocations (clear)")
            .clear();
    }

    /// Free a page, deferring the actual release if any snapshot is
    /// currently live (in-process or cross-process). Called from
    /// B-tree merge paths and overflow / extent deletion paths.
    fn defer_free_page(&self, page_id: PageId) -> Result<(), Error> {
        // Defer if either:
        //   - our process has a live snapshot (cheap atomic read), or
        //   - a peer process has a live snapshot (filesystem scan via
        //     the snapshot registry — `any_external_snapshots_live`).
        // The cheap check is first to skip the FS scan in the common
        // single-process case.
        let in_process_live = self
            .live_snapshots
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0;
        if in_process_live || self.buffer_pool.any_external_snapshots_live() {
            self.pending_free_pages
                .lock()
                .expect("lock poisoned: pending_free_pages")
                .push(page_id);
            Ok(())
        } else {
            self.buffer_pool.free_page(page_id).map_err(Error::Storage)
        }
    }

    /// **CoW MVCC Phase 3.** Get a value by key, traversing from a
    /// caller-supplied root rather than the engine's current
    /// `root_page_id`. This is what `Snapshot::get` calls.
    ///
    /// The caller must have either already captured the root via
    /// `acquire_snapshot` (so deferred frees keep the subtree intact)
    /// or be reading the engine's current root.
    pub fn get_at_root(
        &self,
        root: PageId,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, Error> {
        // Latched traversal — same correctness as the standard
        // `get_internal` path but skips the optimistic latch-free
        // attempt to avoid coupling snapshot reads to the writer's
        // structure_version. A snapshot's root is by construction
        // an immutable subtree once acquire_snapshot has been called,
        // so the latch-free retry-on-version-change isn't needed.
        self.search_node(root, key)
    }

    /// Write a large value to a chain of overflow pages.
    /// Returns the PageId of the first overflow page.
    fn write_overflow_chain(&self, value: &[u8]) -> Result<PageId, Error> {
        let usable = self.max_page_data - OVERFLOW_HEADER_SIZE;
        let mut remaining = value;
        let mut pages: Vec<(PageId, &[u8])> = Vec::new();

        // Allocate all pages first, then link them
        while !remaining.is_empty() {
            let chunk_len = remaining.len().min(usable);
            let chunk = &remaining[..chunk_len];
            remaining = &remaining[chunk_len..];

            let page_arc = self.buffer_pool.new_page()?;
            let page_id = page_arc
                .read()
                .expect("lock poisoned: overflow page read")
                .id;
            self.track_new_page(page_id);
            pages.push((page_id, chunk));
        }

        // Write pages with forward links
        for i in 0..pages.len() {
            let (page_id, chunk) = pages[i];
            let next_id = if i + 1 < pages.len() {
                pages[i + 1].0
            } else {
                0u64
            };

            let page_arc = self.buffer_pool.get_page(page_id)?;
            let mut page = page_arc
                .write()
                .expect("lock poisoned: overflow page write");
            page.page_type = PageType::Overflow;

            let mut data = Vec::with_capacity(OVERFLOW_HEADER_SIZE + chunk.len());
            data.extend_from_slice(&next_id.to_le_bytes());
            data.extend_from_slice(chunk);
            page.data = data;
            page.mark_dirty();
        }

        Ok(pages[0].0)
    }

    /// Read a value from a chain of overflow pages.
    fn read_overflow_chain(
        &self,
        first_page_id: PageId,
        total_len: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut result = Vec::with_capacity(total_len);
        let mut current_page_id = first_page_id;

        while current_page_id != 0 {
            let page_arc = self.buffer_pool.get_page(current_page_id)?;
            let page = page_arc.read().expect("lock poisoned: overflow page read");

            if page.data.len() < OVERFLOW_HEADER_SIZE {
                return Err(Error::Storage(StorageError::Corrupted {
                    page_id: current_page_id,
                    reason: "Overflow page too short".to_string(),
                }));
            }

            let next_id = u64::from_le_bytes([
                page.data[0],
                page.data[1],
                page.data[2],
                page.data[3],
                page.data[4],
                page.data[5],
                page.data[6],
                page.data[7],
            ]);
            result.extend_from_slice(&page.data[OVERFLOW_HEADER_SIZE..]);
            current_page_id = next_id;
        }

        result.truncate(total_len);
        Ok(result)
    }

    /// Free a chain of overflow pages.
    fn free_overflow_chain(&self, first_page_id: PageId) -> Result<(), Error> {
        let mut current = first_page_id;
        while current != 0 {
            let page_arc = self.buffer_pool.get_page(current)?;
            let page = page_arc.read().expect("lock poisoned: overflow page read");
            let next = if page.data.len() >= OVERFLOW_HEADER_SIZE {
                u64::from_le_bytes([
                    page.data[0],
                    page.data[1],
                    page.data[2],
                    page.data[3],
                    page.data[4],
                    page.data[5],
                    page.data[6],
                    page.data[7],
                ])
            } else {
                0
            };
            drop(page);
            // Phase 3: snapshot-aware deferral — see fn defer_free_page.
            self.defer_free_page(current)?;
            current = next;
        }
        Ok(())
    }

    // ── Extent-based large blob storage ──────────────────────────────
    //
    // For values larger than ~256KB, extents are more efficient than
    // overflow page chains. An extent is a contiguous run of pages
    // allocated sequentially, enabling single-seek sequential I/O.
    //
    // Layout:
    //   Page 0 (ExtentHeader): [page_count: u64][total_bytes: u64][data...]
    //   Page 1..N (ExtentData): [data...]
    //
    // The B-tree leaf stores: (VALUE_MARKER_EXTENT, first_page_id, total_bytes)

    /// Minimum value size to use extent storage instead of overflow chains.
    /// Below this threshold, overflow chains are fine (few pages, minimal seeking).
    /// Above this, extents avoid the linked-list traversal penalty.
    const EXTENT_THRESHOLD: usize = 256 * 1024; // 256 KB

    /// Write a large value as a contiguous extent.
    /// Returns the PageId of the extent header page.
    pub fn write_extent(&self, value: &[u8]) -> Result<PageId, Error> {
        let usable_first = self.max_page_data - EXTENT_HEADER_SIZE;
        let usable_rest = self.max_page_data;

        // Calculate number of pages needed
        let page_count = if value.len() <= usable_first {
            1
        } else {
            1 + (value.len() - usable_first + usable_rest - 1) / usable_rest
        };

        // Allocate contiguous pages (single sequential range)
        let first_page_id = self.buffer_pool.allocate_contiguous(page_count)?;

        // Write extent header page (page 0)
        {
            let page_arc = self.buffer_pool.get_page(first_page_id)?;
            let mut page = page_arc.write().expect("lock poisoned");
            page.page_type = PageType::ExtentHeader;

            let first_chunk_len = value.len().min(usable_first);
            let mut data = Vec::with_capacity(EXTENT_HEADER_SIZE + first_chunk_len);
            data.extend_from_slice(&(page_count as u64).to_le_bytes());
            data.extend_from_slice(&(value.len() as u64).to_le_bytes());
            data.extend_from_slice(&value[..first_chunk_len]);
            page.data = data;
            page.mark_dirty();
        }

        // Write extent data pages (contiguous: first+1, first+2, ...)
        let mut offset = usable_first.min(value.len());
        for i in 1..page_count {
            let chunk_len = (value.len() - offset).min(usable_rest);
            let page_arc = self.buffer_pool.get_page(first_page_id + i as u64)?;
            let mut page = page_arc.write().expect("lock poisoned");
            page.page_type = PageType::ExtentData;
            page.data = value[offset..offset + chunk_len].to_vec();
            page.mark_dirty();
            offset += chunk_len;
        }

        Ok(first_page_id)
    }

    /// Read a value from a contiguous extent.
    /// `first_page_id` is the extent header page.
    pub fn read_extent(
        &self,
        first_page_id: PageId,
        total_len: usize,
    ) -> Result<Vec<u8>, Error> {
        // Read header page
        let page_arc = self.buffer_pool.get_page(first_page_id)?;
        let page = page_arc.read().expect("lock poisoned");

        if page.data.len() < EXTENT_HEADER_SIZE {
            return Err(Error::Storage(StorageError::Corrupted {
                page_id: first_page_id,
                reason: "Extent header too short".to_string(),
            }));
        }

        let page_count = u64::from_le_bytes(
            page.data[0..8].try_into().unwrap()
        ) as usize;
        let stored_total = u64::from_le_bytes(
            page.data[8..16].try_into().unwrap()
        ) as usize;

        let expected_len = total_len.min(stored_total);
        let mut result = Vec::with_capacity(expected_len);

        // Copy data from header page (after the 16-byte extent header)
        let first_data = &page.data[EXTENT_HEADER_SIZE..];
        let first_chunk = first_data.len().min(expected_len);
        result.extend_from_slice(&first_data[..first_chunk]);
        drop(page);

        // Read subsequent extent data pages (contiguous page IDs)
        for i in 1..page_count {
            if result.len() >= expected_len {
                break;
            }
            let data_page_id = first_page_id + i as u64;
            let data_page_arc = self.buffer_pool.get_page(data_page_id)?;
            let data_page = data_page_arc.read().expect("lock poisoned");

            let remaining = expected_len - result.len();
            let chunk = data_page.data.len().min(remaining);
            result.extend_from_slice(&data_page.data[..chunk]);
        }

        result.truncate(expected_len);
        Ok(result)
    }

    /// Free a contiguous extent.
    pub fn free_extent(&self, first_page_id: PageId) -> Result<(), Error> {
        // Read header to get page count
        let page_arc = self.buffer_pool.get_page(first_page_id)?;
        let page = page_arc.read().expect("lock poisoned");

        let page_count = if page.data.len() >= 8 {
            u64::from_le_bytes(page.data[0..8].try_into().unwrap()) as usize
        } else {
            1
        };
        drop(page);

        // Free all pages in the extent (contiguous IDs).
        // Phase 3: snapshot-aware deferral — see fn defer_free_page.
        for i in 0..page_count {
            self.defer_free_page(first_page_id + i as u64)?;
        }

        Ok(())
    }

    /// Serialize a node for page storage, writing large values to overflow pages.
    /// Returns the serialized node data with overflow references for large values.
    fn serialize_node_with_overflow(&self, node: &BTreeNode) -> Result<Vec<u8>, Error> {
        let mut buf = Vec::with_capacity(node.serialized_size().min(self.max_page_data));

        buf.push(if node.is_leaf { 1 } else { 0 });
        buf.extend_from_slice(&(node.keys.len() as u32).to_le_bytes());

        for key in &node.keys {
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key);
        }

        if node.is_leaf {
            for value in &node.values {
                match value {
                    None => buf.push(VALUE_MARKER_NONE),
                    Some(v) if v.is_empty() => buf.push(VALUE_MARKER_EMPTY),
                    Some(v) if v.len() >= Self::EXTENT_THRESHOLD => {
                        // Large blob: use contiguous extent (sequential I/O)
                        let first_page = self.write_extent(v)?;
                        buf.push(VALUE_MARKER_EXTENT);
                        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        buf.extend_from_slice(&first_page.to_le_bytes());
                    }
                    Some(v) if v.len() > self.max_inline_value_size => {
                        // Medium blob: use overflow chain (few pages)
                        let first_page = self.write_overflow_chain(v)?;
                        buf.push(VALUE_MARKER_OVERFLOW);
                        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        buf.extend_from_slice(&first_page.to_le_bytes());
                    }
                    Some(v) => {
                        buf.push(VALUE_MARKER_DATA);
                        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        buf.extend_from_slice(v);
                    }
                }
            }
        }

        if !node.is_leaf {
            buf.extend_from_slice(&(node.children.len() as u32).to_le_bytes());
            for child in &node.children {
                buf.extend_from_slice(&child.to_le_bytes());
            }
        }

        Ok(buf)
    }

    /// Deserialize a node from page data, resolving overflow references.
    fn deserialize_node_with_overflow(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> Result<BTreeNode, Error> {
        // First do basic deserialization that handles overflow markers
        if data.is_empty() {
            return Err(Error::Storage(StorageError::Corrupted {
                page_id,
                reason: "Empty node data".to_string(),
            }));
        }

        let mut cursor = 0;
        let is_leaf = data[cursor] != 0;
        cursor += 1;

        if cursor + 4 > data.len() {
            return Err(Error::Storage(StorageError::Corrupted {
                page_id,
                reason: "Missing key count".to_string(),
            }));
        }
        let num_keys = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        let mut keys = Vec::with_capacity(num_keys);
        for _ in 0..num_keys {
            if cursor + 4 > data.len() {
                return Err(Error::Storage(StorageError::Corrupted {
                    page_id,
                    reason: "Key length truncated".to_string(),
                }));
            }
            let klen = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;
            cursor += 4;
            if cursor + klen > data.len() {
                return Err(Error::Storage(StorageError::Corrupted {
                    page_id,
                    reason: "Key data truncated".to_string(),
                }));
            }
            keys.push(data[cursor..cursor + klen].to_vec());
            cursor += klen;
        }

        let mut values = Vec::with_capacity(if is_leaf { num_keys } else { 0 });
        let mut children = Vec::new();

        if is_leaf {
            for _ in 0..num_keys {
                if cursor >= data.len() {
                    return Err(Error::Storage(StorageError::Corrupted {
                        page_id,
                        reason: "Value marker truncated".to_string(),
                    }));
                }
                let marker = data[cursor];
                cursor += 1;
                match marker {
                    VALUE_MARKER_NONE => values.push(None),
                    VALUE_MARKER_EMPTY => values.push(Some(Vec::new())),
                    VALUE_MARKER_DATA => {
                        if cursor + 4 > data.len() {
                            return Err(Error::Storage(StorageError::Corrupted {
                                page_id,
                                reason: "Value length truncated".to_string(),
                            }));
                        }
                        let vlen = u32::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                        ]) as usize;
                        cursor += 4;
                        if cursor + vlen > data.len() {
                            return Err(Error::Storage(StorageError::Corrupted {
                                page_id,
                                reason: "Value data truncated".to_string(),
                            }));
                        }
                        values.push(Some(data[cursor..cursor + vlen].to_vec()));
                        cursor += vlen;
                    }
                    VALUE_MARKER_OVERFLOW => {
                        if cursor + 12 > data.len() {
                            return Err(Error::Storage(StorageError::Corrupted {
                                page_id,
                                reason: "Overflow reference truncated".to_string(),
                            }));
                        }
                        let total_len = u32::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                        ]) as usize;
                        cursor += 4;
                        let first_page = u64::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                            data[cursor + 4],
                            data[cursor + 5],
                            data[cursor + 6],
                            data[cursor + 7],
                        ]);
                        cursor += 8;
                        // Read the full value from overflow pages
                        let full_value = self.read_overflow_chain(first_page, total_len)?;
                        values.push(Some(full_value));
                    }
                    VALUE_MARKER_EXTENT => {
                        if cursor + 12 > data.len() {
                            return Err(Error::Storage(StorageError::Corrupted {
                                page_id,
                                reason: "Extent reference truncated".to_string(),
                            }));
                        }
                        let total_len = u32::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                        ]) as usize;
                        cursor += 4;
                        let first_page = u64::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                            data[cursor + 4],
                            data[cursor + 5],
                            data[cursor + 6],
                            data[cursor + 7],
                        ]);
                        cursor += 8;
                        // Read from contiguous extent (sequential I/O)
                        let full_value = self.read_extent(first_page, total_len)?;
                        values.push(Some(full_value));
                    }
                    _ => {
                        return Err(Error::Storage(StorageError::Corrupted {
                            page_id,
                            reason: format!("Unknown value marker: {}", marker),
                        }));
                    }
                }
            }
        } else {
            if cursor + 4 > data.len() {
                return Err(Error::Storage(StorageError::Corrupted {
                    page_id,
                    reason: "Missing children count".to_string(),
                }));
            }
            let num_children = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;
            cursor += 4;
            children.reserve(num_children);
            for _ in 0..num_children {
                if cursor + 8 > data.len() {
                    return Err(Error::Storage(StorageError::Corrupted {
                        page_id,
                        reason: "Child pointer truncated".to_string(),
                    }));
                }
                let child_id = u64::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                    data[cursor + 4],
                    data[cursor + 5],
                    data[cursor + 6],
                    data[cursor + 7],
                ]);
                cursor += 8;
                children.push(child_id);
            }
        }

        Ok(BTreeNode {
            keys,
            values,
            children,
            is_leaf,
            page_id,
        })
    }

    /// Flush dirty nodes from cache to buffer pool pages.
    /// Called by sync() before buffer pool flush. Serializes only nodes
    /// modified since the last flush (write-back cache).
    fn flush_dirty_nodes(&self) -> Result<(), Error> {
        // Collect dirty entries and mark clean (per-shard locks, released per shard)
        let mut to_flush: Vec<(PageId, Arc<BTreeNode>)> = Vec::new();
        for mut entry in self.node_cache.iter_mut() {
            if entry.dirty {
                to_flush.push((*entry.key(), Arc::clone(&entry.node)));
                entry.dirty = false;
            }
        }

        // Serialize BEFORE acquiring page locks — serialize_node_with_overflow may
        // allocate overflow pages (new_page → evict_one), which would deadlock if
        // the eviction victim's write lock is already held by us.
        for (page_id, node_arc) in to_flush {
            let data = self.serialize_node_with_overflow(&node_arc)?;
            let page_type = if node_arc.is_leaf {
                PageType::BTreeLeaf
            } else {
                PageType::BTreeInternal
            };

            let page_arc = self.buffer_pool.get_page(page_id)?;
            let mut page = page_arc.write().expect("lock poisoned: page write");
            page.data = data;
            page.page_type = page_type;
            page.mark_dirty();
        }
        Ok(())
    }

    /// Range scan over keys
    ///
    /// Returns an iterator over key-value pairs in the specified range.
    ///
    /// # Example
    /// ```ignore
    /// // Scan all keys starting with "user:"
    /// let iter = engine.range(
    ///     Bound::Included(b"user:"),
    ///     Bound::Excluded(b"user;"),  // ';' is after ':' in ASCII
    ///     ScanDirection::Forward,
    /// )?;
    /// for result in iter {
    ///     let entry = result?;
    ///     println!("{:?} = {:?}", entry.key, entry.value);
    /// }
    /// ```
    pub fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<BTreeRangeIterator<'_>, Error> {
        Ok(BTreeRangeIterator::new(self, start, end, direction))
    }

    /// Scan all keys in the database
    pub fn scan(&self, direction: ScanDirection) -> Result<BTreeRangeIterator<'_>, Error> {
        self.range(Bound::Unbounded, Bound::Unbounded, direction)
    }

    /// **CoW MVCC Phase 6.** Range scan pinned to a specific root page
    /// id — the snapshot variant of `range`. Reads traverse only the
    /// subtree reachable from `root`; the engine's mutable
    /// `root_page_id` (which a concurrent writer rotates on every
    /// put) is ignored.
    ///
    /// Used by `Snapshot::range`. Callers must hold a `Snapshot`
    /// to ensure `root` and its descendants stay reachable —
    /// `defer_free_page` guarantees this while any in-process or
    /// peer-process snapshot is live.
    pub fn range_at_root(
        &self,
        root: PageId,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<BTreeRangeIterator<'_>, Error> {
        Ok(BTreeRangeIterator::new_with_root(
            self,
            Some(root),
            start,
            end,
            direction,
        ))
    }

    /// Snapshot-pinned full scan; sibling of `scan` for snapshot reads.
    pub fn scan_at_root(
        &self,
        root: PageId,
        direction: ScanDirection,
    ) -> Result<BTreeRangeIterator<'_>, Error> {
        self.range_at_root(root, Bound::Unbounded, Bound::Unbounded, direction)
    }

    /// Snapshot-pinned prefix scan; sibling of `prefix_scan` for snapshot reads.
    pub fn prefix_scan_at_root(
        &self,
        root: PageId,
        prefix: &[u8],
    ) -> Result<BTreeRangeIterator<'_>, Error> {
        if prefix.is_empty() {
            return self.scan_at_root(root, ScanDirection::Forward);
        }
        let mut end_prefix = prefix.to_vec();
        while let Some(&0xFF) = end_prefix.last() {
            end_prefix.pop();
        }
        if let Some(last) = end_prefix.last_mut() {
            *last += 1;
            self.range_at_root(
                root,
                Bound::Included(prefix),
                Bound::Excluded(end_prefix.as_slice()),
                ScanDirection::Forward,
            )
        } else {
            self.range_at_root(
                root,
                Bound::Included(prefix),
                Bound::Unbounded,
                ScanDirection::Forward,
            )
        }
    }

    /// Count keys matching `prefix` without materialising values.
    ///
    /// Tree-walk that loads each B-tree node once, sums leaf keys
    /// matching the prefix, and prunes subtrees whose key range
    /// doesn't overlap. Designed for hot paths like the row-count
    /// metric where the existing `prefix_scan` iterator's per-key
    /// `key.clone() + value.clone() + IndexEntry::new()` work was
    /// dominating runtime — a 1 M-row table with 4 KB JSON values
    /// would clone ~4 GB of data per scan, plus repeated cache
    /// misses on every cross-leaf transition. Production
    /// observation 2026-05-07: scholar-server's `refresh_row_counts`
    /// took >30 min per scan via `prefix_scan` on a 600 GB DB; the
    /// tree-walk path completes the same 9-table refresh in ~37 s.
    ///
    /// Returns `(count, errors)`. Subtrees that fail (corrupt
    /// internal page, depth-bound exceeded, decode error) are
    /// recorded in `errors` and skipped — the count reflects only
    /// the reachable rows. This matches `prefix_scan`'s observable
    /// behaviour of swallowing per-entry iterator errors, but with
    /// visibility: callers learn which page failed so the corruption
    /// is debuggable instead of silently shrinking the count.
    pub fn prefix_count(&self, prefix: &[u8]) -> Result<PrefixCount, Error> {
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let max_depth = self.tree_height.load(Ordering::Acquire) as usize + 16;
        let mut errors: Vec<(PageId, String)> = Vec::new();
        let mut ancestors: Vec<PageId> = Vec::with_capacity(max_depth);
        let count = self.count_subtree(root_id, prefix, max_depth, &mut ancestors, &mut errors);
        Ok(PrefixCount { count, errors })
    }

    /// Recursive helper: never returns Err. On error in a subtree,
    /// records (page_id, reason) in `errors` and returns 0 for that
    /// subtree, so siblings still contribute to the total.
    ///
    /// `ancestors` is the descent path from the root to (but not
    /// including) `page_id`. If `page_id` already appears in
    /// `ancestors`, we've found a cycle — record it and bail without
    /// recursing further. This prevents the depth-bound from
    /// counting the same leaves multiple times (each cycle traversal
    /// loops up to `depth_remaining` times before erroring, and
    /// each cycle visit re-counts the cycle's leaves; in production
    /// 2026-05-07 a few thousand corrupt pages drove scholar_
    /// citations' count from ~470 K real rows to a reported 1.5 B).
    fn count_subtree(
        &self,
        page_id: PageId,
        prefix: &[u8],
        depth_remaining: usize,
        ancestors: &mut Vec<PageId>,
        errors: &mut Vec<(PageId, String)>,
    ) -> u64 {
        if depth_remaining == 0 {
            errors.push((
                page_id,
                "depth bound exceeded — corrupt internal page chain".into(),
            ));
            return 0;
        }
        if ancestors.contains(&page_id) {
            // Cycle: this page is already on the descent path.
            // Recursing would re-count any leaves underneath it.
            errors.push((page_id, "cycle detected on descent path".into()));
            return 0;
        }
        let node = match self.load_node(page_id) {
            Ok(n) => n,
            Err(e) => {
                errors.push((page_id, format!("load_node: {e}")));
                return 0;
            }
        };
        if node.is_leaf {
            // Sum keys matching prefix. Scholar's row keys are sorted
            // (`row::<table>\x00<pk>`), so we could short-circuit on
            // the first key past the prefix range — but the saving
            // is small relative to the parent walk and the simple
            // scan is easier to verify. Tombstones (value=None) are
            // EXCLUDED from the count to match `prefix_scan`'s
            // observable behaviour.
            let mut count: u64 = 0;
            for (i, key) in node.keys.iter().enumerate() {
                if key.starts_with(prefix) {
                    if let Some(values_slot) = node.values.get(i) {
                        if values_slot.is_some() {
                            count += 1;
                        }
                    }
                }
            }
            return count;
        }
        // Internal node: recurse only into children whose key range
        // could contain a key with the given prefix.
        ancestors.push(page_id);
        let mut total: u64 = 0;
        for (i, child_id) in node.children.iter().enumerate() {
            let lower = if i == 0 { None } else { node.keys.get(i - 1) };
            let upper = node.keys.get(i);
            if !child_range_overlaps_prefix(lower, upper, prefix) {
                continue;
            }
            total +=
                self.count_subtree(*child_id, prefix, depth_remaining - 1, ancestors, errors);
        }
        ancestors.pop();
        total
    }

    /// Get all keys with a given prefix
    ///
    /// This is a convenience method that creates an appropriate range scan.
    pub fn prefix_scan(&self, prefix: &[u8]) -> Result<BTreeRangeIterator<'_>, Error> {
        if prefix.is_empty() {
            return self.scan(ScanDirection::Forward);
        }

        // Compute the successor prefix by incrementing the prefix as a big-endian
        // integer. Strip trailing 0xFF bytes and increment the last non-0xFF byte.
        // e.g., "user:" -> "user;" (';' = ':' + 1)
        // e.g., [0x01, 0xFF] -> [0x02]
        // If ALL bytes are 0xFF, the prefix matches everything from prefix onward.
        let mut end_prefix = prefix.to_vec();
        while let Some(&0xFF) = end_prefix.last() {
            end_prefix.pop();
        }

        if let Some(last) = end_prefix.last_mut() {
            *last += 1;
            self.range(
                Bound::Included(prefix),
                Bound::Excluded(end_prefix.as_slice()),
                ScanDirection::Forward,
            )
        } else {
            // All bytes were 0xFF — scan from prefix to end
            self.range(
                Bound::Included(prefix),
                Bound::Unbounded,
                ScanDirection::Forward,
            )
        }
    }

    // Internal methods

    fn get_internal(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");

        // Optimistic latch-free read with structure version check.
        // Structural changes (splits/merges) are rare (~1 per 256 inserts),
        // so this succeeds on the first attempt in the vast majority of cases.
        for _ in 0..3 {
            let version = self.structure_version.load(Ordering::Acquire);
            match self.search_node_latchfree(root_id, key) {
                Ok(result) => {
                    if self.structure_version.load(Ordering::Acquire) == version {
                        return Ok(result);
                    }
                    // Structure changed during traversal — retry
                    continue;
                }
                Err(_) => break, // fallback to latched path
            }
        }

        // Fallback: latched traversal (guaranteed correct)
        self.search_node(root_id, key)
    }

    /// Latch-free tree traversal for optimistic reads.
    /// Same logic as search_node but without page latch acquisition.
    /// Safe because load_node returns immutable Arc<BTreeNode> snapshots.
    ///
    /// **Iterative + depth-bounded** (2026-05-12): a cycle in the
    /// tree's internal child pointers (corrupt page) would otherwise
    /// recurse forever and abort the process with a stack overflow —
    /// in production this crash-looped scholar-server on every query
    /// that descended through a corrupt subtree. Returns
    /// `Err(StorageError::Corrupted)` instead; `get_internal` already
    /// treats that as a fallback signal, and the latched
    /// `search_node` (also bounded) then returns the same error to
    /// the caller — a graceful failure, not a process abort. Mirrors
    /// the iterator-descent and `count_subtree` hardening.
    fn search_node_latchfree(
        &self,
        mut page_id: PageId,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, Error> {
        let max_depth = self.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let node = self.load_node(page_id)?;
            let idx = node.find_key_index(key);
            if node.is_leaf {
                return Ok(if idx < node.keys.len() && node.keys[idx] == key {
                    node.values[idx].clone()
                } else {
                    None
                });
            }
            page_id = if idx < node.keys.len() && node.keys[idx] == key {
                node.children[idx + 1]
            } else {
                node.children[idx]
            };
        }
        Err(Error::Storage(StorageError::Corrupted {
            page_id,
            reason: "search_node_latchfree: depth bound exceeded — \
                     likely a cycle in the tree's internal child pointers"
                .into(),
        }))
    }

    /// Latched point lookup. **Iterative + depth-bounded** for the
    /// same reason as `search_node_latchfree` — see that fn's doc.
    /// Each loop iteration takes the read latch for the current page,
    /// reads it, releases the latch, and steps to the child (exactly
    /// the latch lifetime the old recursive form had per level).
    fn search_node(&self, mut page_id: PageId, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let max_depth = self.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let step: SearchResult = self.page_latches.with_read(page_id, || -> Result<SearchResult, Error> {
                let node = self.load_node(page_id)?;
                let idx = node.find_key_index(key);
                if node.is_leaf {
                    if idx < node.keys.len() && node.keys[idx] == key {
                        Ok(SearchResult::Found(node.values[idx].clone()))
                    } else {
                        Ok(SearchResult::NotFound)
                    }
                } else {
                    let child_id = if idx < node.keys.len() && node.keys[idx] == key {
                        node.children[idx + 1]
                    } else {
                        node.children[idx]
                    };
                    Ok(SearchResult::Continue(child_id))
                }
            })?;
            match step {
                SearchResult::Found(value) => return Ok(value),
                SearchResult::NotFound => return Ok(None),
                SearchResult::Continue(child_id) => page_id = child_id,
            }
        }
        Err(Error::Storage(StorageError::Corrupted {
            page_id,
            reason: "search_node: depth bound exceeded — \
                     likely a cycle in the tree's internal child pointers"
                .into(),
        }))
    }

    /// Traverse tree from `start_page_id` to find the leaf page for `key`.
    ///
    /// Requires: tree topology is stable (caller holds write_lock.read() or
    /// write_lock.write()). Uses cached nodes — no page latches needed.
    /// Compute tree height by traversing from root to leftmost leaf.
    /// Returns 0 if root is a leaf (no internal levels).
    fn compute_tree_height(&self) -> Result<u32, Error> {
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let mut current = root_id;
        let mut height = 0u32;
        // A B-tree's height is bounded by ~64 even at fanout 2; a much
        // larger cap means a cycle in the leftmost-child chain (corrupt
        // internal page). Erroring out beats spinning forever — and
        // beats returning a huge height that would inflate every
        // `tree_height + 16` depth bound elsewhere.
        for _ in 0..256 {
            let node = self.load_node(current)?;
            if node.is_leaf {
                return Ok(height);
            }
            let Some(&child0) = node.children.first() else {
                return Err(Error::Storage(StorageError::Corrupted {
                    page_id: current,
                    reason: "compute_tree_height: internal node has no children".into(),
                }));
            };
            height += 1;
            current = child0;
        }
        Err(Error::Storage(StorageError::Corrupted {
            page_id: current,
            reason: "compute_tree_height: leftmost-child chain exceeded 256 levels — \
                     likely a cycle in internal child pointers"
                .into(),
        }))
    }

    /// Find the leaf page that would contain `key` by descending through
    /// exactly `tree_height` internal levels. Never loads leaf nodes,
    /// avoiding a race with take_node_for_write in the optimistic path.
    fn find_leaf(&self, start_page_id: PageId, key: &[u8]) -> Result<PageId, Error> {
        let height = self.tree_height.load(Ordering::Acquire);
        if height == 0 {
            // Root is a leaf — return it directly without loading
            return Ok(start_page_id);
        }
        let mut current = start_page_id;
        for _ in 0..height {
            let node = self.load_node(current)?;
            let idx = node.find_key_index(key);
            current = if idx < node.keys.len() && node.keys[idx] == key {
                node.children[idx + 1]
            } else {
                node.children[idx]
            };
        }
        Ok(current)
    }

    fn put_locked(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        // Assumes write_lock is already held
        // let _write_guard = self.write_lock.write().unwrap();

        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let root = self.load_node(root_id)?;

        if root.is_full_by_size(self.max_page_data) {
            drop(root);
            // split_root publishes the new root id internally via root_page_id
            // (CoW: the new root is allocated fresh, the old root is retired
            // to a fresh id too — see fn split_root). update_metadata is
            // called there, so the on-disk meta page reflects the new root.
            self.split_root()?;
        }

        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");

        // CoW cascade: insert returns the (possibly new) page id of the
        // subtree it operated on. If the root's page id changed, rotate
        // root_page_id to the new id and re-publish the meta page.
        // Depth bound mirrors the iterator-side hardening
        // (descend_to_leftmost, etc.). A cycle in the tree's
        // internal-page child pointers would otherwise recurse
        // forever, blow the stack, and abort the process.
        // Production 2026-05-11: ingestd crash-looped here for ~2
        // days on a particular OpenAlex day before this guard
        // landed.
        let max_depth = self.tree_height.load(Ordering::Acquire) as usize + 16;
        let new_root_id = self.insert_non_full_serial(root_id, key, value, max_depth)?;
        if new_root_id != root_id {
            *self
                .root_page_id
                .write()
                .expect("lock poisoned: root_page_id write") = new_root_id;
            self.structure_version.fetch_add(1, Ordering::Release);
            self.update_metadata()?;
        }

        Ok(())
    }

    /// Insert into non-full node (serialized version - caller holds write_lock).
    ///
    /// **CoW**: returns the new page id of the subtree's root. When the
    /// caller is the parent of this subtree, it must rewrite its
    /// `children[idx]` pointer to the returned id and itself be saved
    /// CoW (cascading up to the tree root). Phase 1 of the MVCC refactor
    /// — see `docs/joule-db/cow-mvcc-design.md`.
    fn insert_non_full_serial(
        &self,
        page_id: PageId,
        key: &[u8],
        value: &[u8],
        depth_remaining: usize,
    ) -> Result<PageId, Error> {
        if depth_remaining == 0 {
            return Err(Error::Storage(StorageError::Corrupted {
                page_id,
                reason: "insert_non_full_serial: depth bound exceeded — \
                         likely a cycle in the tree's internal child \
                         pointers"
                    .into(),
            }));
        }
        let node = self.load_node(page_id)?;
        let idx = node.find_key_index(key);

        if node.is_leaf {
            drop(node);
            let mut node = self.take_node_for_write(page_id)?;
            let idx = node.find_key_index(key);
            if idx < node.keys.len() && node.keys[idx] == key {
                node.values[idx] = Some(value.to_vec());
            } else {
                node.keys.insert(idx, key.to_vec());
                node.values.insert(idx, Some(value.to_vec()));
            }
            // CoW: leaf gets a fresh page id; caller cascades.
            self.save_node_cow(node)
        } else {
            let child_idx = if idx < node.keys.len() && node.keys[idx] == key {
                idx + 1
            } else {
                idx
            };

            let child_page_id = node.children[child_idx];
            let child = self.load_node(child_page_id)?;
            let needs_split = child.is_full_by_size(self.max_page_data);
            drop(child);
            drop(node);

            // Take the parent for write so the post-recursion CoW save
            // captures the cascaded child id.
            let mut node = self.take_node_for_write(page_id)?;
            let idx = node.find_key_index(key);
            let child_idx = if idx < node.keys.len() && node.keys[idx] == key {
                idx + 1
            } else {
                idx
            };

            if needs_split {
                // split_child_serial CoW-saves both children and writes
                // their new ids back into node.children[child_idx] and
                // node.children[child_idx + 1]. The parent is held in
                // memory (not yet saved) so we can keep mutating it
                // before the final CoW save below.
                self.split_child_serial(&mut node, child_idx)?;
            }

            // After possible split, find the right child again.
            let new_idx = node.find_key_index(key);
            let target_child = if new_idx < node.keys.len() && node.keys[new_idx] == key {
                new_idx + 1
            } else {
                new_idx
            };
            let target_page_id = node.children[target_child];

            // Recurse — the child returns its new (CoW) page id. Patch
            // the parent's children pointer with the cascaded id before
            // the parent itself is saved.
            let new_target_id =
                self.insert_non_full_serial(target_page_id, key, value, depth_remaining - 1)?;
            if new_target_id != target_page_id {
                node.children[target_child] = new_target_id;
            }

            // Save this node CoW so the entire ancestor chain is fresh.
            self.save_node_cow(node)
        }
    }

    /// Split child node (serialized - caller holds write_lock).
    ///
    /// **CoW**: both halves of the split get fresh page ids via
    /// `save_node_cow`. The parent is mutated in memory (not saved
    /// here) — it stores `[new_left_id, new_right_id]` at
    /// `children[child_idx], children[child_idx + 1]`. The caller is
    /// responsible for CoW-saving the parent after this returns.
    fn split_child_serial(&self, parent: &mut BTreeNode, child_idx: usize) -> Result<(), Error> {
        let child_id = parent.children[child_idx];
        let mut child = self.take_node_for_write(child_id)?;

        // We pass a placeholder page id (0) into BTreeNode::split — the
        // returned right_node carries that placeholder until save_node_cow
        // retags it with a freshly allocated id below. No persistence
        // ever happens at id 0 since save_node_cow allocates first.
        let (right_node, median_key) = child.split(0);

        let new_left_id = self.save_node_cow(child)?;
        let new_right_id = self.save_node_cow(right_node)?;

        parent.keys.insert(child_idx, median_key);
        parent.children[child_idx] = new_left_id;
        parent.children.insert(child_idx + 1, new_right_id);

        self.structure_version.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Split root node (serialized - caller holds write_lock).
    ///
    /// **CoW**: every node born of this split gets a fresh page id via
    /// `save_node_cow` — including the demoted old root, the new
    /// right sibling, and the new root itself. The placeholder id `0`
    /// passed to `BTreeNode::split` and `BTreeNode::new_internal` is
    /// never persisted; `save_node_cow` retags before insertion.
    fn split_root(&self) -> Result<(), Error> {
        let old_root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let mut old_root = self.take_node_for_write(old_root_id)?;

        let (right_node, median_key) = old_root.split(0);

        let new_left_id = self.save_node_cow(old_root)?;
        let new_right_id = self.save_node_cow(right_node)?;

        let mut new_root = BTreeNode::new_internal(0);
        new_root.keys.push(median_key);
        new_root.children.push(new_left_id);
        new_root.children.push(new_right_id);

        let new_root_id = self.save_node_cow(new_root)?;

        *self
            .root_page_id
            .write()
            .expect("lock poisoned: root_page_id write") = new_root_id;
        self.tree_height.fetch_add(1, Ordering::Release);
        self.structure_version.fetch_add(1, Ordering::Release);

        // Persist new root to metadata
        self.update_metadata()?;

        Ok(())
    }

    /// Save a node to cache (write-back). Serialization deferred until flush_dirty_nodes().
    fn save_node_direct(&self, node: BTreeNode) -> Result<(), Error> {
        let page_id = node.page_id;
        self.node_cache.insert(
            page_id,
            CacheEntry {
                node: Arc::new(node),
                dirty: true,
            },
        );
        Ok(())
    }

    /// **Copy-on-Write save.** Allocate a fresh page id, retag the node onto
    /// it, serialise its bytes into the new buffer-pool page, and insert
    /// the node into the engine's deserialised cache. Returns the new
    /// page id so the caller can rewrite parent pointers.
    ///
    /// **Why we serialise eagerly here** rather than deferring to
    /// `flush_dirty_nodes`: the buffer pool's `new_page` seeds the
    /// page with `EMPTY_LEAF_BODY` so an unfortunate eviction doesn't
    /// flush garbage to disk. But the empty placeholder is itself a
    /// valid (zero-key) leaf, so a snapshot reader who hits the
    /// post-eviction disk version sees "no keys" — silent data loss
    /// instead of a corruption error. Under heavy CoW write churn,
    /// the buffer pool can evict newly-created pages before the next
    /// `flush_dirty_nodes` has populated them. Serialising at the
    /// `save_node_cow` site means evictable pages always carry their
    /// real content; `flush_dirty_nodes` becomes a no-op for these
    /// entries (`dirty: false` in the engine's cache, but the page
    /// itself is dirty in the buffer pool until `flush_all` writes it
    /// to disk).
    ///
    /// Caller contract:
    /// 1. Caller must update *every* pointer in the parent that referred
    ///    to `node.page_id` so it now refers to the returned id.
    /// 2. The old page id is **not** added to the free list here — that
    ///    happens at commit time once we know the cascade is durable
    ///    (Phase 2).
    /// 3. Pre-existing entries in `node_cache` for the old page id are
    ///    left untouched. Readers holding a snapshot at the old version
    ///    can still resolve the old page id.
    pub(crate) fn save_node_cow(&self, mut node: BTreeNode) -> Result<PageId, Error> {
        // Allocate a fresh page id via the buffer pool. The pool seeds the
        // backing page with EMPTY_LEAF_BODY; we overwrite that with the
        // real serialised bytes immediately below.
        let new_page_arc = self.buffer_pool.new_page()?;
        let new_page_id = new_page_arc
            .read()
            .expect("lock poisoned: cow new page read")
            .id;
        self.track_new_page(new_page_id);

        node.page_id = new_page_id;

        // Serialise the node directly into the buffer-pool page. This
        // closes a race where eviction would otherwise flush the
        // EMPTY_LEAF_BODY placeholder to disk before `flush_dirty_nodes`
        // gets a chance to populate it (silent data loss for any
        // snapshot reader landing on the eviction-flushed page id).
        // `serialize_node_with_overflow` may itself allocate overflow
        // pages — those are fully owned + written here, no further
        // flush_dirty_nodes work needed for them either.
        let data = self.serialize_node_with_overflow(&node)?;
        let page_type = if node.is_leaf {
            PageType::BTreeLeaf
        } else {
            PageType::BTreeInternal
        };
        {
            let mut page = new_page_arc
                .write()
                .expect("lock poisoned: cow new page write");
            page.data = data;
            page.page_type = page_type;
            page.mark_dirty();
        }

        // Insert into the engine's deserialised cache. `dirty: false`
        // because the buffer-pool page is already in sync with the
        // node — `flush_dirty_nodes` will skip this entry. The
        // buffer-pool page itself is still dirty until `flush_all`
        // writes it to disk.
        self.node_cache.insert(
            new_page_id,
            CacheEntry {
                node: Arc::new(node),
                dirty: false,
            },
        );

        Ok(new_page_id)
    }

    fn delete_locked(&self, key: &[u8]) -> Result<bool, Error> {
        // Assumes write_lock is already held
        // let _write_guard = self.write_lock.write().unwrap();

        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");

        // CoW cascade: delete returns the (possibly new) page id of the
        // subtree's root after every node touched on the path was
        // rewritten via save_node_cow.
        // Depth bound — same cycle protection as the insert path.
        let max_depth = self.tree_height.load(Ordering::Acquire) as usize + 16;
        let (deleted, _, new_root_id) = self.delete_recursive(root_id, key, max_depth)?;
        if new_root_id != root_id {
            *self
                .root_page_id
                .write()
                .expect("lock poisoned: root_page_id write") = new_root_id;
            self.structure_version.fetch_add(1, Ordering::Release);
            self.update_metadata()?;
        }

        // After possible merging, the root itself may collapse: an
        // internal root with zero keys and one remaining child shrinks
        // the tree by one level. Read at the *new* root id.
        let root_id = *self
            .root_page_id
            .read()
            .expect("lock poisoned: root_page_id read");
        let root = self.load_node(root_id)?;
        if !root.is_leaf && root.keys.is_empty() && root.children.len() == 1 {
            let new_root_id = root.children[0];
            *self
                .root_page_id
                .write()
                .expect("lock poisoned: root_page_id write") = new_root_id;
            self.tree_height.fetch_sub(1, Ordering::Release);
            self.structure_version.fetch_add(1, Ordering::Release);
            self.update_metadata()?;
        }

        Ok(deleted)
    }

    /// Recursively delete a key.
    ///
    /// Returns `(was_deleted, is_underflow, new_self_page_id)`.
    /// **CoW**: every node visited on the delete path is rewritten via
    /// `save_node_cow`. The returned `new_self_page_id` is what the
    /// caller (the parent) must store in its `children[child_idx]` slot
    /// before saving itself CoW. See `docs/joule-db/cow-mvcc-design.md`.
    fn delete_recursive(
        &self,
        page_id: PageId,
        key: &[u8],
        depth_remaining: usize,
    ) -> Result<(bool, bool, PageId), Error> {
        if depth_remaining == 0 {
            return Err(Error::Storage(StorageError::Corrupted {
                page_id,
                reason: "delete_recursive: depth bound exceeded — likely a \
                         cycle in the tree's internal child pointers"
                    .into(),
            }));
        }
        let node_arc = self.load_node(page_id)?;
        let idx = node_arc.find_key_index(key);

        if node_arc.is_leaf {
            if idx < node_arc.keys.len() && node_arc.keys[idx] == key {
                drop(node_arc);
                let mut node = self.take_node_for_write(page_id)?;
                let idx = node.find_key_index(key);

                node.keys.remove(idx);
                node.values.remove(idx);

                // Byte-size underflow check — see is_underflowing_by_size
                // for why we no longer use count-based MIN_KEYS here.
                let underflow = node.is_underflowing_by_size(self.max_page_data);
                let new_id = self.save_node_cow(node)?;
                Ok((true, underflow, new_id))
            } else {
                // No mutation occurred on this leaf, so the page id is
                // unchanged. The caller's parent pointer remains valid.
                Ok((false, false, page_id))
            }
        } else {
            // Internal node: determine which child to descend into
            let child_idx = if idx < node_arc.keys.len() && node_arc.keys[idx] == key {
                // Key found in internal node - this shouldn't happen in our design
                // (only leaves have values), so we descend to right child
                idx + 1
            } else {
                idx
            };

            let child_id = node_arc.children[child_idx];
            drop(node_arc);

            // Recursively delete from child — child may return a new id.
            let (deleted, child_underflow, new_child_id) =
                self.delete_recursive(child_id, key, depth_remaining - 1)?;

            // If nothing actually changed (leaf no-op for an absent key
            // plus no underflow), this node is unchanged and we can
            // skip the CoW rewrite.
            if !deleted && !child_underflow && new_child_id == child_id {
                return Ok((false, false, page_id));
            }

            // Take this node for write, patch the child pointer, save CoW.
            let mut node = self.take_node_for_write(page_id)?;
            if new_child_id != child_id {
                node.children[child_idx] = new_child_id;
            }
            let new_self_id = self.save_node_cow(node)?;

            // Handle underflow by rebalancing on the new node id. The
            // rebalance functions return the new id of the parent after
            // borrow / merge cascades.
            let new_self_id = if child_underflow {
                self.rebalance_after_delete(new_self_id, child_idx)?
            } else {
                new_self_id
            };

            // Check if this node now underflows. Byte-size criterion
            // — see is_underflowing_by_size.
            let node = self.load_node(new_self_id)?;
            let underflow = node.is_underflowing_by_size(self.max_page_data);

            Ok((deleted, underflow, new_self_id))
        }
    }

    /// Rebalance after a child underflows.
    ///
    /// **CoW**: returns the new page id of the parent after borrow /
    /// merge cascades. Caller must thread this id up to its own parent.
    /// Tries to:
    /// 1. Borrow from left sibling
    /// 2. Borrow from right sibling
    /// 3. Merge with a sibling
    fn rebalance_after_delete(
        &self,
        parent_id: PageId,
        child_idx: usize,
    ) -> Result<PageId, Error> {
        let parent = self.load_node(parent_id)?;
        let child_id = parent.children[child_idx];
        let child = self.load_node(child_id)?;

        // Skip if child doesn't actually underflow.
        // Byte-size criterion (see is_underflowing_by_size).
        if !child.is_underflowing_by_size(self.max_page_data) {
            return Ok(parent_id);
        }

        // Try to borrow from left sibling. The sibling must have
        // *bytes to spare* — borrowing from a same-sized neighbour
        // just shuffles the underflow around. Use byte-size again:
        // borrow only if the sibling is itself NOT underflowing
        // (i.e. has > 25% fill).
        if child_idx > 0 {
            let left_sibling_id = parent.children[child_idx - 1];
            let left_sibling = self.load_node(left_sibling_id)?;
            if !left_sibling.is_underflowing_by_size(self.max_page_data) {
                return self.borrow_from_left(parent_id, child_idx);
            }
        }

        // Try to borrow from right sibling
        if child_idx < parent.children.len() - 1 {
            let right_sibling_id = parent.children[child_idx + 1];
            let right_sibling = self.load_node(right_sibling_id)?;
            if !right_sibling.is_underflowing_by_size(self.max_page_data) {
                return self.borrow_from_right(parent_id, child_idx);
            }
        }

        // Try to merge — but only if the merged result fits in a page.
        // The historical bug was that merge always proceeded; with
        // byte-fat values (e.g. 2KB), merging two half-leaves could
        // produce a node larger than the page size. The safe-merge
        // check rejects that and falls through to "accept underflow",
        // which is suboptimal-tree-shape but never produces an
        // unwriteable page.
        let separator_for_left: Option<Vec<u8>> = if child_idx > 0 {
            Some(parent.keys[child_idx - 1].clone())
        } else {
            None
        };
        let separator_for_right: Option<Vec<u8>> = if child_idx < parent.keys.len() {
            Some(parent.keys[child_idx].clone())
        } else {
            None
        };

        if child_idx > 0 {
            let left_sibling_id = parent.children[child_idx - 1];
            let left_sibling = self.load_node(left_sibling_id)?;
            let separator = separator_for_left
                .as_deref()
                .expect("separator_for_left must exist when child_idx > 0");
            if left_sibling.can_merge_with(&child, separator, self.max_page_data) {
                drop(left_sibling);
                drop(child);
                drop(parent);
                return self.merge_with_left(parent_id, child_idx);
            }
        }
        if child_idx < parent.children.len() - 1 {
            let right_sibling_id = parent.children[child_idx + 1];
            let right_sibling = self.load_node(right_sibling_id)?;
            let separator = separator_for_right
                .as_deref()
                .expect("separator_for_right must exist when child_idx < len-1");
            if child.can_merge_with(&right_sibling, separator, self.max_page_data) {
                drop(right_sibling);
                drop(child);
                drop(parent);
                return self.merge_with_right(parent_id, child_idx);
            }
        }

        // Neither sibling is willing to lend, and neither merge would
        // fit. Accept the underflow — the tree shape is suboptimal
        // but every node is a valid page. Future inserts can refill
        // this leaf; future merges may become feasible if siblings
        // shrink further.
        Ok(parent_id)
    }

    /// Borrow one key from the left sibling.
    ///
    /// **CoW**: parent, left sibling, and child all get fresh page ids.
    /// Returns the new parent id so the caller can cascade upward.
    fn borrow_from_left(&self, parent_id: PageId, child_idx: usize) -> Result<PageId, Error> {
        let mut parent = self.take_node_for_write(parent_id)?;
        let left_id = parent.children[child_idx - 1];
        let child_id = parent.children[child_idx];
        let mut left = self.take_node_for_write(left_id)?;
        let mut child = self.take_node_for_write(child_id)?;

        if child.is_leaf {
            // For LEAF nodes: move key+value from left to child, update separator
            let borrowed_key = left.keys.pop().expect("left sibling guaranteed non-empty");
            let borrowed_value = left
                .values
                .pop()
                .expect("left sibling values guaranteed non-empty");

            // Insert at beginning of child
            child.keys.insert(0, borrowed_key);
            child.values.insert(0, borrowed_value);

            // Update separator to be the new smallest key in child
            parent.keys[child_idx - 1] = child.keys[0].clone();
        } else {
            // For INTERNAL nodes: rotate through parent
            let parent_key = parent.keys.remove(child_idx - 1);
            child.keys.insert(0, parent_key);

            let borrowed_key = left.keys.pop().expect("left sibling guaranteed non-empty");
            parent.keys.insert(child_idx - 1, borrowed_key);

            let borrowed_child = left
                .children
                .pop()
                .expect("left sibling children guaranteed non-empty");
            child.children.insert(0, borrowed_child);
        }

        // Save the two siblings CoW first; their new ids feed back into
        // the parent's children slots before parent itself is saved CoW.
        let new_left_id = self.save_node_cow(left)?;
        let new_child_id = self.save_node_cow(child)?;
        parent.children[child_idx - 1] = new_left_id;
        parent.children[child_idx] = new_child_id;
        self.save_node_cow(parent)
    }

    /// Borrow one key from the right sibling.
    ///
    /// **CoW**: parent, child, and right sibling all get fresh page ids.
    /// Returns the new parent id so the caller can cascade upward.
    fn borrow_from_right(&self, parent_id: PageId, child_idx: usize) -> Result<PageId, Error> {
        let mut parent = self.take_node_for_write(parent_id)?;
        let child_id = parent.children[child_idx];
        let right_id = parent.children[child_idx + 1];
        let mut child = self.take_node_for_write(child_id)?;
        let mut right = self.take_node_for_write(right_id)?;

        if child.is_leaf {
            // For LEAF nodes: move key+value from right to child, update separator
            let borrowed_key = right.keys.remove(0);
            let borrowed_value = right.values.remove(0);

            child.keys.push(borrowed_key);
            child.values.push(borrowed_value);

            parent.keys[child_idx] = right.keys[0].clone();
        } else {
            // For INTERNAL nodes: rotate through parent
            let parent_key = parent.keys.remove(child_idx);
            child.keys.push(parent_key);

            let borrowed_key = right.keys.remove(0);
            parent.keys.insert(child_idx, borrowed_key);

            let borrowed_child = right.children.remove(0);
            child.children.push(borrowed_child);
        }

        let new_child_id = self.save_node_cow(child)?;
        let new_right_id = self.save_node_cow(right)?;
        parent.children[child_idx] = new_child_id;
        parent.children[child_idx + 1] = new_right_id;
        self.save_node_cow(parent)
    }

    /// Merge child with its left sibling.
    ///
    /// **CoW**: parent and the surviving left sibling both get fresh
    /// page ids; the absorbed child page id is released to the buffer
    /// pool's free list (Phase 4 will gate this on min-live-snapshot).
    /// Returns the new parent id so the caller can cascade upward.
    fn merge_with_left(&self, parent_id: PageId, child_idx: usize) -> Result<PageId, Error> {
        let mut parent = self.take_node_for_write(parent_id)?;
        let left_id = parent.children[child_idx - 1];
        let child_id = parent.children[child_idx];
        let mut left = self.take_node_for_write(left_id)?;
        let child = self.take_node_for_write(child_id)?;

        // Pull down the separator key from parent
        let separator = parent.keys.remove(child_idx - 1);
        parent.children.remove(child_idx);

        // Add separator to left (for internal nodes)
        if !left.is_leaf {
            left.keys.push(separator);
        }

        // Move all keys and values/children from child to left
        left.keys.extend(child.keys);
        if left.is_leaf {
            left.values.extend(child.values);
        } else {
            left.children.extend(child.children);
        }

        let new_left_id = self.save_node_cow(left)?;
        // After remove(child_idx), the slot at child_idx - 1 still maps
        // to `left` — patch it with the new id before saving the parent.
        parent.children[child_idx - 1] = new_left_id;

        // Free the absorbed child's page id. Phase 3 defers this if
        // any snapshot is live (so a snapshot reader can't observe the
        // page reused). Phase 4 will replace with a version-gated
        // free list keyed by min-live-snapshot.
        self.defer_free_page(child_id)?;

        self.save_node_cow(parent)
    }

    /// Merge child with its right sibling.
    ///
    /// **CoW**: parent and the surviving child both get fresh page ids;
    /// the absorbed right sibling's page id is released. Returns the
    /// new parent id so the caller can cascade upward.
    fn merge_with_right(&self, parent_id: PageId, child_idx: usize) -> Result<PageId, Error> {
        let mut parent = self.take_node_for_write(parent_id)?;
        let child_id = parent.children[child_idx];
        let right_id = parent.children[child_idx + 1];
        let mut child = self.take_node_for_write(child_id)?;
        let right = self.take_node_for_write(right_id)?;

        // Pull down the separator key from parent
        let separator = parent.keys.remove(child_idx);
        parent.children.remove(child_idx + 1);

        // Add separator to child (for internal nodes)
        if !child.is_leaf {
            child.keys.push(separator);
        }

        // Move all keys and values/children from right to child
        child.keys.extend(right.keys);
        if child.is_leaf {
            child.values.extend(right.values);
        } else {
            child.children.extend(right.children);
        }

        let new_child_id = self.save_node_cow(child)?;
        parent.children[child_idx] = new_child_id;
        self.structure_version.fetch_add(1, Ordering::Release);

        // Phase 3: snapshot-aware deferral — see fn defer_free_page.
        self.defer_free_page(right_id)?;

        self.save_node_cow(parent)
    }

    /// Load a node from cache or storage
    ///
    /// Returns an Arc to the node, allowing zero-copy sharing between
    /// concurrent readers. The buffer pool handles LRU eviction automatically.
    ///
    /// Note: This uses cache opportunistically. For operations that modify nodes,
    /// use `load_node_fresh` to ensure you have the latest version.
    /// Load a node from storage (via buffer pool)
    ///
    /// Returns an Arc to the node. Note that with the new buffer pool,
    /// we deserialize on every access. In a production system we'd want
    /// a layer that caches deserialized nodes.
    fn load_node(&self, page_id: PageId) -> Result<Arc<BTreeNode>, Error> {
        // Fast path: DashMap per-shard read lock
        if let Some(entry) = self.node_cache.get(&page_id) {
            return Ok(Arc::clone(&entry.node));
        }

        // Slow path: deserialize from buffer pool page with overflow
        // resolution. Both failure modes — `Corrupted` from
        // `Page::decode` and parse errors from
        // `deserialize_node_with_overflow` — propagate as engine
        // errors. The repair path is operator-driven: archive +
        // re-ingest from authoritative sources (see
        // `project_scholar_reingest_repair_2026_05_03.md`); the
        // engine no longer silently substitutes empty leaves.
        let page_arc = self.buffer_pool.get_page(page_id).map_err(Error::Storage)?;
        let page = page_arc.read().expect("lock poisoned: page read");
        let node = Arc::new(self.deserialize_node_with_overflow(page_id, &page.data)?);
        drop(page);

        // Insert into cache (another thread may have beaten us)
        self.node_cache.entry(page_id).or_insert(CacheEntry {
            node: Arc::clone(&node),
            dirty: false,
        });

        Ok(node)
    }

    /// Take a node for writing: removes from cache and returns owned node.
    /// Uses Arc::try_unwrap for zero-copy when possible, falls back to clone.
    fn take_node_for_write(&self, page_id: PageId) -> Result<BTreeNode, Error> {
        if let Some((_, entry)) = self.node_cache.remove(&page_id) {
            match Arc::try_unwrap(entry.node) {
                Ok(node) => Ok(node),
                Err(arc) => Ok((*arc).clone()),
            }
        } else {
            // Same error policy as `load_node`: `Corrupted` and parse
            // errors propagate so the operator runs the archive +
            // re-ingest playbook rather than silently writing over
            // an empty-leaf placeholder.
            let page_arc = self.buffer_pool.get_page(page_id).map_err(Error::Storage)?;
            let page = page_arc.read().expect("lock poisoned: page read");
            self.deserialize_node_with_overflow(page_id, &page.data)
        }
    }

    /// Load a node fresh - same as load_node now
    pub fn load_node_fresh(&self, page_id: PageId) -> Result<Arc<BTreeNode>, Error> {
        self.load_node(page_id)
    }

    /// Internal: Load from storage - same as load_node now
    fn load_node_from_storage(&self, page_id: PageId) -> Result<Arc<BTreeNode>, Error> {
        self.load_node(page_id)
    }

    /// Save a node to storage and update cache
    ///
    /// This is a convenience method that wraps `save_node_direct`.
    pub fn save_node(&self, node: BTreeNode) -> Result<(), Error> {
        self.save_node_direct(node)
    }

    /// Get cache statistics
    pub fn cache_size(&self) -> usize {
        self.buffer_pool.cache_size()
    }

    /// Get number of active page latches
    pub fn active_latches(&self) -> usize {
        self.page_latches.active_latches()
    }

    /// Clear the node cache (useful for testing)
    pub fn clear_cache(&self) {
        let _ = self.flush_dirty_nodes();
        self.buffer_pool.clear();
        self.node_cache.clear();
    }

    /// Release latch for a freed page (call after free_page)
    pub fn release_page_latch(&self, page_id: PageId) {
        self.page_latches.release_page(page_id);
    }
}

// Implement Index trait for Engine

impl Index for Engine {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        Engine::get(self, key)
            .map_err(|e| IndexError::Storage(StorageError::Backend(e.to_string())))
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        Engine::put(self, key, value)
            .map_err(|e| IndexError::Storage(StorageError::Backend(e.to_string())))
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        Engine::delete(self, key)
            .map_err(|e| IndexError::Storage(StorageError::Backend(e.to_string())))
    }

    fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        Ok(Box::new(BTreeRangeIterator::new(
            self, start, end, direction,
        )))
    }
}

// ============================================================================
// Range Iterator Implementation
// ============================================================================

/// Range iterator for B-tree
///
/// Performs in-order traversal of the B-tree within the specified bounds.
/// Uses a stack-based approach for lazy evaluation.
pub struct BTreeRangeIterator<'a> {
    engine: &'a Engine,
    /// **CoW MVCC Phase 6.** Optional pinned root for snapshot reads.
    /// `None` means "use the engine's current `root_page_id`" (the
    /// classic mutable-root iteration); `Some(id)` means "traverse
    /// from this root only" (a snapshot-stable view).
    root_override: Option<PageId>,
    /// Stack of (page_id, key_index) for traversal
    /// Key index points to the next key to return in that node
    stack: Vec<(PageId, usize)>,
    /// Start bound (owned for lifetime simplicity)
    start_bound: Bound<Vec<u8>>,
    /// End bound (owned)
    end_bound: Bound<Vec<u8>>,
    /// Scan direction
    direction: ScanDirection,
    /// Whether we've initialized the starting position
    initialized: bool,
    /// Whether iteration is complete
    finished: bool,
    /// Cached current node to avoid re-loading when iterating within the same node
    cached_node: Option<(PageId, Arc<BTreeNode>)>,
}

impl<'a> BTreeRangeIterator<'a> {
    /// Create a new range iterator
    fn new(
        engine: &'a Engine,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Self {
        Self::new_with_root(engine, None, start, end, direction)
    }

    /// Create a new range iterator pinned to a specific root page id.
    /// Used by `Engine::range_at_root` for snapshot reads — the
    /// iterator will descend from `root` rather than reading the
    /// engine's current `root_page_id`.
    fn new_with_root(
        engine: &'a Engine,
        root_override: Option<PageId>,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Self {
        // Convert to owned bounds
        let start_bound = match start {
            Bound::Included(k) => Bound::Included(k.to_vec()),
            Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end_bound = match end {
            Bound::Included(k) => Bound::Included(k.to_vec()),
            Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
            Bound::Unbounded => Bound::Unbounded,
        };

        Self {
            engine,
            root_override,
            stack: Vec::new(),
            start_bound,
            end_bound,
            direction,
            initialized: false,
            finished: false,
            cached_node: None,
        }
    }

    /// Check if a key is within the start bound
    fn is_after_start(&self, key: &[u8]) -> bool {
        match &self.start_bound {
            Bound::Unbounded => true,
            Bound::Included(start) => key >= start.as_slice(),
            Bound::Excluded(start) => key > start.as_slice(),
        }
    }

    /// Check if a key is within the end bound
    fn is_before_end(&self, key: &[u8]) -> bool {
        match &self.end_bound {
            Bound::Unbounded => true,
            Bound::Included(end) => key <= end.as_slice(),
            Bound::Excluded(end) => key < end.as_slice(),
        }
    }

    /// Check if a key is within both bounds
    fn is_in_range(&self, key: &[u8]) -> bool {
        self.is_after_start(key) && self.is_before_end(key)
    }

    /// Load a node, using the local cache if the page_id matches
    fn get_node(&mut self, page_id: PageId) -> Result<Arc<BTreeNode>, IndexError> {
        if let Some((cached_id, ref node)) = self.cached_node {
            if cached_id == page_id {
                return Ok(Arc::clone(node));
            }
        }
        let node = self
            .engine
            .load_node(page_id)
            .map_err(|e| IndexError::Storage(StorageError::Backend(e.to_string())))?;
        self.cached_node = Some((page_id, Arc::clone(&node)));
        Ok(node)
    }

    /// Initialize the iterator by finding the starting position
    fn initialize_forward(&mut self) -> Result<(), IndexError> {
        let root_id = self.root_override.unwrap_or_else(|| {
            *self
                .engine
                .root_page_id
                .read()
                .expect("lock poisoned: root_page_id read")
        });
        self.descend_to_start_forward(root_id)?;
        self.initialized = true;
        Ok(())
    }

    /// Descend to the starting position for forward iteration.
    ///
    /// Iterative with a depth bound — same protection as
    /// `descend_to_leftmost` (see that function's note for the
    /// production incident this guards against). Sets up the stack so
    /// that `next_forward()` will return the first entry >= start
    /// bound.
    fn descend_to_start_forward(&mut self, mut page_id: PageId) -> Result<(), IndexError> {
        let max_depth = self.engine.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let node = self.get_node(page_id)?;

            if node.is_leaf {
                let start_idx = match &self.start_bound {
                    Bound::Unbounded => 0,
                    Bound::Included(start) => node
                        .keys
                        .iter()
                        .position(|k| k.as_slice() >= start.as_slice())
                        .unwrap_or(node.keys.len()),
                    Bound::Excluded(start) => node
                        .keys
                        .iter()
                        .position(|k| k.as_slice() > start.as_slice())
                        .unwrap_or(node.keys.len()),
                };
                self.stack.push((page_id, start_idx));
                return Ok(());
            }

            // Internal node.
            let child_idx = match &self.start_bound {
                Bound::Unbounded => 0,
                Bound::Included(start) | Bound::Excluded(start) => {
                    node.find_key_index(start)
                }
            };
            // Push this internal node's position so next_forward can
            // continue to the next child after we exit this subtree.
            self.stack.push((page_id, child_idx + 1));

            if child_idx >= node.children.len() {
                return Ok(());
            }
            page_id = node.children[child_idx];
        }
        Err(IndexError::Corrupted {
            reason: format!(
                "descend_to_start_forward: exceeded depth {max_depth} \
                 starting from page {page_id} — likely a corrupt \
                 internal page chain",
            ),
        })
    }

    /// Get the next entry in forward direction
    ///
    /// For B-trees where only leaf nodes have values:
    /// - Internal nodes are purely for navigation
    /// - We iterate through leaf nodes in order, using stack for backtracking
    fn next_forward(&mut self) -> Option<Result<IndexEntry, IndexError>> {
        while let Some((page_id, idx)) = self.stack.pop() {
            let node = match self.get_node(page_id) {
                Ok(n) => n,
                Err(e) => {
                    // Stop the iterator on a get_node failure — the stack
                    // state may include sibling pointers that also fail
                    // and re-firing this branch would loop forever
                    // (production 2026-05-07: corrupt internal page in
                    // scholar_entities subtree caused a 30+ minute spin
                    // with stack growth on every caller `for` step).
                    self.finished = true;
                    return Some(Err(e));
                }
            };

            if node.is_leaf {
                // Leaf node - return entries
                if idx < node.keys.len() {
                    let key = &node.keys[idx];

                    // Check if we've passed the end bound
                    if !self.is_before_end(key) {
                        self.finished = true;
                        return None;
                    }

                    // Push next position for this leaf
                    self.stack.push((page_id, idx + 1));

                    // Check if key is in range
                    if self.is_in_range(key) {
                        if let Some(value) = &node.values[idx] {
                            return Some(Ok(IndexEntry::new(key.clone(), value.clone())));
                        }
                        // Value is None (deleted), continue to next
                    }
                }
                // Leaf exhausted, continue popping stack to find next leaf
            } else {
                // Internal node - use for navigation only
                // idx represents which child we should descend into next
                if idx < node.children.len() {
                    // Push position for next child
                    if idx + 1 <= node.children.len() {
                        self.stack.push((page_id, idx + 1));
                    }

                    // Descend into current child. On error, terminate
                    // the iterator — leaving the stack populated would
                    // cause the next caller iteration to re-enter the
                    // same broken subtree.
                    if let Err(e) = self.descend_to_leftmost(node.children[idx]) {
                        self.finished = true;
                        return Some(Err(e));
                    }
                }
                // If idx >= children.len(), this internal node is exhausted
            }
        }

        self.finished = true;
        None
    }

    /// Descend to the leftmost leaf from a given node.
    ///
    /// Iterative (not recursive) with a depth bound: the engine's
    /// known `tree_height` plus a generous slack. A corrupt internal
    /// page that points back at itself or one of its ancestors would
    /// otherwise loop here forever (production 2026-05-07: scholar-
    /// server's row-count refresher pinned a CPU at 100 % for 18+ min
    /// in this very function, never returning). Return
    /// `IndexError::Corrupted` on overflow so the caller learns and
    /// can decide what to do — e.g. log, partial-count and continue,
    /// or abort.
    fn descend_to_leftmost(&mut self, mut page_id: PageId) -> Result<(), IndexError> {
        let max_depth = self.engine.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let node = self.get_node(page_id)?;
            self.stack.push((page_id, 0));
            if node.is_leaf || node.children.is_empty() {
                return Ok(());
            }
            page_id = node.children[0];
        }
        Err(IndexError::Corrupted {
            reason: format!(
                "descend_to_leftmost: exceeded depth {max_depth} \
                 starting from page {page_id} — likely a corrupt \
                 internal page with a cycle in its leftmost child \
                 chain",
            ),
        })
    }

    /// Initialize backward iteration
    fn initialize_backward(&mut self) -> Result<(), IndexError> {
        let root_id = self.root_override.unwrap_or_else(|| {
            *self
                .engine
                .root_page_id
                .read()
                .expect("lock poisoned: root_page_id read")
        });
        self.descend_to_start_backward(root_id)?;
        self.initialized = true;
        Ok(())
    }

    /// Descend to starting position for backward iteration.
    ///
    /// Iterative with a depth bound — see `descend_to_leftmost` for
    /// rationale.
    fn descend_to_start_backward(&mut self, mut page_id: PageId) -> Result<(), IndexError> {
        let max_depth = self.engine.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let node = self.get_node(page_id)?;

            if node.is_leaf {
                let start_idx = match &self.end_bound {
                    Bound::Unbounded => node.keys.len().saturating_sub(1),
                    Bound::Included(end) => node
                        .keys
                        .iter()
                        .rposition(|k| k.as_slice() <= end.as_slice())
                        .unwrap_or(usize::MAX),
                    Bound::Excluded(end) => node
                        .keys
                        .iter()
                        .rposition(|k| k.as_slice() < end.as_slice())
                        .unwrap_or(usize::MAX),
                };

                if start_idx != usize::MAX {
                    self.stack.push((page_id, start_idx));
                }
                return Ok(());
            }

            // Internal node.
            let child_idx = match &self.end_bound {
                Bound::Unbounded => node.children.len() - 1,
                Bound::Included(end) | Bound::Excluded(end) => {
                    let idx = node.find_key_index(end);
                    idx.min(node.children.len() - 1)
                }
            };
            self.stack.push((page_id, child_idx));

            if child_idx >= node.children.len() {
                return Ok(());
            }
            page_id = node.children[child_idx];
        }
        Err(IndexError::Corrupted {
            reason: format!(
                "descend_to_start_backward: exceeded depth {max_depth} \
                 starting from page {page_id} — likely a corrupt \
                 internal page chain",
            ),
        })
    }

    /// Get the next entry in backward direction
    fn next_backward(&mut self) -> Option<Result<IndexEntry, IndexError>> {
        while let Some((page_id, idx)) = self.stack.pop() {
            // Handle usize::MAX (invalid position)
            if idx == usize::MAX {
                continue;
            }

            let node = match self.get_node(page_id) {
                Ok(n) => n,
                Err(e) => {
                    // Same termination semantics as next_forward — see
                    // that function for context.
                    self.finished = true;
                    return Some(Err(e));
                }
            };

            if node.is_leaf {
                let key = &node.keys[idx];

                // Check if we've passed the start bound
                if !self.is_after_start(key) {
                    self.finished = true;
                    return None;
                }

                // Push previous position
                if idx > 0 {
                    self.stack.push((page_id, idx - 1));
                }

                // Check if key is in range
                if self.is_in_range(key) {
                    if let Some(value) = &node.values[idx] {
                        return Some(Ok(IndexEntry::new(key.clone(), value.clone())));
                    }
                }
            } else {
                // Internal node - backward traversal is more complex
                // For simplicity, collect and reverse (less efficient but correct)
                // A proper implementation would maintain proper backward state
                if idx > 0 {
                    self.stack.push((page_id, idx - 1));
                }

                if idx < node.children.len() {
                    if let Err(e) = self.descend_to_rightmost(node.children[idx]) {
                        self.finished = true;
                        return Some(Err(e));
                    }
                }
            }
        }

        self.finished = true;
        None
    }

    /// Descend to the rightmost leaf from a given node.
    ///
    /// Iterative with a depth bound — see `descend_to_leftmost` for
    /// rationale.
    fn descend_to_rightmost(&mut self, mut page_id: PageId) -> Result<(), IndexError> {
        let max_depth = self.engine.tree_height.load(Ordering::Acquire) as usize + 16;
        for _ in 0..max_depth {
            let node = self.get_node(page_id)?;

            let idx = if node.is_leaf {
                node.keys.len().saturating_sub(1)
            } else {
                node.children.len().saturating_sub(1)
            };

            if idx != usize::MAX || !node.keys.is_empty() {
                self.stack.push((page_id, idx));
            }

            if node.is_leaf || node.children.is_empty() {
                return Ok(());
            }
            page_id = node.children[node.children.len() - 1];
        }
        Err(IndexError::Corrupted {
            reason: format!(
                "descend_to_rightmost: exceeded depth {max_depth} \
                 starting from page {page_id} — likely a corrupt \
                 internal page chain",
            ),
        })
    }
}

impl<'a> IndexIterator for BTreeRangeIterator<'a> {}

impl<'a> Iterator for BTreeRangeIterator<'a> {
    type Item = Result<IndexEntry, IndexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Initialize on first call
        if !self.initialized {
            let result = match self.direction {
                ScanDirection::Forward => self.initialize_forward(),
                ScanDirection::Backward => self.initialize_backward(),
            };
            if let Err(e) = result {
                self.finished = true;
                return Some(Err(e));
            }
        }

        match self.direction {
            ScanDirection::Forward => self.next_forward(),
            ScanDirection::Backward => self.next_backward(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryBackend;

    /// **Corruption-error propagation contract.** Reading a page that
    /// `Page::decode` rejects (bad magic, wrong size, etc.) must
    /// surface `StorageError::Corrupted` to the caller; the engine
    /// must not substitute an empty leaf or any other silent
    /// placeholder. The repair playbook is operator-driven (archive
    /// + re-ingest from authoritative sources). Silent empty-leaf
    /// fallback was removed 2026-05-03 because the placeholder could
    /// be marked dirty by subsequent writes and flushed back to
    /// disk, overwriting the only forensically recoverable bytes.
    #[test]
    fn corruption_propagates_to_caller() {
        use crate::error::StorageError;
        use crate::storage::page::PageType;
        use crate::storage::{Page, PageId, StorageBackend};

        struct CorruptingBackend {
            inner: MemoryBackend,
            poisoned: PageId,
        }
        impl StorageBackend for CorruptingBackend {
            fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
                if page_id == self.poisoned {
                    return Err(StorageError::Corrupted {
                        page_id,
                        reason: "synthetic test corruption".into(),
                    });
                }
                self.inner.read_page(page_id)
            }
            fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
                self.inner.write_page(page)
            }
            fn allocate_page(&mut self) -> Result<PageId, StorageError> {
                self.inner.allocate_page()
            }
            fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
                self.inner.free_page(page_id)
            }
            fn sync(&mut self) -> Result<(), StorageError> {
                self.inner.sync()
            }
        }

        let backend = CorruptingBackend {
            inner: MemoryBackend::new(),
            poisoned: 999_999,
        };
        let engine = Engine::new(backend).unwrap();
        engine.put(b"alpha", b"value").unwrap();

        // load_node on the poisoned id must error, not silently
        // return an empty leaf.
        match engine.load_node(999_999) {
            Err(Error::Storage(StorageError::Corrupted { page_id, .. })) => {
                assert_eq!(page_id, 999_999);
            }
            Err(other) => panic!("expected Corrupted error, got {other:?}"),
            Ok(_) => panic!("load_node on a corrupt page must NOT silently return an empty leaf"),
        }

        // And the in-memory body-level decode failure path
        // (`deserialize_node_with_overflow` "Empty node data") must
        // propagate the same way.
        let arc = engine.buffer_pool.new_page().unwrap();
        let corrupt_id = arc.read().expect("lock poisoned").id;
        {
            let mut p = arc.write().expect("lock poisoned");
            *p = Page::with_data(corrupt_id, PageType::BTreeLeaf, Vec::new());
            p.data.clear();
            p.mark_dirty();
        }
        assert!(
            engine.load_node(corrupt_id).is_err(),
            "decode-failure load must propagate as an error"
        );
    }

    /// Phase 1.1 of the CoW MVCC refactor. Foundational primitive test:
    /// `save_node_cow` must produce a fresh page id distinct from the
    /// node's prior id, and the modified node must be reachable at the
    /// new id while the cache entry at the old id remains untouched.
    ///
    /// "Old id stays untouched" is the in-memory analogue of "readers
    /// holding a snapshot at the old version still see the old root."
    /// At Phase 1 only the cache invariant is testable; on-disk
    /// snapshot semantics arrive in Phase 2 with the meta-page atomic
    /// root swap. See `docs/joule-db/cow-mvcc-design.md`.
    #[test]
    fn save_node_cow_assigns_fresh_page_id() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Seed one key so we have a non-empty root with predictable contents.
        engine.put(b"alpha", b"1").unwrap();

        let original_root_id = engine.root_page_id();

        // Take the root for a CoW write — modify, then save_node_cow.
        let mut node = engine.take_node_for_write(original_root_id).unwrap();
        let original_page_id = node.page_id;
        assert_eq!(original_page_id, original_root_id);
        assert!(
            node.keys.iter().any(|k| k == b"alpha"),
            "seed key missing from root"
        );

        // Mutate: append a new key to the leaf root.
        node.keys.push(b"beta".to_vec());
        node.values.push(Some(b"2".to_vec()));

        let new_page_id = engine.save_node_cow(node).unwrap();

        // Property 1: fresh page id, distinct from original.
        assert_ne!(
            new_page_id, original_page_id,
            "save_node_cow must allocate a new page id, not reuse the input's"
        );
        assert!(
            new_page_id > original_page_id,
            "new page id should monotonically increase from the allocator"
        );

        // Property 2: the modified node is resolvable at the new id with
        // the new contents.
        let new_node = engine.load_node(new_page_id).unwrap();
        assert!(
            new_node.keys.iter().any(|k| k == b"beta"),
            "new page should contain the mutation"
        );
        assert!(
            new_node.keys.iter().any(|k| k == b"alpha"),
            "new page should retain pre-existing keys"
        );

        // Property 3: the original page id, if present in cache (it isn't
        // here because take_node_for_write removed it), is independent
        // of the CoW save. We assert the broader invariant: the new
        // entry was inserted at `new_page_id`, not at `original_page_id`.
        assert!(
            engine.node_cache.contains_key(&new_page_id),
            "save_node_cow must place the node in cache at the new id"
        );
        assert!(
            !engine.node_cache.contains_key(&original_page_id),
            "save_node_cow must not touch the old page id's cache entry; \
             take_node_for_write previously removed it"
        );
    }

    /// Phase 1.3 of the CoW MVCC refactor. After conversion of
    /// `insert_non_full_serial`, `split_child_serial`, and `split_root`
    /// to allocate-new-page semantics, every `put` should produce a
    /// strictly larger `root_page_id` than before — proof that the
    /// write path no longer mutates the existing root in place.
    ///
    /// 100 puts on a previously-empty engine should yield 100 distinct
    /// root ids, monotonically increasing.
    #[test]
    fn put_rotates_root_page_id_on_every_insert() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        let mut seen_root_ids: Vec<PageId> = Vec::with_capacity(101);
        seen_root_ids.push(engine.root_page_id());

        for i in 0..100 {
            let key = format!("k{:04}", i);
            let val = format!("v{}", i);
            engine.put(key.as_bytes(), val.as_bytes()).unwrap();
            seen_root_ids.push(engine.root_page_id());
        }

        // Every consecutive root id must differ — CoW writes never
        // reuse the previous root's id.
        for w in seen_root_ids.windows(2) {
            assert_ne!(
                w[0], w[1],
                "consecutive puts must produce distinct root ids \
                 under CoW; saw {} -> {}",
                w[0], w[1]
            );
        }

        // Strict monotonicity: the page allocator hands out ids in
        // increasing order and we never go backward (free list is empty
        // at this point — Phase 4 will introduce reclamation).
        for w in seen_root_ids.windows(2) {
            assert!(
                w[1] > w[0],
                "root id must increase monotonically; saw {} -> {}",
                w[0],
                w[1]
            );
        }

        // Sanity: every key still readable after the cascade churn.
        for i in 0..100 {
            let key = format!("k{:04}", i);
            let expected = format!("v{}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "key {} lost during CoW write cascade",
                key
            );
        }
    }

    #[test]
    fn test_basic_operations() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Put
        engine.put(b"key1", b"value1").unwrap();
        engine.put(b"key2", b"value2").unwrap();

        // Get
        assert_eq!(engine.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(engine.get(b"key2").unwrap(), Some(b"value2".to_vec()));
        assert_eq!(engine.get(b"key3").unwrap(), None);

        // Update
        engine.put(b"key1", b"updated").unwrap();
        assert_eq!(engine.get(b"key1").unwrap(), Some(b"updated".to_vec()));

        // Delete
        assert!(engine.delete(b"key1").unwrap());
        assert_eq!(engine.get(b"key1").unwrap(), None);
        assert!(!engine.delete(b"key1").unwrap()); // Already deleted
    }

    #[test]
    fn test_empty_value() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Empty value should be distinguishable from None
        engine.put(b"empty", b"").unwrap();
        assert_eq!(engine.get(b"empty").unwrap(), Some(Vec::new()));
        assert_eq!(engine.get(b"missing").unwrap(), None);
    }

    #[test]
    fn test_transaction_commit() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        let mut tx = engine.begin();
        tx.put(b"key1", b"value1").unwrap();
        tx.put(b"key2", b"value2").unwrap();
        tx.commit().unwrap();

        // Values should be visible
        assert_eq!(engine.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(engine.get(b"key2").unwrap(), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_transaction_rollback() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Set initial value
        engine.put(b"key1", b"initial").unwrap();

        // Start transaction, modify, rollback
        let mut tx = engine.begin();
        tx.put(b"key1", b"modified").unwrap();
        tx.put(b"key2", b"new").unwrap();
        tx.rollback().unwrap();

        // Original value should remain, new key should not exist
        assert_eq!(engine.get(b"key1").unwrap(), Some(b"initial".to_vec()));
        assert_eq!(engine.get(b"key2").unwrap(), None);
    }

    #[test]
    fn test_many_keys() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert many keys to trigger splits
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{}", i);
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Verify all keys
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let expected = format!("value{}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "Failed for key {}",
                key
            );
        }
    }

    #[test]
    fn test_large_values() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // 8KB value
        let large_value = vec![42u8; 8192];
        engine.put(b"large", &large_value).unwrap();
        assert_eq!(engine.get(b"large").unwrap(), Some(large_value));
    }

    #[test]
    fn test_concurrent_reads_writes() {
        use std::sync::Arc;
        use std::thread;

        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());

        // Pre-populate some data
        for i in 0..50 {
            let key = format!("key{:03}", i);
            let value = format!("value{}", i);
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let mut handles = vec![];

        // Spawn readers
        for thread_id in 0..4 {
            let engine = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    for i in 0..50 {
                        let key = format!("key{:03}", i);
                        let result = engine.get(key.as_bytes());
                        assert!(
                            result.is_ok(),
                            "Thread {} read failed: {:?}",
                            thread_id,
                            result.err()
                        );
                    }
                }
            }));
        }

        // Spawn writers
        for thread_id in 0..2 {
            let engine = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for round in 0..20 {
                    for i in 50..60 {
                        let key = format!("key{:03}", i);
                        let value = format!("value-{}-{}-{}", thread_id, round, i);
                        let result = engine.put(key.as_bytes(), value.as_bytes());
                        assert!(result.is_ok(), "Thread {} write failed", thread_id);
                    }
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify original data intact
        for i in 0..50 {
            let key = format!("key{:03}", i);
            let expected = format!("value{}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "Original key {} was corrupted",
                key
            );
        }

        // Concurrent-readers-during-writes is the property under test;
        // the data integrity assertions above already verify it. The
        // historical `active_latches() > 0` check tracked the optimistic
        // leaf-only put path, which is gone under CoW MVCC (Phase 1 —
        // see `pub fn put`). Page latches may still be acquired by
        // readers, but their presence is no longer load-bearing for
        // this test.
    }

    #[test]
    fn test_concurrent_splits() {
        use std::sync::Arc;
        use std::thread;

        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());

        let mut handles = vec![];

        // Spawn multiple writers that will trigger splits
        for thread_id in 0..4 {
            let engine = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    // Each thread writes to a different key range
                    let key = format!("t{}-key{:03}", thread_id, i);
                    let value = format!("value{}", i);
                    engine.put(key.as_bytes(), value.as_bytes()).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all data
        for thread_id in 0..4 {
            for i in 0..50 {
                let key = format!("t{}-key{:03}", thread_id, i);
                let expected = format!("value{}", i);
                assert_eq!(
                    engine.get(key.as_bytes()).unwrap(),
                    Some(expected.into_bytes()),
                    "Key {} missing",
                    key
                );
            }
        }

        // Should have multiple pages after splits
        assert!(
            engine.cache_size() > 1,
            "Expected multiple pages after splits"
        );
    }

    // =========================================================================
    // Range Iteration Tests
    // =========================================================================

    #[test]
    fn test_range_scan_forward() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert keys
        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.put(b"d", b"4").unwrap();
        engine.put(b"e", b"5").unwrap();

        // Scan all forward
        let mut iter = engine.scan(ScanDirection::Forward).unwrap();
        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].key, b"a");
        assert_eq!(results[1].key, b"b");
        assert_eq!(results[2].key, b"c");
        assert_eq!(results[3].key, b"d");
        assert_eq!(results[4].key, b"e");
    }

    #[test]
    fn test_range_scan_backward() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.put(b"d", b"4").unwrap();
        engine.put(b"e", b"5").unwrap();

        // Scan all backward
        let mut iter = engine.scan(ScanDirection::Backward).unwrap();
        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].key, b"e");
        assert_eq!(results[1].key, b"d");
        assert_eq!(results[2].key, b"c");
        assert_eq!(results[3].key, b"b");
        assert_eq!(results[4].key, b"a");
    }

    #[test]
    fn test_range_bounded_inclusive() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.put(b"d", b"4").unwrap();
        engine.put(b"e", b"5").unwrap();

        // Range [b, d] inclusive
        let mut iter = engine
            .range(
                Bound::Included(b"b"),
                Bound::Included(b"d"),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, b"b");
        assert_eq!(results[1].key, b"c");
        assert_eq!(results[2].key, b"d");
    }

    #[test]
    fn test_range_bounded_exclusive() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.put(b"d", b"4").unwrap();
        engine.put(b"e", b"5").unwrap();

        // Range (b, d) exclusive
        let mut iter = engine
            .range(
                Bound::Excluded(b"b"),
                Bound::Excluded(b"d"),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, b"c");
    }

    #[test]
    fn test_range_half_bounded() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.put(b"d", b"4").unwrap();
        engine.put(b"e", b"5").unwrap();

        // Range [c, ∞)
        let mut iter = engine
            .range(
                Bound::Included(b"c"),
                Bound::Unbounded,
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, b"c");
        assert_eq!(results[1].key, b"d");
        assert_eq!(results[2].key, b"e");

        // Range (-∞, c]
        let mut iter = engine
            .range(
                Bound::Unbounded,
                Bound::Included(b"c"),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, b"a");
        assert_eq!(results[1].key, b"b");
        assert_eq!(results[2].key, b"c");
    }

    #[test]
    fn test_prefix_scan() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert with different prefixes
        engine.put(b"user:1", b"alice").unwrap();
        engine.put(b"user:2", b"bob").unwrap();
        engine.put(b"user:3", b"charlie").unwrap();
        engine.put(b"post:1", b"hello").unwrap();
        engine.put(b"post:2", b"world").unwrap();

        // Prefix scan for "user:"
        let mut iter = engine.prefix_scan(b"user:").unwrap();
        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|e| e.key.starts_with(b"user:")));

        // Prefix scan for "post:"
        let mut iter = engine.prefix_scan(b"post:").unwrap();
        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.key.starts_with(b"post:")));
    }

    #[test]
    fn test_range_empty_result() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"z", b"2").unwrap();

        // Range in the middle where nothing exists
        let mut iter = engine
            .range(
                Bound::Included(b"m"),
                Bound::Included(b"n"),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }

        assert_eq!(count, 0);
    }

    #[test]
    fn test_range_with_many_keys() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert 100 keys to force tree splits
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{}", i);
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Scan range [key020, key040]
        let mut iter = engine
            .range(
                Bound::Included(b"key020"),
                Bound::Included(b"key040"),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 21); // 020 to 040 inclusive
        assert_eq!(results[0].key, b"key020".to_vec());
        assert_eq!(results[20].key, b"key040".to_vec());

        // Verify order
        for i in 0..results.len() - 1 {
            assert!(
                results[i].key < results[i + 1].key,
                "Keys should be in order"
            );
        }
    }

    #[test]
    fn test_scan_empty_tree() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        let mut iter = engine.scan(ScanDirection::Forward).unwrap();
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_range_single_key() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"only", b"one").unwrap();

        let mut iter = engine.scan(ScanDirection::Forward).unwrap();
        let entry = iter.next().unwrap().unwrap();
        assert_eq!(entry.key, b"only");
        assert_eq!(entry.value, b"one");
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_index_trait_range() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();

        // Test through Index trait
        let index: &dyn Index = &engine;
        let mut iter = index
            .range(
                Bound::Included(b"a".as_slice()),
                Bound::Included(b"b".as_slice()),
                ScanDirection::Forward,
            )
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_index_count() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        for i in 0..50 {
            let key = format!("key{:02}", i);
            engine.put(key.as_bytes(), b"value").unwrap();
        }

        // Test through Index trait
        let index: &dyn Index = &engine;

        // Count all
        let total = index.count().unwrap();
        assert_eq!(total, 50);

        // Count range
        let range_count = index
            .count_range(
                Bound::Included(b"key10".as_slice()),
                Bound::Included(b"key20".as_slice()),
            )
            .unwrap();
        assert_eq!(range_count, 11); // 10 to 20 inclusive
    }

    // =========================================================================
    // Delete with Rebalancing Tests
    // =========================================================================

    #[test]
    fn test_delete_basic() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"a", b"1").unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.put(b"c", b"3").unwrap();

        assert!(engine.delete(b"b").unwrap());
        assert!(!engine.delete(b"b").unwrap()); // Already deleted

        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), None);
        assert_eq!(engine.get(b"c").unwrap(), Some(b"3".to_vec()));
    }

    #[test]
    fn test_delete_all_keys() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        for i in 0..20 {
            let key = format!("key{:02}", i);
            engine.put(key.as_bytes(), b"value").unwrap();
        }

        // Delete all
        for i in 0..20 {
            let key = format!("key{:02}", i);
            assert!(
                engine.delete(key.as_bytes()).unwrap(),
                "Failed to delete {}",
                key
            );
        }

        // Verify all deleted
        for i in 0..20 {
            let key = format!("key{:02}", i);
            assert_eq!(engine.get(key.as_bytes()).unwrap(), None);
        }

        // Count should be 0
        let count = Index::count(&engine).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_delete_with_rebalance() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert enough keys to cause splits (MAX_KEYS = 32)
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{}", i);
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Verify tree has multiple nodes
        assert!(engine.cache_size() > 1, "Tree should have multiple nodes");

        // Delete half the keys
        for i in (0..100).step_by(2) {
            let key = format!("key{:03}", i);
            assert!(
                engine.delete(key.as_bytes()).unwrap(),
                "Failed to delete {}",
                key
            );
        }

        // Verify remaining keys still accessible
        for i in (1..100).step_by(2) {
            let key = format!("key{:03}", i);
            let expected = format!("value{}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "Key {} should still exist",
                key
            );
        }

        // Count should be 50
        let count = Index::count(&engine).unwrap();
        assert_eq!(count, 50);
    }

    #[test]
    fn test_delete_in_reverse_order() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        for i in 0..50 {
            let key = format!("key{:02}", i);
            engine.put(key.as_bytes(), b"v").unwrap();
        }

        // Delete in reverse order (tests different rebalancing scenarios)
        for i in (0..50).rev() {
            let key = format!("key{:02}", i);

            // Check key24 before each delete
            let key24_exists = engine.get(b"key24").unwrap().is_some();

            let deleted = engine.delete(key.as_bytes()).unwrap();

            // Check key24 after each delete
            let key24_exists_after = engine.get(b"key24").unwrap().is_some();

            if key24_exists && !key24_exists_after && i != 24 {
                panic!(
                    "key24 disappeared after deleting key{:02}! count={}",
                    i,
                    Index::count(&engine).unwrap()
                );
            }

            if !deleted {
                let exists = engine.get(key.as_bytes()).unwrap();
                panic!(
                    "Failed to delete key{:02} (i={}), exists={:?}, count={}",
                    i,
                    i,
                    exists,
                    Index::count(&engine).unwrap()
                );
            }
        }

        // Should be empty
        let count = Index::count(&engine).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_delete_and_reinsert() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert, delete, reinsert
        for round in 0..3 {
            for i in 0..30 {
                let key = format!("key{:02}", i);
                let value = format!("round{}-value{}", round, i);
                engine.put(key.as_bytes(), value.as_bytes()).unwrap();
            }

            if round < 2 {
                for i in 0..30 {
                    let key = format!("key{:02}", i);
                    assert!(engine.delete(key.as_bytes()).unwrap());
                }
            }
        }

        // Verify final values
        for i in 0..30 {
            let key = format!("key{:02}", i);
            let expected = format!("round2-value{}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes())
            );
        }
    }

    #[test]
    fn test_delete_maintains_order() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        // Insert keys
        for i in 0..50 {
            let key = format!("key{:02}", i);
            engine.put(key.as_bytes(), b"v").unwrap();
        }

        // Delete every third key
        for i in (0..50).step_by(3) {
            let key = format!("key{:02}", i);
            engine.delete(key.as_bytes()).unwrap();
        }

        // Scan and verify order
        let mut iter = engine.scan(ScanDirection::Forward).unwrap();
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count = 0;

        while let Some(result) = iter.next() {
            let entry = result.unwrap();
            if let Some(prev) = &prev_key {
                assert!(entry.key > *prev, "Keys should be in ascending order");
            }
            prev_key = Some(entry.key);
            count += 1;
        }

        // Should have 50 - 17 = 33 keys (0, 3, 6, ..., 48 deleted = 17 keys)
        assert_eq!(count, 33);
    }

    // ========================================================================
    // Concurrent tests (Phase 4: optimistic latch crabbing)
    // ========================================================================

    #[test]
    fn test_concurrent_puts() {
        let engine = Arc::new(Engine::new(crate::storage::memory::MemoryBackend::new()).unwrap());
        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let eng = Arc::clone(&engine);
                std::thread::spawn(move || {
                    for i in 0..1000u32 {
                        let key = format!("t{tid}-k{i}");
                        eng.put(key.as_bytes(), b"value").unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Verify all 8000 keys present
        for tid in 0..8u32 {
            for i in 0..1000u32 {
                let key = format!("t{tid}-k{i}");
                assert!(
                    engine.get(key.as_bytes()).unwrap().is_some(),
                    "Missing key: {key}"
                );
            }
        }
    }

    #[test]
    fn test_concurrent_mixed_ops() {
        let engine = Arc::new(Engine::new(crate::storage::memory::MemoryBackend::new()).unwrap());
        // Pre-load some keys
        for i in 0..500u32 {
            let key = format!("shared-k{i:05}");
            engine.put(key.as_bytes(), b"initial").unwrap();
        }

        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let eng = Arc::clone(&engine);
                std::thread::spawn(move || {
                    if tid < 4 {
                        // Writers: update existing + insert new
                        for i in 0..500u32 {
                            let key = format!("shared-k{i:05}");
                            eng.put(key.as_bytes(), format!("t{tid}").as_bytes())
                                .unwrap();
                        }
                    } else {
                        // Readers: read keys
                        for i in 0..500u32 {
                            let key = format!("shared-k{i:05}");
                            let _ = eng.get(key.as_bytes());
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Verify all shared keys still present
        for i in 0..500u32 {
            let key = format!("shared-k{i:05}");
            assert!(
                engine.get(key.as_bytes()).unwrap().is_some(),
                "Missing key after concurrent ops: {key}"
            );
        }
    }

    #[test]
    fn test_concurrent_puts_with_splits() {
        // Force many splits: 8 threads each insert 2000 keys = 16K total
        let engine = Arc::new(Engine::new(crate::storage::memory::MemoryBackend::new()).unwrap());
        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let eng = Arc::clone(&engine);
                std::thread::spawn(move || {
                    for i in 0..2000u32 {
                        let key = format!("split-t{tid}-k{i:06}");
                        eng.put(key.as_bytes(), b"v").unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Verify all 16000 keys present
        for tid in 0..8u32 {
            for i in 0..2000u32 {
                let key = format!("split-t{tid}-k{i:06}");
                assert!(
                    engine.get(key.as_bytes()).unwrap().is_some(),
                    "Missing key: {key}"
                );
            }
        }
    }

    #[test]
    fn test_concurrent_deletes() {
        let engine = Arc::new(Engine::new(crate::storage::memory::MemoryBackend::new()).unwrap());
        // Pre-load keys
        for tid in 0..4u32 {
            for i in 0..500u32 {
                let key = format!("del-t{tid}-k{i:05}");
                engine.put(key.as_bytes(), b"value").unwrap();
            }
        }
        // Each thread deletes its own keys concurrently
        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let eng = Arc::clone(&engine);
                std::thread::spawn(move || {
                    for i in 0..500u32 {
                        let key = format!("del-t{tid}-k{i:05}");
                        eng.delete(key.as_bytes()).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Verify all keys deleted
        for tid in 0..4u32 {
            for i in 0..500u32 {
                let key = format!("del-t{tid}-k{i:05}");
                assert!(
                    engine.get(key.as_bytes()).unwrap().is_none(),
                    "Key should be deleted: {key}"
                );
            }
        }
    }

    #[test]
    fn test_large_value_overflow() {
        // Use a small page size to force overflow with modest value sizes
        let backend = MemoryBackend::with_page_size(4096); // 4KB pages
        let engine = Engine::new(backend).unwrap();

        // max_inline_value_size = (4096 - 32) / 4 = 1016 bytes
        // Values larger than this should use overflow pages

        // Small value — inline
        let small_val = vec![0xAA; 500];
        engine.put(b"small", &small_val).unwrap();
        assert_eq!(engine.get(b"small").unwrap(), Some(small_val.clone()));

        // Large value — should trigger overflow
        let large_val = vec![0xBB; 2000];
        engine.put(b"large", &large_val).unwrap();

        // Flush to disk to exercise overflow serialization
        engine.sync().unwrap();

        // Clear cache to force deserialization from disk (with overflow resolution)
        engine.clear_cache();

        // Read back — should reconstruct from overflow pages
        assert_eq!(engine.get(b"large").unwrap(), Some(large_val.clone()));
        assert_eq!(engine.get(b"small").unwrap(), Some(small_val.clone()));

        // Very large value — multi-page overflow chain
        let huge_val = vec![0xCC; 20_000]; // ~5 overflow pages at 4KB
        engine.put(b"huge", &huge_val).unwrap();
        engine.sync().unwrap();
        engine.clear_cache();
        assert_eq!(engine.get(b"huge").unwrap(), Some(huge_val.clone()));

        // Update large value to small — overflow pages should still work
        let updated_val = vec![0xDD; 100];
        engine.put(b"large", &updated_val).unwrap();
        engine.sync().unwrap();
        engine.clear_cache();
        assert_eq!(engine.get(b"large").unwrap(), Some(updated_val));

        // Delete and verify
        assert!(engine.delete(b"huge").unwrap());
        assert_eq!(engine.get(b"huge").unwrap(), None);
    }

    #[test]
    fn test_large_value_with_many_keys() {
        // Verify overflow works with many entries in the B-tree
        let backend = MemoryBackend::with_page_size(4096);
        let engine = Engine::new(backend).unwrap();

        // Insert 100 entries, mix of small and large values
        for i in 0..100u32 {
            let key = format!("key{:04}", i);
            let val = if i % 10 == 0 {
                // Every 10th value is large (triggers overflow)
                vec![(i & 0xFF) as u8; 3000]
            } else {
                vec![(i & 0xFF) as u8; 50]
            };
            engine.put(key.as_bytes(), &val).unwrap();
        }

        // Sync and clear cache to force overflow serialization + deserialization
        engine.sync().unwrap();
        engine.clear_cache();

        // Verify all entries
        for i in 0..100u32 {
            let key = format!("key{:04}", i);
            let expected = if i % 10 == 0 {
                vec![(i & 0xFF) as u8; 3000]
            } else {
                vec![(i & 0xFF) as u8; 50]
            };
            let actual = engine.get(key.as_bytes()).unwrap();
            assert_eq!(actual, Some(expected), "Mismatch at key {}", key);
        }
    }

    /// A `write_transaction` whose closure mutates the tree and then
    /// returns `Err` must be fully invisible — no leaked puts, the
    /// engine still usable. (CoW rollback rail; see
    /// `Engine::abort_uncommitted`.)
    #[test]
    fn write_transaction_rolls_back_on_closure_error() {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).unwrap();

        engine.put(b"keep", b"original").unwrap();
        engine.sync().unwrap();
        let height_before = engine.tree_height.load(Ordering::Acquire);

        let res: Result<(), Error> = engine.write_transaction(|tx| {
            tx.put(b"ghost", b"should-vanish")?;
            tx.put(b"keep", b"should-not-stick")?;
            Err(Error::Storage(StorageError::Backend(
                "simulated mid-cascade failure".into(),
            )))
        });
        assert!(res.is_err());

        assert_eq!(engine.get(b"ghost").unwrap(), None, "failed-tx put leaked");
        assert_eq!(
            engine.get(b"keep").unwrap(),
            Some(b"original".to_vec()),
            "failed-tx overwrite leaked"
        );
        assert_eq!(
            engine.tree_height.load(Ordering::Acquire),
            height_before,
            "tree_height not restored after failed tx"
        );

        // Engine remains usable for subsequent writes, across a sync +
        // cache clear.
        engine.put(b"after", b"works").unwrap();
        engine.sync().unwrap();
        engine.clear_cache();
        assert_eq!(engine.get(b"after").unwrap(), Some(b"works".to_vec()));
        assert_eq!(engine.get(b"keep").unwrap(), Some(b"original".to_vec()));
        assert_eq!(engine.get(b"ghost").unwrap(), None);
    }

    /// Stress: build a B-tree with overflow-sized values, sync, then run
    /// a `write_transaction` that overwrites every key (allocating fresh
    /// CoW pages and overflow chains) and then fails. The committed data
    /// must survive — intact and readable after a sync + cache clear —
    /// proving the failed transaction neither published a half-built
    /// root nor released pages still reachable from the committed root.
    #[test]
    fn failed_write_transaction_preserves_committed_tree() {
        let backend = MemoryBackend::with_page_size(4096);
        let engine = Engine::new(backend).unwrap();

        let orig = |i: u32| vec![(i & 0xFF) as u8; 6000]; // overflow-sized
        for i in 0..64u32 {
            engine.put(format!("k{i:04}").as_bytes(), &orig(i)).unwrap();
        }
        engine.sync().unwrap();
        let pf_before = engine.pending_free_count();

        let res: Result<(), Error> = engine.write_transaction(|tx| {
            for i in 0..64u32 {
                tx.put(format!("k{i:04}").as_bytes(), &vec![0xEE; 7000])?;
            }
            Err(Error::Storage(StorageError::Backend(
                "simulated failure after 64 overwrites".into(),
            )))
        });
        assert!(res.is_err());
        assert_eq!(
            engine.pending_free_count(),
            pf_before,
            "failed tx left deferred frees (for still-live pages) queued"
        );

        engine.sync().unwrap();
        engine.clear_cache();
        for i in 0..64u32 {
            assert_eq!(
                engine.get(format!("k{i:04}").as_bytes()).unwrap(),
                Some(orig(i)),
                "k{i:04} corrupted by failed write_transaction"
            );
        }
        // ...and the engine still accepts new writes against the
        // (correctly-preserved) committed tree.
        engine.put(b"k0000", b"fresh").unwrap();
        engine.sync().unwrap();
        engine.clear_cache();
        assert_eq!(engine.get(b"k0000").unwrap(), Some(b"fresh".to_vec()));
        for i in 1..64u32 {
            assert_eq!(
                engine.get(format!("k{i:04}").as_bytes()).unwrap(),
                Some(orig(i))
            );
        }
    }
}
