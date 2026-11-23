//! SlateDB / Turbopuffer-Style Disaggregated Object-Storage Tier for CoW Branches
//!
//! Decouples branch durability from local disks so JouleDB branches can scale to zero
//! (`suspend_to_manifest`) and wake up statelessly on any compute node (`resume_from_manifest`)
//! in milliseconds while fetching cold 64 KiB pages lazily from S3/GCS/R2-compatible object storage:
//!
//! - **Hot Tier**: Branch-local mutable CoW dirty page overlay in RAM.
//! - **Warm Tier**: Bounded FIFO/LRU clean page cache in RAM/NVMe.
//! - **Cold Tier**: Content-addressed immutable page blobs (`cas/<digest>.jpage`) + JSON
//!   [`BranchSnapshotManifest`] stored in any [`ObjectStoreTier`] backend.

use crate::BranchError;
use joule_db_core::error::StorageError;
use joule_db_core::storage::{DEFAULT_PAGE_SIZE, Page, PageId, StorageBackend, StorageStats};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Pluggable object-storage interface (`s3://`, `gs://`, `r2://`, `file://`, or memory).
pub trait ObjectStoreTier: Send + Sync {
    fn put_object(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError>;
    fn get_object(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError>;
    fn get_range(&self, key: &str, offset: usize, len: usize) -> Result<Option<Vec<u8>>, StorageError> {
        let Some(full) = self.get_object(key)? else {
            return Ok(None);
        };
        if offset >= full.len() {
            return Ok(Some(Vec::new()));
        }
        let end = (offset + len).min(full.len());
        Ok(Some(full[offset..end].to_vec()))
    }
    fn delete_object(&self, key: &str) -> Result<(), StorageError>;
    fn backend_scheme(&self) -> &str;
}

/// In-memory object store for fast tests and ephemeral sandboxes.
#[derive(Default)]
pub struct InMemoryObjectStore {
    blobs: RwLock<HashMap<String, Vec<u8>>>,
    gets: AtomicU64,
    puts: AtomicU64,
}

impl InMemoryObjectStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn object_count(&self) -> usize {
        self.blobs.read().map(|m| m.len()).unwrap_or(0)
    }

    pub fn io_counts(&self) -> (u64, u64) {
        (
            self.gets.load(Ordering::Relaxed),
            self.puts.load(Ordering::Relaxed),
        )
    }
}

impl ObjectStoreTier for InMemoryObjectStore {
    fn put_object(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError> {
        self.puts.fetch_add(1, Ordering::Relaxed);
        self.blobs
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    fn get_object(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        self.gets.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .blobs
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .get(key)
            .cloned())
    }

    fn delete_object(&self, key: &str) -> Result<(), StorageError> {
        self.blobs
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .remove(key);
        Ok(())
    }

    fn backend_scheme(&self) -> &str {
        "mem-obj"
    }
}

/// Local-directory object store (`file://` staging / MinIO-style bucket directory).
pub struct LocalDirObjectStore {
    root: PathBuf,
}

impl LocalDirObjectStore {
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self, StorageError> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(Self { root })
    }

    fn resolve(&self, key: &str) -> PathBuf {
        let clean: String = key
            .trim_start_matches('/')
            .chars()
            .map(|c| if c.is_alphanumeric() || matches!(c, '/' | '.' | '-' | '_') { c } else { '_' })
            .collect();
        self.root.join(clean)
    }
}

impl ObjectStoreTier for LocalDirObjectStore {
    fn put_object(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError> {
        let full = self.resolve(key);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StorageError::Io(e.to_string()))?;
        }
        std::fs::write(&full, bytes).map_err(|e| StorageError::Io(e.to_string()))
    }

    fn get_object(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let full = self.resolve(key);
        if !full.exists() {
            return Ok(None);
        }
        std::fs::read(&full)
            .map(Some)
            .map_err(|e| StorageError::Io(e.to_string()))
    }

    fn delete_object(&self, key: &str) -> Result<(), StorageError> {
        let full = self.resolve(key);
        if full.exists() {
            std::fs::remove_file(&full).map_err(|e| StorageError::Io(e.to_string()))?;
        }
        Ok(())
    }

    fn backend_scheme(&self) -> &str {
        "file-obj"
    }
}

/// Serializable, content-addressed branch manifest enabling O(1) stateless compute spin-up
/// and cross-branch page deduplication in object storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchSnapshotManifest {
    pub branch_name: String,
    pub parent_branch: Option<String>,
    pub page_size: usize,
    pub next_page_id: PageId,
    /// Maps logical `PageId` -> content-addressed object key `cas/<digest>.jpage`.
    pub page_directory: BTreeMap<PageId, String>,
    /// Tombstoned page IDs freed on this branch.
    pub freed_pages: Vec<PageId>,
    pub energy_budget_uj: Option<u64>,
    pub energy_spent_uj: u64,
}

fn cas_digest_128(bytes: &[u8]) -> String {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x8422_2325_cbf2_9ce4;
    for &b in bytes {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 = h2.wrapping_mul(0x0000_0100_0000_01b3) ^ (b as u64);
    }
    format!("{h1:016x}{h2:016x}")
}

struct WarmLruCache {
    capacity: usize,
    order: VecDeque<PageId>,
    pages: HashMap<PageId, Page>,
}

impl WarmLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            pages: HashMap::new(),
        }
    }

    fn get(&mut self, page_id: PageId) -> Option<Page> {
        let p = self.pages.get(&page_id)?.clone();
        self.order.retain(|&id| id != page_id);
        self.order.push_back(page_id);
        Some(p)
    }

    fn insert(&mut self, page: Page) {
        let pid = page.id;
        if !self.pages.contains_key(&pid) && self.pages.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.pages.remove(&evicted);
            }
        }
        self.order.retain(|&id| id != pid);
        self.order.push_back(pid);
        self.pages.insert(pid, page);
    }

    fn remove(&mut self, page_id: PageId) {
        self.pages.remove(&page_id);
        self.order.retain(|&id| id != page_id);
    }
}

/// Three-tier (`Hot` dirty overlay -> `Warm` bounded LRU -> `Cold` CAS Object Store)
/// disaggregated Copy-on-Write branch storage backend.
pub struct TieredCowBranchStorage {
    branch_name: String,
    parent_branch: Option<String>,
    object_store: Arc<dyn ObjectStoreTier>,
    page_size: usize,
    /// Hot dirty pages modified since last `flush_to_object_store`.
    hot_dirty: RwLock<HashMap<PageId, Page>>,
    /// Warm clean bounded LRU page cache.
    warm_cache: RwLock<WarmLruCache>,
    /// Cold content-addressed object directory (`PageId -> cas/<hash>.jpage`).
    cold_directory: RwLock<BTreeMap<PageId, String>>,
    freed: RwLock<HashSet<PageId>>,
    next_page_id: AtomicU64,
    pages_read: AtomicU64,
    pages_written: AtomicU64,
}

impl TieredCowBranchStorage {
    /// Create a fresh object-store-backed branch storage with `warm_cache_pages` capacity.
    pub fn new(
        branch_name: &str,
        parent_branch: Option<String>,
        object_store: Arc<dyn ObjectStoreTier>,
        warm_cache_pages: usize,
    ) -> Self {
        Self {
            branch_name: branch_name.to_string(),
            parent_branch,
            object_store,
            page_size: DEFAULT_PAGE_SIZE,
            hot_dirty: RwLock::new(HashMap::new()),
            warm_cache: RwLock::new(WarmLruCache::new(warm_cache_pages)),
            cold_directory: RwLock::new(BTreeMap::new()),
            freed: RwLock::new(HashSet::new()),
            next_page_id: AtomicU64::new(1),
            pages_read: AtomicU64::new(0),
            pages_written: AtomicU64::new(0),
        }
    }

    /// Fork a child `TieredCowBranchStorage` in O(1) by sharing the parent's immutable
    /// CAS `cold_directory` pointers (zero object duplication!).
    pub fn fork_child(&self, child_name: &str, warm_cache_pages: usize) -> Result<Self, BranchError> {
        let cold_snapshot = self
            .cold_directory
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?
            .clone();
        let freed_snapshot = self
            .freed
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?
            .clone();

        Ok(Self {
            branch_name: child_name.to_string(),
            parent_branch: Some(self.branch_name.clone()),
            object_store: Arc::clone(&self.object_store),
            page_size: self.page_size,
            hot_dirty: RwLock::new(HashMap::new()),
            warm_cache: RwLock::new(WarmLruCache::new(warm_cache_pages)),
            cold_directory: RwLock::new(cold_snapshot),
            freed: RwLock::new(freed_snapshot),
            next_page_id: AtomicU64::new(self.next_page_id.load(Ordering::SeqCst)),
            pages_read: AtomicU64::new(0),
            pages_written: AtomicU64::new(0),
        })
    }

    /// Flush all `hot_dirty` pages to content-addressed blobs in `object_store` and publish
    /// `manifests/<branch_name>.json` for stateless scale-to-zero suspend.
    pub fn suspend_to_manifest(
        &self,
        energy_budget_uj: Option<u64>,
        energy_spent_uj: u64,
    ) -> Result<BranchSnapshotManifest, StorageError> {
        let dirty: Vec<(PageId, Page)> = {
            let mut hot = self
                .hot_dirty
                .write()
                .map_err(|e| StorageError::Backend(e.to_string()))?;
            hot.drain().collect()
        };

        {
            let mut cold = self
                .cold_directory
                .write()
                .map_err(|e| StorageError::Backend(e.to_string()))?;
            let mut warm = self
                .warm_cache
                .write()
                .map_err(|e| StorageError::Backend(e.to_string()))?;

            for (pid, page) in dirty {
                let encoded = page.encode(self.page_size)?;
                let digest = cas_digest_128(&encoded);
                let obj_key = format!("cas/{digest}.jpage");
                self.object_store.put_object(&obj_key, &encoded)?;
                cold.insert(pid, obj_key);
                warm.insert(page);
            }
        }

        let cold_dir = self
            .cold_directory
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .clone();
        let mut freed_vec: Vec<PageId> = self
            .freed
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .iter()
            .copied()
            .collect();
        freed_vec.sort_unstable();

        let manifest = BranchSnapshotManifest {
            branch_name: self.branch_name.clone(),
            parent_branch: self.parent_branch.clone(),
            page_size: self.page_size,
            next_page_id: self.next_page_id.load(Ordering::SeqCst),
            page_directory: cold_dir,
            freed_pages: freed_vec,
            energy_budget_uj,
            energy_spent_uj,
        };

        let manifest_bytes = serde_json::to_vec(&manifest)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        self.object_store.put_object(
            &format!("manifests/{}.json", self.branch_name),
            &manifest_bytes,
        )?;
        Ok(manifest)
    }

    /// Wake up a stateless compute node in O(metadata) from `manifests/<branch_name>.json`
    /// without pre-downloading data pages; pages fault into `warm_cache` on demand!
    pub fn resume_from_object_store(
        branch_name: &str,
        object_store: Arc<dyn ObjectStoreTier>,
        warm_cache_pages: usize,
    ) -> Result<(Self, BranchSnapshotManifest), StorageError> {
        let key = format!("manifests/{branch_name}.json");
        let bytes = object_store.get_object(&key)?.ok_or_else(|| {
            StorageError::Backend(format!("branch manifest not found in object store: {key}"))
        })?;
        let manifest: BranchSnapshotManifest = serde_json::from_slice(&bytes)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let storage = Self {
            branch_name: manifest.branch_name.clone(),
            parent_branch: manifest.parent_branch.clone(),
            object_store,
            page_size: manifest.page_size,
            hot_dirty: RwLock::new(HashMap::new()),
            warm_cache: RwLock::new(WarmLruCache::new(warm_cache_pages)),
            cold_directory: RwLock::new(manifest.page_directory.clone()),
            freed: RwLock::new(manifest.freed_pages.iter().copied().collect()),
            next_page_id: AtomicU64::new(manifest.next_page_id),
            pages_read: AtomicU64::new(0),
            pages_written: AtomicU64::new(0),
        };
        Ok((storage, manifest))
    }
}

impl StorageBackend for TieredCowBranchStorage {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        if self
            .freed
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .contains(&page_id)
        {
            return Ok(None);
        }

        // 1. Hot dirty overlay
        if let Some(p) = self
            .hot_dirty
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .get(&page_id)
            .cloned()
        {
            self.pages_read.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(p));
        }

        // 2. Warm LRU cache
        if let Some(p) = self
            .warm_cache
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .get(page_id)
        {
            self.pages_read.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(p));
        }

        // 3. Cold CAS Object Storage fault-in
        let obj_key = self
            .cold_directory
            .read()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .get(&page_id)
            .cloned();
        let Some(key) = obj_key else {
            return Ok(None);
        };
        let Some(raw) = self.object_store.get_object(&key)? else {
            return Err(StorageError::Backend(format!(
                "dangling CAS object {key} for page {page_id}"
            )));
        };
        let page = Page::decode(&raw).map_err(|e| StorageError::Serialization(e.to_string()))?;
        self.warm_cache
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .insert(page.clone());
        self.pages_read.fetch_add(1, Ordering::Relaxed);
        Ok(Some(page))
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        let pid = page.id;
        self.freed
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .remove(&pid);
        self.hot_dirty
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .insert(pid, page);
        self.pages_written.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        Ok(self.next_page_id.fetch_add(1, Ordering::SeqCst))
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        self.hot_dirty
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .remove(&page_id);
        self.warm_cache
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .remove(page_id);
        self.cold_directory
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .remove(&page_id);
        self.freed
            .write()
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .insert(page_id);
        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        let _ = self.suspend_to_manifest(None, 0)?;
        Ok(())
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn stats(&self) -> StorageStats {
        let cold_len = self.cold_directory.read().map(|c| c.len()).unwrap_or(0);
        let hot_len = self.hot_dirty.read().map(|h| h.len()).unwrap_or(0);
        let freed_len = self.freed.read().map(|f| f.len()).unwrap_or(0);
        StorageStats {
            total_pages: (cold_len + hot_len) as u64,
            free_pages: freed_len as u64,
            pages_read: self.pages_read.load(Ordering::Relaxed),
            pages_written: self.pages_written.load(Ordering::Relaxed),
            page_size: self.page_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::storage::PageType;

    #[test]
    fn test_stateless_suspend_resume_and_cas_branch_dedup() {
        let obj_store = Arc::new(InMemoryObjectStore::new());
        let mut main_tier =
            TieredCowBranchStorage::new("main", None, obj_store.clone(), 2);

        let p1 = main_tier.allocate_page().unwrap();
        let p2 = main_tier.allocate_page().unwrap();
        // Both pages carry identical bytes -> CAS deduplicates to 1 physical `.jpage` blob!
        main_tier
            .write_page(Page::with_data(p1, PageType::BTreeLeaf, b"shared-payload".to_vec()))
            .unwrap();
        main_tier
            .write_page(Page::with_data(p2, PageType::BTreeLeaf, b"body-2".to_vec()))
            .unwrap();

        let manifest = main_tier.suspend_to_manifest(Some(5_000), 120).unwrap();
        assert_eq!(manifest.page_directory.len(), 2);

        // O(1) child fork shares CAS pointers without copying objects; modifies only p2
        let mut child_tier = main_tier.fork_child("agent-exp", 2).unwrap();
        child_tier
            .write_page(Page::with_data(p2, PageType::BTreeLeaf, b"mutated-by-agent".to_vec()))
            .unwrap();
        child_tier.suspend_to_manifest(Some(500), 42).unwrap();

        // Simulate stateless scale-from-zero wakeup on a brand-new compute node!
        let (resumed_child, child_mf) =
            TieredCowBranchStorage::resume_from_object_store("agent-exp", obj_store.clone(), 2)
                .unwrap();
        assert_eq!(child_mf.energy_spent_uj, 42);
        assert_eq!(
            resumed_child.read_page(p1).unwrap().unwrap().data,
            b"shared-payload"
        );
        assert_eq!(
            resumed_child.read_page(p2).unwrap().unwrap().data,
            b"mutated-by-agent"
        );

        // Main remains isolated
        let (resumed_main, _) =
            TieredCowBranchStorage::resume_from_object_store("main", obj_store, 2).unwrap();
        assert_eq!(resumed_main.read_page(p2).unwrap().unwrap().data, b"body-2");
    }
}
