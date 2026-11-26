//! OPFS Virtual Block Device
//!
//! High-performance block-based storage for OPFS, replacing file-per-page approach.
//! Inspired by Phantom Engine's .wpack format for maximum I/O performance.
//!
//! Architecture:
//! - Single large file (data.wdb) containing all pages
//! - Block-based addressing (64KB blocks)
//! - In-memory page table (PageID -> BlockAddress)
//! - Persistent page table stored in meta.wdb

use joule_db_core::storage::PageId;
use joule_db_core::storage::page::DEFAULT_PAGE_SIZE;
use std::collections::HashMap;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

use super::config::PersistenceConfig;
use super::error::PersistenceError;
use super::{AsyncPersistenceBackend, StorageStats};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Block size (64KB) - matches Phantom Engine recommendation
const BLOCK_SIZE: u64 = 64 * 1024;

/// Pages per block (assuming 16KB pages)
const PAGES_PER_BLOCK: u64 = BLOCK_SIZE / DEFAULT_PAGE_SIZE as u64;

/// Block address within the virtual block device
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockAddress {
    /// Block ID (which 64KB block)
    block_id: u64,
    /// Offset within the block (for pages smaller than block size)
    offset: u64,
}

/// OPFS Virtual Block Device Backend
///
/// Stores all pages in a single large file with block-based addressing.
/// This provides 10-100x better I/O performance compared to file-per-page.

pub struct OPFSVirtualBlockDevice {
    root_dir: String,
    #[cfg(target_arch = "wasm32")]
    data_file_handle: Arc<RwLock<Option<JsValue>>>, // Main data file
    #[cfg(target_arch = "wasm32")]
    meta_file_handle: Arc<RwLock<Option<JsValue>>>, // Metadata file
    #[cfg(target_arch = "wasm32")]
    sync_handle: Arc<RwLock<Option<JsValue>>>, // SyncAccessHandle for workers
    initialized: Arc<RwLock<bool>>,
    /// Page table: PageID -> BlockAddress
    page_table: Arc<RwLock<HashMap<PageId, BlockAddress>>>,
    /// Free blocks available for allocation
    free_blocks: Arc<RwLock<Vec<u64>>>,
    /// Next block ID to allocate
    next_block_id: Arc<AtomicU64>,
    /// Cache for recently accessed pages
    cache: Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    /// Statistics
    stats: Arc<RwLock<StorageStats>>,
    config: PersistenceConfig,
}

impl OPFSVirtualBlockDevice {
    /// Create a new OPFS Virtual Block Device
    pub fn new(root_dir: &str) -> Self {
        Self::with_config(PersistenceConfig::new(root_dir), root_dir.to_string())
    }

    /// Create with custom configuration
    pub fn with_config(config: PersistenceConfig, root_dir: String) -> Self {
        Self {
            root_dir,
            #[cfg(target_arch = "wasm32")]
            data_file_handle: Arc::new(RwLock::new(None)),
            #[cfg(target_arch = "wasm32")]
            meta_file_handle: Arc::new(RwLock::new(None)),
            #[cfg(target_arch = "wasm32")]
            sync_handle: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            page_table: Arc::new(RwLock::new(HashMap::new())),
            free_blocks: Arc::new(RwLock::new(Vec::new())),
            next_block_id: Arc::new(AtomicU64::new(1)), // Block 0 is reserved for metadata
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            config,
        }
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Initialize OPFS Virtual Block Device
    #[cfg(target_arch = "wasm32")]
    pub async fn init(&self) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        if *self.initialized.read().unwrap_or_else(|e| e.into_inner()) {
            return Ok(());
        }

        // Get OPFS root directory
        let window = web_sys::window()
            .ok_or_else(|| PersistenceError::NotAvailable("No window object".to_string()))?;

        let navigator = window.navigator();
        let storage_manager = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| {
                PersistenceError::NotAvailable("StorageManager not available".to_string())
            })?;

        let get_directory =
            js_sys::Reflect::get(&storage_manager, &JsValue::from_str("getDirectory"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::NotAvailable("getDirectory not available".to_string())
                })?;

        let root_promise = get_directory.call0(&storage_manager).map_err(|_| {
            PersistenceError::NotAvailable("Failed to get root directory".to_string())
        })?;

        let root_dir = JsFuture::from(
            root_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::NotAvailable("Failed to access OPFS".to_string()))?;

        // Get or create database directory
        let get_directory_handle =
            js_sys::Reflect::get(&root_dir, &JsValue::from_str("getDirectoryHandle"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::NotAvailable("getDirectoryHandle not available".to_string())
                })?;

        let create_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &create_options,
            &JsValue::from_str("create"),
            &JsValue::TRUE,
        )
        .unwrap_or_default();

        let db_dir_promise = get_directory_handle
            .call2(
                &root_dir,
                &JsValue::from_str(&self.root_dir),
                &create_options,
            )
            .map_err(|_| {
                PersistenceError::NotAvailable("Failed to get database directory".to_string())
            })?;

        let db_dir = JsFuture::from(
            db_dir_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| {
            PersistenceError::NotAvailable("Failed to create database directory".to_string())
        })?;

        // Get or create data file (main block device)
        let get_file_handle = js_sys::Reflect::get(&db_dir, &JsValue::from_str("getFileHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| {
                PersistenceError::NotAvailable("getFileHandle not available".to_string())
            })?;

        let data_file_promise = get_file_handle
            .call2(&db_dir, &JsValue::from_str("data.wdb"), &create_options)
            .map_err(|_| PersistenceError::NotAvailable("Failed to get data file".to_string()))?;

        let data_file = JsFuture::from(
            data_file_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::NotAvailable("Failed to create data file".to_string()))?;

        // Get or create metadata file (page table)
        let meta_file_promise = get_file_handle
            .call2(&db_dir, &JsValue::from_str("meta.wdb"), &create_options)
            .map_err(|_| PersistenceError::NotAvailable("Failed to get meta file".to_string()))?;

        let meta_file = JsFuture::from(
            meta_file_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::NotAvailable("Failed to create meta file".to_string()))?;

        // Store handles
        {
            let mut data_handle = self.data_file_handle.write().unwrap_or_else(|e| e.into_inner());
            *data_handle = Some(data_file.clone());
        }

        {
            let mut meta_handle = self.meta_file_handle.write().unwrap_or_else(|e| e.into_inner());
            *meta_handle = Some(meta_file);
        }

        // Try to get sync access handle (for workers)
        if let Ok(sync_handle) = self.try_get_sync_handle(&data_file).await {
            let mut sync = self.sync_handle.write().unwrap_or_else(|e| e.into_inner());
            *sync = Some(sync_handle);
        }

        // Load page table from metadata file
        self.load_page_table().await?;

        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;

        Ok(())
    }

    /// Try to get sync access handle (only works in workers)
    #[cfg(target_arch = "wasm32")]
    async fn try_get_sync_handle(
        &self,
        file_handle: &JsValue,
    ) -> Result<JsValue, PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        let create_sync =
            js_sys::Reflect::get(file_handle, &JsValue::from_str("createSyncAccessHandle"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::NotAvailable(
                        "createSyncAccessHandle not available".to_string(),
                    )
                })?;

        let sync_promise = create_sync.call0(file_handle).map_err(|_| {
            PersistenceError::NotAvailable("Failed to create sync handle".to_string())
        })?;

        JsFuture::from(
            sync_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| {
            PersistenceError::NotAvailable(
                "Sync access not available (must be in worker)".to_string(),
            )
        })
    }

    /// Load page table from metadata file
    #[cfg(target_arch = "wasm32")]
    async fn load_page_table(&self) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        let meta_handle = self.meta_file_handle.read().unwrap_or_else(|e| e.into_inner());
        let meta_handle = meta_handle.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Meta file not initialized".to_string())
        })?;

        // Try to read metadata file
        let get_file = js_sys::Reflect::get(meta_handle, &JsValue::from_str("getFile"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("getFile not available".to_string()))?;

        let file_promise = get_file
            .call0(meta_handle)
            .map_err(|_| PersistenceError::IoError("Failed to get file".to_string()))?;

        let file = JsFuture::from(
            file_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to read file".to_string()))?;

        // Check file size
        let size = js_sys::Reflect::get(&file, &JsValue::from_str("size"))
            .ok()
            .and_then(|s| s.as_f64())
            .unwrap_or(0.0) as u64;

        if size == 0 {
            // New database, no page table yet
            return Ok(());
        }

        // Read metadata
        let array_buffer_func = js_sys::Reflect::get(&file, &JsValue::from_str("arrayBuffer"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("arrayBuffer not available".to_string()))?;

        let array_buffer_promise = array_buffer_func
            .call0(&file)
            .map_err(|_| PersistenceError::IoError("Failed to read".to_string()))?;

        let array_buffer = JsFuture::from(
            array_buffer_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to read data".to_string()))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let data = uint8_array.to_vec();

        // Deserialize page table
        let mut page_table = self.page_table.write().unwrap_or_else(|e| e.into_inner());
        let mut offset = 0;

        // Read next_block_id
        if offset + 8 > data.len() {
            return Ok(()); // Corrupted or empty
        }
        let next_block_id = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;
        self.next_block_id.store(next_block_id, Ordering::SeqCst);

        // Read page table size
        if offset + 8 > data.len() {
            return Ok(());
        }
        let table_size = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;

        // Read page table entries
        for _ in 0..table_size {
            if offset + 24 > data.len() {
                break; // Corrupted
            }

            // Read PageID (8 bytes)
            let page_id = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            // Read BlockAddress (16 bytes: block_id + offset)
            let block_id = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            let block_offset = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            page_table.insert(
                page_id,
                BlockAddress {
                    block_id,
                    offset: block_offset,
                },
            );
        }

        Ok(())
    }

    /// Save page table to metadata file
    #[cfg(target_arch = "wasm32")]
    pub async fn save_page_table(&self) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        let meta_handle = self.meta_file_handle.read().unwrap_or_else(|e| e.into_inner());
        let meta_handle = meta_handle.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Meta file not initialized".to_string())
        })?;

        // Serialize page table
        let page_table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        let next_block_id = self.next_block_id.load(Ordering::SeqCst);

        let mut data = Vec::new();

        // Write next_block_id (8 bytes)
        data.extend_from_slice(&next_block_id.to_le_bytes());

        // Write page table size (8 bytes)
        data.extend_from_slice(&(page_table.len() as u64).to_le_bytes());

        // Write page table entries (24 bytes each: PageID + BlockAddress)
        for (page_id, addr) in page_table.iter() {
            data.extend_from_slice(&page_id.to_le_bytes());
            data.extend_from_slice(&addr.block_id.to_le_bytes());
            data.extend_from_slice(&addr.offset.to_le_bytes());
        }

        // Write to metadata file
        let create_writable =
            js_sys::Reflect::get(meta_handle, &JsValue::from_str("createWritable"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::IoError("createWritable not available".to_string())
                })?;

        let writable_promise = create_writable
            .call0(meta_handle)
            .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

        let writable = JsFuture::from(
            writable_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

        let write_func = js_sys::Reflect::get(&writable, &JsValue::from_str("write"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("write not available".to_string()))?;

        let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(&data);

        let write_promise = write_func
            .call1(&writable, &array.buffer())
            .map_err(|_| PersistenceError::IoError("Failed to write".to_string()))?;

        JsFuture::from(
            write_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Write failed".to_string()))?;

        // Close
        let close_func = js_sys::Reflect::get(&writable, &JsValue::from_str("close"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

        if let Some(close) = close_func {
            if let Ok(close_promise) = close.call0(&writable) {
                if let Ok(promise) = close_promise.dyn_into::<js_sys::Promise>() {
                    let _ = JsFuture::from(promise).await;
                }
            }
        }

        Ok(())
    }

    /// Save page table to metadata file (static version for use from closures)
    /// Takes Arc clones so it can be called from async closures that capture Arc values
    #[cfg(target_arch = "wasm32")]
    pub async fn save_page_table_with_arcs(
        meta_file_handle: Arc<RwLock<Option<JsValue>>>,
        page_table: Arc<RwLock<HashMap<PageId, BlockAddress>>>,
        next_block_id: Arc<AtomicU64>,
    ) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        let meta_handle = meta_file_handle.read().unwrap_or_else(|e| e.into_inner());
        let meta_handle = meta_handle.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Meta file not initialized".to_string())
        })?;

        // Serialize page table
        let page_table_guard = page_table.read().unwrap_or_else(|e| e.into_inner());
        let next_block = next_block_id.load(Ordering::SeqCst);

        let mut data = Vec::new();

        // Write next_block_id (8 bytes)
        data.extend_from_slice(&next_block.to_le_bytes());

        // Write page table size (8 bytes)
        data.extend_from_slice(&(page_table_guard.len() as u64).to_le_bytes());

        // Write page table entries (24 bytes each: PageID + BlockAddress)
        for (page_id, addr) in page_table_guard.iter() {
            data.extend_from_slice(&page_id.to_le_bytes());
            data.extend_from_slice(&addr.block_id.to_le_bytes());
            data.extend_from_slice(&addr.offset.to_le_bytes());
        }

        drop(page_table_guard);

        // Write to metadata file
        let create_writable =
            js_sys::Reflect::get(meta_handle, &JsValue::from_str("createWritable"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::IoError("createWritable not available".to_string())
                })?;

        let writable_promise = create_writable
            .call0(meta_handle)
            .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

        let writable = JsFuture::from(
            writable_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

        let write_func = js_sys::Reflect::get(&writable, &JsValue::from_str("write"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("write not available".to_string()))?;

        let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(&data);

        let write_promise = write_func
            .call1(&writable, &array.buffer())
            .map_err(|_| PersistenceError::IoError("Failed to write".to_string()))?;

        JsFuture::from(
            write_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Write failed".to_string()))?;

        // Close
        let close_func = js_sys::Reflect::get(&writable, &JsValue::from_str("close"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

        if let Some(close) = close_func {
            if let Ok(close_promise) = close.call0(&writable) {
                if let Ok(promise) = close_promise.dyn_into::<js_sys::Promise>() {
                    let _ = JsFuture::from(promise).await;
                }
            }
        }

        Ok(())
    }

    /// Calculate file offset for a block address
    fn block_to_offset(&self, addr: BlockAddress) -> u64 {
        (addr.block_id * BLOCK_SIZE) + addr.offset
    }

    /// Allocate a new block
    fn allocate_block(&self) -> u64 {
        // Try to reuse a free block first
        let mut free = self.free_blocks.write().unwrap_or_else(|e| e.into_inner());
        if let Some(block_id) = free.pop() {
            return block_id;
        }
        drop(free);

        // Allocate new block
        self.next_block_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Free a block (mark for reuse)
    fn free_block(&self, block_id: u64) {
        let mut free = self.free_blocks.write().unwrap_or_else(|e| e.into_inner());
        free.push(block_id);
    }

    /// Insert into cache with eviction if needed
    ///
    /// Evicts random entries when cache exceeds config.cache_size limit.
    fn cache_insert_with_eviction(&self, page_id: PageId, data: Vec<u8>) {
        let max_size = self.config.cache_size;
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());

        // Evict entries if cache is full
        while cache.len() >= max_size && max_size > 0 {
            // Simple eviction: remove a random key
            // Note: For a true LRU, we'd need a LinkedHashMap, but this
            // approach is simpler and still bounds memory usage
            if let Some(key_to_remove) = cache.keys().next().cloned() {
                cache.remove(&key_to_remove);
            } else {
                break;
            }
        }

        cache.insert(page_id, data);
    }
}

/// Standalone cache insertion with eviction (for use in async closures)
fn cache_insert_with_eviction_fn(
    cache: &Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    max_size: usize,
    page_id: PageId,
    data: Vec<u8>,
) {
    let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());

    // Evict entries if cache is full
    while cache_guard.len() >= max_size && max_size > 0 {
        if let Some(key_to_remove) = cache_guard.keys().next().cloned() {
            cache_guard.remove(&key_to_remove);
        } else {
            break;
        }
    }

    cache_guard.insert(page_id, data);
}

impl AsyncPersistenceBackend for OPFSVirtualBlockDevice {
    #[cfg(target_arch = "wasm32")]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + 'static>,
    > {
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let page_table = Arc::clone(&self.page_table);
        let sync_handle = Arc::clone(&self.sync_handle);
        let data_file_handle = Arc::clone(&self.data_file_handle);
        let max_cache_size = self.config.cache_size;

        Box::pin(async move {
            // Check cache first
            {
                let cache_guard = cache.read().unwrap_or_else(|e| e.into_inner());
                if let Some(data) = cache_guard.get(&page_id) {
                    stats.write().unwrap_or_else(|e| e.into_inner()).cache_hits += 1;
                    return Ok(data.clone());
                }
            }
            stats.write().unwrap_or_else(|e| e.into_inner()).cache_misses += 1;

            // Look up block address
            let addr = {
                let page_table_guard = page_table.read().unwrap_or_else(|e| e.into_inner());
                *page_table_guard
                    .get(&page_id)
                    .ok_or_else(|| PersistenceError::PageNotFound(page_id))?
            };

            // Helper function to calculate offset
            let block_to_offset =
                |addr: BlockAddress| -> u64 { addr.block_id * BLOCK_SIZE + addr.offset };

            // Check if we have sync handle (worker context)
            let sync_handle_guard = sync_handle.read().unwrap_or_else(|e| e.into_inner());
            if let Some(sync) = sync_handle_guard.as_ref() {
                // Fast synchronous read
                let offset = block_to_offset(addr);
                let mut buffer = vec![0u8; DEFAULT_PAGE_SIZE];

                let read_func =
                    js_sys::Reflect::get(sync, &wasm_bindgen::JsValue::from_str("read"))
                        .ok()
                        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                        .ok_or_else(|| {
                            PersistenceError::IoError("read not available".to_string())
                        })?;

                let read_options = js_sys::Object::new();
                js_sys::Reflect::set(
                    &read_options,
                    &wasm_bindgen::JsValue::from_str("at"),
                    &wasm_bindgen::JsValue::from_f64(offset as f64),
                )
                .unwrap_or_default();

                read_func
                    .call2(sync, &js_sys::Uint8Array::from(&buffer[..]), &read_options)
                    .map_err(|_| PersistenceError::IoError("Read failed".to_string()))?;

                // Update cache with eviction
                cache_insert_with_eviction_fn(&cache, max_cache_size, page_id, buffer.clone());

                stats.write().unwrap_or_else(|e| e.into_inner()).read_ops += 1;
                return Ok(buffer);
            }

            // Fallback to async read (main thread)
            use wasm_bindgen_futures::JsFuture;

            let data_handle_guard = data_file_handle.read().unwrap_or_else(|e| e.into_inner());
            let data_handle_val = data_handle_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Data file not initialized".to_string())
            })?;

            let offset = block_to_offset(addr);

            // Get file
            let get_file =
                js_sys::Reflect::get(data_handle_val, &wasm_bindgen::JsValue::from_str("getFile"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| {
                        PersistenceError::IoError("getFile not available".to_string())
                    })?;

            let file_promise = get_file
                .call0(data_handle_val)
                .map_err(|_| PersistenceError::IoError("Failed to get file".to_string()))?;

            let file = JsFuture::from(
                file_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to read file".to_string()))?;

            // Read slice using Blob.slice
            let slice_func = js_sys::Reflect::get(&file, &wasm_bindgen::JsValue::from_str("slice"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| PersistenceError::IoError("slice not available".to_string()))?;

            let slice_promise = slice_func
                .call2(
                    &file,
                    &wasm_bindgen::JsValue::from_f64(offset as f64),
                    &wasm_bindgen::JsValue::from_f64((offset + DEFAULT_PAGE_SIZE as u64) as f64),
                )
                .map_err(|_| PersistenceError::IoError("Failed to slice".to_string()))?;

            let blob = JsFuture::from(
                slice_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to slice file".to_string()))?;

            // Convert to ArrayBuffer
            let array_buffer_func =
                js_sys::Reflect::get(&blob, &wasm_bindgen::JsValue::from_str("arrayBuffer"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| {
                        PersistenceError::IoError("arrayBuffer not available".to_string())
                    })?;

            let array_buffer_promise = array_buffer_func
                .call0(&blob)
                .map_err(|_| PersistenceError::IoError("Failed to read".to_string()))?;

            let array_buffer = JsFuture::from(
                array_buffer_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to read data".to_string()))?;

            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let mut data = vec![0u8; uint8_array.length() as usize];
            uint8_array.copy_to(&mut data);

            // Update cache
            {
                let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());
                cache_guard.insert(page_id, data.clone());
            }

            stats.write().unwrap_or_else(|e| e.into_inner()).read_ops += 1;
            Ok(data)
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + Send + 'static>,
    > {
        let cache = Arc::clone(&self.cache);
        Box::pin(async move {
            let cache_guard = cache.read().unwrap_or_else(|e| e.into_inner());
            cache_guard
                .get(&page_id)
                .cloned()
                .ok_or(PersistenceError::PageNotFound(page_id))
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>
    {
        let data = data.to_vec();
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let page_table = Arc::clone(&self.page_table);
        let free_blocks = Arc::clone(&self.free_blocks);
        let next_block_id = Arc::clone(&self.next_block_id);
        let sync_handle = Arc::clone(&self.sync_handle);
        let data_file_handle = Arc::clone(&self.data_file_handle);
        let meta_file_handle = Arc::clone(&self.meta_file_handle);

        Box::pin(async move {
            // Allocate or get block address
            let addr = {
                let mut page_table_guard = page_table.write().unwrap_or_else(|e| e.into_inner());
                if let Some(existing) = page_table_guard.get(&page_id) {
                    *existing // Reuse existing block
                } else {
                    // Allocate new block
                    let block_id = {
                        let mut free = free_blocks.write().unwrap_or_else(|e| e.into_inner());
                        if let Some(id) = free.pop() {
                            id
                        } else {
                            next_block_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        }
                    };
                    let addr = BlockAddress {
                        block_id,
                        offset: 0, // For now, one page per block
                    };
                    page_table_guard.insert(page_id, addr);
                    addr
                }
            };

            // Calculate offset: block_id * BLOCK_SIZE + offset
            let offset = addr.block_id * BLOCK_SIZE as u64 + addr.offset;

            // Check if we have sync handle (worker context)
            let sync_handle_guard = sync_handle.read().unwrap_or_else(|e| e.into_inner());
            if let Some(sync) = sync_handle_guard.as_ref() {
                // Fast synchronous write
                let write_func =
                    js_sys::Reflect::get(sync, &wasm_bindgen::JsValue::from_str("write"))
                        .ok()
                        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                        .ok_or_else(|| {
                            PersistenceError::IoError("write not available".to_string())
                        })?;

                let write_options = js_sys::Object::new();
                js_sys::Reflect::set(
                    &write_options,
                    &wasm_bindgen::JsValue::from_str("at"),
                    &wasm_bindgen::JsValue::from_f64(offset as f64),
                )
                .unwrap_or_default();

                let array = js_sys::Uint8Array::from(data.as_slice());
                write_func
                    .call2(sync, &array, &write_options)
                    .map_err(|_| PersistenceError::IoError("Write failed".to_string()))?;

                // Update cache
                {
                    let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());
                    cache_guard.insert(page_id, data.clone());
                }

                // Save page table periodically (every 100 writes or on sync)
                {
                    let mut stats_guard = stats.write().unwrap_or_else(|e| e.into_inner());
                    stats_guard.write_ops += 1;
                    let should_save = stats_guard.write_ops % 100 == 0;
                    drop(stats_guard);
                    if should_save {
                        // Save page table using static method with Arc clones
                        let _ = Self::save_page_table_with_arcs(
                            Arc::clone(&meta_file_handle),
                            Arc::clone(&page_table),
                            Arc::clone(&next_block_id),
                        )
                        .await;
                    }
                }

                return Ok(());
            }

            // Fallback to async write (main thread)
            use wasm_bindgen_futures::JsFuture;

            let data_handle_guard = data_file_handle.read().unwrap_or_else(|e| e.into_inner());
            let data_handle = data_handle_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Data file not initialized".to_string())
            })?;

            // Create writable stream
            let create_writable = js_sys::Reflect::get(
                data_handle,
                &wasm_bindgen::JsValue::from_str("createWritable"),
            )
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("createWritable not available".to_string()))?;

            let writable_promise = create_writable
                .call0(data_handle)
                .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

            let writable = JsFuture::from(
                writable_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

            // Seek to offset
            let seek_func =
                js_sys::Reflect::get(&writable, &wasm_bindgen::JsValue::from_str("seek"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

            if let Some(seek) = seek_func {
                seek.call1(&writable, &wasm_bindgen::JsValue::from_f64(offset as f64))
                    .ok();
            }

            // Write data
            let write_func =
                js_sys::Reflect::get(&writable, &wasm_bindgen::JsValue::from_str("write"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| PersistenceError::IoError("write not available".to_string()))?;

            let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
            array.copy_from(&data);

            let write_promise = write_func
                .call1(&writable, &array.buffer())
                .map_err(|_| PersistenceError::IoError("Failed to write".to_string()))?;

            JsFuture::from(
                write_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Write failed".to_string()))?;

            // Close
            let close_func =
                js_sys::Reflect::get(&writable, &wasm_bindgen::JsValue::from_str("close"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

            if let Some(close) = close_func {
                if let Ok(close_promise) = close.call0(&writable) {
                    if let Ok(promise) = close_promise.dyn_into::<js_sys::Promise>() {
                        let _ = JsFuture::from(promise).await;
                    }
                }
            }

            // Update cache
            {
                let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());
                cache_guard.insert(page_id, data.clone());
            }

            // Save page table periodically
            {
                let mut stats_guard = stats.write().unwrap_or_else(|e| e.into_inner());
                stats_guard.write_ops += 1;
                let should_save = stats_guard.write_ops % 100 == 0;
                drop(stats_guard);
                if should_save {
                    // Save page table using static method with Arc clones
                    let _ = Self::save_page_table_with_arcs(
                        Arc::clone(&meta_file_handle),
                        Arc::clone(&page_table),
                        Arc::clone(&next_block_id),
                    )
                    .await;
                }
            }

            Ok(())
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    > {
        let data = data.to_vec();
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());
            cache_guard.insert(page_id, data.clone());
            stats.write().unwrap_or_else(|e| e.into_inner()).write_ops += 1;
            Ok(())
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn sync(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>
    {
        let page_table = Arc::clone(&self.page_table);
        let meta_file_handle = Arc::clone(&self.meta_file_handle);
        let sync_handle = Arc::clone(&self.sync_handle);
        let stats = Arc::clone(&self.stats);
        let next_block_id = Arc::clone(&self.next_block_id);

        Box::pin(async move {
            // Save page table on sync
            {
                use wasm_bindgen::JsValue;
                use wasm_bindgen_futures::JsFuture;

                // Serialize page table
                let page_table_guard = page_table.read().unwrap_or_else(|e| e.into_inner());
                let mut data = Vec::new();

                // Write next_block_id (8 bytes)
                let next_id = next_block_id.load(std::sync::atomic::Ordering::SeqCst);
                data.extend_from_slice(&next_id.to_le_bytes());

                // Write page table size (8 bytes)
                let table_size = page_table_guard.len() as u64;
                data.extend_from_slice(&table_size.to_le_bytes());

                // Write page table entries (24 bytes each: PageID + BlockAddress)
                for (&page_id, &addr) in page_table_guard.iter() {
                    data.extend_from_slice(&(page_id as u64).to_le_bytes());
                    data.extend_from_slice(&addr.block_id.to_le_bytes());
                    data.extend_from_slice(&addr.offset.to_le_bytes());
                }
                drop(page_table_guard);

                // Write to metadata file
                let meta_handle_guard = meta_file_handle.read().unwrap_or_else(|e| e.into_inner());
                if let Some(meta_handle) = meta_handle_guard.as_ref() {
                    let create_writable =
                        js_sys::Reflect::get(meta_handle, &JsValue::from_str("createWritable"))
                            .ok()
                            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

                    if let Some(create) = create_writable {
                        if let Ok(writable_promise) = create.call0(meta_handle) {
                            if let Ok(promise) = writable_promise.dyn_into::<js_sys::Promise>() {
                                if let Ok(writable) = JsFuture::from(promise).await {
                                    let write_func = js_sys::Reflect::get(
                                        &writable,
                                        &JsValue::from_str("write"),
                                    )
                                    .ok()
                                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

                                    if let Some(write) = write_func {
                                        let array = js_sys::Uint8Array::from(data.as_slice());
                                        if let Ok(write_promise) =
                                            write.call1(&writable, &array.buffer())
                                        {
                                            if let Ok(p) =
                                                write_promise.dyn_into::<js_sys::Promise>()
                                            {
                                                let _ = JsFuture::from(p).await;
                                            }
                                        }

                                        let close_func = js_sys::Reflect::get(
                                            &writable,
                                            &JsValue::from_str("close"),
                                        )
                                        .ok()
                                        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

                                        if let Some(close) = close_func {
                                            if let Ok(close_promise) = close.call0(&writable) {
                                                if let Ok(p) =
                                                    close_promise.dyn_into::<js_sys::Promise>()
                                                {
                                                    let _ = JsFuture::from(p).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Sync data file if sync handle available
            {
                let sync_handle_guard = sync_handle.read().unwrap_or_else(|e| e.into_inner());
                if let Some(sync) = sync_handle_guard.as_ref() {
                    let flush_func =
                        js_sys::Reflect::get(sync, &wasm_bindgen::JsValue::from_str("flush"))
                            .ok()
                            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

                    if let Some(flush) = flush_func {
                        flush.call0(sync).ok();
                    }
                }
            }

            stats.write().unwrap_or_else(|e| e.into_inner()).sync_ops += 1;
            Ok(())
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sync(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + Send + 'static>,
    > {
        Box::pin(async move { Ok(()) })
    }

    fn stats(&self) -> StorageStats {
        let page_table = self.page_table.read().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.read().unwrap_or_else(|e| e.into_inner()).clone();
        stats.total_pages = page_table.len() as u64;
        stats
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl OPFSVirtualBlockDevice {
    pub async fn init(&self) -> Result<(), PersistenceError> {
        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(())
    }
}
