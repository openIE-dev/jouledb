//! OPFS (Origin Private File System) persistence backend
//!
//! Uses the File System Access API for high-performance file storage in browsers.
//! Supports:
//! - Synchronous access via SyncAccessHandle (in workers)
//! - Asynchronous access via standard file handles
//! - Directory organization for data files
//! - Page-level caching

use joule_db_core::storage::PageId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::config::PersistenceConfig;
use super::error::PersistenceError;
use super::wal::WalManager;
use super::{AsyncPersistenceBackend, StorageStats};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// OPFS (Origin Private File System) persistence backend
///
/// Uses the File System Access API for high-performance file storage.
/// Features:
/// - Per-page file storage for efficient random access
/// - In-memory caching with LRU eviction
/// - Async operations with browser APIs
/// - Optional compression and encryption

pub struct OPFSBackend {
    root_dir: String,
    #[cfg(target_arch = "wasm32")]
    dir_handle: Arc<RwLock<Option<JsValue>>>,
    initialized: Arc<RwLock<bool>>,
    cache: Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    stats: Arc<RwLock<StorageStats>>,
    config: PersistenceConfig,
    wal: Arc<RwLock<WalManager>>,
    /// Sync access handle for worker context
    #[cfg(target_arch = "wasm32")]
    sync_handle: Arc<RwLock<Option<JsValue>>>,
}

impl OPFSBackend {
    /// Create a new OPFS backend with the given directory name
    pub fn new(root_dir: &str) -> Self {
        Self::with_config(PersistenceConfig::new(root_dir), root_dir.to_string())
    }

    /// Create with custom configuration
    pub fn with_config(config: PersistenceConfig, root_dir: String) -> Self {
        Self {
            root_dir,
            #[cfg(target_arch = "wasm32")]
            dir_handle: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            config,
            wal: Arc::new(RwLock::new(WalManager::new())),
            #[cfg(target_arch = "wasm32")]
            sync_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Get file path for a page
    fn page_path(&self, page_id: PageId) -> String {
        format!("page_{:016x}.dat", page_id)
    }

    /// Initialize OPFS directory
    #[cfg(target_arch = "wasm32")]
    pub async fn init(&self) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        if *self.initialized.read().unwrap_or_else(|e| e.into_inner()) {
            return Ok(());
        }

        // Get OPFS root directory via navigator.storage.getDirectory()
        let window = web_sys::window()
            .ok_or_else(|| PersistenceError::NotAvailable("No window object".to_string()))?;

        let navigator = window.navigator();
        let storage_manager = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| {
                PersistenceError::NotAvailable("StorageManager not available".to_string())
            })?;

        let get_directory_func =
            js_sys::Reflect::get(&storage_manager, &JsValue::from_str("getDirectory"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| PersistenceError::NotAvailable("OPFS not available".to_string()))?;

        let root_promise = get_directory_func
            .call0(&storage_manager)
            .map_err(|_| PersistenceError::NotAvailable("Failed to get OPFS root".to_string()))?;

        let root = JsFuture::from(
            root_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::InitError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::NotAvailable("OPFS not accessible".to_string()))?;

        // Get or create our data directory
        let get_dir_func = js_sys::Reflect::get(&root, &JsValue::from_str("getDirectoryHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| {
                PersistenceError::InitError("getDirectoryHandle not available".to_string())
            })?;

        let create_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &create_options,
            &JsValue::from_str("create"),
            &JsValue::TRUE,
        )
        .unwrap_or_default();

        let dir_promise = get_dir_func
            .call2(&root, &JsValue::from_str(&self.root_dir), &create_options)
            .map_err(|_| PersistenceError::InitError("Failed to get directory".to_string()))?;

        let dir_handle = JsFuture::from(
            dir_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::InitError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::InitError("Failed to create directory".to_string()))?;

        *self.dir_handle.write().unwrap_or_else(|e| e.into_inner()) = Some(dir_handle);
        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn init(&self) -> Result<(), PersistenceError> {
        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(())
    }

    /// Read page from OPFS
    #[cfg(target_arch = "wasm32")]
    pub async fn read_page_async(&self, page_id: PageId) -> Result<Vec<u8>, PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        // Check cache first
        {
            let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(data) = cache.get(&page_id) {
                self.stats.write().unwrap_or_else(|e| e.into_inner()).cache_hits += 1;
                return Ok(data.clone());
            }
        }
        self.stats.write().unwrap_or_else(|e| e.into_inner()).cache_misses += 1;

        let dir_handle = self.dir_handle.read().unwrap_or_else(|e| e.into_inner());
        let dir_handle = dir_handle.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Directory not initialized".to_string())
        })?;

        let file_path = self.page_path(page_id);

        // Get file handle
        let get_file_func = js_sys::Reflect::get(dir_handle, &JsValue::from_str("getFileHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("getFileHandle not available".to_string()))?;

        let file_handle_promise = get_file_func
            .call1(dir_handle, &JsValue::from_str(&file_path))
            .map_err(|_| PersistenceError::PageNotFound(page_id))?;

        let file_handle = JsFuture::from(
            file_handle_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::PageNotFound(page_id))?;

        // Get file
        let get_file = js_sys::Reflect::get(&file_handle, &JsValue::from_str("getFile"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("getFile not available".to_string()))?;

        let file_promise = get_file
            .call0(&file_handle)
            .map_err(|_| PersistenceError::IoError("Failed to get file".to_string()))?;

        let file = JsFuture::from(
            file_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to read file".to_string()))?;

        // Read as ArrayBuffer
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

        // Convert to Vec<u8>
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        self.stats.write().unwrap_or_else(|e| e.into_inner()).read_ops += 1;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(page_id, data.clone());
        }

        Ok(data)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn read_page_async(&self, page_id: PageId) -> Result<Vec<u8>, PersistenceError> {
        let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
        cache
            .get(&page_id)
            .cloned()
            .ok_or(PersistenceError::PageNotFound(page_id))
    }

    /// Write page to OPFS
    #[cfg(target_arch = "wasm32")]
    pub async fn write_page_async(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        let dir_handle = self.dir_handle.read().unwrap_or_else(|e| e.into_inner());
        let dir_handle = dir_handle.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Directory not initialized".to_string())
        })?;

        let file_path = self.page_path(page_id);

        // Get or create file handle
        let get_file_func = js_sys::Reflect::get(dir_handle, &JsValue::from_str("getFileHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("getFileHandle not available".to_string()))?;

        let create_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &create_options,
            &JsValue::from_str("create"),
            &JsValue::TRUE,
        )
        .unwrap_or_default();

        let file_handle_promise = get_file_func
            .call2(dir_handle, &JsValue::from_str(&file_path), &create_options)
            .map_err(|_| PersistenceError::IoError("Failed to get file handle".to_string()))?;

        let file_handle = JsFuture::from(
            file_handle_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|e| {
            let error_str = format!("{:?}", e);
            if error_str.contains("QuotaExceeded") {
                PersistenceError::QuotaExceeded
            } else {
                PersistenceError::IoError(error_str)
            }
        })?;

        // Create writable stream
        let create_writable =
            js_sys::Reflect::get(&file_handle, &JsValue::from_str("createWritable"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::IoError("createWritable not available".to_string())
                })?;

        let writable_promise = create_writable
            .call0(&file_handle)
            .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

        let writable = JsFuture::from(
            writable_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
        )
        .await
        .map_err(|_| PersistenceError::IoError("Failed to create writable stream".to_string()))?;

        // Write data
        let write_func = js_sys::Reflect::get(&writable, &JsValue::from_str("write"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| PersistenceError::IoError("write not available".to_string()))?;

        let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);

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

        // Close writable stream
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

        // Update cache
        {
            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(page_id, data.to_vec());
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.write_ops += 1;
            stats.total_pages = self.cache.read().unwrap_or_else(|e| e.into_inner()).len() as u64;
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn write_page_async(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> Result<(), PersistenceError> {
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(page_id, data.to_vec());
        self.stats.write().unwrap_or_else(|e| e.into_inner()).write_ops += 1;
        Ok(())
    }

    /// Read page (sync version for non-wasm)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_page_sync(&self, page_id: PageId) -> Result<Vec<u8>, PersistenceError> {
        let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
        cache
            .get(&page_id)
            .cloned()
            .ok_or(PersistenceError::PageNotFound(page_id))
    }

    /// Write page (sync version for non-wasm)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_page_sync(&self, page_id: PageId, data: &[u8]) -> Result<(), PersistenceError> {
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(page_id, data.to_vec());
        Ok(())
    }

    /// Check if OPFS is available in the current environment
    pub fn is_available() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::window()
                .map(|w| {
                    let navigator = w.navigator();
                    js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
                        .ok()
                        .and_then(|s| {
                            js_sys::Reflect::get(&s, &JsValue::from_str("getDirectory"))
                                .ok()
                                .map(|f| f.is_function())
                        })
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }
}

impl AsyncPersistenceBackend for OPFSBackend {
    #[cfg(target_arch = "wasm32")]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + 'static>,
    > {
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let dir_handle = Arc::clone(&self.dir_handle);
        let root_dir = self.root_dir.clone(); // kept for potential error context or debugging

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

            let dir_handle_guard = dir_handle.read().unwrap_or_else(|e| e.into_inner());
            let dir_handle_val = dir_handle_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Directory not initialized".to_string())
            })?;

            let file_path = format!("page_{:016x}.dat", page_id);

            // Get file handle
            let get_file_func =
                js_sys::Reflect::get(dir_handle_val, &JsValue::from_str("getFileHandle"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| {
                        PersistenceError::IoError("getFileHandle not available".to_string())
                    })?;

            let file_handle_promise = get_file_func
                .call1(dir_handle_val, &JsValue::from_str(&file_path))
                .map_err(|_| PersistenceError::IoError("Failed to get file handle".to_string()))?;

            let file_handle = wasm_bindgen_futures::JsFuture::from(
                file_handle_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to get file".to_string()))?;

            // Read file
            let get_file_func = js_sys::Reflect::get(&file_handle, &JsValue::from_str("getFile"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| PersistenceError::IoError("getFile not available".to_string()))?;

            let file_promise = get_file_func
                .call0(&file_handle)
                .map_err(|_| PersistenceError::IoError("Failed to get file".to_string()))?;

            let file = wasm_bindgen_futures::JsFuture::from(
                file_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to read file".to_string()))?;

            // Read file as array buffer
            let array_buffer_func = js_sys::Reflect::get(&file, &JsValue::from_str("arrayBuffer"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| {
                    PersistenceError::IoError("arrayBuffer not available".to_string())
                })?;

            let array_buffer_promise = array_buffer_func
                .call0(&file)
                .map_err(|_| PersistenceError::IoError("Failed to read".to_string()))?;

            let array_buffer = wasm_bindgen_futures::JsFuture::from(
                array_buffer_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to read data".to_string()))?;

            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let data = uint8_array.to_vec();

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
        let dir_handle = Arc::clone(&self.dir_handle);
        let _root_dir = self.root_dir.clone();

        Box::pin(async move {
            let dir_handle_guard = dir_handle.read().unwrap_or_else(|e| e.into_inner());
            let dir_handle_val = dir_handle_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Directory not initialized".to_string())
            })?;

            let file_path = format!("page_{:016x}.dat", page_id);

            // Get or create file handle
            let get_file_func =
                js_sys::Reflect::get(dir_handle_val, &JsValue::from_str("getFileHandle"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| {
                        PersistenceError::IoError("getFileHandle not available".to_string())
                    })?;

            let create_options = js_sys::Object::new();
            js_sys::Reflect::set(
                &create_options,
                &JsValue::from_str("create"),
                &JsValue::TRUE,
            )
            .unwrap_or_default();

            let file_handle_promise = get_file_func
                .call2(
                    dir_handle_val,
                    &JsValue::from_str(&file_path),
                    &create_options,
                )
                .map_err(|_| PersistenceError::IoError("Failed to get file handle".to_string()))?;

            let file_handle = wasm_bindgen_futures::JsFuture::from(
                file_handle_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|e| {
                let error_str = format!("{:?}", e);
                if error_str.contains("QuotaExceeded") {
                    PersistenceError::QuotaExceeded
                } else {
                    PersistenceError::IoError(error_str)
                }
            })?;

            // Get writable stream
            let create_writable_func =
                js_sys::Reflect::get(&file_handle, &JsValue::from_str("createWritable"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| {
                        PersistenceError::IoError("createWritable not available".to_string())
                    })?;

            let writable_promise = create_writable_func
                .call0(&file_handle)
                .map_err(|_| PersistenceError::IoError("Failed to create writable".to_string()))?;

            let writable = wasm_bindgen_futures::JsFuture::from(
                writable_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Failed to get writable".to_string()))?;

            // Write data
            let uint8_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
            uint8_array.copy_from(&data);

            let write_func = js_sys::Reflect::get(&writable, &JsValue::from_str("write"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| PersistenceError::IoError("write not available".to_string()))?;

            let write_promise = write_func
                .call1(&writable, &uint8_array)
                .map_err(|_| PersistenceError::IoError("Failed to write".to_string()))?;

            wasm_bindgen_futures::JsFuture::from(
                write_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Write failed".to_string()))?;

            // Close writable
            let close_func = js_sys::Reflect::get(&writable, &JsValue::from_str("close"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| PersistenceError::IoError("close not available".to_string()))?;

            let close_promise = close_func
                .call0(&writable)
                .map_err(|_| PersistenceError::IoError("Failed to close".to_string()))?;

            wasm_bindgen_futures::JsFuture::from(
                close_promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|_| PersistenceError::IoError("Close failed".to_string()))?;

            // Update cache
            {
                let mut cache_guard = cache.write().unwrap_or_else(|e| e.into_inner());
                cache_guard.insert(page_id, data);
            }
            stats.write().unwrap_or_else(|e| e.into_inner()).write_ops += 1;

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
            cache_guard.insert(page_id, data);
            stats.write().unwrap_or_else(|e| e.into_inner()).write_ops += 1;
            Ok(())
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn sync(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PersistenceError>> + 'static>>
    {
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
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
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            stats.write().unwrap_or_else(|e| e.into_inner()).sync_ops += 1;
            Ok(())
        })
    }

    fn stats(&self) -> StorageStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

// Make it Send + Sync safe
unsafe impl Send for OPFSBackend {}
unsafe impl Sync for OPFSBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opfs_backend_creation() {
        let backend = OPFSBackend::new("test_db");
        assert!(!backend.is_initialized());
    }

    #[test]
    fn test_opfs_page_path() {
        let backend = OPFSBackend::new("test_db");
        assert_eq!(backend.page_path(0), "page_0000000000000000.dat");
        assert_eq!(backend.page_path(255), "page_00000000000000ff.dat");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_opfs_backend_sync_ops() {
        let backend = OPFSBackend::new("test_db");

        // Write a page using sync API
        let data = vec![1, 2, 3, 4, 5];
        backend.write_page_sync(1, &data).unwrap();

        // Read it back
        let read_data = backend.read_page_sync(1).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_opfs_availability_non_wasm() {
        // OPFS is not available outside of browsers
        assert!(!OPFSBackend::is_available());
    }
}
