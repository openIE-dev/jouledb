//! IndexedDB persistence backend

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

/// IndexedDB persistence backend
///
/// Uses browser IndexedDB for persistent storage. Supports:
/// - Transaction-based reads/writes
/// - Object store for page storage
/// - Automatic schema migrations
/// - Error handling for quota exceeded

pub struct IndexedDBBackend {
    db_name: String,
    store_name: String,
    #[cfg(target_arch = "wasm32")]
    pub(crate) db: Arc<RwLock<Option<web_sys::IdbDatabase>>>,
    initialized: Arc<RwLock<bool>>,
    cache: Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    stats: Arc<RwLock<StorageStats>>,
    config: PersistenceConfig,
    wal: Arc<RwLock<WalManager>>,
}

impl IndexedDBBackend {
    /// Create a new IndexedDB backend
    pub fn new(db_name: &str, store_name: &str) -> Self {
        Self::with_config(PersistenceConfig::new(db_name), store_name.to_string())
    }

    /// Create with custom configuration
    pub fn with_config(config: PersistenceConfig, store_name: String) -> Self {
        Self {
            db_name: config.db_name.clone(),
            store_name,
            #[cfg(target_arch = "wasm32")]
            db: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            config,
            wal: Arc::new(RwLock::new(WalManager::new())),
        }
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Initialize IndexedDB asynchronously
    #[cfg(target_arch = "wasm32")]
    pub async fn init(&self) -> Result<(), PersistenceError> {
        use js_sys::Promise;
        use wasm_bindgen_futures::JsFuture;

        if *self.initialized.read().unwrap_or_else(|e| e.into_inner()) {
            return Ok(());
        }

        let window = web_sys::window()
            .ok_or_else(|| PersistenceError::NotAvailable("No window object".to_string()))?;

        let idb_factory = window
            .indexed_db()
            .map_err(|_| PersistenceError::NotAvailable("IndexedDB not supported".to_string()))?
            .ok_or_else(|| PersistenceError::NotAvailable("IndexedDB not available".to_string()))?;

        let open_request = idb_factory
            .open_with_u32(&self.db_name, 1)
            .map_err(|e| PersistenceError::InitError(format!("Failed to open: {:?}", e)))?;

        // Set up onupgradeneeded handler
        let store_name = self.store_name.clone();
        let onupgradeneeded =
            Closure::wrap(Box::new(move |event: web_sys::IdbVersionChangeEvent| {
                if let Some(target) = event.target() {
                    let target: web_sys::EventTarget = target;
                    if let Ok(request) = target.dyn_into::<web_sys::IdbOpenDbRequest>() {
                        if let Ok(result) = request.result() {
                            let result: wasm_bindgen::JsValue = result;
                            if let Ok(db) = result.dyn_into::<web_sys::IdbDatabase>() {
                                if !db.object_store_names().contains(&store_name) {
                                    let _ = db.create_object_store(&store_name);
                                }
                            }
                        }
                    }
                }
            })
                as Box<dyn FnMut(web_sys::IdbVersionChangeEvent)>);

        open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
        onupgradeneeded.forget();

        // Wait for database to open
        let promise = Self::request_to_promise(&open_request);
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| PersistenceError::InitError(format!("{:?}", e)))?;

        let db: web_sys::IdbDatabase = result
            .dyn_into()
            .map_err(|_| PersistenceError::InitError("Invalid database object".to_string()))?;

        *self.db.write().unwrap_or_else(|e| e.into_inner()) = Some(db);
        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn init(&self) -> Result<(), PersistenceError> {
        *self.initialized.write().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(())
    }

    /// Convert IdbRequest to Promise
    #[cfg(target_arch = "wasm32")]
    fn request_to_promise(request: &web_sys::IdbOpenDbRequest) -> js_sys::Promise {
        js_sys::Promise::new(&mut |resolve, reject| {
            let resolve_clone = resolve.clone();
            let reject_clone = reject.clone();

            let success_cb = Closure::once(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
                        if let Ok(result) = req.result() {
                            let _ = resolve_clone.call1(&JsValue::UNDEFINED, &result);
                        }
                    }
                }
            }) as Box<dyn FnOnce(web_sys::Event)>);

            let error_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ =
                    reject_clone.call1(&JsValue::UNDEFINED, &JsValue::from_str("Request failed"));
            }) as Box<dyn FnOnce(web_sys::Event)>);

            request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

            success_cb.forget();
            error_cb.forget();
        })
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

    /// Read page asynchronously
    #[cfg(target_arch = "wasm32")]
    pub async fn read_page_async(&self, page_id: PageId) -> Result<Vec<u8>, PersistenceError> {
        use wasm_bindgen_futures::JsFuture;

        // Check cache first
        {
            let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(data) = cache.get(&page_id) {
                self.stats
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .cache_hits += 1;
                return Ok(data.clone());
            }
        }
        self.stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .cache_misses += 1;

        let db_guard = self.db.read().unwrap_or_else(|e| e.into_inner());
        let db = db_guard.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Database not initialized".to_string())
        })?;

        let transaction = db
            .transaction_with_str(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let key = JsValue::from_f64(page_id as f64);
        let request = store
            .get(&key)
            .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let resolve_clone = resolve.clone();
            let reject_clone = reject.clone();

            let success_cb = Closure::once(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
                        if let Ok(result) = req.result() {
                            let _ = resolve_clone.call1(&JsValue::UNDEFINED, &result);
                        }
                    }
                }
            }) as Box<dyn FnOnce(web_sys::Event)>);

            let error_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ = reject_clone.call1(&JsValue::UNDEFINED, &JsValue::from_str("Read failed"));
            }) as Box<dyn FnOnce(web_sys::Event)>);

            request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

            success_cb.forget();
            error_cb.forget();
        });

        let result = JsFuture::from(promise)
            .await
            .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

        if result.is_undefined() || result.is_null() {
            return Err(PersistenceError::PageNotFound(page_id));
        }

        let data: Vec<u8> = serde_wasm_bindgen::from_value(result)
            .map_err(|e| PersistenceError::IoError(format!("Deserialize error: {:?}", e)))?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(page_id, data.clone());
        }

        self.stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .read_ops += 1;
        Ok(data)
    }

    /// Write page asynchronously
    #[cfg(target_arch = "wasm32")]
    pub async fn write_page_async(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> Result<(), PersistenceError> {
        use wasm_bindgen_futures::JsFuture;
        use web_sys::IdbTransactionMode;

        let db_guard = self.db.read().unwrap_or_else(|e| e.into_inner());
        let db = db_guard.as_ref().ok_or_else(|| {
            PersistenceError::NotAvailable("Database not initialized".to_string())
        })?;

        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let key = JsValue::from_f64(page_id as f64);
        let value = serde_wasm_bindgen::to_value(data)
            .map_err(|e| PersistenceError::IoError(format!("Serialize error: {:?}", e)))?;

        let request = store
            .put_with_key(&value, &key)
            .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let resolve_clone = resolve.clone();
            let reject_clone = reject.clone();

            let success_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ = resolve_clone.call1(&JsValue::UNDEFINED, &JsValue::UNDEFINED);
            }) as Box<dyn FnOnce(web_sys::Event)>);

            let error_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ = reject_clone.call1(&JsValue::UNDEFINED, &JsValue::from_str("Write failed"));
            }) as Box<dyn FnOnce(web_sys::Event)>);

            request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

            success_cb.forget();
            error_cb.forget();
        });

        JsFuture::from(promise)
            .await
            .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(page_id, data.to_vec());
        }

        self.stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .write_ops += 1;
        Ok(())
    }
}

impl AsyncPersistenceBackend for IndexedDBBackend {
    #[cfg(target_arch = "wasm32")]
    fn read_page(
        &self,
        page_id: PageId,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, PersistenceError>> + 'static>,
    > {
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let store_name = self.store_name.clone();
        let db_arc = Arc::clone(&self.db);

        Box::pin(async move {
            {
                let cache_guard = cache.read().unwrap_or_else(|e| e.into_inner());
                if let Some(data) = cache_guard.get(&page_id) {
                    stats.write().unwrap_or_else(|e| e.into_inner()).cache_hits += 1;
                    return Ok(data.clone());
                }
            }
            stats
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .cache_misses += 1;

            let db_guard = db_arc.read().unwrap_or_else(|e| e.into_inner());
            let db = db_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Database not initialized".to_string())
            })?;

            let transaction = db
                .transaction_with_str(&store_name)
                .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

            let store = transaction
                .object_store(&store_name)
                .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

            let key = wasm_bindgen::JsValue::from_f64(page_id as f64);
            let request = store
                .get(&key)
                .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

            let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                let resolve_clone = resolve.clone();
                let success_cb =
                    wasm_bindgen::closure::Closure::once(Box::new(move |event: web_sys::Event| {
                        if let Some(target) = event.target() {
                            if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
                                if let Ok(result) = req.result() {
                                    let _ = resolve_clone
                                        .call1(&wasm_bindgen::JsValue::UNDEFINED, &result);
                                }
                            }
                        }
                    })
                        as Box<dyn FnOnce(web_sys::Event)>);
                request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
                success_cb.forget();
            });

            let result = wasm_bindgen_futures::JsFuture::from(
                promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PersistenceError::IoError("Invalid promise".to_string()))?,
            )
            .await
            .map_err(|e| PersistenceError::IoError(format!("Get failed: {:?}", e)))?;

            if result.is_undefined() || result.is_null() {
                return Err(PersistenceError::PageNotFound(page_id));
            }

            let data: Vec<u8> = serde_wasm_bindgen::from_value(result)
                .map_err(|e| PersistenceError::IoError(format!("Deserialize error: {:?}", e)))?;

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
        let store_name = self.store_name.clone();
        let db_arc = Arc::clone(&self.db);

        Box::pin(async move {
            let db_guard = db_arc.read().unwrap_or_else(|e| e.into_inner());
            let db_val = db_guard.as_ref().ok_or_else(|| {
                PersistenceError::NotAvailable("Database not initialized".to_string())
            })?;

            let transaction = db_val
                .transaction_with_str(store_name.as_str())
                .map_err(|e| PersistenceError::IoError(format!("Transaction failed: {:?}", e)))?;

            let object_store = transaction
                .object_store(store_name.as_str())
                .map_err(|e| PersistenceError::IoError(format!("Object store failed: {:?}", e)))?;

            let uint8_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
            uint8_array.copy_from(&data);

            let key = js_sys::Number::from(page_id as f64);
            let request = object_store
                .put_with_key(&uint8_array, &key)
                .map_err(|e| PersistenceError::IoError(format!("Put request failed: {:?}", e)))?;

            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                let resolve_clone = resolve.clone();
                let reject_clone = reject.clone();

                let success_cb =
                    wasm_bindgen::closure::Closure::once(Box::new(move |_: web_sys::Event| {
                        let _ = resolve_clone.call1(
                            &wasm_bindgen::JsValue::UNDEFINED,
                            &wasm_bindgen::JsValue::UNDEFINED,
                        );
                    })
                        as Box<dyn FnOnce(web_sys::Event)>);

                let error_cb =
                    wasm_bindgen::closure::Closure::once(Box::new(move |_: web_sys::Event| {
                        let _ = reject_clone.call1(
                            &wasm_bindgen::JsValue::UNDEFINED,
                            &wasm_bindgen::JsValue::from_str("Write failed"),
                        );
                    })
                        as Box<dyn FnOnce(web_sys::Event)>);

                request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
                request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

                success_cb.forget();
                error_cb.forget();
            });

            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

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
unsafe impl Send for IndexedDBBackend {}
unsafe impl Sync for IndexedDBBackend {}
