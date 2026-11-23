//! IndexedDB storage backend
//!
//! Implements async storage using the browser's IndexedDB API.

use js_sys::Promise;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Event, IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransactionMode};

use joule_db_core::error::StorageError;
use joule_db_core::storage::{Page, PageId};

const STORE_NAME: &str = "pages";
const DB_VERSION: u32 = 1;

/// Convert an IdbRequest to a Promise
fn request_to_promise(request: &IdbRequest) -> Promise {
    let _onsuccess = Closure::once(Box::new(|_event: Event| {}) as Box<dyn FnOnce(Event)>);
    let _onerror = Closure::once(Box::new(|_event: Event| {}) as Box<dyn FnOnce(Event)>);

    Promise::new(&mut |resolve, reject| {
        let resolve_clone = resolve.clone();
        let reject_clone = reject.clone();

        let success_cb = Closure::once(Box::new(move |event: Event| {
            let target = event.target().unwrap();
            let req: IdbRequest = target.unchecked_into();
            let result = req.result().unwrap_or(JsValue::UNDEFINED);
            resolve_clone.call1(&JsValue::UNDEFINED, &result).unwrap();
        }) as Box<dyn FnOnce(Event)>);

        let error_cb = Closure::once(Box::new(move |event: Event| {
            let target = event.target().unwrap();
            let req: IdbRequest = target.unchecked_into();
            let error = req.error().unwrap_or(None);
            let error_val = match error {
                Some(e) => JsValue::from(e),
                None => JsValue::from_str("Unknown error"),
            };
            reject_clone.call1(&JsValue::UNDEFINED, &error_val).unwrap();
        }) as Box<dyn FnOnce(Event)>);

        request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
        request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

        success_cb.forget();
        error_cb.forget();
    })
}

/// Convert an IdbOpenDbRequest to a Promise
fn open_request_to_promise(request: &IdbOpenDbRequest, store_name: String) -> Promise {
    Promise::new(&mut |resolve, reject| {
        let resolve_clone = resolve.clone();
        let reject_clone = reject.clone();
        let store_name_clone = store_name.clone();

        let upgrade_cb = Closure::once(Box::new(move |event: Event| {
            let target = event.target().unwrap();
            let req: IdbOpenDbRequest = target.unchecked_into();
            let db: IdbDatabase = req.result().unwrap().unchecked_into();

            if !db.object_store_names().contains(&store_name_clone) {
                let _ = db.create_object_store(&store_name_clone);
            }
        }) as Box<dyn FnOnce(Event)>);

        let success_cb = Closure::once(Box::new(move |event: Event| {
            let target = event.target().unwrap();
            let req: IdbOpenDbRequest = target.unchecked_into();
            let result = req.result().unwrap_or(JsValue::UNDEFINED);
            resolve_clone.call1(&JsValue::UNDEFINED, &result).unwrap();
        }) as Box<dyn FnOnce(Event)>);

        let error_cb = Closure::once(Box::new(move |event: Event| {
            let target = event.target().unwrap();
            let req: IdbOpenDbRequest = target.unchecked_into();
            let error = req.error().unwrap_or(None);
            let error_val = match error {
                Some(e) => JsValue::from(e),
                None => JsValue::from_str("Unknown error"),
            };
            reject_clone.call1(&JsValue::UNDEFINED, &error_val).unwrap();
        }) as Box<dyn FnOnce(Event)>);

        request.set_onupgradeneeded(Some(upgrade_cb.as_ref().unchecked_ref()));
        request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
        request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

        upgrade_cb.forget();
        success_cb.forget();
        error_cb.forget();
    })
}

/// IndexedDB storage backend
pub struct IndexedDbBackend {
    db_name: String,
    db: Arc<RwLock<Option<IdbDatabase>>>,
    cache: Arc<RwLock<HashMap<PageId, Page>>>,
    next_page_id: Arc<RwLock<PageId>>,
}

impl IndexedDbBackend {
    /// Create a new IndexedDB backend
    pub fn new(db_name: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            db: Arc::new(RwLock::new(None)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            next_page_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Open the database
    pub async fn open(&self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window")?;
        let idb_factory = window.indexed_db()?.ok_or("IndexedDB not available")?;

        let open_request = idb_factory.open_with_u32(&self.db_name, DB_VERSION)?;

        let promise = open_request_to_promise(&open_request, STORE_NAME.to_string());
        let db: IdbDatabase = JsFuture::from(promise).await?.unchecked_into();

        // Store the database handle
        let mut db_lock = self.db.write().unwrap();
        *db_lock = Some(db);

        Ok(())
    }

    /// Get the database reference
    fn get_db(&self) -> Result<IdbDatabase, StorageError> {
        self.db
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| StorageError::Backend("Database not opened".to_string()))
    }

    /// Read a page asynchronously
    pub async fn read_page_async(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(page) = cache.get(&page_id) {
                return Ok(Some(page.clone()));
            }
        }

        let db = self.get_db()?;
        let tx = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)
            .map_err(|e| StorageError::Backend(format!("Transaction error: {:?}", e)))?;

        let store = tx
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::Backend(format!("Store error: {:?}", e)))?;

        let key = JsValue::from_f64(page_id as f64);
        let request = store
            .get(&key)
            .map_err(|e| StorageError::Backend(format!("Get error: {:?}", e)))?;

        let promise = request_to_promise(&request);
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Backend(format!("Async error: {:?}", e)))?;

        if result.is_undefined() || result.is_null() {
            return Ok(None);
        }

        // Deserialize the page
        let data: Vec<u8> = serde_wasm_bindgen::from_value(result)
            .map_err(|e| StorageError::Backend(format!("Deserialize error: {:?}", e)))?;

        let page = Page::decode(&data)
            .map_err(|e| StorageError::Backend(format!("Decode error: {:?}", e)))?;

        // Cache it
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(page_id, page.clone());
        }

        Ok(Some(page))
    }

    /// Write a page asynchronously
    pub async fn write_page_async(&self, page: Page) -> Result<(), StorageError> {
        let page_size = joule_db_core::storage::page::DEFAULT_PAGE_SIZE;
        let data = page
            .encode(page_size)
            .map_err(|e| StorageError::Backend(format!("Encode error: {:?}", e)))?;

        let db = self.get_db()?;
        let tx = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::Backend(format!("Transaction error: {:?}", e)))?;

        let store = tx
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::Backend(format!("Store error: {:?}", e)))?;

        let page_id = page.id;
        let key = JsValue::from_f64(page_id as f64);
        let value = serde_wasm_bindgen::to_value(&data)
            .map_err(|e| StorageError::Backend(format!("Serialize error: {:?}", e)))?;

        let request = store
            .put_with_key(&value, &key)
            .map_err(|e| StorageError::Backend(format!("Put error: {:?}", e)))?;

        let promise = request_to_promise(&request);
        JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Backend(format!("Async error: {:?}", e)))?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(page_id, page);
        }

        Ok(())
    }

    /// Allocate a new page
    pub fn allocate_page(&self) -> Result<PageId, StorageError> {
        let mut next_id = self.next_page_id.write().unwrap();
        let page_id = *next_id;
        *next_id += 1;
        Ok(page_id)
    }

    /// Free a page
    pub async fn free_page_async(&self, page_id: PageId) -> Result<(), StorageError> {
        // Remove from cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.remove(&page_id);
        }

        let db = self.get_db()?;
        let tx = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::Backend(format!("Transaction error: {:?}", e)))?;

        let store = tx
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::Backend(format!("Store error: {:?}", e)))?;

        let key = JsValue::from_f64(page_id as f64);
        let request = store
            .delete(&key)
            .map_err(|e| StorageError::Backend(format!("Delete error: {:?}", e)))?;

        let promise = request_to_promise(&request);
        JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Backend(format!("Async error: {:?}", e)))?;

        Ok(())
    }

    /// Sync (no-op for IndexedDB as writes are durable)
    pub async fn sync_async(&self) -> Result<(), StorageError> {
        Ok(())
    }
}
