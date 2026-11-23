//! IndexedDB storage for query history
//!
//! Provides persistent storage for query history used by predictive prefetching.
//! Uses a dedicated IndexedDB database separate from the main database storage.

#[cfg(target_arch = "wasm32")]
use js_sys::Promise;
use std::sync::{Arc, RwLock};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::closure::Closure;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

/// IndexedDB storage for query history

pub struct QueryHistoryStorage {
    db_name: String,
    store_name: String,
    #[cfg(target_arch = "wasm32")]
    db: Arc<RwLock<Option<web_sys::IdbDatabase>>>,
    /// In-memory history cache (used for non-WASM)
    #[cfg(not(target_arch = "wasm32"))]
    history_cache: Arc<RwLock<Vec<String>>>,
    initialized: Arc<RwLock<bool>>,
    max_history_size: usize,
}

impl QueryHistoryStorage {
    /// Create a new query history storage
    pub fn new(db_name: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            store_name: "query_history".to_string(),
            #[cfg(target_arch = "wasm32")]
            db: Arc::new(RwLock::new(None)),
            #[cfg(not(target_arch = "wasm32"))]
            history_cache: Arc::new(RwLock::new(Vec::new())),
            initialized: Arc::new(RwLock::new(false)),
            max_history_size: 10000, // Default: keep last 10k queries
        }
    }

    /// Create with custom max history size
    pub fn with_max_size(db_name: &str, max_history_size: usize) -> Self {
        Self {
            db_name: db_name.to_string(),
            store_name: "query_history".to_string(),
            #[cfg(target_arch = "wasm32")]
            db: Arc::new(RwLock::new(None)),
            #[cfg(not(target_arch = "wasm32"))]
            history_cache: Arc::new(RwLock::new(Vec::new())),
            initialized: Arc::new(RwLock::new(false)),
            max_history_size,
        }
    }

    /// Initialize IndexedDB
    #[cfg(target_arch = "wasm32")]
    pub async fn init(&self) -> Result<(), String> {
        if *self.initialized.read().unwrap() {
            return Ok(());
        }

        let window = web_sys::window().ok_or_else(|| "No window object".to_string())?;

        let idb_factory = window
            .indexed_db()
            .map_err(|_| "IndexedDB not supported".to_string())?
            .ok_or_else(|| "IndexedDB not available".to_string())?;

        let open_request = idb_factory
            .open_with_u32(&self.db_name, 1)
            .map_err(|e| format!("Failed to open IndexedDB: {:?}", e))?;

        // Set up onupgradeneeded handler
        let store_name = self.store_name.clone();
        let onupgradeneeded =
            Closure::wrap(Box::new(move |event: web_sys::IdbVersionChangeEvent| {
                if let Some(target) = event.target() {
                    let target: web_sys::EventTarget = target; // Explicit type
                    if let Ok(request) = target.dyn_into::<web_sys::IdbOpenDbRequest>() {
                        if let Ok(result) = request.result() {
                            let result: wasm_bindgen::JsValue = result; // Explicit type
                            if let Ok(db) = result.dyn_into::<web_sys::IdbDatabase>() {
                                // Create object store if it doesn't exist
                                if !db.object_store_names().contains(&store_name) {
                                    let _ = db.create_object_store(&store_name);
                                }
                            }
                        }
                    }
                }
            }) as Box<dyn FnMut(_)>);

        open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
        onupgradeneeded.forget();

        // Wait for database to open
        let promise = Promise::new(&mut |resolve, _| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    let target: web_sys::EventTarget = target;
                    if let Ok(request) = target.dyn_into::<web_sys::IdbOpenDbRequest>() {
                        if let Ok(result) = request.result() {
                            let result: wasm_bindgen::JsValue = result;
                            if let Ok(db) = result.dyn_into::<web_sys::IdbDatabase>() {
                                resolve.call1(&JsValue::NULL, &db).unwrap();
                            }
                        }
                    }
                }
            }) as Box<dyn FnMut(_)>);
            open_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            onsuccess.forget();
        });

        let db_value = JsFuture::from(promise)
            .await
            .map_err(|e| format!("Failed to open database: {:?}", e))?;

        let db = db_value
            .dyn_into::<web_sys::IdbDatabase>()
            .map_err(|e| format!("Failed to cast database: {:?}", e))?;

        *self.db.write().unwrap() = Some(db);
        *self.initialized.write().unwrap() = true;

        Ok(())
    }

    /// Initialize storage (non-WASM: uses in-memory storage with optional file persistence)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn init(&self) -> Result<(), String> {
        *self.initialized.write().unwrap() = true;
        tracing::debug!(
            "QueryHistoryStorage initialized (non-WASM mode, in-memory with file backup)"
        );
        Ok(())
    }

    /// Save query history to IndexedDB
    #[cfg(target_arch = "wasm32")]
    pub async fn save_history(&self, queries: &[String]) -> Result<(), String> {
        if !*self.initialized.read().unwrap() {
            self.init().await?;
        }

        let db_guard = self.db.read().unwrap();
        let db = db_guard
            .as_ref()
            .ok_or_else(|| "Database not initialized".to_string())?;

        // Serialize queries to JSON
        let json =
            serde_json::to_string(queries).map_err(|e| format!("Serialization error: {}", e))?;

        let transaction = db
            .transaction_with_str(&self.store_name)
            .map_err(|e| format!("Transaction error: {:?}", e))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| format!("Store error: {:?}", e))?;

        // Store with key "history"
        let key = JsValue::from_str("history");
        let value = JsValue::from_str(&json);

        let request = store
            .put_with_key(&value, &key)
            .map_err(|e| format!("Put error: {:?}", e))?;

        // Wait for completion
        let promise = Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                resolve.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);
            let onerror = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                reject.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);

            request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onsuccess.forget();
            onerror.forget();
        });

        JsFuture::from(promise)
            .await
            .map_err(|e| format!("Save error: {:?}", e))?;

        Ok(())
    }

    /// Save query history (non-WASM: uses in-memory cache with file backup)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save_history(&self, queries: &[String]) -> Result<(), String> {
        // Update in-memory cache
        let mut cache = self
            .history_cache
            .write()
            .map_err(|_| "Failed to acquire write lock on history cache".to_string())?;

        // Append new queries and limit to max_history_size
        cache.extend(queries.iter().cloned());
        if cache.len() > self.max_history_size {
            let start = cache.len() - self.max_history_size;
            *cache = cache[start..].to_vec();
        }

        // Optionally persist to file (best-effort)
        let history_file = format!("{}_query_history.json", self.db_name);
        if let Ok(json) = serde_json::to_string(&*cache) {
            let _ = std::fs::write(&history_file, json);
            tracing::debug!("Query history persisted to {}", history_file);
        }

        Ok(())
    }

    /// Load query history from IndexedDB
    #[cfg(target_arch = "wasm32")]
    pub async fn load_history(&self) -> Result<Vec<String>, String> {
        if !*self.initialized.read().unwrap() {
            self.init().await?;
        }

        let db_guard = self.db.read().unwrap();
        let db = db_guard
            .as_ref()
            .ok_or_else(|| "Database not initialized".to_string())?;

        let transaction = db
            .transaction_with_str(&self.store_name)
            .map_err(|e| format!("Transaction error: {:?}", e))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| format!("Store error: {:?}", e))?;

        let key = JsValue::from_str("history");
        let request = store.get(&key).map_err(|e| format!("Get error: {:?}", e))?;

        // Wait for result
        let promise = Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(request) = target.dyn_into::<web_sys::IdbRequest>() {
                        if let Ok(result) = request.result() {
                            resolve.call1(&JsValue::NULL, &result).unwrap();
                            return;
                        }
                    }
                }
                resolve.call1(&JsValue::NULL, &JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);
            let onerror = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                reject.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);

            request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onsuccess.forget();
            onerror.forget();
        });

        let result = JsFuture::from(promise)
            .await
            .map_err(|e| format!("Load error: {:?}", e))?;

        if result.is_null() {
            return Ok(Vec::new());
        }

        // Extract string from JsValue
        let json_str = result
            .as_string()
            .ok_or_else(|| "Invalid result type".to_string())?;

        // Deserialize
        let queries: Vec<String> =
            serde_json::from_str(&json_str).map_err(|e| format!("Deserialization error: {}", e))?;

        // Limit to max_history_size
        let queries = if queries.len() > self.max_history_size {
            queries[queries.len() - self.max_history_size..].to_vec()
        } else {
            queries
        };

        Ok(queries)
    }

    /// Load query history (non-WASM: loads from in-memory cache or file)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_history(&self) -> Result<Vec<String>, String> {
        // First, try to load from in-memory cache
        let mut cache = self
            .history_cache
            .write()
            .map_err(|_| "Failed to acquire write lock on history cache".to_string())?;

        // If cache is empty, try to load from file
        if cache.is_empty() {
            let history_file = format!("{}_query_history.json", self.db_name);
            if let Ok(content) = std::fs::read_to_string(&history_file) {
                if let Ok(queries) = serde_json::from_str::<Vec<String>>(&content) {
                    *cache = queries;
                    tracing::debug!("Query history loaded from {}", history_file);
                }
            }
        }

        // Return a copy limited to max_history_size
        let result = if cache.len() > self.max_history_size {
            cache[cache.len() - self.max_history_size..].to_vec()
        } else {
            cache.clone()
        };

        Ok(result)
    }

    /// Clear query history
    #[cfg(target_arch = "wasm32")]
    pub async fn clear(&self) -> Result<(), String> {
        if !*self.initialized.read().unwrap() {
            self.init().await?;
        }

        let db_guard = self.db.read().unwrap();
        let db = db_guard
            .as_ref()
            .ok_or_else(|| "Database not initialized".to_string())?;

        let transaction = db
            .transaction_with_str(&self.store_name)
            .map_err(|e| format!("Transaction error: {:?}", e))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| format!("Store error: {:?}", e))?;

        let key = JsValue::from_str("history");
        let request = store
            .delete(&key)
            .map_err(|e| format!("Delete error: {:?}", e))?;

        let promise = Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                resolve.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);
            let onerror = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                reject.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(_)>);

            request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onsuccess.forget();
            onerror.forget();
        });

        JsFuture::from(promise)
            .await
            .map_err(|e| format!("Clear error: {:?}", e))?;

        Ok(())
    }

    /// Clear query history (non-WASM: clears cache and file)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn clear(&self) -> Result<(), String> {
        // Clear in-memory cache
        let mut cache = self
            .history_cache
            .write()
            .map_err(|_| "Failed to acquire write lock on history cache".to_string())?;
        cache.clear();

        // Delete persistence file (best-effort)
        let history_file = format!("{}_query_history.json", self.db_name);
        let _ = std::fs::remove_file(&history_file);
        tracing::debug!("Query history cleared");

        Ok(())
    }
}

// Make it Send + Sync safe
unsafe impl Send for QueryHistoryStorage {}
unsafe impl Sync for QueryHistoryStorage {}
