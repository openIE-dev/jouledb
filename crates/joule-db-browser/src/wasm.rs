//! Single WASM Entry Point for JouleDB Browser
//!
//! This is the ONLY file with `#[wasm_bindgen]` attributes in the entire project.
//! All WASM bindings are consolidated here for maintainability and to avoid
//! dependency conflicts.

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

// Re-export all internal types (without WASM bindings)
pub use crate::engine::AsyncEngine;
pub use crate::engine_enhanced::EnhancedAsyncEngine;
pub use crate::persistence::{
    BackendType, CacheStats, DynBackend, IndexedDBBackend, MigrationResult, OPFSBackend,
    OPFSVirtualBlockDevice, PersistenceConfig, PersistenceError, ZeroCopyPageCache, migrate_to_vbd,
};
pub use crate::storage::indexeddb::IndexedDbBackend;
pub use crate::storage::{BatchConfig, BatchStats, IndexedDBBatcher};

// Import internal modules
use crate::engine_enhanced::AsyncRangeIterator;
use joule_db_core::index::{Bound, ScanDirection};
use joule_db_core::{Engine, IsolationLevel as CoreIsolationLevel, MemoryBackend};

// ============================================================================
// Core Database Types
// ============================================================================

/// JouleDB browser database - Main entry point
#[wasm_bindgen]
pub struct JouleDB {
    // For in-memory databases
    engine: Option<Engine>,
    // For persistent databases (OPFS/IndexedDB) - uses AsyncEngine
    #[cfg(target_arch = "wasm32")]
    async_engine: Option<Arc<RwLock<AsyncEngine>>>,
    // Enhanced engine with full features
    #[cfg(target_arch = "wasm32")]
    enhanced_engine: Option<Arc<RwLock<EnhancedJouleDB>>>,
    #[cfg(target_arch = "wasm32")]
    backend_type: BackendType,
    db_name: String,
}

#[wasm_bindgen]
impl JouleDB {
    /// Create a new in-memory database
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<JouleDB, JsValue> {
        let backend = MemoryBackend::new();
        let engine = Engine::new(backend).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JouleDB {
            engine: Some(engine),
            #[cfg(target_arch = "wasm32")]
            async_engine: None,
            #[cfg(target_arch = "wasm32")]
            enhanced_engine: None,
            #[cfg(target_arch = "wasm32")]
            backend_type: BackendType::Memory,
            db_name: String::new(),
        })
    }

    /// Open a persistent database with auto-selected backend
    #[wasm_bindgen]
    pub async fn open(db_name: &str) -> Result<JouleDB, JsValue> {
        use crate::persistence::auto_select_backend;
        let selection = auto_select_backend();
        Self::open_with_backend(db_name, &selection.backend_type.to_string()).await
    }

    /// Open a persistent database with a specific backend
    #[wasm_bindgen]
    pub async fn open_with_backend(db_name: &str, backend_str: &str) -> Result<JouleDB, JsValue> {
        use crate::persistence::{OPFSBackend, OPFSVirtualBlockDevice};

        let backend_type = match backend_str.to_lowercase().as_str() {
            "opfs" => BackendType::OPFS,
            "opfs-vbd" | "opfsvbd" | "vbd" => BackendType::OPFSVBD,
            "indexeddb" | "indexed_db" => BackendType::IndexedDB,
            "memory" => BackendType::Memory,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown backend: {}",
                    backend_str
                )));
            }
        };

        match backend_type {
            BackendType::OPFS => {
                let backend = OPFSBackend::new(db_name);
                backend
                    .init()
                    .await
                    .map_err(|e| JsValue::from_str(&format!("OPFS init failed: {}", e)))?;

                let backend_arc: Arc<RwLock<DynBackend>> = Arc::new(RwLock::new(backend));
                let async_engine = AsyncEngine::open(backend_arc)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("Failed to open engine: {}", e)))?;

                let async_engine_arc = Arc::new(RwLock::new(async_engine));
                let enhanced = EnhancedJouleDB::from_engine(Arc::clone(&async_engine_arc));

                Ok(JouleDB {
                    engine: None,
                    #[cfg(target_arch = "wasm32")]
                    async_engine: Some(async_engine_arc),
                    #[cfg(target_arch = "wasm32")]
                    enhanced_engine: Some(Arc::new(RwLock::new(enhanced))),
                    #[cfg(target_arch = "wasm32")]
                    backend_type: BackendType::OPFS,
                    db_name: db_name.to_string(),
                })
            }
            BackendType::OPFSVBD => {
                let backend = OPFSVirtualBlockDevice::new(db_name);
                backend
                    .init()
                    .await
                    .map_err(|e| JsValue::from_str(&format!("OPFS VBD init failed: {}", e)))?;

                let backend_arc: Arc<RwLock<DynBackend>> = Arc::new(RwLock::new(backend));
                let async_engine = AsyncEngine::open(backend_arc)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("Failed to open engine: {}", e)))?;

                let async_engine_arc = Arc::new(RwLock::new(async_engine));
                let enhanced = EnhancedJouleDB::from_engine(Arc::clone(&async_engine_arc));

                Ok(JouleDB {
                    engine: None,
                    #[cfg(target_arch = "wasm32")]
                    async_engine: Some(async_engine_arc),
                    #[cfg(target_arch = "wasm32")]
                    enhanced_engine: Some(Arc::new(RwLock::new(enhanced))),
                    #[cfg(target_arch = "wasm32")]
                    backend_type: BackendType::OPFSVBD,
                    db_name: db_name.to_string(),
                })
            }
            BackendType::IndexedDB => {
                let backend = IndexedDBBackend::new(db_name, "pages");
                backend
                    .init()
                    .await
                    .map_err(|e| JsValue::from_str(&format!("IndexedDB init failed: {}", e)))?;

                let backend_arc: Arc<RwLock<DynBackend>> = Arc::new(RwLock::new(backend));
                let async_engine = AsyncEngine::open(backend_arc)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("Failed to open engine: {}", e)))?;

                let async_engine_arc = Arc::new(RwLock::new(async_engine));
                let enhanced = EnhancedJouleDB::from_engine(Arc::clone(&async_engine_arc));

                Ok(JouleDB {
                    engine: None,
                    #[cfg(target_arch = "wasm32")]
                    async_engine: Some(async_engine_arc),
                    #[cfg(target_arch = "wasm32")]
                    enhanced_engine: Some(Arc::new(RwLock::new(enhanced))),
                    #[cfg(target_arch = "wasm32")]
                    backend_type: BackendType::IndexedDB,
                    db_name: db_name.to_string(),
                })
            }
            BackendType::Memory => Self::new(),
        }
    }

    /// Check if OPFS is available
    #[wasm_bindgen]
    pub fn is_opfs_available() -> bool {
        OPFSBackend::is_available()
    }

    /// Check if IndexedDB is available
    #[wasm_bindgen]
    pub fn is_indexeddb_available() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::window()
                .and_then(|w| w.indexed_db().ok())
                .flatten()
                .is_some()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Get a value by key
    #[wasm_bindgen]
    pub async fn get(&self, key: &str) -> Result<JsValue, JsValue> {
        if let Some(ref engine) = self.engine {
            let res: Result<Option<Vec<u8>>, joule_db_core::Error> = engine.get(key.as_bytes());
            match res {
                Ok(Some(value)) => match String::from_utf8(value) {
                    Ok(s) => Ok(JsValue::from_str(&s)),
                    Err(_) => Err(JsValue::from_str("Invalid UTF-8")),
                },
                Ok(None) => Ok(JsValue::NULL),
                Err(e) => Err(JsValue::from_str(&e.to_string())),
            }
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(ref async_engine) = self.async_engine {
                    let engine = async_engine.read().unwrap();
                    match engine.get(key.as_bytes()).await {
                        Ok(Some(value)) => match String::from_utf8(value) {
                            Ok(s) => Ok(JsValue::from_str(&s)),
                            Err(_) => Err(JsValue::from_str("Invalid UTF-8")),
                        },
                        Ok(None) => Ok(JsValue::NULL),
                        Err(e) => Err(JsValue::from_str(&e.to_string())),
                    }
                } else {
                    Err(JsValue::from_str("No engine available"))
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                Err(JsValue::from_str("Persistent backends not available"))
            }
        }
    }

    /// Put a key-value pair
    #[wasm_bindgen]
    pub async fn put(&self, key: &str, value: &str) -> Result<(), JsValue> {
        if let Some(ref engine) = self.engine {
            let res: Result<(), joule_db_core::Error> =
                engine.put(key.as_bytes(), value.as_bytes());
            res.map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(ref async_engine) = self.async_engine {
                    let engine = async_engine.read().unwrap();
                    let fut = engine.put(key.as_bytes(), value.as_bytes());
                    fut.await.map_err(|e| JsValue::from_str(&e.to_string()))
                } else {
                    Err(JsValue::from_str("No engine available"))
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                Err(JsValue::from_str("Persistent backends not available"))
            }
        }
    }

    /// Delete a key
    #[wasm_bindgen]
    pub async fn delete(&self, key: &str) -> Result<(), JsValue> {
        if let Some(ref engine) = self.engine {
            let res: Result<bool, joule_db_core::Error> = engine.delete(key.as_bytes());
            res.map(|_| ())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(ref async_engine) = self.async_engine {
                    let engine = async_engine.read().unwrap();
                    let fut = engine.delete(key.as_bytes());
                    fut.await
                        .map(|_| ()) // Ignore boolean result
                        .map_err(|e| JsValue::from_str(&e.to_string()))
                } else {
                    Err(JsValue::from_str("No engine available"))
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                Err(JsValue::from_str("Persistent backends not available"))
            }
        }
    }
}

// ============================================================================
// Enhanced Database Types
// ============================================================================

/// WASM binding for range query bounds
#[wasm_bindgen]
pub enum RangeBound {
    Included,
    Excluded,
    Unbounded,
}

/// WASM binding for scan direction
#[wasm_bindgen]
pub enum ScanDir {
    Forward,
    Backward,
}

/// WASM binding for transaction isolation level
#[wasm_bindgen]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
    Snapshot,
}

impl From<IsolationLevel> for CoreIsolationLevel {
    fn from(level: IsolationLevel) -> Self {
        match level {
            IsolationLevel::ReadUncommitted => CoreIsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted => CoreIsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead => CoreIsolationLevel::RepeatableRead,
            IsolationLevel::Serializable => CoreIsolationLevel::Serializable,
            IsolationLevel::Snapshot => CoreIsolationLevel::Snapshot,
        }
    }
}

/// WASM binding for range iterator
#[wasm_bindgen]
pub struct RangeIterator {
    inner: Arc<RwLock<AsyncRangeIterator>>,
}

#[wasm_bindgen]
impl RangeIterator {
    /// Get next entry
    pub async fn next(&mut self) -> Result<Option<RangeEntry>, JsValue> {
        let mut iter = self.inner.write().unwrap();
        match iter.next().await {
            Ok(Some(entry)) => Ok(Some(RangeEntry {
                key: entry.key,
                value: entry.value,
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Collect all remaining entries
    pub async fn collect_all(&mut self) -> Result<js_sys::Array, JsValue> {
        let mut iter = self.inner.write().unwrap();
        let entries = iter
            .collect_all()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let array = js_sys::Array::new();
        for entry in entries {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("key"),
                &js_sys::Uint8Array::from(&entry.key[..]),
            )?;
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("value"),
                &js_sys::Uint8Array::from(&entry.value[..]),
            )?;
            array.push(&obj);
        }
        Ok(array)
    }
}

/// WASM binding for range entry
#[wasm_bindgen]
pub struct RangeEntry {
    key: Vec<u8>,
    value: Vec<u8>,
}

#[wasm_bindgen]
impl RangeEntry {
    /// Get key as Uint8Array
    pub fn key(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.key[..])
    }

    /// Get value as Uint8Array
    pub fn value(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.value[..])
    }

    /// Get key as string (UTF-8)
    pub fn key_string(&self) -> Result<String, JsValue> {
        String::from_utf8(self.key.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8: {}", e)))
    }

    /// Get value as string (UTF-8)
    pub fn value_string(&self) -> Result<String, JsValue> {
        String::from_utf8(self.value.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8: {}", e)))
    }
}

/// WASM binding for query result
#[wasm_bindgen]
pub struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Vec<js_sys::Object>>,
    affected_rows: usize,
    execution_time_ms: u64,
}

#[wasm_bindgen]
impl QueryResult {
    /// Get column names
    pub fn columns(&self) -> js_sys::Array {
        let array = js_sys::Array::new();
        for col in &self.columns {
            array.push(&JsValue::from_str(col));
        }
        array
    }

    /// Get rows as array
    pub fn rows(&self) -> js_sys::Array {
        let array = js_sys::Array::new();
        for row in &self.rows {
            let row_array = js_sys::Array::new();
            for cell in row {
                row_array.push(cell);
            }
            array.push(&row_array);
        }
        array
    }

    /// Get affected rows count
    pub fn affected_rows(&self) -> usize {
        self.affected_rows
    }

    /// Get execution time in milliseconds
    pub fn execution_time_ms(&self) -> u64 {
        self.execution_time_ms
    }
}

/// WASM binding for enhanced database
#[wasm_bindgen]
pub struct EnhancedJouleDB {
    engine: Arc<RwLock<EnhancedAsyncEngine>>,
}

impl EnhancedJouleDB {
    /// Create enhanced database from base engine (Internal use only)
    pub(crate) fn from_engine(base: Arc<RwLock<AsyncEngine>>) -> Self {
        let enhanced = EnhancedAsyncEngine::new(base);
        Self {
            engine: Arc::new(RwLock::new(enhanced)),
        }
    }
}

#[wasm_bindgen]
impl EnhancedJouleDB {
    /// Begin a transaction

    /// Begin a transaction
    pub async fn begin(&self) -> Result<u64, JsValue> {
        let engine = self.engine.read().unwrap();
        engine
            .begin()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Begin transaction with isolation level
    pub async fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<u64, JsValue> {
        let engine = self.engine.read().unwrap();
        let core_isolation: CoreIsolationLevel = isolation.into();
        engine
            .begin_with_isolation(core_isolation)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Commit transaction
    pub async fn commit(&self, tx_id: u64) -> Result<(), JsValue> {
        let engine = self.engine.read().unwrap();
        engine
            .commit(tx_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Rollback transaction
    pub async fn rollback(&self, tx_id: u64) -> Result<(), JsValue> {
        let engine = self.engine.read().unwrap();
        engine
            .rollback(tx_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Range query
    pub async fn range(
        &self,
        start_bound: RangeBound,
        start_key: Option<js_sys::Uint8Array>,
        end_bound: RangeBound,
        end_key: Option<js_sys::Uint8Array>,
        direction: ScanDir,
        tx_id: Option<u64>,
    ) -> Result<RangeIterator, JsValue> {
        let engine = self.engine.read().unwrap();

        let start = match start_bound {
            RangeBound::Included => {
                if let Some(key) = start_key {
                    let mut vec = vec![0u8; key.length() as usize];
                    key.copy_to(&mut vec);
                    Bound::Included(vec)
                } else {
                    return Err(JsValue::from_str("Included/Excluded bound requires key"));
                }
            }
            RangeBound::Excluded => {
                if let Some(key) = start_key {
                    let mut vec = vec![0u8; key.length() as usize];
                    key.copy_to(&mut vec);
                    Bound::Excluded(vec)
                } else {
                    return Err(JsValue::from_str("Included/Excluded bound requires key"));
                }
            }
            RangeBound::Unbounded => Bound::Unbounded,
        };

        let end = match end_bound {
            RangeBound::Included => {
                if let Some(key) = end_key {
                    let mut vec = vec![0u8; key.length() as usize];
                    key.copy_to(&mut vec);
                    Bound::Included(vec)
                } else {
                    return Err(JsValue::from_str("Included/Excluded bound requires key"));
                }
            }
            RangeBound::Excluded => {
                if let Some(key) = end_key {
                    let mut vec = vec![0u8; key.length() as usize];
                    key.copy_to(&mut vec);
                    Bound::Excluded(vec)
                } else {
                    return Err(JsValue::from_str("Included/Excluded bound requires key"));
                }
            }
            RangeBound::Unbounded => Bound::Unbounded,
        };

        let dir = match direction {
            ScanDir::Forward => ScanDirection::Forward,
            ScanDir::Backward => ScanDirection::Backward,
        };

        let iter = engine
            .range(start, end, dir, tx_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(RangeIterator {
            inner: Arc::new(RwLock::new(iter)),
        })
    }

    /// Prefix scan
    pub async fn prefix_scan(
        &self,
        prefix: js_sys::Uint8Array,
        tx_id: Option<u64>,
    ) -> Result<RangeIterator, JsValue> {
        let mut prefix_vec = vec![0u8; prefix.length() as usize];
        prefix.copy_to(&mut prefix_vec);

        let engine = self.engine.read().unwrap();
        let iter = engine
            .prefix_scan(&prefix_vec, tx_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(RangeIterator {
            inner: Arc::new(RwLock::new(iter)),
        })
    }

    /// Execute SQL query
    #[wasm_bindgen]
    pub async fn execute_sql(&self, sql: &str, tx_id: Option<u64>) -> Result<QueryResult, JsValue> {
        let engine = self.engine.read().unwrap();
        let result = engine
            .execute_sql(sql, tx_id)
            .await
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        // Convert ResultSet to WASM-friendly format
        let mut rows = Vec::new();
        for row in result.rows {
            let mut row_array = Vec::new();
            for value in row.values {
                let js_value = match value {
                    joule_db_query::ast::Value::Null => JsValue::NULL,
                    joule_db_query::ast::Value::Bool(b) => JsValue::from_bool(b),
                    joule_db_query::ast::Value::Int(i) => JsValue::from_f64(i as f64),
                    joule_db_query::ast::Value::Float(f) => JsValue::from_f64(f),
                    joule_db_query::ast::Value::String(s) => JsValue::from_str(&s),
                    joule_db_query::ast::Value::Bytes(b) => js_sys::Uint8Array::from(&b[..]).into(),
                    _ => JsValue::NULL,
                };
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &JsValue::from_str("value"), &js_value)?;
                row_array.push(obj);
            }
            rows.push(row_array);
        }

        Ok(QueryResult {
            columns: result.columns,
            rows,
            affected_rows: result.affected_rows,
            execution_time_ms: result.execution_time_ms,
        })
    }

    /// Create index
    #[wasm_bindgen]
    pub async fn create_index(&self, name: &str, unique: bool) -> Result<(), JsValue> {
        let engine = self.engine.read().unwrap();
        engine
            .create_index(name, unique)
            .await
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Drop index
    #[wasm_bindgen]
    pub async fn drop_index(&self, name: &str) -> Result<(), JsValue> {
        let engine = self.engine.read().unwrap();
        engine
            .drop_index(name)
            .await
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }
}

// ============================================================================
// HTAP Types (if feature enabled)
// ============================================================================

mod htap_bindings {
    use super::*;
    use crate::htap::{HTAPConfig as InternalHTAPConfig, HTAPStorage as InternalHTAPStorage};

    /// HTAP storage engine
    #[wasm_bindgen]
    pub struct HTAPStorage {
        inner: Arc<InternalHTAPStorage>,
    }

    #[wasm_bindgen]
    impl HTAPStorage {
        /// Create new HTAP storage asynchronously
        pub async fn new(config: JsValue) -> Result<HTAPStorage, JsValue> {
            let config: InternalHTAPConfig = serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsValue::from_str(&format!("Invalid config: {:?}", e)))?;

            let inner = InternalHTAPStorage::new(config).await.map_err(|e| {
                JsValue::from_str(&format!("Failed to create HTAP storage: {:?}", e))
            })?;

            Ok(HTAPStorage {
                inner: Arc::new(inner),
            })
        }

        /// Execute query (automatic routing)
        #[wasm_bindgen]
        pub async fn query(&self, query: &str) -> Result<JsValue, JsValue> {
            match self.inner.query(query).await {
                Ok(result) => Ok(JsValue::from_str(&result)),
                Err(e) => Err(JsValue::from_str(&e)),
            }
        }
    }
}

pub use htap_bindings::*;

// ============================================================================
// Learned Index Types (if feature enabled)
// ============================================================================

#[cfg(feature = "learned")]
mod learned_bindings {
    use super::*;
    use crate::learned::{
        LearnedIndex as InternalLearnedIndex, LearnedIndexConfig as InternalLearnedIndexConfig,
    };

    /// Learned index
    #[wasm_bindgen]
    pub struct LearnedIndex {
        inner: Arc<InternalLearnedIndex>,
    }

    #[wasm_bindgen]
    impl LearnedIndex {
        /// Create new learned index
        #[wasm_bindgen(constructor)]
        pub async fn new(config: JsValue) -> Result<LearnedIndex, JsValue> {
            let config: InternalLearnedIndexConfig = serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsValue::from_str(&format!("Invalid config: {:?}", e)))?;

            let inner = InternalLearnedIndex::new(config).await.map_err(|e| {
                JsValue::from_str(&format!("Failed to create learned index: {:?}", e))
            })?;

            Ok(LearnedIndex {
                inner: Arc::new(inner),
            })
        }

        /// Get value by key
        pub async fn get(&self, key: &str) -> Result<JsValue, JsValue> {
            self.inner.get(key).await
        }
    }
}

#[cfg(feature = "learned")]
pub use learned_bindings::*;

// ============================================================================
// Deterministic Execution Types (if feature enabled)
// ============================================================================

#[cfg(feature = "deterministic")]
mod deterministic_bindings {
    use super::*;
    use crate::deterministic::{
        DeterministicTransactionManager as InternalDeterministicTransactionManager,
        MultiTabCoordinator as InternalMultiTabCoordinator,
    };

    /// Deterministic transaction manager
    #[wasm_bindgen]
    pub struct DeterministicTransactionManager {
        inner: Arc<InternalDeterministicTransactionManager>,
    }

    #[wasm_bindgen]
    impl DeterministicTransactionManager {
        /// Create new deterministic transaction manager
        #[wasm_bindgen(constructor)]
        pub async fn new() -> Result<DeterministicTransactionManager, JsValue> {
            let inner = InternalDeterministicTransactionManager::new()
                .await
                .map_err(|e| JsValue::from_str(&format!("Failed to create manager: {:?}", e)))?;

            let inner_arc = Arc::new(inner);
            Ok(DeterministicTransactionManager { inner: inner_arc })
        }
    }

    /// Multi-tab coordinator
    #[wasm_bindgen]
    pub struct MultiTabCoordinator {
        inner: Arc<InternalMultiTabCoordinator>,
    }

    #[wasm_bindgen]
    impl MultiTabCoordinator {
        /// Create new coordinator
        #[wasm_bindgen(constructor)]
        pub async fn new(strategy: String) -> Result<MultiTabCoordinator, JsValue> {
            let inner = InternalMultiTabCoordinator::new(strategy)
                .await
                .map_err(|e| {
                    JsValue::from_str(&format!("Failed to create coordinator: {:?}", e))
                })?;

            Ok(MultiTabCoordinator {
                inner: Arc::new(inner),
            })
        }
    }
}

#[cfg(feature = "deterministic")]
pub use deterministic_bindings::*;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    // Panic hook initialization can be added via wasm-pack or manually
    #[cfg(feature = "console_error_panic_hook")]
    {
        console_error_panic_hook::set_once();
    }
}
