//! Browser Integration for Holographic Memory
//!
//! Provides WASM bindings and OPFS persistence for holographic key-value stores.
//! Core algorithms remain in joule-db-novel (pure Rust).

use crate::novel::holographic_serialization::{
    deserialize_turbo_holographic, reconstruct_turbo_holographic, serialize_turbo_holographic,
};
use crate::persistence::OPFSBackend;
#[cfg(feature = "novel")]
use joule_db_hdc::turbo_holographic::TurboHolographic;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

/// Browser-specific holographic store with OPFS persistence
pub struct BrowserHolographicStore {
    /// Core holographic store (pure Rust, no WASM)
    store: Arc<RwLock<TurboHolographic>>,
    /// Optional persistent backend for long-term storage
    backend: Option<Arc<RwLock<OPFSBackend>>>,
    /// Dimension of hypervectors
    dimension: usize,
    /// Track key-value pairs for serialization (since we can't access bundle directly)
    key_value_cache: Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>,
    /// Page ID for storing holographic data
    persistence_page_id: u64,
}

/// Configuration for browser holographic store
#[derive(Debug, Clone)]
pub struct HolographicStoreConfig {
    /// Dimension (4096, 8192, 16384, 32768 recommended)
    pub dimension: usize,
    /// Enable OPFS persistence
    pub persistent: bool,
    /// Database name for persistence
    pub db_name: String,
}

impl Default for HolographicStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 8192,
            persistent: false,
            db_name: "holographic_db".to_string(),
        }
    }
}

impl BrowserHolographicStore {
    /// Create a new in-memory holographic store
    pub fn new(dimension: usize) -> Self {
        let store = TurboHolographic::new(dimension);
        Self {
            store: Arc::new(RwLock::new(store)),
            backend: None,
            dimension,
            key_value_cache: Arc::new(RwLock::new(HashMap::new())),
            persistence_page_id: 1, // Use page 1 for holographic store data
        }
    }

    /// Create with configuration
    pub fn with_config(config: HolographicStoreConfig) -> Self {
        let store = TurboHolographic::new(config.dimension);
        let backend = if config.persistent {
            Some(Arc::new(RwLock::new(OPFSBackend::new(&config.db_name))))
        } else {
            None
        };
        Self {
            store: Arc::new(RwLock::new(store)),
            backend,
            dimension: config.dimension,
            key_value_cache: Arc::new(RwLock::new(HashMap::new())),
            persistence_page_id: 1,
        }
    }

    /// Create with OPFS persistence
    pub async fn with_persistence(dimension: usize, db_name: &str) -> Result<Self, JsValue> {
        let backend = Arc::new(RwLock::new(OPFSBackend::new(db_name)));

        // Initialize backend if needed
        {
            let backend_guard = backend.read().unwrap();
            if !backend_guard.is_initialized() {
                drop(backend_guard); // Release read lock
                let backend_write = backend.write().unwrap();
                // Get init future and drop the lock before awaiting
                drop(backend_write);
                let backend_ref = backend.read().unwrap();
                let init_future = backend_ref.init();
                // drop(backend_ref); // Removed to keep borrowing valid
                init_future.await.map_err(|e| {
                    JsValue::from_str(&format!("Failed to initialize OPFS: {:?}", e))
                })?;
            }
        }

        // Try to load existing data
        use crate::persistence::AsyncPersistenceBackend;
        let (store, cache) = {
            let backend_guard = backend.read().unwrap();
            let read_future = backend_guard.read_page(1);
            drop(backend_guard); // Release lock before awaiting

            if let Ok(data) = read_future.await {
                // Deserialize and reconstruct
                match deserialize_turbo_holographic(&data) {
                    Ok(serialized) => {
                        let mut cache_map = HashMap::new();
                        for (key, value) in &serialized.key_value_pairs {
                            cache_map.insert(key.clone(), value.clone());
                        }
                        let reconstructed = reconstruct_turbo_holographic(&serialized);
                        (
                            Arc::new(RwLock::new(reconstructed)),
                            Arc::new(RwLock::new(cache_map)),
                        )
                    }
                    Err(_) => {
                        // If deserialization fails, create new store
                        (
                            Arc::new(RwLock::new(TurboHolographic::new(dimension))),
                            Arc::new(RwLock::new(HashMap::new())),
                        )
                    }
                }
            } else {
                // No existing data, create new store
                (
                    Arc::new(RwLock::new(TurboHolographic::new(dimension))),
                    Arc::new(RwLock::new(HashMap::new())),
                )
            }
        };

        Ok(Self {
            store,
            backend: Some(backend),
            dimension,
            key_value_cache: cache,
            persistence_page_id: 1,
        })
    }

    /// Store a key-value pair
    pub fn put(&self, key: &[u8], value: &[u8]) {
        let mut store = self.store.write().unwrap();
        store.put(key, value);

        // Update cache for serialization
        let mut cache = self.key_value_cache.write().unwrap();
        cache.insert(key.to_vec(), value.to_vec());
    }

    /// Persist changes to OPFS (async, call periodically or on close)
    pub async fn persist(&self) -> Result<(), JsValue> {
        if let Some(ref backend) = self.backend {
            let store = self.store.read().unwrap();
            let cache = self.key_value_cache.read().unwrap();

            // Convert HashMap to Vec for serialization
            let pairs: Vec<(Vec<u8>, Vec<u8>)> =
                cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            // Serialize
            let serialized = serialize_turbo_holographic(&store, &pairs);

            // Write to OPFS using AsyncPersistenceBackend trait
            use crate::persistence::AsyncPersistenceBackend;
            let backend_guard = backend.read().unwrap();
            let write_future = backend_guard.write_page(self.persistence_page_id, &serialized);
            drop(backend_guard); // Release lock before awaiting

            write_future
                .await
                .map_err(|e| JsValue::from_str(&format!("Persistence error: {}", e)))?;
        }
        Ok(())
    }

    /// Retrieve a value by key (approximate)
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let store = self.store.read().unwrap();
        store.get(key)
    }

    /// Check if key exists (probabilistic)
    pub fn contains(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Get number of items stored
    pub fn len(&self) -> usize {
        let store = self.store.read().unwrap();
        store.len()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get capacity estimate
    pub fn capacity(&self) -> usize {
        // sqrt(dimension) for reliable retrieval
        (self.dimension as f64).sqrt() as usize
    }

    /// Get load factor
    pub fn load_factor(&self) -> f32 {
        let len = self.len();
        let cap = self.capacity();
        if cap == 0 {
            0.0
        } else {
            len as f32 / cap as f32
        }
    }
}

/// WASM bindings for holographic store

pub struct WasmHolographicStore {
    inner: BrowserHolographicStore,
}

impl WasmHolographicStore {
    /// Create a new holographic store

    pub fn new(dimension: usize) -> Self {
        Self {
            inner: BrowserHolographicStore::new(dimension),
        }
    }

    /// Store a key-value pair
    pub fn put(&self, key: &[u8], value: &[u8]) {
        self.inner.put(key, value);
    }

    /// Retrieve a value by key
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.get(key)
    }

    /// Check if key exists
    pub fn contains(&self, key: &[u8]) -> bool {
        self.inner.contains(key)
    }

    /// Get number of items
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Get load factor
    pub fn load_factor(&self) -> f32 {
        self.inner.load_factor()
    }
}
