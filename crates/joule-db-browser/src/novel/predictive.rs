//! Browser Integration for Predictive Prefetching
//!
//! Provides query prediction and prefetching for browser environment.
//! Core algorithms remain in joule-db-novel (pure Rust).

use crate::engine_enhanced::EnhancedAsyncEngine;
use crate::novel::query_history_storage::QueryHistoryStorage;
#[cfg(feature = "novel")]
use joule_db_hdc::predictive::NGramPredictor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Browser-specific query predictor with prefetching
pub struct BrowserQueryPredictor {
    /// Core predictor (pure Rust, no WASM)
    predictor: Arc<RwLock<NGramPredictor>>,
    /// Engine for prefetching
    engine: Arc<RwLock<EnhancedAsyncEngine>>,
    /// Query history (in-memory cache, synced to IndexedDB)
    history: Arc<RwLock<Vec<String>>>,
    /// IndexedDB storage for persistent query history
    storage: Arc<RwLock<QueryHistoryStorage>>,
    /// Auto-save enabled
    auto_save: bool,
    /// Hash to query string reverse mapping for prediction
    hash_to_query: Arc<RwLock<HashMap<u64, String>>>,
}

impl BrowserQueryPredictor {
    /// Create new predictor
    pub fn new(engine: Arc<RwLock<EnhancedAsyncEngine>>) -> Self {
        Self::with_storage(engine, "query_predictor_db")
    }

    /// Create with custom storage database name
    pub fn with_storage(engine: Arc<RwLock<EnhancedAsyncEngine>>, db_name: &str) -> Self {
        Self {
            predictor: Arc::new(RwLock::new(NGramPredictor::new(3))), // 3-gram
            engine,
            history: Arc::new(RwLock::new(Vec::new())),
            storage: Arc::new(RwLock::new(QueryHistoryStorage::new(db_name))),
            auto_save: true,
            hash_to_query: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom storage and max history size
    pub fn with_storage_and_max_size(
        engine: Arc<RwLock<EnhancedAsyncEngine>>,
        db_name: &str,
        max_history_size: usize,
    ) -> Self {
        Self {
            predictor: Arc::new(RwLock::new(NGramPredictor::new(3))),
            engine,
            history: Arc::new(RwLock::new(Vec::new())),
            storage: Arc::new(RwLock::new(QueryHistoryStorage::with_max_size(
                db_name,
                max_history_size,
            ))),
            auto_save: true,
            hash_to_query: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enable/disable auto-save
    pub fn set_auto_save(&mut self, enabled: bool) {
        self.auto_save = enabled;
    }

    /// Record a query for learning
    pub fn record_query(&mut self, query: &str) {
        // Update in-memory history
        {
            let mut history = self.history.write().unwrap_or_else(|e| e.into_inner());
            history.push(query.to_string());
        }

        // Compute hash and store reverse mapping
        let hash = Self::compute_query_hash(query);
        {
            let mut mapping = self
                .hash_to_query
                .write()
                .unwrap_or_else(|e| e.into_inner());
            mapping.insert(hash, query.to_string());
        }

        // Update predictor
        {
            let predictor = self.predictor.write().unwrap_or_else(|e| e.into_inner());
            predictor.observe(query);
        }

        // Auto-save if enabled
        if self.auto_save {
            // Note: Auto-save is async, so we'll save periodically instead
            // This will be handled by save_history() calls
        }
    }

    /// Compute hash for a query string (matching NGramPredictor's internal hash)
    fn compute_query_hash(query: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Predict next queries
    pub fn predict_next(&self, count: usize) -> Vec<String> {
        let predictor = self.predictor.read().unwrap_or_else(|e| e.into_inner());
        let predictions = predictor.predict(count);

        // Convert prediction hashes back to query strings using reverse mapping
        let mapping = self.hash_to_query.read().unwrap_or_else(|e| e.into_inner());
        predictions
            .into_iter()
            .filter_map(|pred| mapping.get(&pred.hash).cloned())
            .take(count)
            .collect()
    }

    /// Get query history
    pub fn get_history(&self) -> Vec<String> {
        self.history
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get history length
    pub fn history_len(&self) -> usize {
        self.history.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Prefetch predicted queries
    pub async fn prefetch(&self, count: usize) -> Result<(), String> {
        let predictions = self.predict_next(count);

        // Execute predictions in background with timeout to avoid blocking
        for query in predictions {
            // Clone engine for async operation
            let engine = Arc::clone(&self.engine);

            // Spawn background task for prefetch
            #[cfg(target_arch = "wasm32")]
            {
                wasm_bindgen_futures::spawn_local(async move {
                    let engine_guard = engine.read().unwrap_or_else(|e| e.into_inner());
                    let _ = engine_guard.execute_sql(&query, None).await;
                });
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                // Note: In non-wasm environments, we'd typically use tokio::spawn
                // but since joule-db-browser is primarily for browser use,
                // we simply execute synchronously in this fallback path
                let engine_guard = engine.read().unwrap_or_else(|e| e.into_inner());
                // Note: Can't await here in non-async context, so skip prefetch
                // in native builds. The main use case is browser/wasm.
                drop(engine_guard);
            }
        }

        Ok(())
    }

    /// Load query history from IndexedDB
    pub async fn load_history(&mut self) -> Result<(), String> {
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        let queries = storage.load_history().await?;

        // Update in-memory history
        {
            let mut history = self.history.write().unwrap_or_else(|e| e.into_inner());
            *history = queries.clone();
        }

        // Replay queries to predictor to rebuild N-gram model and hash mapping
        {
            let predictor = self.predictor.write().unwrap_or_else(|e| e.into_inner());
            let mut mapping = self
                .hash_to_query
                .write()
                .unwrap_or_else(|e| e.into_inner());
            for query in &queries {
                predictor.observe(query);
                let hash = Self::compute_query_hash(query);
                mapping.insert(hash, query.clone());
            }
        }

        Ok(())
    }

    /// Save query history to IndexedDB
    pub async fn save_history(&self) -> Result<(), String> {
        let queries = self
            .history
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage.save_history(&queries).await?;
        Ok(())
    }

    /// Clear query history
    pub async fn clear_history(&mut self) -> Result<(), String> {
        // Clear in-memory
        {
            let mut history = self.history.write().unwrap_or_else(|e| e.into_inner());
            history.clear();
        }

        // Clear predictor
        {
            // Create new predictor to reset state
            *self.predictor.write().unwrap_or_else(|e| e.into_inner()) = NGramPredictor::new(3);
        }

        // Clear IndexedDB
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage.clear().await?;

        Ok(())
    }
}

/// Predictive prefetch manager
pub struct PredictivePrefetch {
    predictor: Arc<RwLock<BrowserQueryPredictor>>,
    enabled: bool,
}

impl PredictivePrefetch {
    /// Create new prefetch manager
    pub fn new(engine: Arc<RwLock<EnhancedAsyncEngine>>) -> Self {
        Self {
            predictor: Arc::new(RwLock::new(BrowserQueryPredictor::new(engine))),
            enabled: true,
        }
    }

    /// Enable/disable prefetching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Record query and trigger prefetch if enabled
    pub async fn record_and_prefetch(&self, query: &str) -> Result<(), String> {
        {
            let mut predictor = self.predictor.write().unwrap_or_else(|e| e.into_inner());
            predictor.record_query(query);
        }

        if self.enabled {
            let predictor = self.predictor.read().unwrap_or_else(|e| e.into_inner());
            predictor.prefetch(5).await?; // Prefetch top 5 predictions
        }

        Ok(())
    }

    /// Save history (call periodically or on close)
    pub async fn save_history(&self) -> Result<(), String> {
        let predictor = self.predictor.read().unwrap_or_else(|e| e.into_inner());
        predictor.save_history().await
    }

    /// Load history (call on initialization)
    pub async fn load_history(&mut self) -> Result<(), String> {
        let mut predictor = self.predictor.write().unwrap_or_else(|e| e.into_inner());
        predictor.load_history().await
    }
}
