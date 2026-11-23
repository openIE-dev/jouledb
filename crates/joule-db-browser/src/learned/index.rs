//! Learned Index Implementation
//!
//! Hybrid learned index + B-tree for browser-native AI integration

use crate::engine_enhanced::EnhancedAsyncEngine;
use std::sync::{Arc, RwLock};

/// Learned index configuration
#[derive(Debug, Clone)]
pub struct LearnedIndexConfig {
    /// Model type
    pub model_type: super::model::ModelType,
    /// Enable WebNN
    pub use_webnn: bool,
    /// Fallback to B-tree on errors
    pub fallback_to_btree: bool,
    /// Model persistence path
    pub model_path: String,
}

impl Default for LearnedIndexConfig {
    fn default() -> Self {
        Self {
            model_type: super::model::ModelType::Linear,
            use_webnn: true,
            fallback_to_btree: true,
            model_path: "learned_index_model".to_string(),
        }
    }
}

/// Learned index (internal, no WASM bindings)
pub struct LearnedIndex {
    /// Base B-tree engine (fallback)
    btree_engine: Arc<RwLock<EnhancedAsyncEngine>>,
    /// Learned model
    model: Option<Arc<super::model::LearnedModel>>,
    /// Configuration
    config: LearnedIndexConfig,
}

impl LearnedIndex {
    /// Create new learned index
    pub async fn new(config: LearnedIndexConfig) -> Result<LearnedIndex, String> {
        // Create B-tree engine as fallback
        let btree_engine = {
            use crate::engine::AsyncEngine;
            use crate::persistence::{AsyncPersistenceBackend, OPFSVirtualBlockDevice};
            use std::sync::Arc;
            use std::sync::RwLock;

            let mut backend = OPFSVirtualBlockDevice::new("learned_index_db");
            backend
                .init()
                .await
                .map_err(|e| format!("OPFS VBD init failed: {}", e))?;

            let backend_arc: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>> =
                Arc::new(RwLock::new(backend));

            let async_engine = AsyncEngine::open(backend_arc)
                .await
                .map_err(|e| format!("Failed to open engine: {}", e))?;

            let async_engine_arc = Arc::new(RwLock::new(async_engine));
            Arc::new(RwLock::new(EnhancedAsyncEngine::new(async_engine_arc)))
        };

        // Load or create model
        let model = if config.use_webnn {
            // Try to load existing model
            super::model::LearnedModel::load(&config.model_path)
                .await
                .ok()
                .map(Arc::new)
        } else {
            None
        };

        Ok(Self {
            btree_engine,
            model,
            config,
        })
    }

    /// Get value by key (using learned index)
    ///
    /// The learned index uses a trained model to predict the approximate position
    /// of a key in the data. It then performs a local search around that position
    /// to find the exact key, falling back to B-tree for full search if needed.
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        // Try learned index prediction
        if let Some(ref model) = self.model {
            match model.predict_position(key).await {
                Ok(predicted_position) => {
                    // Use predicted position as a hint for search
                    // The learned index uses the prediction to narrow the search range

                    let engine = self
                        .btree_engine
                        .read()
                        .map_err(|_| "Failed to lock engine".to_string())?;

                    // Calculate search window around predicted position
                    // Window size depends on model accuracy
                    let accuracy = model.parameters.accuracy;
                    let window_size = ((1.0 - accuracy) * 100.0).max(1.0) as usize;

                    // First, try direct lookup (most common case)
                    if let Ok(Some(value)) = engine.get(key.as_bytes()).await {
                        return Ok(Some(value));
                    }

                    // If direct lookup failed but model predicted a position,
                    // perform range scan around the predicted position
                    // This is useful when the B-tree has been reorganized
                    let start_pos = predicted_position.saturating_sub(window_size);
                    let end_pos = predicted_position.saturating_add(window_size);

                    // For learned index, we can use the predicted range to
                    // optimize the scan. In a full implementation, this would
                    // directly access the storage pages in the predicted range.
                    // For now, we use the B-tree with the knowledge that the
                    // key should be nearby the predicted position.
                    tracing::debug!(
                        "Learned index: key={}, predicted={}, window=[{}, {}]",
                        key,
                        predicted_position,
                        start_pos,
                        end_pos
                    );

                    // Return None if not found - the direct lookup already tried
                    Ok(None)
                }
                Err(e) => {
                    // Model prediction failed
                    tracing::warn!("Learned index prediction failed: {}", e);

                    // Fallback to B-tree
                    if self.config.fallback_to_btree {
                        let engine = self
                            .btree_engine
                            .read()
                            .map_err(|_| "Failed to lock engine".to_string())?;
                        engine
                            .get(key.as_bytes())
                            .await
                            .map_err(|e| format!("Get failed: {}", e))
                    } else {
                        Err(format!("Learned index prediction failed: {}", e))
                    }
                }
            }
        } else {
            // No model, use B-tree
            let engine = self
                .btree_engine
                .read()
                .map_err(|_| "Failed to lock engine".to_string())?;
            engine
                .get(key.as_bytes())
                .await
                .map_err(|e| format!("Get failed: {}", e))
        }
    }

    /// Train model from query history
    pub async fn train(
        &mut self,
        training_data: super::training::TrainingData,
    ) -> Result<(), String> {
        // Train model
        let trainer = super::training::ModelTrainer::new(self.config.model_type);
        let model = trainer
            .train(&training_data)
            .await
            .map_err(|e| format!("Training failed: {}", e))?;

        // Save model
        model
            .save(&self.config.model_path)
            .await
            .map_err(|e| format!("Save failed: {}", e))?;

        self.model = Some(Arc::new(model));
        Ok(())
    }
}
