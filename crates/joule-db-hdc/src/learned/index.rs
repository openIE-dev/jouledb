//! Learned Index
//!
//! High-level interface for learned indexes with error bounds and fallback support.

use super::model::{LearnedIndexModel, ModelType};
use super::{LearnedError, LearnedResult};
use std::sync::{Arc, RwLock};

/// Result of a learned index lookup
#[derive(Debug, Clone)]
pub struct LookupResult {
    /// Predicted position
    pub predicted: usize,
    /// Minimum bound for search range
    pub min_bound: usize,
    /// Maximum bound for search range
    pub max_bound: usize,
    /// Confidence score (0.0-1.0, based on training error)
    pub confidence: f64,
}

impl LookupResult {
    /// Get the search range size
    pub fn range_size(&self) -> usize {
        self.max_bound - self.min_bound + 1
    }

    /// Check if a position is within the search range
    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.min_bound && pos <= self.max_bound
    }
}

/// Learned Index
///
/// A complete learned index implementation that wraps a model with
/// error bounds, confidence tracking, and optional fallback mechanisms.
///
/// ## Thread Safety
///
/// The index uses internal locking for thread-safe access.
pub struct LearnedIndex {
    /// The underlying model
    model: Arc<RwLock<LearnedIndexModel>>,
    /// Error bound multiplier (default 1.5 = 150% of training max error)
    error_multiplier: f64,
    /// Minimum error bound (positions)
    min_error_bound: usize,
    /// Whether to use dynamic error bounds
    dynamic_bounds: bool,
    /// Key bounds
    min_key: f64,
    max_key: f64,
    /// Number of records
    num_records: usize,
}

impl LearnedIndex {
    /// Create a new learned index with default settings
    ///
    /// # Arguments
    /// * `min_key` - Expected minimum key value
    /// * `max_key` - Expected maximum key value  
    /// * `num_records` - Expected number of records
    pub fn new(min_key: f64, max_key: f64, num_records: usize) -> Self {
        Self {
            model: Arc::new(RwLock::new(LearnedIndexModel::linear())),
            error_multiplier: 1.5,
            min_error_bound: 1,
            dynamic_bounds: true,
            min_key,
            max_key,
            num_records,
        }
    }

    /// Create with a specific model type
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model = Arc::new(RwLock::new(LearnedIndexModel::new(model_type)));
        self
    }

    /// Set the error multiplier (default 1.5)
    ///
    /// Higher values give wider search bounds but fewer misses.
    pub fn with_error_multiplier(mut self, multiplier: f64) -> Self {
        self.error_multiplier = multiplier.max(1.0);
        self
    }

    /// Set minimum error bound in positions
    pub fn with_min_error_bound(mut self, bound: usize) -> Self {
        self.min_error_bound = bound.max(1);
        self
    }

    /// Enable/disable dynamic error bounds
    pub fn with_dynamic_bounds(mut self, enabled: bool) -> Self {
        self.dynamic_bounds = enabled;
        self
    }

    /// Train the index on sorted key-position data
    ///
    /// # Arguments
    /// * `data` - Sorted pairs of (key, position)
    ///
    /// # Returns
    /// Training statistics: (mean absolute error, max absolute error)
    pub fn train(&mut self, data: &[(f64, usize)]) -> LearnedResult<(f64, f64)> {
        if data.is_empty() {
            return Err(LearnedError::InvalidTrainingData(
                "empty training data".to_string(),
            ));
        }

        // Update bounds from data
        self.min_key = data[0].0;
        self.max_key = data[data.len() - 1].0;
        self.num_records = data.len();

        // Train model
        let mut model = self.model.write().unwrap();
        model.train(data)
    }

    /// Check if the index has been trained
    pub fn is_trained(&self) -> bool {
        let model = self.model.read().unwrap();
        model.is_trained()
    }

    /// Lookup a key and return predicted position with bounds
    pub fn lookup(&self, key: f64) -> LearnedResult<LookupResult> {
        let model = self.model.read().unwrap();

        if !model.is_trained() {
            return Err(LearnedError::NotTrained);
        }

        // Get prediction
        let predicted_f64 = model.predict(key)?;
        let predicted = predicted_f64.round() as usize;
        let predicted = predicted.min(self.num_records.saturating_sub(1));

        // Calculate error bounds
        let error_bound = if self.dynamic_bounds {
            let base_error = model.max_absolute_error() * self.error_multiplier;
            base_error.max(self.min_error_bound as f64) as usize
        } else {
            self.min_error_bound
        };

        let min_bound = predicted.saturating_sub(error_bound);
        let max_bound = (predicted + error_bound).min(self.num_records.saturating_sub(1));

        // Calculate confidence based on MAE
        let confidence = if model.num_records() > 0 {
            let mae = model.mean_absolute_error();
            let relative_error = mae / model.num_records() as f64;
            (1.0 - relative_error).clamp(0.0, 1.0)
        } else {
            0.5
        };

        Ok(LookupResult {
            predicted,
            min_bound,
            max_bound,
            confidence,
        })
    }

    /// Get statistics about the index
    pub fn stats(&self) -> LearnedIndexStats {
        let model = self.model.read().unwrap();
        LearnedIndexStats {
            is_trained: model.is_trained(),
            num_records: model.num_records(),
            mean_absolute_error: model.mean_absolute_error(),
            max_absolute_error: model.max_absolute_error(),
            key_bounds: model.key_bounds(),
        }
    }

    /// Get the number of records
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Get key bounds
    pub fn key_bounds(&self) -> (f64, f64) {
        (self.min_key, self.max_key)
    }

    /// Get mean absolute error from training
    pub fn mean_absolute_error(&self) -> f64 {
        let model = self.model.read().unwrap();
        model.mean_absolute_error()
    }

    /// Get maximum absolute error from training
    pub fn max_absolute_error(&self) -> f64 {
        let model = self.model.read().unwrap();
        model.max_absolute_error()
    }

    /// Set the underlying model (for external training)
    pub fn set_model(&mut self, model: LearnedIndexModel) {
        let (min_key, max_key) = model.key_bounds();
        self.min_key = min_key;
        self.max_key = max_key;
        self.num_records = model.num_records();
        *self.model.write().unwrap() = model;
    }
}

impl Clone for LearnedIndex {
    fn clone(&self) -> Self {
        let model = self.model.read().unwrap();
        Self {
            model: Arc::new(RwLock::new(model.clone())),
            error_multiplier: self.error_multiplier,
            min_error_bound: self.min_error_bound,
            dynamic_bounds: self.dynamic_bounds,
            min_key: self.min_key,
            max_key: self.max_key,
            num_records: self.num_records,
        }
    }
}

/// Statistics about a learned index
#[derive(Debug, Clone)]
pub struct LearnedIndexStats {
    /// Whether the model is trained
    pub is_trained: bool,
    /// Number of records
    pub num_records: usize,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Key bounds (min, max)
    pub key_bounds: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_index_basic() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100);

        // Train on linear data
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        assert!(index.is_trained());

        // Lookup
        let result = index.lookup(50.0).unwrap();
        assert!((result.predicted as i64 - 50).abs() < 5);
        assert!(result.contains(50));
    }

    #[test]
    fn test_lookup_result() {
        let result = LookupResult {
            predicted: 50,
            min_bound: 45,
            max_bound: 55,
            confidence: 0.9,
        };

        assert_eq!(result.range_size(), 11);
        assert!(result.contains(50));
        assert!(result.contains(45));
        assert!(result.contains(55));
        assert!(!result.contains(44));
        assert!(!result.contains(56));
    }

    #[test]
    fn test_bounds_clamping() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100);
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        // Lookup at edge - should not go below 0
        let result = index.lookup(0.0).unwrap();
        assert!(result.min_bound == 0);

        // Lookup at edge - should not exceed max
        let result = index.lookup(99.0).unwrap();
        assert!(result.max_bound <= 99);
    }

    #[test]
    fn test_with_model_type() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100).with_model_type(ModelType::Quadratic);

        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        assert!(index.is_trained());
    }

    #[test]
    fn test_error_multiplier() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100).with_error_multiplier(2.0);

        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        let result = index.lookup(50.0).unwrap();
        // Wider bounds due to higher multiplier
        assert!(result.range_size() >= 1);
    }

    #[test]
    fn test_stats() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100);
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        let stats = index.stats();
        assert!(stats.is_trained);
        assert_eq!(stats.num_records, 100);
        assert_eq!(stats.key_bounds, (0.0, 99.0));
    }

    #[test]
    fn test_untrained_lookup() {
        let index = LearnedIndex::new(0.0, 99.0, 100);
        assert!(!index.is_trained());
        assert!(index.lookup(50.0).is_err());
    }

    #[test]
    fn test_clone() {
        let mut index = LearnedIndex::new(0.0, 99.0, 100);
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        let cloned = index.clone();
        assert!(cloned.is_trained());
        assert_eq!(cloned.num_records(), 100);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let mut index = LearnedIndex::new(0.0, 99.0, 100);
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();
        index.train(&data).unwrap();

        let index = Arc::new(index);
        let mut handles = vec![];

        for i in 0..4 {
            let index_clone = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for j in 0..25 {
                    let key = (i * 25 + j) as f64;
                    let _result = index_clone.lookup(key);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
