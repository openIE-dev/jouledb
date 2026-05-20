//! Learned Indexes
//!
//! Machine learning models for optimized data access patterns.
//!
//! Instead of traditional B-Tree indexes that have O(log n) lookup, learned indexes
//! use statistical models to predict the position of data in O(1) time with
//! bounded error correction.
//!
//! ## Key Concepts
//!
//! - **Model Types**: Linear, polynomial, or piecewise linear regression
//! - **Error Bounds**: Guaranteed maximum prediction error for fallback search
//! - **Training**: Fit model to existing key distribution
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::learned::{LearnedIndex, LearnedIndexModel, ModelType};
//!
//! // Create and train a learned index
//! let mut index = LearnedIndex::new(0.0, 1000.0, 10000);
//!
//! // Train on sorted key-position pairs
//! let training_data: Vec<(f64, usize)> = (0..10000)
//!     .map(|i| (i as f64, i))
//!     .collect();
//! index.train(&training_data).unwrap();
//!
//! // Lookup - returns predicted position and search bounds
//! let result = index.lookup(500.0);
//! println!("Position: ~{}, search range: [{}, {}]",
//!     result.predicted, result.min_bound, result.max_bound);
//! ```

mod index;
mod model;

pub use index::{LearnedIndex, LookupResult};
pub use model::{LearnedIndexModel, ModelType};

use thiserror::Error;

/// Errors for learned index operations
#[derive(Error, Debug, Clone)]
pub enum LearnedError {
    /// Model not trained
    #[error("model not trained - call train() first")]
    NotTrained,

    /// Invalid training data
    #[error("invalid training data: {0}")]
    InvalidTrainingData(String),

    /// Key out of bounds
    #[error("key {0} out of bounds [{1}, {2}]")]
    KeyOutOfBounds(f64, f64, f64),

    /// Model fitting failed
    #[error("model fitting failed: {0}")]
    FittingFailed(String),

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for learned index operations
pub type LearnedResult<T> = Result<T, LearnedError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LearnedError::NotTrained;
        assert!(err.to_string().contains("not trained"));

        let err = LearnedError::KeyOutOfBounds(50.0, 0.0, 100.0);
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_module_exports() {
        // Verify all exports are accessible
        let _model_type = ModelType::Linear;
        let model = LearnedIndexModel::new(ModelType::Linear);
        assert!(!model.is_trained());
    }
}
