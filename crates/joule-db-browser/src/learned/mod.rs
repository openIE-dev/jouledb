//! Learned Indexes Module
//!
//! Implements learned indexes using WebNN for browser-native AI integration

pub mod index;
pub mod model;
pub mod training;

pub use index::{LearnedIndex, LearnedIndexConfig};
pub use model::{LearnedModel, ModelType};
pub use training::{ModelTrainer, TrainingData};
