//! # JouleDB Features
//!
//! Specialized feature modules for JouleDB.
//!
//! ## Features
//!
//! - **Time Series** - Time-partitioned data with downsampling
//! - **Graph** - Graph database with traversals
//! - **Vector** - Vector similarity search (ANN)
//! - **Full-Text Search** - Text indexing and search
//! - **Embeddings** - Vector embeddings support
//! - **Columnar** - Column-oriented storage
//! - **SIMD** - SIMD-optimized distance calculations

/// SIMD-optimized distance calculations for vector similarity search.
/// Available on all platforms with fallback scalar implementations.
pub mod simd;

/// Persistence layer for durable storage of feature data structures.
pub mod persistence;

#[cfg(feature = "timeseries")]
pub mod timeseries;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "vector")]
pub mod vector;

#[cfg(feature = "fulltext")]
pub mod fulltext;

#[cfg(feature = "embeddings")]
pub mod embeddings;

#[cfg(feature = "columnar")]
pub mod columnar;

// Re-exports
#[cfg(feature = "timeseries")]
pub use timeseries::{Aggregation, DataPoint, DownsamplePolicy, TimeSeriesConfig, TimeSeriesStore};

#[cfg(feature = "graph")]
pub use graph::{Edge, GraphConfig, GraphStore, Node, PathResult, Traversal};

#[cfg(feature = "vector")]
pub use vector::{SearchResult, SimilarityMetric, VectorConfig, VectorIndex};

#[cfg(feature = "fulltext")]
pub use fulltext::{FullTextConfig, FullTextIndex, SearchHit, SearchQuery};

#[cfg(feature = "embeddings")]
pub use embeddings::{
    EmbeddingConfig, EmbeddingError, EmbeddingModel, EmbeddingStore, SimilarityResult,
};

#[cfg(feature = "columnar")]
pub use columnar::{
    Aggregation as ColumnarAggregation, Column, ColumnStore, ColumnStoreBuilder, DataType, Value,
};

// Persistence re-exports
pub use persistence::{
    FullTextPersistence, GraphPersistence, PersistResult, PersistedDataPoint, PersistedDocument,
    PersistedEdge, PersistedNode, PersistedPosting, PersistedVector, PersistenceError,
    StorageEngine, TimeSeriesPersistence, VectorPersistence,
};
