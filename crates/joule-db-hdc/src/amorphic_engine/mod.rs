//! Amorphic Engine
//!
//! Unified API combining all HDC database technologies into a single engine.
//!
//! ## Core Innovations
//!
//! 1. **Sparse Distributed Memory** - Content-addressable by default
//! 2. **Holographic Storage** - Every node contains complete information
//! 3. **Predictive Prefetching** - High cache hit rate
//! 4. **Information Manifold** - O(1) similarity search via geodesics
//! 5. **Thermodynamic Optimizer** - Self-tuning query execution
//! 6. **Hyperdimensional Computing** - 10,000-dim vector operations
//! 7. **Spiking Neural Networks** - Temporal data processing
//! 8. **Learned Indexes** - ML-optimized data access
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::amorphic_engine::{AmorphicEngine, AmorphicEngineConfig};
//!
//! let config = AmorphicEngineConfig::minimal();
//! let db = AmorphicEngine::with_config(config, "node_1");
//!
//! // Write data
//! db.write("key1", b"value1")?;
//!
//! // Read with predictive caching
//! let value = db.read("key1");
//!
//! // Content-based lookup
//! let similar = db.content_lookup(b"value1");
//! ```

mod distributed;
mod unified;

pub use distributed::{DistributedCluster, DistributedNode, LWWRegister, VectorClock};
pub use unified::{AmorphicEngine, AmorphicEngineConfig, AmorphicEngineStats};

use thiserror::Error;

/// Errors for AmorphicEngine operations
#[derive(Error, Debug, Clone)]
pub enum AmorphicEngineError {
    /// SDM error
    #[error("SDM error: {0}")]
    Sdm(String),

    /// Holographic error
    #[error("Holographic error: {0}")]
    Holographic(String),

    /// Hyperdimensional error
    #[error("Hyperdimensional error: {0}")]
    Hyperdimensional(String),

    /// Manifold error
    #[error("Manifold error: {0}")]
    Manifold(String),

    /// Predictor error
    #[error("Predictor error: {0}")]
    Predictor(String),

    /// Optimizer error
    #[error("Optimizer error: {0}")]
    Optimizer(String),

    /// SNN error
    #[error("SNN error: {0}")]
    Snn(String),

    /// Learned index error
    #[error("Learned index error: {0}")]
    Learned(String),

    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Lock error
    #[error("Lock error: {0}")]
    LockError(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for AmorphicEngine operations
pub type AmorphicEngineResult<T> = Result<T, AmorphicEngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AmorphicEngineError::Sdm("test error".to_string());
        assert!(err.to_string().contains("SDM"));
        assert!(err.to_string().contains("test error"));
    }
}
