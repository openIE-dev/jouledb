//! Neurosymbolic Integration
//!
//! Combines neural pattern matching with symbolic reasoning for hybrid AI queries.
//!
//! ## Key Concepts
//!
//! - **Neural Layer**: Pattern matching using hyperdimensional vectors
//! - **Symbolic Layer**: Rule-based forward chaining reasoner
//! - **Hybrid Integration**: Combines results from both layers
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::neurosymbolic::{NeurosymbolicDB, QueryType};
//!
//! let mut db = NeurosymbolicDB::new(1000); // 1000-dim vectors
//!
//! // Add symbolic rules
//! db.add_rule("human(X)", "mortal(X)");
//! db.add_fact("human", vec!["socrates"]);
//!
//! // Add neural patterns
//! db.add_pattern("cat", &[0.1, 0.2, 0.8, ...]);
//!
//! // Hybrid query
//! let results = db.query("mortal(?X)", QueryType::Hybrid)?;
//! ```

mod integration;
mod neural;
mod symbolic;

pub use integration::{HybridResult, NeurosymbolicDB, QueryType};
pub use neural::{NeuralLayer, PatternMatch};
pub use symbolic::{Binding, Fact, Rule, SymbolicReasoner};

use thiserror::Error;

/// Errors for neurosymbolic operations
#[derive(Error, Debug, Clone)]
pub enum NeurosymbolicError {
    /// Pattern not found
    #[error("pattern not found: {0}")]
    PatternNotFound(String),

    /// Rule parse error
    #[error("invalid rule syntax: {0}")]
    InvalidRule(String),

    /// Fact parse error
    #[error("invalid fact syntax: {0}")]
    InvalidFact(String),

    /// Query error
    #[error("query error: {0}")]
    QueryError(String),

    /// Dimension mismatch
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Lock error
    #[error("lock error: {0}")]
    LockError(String),
}

/// Result type for neurosymbolic operations
pub type NeurosymbolicResult<T> = Result<T, NeurosymbolicError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NeurosymbolicError::PatternNotFound("cat".to_string());
        assert!(err.to_string().contains("cat"));
    }
}
