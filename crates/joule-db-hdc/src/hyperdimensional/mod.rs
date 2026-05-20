//! Hyperdimensional Computing / Vector Symbolic Architecture (VSA)
//!
//! High-dimensional random vectors for symbolic computation.
//!
//! # Overview
//!
//! Hyperdimensional computing uses vectors with thousands of dimensions
//! (typically 10,000) to represent and manipulate symbolic information.
//! Key operations include:
//!
//! - **Binding** - Circular convolution to associate concepts
//! - **Bundling** - Element-wise addition for set union
//! - **Similarity** - Cosine similarity for matching
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_hdc::hyperdimensional::{HyperVector, HyperdimensionalStorage};
//!
//! let storage = HyperdimensionalStorage::new(10000);
//!
//! // Create and store vectors
//! let v1 = HyperVector::random(10000, 42);
//! storage.add_vector(v1.clone())?;
//!
//! // Search for similar vectors
//! let results = storage.similarity_search(&v1, 5)?;
//! ```

mod storage;
mod vector;

pub use storage::{HyperdimensionalStorage, SimilarityMatch};
pub use vector::{HDError, HyperVector};
