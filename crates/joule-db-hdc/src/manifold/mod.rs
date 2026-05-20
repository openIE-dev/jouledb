//! Information Manifold
//!
//! Geodesic-based indexing treating data as points on a Riemannian manifold.
//! Similar data points are geodesically close.
//!
//! # Overview
//!
//! The Information Manifold provides:
//! - **Geodesic distance** - Natural distance on curved space
//! - **K-NN search** - Find k nearest neighbors
//! - **Range queries** - Find all points within radius
//! - **LSH indexing** - O(1) approximate nearest neighbor
//! - **HNSW indexing** - O(log n) accurate nearest neighbor
//!
//! # Example
//!
//! ```rust
//! use joule_db_hdc::manifold::{InformationManifold, ManifoldPoint};
//!
//! let manifold = InformationManifold::new(64);
//!
//! // Insert data points
//! manifold.insert("doc1", b"hello world");
//! manifold.insert("doc2", b"hello there");
//!
//! // Find similar points
//! let neighbors = manifold.nearest_by_data(b"hello", 5);
//! ```

mod hnsw;
mod index;
pub mod ivf;
mod lsh;
mod point;
pub mod quantization;

pub use hnsw::{DistanceMetric, HNSWIndex, HNSWResult};
pub use index::{InformationManifold, ManifoldError, ManifoldStats, NeighborResult};
pub use ivf::{IVFIndex, IVFResult};
pub use lsh::{LSHIndex, LSHResult, LSHTable};
pub use point::ManifoldPoint;
pub use quantization::{PQResult, ProductQuantizer, SQResult, ScalarQuantizer};
