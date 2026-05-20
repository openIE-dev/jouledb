//! Index abstractions for JouleDB
//!
//! Provides traits for different index types (B-tree, Hash, etc.)

pub mod art;
pub mod gin;
pub mod hnsw;
pub mod kdtree3;
pub mod minhash_lsh;
pub mod octree3;
pub mod rtree3;
mod btree;
mod traits;

pub use art::AdaptiveRadixTree;
pub use btree::{BTreeIndex, BTreeUniqueIndex, HashUniqueIndex};
pub use kdtree3::KdTree3;
pub use minhash_lsh::MinHashLshIndex;
pub use octree3::Octree3;
pub use rtree3::RTree3;
pub use traits::{
    Bound, Index, IndexEntry, IndexIterator, OrderedIndex, ScanDirection, SimilarityIndex,
    UniqueIndex,
};
