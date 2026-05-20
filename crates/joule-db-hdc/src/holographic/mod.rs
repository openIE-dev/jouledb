//! Holographic Associative Memory (HAM)
//!
//! Brain-inspired storage using complex vectors and interference patterns.
//!
//! # Overview
//!
//! HAM stores data as interference patterns between reference and object beams,
//! similar to optical holograms. This enables:
//!
//! - Associative recall from partial cues
//! - Pattern completion
//! - Similarity-based search
//! - Content-addressable storage
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_hdc::holographic::{HolographicStorage, Complex};
//!
//! let storage = HolographicStorage::new(512);
//!
//! // Store a pattern (as complex vector pairs)
//! let data: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
//! storage.store_pattern("pattern1".into(), &data).unwrap();
//!
//! // Recall from partial cue
//! let partial = &data[..256];
//! let recalled = storage.recall_pattern("pattern1", partial).unwrap();
//! ```

mod complex;
pub mod index;
mod interference;
mod storage;

pub use complex::Complex;
pub use interference::InterferencePattern;
pub use storage::{HolographicError, HolographicStats, HolographicStorage, SimilarityResult};
