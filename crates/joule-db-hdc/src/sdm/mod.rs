//! Sparse Distributed Memory (SDM)
//!
//! High-dimensional content-addressable memory based on Kanerva's work.
//! Provides O(1) approximate lookup and graceful degradation.
//!
//! # Overview
//!
//! SDM is a mathematical model of human long-term memory. It stores data
//! in a high-dimensional binary space where similar patterns activate
//! similar locations, enabling:
//!
//! - Content-addressable retrieval
//! - Fault tolerance (graceful degradation)
//! - Pattern completion from partial cues
//! - Automatic generalization
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_hdc::sdm::{SparseDistributedMemory, SDMAddress};
//!
//! // Create SDM with 1000 locations, 256-bit addresses, 128-byte data
//! let sdm = SparseDistributedMemory::new(1000, 256, 128);
//!
//! // Write data to content-based address
//! let data = b"Hello, SDM!";
//! let address = SDMAddress::from_data(data, 256);
//! sdm.write_bytes(&address, data).unwrap();
//!
//! // Read back (will be approximate but similar)
//! let recalled = sdm.read(&address);
//! ```

mod memory;

/// Attention-as-SDM bridge (Bricken & Pehlevan, NeurIPS 2021).
/// Transformer attention ≈ SDM read under L2 normalization.
pub mod attention;

pub use attention::AttentionSDM;
pub use memory::{NearestLocation, SDMAddress, SDMError, SDMStats, SparseDistributedMemory};
