//! # joule-db-ledger
//!
//! Blockchain-anchored energy receipt layer for JouleDB.
//!
//! Every query executed through the energy-aware executor produces an
//! `LedgerEnergyReceipt` with per-query energy data, carbon estimates,
//! and device context. Receipts are batched into Merkle trees and committed
//! to a pluggable backend (file, permissioned chain, or public L2) for
//! tamper-proof attestation.
//!
//! ## Architecture
//!
//! ```text
//! Query Executor
//!     │
//!     ▼
//! ReceiptCollector ──mpsc──▶ BatchCommitter
//!                                 │
//!                                 ▼
//!                           MerkleTree::from_leaves()
//!                                 │
//!                                 ▼
//!                           LedgerBackend::commit_batch()
//!                                 │
//!                           ┌─────┴─────┐
//!                      MemoryBackend  FileBackend  (future: EthBackend)
//! ```
//!
//! ## Verification
//!
//! The HTTP endpoint `GET /api/v1/ledger/receipts/{id}/verify` returns
//! the receipt, its Merkle inclusion proof, the batch summary, and the
//! backend commitment reference. Third parties can independently verify
//! receipt integrity by recomputing the proof against the published root.

pub mod backend;
#[cfg(feature = "ethereum")]
pub mod backend_eth;
pub mod backend_file;
pub mod backend_memory;
pub mod batch;
pub mod carbon;
pub mod collector;
pub mod committer;
pub mod error;
#[cfg(feature = "http")]
pub mod http;
pub mod merkle;
pub mod receipt;

// Re-exports for convenience
pub use backend::LedgerBackend;
#[cfg(feature = "ethereum")]
pub use backend_eth::{EthBackendConfig, EthLedgerBackend};
pub use backend_file::FileLedgerBackend;
pub use backend_memory::MemoryLedgerBackend;
pub use batch::{BatchCommitment, ReceiptBatch};
pub use carbon::{
    CarbonConfig, CarbonDataSource, CarbonIntensityCache, StaticCarbonSource, joules_to_kg_co2e,
    joules_to_kg_co2e_dynamic, joules_to_kwh,
};
pub use collector::{CollectorMetrics, LedgerConfig, ReceiptCollector};
pub use committer::{BatchCommitter, CommitterMetrics, ReceiptStore};
pub use error::LedgerError;
pub use merkle::{MerkleProof, MerkleTree};
pub use receipt::{ExecutionStage, LedgerEnergyReceipt};
