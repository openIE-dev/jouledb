use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::error::LedgerError;

/// Pluggable backend for committing Merkle roots.
///
/// Mirrors the on-chain ABI contract:
/// `commit_batch(batchId, merkleRoot, timeStart, timeEnd, aggregateKWh, aggregateKgCO2e, issuer)`
///
/// Implementations include in-memory (tests), file-backed (local production),
/// and future blockchain backends (Ethereum, Solana, Hyperledger).
pub trait LedgerBackend: Send + Sync {
    /// Backend name (e.g., "memory", "file", "ethereum").
    fn name(&self) -> &str;

    /// Commit a batch's Merkle root to the backend.
    fn commit_batch(
        &self,
        batch: &ReceiptBatch,
    ) -> impl std::future::Future<Output = Result<BatchCommitment, LedgerError>> + Send;

    /// Look up a previously committed batch by batch_id.
    fn get_commitment(
        &self,
        batch_id: &str,
    ) -> impl std::future::Future<Output = Result<Option<BatchCommitment>, LedgerError>> + Send;

    /// List committed batches (most recent first).
    fn list_commitments(
        &self,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<BatchCommitment>, LedgerError>> + Send;
}
