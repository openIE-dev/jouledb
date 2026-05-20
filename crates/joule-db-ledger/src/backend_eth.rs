//! Ethereum L2 stub backend for energy receipt attestation.
//!
//! This is a stub implementation that simulates committing batch Merkle roots
//! to an Ethereum L2 smart contract. In production, this would use ethers-rs
//! or alloy to submit transactions.
//!
//! ## Target contract ABI
//!
//! ```solidity
//! // SPDX-License-Identifier: MIT
//! pragma solidity ^0.8.20;
//!
//! contract EnergyLedger {
//!     event BatchCommitted(
//!         bytes32 indexed batchId,
//!         bytes32 merkleRoot,
//!         uint256 timeStart,
//!         uint256 timeEnd,
//!         uint256 aggregateKwhMicro,
//!         uint256 aggregateKgCo2eMicro
//!     );
//!
//!     function commitBatch(
//!         bytes32 batchId,
//!         bytes32 merkleRoot,
//!         uint256 timeStart,
//!         uint256 timeEnd,
//!         uint256 aggregateKwhMicro,
//!         uint256 aggregateKgCo2eMicro
//!     ) external;
//! }
//! ```

use chrono::Utc;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

use crate::backend::LedgerBackend;
use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::error::LedgerError;

/// Configuration for the Ethereum L2 backend.
#[derive(Debug, Clone)]
pub struct EthBackendConfig {
    /// JSON-RPC endpoint URL for the L2 chain.
    pub rpc_url: String,
    /// Contract address (hex, 0x-prefixed).
    pub contract_address: String,
    /// Chain ID of the target L2.
    pub chain_id: u64,
    /// Signer private key (hex). None for read-only / stub mode.
    pub signer_key: Option<String>,
}

/// Stub Ethereum L2 backend.
///
/// Simulates on-chain commitment by computing a synthetic transaction hash
/// and storing commitments in memory. Replace with real ethers-rs calls for
/// production use.
pub struct EthLedgerBackend {
    config: EthBackendConfig,
    commitments: RwLock<Vec<BatchCommitment>>,
    seq: AtomicU64,
}

impl EthLedgerBackend {
    /// Create a new stub Ethereum backend.
    pub fn new(config: EthBackendConfig) -> Self {
        Self {
            config,
            commitments: RwLock::new(Vec::new()),
            seq: AtomicU64::new(0),
        }
    }

    /// Compute a synthetic transaction hash from batch_id and sequence.
    fn synthetic_tx_hash(batch_id: &str, seq: u64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}", batch_id, seq).as_bytes());
        let hash = hasher.finalize();
        format!("0x{}", hex::encode(hash))
    }
}

impl LedgerBackend for EthLedgerBackend {
    fn name(&self) -> &str {
        "ethereum-l2-stub"
    }

    async fn commit_batch(&self, batch: &ReceiptBatch) -> Result<BatchCommitment, LedgerError> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let tx_hash = Self::synthetic_tx_hash(&batch.batch_id, seq);

        tracing::info!(
            batch_id = %batch.batch_id,
            chain_id = self.config.chain_id,
            contract = %self.config.contract_address,
            tx_hash = %tx_hash,
            receipt_count = batch.receipt_count,
            "Stub: would commit batch to Ethereum L2"
        );

        let commitment = BatchCommitment {
            batch_id: batch.batch_id.clone(),
            tx_ref: tx_hash,
            backend_name: format!("ethereum-l2-stub:{}", self.config.chain_id),
            committed_at: Utc::now(),
            block_number: Some(seq + 1),
            chain_overhead_joules: Some(0.05), // Stub: ~50mJ per L2 batch tx
            amortized_overhead_joules: None,   // Computed by committer
        };

        self.commitments.write().await.push(commitment.clone());
        Ok(commitment)
    }

    async fn get_commitment(&self, batch_id: &str) -> Result<Option<BatchCommitment>, LedgerError> {
        let store = self.commitments.read().await;
        Ok(store.iter().find(|c| c.batch_id == batch_id).cloned())
    }

    async fn list_commitments(&self, limit: usize) -> Result<Vec<BatchCommitment>, LedgerError> {
        let store = self.commitments.read().await;
        let mut result: Vec<_> = store.iter().rev().take(limit).cloned().collect();
        result.reverse();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn test_config() -> EthBackendConfig {
        EthBackendConfig {
            rpc_url: "https://localhost:8545".to_string(),
            contract_address: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
            chain_id: 42161, // Arbitrum One
            signer_key: None,
        }
    }

    fn sample_batch(id: &str) -> ReceiptBatch {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        ReceiptBatch {
            batch_id: id.to_string(),
            merkle_root: id.to_string(),
            receipt_count: 5,
            time_start: ts,
            time_end: ts + chrono::Duration::seconds(60),
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "test".to_string(),
            receipt_ids: vec!["r1".into(), "r2".into()],
            tenant_id: None,
        }
    }

    #[tokio::test]
    async fn commit_and_get() {
        let backend = EthLedgerBackend::new(test_config());
        let commitment = backend.commit_batch(&sample_batch("batch1")).await.unwrap();
        assert_eq!(commitment.batch_id, "batch1");

        let found = backend.get_commitment("batch1").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().batch_id, "batch1");
    }

    #[tokio::test]
    async fn block_numbers_increment() {
        let backend = EthLedgerBackend::new(test_config());
        let c1 = backend.commit_batch(&sample_batch("b1")).await.unwrap();
        let c2 = backend.commit_batch(&sample_batch("b2")).await.unwrap();
        let c3 = backend.commit_batch(&sample_batch("b3")).await.unwrap();
        assert_eq!(c1.block_number, Some(1));
        assert_eq!(c2.block_number, Some(2));
        assert_eq!(c3.block_number, Some(3));
    }

    #[tokio::test]
    async fn backend_name_includes_chain_id() {
        let backend = EthLedgerBackend::new(test_config());
        let commitment = backend.commit_batch(&sample_batch("b1")).await.unwrap();
        assert_eq!(commitment.backend_name, "ethereum-l2-stub:42161");
    }

    #[tokio::test]
    async fn list_ordering() {
        let backend = EthLedgerBackend::new(test_config());
        for i in 0..5 {
            backend
                .commit_batch(&sample_batch(&format!("b{}", i)))
                .await
                .unwrap();
        }
        let all = backend.list_commitments(10).await.unwrap();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0].batch_id, "b0");
        assert_eq!(all[4].batch_id, "b4");

        let limited = backend.list_commitments(3).await.unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[tokio::test]
    async fn tx_ref_format() {
        let backend = EthLedgerBackend::new(test_config());
        let commitment = backend.commit_batch(&sample_batch("b1")).await.unwrap();
        // Should be 0x-prefixed hex SHA256 (66 chars: 0x + 64 hex)
        assert!(commitment.tx_ref.starts_with("0x"));
        assert_eq!(commitment.tx_ref.len(), 66);

        // Reports chain overhead
        assert!(commitment.chain_overhead_joules.is_some());
    }
}
