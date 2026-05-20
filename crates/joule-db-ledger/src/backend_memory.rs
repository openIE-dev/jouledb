use chrono::Utc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

use crate::backend::LedgerBackend;
use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::error::LedgerError;

/// In-memory ledger backend for testing.
///
/// Stores commitments in a `Vec` protected by a `RwLock`. No persistence.
pub struct MemoryLedgerBackend {
    commitments: RwLock<Vec<BatchCommitment>>,
    seq: AtomicU64,
}

impl Default for MemoryLedgerBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryLedgerBackend {
    pub fn new() -> Self {
        Self {
            commitments: RwLock::new(Vec::new()),
            seq: AtomicU64::new(0),
        }
    }

    /// Return the total number of committed batches.
    pub async fn len(&self) -> usize {
        self.commitments.read().await.len()
    }
}

impl LedgerBackend for MemoryLedgerBackend {
    fn name(&self) -> &str {
        "memory"
    }

    async fn commit_batch(&self, batch: &ReceiptBatch) -> Result<BatchCommitment, LedgerError> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let commitment = BatchCommitment {
            batch_id: batch.batch_id.clone(),
            tx_ref: format!("mem:{}", seq),
            backend_name: "memory".to_string(),
            committed_at: Utc::now(),
            block_number: None,
            chain_overhead_joules: None,
            amortized_overhead_joules: None,
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

    fn sample_batch(id: &str) -> ReceiptBatch {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        ReceiptBatch {
            batch_id: id.to_string(),
            merkle_root: id.to_string(),
            receipt_count: 1,
            time_start: ts,
            time_end: ts + chrono::Duration::seconds(60),
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "test".to_string(),
            receipt_ids: vec!["r1".into()],
            tenant_id: None,
        }
    }

    #[tokio::test]
    async fn commit_and_get() {
        let backend = MemoryLedgerBackend::new();
        let batch = sample_batch("batch1");

        let commitment = backend.commit_batch(&batch).await.unwrap();
        assert_eq!(commitment.batch_id, "batch1");
        assert_eq!(commitment.backend_name, "memory");

        let found = backend.get_commitment("batch1").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().batch_id, "batch1");
    }

    #[tokio::test]
    async fn get_not_found() {
        let backend = MemoryLedgerBackend::new();
        let found = backend.get_commitment("nonexistent").await.unwrap();
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn list_commitments_ordering() {
        let backend = MemoryLedgerBackend::new();
        for i in 0..5 {
            let batch = sample_batch(&format!("batch{}", i));
            backend.commit_batch(&batch).await.unwrap();
        }

        let all = backend.list_commitments(10).await.unwrap();
        assert_eq!(all.len(), 5);

        let limited = backend.list_commitments(3).await.unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[tokio::test]
    async fn sequential_tx_refs() {
        let backend = MemoryLedgerBackend::new();
        let c1 = backend.commit_batch(&sample_batch("b1")).await.unwrap();
        let c2 = backend.commit_batch(&sample_batch("b2")).await.unwrap();
        assert_eq!(c1.tx_ref, "mem:0");
        assert_eq!(c2.tx_ref, "mem:1");
    }
}
