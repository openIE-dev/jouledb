use chrono::Utc;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

use crate::backend::LedgerBackend;
use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::error::LedgerError;

/// File-backed append-only ledger backend for local production use.
///
/// Stores one `BatchCommitment` per line in JSON-lines format.
/// Each write is followed by an fsync for durability.
pub struct FileLedgerBackend {
    path: PathBuf,
    /// In-memory cache of commitments (loaded on creation).
    cache: RwLock<Vec<BatchCommitment>>,
    seq: AtomicU64,
}

impl FileLedgerBackend {
    /// Create or open the file-backed backend.
    ///
    /// If the file exists, loads all prior commitments into memory.
    pub fn new(path: PathBuf) -> Result<Self, LedgerError> {
        let mut cache = Vec::new();
        let mut max_seq: u64 = 0;

        if path.exists() {
            let file = std::fs::File::open(&path)?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let commitment: BatchCommitment =
                    serde_json::from_str(&line).map_err(LedgerError::Json)?;
                // Extract sequence from tx_ref "file:N"
                if let Some(n) = commitment.tx_ref.strip_prefix("file:") {
                    if let Ok(seq) = n.parse::<u64>() {
                        max_seq = max_seq.max(seq + 1);
                    }
                }
                cache.push(commitment);
            }
        }

        Ok(Self {
            path,
            cache: RwLock::new(cache),
            seq: AtomicU64::new(max_seq),
        })
    }

    fn append_to_file(&self, commitment: &BatchCommitment) -> Result<(), LedgerError> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let json = serde_json::to_string(commitment)?;
        writeln!(file, "{}", json)?;
        file.sync_all()?;
        Ok(())
    }
}

impl LedgerBackend for FileLedgerBackend {
    fn name(&self) -> &str {
        "file"
    }

    async fn commit_batch(&self, batch: &ReceiptBatch) -> Result<BatchCommitment, LedgerError> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let commitment = BatchCommitment {
            batch_id: batch.batch_id.clone(),
            tx_ref: format!("file:{}", seq),
            backend_name: "file".to_string(),
            committed_at: Utc::now(),
            block_number: None,
            chain_overhead_joules: None,
            amortized_overhead_joules: None,
        };

        self.append_to_file(&commitment)?;
        self.cache.write().await.push(commitment.clone());
        Ok(commitment)
    }

    async fn get_commitment(&self, batch_id: &str) -> Result<Option<BatchCommitment>, LedgerError> {
        let cache = self.cache.read().await;
        Ok(cache.iter().find(|c| c.batch_id == batch_id).cloned())
    }

    async fn list_commitments(&self, limit: usize) -> Result<Vec<BatchCommitment>, LedgerError> {
        let cache = self.cache.read().await;
        let mut result: Vec<_> = cache.iter().rev().take(limit).cloned().collect();
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
    async fn commit_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commitments.jsonl");

        // Write
        {
            let backend = FileLedgerBackend::new(path.clone()).unwrap();
            backend.commit_batch(&sample_batch("batch1")).await.unwrap();
            backend.commit_batch(&sample_batch("batch2")).await.unwrap();
        }

        // Reload
        let backend = FileLedgerBackend::new(path).unwrap();
        let all = backend.list_commitments(10).await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].batch_id, "batch1");
        assert_eq!(all[1].batch_id, "batch2");
    }

    #[tokio::test]
    async fn get_commitment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commitments.jsonl");
        let backend = FileLedgerBackend::new(path).unwrap();

        backend.commit_batch(&sample_batch("batch1")).await.unwrap();

        let found = backend.get_commitment("batch1").await.unwrap();
        assert!(found.is_some());

        let not_found = backend.get_commitment("nope").await.unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn sequence_continues_after_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commitments.jsonl");

        {
            let backend = FileLedgerBackend::new(path.clone()).unwrap();
            let c = backend.commit_batch(&sample_batch("b1")).await.unwrap();
            assert_eq!(c.tx_ref, "file:0");
        }

        let backend = FileLedgerBackend::new(path).unwrap();
        let c = backend.commit_batch(&sample_batch("b2")).await.unwrap();
        assert_eq!(c.tx_ref, "file:1");
    }

    #[tokio::test]
    async fn empty_file_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commitments.jsonl");
        let backend = FileLedgerBackend::new(path).unwrap();
        let all = backend.list_commitments(10).await.unwrap();
        assert!(all.is_empty());
    }
}
