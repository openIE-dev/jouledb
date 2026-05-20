//! Adversarial and stress tests for joule-db-ledger.
//!
//! Covers clock skew, duplicates, backend errors, sustained load,
//! malformed data, Merkle edge cases, and file backend corruption.

use chrono::{TimeZone, Utc};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

use joule_db_ledger::*;

// =========================================================================
// Helpers
// =========================================================================

fn make_leaf(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

// =========================================================================
// Clock skew tests
// =========================================================================

#[tokio::test]
async fn clock_skew_end_before_start() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);

    let start = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    let end = start - chrono::Duration::seconds(10); // end before start

    // Should still succeed — receipts record whatever timestamps they're given
    let receipt = collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", start, end)
        .unwrap();
    assert_eq!(receipt.timestamp_start, start);
    assert_eq!(receipt.timestamp_end, end);
}

#[tokio::test]
async fn clock_skew_epoch_zero() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);

    let epoch = Utc.with_ymd_and_hms(1970, 1, 1, 0, 0, 0).unwrap();
    let receipt = collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", epoch, epoch)
        .unwrap();
    assert_eq!(receipt.timestamp_start, epoch);
}

#[tokio::test]
async fn clock_skew_far_future() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);

    let future = Utc.with_ymd_and_hms(9999, 12, 31, 23, 59, 59).unwrap();
    let receipt = collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", future, future)
        .unwrap();
    assert_eq!(receipt.timestamp_start, future);
}

#[tokio::test]
async fn clock_skew_same_id_different_timestamp() {
    // Two receipts with same qid+tenant but different timestamps → different receipt_ids
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);

    let ts1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let ts2 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 1).unwrap();

    let r1 = collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", ts1, ts1)
        .unwrap();
    let r2 = collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", ts2, ts2)
        .unwrap();
    assert_ne!(r1.receipt_id, r2.receipt_id);
}

// =========================================================================
// Duplicate receipt tests
// =========================================================================

#[tokio::test]
async fn duplicate_receipts_deduped_in_committer() {
    let config = LedgerConfig {
        batch_max_receipts: 1000,
        batch_max_interval_secs: 300,
        ..Default::default()
    };

    let backend = Arc::new(MemoryLedgerBackend::new());
    let (tx, rx) = mpsc::channel(100);
    let (handle, store) = BatchCommitter::spawn(config, backend, rx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    let receipt_id = LedgerEnergyReceipt::compute_id("q1", "t1", &ts);

    // Send 10 identical receipts directly
    for _ in 0..10 {
        let receipt = LedgerEnergyReceipt {
            receipt_id: receipt_id.clone(),
            qid: "q1".to_string(),
            tenant_id: "t1".to_string(),
            workload_tag: None,
            energy_joules_total: 0.005,
            energy_joules_by_stage: Default::default(),
            kwh: 0.005 / 3_600_000.0,
            kg_co2e: (0.005 / 3_600_000.0) * 0.4,
            grid_region: "UNKNOWN".to_string(),
            grid_factor_source: "test".to_string(),
            timestamp_start: ts,
            timestamp_end: ts,
            device_target: "cpu".to_string(),
            algorithm_type: "btree".to_string(),
        };
        tx.send(receipt).await.unwrap();
    }

    drop(tx);
    handle.await.unwrap();

    let s = store.read().await;
    assert_eq!(s.receipts.len(), 1);
}

#[tokio::test]
async fn same_id_different_content_still_deduped() {
    let config = LedgerConfig {
        batch_max_receipts: 1000,
        batch_max_interval_secs: 300,
        ..Default::default()
    };

    let backend = Arc::new(MemoryLedgerBackend::new());
    let (tx, rx) = mpsc::channel(100);
    let (handle, store) = BatchCommitter::spawn(config, backend, rx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    let receipt_id = LedgerEnergyReceipt::compute_id("q1", "t1", &ts);

    // Send same receipt_id but different energy values
    for i in 0..3 {
        let receipt = LedgerEnergyReceipt {
            receipt_id: receipt_id.clone(),
            qid: "q1".to_string(),
            tenant_id: "t1".to_string(),
            workload_tag: None,
            energy_joules_total: 0.001 * (i as f64 + 1.0),
            energy_joules_by_stage: Default::default(),
            kwh: 0.001 / 3_600_000.0,
            kg_co2e: 0.0,
            grid_region: "UNKNOWN".to_string(),
            grid_factor_source: "test".to_string(),
            timestamp_start: ts,
            timestamp_end: ts,
            device_target: "cpu".to_string(),
            algorithm_type: "btree".to_string(),
        };
        tx.send(receipt).await.unwrap();
    }

    drop(tx);
    handle.await.unwrap();

    let s = store.read().await;
    // Dedup is by receipt_id — first wins
    assert_eq!(s.receipts.len(), 1);
}

// =========================================================================
// Backend error tests
// =========================================================================

/// A backend that fails every Nth commit.
struct FailingBackend {
    inner: MemoryLedgerBackend,
    call_count: AtomicU64,
    fail_every: u64,
}

impl FailingBackend {
    fn new(fail_every: u64) -> Self {
        Self {
            inner: MemoryLedgerBackend::new(),
            call_count: AtomicU64::new(0),
            fail_every,
        }
    }
}

impl LedgerBackend for FailingBackend {
    fn name(&self) -> &str {
        "failing"
    }

    async fn commit_batch(
        &self,
        batch: &ReceiptBatch,
    ) -> Result<BatchCommitment, joule_db_ledger::LedgerError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
        if n % self.fail_every == 0 {
            return Err(joule_db_ledger::LedgerError::CommitError(
                "simulated failure".to_string(),
            ));
        }
        self.inner.commit_batch(batch).await
    }

    async fn get_commitment(
        &self,
        batch_id: &str,
    ) -> Result<Option<BatchCommitment>, joule_db_ledger::LedgerError> {
        self.inner.get_commitment(batch_id).await
    }

    async fn list_commitments(
        &self,
        limit: usize,
    ) -> Result<Vec<BatchCommitment>, joule_db_ledger::LedgerError> {
        self.inner.list_commitments(limit).await
    }
}

#[tokio::test]
async fn committer_survives_backend_failure() {
    let config = LedgerConfig {
        batch_max_receipts: 2,
        batch_max_interval_secs: 300,
        ..Default::default()
    };

    // Fail every 2nd commit
    let backend = Arc::new(FailingBackend::new(2));
    let (tx, rx) = mpsc::channel(100);
    let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
    let collector = ReceiptCollector::new(config, tx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    // Send 6 receipts → 3 batches of 2. Batch 2 will fail.
    for i in 0..6 {
        collector
            .record(
                &format!("q{}", i),
                "t1",
                None,
                0.001,
                "cpu",
                "btree",
                ts + chrono::Duration::milliseconds(i as i64),
                ts + chrono::Duration::milliseconds(i as i64 + 1),
            )
            .unwrap();
    }

    // Give committer time to process
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    drop(collector);
    handle.await.unwrap();

    let s = store.read().await;
    // With the retry queue, failed batch 2 is retried and succeeds.
    // All 6 receipts are eventually committed across 3 batches.
    assert_eq!(s.receipts.len(), 6);
    assert_eq!(s.batches.len(), 3);
}

#[tokio::test]
async fn failed_batch_not_in_store() {
    let config = LedgerConfig {
        batch_max_receipts: 1,
        batch_max_interval_secs: 300,
        ..Default::default()
    };

    // Fail every 1st commit (all fail)
    let backend = Arc::new(FailingBackend::new(1));
    let (tx, rx) = mpsc::channel(100);
    let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
    let collector = ReceiptCollector::new(config, tx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    collector
        .record("q1", "t1", None, 0.001, "cpu", "btree", ts, ts)
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    drop(collector);
    handle.await.unwrap();

    let s = store.read().await;
    assert_eq!(s.receipts.len(), 0);
    assert_eq!(s.batches.len(), 0);
}

// =========================================================================
// Sustained load tests
// =========================================================================

#[tokio::test]
async fn sustained_10k_receipts() {
    let config = LedgerConfig {
        batch_max_receipts: 100,
        batch_max_interval_secs: 60,
        ..Default::default()
    };

    let backend = Arc::new(MemoryLedgerBackend::new());
    let (tx, rx) = mpsc::channel(12_000); // large enough for all receipts
    let (handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
    let collector = ReceiptCollector::new(config, tx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    for i in 0..10_000 {
        collector
            .record(
                &format!("q{}", i),
                "t1",
                None,
                0.001,
                "cpu",
                "btree",
                ts + chrono::Duration::microseconds(i as i64),
                ts + chrono::Duration::microseconds(i as i64 + 1),
            )
            .unwrap();
    }

    let metrics = collector.metrics();
    assert_eq!(metrics.receipts_sent(), 10_000);
    assert_eq!(metrics.receipts_dropped(), 0);

    drop(collector);
    handle.await.unwrap();

    let s = store.read().await;
    assert_eq!(s.receipts.len(), 10_000);
    // 10_000 / 100 = 100 batches
    assert_eq!(s.batches.len(), 100);
    assert_eq!(backend.len().await, 100);
}

#[tokio::test]
async fn backpressure_counter_under_load() {
    let config = LedgerConfig {
        batch_max_receipts: 1000,
        batch_max_interval_secs: 300,
        ..Default::default()
    };

    // Tiny channel to force backpressure
    let (tx, rx) = mpsc::channel(2);
    let backend = Arc::new(MemoryLedgerBackend::new());
    let (_handle, _store) = BatchCommitter::spawn(config.clone(), backend, rx);
    let collector = ReceiptCollector::new(config, tx);

    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    for i in 0..100 {
        let _ = collector.record(
            &format!("q{}", i),
            "t1",
            None,
            0.001,
            "cpu",
            "btree",
            ts + chrono::Duration::microseconds(i as i64),
            ts + chrono::Duration::microseconds(i as i64 + 1),
        );
    }

    let metrics = collector.metrics();
    let total = metrics.receipts_sent() + metrics.receipts_dropped();
    assert_eq!(total, 100);
    assert!(metrics.receipts_dropped() > 0, "some should be dropped");
}

// =========================================================================
// Malformed data tests
// =========================================================================

#[tokio::test]
async fn nan_energy_accepted() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

    let receipt = collector
        .record("q1", "t1", None, f64::NAN, "cpu", "btree", ts, ts)
        .unwrap();
    assert!(receipt.energy_joules_total.is_nan());
    assert!(receipt.kwh.is_nan());
}

#[tokio::test]
async fn infinity_energy_accepted() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

    let receipt = collector
        .record("q1", "t1", None, f64::INFINITY, "cpu", "btree", ts, ts)
        .unwrap();
    assert!(receipt.energy_joules_total.is_infinite());
}

#[tokio::test]
async fn empty_strings_accepted() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

    let receipt = collector
        .record("", "", None, 0.001, "", "", ts, ts)
        .unwrap();
    assert_eq!(receipt.qid, "");
    assert_eq!(receipt.tenant_id, "");
}

#[tokio::test]
async fn large_string_fields() {
    let (tx, _rx) = mpsc::channel(100);
    let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

    let big_str = "x".repeat(1_000_000);
    let receipt = collector
        .record(
            &big_str,
            &big_str,
            Some(&big_str),
            0.001,
            &big_str,
            &big_str,
            ts,
            ts,
        )
        .unwrap();
    assert_eq!(receipt.qid.len(), 1_000_000);
}

// =========================================================================
// Merkle edge case tests
// =========================================================================

#[test]
fn merkle_1000_leaves_all_proofs_valid() {
    let leaves: Vec<[u8; 32]> = (0..1000u32).map(|i| make_leaf(&i.to_le_bytes())).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    let root = tree.root();

    for i in 0..1000 {
        let proof = tree.proof(i).unwrap();
        assert!(proof.verify(&root), "proof for leaf {} failed", i);
    }
}

#[test]
fn merkle_1023_leaves_heavy_padding() {
    // 1023 leaves → padded to 1024 (2^10)
    let leaves: Vec<[u8; 32]> = (0..1023u32).map(|i| make_leaf(&i.to_le_bytes())).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    assert_eq!(tree.leaf_count(), 1023);

    let root = tree.root();
    // Check first, middle, and last
    for &idx in &[0, 511, 1022] {
        let proof = tree.proof(idx).unwrap();
        assert!(proof.verify(&root), "proof for leaf {} failed", idx);
        // Depth should be 10 (log2(1024))
        assert_eq!(proof.siblings.len(), 10);
    }
}

#[test]
fn merkle_tampered_sibling_fails() {
    let leaves: Vec<[u8; 32]> = (0..8).map(|i| make_leaf(&[i])).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    let root = tree.root();

    let mut proof = tree.proof(3).unwrap();
    assert!(proof.verify(&root));

    // Tamper with a sibling by flipping a hex digit
    let original = proof.siblings[0].hash.clone();
    let tampered = if original.starts_with('0') {
        format!("f{}", &original[1..])
    } else {
        format!("0{}", &original[1..])
    };
    proof.siblings[0].hash = tampered;
    assert!(!proof.verify(&root), "tampered proof should fail");
}

// =========================================================================
// File backend corruption tests
// =========================================================================

#[tokio::test]
async fn file_backend_corrupted_json_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("commitments.jsonl");

    // Write a valid commitment then a corrupted line
    {
        let backend = FileLedgerBackend::new(path.clone()).unwrap();
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let batch = ReceiptBatch {
            batch_id: "good".to_string(),
            merkle_root: "good".to_string(),
            receipt_count: 1,
            time_start: ts,
            time_end: ts,
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "test".to_string(),
            receipt_ids: vec!["r1".into()],
            tenant_id: None,
        };
        backend.commit_batch(&batch).await.unwrap();
    }

    // Append a corrupted line
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"NOT VALID JSON\n")
        .unwrap();

    use std::io::Write;

    // Re-opening should fail (can't parse corrupted line)
    let result = FileLedgerBackend::new(path);
    assert!(result.is_err());
}

#[tokio::test]
async fn file_backend_empty_lines_tolerated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("commitments.jsonl");

    {
        let backend = FileLedgerBackend::new(path.clone()).unwrap();
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let batch = ReceiptBatch {
            batch_id: "b1".to_string(),
            merkle_root: "b1".to_string(),
            receipt_count: 1,
            time_start: ts,
            time_end: ts,
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "test".to_string(),
            receipt_ids: vec!["r1".into()],
            tenant_id: None,
        };
        backend.commit_batch(&batch).await.unwrap();
    }

    // Append empty lines
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    writeln!(file).unwrap();
    writeln!(file).unwrap();
    writeln!(file, "   ").unwrap(); // whitespace-only line

    // Should still load fine (empty lines skipped)
    let backend = FileLedgerBackend::new(path).unwrap();
    let all = backend.list_commitments(10).await.unwrap();
    assert_eq!(all.len(), 1);
}

#[tokio::test]
async fn file_backend_read_only_fails_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("commitments.jsonl");

    // Create file
    std::fs::write(&path, "").unwrap();

    // Make read-only
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&path, perms.clone()).unwrap();

    let backend = FileLedgerBackend::new(path.clone()).unwrap();
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    let batch = ReceiptBatch {
        batch_id: "b1".to_string(),
        merkle_root: "b1".to_string(),
        receipt_count: 1,
        time_start: ts,
        time_end: ts,
        aggregate_kwh: 0.001,
        aggregate_kg_co2e: 0.0004,
        issuer: "test".to_string(),
        receipt_ids: vec!["r1".into()],
        tenant_id: None,
    };

    let result = backend.commit_batch(&batch).await;
    assert!(result.is_err());

    // Restore permissions for cleanup
    perms.set_readonly(false);
    std::fs::set_permissions(&path, perms).unwrap();
}

// =========================================================================
// Stage energy edge cases
// =========================================================================

#[test]
fn stage_energy_all_stages_set() {
    let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
    let mut receipt = LedgerEnergyReceipt {
        receipt_id: "test".to_string(),
        qid: "q1".to_string(),
        tenant_id: "t1".to_string(),
        workload_tag: None,
        energy_joules_total: 1.0,
        energy_joules_by_stage: Default::default(),
        kwh: 0.0,
        kg_co2e: 0.0,
        grid_region: "US".to_string(),
        grid_factor_source: "test".to_string(),
        timestamp_start: ts,
        timestamp_end: ts,
        device_target: "cpu".to_string(),
        algorithm_type: "btree".to_string(),
    };

    // Set all stages
    let mut stages = HashMap::new();
    for stage in ExecutionStage::ALL {
        stages.insert(*stage, 0.125); // 1.0 / 8 stages
    }
    receipt.set_stage_energy(stages);

    let typed = receipt.typed_stage_energy();
    assert_eq!(typed.len(), 8);
    let total: f64 = typed.values().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

// =========================================================================
// Carbon conversion edge cases
// =========================================================================

#[test]
fn carbon_negative_joules() {
    // Negative energy is mathematically valid (credit?)
    let kwh = joules_to_kwh(-3_600_000.0);
    assert!((kwh - (-1.0)).abs() < 1e-10);

    let co2 = joules_to_kg_co2e(-3_600_000.0, &CarbonConfig::default());
    assert!((co2 - (-0.4)).abs() < 1e-10);
}

#[test]
fn carbon_dynamic_zero_intensity() {
    let result = joules_to_kg_co2e_dynamic(3_600_000.0, 0.0);
    assert_eq!(result, 0.0);
}
