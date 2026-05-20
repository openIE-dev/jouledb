use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, mpsc};

/// Maximum size of the deduplication seen-set before clearing.
const MAX_SEEN_SET_SIZE: usize = 100_000;

use crate::backend::LedgerBackend;
use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::collector::LedgerConfig;
use crate::merkle::MerkleTree;
use crate::receipt::LedgerEnergyReceipt;

/// In-memory store for receipts, Merkle trees, and commitments.
///
/// Used by the verification endpoint to look up receipts and generate proofs.
pub struct ReceiptStore {
    /// receipt_id -> (receipt, batch_id, leaf_index)
    pub receipts: HashMap<String, (LedgerEnergyReceipt, String, usize)>,
    /// batch_id -> MerkleTree
    pub trees: HashMap<String, MerkleTree>,
    /// batch_id -> BatchCommitment
    pub commitments: HashMap<String, BatchCommitment>,
    /// batch_id -> ReceiptBatch
    pub batches: HashMap<String, ReceiptBatch>,
    /// Committer operational metrics (shared with callers via Arc).
    pub committer_metrics: CommitterMetrics,
}

impl ReceiptStore {
    pub fn new() -> Self {
        Self {
            receipts: HashMap::new(),
            trees: HashMap::new(),
            commitments: HashMap::new(),
            batches: HashMap::new(),
            committer_metrics: CommitterMetrics::default(),
        }
    }
}

impl Default for ReceiptStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Operational metrics for the batch committer.
#[derive(Debug, Clone)]
pub struct CommitterMetrics {
    pub commits_failed: Arc<AtomicU64>,
    pub commits_retried: Arc<AtomicU64>,
    pub retry_queue_depth: Arc<AtomicU64>,
    pub permanently_dropped: Arc<AtomicU64>,
}

impl Default for CommitterMetrics {
    fn default() -> Self {
        Self {
            commits_failed: Arc::new(AtomicU64::new(0)),
            commits_retried: Arc::new(AtomicU64::new(0)),
            retry_queue_depth: Arc::new(AtomicU64::new(0)),
            permanently_dropped: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl CommitterMetrics {
    pub fn commits_failed(&self) -> u64 {
        self.commits_failed.load(Ordering::Relaxed)
    }
    pub fn commits_retried(&self) -> u64 {
        self.commits_retried.load(Ordering::Relaxed)
    }
    pub fn retry_queue_depth(&self) -> u64 {
        self.retry_queue_depth.load(Ordering::Relaxed)
    }
    pub fn permanently_dropped(&self) -> u64 {
        self.permanently_dropped.load(Ordering::Relaxed)
    }
}

/// A failed batch commit waiting to be retried.
struct RetryEntry {
    batch: ReceiptBatch,
    receipts: Vec<LedgerEnergyReceipt>,
    tree: MerkleTree,
    attempts: u32,
    next_retry: tokio::time::Instant,
}

/// Bounded in-memory retry queue with exponential backoff.
struct RetryQueue {
    entries: VecDeque<RetryEntry>,
    max_entries: usize,
    max_attempts: u32,
    dropped_count: u64,
}

impl RetryQueue {
    fn new(max_entries: usize, max_attempts: u32) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_attempts,
            dropped_count: 0,
        }
    }

    /// Enqueue a failed batch for retry. If the queue is full, drops the oldest entry.
    fn enqueue(
        &mut self,
        batch: ReceiptBatch,
        receipts: Vec<LedgerEnergyReceipt>,
        tree: MerkleTree,
    ) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.pop_front() {
                tracing::error!(
                    batch_id = %oldest.batch.batch_id,
                    receipts = oldest.receipts.len(),
                    "Retry queue full — permanently dropping oldest batch"
                );
                self.dropped_count += 1;
            }
        }
        self.entries.push_back(RetryEntry {
            batch,
            receipts,
            tree,
            attempts: 0,
            next_retry: tokio::time::Instant::now(),
        });
    }

    /// Drain entries whose next_retry time has passed.
    fn drain_ready(&mut self) -> Vec<RetryEntry> {
        let now = tokio::time::Instant::now();
        let mut ready = Vec::new();
        let mut remaining = VecDeque::new();
        for entry in self.entries.drain(..) {
            if entry.next_retry <= now {
                ready.push(entry);
            } else {
                remaining.push_back(entry);
            }
        }
        self.entries = remaining;
        ready
    }

    /// Re-enqueue a failed retry with incremented attempt count and exponential backoff.
    /// Drops permanently if max_attempts reached.
    fn re_enqueue(&mut self, mut entry: RetryEntry) {
        entry.attempts += 1;
        if entry.attempts >= self.max_attempts {
            tracing::error!(
                batch_id = %entry.batch.batch_id,
                attempts = entry.attempts,
                receipts = entry.receipts.len(),
                "Batch permanently dropped after max retry attempts"
            );
            self.dropped_count += 1;
            return;
        }
        // Exponential backoff: 2^attempts seconds, capped at 60s
        let backoff_secs = (1u64 << entry.attempts).min(60);
        entry.next_retry =
            tokio::time::Instant::now() + std::time::Duration::from_secs(backoff_secs);
        self.entries.push_back(entry);
    }

    /// Earliest retry instant in the queue, for sleep scheduling.
    fn next_retry_instant(&self) -> Option<tokio::time::Instant> {
        self.entries.iter().map(|e| e.next_retry).min()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn dropped_count(&self) -> u64 {
        self.dropped_count
    }
}

/// Background batch committer.
///
/// Receives receipts from the `ReceiptCollector` via an mpsc channel,
/// accumulates them into batches (by count or time), builds Merkle trees,
/// and commits roots via the configured `LedgerBackend`.
pub struct BatchCommitter;

impl BatchCommitter {
    /// Spawn the background committer task.
    ///
    /// Returns the join handle and a shared reference to the receipt store
    /// (used by the verification endpoint).
    pub fn spawn<B: LedgerBackend + 'static>(
        config: LedgerConfig,
        backend: Arc<B>,
        rx: mpsc::Receiver<LedgerEnergyReceipt>,
    ) -> (tokio::task::JoinHandle<()>, Arc<RwLock<ReceiptStore>>) {
        let store = Arc::new(RwLock::new(ReceiptStore::new()));
        let store_clone = store.clone();

        let handle = tokio::spawn(async move {
            run_committer(config, backend, rx, store_clone).await;
        });

        (handle, store)
    }
}

async fn run_committer<B: LedgerBackend>(
    config: LedgerConfig,
    backend: Arc<B>,
    rx: mpsc::Receiver<LedgerEnergyReceipt>,
    store: Arc<RwLock<ReceiptStore>>,
) {
    if config.enable_tenant_isolation {
        run_committer_tenant_isolated(config, backend, rx, store).await;
    } else {
        run_committer_single_stream(config, backend, rx, store).await;
    }
}

/// Single-stream committer: all receipts go into one batch regardless of tenant.
async fn run_committer_single_stream<B: LedgerBackend>(
    config: LedgerConfig,
    backend: Arc<B>,
    mut rx: mpsc::Receiver<LedgerEnergyReceipt>,
    store: Arc<RwLock<ReceiptStore>>,
) {
    let mut pending: Vec<LedgerEnergyReceipt> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let interval = std::time::Duration::from_secs(config.batch_max_interval_secs);
    let mut deadline = tokio::time::Instant::now() + interval;
    let mut retry_queue = RetryQueue::new(config.retry_max_queue, config.retry_max_attempts);
    let metrics = {
        let s = store.read().await;
        s.committer_metrics.clone()
    };

    loop {
        // Process ready retries before accepting new work
        process_retries(&backend, &store, &mut retry_queue, &metrics).await;

        // Sleep target: min of batch deadline and next retry instant
        let sleep_target = match retry_queue.next_retry_instant() {
            Some(next) => deadline.min(next),
            None => deadline,
        };

        tokio::select! {
            maybe_receipt = rx.recv() => {
                match maybe_receipt {
                    Some(receipt) => {
                        // Deduplication check
                        if !seen.insert(receipt.receipt_id.clone()) {
                            tracing::warn!(
                                receipt_id = %receipt.receipt_id,
                                "Duplicate receipt dropped"
                            );
                            continue;
                        }
                        pending.push(receipt);
                        if pending.len() >= config.batch_max_receipts {
                            if let Some((batch, receipts, tree)) =
                                commit_batch(&config, &backend, &store, &mut pending, None).await
                            {
                                metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                                retry_queue.enqueue(batch, receipts, tree);
                            }
                            if seen.len() > MAX_SEEN_SET_SIZE {
                                seen.clear();
                            }
                            deadline = tokio::time::Instant::now() + interval;
                        }
                    }
                    None => {
                        // Channel closed — flush remaining
                        if !pending.is_empty() {
                            if let Some((batch, receipts, tree)) =
                                commit_batch(&config, &backend, &store, &mut pending, None).await
                            {
                                metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                                retry_queue.enqueue(batch, receipts, tree);
                            }
                        }
                        // Final retry drain at shutdown
                        process_retries(&backend, &store, &mut retry_queue, &metrics).await;
                        break;
                    }
                }
            }
            _ = tokio::time::sleep_until(sleep_target) => {
                if !pending.is_empty() {
                    if let Some((batch, receipts, tree)) =
                        commit_batch(&config, &backend, &store, &mut pending, None).await
                    {
                        metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                        retry_queue.enqueue(batch, receipts, tree);
                    }
                    if seen.len() > MAX_SEEN_SET_SIZE {
                        seen.clear();
                    }
                }
                deadline = tokio::time::Instant::now() + interval;
            }
        }
    }
}

/// Tenant-isolated committer: groups receipts by tenant_id and commits
/// separate batches per tenant.
async fn run_committer_tenant_isolated<B: LedgerBackend>(
    config: LedgerConfig,
    backend: Arc<B>,
    mut rx: mpsc::Receiver<LedgerEnergyReceipt>,
    store: Arc<RwLock<ReceiptStore>>,
) {
    let mut pending: HashMap<String, Vec<LedgerEnergyReceipt>> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    let interval = std::time::Duration::from_secs(config.batch_max_interval_secs);
    let mut deadline = tokio::time::Instant::now() + interval;
    let mut retry_queue = RetryQueue::new(config.retry_max_queue, config.retry_max_attempts);
    let metrics = {
        let s = store.read().await;
        s.committer_metrics.clone()
    };

    loop {
        // Process ready retries before accepting new work
        process_retries(&backend, &store, &mut retry_queue, &metrics).await;

        let sleep_target = match retry_queue.next_retry_instant() {
            Some(next) => deadline.min(next),
            None => deadline,
        };

        tokio::select! {
            maybe_receipt = rx.recv() => {
                match maybe_receipt {
                    Some(receipt) => {
                        if !seen.insert(receipt.receipt_id.clone()) {
                            tracing::warn!(
                                receipt_id = %receipt.receipt_id,
                                "Duplicate receipt dropped"
                            );
                            continue;
                        }
                        let tenant = receipt.tenant_id.clone();
                        let tenant_pending = pending.entry(tenant.clone()).or_default();
                        tenant_pending.push(receipt);
                        if tenant_pending.len() >= config.batch_max_receipts {
                            let mut batch_receipts: Vec<LedgerEnergyReceipt> =
                                pending.remove(&tenant).unwrap_or_default();
                            if let Some((batch, receipts, tree)) = commit_batch(
                                &config, &backend, &store, &mut batch_receipts, Some(&tenant),
                            )
                            .await
                            {
                                metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                                retry_queue.enqueue(batch, receipts, tree);
                            }
                            if seen.len() > MAX_SEEN_SET_SIZE {
                                seen.clear();
                            }
                            deadline = tokio::time::Instant::now() + interval;
                        }
                    }
                    None => {
                        // Channel closed — flush all tenant queues
                        for (tenant, mut receipts) in pending.drain() {
                            if !receipts.is_empty() {
                                if let Some((batch, recs, tree)) = commit_batch(
                                    &config, &backend, &store, &mut receipts, Some(&tenant),
                                )
                                .await
                                {
                                    metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                                    retry_queue.enqueue(batch, recs, tree);
                                }
                            }
                        }
                        // Final retry drain at shutdown
                        process_retries(&backend, &store, &mut retry_queue, &metrics).await;
                        break;
                    }
                }
            }
            _ = tokio::time::sleep_until(sleep_target) => {
                // Timer fired — flush all tenant queues
                let tenants: Vec<String> = pending.keys().cloned().collect();
                for tenant in tenants {
                    if let Some(mut receipts) = pending.remove(&tenant) {
                        if !receipts.is_empty() {
                            if let Some((batch, recs, tree)) = commit_batch(
                                &config, &backend, &store, &mut receipts, Some(&tenant),
                            )
                            .await
                            {
                                metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
                                retry_queue.enqueue(batch, recs, tree);
                            }
                        }
                    }
                }
                if seen.len() > MAX_SEEN_SET_SIZE {
                    seen.clear();
                }
                deadline = tokio::time::Instant::now() + interval;
            }
        }
    }
}

/// Build a batch from pending receipts and attempt to commit it.
///
/// Returns `None` on success (data stored), or `Some((batch, receipts, tree))`
/// on failure so the caller can enqueue for retry.
async fn commit_batch<B: LedgerBackend>(
    config: &LedgerConfig,
    backend: &Arc<B>,
    store: &Arc<RwLock<ReceiptStore>>,
    pending: &mut Vec<LedgerEnergyReceipt>,
    tenant_id: Option<&str>,
) -> Option<(ReceiptBatch, Vec<LedgerEnergyReceipt>, MerkleTree)> {
    let receipts: Vec<LedgerEnergyReceipt> = pending.drain(..).collect();

    // Build Merkle tree from receipt content hashes
    let leaves: Vec<[u8; 32]> = receipts.iter().map(|r| r.content_hash()).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    let root_hex = tree.root_hex();

    let time_start = receipts
        .iter()
        .map(|r| r.timestamp_start)
        .min()
        .expect("non-empty batch");
    let time_end = receipts
        .iter()
        .map(|r| r.timestamp_end)
        .max()
        .expect("non-empty batch");
    let aggregate_kwh: f64 = receipts.iter().map(|r| r.kwh).sum();
    let aggregate_kg_co2e: f64 = receipts.iter().map(|r| r.kg_co2e).sum();

    let batch = ReceiptBatch {
        batch_id: root_hex.clone(),
        merkle_root: root_hex.clone(),
        receipt_count: receipts.len(),
        time_start,
        time_end,
        aggregate_kwh,
        aggregate_kg_co2e,
        issuer: config.issuer.clone(),
        receipt_ids: receipts.iter().map(|r| r.receipt_id.clone()).collect(),
        tenant_id: tenant_id.map(String::from),
    };

    if store_commitment(backend, store, &batch, &receipts, &root_hex, tree).await {
        tracing::info!(
            batch_id = %root_hex,
            receipts = receipts.len(),
            aggregate_kwh = aggregate_kwh,
            aggregate_kg_co2e = aggregate_kg_co2e,
            "Batch committed to ledger"
        );
        None
    } else {
        // Return data for retry queue
        let tree = MerkleTree::from_leaves(&leaves);
        Some((batch, receipts, tree))
    }
}

/// Attempt to commit a batch to the backend and store results on success.
/// Returns `true` on success, `false` on failure.
async fn store_commitment<B: LedgerBackend>(
    backend: &Arc<B>,
    store: &Arc<RwLock<ReceiptStore>>,
    batch: &ReceiptBatch,
    receipts: &[LedgerEnergyReceipt],
    root_hex: &str,
    tree: MerkleTree,
) -> bool {
    match backend.commit_batch(batch).await {
        Ok(mut commitment) => {
            if let Some(overhead) = commitment.chain_overhead_joules {
                if batch.receipt_count > 0 {
                    commitment.amortized_overhead_joules =
                        Some(overhead / batch.receipt_count as f64);
                }
            }
            let mut s = store.write().await;
            for (i, receipt) in receipts.iter().enumerate() {
                s.receipts.insert(
                    receipt.receipt_id.clone(),
                    (receipt.clone(), root_hex.to_string(), i),
                );
            }
            s.trees.insert(root_hex.to_string(), tree);
            s.commitments.insert(root_hex.to_string(), commitment);
            s.batches.insert(root_hex.to_string(), batch.clone());
            true
        }
        Err(e) => {
            tracing::error!("Ledger commit failed: {}", e);
            false
        }
    }
}

/// Retry a previously failed batch commit.
/// Returns `None` on success, `Some(entry)` on failure for re-enqueue.
async fn retry_commit<B: LedgerBackend>(
    backend: &Arc<B>,
    store: &Arc<RwLock<ReceiptStore>>,
    entry: RetryEntry,
) -> Option<RetryEntry> {
    let root_hex = entry.batch.merkle_root.clone();
    let leaves: Vec<[u8; 32]> = entry.receipts.iter().map(|r| r.content_hash()).collect();
    let tree = MerkleTree::from_leaves(&leaves);

    if store_commitment(
        backend,
        store,
        &entry.batch,
        &entry.receipts,
        &root_hex,
        tree,
    )
    .await
    {
        tracing::info!(
            batch_id = %entry.batch.batch_id,
            attempt = entry.attempts + 1,
            "Retry commit succeeded"
        );
        None
    } else {
        tracing::warn!(
            batch_id = %entry.batch.batch_id,
            attempt = entry.attempts + 1,
            "Retry commit failed"
        );
        Some(entry)
    }
}

/// Process ready retries from the queue, updating metrics.
async fn process_retries<B: LedgerBackend>(
    backend: &Arc<B>,
    store: &Arc<RwLock<ReceiptStore>>,
    retry_queue: &mut RetryQueue,
    metrics: &CommitterMetrics,
) {
    let ready = retry_queue.drain_ready();
    for entry in ready {
        metrics.commits_retried.fetch_add(1, Ordering::Relaxed);
        if let Some(failed) = retry_commit(backend, store, entry).await {
            metrics.commits_failed.fetch_add(1, Ordering::Relaxed);
            retry_queue.re_enqueue(failed);
        }
    }
    metrics
        .retry_queue_depth
        .store(retry_queue.len() as u64, Ordering::Relaxed);
    metrics
        .permanently_dropped
        .store(retry_queue.dropped_count(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_memory::MemoryLedgerBackend;
    use crate::batch::BatchCommitment;
    use crate::collector::ReceiptCollector;
    use crate::error::LedgerError;
    use chrono::{TimeZone, Utc};

    /// Backend that fails the first N calls, then delegates to MemoryLedgerBackend.
    struct FailNTimesBackend {
        inner: MemoryLedgerBackend,
        remaining_failures: std::sync::atomic::AtomicU32,
    }

    impl FailNTimesBackend {
        fn new(fail_count: u32) -> Self {
            Self {
                inner: MemoryLedgerBackend::new(),
                remaining_failures: std::sync::atomic::AtomicU32::new(fail_count),
            }
        }
    }

    impl LedgerBackend for FailNTimesBackend {
        fn name(&self) -> &str {
            "fail-n-times"
        }

        async fn commit_batch(
            &self,
            batch: &crate::batch::ReceiptBatch,
        ) -> Result<BatchCommitment, LedgerError> {
            let prev =
                self.remaining_failures
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                        if n > 0 { Some(n - 1) } else { None }
                    });
            if prev.is_ok() {
                return Err(LedgerError::CommitError("simulated failure".into()));
            }
            self.inner.commit_batch(batch).await
        }

        async fn get_commitment(
            &self,
            batch_id: &str,
        ) -> Result<Option<BatchCommitment>, LedgerError> {
            self.inner.get_commitment(batch_id).await
        }

        async fn list_commitments(
            &self,
            limit: usize,
        ) -> Result<Vec<BatchCommitment>, LedgerError> {
            self.inner.list_commitments(limit).await
        }
    }

    /// Backend that always fails.
    struct AlwaysFailBackend;

    impl LedgerBackend for AlwaysFailBackend {
        fn name(&self) -> &str {
            "always-fail"
        }
        async fn commit_batch(
            &self,
            _batch: &crate::batch::ReceiptBatch,
        ) -> Result<BatchCommitment, LedgerError> {
            Err(LedgerError::CommitError("permanent failure".into()))
        }
        async fn get_commitment(
            &self,
            _batch_id: &str,
        ) -> Result<Option<BatchCommitment>, LedgerError> {
            Ok(None)
        }
        async fn list_commitments(
            &self,
            _limit: usize,
        ) -> Result<Vec<BatchCommitment>, LedgerError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn count_triggered_batch() {
        let config = LedgerConfig {
            batch_max_receipts: 3,
            batch_max_interval_secs: 300, // long interval so count triggers first
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        for i in 0..3 {
            collector
                .record(
                    &format!("q{}", i),
                    "tenant1",
                    None,
                    0.001 * (i as f64 + 1.0),
                    "cpu",
                    "btree",
                    ts + chrono::Duration::milliseconds(i as i64 * 100),
                    ts + chrono::Duration::milliseconds(i as i64 * 100 + 50),
                )
                .unwrap();
        }

        // Give the committer time to process
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 3);
        assert_eq!(s.trees.len(), 1);
        assert_eq!(s.commitments.len(), 1);

        // Verify all receipts are in the store
        for i in 0..3 {
            let receipt_id = LedgerEnergyReceipt::compute_id(
                &format!("q{}", i),
                "tenant1",
                &(ts + chrono::Duration::milliseconds(i as i64 * 100)),
            );
            assert!(s.receipts.contains_key(&receipt_id));
        }

        drop(s);
        // Verify backend has 1 committed batch
        assert_eq!(backend.len().await, 1);

        // Clean up
        drop(collector);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn time_triggered_batch() {
        let config = LedgerConfig {
            batch_max_receipts: 1000, // high count so time triggers first
            batch_max_interval_secs: 1,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (_handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "t1", None, 0.005, "cpu", "btree", ts, ts)
            .unwrap();

        // Wait for the time interval to trigger
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 1);
        assert_eq!(s.trees.len(), 1);
    }

    #[tokio::test]
    async fn shutdown_flushes_remaining() {
        let config = LedgerConfig {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 300,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "t1", None, 0.005, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record("q2", "t1", None, 0.003, "gpu", "hdc", ts, ts)
            .unwrap();

        // Drop collector to close the channel
        drop(collector);

        // Wait for committer to flush and exit
        handle.await.unwrap();

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 2);
        assert_eq!(s.trees.len(), 1);
    }

    #[tokio::test]
    async fn duplicate_receipt_dropped() {
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

        // Build the same receipt 3 times and send directly
        for _ in 0..3 {
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

        // Drop sender to trigger flush
        drop(tx);
        handle.await.unwrap();

        let s = store.read().await;
        // Only 1 receipt should be stored (duplicates dropped)
        assert_eq!(s.receipts.len(), 1);
    }

    #[tokio::test]
    async fn different_receipts_not_dropped() {
        let config = LedgerConfig {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 300,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        for i in 0..5 {
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

        drop(collector);
        handle.await.unwrap();

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 5);
    }

    #[tokio::test]
    async fn proof_lookup_after_commit() {
        let config = LedgerConfig {
            batch_max_receipts: 5,
            batch_max_interval_secs: 300,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let mut receipt_ids = Vec::new();
        for i in 0..5 {
            let r = collector
                .record(
                    &format!("q{}", i),
                    "tenant1",
                    None,
                    0.001 * (i as f64 + 1.0),
                    "cpu",
                    "btree",
                    ts + chrono::Duration::milliseconds(i as i64),
                    ts + chrono::Duration::milliseconds(i as i64 + 50),
                )
                .unwrap();
            receipt_ids.push(r.receipt_id);
        }

        // Wait for batch to commit
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let s = store.read().await;

        // Verify proof generation for each receipt
        for (i, rid) in receipt_ids.iter().enumerate() {
            let (receipt, batch_id, leaf_index) = s.receipts.get(rid).unwrap();
            assert_eq!(*leaf_index, i);

            let tree = s.trees.get(batch_id).unwrap();
            let proof = tree.proof(*leaf_index).unwrap();

            // Proof should verify against the tree root
            assert!(proof.verify(&tree.root()));

            // Content hash should match the leaf hash in the proof
            let content_hash = hex::encode(receipt.content_hash());
            assert_eq!(proof.leaf_hash, content_hash);
        }

        drop(s);
        drop(collector);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn tenant_isolation_separate_batches() {
        let config = LedgerConfig {
            batch_max_receipts: 2,
            batch_max_interval_secs: 300,
            enable_tenant_isolation: true,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend.clone(), rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        // 2 receipts for tenant_a → triggers batch
        collector
            .record("q1", "tenant_a", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record(
                "q2",
                "tenant_a",
                None,
                0.002,
                "cpu",
                "btree",
                ts + chrono::Duration::milliseconds(1),
                ts + chrono::Duration::milliseconds(2),
            )
            .unwrap();
        // 2 receipts for tenant_b → triggers batch
        collector
            .record(
                "q3",
                "tenant_b",
                None,
                0.003,
                "gpu",
                "hdc",
                ts + chrono::Duration::milliseconds(3),
                ts + chrono::Duration::milliseconds(4),
            )
            .unwrap();
        collector
            .record(
                "q4",
                "tenant_b",
                None,
                0.004,
                "gpu",
                "hdc",
                ts + chrono::Duration::milliseconds(5),
                ts + chrono::Duration::milliseconds(6),
            )
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 4);
        // Should have 2 separate batches (one per tenant)
        assert_eq!(s.batches.len(), 2);

        // Verify tenant_id is set on each batch
        for batch in s.batches.values() {
            assert!(batch.tenant_id.is_some());
            let tid = batch.tenant_id.as_ref().unwrap();
            assert!(tid == "tenant_a" || tid == "tenant_b");
            assert_eq!(batch.receipt_count, 2);
        }

        drop(s);
        drop(collector);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn tenant_isolation_timer_flushes_all() {
        let config = LedgerConfig {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 1,
            enable_tenant_isolation: true,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (_handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "tenant_a", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record(
                "q2",
                "tenant_b",
                None,
                0.002,
                "gpu",
                "hdc",
                ts + chrono::Duration::milliseconds(1),
                ts + chrono::Duration::milliseconds(2),
            )
            .unwrap();

        // Wait for timer to fire
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 2);
        // Both tenants flushed → 2 separate batches
        assert_eq!(s.batches.len(), 2);
    }

    #[tokio::test]
    async fn disabled_tenant_isolation_mixes_tenants() {
        let config = LedgerConfig {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 300,
            enable_tenant_isolation: false,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "tenant_a", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record(
                "q2",
                "tenant_b",
                None,
                0.002,
                "gpu",
                "hdc",
                ts + chrono::Duration::milliseconds(1),
                ts + chrono::Duration::milliseconds(2),
            )
            .unwrap();

        drop(collector);
        handle.await.unwrap();

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 2);
        // Single batch containing both tenants
        assert_eq!(s.batches.len(), 1);
        let batch = s.batches.values().next().unwrap();
        assert!(batch.tenant_id.is_none());
        assert_eq!(batch.receipt_count, 2);
    }

    #[tokio::test]
    async fn tenant_isolation_single_tenant_equals_single_stream() {
        let config = LedgerConfig {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 300,
            enable_tenant_isolation: true,
            ..Default::default()
        };

        let backend = Arc::new(MemoryLedgerBackend::new());
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        for i in 0..5 {
            collector
                .record(
                    &format!("q{}", i),
                    "only_tenant",
                    None,
                    0.001,
                    "cpu",
                    "btree",
                    ts + chrono::Duration::milliseconds(i as i64),
                    ts + chrono::Duration::milliseconds(i as i64 + 1),
                )
                .unwrap();
        }

        drop(collector);
        handle.await.unwrap();

        let s = store.read().await;
        assert_eq!(s.receipts.len(), 5);
        // All in one batch (same tenant)
        assert_eq!(s.batches.len(), 1);
        let batch = s.batches.values().next().unwrap();
        assert_eq!(batch.tenant_id.as_deref(), Some("only_tenant"));
        assert_eq!(batch.receipt_count, 5);
    }

    // ---- Retry queue unit tests ----

    fn sample_receipt(id: &str) -> LedgerEnergyReceipt {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        LedgerEnergyReceipt {
            receipt_id: id.to_string(),
            qid: id.to_string(),
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
        }
    }

    fn sample_batch_data(id: &str) -> (ReceiptBatch, Vec<LedgerEnergyReceipt>, MerkleTree) {
        let receipts = vec![sample_receipt(id)];
        let leaves: Vec<[u8; 32]> = receipts.iter().map(|r| r.content_hash()).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let batch = ReceiptBatch {
            batch_id: id.to_string(),
            merkle_root: tree.root_hex(),
            receipt_count: 1,
            time_start: ts,
            time_end: ts,
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "test".to_string(),
            receipt_ids: vec![id.to_string()],
            tenant_id: None,
        };
        (batch, receipts, tree)
    }

    #[tokio::test]
    async fn test_retry_queue_enqueue_and_drain() {
        let mut queue = RetryQueue::new(64, 5);
        assert_eq!(queue.len(), 0);

        let (batch, receipts, tree) = sample_batch_data("b1");
        queue.enqueue(batch, receipts, tree);
        assert_eq!(queue.len(), 1);

        // Entries with next_retry = now should be immediately drainable
        let ready = queue.drain_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(queue.len(), 0);
        assert_eq!(ready[0].batch.batch_id, "b1");
        assert_eq!(ready[0].attempts, 0);
    }

    #[tokio::test]
    async fn test_retry_backoff_timing() {
        let mut queue = RetryQueue::new(64, 5);
        let (batch, receipts, tree) = sample_batch_data("b1");
        queue.enqueue(batch, receipts, tree);

        // Drain the entry (it's ready immediately)
        let mut ready = queue.drain_ready();
        assert_eq!(ready.len(), 1);

        // Re-enqueue with backoff (attempt 1 → 2^1 = 2 second backoff)
        let entry = ready.pop().unwrap();
        queue.re_enqueue(entry);
        assert_eq!(queue.len(), 1);

        // Should NOT be ready yet (backoff is 2 seconds)
        let ready = queue.drain_ready();
        assert!(ready.is_empty());
        assert_eq!(queue.len(), 1);
    }

    #[tokio::test]
    async fn test_retry_max_attempts_drops() {
        let mut queue = RetryQueue::new(64, 3); // max 3 attempts
        let (batch, receipts, tree) = sample_batch_data("b1");
        queue.enqueue(batch, receipts, tree);

        // Simulate 3 failed attempts
        let mut entry = queue.drain_ready().pop().unwrap();
        for _ in 0..2 {
            queue.re_enqueue(entry);
            // Force immediate drain by manipulating next_retry
            if let Some(e) = queue.entries.back_mut() {
                e.next_retry = tokio::time::Instant::now();
            }
            entry = queue.drain_ready().pop().unwrap();
        }

        // Third re_enqueue should drop (attempts == max_attempts)
        queue.re_enqueue(entry);
        // Entry should be dropped, not in queue
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[tokio::test]
    async fn test_commit_failure_enters_retry_queue() {
        let config = LedgerConfig {
            batch_max_receipts: 2,
            batch_max_interval_secs: 300,
            retry_max_attempts: 5,
            retry_max_queue: 64,
            ..Default::default()
        };

        // Backend fails first call, succeeds after
        let backend = Arc::new(AlwaysFailBackend);
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "t1", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record(
                "q2",
                "t1",
                None,
                0.002,
                "cpu",
                "btree",
                ts + chrono::Duration::milliseconds(1),
                ts + chrono::Duration::milliseconds(2),
            )
            .unwrap();

        // Wait for batch attempt
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Store should be empty (commit failed)
        let s = store.read().await;
        assert_eq!(s.receipts.len(), 0);
        // Metrics should show failure
        assert!(s.committer_metrics.commits_failed() >= 1);
        drop(s);

        drop(collector);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn test_retry_succeeds_on_second_attempt() {
        let config = LedgerConfig {
            batch_max_receipts: 2,
            batch_max_interval_secs: 1, // short interval to trigger retry processing
            retry_max_attempts: 5,
            retry_max_queue: 64,
            ..Default::default()
        };

        // Fail first call, succeed after
        let backend = Arc::new(FailNTimesBackend::new(1));
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        collector
            .record("q1", "t1", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();
        collector
            .record(
                "q2",
                "t1",
                None,
                0.002,
                "cpu",
                "btree",
                ts + chrono::Duration::milliseconds(1),
                ts + chrono::Duration::milliseconds(2),
            )
            .unwrap();

        // Wait for first attempt (fails) + retry backoff (2s) + processing
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;

        let s = store.read().await;
        // Retry should have succeeded — receipts now in store
        assert_eq!(s.receipts.len(), 2);
        assert!(s.committer_metrics.commits_retried() >= 1);

        drop(s);
        drop(collector);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn test_retry_queue_bounded() {
        let mut queue = RetryQueue::new(2, 5); // max 2 entries

        let (b1, r1, t1) = sample_batch_data("b1");
        let (b2, r2, t2) = sample_batch_data("b2");
        let (b3, r3, t3) = sample_batch_data("b3");

        queue.enqueue(b1, r1, t1);
        queue.enqueue(b2, r2, t2);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dropped_count(), 0);

        // Enqueue a third — should drop the oldest (b1)
        queue.enqueue(b3, r3, t3);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dropped_count(), 1);

        // Remaining entries should be b2 and b3
        let ready = queue.drain_ready();
        let ids: Vec<&str> = ready.iter().map(|e| e.batch.batch_id.as_str()).collect();
        assert!(ids.contains(&"b2"));
        assert!(ids.contains(&"b3"));
    }

    #[tokio::test]
    async fn test_metrics_track_failures() {
        let config = LedgerConfig {
            batch_max_receipts: 1,
            batch_max_interval_secs: 300,
            retry_max_attempts: 2,
            retry_max_queue: 64,
            ..Default::default()
        };

        let backend = Arc::new(AlwaysFailBackend);
        let (tx, rx) = mpsc::channel(100);
        let (handle, store) = BatchCommitter::spawn(config.clone(), backend, rx);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        // Send 1 receipt — triggers immediate batch (batch_max_receipts=1)
        collector
            .record("q1", "t1", None, 0.001, "cpu", "btree", ts, ts)
            .unwrap();

        // Wait for initial failure + retry attempts to exhaust
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        drop(collector);
        handle.await.unwrap();

        let s = store.read().await;
        // Should have recorded at least 1 initial failure
        assert!(s.committer_metrics.commits_failed() >= 1);
        // After max_attempts (2), the batch should be permanently dropped
        assert!(s.committer_metrics.permanently_dropped() >= 1);
        // No receipts should be in the store (all commits failed)
        assert_eq!(s.receipts.len(), 0);
    }
}
