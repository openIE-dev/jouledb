//! Group Commit for Write Throughput Optimization
//!
//! Implements group commit to batch multiple transactions into a single
//! WAL flush, significantly improving write throughput.
//!
//! ## Design
//!
//! - Transactions wait briefly for other transactions to commit
//! - Multiple transactions are batched into a single WAL entry
//! - Single flush operation for the entire batch
//! - Reduces lock contention and I/O overhead

use crate::error::TransactionError;
use crate::persistence::{LSN, TxId, WalBackend, WalEntry};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Group commit configuration
#[derive(Debug, Clone)]
pub struct GroupCommitConfig {
    /// Maximum time to wait for other transactions (in milliseconds)
    pub max_wait_ms: u64,
    /// Maximum number of transactions per batch
    pub max_batch_size: usize,
    /// Minimum batch size before committing (0 = commit immediately)
    pub min_batch_size: usize,
    /// Enable group commit
    pub enabled: bool,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_wait_ms: 10, // 10ms wait for batching
            max_batch_size: 100,
            min_batch_size: 1,
            enabled: true,
        }
    }
}

/// Pending transaction in the commit queue
#[derive(Debug)]
struct PendingTransaction {
    /// Transaction ID
    tx_id: TxId,
    /// WAL entries for this transaction
    entries: Vec<WalEntry>,
    /// Timestamp when transaction requested commit
    commit_time: Instant,
    /// Result channel
    result_tx: Option<std::sync::mpsc::Sender<Result<LSN, TransactionError>>>,
}

/// Group commit manager
pub struct GroupCommitManager<W: WalBackend> {
    config: GroupCommitConfig,
    wal: Arc<W>,
    /// Queue of pending transactions
    queue: Arc<Mutex<VecDeque<PendingTransaction>>>,
    /// Condition variable to wake up commit thread
    condvar: Arc<Condvar>,
    /// Current batch being built
    current_batch: Arc<Mutex<Vec<PendingTransaction>>>,
    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl<W: WalBackend> GroupCommitManager<W> {
    /// Create a new group commit manager
    pub fn new(wal: Arc<W>, config: GroupCommitConfig) -> Self {
        let manager = Self {
            config,
            wal,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            condvar: Arc::new(Condvar::new()),
            current_batch: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };

        // Start commit thread
        manager.start_commit_thread();

        manager
    }

    /// Start the background commit thread
    fn start_commit_thread(&self) {
        let queue = self.queue.clone();
        let condvar = self.condvar.clone();
        let current_batch = self.current_batch.clone();
        let wal = self.wal.clone();
        let config = self.config.clone();
        let running = self.running.clone();

        std::thread::spawn(move || {
            while running.load(std::sync::atomic::Ordering::Relaxed) {
                let mut batch = Vec::new();
                let wait_start = Instant::now();

                // Collect transactions for batching
                loop {
                    let mut queue_guard = queue.lock().expect("lock poisoned: commit queue");

                    // Check if we have enough transactions or timeout
                    let elapsed = wait_start.elapsed();
                    let should_commit = queue_guard.len() >= config.min_batch_size
                        || (!batch.is_empty() && elapsed.as_millis() as u64 >= config.max_wait_ms)
                        || batch.len() >= config.max_batch_size;

                    if should_commit && !batch.is_empty() {
                        break;
                    }

                    // Wait for transactions or timeout
                    if queue_guard.is_empty() {
                        let timeout = Duration::from_millis(config.max_wait_ms);
                        let (guard, timeout_result) = condvar
                            .wait_timeout(queue_guard, timeout)
                            .expect("lock poisoned: commit queue condvar wait");
                        queue_guard = guard;

                        if timeout_result.timed_out() && !batch.is_empty() {
                            break;
                        }
                    }

                    // Take transactions from queue
                    while batch.len() < config.max_batch_size {
                        if let Some(tx) = queue_guard.pop_front() {
                            batch.push(tx);
                        } else {
                            break;
                        }
                    }

                    drop(queue_guard);

                    // If we have minimum batch size, commit
                    if batch.len() >= config.min_batch_size {
                        break;
                    }
                }

                // Commit the batch
                if !batch.is_empty() {
                    Self::commit_batch(&wal, batch);
                }
            }
        });
    }

    /// Commit a batch of transactions
    fn commit_batch(wal: &Arc<W>, batch: Vec<PendingTransaction>) {
        let mut all_entries = Vec::new();
        let mut results = Vec::new();

        // Collect all WAL entries
        for pending in &batch {
            all_entries.extend(pending.entries.clone());
            results.push((pending.tx_id, pending.result_tx.clone()));
        }

        // Write all entries in a single WAL operation
        match wal.append_entries(&all_entries) {
            Ok(lsn) => {
                // Notify all transactions of success
                for (tx_id, result_tx) in results {
                    if let Some(tx) = result_tx {
                        let _ = tx.send(Ok(lsn));
                    }
                }
            }
            Err(e) => {
                // Notify all transactions of failure
                for (tx_id, result_tx) in results {
                    if let Some(tx) = result_tx {
                        let _ = tx.send(Err(TransactionError::SerializationFailure {
                            reason: format!("WAL write failed: {}", e),
                        }));
                    }
                }
            }
        }
    }

    /// Queue a transaction for group commit
    pub fn queue_commit(
        &self,
        tx_id: TxId,
        entries: Vec<WalEntry>,
    ) -> std::sync::mpsc::Receiver<Result<LSN, TransactionError>> {
        let (tx, rx) = std::sync::mpsc::channel();

        let pending = PendingTransaction {
            tx_id,
            entries,
            commit_time: Instant::now(),
            result_tx: Some(tx),
        };

        {
            let mut queue = self.queue.lock().expect("lock poisoned: commit queue");
            queue.push_back(pending);
        }

        // Wake up commit thread
        self.condvar.notify_one();

        rx
    }

    /// Shutdown the group commit manager
    pub fn shutdown(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.condvar.notify_all();
    }
}

impl<W: WalBackend> Drop for GroupCommitManager<W> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::traits::WalBackend;
    use std::sync::Arc;

    struct MockWalBackend {
        entries: Arc<Mutex<Vec<WalEntry>>>,
    }

    impl WalBackend for MockWalBackend {
        fn append_entry(&self, entry: &WalEntry) -> Result<LSN, crate::error::Error> {
            let mut entries = self.entries.lock().unwrap();
            entries.push(entry.clone());
            Ok(entries.len() as u64)
        }

        fn append_entries(&self, entries: &[WalEntry]) -> Result<LSN, crate::error::Error> {
            let mut stored = self.entries.lock().unwrap();
            stored.extend_from_slice(entries);
            Ok(stored.len() as u64)
        }

        fn read_entries(
            &self,
            _from_lsn: LSN,
            _limit: usize,
        ) -> Result<Vec<WalEntry>, crate::error::Error> {
            Ok(self.entries.lock().unwrap().clone())
        }

        fn flush(&self) -> Result<(), crate::error::Error> {
            Ok(())
        }

        fn sync(&self) -> Result<(), crate::error::Error> {
            Ok(())
        }

        fn current_lsn(&self) -> LSN {
            self.entries.lock().unwrap().len() as u64
        }
    }

    #[test]
    fn test_group_commit_batching() {
        let wal = Arc::new(MockWalBackend {
            entries: Arc::new(Mutex::new(Vec::new())),
        });

        let config = GroupCommitConfig {
            max_wait_ms: 100,
            max_batch_size: 10,
            min_batch_size: 3,
            enabled: true,
        };

        let manager = GroupCommitManager::new(wal.clone(), config);

        // Queue multiple transactions
        let mut receivers = Vec::new();
        for i in 0..5 {
            let entries = vec![WalEntry::page_write(0, i, 0, vec![i as u8])];
            receivers.push(manager.queue_commit(i, entries));
        }

        // Wait for commits
        std::thread::sleep(Duration::from_millis(150));

        // Check that all transactions were committed
        for rx in receivers {
            assert!(rx.try_recv().is_ok());
        }

        // Check that entries were written
        let entries = wal.entries.lock().unwrap();
        assert!(entries.len() >= 5);
    }
}
