//! Transaction pool/mempool — transaction submission, priority ordering by
//! gas price, nonce ordering per account, pool size limits, eviction policy,
//! transaction validation, pending/queued separation, and pool statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from transaction pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxPoolError {
    /// Pool is full.
    PoolFull { capacity: usize, current: usize },
    /// Transaction already exists in pool.
    DuplicateTx(String),
    /// Invalid gas price (must be > 0).
    InvalidGasPrice,
    /// Invalid nonce (less than expected).
    NonceTooLow { account: String, expected: u64, got: u64 },
    /// Transaction not found.
    TxNotFound(String),
    /// Gas limit too high.
    GasLimitExceeded { limit: u64, max: u64 },
    /// Sender balance too low to cover gas.
    InsufficientFunds { sender: String },
}

impl fmt::Display for TxPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolFull { capacity, current } => {
                write!(f, "pool full: capacity {capacity}, current {current}")
            }
            Self::DuplicateTx(hash) => write!(f, "duplicate transaction: {hash}"),
            Self::InvalidGasPrice => write!(f, "gas price must be greater than zero"),
            Self::NonceTooLow { account, expected, got } => {
                write!(f, "nonce too low for {account}: expected {expected}, got {got}")
            }
            Self::TxNotFound(hash) => write!(f, "transaction not found: {hash}"),
            Self::GasLimitExceeded { limit, max } => {
                write!(f, "gas limit {limit} exceeds max {max}")
            }
            Self::InsufficientFunds { sender } => {
                write!(f, "insufficient funds for sender: {sender}")
            }
        }
    }
}

impl std::error::Error for TxPoolError {}

// ── Transaction ─────────────────────────────────────────────────────────────

/// Status of a transaction in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxStatus {
    /// Ready to be included in a block (correct nonce sequence).
    Pending,
    /// Waiting for a preceding nonce to arrive.
    Queued,
}

impl fmt::Display for TxStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Queued => write!(f, "queued"),
        }
    }
}

/// A transaction in the mempool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolTransaction {
    /// Unique transaction hash.
    pub hash: String,
    /// Sender address.
    pub sender: String,
    /// Recipient address (None for contract creation).
    pub to: Option<String>,
    /// Value transferred.
    pub value: u64,
    /// Gas price (priority factor).
    pub gas_price: u64,
    /// Gas limit for execution.
    pub gas_limit: u64,
    /// Nonce for the sender account.
    pub nonce: u64,
    /// Arbitrary data payload.
    pub data: Vec<u8>,
    /// Submission timestamp.
    pub submitted_at: u64,
    /// Current status in the pool.
    pub status: TxStatus,
}

impl PoolTransaction {
    /// Create a new transaction.
    pub fn new(
        hash: impl Into<String>,
        sender: impl Into<String>,
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
        value: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            hash: hash.into(),
            sender: sender.into(),
            to: None,
            value,
            gas_price,
            gas_limit,
            nonce,
            data: Vec::new(),
            submitted_at: timestamp,
            status: TxStatus::Queued,
        }
    }

    /// Set the recipient.
    pub fn with_to(mut self, to: impl Into<String>) -> Self {
        self.to = Some(to.into());
        self
    }

    /// Set the data payload.
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Effective priority score (higher = more likely to be included).
    pub fn priority(&self) -> u64 {
        self.gas_price
    }
}

// ── Pool Statistics ─────────────────────────────────────────────────────────

/// Statistics about the transaction pool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PoolStats {
    /// Total transactions in pool.
    pub total: usize,
    /// Pending (ready) transactions.
    pub pending: usize,
    /// Queued (waiting) transactions.
    pub queued: usize,
    /// Number of unique senders.
    pub senders: usize,
    /// Pool capacity.
    pub capacity: usize,
}

// ── Account Nonce Tracker ───────────────────────────────────────────────────

/// Tracks the next expected nonce for each account.
#[derive(Debug, Clone, Default)]
struct NonceTracker {
    /// Next expected nonce per sender.
    nonces: HashMap<String, u64>,
}

impl NonceTracker {
    fn expected_nonce(&self, sender: &str) -> u64 {
        self.nonces.get(sender).copied().unwrap_or(0)
    }

    fn set_nonce(&mut self, sender: impl Into<String>, nonce: u64) {
        self.nonces.insert(sender.into(), nonce);
    }

    fn advance(&mut self, sender: &str) {
        let current = self.expected_nonce(sender);
        self.nonces.insert(sender.to_string(), current + 1);
    }
}

// ── Transaction Pool ────────────────────────────────────────────────────────

/// A transaction pool (mempool) with priority ordering, nonce sequencing,
/// size limits, and eviction.
#[derive(Debug, Clone)]
pub struct TxPool {
    /// All transactions keyed by hash.
    transactions: HashMap<String, PoolTransaction>,
    /// Per-sender transaction lists (hash -> nonce).
    sender_txs: HashMap<String, Vec<String>>,
    /// Nonce tracking.
    nonce_tracker: NonceTracker,
    /// Maximum pool capacity.
    pub capacity: usize,
    /// Maximum gas limit per transaction.
    pub max_gas_limit: u64,
    /// Total evicted transaction count.
    pub evictions: u64,
}

impl TxPool {
    /// Create a new transaction pool.
    pub fn new(capacity: usize, max_gas_limit: u64) -> Self {
        Self {
            transactions: HashMap::new(),
            sender_txs: HashMap::new(),
            nonce_tracker: NonceTracker::default(),
            capacity,
            max_gas_limit,
            evictions: 0,
        }
    }

    /// Get the number of transactions in the pool.
    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    /// Submit a transaction to the pool.
    pub fn submit(&mut self, tx: PoolTransaction) -> Result<(), TxPoolError> {
        // Validate gas price
        if tx.gas_price == 0 {
            return Err(TxPoolError::InvalidGasPrice);
        }

        // Validate gas limit
        if tx.gas_limit > self.max_gas_limit {
            return Err(TxPoolError::GasLimitExceeded {
                limit: tx.gas_limit,
                max: self.max_gas_limit,
            });
        }

        // Check for duplicate
        if self.transactions.contains_key(&tx.hash) {
            return Err(TxPoolError::DuplicateTx(tx.hash.clone()));
        }

        // Check nonce isn't too low
        let expected = self.nonce_tracker.expected_nonce(&tx.sender);
        if tx.nonce < expected {
            return Err(TxPoolError::NonceTooLow {
                account: tx.sender.clone(),
                expected,
                got: tx.nonce,
            });
        }

        // Evict if at capacity
        if self.transactions.len() >= self.capacity {
            if !self.evict_lowest_priority() {
                return Err(TxPoolError::PoolFull {
                    capacity: self.capacity,
                    current: self.transactions.len(),
                });
            }
        }

        let hash = tx.hash.clone();
        let sender = tx.sender.clone();

        // Determine status
        let mut tx = tx;
        if tx.nonce == expected {
            tx.status = TxStatus::Pending;
        } else {
            tx.status = TxStatus::Queued;
        }

        self.transactions.insert(hash.clone(), tx);
        self.sender_txs
            .entry(sender)
            .or_default()
            .push(hash);

        // Promote queued transactions if this filled a nonce gap
        self.promote_queued();

        Ok(())
    }

    /// Remove a transaction from the pool.
    pub fn remove(&mut self, hash: &str) -> Result<PoolTransaction, TxPoolError> {
        let tx = self
            .transactions
            .remove(hash)
            .ok_or_else(|| TxPoolError::TxNotFound(hash.to_string()))?;

        if let Some(sender_hashes) = self.sender_txs.get_mut(&tx.sender) {
            sender_hashes.retain(|h| h != hash);
        }

        Ok(tx)
    }

    /// Get a transaction by hash.
    pub fn get(&self, hash: &str) -> Option<&PoolTransaction> {
        self.transactions.get(hash)
    }

    /// Mark a transaction as confirmed (remove from pool, advance nonce).
    pub fn confirm(&mut self, hash: &str) -> Result<PoolTransaction, TxPoolError> {
        let tx = self.remove(hash)?;
        self.nonce_tracker.advance(&tx.sender);
        self.promote_queued();
        Ok(tx)
    }

    /// Get all pending (ready) transactions sorted by gas price (descending).
    pub fn pending(&self) -> Vec<&PoolTransaction> {
        let mut pending: Vec<&PoolTransaction> = self
            .transactions
            .values()
            .filter(|tx| tx.status == TxStatus::Pending)
            .collect();
        pending.sort_by(|a, b| b.gas_price.cmp(&a.gas_price));
        pending
    }

    /// Get all queued transactions.
    pub fn queued(&self) -> Vec<&PoolTransaction> {
        self.transactions
            .values()
            .filter(|tx| tx.status == TxStatus::Queued)
            .collect()
    }

    /// Get transactions for a specific sender, sorted by nonce.
    pub fn sender_transactions(&self, sender: &str) -> Vec<&PoolTransaction> {
        let mut txs: Vec<&PoolTransaction> = self.sender_txs
            .get(sender)
            .map(|hashes| {
                hashes
                    .iter()
                    .filter_map(|h| self.transactions.get(h))
                    .collect()
            })
            .unwrap_or_default();
        txs.sort_by_key(|tx| tx.nonce);
        txs
    }

    /// Set the expected nonce for an account (e.g., after loading chain state).
    pub fn set_account_nonce(&mut self, sender: impl Into<String>, nonce: u64) {
        self.nonce_tracker.set_nonce(sender, nonce);
        self.promote_queued();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let pending = self
            .transactions
            .values()
            .filter(|tx| tx.status == TxStatus::Pending)
            .count();
        let queued = self
            .transactions
            .values()
            .filter(|tx| tx.status == TxStatus::Queued)
            .count();

        PoolStats {
            total: self.transactions.len(),
            pending,
            queued,
            senders: self.sender_txs.len(),
            capacity: self.capacity,
        }
    }

    /// Clear all transactions from the pool.
    pub fn clear(&mut self) {
        self.transactions.clear();
        self.sender_txs.clear();
    }

    /// Promote queued transactions to pending when their nonce is next.
    fn promote_queued(&mut self) {
        let senders: Vec<String> = self.sender_txs.keys().cloned().collect();

        for sender in senders {
            loop {
                let expected = self.nonce_tracker.expected_nonce(&sender);
                let hashes = match self.sender_txs.get(&sender) {
                    Some(h) => h.clone(),
                    None => break,
                };

                let promoted = hashes.iter().find(|h| {
                    self.transactions
                        .get(*h)
                        .map(|tx| tx.nonce == expected && tx.status == TxStatus::Queued)
                        .unwrap_or(false)
                });

                match promoted {
                    Some(hash) => {
                        let hash = hash.clone();
                        if let Some(tx) = self.transactions.get_mut(&hash) {
                            tx.status = TxStatus::Pending;
                        }
                        // Don't advance nonce here — that happens on confirm
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    /// Evict the lowest-priority transaction. Returns true if something was evicted.
    fn evict_lowest_priority(&mut self) -> bool {
        // Find the queued tx with the lowest gas price. If no queued, evict lowest pending.
        let lowest_hash = {
            let mut lowest: Option<(&str, u64)> = None;
            // Prefer evicting queued first
            for tx in self.transactions.values() {
                if tx.status == TxStatus::Queued {
                    match lowest {
                        None => lowest = Some((&tx.hash, tx.gas_price)),
                        Some((_, price)) if tx.gas_price < price => {
                            lowest = Some((&tx.hash, tx.gas_price));
                        }
                        _ => {}
                    }
                }
            }
            // If no queued, evict lowest pending
            if lowest.is_none() {
                for tx in self.transactions.values() {
                    match lowest {
                        None => lowest = Some((&tx.hash, tx.gas_price)),
                        Some((_, price)) if tx.gas_price < price => {
                            lowest = Some((&tx.hash, tx.gas_price));
                        }
                        _ => {}
                    }
                }
            }
            lowest.map(|(h, _)| h.to_string())
        };

        if let Some(hash) = lowest_hash {
            let _ = self.remove(&hash);
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Get the top N highest-priority pending transactions.
    pub fn top_pending(&self, n: usize) -> Vec<&PoolTransaction> {
        let mut pending = self.pending();
        pending.truncate(n);
        pending
    }
}

impl Default for TxPool {
    fn default() -> Self {
        Self::new(4096, 30_000_000)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tx(hash: &str, sender: &str, nonce: u64, gas_price: u64) -> PoolTransaction {
        PoolTransaction::new(hash, sender, nonce, gas_price, 21000, 0, 100)
    }

    #[test]
    fn test_submit_transaction() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_duplicate_tx_error() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        let err = pool.submit(make_tx("tx1", "alice", 1, 10)).unwrap_err();
        assert_eq!(err, TxPoolError::DuplicateTx("tx1".to_string()));
    }

    #[test]
    fn test_zero_gas_price_error() {
        let mut pool = TxPool::new(100, 1_000_000);
        let err = pool.submit(make_tx("tx1", "alice", 0, 0)).unwrap_err();
        assert_eq!(err, TxPoolError::InvalidGasPrice);
    }

    #[test]
    fn test_gas_limit_exceeded() {
        let mut pool = TxPool::new(100, 100);
        let mut tx = make_tx("tx1", "alice", 0, 10);
        tx.gas_limit = 200;
        let err = pool.submit(tx).unwrap_err();
        assert!(matches!(err, TxPoolError::GasLimitExceeded { .. }));
    }

    #[test]
    fn test_nonce_too_low() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.set_account_nonce("alice", 5);
        let err = pool.submit(make_tx("tx1", "alice", 3, 10)).unwrap_err();
        assert!(matches!(err, TxPoolError::NonceTooLow { .. }));
    }

    #[test]
    fn test_pending_status_on_correct_nonce() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        let tx = pool.get("tx1").unwrap();
        assert_eq!(tx.status, TxStatus::Pending);
    }

    #[test]
    fn test_queued_status_on_future_nonce() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 5, 10)).unwrap();
        let tx = pool.get("tx1").unwrap();
        assert_eq!(tx.status, TxStatus::Queued);
    }

    #[test]
    fn test_pending_ordered_by_gas_price() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "a", 0, 5)).unwrap();
        pool.submit(make_tx("tx2", "b", 0, 20)).unwrap();
        pool.submit(make_tx("tx3", "c", 0, 10)).unwrap();
        let pending = pool.pending();
        assert_eq!(pending[0].gas_price, 20);
        assert_eq!(pending[1].gas_price, 10);
        assert_eq!(pending[2].gas_price, 5);
    }

    #[test]
    fn test_confirm_advances_nonce() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        pool.submit(make_tx("tx2", "alice", 1, 10)).unwrap();
        pool.confirm("tx1").unwrap();
        assert_eq!(pool.len(), 1);
        let tx2 = pool.get("tx2").unwrap();
        assert_eq!(tx2.status, TxStatus::Pending);
    }

    #[test]
    fn test_remove_transaction() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        let removed = pool.remove("tx1").unwrap();
        assert_eq!(removed.hash, "tx1");
        assert!(pool.is_empty());
    }

    #[test]
    fn test_remove_not_found() {
        let mut pool = TxPool::new(100, 1_000_000);
        let err = pool.remove("missing").unwrap_err();
        assert_eq!(err, TxPoolError::TxNotFound("missing".to_string()));
    }

    #[test]
    fn test_pool_capacity_eviction() {
        let mut pool = TxPool::new(2, 1_000_000);
        pool.submit(make_tx("tx1", "a", 0, 5)).unwrap();
        pool.submit(make_tx("tx2", "b", 0, 20)).unwrap();
        // Pool is full; this should evict the lowest priority
        pool.submit(make_tx("tx3", "c", 0, 15)).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(pool.evictions >= 1);
        // tx1 (gas_price=5) should have been evicted
        assert!(pool.get("tx1").is_none());
    }

    #[test]
    fn test_sender_transactions_sorted_by_nonce() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx3", "alice", 2, 10)).unwrap();
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        pool.submit(make_tx("tx2", "alice", 1, 10)).unwrap();
        let txs = pool.sender_transactions("alice");
        assert_eq!(txs[0].nonce, 0);
        assert_eq!(txs[1].nonce, 1);
        assert_eq!(txs[2].nonce, 2);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        pool.submit(make_tx("tx2", "alice", 5, 10)).unwrap(); // queued (nonce gap)
        pool.submit(make_tx("tx3", "bob", 0, 10)).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.senders, 2);
    }

    #[test]
    fn test_clear_pool() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "alice", 0, 10)).unwrap();
        pool.submit(make_tx("tx2", "bob", 0, 10)).unwrap();
        pool.clear();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_top_pending() {
        let mut pool = TxPool::new(100, 1_000_000);
        pool.submit(make_tx("tx1", "a", 0, 5)).unwrap();
        pool.submit(make_tx("tx2", "b", 0, 20)).unwrap();
        pool.submit(make_tx("tx3", "c", 0, 10)).unwrap();
        let top = pool.top_pending(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].gas_price, 20);
        assert_eq!(top[1].gas_price, 10);
    }

    #[test]
    fn test_tx_with_recipient() {
        let tx = make_tx("tx1", "alice", 0, 10).with_to("bob");
        assert_eq!(tx.to.as_deref(), Some("bob"));
    }

    #[test]
    fn test_tx_with_data() {
        let tx = make_tx("tx1", "alice", 0, 10).with_data(vec![1, 2, 3]);
        assert_eq!(tx.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_default_pool() {
        let pool = TxPool::default();
        assert_eq!(pool.capacity, 4096);
        assert_eq!(pool.max_gas_limit, 30_000_000);
    }

    #[test]
    fn test_tx_status_display() {
        assert_eq!(format!("{}", TxStatus::Pending), "pending");
        assert_eq!(format!("{}", TxStatus::Queued), "queued");
    }

    #[test]
    fn test_pool_error_display() {
        let err = TxPoolError::PoolFull { capacity: 10, current: 10 };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
    }
}
