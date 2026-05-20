//! Multi-Version Concurrency Control (MVCC) with Version Chains
//!
//! Implements true MVCC where each key maintains a chain of historical versions,
//! enabling:
//! - Point-in-time reads at any timestamp
//! - Non-blocking reads (readers never block writers)
//! - Snapshot isolation guarantees
//! - Automatic garbage collection of old versions
//!
//! ## Version Chain Structure
//!
//! ```text
//! Key -> [Version@T=100] -> [Version@T=90] -> [Version@T=50] -> NULL
//!        (newest)           (older)          (oldest visible)
//! ```
//!
//! ## Visibility Rules
//!
//! A version is visible to transaction T if:
//! - version.write_ts <= T.read_ts (written before transaction started)
//! - version.write_ts is committed (not aborted)
//! - No newer version exists that is also visible

use crate::error::TransactionError;
use crate::tx::{TxId, TxState};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Global timestamp oracle
static TIMESTAMP_ORACLE: AtomicU64 = AtomicU64::new(1);

/// Get next timestamp from oracle
fn next_timestamp() -> u64 {
    TIMESTAMP_ORACLE.fetch_add(1, Ordering::SeqCst)
}

/// A single version in the version chain
#[derive(Debug, Clone)]
pub struct Version {
    /// The actual data (None = tombstone for delete)
    pub data: Option<Vec<u8>>,
    /// Timestamp when this version was created
    pub write_ts: u64,
    /// Transaction ID that created this version
    pub created_by: TxId,
    /// Whether the creating transaction has committed
    pub committed: bool,
}

impl Version {
    /// Create a new version
    pub fn new(data: Option<Vec<u8>>, write_ts: u64, created_by: TxId) -> Self {
        Self {
            data,
            write_ts,
            created_by,
            committed: false,
        }
    }

    /// Check if this version is a tombstone (deleted)
    pub fn is_tombstone(&self) -> bool {
        self.data.is_none()
    }
}

/// Version chain for a single key
#[derive(Debug, Clone)]
pub struct VersionChain {
    /// Versions stored newest-last (push to end is O(1), iterate with .rev())
    versions: Vec<Version>,
    /// Minimum timestamp that must be retained (for garbage collection)
    min_retain_ts: u64,
    /// Cached latest committed write_ts (updated on commit, recalculated after GC)
    cached_latest_committed_ts: Option<u64>,
}

impl Default for VersionChain {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionChain {
    /// Create a new empty version chain
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
            min_retain_ts: 0,
            cached_latest_committed_ts: None,
        }
    }

    /// Add a new version to the chain
    pub fn add_version(&mut self, version: Version) {
        self.versions.push(version); // O(1) amortized
    }

    /// Find the visible version for a given read timestamp
    pub fn find_visible(&self, read_ts: u64, active_txns: &HashSet<TxId>) -> Option<&Version> {
        // Iterate newest-first (from the end)
        for version in self.versions.iter().rev() {
            // Skip uncommitted versions from other transactions
            if !version.committed && active_txns.contains(&version.created_by) {
                continue;
            }

            // Version must be committed and written before our read timestamp
            if version.committed && version.write_ts <= read_ts {
                return Some(version);
            }
        }
        None
    }

    /// Find version created by a specific transaction (for read-your-writes)
    pub fn find_by_txn(&self, tx_id: TxId) -> Option<&Version> {
        self.versions.iter().rev().find(|v| v.created_by == tx_id)
    }

    /// Find version created by a specific transaction (mutable)
    pub fn find_by_txn_mut(&mut self, tx_id: TxId) -> Option<&mut Version> {
        self.versions
            .iter_mut()
            .rev()
            .find(|v| v.created_by == tx_id)
    }

    /// Mark a version as committed
    pub fn commit_version(&mut self, tx_id: TxId) {
        let mut committed_ts = None;
        if let Some(v) = self.find_by_txn_mut(tx_id) {
            v.committed = true;
            committed_ts = Some(v.write_ts);
        }
        if let Some(ts) = committed_ts {
            match self.cached_latest_committed_ts {
                Some(cached) if cached >= ts => {}
                _ => self.cached_latest_committed_ts = Some(ts),
            }
        }
    }

    /// Mark a version as committed with a specific timestamp
    pub fn commit_version_with_ts(&mut self, tx_id: TxId, commit_ts: u64) {
        if let Some(v) = self.find_by_txn_mut(tx_id) {
            v.write_ts = commit_ts;
            v.committed = true;
        }
        // Update cached ts (commit_ts comes from monotonic counter, so just compare)
        match self.cached_latest_committed_ts {
            Some(cached) if cached >= commit_ts => {}
            _ => self.cached_latest_committed_ts = Some(commit_ts),
        }
    }

    /// Remove uncommitted version from aborted transaction
    pub fn abort_version(&mut self, tx_id: TxId) {
        self.versions
            .retain(|v| v.created_by != tx_id || v.committed);
    }

    /// Garbage collect old versions
    pub fn gc(&mut self, oldest_active_ts: u64) {
        self.min_retain_ts = oldest_active_ts;

        // With newest-last ordering, old versions are at the front.
        // Find the most recent committed version older than oldest_active_ts (keeper).
        let keeper_idx: Option<usize> = self
            .versions
            .iter()
            .enumerate()
            .rev()
            .find(|(_, v)| v.committed && v.write_ts < oldest_active_ts)
            .map(|(i, _)| i);

        let mut idx = 0usize;
        self.versions.retain(|v| {
            let i = idx;
            idx += 1;

            if !v.committed {
                return true; // Keep uncommitted for now
            }
            if v.write_ts >= oldest_active_ts {
                return true; // Potentially visible
            }
            // Old committed version — only keep the designated keeper
            Some(i) == keeper_idx
        });

        // Recalculate cached ts after GC
        self.cached_latest_committed_ts = self
            .versions
            .iter()
            .filter(|v| v.committed)
            .map(|v| v.write_ts)
            .max();
    }

    /// Check if there's a write conflict with pending uncommitted version
    pub fn has_write_conflict(&self, tx_id: TxId, read_ts: u64) -> bool {
        for version in self.versions.iter().rev() {
            // Conflict if another uncommitted version exists
            if !version.committed && version.created_by != tx_id {
                return true;
            }

            // Conflict if committed version written after our read timestamp
            if version.committed && version.write_ts > read_ts {
                return true;
            }
        }
        false
    }

    /// Get the latest committed version's timestamp (O(1) cached)
    pub fn latest_committed_ts(&self) -> Option<u64> {
        self.cached_latest_committed_ts
    }

    /// Check if chain is empty (all versions garbage collected)
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Number of versions in chain
    pub fn len(&self) -> usize {
        self.versions.len()
    }
}

/// MVCC transaction state
#[derive(Debug)]
pub struct MvccTransaction {
    /// Transaction ID
    id: TxId,
    /// Read timestamp (snapshot point)
    read_ts: u64,
    /// Commit timestamp (assigned at commit)
    commit_ts: Option<u64>,
    /// Current state
    state: TxState,
    /// Keys read during transaction (for validation)
    read_set: HashMap<Vec<u8>, u64>, // key -> version_ts read
    /// Keys written during transaction
    write_set: HashMap<Vec<u8>, Option<Vec<u8>>>, // key -> value (None = delete)
    /// Reference to version store
    store: Arc<MvccStore>,
}

impl MvccTransaction {
    /// Create new MVCC transaction
    fn new(store: Arc<MvccStore>) -> Self {
        let ts = next_timestamp();
        let id = ts;
        let read_ts = ts;

        // Register as active transaction
        store.register_active(id);

        Self {
            id,
            read_ts,
            commit_ts: None,
            state: TxState::Active,
            read_set: HashMap::new(),
            write_set: HashMap::new(),
            store,
        }
    }

    /// Get transaction ID
    pub fn id(&self) -> TxId {
        self.id
    }

    /// Get read timestamp
    pub fn read_ts(&self) -> u64 {
        self.read_ts
    }

    /// Get commit timestamp
    pub fn commit_ts(&self) -> Option<u64> {
        self.commit_ts
    }

    /// Get current state
    pub fn state(&self) -> TxState {
        self.state
    }

    /// Get reference to write set
    pub fn write_set(&self) -> &HashMap<Vec<u8>, Option<Vec<u8>>> {
        &self.write_set
    }

    /// Snapshot the current write_set for savepoint
    pub fn write_set_snapshot(&self) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        self.write_set
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Restore write_set from a savepoint snapshot
    pub fn restore_write_set(&mut self, snapshot: Vec<(Vec<u8>, Option<Vec<u8>>)>) {
        self.write_set.clear();
        for (k, v) in snapshot {
            self.write_set.insert(k, v);
        }
    }

    /// Read a value
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check local write set first (read-your-writes)
        if let Some(local_value) = self.write_set.get(key) {
            return Ok(local_value.clone());
        }

        // Read from version store
        let (value, version_ts) = self.store.read(key, self.read_ts, self.id)?;

        // Track in read set for validation
        if let Some(ts) = version_ts {
            self.read_set.insert(key.to_vec(), ts);
        }

        Ok(value)
    }

    /// Write a value
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check for write-write conflicts with other uncommitted transactions
        if let Some(holder) = self.store.check_write_conflict_detailed(key, self.id)? {
            self.state = TxState::Aborted;
            return Err(TransactionError::WriteConflict {
                key: key.to_vec(),
                holder_tx_id: holder,
            });
        }

        // Write intent version to store (uncommitted)
        self.store
            .write_intent(key, Some(value.to_vec()), self.read_ts, self.id)?;

        // Add to write set for tracking
        self.write_set.insert(key.to_vec(), Some(value.to_vec()));

        Ok(())
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Check if key exists
        let existed = self.get(key)?.is_some();

        // Check for write-write conflicts with other uncommitted transactions
        if let Some(holder) = self.store.check_write_conflict_detailed(key, self.id)? {
            self.state = TxState::Aborted;
            return Err(TransactionError::WriteConflict {
                key: key.to_vec(),
                holder_tx_id: holder,
            });
        }

        // Write tombstone intent to store (uncommitted)
        self.store.write_intent(key, None, self.read_ts, self.id)?;

        // Add tombstone to write set
        self.write_set.insert(key.to_vec(), None);

        Ok(existed)
    }

    /// Scan all keys with a given prefix, returning visible (key, value) pairs.
    /// Merges store-visible versions with the local write set.
    /// Tombstones (deletes) are excluded from the result.
    pub fn scan_prefix(
        &mut self,
        prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Read committed versions from the store
        let mut result_map: HashMap<Vec<u8>, Option<Vec<u8>>> =
            self.store.scan_prefix(prefix, self.read_ts, self.id)?;

        // Overlay local write set (read-your-writes)
        for (key, value) in &self.write_set {
            if key.starts_with(prefix) {
                result_map.insert(key.clone(), value.clone());
            }
        }

        // Track reads for validation
        // (We don't track individual version_ts for prefix scans, but record
        // that we scanned this prefix range — fine for snapshot isolation)

        // Collect non-tombstone entries
        Ok(result_map
            .into_iter()
            .filter_map(|(k, v)| v.map(|data| (k, data)))
            .collect())
    }

    /// Validate transaction (check for serialization conflicts)
    fn validate(&self) -> Result<bool, TransactionError> {
        // For snapshot isolation with first-committer-wins:
        // Check if any key we're WRITING TO was modified by another
        // committed transaction after our read timestamp
        for key in self.write_set.keys() {
            if let Some(latest_ts) = self.store.get_latest_committed_ts(key)? {
                // If someone committed a newer version to a key we're writing,
                // that's a write-write conflict (first-committer-wins)
                if latest_ts > self.read_ts {
                    return Ok(false);
                }
            }
        }

        // Note: Pure read keys don't cause conflicts in snapshot isolation.
        // The transaction sees a consistent snapshot from read_ts.

        Ok(true)
    }

    /// Commit the transaction
    pub fn commit(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyCommitted { tx_id: self.id });
        }

        // Validate for serialization conflicts
        if !self.validate()? {
            self.state = TxState::Aborted;
            self.store.unregister_active(self.id);
            let keys: Vec<Vec<u8>> = self.write_set.keys().cloned().collect();
            self.store.abort_transaction_keys(self.id, &keys)?;
            return Err(TransactionError::SerializationFailure {
                reason: "write-write conflict detected".to_string(),
            });
        }

        // Get commit timestamp
        let commit_ts = next_timestamp();
        self.commit_ts = Some(commit_ts);

        // Update intent versions with commit timestamp and mark committed
        // Only visit the keys this transaction actually wrote (O(write_set) not O(total_keys))
        let keys: Vec<Vec<u8>> = self.write_set.keys().cloned().collect();
        self.store
            .commit_transaction_with_ts_keys(self.id, commit_ts, &keys)?;

        // Unregister as active
        self.store.unregister_active(self.id);

        self.state = TxState::Committed;
        Ok(())
    }

    /// Rollback the transaction
    pub fn rollback(mut self) -> Result<(), TransactionError> {
        if self.state != TxState::Active {
            return Err(TransactionError::AlreadyAborted { tx_id: self.id });
        }

        // Abort only the versions this transaction wrote (O(write_set) not O(total_keys))
        let keys: Vec<Vec<u8>> = self.write_set.keys().cloned().collect();
        self.store.abort_transaction_keys(self.id, &keys)?;

        // Unregister as active
        self.store.unregister_active(self.id);

        self.state = TxState::Aborted;
        Ok(())
    }
}

/// MVCC version store
#[derive(Debug)]
pub struct MvccStore {
    /// Version chains keyed by key
    chains: RwLock<HashMap<Vec<u8>, VersionChain>>,
    /// Active transaction IDs
    active_txns: Mutex<HashSet<TxId>>,
    /// Oldest active transaction read timestamp (for GC)
    oldest_active_ts: AtomicU64,
    /// Garbage collection threshold (number of versions before GC runs)
    gc_threshold: usize,
    /// Total version count (for triggering GC)
    version_count: AtomicU64,
}

impl MvccStore {
    /// Create a new MVCC store
    pub fn new() -> Self {
        Self {
            chains: RwLock::new(HashMap::new()),
            active_txns: Mutex::new(HashSet::new()),
            oldest_active_ts: AtomicU64::new(0),
            gc_threshold: 10000,
            version_count: AtomicU64::new(0),
        }
    }

    /// Create with custom GC threshold
    pub fn with_gc_threshold(gc_threshold: usize) -> Self {
        Self {
            chains: RwLock::new(HashMap::new()),
            active_txns: Mutex::new(HashSet::new()),
            oldest_active_ts: AtomicU64::new(0),
            gc_threshold,
            version_count: AtomicU64::new(0),
        }
    }

    /// Register an active transaction
    fn register_active(&self, tx_id: TxId) {
        let mut active = self.active_txns.lock().expect("lock poisoned: active_txns");
        active.insert(tx_id);
        // Update oldest if needed (tx_id is also read_ts in our impl)
        let current_oldest = self.oldest_active_ts.load(Ordering::Relaxed);
        if current_oldest == 0 || tx_id < current_oldest {
            self.oldest_active_ts.store(tx_id, Ordering::Relaxed);
        }
    }

    /// Unregister an active transaction
    fn unregister_active(&self, tx_id: TxId) {
        let mut active = self.active_txns.lock().expect("lock poisoned: active_txns");
        active.remove(&tx_id);

        // Update oldest active timestamp
        let new_oldest = active.iter().min().copied().unwrap_or(0);
        self.oldest_active_ts.store(new_oldest, Ordering::Relaxed);
    }

    /// Get active transaction IDs
    fn get_active_txns(&self) -> HashSet<TxId> {
        self.active_txns
            .lock()
            .expect("lock poisoned: active_txns")
            .clone()
    }

    /// Read a value at given timestamp
    fn read(
        &self,
        key: &[u8],
        read_ts: u64,
        tx_id: TxId,
    ) -> Result<(Option<Vec<u8>>, Option<u64>), TransactionError> {
        let chains = self
            .chains
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        let active = self.get_active_txns();

        if let Some(chain) = chains.get(key) {
            // First check for our own uncommitted write
            if let Some(version) = chain.find_by_txn(tx_id) {
                return Ok((version.data.clone(), Some(version.write_ts)));
            }

            // Find visible committed version
            if let Some(version) = chain.find_visible(read_ts, &active) {
                if version.is_tombstone() {
                    return Ok((None, Some(version.write_ts)));
                }
                return Ok((version.data.clone(), Some(version.write_ts)));
            }
        }

        Ok((None, None))
    }

    /// Scan all keys with a given prefix, returning visible (key, value|None) pairs.
    /// Returns a map so the caller can overlay local writes.
    fn scan_prefix(
        &self,
        prefix: &[u8],
        read_ts: u64,
        tx_id: TxId,
    ) -> Result<HashMap<Vec<u8>, Option<Vec<u8>>>, TransactionError> {
        let chains = self
            .chains
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        let active = self.get_active_txns();
        let mut results = HashMap::new();

        for (key, chain) in chains.iter() {
            if !key.starts_with(prefix) {
                continue;
            }

            // Check for our own uncommitted write first
            if let Some(version) = chain.find_by_txn(tx_id) {
                results.insert(key.clone(), version.data.clone());
                continue;
            }

            // Find visible committed version
            if let Some(version) = chain.find_visible(read_ts, &active) {
                results.insert(key.clone(), version.data.clone());
            }
        }

        Ok(results)
    }

    /// Write a new version
    fn write(
        &self,
        key: &[u8],
        value: Option<Vec<u8>>,
        write_ts: u64,
        tx_id: TxId,
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        let chain = chains.entry(key.to_vec()).or_insert_with(VersionChain::new);
        let version = Version::new(value, write_ts, tx_id);
        chain.add_version(version);

        // Track version count for GC
        let count = self.version_count.fetch_add(1, Ordering::Relaxed);
        if count > self.gc_threshold as u64 {
            drop(chains);
            self.run_gc();
        }

        Ok(())
    }

    /// Check for write-write conflict (returns holder tx_id if conflict)
    fn check_write_conflict_detailed(
        &self,
        key: &[u8],
        tx_id: TxId,
    ) -> Result<Option<TxId>, TransactionError> {
        let chains = self
            .chains
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        if let Some(chain) = chains.get(key) {
            // Look for uncommitted version from another transaction
            for version in chain.versions.iter().rev() {
                if !version.committed && version.created_by != tx_id {
                    return Ok(Some(version.created_by));
                }
            }
        }

        Ok(None)
    }

    /// Write an intent (uncommitted version) to the store
    fn write_intent(
        &self,
        key: &[u8],
        value: Option<Vec<u8>>,
        write_ts: u64,
        tx_id: TxId,
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        let chain = chains.entry(key.to_vec()).or_insert_with(VersionChain::new);

        // Check if we already have an intent from this transaction
        if let Some(existing) = chain.find_by_txn_mut(tx_id) {
            // Update existing intent
            existing.data = value;
            existing.write_ts = write_ts;
            return Ok(());
        }

        // Create new intent version (uncommitted)
        let version = Version::new(value, write_ts, tx_id);
        chain.add_version(version);

        // Track version count for GC
        let count = self.version_count.fetch_add(1, Ordering::Relaxed);
        if count > self.gc_threshold as u64 {
            drop(chains);
            self.run_gc();
        }

        Ok(())
    }

    /// Get latest committed timestamp for a key
    fn get_latest_committed_ts(&self, key: &[u8]) -> Result<Option<u64>, TransactionError> {
        let chains = self
            .chains
            .read()
            .map_err(|_| TransactionError::SerializationFailure {
                reason: "lock poisoned".to_string(),
            })?;

        Ok(chains.get(key).and_then(|c| c.latest_committed_ts()))
    }

    /// Commit all versions for a transaction
    fn commit_transaction(&self, tx_id: TxId) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        for chain in chains.values_mut() {
            chain.commit_version(tx_id);
        }

        Ok(())
    }

    /// Commit all versions for a transaction with a specific commit timestamp
    fn commit_transaction_with_ts(
        &self,
        tx_id: TxId,
        commit_ts: u64,
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        for chain in chains.values_mut() {
            chain.commit_version_with_ts(tx_id, commit_ts);
        }

        Ok(())
    }

    /// Abort all versions for a transaction
    fn abort_transaction(&self, tx_id: TxId) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        for chain in chains.values_mut() {
            chain.abort_version(tx_id);
        }

        Ok(())
    }

    /// Commit versions for a transaction, targeting only the given keys.
    /// O(write_set_size) instead of O(total_keys).
    fn commit_transaction_with_ts_keys(
        &self,
        tx_id: TxId,
        commit_ts: u64,
        keys: &[Vec<u8>],
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        for key in keys {
            if let Some(chain) = chains.get_mut(key.as_slice()) {
                chain.commit_version_with_ts(tx_id, commit_ts);
            }
        }

        Ok(())
    }

    /// Abort versions for a transaction, targeting only the given keys.
    /// O(write_set_size) instead of O(total_keys).
    fn abort_transaction_keys(
        &self,
        tx_id: TxId,
        keys: &[Vec<u8>],
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        for key in keys {
            if let Some(chain) = chains.get_mut(key.as_slice()) {
                chain.abort_version(tx_id);
            }
        }

        Ok(())
    }

    /// Run garbage collection
    fn run_gc(&self) {
        let oldest_ts = self.oldest_active_ts.load(Ordering::Relaxed);
        if oldest_ts == 0 {
            return; // No active transactions
        }

        if let Ok(mut chains) = self.chains.write() {
            let mut total_versions = 0u64;
            let mut empty_keys = Vec::new();

            for (key, chain) in chains.iter_mut() {
                chain.gc(oldest_ts);
                if chain.is_empty() {
                    empty_keys.push(key.clone());
                }
                total_versions += chain.len() as u64;
            }

            // Remove empty chains
            for key in empty_keys {
                chains.remove(&key);
            }

            self.version_count.store(total_versions, Ordering::Relaxed);
        }
    }

    /// Get a value outside of a transaction (for testing/debugging)
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let chains = self.chains.read().ok()?;
        let active = self.get_active_txns();
        let current_ts = next_timestamp();

        chains.get(key).and_then(|chain| {
            chain
                .find_visible(current_ts, &active)
                .and_then(|v| v.data.clone())
        })
    }

    /// Get statistics
    pub fn stats(&self) -> MvccStats {
        let chains = self.chains.read().expect("lock poisoned: chains read");
        let active = self.active_txns.lock().expect("lock poisoned: active_txns");

        MvccStats {
            key_count: chains.len(),
            version_count: chains.values().map(|c| c.len()).sum(),
            active_transaction_count: active.len(),
            oldest_active_ts: self.oldest_active_ts.load(Ordering::Relaxed),
        }
    }
}

impl Default for MvccStore {
    fn default() -> Self {
        Self::new()
    }
}

/// MVCC statistics
#[derive(Debug, Clone)]
pub struct MvccStats {
    /// Number of keys with version chains
    pub key_count: usize,
    /// Total number of versions across all chains
    pub version_count: usize,
    /// Number of active transactions
    pub active_transaction_count: usize,
    /// Oldest active transaction timestamp
    pub oldest_active_ts: u64,
}

/// MVCC transaction manager
#[derive(Clone)]
pub struct MvccTransactionManager {
    store: Arc<MvccStore>,
}

impl MvccTransactionManager {
    /// Create new transaction manager
    pub fn new() -> Self {
        Self {
            store: Arc::new(MvccStore::new()),
        }
    }

    /// Create with custom GC threshold
    pub fn with_gc_threshold(gc_threshold: usize) -> Self {
        Self {
            store: Arc::new(MvccStore::with_gc_threshold(gc_threshold)),
        }
    }

    /// Begin a new transaction
    pub fn begin(&self) -> MvccTransaction {
        MvccTransaction::new(self.store.clone())
    }

    /// Get store statistics
    pub fn stats(&self) -> MvccStats {
        self.store.stats()
    }

    /// Force garbage collection
    pub fn gc(&self) {
        self.store.run_gc();
    }

    /// Get a value outside transaction (for testing)
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.store.get(key)
    }
}

impl Default for MvccTransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Bootstrap and store-access methods.
impl MvccStore {
    /// Insert a pre-committed version directly into the store (for bootstrapping).
    /// This bypasses the transaction lifecycle — use only for seeding initial state.
    pub fn insert_committed(
        &self,
        key: &[u8],
        value: Vec<u8>,
        commit_ts: u64,
    ) -> Result<(), TransactionError> {
        let mut chains =
            self.chains
                .write()
                .map_err(|_| TransactionError::SerializationFailure {
                    reason: "lock poisoned".to_string(),
                })?;

        let chain = chains.entry(key.to_vec()).or_insert_with(VersionChain::new);
        let mut version = Version::new(Some(value), commit_ts, 0); // tx_id 0 = bootstrap
        version.committed = true;
        chain.add_version(version);
        // Update cached ts
        match chain.cached_latest_committed_ts {
            Some(cached) if cached >= commit_ts => {}
            _ => chain.cached_latest_committed_ts = Some(commit_ts),
        }

        Ok(())
    }
}

impl MvccTransactionManager {
    /// Get a reference to the underlying store (for bootstrap operations)
    pub fn store(&self) -> &Arc<MvccStore> {
        &self.store
    }

    /// Time-travel scan: read all keys with a given prefix visible at `ts`.
    /// Returns committed versions only (no transaction context).
    pub fn scan_prefix_at(
        &self,
        prefix: &[u8],
        ts: u64,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, TransactionError> {
        // Use tx_id 0 (no active transaction) so only committed versions are visible
        let result_map = self.store.scan_prefix(prefix, ts, 0)?;
        Ok(result_map
            .into_iter()
            .filter_map(|(k, v)| v.map(|data| (k, data)))
            .collect())
    }
}

// --- Key encoding helpers for table/record mapping ---

/// Encode a table name + record ID into an MVCC key.
/// Format: `<table_name>\0<record_id as big-endian u64>`
pub fn encode_record_key(table: &str, record_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + 1 + 8);
    key.extend_from_slice(table.as_bytes());
    key.push(0); // null separator
    key.extend_from_slice(&record_id.to_be_bytes());
    key
}

/// Encode a table prefix for prefix scanning.
/// Format: `<table_name>\0`
pub fn encode_table_prefix(table: &str) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(table.len() + 1);
    prefix.extend_from_slice(table.as_bytes());
    prefix.push(0); // null separator
    prefix
}

/// Decode the record ID from an MVCC key, given the table name.
/// Returns `None` if the key doesn't match the expected format.
pub fn decode_record_id(key: &[u8], table: &str) -> Option<u64> {
    let prefix = encode_table_prefix(table);
    if key.len() != prefix.len() + 8 || !key.starts_with(&prefix) {
        return None;
    }
    let id_bytes: [u8; 8] = key[prefix.len()..].try_into().ok()?;
    Some(u64::from_be_bytes(id_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvcc_basic_read_write() {
        let manager = MvccTransactionManager::new();

        // Transaction 1: Write
        let mut tx1 = manager.begin();
        tx1.put(b"key1", b"value1").unwrap();
        tx1.put(b"key2", b"value2").unwrap();
        tx1.commit().unwrap();

        // Transaction 2: Read
        let mut tx2 = manager.begin();
        assert_eq!(tx2.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(tx2.get(b"key2").unwrap(), Some(b"value2".to_vec()));
        tx2.commit().unwrap();
    }

    #[test]
    fn test_mvcc_read_your_writes() {
        let manager = MvccTransactionManager::new();

        let mut tx = manager.begin();

        // Write then read in same transaction
        tx.put(b"key", b"value").unwrap();
        assert_eq!(tx.get(b"key").unwrap(), Some(b"value".to_vec()));

        // Overwrite and read again
        tx.put(b"key", b"updated").unwrap();
        assert_eq!(tx.get(b"key").unwrap(), Some(b"updated".to_vec()));

        tx.commit().unwrap();
    }

    #[test]
    fn test_mvcc_snapshot_isolation() {
        let manager = MvccTransactionManager::new();

        // Transaction 1: Write initial value
        let mut tx1 = manager.begin();
        tx1.put(b"key", b"v1").unwrap();
        tx1.commit().unwrap();

        // Transaction 2: Start and read
        let mut tx2 = manager.begin();
        assert_eq!(tx2.get(b"key").unwrap(), Some(b"v1".to_vec()));

        // Transaction 3: Update the key
        let mut tx3 = manager.begin();
        tx3.put(b"key", b"v2").unwrap();
        tx3.commit().unwrap();

        // Transaction 2: Should still see v1 (snapshot isolation)
        assert_eq!(tx2.get(b"key").unwrap(), Some(b"v1".to_vec()));
        tx2.commit().unwrap();

        // New transaction: Should see v2
        let mut tx4 = manager.begin();
        assert_eq!(tx4.get(b"key").unwrap(), Some(b"v2".to_vec()));
        tx4.commit().unwrap();
    }

    #[test]
    fn test_mvcc_write_conflict() {
        let manager = MvccTransactionManager::new();

        // Transaction 1: Start
        let mut tx1 = manager.begin();
        tx1.put(b"key", b"v1").unwrap();

        // Transaction 2: Try to write same key (conflict)
        let mut tx2 = manager.begin();
        let result = tx2.put(b"key", b"v2");
        assert!(result.is_err());

        // Transaction 1 can still commit
        tx1.commit().unwrap();
    }

    #[test]
    fn test_mvcc_delete() {
        let manager = MvccTransactionManager::new();

        // Write
        let mut tx1 = manager.begin();
        tx1.put(b"key", b"value").unwrap();
        tx1.commit().unwrap();

        // Verify exists
        let mut tx2 = manager.begin();
        assert_eq!(tx2.get(b"key").unwrap(), Some(b"value".to_vec()));
        tx2.commit().unwrap();

        // Delete
        let mut tx3 = manager.begin();
        assert!(tx3.delete(b"key").unwrap()); // Returns true (existed)
        tx3.commit().unwrap();

        // Verify deleted
        let mut tx4 = manager.begin();
        assert_eq!(tx4.get(b"key").unwrap(), None);
        tx4.commit().unwrap();
    }

    #[test]
    fn test_mvcc_rollback() {
        let manager = MvccTransactionManager::new();

        // Write
        let mut tx1 = manager.begin();
        tx1.put(b"key", b"value").unwrap();
        tx1.commit().unwrap();

        // Start transaction, write, then rollback
        let mut tx2 = manager.begin();
        tx2.put(b"key", b"updated").unwrap();
        tx2.rollback().unwrap();

        // Value should be unchanged
        assert_eq!(manager.get(b"key"), Some(b"value".to_vec()));
    }

    #[test]
    fn test_mvcc_gc() {
        let manager = MvccTransactionManager::with_gc_threshold(5);

        // Create many versions
        for i in 0..20 {
            let mut tx = manager.begin();
            tx.put(b"key", format!("v{}", i).as_bytes()).unwrap();
            tx.commit().unwrap();
        }

        // Force GC
        manager.gc();

        // Value should still be readable
        assert!(manager.get(b"key").is_some());

        // Stats should show reduced version count
        let stats = manager.stats();
        assert!(stats.version_count < 20);
    }

    #[test]
    fn test_mvcc_stats() {
        let manager = MvccTransactionManager::new();

        let stats = manager.stats();
        assert_eq!(stats.key_count, 0);
        assert_eq!(stats.version_count, 0);
        assert_eq!(stats.active_transaction_count, 0);

        // Start a transaction
        let tx = manager.begin();
        let stats = manager.stats();
        assert_eq!(stats.active_transaction_count, 1);

        drop(tx); // Will not commit, but transaction goes out of scope
    }

    #[test]
    fn test_version_chain() {
        let mut chain = VersionChain::new();

        // Add versions
        let mut v1 = Version::new(Some(b"v1".to_vec()), 10, 1);
        v1.committed = true;
        chain.add_version(v1);

        let mut v2 = Version::new(Some(b"v2".to_vec()), 20, 2);
        v2.committed = true;
        chain.add_version(v2);

        let mut v3 = Version::new(Some(b"v3".to_vec()), 30, 3);
        v3.committed = true;
        chain.add_version(v3);

        assert_eq!(chain.len(), 3);

        // Find visible at different timestamps
        let active = HashSet::new();
        assert_eq!(
            chain.find_visible(15, &active).unwrap().data,
            Some(b"v1".to_vec())
        );
        assert_eq!(
            chain.find_visible(25, &active).unwrap().data,
            Some(b"v2".to_vec())
        );
        assert_eq!(
            chain.find_visible(35, &active).unwrap().data,
            Some(b"v3".to_vec())
        );

        // GC old versions
        chain.gc(25);
        assert_eq!(chain.len(), 2); // v1 should be GC'd, but one old kept
    }

    #[test]
    fn test_scan_prefix_basic() {
        let manager = MvccTransactionManager::new();

        // Insert records under two tables
        let mut tx = manager.begin();
        tx.put(&encode_record_key("users", 1), b"alice").unwrap();
        tx.put(&encode_record_key("users", 2), b"bob").unwrap();
        tx.put(&encode_record_key("orders", 1), b"order1").unwrap();
        tx.commit().unwrap();

        // Scan only users
        let mut tx2 = manager.begin();
        let users = tx2.scan_prefix(&encode_table_prefix("users")).unwrap();
        assert_eq!(users.len(), 2);
        // Verify both user records are present
        let values: Vec<&[u8]> = users.iter().map(|(_, v)| v.as_slice()).collect();
        assert!(values.contains(&b"alice".as_slice()));
        assert!(values.contains(&b"bob".as_slice()));

        // Scan orders — should be 1
        let orders = tx2.scan_prefix(&encode_table_prefix("orders")).unwrap();
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].1, b"order1");
        tx2.commit().unwrap();
    }

    #[test]
    fn test_scan_prefix_snapshot_visibility() {
        let manager = MvccTransactionManager::new();

        // Commit initial data
        let mut tx1 = manager.begin();
        tx1.put(&encode_record_key("t", 1), b"v1").unwrap();
        tx1.put(&encode_record_key("t", 2), b"v2").unwrap();
        tx1.commit().unwrap();

        // Start tx2 (takes snapshot)
        let mut tx2 = manager.begin();

        // tx3 adds a new record AFTER tx2's snapshot
        let mut tx3 = manager.begin();
        tx3.put(&encode_record_key("t", 3), b"v3").unwrap();
        tx3.commit().unwrap();

        // tx2 should only see records 1 and 2 (snapshot isolation)
        let results = tx2.scan_prefix(&encode_table_prefix("t")).unwrap();
        assert_eq!(results.len(), 2);
        tx2.commit().unwrap();

        // New tx should see all 3
        let mut tx4 = manager.begin();
        let results = tx4.scan_prefix(&encode_table_prefix("t")).unwrap();
        assert_eq!(results.len(), 3);
        tx4.commit().unwrap();
    }

    #[test]
    fn test_scan_prefix_read_your_writes() {
        let manager = MvccTransactionManager::new();

        // Commit one record
        let mut tx1 = manager.begin();
        tx1.put(&encode_record_key("t", 1), b"committed").unwrap();
        tx1.commit().unwrap();

        // Start tx2, add a local write, scan
        let mut tx2 = manager.begin();
        tx2.put(&encode_record_key("t", 2), b"local").unwrap();
        let results = tx2.scan_prefix(&encode_table_prefix("t")).unwrap();
        // Should see both: committed record + local write
        assert_eq!(results.len(), 2);
        let values: Vec<&[u8]> = results.iter().map(|(_, v)| v.as_slice()).collect();
        assert!(values.contains(&b"committed".as_slice()));
        assert!(values.contains(&b"local".as_slice()));
        tx2.rollback().unwrap();
    }

    #[test]
    fn test_scan_prefix_tombstones_excluded() {
        let manager = MvccTransactionManager::new();

        // Commit two records
        let mut tx1 = manager.begin();
        tx1.put(&encode_record_key("t", 1), b"keep").unwrap();
        tx1.put(&encode_record_key("t", 2), b"delete_me").unwrap();
        tx1.commit().unwrap();

        // Delete one
        let mut tx2 = manager.begin();
        tx2.delete(&encode_record_key("t", 2)).unwrap();
        // Scan should exclude the tombstone
        let results = tx2.scan_prefix(&encode_table_prefix("t")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, b"keep");
        tx2.commit().unwrap();

        // Subsequent scan should also exclude
        let mut tx3 = manager.begin();
        let results = tx3.scan_prefix(&encode_table_prefix("t")).unwrap();
        assert_eq!(results.len(), 1);
        tx3.commit().unwrap();
    }

    #[test]
    fn test_key_encode_decode_roundtrip() {
        let key = encode_record_key("users", 42);
        assert_eq!(decode_record_id(&key, "users"), Some(42));

        let key2 = encode_record_key("orders", u64::MAX);
        assert_eq!(decode_record_id(&key2, "orders"), Some(u64::MAX));

        // Wrong table name returns None
        assert_eq!(decode_record_id(&key, "orders"), None);

        // Truncated key returns None
        assert_eq!(decode_record_id(&key[..key.len() - 1], "users"), None);
    }

    #[test]
    fn test_table_prefix_matching() {
        let prefix = encode_table_prefix("users");
        let key1 = encode_record_key("users", 1);
        let key2 = encode_record_key("users_archive", 1);
        let key3 = encode_record_key("orders", 1);

        assert!(key1.starts_with(&prefix));
        // "users_archive" should NOT match "users\0" prefix
        assert!(!key2.starts_with(&prefix));
        assert!(!key3.starts_with(&prefix));
    }

    #[test]
    fn test_insert_committed_bootstrap() {
        let manager = MvccTransactionManager::new();
        let store = manager.store();

        // Bootstrap some committed data
        store
            .insert_committed(&encode_record_key("t", 1), b"bootstrapped".to_vec(), 0)
            .unwrap();

        // Should be visible to a new transaction
        let mut tx = manager.begin();
        let val = tx.get(&encode_record_key("t", 1)).unwrap();
        assert_eq!(val, Some(b"bootstrapped".to_vec()));

        // Should appear in prefix scan
        let results = tx.scan_prefix(&encode_table_prefix("t")).unwrap();
        assert_eq!(results.len(), 1);
        tx.commit().unwrap();
    }
}
