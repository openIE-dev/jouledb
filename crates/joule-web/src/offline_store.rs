//! Offline-first data store: in-memory CRUD with pending change queue,
//! flush-on-connect, optimistic updates with rollback, retry queue,
//! and change coalescing.

use serde_json::Value;
use std::collections::HashMap;

// ── Pending Change ───────────────────────────────────────────────

/// The type of pending change operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeOp {
    Create,
    Update,
    Delete,
}

/// A pending change queued while offline.
#[derive(Debug, Clone)]
pub struct PendingChange {
    pub op: ChangeOp,
    pub key: String,
    pub data: Option<Value>,
    pub timestamp: i64,
}

impl PendingChange {
    pub fn create(key: &str, data: Value, timestamp: i64) -> Self {
        Self {
            op: ChangeOp::Create,
            key: key.to_string(),
            data: Some(data),
            timestamp,
        }
    }

    pub fn update(key: &str, data: Value, timestamp: i64) -> Self {
        Self {
            op: ChangeOp::Update,
            key: key.to_string(),
            data: Some(data),
            timestamp,
        }
    }

    pub fn delete(key: &str, timestamp: i64) -> Self {
        Self {
            op: ChangeOp::Delete,
            key: key.to_string(),
            data: None,
            timestamp,
        }
    }
}

// ── Flush Result ─────────────────────────────────────────────────

/// Result of flushing a single change. The callback returns this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlushOutcome {
    /// Change accepted by server.
    Accepted,
    /// Change rejected — rollback needed.
    Rejected,
    /// Transient failure — add to retry queue.
    RetryLater,
}

// ── Offline Store ────────────────────────────────────────────────

/// An offline-first key-value store.
#[derive(Debug)]
pub struct OfflineStore {
    data: HashMap<String, Value>,
    pending_changes: Vec<PendingChange>,
    retry_queue: Vec<PendingChange>,
    /// Snapshot of data before pending changes, for rollback.
    snapshots: HashMap<String, Option<Value>>,
    online: bool,
}

impl OfflineStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            pending_changes: Vec::new(),
            retry_queue: Vec::new(),
            snapshots: HashMap::new(),
            online: false,
        }
    }

    pub fn is_online(&self) -> bool {
        self.online
    }

    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Create a new record. Queues a pending change if offline.
    pub fn create(&mut self, key: &str, data: Value, timestamp: i64) {
        // Save snapshot for rollback
        if !self.snapshots.contains_key(key) {
            self.snapshots.insert(key.to_string(), self.data.get(key).cloned());
        }
        // Optimistic update
        self.data.insert(key.to_string(), data.clone());
        self.pending_changes.push(PendingChange::create(key, data, timestamp));
    }

    /// Update an existing record.
    pub fn update(&mut self, key: &str, data: Value, timestamp: i64) {
        if !self.snapshots.contains_key(key) {
            self.snapshots.insert(key.to_string(), self.data.get(key).cloned());
        }
        self.data.insert(key.to_string(), data.clone());
        self.pending_changes.push(PendingChange::update(key, data, timestamp));
    }

    /// Delete a record.
    pub fn delete(&mut self, key: &str, timestamp: i64) {
        if !self.snapshots.contains_key(key) {
            self.snapshots.insert(key.to_string(), self.data.get(key).cloned());
        }
        self.data.remove(key);
        self.pending_changes.push(PendingChange::delete(key, timestamp));
    }

    /// Number of pending changes.
    pub fn pending_count(&self) -> usize {
        self.pending_changes.len()
    }

    /// Number of changes in the retry queue.
    pub fn retry_count(&self) -> usize {
        self.retry_queue.len()
    }

    /// Coalesce pending changes: for each key, keep only the latest change.
    pub fn coalesce(&mut self) {
        let mut latest: HashMap<String, PendingChange> = HashMap::new();
        for change in self.pending_changes.drain(..) {
            // Always replace with the latest
            latest.insert(change.key.clone(), change);
        }
        self.pending_changes = latest.into_values().collect();
        // Sort by timestamp for deterministic ordering
        self.pending_changes.sort_by_key(|c| c.timestamp);
    }

    /// Flush pending changes through a callback. The callback decides
    /// each change's fate (Accepted / Rejected / RetryLater).
    pub fn flush<F>(&mut self, mut handler: F)
    where
        F: FnMut(&PendingChange) -> FlushOutcome,
    {
        let changes = std::mem::take(&mut self.pending_changes);
        for change in changes {
            match handler(&change) {
                FlushOutcome::Accepted => {
                    // Remove snapshot, change is committed
                    self.snapshots.remove(&change.key);
                }
                FlushOutcome::Rejected => {
                    // Rollback
                    if let Some(snapshot) = self.snapshots.remove(&change.key) {
                        match snapshot {
                            Some(old_val) => {
                                self.data.insert(change.key, old_val);
                            }
                            None => {
                                self.data.remove(&change.key);
                            }
                        }
                    }
                }
                FlushOutcome::RetryLater => {
                    self.retry_queue.push(change);
                }
            }
        }
    }

    /// Retry all changes in the retry queue.
    pub fn retry<F>(&mut self, mut handler: F)
    where
        F: FnMut(&PendingChange) -> FlushOutcome,
    {
        let retries = std::mem::take(&mut self.retry_queue);
        for change in retries {
            match handler(&change) {
                FlushOutcome::Accepted => {
                    self.snapshots.remove(&change.key);
                }
                FlushOutcome::Rejected => {
                    if let Some(snapshot) = self.snapshots.remove(&change.key) {
                        match snapshot {
                            Some(old_val) => {
                                self.data.insert(change.key, old_val);
                            }
                            None => {
                                self.data.remove(&change.key);
                            }
                        }
                    }
                }
                FlushOutcome::RetryLater => {
                    self.retry_queue.push(change);
                }
            }
        }
    }

    /// Apply server-authoritative updates (e.g., from a pull).
    pub fn apply_server_updates(&mut self, updates: Vec<(String, Option<Value>)>) {
        for (key, value) in updates {
            match value {
                Some(v) => {
                    self.data.insert(key, v);
                }
                None => {
                    self.data.remove(&key);
                }
            }
        }
    }

    /// All keys in the store.
    pub fn keys(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for OfflineStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_and_get() {
        let mut store = OfflineStore::new();
        store.create("a", json!({"x": 1}), 100);
        assert_eq!(store.get("a").unwrap(), &json!({"x": 1}));
        assert_eq!(store.pending_count(), 1);
    }

    #[test]
    fn test_update() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.update("a", json!(2), 200);
        assert_eq!(store.get("a").unwrap(), &json!(2));
        assert_eq!(store.pending_count(), 2);
    }

    #[test]
    fn test_delete() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.delete("a", 200);
        assert!(store.get("a").is_none());
    }

    #[test]
    fn test_flush_accepted() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.flush(|_| FlushOutcome::Accepted);
        assert_eq!(store.pending_count(), 0);
        assert_eq!(store.get("a").unwrap(), &json!(1));
    }

    #[test]
    fn test_flush_rejected_rollback() {
        let mut store = OfflineStore::new();
        store.create("a", json!("original"), 100);
        store.flush(|_| FlushOutcome::Accepted);

        store.update("a", json!("modified"), 200);
        assert_eq!(store.get("a").unwrap(), &json!("modified"));

        store.flush(|_| FlushOutcome::Rejected);
        // Should rollback to "original"
        assert_eq!(store.get("a").unwrap(), &json!("original"));
    }

    #[test]
    fn test_flush_rejected_create_rollback() {
        let mut store = OfflineStore::new();
        store.create("new_key", json!(42), 100);
        assert!(store.get("new_key").is_some());

        store.flush(|_| FlushOutcome::Rejected);
        // Key didn't exist before, should be removed
        assert!(store.get("new_key").is_none());
    }

    #[test]
    fn test_flush_retry_later() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.flush(|_| FlushOutcome::RetryLater);
        assert_eq!(store.pending_count(), 0);
        assert_eq!(store.retry_count(), 1);
        // Data is still optimistically present
        assert_eq!(store.get("a").unwrap(), &json!(1));
    }

    #[test]
    fn test_retry_accepted() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.flush(|_| FlushOutcome::RetryLater);
        assert_eq!(store.retry_count(), 1);

        store.retry(|_| FlushOutcome::Accepted);
        assert_eq!(store.retry_count(), 0);
    }

    #[test]
    fn test_coalesce() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.update("a", json!(2), 200);
        store.update("a", json!(3), 300);
        assert_eq!(store.pending_count(), 3);

        store.coalesce();
        assert_eq!(store.pending_count(), 1);
        // The latest change should be kept
        assert_eq!(store.pending_changes[0].timestamp, 300);
    }

    #[test]
    fn test_coalesce_multiple_keys() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.create("b", json!(2), 200);
        store.update("a", json!(3), 300);
        store.coalesce();
        assert_eq!(store.pending_count(), 2);
    }

    #[test]
    fn test_apply_server_updates() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.flush(|_| FlushOutcome::Accepted);

        store.apply_server_updates(vec![
            ("a".to_string(), Some(json!(99))),
            ("b".to_string(), Some(json!(42))),
        ]);
        assert_eq!(store.get("a").unwrap(), &json!(99));
        assert_eq!(store.get("b").unwrap(), &json!(42));
    }

    #[test]
    fn test_apply_server_delete() {
        let mut store = OfflineStore::new();
        store.create("a", json!(1), 100);
        store.flush(|_| FlushOutcome::Accepted);

        store.apply_server_updates(vec![("a".to_string(), None)]);
        assert!(store.get("a").is_none());
    }

    #[test]
    fn test_len_and_empty() {
        let mut store = OfflineStore::new();
        assert!(store.is_empty());
        store.create("a", json!(1), 100);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_online_flag() {
        let mut store = OfflineStore::new();
        assert!(!store.is_online());
        store.set_online(true);
        assert!(store.is_online());
    }
}
