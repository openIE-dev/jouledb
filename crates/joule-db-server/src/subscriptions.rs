//! Real-time Subscription System for JouleDB
//!
//! Provides pub/sub functionality for real-time change notifications.
//! Clients can subscribe to patterns and receive notifications when
//! matching data changes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, broadcast, mpsc};

use crate::subscription_hdc::HdcSubscriptionIndex;

/// Unique subscription ID
pub type SubscriptionId = u64;

/// Change operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeOperation {
    /// Key was inserted
    Insert,
    /// Key was updated
    Update,
    /// Key was deleted
    Delete,
}

/// Change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    /// Unique event ID
    pub id: u64,
    /// Operation type
    pub operation: ChangeOperation,
    /// Key that changed
    pub key: String,
    /// New value (None for deletes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Vec<u8>>,
    /// Old value (for updates and deletes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_value: Option<Vec<u8>>,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Table name (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
}

impl ChangeEvent {
    /// Create an insert event
    pub fn insert(key: String, value: Vec<u8>) -> Self {
        Self {
            id: 0, // Will be set by manager
            operation: ChangeOperation::Insert,
            key,
            value: Some(value),
            old_value: None,
            timestamp: current_timestamp_ms(),
            table: None,
        }
    }

    /// Create an update event
    pub fn update(key: String, old_value: Vec<u8>, new_value: Vec<u8>) -> Self {
        Self {
            id: 0,
            operation: ChangeOperation::Update,
            key,
            value: Some(new_value),
            old_value: Some(old_value),
            timestamp: current_timestamp_ms(),
            table: None,
        }
    }

    /// Create a delete event
    pub fn delete(key: String, old_value: Option<Vec<u8>>) -> Self {
        Self {
            id: 0,
            operation: ChangeOperation::Delete,
            key,
            value: None,
            old_value,
            timestamp: current_timestamp_ms(),
            table: None,
        }
    }

    /// Set table name
    pub fn with_table(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }
}

/// Subscription pattern
#[derive(Debug, Clone)]
pub struct SubscriptionPattern {
    /// Pattern string (supports * wildcards)
    pub pattern: String,
    /// Compiled regex for matching
    regex: Option<regex::Regex>,
}

impl SubscriptionPattern {
    /// Create new pattern
    pub fn new(pattern: &str) -> Self {
        // Convert glob pattern to regex
        let regex_pattern = pattern
            .replace('.', "\\.")
            .replace('*', ".*")
            .replace('?', ".");

        let regex = regex::Regex::new(&format!("^{}$", regex_pattern)).ok();

        Self {
            pattern: pattern.to_string(),
            regex,
        }
    }

    /// Check if a key matches this pattern
    pub fn matches(&self, key: &str) -> bool {
        if let Some(ref regex) = self.regex {
            regex.is_match(key)
        } else {
            // Fallback to exact match
            self.pattern == key
        }
    }
}

/// Subscription info
#[derive(Debug)]
struct Subscription {
    id: SubscriptionId,
    pattern: SubscriptionPattern,
    sender: mpsc::UnboundedSender<ChangeEvent>,
}

/// Minimum subscriptions before HDC pre-filtering kicks in
const HDC_PREFILTER_THRESHOLD: usize = 50;

/// Subscription manager
pub struct SubscriptionManager {
    /// Next subscription ID
    next_id: AtomicU64,
    /// Next event ID
    next_event_id: AtomicU64,
    /// Active subscriptions by ID
    subscriptions: RwLock<HashMap<SubscriptionId, Subscription>>,
    /// Pattern index: pattern -> subscription IDs
    pattern_index: RwLock<HashMap<String, Vec<SubscriptionId>>>,
    /// HDC-based pre-filter for fast approximate matching
    hdc_index: HdcSubscriptionIndex,
    /// Broadcast channel for all events (for logging/debugging)
    broadcast: broadcast::Sender<ChangeEvent>,
    /// Statistics
    stats: SubscriptionStats,
}

/// Subscription statistics
#[derive(Debug, Default)]
pub struct SubscriptionStats {
    /// Total subscriptions created
    pub total_subscriptions: AtomicU64,
    /// Currently active subscriptions
    pub active_subscriptions: AtomicU64,
    /// Total events published
    pub total_events: AtomicU64,
    /// Total events delivered
    pub total_deliveries: AtomicU64,
}

impl SubscriptionStats {
    pub fn snapshot(&self) -> SubscriptionStatsSnapshot {
        SubscriptionStatsSnapshot {
            total_subscriptions: self.total_subscriptions.load(Ordering::Relaxed),
            active_subscriptions: self.active_subscriptions.load(Ordering::Relaxed),
            total_events: self.total_events.load(Ordering::Relaxed),
            total_deliveries: self.total_deliveries.load(Ordering::Relaxed),
        }
    }
}

/// Stats snapshot
#[derive(Debug, Clone, Serialize)]
pub struct SubscriptionStatsSnapshot {
    pub total_subscriptions: u64,
    pub active_subscriptions: u64,
    pub total_events: u64,
    pub total_deliveries: u64,
}

impl SubscriptionManager {
    /// Create new subscription manager
    pub fn new() -> Self {
        let (broadcast, _) = broadcast::channel(1024);

        Self {
            next_id: AtomicU64::new(1),
            next_event_id: AtomicU64::new(1),
            subscriptions: RwLock::new(HashMap::new()),
            pattern_index: RwLock::new(HashMap::new()),
            hdc_index: HdcSubscriptionIndex::new(),
            broadcast,
            stats: SubscriptionStats::default(),
        }
    }

    /// Maximum total subscriptions (across all clients).
    const MAX_SUBSCRIPTIONS: usize = 100_000;

    /// Subscribe to a pattern.
    ///
    /// Returns an error if the maximum subscription count is exceeded.
    pub async fn subscribe(
        &self,
        pattern: &str,
    ) -> Result<(SubscriptionId, mpsc::UnboundedReceiver<ChangeEvent>), String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::unbounded_channel();

        let subscription = Subscription {
            id,
            pattern: SubscriptionPattern::new(pattern),
            sender,
        };

        // Add to subscriptions (with capacity check)
        {
            let mut subs = self.subscriptions.write().await;
            if subs.len() >= Self::MAX_SUBSCRIPTIONS {
                return Err(format!(
                    "Maximum subscription count ({}) exceeded",
                    Self::MAX_SUBSCRIPTIONS
                ));
            }
            subs.insert(id, subscription);
        }

        // Add to pattern index
        {
            let mut index = self.pattern_index.write().await;
            index
                .entry(pattern.to_string())
                .or_insert_with(Vec::new)
                .push(id);
        }

        // Add to HDC index for fast pre-filtering
        self.hdc_index.add_pattern(id, pattern).await;

        self.stats
            .total_subscriptions
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .active_subscriptions
            .fetch_add(1, Ordering::Relaxed);

        Ok((id, receiver))
    }

    /// Unsubscribe
    pub async fn unsubscribe(&self, id: SubscriptionId) -> bool {
        let mut subs = self.subscriptions.write().await;

        if let Some(sub) = subs.remove(&id) {
            // Remove from pattern index
            let mut index = self.pattern_index.write().await;
            if let Some(ids) = index.get_mut(&sub.pattern.pattern) {
                ids.retain(|&x| x != id);
                if ids.is_empty() {
                    index.remove(&sub.pattern.pattern);
                }
            }

            // Remove from HDC index
            self.hdc_index.remove_pattern(id).await;

            self.stats
                .active_subscriptions
                .fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Notify subscribers of a change
    pub async fn notify(&self, mut event: ChangeEvent) {
        // Assign event ID
        event.id = self.next_event_id.fetch_add(1, Ordering::Relaxed);

        self.stats.total_events.fetch_add(1, Ordering::Relaxed);

        // Broadcast to all listeners
        let _ = self.broadcast.send(event.clone());

        // Find matching subscriptions
        let subs = self.subscriptions.read().await;

        // Use HDC pre-filtering when there are enough subscriptions
        if subs.len() >= HDC_PREFILTER_THRESHOLD {
            // Fast path: HDC narrows candidates, then verify with exact regex
            let candidates: HashSet<u64> = self
                .hdc_index
                .find_candidates(&event.key)
                .await
                .into_iter()
                .collect();

            for sub in subs.values() {
                if candidates.contains(&sub.id) && sub.pattern.matches(&event.key) {
                    if sub.sender.send(event.clone()).is_ok() {
                        self.stats.total_deliveries.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        } else {
            // Small number of subscriptions: direct regex matching is fine
            for sub in subs.values() {
                if sub.pattern.matches(&event.key) {
                    if sub.sender.send(event.clone()).is_ok() {
                        self.stats.total_deliveries.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Notify insert
    pub async fn notify_insert(&self, key: &str, value: &[u8]) {
        self.notify(ChangeEvent::insert(key.to_string(), value.to_vec()))
            .await;
    }

    /// Notify update
    pub async fn notify_update(&self, key: &str, old_value: &[u8], new_value: &[u8]) {
        self.notify(ChangeEvent::update(
            key.to_string(),
            old_value.to_vec(),
            new_value.to_vec(),
        ))
        .await;
    }

    /// Notify delete
    pub async fn notify_delete(&self, key: &str, old_value: Option<&[u8]>) {
        self.notify(ChangeEvent::delete(
            key.to_string(),
            old_value.map(|v| v.to_vec()),
        ))
        .await;
    }

    /// Subscribe to broadcast (receives all events)
    pub fn subscribe_broadcast(&self) -> broadcast::Receiver<ChangeEvent> {
        self.broadcast.subscribe()
    }

    /// Get statistics
    pub fn stats(&self) -> SubscriptionStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get active subscription count
    pub async fn active_count(&self) -> usize {
        self.subscriptions.read().await.len()
    }

    /// List active patterns
    pub async fn list_patterns(&self) -> Vec<String> {
        self.pattern_index.read().await.keys().cloned().collect()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Subscription handle - RAII guard that unsubscribes on drop
pub struct SubscriptionHandle {
    id: SubscriptionId,
    manager: Arc<SubscriptionManager>,
    receiver: Option<mpsc::UnboundedReceiver<ChangeEvent>>,
}

impl SubscriptionHandle {
    /// Create new handle
    pub fn new(
        id: SubscriptionId,
        manager: Arc<SubscriptionManager>,
        receiver: mpsc::UnboundedReceiver<ChangeEvent>,
    ) -> Self {
        Self {
            id,
            manager,
            receiver: Some(receiver),
        }
    }

    /// Get subscription ID
    pub fn id(&self) -> SubscriptionId {
        self.id
    }

    /// Take the receiver (can only be called once)
    pub fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<ChangeEvent>> {
        self.receiver.take()
    }

    /// Receive next event
    pub async fn recv(&mut self) -> Option<ChangeEvent> {
        if let Some(ref mut receiver) = self.receiver {
            receiver.recv().await
        } else {
            None
        }
    }
}

impl Drop for SubscriptionHandle {
    fn drop(&mut self) {
        let id = self.id;
        let manager = self.manager.clone();

        // Spawn unsubscribe task
        tokio::spawn(async move {
            manager.unsubscribe(id).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        let pattern = SubscriptionPattern::new("users:*");
        assert!(pattern.matches("users:123"));
        assert!(pattern.matches("users:alice"));
        assert!(!pattern.matches("orders:123"));

        let pattern = SubscriptionPattern::new("*:created");
        assert!(pattern.matches("user:created"));
        assert!(pattern.matches("order:created"));
        assert!(!pattern.matches("user:updated"));

        let pattern = SubscriptionPattern::new("exact-key");
        assert!(pattern.matches("exact-key"));
        assert!(!pattern.matches("exact-key-2"));
    }

    #[tokio::test]
    async fn test_subscribe_and_notify() {
        let manager = SubscriptionManager::new();

        let (id, mut receiver) = manager.subscribe("users:*").await.unwrap();
        assert_eq!(manager.active_count().await, 1);

        // Notify matching event
        manager.notify_insert("users:123", b"test data").await;

        // Should receive event
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.key, "users:123");
        assert_eq!(event.operation, ChangeOperation::Insert);
        assert_eq!(event.value, Some(b"test data".to_vec()));

        // Notify non-matching event
        manager.notify_insert("orders:456", b"other data").await;

        // Should not receive (use try_recv to avoid blocking)
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(receiver.try_recv().is_err());

        // Unsubscribe
        assert!(manager.unsubscribe(id).await);
        assert_eq!(manager.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let manager = Arc::new(SubscriptionManager::new());

        let (_, mut rx1) = manager.subscribe("users:*").await.unwrap();
        let (_, mut rx2) = manager.subscribe("users:*").await.unwrap();
        let (_, mut rx3) = manager.subscribe("orders:*").await.unwrap();

        // Notify users event
        manager.notify_insert("users:123", b"data").await;

        // rx1 and rx2 should receive
        assert!(rx1.recv().await.is_some());
        assert!(rx2.recv().await.is_some());

        // rx3 should not receive
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(rx3.try_recv().is_err());

        // Check stats
        let stats = manager.stats();
        assert_eq!(stats.active_subscriptions, 3);
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.total_deliveries, 2);
    }

    #[tokio::test]
    async fn test_update_and_delete_events() {
        let manager = SubscriptionManager::new();
        let (_, mut receiver) = manager.subscribe("*").await.unwrap();

        // Update
        manager.notify_update("key1", b"old", b"new").await;
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.operation, ChangeOperation::Update);
        assert_eq!(event.value, Some(b"new".to_vec()));
        assert_eq!(event.old_value, Some(b"old".to_vec()));

        // Delete
        manager.notify_delete("key2", Some(b"deleted")).await;
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.operation, ChangeOperation::Delete);
        assert_eq!(event.value, None);
        assert_eq!(event.old_value, Some(b"deleted".to_vec()));
    }

    #[tokio::test]
    async fn test_broadcast() {
        let manager = SubscriptionManager::new();
        let mut broadcast_rx = manager.subscribe_broadcast();

        manager.notify_insert("any-key", b"data").await;

        let event = broadcast_rx.recv().await.unwrap();
        assert_eq!(event.key, "any-key");
    }

    #[tokio::test]
    async fn test_subscription_handle() {
        let manager = Arc::new(SubscriptionManager::new());

        {
            let (id, receiver) = manager.subscribe("test:*").await.unwrap();
            let _handle = SubscriptionHandle::new(id, manager.clone(), receiver);
            assert_eq!(manager.active_count().await, 1);
        }

        // Handle dropped, should unsubscribe
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(manager.active_count().await, 0);
    }

    #[test]
    fn test_change_event_builders() {
        let insert = ChangeEvent::insert("key".to_string(), vec![1, 2, 3]);
        assert_eq!(insert.operation, ChangeOperation::Insert);
        assert!(insert.value.is_some());
        assert!(insert.old_value.is_none());

        let update = ChangeEvent::update("key".to_string(), vec![1], vec![2]);
        assert_eq!(update.operation, ChangeOperation::Update);
        assert!(update.value.is_some());
        assert!(update.old_value.is_some());

        let delete = ChangeEvent::delete("key".to_string(), Some(vec![1]));
        assert_eq!(delete.operation, ChangeOperation::Delete);
        assert!(delete.value.is_none());
        assert!(delete.old_value.is_some());

        let with_table = ChangeEvent::insert("key".to_string(), vec![]).with_table("users");
        assert_eq!(with_table.table, Some("users".to_string()));
    }
}
