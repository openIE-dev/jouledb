//! Change Streams API (MongoDB-style)
//!
//! Provides resumable change streams for real-time data change notifications.
//! Supports filtering, resumable tokens, and event replay.

use crate::subscriptions::{ChangeEvent, ChangeOperation, SubscriptionId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, broadcast};

/// Change stream token for resumable streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeStreamToken {
    /// WAL offset
    pub wal_offset: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Checksum for validation
    pub checksum: u32,
}

/// Change stream filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeStreamFilter {
    /// Operation types to include
    pub operations: Vec<ChangeOperation>,
    /// Key pattern filter (supports wildcards)
    pub key_pattern: Option<String>,
    /// Table name filter
    pub table: Option<String>,
}

impl Default for ChangeStreamFilter {
    fn default() -> Self {
        Self {
            operations: vec![
                ChangeOperation::Insert,
                ChangeOperation::Update,
                ChangeOperation::Delete,
            ],
            key_pattern: None,
            table: None,
        }
    }
}

/// Change stream options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeStreamOptions {
    /// Filter
    pub filter: ChangeStreamFilter,
    /// Batch size
    pub batch_size: usize,
    /// Maximum await time (ms)
    pub max_await_time_ms: u64,
    /// Start from token (for resuming)
    pub resume_token: Option<ChangeStreamToken>,
}

impl Default for ChangeStreamOptions {
    fn default() -> Self {
        Self {
            filter: ChangeStreamFilter::default(),
            batch_size: 100,
            max_await_time_ms: 1000,
            resume_token: None,
        }
    }
}

/// Change stream
pub struct ChangeStream {
    /// Stream ID
    id: SubscriptionId,
    /// Options
    options: ChangeStreamOptions,
    /// Current token
    current_token: Arc<RwLock<ChangeStreamToken>>,
    /// Event receiver
    receiver: broadcast::Receiver<ChangeEvent>,
    /// Event buffer
    buffer: Arc<RwLock<Vec<ChangeEvent>>>,
    /// Next event ID
    next_event_id: Arc<AtomicU64>,
}

impl ChangeStream {
    /// Create new change stream
    pub fn new(
        id: SubscriptionId,
        options: ChangeStreamOptions,
        receiver: broadcast::Receiver<ChangeEvent>,
    ) -> Self {
        let current_token = if let Some(ref token) = options.resume_token {
            token.clone()
        } else {
            ChangeStreamToken {
                wal_offset: 0,
                timestamp: 0,
                checksum: 0,
            }
        };

        Self {
            id,
            options,
            current_token: Arc::new(RwLock::new(current_token)),
            receiver,
            buffer: Arc::new(RwLock::new(Vec::new())),
            next_event_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Get next batch of events
    pub async fn next_batch(&mut self) -> Result<Vec<ChangeEvent>, String> {
        let mut events = Vec::new();
        let mut timeout = tokio::time::Duration::from_millis(self.options.max_await_time_ms);

        // Check buffer first
        {
            let mut buffer = self.buffer.write().await;
            if !buffer.is_empty() {
                let batch_size = self.options.batch_size.min(buffer.len());
                events = buffer.drain(..batch_size).collect();
            }
        }

        // Receive new events
        while events.len() < self.options.batch_size {
            match tokio::time::timeout(timeout, self.receiver.recv()).await {
                Ok(Ok(event)) => {
                    if self.matches_filter(&event) {
                        events.push(event);
                    }
                    timeout = tokio::time::Duration::from_millis(100); // Shorter timeout for subsequent events
                }
                Ok(Err(_)) => break, // Channel closed
                Err(_) => break,     // Timeout
            }
        }

        // Update token
        if let Some(last_event) = events.last() {
            let mut token = self.current_token.write().await;
            token.wal_offset = last_event.id;
            token.timestamp = last_event.timestamp;
            token.checksum = self.compute_checksum(&events);
        }

        Ok(events)
    }

    /// Get current resume token
    pub async fn resume_token(&self) -> ChangeStreamToken {
        self.current_token.read().await.clone()
    }

    /// Check if event matches filter
    fn matches_filter(&self, event: &ChangeEvent) -> bool {
        // Check operation type
        if !self.options.filter.operations.contains(&event.operation) {
            return false;
        }

        // Check key pattern
        if let Some(ref pattern) = self.options.filter.key_pattern {
            if !self.matches_pattern(&event.key, pattern) {
                return false;
            }
        }

        // Check table
        if let Some(ref filter_table) = self.options.filter.table {
            if let Some(ref event_table) = event.table {
                if filter_table != event_table {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Match key against pattern (supports * and ? wildcards)
    fn matches_pattern(&self, key: &str, pattern: &str) -> bool {
        // Simple wildcard matching
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let key_chars: Vec<char> = key.chars().collect();
        self.match_pattern_recursive(&key_chars, &pattern_chars, 0, 0)
    }

    fn match_pattern_recursive(
        &self,
        key: &[char],
        pattern: &[char],
        key_idx: usize,
        pattern_idx: usize,
    ) -> bool {
        if pattern_idx >= pattern.len() {
            return key_idx >= key.len();
        }

        match pattern[pattern_idx] {
            '*' => {
                // Match zero or more characters
                for i in key_idx..=key.len() {
                    if self.match_pattern_recursive(key, pattern, i, pattern_idx + 1) {
                        return true;
                    }
                }
                false
            }
            '?' => {
                // Match exactly one character
                if key_idx < key.len() {
                    self.match_pattern_recursive(key, pattern, key_idx + 1, pattern_idx + 1)
                } else {
                    false
                }
            }
            c => {
                // Match exact character
                if key_idx < key.len() && key[key_idx] == c {
                    self.match_pattern_recursive(key, pattern, key_idx + 1, pattern_idx + 1)
                } else {
                    false
                }
            }
        }
    }

    /// Compute checksum for events
    fn compute_checksum(&self, events: &[ChangeEvent]) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for event in events {
            event.id.hash(&mut hasher);
            event.timestamp.hash(&mut hasher);
        }
        hasher.finish() as u32
    }
}

/// Change stream manager
pub struct ChangeStreamManager {
    /// Active streams
    streams: Arc<RwLock<HashMap<SubscriptionId, Arc<RwLock<ChangeStream>>>>>,
    /// Event broadcaster
    broadcaster: broadcast::Sender<ChangeEvent>,
    /// Next stream ID
    next_id: Arc<AtomicU64>,
}

impl ChangeStreamManager {
    /// Create new change stream manager
    pub fn new(broadcaster: broadcast::Sender<ChangeEvent>) -> Self {
        Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            broadcaster,
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Create new change stream
    pub async fn create_stream(&self, options: ChangeStreamOptions) -> SubscriptionId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let receiver = self.broadcaster.subscribe();
        let stream = ChangeStream::new(id, options, receiver);

        let mut streams = self.streams.write().await;
        streams.insert(id, Arc::new(RwLock::new(stream)));

        id
    }

    /// Get stream by ID
    pub async fn get_stream(&self, id: SubscriptionId) -> Option<Arc<RwLock<ChangeStream>>> {
        let streams = self.streams.read().await;
        streams.get(&id).cloned()
    }

    /// Close stream
    pub async fn close_stream(&self, id: SubscriptionId) {
        let mut streams = self.streams.write().await;
        streams.remove(&id);
    }

    /// Broadcast event to all streams
    pub fn broadcast_event(&self, event: ChangeEvent) {
        let _ = self.broadcaster.send(event);
    }
}
