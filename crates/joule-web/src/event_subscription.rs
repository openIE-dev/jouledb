//! Event subscriptions — persistent subscription with checkpoint, competing
//! consumers (partition-based), subscription filter, subscription position
//! management, catch-up subscription, and live subscription.
//!
//! Replaces JS subscription libraries (EventStoreDB subscriptions, Kafka
//! consumers) with a pure-Rust event subscription engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Subscription errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptionError {
    /// Subscription not found.
    NotFound(String),
    /// Subscription already exists.
    AlreadyExists(String),
    /// Consumer group not found.
    ConsumerGroupNotFound(String),
    /// Consumer not found.
    ConsumerNotFound { group_id: String, consumer_id: String },
    /// Partition not found.
    PartitionNotFound { group_id: String, partition: u32 },
    /// Subscription is not active.
    NotActive(String),
    /// Subscription is already active.
    AlreadyActive(String),
    /// Checkpoint error.
    CheckpointError(String),
    /// Filter error.
    FilterError(String),
}

impl std::fmt::Display for SubscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "subscription not found: {id}"),
            Self::AlreadyExists(id) => write!(f, "subscription already exists: {id}"),
            Self::ConsumerGroupNotFound(id) => write!(f, "consumer group not found: {id}"),
            Self::ConsumerNotFound { group_id, consumer_id } => {
                write!(f, "consumer {consumer_id} not found in group {group_id}")
            }
            Self::PartitionNotFound { group_id, partition } => {
                write!(f, "partition {partition} not found in group {group_id}")
            }
            Self::NotActive(id) => write!(f, "subscription {id} is not active"),
            Self::AlreadyActive(id) => write!(f, "subscription {id} is already active"),
            Self::CheckpointError(msg) => write!(f, "checkpoint error: {msg}"),
            Self::FilterError(msg) => write!(f, "filter error: {msg}"),
        }
    }
}

impl std::error::Error for SubscriptionError {}

// ── Subscription Mode ───────────────────────────────────────────

/// Mode of a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubscriptionMode {
    /// Catch-up: reads from a stored position, catches up to head.
    CatchUp,
    /// Live: only receives new events from this point forward.
    Live,
    /// Catch-up then live: catches up, then switches to live.
    CatchUpThenLive,
}

// ── Subscription Status ─────────────────────────────────────────

/// Status of a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    Created,
    Active,
    Paused,
    Stopped,
    CatchingUp,
    Live,
    Faulted,
}

// ── Checkpoint ──────────────────────────────────────────────────

/// A persistent checkpoint for a subscription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub subscription_id: String,
    pub position: u64,
    pub updated_at: DateTime<Utc>,
}

impl Checkpoint {
    pub fn new(subscription_id: impl Into<String>) -> Self {
        Self {
            subscription_id: subscription_id.into(),
            position: 0,
            updated_at: Utc::now(),
        }
    }

    pub fn advance(&mut self, position: u64) {
        if position > self.position {
            self.position = position;
            self.updated_at = Utc::now();
        }
    }
}

// ── Subscription Filter ─────────────────────────────────────────

/// Filters events for a subscription.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    /// Include only events from these streams (empty = all).
    pub stream_ids: Vec<String>,
    /// Include only these event types (empty = all).
    pub event_types: Vec<String>,
    /// Exclude these event types.
    pub exclude_event_types: Vec<String>,
    /// Stream ID prefix filter.
    pub stream_prefix: Option<String>,
}

impl SubscriptionFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_streams(mut self, streams: Vec<String>) -> Self {
        self.stream_ids = streams;
        self
    }

    pub fn with_event_types(mut self, types: Vec<String>) -> Self {
        self.event_types = types;
        self
    }

    pub fn with_exclude_event_types(mut self, types: Vec<String>) -> Self {
        self.exclude_event_types = types;
        self
    }

    pub fn with_stream_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.stream_prefix = Some(prefix.into());
        self
    }

    /// Check if an event passes this filter.
    pub fn matches(&self, stream_id: &str, event_type: &str) -> bool {
        // Stream filter.
        if !self.stream_ids.is_empty() && !self.stream_ids.iter().any(|s| s == stream_id) {
            return false;
        }

        // Stream prefix filter.
        if let Some(prefix) = &self.stream_prefix {
            if !stream_id.starts_with(prefix.as_str()) {
                return false;
            }
        }

        // Event type include filter.
        if !self.event_types.is_empty() && !self.event_types.iter().any(|t| t == event_type) {
            return false;
        }

        // Event type exclude filter.
        if self.exclude_event_types.iter().any(|t| t == event_type) {
            return false;
        }

        true
    }
}

// ── Subscription Event ──────────────────────────────────────────

/// An event delivered to a subscription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscriptionEvent {
    pub global_position: u64,
    pub stream_id: String,
    pub event_type: String,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl SubscriptionEvent {
    pub fn new(
        global_position: u64,
        stream_id: impl Into<String>,
        event_type: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        Self {
            global_position,
            stream_id: stream_id.into(),
            event_type: event_type.into(),
            data,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
}

// ── Persistent Subscription ─────────────────────────────────────

/// A persistent subscription with checkpoint and filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSubscription {
    pub subscription_id: String,
    pub mode: SubscriptionMode,
    pub status: SubscriptionStatus,
    pub filter: SubscriptionFilter,
    pub checkpoint: Checkpoint,
    pub events_delivered: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
    /// Position at which this subscription considers itself "caught up".
    pub live_position: Option<u64>,
}

impl PersistentSubscription {
    pub fn new(
        subscription_id: impl Into<String>,
        mode: SubscriptionMode,
    ) -> Self {
        let sid = subscription_id.into();
        let now = Utc::now();
        Self {
            subscription_id: sid.clone(),
            mode,
            status: SubscriptionStatus::Created,
            filter: SubscriptionFilter::new(),
            checkpoint: Checkpoint::new(sid),
            events_delivered: 0,
            created_at: now,
            updated_at: now,
            error_message: None,
            live_position: None,
        }
    }

    pub fn with_filter(mut self, filter: SubscriptionFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Activate the subscription.
    pub fn activate(&mut self) -> Result<(), SubscriptionError> {
        if self.status == SubscriptionStatus::Active || self.status == SubscriptionStatus::Live {
            return Err(SubscriptionError::AlreadyActive(self.subscription_id.clone()));
        }
        self.status = match self.mode {
            SubscriptionMode::CatchUp | SubscriptionMode::CatchUpThenLive => {
                SubscriptionStatus::CatchingUp
            }
            SubscriptionMode::Live => SubscriptionStatus::Live,
        };
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Pause the subscription.
    pub fn pause(&mut self) {
        self.status = SubscriptionStatus::Paused;
        self.updated_at = Utc::now();
    }

    /// Stop the subscription.
    pub fn stop(&mut self) {
        self.status = SubscriptionStatus::Stopped;
        self.updated_at = Utc::now();
    }

    /// Mark that the subscription has caught up to the head.
    pub fn mark_caught_up(&mut self) {
        match self.mode {
            SubscriptionMode::CatchUpThenLive => {
                self.status = SubscriptionStatus::Live;
                self.live_position = Some(self.checkpoint.position);
            }
            SubscriptionMode::CatchUp => {
                self.status = SubscriptionStatus::Active;
            }
            SubscriptionMode::Live => {}
        }
        self.updated_at = Utc::now();
    }

    /// Process events and advance checkpoint.
    pub fn receive_events(&mut self, events: &[SubscriptionEvent]) -> Vec<SubscriptionEvent> {
        let is_active = matches!(
            self.status,
            SubscriptionStatus::Active
                | SubscriptionStatus::CatchingUp
                | SubscriptionStatus::Live
        );
        if !is_active {
            return Vec::new();
        }

        let mut delivered = Vec::new();
        for event in events {
            if event.global_position < self.checkpoint.position {
                continue;
            }
            if !self.filter.matches(&event.stream_id, &event.event_type) {
                continue;
            }
            delivered.push(event.clone());
            self.checkpoint.advance(event.global_position + 1);
            self.events_delivered += 1;
        }
        self.updated_at = Utc::now();
        delivered
    }

    /// Set checkpoint manually.
    pub fn set_checkpoint(&mut self, position: u64) {
        self.checkpoint.position = position;
        self.checkpoint.updated_at = Utc::now();
        self.updated_at = Utc::now();
    }

    /// Check if the subscription is receiving events.
    pub fn is_receiving(&self) -> bool {
        matches!(
            self.status,
            SubscriptionStatus::Active
                | SubscriptionStatus::CatchingUp
                | SubscriptionStatus::Live
        )
    }
}

// ── Consumer Assignment ─────────────────────────────────────────

/// A consumer in a consumer group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consumer {
    pub consumer_id: String,
    pub assigned_partitions: Vec<u32>,
    pub last_heartbeat: DateTime<Utc>,
    pub events_processed: u64,
}

impl Consumer {
    pub fn new(consumer_id: impl Into<String>) -> Self {
        Self {
            consumer_id: consumer_id.into(),
            assigned_partitions: Vec::new(),
            last_heartbeat: Utc::now(),
            events_processed: 0,
        }
    }

    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Utc::now();
    }
}

// ── Consumer Group ──────────────────────────────────────────────

/// A group of competing consumers sharing partitions.
#[derive(Debug, Clone)]
pub struct ConsumerGroup {
    pub group_id: String,
    pub partition_count: u32,
    consumers: Vec<Consumer>,
    /// Per-partition checkpoints.
    partition_checkpoints: HashMap<u32, u64>,
}

impl ConsumerGroup {
    pub fn new(group_id: impl Into<String>, partition_count: u32) -> Self {
        let mut checkpoints = HashMap::new();
        for p in 0..partition_count {
            checkpoints.insert(p, 0);
        }
        Self {
            group_id: group_id.into(),
            partition_count,
            consumers: Vec::new(),
            partition_checkpoints: checkpoints,
        }
    }

    /// Add a consumer and rebalance partitions.
    pub fn add_consumer(&mut self, consumer_id: impl Into<String>) {
        let consumer = Consumer::new(consumer_id);
        self.consumers.push(consumer);
        self.rebalance();
    }

    /// Remove a consumer and rebalance.
    pub fn remove_consumer(&mut self, consumer_id: &str) -> Option<Consumer> {
        let pos = self.consumers.iter().position(|c| c.consumer_id == consumer_id);
        if let Some(idx) = pos {
            let removed = self.consumers.remove(idx);
            self.rebalance();
            Some(removed)
        } else {
            None
        }
    }

    /// Rebalance partitions across consumers (round-robin).
    fn rebalance(&mut self) {
        // Clear all assignments.
        for consumer in &mut self.consumers {
            consumer.assigned_partitions.clear();
        }

        if self.consumers.is_empty() {
            return;
        }

        // Round-robin assignment.
        for p in 0..self.partition_count {
            let consumer_idx = (p as usize) % self.consumers.len();
            self.consumers[consumer_idx]
                .assigned_partitions
                .push(p);
        }
    }

    /// Determine partition for an event (by stream_id hash).
    pub fn partition_for(&self, stream_id: &str) -> u32 {
        if self.partition_count == 0 {
            return 0;
        }
        let hash = stream_id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash % (self.partition_count as u64)) as u32
    }

    /// Get the consumer assigned to a partition.
    pub fn consumer_for_partition(&self, partition: u32) -> Option<&Consumer> {
        self.consumers
            .iter()
            .find(|c| c.assigned_partitions.contains(&partition))
    }

    /// Checkpoint a partition.
    pub fn checkpoint_partition(&mut self, partition: u32, position: u64) -> Result<(), SubscriptionError> {
        let checkpoint = self
            .partition_checkpoints
            .get_mut(&partition)
            .ok_or_else(|| SubscriptionError::PartitionNotFound {
                group_id: self.group_id.clone(),
                partition,
            })?;
        if position > *checkpoint {
            *checkpoint = position;
        }
        Ok(())
    }

    /// Get partition checkpoint.
    pub fn partition_checkpoint(&self, partition: u32) -> Option<u64> {
        self.partition_checkpoints.get(&partition).copied()
    }

    /// Consumer count.
    pub fn consumer_count(&self) -> usize {
        self.consumers.len()
    }

    /// Get all consumers.
    pub fn consumers(&self) -> &[Consumer] {
        &self.consumers
    }

    /// Heartbeat a consumer.
    pub fn heartbeat(&mut self, consumer_id: &str) -> Result<(), SubscriptionError> {
        let consumer = self
            .consumers
            .iter_mut()
            .find(|c| c.consumer_id == consumer_id)
            .ok_or_else(|| SubscriptionError::ConsumerNotFound {
                group_id: self.group_id.clone(),
                consumer_id: consumer_id.to_string(),
            })?;
        consumer.heartbeat();
        Ok(())
    }

    /// Record events processed by a consumer.
    pub fn record_processed(&mut self, consumer_id: &str, count: u64) -> Result<(), SubscriptionError> {
        let consumer = self
            .consumers
            .iter_mut()
            .find(|c| c.consumer_id == consumer_id)
            .ok_or_else(|| SubscriptionError::ConsumerNotFound {
                group_id: self.group_id.clone(),
                consumer_id: consumer_id.to_string(),
            })?;
        consumer.events_processed += count;
        Ok(())
    }
}

// ── Subscription Manager ────────────────────────────────────────

/// Manages persistent subscriptions and consumer groups.
#[derive(Debug)]
pub struct SubscriptionManager {
    subscriptions: HashMap<String, PersistentSubscription>,
    consumer_groups: HashMap<String, ConsumerGroup>,
}

impl SubscriptionManager {
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            consumer_groups: HashMap::new(),
        }
    }

    /// Create a persistent subscription.
    pub fn create_subscription(
        &mut self,
        subscription: PersistentSubscription,
    ) -> Result<(), SubscriptionError> {
        if self.subscriptions.contains_key(&subscription.subscription_id) {
            return Err(SubscriptionError::AlreadyExists(
                subscription.subscription_id.clone(),
            ));
        }
        self.subscriptions
            .insert(subscription.subscription_id.clone(), subscription);
        Ok(())
    }

    /// Get a subscription.
    pub fn get_subscription(&self, id: &str) -> Option<&PersistentSubscription> {
        self.subscriptions.get(id)
    }

    /// Get a mutable subscription.
    pub fn get_subscription_mut(&mut self, id: &str) -> Option<&mut PersistentSubscription> {
        self.subscriptions.get_mut(id)
    }

    /// Remove a subscription.
    pub fn remove_subscription(&mut self, id: &str) -> Result<PersistentSubscription, SubscriptionError> {
        self.subscriptions
            .remove(id)
            .ok_or_else(|| SubscriptionError::NotFound(id.to_string()))
    }

    /// Create a consumer group.
    pub fn create_consumer_group(&mut self, group: ConsumerGroup) -> Result<(), SubscriptionError> {
        if self.consumer_groups.contains_key(&group.group_id) {
            return Err(SubscriptionError::AlreadyExists(group.group_id.clone()));
        }
        self.consumer_groups
            .insert(group.group_id.clone(), group);
        Ok(())
    }

    /// Get a consumer group.
    pub fn get_consumer_group(&self, id: &str) -> Option<&ConsumerGroup> {
        self.consumer_groups.get(id)
    }

    /// Get a mutable consumer group.
    pub fn get_consumer_group_mut(&mut self, id: &str) -> Option<&mut ConsumerGroup> {
        self.consumer_groups.get_mut(id)
    }

    /// Dispatch events to all active subscriptions.
    pub fn dispatch(
        &mut self,
        events: &[SubscriptionEvent],
    ) -> Vec<(String, Vec<SubscriptionEvent>)> {
        let ids: Vec<String> = self.subscriptions.keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            if let Some(sub) = self.subscriptions.get_mut(&id) {
                let delivered = sub.receive_events(events);
                if !delivered.is_empty() {
                    results.push((id, delivered));
                }
            }
        }
        results
    }

    /// List subscription IDs (sorted).
    pub fn subscription_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.subscriptions.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// List consumer group IDs (sorted).
    pub fn consumer_group_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.consumer_groups.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Count subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Count consumer groups.
    pub fn consumer_group_count(&self) -> usize {
        self.consumer_groups.len()
    }

    /// Count subscriptions by status.
    pub fn count_by_status(&self, status: SubscriptionStatus) -> usize {
        self.subscriptions.values().filter(|s| s.status == status).count()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(pos: u64, stream: &str, event_type: &str) -> SubscriptionEvent {
        SubscriptionEvent::new(pos, stream, event_type, HashMap::new())
    }

    #[test]
    fn test_persistent_subscription_catch_up() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp);
        sub.activate().unwrap();
        assert_eq!(sub.status, SubscriptionStatus::CatchingUp);

        let events = vec![
            make_event(0, "s1", "E1"),
            make_event(1, "s1", "E2"),
            make_event(2, "s1", "E3"),
        ];
        let delivered = sub.receive_events(&events);
        assert_eq!(delivered.len(), 3);
        assert_eq!(sub.checkpoint.position, 3);
        assert_eq!(sub.events_delivered, 3);
    }

    #[test]
    fn test_persistent_subscription_live() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::Live);
        sub.set_checkpoint(10);
        sub.activate().unwrap();
        assert_eq!(sub.status, SubscriptionStatus::Live);

        // Events before checkpoint are skipped.
        let events = vec![
            make_event(8, "s1", "E1"),
            make_event(9, "s1", "E2"),
            make_event(10, "s1", "E3"),
            make_event(11, "s1", "E4"),
        ];
        let delivered = sub.receive_events(&events);
        assert_eq!(delivered.len(), 2); // Only pos 10 and 11.
    }

    #[test]
    fn test_catch_up_then_live() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUpThenLive);
        sub.activate().unwrap();
        assert_eq!(sub.status, SubscriptionStatus::CatchingUp);

        let events = vec![make_event(0, "s1", "E1"), make_event(1, "s1", "E2")];
        sub.receive_events(&events);

        sub.mark_caught_up();
        assert_eq!(sub.status, SubscriptionStatus::Live);
        assert!(sub.live_position.is_some());
    }

    #[test]
    fn test_subscription_filter_streams() {
        let filter = SubscriptionFilter::new()
            .with_streams(vec!["s1".to_string(), "s2".to_string()]);

        assert!(filter.matches("s1", "E1"));
        assert!(filter.matches("s2", "E1"));
        assert!(!filter.matches("s3", "E1"));
    }

    #[test]
    fn test_subscription_filter_event_types() {
        let filter = SubscriptionFilter::new()
            .with_event_types(vec!["Created".to_string(), "Updated".to_string()]);

        assert!(filter.matches("s1", "Created"));
        assert!(filter.matches("s1", "Updated"));
        assert!(!filter.matches("s1", "Deleted"));
    }

    #[test]
    fn test_subscription_filter_exclude() {
        let filter = SubscriptionFilter::new()
            .with_exclude_event_types(vec!["Heartbeat".to_string()]);

        assert!(filter.matches("s1", "Created"));
        assert!(!filter.matches("s1", "Heartbeat"));
    }

    #[test]
    fn test_subscription_filter_stream_prefix() {
        let filter = SubscriptionFilter::new()
            .with_stream_prefix("order-");

        assert!(filter.matches("order-123", "E1"));
        assert!(filter.matches("order-456", "E1"));
        assert!(!filter.matches("user-789", "E1"));
    }

    #[test]
    fn test_subscription_filter_combined() {
        let filter = SubscriptionFilter::new()
            .with_stream_prefix("order-")
            .with_event_types(vec!["OrderPlaced".to_string()])
            .with_exclude_event_types(vec!["OrderCancelled".to_string()]);

        assert!(filter.matches("order-1", "OrderPlaced"));
        assert!(!filter.matches("order-1", "OrderCancelled"));
        assert!(!filter.matches("user-1", "OrderPlaced"));
        assert!(!filter.matches("order-1", "OrderShipped"));
    }

    #[test]
    fn test_subscription_with_filter() {
        let filter = SubscriptionFilter::new()
            .with_event_types(vec!["Created".to_string()]);
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp)
            .with_filter(filter);
        sub.activate().unwrap();

        let events = vec![
            make_event(0, "s1", "Created"),
            make_event(1, "s1", "Updated"),
            make_event(2, "s1", "Created"),
        ];
        let delivered = sub.receive_events(&events);
        assert_eq!(delivered.len(), 2);
        assert!(delivered.iter().all(|e| e.event_type == "Created"));
    }

    #[test]
    fn test_subscription_pause_resume() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp);
        sub.activate().unwrap();
        sub.receive_events(&[make_event(0, "s1", "E1")]);

        sub.pause();
        assert_eq!(sub.status, SubscriptionStatus::Paused);
        assert!(sub.receive_events(&[make_event(1, "s1", "E2")]).is_empty());

        sub.activate().unwrap();
        let delivered = sub.receive_events(&[make_event(1, "s1", "E2")]);
        assert_eq!(delivered.len(), 1);
    }

    #[test]
    fn test_subscription_stop() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::Live);
        sub.activate().unwrap();
        sub.stop();
        assert_eq!(sub.status, SubscriptionStatus::Stopped);
        assert!(!sub.is_receiving());
    }

    #[test]
    fn test_subscription_double_activate() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::Live);
        sub.activate().unwrap();
        let err = sub.activate().unwrap_err();
        assert!(matches!(err, SubscriptionError::AlreadyActive(_)));
    }

    #[test]
    fn test_checkpoint_advance() {
        let mut cp = Checkpoint::new("sub-1");
        assert_eq!(cp.position, 0);
        cp.advance(5);
        assert_eq!(cp.position, 5);
        cp.advance(3); // Should not go backwards.
        assert_eq!(cp.position, 5);
        cp.advance(10);
        assert_eq!(cp.position, 10);
    }

    #[test]
    fn test_consumer_group_creation() {
        let group = ConsumerGroup::new("group-1", 4);
        assert_eq!(group.partition_count, 4);
        assert_eq!(group.consumer_count(), 0);
    }

    #[test]
    fn test_consumer_group_add_remove() {
        let mut group = ConsumerGroup::new("g1", 4);
        group.add_consumer("c1");
        group.add_consumer("c2");
        assert_eq!(group.consumer_count(), 2);

        // c1 should have partitions 0, 2; c2 should have 1, 3.
        let c1 = &group.consumers()[0];
        let c2 = &group.consumers()[1];
        assert_eq!(c1.assigned_partitions, vec![0, 2]);
        assert_eq!(c2.assigned_partitions, vec![1, 3]);

        group.remove_consumer("c1");
        assert_eq!(group.consumer_count(), 1);
        // All partitions assigned to remaining consumer.
        assert_eq!(group.consumers()[0].assigned_partitions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_consumer_group_partition_for() {
        let group = ConsumerGroup::new("g1", 4);
        let p1 = group.partition_for("stream-a");
        let p2 = group.partition_for("stream-b");
        assert!(p1 < 4);
        assert!(p2 < 4);
        // Same stream always maps to same partition.
        assert_eq!(group.partition_for("stream-a"), p1);
    }

    #[test]
    fn test_consumer_group_checkpoint() {
        let mut group = ConsumerGroup::new("g1", 2);
        group.checkpoint_partition(0, 10).unwrap();
        group.checkpoint_partition(1, 20).unwrap();

        assert_eq!(group.partition_checkpoint(0), Some(10));
        assert_eq!(group.partition_checkpoint(1), Some(20));
    }

    #[test]
    fn test_consumer_group_checkpoint_invalid_partition() {
        let mut group = ConsumerGroup::new("g1", 2);
        let err = group.checkpoint_partition(99, 10).unwrap_err();
        assert!(matches!(err, SubscriptionError::PartitionNotFound { .. }));
    }

    #[test]
    fn test_consumer_heartbeat() {
        let mut group = ConsumerGroup::new("g1", 2);
        group.add_consumer("c1");
        group.heartbeat("c1").unwrap();
        let err = group.heartbeat("ghost").unwrap_err();
        assert!(matches!(err, SubscriptionError::ConsumerNotFound { .. }));
    }

    #[test]
    fn test_consumer_record_processed() {
        let mut group = ConsumerGroup::new("g1", 2);
        group.add_consumer("c1");
        group.record_processed("c1", 10).unwrap();
        assert_eq!(group.consumers()[0].events_processed, 10);
    }

    #[test]
    fn test_manager_create_subscription() {
        let mut mgr = SubscriptionManager::new();
        let sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp);
        mgr.create_subscription(sub).unwrap();
        assert!(mgr.get_subscription("sub-1").is_some());
    }

    #[test]
    fn test_manager_duplicate_subscription() {
        let mut mgr = SubscriptionManager::new();
        mgr.create_subscription(PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp))
            .unwrap();
        let err = mgr
            .create_subscription(PersistentSubscription::new("sub-1", SubscriptionMode::Live))
            .unwrap_err();
        assert!(matches!(err, SubscriptionError::AlreadyExists(_)));
    }

    #[test]
    fn test_manager_dispatch() {
        let mut mgr = SubscriptionManager::new();

        let mut s1 = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp);
        s1.activate().unwrap();
        mgr.create_subscription(s1).unwrap();

        let mut s2 = PersistentSubscription::new("sub-2", SubscriptionMode::CatchUp);
        s2.activate().unwrap();
        mgr.create_subscription(s2).unwrap();

        let events = vec![make_event(0, "s1", "E1")];
        let results = mgr.dispatch(&events);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_manager_remove_subscription() {
        let mut mgr = SubscriptionManager::new();
        mgr.create_subscription(PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp))
            .unwrap();
        let removed = mgr.remove_subscription("sub-1").unwrap();
        assert_eq!(removed.subscription_id, "sub-1");
        assert!(mgr.get_subscription("sub-1").is_none());
    }

    #[test]
    fn test_manager_remove_not_found() {
        let mut mgr = SubscriptionManager::new();
        let err = mgr.remove_subscription("ghost").unwrap_err();
        assert!(matches!(err, SubscriptionError::NotFound(_)));
    }

    #[test]
    fn test_manager_subscription_ids_sorted() {
        let mut mgr = SubscriptionManager::new();
        mgr.create_subscription(PersistentSubscription::new("zulu", SubscriptionMode::CatchUp))
            .unwrap();
        mgr.create_subscription(PersistentSubscription::new("alpha", SubscriptionMode::CatchUp))
            .unwrap();
        assert_eq!(mgr.subscription_ids(), vec!["alpha", "zulu"]);
    }

    #[test]
    fn test_manager_consumer_group() {
        let mut mgr = SubscriptionManager::new();
        let group = ConsumerGroup::new("g1", 4);
        mgr.create_consumer_group(group).unwrap();
        assert!(mgr.get_consumer_group("g1").is_some());
        assert_eq!(mgr.consumer_group_count(), 1);
    }

    #[test]
    fn test_manager_count_by_status() {
        let mut mgr = SubscriptionManager::new();
        mgr.create_subscription(PersistentSubscription::new("s1", SubscriptionMode::CatchUp))
            .unwrap();
        mgr.create_subscription(PersistentSubscription::new("s2", SubscriptionMode::CatchUp))
            .unwrap();
        assert_eq!(mgr.count_by_status(SubscriptionStatus::Created), 2);
    }

    #[test]
    fn test_consumer_for_partition() {
        let mut group = ConsumerGroup::new("g1", 4);
        group.add_consumer("c1");
        group.add_consumer("c2");

        let c = group.consumer_for_partition(0).unwrap();
        assert_eq!(c.consumer_id, "c1");
        let c = group.consumer_for_partition(1).unwrap();
        assert_eq!(c.consumer_id, "c2");
    }

    #[test]
    fn test_set_checkpoint_manually() {
        let mut sub = PersistentSubscription::new("sub-1", SubscriptionMode::CatchUp);
        sub.set_checkpoint(42);
        assert_eq!(sub.checkpoint.position, 42);
    }

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = SubscriptionFilter::new();
        assert!(filter.matches("any-stream", "AnyEvent"));
    }
}
