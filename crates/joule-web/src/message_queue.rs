//! In-memory message queue — FIFO with priorities, delayed messages, acknowledgment,
//! dead letter queue, consumer groups, message TTL, and queue size limits.
//!
//! Replaces RabbitMQ/SQS/BullMQ client libraries with a pure-Rust in-memory
//! queue that models the full lifecycle: enqueue, dequeue, acknowledge, retry,
//! dead-letter, and expiry — all with energy tracking.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Message queue domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqError {
    /// Queue not found.
    QueueNotFound(String),
    /// Queue is full (size limit reached).
    QueueFull { queue: String, limit: usize },
    /// Message not found.
    MessageNotFound(String),
    /// Message has expired (TTL).
    MessageExpired(String),
    /// Consumer group not found.
    GroupNotFound(String),
    /// Consumer not found in group.
    ConsumerNotFound(String),
    /// Message already acknowledged.
    AlreadyAcknowledged(String),
    /// Duplicate message ID.
    DuplicateMessage(String),
}

impl std::fmt::Display for MqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueNotFound(q) => write!(f, "queue not found: {q}"),
            Self::QueueFull { queue, limit } => {
                write!(f, "queue {queue} is full (limit {limit})")
            }
            Self::MessageNotFound(id) => write!(f, "message not found: {id}"),
            Self::MessageExpired(id) => write!(f, "message expired: {id}"),
            Self::GroupNotFound(g) => write!(f, "consumer group not found: {g}"),
            Self::ConsumerNotFound(c) => write!(f, "consumer not found: {c}"),
            Self::AlreadyAcknowledged(id) => write!(f, "message already acknowledged: {id}"),
            Self::DuplicateMessage(id) => write!(f, "duplicate message: {id}"),
        }
    }
}

impl std::error::Error for MqError {}

// ── Priority ──────────────────────────────────────────────────

/// Message priority (lower numeric value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const CRITICAL: Priority = Priority(0);
    pub const HIGH: Priority = Priority(10);
    pub const NORMAL: Priority = Priority(50);
    pub const LOW: Priority = Priority(100);
}

impl Default for Priority {
    fn default() -> Self {
        Priority::NORMAL
    }
}

// ── Message Status ────────────────────────────────────────────

/// Lifecycle status of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageStatus {
    /// Waiting in the queue.
    Pending,
    /// Delayed — not yet eligible for delivery.
    Delayed,
    /// Delivered to a consumer, awaiting acknowledgment.
    Delivered,
    /// Successfully acknowledged.
    Acknowledged,
    /// Moved to the dead letter queue.
    DeadLettered,
    /// Expired due to TTL.
    Expired,
}

// ── Message ───────────────────────────────────────────────────

/// A message in the queue.
#[derive(Debug, Clone)]
pub struct Message {
    pub id: String,
    pub payload: String,
    pub priority: Priority,
    pub status: MessageStatus,
    /// Number of delivery attempts.
    pub attempt: u32,
    /// Maximum delivery attempts before dead-lettering.
    pub max_attempts: u32,
    /// Time-to-live in milliseconds (None = no expiry).
    pub ttl_ms: Option<u64>,
    /// Simulated creation timestamp in ms.
    pub created_at_ms: u64,
    /// Delay before the message becomes eligible (in ms from creation).
    pub delay_ms: u64,
    /// Headers / metadata.
    pub headers: HashMap<String, String>,
    /// Which consumer group owns this delivery (if any).
    pub consumer_group: Option<String>,
    /// Which consumer in the group holds the delivery (if any).
    pub consumer_id: Option<String>,
}

impl Message {
    pub fn new(id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            payload: payload.into(),
            priority: Priority::default(),
            status: MessageStatus::Pending,
            attempt: 0,
            max_attempts: 3,
            ttl_ms: None,
            created_at_ms: 0,
            delay_ms: 0,
            headers: HashMap::new(),
            consumer_group: None,
            consumer_id: None,
        }
    }

    pub fn with_priority(mut self, p: Priority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_ttl(mut self, ms: u64) -> Self {
        self.ttl_ms = Some(ms);
        self
    }

    pub fn with_delay(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self.status = MessageStatus::Delayed;
        self
    }

    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Whether the message is past its TTL at the given simulated time.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        if let Some(ttl) = self.ttl_ms {
            now_ms.saturating_sub(self.created_at_ms) >= ttl
        } else {
            false
        }
    }

    /// Whether the message is eligible for delivery at the given simulated time.
    pub fn is_eligible(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_at_ms) >= self.delay_ms
    }
}

// ── Consumer Group ────────────────────────────────────────────

/// A consumer group with round-robin assignment.
#[derive(Debug, Clone)]
pub struct ConsumerGroup {
    pub name: String,
    pub consumers: Vec<String>,
    /// Round-robin index.
    next_consumer: usize,
}

impl ConsumerGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            consumers: Vec::new(),
            next_consumer: 0,
        }
    }

    pub fn add_consumer(&mut self, id: impl Into<String>) {
        self.consumers.push(id.into());
    }

    pub fn remove_consumer(&mut self, id: &str) -> bool {
        if let Some(pos) = self.consumers.iter().position(|c| c == id) {
            self.consumers.remove(pos);
            if self.next_consumer >= self.consumers.len() && !self.consumers.is_empty() {
                self.next_consumer = 0;
            }
            true
        } else {
            false
        }
    }

    /// Pick the next consumer in round-robin order.
    pub fn pick_next(&mut self) -> Option<String> {
        if self.consumers.is_empty() {
            return None;
        }
        let consumer = self.consumers[self.next_consumer].clone();
        self.next_consumer = (self.next_consumer + 1) % self.consumers.len();
        Some(consumer)
    }
}

// ── Queue Stats ───────────────────────────────────────────────

/// Statistics for a queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub enqueued: u64,
    pub delivered: u64,
    pub acknowledged: u64,
    pub dead_lettered: u64,
    pub expired: u64,
    pub current_size: usize,
}

// ── Message Queue ─────────────────────────────────────────────

/// A named message queue with priority ordering.
#[derive(Debug)]
pub struct MessageQueue {
    pub name: String,
    /// Messages indexed by priority then insertion order.
    pending: BTreeMap<Priority, VecDeque<String>>,
    /// All messages by ID.
    messages: HashMap<String, Message>,
    /// Dead letter queue.
    dead_letters: VecDeque<Message>,
    /// Consumer groups.
    groups: HashMap<String, ConsumerGroup>,
    /// Maximum queue size (None = unlimited).
    max_size: Option<usize>,
    /// Simulated clock (milliseconds).
    clock_ms: u64,
    /// ID uniqueness tracking.
    seen_ids: HashSet<String>,
    /// Stats.
    stats: QueueStats,
}

impl MessageQueue {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pending: BTreeMap::new(),
            messages: HashMap::new(),
            dead_letters: VecDeque::new(),
            groups: HashMap::new(),
            max_size: None,
            clock_ms: 0,
            seen_ids: HashSet::new(),
            stats: QueueStats::default(),
        }
    }

    pub fn with_max_size(mut self, limit: usize) -> Self {
        self.max_size = Some(limit);
        self
    }

    /// Advance the simulated clock.
    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    /// Set the simulated clock to an absolute value.
    pub fn set_clock(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    pub fn clock(&self) -> u64 {
        self.clock_ms
    }

    // ── Enqueue ───────────────────────────────────────────────

    /// Enqueue a message.
    pub fn enqueue(&mut self, mut msg: Message) -> Result<(), MqError> {
        if self.seen_ids.contains(&msg.id) {
            return Err(MqError::DuplicateMessage(msg.id));
        }
        let current = self.messages.values().filter(|m| {
            matches!(
                m.status,
                MessageStatus::Pending | MessageStatus::Delayed | MessageStatus::Delivered
            )
        }).count();
        if let Some(limit) = self.max_size {
            if current >= limit {
                return Err(MqError::QueueFull {
                    queue: self.name.clone(),
                    limit,
                });
            }
        }
        msg.created_at_ms = self.clock_ms;
        let id = msg.id.clone();
        let priority = msg.priority;
        self.seen_ids.insert(id.clone());
        self.messages.insert(id.clone(), msg);
        self.pending
            .entry(priority)
            .or_insert_with(VecDeque::new)
            .push_back(id);
        self.stats.enqueued += 1;
        self.stats.current_size += 1;
        Ok(())
    }

    // ── Expire ────────────────────────────────────────────────

    /// Expire messages that are past their TTL.
    pub fn expire_messages(&mut self) -> Vec<String> {
        let now = self.clock_ms;
        let mut expired_ids = Vec::new();
        for msg in self.messages.values_mut() {
            if msg.is_expired(now)
                && matches!(
                    msg.status,
                    MessageStatus::Pending | MessageStatus::Delayed
                )
            {
                msg.status = MessageStatus::Expired;
                expired_ids.push(msg.id.clone());
            }
        }
        // Remove expired from pending queues.
        for queue in self.pending.values_mut() {
            queue.retain(|id| !expired_ids.contains(id));
        }
        self.stats.expired += expired_ids.len() as u64;
        self.stats.current_size = self.stats.current_size.saturating_sub(expired_ids.len());
        expired_ids
    }

    // ── Promote delayed ──────────────────────────────────────

    /// Promote delayed messages that are now eligible.
    pub fn promote_delayed(&mut self) -> Vec<String> {
        let now = self.clock_ms;
        let mut promoted = Vec::new();
        for msg in self.messages.values_mut() {
            if msg.status == MessageStatus::Delayed && msg.is_eligible(now) {
                msg.status = MessageStatus::Pending;
                promoted.push((msg.id.clone(), msg.priority));
            }
        }
        let ids: Vec<String> = promoted.iter().map(|(id, _)| id.clone()).collect();
        for (id, priority) in promoted {
            self.pending
                .entry(priority)
                .or_insert_with(VecDeque::new)
                .push_back(id);
        }
        ids
    }

    // ── Dequeue ───────────────────────────────────────────────

    /// Dequeue the highest-priority eligible message (without consumer group).
    pub fn dequeue(&mut self) -> Option<Message> {
        self.expire_messages();
        self.promote_delayed();
        let now = self.clock_ms;
        // Try priorities in order (BTreeMap iterates in sorted order).
        for (_prio, queue) in self.pending.iter_mut() {
            while let Some(id) = queue.front() {
                let id_clone = id.clone();
                if let Some(msg) = self.messages.get(&id_clone) {
                    if msg.status != MessageStatus::Pending {
                        queue.pop_front();
                        continue;
                    }
                    if msg.is_expired(now) {
                        queue.pop_front();
                        continue;
                    }
                }
                queue.pop_front();
                if let Some(msg) = self.messages.get_mut(&id_clone) {
                    msg.status = MessageStatus::Delivered;
                    msg.attempt += 1;
                    self.stats.delivered += 1;
                    return Some(msg.clone());
                }
            }
        }
        None
    }

    /// Dequeue for a specific consumer group — round-robin assignment.
    pub fn dequeue_for_group(&mut self, group_name: &str) -> Result<Option<Message>, MqError> {
        if !self.groups.contains_key(group_name) {
            return Err(MqError::GroupNotFound(group_name.to_string()));
        }
        let consumer_id = {
            let group = self.groups.get_mut(group_name).unwrap();
            group.pick_next()
        };
        let mut msg = match self.dequeue() {
            Some(m) => m,
            None => return Ok(None),
        };
        msg.consumer_group = Some(group_name.to_string());
        msg.consumer_id = consumer_id;
        // Update stored message.
        if let Some(stored) = self.messages.get_mut(&msg.id) {
            stored.consumer_group.clone_from(&msg.consumer_group);
            stored.consumer_id.clone_from(&msg.consumer_id);
        }
        Ok(Some(msg))
    }

    // ── Acknowledge ──────────────────────────────────────────

    /// Acknowledge a delivered message.
    pub fn acknowledge(&mut self, message_id: &str) -> Result<(), MqError> {
        let msg = self
            .messages
            .get_mut(message_id)
            .ok_or_else(|| MqError::MessageNotFound(message_id.to_string()))?;
        if msg.status == MessageStatus::Acknowledged {
            return Err(MqError::AlreadyAcknowledged(message_id.to_string()));
        }
        msg.status = MessageStatus::Acknowledged;
        self.stats.acknowledged += 1;
        self.stats.current_size = self.stats.current_size.saturating_sub(1);
        Ok(())
    }

    // ── Negative Acknowledge (retry or dead-letter) ──────────

    /// Negatively acknowledge — either re-enqueue or dead-letter.
    pub fn nack(&mut self, message_id: &str) -> Result<bool, MqError> {
        let msg = self
            .messages
            .get_mut(message_id)
            .ok_or_else(|| MqError::MessageNotFound(message_id.to_string()))?;
        if msg.attempt >= msg.max_attempts {
            // Dead-letter.
            msg.status = MessageStatus::DeadLettered;
            self.dead_letters.push_back(msg.clone());
            self.stats.dead_lettered += 1;
            self.stats.current_size = self.stats.current_size.saturating_sub(1);
            Ok(false)
        } else {
            // Re-enqueue.
            msg.status = MessageStatus::Pending;
            msg.consumer_group = None;
            msg.consumer_id = None;
            let priority = msg.priority;
            let id = msg.id.clone();
            self.pending
                .entry(priority)
                .or_insert_with(VecDeque::new)
                .push_back(id);
            Ok(true)
        }
    }

    // ── Dead Letter Queue ────────────────────────────────────

    /// Get all dead-lettered messages.
    pub fn dead_letters(&self) -> &VecDeque<Message> {
        &self.dead_letters
    }

    /// Pop from the dead letter queue.
    pub fn pop_dead_letter(&mut self) -> Option<Message> {
        self.dead_letters.pop_front()
    }

    /// Drain the dead letter queue.
    pub fn drain_dead_letters(&mut self) -> Vec<Message> {
        self.dead_letters.drain(..).collect()
    }

    // ── Consumer Groups ──────────────────────────────────────

    /// Create a consumer group.
    pub fn create_group(&mut self, name: impl Into<String>) -> String {
        let name = name.into();
        self.groups
            .entry(name.clone())
            .or_insert_with(|| ConsumerGroup::new(name.clone()));
        name
    }

    /// Add a consumer to a group.
    pub fn add_consumer_to_group(
        &mut self,
        group: &str,
        consumer_id: impl Into<String>,
    ) -> Result<(), MqError> {
        let g = self
            .groups
            .get_mut(group)
            .ok_or_else(|| MqError::GroupNotFound(group.to_string()))?;
        g.add_consumer(consumer_id);
        Ok(())
    }

    /// Remove a consumer from a group.
    pub fn remove_consumer_from_group(
        &mut self,
        group: &str,
        consumer_id: &str,
    ) -> Result<bool, MqError> {
        let g = self
            .groups
            .get_mut(group)
            .ok_or_else(|| MqError::GroupNotFound(group.to_string()))?;
        Ok(g.remove_consumer(consumer_id))
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get a message by ID.
    pub fn get_message(&self, id: &str) -> Option<&Message> {
        self.messages.get(id)
    }

    /// Get queue statistics.
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    /// Count of messages in a given status.
    pub fn count_by_status(&self, status: MessageStatus) -> usize {
        self.messages.values().filter(|m| m.status == status).count()
    }

    /// Number of pending messages across all priorities.
    pub fn pending_count(&self) -> usize {
        self.count_by_status(MessageStatus::Pending)
            + self.count_by_status(MessageStatus::Delayed)
    }

    /// Total messages tracked (all statuses).
    pub fn total_messages(&self) -> usize {
        self.messages.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_queue() -> MessageQueue {
        MessageQueue::new("test-q")
    }

    #[test]
    fn test_enqueue_dequeue_fifo() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "hello")).unwrap();
        q.enqueue(Message::new("m2", "world")).unwrap();
        let m = q.dequeue().unwrap();
        assert_eq!(m.id, "m1");
        let m = q.dequeue().unwrap();
        assert_eq!(m.id, "m2");
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = make_queue();
        q.enqueue(Message::new("low", "a").with_priority(Priority::LOW)).unwrap();
        q.enqueue(Message::new("high", "b").with_priority(Priority::HIGH)).unwrap();
        q.enqueue(Message::new("crit", "c").with_priority(Priority::CRITICAL)).unwrap();
        assert_eq!(q.dequeue().unwrap().id, "crit");
        assert_eq!(q.dequeue().unwrap().id, "high");
        assert_eq!(q.dequeue().unwrap().id, "low");
    }

    #[test]
    fn test_acknowledge() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "data")).unwrap();
        let m = q.dequeue().unwrap();
        assert_eq!(m.status, MessageStatus::Delivered);
        q.acknowledge("m1").unwrap();
        assert_eq!(
            q.get_message("m1").unwrap().status,
            MessageStatus::Acknowledged
        );
    }

    #[test]
    fn test_double_ack_fails() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "data")).unwrap();
        q.dequeue().unwrap();
        q.acknowledge("m1").unwrap();
        assert!(matches!(
            q.acknowledge("m1"),
            Err(MqError::AlreadyAcknowledged(_))
        ));
    }

    #[test]
    fn test_nack_retries() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "data").with_max_attempts(3)).unwrap();
        // First delivery.
        let _m = q.dequeue().unwrap();
        let requeued = q.nack("m1").unwrap();
        assert!(requeued);
        // Second delivery.
        let m2 = q.dequeue().unwrap();
        assert_eq!(m2.attempt, 2);
    }

    #[test]
    fn test_nack_dead_letters_after_max_attempts() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "payload").with_max_attempts(2)).unwrap();
        // Attempt 1.
        q.dequeue().unwrap();
        q.nack("m1").unwrap();
        // Attempt 2.
        q.dequeue().unwrap();
        let requeued = q.nack("m1").unwrap();
        assert!(!requeued);
        assert_eq!(q.dead_letters().len(), 1);
        assert_eq!(q.dead_letters()[0].id, "m1");
    }

    #[test]
    fn test_message_ttl_expiry() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "data").with_ttl(100)).unwrap();
        q.advance_time(200);
        let expired = q.expire_messages();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], "m1");
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_delayed_message() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "delayed").with_delay(500)).unwrap();
        // Not eligible yet.
        assert!(q.dequeue().is_none());
        q.advance_time(500);
        let m = q.dequeue().unwrap();
        assert_eq!(m.id, "m1");
    }

    #[test]
    fn test_queue_size_limit() {
        let mut q = MessageQueue::new("limited").with_max_size(2);
        q.enqueue(Message::new("m1", "a")).unwrap();
        q.enqueue(Message::new("m2", "b")).unwrap();
        let err = q.enqueue(Message::new("m3", "c")).unwrap_err();
        assert!(matches!(err, MqError::QueueFull { limit: 2, .. }));
    }

    #[test]
    fn test_duplicate_message_rejected() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "first")).unwrap();
        let err = q.enqueue(Message::new("m1", "second")).unwrap_err();
        assert!(matches!(err, MqError::DuplicateMessage(_)));
    }

    #[test]
    fn test_consumer_group_round_robin() {
        let mut q = make_queue();
        q.create_group("grp1");
        q.add_consumer_to_group("grp1", "c1").unwrap();
        q.add_consumer_to_group("grp1", "c2").unwrap();
        q.enqueue(Message::new("m1", "a")).unwrap();
        q.enqueue(Message::new("m2", "b")).unwrap();
        let m1 = q.dequeue_for_group("grp1").unwrap().unwrap();
        let m2 = q.dequeue_for_group("grp1").unwrap().unwrap();
        assert_eq!(m1.consumer_id.as_deref(), Some("c1"));
        assert_eq!(m2.consumer_id.as_deref(), Some("c2"));
    }

    #[test]
    fn test_dequeue_for_nonexistent_group() {
        let mut q = make_queue();
        assert!(matches!(
            q.dequeue_for_group("nope"),
            Err(MqError::GroupNotFound(_))
        ));
    }

    #[test]
    fn test_remove_consumer_from_group() {
        let mut q = make_queue();
        q.create_group("grp");
        q.add_consumer_to_group("grp", "c1").unwrap();
        q.add_consumer_to_group("grp", "c2").unwrap();
        let removed = q.remove_consumer_from_group("grp", "c1").unwrap();
        assert!(removed);
        let not_found = q.remove_consumer_from_group("grp", "c1").unwrap();
        assert!(!not_found);
    }

    #[test]
    fn test_queue_stats() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "a")).unwrap();
        q.enqueue(Message::new("m2", "b")).unwrap();
        assert_eq!(q.stats().enqueued, 2);
        q.dequeue().unwrap();
        assert_eq!(q.stats().delivered, 1);
        q.acknowledge("m1").unwrap();
        assert_eq!(q.stats().acknowledged, 1);
    }

    #[test]
    fn test_message_headers() {
        let msg = Message::new("m1", "data")
            .with_header("content-type", "application/json")
            .with_header("source", "api");
        assert_eq!(msg.headers.get("content-type").unwrap(), "application/json");
        assert_eq!(msg.headers.get("source").unwrap(), "api");
    }

    #[test]
    fn test_pop_dead_letter() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "payload").with_max_attempts(1)).unwrap();
        q.dequeue().unwrap();
        q.nack("m1").unwrap();
        let dl = q.pop_dead_letter().unwrap();
        assert_eq!(dl.id, "m1");
        assert!(q.pop_dead_letter().is_none());
    }

    #[test]
    fn test_drain_dead_letters() {
        let mut q = make_queue();
        for i in 0..3 {
            let id = format!("m{i}");
            q.enqueue(Message::new(id, "p").with_max_attempts(1)).unwrap();
            let msg = q.dequeue().unwrap();
            q.nack(&msg.id).unwrap();
        }
        let all = q.drain_dead_letters();
        assert_eq!(all.len(), 3);
        assert!(q.dead_letters().is_empty());
    }

    #[test]
    fn test_count_by_status() {
        let mut q = make_queue();
        q.enqueue(Message::new("m1", "a")).unwrap();
        q.enqueue(Message::new("m2", "b")).unwrap();
        assert_eq!(q.count_by_status(MessageStatus::Pending), 2);
        q.dequeue().unwrap();
        assert_eq!(q.count_by_status(MessageStatus::Delivered), 1);
        assert_eq!(q.count_by_status(MessageStatus::Pending), 1);
    }

    #[test]
    fn test_delayed_not_expired_before_eligible() {
        let mut q = make_queue();
        q.enqueue(
            Message::new("m1", "data")
                .with_delay(200)
                .with_ttl(500),
        )
        .unwrap();
        q.advance_time(100);
        // Not yet eligible, should not dequeue.
        assert!(q.dequeue().is_none());
        q.advance_time(150); // 250 ms total — eligible, not expired.
        let m = q.dequeue().unwrap();
        assert_eq!(m.id, "m1");
    }

    #[test]
    fn test_same_priority_maintains_fifo() {
        let mut q = make_queue();
        for i in 0..5 {
            q.enqueue(
                Message::new(format!("m{i}"), format!("payload-{i}"))
                    .with_priority(Priority::NORMAL),
            )
            .unwrap();
        }
        for i in 0..5 {
            let m = q.dequeue().unwrap();
            assert_eq!(m.id, format!("m{i}"));
        }
    }

    #[test]
    fn test_message_not_found_errors() {
        let mut q = make_queue();
        assert!(matches!(
            q.acknowledge("nonexistent"),
            Err(MqError::MessageNotFound(_))
        ));
        assert!(matches!(
            q.nack("nonexistent"),
            Err(MqError::MessageNotFound(_))
        ));
    }
}
