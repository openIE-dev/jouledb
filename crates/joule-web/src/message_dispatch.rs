//! Message dispatching and routing — envelopes, queues, TTL, priority, retry.
//!
//! Replaces RabbitMQ / NATS message routing with pure Rust.
//! Message envelopes with sender/recipient/priority, per-recipient queues,
//! TTL-based expiration, priority ordering, routing rules, delivery
//! confirmation, retry on failure, and throughput/latency statistics.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    RecipientNotFound(String),
    QueueFull(String),
    MessageExpired(u64),
    DeliveryFailed { message_id: u64, reason: String },
    RuleNotFound(String),
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RecipientNotFound(r) => write!(f, "recipient not found: {r}"),
            Self::QueueFull(r) => write!(f, "queue full for: {r}"),
            Self::MessageExpired(id) => write!(f, "message expired: {id}"),
            Self::DeliveryFailed { message_id, reason } => {
                write!(f, "delivery failed for {message_id}: {reason}")
            }
            Self::RuleNotFound(id) => write!(f, "rule not found: {id}"),
        }
    }
}

impl std::error::Error for DispatchError {}

// ── Priority ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

impl Priority {
    fn ordinal(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ordinal().cmp(&other.ordinal())
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ── Recipient ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Recipient {
    User(String),
    Channel(String),
    Broadcast,
}

impl fmt::Display for Recipient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(u) => write!(f, "user:{u}"),
            Self::Channel(c) => write!(f, "channel:{c}"),
            Self::Broadcast => write!(f, "broadcast"),
        }
    }
}

// ── DeliveryStatus ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryStatus {
    Queued,
    Delivered,
    Failed,
    Expired,
}

impl fmt::Display for DeliveryStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Delivered => write!(f, "delivered"),
            Self::Failed => write!(f, "failed"),
            Self::Expired => write!(f, "expired"),
        }
    }
}

// ── MessageEnvelope ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MessageEnvelope {
    pub id: u64,
    pub sender: String,
    pub recipient: Recipient,
    pub payload: String,
    pub priority: Priority,
    pub created_at: u64,
    pub ttl_secs: Option<u64>,
    pub status: DeliveryStatus,
    pub retry_count: u32,
    pub max_retries: u32,
    pub delivered_at: Option<u64>,
}

impl MessageEnvelope {
    pub fn new(id: u64, sender: &str, recipient: Recipient, payload: &str, priority: Priority, created_at: u64) -> Self {
        Self {
            id,
            sender: sender.to_string(),
            recipient,
            payload: payload.to_string(),
            priority,
            created_at,
            ttl_secs: None,
            status: DeliveryStatus::Queued,
            retry_count: 0,
            max_retries: 3,
            delivered_at: None,
        }
    }

    pub fn with_ttl(mut self, secs: u64) -> Self {
        self.ttl_secs = Some(secs);
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn is_expired(&self, now: u64) -> bool {
        if let Some(ttl) = self.ttl_secs {
            now.saturating_sub(self.created_at) > ttl
        } else {
            false
        }
    }

    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    pub fn latency(&self) -> Option<u64> {
        self.delivered_at.map(|d| d.saturating_sub(self.created_at))
    }
}

impl fmt::Display for MessageEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "msg#{} {} -> {} [{}] ({})",
            self.id, self.sender, self.recipient, self.priority, self.status
        )
    }
}

// ── PriorityEntry (for BinaryHeap) ─────────────────────────────

#[derive(Debug, Clone, Eq, PartialEq)]
struct PriorityEntry {
    priority: Priority,
    created_at: u64,
    message_id: u64,
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
            .then_with(|| other.created_at.cmp(&self.created_at))
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── RoutingRule ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub id: String,
    pub payload_contains: Option<String>,
    pub min_priority: Option<Priority>,
    pub redirect_to: Recipient,
}

impl RoutingRule {
    pub fn new(id: &str, redirect_to: Recipient) -> Self {
        Self {
            id: id.to_string(),
            payload_contains: None,
            min_priority: None,
            redirect_to,
        }
    }

    pub fn with_payload_filter(mut self, contains: &str) -> Self {
        self.payload_contains = Some(contains.to_string());
        self
    }

    pub fn with_min_priority(mut self, p: Priority) -> Self {
        self.min_priority = Some(p);
        self
    }

    pub fn matches(&self, envelope: &MessageEnvelope) -> bool {
        if let Some(ref pattern) = self.payload_contains {
            if !envelope.payload.contains(pattern.as_str()) {
                return false;
            }
        }
        if let Some(min_p) = self.min_priority {
            if envelope.priority < min_p {
                return false;
            }
        }
        true
    }
}

// ── DispatchStats ───────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct DispatchStats {
    pub total_dispatched: u64,
    pub total_delivered: u64,
    pub total_failed: u64,
    pub total_expired: u64,
    pub total_dropped: u64,
    pub total_latency: u64,
    delivered_count_for_avg: u64,
}

impl DispatchStats {
    pub fn avg_latency(&self) -> f64 {
        if self.delivered_count_for_avg == 0 {
            0.0
        } else {
            self.total_latency as f64 / self.delivered_count_for_avg as f64
        }
    }

    pub fn throughput(&self) -> u64 {
        self.total_delivered
    }
}

// ── MessageDispatcher ───────────────────────────────────────────

#[derive(Debug)]
pub struct MessageDispatcher {
    next_id: u64,
    messages: HashMap<u64, MessageEnvelope>,
    queues: HashMap<String, VecDeque<u64>>,
    priority_heap: BinaryHeap<PriorityEntry>,
    rules: Vec<RoutingRule>,
    max_queue_size: usize,
    stats: DispatchStats,
    delivered: Vec<u64>,
}

impl MessageDispatcher {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            next_id: 1,
            messages: HashMap::new(),
            queues: HashMap::new(),
            priority_heap: BinaryHeap::new(),
            rules: Vec::new(),
            max_queue_size,
            stats: DispatchStats::default(),
            delivered: Vec::new(),
        }
    }

    pub fn register_recipient(&mut self, id: &str) {
        self.queues.entry(id.to_string()).or_default();
    }

    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
    }

    pub fn remove_rule(&mut self, rule_id: &str) -> Result<(), DispatchError> {
        let idx = self.rules.iter().position(|r| r.id == rule_id)
            .ok_or_else(|| DispatchError::RuleNotFound(rule_id.to_string()))?;
        self.rules.remove(idx);
        Ok(())
    }

    /// Dispatch a message. Applies routing rules, enqueues to recipient.
    pub fn dispatch(
        &mut self,
        sender: &str,
        recipient: Recipient,
        payload: &str,
        priority: Priority,
        created_at: u64,
        ttl_secs: Option<u64>,
    ) -> Result<u64, DispatchError> {
        let id = self.next_id;
        self.next_id += 1;

        let mut envelope = MessageEnvelope::new(id, sender, recipient, payload, priority, created_at);
        if let Some(ttl) = ttl_secs {
            envelope = envelope.with_ttl(ttl);
        }

        // Apply routing rules
        let mut final_recipient = envelope.recipient.clone();
        for rule in &self.rules {
            if rule.matches(&envelope) {
                final_recipient = rule.redirect_to.clone();
                break;
            }
        }
        envelope.recipient = final_recipient.clone();

        // Determine queue key
        let queue_keys = self.resolve_queue_keys(&final_recipient);

        for key in &queue_keys {
            if let Some(queue) = self.queues.get(key) {
                if queue.len() >= self.max_queue_size {
                    self.stats.total_dropped += 1;
                    return Err(DispatchError::QueueFull(key.clone()));
                }
            }
        }

        self.priority_heap.push(PriorityEntry {
            priority: envelope.priority,
            created_at: envelope.created_at,
            message_id: id,
        });

        self.messages.insert(id, envelope);

        for key in queue_keys {
            self.queues.entry(key).or_default().push_back(id);
        }

        self.stats.total_dispatched += 1;
        Ok(id)
    }

    fn resolve_queue_keys(&self, recipient: &Recipient) -> Vec<String> {
        match recipient {
            Recipient::User(u) => vec![u.clone()],
            Recipient::Channel(c) => vec![c.clone()],
            Recipient::Broadcast => self.queues.keys().cloned().collect(),
        }
    }

    /// Deliver the next message from a recipient's queue.
    pub fn deliver(&mut self, recipient_id: &str, now: u64) -> Result<Option<MessageEnvelope>, DispatchError> {
        let queue = self.queues.get_mut(recipient_id)
            .ok_or_else(|| DispatchError::RecipientNotFound(recipient_id.to_string()))?;

        while let Some(msg_id) = queue.pop_front() {
            if let Some(msg) = self.messages.get_mut(&msg_id) {
                if msg.is_expired(now) {
                    msg.status = DeliveryStatus::Expired;
                    self.stats.total_expired += 1;
                    continue;
                }
                msg.status = DeliveryStatus::Delivered;
                msg.delivered_at = Some(now);
                self.stats.total_delivered += 1;
                if let Some(latency) = msg.latency() {
                    self.stats.total_latency += latency;
                    self.stats.delivered_count_for_avg += 1;
                }
                self.delivered.push(msg_id);
                return Ok(Some(msg.clone()));
            }
        }
        Ok(None)
    }

    /// Simulate delivery failure — re-enqueue if retries remain.
    pub fn fail_delivery(&mut self, message_id: u64, reason: &str) -> Result<bool, DispatchError> {
        let msg = self.messages.get_mut(&message_id)
            .ok_or_else(|| DispatchError::DeliveryFailed { message_id, reason: "not found".to_string() })?;
        msg.retry_count += 1;
        if msg.can_retry() {
            msg.status = DeliveryStatus::Queued;
            let key = match &msg.recipient {
                Recipient::User(u) => u.clone(),
                Recipient::Channel(c) => c.clone(),
                Recipient::Broadcast => return Ok(false),
            };
            self.queues.entry(key).or_default().push_back(message_id);
            Ok(true) // retried
        } else {
            msg.status = DeliveryStatus::Failed;
            self.stats.total_failed += 1;
            Ok(false) // permanent failure
        }
    }

    /// Expire old messages from all queues.
    pub fn expire_messages(&mut self, now: u64) -> usize {
        let mut expired_count = 0;
        for queue in self.queues.values_mut() {
            queue.retain(|msg_id| {
                if let Some(msg) = self.messages.get(&msg_id) {
                    if msg.is_expired(now) {
                        expired_count += 1;
                        return false;
                    }
                }
                true
            });
        }
        // We can't mutably borrow self.messages while iterating self.queues above,
        // so we mark expired messages in a second pass.
        for msg in self.messages.values_mut() {
            if msg.is_expired(now) && msg.status == DeliveryStatus::Queued {
                msg.status = DeliveryStatus::Expired;
                self.stats.total_expired += 1;
            }
        }
        expired_count
    }

    pub fn queue_depth(&self, recipient_id: &str) -> usize {
        self.queues.get(recipient_id).map_or(0, |q| q.len())
    }

    pub fn get_message(&self, id: u64) -> Option<&MessageEnvelope> {
        self.messages.get(&id)
    }

    pub fn stats(&self) -> &DispatchStats {
        &self.stats
    }

    pub fn delivered_messages(&self) -> &[u64] {
        &self.delivered
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatcher() -> MessageDispatcher {
        let mut d = MessageDispatcher::new(100);
        d.register_recipient("alice");
        d.register_recipient("bob");
        d.register_recipient("general");
        d
    }

    #[test]
    fn test_dispatch_and_deliver() {
        let mut d = dispatcher();
        let id = d.dispatch("bob", Recipient::User("alice".into()), "hello", Priority::Normal, 100, None).unwrap();
        let msg = d.deliver("alice", 101).unwrap().unwrap();
        assert_eq!(msg.id, id);
        assert_eq!(msg.status, DeliveryStatus::Delivered);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", Priority::Critical), "critical");
    }

    #[test]
    fn test_message_ttl() {
        let mut d = dispatcher();
        d.dispatch("bob", Recipient::User("alice".into()), "urgent", Priority::High, 100, Some(10)).unwrap();
        let msg = d.deliver("alice", 200);
        // Should be expired (200 - 100 = 100 > ttl 10)
        assert!(msg.unwrap().is_none());
    }

    #[test]
    fn test_broadcast() {
        let mut d = dispatcher();
        d.dispatch("admin", Recipient::Broadcast, "announcement", Priority::High, 100, None).unwrap();
        let a = d.deliver("alice", 101).unwrap();
        let b = d.deliver("bob", 101).unwrap();
        assert!(a.is_some());
        assert!(b.is_some());
    }

    #[test]
    fn test_channel_dispatch() {
        let mut d = dispatcher();
        d.dispatch("bob", Recipient::Channel("general".into()), "hi all", Priority::Normal, 100, None).unwrap();
        let msg = d.deliver("general", 101).unwrap().unwrap();
        assert_eq!(msg.payload, "hi all");
    }

    #[test]
    fn test_queue_full() {
        let mut d = MessageDispatcher::new(1);
        d.register_recipient("alice");
        d.dispatch("bob", Recipient::User("alice".into()), "m1", Priority::Normal, 100, None).unwrap();
        let err = d.dispatch("bob", Recipient::User("alice".into()), "m2", Priority::Normal, 101, None);
        assert!(err.is_err());
    }

    #[test]
    fn test_delivery_failure_retry() {
        let mut d = dispatcher();
        let id = d.dispatch("bob", Recipient::User("alice".into()), "hello", Priority::Normal, 100, None).unwrap();
        d.deliver("alice", 101).unwrap();
        let retried = d.fail_delivery(id, "timeout").unwrap();
        assert!(retried);
    }

    #[test]
    fn test_max_retries_exhausted() {
        let mut d = dispatcher();
        let id = d.dispatch("bob", Recipient::User("alice".into()), "hello", Priority::Normal, 100, None).unwrap();
        d.deliver("alice", 101).unwrap();
        for _ in 0..3 {
            d.fail_delivery(id, "timeout").unwrap();
        }
        let retried = d.fail_delivery(id, "timeout").unwrap();
        assert!(!retried);
    }

    #[test]
    fn test_routing_rule() {
        let mut d = dispatcher();
        d.register_recipient("support");
        d.add_rule(
            RoutingRule::new("r1", Recipient::Channel("support".into()))
                .with_payload_filter("help"),
        );
        d.dispatch("bob", Recipient::User("alice".into()), "I need help", Priority::Normal, 100, None).unwrap();
        // Message should be routed to support channel
        let msg = d.deliver("support", 101).unwrap();
        assert!(msg.is_some());
    }

    #[test]
    fn test_routing_rule_priority() {
        let mut d = dispatcher();
        d.register_recipient("escalation");
        d.add_rule(
            RoutingRule::new("r1", Recipient::Channel("escalation".into()))
                .with_min_priority(Priority::High),
        );
        d.dispatch("bob", Recipient::User("alice".into()), "urgent", Priority::Critical, 100, None).unwrap();
        let msg = d.deliver("escalation", 101).unwrap();
        assert!(msg.is_some());
    }

    #[test]
    fn test_remove_rule() {
        let mut d = dispatcher();
        d.add_rule(RoutingRule::new("r1", Recipient::Broadcast));
        d.remove_rule("r1").unwrap();
        assert_eq!(d.rule_count(), 0);
    }

    #[test]
    fn test_stats() {
        let mut d = dispatcher();
        d.dispatch("bob", Recipient::User("alice".into()), "m1", Priority::Normal, 100, None).unwrap();
        d.deliver("alice", 105).unwrap();
        let s = d.stats();
        assert_eq!(s.total_dispatched, 1);
        assert_eq!(s.total_delivered, 1);
        assert_eq!(s.avg_latency(), 5.0);
    }

    #[test]
    fn test_queue_depth() {
        let mut d = dispatcher();
        d.dispatch("bob", Recipient::User("alice".into()), "m1", Priority::Normal, 100, None).unwrap();
        d.dispatch("bob", Recipient::User("alice".into()), "m2", Priority::Normal, 101, None).unwrap();
        assert_eq!(d.queue_depth("alice"), 2);
    }

    #[test]
    fn test_empty_deliver() {
        let mut d = dispatcher();
        let msg = d.deliver("alice", 100).unwrap();
        assert!(msg.is_none());
    }

    #[test]
    fn test_recipient_not_found() {
        let mut d = dispatcher();
        assert!(d.deliver("ghost", 100).is_err());
    }

    #[test]
    fn test_display_envelope() {
        let env = MessageEnvelope::new(1, "bob", Recipient::User("alice".into()), "hi", Priority::Normal, 100);
        let s = format!("{env}");
        assert!(s.contains("bob"));
        assert!(s.contains("alice"));
        assert!(s.contains("normal"));
    }

    #[test]
    fn test_display_recipient() {
        assert_eq!(format!("{}", Recipient::User("alice".into())), "user:alice");
        assert_eq!(format!("{}", Recipient::Channel("gen".into())), "channel:gen");
        assert_eq!(format!("{}", Recipient::Broadcast), "broadcast");
    }

    #[test]
    fn test_envelope_latency() {
        let mut env = MessageEnvelope::new(1, "bob", Recipient::User("alice".into()), "hi", Priority::Normal, 100);
        assert!(env.latency().is_none());
        env.delivered_at = Some(110);
        assert_eq!(env.latency(), Some(10));
    }

    #[test]
    fn test_envelope_can_retry() {
        let mut env = MessageEnvelope::new(1, "bob", Recipient::User("alice".into()), "hi", Priority::Normal, 100)
            .with_max_retries(1);
        assert!(env.can_retry());
        env.retry_count = 1;
        assert!(!env.can_retry());
    }

    #[test]
    fn test_expire_messages() {
        let mut d = dispatcher();
        d.dispatch("bob", Recipient::User("alice".into()), "old", Priority::Normal, 100, Some(10)).unwrap();
        d.dispatch("bob", Recipient::User("alice".into()), "new", Priority::Normal, 200, Some(1000)).unwrap();
        let expired = d.expire_messages(200);
        assert_eq!(expired, 1);
    }
}
