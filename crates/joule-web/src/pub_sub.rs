//! Publish-subscribe messaging — topics with glob pattern matching, subscriber
//! registration, message fan-out, delivery semantics, filtering, and hierarchy.
//!
//! Replaces JS pub/sub libraries (EventEmitter3, Postal.js, PubSubJS) with a
//! pure-Rust pub/sub engine supporting glob topic patterns, at-most-once and
//! at-least-once delivery simulation, and hierarchical topic trees.

use std::collections::{HashMap, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Pub/sub domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PubSubError {
    /// Subscriber not found.
    SubscriberNotFound(String),
    /// Topic not found.
    TopicNotFound(String),
    /// Duplicate subscriber ID.
    DuplicateSubscriber(String),
    /// Filter rejected the message.
    FilterRejected,
    /// Subscriber is inactive.
    SubscriberInactive(String),
}

impl std::fmt::Display for PubSubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubscriberNotFound(id) => write!(f, "subscriber not found: {id}"),
            Self::TopicNotFound(t) => write!(f, "topic not found: {t}"),
            Self::DuplicateSubscriber(id) => write!(f, "duplicate subscriber: {id}"),
            Self::FilterRejected => write!(f, "message rejected by filter"),
            Self::SubscriberInactive(id) => write!(f, "subscriber inactive: {id}"),
        }
    }
}

impl std::error::Error for PubSubError {}

// ── Delivery Guarantee ────────────────────────────────────────

/// Delivery semantics for a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeliveryGuarantee {
    /// Fire and forget — no retry.
    AtMostOnce,
    /// Simulate retry until acknowledged (up to max retries).
    AtLeastOnce,
}

impl Default for DeliveryGuarantee {
    fn default() -> Self {
        Self::AtMostOnce
    }
}

// ── Published Message ─────────────────────────────────────────

/// A message published to a topic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PubMessage {
    pub id: String,
    pub topic: String,
    pub payload: String,
    pub headers: HashMap<String, String>,
    pub timestamp_ms: u64,
}

impl PubMessage {
    pub fn new(id: impl Into<String>, topic: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            topic: topic.into(),
            payload: payload.into(),
            headers: HashMap::new(),
            timestamp_ms: 0,
        }
    }

    pub fn with_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.headers.insert(k.into(), v.into());
        self
    }
}

// ── Delivery Record ───────────────────────────────────────────

/// Tracks a delivery attempt to a subscriber.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryRecord {
    pub message_id: String,
    pub subscriber_id: String,
    pub attempts: u32,
    pub acknowledged: bool,
    pub delivered: bool,
}

// ── Filter ────────────────────────────────────────────────────

/// A message filter predicate (by header key/value match).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageFilter {
    /// Only deliver if these header key/value pairs are present.
    pub required_headers: HashMap<String, String>,
}

impl MessageFilter {
    pub fn new() -> Self {
        Self {
            required_headers: HashMap::new(),
        }
    }

    pub fn require_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.required_headers.insert(key.into(), value.into());
        self
    }

    /// Check if a message passes this filter.
    pub fn matches(&self, msg: &PubMessage) -> bool {
        self.required_headers.iter().all(|(k, v)| {
            msg.headers.get(k).map_or(false, |mv| mv == v)
        })
    }
}

impl Default for MessageFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Subscriber ────────────────────────────────────────────────

/// A subscriber to one or more topic patterns.
#[derive(Debug, Clone)]
pub struct Subscriber {
    pub id: String,
    /// Glob-style topic pattern (supports `*` and `**`).
    pub pattern: String,
    pub delivery: DeliveryGuarantee,
    pub max_retries: u32,
    pub filter: Option<MessageFilter>,
    pub active: bool,
    /// Messages delivered to this subscriber.
    pub inbox: VecDeque<PubMessage>,
    /// Delivery records.
    pub deliveries: Vec<DeliveryRecord>,
}

impl Subscriber {
    pub fn new(id: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            pattern: pattern.into(),
            delivery: DeliveryGuarantee::AtMostOnce,
            max_retries: 3,
            filter: None,
            active: true,
            inbox: VecDeque::new(),
            deliveries: Vec::new(),
        }
    }

    pub fn with_delivery(mut self, guarantee: DeliveryGuarantee) -> Self {
        self.delivery = guarantee;
        self
    }

    pub fn with_filter(mut self, filter: MessageFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }
}

// ── Glob Matching ─────────────────────────────────────────────

/// Match a topic against a glob-style pattern.
///
/// Rules:
/// - `*` matches exactly one level (segment between `/`).
/// - `**` matches zero or more levels.
/// - Literal segments must match exactly.
fn topic_matches(pattern: &str, topic: &str) -> bool {
    let pat_parts: Vec<&str> = pattern.split('/').collect();
    let top_parts: Vec<&str> = topic.split('/').collect();
    glob_match_parts(&pat_parts, &top_parts)
}

fn glob_match_parts(pattern: &[&str], topic: &[&str]) -> bool {
    if pattern.is_empty() && topic.is_empty() {
        return true;
    }
    if pattern.is_empty() {
        return false;
    }
    if pattern[0] == "**" {
        // `**` can match zero or more segments.
        for i in 0..=topic.len() {
            if glob_match_parts(&pattern[1..], &topic[i..]) {
                return true;
            }
        }
        return false;
    }
    if topic.is_empty() {
        return false;
    }
    if pattern[0] == "*" || pattern[0] == topic[0] {
        return glob_match_parts(&pattern[1..], &topic[1..]);
    }
    false
}

// ── Topic Stats ───────────────────────────────────────────────

/// Per-topic statistics.
#[derive(Debug, Clone, Default)]
pub struct TopicStats {
    pub published: u64,
    pub delivered: u64,
    pub filtered: u64,
}

// ── PubSub Engine ─────────────────────────────────────────────

/// The publish-subscribe engine.
#[derive(Debug)]
pub struct PubSub {
    subscribers: HashMap<String, Subscriber>,
    /// History of published messages (for replay / audit).
    history: Vec<PubMessage>,
    /// Per-topic stats.
    topic_stats: HashMap<String, TopicStats>,
    /// Simulated clock.
    clock_ms: u64,
    next_msg_id: u64,
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}

impl PubSub {
    pub fn new() -> Self {
        Self {
            subscribers: HashMap::new(),
            history: Vec::new(),
            topic_stats: HashMap::new(),
            clock_ms: 0,
            next_msg_id: 0,
        }
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    // ── Subscribe ─────────────────────────────────────────────

    /// Register a subscriber.
    pub fn subscribe(&mut self, sub: Subscriber) -> Result<(), PubSubError> {
        if self.subscribers.contains_key(&sub.id) {
            return Err(PubSubError::DuplicateSubscriber(sub.id));
        }
        self.subscribers.insert(sub.id.clone(), sub);
        Ok(())
    }

    /// Unsubscribe by ID.
    pub fn unsubscribe(&mut self, id: &str) -> Result<(), PubSubError> {
        self.subscribers
            .remove(id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(id.to_string()))?;
        Ok(())
    }

    /// Deactivate a subscriber (stop delivering but keep state).
    pub fn deactivate(&mut self, id: &str) -> Result<(), PubSubError> {
        let sub = self
            .subscribers
            .get_mut(id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(id.to_string()))?;
        sub.active = false;
        Ok(())
    }

    /// Reactivate a subscriber.
    pub fn activate(&mut self, id: &str) -> Result<(), PubSubError> {
        let sub = self
            .subscribers
            .get_mut(id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(id.to_string()))?;
        sub.active = true;
        Ok(())
    }

    // ── Publish ───────────────────────────────────────────────

    /// Publish a message to a topic. Returns list of subscriber IDs that received it.
    pub fn publish(&mut self, topic: impl Into<String>, payload: impl Into<String>) -> Vec<String> {
        let topic = topic.into();
        let payload = payload.into();
        let msg_id = format!("msg-{}", self.next_msg_id);
        self.next_msg_id += 1;
        let msg = PubMessage {
            id: msg_id,
            topic: topic.clone(),
            payload,
            headers: HashMap::new(),
            timestamp_ms: self.clock_ms,
        };
        self.publish_message(msg)
    }

    /// Publish a pre-built message. Returns list of subscriber IDs that received it.
    pub fn publish_message(&mut self, mut msg: PubMessage) -> Vec<String> {
        msg.timestamp_ms = self.clock_ms;
        let topic = msg.topic.clone();
        self.history.push(msg.clone());

        let stats = self.topic_stats.entry(topic.clone()).or_default();
        stats.published += 1;

        let mut delivered_to = Vec::new();

        // Collect matching subscriber IDs first to avoid borrow issues.
        let matching_ids: Vec<String> = self
            .subscribers
            .values()
            .filter(|s| s.active && topic_matches(&s.pattern, &topic))
            .map(|s| s.id.clone())
            .collect();

        for sub_id in matching_ids {
            let sub = self.subscribers.get_mut(&sub_id).unwrap();

            // Apply filter.
            if let Some(filter) = &sub.filter {
                if !filter.matches(&msg) {
                    let tstats = self.topic_stats.entry(topic.clone()).or_default();
                    tstats.filtered += 1;
                    continue;
                }
            }

            // Deliver.
            let record = DeliveryRecord {
                message_id: msg.id.clone(),
                subscriber_id: sub.id.clone(),
                attempts: 1,
                acknowledged: sub.delivery == DeliveryGuarantee::AtMostOnce,
                delivered: true,
            };
            sub.deliveries.push(record);
            sub.inbox.push_back(msg.clone());
            delivered_to.push(sub.id.clone());

            let tstats = self.topic_stats.entry(topic.clone()).or_default();
            tstats.delivered += 1;
        }

        delivered_to
    }

    /// Publish with headers.
    pub fn publish_with_headers(
        &mut self,
        topic: impl Into<String>,
        payload: impl Into<String>,
        headers: HashMap<String, String>,
    ) -> Vec<String> {
        let topic = topic.into();
        let payload = payload.into();
        let msg_id = format!("msg-{}", self.next_msg_id);
        self.next_msg_id += 1;
        let msg = PubMessage {
            id: msg_id,
            topic,
            payload,
            headers,
            timestamp_ms: self.clock_ms,
        };
        self.publish_message(msg)
    }

    // ── Acknowledge (for at-least-once) ──────────────────────

    /// Acknowledge a message for an at-least-once subscriber.
    pub fn acknowledge(
        &mut self,
        subscriber_id: &str,
        message_id: &str,
    ) -> Result<(), PubSubError> {
        let sub = self
            .subscribers
            .get_mut(subscriber_id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(subscriber_id.to_string()))?;
        for record in &mut sub.deliveries {
            if record.message_id == message_id {
                record.acknowledged = true;
                return Ok(());
            }
        }
        Ok(())
    }

    /// Get unacknowledged deliveries for a subscriber.
    pub fn unacknowledged(&self, subscriber_id: &str) -> Result<Vec<&DeliveryRecord>, PubSubError> {
        let sub = self
            .subscribers
            .get(subscriber_id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(subscriber_id.to_string()))?;
        Ok(sub.deliveries.iter().filter(|d| !d.acknowledged).collect())
    }

    /// Retry unacknowledged deliveries for a subscriber (at-least-once).
    /// Returns number of retried deliveries.
    pub fn retry_unacknowledged(&mut self, subscriber_id: &str) -> Result<u32, PubSubError> {
        let sub = self
            .subscribers
            .get_mut(subscriber_id)
            .ok_or_else(|| PubSubError::SubscriberNotFound(subscriber_id.to_string()))?;
        let mut retried = 0u32;
        for record in &mut sub.deliveries {
            if !record.acknowledged && record.attempts < sub.max_retries {
                record.attempts += 1;
                retried += 1;
            }
        }
        Ok(retried)
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get a subscriber by ID.
    pub fn get_subscriber(&self, id: &str) -> Option<&Subscriber> {
        self.subscribers.get(id)
    }

    /// Get the subscriber's inbox.
    pub fn inbox(&self, id: &str) -> Option<&VecDeque<PubMessage>> {
        self.subscribers.get(id).map(|s| &s.inbox)
    }

    /// History of all published messages.
    pub fn history(&self) -> &[PubMessage] {
        &self.history
    }

    /// Per-topic stats.
    pub fn topic_stats(&self, topic: &str) -> Option<&TopicStats> {
        self.topic_stats.get(topic)
    }

    /// Number of active subscribers.
    pub fn active_subscriber_count(&self) -> usize {
        self.subscribers.values().filter(|s| s.active).count()
    }

    /// Total subscriber count.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// List topics that have been published to.
    pub fn topics(&self) -> Vec<String> {
        let mut topics: Vec<String> = self.topic_stats.keys().cloned().collect();
        topics.sort();
        topics
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_publish_subscribe() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "events/user")).unwrap();
        let delivered = ps.publish("events/user", "hello");
        assert_eq!(delivered, vec!["s1"]);
        assert_eq!(ps.inbox("s1").unwrap().len(), 1);
    }

    #[test]
    fn test_glob_single_wildcard() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "events/*")).unwrap();
        let d1 = ps.publish("events/user", "a");
        let d2 = ps.publish("events/order", "b");
        let d3 = ps.publish("events/user/login", "c");
        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 1);
        assert_eq!(d3.len(), 0); // single * doesn't match nested
    }

    #[test]
    fn test_glob_double_wildcard() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "events/**")).unwrap();
        let d1 = ps.publish("events/user", "a");
        let d2 = ps.publish("events/user/login", "b");
        let d3 = ps.publish("events/order/placed/123", "c");
        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 1);
        assert_eq!(d3.len(), 1);
    }

    #[test]
    fn test_exact_topic_no_match() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "orders/placed")).unwrap();
        let d = ps.publish("orders/shipped", "x");
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_fan_out() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "news")).unwrap();
        ps.subscribe(Subscriber::new("s2", "news")).unwrap();
        ps.subscribe(Subscriber::new("s3", "news")).unwrap();
        let delivered = ps.publish("news", "breaking");
        assert_eq!(delivered.len(), 3);
    }

    #[test]
    fn test_message_filter() {
        let mut ps = PubSub::new();
        let filter = MessageFilter::new().require_header("priority", "high");
        ps.subscribe(Subscriber::new("s1", "alerts").with_filter(filter)).unwrap();

        // Without matching header — not delivered.
        let d1 = ps.publish("alerts", "low priority");
        assert_eq!(d1.len(), 0);

        // With matching header — delivered.
        let mut headers = HashMap::new();
        headers.insert("priority".to_string(), "high".to_string());
        let d2 = ps.publish_with_headers("alerts", "urgent", headers);
        assert_eq!(d2.len(), 1);
    }

    #[test]
    fn test_unsubscribe() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "topic")).unwrap();
        ps.unsubscribe("s1").unwrap();
        let d = ps.publish("topic", "gone");
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_unsubscribe_not_found() {
        let mut ps = PubSub::new();
        assert!(matches!(
            ps.unsubscribe("nope"),
            Err(PubSubError::SubscriberNotFound(_))
        ));
    }

    #[test]
    fn test_duplicate_subscriber() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "topic")).unwrap();
        assert!(matches!(
            ps.subscribe(Subscriber::new("s1", "other")),
            Err(PubSubError::DuplicateSubscriber(_))
        ));
    }

    #[test]
    fn test_deactivate_activate() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "topic")).unwrap();
        ps.deactivate("s1").unwrap();
        let d1 = ps.publish("topic", "hidden");
        assert_eq!(d1.len(), 0);

        ps.activate("s1").unwrap();
        let d2 = ps.publish("topic", "visible");
        assert_eq!(d2.len(), 1);
    }

    #[test]
    fn test_at_least_once_delivery() {
        let mut ps = PubSub::new();
        ps.subscribe(
            Subscriber::new("s1", "topic")
                .with_delivery(DeliveryGuarantee::AtLeastOnce)
                .with_max_retries(3),
        )
        .unwrap();
        ps.publish("topic", "important");

        // Should have unacknowledged record.
        let unacked = ps.unacknowledged("s1").unwrap();
        assert_eq!(unacked.len(), 1);

        // Acknowledge.
        let msg_id = unacked[0].message_id.clone();
        ps.acknowledge("s1", &msg_id).unwrap();
        let unacked = ps.unacknowledged("s1").unwrap();
        assert_eq!(unacked.len(), 0);
    }

    #[test]
    fn test_retry_unacknowledged() {
        let mut ps = PubSub::new();
        ps.subscribe(
            Subscriber::new("s1", "topic")
                .with_delivery(DeliveryGuarantee::AtLeastOnce)
                .with_max_retries(5),
        )
        .unwrap();
        ps.publish("topic", "msg");
        let retried = ps.retry_unacknowledged("s1").unwrap();
        assert_eq!(retried, 1);

        // Check attempts incremented.
        let sub = ps.get_subscriber("s1").unwrap();
        assert_eq!(sub.deliveries[0].attempts, 2);
    }

    #[test]
    fn test_topic_stats() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "t")).unwrap();
        ps.publish("t", "a");
        ps.publish("t", "b");
        let stats = ps.topic_stats("t").unwrap();
        assert_eq!(stats.published, 2);
        assert_eq!(stats.delivered, 2);
    }

    #[test]
    fn test_history() {
        let mut ps = PubSub::new();
        ps.publish("a", "1");
        ps.publish("b", "2");
        assert_eq!(ps.history().len(), 2);
        assert_eq!(ps.history()[0].topic, "a");
        assert_eq!(ps.history()[1].topic, "b");
    }

    #[test]
    fn test_topic_hierarchy_matching() {
        assert!(topic_matches("a/b/c", "a/b/c"));
        assert!(!topic_matches("a/b/c", "a/b/d"));
        assert!(topic_matches("a/*/c", "a/x/c"));
        assert!(!topic_matches("a/*/c", "a/x/y/c"));
        assert!(topic_matches("a/**/c", "a/x/y/c"));
        assert!(topic_matches("a/**", "a/b/c/d"));
        assert!(topic_matches("**", "anything/at/all"));
    }

    #[test]
    fn test_inactive_subscriber_skipped() {
        let mut ps = PubSub::new();
        let mut sub = Subscriber::new("s1", "topic");
        sub.active = false;
        ps.subscribe(sub).unwrap();
        let d = ps.publish("topic", "msg");
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_topics_list() {
        let mut ps = PubSub::new();
        ps.publish("beta", "1");
        ps.publish("alpha", "2");
        let topics = ps.topics();
        assert_eq!(topics, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_subscriber_count() {
        let mut ps = PubSub::new();
        ps.subscribe(Subscriber::new("s1", "a")).unwrap();
        ps.subscribe(Subscriber::new("s2", "b")).unwrap();
        assert_eq!(ps.subscriber_count(), 2);
        assert_eq!(ps.active_subscriber_count(), 2);
        ps.deactivate("s1").unwrap();
        assert_eq!(ps.active_subscriber_count(), 1);
    }
}
