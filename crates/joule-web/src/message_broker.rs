//! Message broker — exchange types (direct/fanout/topic), routing keys, queue
//! binding, in-memory persistence, consumer acknowledgment, prefetch limits,
//! and dead letter routing.
//!
//! Replaces AMQP/RabbitMQ client libraries with a pure-Rust message broker
//! engine supporting exchange-to-queue routing, consumer prefetch, and dead
//! letter exchange (DLX) patterns.

use std::collections::{HashMap, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Message broker domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerError {
    /// Exchange not found.
    ExchangeNotFound(String),
    /// Queue not found.
    QueueNotFound(String),
    /// Consumer not found.
    ConsumerNotFound(String),
    /// Message not found.
    MessageNotFound(String),
    /// Queue already exists.
    DuplicateQueue(String),
    /// Exchange already exists.
    DuplicateExchange(String),
    /// Prefetch limit reached.
    PrefetchLimitReached { consumer: String, limit: usize },
    /// No route for routing key.
    NoRoute { exchange: String, routing_key: String },
    /// Binding already exists.
    DuplicateBinding { exchange: String, queue: String },
}

impl std::fmt::Display for BrokerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExchangeNotFound(e) => write!(f, "exchange not found: {e}"),
            Self::QueueNotFound(q) => write!(f, "queue not found: {q}"),
            Self::ConsumerNotFound(c) => write!(f, "consumer not found: {c}"),
            Self::MessageNotFound(id) => write!(f, "message not found: {id}"),
            Self::DuplicateQueue(q) => write!(f, "duplicate queue: {q}"),
            Self::DuplicateExchange(e) => write!(f, "duplicate exchange: {e}"),
            Self::PrefetchLimitReached { consumer, limit } => {
                write!(f, "prefetch limit {limit} reached for consumer {consumer}")
            }
            Self::NoRoute {
                exchange,
                routing_key,
            } => write!(f, "no route from {exchange} for key {routing_key}"),
            Self::DuplicateBinding { exchange, queue } => {
                write!(f, "binding already exists: {exchange} -> {queue}")
            }
        }
    }
}

impl std::error::Error for BrokerError {}

// ── Exchange Type ─────────────────────────────────────────────

/// Type of exchange routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangeType {
    /// Routes to the queue whose binding key exactly matches the routing key.
    Direct,
    /// Routes to all bound queues regardless of routing key.
    Fanout,
    /// Routes using dot-delimited pattern matching (`*` = one word, `#` = zero or more).
    Topic,
}

// ── Exchange ──────────────────────────────────────────────────

/// An exchange that routes messages to queues.
#[derive(Debug, Clone)]
pub struct Exchange {
    pub name: String,
    pub exchange_type: ExchangeType,
    /// Bindings: (queue_name, binding_key).
    pub bindings: Vec<(String, String)>,
    pub message_count: u64,
}

impl Exchange {
    pub fn new(name: impl Into<String>, exchange_type: ExchangeType) -> Self {
        Self {
            name: name.into(),
            exchange_type,
            bindings: Vec::new(),
            message_count: 0,
        }
    }
}

// ── Broker Message ────────────────────────────────────────────

/// A message in the broker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerMessage {
    pub id: String,
    pub routing_key: String,
    pub payload: String,
    pub exchange: String,
    pub headers: HashMap<String, String>,
    pub timestamp_ms: u64,
    pub delivery_count: u32,
    pub acknowledged: bool,
    /// Which consumer holds this delivery.
    pub consumer_id: Option<String>,
}

impl BrokerMessage {
    pub fn new(
        id: impl Into<String>,
        routing_key: impl Into<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            routing_key: routing_key.into(),
            payload: payload.into(),
            exchange: String::new(),
            headers: HashMap::new(),
            timestamp_ms: 0,
            delivery_count: 0,
            acknowledged: false,
            consumer_id: None,
        }
    }

    pub fn with_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.headers.insert(k.into(), v.into());
        self
    }
}

// ── Queue ─────────────────────────────────────────────────────

/// A queue that holds messages for consumers.
#[derive(Debug, Clone)]
pub struct BrokerQueue {
    pub name: String,
    pub messages: VecDeque<BrokerMessage>,
    /// Dead letter exchange name (if configured).
    pub dead_letter_exchange: Option<String>,
    /// Dead letter routing key.
    pub dead_letter_routing_key: Option<String>,
    /// Maximum delivery attempts before dead-lettering.
    pub max_delivery_attempts: u32,
    pub total_enqueued: u64,
    pub total_acknowledged: u64,
    pub total_dead_lettered: u64,
}

impl BrokerQueue {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            messages: VecDeque::new(),
            dead_letter_exchange: None,
            dead_letter_routing_key: None,
            max_delivery_attempts: 3,
            total_enqueued: 0,
            total_acknowledged: 0,
            total_dead_lettered: 0,
        }
    }

    pub fn with_dead_letter(
        mut self,
        exchange: impl Into<String>,
        routing_key: impl Into<String>,
    ) -> Self {
        self.dead_letter_exchange = Some(exchange.into());
        self.dead_letter_routing_key = Some(routing_key.into());
        self
    }

    pub fn with_max_delivery_attempts(mut self, n: u32) -> Self {
        self.max_delivery_attempts = n;
        self
    }
}

// ── Consumer ──────────────────────────────────────────────────

/// A consumer attached to a queue.
#[derive(Debug, Clone)]
pub struct Consumer {
    pub id: String,
    pub queue_name: String,
    pub prefetch_limit: usize,
    /// Messages currently held (unacknowledged).
    pub unacked: Vec<String>,
    pub total_consumed: u64,
}

impl Consumer {
    pub fn new(id: impl Into<String>, queue_name: impl Into<String>, prefetch: usize) -> Self {
        Self {
            id: id.into(),
            queue_name: queue_name.into(),
            prefetch_limit: prefetch,
            unacked: Vec::new(),
            total_consumed: 0,
        }
    }
}

// ── Topic Pattern Matching ────────────────────────────────────

/// Match a routing key against a topic binding pattern.
/// `*` matches exactly one word, `#` matches zero or more words.
fn topic_matches(pattern: &str, routing_key: &str) -> bool {
    let pat_parts: Vec<&str> = pattern.split('.').collect();
    let key_parts: Vec<&str> = routing_key.split('.').collect();
    topic_match_parts(&pat_parts, &key_parts)
}

fn topic_match_parts(pattern: &[&str], key: &[&str]) -> bool {
    if pattern.is_empty() && key.is_empty() {
        return true;
    }
    if pattern.is_empty() {
        return false;
    }
    if pattern[0] == "#" {
        // `#` matches zero or more words.
        for i in 0..=key.len() {
            if topic_match_parts(&pattern[1..], &key[i..]) {
                return true;
            }
        }
        return false;
    }
    if key.is_empty() {
        return false;
    }
    if pattern[0] == "*" || pattern[0] == key[0] {
        return topic_match_parts(&pattern[1..], &key[1..]);
    }
    false
}

// ── Message Broker ────────────────────────────────────────────

/// In-memory message broker with exchanges, queues, and consumers.
#[derive(Debug)]
pub struct MessageBroker {
    exchanges: HashMap<String, Exchange>,
    queues: HashMap<String, BrokerQueue>,
    consumers: HashMap<String, Consumer>,
    /// Message log (in-memory persistence).
    message_log: Vec<BrokerMessage>,
    /// Simulated clock.
    clock_ms: u64,
    next_msg_id: u64,
}

impl Default for MessageBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageBroker {
    pub fn new() -> Self {
        Self {
            exchanges: HashMap::new(),
            queues: HashMap::new(),
            consumers: HashMap::new(),
            message_log: Vec::new(),
            clock_ms: 0,
            next_msg_id: 1,
        }
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    // ── Exchange Management ──────────────────────────────────

    /// Declare an exchange.
    pub fn declare_exchange(
        &mut self,
        name: impl Into<String>,
        exchange_type: ExchangeType,
    ) -> Result<(), BrokerError> {
        let name = name.into();
        if self.exchanges.contains_key(&name) {
            return Err(BrokerError::DuplicateExchange(name));
        }
        self.exchanges
            .insert(name.clone(), Exchange::new(name, exchange_type));
        Ok(())
    }

    /// Delete an exchange.
    pub fn delete_exchange(&mut self, name: &str) -> Result<(), BrokerError> {
        self.exchanges
            .remove(name)
            .ok_or_else(|| BrokerError::ExchangeNotFound(name.to_string()))?;
        Ok(())
    }

    // ── Queue Management ─────────────────────────────────────

    /// Declare a queue.
    pub fn declare_queue(&mut self, queue: BrokerQueue) -> Result<(), BrokerError> {
        if self.queues.contains_key(&queue.name) {
            return Err(BrokerError::DuplicateQueue(queue.name));
        }
        self.queues.insert(queue.name.clone(), queue);
        Ok(())
    }

    /// Delete a queue.
    pub fn delete_queue(&mut self, name: &str) -> Result<(), BrokerError> {
        self.queues
            .remove(name)
            .ok_or_else(|| BrokerError::QueueNotFound(name.to_string()))?;
        // Remove consumers attached to this queue.
        self.consumers.retain(|_, c| c.queue_name != name);
        Ok(())
    }

    // ── Binding ──────────────────────────────────────────────

    /// Bind a queue to an exchange with a routing/binding key.
    pub fn bind(
        &mut self,
        exchange_name: &str,
        queue_name: &str,
        binding_key: impl Into<String>,
    ) -> Result<(), BrokerError> {
        if !self.queues.contains_key(queue_name) {
            return Err(BrokerError::QueueNotFound(queue_name.to_string()));
        }
        let exchange = self
            .exchanges
            .get_mut(exchange_name)
            .ok_or_else(|| BrokerError::ExchangeNotFound(exchange_name.to_string()))?;
        let bk = binding_key.into();
        let already = exchange
            .bindings
            .iter()
            .any(|(q, k)| q == queue_name && k == &bk);
        if already {
            return Err(BrokerError::DuplicateBinding {
                exchange: exchange_name.to_string(),
                queue: queue_name.to_string(),
            });
        }
        exchange.bindings.push((queue_name.to_string(), bk));
        Ok(())
    }

    /// Unbind a queue from an exchange.
    pub fn unbind(
        &mut self,
        exchange_name: &str,
        queue_name: &str,
    ) -> Result<(), BrokerError> {
        let exchange = self
            .exchanges
            .get_mut(exchange_name)
            .ok_or_else(|| BrokerError::ExchangeNotFound(exchange_name.to_string()))?;
        let before = exchange.bindings.len();
        exchange.bindings.retain(|(q, _)| q != queue_name);
        if exchange.bindings.len() == before {
            return Err(BrokerError::QueueNotFound(queue_name.to_string()));
        }
        Ok(())
    }

    // ── Consumer Management ──────────────────────────────────

    /// Register a consumer for a queue.
    pub fn register_consumer(&mut self, consumer: Consumer) -> Result<(), BrokerError> {
        if !self.queues.contains_key(&consumer.queue_name) {
            return Err(BrokerError::QueueNotFound(consumer.queue_name.clone()));
        }
        self.consumers.insert(consumer.id.clone(), consumer);
        Ok(())
    }

    /// Unregister a consumer.
    pub fn unregister_consumer(&mut self, consumer_id: &str) -> Result<(), BrokerError> {
        self.consumers
            .remove(consumer_id)
            .ok_or_else(|| BrokerError::ConsumerNotFound(consumer_id.to_string()))?;
        Ok(())
    }

    // ── Publish ──────────────────────────────────────────────

    /// Publish a message to an exchange.
    pub fn publish(
        &mut self,
        exchange_name: &str,
        routing_key: impl Into<String>,
        payload: impl Into<String>,
    ) -> Result<Vec<String>, BrokerError> {
        let routing_key = routing_key.into();
        let payload = payload.into();
        let exchange = self
            .exchanges
            .get(exchange_name)
            .ok_or_else(|| BrokerError::ExchangeNotFound(exchange_name.to_string()))?;
        let exchange_type = exchange.exchange_type;
        let bindings = exchange.bindings.clone();

        // Find matching queues.
        let target_queues: Vec<String> = match exchange_type {
            ExchangeType::Direct => bindings
                .iter()
                .filter(|(_, bk)| bk == &routing_key)
                .map(|(q, _)| q.clone())
                .collect(),
            ExchangeType::Fanout => bindings.iter().map(|(q, _)| q.clone()).collect(),
            ExchangeType::Topic => bindings
                .iter()
                .filter(|(_, bk)| topic_matches(bk, &routing_key))
                .map(|(q, _)| q.clone())
                .collect(),
        };

        // Create message for each target queue.
        let mut delivered_to = Vec::new();
        for queue_name in &target_queues {
            let msg_id = format!("msg-{}", self.next_msg_id);
            self.next_msg_id += 1;
            let msg = BrokerMessage {
                id: msg_id.clone(),
                routing_key: routing_key.clone(),
                payload: payload.clone(),
                exchange: exchange_name.to_string(),
                headers: HashMap::new(),
                timestamp_ms: self.clock_ms,
                delivery_count: 0,
                acknowledged: false,
                consumer_id: None,
            };
            self.message_log.push(msg.clone());
            if let Some(queue) = self.queues.get_mut(queue_name) {
                queue.messages.push_back(msg);
                queue.total_enqueued += 1;
                delivered_to.push(queue_name.clone());
            }
        }

        // Update exchange counter.
        if let Some(ex) = self.exchanges.get_mut(exchange_name) {
            ex.message_count += 1;
        }

        Ok(delivered_to)
    }

    // ── Consume ──────────────────────────────────────────────

    /// Consume a message for a consumer (respects prefetch).
    pub fn consume(&mut self, consumer_id: &str) -> Result<Option<BrokerMessage>, BrokerError> {
        let consumer = self
            .consumers
            .get(consumer_id)
            .ok_or_else(|| BrokerError::ConsumerNotFound(consumer_id.to_string()))?;
        if consumer.unacked.len() >= consumer.prefetch_limit {
            return Err(BrokerError::PrefetchLimitReached {
                consumer: consumer_id.to_string(),
                limit: consumer.prefetch_limit,
            });
        }
        let queue_name = consumer.queue_name.clone();
        let queue = self
            .queues
            .get_mut(&queue_name)
            .ok_or_else(|| BrokerError::QueueNotFound(queue_name.clone()))?;
        if let Some(mut msg) = queue.messages.pop_front() {
            msg.delivery_count += 1;
            msg.consumer_id = Some(consumer_id.to_string());
            let msg_id = msg.id.clone();
            let consumer = self.consumers.get_mut(consumer_id).unwrap();
            consumer.unacked.push(msg_id);
            consumer.total_consumed += 1;
            Ok(Some(msg))
        } else {
            Ok(None)
        }
    }

    // ── Acknowledge ──────────────────────────────────────────

    /// Acknowledge a message.
    pub fn acknowledge(
        &mut self,
        consumer_id: &str,
        message_id: &str,
    ) -> Result<(), BrokerError> {
        let consumer = self
            .consumers
            .get_mut(consumer_id)
            .ok_or_else(|| BrokerError::ConsumerNotFound(consumer_id.to_string()))?;
        let pos = consumer
            .unacked
            .iter()
            .position(|id| id == message_id)
            .ok_or_else(|| BrokerError::MessageNotFound(message_id.to_string()))?;
        consumer.unacked.remove(pos);
        let queue_name = consumer.queue_name.clone();
        if let Some(queue) = self.queues.get_mut(&queue_name) {
            queue.total_acknowledged += 1;
        }
        Ok(())
    }

    /// Negative acknowledge — requeue or dead-letter.
    pub fn nack(
        &mut self,
        consumer_id: &str,
        message_id: &str,
        mut message: BrokerMessage,
    ) -> Result<bool, BrokerError> {
        let consumer = self
            .consumers
            .get_mut(consumer_id)
            .ok_or_else(|| BrokerError::ConsumerNotFound(consumer_id.to_string()))?;
        consumer.unacked.retain(|id| id != message_id);
        let queue_name = consumer.queue_name.clone();
        let queue = self
            .queues
            .get(&queue_name)
            .ok_or_else(|| BrokerError::QueueNotFound(queue_name.clone()))?;
        let max_attempts = queue.max_delivery_attempts;
        let dlx = queue.dead_letter_exchange.clone();
        let dlrk = queue.dead_letter_routing_key.clone();

        if message.delivery_count >= max_attempts {
            // Dead-letter.
            if let (Some(dlx_name), Some(dlrk_val)) = (dlx, dlrk) {
                // Route to dead letter exchange.
                let _ = self.publish(&dlx_name, dlrk_val, message.payload.clone());
            }
            if let Some(queue) = self.queues.get_mut(&queue_name) {
                queue.total_dead_lettered += 1;
            }
            Ok(false)
        } else {
            // Requeue.
            message.consumer_id = None;
            if let Some(queue) = self.queues.get_mut(&queue_name) {
                queue.messages.push_back(message);
            }
            Ok(true)
        }
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get exchange by name.
    pub fn get_exchange(&self, name: &str) -> Option<&Exchange> {
        self.exchanges.get(name)
    }

    /// Get queue by name.
    pub fn get_queue(&self, name: &str) -> Option<&BrokerQueue> {
        self.queues.get(name)
    }

    /// Get consumer by ID.
    pub fn get_consumer(&self, id: &str) -> Option<&Consumer> {
        self.consumers.get(id)
    }

    /// Message log.
    pub fn message_log(&self) -> &[BrokerMessage] {
        &self.message_log
    }

    /// Queue depth (pending messages).
    pub fn queue_depth(&self, queue_name: &str) -> Option<usize> {
        self.queues.get(queue_name).map(|q| q.messages.len())
    }

    /// Total exchanges.
    pub fn exchange_count(&self) -> usize {
        self.exchanges.len()
    }

    /// Total queues.
    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    /// Total consumers.
    pub fn consumer_count(&self) -> usize {
        self.consumers.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_direct() -> MessageBroker {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("orders", ExchangeType::Direct).unwrap();
        broker.declare_queue(BrokerQueue::new("order-queue")).unwrap();
        broker.bind("orders", "order-queue", "order.created").unwrap();
        broker
    }

    #[test]
    fn test_direct_exchange_routing() {
        let mut broker = setup_direct();
        let queues = broker
            .publish("orders", "order.created", "payload")
            .unwrap();
        assert_eq!(queues, vec!["order-queue"]);
        assert_eq!(broker.queue_depth("order-queue"), Some(1));
    }

    #[test]
    fn test_direct_no_match() {
        let mut broker = setup_direct();
        let queues = broker
            .publish("orders", "order.shipped", "payload")
            .unwrap();
        assert!(queues.is_empty());
    }

    #[test]
    fn test_fanout_exchange() {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("events", ExchangeType::Fanout).unwrap();
        broker.declare_queue(BrokerQueue::new("q1")).unwrap();
        broker.declare_queue(BrokerQueue::new("q2")).unwrap();
        broker.bind("events", "q1", "").unwrap();
        broker.bind("events", "q2", "").unwrap();
        let queues = broker.publish("events", "anything", "data").unwrap();
        assert_eq!(queues.len(), 2);
    }

    #[test]
    fn test_topic_exchange() {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("logs", ExchangeType::Topic).unwrap();
        broker.declare_queue(BrokerQueue::new("all-logs")).unwrap();
        broker.declare_queue(BrokerQueue::new("error-logs")).unwrap();
        broker.bind("logs", "all-logs", "#").unwrap();
        broker.bind("logs", "error-logs", "*.error").unwrap();
        let q1 = broker.publish("logs", "app.error", "err").unwrap();
        assert!(q1.contains(&"all-logs".to_string()));
        assert!(q1.contains(&"error-logs".to_string()));
        let q2 = broker.publish("logs", "app.info", "info").unwrap();
        assert!(q2.contains(&"all-logs".to_string()));
        assert!(!q2.contains(&"error-logs".to_string()));
    }

    #[test]
    fn test_topic_pattern_matching() {
        assert!(topic_matches("#", "a.b.c"));
        assert!(topic_matches("a.#", "a.b.c"));
        assert!(topic_matches("a.*", "a.b"));
        assert!(!topic_matches("a.*", "a.b.c"));
        assert!(topic_matches("*.b.*", "a.b.c"));
        assert!(topic_matches("a.#.c", "a.b.c"));
    }

    #[test]
    fn test_consumer_prefetch() {
        let mut broker = setup_direct();
        broker
            .register_consumer(Consumer::new("c1", "order-queue", 1))
            .unwrap();
        broker
            .publish("orders", "order.created", "p1")
            .unwrap();
        broker
            .publish("orders", "order.created", "p2")
            .unwrap();
        broker.consume("c1").unwrap(); // takes one
        assert!(matches!(
            broker.consume("c1"),
            Err(BrokerError::PrefetchLimitReached { .. })
        ));
    }

    #[test]
    fn test_acknowledge() {
        let mut broker = setup_direct();
        broker
            .register_consumer(Consumer::new("c1", "order-queue", 10))
            .unwrap();
        broker
            .publish("orders", "order.created", "data")
            .unwrap();
        let msg = broker.consume("c1").unwrap().unwrap();
        broker.acknowledge("c1", &msg.id).unwrap();
        assert!(broker.get_consumer("c1").unwrap().unacked.is_empty());
    }

    #[test]
    fn test_nack_requeue() {
        let mut broker = setup_direct();
        broker
            .register_consumer(Consumer::new("c1", "order-queue", 10))
            .unwrap();
        broker
            .publish("orders", "order.created", "data")
            .unwrap();
        let msg = broker.consume("c1").unwrap().unwrap();
        let msg_id = msg.id.clone();
        let requeued = broker.nack("c1", &msg_id, msg).unwrap();
        assert!(requeued);
        assert_eq!(broker.queue_depth("order-queue"), Some(1));
    }

    #[test]
    fn test_dead_letter_routing() {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("main", ExchangeType::Direct).unwrap();
        broker.declare_exchange("dlx", ExchangeType::Direct).unwrap();
        broker
            .declare_queue(
                BrokerQueue::new("work")
                    .with_dead_letter("dlx", "dead")
                    .with_max_delivery_attempts(1),
            )
            .unwrap();
        broker.declare_queue(BrokerQueue::new("dead-letters")).unwrap();
        broker.bind("main", "work", "job").unwrap();
        broker.bind("dlx", "dead-letters", "dead").unwrap();
        broker
            .register_consumer(Consumer::new("c1", "work", 10))
            .unwrap();
        broker.publish("main", "job", "data").unwrap();
        let msg = broker.consume("c1").unwrap().unwrap();
        let msg_id = msg.id.clone();
        // Already at max attempts (1), should dead-letter.
        let requeued = broker.nack("c1", &msg_id, msg).unwrap();
        assert!(!requeued);
        assert_eq!(broker.queue_depth("dead-letters"), Some(1));
    }

    #[test]
    fn test_duplicate_exchange() {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("ex", ExchangeType::Direct).unwrap();
        assert!(matches!(
            broker.declare_exchange("ex", ExchangeType::Fanout),
            Err(BrokerError::DuplicateExchange(_))
        ));
    }

    #[test]
    fn test_duplicate_queue() {
        let mut broker = MessageBroker::new();
        broker.declare_queue(BrokerQueue::new("q")).unwrap();
        assert!(matches!(
            broker.declare_queue(BrokerQueue::new("q")),
            Err(BrokerError::DuplicateQueue(_))
        ));
    }

    #[test]
    fn test_delete_exchange() {
        let mut broker = MessageBroker::new();
        broker.declare_exchange("ex", ExchangeType::Direct).unwrap();
        broker.delete_exchange("ex").unwrap();
        assert!(matches!(
            broker.delete_exchange("ex"),
            Err(BrokerError::ExchangeNotFound(_))
        ));
    }

    #[test]
    fn test_delete_queue_removes_consumers() {
        let mut broker = MessageBroker::new();
        broker.declare_queue(BrokerQueue::new("q")).unwrap();
        broker
            .register_consumer(Consumer::new("c1", "q", 10))
            .unwrap();
        broker.delete_queue("q").unwrap();
        assert_eq!(broker.consumer_count(), 0);
    }

    #[test]
    fn test_unbind() {
        let mut broker = setup_direct();
        broker.unbind("orders", "order-queue").unwrap();
        let queues = broker
            .publish("orders", "order.created", "data")
            .unwrap();
        assert!(queues.is_empty());
    }

    #[test]
    fn test_message_log() {
        let mut broker = setup_direct();
        broker.publish("orders", "order.created", "p1").unwrap();
        broker.publish("orders", "order.created", "p2").unwrap();
        assert_eq!(broker.message_log().len(), 2);
    }

    #[test]
    fn test_exchange_not_found() {
        let mut broker = MessageBroker::new();
        assert!(matches!(
            broker.publish("nope", "key", "data"),
            Err(BrokerError::ExchangeNotFound(_))
        ));
    }

    #[test]
    fn test_consumer_not_found() {
        let mut broker = MessageBroker::new();
        assert!(matches!(
            broker.consume("nope"),
            Err(BrokerError::ConsumerNotFound(_))
        ));
    }

    #[test]
    fn test_duplicate_binding() {
        let mut broker = setup_direct();
        assert!(matches!(
            broker.bind("orders", "order-queue", "order.created"),
            Err(BrokerError::DuplicateBinding { .. })
        ));
    }
}
