//! Mediator pattern — decoupled colleague communication via a central coordinator.
//!
//! The `Mediator` orchestrates communication between registered `Colleague`s.
//! Colleagues send messages through the mediator, which routes them as
//! broadcasts or directed messages. The mediator maintains its own state
//! and an event log.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Message ────────────────────────────────────────────────────────

/// A message exchanged through the mediator.
#[derive(Debug, Clone)]
pub struct Message {
    pub from: String,
    pub to: MessageTarget,
    pub event: String,
    pub payload: Value,
}

/// Where a message is directed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageTarget {
    /// Broadcast to all colleagues (except sender).
    Broadcast,
    /// Directed to a specific colleague by ID.
    Direct(String),
}

impl Message {
    /// Create a broadcast message.
    pub fn broadcast(from: impl Into<String>, event: impl Into<String>, payload: Value) -> Self {
        Self {
            from: from.into(),
            to: MessageTarget::Broadcast,
            event: event.into(),
            payload,
        }
    }

    /// Create a directed message.
    pub fn direct(
        from: impl Into<String>,
        to: impl Into<String>,
        event: impl Into<String>,
        payload: Value,
    ) -> Self {
        Self {
            from: from.into(),
            to: MessageTarget::Direct(to.into()),
            event: event.into(),
            payload,
        }
    }
}

// ── Colleague ──────────────────────────────────────────────────────

/// A participant in the mediator network.
pub struct Colleague {
    id: String,
    description: String,
    inbox: Vec<Message>,
    handler: Box<dyn Fn(&Message) -> Option<Value>>,
}

impl Colleague {
    /// Create a colleague with a handler that optionally returns a response value.
    pub fn new(
        id: impl Into<String>,
        handler: impl Fn(&Message) -> Option<Value> + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            inbox: Vec::new(),
            handler: Box::new(handler),
        }
    }

    /// Set a description for this colleague.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// The colleague's unique ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Number of messages received so far.
    pub fn inbox_count(&self) -> usize {
        self.inbox.len()
    }

    /// Get a reference to all received messages.
    pub fn inbox(&self) -> &[Message] {
        &self.inbox
    }

    /// Receive a message: store it and invoke the handler.
    fn receive(&mut self, msg: &Message) -> Option<Value> {
        self.inbox.push(msg.clone());
        (self.handler)(msg)
    }
}

impl fmt::Debug for Colleague {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Colleague")
            .field("id", &self.id)
            .field("description", &self.description)
            .field("inbox_count", &self.inbox.len())
            .finish()
    }
}

// ── Colleague info ─────────────────────────────────────────────────

/// A lightweight snapshot of a colleague's state (no closures).
#[derive(Debug, Clone)]
pub struct ColleagueInfo {
    pub id: String,
    pub description: String,
    pub inbox_count: usize,
}

// ── Event log entry ────────────────────────────────────────────────

/// A recorded event in the mediator's log.
#[derive(Debug, Clone)]
pub struct EventEntry {
    pub sequence: u64,
    pub message: Message,
    pub delivered_to: Vec<String>,
}

// ── Mediator ───────────────────────────────────────────────────────

/// Central coordinator that routes messages between colleagues.
pub struct Mediator {
    colleagues: Vec<Colleague>,
    state: HashMap<String, Value>,
    event_log: Vec<EventEntry>,
    sequence: u64,
}

impl Mediator {
    /// Create an empty mediator.
    pub fn new() -> Self {
        Self {
            colleagues: Vec::new(),
            state: HashMap::new(),
            event_log: Vec::new(),
            sequence: 0,
        }
    }

    /// Register a colleague.
    pub fn register(&mut self, colleague: Colleague) {
        // Prevent duplicate IDs.
        self.colleagues.retain(|c| c.id != colleague.id);
        self.colleagues.push(colleague);
    }

    /// Unregister a colleague by ID. Returns true if found.
    pub fn unregister(&mut self, id: &str) -> bool {
        let before = self.colleagues.len();
        self.colleagues.retain(|c| c.id != id);
        self.colleagues.len() < before
    }

    /// Number of registered colleagues.
    pub fn colleague_count(&self) -> usize {
        self.colleagues.len()
    }

    /// Whether a colleague with the given ID is registered.
    pub fn has_colleague(&self, id: &str) -> bool {
        self.colleagues.iter().any(|c| c.id == id)
    }

    /// Snapshot info for all colleagues.
    pub fn colleague_info(&self) -> Vec<ColleagueInfo> {
        self.colleagues
            .iter()
            .map(|c| ColleagueInfo {
                id: c.id.clone(),
                description: c.description.clone(),
                inbox_count: c.inbox_count(),
            })
            .collect()
    }

    /// Set mediator-level state.
    pub fn set_state(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    /// Get mediator-level state.
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Full event log.
    pub fn event_log(&self) -> &[EventEntry] {
        &self.event_log
    }

    /// Total number of messages routed.
    pub fn message_count(&self) -> u64 {
        self.sequence
    }

    /// Send a message through the mediator, routing to appropriate colleagues.
    /// Returns a list of (colleague_id, optional_response) pairs.
    pub fn send(&mut self, msg: Message) -> Vec<(String, Option<Value>)> {
        self.sequence += 1;
        let seq = self.sequence;
        let mut delivered_to = Vec::new();
        let mut results = Vec::new();

        match &msg.to {
            MessageTarget::Broadcast => {
                // Collect indices that should receive this message.
                let indices: Vec<usize> = self
                    .colleagues
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.id != msg.from)
                    .map(|(i, _)| i)
                    .collect();

                for idx in indices {
                    let id = self.colleagues[idx].id.clone();
                    let response = self.colleagues[idx].receive(&msg);
                    delivered_to.push(id.clone());
                    results.push((id, response));
                }
            }
            MessageTarget::Direct(target) => {
                if let Some(idx) = self.colleagues.iter().position(|c| c.id == *target) {
                    let id = self.colleagues[idx].id.clone();
                    let response = self.colleagues[idx].receive(&msg);
                    delivered_to.push(id.clone());
                    results.push((id, response));
                }
            }
        }

        self.event_log.push(EventEntry {
            sequence: seq,
            message: msg,
            delivered_to,
        });

        results
    }

    /// Get a reference to a colleague's inbox by ID.
    pub fn colleague_inbox(&self, id: &str) -> Option<&[Message]> {
        self.colleagues
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.inbox())
    }
}

impl Default for Mediator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_colleague(id: &str) -> Colleague {
        let id_owned = id.to_string();
        Colleague::new(id, move |msg| {
            Some(Value::String(format!("{} received: {}", id_owned, msg.event)))
        })
    }

    fn silent_colleague(id: &str) -> Colleague {
        Colleague::new(id, |_| None)
    }

    #[test]
    fn register_and_count() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("b"));
        assert_eq!(m.colleague_count(), 2);
    }

    #[test]
    fn register_dedup_by_id() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("a"));
        assert_eq!(m.colleague_count(), 1);
    }

    #[test]
    fn unregister() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        assert!(m.unregister("a"));
        assert_eq!(m.colleague_count(), 0);
        assert!(!m.unregister("a"));
    }

    #[test]
    fn has_colleague() {
        let mut m = Mediator::new();
        m.register(echo_colleague("x"));
        assert!(m.has_colleague("x"));
        assert!(!m.has_colleague("y"));
    }

    #[test]
    fn broadcast_message() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("b"));
        m.register(echo_colleague("c"));

        let msg = Message::broadcast("a", "hello", Value::Null);
        let results = m.send(msg);

        // "a" should NOT receive it (sender excluded).
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert!(!ids.contains(&"a"));
    }

    #[test]
    fn directed_message() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("b"));

        let msg = Message::direct("a", "b", "ping", Value::Null);
        let results = m.send(msg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn directed_to_nonexistent() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));

        let msg = Message::direct("a", "z", "ping", Value::Null);
        let results = m.send(msg);
        assert!(results.is_empty());
    }

    #[test]
    fn response_values() {
        let mut m = Mediator::new();
        m.register(echo_colleague("srv"));

        let msg = Message::direct("client", "srv", "query", Value::Null);
        let results = m.send(msg);
        assert_eq!(results.len(), 1);
        let resp = results[0].1.as_ref().unwrap();
        assert_eq!(resp, &Value::String("srv received: query".to_string()));
    }

    #[test]
    fn silent_colleague_no_response() {
        let mut m = Mediator::new();
        m.register(silent_colleague("quiet"));

        let msg = Message::direct("sender", "quiet", "test", Value::Null);
        let results = m.send(msg);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_none());
    }

    #[test]
    fn event_log_recorded() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("b"));

        m.send(Message::broadcast("a", "evt1", Value::Null));
        m.send(Message::direct("b", "a", "evt2", Value::Null));

        let log = m.event_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].sequence, 1);
        assert_eq!(log[0].message.event, "evt1");
        assert_eq!(log[1].sequence, 2);
        assert_eq!(log[1].message.event, "evt2");
    }

    #[test]
    fn event_log_delivered_to() {
        let mut m = Mediator::new();
        m.register(echo_colleague("x"));
        m.register(echo_colleague("y"));

        m.send(Message::broadcast("x", "ping", Value::Null));

        let entry = &m.event_log()[0];
        assert_eq!(entry.delivered_to, vec!["y".to_string()]);
    }

    #[test]
    fn mediator_state() {
        let mut m = Mediator::new();
        m.set_state("mode", Value::String("active".to_string()));
        assert_eq!(
            m.get_state("mode"),
            Some(&Value::String("active".to_string()))
        );
        assert!(m.get_state("missing").is_none());
    }

    #[test]
    fn message_count() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        assert_eq!(m.message_count(), 0);
        m.send(Message::broadcast("a", "e", Value::Null));
        m.send(Message::broadcast("a", "e", Value::Null));
        assert_eq!(m.message_count(), 2);
    }

    #[test]
    fn colleague_inbox() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a"));
        m.register(echo_colleague("b"));

        m.send(Message::direct("a", "b", "first", Value::Null));
        m.send(Message::direct("a", "b", "second", Value::Null));

        let inbox = m.colleague_inbox("b").unwrap();
        assert_eq!(inbox.len(), 2);
        assert_eq!(inbox[0].event, "first");
        assert_eq!(inbox[1].event, "second");
    }

    #[test]
    fn colleague_inbox_nonexistent() {
        let m = Mediator::new();
        assert!(m.colleague_inbox("z").is_none());
    }

    #[test]
    fn colleague_info_snapshot() {
        let mut m = Mediator::new();
        m.register(echo_colleague("a").with_description("Agent A"));
        m.register(echo_colleague("b"));

        let info = m.colleague_info();
        assert_eq!(info.len(), 2);
        let a_info = info.iter().find(|i| i.id == "a").unwrap();
        assert_eq!(a_info.description, "Agent A");
    }

    #[test]
    fn colleague_description() {
        let c = echo_colleague("test").with_description("My test colleague");
        assert_eq!(c.description(), "My test colleague");
        assert_eq!(c.id(), "test");
    }

    #[test]
    fn empty_mediator_broadcast() {
        let mut m = Mediator::new();
        let msg = Message::broadcast("nobody", "ping", Value::Null);
        let results = m.send(msg);
        assert!(results.is_empty());
    }
}
