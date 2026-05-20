//! Lamport timestamps — logical clock, send/receive event handling,
//! happens-before relation, total ordering with tie-breaking, event log,
//! clock synchronization, distributed event ordering.

use std::collections::HashMap;

// ── Lamport Clock ────────────────────────────────────────────────────────────

/// A Lamport logical clock for a single process/node.
#[derive(Debug, Clone)]
pub struct LamportClock {
    /// Current timestamp value.
    timestamp: u64,
    /// Node identifier for tie-breaking in total order.
    node_id: String,
}

impl LamportClock {
    /// Create a new Lamport clock for the given node.
    pub fn new(node_id: &str) -> Self {
        Self {
            timestamp: 0,
            node_id: node_id.to_string(),
        }
    }

    /// Create a Lamport clock with an initial timestamp.
    pub fn with_timestamp(node_id: &str, timestamp: u64) -> Self {
        Self {
            timestamp,
            node_id: node_id.to_string(),
        }
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get the node id.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Tick: increment the clock for a local event. Returns the new timestamp.
    pub fn tick(&mut self) -> u64 {
        self.timestamp += 1;
        self.timestamp
    }

    /// Send event: tick the clock and return a timestamped message.
    pub fn send(&mut self) -> u64 {
        self.tick()
    }

    /// Receive event: update the clock based on the received timestamp,
    /// then tick. Returns the new timestamp.
    pub fn receive(&mut self, received_ts: u64) -> u64 {
        if received_ts > self.timestamp {
            self.timestamp = received_ts;
        }
        self.tick()
    }

    /// Synchronize with a set of remote timestamps (e.g., during gossip).
    /// Takes the max and then ticks.
    pub fn sync(&mut self, remote_timestamps: &[u64]) -> u64 {
        for &ts in remote_timestamps {
            if ts > self.timestamp {
                self.timestamp = ts;
            }
        }
        self.tick()
    }

    /// Reset the clock (for testing or reinitialization).
    pub fn reset(&mut self) {
        self.timestamp = 0;
    }
}

// ── Stamped Event ────────────────────────────────────────────────────────────

/// An event stamped with a Lamport timestamp and source node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StampedEvent {
    /// Lamport timestamp.
    pub timestamp: u64,
    /// Node that generated the event.
    pub node_id: String,
    /// Event payload.
    pub payload: String,
    /// Event kind.
    pub kind: EventKind,
}

/// The kind of event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// A local event (internal computation).
    Local,
    /// A send event (message sent to another node).
    Send,
    /// A receive event (message received from another node).
    Receive,
}

impl StampedEvent {
    /// Create a new stamped event.
    pub fn new(timestamp: u64, node_id: &str, payload: &str, kind: EventKind) -> Self {
        Self {
            timestamp,
            node_id: node_id.to_string(),
            payload: payload.to_string(),
            kind,
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

/// Relation between two events according to Lamport happens-before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HappensBefore {
    /// Event A happens before Event B.
    Before,
    /// Event A happens after Event B.
    After,
    /// Events are concurrent (cannot determine ordering from timestamps alone).
    Concurrent,
    /// Events are the same (same timestamp and node).
    Equal,
}

/// Compare two stamped events using the Lamport happens-before relation.
/// Note: Lamport timestamps only provide a necessary condition for happens-before;
/// if ts(a) < ts(b), a *might* happen before b. If ts(a) >= ts(b), a does NOT
/// happen before b. For a definitive happens-before we'd need vector clocks.
/// Here we provide the total order comparison.
pub fn compare_events(a: &StampedEvent, b: &StampedEvent) -> HappensBefore {
    if a.timestamp == b.timestamp && a.node_id == b.node_id {
        return HappensBefore::Equal;
    }
    if a.timestamp < b.timestamp {
        HappensBefore::Before
    } else if a.timestamp > b.timestamp {
        HappensBefore::After
    } else {
        // Same timestamp, different nodes — concurrent in Lamport semantics.
        HappensBefore::Concurrent
    }
}

/// Total order comparison: first by timestamp, then by node_id.
/// Returns std::cmp::Ordering for use in sorting.
pub fn total_order(a: &StampedEvent, b: &StampedEvent) -> std::cmp::Ordering {
    a.timestamp.cmp(&b.timestamp)
        .then(a.node_id.cmp(&b.node_id))
}

// ── Event Log ────────────────────────────────────────────────────────────────

/// An event log that records stamped events and supports ordering queries.
#[derive(Debug, Clone)]
pub struct EventLog {
    events: Vec<StampedEvent>,
}

impl EventLog {
    /// Create a new empty event log.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }

    /// Record an event.
    pub fn record(&mut self, event: StampedEvent) {
        self.events.push(event);
    }

    /// Get all events in insertion order.
    pub fn events(&self) -> &[StampedEvent] {
        &self.events
    }

    /// Get the number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get all events sorted by total order.
    pub fn sorted_events(&self) -> Vec<&StampedEvent> {
        let mut sorted: Vec<&StampedEvent> = self.events.iter().collect();
        sorted.sort_by(|a, b| total_order(a, b));
        sorted
    }

    /// Get all events from a specific node, in timestamp order.
    pub fn events_for_node(&self, node_id: &str) -> Vec<&StampedEvent> {
        let mut events: Vec<&StampedEvent> = self.events
            .iter()
            .filter(|e| e.node_id == node_id)
            .collect();
        events.sort_by_key(|e| e.timestamp);
        events
    }

    /// Get all events within a timestamp range (inclusive).
    pub fn events_in_range(&self, from: u64, to: u64) -> Vec<&StampedEvent> {
        let mut events: Vec<&StampedEvent> = self.events
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect();
        events.sort_by(|a, b| total_order(a, b));
        events
    }

    /// Get the maximum timestamp in the log.
    pub fn max_timestamp(&self) -> u64 {
        self.events.iter().map(|e| e.timestamp).max().unwrap_or(0)
    }

    /// Get the minimum timestamp in the log.
    pub fn min_timestamp(&self) -> u64 {
        self.events.iter().map(|e| e.timestamp).min().unwrap_or(0)
    }

    /// Get events grouped by node.
    pub fn events_by_node(&self) -> HashMap<&str, Vec<&StampedEvent>> {
        let mut map: HashMap<&str, Vec<&StampedEvent>> = HashMap::new();
        for event in &self.events {
            map.entry(event.node_id.as_str())
                .or_default()
                .push(event);
        }
        // Sort each node's events by timestamp.
        for events in map.values_mut() {
            events.sort_by_key(|e| e.timestamp);
        }
        map
    }

    /// Count events by kind.
    pub fn count_by_kind(&self) -> (usize, usize, usize) {
        let mut local = 0;
        let mut send = 0;
        let mut receive = 0;
        for event in &self.events {
            match event.kind {
                EventKind::Local => local += 1,
                EventKind::Send => send += 1,
                EventKind::Receive => receive += 1,
            }
        }
        (local, send, receive)
    }

    /// Merge another event log into this one (deduplication by timestamp+node+payload).
    pub fn merge(&mut self, other: &EventLog) {
        for event in &other.events {
            let already_present = self.events.iter().any(|e| {
                e.timestamp == event.timestamp
                    && e.node_id == event.node_id
                    && e.payload == event.payload
            });
            if !already_present {
                self.events.push(event.clone());
            }
        }
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Distributed Clock System ─────────────────────────────────────────────────

/// A system of Lamport clocks for multiple nodes, with an event log.
#[derive(Debug, Clone)]
pub struct ClockSystem {
    /// Clocks for each node.
    clocks: HashMap<String, LamportClock>,
    /// Global event log.
    log: EventLog,
}

impl ClockSystem {
    /// Create a new clock system.
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
            log: EventLog::new(),
        }
    }

    /// Register a node in the system.
    pub fn add_node(&mut self, node_id: &str) {
        self.clocks.insert(
            node_id.to_string(),
            LamportClock::new(node_id),
        );
    }

    /// Record a local event on a node.
    pub fn local_event(&mut self, node_id: &str, payload: &str) -> Option<u64> {
        let clock = self.clocks.get_mut(node_id)?;
        let ts = clock.tick();
        self.log.record(StampedEvent::new(ts, node_id, payload, EventKind::Local));
        Some(ts)
    }

    /// Record a send event: node `from` sends a message.
    /// Returns the timestamp to include in the message.
    pub fn send_event(&mut self, from: &str, payload: &str) -> Option<u64> {
        let clock = self.clocks.get_mut(from)?;
        let ts = clock.send();
        self.log.record(StampedEvent::new(ts, from, payload, EventKind::Send));
        Some(ts)
    }

    /// Record a receive event: node `to` receives a message with `sent_ts`.
    pub fn receive_event(&mut self, to: &str, sent_ts: u64, payload: &str) -> Option<u64> {
        let clock = self.clocks.get_mut(to)?;
        let ts = clock.receive(sent_ts);
        self.log.record(StampedEvent::new(ts, to, payload, EventKind::Receive));
        Some(ts)
    }

    /// Get the current timestamp for a node.
    pub fn timestamp_for(&self, node_id: &str) -> Option<u64> {
        self.clocks.get(node_id).map(|c| c.timestamp())
    }

    /// Get the event log.
    pub fn log(&self) -> &EventLog {
        &self.log
    }

    /// Get the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.clocks.len()
    }
}

impl Default for ClockSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_new_starts_at_zero() {
        let c = LamportClock::new("n1");
        assert_eq!(c.timestamp(), 0);
        assert_eq!(c.node_id(), "n1");
    }

    #[test]
    fn clock_tick() {
        let mut c = LamportClock::new("n1");
        assert_eq!(c.tick(), 1);
        assert_eq!(c.tick(), 2);
        assert_eq!(c.tick(), 3);
    }

    #[test]
    fn clock_send() {
        let mut c = LamportClock::new("n1");
        let ts = c.send();
        assert_eq!(ts, 1);
    }

    #[test]
    fn clock_receive_from_ahead() {
        let mut c = LamportClock::new("n1");
        c.tick(); // ts = 1
        let ts = c.receive(10); // max(1,10) + 1 = 11
        assert_eq!(ts, 11);
    }

    #[test]
    fn clock_receive_from_behind() {
        let mut c = LamportClock::new("n1");
        c.tick(); // ts = 1
        c.tick(); // ts = 2
        c.tick(); // ts = 3
        let ts = c.receive(1); // max(3,1) + 1 = 4
        assert_eq!(ts, 4);
    }

    #[test]
    fn clock_sync_with_multiple() {
        let mut c = LamportClock::new("n1");
        let ts = c.sync(&[5, 3, 10, 2]);
        assert_eq!(ts, 11); // max(0,5,3,10,2) + 1 = 11
    }

    #[test]
    fn clock_reset() {
        let mut c = LamportClock::new("n1");
        c.tick();
        c.tick();
        c.reset();
        assert_eq!(c.timestamp(), 0);
    }

    #[test]
    fn clock_with_timestamp() {
        let c = LamportClock::with_timestamp("n1", 42);
        assert_eq!(c.timestamp(), 42);
    }

    #[test]
    fn compare_events_before() {
        let a = StampedEvent::new(1, "n1", "a", EventKind::Local);
        let b = StampedEvent::new(2, "n2", "b", EventKind::Local);
        assert_eq!(compare_events(&a, &b), HappensBefore::Before);
    }

    #[test]
    fn compare_events_after() {
        let a = StampedEvent::new(5, "n1", "a", EventKind::Local);
        let b = StampedEvent::new(2, "n2", "b", EventKind::Local);
        assert_eq!(compare_events(&a, &b), HappensBefore::After);
    }

    #[test]
    fn compare_events_concurrent() {
        let a = StampedEvent::new(3, "n1", "a", EventKind::Local);
        let b = StampedEvent::new(3, "n2", "b", EventKind::Local);
        assert_eq!(compare_events(&a, &b), HappensBefore::Concurrent);
    }

    #[test]
    fn compare_events_equal() {
        let a = StampedEvent::new(3, "n1", "a", EventKind::Local);
        let b = StampedEvent::new(3, "n1", "b", EventKind::Local);
        assert_eq!(compare_events(&a, &b), HappensBefore::Equal);
    }

    #[test]
    fn total_order_deterministic() {
        let a = StampedEvent::new(3, "n1", "a", EventKind::Local);
        let b = StampedEvent::new(3, "n2", "b", EventKind::Local);
        assert_eq!(total_order(&a, &b), std::cmp::Ordering::Less);
    }

    #[test]
    fn event_log_record_and_query() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log.record(StampedEvent::new(2, "n2", "b", EventKind::Send));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn event_log_sorted_events() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(3, "n1", "c", EventKind::Local));
        log.record(StampedEvent::new(1, "n2", "a", EventKind::Local));
        log.record(StampedEvent::new(2, "n1", "b", EventKind::Local));
        let sorted = log.sorted_events();
        assert_eq!(sorted[0].timestamp, 1);
        assert_eq!(sorted[1].timestamp, 2);
        assert_eq!(sorted[2].timestamp, 3);
    }

    #[test]
    fn event_log_events_for_node() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log.record(StampedEvent::new(2, "n2", "b", EventKind::Local));
        log.record(StampedEvent::new(3, "n1", "c", EventKind::Local));
        let n1_events = log.events_for_node("n1");
        assert_eq!(n1_events.len(), 2);
        assert_eq!(n1_events[0].payload, "a");
        assert_eq!(n1_events[1].payload, "c");
    }

    #[test]
    fn event_log_range_query() {
        let mut log = EventLog::new();
        for i in 1..=10 {
            log.record(StampedEvent::new(i, "n1", &format!("e{}", i), EventKind::Local));
        }
        let range = log.events_in_range(3, 7);
        assert_eq!(range.len(), 5);
        assert_eq!(range[0].timestamp, 3);
        assert_eq!(range[4].timestamp, 7);
    }

    #[test]
    fn event_log_min_max_timestamp() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(5, "n1", "a", EventKind::Local));
        log.record(StampedEvent::new(2, "n2", "b", EventKind::Local));
        log.record(StampedEvent::new(8, "n3", "c", EventKind::Local));
        assert_eq!(log.max_timestamp(), 8);
        assert_eq!(log.min_timestamp(), 2);
    }

    #[test]
    fn event_log_count_by_kind() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log.record(StampedEvent::new(2, "n1", "b", EventKind::Send));
        log.record(StampedEvent::new(3, "n2", "c", EventKind::Receive));
        log.record(StampedEvent::new(4, "n1", "d", EventKind::Local));
        let (local, send, receive) = log.count_by_kind();
        assert_eq!(local, 2);
        assert_eq!(send, 1);
        assert_eq!(receive, 1);
    }

    #[test]
    fn event_log_merge() {
        let mut log1 = EventLog::new();
        let mut log2 = EventLog::new();
        log1.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log2.record(StampedEvent::new(2, "n2", "b", EventKind::Local));
        log1.merge(&log2);
        assert_eq!(log1.len(), 2);
    }

    #[test]
    fn event_log_merge_deduplicates() {
        let mut log1 = EventLog::new();
        let mut log2 = EventLog::new();
        log1.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log2.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log1.merge(&log2);
        assert_eq!(log1.len(), 1);
    }

    #[test]
    fn clock_system_send_receive() {
        let mut sys = ClockSystem::new();
        sys.add_node("n1");
        sys.add_node("n2");
        let ts = sys.send_event("n1", "hello").unwrap();
        assert_eq!(ts, 1);
        let recv_ts = sys.receive_event("n2", ts, "hello").unwrap();
        assert_eq!(recv_ts, 2); // max(0,1) + 1 = 2
    }

    #[test]
    fn clock_system_local_event() {
        let mut sys = ClockSystem::new();
        sys.add_node("n1");
        let ts = sys.local_event("n1", "work").unwrap();
        assert_eq!(ts, 1);
        let ts2 = sys.local_event("n1", "more").unwrap();
        assert_eq!(ts2, 2);
    }

    #[test]
    fn clock_system_nonexistent_node() {
        let mut sys = ClockSystem::new();
        assert!(sys.local_event("n1", "x").is_none());
    }

    #[test]
    fn event_log_clear() {
        let mut log = EventLog::new();
        log.record(StampedEvent::new(1, "n1", "a", EventKind::Local));
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn clock_system_node_count() {
        let mut sys = ClockSystem::new();
        sys.add_node("n1");
        sys.add_node("n2");
        assert_eq!(sys.node_count(), 2);
    }
}
