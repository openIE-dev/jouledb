//! Vector clocks — logical clocks for distributed systems.
//!
//! Supports increment, merge, happens-before comparison, concurrent detection,
//! causal ordering of events, serialization, and timeline reconstruction.

use std::collections::HashMap;

// ── Ordering ─────────────────────────────────────────────────────────────────

/// Comparison result between two vector clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockOrder {
    /// First happened before second.
    Before,
    /// First happened after second.
    After,
    /// Events are concurrent (neither happened before the other).
    Concurrent,
    /// Clocks are identical.
    Equal,
}

// ── Vector Clock ─────────────────────────────────────────────────────────────

/// A vector clock mapping node ids to logical timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    clock: HashMap<String, u64>,
}

impl VectorClock {
    /// Create a new empty vector clock.
    pub fn new() -> Self {
        Self {
            clock: HashMap::new(),
        }
    }

    /// Create a vector clock with initial entries.
    pub fn from_entries(entries: &[(&str, u64)]) -> Self {
        let mut clock = HashMap::new();
        for (node, ts) in entries {
            clock.insert(node.to_string(), *ts);
        }
        Self { clock }
    }

    /// Get the timestamp for a given node.
    pub fn get(&self, node: &str) -> u64 {
        self.clock.get(node).copied().unwrap_or(0)
    }

    /// Set the timestamp for a given node.
    pub fn set(&mut self, node: &str, value: u64) {
        if value == 0 {
            self.clock.remove(node);
        } else {
            self.clock.insert(node.to_string(), value);
        }
    }

    /// Increment the timestamp for a node (local event).
    pub fn increment(&mut self, node: &str) {
        let val = self.get(node) + 1;
        self.clock.insert(node.to_string(), val);
    }

    /// Merge with another clock (element-wise max).
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &ts) in &other.clock {
            let current = self.get(node);
            if ts > current {
                self.clock.insert(node.clone(), ts);
            }
        }
    }

    /// Create a merged clock without mutating self.
    pub fn merged(&self, other: &VectorClock) -> VectorClock {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Compare two vector clocks.
    pub fn compare(&self, other: &VectorClock) -> ClockOrder {
        let all_nodes: Vec<String> = {
            let mut set: std::collections::HashSet<String> = self.clock.keys().cloned().collect();
            for k in other.clock.keys() {
                set.insert(k.clone());
            }
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        };

        let mut self_less = false;
        let mut other_less = false;

        for node in &all_nodes {
            let a = self.get(node);
            let b = other.get(node);
            if a < b {
                self_less = true;
            }
            if a > b {
                other_less = true;
            }
        }

        match (self_less, other_less) {
            (false, false) => ClockOrder::Equal,
            (true, false) => ClockOrder::Before,
            (false, true) => ClockOrder::After,
            (true, true) => ClockOrder::Concurrent,
        }
    }

    /// Check if this clock happened before the other.
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrder::Before
    }

    /// Check if this clock happened after the other.
    pub fn happened_after(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrder::After
    }

    /// Check if this clock is concurrent with the other.
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrder::Concurrent
    }

    /// Number of entries in the clock.
    pub fn len(&self) -> usize {
        self.clock.len()
    }

    /// Whether the clock is empty.
    pub fn is_empty(&self) -> bool {
        self.clock.is_empty()
    }

    /// All node ids in the clock.
    pub fn nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self.clock.keys().cloned().collect();
        nodes.sort();
        nodes
    }

    /// Serialize the clock to a JSON-like string representation.
    pub fn serialize(&self) -> String {
        let mut entries: Vec<(&str, u64)> = self
            .clock
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let parts: Vec<String> = entries
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect();
        parts.join(",")
    }

    /// Deserialize from the serialize() format.
    pub fn deserialize(s: &str) -> Self {
        let mut clock = HashMap::new();
        if s.is_empty() {
            return Self { clock };
        }
        for part in s.split(',') {
            let mut split = part.splitn(2, ':');
            if let (Some(node), Some(ts_str)) = (split.next(), split.next()) {
                if let Ok(ts) = ts_str.parse::<u64>() {
                    clock.insert(node.to_string(), ts);
                }
            }
        }
        Self { clock }
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

// ── Causally Ordered Event ───────────────────────────────────────────────────

/// An event tagged with a vector clock for causal ordering.
#[derive(Debug, Clone)]
pub struct CausalEvent {
    /// The node that generated this event.
    pub node: String,
    /// The vector clock at the time of this event.
    pub clock: VectorClock,
    /// Event payload.
    pub payload: String,
}

// ── Event Timeline ───────────────────────────────────────────────────────────

/// Tracks causally ordered events across multiple nodes.
#[derive(Debug, Clone)]
pub struct EventTimeline {
    events: Vec<CausalEvent>,
    /// Current vector clock per node.
    clocks: HashMap<String, VectorClock>,
}

impl EventTimeline {
    /// Create a new empty timeline.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            clocks: HashMap::new(),
        }
    }

    /// Record a local event at a node.
    pub fn record_event(&mut self, node: &str, payload: &str) -> VectorClock {
        let clock = self.clocks.entry(node.to_string()).or_default();
        clock.increment(node);
        let snapshot = clock.clone();
        self.events.push(CausalEvent {
            node: node.to_string(),
            clock: snapshot.clone(),
            payload: payload.to_string(),
        });
        snapshot
    }

    /// Record a receive event: the receiving node merges clocks, then increments.
    pub fn record_receive(
        &mut self,
        receiver: &str,
        sender_clock: &VectorClock,
        payload: &str,
    ) -> VectorClock {
        let clock = self.clocks.entry(receiver.to_string()).or_default();
        clock.merge(sender_clock);
        clock.increment(receiver);
        let snapshot = clock.clone();
        self.events.push(CausalEvent {
            node: receiver.to_string(),
            clock: snapshot.clone(),
            payload: payload.to_string(),
        });
        snapshot
    }

    /// Get the current clock for a node.
    pub fn clock_for(&self, node: &str) -> VectorClock {
        self.clocks.get(node).cloned().unwrap_or_default()
    }

    /// All events in insertion order.
    pub fn events(&self) -> &[CausalEvent] {
        &self.events
    }

    /// Sort events in causal order. Events that happened-before come first;
    /// concurrent events maintain their original order.
    pub fn causal_order(&self) -> Vec<CausalEvent> {
        let mut events = self.events.clone();
        let n = events.len();
        // Stable topological sort using insertion sort
        for i in 1..n {
            let mut j = i;
            while j > 0 {
                let order = events[j].clock.compare(&events[j - 1].clock);
                if order == ClockOrder::Before {
                    events.swap(j, j - 1);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        events
    }

    /// Number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Find all events concurrent with a given clock.
    pub fn find_concurrent(&self, clock: &VectorClock) -> Vec<&CausalEvent> {
        self.events
            .iter()
            .filter(|e| e.clock.is_concurrent(clock))
            .collect()
    }

    /// Find all events that happened before a given clock.
    pub fn find_before(&self, clock: &VectorClock) -> Vec<&CausalEvent> {
        self.events
            .iter()
            .filter(|e| e.clock.happened_before(clock))
            .collect()
    }
}

impl Default for EventTimeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clock() {
        let vc = VectorClock::new();
        assert!(vc.is_empty());
        assert_eq!(vc.get("A"), 0);
    }

    #[test]
    fn test_increment() {
        let mut vc = VectorClock::new();
        vc.increment("A");
        assert_eq!(vc.get("A"), 1);
        vc.increment("A");
        assert_eq!(vc.get("A"), 2);
    }

    #[test]
    fn test_merge() {
        let mut a = VectorClock::from_entries(&[("A", 2), ("B", 1)]);
        let b = VectorClock::from_entries(&[("A", 1), ("B", 3), ("C", 1)]);
        a.merge(&b);
        assert_eq!(a.get("A"), 2);
        assert_eq!(a.get("B"), 3);
        assert_eq!(a.get("C"), 1);
    }

    #[test]
    fn test_merged_non_mutating() {
        let a = VectorClock::from_entries(&[("A", 2)]);
        let b = VectorClock::from_entries(&[("B", 3)]);
        let c = a.merged(&b);
        assert_eq!(c.get("A"), 2);
        assert_eq!(c.get("B"), 3);
        assert_eq!(a.get("B"), 0); // a is unchanged
    }

    #[test]
    fn test_equal_clocks() {
        let a = VectorClock::from_entries(&[("A", 1), ("B", 2)]);
        let b = VectorClock::from_entries(&[("A", 1), ("B", 2)]);
        assert_eq!(a.compare(&b), ClockOrder::Equal);
    }

    #[test]
    fn test_happened_before() {
        let a = VectorClock::from_entries(&[("A", 1), ("B", 1)]);
        let b = VectorClock::from_entries(&[("A", 2), ("B", 2)]);
        assert!(a.happened_before(&b));
        assert!(b.happened_after(&a));
    }

    #[test]
    fn test_concurrent() {
        let a = VectorClock::from_entries(&[("A", 2), ("B", 1)]);
        let b = VectorClock::from_entries(&[("A", 1), ("B", 2)]);
        assert!(a.is_concurrent(&b));
        assert!(b.is_concurrent(&a));
    }

    #[test]
    fn test_not_concurrent_when_ordered() {
        let a = VectorClock::from_entries(&[("A", 1)]);
        let b = VectorClock::from_entries(&[("A", 2)]);
        assert!(!a.is_concurrent(&b));
    }

    #[test]
    fn test_serialize_deserialize() {
        let vc = VectorClock::from_entries(&[("A", 3), ("B", 1)]);
        let s = vc.serialize();
        let vc2 = VectorClock::deserialize(&s);
        assert_eq!(vc, vc2);
    }

    #[test]
    fn test_serialize_empty() {
        let vc = VectorClock::new();
        let s = vc.serialize();
        assert_eq!(s, "");
        let vc2 = VectorClock::deserialize(&s);
        assert_eq!(vc, vc2);
    }

    #[test]
    fn test_nodes_sorted() {
        let vc = VectorClock::from_entries(&[("C", 1), ("A", 2), ("B", 3)]);
        assert_eq!(vc.nodes(), vec!["A", "B", "C"]);
    }

    #[test]
    fn test_event_timeline_basic() {
        let mut tl = EventTimeline::new();
        tl.record_event("A", "write x=1");
        tl.record_event("B", "write y=2");
        assert_eq!(tl.event_count(), 2);
    }

    #[test]
    fn test_event_timeline_send_receive() {
        let mut tl = EventTimeline::new();
        let clock_a = tl.record_event("A", "send msg");
        tl.record_receive("B", &clock_a, "recv msg");
        let clock_b = tl.clock_for("B");
        assert_eq!(clock_b.get("A"), 1);
        assert_eq!(clock_b.get("B"), 1);
    }

    #[test]
    fn test_causal_ordering() {
        let mut tl = EventTimeline::new();
        let c1 = tl.record_event("A", "first");
        tl.record_receive("B", &c1, "second");
        let ordered = tl.causal_order();
        assert_eq!(ordered[0].payload, "first");
        assert_eq!(ordered[1].payload, "second");
    }

    #[test]
    fn test_find_concurrent() {
        let mut tl = EventTimeline::new();
        tl.record_event("A", "e1");
        tl.record_event("B", "e2");
        let clock_a = tl.clock_for("A");
        let concurrent = tl.find_concurrent(&clock_a);
        // B's event is concurrent with A's clock
        assert_eq!(concurrent.len(), 1);
        assert_eq!(concurrent[0].payload, "e2");
    }

    #[test]
    fn test_find_before() {
        let mut tl = EventTimeline::new();
        let c1 = tl.record_event("A", "e1");
        tl.record_receive("B", &c1, "e2");
        let clock_b = tl.clock_for("B");
        let before = tl.find_before(&clock_b);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].payload, "e1");
    }

    #[test]
    fn test_set_clock_value() {
        let mut vc = VectorClock::new();
        vc.set("A", 5);
        assert_eq!(vc.get("A"), 5);
        vc.set("A", 0);
        assert_eq!(vc.get("A"), 0);
        assert!(vc.is_empty());
    }

    #[test]
    fn test_len() {
        let vc = VectorClock::from_entries(&[("A", 1), ("B", 2), ("C", 3)]);
        assert_eq!(vc.len(), 3);
    }

    #[test]
    fn test_default_clock() {
        let vc = VectorClock::default();
        assert!(vc.is_empty());
    }

    #[test]
    fn test_default_timeline() {
        let tl = EventTimeline::default();
        assert_eq!(tl.event_count(), 0);
    }

    #[test]
    fn test_complex_scenario() {
        // A sends to B, B sends to C, C's clock should reflect all
        let mut tl = EventTimeline::new();
        let ca = tl.record_event("A", "a1");
        let cb = tl.record_receive("B", &ca, "b1");
        tl.record_receive("C", &cb, "c1");
        let cc = tl.clock_for("C");
        assert_eq!(cc.get("A"), 1);
        assert_eq!(cc.get("B"), 1);
        assert_eq!(cc.get("C"), 1);
    }
}
