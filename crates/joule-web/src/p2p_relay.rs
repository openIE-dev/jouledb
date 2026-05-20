//! P2P relay protocol for message forwarding — multi-hop relay, route
//! discovery, message deduplication, hop count limiting, relay path
//! optimization, bidirectional relay, and relay statistics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── RelayMessage ────────────────────────────────────────────────────────────

/// A message being relayed through the P2P network.
#[derive(Debug, Clone)]
pub struct RelayMessage {
    pub id: u64,
    pub source: String,
    pub destination: String,
    pub payload: Vec<u8>,
    pub hops: Vec<String>,
    pub ttl: u32,
    pub created_at: u64,
}

impl RelayMessage {
    pub fn new(
        id: u64,
        source: impl Into<String>,
        destination: impl Into<String>,
        payload: Vec<u8>,
        ttl: u32,
        now: u64,
    ) -> Self {
        let src = source.into();
        Self {
            id,
            source: src.clone(),
            destination: destination.into(),
            payload,
            hops: vec![src],
            ttl,
            created_at: now,
        }
    }

    /// Record a hop through a relay node.
    pub fn record_hop(&mut self, node_id: impl Into<String>) {
        self.hops.push(node_id.into());
        self.ttl = self.ttl.saturating_sub(1);
    }

    /// Whether the message has exhausted its TTL.
    pub fn is_expired(&self) -> bool {
        self.ttl == 0
    }

    /// Number of hops taken so far.
    pub fn hop_count(&self) -> usize {
        self.hops.len().saturating_sub(1) // first entry is source
    }

    /// Whether the message has reached its destination.
    pub fn reached_destination(&self) -> bool {
        self.hops.last().map(|h| h == &self.destination).unwrap_or(false)
    }
}

impl fmt::Display for RelayMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RelayMsg(id={}, {}→{}, hops={}, ttl={})",
            self.id, self.source, self.destination, self.hop_count(), self.ttl,
        )
    }
}

// ── RouteEntry ──────────────────────────────────────────────────────────────

/// A route to a destination through a next-hop node.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    pub destination: String,
    pub next_hop: String,
    pub hop_count: u32,
    pub last_updated: u64,
}

// ── RelayStats ──────────────────────────────────────────────────────────────

/// Statistics for a relay node.
#[derive(Debug, Clone, Default)]
pub struct RelayStats {
    pub messages_forwarded: u64,
    pub messages_delivered: u64,
    pub messages_dropped_ttl: u64,
    pub messages_dropped_dup: u64,
    pub total_hops: u64,
}

impl RelayStats {
    /// Average hops per delivered message.
    pub fn avg_hops(&self) -> f64 {
        if self.messages_delivered == 0 {
            0.0
        } else {
            self.total_hops as f64 / self.messages_delivered as f64
        }
    }
}

impl fmt::Display for RelayStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RelayStats(fwd={}, delivered={}, dropped_ttl={}, dropped_dup={}, avg_hops={:.1})",
            self.messages_forwarded, self.messages_delivered,
            self.messages_dropped_ttl, self.messages_dropped_dup,
            self.avg_hops(),
        )
    }
}

// ── RelayNode ───────────────────────────────────────────────────────────────

/// A node in the P2P relay network.
pub struct RelayNode {
    pub id: String,
    neighbors: HashSet<String>,
    routes: HashMap<String, RouteEntry>,
    seen_messages: HashSet<u64>,
    seen_limit: usize,
    outbox: VecDeque<RelayMessage>,
    delivered: Vec<RelayMessage>,
    stats: RelayStats,
    default_ttl: u32,
    current_tick: u64,
    next_msg_id: u64,
}

impl RelayNode {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            neighbors: HashSet::new(),
            routes: HashMap::new(),
            seen_messages: HashSet::new(),
            seen_limit: 10_000,
            outbox: VecDeque::new(),
            delivered: Vec::new(),
            stats: RelayStats::default(),
            default_ttl: 10,
            current_tick: 0,
            next_msg_id: 1,
        }
    }

    pub fn with_default_ttl(mut self, ttl: u32) -> Self {
        self.default_ttl = ttl;
        self
    }

    pub fn with_seen_limit(mut self, limit: usize) -> Self {
        self.seen_limit = limit;
        self
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Add a neighbor (direct link).
    pub fn add_neighbor(&mut self, neighbor_id: impl Into<String>) {
        let nid = neighbor_id.into();
        self.neighbors.insert(nid.clone());
        // Direct neighbor is 1 hop
        self.routes.insert(nid.clone(), RouteEntry {
            destination: nid.clone(),
            next_hop: nid,
            hop_count: 1,
            last_updated: self.current_tick,
        });
    }

    /// Remove a neighbor.
    pub fn remove_neighbor(&mut self, neighbor_id: &str) {
        self.neighbors.remove(neighbor_id);
        // Remove routes that go through this neighbor
        self.routes.retain(|_, r| r.next_hop != neighbor_id);
    }

    /// Learn a route from an incoming route advertisement.
    pub fn learn_route(&mut self, dest: impl Into<String>, via: impl Into<String>, hops: u32) {
        let d = dest.into();
        let v = via.into();
        let should_update = match self.routes.get(&d) {
            Some(existing) => hops + 1 < existing.hop_count,
            None => true,
        };
        if should_update {
            self.routes.insert(d.clone(), RouteEntry {
                destination: d,
                next_hop: v,
                hop_count: hops + 1,
                last_updated: self.current_tick,
            });
        }
    }

    /// Get the next hop for a destination.
    pub fn next_hop_for(&self, destination: &str) -> Option<&str> {
        self.routes.get(destination).map(|r| r.next_hop.as_str())
    }

    /// Create a new message from this node to a destination.
    pub fn create_message(&mut self, destination: impl Into<String>, payload: Vec<u8>) -> RelayMessage {
        let id = self.next_msg_id;
        self.next_msg_id += 1;
        RelayMessage::new(id, &self.id, destination, payload, self.default_ttl, self.current_tick)
    }

    /// Process an incoming relay message. Returns the next-hop node id if forwarding.
    pub fn process_message(&mut self, mut msg: RelayMessage) -> Result<Option<String>, String> {
        // Deduplication check
        if self.seen_messages.contains(&msg.id) {
            self.stats.messages_dropped_dup += 1;
            return Ok(None);
        }
        self.seen_messages.insert(msg.id);
        // Trim seen cache
        if self.seen_messages.len() > self.seen_limit {
            if let Some(&oldest) = self.seen_messages.iter().next() {
                self.seen_messages.remove(&oldest);
            }
        }

        // TTL check
        if msg.is_expired() {
            self.stats.messages_dropped_ttl += 1;
            return Err("TTL expired".into());
        }

        // Check if we are the destination
        if msg.destination == self.id {
            let hops = msg.hop_count();
            self.stats.messages_delivered += 1;
            self.stats.total_hops += hops as u64;
            msg.record_hop(&self.id);
            self.delivered.push(msg);
            return Ok(None);
        }

        // Forward toward destination
        msg.record_hop(&self.id);
        self.stats.messages_forwarded += 1;

        if let Some(route) = self.routes.get(&msg.destination) {
            let next = route.next_hop.clone();
            self.outbox.push_back(msg);
            Ok(Some(next))
        } else {
            // No route — drop
            self.stats.messages_dropped_ttl += 1;
            Err("no route to destination".into())
        }
    }

    /// Drain the outbox.
    pub fn drain_outbox(&mut self) -> Vec<RelayMessage> {
        self.outbox.drain(..).collect()
    }

    /// Get delivered messages.
    pub fn delivered(&self) -> &[RelayMessage] {
        &self.delivered
    }

    /// Neighbor count.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.len()
    }

    /// Route count.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Statistics.
    pub fn stats(&self) -> &RelayStats {
        &self.stats
    }

    /// Neighbors list.
    pub fn neighbors(&self) -> Vec<&str> {
        self.neighbors.iter().map(|s| s.as_str()).collect()
    }

    /// Setup bidirectional relay by adding each other as neighbor.
    pub fn setup_bidirectional(a: &mut RelayNode, b: &mut RelayNode) {
        a.add_neighbor(&b.id);
        b.add_neighbor(&a.id);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_message_new() {
        let msg = RelayMessage::new(1, "alice", "bob", b"hi".to_vec(), 5, 0);
        assert_eq!(msg.hop_count(), 0);
        assert_eq!(msg.ttl, 5);
        assert!(!msg.is_expired());
    }

    #[test]
    fn test_relay_message_hop() {
        let mut msg = RelayMessage::new(1, "a", "c", vec![], 3, 0);
        msg.record_hop("b");
        assert_eq!(msg.hop_count(), 1);
        assert_eq!(msg.ttl, 2);
    }

    #[test]
    fn test_relay_message_ttl_expire() {
        let mut msg = RelayMessage::new(1, "a", "z", vec![], 1, 0);
        msg.record_hop("b");
        assert!(msg.is_expired());
    }

    #[test]
    fn test_relay_message_display() {
        let msg = RelayMessage::new(42, "src", "dst", vec![], 5, 0);
        let s = format!("{}", msg);
        assert!(s.contains("42"));
        assert!(s.contains("src"));
        assert!(s.contains("dst"));
    }

    #[test]
    fn test_relay_message_reached_dest() {
        let mut msg = RelayMessage::new(1, "a", "b", vec![], 5, 0);
        assert!(!msg.reached_destination());
        msg.record_hop("b");
        assert!(msg.reached_destination());
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = RelayNode::new("a");
        node.add_neighbor("b");
        assert_eq!(node.neighbor_count(), 1);
        assert_eq!(node.route_count(), 1);
    }

    #[test]
    fn test_remove_neighbor() {
        let mut node = RelayNode::new("a");
        node.add_neighbor("b");
        node.remove_neighbor("b");
        assert_eq!(node.neighbor_count(), 0);
        assert_eq!(node.route_count(), 0);
    }

    #[test]
    fn test_next_hop() {
        let mut node = RelayNode::new("a");
        node.add_neighbor("b");
        assert_eq!(node.next_hop_for("b"), Some("b"));
        assert_eq!(node.next_hop_for("unknown"), None);
    }

    #[test]
    fn test_learn_route() {
        let mut node = RelayNode::new("a");
        node.add_neighbor("b");
        node.learn_route("c", "b", 1);
        assert_eq!(node.next_hop_for("c"), Some("b"));
        assert_eq!(node.route_count(), 2);
    }

    #[test]
    fn test_learn_route_shorter_wins() {
        let mut node = RelayNode::new("a");
        node.add_neighbor("b");
        node.learn_route("d", "b", 3);
        node.add_neighbor("c");
        node.learn_route("d", "c", 1);
        assert_eq!(node.next_hop_for("d"), Some("c"));
    }

    #[test]
    fn test_process_message_delivered() {
        let mut node = RelayNode::new("bob");
        let msg = RelayMessage::new(1, "alice", "bob", b"hi".to_vec(), 5, 0);
        let result = node.process_message(msg);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
        assert_eq!(node.delivered().len(), 1);
        assert_eq!(node.stats().messages_delivered, 1);
    }

    #[test]
    fn test_process_message_forwarded() {
        let mut node = RelayNode::new("relay");
        node.add_neighbor("bob");
        let msg = RelayMessage::new(1, "alice", "bob", b"hi".to_vec(), 5, 0);
        let result = node.process_message(msg);
        assert_eq!(result.unwrap(), Some("bob".to_string()));
        assert_eq!(node.stats().messages_forwarded, 1);
    }

    #[test]
    fn test_process_message_dedup() {
        let mut node = RelayNode::new("bob");
        let msg = RelayMessage::new(1, "alice", "bob", b"hi".to_vec(), 5, 0);
        node.process_message(msg.clone()).unwrap();
        let dup = node.process_message(msg);
        assert_eq!(dup.unwrap(), None);
        assert_eq!(node.stats().messages_dropped_dup, 1);
    }

    #[test]
    fn test_process_message_ttl_expired() {
        let mut node = RelayNode::new("relay");
        let msg = RelayMessage::new(1, "a", "z", vec![], 0, 0);
        assert!(node.process_message(msg).is_err());
        assert_eq!(node.stats().messages_dropped_ttl, 1);
    }

    #[test]
    fn test_process_message_no_route() {
        let mut node = RelayNode::new("relay");
        let msg = RelayMessage::new(1, "a", "unknown", vec![], 5, 0);
        assert!(node.process_message(msg).is_err());
    }

    #[test]
    fn test_bidirectional_setup() {
        let mut a = RelayNode::new("a");
        let mut b = RelayNode::new("b");
        RelayNode::setup_bidirectional(&mut a, &mut b);
        assert_eq!(a.neighbor_count(), 1);
        assert_eq!(b.neighbor_count(), 1);
        assert_eq!(a.next_hop_for("b"), Some("b"));
        assert_eq!(b.next_hop_for("a"), Some("a"));
    }

    #[test]
    fn test_create_message() {
        let mut node = RelayNode::new("a");
        let msg = node.create_message("b", b"data".to_vec());
        assert_eq!(msg.source, "a");
        assert_eq!(msg.destination, "b");
    }

    #[test]
    fn test_drain_outbox() {
        let mut node = RelayNode::new("relay");
        node.add_neighbor("bob");
        let msg = RelayMessage::new(1, "alice", "bob", vec![], 5, 0);
        node.process_message(msg).unwrap();
        let out = node.drain_outbox();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_relay_stats_avg_hops() {
        let s = RelayStats { messages_delivered: 4, total_hops: 12, ..Default::default() };
        assert!((s.avg_hops() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_relay_stats_display() {
        let s = RelayStats::default();
        let display = format!("{}", s);
        assert!(display.contains("RelayStats"));
    }
}
