//! Gossip/epidemic protocol — membership, push/pull gossip, infection-style
//! dissemination, failure detection, suspicion mechanism, message deduplication,
//! and convergence tracking.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Member State ─────────────────────────────────────────────────────────────

/// Health state of a cluster member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberState {
    Alive,
    Suspected,
    Dead,
}

// ── Member Info ──────────────────────────────────────────────────────────────

/// Information about a cluster member.
#[derive(Debug, Clone)]
pub struct MemberInfo {
    /// Unique node identifier.
    pub node_id: String,
    /// Address (e.g., "host:port").
    pub address: String,
    /// Current state.
    pub state: MemberState,
    /// Incarnation number (monotonically increasing per node).
    pub incarnation: u64,
    /// Tick of last heartbeat received.
    pub last_heartbeat: u64,
    /// Tick when suspicion started (if suspected).
    pub suspicion_start: Option<u64>,
}

// ── Gossip Message ───────────────────────────────────────────────────────────

/// A gossip message carrying membership updates.
#[derive(Debug, Clone)]
pub struct GossipMessage {
    /// Unique message id for deduplication.
    pub msg_id: u64,
    /// Sender's node id.
    pub sender: String,
    /// Membership updates in this message.
    pub updates: Vec<MemberUpdate>,
    /// Message generation (for convergence tracking).
    pub generation: u64,
}

/// A single membership update.
#[derive(Debug, Clone)]
pub struct MemberUpdate {
    pub node_id: String,
    pub address: String,
    pub state: MemberState,
    pub incarnation: u64,
}

// ── Gossip Node ──────────────────────────────────────────────────────────────

/// A gossip protocol node managing cluster membership.
#[derive(Debug, Clone)]
pub struct GossipNode {
    /// This node's id.
    pub node_id: String,
    /// This node's address.
    pub address: String,
    /// Incarnation number for this node.
    pub incarnation: u64,
    /// Known members.
    pub members: HashMap<String, MemberInfo>,
    /// Current logical tick (for heartbeat tracking).
    pub current_tick: u64,
    /// Heartbeat timeout in ticks before marking as suspected.
    pub heartbeat_timeout: u64,
    /// Suspicion timeout in ticks before marking as dead.
    pub suspicion_timeout: u64,
    /// Seen message ids for deduplication.
    seen_messages: HashSet<u64>,
    /// Pending outgoing messages.
    outbox: VecDeque<GossipMessage>,
    /// Next message id.
    next_msg_id: u64,
    /// Messages received count (for convergence).
    pub messages_received: u64,
    /// Messages sent count.
    pub messages_sent: u64,
    /// Generation counter (incremented on each gossip round).
    pub generation: u64,
    /// Number of nodes that have converged (seen all updates).
    convergence_tracker: HashMap<String, u64>,
}

impl GossipNode {
    /// Create a new gossip node.
    pub fn new(
        node_id: &str,
        address: &str,
        heartbeat_timeout: u64,
        suspicion_timeout: u64,
    ) -> Self {
        let mut members = HashMap::new();
        members.insert(
            node_id.to_string(),
            MemberInfo {
                node_id: node_id.to_string(),
                address: address.to_string(),
                state: MemberState::Alive,
                incarnation: 0,
                last_heartbeat: 0,
                suspicion_start: None,
            },
        );

        Self {
            node_id: node_id.to_string(),
            address: address.to_string(),
            incarnation: 0,
            members,
            current_tick: 0,
            heartbeat_timeout,
            suspicion_timeout,
            seen_messages: HashSet::new(),
            outbox: VecDeque::new(),
            next_msg_id: 0,
            messages_received: 0,
            messages_sent: 0,
            generation: 0,
            convergence_tracker: HashMap::new(),
        }
    }

    /// Number of known members (alive + suspected).
    pub fn member_count(&self) -> usize {
        self.members
            .values()
            .filter(|m| m.state != MemberState::Dead)
            .count()
    }

    /// All known alive members.
    pub fn alive_members(&self) -> Vec<&MemberInfo> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Alive)
            .collect()
    }

    /// All suspected members.
    pub fn suspected_members(&self) -> Vec<&MemberInfo> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Suspected)
            .collect()
    }

    /// All dead members.
    pub fn dead_members(&self) -> Vec<&MemberInfo> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Dead)
            .collect()
    }

    /// Join the cluster by adding a seed node.
    pub fn join(&mut self, seed_id: &str, seed_address: &str) {
        self.members.entry(seed_id.to_string()).or_insert(MemberInfo {
            node_id: seed_id.to_string(),
            address: seed_address.to_string(),
            state: MemberState::Alive,
            incarnation: 0,
            last_heartbeat: self.current_tick,
            suspicion_start: None,
        });
    }

    /// Advance the logical clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Perform failure detection: check heartbeats and advance suspicions.
    pub fn detect_failures(&mut self) -> Vec<String> {
        let mut changes = Vec::new();
        let current = self.current_tick;
        let hb_timeout = self.heartbeat_timeout;
        let sus_timeout = self.suspicion_timeout;
        let self_id = self.node_id.clone();

        let node_ids: Vec<String> = self.members.keys().cloned().collect();
        for node_id in node_ids {
            if node_id == self_id {
                continue;
            }
            let member = self.members.get_mut(&node_id).unwrap();
            match member.state {
                MemberState::Alive => {
                    if current > member.last_heartbeat + hb_timeout {
                        member.state = MemberState::Suspected;
                        member.suspicion_start = Some(current);
                        changes.push(node_id);
                    }
                }
                MemberState::Suspected => {
                    if let Some(start) = member.suspicion_start {
                        if current > start + sus_timeout {
                            member.state = MemberState::Dead;
                            changes.push(node_id);
                        }
                    }
                }
                MemberState::Dead => {}
            }
        }
        changes
    }

    /// Create a push gossip message with this node's membership knowledge.
    pub fn create_push_message(&mut self) -> GossipMessage {
        self.generation += 1;
        let updates: Vec<MemberUpdate> = self
            .members
            .values()
            .map(|m| MemberUpdate {
                node_id: m.node_id.clone(),
                address: m.address.clone(),
                state: m.state,
                incarnation: m.incarnation,
            })
            .collect();

        let msg_id = self.next_msg_id;
        self.next_msg_id += 1;
        self.seen_messages.insert(msg_id);
        self.messages_sent += 1;

        GossipMessage {
            msg_id,
            sender: self.node_id.clone(),
            updates,
            generation: self.generation,
        }
    }

    /// Create a pull request (empty message asking for updates).
    pub fn create_pull_request(&mut self) -> GossipMessage {
        let msg_id = self.next_msg_id;
        self.next_msg_id += 1;
        self.seen_messages.insert(msg_id);
        self.messages_sent += 1;

        GossipMessage {
            msg_id,
            sender: self.node_id.clone(),
            updates: Vec::new(),
            generation: self.generation,
        }
    }

    /// Respond to a pull request with a push message.
    pub fn respond_to_pull(&mut self, _request: &GossipMessage) -> GossipMessage {
        self.create_push_message()
    }

    /// Handle a received gossip message. Returns true if new information was learned.
    pub fn handle_message(&mut self, msg: &GossipMessage) -> bool {
        // Deduplication
        if self.seen_messages.contains(&msg.msg_id) {
            return false;
        }
        self.seen_messages.insert(msg.msg_id);
        self.messages_received += 1;

        // Track convergence
        self.convergence_tracker
            .insert(msg.sender.clone(), msg.generation);

        let mut learned = false;

        for update in &msg.updates {
            let should_apply = match self.members.get(&update.node_id) {
                None => true, // New member
                Some(existing) => {
                    // Apply if higher incarnation, or same incarnation with state progression
                    if update.incarnation > existing.incarnation {
                        true
                    } else if update.incarnation == existing.incarnation {
                        state_priority(update.state) > state_priority(existing.state)
                    } else {
                        false
                    }
                }
            };

            if should_apply {
                let heartbeat_tick = if update.state == MemberState::Alive {
                    self.current_tick
                } else {
                    self.members
                        .get(&update.node_id)
                        .map(|m| m.last_heartbeat)
                        .unwrap_or(self.current_tick)
                };

                self.members.insert(
                    update.node_id.clone(),
                    MemberInfo {
                        node_id: update.node_id.clone(),
                        address: update.address.clone(),
                        state: update.state,
                        incarnation: update.incarnation,
                        last_heartbeat: heartbeat_tick,
                        suspicion_start: if update.state == MemberState::Suspected {
                            Some(self.current_tick)
                        } else {
                            None
                        },
                    },
                );
                learned = true;
            }
        }

        // Update sender as alive
        if let Some(sender_info) = self.members.get_mut(&msg.sender) {
            sender_info.last_heartbeat = self.current_tick;
            if sender_info.state == MemberState::Suspected {
                sender_info.state = MemberState::Alive;
                sender_info.suspicion_start = None;
                learned = true;
            }
        }

        learned
    }

    /// Refute a suspicion about this node by incrementing incarnation.
    pub fn refute_suspicion(&mut self) -> MemberUpdate {
        self.incarnation += 1;
        if let Some(info) = self.members.get_mut(&self.node_id) {
            info.incarnation = self.incarnation;
            info.state = MemberState::Alive;
            info.suspicion_start = None;
        }
        MemberUpdate {
            node_id: self.node_id.clone(),
            address: self.address.clone(),
            state: MemberState::Alive,
            incarnation: self.incarnation,
        }
    }

    /// Queue an outgoing message.
    pub fn enqueue_message(&mut self, msg: GossipMessage) {
        self.outbox.push_back(msg);
    }

    /// Drain outgoing messages.
    pub fn drain_outbox(&mut self) -> Vec<GossipMessage> {
        self.outbox.drain(..).collect()
    }

    /// Select random peers to gossip with (returns up to `count` peer ids).
    /// Uses a simple round-robin-like selection for determinism in simulation.
    pub fn select_peers(&self, count: usize) -> Vec<String> {
        let mut peers: Vec<String> = self
            .members
            .values()
            .filter(|m| m.node_id != self.node_id && m.state != MemberState::Dead)
            .map(|m| m.node_id.clone())
            .collect();
        peers.sort();
        peers.truncate(count);
        peers
    }

    /// Check convergence: fraction of known alive members that have been heard from
    /// within the last `window` generations.
    pub fn convergence_ratio(&self) -> f64 {
        let alive: Vec<&String> = self
            .members
            .keys()
            .filter(|k| {
                let m = &self.members[*k];
                m.state == MemberState::Alive && m.node_id != self.node_id
            })
            .collect();

        if alive.is_empty() {
            return 1.0;
        }

        let heard_from = alive
            .iter()
            .filter(|id| self.convergence_tracker.contains_key(**id))
            .count();

        heard_from as f64 / alive.len() as f64
    }

    /// Number of deduplication entries (seen message ids).
    pub fn seen_count(&self) -> usize {
        self.seen_messages.len()
    }

    /// Prune dead members older than a given number of ticks.
    pub fn prune_dead(&mut self, max_age: u64) {
        let cutoff = self.current_tick.saturating_sub(max_age);
        self.members
            .retain(|id, m| *id == self.node_id || m.state != MemberState::Dead || m.last_heartbeat >= cutoff);
    }
}

/// State priority for conflict resolution (higher = more authoritative).
fn state_priority(state: MemberState) -> u8 {
    match state {
        MemberState::Alive => 0,
        MemberState::Suspected => 1,
        MemberState::Dead => 2,
    }
}

// ── Gossip Simulator ─────────────────────────────────────────────────────────

/// Simulates a gossip cluster for testing convergence.
#[derive(Debug)]
pub struct GossipCluster {
    pub nodes: HashMap<String, GossipNode>,
}

impl GossipCluster {
    /// Create a cluster with the given node ids and addresses.
    pub fn new(
        node_configs: &[(&str, &str)],
        heartbeat_timeout: u64,
        suspicion_timeout: u64,
    ) -> Self {
        let mut nodes = HashMap::new();
        for (id, addr) in node_configs {
            let mut node = GossipNode::new(id, addr, heartbeat_timeout, suspicion_timeout);
            // Each node knows about all others initially
            for (other_id, other_addr) in node_configs {
                if *other_id != *id {
                    node.join(other_id, other_addr);
                }
            }
            nodes.insert(id.to_string(), node);
        }
        Self { nodes }
    }

    /// Run one round of push gossip: each node sends to one peer.
    pub fn gossip_round(&mut self) {
        // Collect messages
        let mut messages: Vec<(String, GossipMessage)> = Vec::new();
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        for node_id in &node_ids {
            let node = self.nodes.get_mut(node_id).unwrap();
            node.tick();
            let peers = node.select_peers(1);
            if let Some(peer_id) = peers.first() {
                let msg = node.create_push_message();
                messages.push((peer_id.clone(), msg));
            }
        }

        // Deliver messages
        for (target, msg) in messages {
            if let Some(node) = self.nodes.get_mut(&target) {
                node.handle_message(&msg);
            }
        }
    }

    /// Run failure detection on all nodes.
    pub fn detect_failures(&mut self) {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for node_id in node_ids {
            let node = self.nodes.get_mut(&node_id).unwrap();
            node.detect_failures();
        }
    }

    /// Check if all nodes agree on membership.
    pub fn is_converged(&self) -> bool {
        let node_ids: Vec<&String> = self.nodes.keys().collect();
        if node_ids.len() < 2 {
            return true;
        }
        let first = &self.nodes[node_ids[0]];
        let first_alive: HashSet<String> = first
            .alive_members()
            .iter()
            .map(|m| m.node_id.clone())
            .collect();

        for id in &node_ids[1..] {
            let node = &self.nodes[*id];
            let node_alive: HashSet<String> = node
                .alive_members()
                .iter()
                .map(|m| m.node_id.clone())
                .collect();
            if node_alive != first_alive {
                return false;
            }
        }
        true
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node() {
        let node = GossipNode::new("node1", "localhost:8001", 5, 10);
        assert_eq!(node.node_id, "node1");
        assert_eq!(node.member_count(), 1);
        assert_eq!(node.alive_members().len(), 1);
    }

    #[test]
    fn test_join() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        node.join("node2", "localhost:8002");
        assert_eq!(node.member_count(), 2);
    }

    #[test]
    fn test_join_idempotent() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        node.join("node2", "localhost:8002");
        node.join("node2", "localhost:8002");
        assert_eq!(node.member_count(), 2);
    }

    #[test]
    fn test_create_push_message() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        node.join("node2", "localhost:8002");
        let msg = node.create_push_message();
        assert_eq!(msg.sender, "node1");
        assert_eq!(msg.updates.len(), 2);
        assert_eq!(node.messages_sent, 1);
    }

    #[test]
    fn test_handle_message_new_member() {
        let mut node1 = GossipNode::new("node1", "localhost:8001", 5, 10);
        let mut node2 = GossipNode::new("node2", "localhost:8002", 5, 10);
        node2.join("node3", "localhost:8003");
        let msg = node2.create_push_message();
        let learned = node1.handle_message(&msg);
        assert!(learned);
        assert!(node1.members.contains_key("node3"));
    }

    #[test]
    fn test_message_deduplication() {
        let mut node1 = GossipNode::new("node1", "localhost:8001", 5, 10);
        let mut node2 = GossipNode::new("node2", "localhost:8002", 5, 10);
        let msg = node2.create_push_message();
        node1.handle_message(&msg);
        let learned = node1.handle_message(&msg);
        assert!(!learned);
        assert_eq!(node1.messages_received, 1);
    }

    #[test]
    fn test_failure_detection_suspicion() {
        let mut node = GossipNode::new("node1", "localhost:8001", 3, 5);
        node.join("node2", "localhost:8002");
        // Advance ticks past heartbeat timeout
        for _ in 0..5 {
            node.tick();
        }
        let changes = node.detect_failures();
        assert!(!changes.is_empty());
        assert_eq!(node.suspected_members().len(), 1);
    }

    #[test]
    fn test_failure_detection_dead() {
        let mut node = GossipNode::new("node1", "localhost:8001", 2, 3);
        node.join("node2", "localhost:8002");
        // Advance past heartbeat timeout to suspect
        for _ in 0..4 {
            node.tick();
        }
        node.detect_failures();
        assert_eq!(node.suspected_members().len(), 1);
        // Advance past suspicion timeout
        for _ in 0..5 {
            node.tick();
        }
        node.detect_failures();
        assert_eq!(node.dead_members().len(), 1);
        assert_eq!(node.suspected_members().len(), 0);
    }

    #[test]
    fn test_refute_suspicion() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        let update = node.refute_suspicion();
        assert_eq!(update.incarnation, 1);
        assert_eq!(update.state, MemberState::Alive);
        assert_eq!(node.incarnation, 1);
    }

    #[test]
    fn test_incarnation_override() {
        let mut node1 = GossipNode::new("node1", "localhost:8001", 5, 10);
        node1.join("node2", "localhost:8002");

        // Simulate receiving a higher incarnation
        let msg = GossipMessage {
            msg_id: 999,
            sender: "node2".to_string(),
            updates: vec![MemberUpdate {
                node_id: "node2".to_string(),
                address: "localhost:8002".to_string(),
                state: MemberState::Alive,
                incarnation: 5,
            }],
            generation: 1,
        };
        node1.handle_message(&msg);
        let member = &node1.members["node2"];
        assert_eq!(member.incarnation, 5);
    }

    #[test]
    fn test_select_peers() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        node.join("node2", "localhost:8002");
        node.join("node3", "localhost:8003");
        let peers = node.select_peers(2);
        assert_eq!(peers.len(), 2);
        assert!(!peers.contains(&"node1".to_string()));
    }

    #[test]
    fn test_outbox() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        let msg = node.create_push_message();
        node.enqueue_message(msg);
        let drained = node.drain_outbox();
        assert_eq!(drained.len(), 1);
        assert_eq!(node.drain_outbox().len(), 0);
    }

    #[test]
    fn test_convergence_ratio() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        // No peers = perfect convergence
        assert_eq!(node.convergence_ratio(), 1.0);
        node.join("node2", "localhost:8002");
        // Haven't heard from node2 yet
        assert_eq!(node.convergence_ratio(), 0.0);
    }

    #[test]
    fn test_cluster_creation() {
        let cluster = GossipCluster::new(
            &[("n1", "addr1"), ("n2", "addr2"), ("n3", "addr3")],
            5,
            10,
        );
        assert_eq!(cluster.nodes.len(), 3);
        for (_, node) in &cluster.nodes {
            assert_eq!(node.member_count(), 3);
        }
    }

    #[test]
    fn test_cluster_gossip_round() {
        let mut cluster = GossipCluster::new(
            &[("n1", "addr1"), ("n2", "addr2"), ("n3", "addr3")],
            5,
            10,
        );
        cluster.gossip_round();
        // After a round, messages should have been exchanged
        let total_sent: u64 = cluster.nodes.values().map(|n| n.messages_sent).sum();
        assert!(total_sent > 0);
    }

    #[test]
    fn test_prune_dead() {
        let mut node = GossipNode::new("node1", "localhost:8001", 1, 1);
        node.join("node2", "localhost:8002");
        // Force node2 to dead state:
        // Advance past heartbeat_timeout so detect_failures suspects node2.
        for _ in 0..5 {
            node.tick();
        }
        node.detect_failures(); // node2: Alive -> Suspected at tick 5
        // Advance past suspicion_timeout so next detect_failures marks node2 dead.
        for _ in 0..5 {
            node.tick();
        }
        node.detect_failures(); // node2: Suspected -> Dead (tick 10 > 5 + 1)
        assert_eq!(node.dead_members().len(), 1);
        node.prune_dead(5);
        // Should have been pruned (last_heartbeat was tick 0, current is 10)
        assert!(!node.members.contains_key("node2"));
    }

    #[test]
    fn test_pull_request() {
        let mut node1 = GossipNode::new("node1", "localhost:8001", 5, 10);
        let mut node2 = GossipNode::new("node2", "localhost:8002", 5, 10);
        node2.join("node3", "localhost:8003");

        let pull = node1.create_pull_request();
        assert!(pull.updates.is_empty());

        let response = node2.respond_to_pull(&pull);
        assert!(!response.updates.is_empty());
    }

    #[test]
    fn test_seen_count() {
        let mut node = GossipNode::new("node1", "localhost:8001", 5, 10);
        assert_eq!(node.seen_count(), 0);
        node.create_push_message();
        assert_eq!(node.seen_count(), 1);
    }

    #[test]
    fn test_state_priority() {
        assert!(state_priority(MemberState::Dead) > state_priority(MemberState::Suspected));
        assert!(state_priority(MemberState::Suspected) > state_priority(MemberState::Alive));
    }
}
