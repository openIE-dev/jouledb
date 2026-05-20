//! Cluster membership — SWIM-like protocol, join/leave, suspicion mechanism,
//! incarnation numbers, indirect probing, state dissemination, membership list,
//! split-brain detection.

use std::collections::{HashMap, HashSet};

// ── Member State ─────────────────────────────────────────────────────────────

/// Health state of a cluster member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemberState {
    /// Node is alive and healthy.
    Alive,
    /// Node is suspected of failure but not confirmed.
    Suspect,
    /// Node has been confirmed dead.
    Dead,
    /// Node has gracefully left the cluster.
    Left,
}

// ── Member Info ──────────────────────────────────────────────────────────────

/// Information about a cluster member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    /// Unique node identifier.
    pub node_id: String,
    /// Network address.
    pub address: String,
    /// Current health state.
    pub state: MemberState,
    /// Incarnation number — monotonically increasing, used to refute suspicion.
    pub incarnation: u64,
    /// Tick of the last acknowledgement from this member.
    pub last_ack_tick: u64,
    /// Tick when suspicion was initiated (if state == Suspect).
    pub suspicion_start_tick: Option<u64>,
    /// Metadata tags (key-value).
    pub metadata: HashMap<String, String>,
}

impl Member {
    /// Create a new alive member.
    pub fn new(node_id: &str, address: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            address: address.to_string(),
            state: MemberState::Alive,
            incarnation: 0,
            last_ack_tick: 0,
            suspicion_start_tick: None,
            metadata: HashMap::new(),
        }
    }
}

// ── Probe Message ────────────────────────────────────────────────────────────

/// Types of membership protocol messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipMessage {
    /// Direct ping to check liveness.
    Ping { from: String, to: String, seq: u64 },
    /// Acknowledgement of a ping.
    Ack { from: String, to: String, seq: u64 },
    /// Request an indirect probe via a relay node.
    PingReq {
        from: String,
        relay: String,
        target: String,
        seq: u64,
    },
    /// Join request.
    Join { node_id: String, address: String },
    /// Graceful leave notification.
    Leave { node_id: String },
    /// State update dissemination.
    StateUpdate {
        node_id: String,
        state: MemberState,
        incarnation: u64,
    },
}

// ── Membership Statistics ────────────────────────────────────────────────────

/// Statistics about the membership list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipStats {
    /// Total number of known members.
    pub total: usize,
    /// Number of alive members.
    pub alive: usize,
    /// Number of suspected members.
    pub suspect: usize,
    /// Number of dead members.
    pub dead: usize,
    /// Number of members that left.
    pub left: usize,
    /// Current tick.
    pub current_tick: u64,
}

// ── Cluster Membership ───────────────────────────────────────────────────────

/// A SWIM-like cluster membership manager.
#[derive(Debug, Clone)]
pub struct Membership {
    /// This node's id.
    node_id: String,
    /// This node's address.
    address: String,
    /// This node's incarnation number.
    incarnation: u64,
    /// Known members.
    members: HashMap<String, Member>,
    /// Current logical tick.
    current_tick: u64,
    /// Number of ticks before a non-responding node becomes suspected.
    ping_timeout: u64,
    /// Number of ticks a node can remain suspected before being declared dead.
    suspicion_timeout: u64,
    /// Number of indirect probe relays to use.
    indirect_probe_count: usize,
    /// Sequence number for ping messages.
    ping_seq: u64,
    /// Pending pings awaiting ack: seq -> (target_node_id, send_tick).
    pending_pings: HashMap<u64, (String, u64)>,
}

impl Membership {
    /// Create a new membership manager.
    pub fn new(node_id: &str, address: &str) -> Self {
        let mut members = HashMap::new();
        members.insert(node_id.to_string(), Member::new(node_id, address));
        Self {
            node_id: node_id.to_string(),
            address: address.to_string(),
            incarnation: 0,
            members,
            current_tick: 0,
            ping_timeout: 5,
            suspicion_timeout: 10,
            indirect_probe_count: 3,
            ping_seq: 0,
            pending_pings: HashMap::new(),
        }
    }

    /// Create with custom timeout configuration.
    pub fn with_config(
        node_id: &str,
        address: &str,
        ping_timeout: u64,
        suspicion_timeout: u64,
        indirect_probe_count: usize,
    ) -> Self {
        let mut m = Self::new(node_id, address);
        m.ping_timeout = ping_timeout;
        m.suspicion_timeout = suspicion_timeout;
        m.indirect_probe_count = indirect_probe_count;
        m
    }

    /// Get this node's id.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get this node's incarnation.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Advance the tick by one and check for timeouts.
    /// Returns messages to send (pings, state updates, etc.).
    pub fn tick(&mut self) -> Vec<MembershipMessage> {
        self.current_tick += 1;
        let mut messages = Vec::new();
        let tick = self.current_tick;

        // Check pending pings for timeouts.
        let timed_out: Vec<u64> = self.pending_pings.iter()
            .filter(|(_, (_, send_tick))| tick - send_tick >= self.ping_timeout)
            .map(|(seq, _)| *seq)
            .collect();

        for seq in &timed_out {
            if let Some((target, _)) = self.pending_pings.remove(seq) {
                // Mark the target as suspected.
                if let Some(member) = self.members.get_mut(&target) {
                    if member.state == MemberState::Alive {
                        member.state = MemberState::Suspect;
                        member.suspicion_start_tick = Some(tick);
                        let incarnation = member.incarnation;
                        messages.push(MembershipMessage::StateUpdate {
                            node_id: target,
                            state: MemberState::Suspect,
                            incarnation,
                        });
                    }
                }
            }
        }

        // Check suspicion timeouts.
        let suspect_timeout = self.suspicion_timeout;
        let dead_nodes: Vec<String> = self.members.values()
            .filter(|m| m.state == MemberState::Suspect)
            .filter(|m| {
                m.suspicion_start_tick
                    .map_or(false, |start| tick - start >= suspect_timeout)
            })
            .map(|m| m.node_id.clone())
            .collect();

        for node_id in dead_nodes {
            if let Some(member) = self.members.get_mut(&node_id) {
                member.state = MemberState::Dead;
                member.suspicion_start_tick = None;
                let incarnation = member.incarnation;
                messages.push(MembershipMessage::StateUpdate {
                    node_id,
                    state: MemberState::Dead,
                    incarnation,
                });
            }
        }

        messages
    }

    /// Handle a join request.
    pub fn handle_join(&mut self, node_id: &str, address: &str) -> Vec<MembershipMessage> {
        let member = Member {
            node_id: node_id.to_string(),
            address: address.to_string(),
            state: MemberState::Alive,
            incarnation: 0,
            last_ack_tick: self.current_tick,
            suspicion_start_tick: None,
            metadata: HashMap::new(),
        };
        self.members.insert(node_id.to_string(), member);

        // Disseminate the join to all alive members.
        vec![MembershipMessage::StateUpdate {
            node_id: node_id.to_string(),
            state: MemberState::Alive,
            incarnation: 0,
        }]
    }

    /// Handle a leave notification.
    pub fn handle_leave(&mut self, node_id: &str) -> Vec<MembershipMessage> {
        if let Some(member) = self.members.get_mut(node_id) {
            member.state = MemberState::Left;
            member.suspicion_start_tick = None;
            let incarnation = member.incarnation;
            return vec![MembershipMessage::StateUpdate {
                node_id: node_id.to_string(),
                state: MemberState::Left,
                incarnation,
            }];
        }
        Vec::new()
    }

    /// Gracefully leave the cluster. Returns a Leave message to broadcast.
    pub fn leave(&mut self) -> MembershipMessage {
        if let Some(member) = self.members.get_mut(&self.node_id) {
            member.state = MemberState::Left;
        }
        MembershipMessage::Leave {
            node_id: self.node_id.clone(),
        }
    }

    /// Send a ping to a target node. Returns the Ping message.
    pub fn ping(&mut self, target: &str) -> MembershipMessage {
        self.ping_seq += 1;
        let seq = self.ping_seq;
        self.pending_pings.insert(seq, (target.to_string(), self.current_tick));
        MembershipMessage::Ping {
            from: self.node_id.clone(),
            to: target.to_string(),
            seq,
        }
    }

    /// Handle a received ping. Returns an Ack.
    pub fn handle_ping(&self, from: &str, seq: u64) -> MembershipMessage {
        MembershipMessage::Ack {
            from: self.node_id.clone(),
            to: from.to_string(),
            seq,
        }
    }

    /// Handle a received ack. Updates the member's last_ack_tick and clears
    /// suspicion if applicable.
    pub fn handle_ack(&mut self, from: &str, seq: u64) {
        self.pending_pings.remove(&seq);
        if let Some(member) = self.members.get_mut(from) {
            member.last_ack_tick = self.current_tick;
            if member.state == MemberState::Suspect {
                member.state = MemberState::Alive;
                member.suspicion_start_tick = None;
            }
        }
    }

    /// Request an indirect probe for a non-responding target.
    /// Picks up to `indirect_probe_count` relay nodes.
    pub fn indirect_probe(&mut self, target: &str) -> Vec<MembershipMessage> {
        let self_id = self.node_id.clone();
        let relays: Vec<String> = self.members.values()
            .filter(|m| {
                m.state == MemberState::Alive
                    && m.node_id != self_id
                    && m.node_id != target
            })
            .take(self.indirect_probe_count)
            .map(|m| m.node_id.clone())
            .collect();

        self.ping_seq += 1;
        let seq = self.ping_seq;
        let mut messages = Vec::new();
        for relay in relays {
            messages.push(MembershipMessage::PingReq {
                from: self.node_id.clone(),
                relay,
                target: target.to_string(),
                seq,
            });
        }
        messages
    }

    /// Handle a state update from another node.
    pub fn handle_state_update(&mut self, node_id: &str, state: MemberState, incarnation: u64) {
        if let Some(member) = self.members.get_mut(node_id) {
            // Only accept updates with higher or equal incarnation.
            if incarnation >= member.incarnation {
                // Alive with higher incarnation trumps Suspect.
                if state == MemberState::Alive && incarnation > member.incarnation {
                    member.state = MemberState::Alive;
                    member.incarnation = incarnation;
                    member.suspicion_start_tick = None;
                } else if incarnation > member.incarnation || state_priority(state) >= state_priority(member.state) {
                    member.state = state;
                    member.incarnation = incarnation;
                    if state != MemberState::Suspect {
                        member.suspicion_start_tick = None;
                    }
                }
            }
        } else {
            // Unknown node — add it.
            let mut member = Member::new(node_id, "unknown");
            member.state = state;
            member.incarnation = incarnation;
            self.members.insert(node_id.to_string(), member);
        }
    }

    /// Refute suspicion about this node by incrementing our incarnation number.
    /// Returns a StateUpdate to broadcast.
    pub fn refute_suspicion(&mut self) -> MembershipMessage {
        self.incarnation += 1;
        if let Some(member) = self.members.get_mut(&self.node_id) {
            member.incarnation = self.incarnation;
            member.state = MemberState::Alive;
            member.suspicion_start_tick = None;
        }
        MembershipMessage::StateUpdate {
            node_id: self.node_id.clone(),
            state: MemberState::Alive,
            incarnation: self.incarnation,
        }
    }

    /// Get a member by id.
    pub fn get_member(&self, node_id: &str) -> Option<&Member> {
        self.members.get(node_id)
    }

    /// Get all alive members (sorted by node_id for determinism).
    pub fn alive_members(&self) -> Vec<&Member> {
        let mut members: Vec<&Member> = self.members.values()
            .filter(|m| m.state == MemberState::Alive)
            .collect();
        members.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        members
    }

    /// Get all members in a specific state.
    pub fn members_in_state(&self, state: MemberState) -> Vec<&Member> {
        let mut members: Vec<&Member> = self.members.values()
            .filter(|m| m.state == state)
            .collect();
        members.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        members
    }

    /// Get the total number of known members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Detect potential split-brain: if the number of alive members is less
    /// than half of total known members, we may have a split-brain.
    pub fn detect_split_brain(&self) -> bool {
        let alive = self.members.values()
            .filter(|m| m.state == MemberState::Alive)
            .count();
        let total = self.members.len();
        if total <= 1 {
            return false;
        }
        alive * 2 < total
    }

    /// Get membership statistics.
    pub fn stats(&self) -> MembershipStats {
        let mut alive = 0;
        let mut suspect = 0;
        let mut dead = 0;
        let mut left = 0;
        for member in self.members.values() {
            match member.state {
                MemberState::Alive => alive += 1,
                MemberState::Suspect => suspect += 1,
                MemberState::Dead => dead += 1,
                MemberState::Left => left += 1,
            }
        }
        MembershipStats {
            total: self.members.len(),
            alive,
            suspect,
            dead,
            left,
            current_tick: self.current_tick,
        }
    }

    /// Set metadata on this node's member entry.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        let nid = self.node_id.clone();
        if let Some(member) = self.members.get_mut(&nid) {
            member.metadata.insert(key.to_string(), value.to_string());
        }
    }

    /// Remove dead and left members from the membership list.
    pub fn prune_dead(&mut self) -> usize {
        let before = self.members.len();
        let self_id = self.node_id.clone();
        self.members.retain(|id, m| {
            *id == self_id || (m.state != MemberState::Dead && m.state != MemberState::Left)
        });
        before - self.members.len()
    }
}

/// Priority ordering for state transitions (higher = more authoritative).
fn state_priority(state: MemberState) -> u8 {
    match state {
        MemberState::Alive => 0,
        MemberState::Suspect => 1,
        MemberState::Dead => 2,
        MemberState::Left => 3,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_membership_has_self() {
        let m = Membership::new("n1", "127.0.0.1:8000");
        assert_eq!(m.member_count(), 1);
        let member = m.get_member("n1").unwrap();
        assert_eq!(member.state, MemberState::Alive);
    }

    #[test]
    fn handle_join() {
        let mut m = Membership::new("n1", "127.0.0.1:8000");
        let msgs = m.handle_join("n2", "127.0.0.1:8001");
        assert_eq!(m.member_count(), 2);
        assert!(!msgs.is_empty());
        let member = m.get_member("n2").unwrap();
        assert_eq!(member.state, MemberState::Alive);
    }

    #[test]
    fn handle_leave() {
        let mut m = Membership::new("n1", "127.0.0.1:8000");
        m.handle_join("n2", "127.0.0.1:8001");
        let msgs = m.handle_leave("n2");
        assert!(!msgs.is_empty());
        let member = m.get_member("n2").unwrap();
        assert_eq!(member.state, MemberState::Left);
    }

    #[test]
    fn graceful_leave() {
        let mut m = Membership::new("n1", "addr");
        let msg = m.leave();
        assert!(matches!(msg, MembershipMessage::Leave { ref node_id } if node_id == "n1"));
        let member = m.get_member("n1").unwrap();
        assert_eq!(member.state, MemberState::Left);
    }

    #[test]
    fn ping_and_ack() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        let ping = m.ping("n2");
        let seq = match &ping {
            MembershipMessage::Ping { seq, .. } => *seq,
            _ => panic!("expected Ping"),
        };
        let ack = m.handle_ping("n2", seq);
        assert!(matches!(ack, MembershipMessage::Ack { .. }));
        m.handle_ack("n2", seq);
        assert!(m.pending_pings.is_empty());
    }

    #[test]
    fn ping_timeout_causes_suspicion() {
        let mut m = Membership::with_config("n1", "addr", 2, 5, 3);
        m.handle_join("n2", "addr2");
        m.ping("n2");
        // Advance ticks past the ping timeout without ack.
        m.tick();
        let msgs = m.tick(); // tick 2 = ping_timeout reached.
        let member = m.get_member("n2").unwrap();
        assert_eq!(member.state, MemberState::Suspect);
        assert!(!msgs.is_empty());
    }

    #[test]
    fn suspicion_timeout_causes_death() {
        let mut m = Membership::with_config("n1", "addr", 1, 3, 3);
        m.handle_join("n2", "addr2");
        m.ping("n2");
        m.tick(); // ping timeout -> suspect.
        m.tick();
        m.tick();
        let msgs = m.tick(); // suspicion timeout -> dead.
        let member = m.get_member("n2").unwrap();
        assert_eq!(member.state, MemberState::Dead);
        let has_dead_update = msgs.iter().any(|msg| matches!(
            msg,
            MembershipMessage::StateUpdate { state: MemberState::Dead, .. }
        ));
        assert!(has_dead_update);
    }

    #[test]
    fn ack_clears_suspicion() {
        let mut m = Membership::with_config("n1", "addr", 1, 10, 3);
        m.handle_join("n2", "addr2");
        let ping = m.ping("n2");
        m.tick(); // Suspect.
        assert_eq!(m.get_member("n2").unwrap().state, MemberState::Suspect);
        // Now receive an ack (with a new ping's seq).
        let new_ping = m.ping("n2");
        let seq = match &new_ping {
            MembershipMessage::Ping { seq, .. } => *seq,
            _ => panic!(),
        };
        m.handle_ack("n2", seq);
        assert_eq!(m.get_member("n2").unwrap().state, MemberState::Alive);
    }

    #[test]
    fn refute_suspicion() {
        let mut m = Membership::new("n1", "addr");
        assert_eq!(m.incarnation(), 0);
        let msg = m.refute_suspicion();
        assert_eq!(m.incarnation(), 1);
        match msg {
            MembershipMessage::StateUpdate { incarnation, state, .. } => {
                assert_eq!(incarnation, 1);
                assert_eq!(state, MemberState::Alive);
            }
            _ => panic!("expected StateUpdate"),
        }
    }

    #[test]
    fn state_update_higher_incarnation() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_state_update("n2", MemberState::Suspect, 0);
        assert_eq!(m.get_member("n2").unwrap().state, MemberState::Suspect);
        // Alive with higher incarnation refutes suspicion.
        m.handle_state_update("n2", MemberState::Alive, 1);
        assert_eq!(m.get_member("n2").unwrap().state, MemberState::Alive);
    }

    #[test]
    fn state_update_lower_incarnation_ignored() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_state_update("n2", MemberState::Alive, 5);
        // Lower incarnation should not override.
        m.handle_state_update("n2", MemberState::Dead, 3);
        assert_eq!(m.get_member("n2").unwrap().state, MemberState::Alive);
    }

    #[test]
    fn indirect_probe() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_join("n3", "addr3");
        m.handle_join("n4", "addr4");
        let msgs = m.indirect_probe("n2");
        assert!(!msgs.is_empty());
        for msg in &msgs {
            match msg {
                MembershipMessage::PingReq { target, .. } => {
                    assert_eq!(target, "n2");
                }
                _ => panic!("expected PingReq"),
            }
        }
    }

    #[test]
    fn alive_members() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_join("n3", "addr3");
        m.handle_leave("n3");
        let alive = m.alive_members();
        assert_eq!(alive.len(), 2);
    }

    #[test]
    fn detect_split_brain() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_join("n3", "addr3");
        m.handle_join("n4", "addr4");
        // All alive: 4/4 => no split brain.
        assert!(!m.detect_split_brain());
        // Kill 3 of 4.
        m.handle_state_update("n2", MemberState::Dead, 0);
        m.handle_state_update("n3", MemberState::Dead, 0);
        m.handle_state_update("n4", MemberState::Dead, 0);
        // 1/4 alive => split brain.
        assert!(m.detect_split_brain());
    }

    #[test]
    fn stats() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_join("n3", "addr3");
        m.handle_leave("n3");
        let s = m.stats();
        assert_eq!(s.total, 3);
        assert_eq!(s.alive, 2);
        assert_eq!(s.left, 1);
    }

    #[test]
    fn set_metadata() {
        let mut m = Membership::new("n1", "addr");
        m.set_metadata("role", "leader");
        let member = m.get_member("n1").unwrap();
        assert_eq!(member.metadata.get("role").unwrap(), "leader");
    }

    #[test]
    fn prune_dead() {
        let mut m = Membership::new("n1", "addr");
        m.handle_join("n2", "addr2");
        m.handle_join("n3", "addr3");
        m.handle_leave("n2");
        m.handle_state_update("n3", MemberState::Dead, 0);
        let pruned = m.prune_dead();
        assert_eq!(pruned, 2);
        assert_eq!(m.member_count(), 1);
    }

    #[test]
    fn handle_unknown_node_state_update() {
        let mut m = Membership::new("n1", "addr");
        m.handle_state_update("n99", MemberState::Alive, 0);
        assert!(m.get_member("n99").is_some());
    }
}
