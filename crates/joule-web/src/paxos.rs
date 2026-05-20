//! Paxos protocol simulation — Prepare/Promise/Accept/Accepted phases, proposal
//! numbers, majority quorum, multi-decree Paxos, leader election, instance log,
//! and message handling.

use std::collections::HashMap;

// ── Proposal Number ──────────────────────────────────────────────────────────

/// A globally unique proposal number combining a round and a node id.
/// Ordering: first by round, then by node_id for tie-breaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProposalNumber {
    /// Monotonically increasing round.
    pub round: u64,
    /// Node id for tie-breaking.
    pub node_id: u64,
}

impl ProposalNumber {
    /// Create a new proposal number.
    pub fn new(round: u64, node_id: u64) -> Self {
        Self { round, node_id }
    }

    /// The "zero" proposal — lower than any real proposal.
    pub fn zero() -> Self {
        Self { round: 0, node_id: 0 }
    }
}

impl PartialOrd for ProposalNumber {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProposalNumber {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.round.cmp(&other.round)
            .then(self.node_id.cmp(&other.node_id))
    }
}

// ── Messages ─────────────────────────────────────────────────────────────────

/// Phase 1a: Prepare request from a proposer.
#[derive(Debug, Clone)]
pub struct Prepare {
    pub instance_id: u64,
    pub proposal: ProposalNumber,
}

/// Phase 1b: Promise response from an acceptor.
#[derive(Debug, Clone)]
pub struct Promise {
    pub instance_id: u64,
    pub proposal: ProposalNumber,
    /// The highest-numbered proposal previously accepted, if any.
    pub accepted_proposal: Option<ProposalNumber>,
    /// The value of the previously accepted proposal, if any.
    pub accepted_value: Option<String>,
    /// True if the promise is granted.
    pub ok: bool,
}

/// Phase 2a: Accept request from a proposer.
#[derive(Debug, Clone)]
pub struct Accept {
    pub instance_id: u64,
    pub proposal: ProposalNumber,
    pub value: String,
}

/// Phase 2b: Accepted response from an acceptor.
#[derive(Debug, Clone)]
pub struct Accepted {
    pub instance_id: u64,
    pub proposal: ProposalNumber,
    pub ok: bool,
}

/// Envelope for all Paxos messages.
#[derive(Debug, Clone)]
pub enum PaxosMessage {
    Prepare(Prepare),
    Promise(Promise),
    Accept(Accept),
    Accepted(Accepted),
}

// ── Instance State ───────────────────────────────────────────────────────────

/// The state of a single Paxos instance (decree) from the perspective of a node.
#[derive(Debug, Clone)]
pub struct InstanceState {
    /// The instance (slot) id.
    pub instance_id: u64,
    /// Highest proposal number promised (as acceptor).
    pub promised: ProposalNumber,
    /// Highest proposal number accepted (as acceptor).
    pub accepted_proposal: ProposalNumber,
    /// Value of the accepted proposal (as acceptor).
    pub accepted_value: Option<String>,
    /// Decided value, if consensus has been reached.
    pub decided_value: Option<String>,
}

impl InstanceState {
    fn new(instance_id: u64) -> Self {
        Self {
            instance_id,
            promised: ProposalNumber::zero(),
            accepted_proposal: ProposalNumber::zero(),
            accepted_value: None,
            decided_value: None,
        }
    }
}

// ── Proposer State ───────────────────────────────────────────────────────────

/// Tracks proposer state for a specific instance.
#[derive(Debug, Clone)]
struct ProposerState {
    proposal: ProposalNumber,
    value: String,
    promises: Vec<Promise>,
    accepted_count: usize,
    phase: ProposerPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposerPhase {
    Preparing,
    Accepting,
    Decided,
}

// ── Paxos Node ───────────────────────────────────────────────────────────────

/// A Paxos node that can act as proposer, acceptor, and learner.
#[derive(Debug, Clone)]
pub struct PaxosNode {
    /// This node's id.
    pub node_id: u64,
    /// Total number of nodes in the cluster.
    pub cluster_size: usize,
    /// Instance log: instance_id -> state.
    instances: HashMap<u64, InstanceState>,
    /// Proposer state per instance.
    proposer_states: HashMap<u64, ProposerState>,
    /// Current round counter for generating proposal numbers.
    current_round: u64,
    /// Whether this node considers itself the leader.
    pub is_leader: bool,
    /// Leader node id (if known).
    pub leader_id: Option<u64>,
    /// Next instance id to use for new proposals.
    next_instance_id: u64,
}

impl PaxosNode {
    /// Create a new Paxos node.
    pub fn new(node_id: u64, cluster_size: usize) -> Self {
        Self {
            node_id,
            cluster_size,
            instances: HashMap::new(),
            proposer_states: HashMap::new(),
            current_round: 0,
            is_leader: false,
            leader_id: None,
            next_instance_id: 1,
        }
    }

    /// Compute the quorum size (majority).
    pub fn quorum_size(&self) -> usize {
        self.cluster_size / 2 + 1
    }

    /// Get or create instance state.
    fn get_instance(&mut self, instance_id: u64) -> &mut InstanceState {
        self.instances
            .entry(instance_id)
            .or_insert_with(|| InstanceState::new(instance_id))
    }

    /// Propose a value. Returns Prepare messages to send to all acceptors.
    pub fn propose(&mut self, value: &str) -> Vec<PaxosMessage> {
        let instance_id = self.next_instance_id;
        self.next_instance_id += 1;
        self.propose_for_instance(instance_id, value)
    }

    /// Propose a value for a specific instance.
    pub fn propose_for_instance(&mut self, instance_id: u64, value: &str) -> Vec<PaxosMessage> {
        self.current_round += 1;
        let proposal = ProposalNumber::new(self.current_round, self.node_id);
        self.proposer_states.insert(instance_id, ProposerState {
            proposal,
            value: value.to_string(),
            promises: Vec::new(),
            accepted_count: 0,
            phase: ProposerPhase::Preparing,
        });
        // Ensure instance state exists.
        let _ = self.get_instance(instance_id);
        // Generate Prepare messages for all nodes.
        vec![PaxosMessage::Prepare(Prepare {
            instance_id,
            proposal,
        })]
    }

    /// Handle a Prepare message (as acceptor). Returns a Promise.
    pub fn handle_prepare(&mut self, prepare: &Prepare) -> PaxosMessage {
        let inst = self.get_instance(prepare.instance_id);
        if prepare.proposal > inst.promised {
            inst.promised = prepare.proposal;
            PaxosMessage::Promise(Promise {
                instance_id: prepare.instance_id,
                proposal: prepare.proposal,
                accepted_proposal: if inst.accepted_proposal > ProposalNumber::zero() {
                    Some(inst.accepted_proposal)
                } else {
                    None
                },
                accepted_value: inst.accepted_value.clone(),
                ok: true,
            })
        } else {
            PaxosMessage::Promise(Promise {
                instance_id: prepare.instance_id,
                proposal: prepare.proposal,
                accepted_proposal: None,
                accepted_value: None,
                ok: false,
            })
        }
    }

    /// Handle a Promise message (as proposer). Returns Accept messages if quorum
    /// reached, or empty vec if still waiting.
    pub fn handle_promise(&mut self, promise: &Promise) -> Vec<PaxosMessage> {
        let quorum = self.quorum_size();
        let ps = match self.proposer_states.get_mut(&promise.instance_id) {
            Some(ps) if ps.phase == ProposerPhase::Preparing => ps,
            _ => return Vec::new(),
        };

        if !promise.ok {
            return Vec::new();
        }

        ps.promises.push(promise.clone());

        if ps.promises.len() >= quorum {
            // Pick the value from the highest-numbered accepted proposal, if any.
            let mut highest: Option<(ProposalNumber, String)> = None;
            for p in &ps.promises {
                if let (Some(ap), Some(av)) = (p.accepted_proposal, p.accepted_value.as_ref()) {
                    match &highest {
                        None => highest = Some((ap, av.clone())),
                        Some((h, _)) if ap > *h => highest = Some((ap, av.clone())),
                        _ => {}
                    }
                }
            }
            let value = match highest {
                Some((_, v)) => v,
                None => ps.value.clone(),
            };
            ps.value = value.clone();
            ps.phase = ProposerPhase::Accepting;
            let proposal = ps.proposal;
            let instance_id = promise.instance_id;
            vec![PaxosMessage::Accept(Accept {
                instance_id,
                proposal,
                value,
            })]
        } else {
            Vec::new()
        }
    }

    /// Handle an Accept message (as acceptor). Returns an Accepted message.
    pub fn handle_accept(&mut self, accept: &Accept) -> PaxosMessage {
        let inst = self.get_instance(accept.instance_id);
        if accept.proposal >= inst.promised {
            inst.promised = accept.proposal;
            inst.accepted_proposal = accept.proposal;
            inst.accepted_value = Some(accept.value.clone());
            PaxosMessage::Accepted(Accepted {
                instance_id: accept.instance_id,
                proposal: accept.proposal,
                ok: true,
            })
        } else {
            PaxosMessage::Accepted(Accepted {
                instance_id: accept.instance_id,
                proposal: accept.proposal,
                ok: false,
            })
        }
    }

    /// Handle an Accepted message (as proposer/learner). Returns true if the
    /// instance has been decided.
    pub fn handle_accepted(&mut self, accepted: &Accepted) -> bool {
        let quorum = self.quorum_size();
        let decided = {
            let ps = match self.proposer_states.get_mut(&accepted.instance_id) {
                Some(ps) if ps.phase == ProposerPhase::Accepting => ps,
                _ => return false,
            };
            if !accepted.ok {
                return false;
            }
            ps.accepted_count += 1;
            if ps.accepted_count >= quorum {
                ps.phase = ProposerPhase::Decided;
                Some(ps.value.clone())
            } else {
                None
            }
        };
        if let Some(value) = decided {
            let inst = self.get_instance(accepted.instance_id);
            inst.decided_value = Some(value);
            true
        } else {
            false
        }
    }

    /// Process any PaxosMessage, dispatching to the appropriate handler.
    pub fn process_message(&mut self, msg: &PaxosMessage) -> Vec<PaxosMessage> {
        match msg {
            PaxosMessage::Prepare(p) => vec![self.handle_prepare(p)],
            PaxosMessage::Promise(p) => self.handle_promise(p),
            PaxosMessage::Accept(a) => vec![self.handle_accept(a)],
            PaxosMessage::Accepted(a) => {
                self.handle_accepted(a);
                Vec::new()
            }
        }
    }

    /// Get the decided value for an instance, if any.
    pub fn decided_value(&self, instance_id: u64) -> Option<&str> {
        self.instances.get(&instance_id)
            .and_then(|i| i.decided_value.as_deref())
    }

    /// Get the instance state for a given instance.
    pub fn instance_state(&self, instance_id: u64) -> Option<&InstanceState> {
        self.instances.get(&instance_id)
    }

    /// Get all decided instances as (instance_id, value) pairs.
    pub fn decided_instances(&self) -> Vec<(u64, &str)> {
        let mut result: Vec<(u64, &str)> = self.instances.iter()
            .filter_map(|(id, inst)| {
                inst.decided_value.as_deref().map(|v| (*id, v))
            })
            .collect();
        result.sort_by_key(|(id, _)| *id);
        result
    }

    /// Get the number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Attempt to become leader. In Multi-Paxos, the leader can skip Phase 1
    /// for subsequent instances. Returns true if leader status is set.
    pub fn become_leader(&mut self) -> bool {
        self.is_leader = true;
        self.leader_id = Some(self.node_id);
        true
    }

    /// Step down from leader.
    pub fn step_down(&mut self) {
        self.is_leader = false;
    }

    /// Set the known leader.
    pub fn set_leader(&mut self, leader_id: u64) {
        self.leader_id = Some(leader_id);
        if leader_id != self.node_id {
            self.is_leader = false;
        }
    }

    /// Get the next instance id that would be used for a new proposal.
    pub fn next_instance_id(&self) -> u64 {
        self.next_instance_id
    }

    /// Manually mark an instance as decided (e.g., learning from another node).
    pub fn learn_value(&mut self, instance_id: u64, value: &str) {
        let inst = self.get_instance(instance_id);
        inst.decided_value = Some(value.to_string());
    }

    /// Check if a proposal number is stale relative to what this node has promised
    /// for the given instance.
    pub fn is_stale(&self, instance_id: u64, proposal: ProposalNumber) -> bool {
        match self.instances.get(&instance_id) {
            Some(inst) => proposal < inst.promised,
            None => false,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_number_ordering() {
        let p1 = ProposalNumber::new(1, 1);
        let p2 = ProposalNumber::new(1, 2);
        let p3 = ProposalNumber::new(2, 1);
        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(ProposalNumber::zero() < p1);
    }

    #[test]
    fn proposal_number_equality() {
        let a = ProposalNumber::new(5, 3);
        let b = ProposalNumber::new(5, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn quorum_size_odd_cluster() {
        let node = PaxosNode::new(1, 3);
        assert_eq!(node.quorum_size(), 2);
    }

    #[test]
    fn quorum_size_even_cluster() {
        let node = PaxosNode::new(1, 4);
        assert_eq!(node.quorum_size(), 3);
    }

    #[test]
    fn single_instance_consensus() {
        let mut proposer = PaxosNode::new(1, 3);
        let mut acceptor1 = PaxosNode::new(2, 3);
        let mut acceptor2 = PaxosNode::new(3, 3);

        // Phase 1: Propose.
        let prepares = proposer.propose("hello");
        assert_eq!(prepares.len(), 1);

        // Acceptors handle Prepare, return Promise.
        let promise1 = acceptor1.process_message(&prepares[0]);
        let promise2 = acceptor2.process_message(&prepares[0]);
        assert_eq!(promise1.len(), 1);
        assert_eq!(promise2.len(), 1);

        // Proposer collects Promises — quorum after 2.
        let accepts1 = proposer.process_message(&promise1[0]);
        assert!(accepts1.is_empty()); // Not yet quorum.
        let accepts2 = proposer.process_message(&promise2[0]);
        assert_eq!(accepts2.len(), 1); // Quorum reached, Accept sent.

        // Phase 2: Acceptors handle Accept.
        let accepted1 = acceptor1.process_message(&accepts2[0]);
        let accepted2 = acceptor2.process_message(&accepts2[0]);
        assert_eq!(accepted1.len(), 1);
        assert_eq!(accepted2.len(), 1);

        // Proposer collects Accepted.
        proposer.process_message(&accepted1[0]);
        let decided = proposer.handle_accepted(match &accepted2[0] {
            PaxosMessage::Accepted(a) => a,
            _ => panic!("expected Accepted"),
        });
        assert!(decided);
        assert_eq!(proposer.decided_value(1), Some("hello"));
    }

    #[test]
    fn promise_rejected_for_lower_proposal() {
        let mut acceptor = PaxosNode::new(1, 3);
        // First prepare with high proposal.
        let high = Prepare {
            instance_id: 1,
            proposal: ProposalNumber::new(10, 1),
        };
        acceptor.handle_prepare(&high);
        // Second prepare with lower proposal should be rejected.
        let low = Prepare {
            instance_id: 1,
            proposal: ProposalNumber::new(5, 2),
        };
        let response = acceptor.handle_prepare(&low);
        match response {
            PaxosMessage::Promise(p) => assert!(!p.ok),
            _ => panic!("expected Promise"),
        }
    }

    #[test]
    fn accept_rejected_for_lower_proposal() {
        let mut acceptor = PaxosNode::new(1, 3);
        // Promise a high proposal.
        let prep = Prepare {
            instance_id: 1,
            proposal: ProposalNumber::new(10, 1),
        };
        acceptor.handle_prepare(&prep);
        // Try to accept with a lower proposal.
        let accept = Accept {
            instance_id: 1,
            proposal: ProposalNumber::new(5, 2),
            value: "old".into(),
        };
        let response = acceptor.handle_accept(&accept);
        match response {
            PaxosMessage::Accepted(a) => assert!(!a.ok),
            _ => panic!("expected Accepted"),
        }
    }

    #[test]
    fn promise_carries_previously_accepted_value() {
        let mut acceptor = PaxosNode::new(1, 3);
        // Accept a value at proposal (1,1).
        let accept = Accept {
            instance_id: 1,
            proposal: ProposalNumber::new(1, 1),
            value: "first".into(),
        };
        acceptor.handle_accept(&accept);
        // New prepare at (2,2).
        let prep = Prepare {
            instance_id: 1,
            proposal: ProposalNumber::new(2, 2),
        };
        let response = acceptor.handle_prepare(&prep);
        match response {
            PaxosMessage::Promise(p) => {
                assert!(p.ok);
                assert_eq!(p.accepted_proposal, Some(ProposalNumber::new(1, 1)));
                assert_eq!(p.accepted_value.as_deref(), Some("first"));
            }
            _ => panic!("expected Promise"),
        }
    }

    #[test]
    fn multi_decree_separate_instances() {
        let mut proposer = PaxosNode::new(1, 3);
        let p1 = proposer.propose("value_a");
        let p2 = proposer.propose("value_b");
        // They should target different instances.
        let inst1 = match &p1[0] {
            PaxosMessage::Prepare(p) => p.instance_id,
            _ => panic!(),
        };
        let inst2 = match &p2[0] {
            PaxosMessage::Prepare(p) => p.instance_id,
            _ => panic!(),
        };
        assert_ne!(inst1, inst2);
    }

    #[test]
    fn proposer_adopts_highest_accepted_value() {
        let mut proposer = PaxosNode::new(1, 3);
        let prepares = proposer.propose("my_value");

        // Simulate two promises, one carrying a previously accepted value.
        let promise_no_val = Promise {
            instance_id: 1,
            proposal: match &prepares[0] {
                PaxosMessage::Prepare(p) => p.proposal,
                _ => panic!(),
            },
            accepted_proposal: None,
            accepted_value: None,
            ok: true,
        };
        let promise_with_val = Promise {
            instance_id: 1,
            proposal: match &prepares[0] {
                PaxosMessage::Prepare(p) => p.proposal,
                _ => panic!(),
            },
            accepted_proposal: Some(ProposalNumber::new(5, 2)),
            accepted_value: Some("adopted".into()),
            ok: true,
        };

        proposer.handle_promise(&promise_no_val);
        let accepts = proposer.handle_promise(&promise_with_val);
        assert_eq!(accepts.len(), 1);
        match &accepts[0] {
            PaxosMessage::Accept(a) => assert_eq!(a.value, "adopted"),
            _ => panic!("expected Accept"),
        }
    }

    #[test]
    fn leader_election() {
        let mut node = PaxosNode::new(1, 3);
        assert!(!node.is_leader);
        node.become_leader();
        assert!(node.is_leader);
        assert_eq!(node.leader_id, Some(1));
        node.step_down();
        assert!(!node.is_leader);
    }

    #[test]
    fn set_leader_from_remote() {
        let mut node = PaxosNode::new(2, 3);
        node.set_leader(1);
        assert_eq!(node.leader_id, Some(1));
        assert!(!node.is_leader);
    }

    #[test]
    fn learn_value() {
        let mut node = PaxosNode::new(1, 3);
        node.learn_value(5, "learned");
        assert_eq!(node.decided_value(5), Some("learned"));
    }

    #[test]
    fn decided_instances() {
        let mut node = PaxosNode::new(1, 3);
        node.learn_value(3, "c");
        node.learn_value(1, "a");
        node.learn_value(2, "b");
        let decided = node.decided_instances();
        assert_eq!(decided.len(), 3);
        assert_eq!(decided[0], (1, "a"));
        assert_eq!(decided[1], (2, "b"));
        assert_eq!(decided[2], (3, "c"));
    }

    #[test]
    fn is_stale_check() {
        let mut node = PaxosNode::new(1, 3);
        let prep = Prepare {
            instance_id: 1,
            proposal: ProposalNumber::new(10, 1),
        };
        node.handle_prepare(&prep);
        assert!(node.is_stale(1, ProposalNumber::new(5, 1)));
        assert!(!node.is_stale(1, ProposalNumber::new(10, 1)));
        assert!(!node.is_stale(1, ProposalNumber::new(15, 1)));
        // Unknown instance is never stale.
        assert!(!node.is_stale(99, ProposalNumber::new(1, 1)));
    }

    #[test]
    fn instance_count() {
        let mut node = PaxosNode::new(1, 3);
        assert_eq!(node.instance_count(), 0);
        node.propose("x");
        assert_eq!(node.instance_count(), 1);
        node.propose("y");
        assert_eq!(node.instance_count(), 2);
    }

    #[test]
    fn next_instance_id_advances() {
        let mut node = PaxosNode::new(1, 3);
        assert_eq!(node.next_instance_id(), 1);
        node.propose("a");
        assert_eq!(node.next_instance_id(), 2);
        node.propose("b");
        assert_eq!(node.next_instance_id(), 3);
    }

    #[test]
    fn propose_for_specific_instance() {
        let mut node = PaxosNode::new(1, 3);
        let msgs = node.propose_for_instance(42, "value_42");
        match &msgs[0] {
            PaxosMessage::Prepare(p) => assert_eq!(p.instance_id, 42),
            _ => panic!("expected Prepare"),
        }
        assert!(node.instance_state(42).is_some());
    }

    #[test]
    fn full_three_node_consensus_end_to_end() {
        let mut nodes = vec![
            PaxosNode::new(0, 3),
            PaxosNode::new(1, 3),
            PaxosNode::new(2, 3),
        ];

        // Node 0 proposes "world".
        let prepares = nodes[0].propose("world");
        let prepare = match &prepares[0] {
            PaxosMessage::Prepare(p) => p.clone(),
            _ => panic!(),
        };

        // All 3 acceptors handle Prepare.
        let mut promises = Vec::new();
        for i in 0..3 {
            let resp = nodes[i].handle_prepare(&prepare);
            promises.push(resp);
        }

        // Feed promises to proposer (node 0).
        let mut accept_msgs = Vec::new();
        for p in &promises {
            let r = nodes[0].process_message(p);
            accept_msgs.extend(r);
        }
        assert!(!accept_msgs.is_empty());

        let accept = match &accept_msgs[0] {
            PaxosMessage::Accept(a) => a.clone(),
            _ => panic!(),
        };

        // All 3 acceptors handle Accept.
        let mut accepted_msgs = Vec::new();
        for i in 0..3 {
            let resp = nodes[i].handle_accept(&accept);
            accepted_msgs.push(resp);
        }

        // Feed accepted to proposer (node 0).
        let mut decided = false;
        for a in &accepted_msgs {
            match a {
                PaxosMessage::Accepted(acc) => {
                    if nodes[0].handle_accepted(acc) {
                        decided = true;
                    }
                }
                _ => panic!(),
            }
        }
        assert!(decided);
        assert_eq!(nodes[0].decided_value(1), Some("world"));
    }
}
