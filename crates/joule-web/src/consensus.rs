//! Distributed consensus simulation — Raft-like leader election, log replication,
//! commit index tracking, state machine, and cluster membership.

use std::collections::{HashMap, HashSet};

// ── Node State ───────────────────────────────────────────────────────────────

/// Raft node role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Follower,
    Candidate,
    Leader,
}

// ── Log Entry ────────────────────────────────────────────────────────────────

/// A replicated log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Term in which the entry was created.
    pub term: u64,
    /// Index in the log (1-based).
    pub index: u64,
    /// Command payload.
    pub command: String,
}

// ── Vote Request / Response ──────────────────────────────────────────────────

/// RequestVote RPC arguments.
#[derive(Debug, Clone)]
pub struct VoteRequest {
    /// Candidate's term.
    pub term: u64,
    /// Candidate's id.
    pub candidate_id: u64,
    /// Index of candidate's last log entry.
    pub last_log_index: u64,
    /// Term of candidate's last log entry.
    pub last_log_term: u64,
}

/// RequestVote RPC response.
#[derive(Debug, Clone)]
pub struct VoteResponse {
    /// Current term of the responder (for candidate to update).
    pub term: u64,
    /// True if vote was granted.
    pub vote_granted: bool,
}

// ── Append Entries Request / Response ────────────────────────────────────────

/// AppendEntries RPC arguments.
#[derive(Debug, Clone)]
pub struct AppendEntriesRequest {
    /// Leader's term.
    pub term: u64,
    /// Leader's id.
    pub leader_id: u64,
    /// Index of log entry immediately preceding new entries.
    pub prev_log_index: u64,
    /// Term of prev_log_index entry.
    pub prev_log_term: u64,
    /// Log entries to store (empty for heartbeat).
    pub entries: Vec<LogEntry>,
    /// Leader's commit index.
    pub leader_commit: u64,
}

/// AppendEntries RPC response.
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    /// Current term of the responder.
    pub term: u64,
    /// True if entries were accepted.
    pub success: bool,
    /// Hint: the index of the conflicting entry for faster backtracking.
    pub match_index: u64,
}

// ── Raft Node ────────────────────────────────────────────────────────────────

/// A simulated Raft consensus node.
#[derive(Debug, Clone)]
pub struct RaftNode {
    /// This node's unique id.
    pub id: u64,
    /// Current term.
    pub current_term: u64,
    /// Candidate id that received vote in current term.
    pub voted_for: Option<u64>,
    /// Log entries.
    pub log: Vec<LogEntry>,
    /// Index of highest log entry known to be committed.
    pub commit_index: u64,
    /// Index of highest log entry applied to state machine.
    pub last_applied: u64,
    /// Current role.
    pub role: NodeRole,
    /// For leader: next index to send to each follower.
    pub next_index: HashMap<u64, u64>,
    /// For leader: highest index known to be replicated on each follower.
    pub match_index: HashMap<u64, u64>,
    /// Set of peers (not including self).
    pub peers: HashSet<u64>,
    /// Votes received in current election.
    pub votes_received: HashSet<u64>,
    /// Applied state machine entries (for verification).
    pub state_machine: Vec<String>,
    /// Ticks since last heartbeat or election timeout.
    pub election_ticks: u64,
    /// Election timeout in ticks.
    pub election_timeout: u64,
}

impl RaftNode {
    /// Create a new Raft node in the Follower state.
    pub fn new(id: u64, peers: &[u64], election_timeout: u64) -> Self {
        Self {
            id,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            role: NodeRole::Follower,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            peers: peers.iter().copied().collect(),
            votes_received: HashSet::new(),
            state_machine: Vec::new(),
            election_ticks: 0,
            election_timeout,
        }
    }

    /// Get the last log index (0 if empty).
    pub fn last_log_index(&self) -> u64 {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Get the last log term (0 if empty).
    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Cluster size (including self).
    pub fn cluster_size(&self) -> usize {
        self.peers.len() + 1
    }

    /// Majority quorum size.
    pub fn quorum_size(&self) -> usize {
        self.cluster_size() / 2 + 1
    }

    /// Get log entry at the given index (1-based).
    pub fn log_entry(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 || index as usize > self.log.len() {
            None
        } else {
            Some(&self.log[(index - 1) as usize])
        }
    }

    /// Get the term of a log entry at the given index.
    fn log_term_at(&self, index: u64) -> u64 {
        if index == 0 {
            0
        } else if (index as usize) <= self.log.len() {
            self.log[(index - 1) as usize].term
        } else {
            0
        }
    }

    /// Become a follower for the given term.
    pub fn become_follower(&mut self, term: u64) {
        self.current_term = term;
        self.role = NodeRole::Follower;
        self.voted_for = None;
        self.votes_received.clear();
        self.election_ticks = 0;
    }

    /// Start an election: transition to Candidate.
    pub fn start_election(&mut self) -> VoteRequest {
        self.current_term += 1;
        self.role = NodeRole::Candidate;
        self.voted_for = Some(self.id);
        self.votes_received.clear();
        self.votes_received.insert(self.id);
        self.election_ticks = 0;

        VoteRequest {
            term: self.current_term,
            candidate_id: self.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Handle a RequestVote RPC.
    pub fn handle_vote_request(&mut self, req: &VoteRequest) -> VoteResponse {
        // If request term is greater, become follower
        if req.term > self.current_term {
            self.become_follower(req.term);
        }

        let vote_granted = if req.term < self.current_term {
            false
        } else if self.voted_for.is_some() && self.voted_for != Some(req.candidate_id) {
            false
        } else {
            // Check if candidate's log is at least as up-to-date
            let my_last_term = self.last_log_term();
            let my_last_index = self.last_log_index();
            if req.last_log_term > my_last_term
                || (req.last_log_term == my_last_term && req.last_log_index >= my_last_index)
            {
                self.voted_for = Some(req.candidate_id);
                self.election_ticks = 0;
                true
            } else {
                false
            }
        };

        VoteResponse {
            term: self.current_term,
            vote_granted,
        }
    }

    /// Handle a vote response (as a Candidate).
    pub fn handle_vote_response(&mut self, resp: &VoteResponse) -> bool {
        if resp.term > self.current_term {
            self.become_follower(resp.term);
            return false;
        }

        if self.role != NodeRole::Candidate {
            return false;
        }

        if resp.vote_granted {
            // We track votes by counting; in a real system we'd track by peer id
            // Here we just check if we have a quorum
            // Since we may not know which peer responded, just increment
        }

        // Check if we have a quorum
        self.votes_received.len() >= self.quorum_size()
    }

    /// Record a vote from a specific peer.
    pub fn record_vote(&mut self, peer_id: u64) {
        self.votes_received.insert(peer_id);
    }

    /// Check if this candidate has won the election.
    pub fn has_won_election(&self) -> bool {
        self.role == NodeRole::Candidate && self.votes_received.len() >= self.quorum_size()
    }

    /// Become leader after winning election.
    pub fn become_leader(&mut self) {
        self.role = NodeRole::Leader;
        self.next_index.clear();
        self.match_index.clear();
        let last_idx = self.last_log_index() + 1;
        for &peer in &self.peers {
            self.next_index.insert(peer, last_idx);
            self.match_index.insert(peer, 0);
        }
    }

    /// As leader, append a new command to the log.
    pub fn leader_append(&mut self, command: String) -> Option<u64> {
        if self.role != NodeRole::Leader {
            return None;
        }
        let index = self.last_log_index() + 1;
        self.log.push(LogEntry {
            term: self.current_term,
            index,
            command,
        });
        Some(index)
    }

    /// Create an AppendEntries request for a specific peer.
    pub fn create_append_entries(&self, peer_id: u64) -> Option<AppendEntriesRequest> {
        if self.role != NodeRole::Leader {
            return None;
        }
        let next_idx = self.next_index.get(&peer_id).copied().unwrap_or(1);
        let prev_log_index = next_idx.saturating_sub(1);
        let prev_log_term = self.log_term_at(prev_log_index);

        let entries: Vec<LogEntry> = self
            .log
            .iter()
            .filter(|e| e.index >= next_idx)
            .cloned()
            .collect();

        Some(AppendEntriesRequest {
            term: self.current_term,
            leader_id: self.id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        })
    }

    /// Handle an AppendEntries RPC (as a follower).
    pub fn handle_append_entries(&mut self, req: &AppendEntriesRequest) -> AppendEntriesResponse {
        if req.term > self.current_term {
            self.become_follower(req.term);
        }

        if req.term < self.current_term {
            return AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
            };
        }

        // Valid leader heartbeat/append — reset election timer
        self.election_ticks = 0;
        if self.role == NodeRole::Candidate {
            self.become_follower(req.term);
        }

        // Check prev_log consistency
        if req.prev_log_index > 0 {
            let my_term = self.log_term_at(req.prev_log_index);
            if req.prev_log_index as usize > self.log.len() || my_term != req.prev_log_term {
                return AppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_log_index(),
                };
            }
        }

        // Append/overwrite entries
        for entry in &req.entries {
            let idx = entry.index as usize;
            if idx <= self.log.len() {
                // Overwrite if term conflict
                if self.log[idx - 1].term != entry.term {
                    self.log.truncate(idx - 1);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Update commit index
        if req.leader_commit > self.commit_index {
            self.commit_index = std::cmp::min(req.leader_commit, self.last_log_index());
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            match_index: self.last_log_index(),
        }
    }

    /// Handle an AppendEntries response (as leader).
    pub fn handle_append_response(&mut self, peer_id: u64, resp: &AppendEntriesResponse) {
        if resp.term > self.current_term {
            self.become_follower(resp.term);
            return;
        }
        if self.role != NodeRole::Leader {
            return;
        }
        if resp.success {
            self.match_index.insert(peer_id, resp.match_index);
            self.next_index.insert(peer_id, resp.match_index + 1);
        } else {
            // Decrement next_index for retry
            let current = self.next_index.get(&peer_id).copied().unwrap_or(1);
            if current > 1 {
                self.next_index.insert(peer_id, current - 1);
            }
        }
    }

    /// As leader, try to advance commit index based on majority replication.
    pub fn try_advance_commit(&mut self) -> bool {
        if self.role != NodeRole::Leader {
            return false;
        }

        let old_commit = self.commit_index;
        // Try each possible commit index from high to low
        let last = self.last_log_index();
        for n in (old_commit + 1..=last).rev() {
            if self.log_term_at(n) != self.current_term {
                continue; // Can only commit entries from current term
            }
            // Count replications (self + peers with match_index >= n)
            let mut count = 1; // self
            for &mi in self.match_index.values() {
                if mi >= n {
                    count += 1;
                }
            }
            if count >= self.quorum_size() {
                self.commit_index = n;
                return true;
            }
        }

        false
    }

    /// Apply committed entries to the state machine.
    pub fn apply_committed(&mut self) -> Vec<String> {
        let mut applied = Vec::new();
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log_entry(self.last_applied) {
                let cmd = entry.command.clone();
                self.state_machine.push(cmd.clone());
                applied.push(cmd);
            }
        }
        applied
    }

    /// Tick the election timer. Returns true if timeout expired.
    pub fn tick(&mut self) -> bool {
        self.election_ticks += 1;
        self.election_ticks >= self.election_timeout
    }

    /// Add a new peer to the cluster.
    pub fn add_peer(&mut self, peer_id: u64) {
        self.peers.insert(peer_id);
        if self.role == NodeRole::Leader {
            let last_idx = self.last_log_index() + 1;
            self.next_index.insert(peer_id, last_idx);
            self.match_index.insert(peer_id, 0);
        }
    }

    /// Remove a peer from the cluster.
    pub fn remove_peer(&mut self, peer_id: u64) {
        self.peers.remove(&peer_id);
        self.next_index.remove(&peer_id);
        self.match_index.remove(&peer_id);
    }
}

// ── Cluster (simulation helper) ──────────────────────────────────────────────

/// A simulated Raft cluster for testing consensus.
#[derive(Debug)]
pub struct RaftCluster {
    pub nodes: HashMap<u64, RaftNode>,
}

impl RaftCluster {
    /// Create a new cluster with `n` nodes (ids 0..n).
    pub fn new(n: u64, election_timeout: u64) -> Self {
        let all_ids: Vec<u64> = (0..n).collect();
        let mut nodes = HashMap::new();
        for &id in &all_ids {
            let peers: Vec<u64> = all_ids.iter().copied().filter(|p| *p != id).collect();
            nodes.insert(id, RaftNode::new(id, &peers, election_timeout));
        }
        Self { nodes }
    }

    /// Find the current leader, if any.
    pub fn leader(&self) -> Option<u64> {
        self.nodes
            .values()
            .find(|n| n.role == NodeRole::Leader)
            .map(|n| n.id)
    }

    /// Simulate a full election by a single candidate.
    pub fn run_election(&mut self, candidate_id: u64) -> bool {
        let vote_req = {
            let node = self.nodes.get_mut(&candidate_id).unwrap();
            node.start_election()
        };

        let peers: Vec<u64> = self.nodes.get(&candidate_id).unwrap().peers.iter().copied().collect();

        for &peer_id in &peers {
            let resp = {
                let peer = self.nodes.get_mut(&peer_id).unwrap();
                peer.handle_vote_request(&vote_req)
            };
            if resp.vote_granted {
                let node = self.nodes.get_mut(&candidate_id).unwrap();
                node.record_vote(peer_id);
            }
        }

        let node = self.nodes.get_mut(&candidate_id).unwrap();
        if node.has_won_election() {
            node.become_leader();
            true
        } else {
            false
        }
    }

    /// Replicate a command from the leader to all followers.
    pub fn replicate(&mut self, leader_id: u64) {
        let peers: Vec<u64> = self.nodes.get(&leader_id).unwrap().peers.iter().copied().collect();

        for &peer_id in &peers {
            let req = {
                let leader = self.nodes.get(&leader_id).unwrap();
                leader.create_append_entries(peer_id)
            };
            if let Some(req) = req {
                let resp = {
                    let follower = self.nodes.get_mut(&peer_id).unwrap();
                    follower.handle_append_entries(&req)
                };
                let leader = self.nodes.get_mut(&leader_id).unwrap();
                leader.handle_append_response(peer_id, &resp);
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_is_follower() {
        let node = RaftNode::new(1, &[2, 3], 10);
        assert_eq!(node.role, NodeRole::Follower);
        assert_eq!(node.current_term, 0);
        assert_eq!(node.cluster_size(), 3);
    }

    #[test]
    fn test_start_election() {
        let mut node = RaftNode::new(1, &[2, 3], 10);
        let req = node.start_election();
        assert_eq!(node.role, NodeRole::Candidate);
        assert_eq!(node.current_term, 1);
        assert_eq!(req.candidate_id, 1);
        assert!(node.votes_received.contains(&1));
    }

    #[test]
    fn test_vote_granted() {
        let mut follower = RaftNode::new(2, &[1, 3], 10);
        let req = VoteRequest {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = follower.handle_vote_request(&req);
        assert!(resp.vote_granted);
        assert_eq!(follower.voted_for, Some(1));
    }

    #[test]
    fn test_vote_denied_older_term() {
        let mut follower = RaftNode::new(2, &[1], 10);
        follower.current_term = 5;
        let req = VoteRequest {
            term: 3,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = follower.handle_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn test_vote_denied_already_voted() {
        let mut follower = RaftNode::new(2, &[1, 3], 10);
        follower.current_term = 1;
        follower.voted_for = Some(3);
        let req = VoteRequest {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = follower.handle_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn test_election_win() {
        let mut cluster = RaftCluster::new(3, 10);
        let won = cluster.run_election(0);
        assert!(won);
        assert_eq!(cluster.leader(), Some(0));
    }

    #[test]
    fn test_leader_append() {
        let mut node = RaftNode::new(1, &[2, 3], 10);
        node.start_election();
        node.record_vote(2);
        node.become_leader();
        let idx = node.leader_append("SET x 1".to_string());
        assert_eq!(idx, Some(1));
        assert_eq!(node.log.len(), 1);
    }

    #[test]
    fn test_append_entries() {
        let mut follower = RaftNode::new(2, &[1], 10);
        let req = AppendEntriesRequest {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                command: "SET x 1".to_string(),
            }],
            leader_commit: 0,
        };
        let resp = follower.handle_append_entries(&req);
        assert!(resp.success);
        assert_eq!(follower.log.len(), 1);
    }

    #[test]
    fn test_log_replication() {
        let mut cluster = RaftCluster::new(3, 10);
        cluster.run_election(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.leader_append("SET x 1".to_string());
        }
        cluster.replicate(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.try_advance_commit();
        }
        let leader = cluster.nodes.get(&0).unwrap();
        assert_eq!(leader.commit_index, 1);
    }

    #[test]
    fn test_apply_committed() {
        let mut cluster = RaftCluster::new(3, 10);
        cluster.run_election(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.leader_append("CMD1".to_string());
        }
        cluster.replicate(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.try_advance_commit();
            let applied = leader.apply_committed();
            assert_eq!(applied, vec!["CMD1".to_string()]);
        }
    }

    #[test]
    fn test_follower_commit_update() {
        let mut cluster = RaftCluster::new(3, 10);
        cluster.run_election(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.leader_append("CMD1".to_string());
        }
        cluster.replicate(0);
        {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.try_advance_commit();
        }
        // Replicate again so followers learn about the commit
        cluster.replicate(0);
        let follower = cluster.nodes.get(&1).unwrap();
        assert_eq!(follower.commit_index, 1);
    }

    #[test]
    fn test_quorum_size() {
        let node3 = RaftNode::new(1, &[2, 3], 10);
        assert_eq!(node3.quorum_size(), 2);
        let node5 = RaftNode::new(1, &[2, 3, 4, 5], 10);
        assert_eq!(node5.quorum_size(), 3);
    }

    #[test]
    fn test_become_follower_on_higher_term() {
        let mut node = RaftNode::new(1, &[2], 10);
        node.start_election();
        assert_eq!(node.role, NodeRole::Candidate);
        let req = AppendEntriesRequest {
            term: 5,
            leader_id: 2,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: Vec::new(),
            leader_commit: 0,
        };
        node.handle_append_entries(&req);
        assert_eq!(node.role, NodeRole::Follower);
        assert_eq!(node.current_term, 5);
    }

    #[test]
    fn test_election_timeout_tick() {
        let mut node = RaftNode::new(1, &[2], 5);
        for _ in 0..4 {
            assert!(!node.tick());
        }
        assert!(node.tick());
    }

    #[test]
    fn test_add_remove_peer() {
        let mut node = RaftNode::new(1, &[2], 10);
        assert_eq!(node.cluster_size(), 2);
        node.add_peer(3);
        assert_eq!(node.cluster_size(), 3);
        node.remove_peer(3);
        assert_eq!(node.cluster_size(), 2);
    }

    #[test]
    fn test_log_entry_access() {
        let mut node = RaftNode::new(1, &[2, 3], 10);
        node.start_election();
        node.record_vote(2);
        node.become_leader();
        node.leader_append("CMD".to_string());
        assert!(node.log_entry(1).is_some());
        assert!(node.log_entry(0).is_none());
        assert!(node.log_entry(2).is_none());
    }

    #[test]
    fn test_multiple_commands() {
        let mut cluster = RaftCluster::new(3, 10);
        cluster.run_election(0);
        for i in 0..3 {
            let leader = cluster.nodes.get_mut(&0).unwrap();
            leader.leader_append(format!("CMD{}", i));
        }
        cluster.replicate(0);
        let leader = cluster.nodes.get_mut(&0).unwrap();
        leader.try_advance_commit();
        assert_eq!(leader.commit_index, 3);
    }
}
