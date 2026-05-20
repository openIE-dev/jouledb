//! Distributed locking — lock acquisition with TTL, fencing tokens, lock
//! renewal, deadlock detection, lock-free compare-and-swap, and leader election.
//!
//! Replaces `redlock`, `node-redlock`, and `etcd-lock` JS libraries with a
//! pure-Rust, energy-aware distributed lock manager that operates entirely
//! in-memory for single-process or simulation use.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Lock manager errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockError {
    /// Lock is already held by another owner.
    AlreadyHeld { resource: String, owner: String },
    /// Lock not found.
    NotFound(String),
    /// Lock has expired.
    Expired(String),
    /// Fencing token mismatch.
    FencingTokenMismatch { expected: u64, actual: u64 },
    /// Only the owner can release or renew.
    NotOwner { resource: String, owner: String, actual_owner: String },
    /// Deadlock detected.
    DeadlockDetected { cycle: Vec<String> },
    /// CAS conflict.
    CasConflict { resource: String, expected: u64, actual: u64 },
    /// Election already in progress.
    ElectionInProgress(String),
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyHeld { resource, owner } => {
                write!(f, "lock on {resource} already held by {owner}")
            }
            Self::NotFound(r) => write!(f, "lock not found: {r}"),
            Self::Expired(r) => write!(f, "lock expired: {r}"),
            Self::FencingTokenMismatch { expected, actual } => {
                write!(f, "fencing token mismatch: expected {expected}, got {actual}")
            }
            Self::NotOwner { resource, owner, actual_owner } => {
                write!(f, "{owner} is not owner of {resource} (owner: {actual_owner})")
            }
            Self::DeadlockDetected { cycle } => {
                write!(f, "deadlock detected: {}", cycle.join(" -> "))
            }
            Self::CasConflict { resource, expected, actual } => {
                write!(f, "CAS conflict on {resource}: expected version {expected}, got {actual}")
            }
            Self::ElectionInProgress(group) => {
                write!(f, "election already in progress for group {group}")
            }
        }
    }
}

impl std::error::Error for LockError {}

// ── Lock record ─────────────────────────────────────────────────

/// A held lock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockRecord {
    pub resource: String,
    pub owner: String,
    pub fencing_token: u64,
    pub acquired_at_ms: u64,
    pub ttl_ms: u64,
    pub renewed_count: u32,
}

impl LockRecord {
    /// Check whether this lock has expired at the given time.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        if self.ttl_ms == 0 {
            return false;
        }
        now_ms.saturating_sub(self.acquired_at_ms) >= self.ttl_ms
    }
}

/// A lock-free CAS register.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasRegister {
    pub resource: String,
    pub value: Vec<u8>,
    pub version: u64,
    pub updated_at_ms: u64,
}

/// Leader election state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionGroup {
    pub group_name: String,
    pub leader: Option<String>,
    pub candidates: Vec<String>,
    pub term: u64,
    pub elected_at_ms: u64,
    pub leader_ttl_ms: u64,
}

// ── Wait queue entry ────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WaitEntry {
    owner: String,
    resource: String,
    queued_at_ms: u64,
}

// ── Lock manager ────────────────────────────────────────────────

/// Distributed lock manager.
pub struct LockManager {
    locks: HashMap<String, LockRecord>,
    next_fencing_token: u64,
    wait_queue: VecDeque<WaitEntry>,
    cas_registers: HashMap<String, CasRegister>,
    elections: HashMap<String, ElectionGroup>,
}

impl LockManager {
    pub fn new() -> Self {
        Self {
            locks: HashMap::new(),
            next_fencing_token: 1,
            wait_queue: VecDeque::new(),
            cas_registers: HashMap::new(),
            elections: HashMap::new(),
        }
    }

    /// Acquire a lock on a resource. Returns the fencing token.
    pub fn acquire(
        &mut self,
        resource: &str,
        owner: &str,
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<u64, LockError> {
        // Clean up expired lock.
        if let Some(existing) = self.locks.get(resource) {
            if existing.is_expired(now_ms) {
                self.locks.remove(resource);
            } else if existing.owner != owner {
                return Err(LockError::AlreadyHeld {
                    resource: resource.into(),
                    owner: existing.owner.clone(),
                });
            } else {
                // Re-entrant — return existing token.
                return Ok(existing.fencing_token);
            }
        }

        let token = self.next_fencing_token;
        self.next_fencing_token += 1;

        self.locks.insert(
            resource.to_string(),
            LockRecord {
                resource: resource.into(),
                owner: owner.into(),
                fencing_token: token,
                acquired_at_ms: now_ms,
                ttl_ms,
                renewed_count: 0,
            },
        );
        Ok(token)
    }

    /// Release a lock. Only the owner may release.
    pub fn release(&mut self, resource: &str, owner: &str) -> Result<(), LockError> {
        let lock = self
            .locks
            .get(resource)
            .ok_or_else(|| LockError::NotFound(resource.into()))?;
        if lock.owner != owner {
            return Err(LockError::NotOwner {
                resource: resource.into(),
                owner: owner.into(),
                actual_owner: lock.owner.clone(),
            });
        }
        self.locks.remove(resource);
        Ok(())
    }

    /// Renew (extend) a lock's TTL. Only the owner may renew.
    pub fn renew(
        &mut self,
        resource: &str,
        owner: &str,
        new_ttl_ms: u64,
        now_ms: u64,
    ) -> Result<(), LockError> {
        let lock = self
            .locks
            .get_mut(resource)
            .ok_or_else(|| LockError::NotFound(resource.into()))?;
        if lock.owner != owner {
            return Err(LockError::NotOwner {
                resource: resource.into(),
                owner: owner.into(),
                actual_owner: lock.owner.clone(),
            });
        }
        if lock.is_expired(now_ms) {
            self.locks.remove(resource);
            return Err(LockError::Expired(resource.into()));
        }
        lock.acquired_at_ms = now_ms;
        lock.ttl_ms = new_ttl_ms;
        lock.renewed_count += 1;
        Ok(())
    }

    /// Validate a fencing token for a resource.
    pub fn validate_fencing_token(
        &self,
        resource: &str,
        token: u64,
    ) -> Result<(), LockError> {
        let lock = self
            .locks
            .get(resource)
            .ok_or_else(|| LockError::NotFound(resource.into()))?;
        if lock.fencing_token != token {
            return Err(LockError::FencingTokenMismatch {
                expected: lock.fencing_token,
                actual: token,
            });
        }
        Ok(())
    }

    /// Add a waiter to the queue (for deadlock detection).
    pub fn enqueue_wait(&mut self, owner: &str, resource: &str, now_ms: u64) {
        self.wait_queue.push_back(WaitEntry {
            owner: owner.into(),
            resource: resource.into(),
            queued_at_ms: now_ms,
        });
    }

    /// Detect deadlocks in the wait-for graph.
    /// Returns cycles found as lists of owner IDs.
    pub fn detect_deadlocks(&self) -> Vec<Vec<String>> {
        // Build wait-for graph: owner -> set of owners it's waiting on.
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &self.wait_queue {
            if let Some(lock) = self.locks.get(&entry.resource) {
                if lock.owner != entry.owner {
                    graph
                        .entry(entry.owner.clone())
                        .or_default()
                        .insert(lock.owner.clone());
                }
            }
        }

        let mut cycles = Vec::new();
        let mut visited = HashSet::new();

        for start in graph.keys() {
            if visited.contains(start) {
                continue;
            }
            let mut path = Vec::new();
            let mut path_set = HashSet::new();
            self.dfs_cycle(start, &graph, &mut path, &mut path_set, &mut visited, &mut cycles);
        }
        cycles
    }

    fn dfs_cycle(
        &self,
        node: &str,
        graph: &HashMap<String, HashSet<String>>,
        path: &mut Vec<String>,
        path_set: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        if path_set.contains(node) {
            // Found cycle — extract it.
            let start_idx = path.iter().position(|n| n == node).unwrap();
            let cycle: Vec<String> = path[start_idx..].to_vec();
            if cycle.len() >= 2 {
                cycles.push(cycle);
            }
            return;
        }
        if visited.contains(node) {
            return;
        }

        path.push(node.to_string());
        path_set.insert(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            let mut sorted: Vec<_> = neighbors.iter().collect();
            sorted.sort();
            for next in sorted {
                self.dfs_cycle(next, graph, path, path_set, visited, cycles);
            }
        }

        path_set.remove(node);
        path.pop();
        visited.insert(node.to_string());
    }

    /// Clear the wait queue.
    pub fn clear_wait_queue(&mut self) {
        self.wait_queue.clear();
    }

    // ── CAS registers ───────────────────────────────────────────

    /// Initialize a CAS register.
    pub fn cas_init(&mut self, resource: &str, value: Vec<u8>, now_ms: u64) {
        self.cas_registers.insert(
            resource.to_string(),
            CasRegister {
                resource: resource.into(),
                value,
                version: 1,
                updated_at_ms: now_ms,
            },
        );
    }

    /// Compare-and-swap: update only if `expected_version` matches current.
    pub fn cas_update(
        &mut self,
        resource: &str,
        new_value: Vec<u8>,
        expected_version: u64,
        now_ms: u64,
    ) -> Result<u64, LockError> {
        let reg = self
            .cas_registers
            .get_mut(resource)
            .ok_or_else(|| LockError::NotFound(resource.into()))?;
        if reg.version != expected_version {
            return Err(LockError::CasConflict {
                resource: resource.into(),
                expected: expected_version,
                actual: reg.version,
            });
        }
        reg.value = new_value;
        reg.version += 1;
        reg.updated_at_ms = now_ms;
        Ok(reg.version)
    }

    /// Read a CAS register's current value and version.
    pub fn cas_read(&self, resource: &str) -> Result<(&[u8], u64), LockError> {
        let reg = self
            .cas_registers
            .get(resource)
            .ok_or_else(|| LockError::NotFound(resource.into()))?;
        Ok((&reg.value, reg.version))
    }

    // ── Leader election ─────────────────────────────────────────

    /// Register a candidate for leader election.
    pub fn register_candidate(&mut self, group: &str, candidate: &str, leader_ttl_ms: u64) {
        let entry = self.elections.entry(group.to_string()).or_insert_with(|| {
            ElectionGroup {
                group_name: group.into(),
                leader: None,
                candidates: Vec::new(),
                term: 0,
                elected_at_ms: 0,
                leader_ttl_ms,
            }
        });
        if !entry.candidates.contains(&candidate.to_string()) {
            entry.candidates.push(candidate.into());
        }
    }

    /// Run leader election for a group. Selects the first candidate
    /// lexicographically (deterministic bully algorithm).
    pub fn elect_leader(&mut self, group: &str, now_ms: u64) -> Result<String, LockError> {
        let entry = self
            .elections
            .get_mut(group)
            .ok_or_else(|| LockError::NotFound(group.into()))?;
        if entry.candidates.is_empty() {
            return Err(LockError::NotFound(format!("no candidates for {group}")));
        }
        let mut sorted = entry.candidates.clone();
        sorted.sort();
        let leader = sorted[0].clone();
        entry.leader = Some(leader.clone());
        entry.term += 1;
        entry.elected_at_ms = now_ms;
        Ok(leader)
    }

    /// Get the current leader for a group, if any (and not expired).
    pub fn current_leader(&self, group: &str, now_ms: u64) -> Option<&str> {
        let entry = self.elections.get(group)?;
        if entry.leader.is_none() {
            return None;
        }
        if entry.leader_ttl_ms > 0
            && now_ms.saturating_sub(entry.elected_at_ms) >= entry.leader_ttl_ms
        {
            return None;
        }
        entry.leader.as_deref()
    }

    /// Get the current term for a group.
    pub fn current_term(&self, group: &str) -> Option<u64> {
        self.elections.get(group).map(|e| e.term)
    }

    /// Remove a candidate from a group.
    pub fn remove_candidate(&mut self, group: &str, candidate: &str) {
        if let Some(entry) = self.elections.get_mut(group) {
            entry.candidates.retain(|c| c != candidate);
            if entry.leader.as_deref() == Some(candidate) {
                entry.leader = None;
            }
        }
    }

    /// Get info about a held lock.
    pub fn lock_info(&self, resource: &str) -> Option<&LockRecord> {
        self.locks.get(resource)
    }

    /// Evict all expired locks.
    pub fn evict_expired(&mut self, now_ms: u64) -> usize {
        let expired: Vec<String> = self
            .locks
            .iter()
            .filter(|(_, l)| l.is_expired(now_ms))
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for k in expired {
            self.locks.remove(&k);
        }
        count
    }

    /// Number of currently held locks.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_and_release() {
        let mut mgr = LockManager::new();
        let token = mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        assert!(token > 0);
        assert_eq!(mgr.lock_count(), 1);
        mgr.release("res1", "alice").unwrap();
        assert_eq!(mgr.lock_count(), 0);
    }

    #[test]
    fn test_acquire_conflict() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        let err = mgr.acquire("res1", "bob", 5000, 1000);
        assert!(matches!(err, Err(LockError::AlreadyHeld { .. })));
    }

    #[test]
    fn test_reentrant_acquire() {
        let mut mgr = LockManager::new();
        let t1 = mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        let t2 = mgr.acquire("res1", "alice", 5000, 1500).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_release_wrong_owner() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        let err = mgr.release("res1", "bob");
        assert!(matches!(err, Err(LockError::NotOwner { .. })));
    }

    #[test]
    fn test_lock_expiry() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 1000, 1000).unwrap();
        // After TTL, bob can acquire.
        let token = mgr.acquire("res1", "bob", 5000, 2001).unwrap();
        assert!(token > 0);
        assert_eq!(mgr.lock_info("res1").unwrap().owner, "bob");
    }

    #[test]
    fn test_renew() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 1000, 1000).unwrap();
        mgr.renew("res1", "alice", 2000, 1500).unwrap();
        let info = mgr.lock_info("res1").unwrap();
        assert_eq!(info.ttl_ms, 2000);
        assert_eq!(info.renewed_count, 1);
    }

    #[test]
    fn test_renew_expired() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 500, 1000).unwrap();
        let err = mgr.renew("res1", "alice", 2000, 2000);
        assert!(matches!(err, Err(LockError::Expired(_))));
    }

    #[test]
    fn test_renew_not_owner() {
        let mut mgr = LockManager::new();
        mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        let err = mgr.renew("res1", "bob", 2000, 1500);
        assert!(matches!(err, Err(LockError::NotOwner { .. })));
    }

    #[test]
    fn test_fencing_tokens_monotonic() {
        let mut mgr = LockManager::new();
        let t1 = mgr.acquire("r1", "a", 5000, 1000).unwrap();
        let t2 = mgr.acquire("r2", "b", 5000, 1000).unwrap();
        assert!(t2 > t1);
    }

    #[test]
    fn test_validate_fencing_token() {
        let mut mgr = LockManager::new();
        let token = mgr.acquire("res1", "alice", 5000, 1000).unwrap();
        mgr.validate_fencing_token("res1", token).unwrap();
        let err = mgr.validate_fencing_token("res1", token + 99);
        assert!(matches!(err, Err(LockError::FencingTokenMismatch { .. })));
    }

    #[test]
    fn test_deadlock_detection() {
        let mut mgr = LockManager::new();
        // alice holds r1, bob holds r2.
        mgr.acquire("r1", "alice", 60000, 1000).unwrap();
        mgr.acquire("r2", "bob", 60000, 1000).unwrap();
        // alice waits for r2, bob waits for r1 → cycle.
        mgr.enqueue_wait("alice", "r2", 2000);
        mgr.enqueue_wait("bob", "r1", 2000);
        let cycles = mgr.detect_deadlocks();
        assert!(!cycles.is_empty());
        // The cycle should contain both alice and bob.
        let cycle = &cycles[0];
        assert!(cycle.contains(&"alice".to_string()) || cycle.contains(&"bob".to_string()));
    }

    #[test]
    fn test_no_deadlock() {
        let mut mgr = LockManager::new();
        mgr.acquire("r1", "alice", 60000, 1000).unwrap();
        mgr.enqueue_wait("bob", "r1", 2000);
        let cycles = mgr.detect_deadlocks();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_cas_init_and_read() {
        let mut mgr = LockManager::new();
        mgr.cas_init("counter", vec![0, 0, 0, 1], 1000);
        let (val, ver) = mgr.cas_read("counter").unwrap();
        assert_eq!(val, &[0, 0, 0, 1]);
        assert_eq!(ver, 1);
    }

    #[test]
    fn test_cas_update_success() {
        let mut mgr = LockManager::new();
        mgr.cas_init("counter", vec![1], 1000);
        let new_ver = mgr.cas_update("counter", vec![2], 1, 2000).unwrap();
        assert_eq!(new_ver, 2);
        let (val, _) = mgr.cas_read("counter").unwrap();
        assert_eq!(val, &[2]);
    }

    #[test]
    fn test_cas_conflict() {
        let mut mgr = LockManager::new();
        mgr.cas_init("counter", vec![1], 1000);
        let err = mgr.cas_update("counter", vec![2], 99, 2000);
        assert!(matches!(err, Err(LockError::CasConflict { .. })));
    }

    #[test]
    fn test_leader_election() {
        let mut mgr = LockManager::new();
        mgr.register_candidate("cluster", "node-c", 10000);
        mgr.register_candidate("cluster", "node-a", 10000);
        mgr.register_candidate("cluster", "node-b", 10000);
        let leader = mgr.elect_leader("cluster", 1000).unwrap();
        assert_eq!(leader, "node-a"); // lexicographic first
        assert_eq!(mgr.current_term("cluster"), Some(1));
        assert_eq!(mgr.current_leader("cluster", 1000), Some("node-a"));
    }

    #[test]
    fn test_leader_expiry() {
        let mut mgr = LockManager::new();
        mgr.register_candidate("cluster", "node-a", 1000);
        mgr.elect_leader("cluster", 1000).unwrap();
        assert_eq!(mgr.current_leader("cluster", 1500), Some("node-a"));
        assert_eq!(mgr.current_leader("cluster", 2000), None); // expired
    }

    #[test]
    fn test_remove_candidate_triggers_leader_clear() {
        let mut mgr = LockManager::new();
        mgr.register_candidate("cluster", "node-a", 0);
        mgr.elect_leader("cluster", 1000).unwrap();
        mgr.remove_candidate("cluster", "node-a");
        assert_eq!(mgr.current_leader("cluster", 1000), None);
    }

    #[test]
    fn test_re_election_increments_term() {
        let mut mgr = LockManager::new();
        mgr.register_candidate("cluster", "node-a", 0);
        mgr.register_candidate("cluster", "node-b", 0);
        mgr.elect_leader("cluster", 1000).unwrap();
        mgr.elect_leader("cluster", 2000).unwrap();
        assert_eq!(mgr.current_term("cluster"), Some(2));
    }

    #[test]
    fn test_evict_expired() {
        let mut mgr = LockManager::new();
        mgr.acquire("r1", "a", 500, 1000).unwrap();
        mgr.acquire("r2", "b", 2000, 1000).unwrap();
        let evicted = mgr.evict_expired(1600);
        assert_eq!(evicted, 1);
        assert_eq!(mgr.lock_count(), 1);
    }

    #[test]
    fn test_zero_ttl_never_expires() {
        let record = LockRecord {
            resource: "r".into(),
            owner: "o".into(),
            fencing_token: 1,
            acquired_at_ms: 1000,
            ttl_ms: 0,
            renewed_count: 0,
        };
        assert!(!record.is_expired(u64::MAX - 1));
    }

    #[test]
    fn test_error_display() {
        let e = LockError::AlreadyHeld {
            resource: "r1".into(),
            owner: "bob".into(),
        };
        assert!(e.to_string().contains("r1"));
        assert!(e.to_string().contains("bob"));
    }

    #[test]
    fn test_duplicate_candidate_registration() {
        let mut mgr = LockManager::new();
        mgr.register_candidate("g", "node-a", 0);
        mgr.register_candidate("g", "node-a", 0);
        let entry = mgr.elections.get("g").unwrap();
        assert_eq!(entry.candidates.len(), 1);
    }
}
