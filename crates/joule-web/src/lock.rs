//! Web Locks API — cooperative resource locking.
//!
//! Implements a headless version of the Web Locks API with shared/exclusive
//! modes, FIFO grant ordering, and snapshot queries.

use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

// ── Types ───────────────────────────────────────────────────────

/// Lock acquisition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Shared,
    Exclusive,
}

/// A pending lock request.
#[derive(Debug, Clone)]
pub struct LockRequest {
    pub name: String,
    pub mode: LockMode,
    pub requester_id: Uuid,
}

/// An active lock holder.
#[derive(Debug, Clone)]
pub struct LockHolder {
    pub id: Uuid,
    pub mode: LockMode,
    pub acquired_at: DateTime<Utc>,
}

/// Result of a lock request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockResult {
    Acquired,
    Waiting,
}

/// Information about a lock for snapshots.
#[derive(Debug, Clone)]
pub struct LockInfo {
    pub name: String,
    pub mode: LockMode,
    pub requester_id: Uuid,
}

/// A snapshot of the lock manager state.
#[derive(Debug, Clone)]
pub struct LockSnapshot {
    pub held: Vec<LockInfo>,
    pub pending: Vec<LockInfo>,
}

// ── LockState ───────────────────────────────────────────────────

/// Lock manager tracking held and waiting locks.
pub struct LockState {
    held: HashMap<String, Vec<LockHolder>>,
    waiting: HashMap<String, VecDeque<LockRequest>>,
}

impl LockState {
    /// Create an empty lock state.
    pub fn new() -> Self {
        Self {
            held: HashMap::new(),
            waiting: HashMap::new(),
        }
    }

    /// Request a lock. Returns `Acquired` if granted immediately, `Waiting` otherwise.
    pub fn request(&mut self, name: &str, mode: LockMode, requester_id: Uuid) -> LockResult {
        let holders = self.held.entry(name.to_string()).or_default();

        if Self::can_grant(holders, mode) {
            holders.push(LockHolder {
                id: requester_id,
                mode,
                acquired_at: Utc::now(),
            });
            LockResult::Acquired
        } else {
            self.waiting
                .entry(name.to_string())
                .or_default()
                .push_back(LockRequest {
                    name: name.to_string(),
                    mode,
                    requester_id,
                });
            LockResult::Waiting
        }
    }

    /// Release a lock and grant waiting requests. Returns IDs of newly granted holders.
    pub fn release(&mut self, name: &str, holder_id: Uuid) -> Vec<Uuid> {
        let mut granted = Vec::new();

        if let Some(holders) = self.held.get_mut(name) {
            holders.retain(|h| h.id != holder_id);

            // Try to grant waiting requests.
            if let Some(waiters) = self.waiting.get_mut(name) {
                let mut remaining = VecDeque::new();
                while let Some(req) = waiters.pop_front() {
                    if Self::can_grant(holders, req.mode) {
                        holders.push(LockHolder {
                            id: req.requester_id,
                            mode: req.mode,
                            acquired_at: Utc::now(),
                        });
                        granted.push(req.requester_id);
                    } else {
                        remaining.push_back(req);
                        // Stop granting — FIFO means we can't skip.
                        break;
                    }
                }
                // Put remaining waiters back, plus any we haven't popped.
                while let Some(req) = waiters.pop_front() {
                    remaining.push_back(req);
                }
                *waiters = remaining;
            }

            // Clean up empty entries.
            if holders.is_empty() {
                self.held.remove(name);
            }
        }

        granted
    }

    /// Query the current lock state.
    pub fn query(&self) -> LockSnapshot {
        let mut held = Vec::new();
        let mut pending = Vec::new();

        for (name, holders) in &self.held {
            for h in holders {
                held.push(LockInfo {
                    name: name.clone(),
                    mode: h.mode,
                    requester_id: h.id,
                });
            }
        }

        for (name, waiters) in &self.waiting {
            for w in waiters {
                pending.push(LockInfo {
                    name: name.clone(),
                    mode: w.mode,
                    requester_id: w.requester_id,
                });
            }
        }

        LockSnapshot { held, pending }
    }

    /// Check if a lock with the given mode can be granted given current holders.
    fn can_grant(holders: &[LockHolder], mode: LockMode) -> bool {
        if holders.is_empty() {
            return true;
        }
        match mode {
            LockMode::Shared => holders.iter().all(|h| h.mode == LockMode::Shared),
            LockMode::Exclusive => false,
        }
    }
}

impl Default for LockState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_exclusive() {
        let mut state = LockState::new();
        let id = Uuid::new_v4();
        let result = state.request("res", LockMode::Exclusive, id);
        assert_eq!(result, LockResult::Acquired);
    }

    #[test]
    fn shared_coexist() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        assert_eq!(
            state.request("res", LockMode::Shared, id1),
            LockResult::Acquired
        );
        assert_eq!(
            state.request("res", LockMode::Shared, id2),
            LockResult::Acquired
        );
    }

    #[test]
    fn exclusive_blocks_shared() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        assert_eq!(
            state.request("res", LockMode::Exclusive, id1),
            LockResult::Acquired
        );
        assert_eq!(
            state.request("res", LockMode::Shared, id2),
            LockResult::Waiting
        );
    }

    #[test]
    fn shared_blocks_exclusive() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        assert_eq!(
            state.request("res", LockMode::Shared, id1),
            LockResult::Acquired
        );
        assert_eq!(
            state.request("res", LockMode::Exclusive, id2),
            LockResult::Waiting
        );
    }

    #[test]
    fn release_grants_next() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        state.request("res", LockMode::Exclusive, id1);
        state.request("res", LockMode::Exclusive, id2);

        let granted = state.release("res", id1);
        assert_eq!(granted, vec![id2]);
    }

    #[test]
    fn query_snapshot() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        state.request("res", LockMode::Exclusive, id1);
        state.request("res", LockMode::Shared, id2);

        let snap = state.query();
        assert_eq!(snap.held.len(), 1);
        assert_eq!(snap.pending.len(), 1);
        assert_eq!(snap.held[0].requester_id, id1);
        assert_eq!(snap.pending[0].requester_id, id2);
    }

    #[test]
    fn fifo_ordering() {
        let mut state = LockState::new();
        let holder = Uuid::new_v4();
        let w1 = Uuid::new_v4();
        let w2 = Uuid::new_v4();
        let w3 = Uuid::new_v4();

        state.request("res", LockMode::Exclusive, holder);
        state.request("res", LockMode::Exclusive, w1);
        state.request("res", LockMode::Exclusive, w2);
        state.request("res", LockMode::Exclusive, w3);

        let granted = state.release("res", holder);
        assert_eq!(granted, vec![w1]);

        let granted = state.release("res", w1);
        assert_eq!(granted, vec![w2]);
    }

    #[test]
    fn release_nonexistent() {
        let mut state = LockState::new();
        let granted = state.release("nope", Uuid::new_v4());
        assert!(granted.is_empty());
    }

    #[test]
    fn release_grants_multiple_shared() {
        let mut state = LockState::new();
        let holder = Uuid::new_v4();
        let w1 = Uuid::new_v4();
        let w2 = Uuid::new_v4();

        state.request("res", LockMode::Exclusive, holder);
        state.request("res", LockMode::Shared, w1);
        state.request("res", LockMode::Shared, w2);

        let granted = state.release("res", holder);
        assert_eq!(granted.len(), 2);
        assert!(granted.contains(&w1));
        assert!(granted.contains(&w2));
    }

    #[test]
    fn independent_resources() {
        let mut state = LockState::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        assert_eq!(
            state.request("a", LockMode::Exclusive, id1),
            LockResult::Acquired
        );
        assert_eq!(
            state.request("b", LockMode::Exclusive, id2),
            LockResult::Acquired
        );
    }
}
