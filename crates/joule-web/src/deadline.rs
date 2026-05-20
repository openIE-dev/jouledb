//! Deadline/timeout manager — create, check, propagate, and cascade deadlines.
//!
//! Replaces ad-hoc timeout tracking with a structured deadline system. Supports
//! parent-child relationships where children inherit the tighter of their own
//! deadline or the parent's, and cascading cancellation through the tree.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Deadline ID ───────────────────────────────────────────────

/// Unique identifier for a deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeadlineId(pub u64);

// ── Deadline State ────────────────────────────────────────────

/// State of a deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlineState {
    /// Deadline is active and has not yet expired.
    Active,
    /// Deadline has expired naturally.
    Expired,
    /// Deadline was explicitly cancelled.
    Cancelled,
}

// ── Deadline Entry ────────────────────────────────────────────

/// A single deadline entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineEntry {
    pub id: DeadlineId,
    /// Label for identification.
    pub label: String,
    /// The absolute timestamp (in ms) at which this deadline expires.
    pub deadline_ms: u64,
    /// When this deadline was created (in ms).
    pub created_ms: u64,
    /// Current state.
    pub state: DeadlineState,
    /// Parent deadline ID, if any.
    pub parent: Option<DeadlineId>,
    /// Child deadline IDs.
    pub children: Vec<DeadlineId>,
}

impl DeadlineEntry {
    /// Original duration of this deadline in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.deadline_ms.saturating_sub(self.created_ms)
    }

    /// Time remaining given the current clock.
    pub fn remaining_ms(&self, now_ms: u64) -> u64 {
        self.deadline_ms.saturating_sub(now_ms)
    }

    /// Whether this deadline has expired at the given time.
    pub fn is_expired_at(&self, now_ms: u64) -> bool {
        self.state == DeadlineState::Expired
            || self.state == DeadlineState::Cancelled
            || now_ms >= self.deadline_ms
    }
}

// ── Deadline Statistics ───────────────────────────────────────

/// Statistics about deadline usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeadlineStats {
    pub total_created: u64,
    pub total_expired: u64,
    pub total_cancelled: u64,
    pub active_count: u64,
    pub cascade_cancellations: u64,
    pub child_deadlines_created: u64,
}

// ── Deadline Manager ──────────────────────────────────────────

/// Manages deadlines with parent-child relationships and cascading cancellation.
#[derive(Debug)]
pub struct DeadlineManager {
    deadlines: HashMap<DeadlineId, DeadlineEntry>,
    next_id: u64,
    stats: DeadlineStats,
}

impl DeadlineManager {
    /// Create a new deadline manager.
    pub fn new() -> Self {
        Self {
            deadlines: HashMap::new(),
            next_id: 1,
            stats: DeadlineStats::default(),
        }
    }

    /// Create a deadline from a duration.
    ///
    /// `now_ms` is the current clock time, `duration_ms` is how long until expiry.
    pub fn create(&mut self, now_ms: u64, duration_ms: u64, label: &str) -> DeadlineId {
        let id = DeadlineId(self.next_id);
        self.next_id += 1;

        let entry = DeadlineEntry {
            id,
            label: label.to_string(),
            deadline_ms: now_ms.saturating_add(duration_ms),
            created_ms: now_ms,
            state: DeadlineState::Active,
            parent: None,
            children: Vec::new(),
        };
        self.deadlines.insert(id, entry);
        self.stats.total_created += 1;
        self.stats.active_count += 1;
        id
    }

    /// Create a child deadline that inherits the tighter of its own or parent's deadline.
    pub fn create_child(
        &mut self,
        parent_id: DeadlineId,
        now_ms: u64,
        duration_ms: u64,
        label: &str,
    ) -> Option<DeadlineId> {
        // Read parent info first.
        let parent_deadline_ms = self.deadlines.get(&parent_id)?.deadline_ms;
        let parent_state = self.deadlines.get(&parent_id)?.state;

        // Parent must be active.
        if parent_state != DeadlineState::Active {
            return None;
        }

        let child_candidate = now_ms.saturating_add(duration_ms);
        // Inherit the tighter deadline.
        let effective_deadline = child_candidate.min(parent_deadline_ms);

        let child_id = DeadlineId(self.next_id);
        self.next_id += 1;

        let entry = DeadlineEntry {
            id: child_id,
            label: label.to_string(),
            deadline_ms: effective_deadline,
            created_ms: now_ms,
            state: DeadlineState::Active,
            parent: Some(parent_id),
            children: Vec::new(),
        };
        self.deadlines.insert(child_id, entry);

        // Register child with parent.
        if let Some(parent) = self.deadlines.get_mut(&parent_id) {
            parent.children.push(child_id);
        }

        self.stats.total_created += 1;
        self.stats.active_count += 1;
        self.stats.child_deadlines_created += 1;
        Some(child_id)
    }

    /// Check if a deadline has expired at the given time.
    pub fn is_expired(&self, id: DeadlineId, now_ms: u64) -> Option<bool> {
        self.deadlines.get(&id).map(|e| e.is_expired_at(now_ms))
    }

    /// Get the remaining time for a deadline.
    pub fn remaining(&self, id: DeadlineId, now_ms: u64) -> Option<u64> {
        self.deadlines.get(&id).map(|e| {
            if e.is_expired_at(now_ms) {
                0
            } else {
                e.remaining_ms(now_ms)
            }
        })
    }

    /// Get a deadline entry.
    pub fn get(&self, id: DeadlineId) -> Option<&DeadlineEntry> {
        self.deadlines.get(&id)
    }

    /// Tick the deadline manager — check all active deadlines and expire those past due.
    /// Returns IDs of newly expired deadlines.
    pub fn tick(&mut self, now_ms: u64) -> Vec<DeadlineId> {
        let mut newly_expired = Vec::new();

        let ids: Vec<DeadlineId> = self.deadlines.keys().copied().collect();
        for id in ids {
            let should_expire = {
                let entry = &self.deadlines[&id];
                entry.state == DeadlineState::Active && now_ms >= entry.deadline_ms
            };
            if should_expire {
                if let Some(entry) = self.deadlines.get_mut(&id) {
                    entry.state = DeadlineState::Expired;
                    self.stats.total_expired += 1;
                    self.stats.active_count = self.stats.active_count.saturating_sub(1);
                    newly_expired.push(id);
                }
            }
        }

        // Cascade: expire children of expired deadlines.
        let mut cascade_queue: Vec<DeadlineId> = newly_expired.clone();
        while let Some(parent_id) = cascade_queue.pop() {
            let children: Vec<DeadlineId> = self.deadlines
                .get(&parent_id)
                .map(|e| e.children.clone())
                .unwrap_or_default();

            for child_id in children {
                let should_cascade = self.deadlines
                    .get(&child_id)
                    .map(|e| e.state == DeadlineState::Active)
                    .unwrap_or(false);

                if should_cascade {
                    if let Some(entry) = self.deadlines.get_mut(&child_id) {
                        entry.state = DeadlineState::Expired;
                        self.stats.total_expired += 1;
                        self.stats.active_count = self.stats.active_count.saturating_sub(1);
                        self.stats.cascade_cancellations += 1;
                        newly_expired.push(child_id);
                        cascade_queue.push(child_id);
                    }
                }
            }
        }

        newly_expired
    }

    /// Cancel a deadline and all its children (cascading cancellation).
    pub fn cancel(&mut self, id: DeadlineId) -> bool {
        let exists = self.deadlines
            .get(&id)
            .map(|e| e.state == DeadlineState::Active)
            .unwrap_or(false);

        if !exists {
            return false;
        }

        let mut queue = vec![id];
        while let Some(current) = queue.pop() {
            let children: Vec<DeadlineId> = self.deadlines
                .get(&current)
                .map(|e| e.children.clone())
                .unwrap_or_default();

            if let Some(entry) = self.deadlines.get_mut(&current) {
                if entry.state == DeadlineState::Active {
                    entry.state = DeadlineState::Cancelled;
                    self.stats.total_cancelled += 1;
                    self.stats.active_count = self.stats.active_count.saturating_sub(1);
                    if current != id {
                        self.stats.cascade_cancellations += 1;
                    }
                }
            }
            queue.extend(children);
        }

        true
    }

    /// Get all active deadline IDs.
    pub fn active_deadlines(&self) -> Vec<DeadlineId> {
        self.deadlines
            .values()
            .filter(|e| e.state == DeadlineState::Active)
            .map(|e| e.id)
            .collect()
    }

    /// Get all deadlines that will expire within `window_ms` of the given time.
    pub fn expiring_soon(&self, now_ms: u64, window_ms: u64) -> Vec<DeadlineId> {
        let cutoff = now_ms.saturating_add(window_ms);
        self.deadlines
            .values()
            .filter(|e| {
                e.state == DeadlineState::Active
                    && e.deadline_ms >= now_ms
                    && e.deadline_ms <= cutoff
            })
            .map(|e| e.id)
            .collect()
    }

    /// Get the number of active deadlines.
    pub fn active_count(&self) -> usize {
        self.deadlines
            .values()
            .filter(|e| e.state == DeadlineState::Active)
            .count()
    }

    /// Get deadline statistics.
    pub fn stats(&self) -> &DeadlineStats {
        &self.stats
    }

    /// Get total number of deadlines (all states).
    pub fn total_count(&self) -> usize {
        self.deadlines.len()
    }
}

impl Default for DeadlineManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_deadline() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(1000, 500, "test");
        let entry = mgr.get(id).unwrap();
        assert_eq!(entry.deadline_ms, 1500);
        assert_eq!(entry.duration_ms(), 500);
        assert_eq!(entry.state, DeadlineState::Active);
    }

    #[test]
    fn test_deadline_not_expired() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(1000, 500, "test");
        assert_eq!(mgr.is_expired(id, 1200), Some(false));
        assert_eq!(mgr.remaining(id, 1200), Some(300));
    }

    #[test]
    fn test_deadline_expired() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(1000, 500, "test");
        assert_eq!(mgr.is_expired(id, 1500), Some(true));
        assert_eq!(mgr.remaining(id, 1500), Some(0));
    }

    #[test]
    fn test_tick_expires_deadlines() {
        let mut mgr = DeadlineManager::new();
        let id1 = mgr.create(0, 100, "short");
        let _id2 = mgr.create(0, 200, "long");
        let expired = mgr.tick(150);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], id1);
    }

    #[test]
    fn test_tick_expires_multiple() {
        let mut mgr = DeadlineManager::new();
        mgr.create(0, 100, "a");
        mgr.create(0, 100, "b");
        let expired = mgr.tick(100);
        assert_eq!(expired.len(), 2);
    }

    #[test]
    fn test_cancel_deadline() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(0, 1000, "cancel-me");
        assert!(mgr.cancel(id));
        let entry = mgr.get(id).unwrap();
        assert_eq!(entry.state, DeadlineState::Cancelled);
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut mgr = DeadlineManager::new();
        assert!(!mgr.cancel(DeadlineId(999)));
    }

    #[test]
    fn test_cancel_already_expired() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(0, 100, "expire");
        mgr.tick(200);
        assert!(!mgr.cancel(id));
    }

    #[test]
    fn test_child_deadline_inherits_parent() {
        let mut mgr = DeadlineManager::new();
        let parent = mgr.create(0, 100, "parent");
        // Child wants 200ms but parent only has 100ms — should inherit 100ms.
        let child = mgr.create_child(parent, 0, 200, "child").unwrap();
        let entry = mgr.get(child).unwrap();
        assert_eq!(entry.deadline_ms, 100); // Inherited from parent.
    }

    #[test]
    fn test_child_deadline_own_shorter() {
        let mut mgr = DeadlineManager::new();
        let parent = mgr.create(0, 1000, "parent");
        // Child wants 50ms — shorter than parent, keeps its own.
        let child = mgr.create_child(parent, 0, 50, "child").unwrap();
        let entry = mgr.get(child).unwrap();
        assert_eq!(entry.deadline_ms, 50);
    }

    #[test]
    fn test_cascade_cancellation() {
        let mut mgr = DeadlineManager::new();
        let parent = mgr.create(0, 1000, "parent");
        let child1 = mgr.create_child(parent, 0, 500, "child1").unwrap();
        let grandchild = mgr.create_child(child1, 0, 300, "grandchild").unwrap();

        mgr.cancel(parent);
        assert_eq!(mgr.get(parent).unwrap().state, DeadlineState::Cancelled);
        assert_eq!(mgr.get(child1).unwrap().state, DeadlineState::Cancelled);
        assert_eq!(mgr.get(grandchild).unwrap().state, DeadlineState::Cancelled);
    }

    #[test]
    fn test_cascade_expiry_propagates() {
        let mut mgr = DeadlineManager::new();
        let parent = mgr.create(0, 100, "parent");
        let child = mgr.create_child(parent, 0, 200, "child").unwrap();

        let expired = mgr.tick(100);
        // Both should expire — child inherited parent's deadline.
        assert!(expired.contains(&parent));
        assert!(expired.contains(&child));
    }

    #[test]
    fn test_active_deadlines() {
        let mut mgr = DeadlineManager::new();
        mgr.create(0, 100, "a");
        let id_b = mgr.create(0, 200, "b");
        mgr.create(0, 300, "c");
        mgr.tick(150); // Expires "a"
        let active = mgr.active_deadlines();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&id_b));
    }

    #[test]
    fn test_expiring_soon() {
        let mut mgr = DeadlineManager::new();
        mgr.create(0, 100, "soon");
        mgr.create(0, 500, "later");
        let soon = mgr.expiring_soon(0, 200);
        assert_eq!(soon.len(), 1);
    }

    #[test]
    fn test_stats_tracking() {
        let mut mgr = DeadlineManager::new();
        mgr.create(0, 100, "a");
        mgr.create(0, 200, "b");
        let id_c = mgr.create(0, 300, "c");
        assert_eq!(mgr.stats().total_created, 3);
        assert_eq!(mgr.stats().active_count, 3);
        mgr.tick(150);
        assert_eq!(mgr.stats().total_expired, 1);
        mgr.cancel(id_c);
        assert_eq!(mgr.stats().total_cancelled, 1);
    }

    #[test]
    fn test_child_of_expired_parent_fails() {
        let mut mgr = DeadlineManager::new();
        let parent = mgr.create(0, 100, "parent");
        mgr.tick(200); // Expire parent
        let child = mgr.create_child(parent, 200, 50, "child");
        assert!(child.is_none());
    }

    #[test]
    fn test_default_constructor() {
        let mgr = DeadlineManager::default();
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.total_count(), 0);
    }

    #[test]
    fn test_remaining_after_cancel() {
        let mut mgr = DeadlineManager::new();
        let id = mgr.create(0, 1000, "test");
        mgr.cancel(id);
        assert_eq!(mgr.remaining(id, 50), Some(0));
    }
}
