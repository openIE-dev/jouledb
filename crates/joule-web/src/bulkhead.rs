//! Bulkhead pattern — concurrency isolation per resource with semaphore-based
//! limiting, bounded queuing, named bulkheads, hierarchical bulkheads, and metrics.
//!
//! Pure Rust implementation of the bulkhead pattern for resilience.
//! Prevents a single failing resource from consuming all capacity.

use std::collections::HashMap;

// ── Bulkhead Decision ───────────────────────────────────────────

/// Decision made by a bulkhead when a request arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkheadDecision {
    /// Request admitted — a permit was acquired.
    Admitted,
    /// Request queued — waiting for a permit.
    Queued,
    /// Request rejected — no capacity and queue full.
    Rejected,
}

// ── Bulkhead Metrics ────────────────────────────────────────────

/// Metrics for a single bulkhead.
#[derive(Debug, Clone, Default)]
pub struct BulkheadMetrics {
    pub total_admitted: u64,
    pub total_rejected: u64,
    pub total_queued: u64,
    pub total_completed: u64,
    pub total_dequeued: u64,
    pub peak_concurrent: usize,
    pub peak_queue_depth: usize,
}

impl BulkheadMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rejection_rate(&self) -> f64 {
        let total = self.total_admitted + self.total_rejected + self.total_queued;
        if total == 0 {
            return 0.0;
        }
        self.total_rejected as f64 / total as f64
    }

    pub fn utilization(&self, max_concurrent: usize) -> f64 {
        if max_concurrent == 0 {
            return 0.0;
        }
        self.peak_concurrent as f64 / max_concurrent as f64
    }
}

// ── Semaphore ───────────────────────────────────────────────────

/// Simple counting semaphore for concurrency control.
#[derive(Debug)]
pub struct Semaphore {
    max_permits: usize,
    available: usize,
}

impl Semaphore {
    pub fn new(max_permits: usize) -> Self {
        Self {
            max_permits: max_permits.max(1),
            available: max_permits.max(1),
        }
    }

    /// Try to acquire one permit. Returns true if acquired.
    pub fn try_acquire(&mut self) -> bool {
        if self.available > 0 {
            self.available -= 1;
            true
        } else {
            false
        }
    }

    /// Release one permit.
    pub fn release(&mut self) {
        if self.available < self.max_permits {
            self.available += 1;
        }
    }

    pub fn available(&self) -> usize {
        self.available
    }

    pub fn max_permits(&self) -> usize {
        self.max_permits
    }

    pub fn in_use(&self) -> usize {
        self.max_permits - self.available
    }
}

// ── Bulkhead ────────────────────────────────────────────────────

/// A single bulkhead for concurrency isolation.
#[derive(Debug)]
pub struct Bulkhead {
    name: String,
    semaphore: Semaphore,
    queue_capacity: usize,
    queue_depth: usize,
    metrics: BulkheadMetrics,
}

impl Bulkhead {
    pub fn new(name: impl Into<String>, max_concurrent: usize, queue_capacity: usize) -> Self {
        Self {
            name: name.into(),
            semaphore: Semaphore::new(max_concurrent),
            queue_capacity,
            queue_depth: 0,
            metrics: BulkheadMetrics::new(),
        }
    }

    /// Try to enter the bulkhead.
    pub fn try_enter(&mut self) -> BulkheadDecision {
        if self.semaphore.try_acquire() {
            self.metrics.total_admitted += 1;
            let concurrent = self.semaphore.in_use();
            if concurrent > self.metrics.peak_concurrent {
                self.metrics.peak_concurrent = concurrent;
            }
            BulkheadDecision::Admitted
        } else if self.queue_depth < self.queue_capacity {
            self.queue_depth += 1;
            self.metrics.total_queued += 1;
            if self.queue_depth > self.metrics.peak_queue_depth {
                self.metrics.peak_queue_depth = self.queue_depth;
            }
            BulkheadDecision::Queued
        } else {
            self.metrics.total_rejected += 1;
            BulkheadDecision::Rejected
        }
    }

    /// Complete a request — release the permit.
    pub fn complete(&mut self) {
        self.semaphore.release();
        self.metrics.total_completed += 1;

        // If anything is queued, admit it.
        if self.queue_depth > 0 && self.semaphore.try_acquire() {
            self.queue_depth -= 1;
            self.metrics.total_dequeued += 1;
            self.metrics.total_admitted += 1;
            let concurrent = self.semaphore.in_use();
            if concurrent > self.metrics.peak_concurrent {
                self.metrics.peak_concurrent = concurrent;
            }
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn active_count(&self) -> usize {
        self.semaphore.in_use()
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available()
    }

    pub fn max_concurrent(&self) -> usize {
        self.semaphore.max_permits()
    }

    pub fn queue_capacity(&self) -> usize {
        self.queue_capacity
    }

    pub fn is_full(&self) -> bool {
        self.semaphore.available() == 0 && self.queue_depth >= self.queue_capacity
    }

    pub fn metrics(&self) -> &BulkheadMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = BulkheadMetrics::new();
    }
}

// ── Bulkhead Registry ───────────────────────────────────────────

/// Registry of named bulkheads for managing multiple resources.
#[derive(Debug)]
pub struct BulkheadRegistry {
    bulkheads: HashMap<String, Bulkhead>,
}

impl BulkheadRegistry {
    pub fn new() -> Self {
        Self {
            bulkheads: HashMap::new(),
        }
    }

    /// Register a new bulkhead.
    pub fn register(&mut self, name: impl Into<String>, max_concurrent: usize, queue_capacity: usize) {
        let n = name.into();
        self.bulkheads
            .insert(n.clone(), Bulkhead::new(n, max_concurrent, queue_capacity));
    }

    /// Try to enter a named bulkhead.
    pub fn try_enter(&mut self, name: &str) -> Option<BulkheadDecision> {
        self.bulkheads.get_mut(name).map(|b| b.try_enter())
    }

    /// Complete a request on a named bulkhead.
    pub fn complete(&mut self, name: &str) -> bool {
        if let Some(b) = self.bulkheads.get_mut(name) {
            b.complete();
            true
        } else {
            false
        }
    }

    /// Get metrics for a named bulkhead.
    pub fn metrics(&self, name: &str) -> Option<&BulkheadMetrics> {
        self.bulkheads.get(name).map(|b| b.metrics())
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.bulkheads.keys().map(|k| k.as_str()).collect();
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.bulkheads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bulkheads.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Bulkhead> {
        self.bulkheads.get(name)
    }
}

impl Default for BulkheadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hierarchical Bulkhead ───────────────────────────────────────

/// A hierarchical bulkhead: parent + child. Both must admit for access.
#[derive(Debug)]
pub struct HierarchicalBulkhead {
    parent: Bulkhead,
    children: HashMap<String, Bulkhead>,
}

impl HierarchicalBulkhead {
    pub fn new(parent_name: impl Into<String>, parent_max: usize, parent_queue: usize) -> Self {
        Self {
            parent: Bulkhead::new(parent_name, parent_max, parent_queue),
            children: HashMap::new(),
        }
    }

    /// Add a child bulkhead.
    pub fn add_child(
        &mut self,
        name: impl Into<String>,
        max_concurrent: usize,
        queue_capacity: usize,
    ) {
        let n = name.into();
        self.children
            .insert(n.clone(), Bulkhead::new(n, max_concurrent, queue_capacity));
    }

    /// Try to enter via child. Parent must also admit.
    pub fn try_enter(&mut self, child_name: &str) -> Option<BulkheadDecision> {
        // First check parent.
        let parent_decision = self.parent.try_enter();
        match parent_decision {
            BulkheadDecision::Rejected => return Some(BulkheadDecision::Rejected),
            BulkheadDecision::Queued => return Some(BulkheadDecision::Queued),
            BulkheadDecision::Admitted => {}
        }

        // Then check child.
        if let Some(child) = self.children.get_mut(child_name) {
            let child_decision = child.try_enter();
            if child_decision == BulkheadDecision::Rejected {
                // Release parent since child rejected.
                self.parent.complete();
            }
            Some(child_decision)
        } else {
            // Unknown child — release parent.
            self.parent.complete();
            None
        }
    }

    /// Complete a request through a child.
    pub fn complete(&mut self, child_name: &str) -> bool {
        if let Some(child) = self.children.get_mut(child_name) {
            child.complete();
            self.parent.complete();
            true
        } else {
            false
        }
    }

    pub fn parent(&self) -> &Bulkhead {
        &self.parent
    }

    pub fn child(&self, name: &str) -> Option<&Bulkhead> {
        self.children.get(name)
    }

    pub fn child_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.children.keys().map(|k| k.as_str()).collect();
        names.sort();
        names
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semaphore_basic() {
        let mut sem = Semaphore::new(2);
        assert_eq!(sem.available(), 2);
        assert!(sem.try_acquire());
        assert_eq!(sem.available(), 1);
        assert!(sem.try_acquire());
        assert!(!sem.try_acquire()); // Exhausted.
        sem.release();
        assert_eq!(sem.available(), 1);
        assert!(sem.try_acquire());
    }

    #[test]
    fn test_semaphore_release_cap() {
        let mut sem = Semaphore::new(1);
        // Extra releases should not exceed max.
        sem.release();
        sem.release();
        assert_eq!(sem.available(), 1);
    }

    #[test]
    fn test_bulkhead_admit() {
        let mut bh = Bulkhead::new("test", 2, 0);
        assert_eq!(bh.try_enter(), BulkheadDecision::Admitted);
        assert_eq!(bh.try_enter(), BulkheadDecision::Admitted);
        assert_eq!(bh.try_enter(), BulkheadDecision::Rejected);
        assert_eq!(bh.active_count(), 2);
    }

    #[test]
    fn test_bulkhead_queue() {
        let mut bh = Bulkhead::new("test", 1, 2);
        assert_eq!(bh.try_enter(), BulkheadDecision::Admitted);
        assert_eq!(bh.try_enter(), BulkheadDecision::Queued);
        assert_eq!(bh.try_enter(), BulkheadDecision::Queued);
        assert_eq!(bh.try_enter(), BulkheadDecision::Rejected); // Queue full.
        assert_eq!(bh.queue_depth(), 2);
    }

    #[test]
    fn test_bulkhead_complete_dequeues() {
        let mut bh = Bulkhead::new("test", 1, 2);
        bh.try_enter(); // Admitted.
        bh.try_enter(); // Queued.
        assert_eq!(bh.queue_depth(), 1);
        bh.complete(); // Should dequeue.
        assert_eq!(bh.queue_depth(), 0);
        assert_eq!(bh.metrics().total_dequeued, 1);
    }

    #[test]
    fn test_bulkhead_is_full() {
        let mut bh = Bulkhead::new("test", 1, 1);
        bh.try_enter();
        assert!(!bh.is_full());
        bh.try_enter(); // Queue.
        assert!(bh.is_full());
    }

    #[test]
    fn test_bulkhead_metrics() {
        let mut bh = Bulkhead::new("test", 1, 0);
        bh.try_enter(); // Admitted.
        bh.try_enter(); // Rejected.
        bh.complete();
        assert_eq!(bh.metrics().total_admitted, 1);
        assert_eq!(bh.metrics().total_rejected, 1);
        assert_eq!(bh.metrics().total_completed, 1);
        assert_eq!(bh.metrics().peak_concurrent, 1);
    }

    #[test]
    fn test_bulkhead_rejection_rate() {
        let mut bh = Bulkhead::new("test", 1, 0);
        bh.try_enter();
        bh.try_enter();
        bh.try_enter();
        // 1 admitted, 2 rejected.
        let rate = bh.metrics().rejection_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bulkhead_utilization() {
        let mut bh = Bulkhead::new("test", 4, 0);
        bh.try_enter();
        bh.try_enter();
        let util = bh.metrics().utilization(4);
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_registry_basic() {
        let mut reg = BulkheadRegistry::new();
        reg.register("db", 5, 10);
        reg.register("api", 3, 5);
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.try_enter("db"), Some(BulkheadDecision::Admitted));
        assert!(reg.complete("db"));
        assert!(reg.try_enter("nonexistent").is_none());
    }

    #[test]
    fn test_registry_names_sorted() {
        let mut reg = BulkheadRegistry::new();
        reg.register("zebra", 1, 0);
        reg.register("alpha", 1, 0);
        assert_eq!(reg.names(), vec!["alpha", "zebra"]);
    }

    #[test]
    fn test_registry_metrics() {
        let mut reg = BulkheadRegistry::new();
        reg.register("svc", 1, 0);
        reg.try_enter("svc");
        let m = reg.metrics("svc").unwrap();
        assert_eq!(m.total_admitted, 1);
        assert!(reg.metrics("unknown").is_none());
    }

    #[test]
    fn test_hierarchical_both_admit() {
        let mut hb = HierarchicalBulkhead::new("parent", 10, 0);
        hb.add_child("child_a", 2, 0);
        assert_eq!(hb.try_enter("child_a"), Some(BulkheadDecision::Admitted));
        assert_eq!(hb.parent().active_count(), 1);
        assert_eq!(hb.child("child_a").unwrap().active_count(), 1);
    }

    #[test]
    fn test_hierarchical_child_rejects() {
        let mut hb = HierarchicalBulkhead::new("parent", 10, 0);
        hb.add_child("child_a", 1, 0);
        hb.try_enter("child_a"); // Admitted.
        let decision = hb.try_enter("child_a").unwrap(); // Child full.
        assert_eq!(decision, BulkheadDecision::Rejected);
        // Parent should have been released.
        assert_eq!(hb.parent().active_count(), 1);
    }

    #[test]
    fn test_hierarchical_parent_rejects() {
        let mut hb = HierarchicalBulkhead::new("parent", 1, 0);
        hb.add_child("child_a", 5, 0);
        hb.try_enter("child_a"); // Admitted.
        let decision = hb.try_enter("child_a").unwrap(); // Parent full.
        assert_eq!(decision, BulkheadDecision::Rejected);
    }

    #[test]
    fn test_hierarchical_complete() {
        let mut hb = HierarchicalBulkhead::new("parent", 1, 0);
        hb.add_child("child_a", 1, 0);
        hb.try_enter("child_a");
        assert!(hb.complete("child_a"));
        assert_eq!(hb.parent().active_count(), 0);
        assert_eq!(hb.child("child_a").unwrap().active_count(), 0);
    }

    #[test]
    fn test_hierarchical_unknown_child() {
        let mut hb = HierarchicalBulkhead::new("parent", 10, 0);
        assert!(hb.try_enter("nonexistent").is_none());
        // Parent should not have been consumed.
        assert_eq!(hb.parent().active_count(), 0);
    }

    #[test]
    fn test_hierarchical_child_names() {
        let mut hb = HierarchicalBulkhead::new("parent", 10, 0);
        hb.add_child("beta", 1, 0);
        hb.add_child("alpha", 1, 0);
        assert_eq!(hb.child_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_bulkhead_reset_metrics() {
        let mut bh = Bulkhead::new("test", 2, 0);
        bh.try_enter();
        bh.try_enter();
        bh.try_enter();
        bh.reset_metrics();
        assert_eq!(bh.metrics().total_admitted, 0);
        assert_eq!(bh.metrics().total_rejected, 0);
    }

    #[test]
    fn test_bulkhead_name() {
        let bh = Bulkhead::new("my_service", 5, 10);
        assert_eq!(bh.name(), "my_service");
        assert_eq!(bh.max_concurrent(), 5);
        assert_eq!(bh.queue_capacity(), 10);
    }

    #[test]
    fn test_peak_queue_depth() {
        let mut bh = Bulkhead::new("test", 1, 5);
        bh.try_enter(); // Admitted.
        bh.try_enter(); // Queued.
        bh.try_enter(); // Queued.
        bh.try_enter(); // Queued.
        assert_eq!(bh.metrics().peak_queue_depth, 3);
        bh.complete(); // Dequeue one.
        bh.complete();
        bh.complete();
        // Peak should still be 3.
        assert_eq!(bh.metrics().peak_queue_depth, 3);
    }
}
