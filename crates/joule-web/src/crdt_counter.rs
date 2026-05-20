//! CRDT counters — G-Counter (grow-only), PN-Counter (positive-negative),
//! state-based merge, operation-based increment/decrement, causally consistent,
//! replica management, counter value query.

use std::collections::HashMap;

// ── G-Counter (Grow-Only Counter) ────────────────────────────────────────────

/// A state-based grow-only counter (G-Counter).
/// Each replica maintains a separate counter. The value is the sum of all replicas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GCounter {
    /// This replica's id.
    replica_id: String,
    /// Map from replica id to that replica's count.
    counts: HashMap<String, u64>,
}

impl GCounter {
    /// Create a new G-Counter for the given replica.
    pub fn new(replica_id: &str) -> Self {
        Self {
            replica_id: replica_id.to_string(),
            counts: HashMap::new(),
        }
    }

    /// Create a G-Counter with initial state.
    pub fn from_state(replica_id: &str, state: &[(&str, u64)]) -> Self {
        let mut counts = HashMap::new();
        for (rid, val) in state {
            counts.insert(rid.to_string(), *val);
        }
        Self {
            replica_id: replica_id.to_string(),
            counts,
        }
    }

    /// Increment the counter on this replica by 1.
    pub fn increment(&mut self) {
        let entry = self.counts.entry(self.replica_id.clone()).or_insert(0);
        *entry += 1;
    }

    /// Increment the counter on this replica by `amount`.
    pub fn increment_by(&mut self, amount: u64) {
        let entry = self.counts.entry(self.replica_id.clone()).or_insert(0);
        *entry += amount;
    }

    /// Get the total value (sum of all replicas).
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Get the local value for this replica.
    pub fn local_value(&self) -> u64 {
        self.counts.get(&self.replica_id).copied().unwrap_or(0)
    }

    /// Get the value for a specific replica.
    pub fn value_for(&self, replica_id: &str) -> u64 {
        self.counts.get(replica_id).copied().unwrap_or(0)
    }

    /// Merge with another G-Counter (element-wise max).
    pub fn merge(&mut self, other: &GCounter) {
        for (rid, &val) in &other.counts {
            let entry = self.counts.entry(rid.clone()).or_insert(0);
            if val > *entry {
                *entry = val;
            }
        }
    }

    /// Create a merged counter without mutating self.
    pub fn merged(&self, other: &GCounter) -> GCounter {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Get the replica id.
    pub fn replica_id(&self) -> &str {
        &self.replica_id
    }

    /// Get the number of replicas that have contributed.
    pub fn replica_count(&self) -> usize {
        self.counts.len()
    }

    /// Check if the counter has ever been incremented.
    pub fn is_zero(&self) -> bool {
        self.counts.values().all(|v| *v == 0)
    }

    /// Compare: returns true if self <= other (every component is <=).
    pub fn less_than_or_equal(&self, other: &GCounter) -> bool {
        for (rid, &val) in &self.counts {
            if val > other.value_for(rid) {
                return false;
            }
        }
        true
    }
}

// ── PN-Counter (Positive-Negative Counter) ───────────────────────────────────

/// A state-based counter supporting both increment and decrement.
/// Internally uses two G-Counters: one for increments, one for decrements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PNCounter {
    /// This replica's id.
    replica_id: String,
    /// G-Counter for positive increments.
    positive: GCounter,
    /// G-Counter for negative decrements.
    negative: GCounter,
}

impl PNCounter {
    /// Create a new PN-Counter for the given replica.
    pub fn new(replica_id: &str) -> Self {
        Self {
            replica_id: replica_id.to_string(),
            positive: GCounter::new(replica_id),
            negative: GCounter::new(replica_id),
        }
    }

    /// Increment by 1.
    pub fn increment(&mut self) {
        self.positive.increment();
    }

    /// Increment by `amount`.
    pub fn increment_by(&mut self, amount: u64) {
        self.positive.increment_by(amount);
    }

    /// Decrement by 1.
    pub fn decrement(&mut self) {
        self.negative.increment();
    }

    /// Decrement by `amount`.
    pub fn decrement_by(&mut self, amount: u64) {
        self.negative.increment_by(amount);
    }

    /// Get the current value (positive - negative). Returns a signed integer.
    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    /// Get the total positive count.
    pub fn positive_value(&self) -> u64 {
        self.positive.value()
    }

    /// Get the total negative count.
    pub fn negative_value(&self) -> u64 {
        self.negative.value()
    }

    /// Merge with another PN-Counter.
    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }

    /// Create a merged counter without mutating self.
    pub fn merged(&self, other: &PNCounter) -> PNCounter {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Get the replica id.
    pub fn replica_id(&self) -> &str {
        &self.replica_id
    }

    /// Check if the counter is at zero (positive == negative).
    pub fn is_zero(&self) -> bool {
        self.value() == 0
    }

    /// Compare: returns true if self <= other for both positive and negative
    /// components.
    pub fn less_than_or_equal(&self, other: &PNCounter) -> bool {
        self.positive.less_than_or_equal(&other.positive)
            && self.negative.less_than_or_equal(&other.negative)
    }

    /// Get the number of distinct replicas contributing to increments.
    pub fn increment_replica_count(&self) -> usize {
        self.positive.replica_count()
    }

    /// Get the number of distinct replicas contributing to decrements.
    pub fn decrement_replica_count(&self) -> usize {
        self.negative.replica_count()
    }
}

// ── Operation-Based Counter ──────────────────────────────────────────────────

/// An operation to apply to a counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterOp {
    Increment(u64),
    Decrement(u64),
}

/// An operation-based counter that tracks operations for dissemination.
#[derive(Debug, Clone)]
pub struct OpCounter {
    /// Current value.
    value: i64,
    /// Pending operations not yet disseminated.
    pending_ops: Vec<CounterOp>,
    /// Replica id.
    replica_id: String,
    /// Sequence number for operations.
    seq: u64,
}

impl OpCounter {
    /// Create a new operation-based counter.
    pub fn new(replica_id: &str) -> Self {
        Self {
            value: 0,
            pending_ops: Vec::new(),
            replica_id: replica_id.to_string(),
            seq: 0,
        }
    }

    /// Increment and record the operation.
    pub fn increment(&mut self) {
        self.value += 1;
        self.seq += 1;
        self.pending_ops.push(CounterOp::Increment(1));
    }

    /// Increment by amount.
    pub fn increment_by(&mut self, amount: u64) {
        self.value += amount as i64;
        self.seq += 1;
        self.pending_ops.push(CounterOp::Increment(amount));
    }

    /// Decrement and record the operation.
    pub fn decrement(&mut self) {
        self.value -= 1;
        self.seq += 1;
        self.pending_ops.push(CounterOp::Decrement(1));
    }

    /// Decrement by amount.
    pub fn decrement_by(&mut self, amount: u64) {
        self.value -= amount as i64;
        self.seq += 1;
        self.pending_ops.push(CounterOp::Decrement(amount));
    }

    /// Get the current value.
    pub fn value(&self) -> i64 {
        self.value
    }

    /// Apply an operation from a remote replica.
    pub fn apply(&mut self, op: &CounterOp) {
        match op {
            CounterOp::Increment(n) => self.value += *n as i64,
            CounterOp::Decrement(n) => self.value -= *n as i64,
        }
    }

    /// Drain and return pending operations.
    pub fn drain_pending(&mut self) -> Vec<CounterOp> {
        std::mem::take(&mut self.pending_ops)
    }

    /// Get pending operations without draining.
    pub fn pending_ops(&self) -> &[CounterOp] {
        &self.pending_ops
    }

    /// Get the replica id.
    pub fn replica_id(&self) -> &str {
        &self.replica_id
    }

    /// Get the current sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── G-Counter tests ──

    #[test]
    fn gcounter_new_is_zero() {
        let c = GCounter::new("r1");
        assert_eq!(c.value(), 0);
        assert!(c.is_zero());
    }

    #[test]
    fn gcounter_increment() {
        let mut c = GCounter::new("r1");
        c.increment();
        c.increment();
        assert_eq!(c.value(), 2);
        assert_eq!(c.local_value(), 2);
    }

    #[test]
    fn gcounter_increment_by() {
        let mut c = GCounter::new("r1");
        c.increment_by(10);
        assert_eq!(c.value(), 10);
    }

    #[test]
    fn gcounter_merge_two_replicas() {
        let mut r1 = GCounter::new("r1");
        let mut r2 = GCounter::new("r2");
        r1.increment_by(3);
        r2.increment_by(5);
        r1.merge(&r2);
        assert_eq!(r1.value(), 8);
        assert_eq!(r1.value_for("r1"), 3);
        assert_eq!(r1.value_for("r2"), 5);
    }

    #[test]
    fn gcounter_merge_is_idempotent() {
        let mut r1 = GCounter::new("r1");
        let r2 = GCounter::from_state("r2", &[("r2", 5)]);
        r1.increment_by(3);
        r1.merge(&r2);
        r1.merge(&r2); // Merge again — should be idempotent.
        assert_eq!(r1.value(), 8);
    }

    #[test]
    fn gcounter_merge_is_commutative() {
        let mut r1 = GCounter::from_state("r1", &[("r1", 3)]);
        let mut r2 = GCounter::from_state("r2", &[("r2", 5)]);
        let m1 = r1.merged(&r2);
        r2.merge(&r1);
        assert_eq!(m1.value(), r2.value());
    }

    #[test]
    fn gcounter_less_than_or_equal() {
        let r1 = GCounter::from_state("r1", &[("r1", 3), ("r2", 2)]);
        let r2 = GCounter::from_state("r2", &[("r1", 3), ("r2", 5)]);
        assert!(r1.less_than_or_equal(&r2));
        assert!(!r2.less_than_or_equal(&r1));
    }

    #[test]
    fn gcounter_replica_count() {
        let mut c = GCounter::new("r1");
        c.increment();
        assert_eq!(c.replica_count(), 1);
        let r2 = GCounter::from_state("r2", &[("r2", 1)]);
        c.merge(&r2);
        assert_eq!(c.replica_count(), 2);
    }

    // ── PN-Counter tests ──

    #[test]
    fn pncounter_increment_and_decrement() {
        let mut c = PNCounter::new("r1");
        c.increment();
        c.increment();
        c.decrement();
        assert_eq!(c.value(), 1);
    }

    #[test]
    fn pncounter_negative_value() {
        let mut c = PNCounter::new("r1");
        c.decrement_by(5);
        assert_eq!(c.value(), -5);
    }

    #[test]
    fn pncounter_merge() {
        let mut r1 = PNCounter::new("r1");
        let mut r2 = PNCounter::new("r2");
        r1.increment_by(10);
        r1.decrement_by(3);
        r2.increment_by(5);
        r2.decrement_by(2);
        r1.merge(&r2);
        // r1: pos = 10+5=15, neg = 3+2=5 => value = 10.
        assert_eq!(r1.value(), 10);
    }

    #[test]
    fn pncounter_merge_is_commutative() {
        let mut r1 = PNCounter::new("r1");
        let mut r2 = PNCounter::new("r2");
        r1.increment_by(3);
        r2.decrement_by(2);
        let m1 = r1.merged(&r2);
        let m2 = r2.merged(&r1);
        assert_eq!(m1.value(), m2.value());
    }

    #[test]
    fn pncounter_is_zero() {
        let mut c = PNCounter::new("r1");
        assert!(c.is_zero());
        c.increment();
        c.decrement();
        assert!(c.is_zero());
    }

    #[test]
    fn pncounter_positive_negative_values() {
        let mut c = PNCounter::new("r1");
        c.increment_by(7);
        c.decrement_by(3);
        assert_eq!(c.positive_value(), 7);
        assert_eq!(c.negative_value(), 3);
    }

    #[test]
    fn pncounter_less_than_or_equal() {
        let mut r1 = PNCounter::new("r1");
        let mut r2 = PNCounter::new("r1");
        r1.increment_by(3);
        r2.increment_by(5);
        assert!(r1.less_than_or_equal(&r2));
    }

    // ── Op-Counter tests ──

    #[test]
    fn opcounter_basic() {
        let mut c = OpCounter::new("r1");
        c.increment();
        c.increment();
        c.decrement();
        assert_eq!(c.value(), 1);
    }

    #[test]
    fn opcounter_apply_remote() {
        let mut c = OpCounter::new("r1");
        c.apply(&CounterOp::Increment(5));
        c.apply(&CounterOp::Decrement(2));
        assert_eq!(c.value(), 3);
    }

    #[test]
    fn opcounter_drain_pending() {
        let mut c = OpCounter::new("r1");
        c.increment();
        c.decrement();
        let ops = c.drain_pending();
        assert_eq!(ops.len(), 2);
        assert!(c.pending_ops().is_empty());
    }

    #[test]
    fn opcounter_seq_advances() {
        let mut c = OpCounter::new("r1");
        assert_eq!(c.seq(), 0);
        c.increment();
        assert_eq!(c.seq(), 1);
        c.decrement_by(3);
        assert_eq!(c.seq(), 2);
    }

    #[test]
    fn gcounter_from_state() {
        let c = GCounter::from_state("r1", &[("a", 2), ("b", 3)]);
        assert_eq!(c.value(), 5);
        assert_eq!(c.value_for("a"), 2);
        assert_eq!(c.value_for("b"), 3);
    }

    #[test]
    fn three_replica_merge() {
        let mut r1 = GCounter::new("r1");
        let mut r2 = GCounter::new("r2");
        let mut r3 = GCounter::new("r3");
        r1.increment_by(1);
        r2.increment_by(2);
        r3.increment_by(3);
        r1.merge(&r2);
        r1.merge(&r3);
        assert_eq!(r1.value(), 6);
        assert_eq!(r1.replica_count(), 3);
    }
}
