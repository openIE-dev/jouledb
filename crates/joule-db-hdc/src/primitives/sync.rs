//! Primitive 15: Synchronize — temporal alignment / coordination barrier.
//!
//! Coordinates *when* things happen, not *what* they contain.
//! In concurrent HDC systems, multiple entities (spawned via Spawn)
//! operate independently. Synchronize is the primitive that brings
//! them into temporal alignment.
//!
//! Neuroscience: gamma-frequency synchronization binds distributed
//! representations into a coherent percept.
//! Distributed systems: barrier synchronization, consensus.

use crate::turbo_holographic::{BinaryHV, BundleAccumulator};

/// A synchronization barrier for multiple HDC entities.
#[derive(Clone, Debug)]
pub struct Barrier {
    /// Expected number of participants.
    pub expected: usize,
    /// Vectors that have arrived at the barrier.
    arrivals: Vec<(u64, BinaryHV)>, // (entity_id, vector)
    /// Whether the barrier has been released.
    pub released: bool,
    /// The synchronized result (available after release).
    result: Option<BinaryHV>,
}

impl Barrier {
    /// Create a barrier expecting `n` participants.
    pub fn new(expected: usize) -> Self {
        Self {
            expected,
            arrivals: Vec::with_capacity(expected),
            released: false,
            result: None,
        }
    }

    /// An entity arrives at the barrier with its current vector.
    /// Returns true if this arrival triggered the barrier release.
    pub fn arrive(&mut self, entity_id: u64, vector: BinaryHV) -> bool {
        if self.released {
            return false;
        }

        // Don't double-count
        if self.arrivals.iter().any(|(id, _)| *id == entity_id) {
            return false;
        }

        self.arrivals.push((entity_id, vector));

        if self.arrivals.len() >= self.expected {
            self.release();
            true
        } else {
            false
        }
    }

    /// How many entities have arrived.
    pub fn arrived_count(&self) -> usize {
        self.arrivals.len()
    }

    /// How many entities are still missing.
    pub fn remaining(&self) -> usize {
        self.expected.saturating_sub(self.arrivals.len())
    }

    /// Force-release the barrier even if not all participants arrived.
    /// Useful for timeouts.
    pub fn force_release(&mut self) {
        if !self.released {
            self.release();
        }
    }

    /// Get the synchronized result: the bundle of all arrived vectors.
    /// Only available after the barrier is released.
    pub fn synchronized(&self) -> Option<&BinaryHV> {
        self.result.as_ref()
    }

    /// Get individual arrivals (entity_id, vector).
    pub fn arrivals(&self) -> &[(u64, BinaryHV)] {
        &self.arrivals
    }

    /// Internal: compute the synchronized result via majority-vote bundle.
    fn release(&mut self) {
        if self.arrivals.is_empty() {
            self.released = true;
            return;
        }

        let dim = self.arrivals[0].1.dimension();
        let mut acc = BundleAccumulator::new(dim);
        for (_, vec) in &self.arrivals {
            acc.add(vec);
        }
        self.result = Some(acc.threshold());
        self.released = true;
    }
}

/// A group of named synchronization points.
/// Entities can synchronize at named checkpoints independently.
#[derive(Clone, Debug)]
pub struct SyncGroup {
    barriers: Vec<(String, Barrier)>,
}

impl SyncGroup {
    /// Create a new sync group.
    pub fn new() -> Self {
        Self {
            barriers: Vec::new(),
        }
    }

    /// Create or get a named barrier.
    pub fn barrier(&mut self, name: &str, expected: usize) -> &mut Barrier {
        if let Some(pos) = self.barriers.iter().position(|(n, _)| n == name) {
            &mut self.barriers[pos].1
        } else {
            self.barriers.push((name.to_string(), Barrier::new(expected)));
            let last = self.barriers.len() - 1;
            &mut self.barriers[last].1
        }
    }

    /// Check if a named barrier has been released.
    pub fn is_released(&self, name: &str) -> bool {
        self.barriers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.released)
            .unwrap_or(false)
    }

    /// Get the synchronized result from a named barrier.
    pub fn get_synchronized(&self, name: &str) -> Option<&BinaryHV> {
        self.barriers
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, b)| b.synchronized())
    }

    /// How many barriers exist in this group.
    pub fn barrier_count(&self) -> usize {
        self.barriers.len()
    }
}

impl Default for SyncGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for entities that can participate in synchronization.
pub trait Synchronizable {
    /// Get this entity's current synchronization vector.
    fn sync_vector(&self) -> BinaryHV;

    /// Get this entity's ID for barrier tracking.
    fn sync_id(&self) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_basic() {
        let mut barrier = Barrier::new(3);
        assert_eq!(barrier.remaining(), 3);
        assert!(!barrier.released);

        let a = BinaryHV::random(1000, 1);
        let b = BinaryHV::random(1000, 2);
        let c = BinaryHV::random(1000, 3);

        assert!(!barrier.arrive(1, a));
        assert_eq!(barrier.remaining(), 2);

        assert!(!barrier.arrive(2, b));
        assert_eq!(barrier.remaining(), 1);

        assert!(barrier.arrive(3, c)); // This one triggers release
        assert!(barrier.released);
        assert!(barrier.synchronized().is_some());
    }

    #[test]
    fn test_barrier_no_double_count() {
        let mut barrier = Barrier::new(2);
        let a = BinaryHV::random(1000, 1);

        barrier.arrive(1, a.clone());
        barrier.arrive(1, a); // Same entity, should be ignored
        assert_eq!(barrier.arrived_count(), 1);
        assert!(!barrier.released);
    }

    #[test]
    fn test_barrier_result_is_bundle() {
        let mut barrier = Barrier::new(2);
        let a = BinaryHV::random(1000, 1);
        let b = BinaryHV::random(1000, 2);

        barrier.arrive(1, a.clone());
        barrier.arrive(2, b.clone());

        let result = barrier.synchronized().unwrap();
        // Result should be similar to both inputs (it's a bundle)
        let sim_a = result.similarity(&a);
        let sim_b = result.similarity(&b);
        assert!(sim_a > 0.4);
        assert!(sim_b > 0.4);
    }

    #[test]
    fn test_force_release() {
        let mut barrier = Barrier::new(5);
        let a = BinaryHV::random(1000, 1);
        barrier.arrive(1, a);
        assert!(!barrier.released);

        barrier.force_release();
        assert!(barrier.released);
        assert!(barrier.synchronized().is_some());
    }

    #[test]
    fn test_sync_group() {
        let mut group = SyncGroup::new();

        // Two independent sync points
        let a = BinaryHV::random(1000, 1);
        let b = BinaryHV::random(1000, 2);

        group.barrier("phase1", 2).arrive(1, a.clone());
        group.barrier("phase1", 2).arrive(2, b.clone());

        assert!(group.is_released("phase1"));
        assert!(!group.is_released("phase2"));
        assert!(group.get_synchronized("phase1").is_some());
    }

    #[test]
    fn test_sync_group_multiple_barriers() {
        let mut group = SyncGroup::new();

        group.barrier("alpha", 1).arrive(1, BinaryHV::random(1000, 1));
        group.barrier("beta", 1).arrive(2, BinaryHV::random(1000, 2));

        assert_eq!(group.barrier_count(), 2);
        assert!(group.is_released("alpha"));
        assert!(group.is_released("beta"));
    }
}
