//! Primitive 12: Spawn — create a new independent computational entity.
//!
//! None of the other primitives produce an autonomous process.
//! Encode maps data. Bind creates relationships. Merge combines.
//! Spawn creates something that *runs on its own*.
//!
//! In HDC terms: generate a fresh random hypervector codebook and
//! give it its own update loop. The spawned entity has its own
//! centroid, its own contrast history, its own decay schedule.
//!
//! Biological analogy: clonal expansion in the immune system.
//! When a B-cell recognizes a novel antigen, it doesn't just
//! bind — it *proliferates*, creating independent copies that
//! each evolve separately via somatic hypermutation.

use crate::turbo_holographic::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

/// A spawned computational entity with its own state.
#[derive(Clone, Debug)]
pub struct SpawnedEntity {
    /// Unique identity vector — random, near-orthogonal to all other entities.
    pub identity: BinaryHV,
    /// The entity's centroid: its model of "normal."
    centroid_acc: BundleAccumulator,
    centroid: Option<BinaryHV>,
    /// Records ingested by this entity.
    pub record_count: u64,
    /// The entity's contrast history: novelty magnitudes.
    pub contrast_history: Vec<f64>,
    /// The parent entity's identity (if spawned from another).
    pub parent: Option<BinaryHV>,
    /// Named slots: the entity's learned bindings.
    pub slots: HashMap<String, BinaryHV>,
    /// Generation: how many spawns deep from the root.
    pub generation: u32,
    /// Whether this entity is active.
    pub active: bool,
}

impl SpawnedEntity {
    /// Create a root entity (no parent).
    pub fn root(dimension: usize, seed: u64) -> Self {
        Self {
            identity: BinaryHV::random(dimension, seed),
            centroid_acc: BundleAccumulator::new(dimension),
            centroid: None,
            record_count: 0,
            contrast_history: Vec::new(),
            parent: None,
            slots: HashMap::new(),
            generation: 0,
            active: true,
        }
    }

    /// Ingest a vector: update centroid and record contrast.
    pub fn ingest(&mut self, hv: &BinaryHV) -> f64 {
        let novelty = match &self.centroid {
            Some(c) => 1.0 - hv.similarity(c) as f64,
            None => 1.0,
        };

        self.centroid_acc.add(hv);
        self.record_count += 1;
        if self.record_count <= 1 || self.record_count % 10 == 0 {
            self.centroid = Some(self.centroid_acc.threshold());
        }

        self.contrast_history.push(novelty);
        novelty
    }

    /// Get this entity's centroid (what it considers "normal").
    pub fn centroid(&self) -> Option<&BinaryHV> {
        self.centroid.as_ref()
    }

    /// Bind a named concept to this entity's slot.
    pub fn bind_slot(&mut self, name: &str, value: BinaryHV) {
        self.slots.insert(name.to_string(), value);
    }

    /// Average contrast seen by this entity.
    pub fn average_contrast(&self) -> f64 {
        if self.contrast_history.is_empty() {
            return 0.0;
        }
        self.contrast_history.iter().sum::<f64>() / self.contrast_history.len() as f64
    }

    /// Deactivate this entity (soft delete — identity preserved).
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Trait for things that can spawn new entities.
pub trait Spawner {
    /// Spawn a child entity. The child gets:
    /// - A new random identity (near-orthogonal to parent)
    /// - A copy of the parent's centroid as starting context
    /// - Parent's generation + 1
    fn spawn(&self, child_seed: u64) -> SpawnedEntity;

    /// Spawn N children (clonal expansion).
    fn spawn_n(&self, n: usize, base_seed: u64) -> Vec<SpawnedEntity>;
}

impl Spawner for SpawnedEntity {
    fn spawn(&self, child_seed: u64) -> SpawnedEntity {
        let dim = self.identity.dimension();
        let mut child = SpawnedEntity {
            identity: BinaryHV::random(dim, child_seed),
            centroid_acc: BundleAccumulator::new(dim),
            centroid: self.centroid.clone(),
            record_count: 0,
            contrast_history: Vec::new(),
            parent: Some(self.identity.clone()),
            slots: HashMap::new(),
            generation: self.generation + 1,
            active: true,
        };

        // If parent has a centroid, prime the child's accumulator
        if let Some(ref c) = self.centroid {
            child.centroid_acc.add(c);
        }

        child
    }

    fn spawn_n(&self, n: usize, base_seed: u64) -> Vec<SpawnedEntity> {
        (0..n)
            .map(|i| self.spawn(base_seed.wrapping_add(i as u64)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_entity() {
        let entity = SpawnedEntity::root(10000, 42);
        assert_eq!(entity.generation, 0);
        assert!(entity.parent.is_none());
        assert!(entity.active);
    }

    #[test]
    fn test_spawn_child() {
        let parent = SpawnedEntity::root(10000, 42);
        let child = parent.spawn(99);
        assert_eq!(child.generation, 1);
        assert!(child.parent.is_some());
        // Child identity should be different from parent
        let sim = child.identity.similarity(&parent.identity);
        assert!(sim < 0.6, "child should be near-orthogonal to parent");
    }

    #[test]
    fn test_clonal_expansion() {
        let parent = SpawnedEntity::root(10000, 42);
        let children = parent.spawn_n(5, 100);
        assert_eq!(children.len(), 5);
        // All children should be near-orthogonal to each other
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                let sim = children[i].identity.similarity(&children[j].identity);
                assert!(sim < 0.6, "siblings should be near-orthogonal");
            }
        }
    }

    #[test]
    fn test_entity_ingest_and_contrast() {
        let mut entity = SpawnedEntity::root(10000, 42);
        let hv1 = BinaryHV::random(10000, 1);
        let hv2 = BinaryHV::random(10000, 2);

        let n1 = entity.ingest(&hv1);
        assert!((n1 - 1.0).abs() < 0.01); // First record: max novelty

        let n2 = entity.ingest(&hv2);
        assert!(n2 > 0.0); // Should be novel

        assert_eq!(entity.record_count, 2);
        assert!(entity.centroid().is_some());
    }

    #[test]
    fn test_child_inherits_centroid() {
        let mut parent = SpawnedEntity::root(10000, 42);
        let hv = BinaryHV::random(10000, 1);
        parent.ingest(&hv);

        let child = parent.spawn(99);
        assert!(child.centroid().is_some());
    }
}
