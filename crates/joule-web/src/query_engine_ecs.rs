//! ECS query system for matching entities by component combinations.
//!
//! Queries specify required, optional, and excluded component types.
//! Matching is done efficiently against archetype bitmasks so that only
//! archetypes containing all required types (and none of the excluded types)
//! are visited. Supports change-tracking filters (Added, Changed since tick).

use std::any::TypeId;
use std::collections::{BTreeSet, HashMap, HashSet};

// ── QueryFilter ──

/// Filter for matching archetypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryFilter {
    /// Entity must have this component.
    With(TypeId),
    /// Entity must NOT have this component.
    Without(TypeId),
    /// Component is optional — included in results if present.
    Optional(TypeId),
    /// Only entities whose component was added after the given tick.
    AddedSince(TypeId, u64),
    /// Only entities whose component was changed after the given tick.
    ChangedSince(TypeId, u64),
}

// ── QueryDescriptor ──

/// Describes what a query is looking for.
#[derive(Debug, Clone)]
pub struct QueryDescriptor {
    pub required: BTreeSet<TypeId>,
    pub excluded: BTreeSet<TypeId>,
    pub optional: BTreeSet<TypeId>,
    pub change_filters: Vec<(TypeId, ChangeFilterKind, u64)>,
}

/// Kind of change filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeFilterKind {
    Added,
    Changed,
}

impl QueryDescriptor {
    pub fn new() -> Self {
        Self {
            required: BTreeSet::new(),
            excluded: BTreeSet::new(),
            optional: BTreeSet::new(),
            change_filters: Vec::new(),
        }
    }

    pub fn with(mut self, tid: TypeId) -> Self {
        self.required.insert(tid);
        self
    }

    pub fn without(mut self, tid: TypeId) -> Self {
        self.excluded.insert(tid);
        self
    }

    pub fn optional(mut self, tid: TypeId) -> Self {
        self.optional.insert(tid);
        self
    }

    pub fn added_since(mut self, tid: TypeId, tick: u64) -> Self {
        self.change_filters.push((tid, ChangeFilterKind::Added, tick));
        self
    }

    pub fn changed_since(mut self, tid: TypeId, tick: u64) -> Self {
        self.change_filters
            .push((tid, ChangeFilterKind::Changed, tick));
        self
    }

    /// Build filters from this descriptor.
    pub fn to_filters(&self) -> Vec<QueryFilter> {
        let mut filters = Vec::new();
        for tid in &self.required {
            filters.push(QueryFilter::With(*tid));
        }
        for tid in &self.excluded {
            filters.push(QueryFilter::Without(*tid));
        }
        for tid in &self.optional {
            filters.push(QueryFilter::Optional(*tid));
        }
        for (tid, kind, tick) in &self.change_filters {
            match kind {
                ChangeFilterKind::Added => filters.push(QueryFilter::AddedSince(*tid, *tick)),
                ChangeFilterKind::Changed => filters.push(QueryFilter::ChangedSince(*tid, *tick)),
            }
        }
        filters
    }
}

impl Default for QueryDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

// ── ArchetypeMask ──

/// Bitmask representation of an archetype's component set for fast matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchetypeMask {
    /// Bit index assigned to each TypeId.
    bits: u128,
}

impl ArchetypeMask {
    pub fn empty() -> Self {
        Self { bits: 0 }
    }

    pub fn from_bits(bits: u128) -> Self {
        Self { bits }
    }

    pub fn set(&mut self, bit: u8) {
        self.bits |= 1u128 << bit;
    }

    pub fn has(&self, bit: u8) -> bool {
        (self.bits >> bit) & 1 == 1
    }

    pub fn contains_all(&self, other: &ArchetypeMask) -> bool {
        (self.bits & other.bits) == other.bits
    }

    pub fn contains_none(&self, other: &ArchetypeMask) -> bool {
        (self.bits & other.bits) == 0
    }

    pub fn bit_count(&self) -> u32 {
        self.bits.count_ones()
    }

    pub fn bits(&self) -> u128 {
        self.bits
    }
}

// ── MaskRegistry ──

/// Assigns bit indices to component TypeIds for bitmask-based queries.
pub struct MaskRegistry {
    type_to_bit: HashMap<TypeId, u8>,
    next_bit: u8,
}

impl MaskRegistry {
    pub fn new() -> Self {
        Self {
            type_to_bit: HashMap::new(),
            next_bit: 0,
        }
    }

    /// Get or assign a bit index for a type.
    pub fn bit_for(&mut self, tid: TypeId) -> u8 {
        if let Some(&bit) = self.type_to_bit.get(&tid) {
            return bit;
        }
        let bit = self.next_bit;
        assert!(bit < 128, "exceeds 128 component type limit");
        self.next_bit += 1;
        self.type_to_bit.insert(tid, bit);
        bit
    }

    /// Get the bit index for a type (if registered).
    pub fn get_bit(&self, tid: &TypeId) -> Option<u8> {
        self.type_to_bit.get(tid).copied()
    }

    /// Build a mask for a set of types.
    pub fn mask_for(&mut self, types: &BTreeSet<TypeId>) -> ArchetypeMask {
        let mut mask = ArchetypeMask::empty();
        for tid in types {
            mask.set(self.bit_for(*tid));
        }
        mask
    }

    /// Number of registered types.
    pub fn type_count(&self) -> usize {
        self.type_to_bit.len()
    }
}

impl Default for MaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── ChangeTracker ──

/// Tracks per-entity, per-component change ticks.
pub struct ChangeTracker {
    /// (entity_id, TypeId) → (added_tick, last_changed_tick).
    ticks: HashMap<(u64, TypeId), (u64, u64)>,
}

impl ChangeTracker {
    pub fn new() -> Self {
        Self {
            ticks: HashMap::new(),
        }
    }

    /// Record that a component was added at the given tick.
    pub fn record_added(&mut self, entity_id: u64, tid: TypeId, tick: u64) {
        self.ticks.insert((entity_id, tid), (tick, tick));
    }

    /// Record that a component was changed at the given tick.
    pub fn record_changed(&mut self, entity_id: u64, tid: TypeId, tick: u64) {
        let entry = self.ticks.entry((entity_id, tid)).or_insert((tick, tick));
        entry.1 = tick;
    }

    /// Remove tracking for an entity+component.
    pub fn remove(&mut self, entity_id: u64, tid: TypeId) {
        self.ticks.remove(&(entity_id, tid));
    }

    /// Remove all tracking for an entity.
    pub fn remove_entity(&mut self, entity_id: u64) {
        self.ticks.retain(|(eid, _), _| *eid != entity_id);
    }

    /// Was the component added after the given tick?
    pub fn was_added_since(&self, entity_id: u64, tid: TypeId, tick: u64) -> bool {
        self.ticks
            .get(&(entity_id, tid))
            .map(|(added, _)| *added > tick)
            .unwrap_or(false)
    }

    /// Was the component changed after the given tick?
    pub fn was_changed_since(&self, entity_id: u64, tid: TypeId, tick: u64) -> bool {
        self.ticks
            .get(&(entity_id, tid))
            .map(|(_, changed)| *changed > tick)
            .unwrap_or(false)
    }

    /// Get the added tick.
    pub fn added_tick(&self, entity_id: u64, tid: TypeId) -> Option<u64> {
        self.ticks.get(&(entity_id, tid)).map(|(a, _)| *a)
    }

    /// Get the last-changed tick.
    pub fn changed_tick(&self, entity_id: u64, tid: TypeId) -> Option<u64> {
        self.ticks.get(&(entity_id, tid)).map(|(_, c)| *c)
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── QueryEngine ──

/// ECS query engine that matches entities against archetype masks and
/// change-tracking filters.
pub struct QueryEngine {
    pub mask_registry: MaskRegistry,
    pub change_tracker: ChangeTracker,
    /// archetype_id → (mask, entity_ids).
    archetypes: HashMap<u64, (ArchetypeMask, Vec<u64>)>,
    next_archetype_id: u64,
}

impl QueryEngine {
    pub fn new() -> Self {
        Self {
            mask_registry: MaskRegistry::new(),
            change_tracker: ChangeTracker::new(),
            archetypes: HashMap::new(),
            next_archetype_id: 0,
        }
    }

    /// Register an archetype with a given set of component types and entities.
    pub fn register_archetype(
        &mut self,
        types: &BTreeSet<TypeId>,
        entities: Vec<u64>,
    ) -> u64 {
        let mask = self.mask_registry.mask_for(types);
        let id = self.next_archetype_id;
        self.next_archetype_id += 1;
        self.archetypes.insert(id, (mask, entities));
        id
    }

    /// Update entities in an archetype.
    pub fn update_archetype(&mut self, arch_id: u64, entities: Vec<u64>) {
        if let Some((_, ents)) = self.archetypes.get_mut(&arch_id) {
            *ents = entities;
        }
    }

    /// Remove an archetype.
    pub fn remove_archetype(&mut self, arch_id: u64) -> bool {
        self.archetypes.remove(&arch_id).is_some()
    }

    /// Execute a query, returning matching entity IDs.
    pub fn query(&self, desc: &QueryDescriptor) -> Vec<u64> {
        // Build required and excluded masks.
        let mut req_mask = ArchetypeMask::empty();
        for tid in &desc.required {
            if let Some(bit) = self.mask_registry.get_bit(tid) {
                req_mask.set(bit);
            } else {
                // Required type not registered => no matches.
                return Vec::new();
            }
        }

        let mut excl_mask = ArchetypeMask::empty();
        for tid in &desc.excluded {
            if let Some(bit) = self.mask_registry.get_bit(tid) {
                excl_mask.set(bit);
            }
            // If excluded type not registered, no archetype can have it, so skip.
        }

        let mut result = Vec::new();
        for (_, (mask, entities)) in &self.archetypes {
            if mask.contains_all(&req_mask) && mask.contains_none(&excl_mask) {
                // Apply change filters per-entity.
                if desc.change_filters.is_empty() {
                    result.extend(entities.iter().copied());
                } else {
                    for &eid in entities {
                        let passes = desc.change_filters.iter().all(|(tid, kind, tick)| {
                            match kind {
                                ChangeFilterKind::Added => {
                                    self.change_tracker.was_added_since(eid, *tid, *tick)
                                }
                                ChangeFilterKind::Changed => {
                                    self.change_tracker.was_changed_since(eid, *tid, *tick)
                                }
                            }
                        });
                        if passes {
                            result.push(eid);
                        }
                    }
                }
            }
        }
        result
    }

    /// Convenience: query with only required types.
    pub fn query_with(&self, types: &[TypeId]) -> Vec<u64> {
        let desc = types
            .iter()
            .fold(QueryDescriptor::new(), |d, tid| d.with(*tid));
        self.query(&desc)
    }

    /// Convenience: query with required + excluded.
    pub fn query_filtered(&self, required: &[TypeId], excluded: &[TypeId]) -> Vec<u64> {
        let mut desc = QueryDescriptor::new();
        for tid in required {
            desc = desc.with(*tid);
        }
        for tid in excluded {
            desc = desc.without(*tid);
        }
        self.query(&desc)
    }

    /// Number of registered archetypes.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    struct Pos;
    struct Vel;
    struct Hp;
    struct Shield;

    #[test]
    fn mask_set_and_has() {
        let mut mask = ArchetypeMask::empty();
        mask.set(0);
        mask.set(5);
        assert!(mask.has(0));
        assert!(mask.has(5));
        assert!(!mask.has(1));
    }

    #[test]
    fn mask_contains_all() {
        let mut full = ArchetypeMask::empty();
        full.set(0);
        full.set(1);
        full.set(2);
        let mut sub = ArchetypeMask::empty();
        sub.set(0);
        sub.set(2);
        assert!(full.contains_all(&sub));
        sub.set(5);
        assert!(!full.contains_all(&sub));
    }

    #[test]
    fn mask_contains_none() {
        let mut a = ArchetypeMask::empty();
        a.set(0);
        a.set(1);
        let mut b = ArchetypeMask::empty();
        b.set(2);
        b.set(3);
        assert!(a.contains_none(&b));
        b.set(1);
        assert!(!a.contains_none(&b));
    }

    #[test]
    fn mask_bit_count() {
        let mut m = ArchetypeMask::empty();
        m.set(0);
        m.set(7);
        m.set(127);
        assert_eq!(m.bit_count(), 3);
    }

    #[test]
    fn mask_registry_assigns_bits() {
        let mut reg = MaskRegistry::new();
        let b1 = reg.bit_for(TypeId::of::<Pos>());
        let b2 = reg.bit_for(TypeId::of::<Vel>());
        assert_ne!(b1, b2);
        // Same type gets same bit.
        assert_eq!(reg.bit_for(TypeId::of::<Pos>()), b1);
    }

    #[test]
    fn mask_registry_mask_for() {
        let mut reg = MaskRegistry::new();
        let types: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>()].into_iter().collect();
        let mask = reg.mask_for(&types);
        assert_eq!(mask.bit_count(), 2);
    }

    #[test]
    fn change_tracker_added() {
        let mut ct = ChangeTracker::new();
        ct.record_added(1, TypeId::of::<Pos>(), 5);
        assert!(ct.was_added_since(1, TypeId::of::<Pos>(), 4));
        assert!(!ct.was_added_since(1, TypeId::of::<Pos>(), 5));
        assert!(!ct.was_added_since(1, TypeId::of::<Pos>(), 6));
    }

    #[test]
    fn change_tracker_changed() {
        let mut ct = ChangeTracker::new();
        ct.record_added(1, TypeId::of::<Pos>(), 5);
        ct.record_changed(1, TypeId::of::<Pos>(), 10);
        assert!(ct.was_changed_since(1, TypeId::of::<Pos>(), 7));
        assert!(!ct.was_changed_since(1, TypeId::of::<Pos>(), 10));
    }

    #[test]
    fn change_tracker_remove() {
        let mut ct = ChangeTracker::new();
        ct.record_added(1, TypeId::of::<Pos>(), 5);
        ct.remove(1, TypeId::of::<Pos>());
        assert!(!ct.was_added_since(1, TypeId::of::<Pos>(), 0));
    }

    #[test]
    fn change_tracker_remove_entity() {
        let mut ct = ChangeTracker::new();
        ct.record_added(1, TypeId::of::<Pos>(), 5);
        ct.record_added(1, TypeId::of::<Vel>(), 5);
        ct.remove_entity(1);
        assert!(ct.added_tick(1, TypeId::of::<Pos>()).is_none());
        assert!(ct.added_tick(1, TypeId::of::<Vel>()).is_none());
    }

    #[test]
    fn query_with_required() {
        let mut engine = QueryEngine::new();
        let types_pv: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>()].into_iter().collect();
        let types_p: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        engine.register_archetype(&types_pv, vec![1, 2]);
        engine.register_archetype(&types_p, vec![3]);
        // Query for Pos+Vel.
        let mut result = engine.query_with(&[TypeId::of::<Pos>(), TypeId::of::<Vel>()]);
        result.sort();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn query_with_exclusion() {
        let mut engine = QueryEngine::new();
        let types_ph: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Hp>()].into_iter().collect();
        let types_p: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        engine.register_archetype(&types_ph, vec![1]);
        engine.register_archetype(&types_p, vec![2]);
        let result = engine.query_filtered(
            &[TypeId::of::<Pos>()],
            &[TypeId::of::<Hp>()],
        );
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn query_with_change_filter() {
        let mut engine = QueryEngine::new();
        let types: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        engine.register_archetype(&types, vec![1, 2, 3]);
        // Only entity 2 was added recently.
        engine.change_tracker.record_added(1, TypeId::of::<Pos>(), 1);
        engine.change_tracker.record_added(2, TypeId::of::<Pos>(), 5);
        engine.change_tracker.record_added(3, TypeId::of::<Pos>(), 2);
        let desc = QueryDescriptor::new()
            .with(TypeId::of::<Pos>())
            .added_since(TypeId::of::<Pos>(), 3);
        let result = engine.query(&desc);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn query_empty_required_type() {
        let engine = QueryEngine::new();
        // TypeId not registered at all.
        let result = engine.query_with(&[TypeId::of::<Shield>()]);
        assert!(result.is_empty());
    }

    #[test]
    fn query_descriptor_to_filters() {
        let desc = QueryDescriptor::new()
            .with(TypeId::of::<Pos>())
            .without(TypeId::of::<Vel>())
            .optional(TypeId::of::<Hp>());
        let filters = desc.to_filters();
        assert_eq!(filters.len(), 3);
    }

    #[test]
    fn update_archetype() {
        let mut engine = QueryEngine::new();
        let types: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        let id = engine.register_archetype(&types, vec![1]);
        engine.update_archetype(id, vec![1, 2, 3]);
        let result = engine.query_with(&[TypeId::of::<Pos>()]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn remove_archetype() {
        let mut engine = QueryEngine::new();
        let types: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        let id = engine.register_archetype(&types, vec![1]);
        assert!(engine.remove_archetype(id));
        assert!(!engine.remove_archetype(id));
        assert_eq!(engine.archetype_count(), 0);
    }

    #[test]
    fn query_multiple_archetypes_match() {
        let mut engine = QueryEngine::new();
        let types1: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>()].into_iter().collect();
        let types2: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>(), TypeId::of::<Hp>()]
                .into_iter()
                .collect();
        engine.register_archetype(&types1, vec![1, 2]);
        engine.register_archetype(&types2, vec![3]);
        // Querying Pos+Vel should match both archetypes.
        let mut result = engine.query_with(&[TypeId::of::<Pos>(), TypeId::of::<Vel>()]);
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn change_tracker_ticks() {
        let mut ct = ChangeTracker::new();
        ct.record_added(1, TypeId::of::<Pos>(), 10);
        assert_eq!(ct.added_tick(1, TypeId::of::<Pos>()), Some(10));
        assert_eq!(ct.changed_tick(1, TypeId::of::<Pos>()), Some(10));
        ct.record_changed(1, TypeId::of::<Pos>(), 20);
        assert_eq!(ct.added_tick(1, TypeId::of::<Pos>()), Some(10));
        assert_eq!(ct.changed_tick(1, TypeId::of::<Pos>()), Some(20));
    }

    #[test]
    fn change_filter_combined() {
        let mut engine = QueryEngine::new();
        let types: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>()].into_iter().collect();
        engine.register_archetype(&types, vec![1, 2]);
        engine.change_tracker.record_added(1, TypeId::of::<Pos>(), 5);
        engine.change_tracker.record_changed(1, TypeId::of::<Vel>(), 8);
        engine.change_tracker.record_added(2, TypeId::of::<Pos>(), 5);
        engine.change_tracker.record_changed(2, TypeId::of::<Vel>(), 3);
        let desc = QueryDescriptor::new()
            .with(TypeId::of::<Pos>())
            .changed_since(TypeId::of::<Vel>(), 5);
        let result = engine.query(&desc);
        assert_eq!(result, vec![1]);
    }
}
