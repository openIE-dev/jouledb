//! Entity storage with generational indices for a game-engine ECS.
//!
//! Entities are `u64` IDs paired with generation counters so that stale
//! references (dangling entity handles from destroyed entities) are detected
//! cheaply at runtime. Destroyed slots are recycled via a free-list.
//!
//! Two storage strategies are provided:
//! - **Dense** — contiguous `Vec` of live entities, good for iteration.
//! - **Sparse** — slot-map indexed by raw entity index, good for random access.

use std::collections::HashMap;

// ── Entity handle ──

/// An opaque entity handle combining a slot index and a generation counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    /// Slot index into the allocator's entry array.
    pub index: u32,
    /// Generation at creation time — must match the allocator's current
    /// generation for the same slot to be considered alive.
    pub generation: u32,
}

impl Entity {
    /// Pack index + generation into a single `u64` for storage/serialization.
    pub fn to_u64(self) -> u64 {
        ((self.generation as u64) << 32) | (self.index as u64)
    }

    /// Unpack a `u64` back into an `Entity`.
    pub fn from_u64(bits: u64) -> Self {
        Self {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

// ── Slot entry ──

#[derive(Debug, Clone)]
struct AllocEntry {
    generation: u32,
    alive: bool,
}

// ── EntityAllocator ──

/// Generational entity allocator with free-list recycling.
#[derive(Debug)]
pub struct EntityAllocator {
    entries: Vec<AllocEntry>,
    free_list: Vec<u32>,
    live_count: usize,
}

impl EntityAllocator {
    /// Create a new empty allocator.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_list: Vec::new(),
            live_count: 0,
        }
    }

    /// Create a new allocator with a pre-allocated capacity hint.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
            free_list: Vec::new(),
            live_count: 0,
        }
    }

    /// Allocate a fresh entity, recycling a destroyed slot if available.
    pub fn create(&mut self) -> Entity {
        self.live_count += 1;
        if let Some(index) = self.free_list.pop() {
            let entry = &mut self.entries[index as usize];
            entry.alive = true;
            Entity {
                index,
                generation: entry.generation,
            }
        } else {
            let index = self.entries.len() as u32;
            self.entries.push(AllocEntry {
                generation: 0,
                alive: true,
            });
            Entity {
                index,
                generation: 0,
            }
        }
    }

    /// Destroy an entity. Returns `true` if it was alive.
    pub fn destroy(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let entry = &mut self.entries[entity.index as usize];
        entry.alive = false;
        entry.generation = entry.generation.wrapping_add(1);
        self.free_list.push(entity.index);
        self.live_count -= 1;
        true
    }

    /// Check whether an entity handle is still valid.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index as usize;
        if idx >= self.entries.len() {
            return false;
        }
        let entry = &self.entries[idx];
        entry.alive && entry.generation == entity.generation
    }

    /// Number of currently alive entities.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Total slots ever allocated (including destroyed ones).
    pub fn total_slots(&self) -> usize {
        self.entries.len()
    }

    /// Number of recycled slots waiting to be reused.
    pub fn free_slots(&self) -> usize {
        self.free_list.len()
    }

    /// Iterator over all currently alive entities.
    pub fn alive_entities(&self) -> Vec<Entity> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.alive)
            .map(|(i, e)| Entity {
                index: i as u32,
                generation: e.generation,
            })
            .collect()
    }

    /// Reset the allocator, destroying everything.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            if entry.alive {
                entry.alive = false;
                entry.generation = entry.generation.wrapping_add(1);
            }
        }
        self.free_list.clear();
        // Push all slots onto free list in reverse order so index 0 is reused first.
        for i in (0..self.entries.len()).rev() {
            self.free_list.push(i as u32);
        }
        self.live_count = 0;
    }
}

impl Default for EntityAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Dense storage ──

/// Dense entity storage — live entities stored contiguously for fast iteration.
/// Maintains a mapping from entity index → dense index and back.
#[derive(Debug)]
pub struct DenseEntityStore {
    allocator: EntityAllocator,
    /// Sparse → dense mapping. Indexed by entity slot index.
    sparse: Vec<Option<usize>>,
    /// Dense → Entity mapping.
    dense: Vec<Entity>,
}

impl DenseEntityStore {
    pub fn new() -> Self {
        Self {
            allocator: EntityAllocator::new(),
            sparse: Vec::new(),
            dense: Vec::new(),
        }
    }

    pub fn create(&mut self) -> Entity {
        let entity = self.allocator.create();
        let idx = entity.index as usize;
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, None);
        }
        let dense_idx = self.dense.len();
        self.sparse[idx] = Some(dense_idx);
        self.dense.push(entity);
        entity
    }

    pub fn destroy(&mut self, entity: Entity) -> bool {
        if !self.allocator.is_alive(entity) {
            return false;
        }
        let idx = entity.index as usize;
        if let Some(dense_idx) = self.sparse[idx] {
            // Swap-remove from dense array.
            let last = self.dense.len() - 1;
            if dense_idx != last {
                let moved = self.dense[last];
                self.dense[dense_idx] = moved;
                self.sparse[moved.index as usize] = Some(dense_idx);
            }
            self.dense.pop();
            self.sparse[idx] = None;
        }
        self.allocator.destroy(entity);
        true
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.allocator.is_alive(entity)
    }

    /// Iterate over all live entities contiguously.
    pub fn iter(&self) -> &[Entity] {
        &self.dense
    }

    pub fn live_count(&self) -> usize {
        self.dense.len()
    }
}

impl Default for DenseEntityStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sparse storage ──

/// Sparse entity storage using a HashMap for O(1) lookup by entity ID.
#[derive(Debug)]
pub struct SparseEntityStore {
    allocator: EntityAllocator,
    /// Maps entity packed u64 → arbitrary payload tag (e.g., archetype id).
    tags: HashMap<u64, u64>,
}

impl SparseEntityStore {
    pub fn new() -> Self {
        Self {
            allocator: EntityAllocator::new(),
            tags: HashMap::new(),
        }
    }

    pub fn create(&mut self) -> Entity {
        let entity = self.allocator.create();
        self.tags.insert(entity.to_u64(), 0);
        entity
    }

    pub fn create_with_tag(&mut self, tag: u64) -> Entity {
        let entity = self.allocator.create();
        self.tags.insert(entity.to_u64(), tag);
        entity
    }

    pub fn destroy(&mut self, entity: Entity) -> bool {
        if !self.allocator.is_alive(entity) {
            return false;
        }
        self.tags.remove(&entity.to_u64());
        self.allocator.destroy(entity);
        true
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.allocator.is_alive(entity)
    }

    pub fn get_tag(&self, entity: Entity) -> Option<u64> {
        if !self.is_alive(entity) {
            return None;
        }
        self.tags.get(&entity.to_u64()).copied()
    }

    pub fn set_tag(&mut self, entity: Entity, tag: u64) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.tags.insert(entity.to_u64(), tag);
        true
    }

    pub fn live_count(&self) -> usize {
        self.allocator.live_count()
    }

    /// Collect all alive entities (order not guaranteed).
    pub fn alive_entities(&self) -> Vec<Entity> {
        self.allocator.alive_entities()
    }
}

impl Default for SparseEntityStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // -- EntityAllocator --

    #[test]
    fn alloc_create_single() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.create();
        assert_eq!(e.index, 0);
        assert_eq!(e.generation, 0);
        assert!(alloc.is_alive(e));
        assert_eq!(alloc.live_count(), 1);
    }

    #[test]
    fn alloc_create_multiple() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.create();
        let b = alloc.create();
        let c = alloc.create();
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(c.index, 2);
        assert_eq!(alloc.live_count(), 3);
        assert_eq!(alloc.total_slots(), 3);
    }

    #[test]
    fn alloc_destroy_and_detect_stale() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.create();
        assert!(alloc.destroy(e));
        assert!(!alloc.is_alive(e));
        assert_eq!(alloc.live_count(), 0);
    }

    #[test]
    fn alloc_double_destroy() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.create();
        assert!(alloc.destroy(e));
        assert!(!alloc.destroy(e));
    }

    #[test]
    fn alloc_recycle_slot() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.create();
        alloc.destroy(a);
        let b = alloc.create();
        // Reuses slot 0, but with incremented generation.
        assert_eq!(b.index, 0);
        assert_eq!(b.generation, 1);
        assert!(!alloc.is_alive(a));
        assert!(alloc.is_alive(b));
    }

    #[test]
    fn alloc_stale_handle_after_recycle() {
        let mut alloc = EntityAllocator::new();
        let old = alloc.create();
        alloc.destroy(old);
        let _new = alloc.create();
        // Old handle has generation 0, slot now has generation 1.
        assert!(!alloc.is_alive(old));
    }

    #[test]
    fn alloc_with_capacity() {
        let alloc = EntityAllocator::with_capacity(100);
        assert_eq!(alloc.live_count(), 0);
        assert_eq!(alloc.total_slots(), 0);
    }

    #[test]
    fn alloc_alive_entities() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.create();
        let b = alloc.create();
        let c = alloc.create();
        alloc.destroy(b);
        let alive = alloc.alive_entities();
        assert_eq!(alive.len(), 2);
        assert!(alive.contains(&a));
        assert!(alive.contains(&c));
    }

    #[test]
    fn alloc_clear() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.create();
        let b = alloc.create();
        alloc.clear();
        assert!(!alloc.is_alive(a));
        assert!(!alloc.is_alive(b));
        assert_eq!(alloc.live_count(), 0);
        assert_eq!(alloc.free_slots(), 2);
    }

    #[test]
    fn alloc_free_slots_count() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.create();
        let b = alloc.create();
        alloc.destroy(a);
        assert_eq!(alloc.free_slots(), 1);
        alloc.destroy(b);
        assert_eq!(alloc.free_slots(), 2);
    }

    #[test]
    fn entity_u64_roundtrip() {
        let e = Entity {
            index: 42,
            generation: 7,
        };
        let packed = e.to_u64();
        let unpacked = Entity::from_u64(packed);
        assert_eq!(unpacked, e);
    }

    #[test]
    fn entity_u64_max_values() {
        let e = Entity {
            index: u32::MAX,
            generation: u32::MAX,
        };
        let unpacked = Entity::from_u64(e.to_u64());
        assert_eq!(unpacked, e);
    }

    #[test]
    fn alloc_out_of_bounds_entity() {
        let alloc = EntityAllocator::new();
        let fake = Entity {
            index: 999,
            generation: 0,
        };
        assert!(!alloc.is_alive(fake));
    }

    #[test]
    fn alloc_generation_wraps() {
        let mut alloc = EntityAllocator::new();
        // Force many create/destroy cycles on the same slot.
        for _ in 0..300 {
            let e = alloc.create();
            alloc.destroy(e);
        }
        let e = alloc.create();
        assert!(alloc.is_alive(e));
        assert_eq!(e.generation, 300);
    }

    // -- DenseEntityStore --

    #[test]
    fn dense_create_and_iterate() {
        let mut store = DenseEntityStore::new();
        let a = store.create();
        let b = store.create();
        let c = store.create();
        let slice = store.iter();
        assert_eq!(slice.len(), 3);
        assert!(slice.contains(&a));
        assert!(slice.contains(&b));
        assert!(slice.contains(&c));
    }

    #[test]
    fn dense_destroy_swap_remove() {
        let mut store = DenseEntityStore::new();
        let a = store.create();
        let _b = store.create();
        let c = store.create();
        store.destroy(a);
        assert_eq!(store.live_count(), 2);
        // After swap-remove, only b and c remain.
        let slice = store.iter();
        assert_eq!(slice.len(), 2);
        assert!(slice.contains(&c));
        assert!(!slice.contains(&a));
    }

    #[test]
    fn dense_destroy_last() {
        let mut store = DenseEntityStore::new();
        let _a = store.create();
        let b = store.create();
        store.destroy(b);
        assert_eq!(store.live_count(), 1);
    }

    #[test]
    fn dense_stale_handle() {
        let mut store = DenseEntityStore::new();
        let e = store.create();
        store.destroy(e);
        assert!(!store.is_alive(e));
    }

    // -- SparseEntityStore --

    #[test]
    fn sparse_create_and_tag() {
        let mut store = SparseEntityStore::new();
        let e = store.create_with_tag(42);
        assert_eq!(store.get_tag(e), Some(42));
    }

    #[test]
    fn sparse_default_tag_zero() {
        let mut store = SparseEntityStore::new();
        let e = store.create();
        assert_eq!(store.get_tag(e), Some(0));
    }

    #[test]
    fn sparse_set_tag() {
        let mut store = SparseEntityStore::new();
        let e = store.create();
        assert!(store.set_tag(e, 99));
        assert_eq!(store.get_tag(e), Some(99));
    }

    #[test]
    fn sparse_destroy_removes_tag() {
        let mut store = SparseEntityStore::new();
        let e = store.create();
        store.destroy(e);
        assert_eq!(store.get_tag(e), None);
    }

    #[test]
    fn sparse_set_tag_dead_entity() {
        let mut store = SparseEntityStore::new();
        let e = store.create();
        store.destroy(e);
        assert!(!store.set_tag(e, 10));
    }

    #[test]
    fn sparse_alive_entities() {
        let mut store = SparseEntityStore::new();
        let a = store.create();
        let b = store.create();
        let c = store.create();
        store.destroy(b);
        let alive = store.alive_entities();
        assert_eq!(alive.len(), 2);
        assert!(alive.contains(&a));
        assert!(alive.contains(&c));
        assert_eq!(store.live_count(), 2);
    }
}
