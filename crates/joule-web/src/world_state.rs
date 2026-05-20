//! ECS world container — the root of all ECS state.
//!
//! Owns entity allocation, component storage, singleton resources, and frame
//! timing. Provides a unified API for spawning/despawning entities, inserting/
//! getting components, and managing global resources.

use std::any::{Any, TypeId};
use std::collections::HashMap;

// ── Entity ──

/// Lightweight entity handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub id: u64,
    pub generation: u32,
}

// ── Slot ──

#[derive(Debug)]
struct Slot {
    generation: u32,
    alive: bool,
}

// ── Component storage ──

struct ComponentStore {
    /// entity_id → boxed component.
    data: HashMap<u64, Box<dyn Any>>,
}

impl ComponentStore {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

// ── Resource ──

/// A singleton resource (globally unique per type).
struct ResourceEntry {
    value: Box<dyn Any>,
}

// ── FrameTiming ──

/// Frame counter and delta time tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameTiming {
    pub frame: u64,
    /// Delta time in seconds since last frame.
    pub delta_secs: f64,
    /// Total elapsed time in seconds.
    pub total_secs: f64,
}

impl FrameTiming {
    pub fn new() -> Self {
        Self {
            frame: 0,
            delta_secs: 0.0,
            total_secs: 0.0,
        }
    }

    /// Advance to the next frame with the given delta time.
    pub fn advance(&mut self, delta: f64) {
        self.frame += 1;
        self.delta_secs = delta;
        self.total_secs += delta;
    }
}

impl Default for FrameTiming {
    fn default() -> Self {
        Self::new()
    }
}

// ── World ──

/// The ECS world: owns entities, components, resources, and frame timing.
pub struct World {
    slots: Vec<Slot>,
    free_list: Vec<u32>,
    components: HashMap<TypeId, ComponentStore>,
    resources: HashMap<TypeId, ResourceEntry>,
    pub timing: FrameTiming,
    /// Next entity ID (monotonic for external reference).
    next_id: u64,
    /// Maps entity id → slot index.
    id_to_slot: HashMap<u64, u32>,
    live_count: usize,
}

impl World {
    /// Create a new empty world.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            components: HashMap::new(),
            resources: HashMap::new(),
            timing: FrameTiming::new(),
            next_id: 0,
            id_to_slot: HashMap::new(),
            live_count: 0,
        }
    }

    // ── Entity management ──

    /// Spawn a new entity with no components.
    pub fn spawn(&mut self) -> Entity {
        let id = self.next_id;
        self.next_id += 1;
        let (slot_idx, generation) = if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.alive = true;
            (idx, slot.generation)
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                alive: true,
            });
            (idx, 0)
        };
        self.id_to_slot.insert(id, slot_idx);
        self.live_count += 1;
        Entity {
            id,
            generation,
        }
    }

    /// Despawn an entity, removing all its components.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let slot_idx = match self.id_to_slot.remove(&entity.id) {
            Some(idx) => idx,
            None => return false,
        };
        let slot = &mut self.slots[slot_idx as usize];
        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free_list.push(slot_idx);
        self.live_count -= 1;

        // Remove all components for this entity.
        for store in self.components.values_mut() {
            store.data.remove(&entity.id);
        }
        true
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        if let Some(&slot_idx) = self.id_to_slot.get(&entity.id) {
            let slot = &self.slots[slot_idx as usize];
            slot.alive && slot.generation == entity.generation
        } else {
            false
        }
    }

    /// Number of alive entities.
    pub fn entity_count(&self) -> usize {
        self.live_count
    }

    /// Collect all alive entity IDs.
    pub fn alive_entities(&self) -> Vec<Entity> {
        self.id_to_slot
            .iter()
            .filter_map(|(id, slot_idx)| {
                let slot = &self.slots[*slot_idx as usize];
                if slot.alive {
                    Some(Entity {
                        id: *id,
                        generation: slot.generation,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    // ── Component management ──

    /// Insert a component on an entity. Returns previous value if any.
    pub fn insert_component<C: 'static>(&mut self, entity: Entity, component: C) -> Option<C> {
        if !self.is_alive(entity) {
            return None;
        }
        let tid = TypeId::of::<C>();
        let store = self
            .components
            .entry(tid)
            .or_insert_with(ComponentStore::new);
        let prev = store
            .data
            .remove(&entity.id)
            .and_then(|b| b.downcast::<C>().ok().map(|b| *b));
        store.data.insert(entity.id, Box::new(component));
        prev
    }

    /// Get a shared reference to a component.
    pub fn get_component<C: 'static>(&self, entity: Entity) -> Option<&C> {
        if !self.is_alive(entity) {
            return None;
        }
        let tid = TypeId::of::<C>();
        self.components
            .get(&tid)?
            .data
            .get(&entity.id)?
            .downcast_ref::<C>()
    }

    /// Get a mutable reference to a component.
    pub fn get_component_mut<C: 'static>(&mut self, entity: Entity) -> Option<&mut C> {
        if !self.is_alive(entity) {
            return None;
        }
        let tid = TypeId::of::<C>();
        self.components
            .get_mut(&tid)?
            .data
            .get_mut(&entity.id)?
            .downcast_mut::<C>()
    }

    /// Remove a component from an entity.
    pub fn remove_component<C: 'static>(&mut self, entity: Entity) -> Option<C> {
        if !self.is_alive(entity) {
            return None;
        }
        let tid = TypeId::of::<C>();
        self.components
            .get_mut(&tid)?
            .data
            .remove(&entity.id)?
            .downcast::<C>()
            .ok()
            .map(|b| *b)
    }

    /// Check if an entity has a component.
    pub fn has_component<C: 'static>(&self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let tid = TypeId::of::<C>();
        self.components
            .get(&tid)
            .map(|s| s.data.contains_key(&entity.id))
            .unwrap_or(false)
    }

    /// Get all entity IDs with a given component type.
    pub fn entities_with_component<C: 'static>(&self) -> Vec<u64> {
        let tid = TypeId::of::<C>();
        self.components
            .get(&tid)
            .map(|s| s.data.keys().copied().collect())
            .unwrap_or_default()
    }

    // ── Resource management ──

    /// Add or replace a singleton resource.
    pub fn insert_resource<R: 'static>(&mut self, resource: R) {
        let tid = TypeId::of::<R>();
        self.resources.insert(tid, ResourceEntry {
            value: Box::new(resource),
        });
    }

    /// Get a shared reference to a resource.
    pub fn get_resource<R: 'static>(&self) -> Option<&R> {
        let tid = TypeId::of::<R>();
        self.resources
            .get(&tid)?
            .value
            .downcast_ref::<R>()
    }

    /// Get a mutable reference to a resource.
    pub fn get_resource_mut<R: 'static>(&mut self) -> Option<&mut R> {
        let tid = TypeId::of::<R>();
        self.resources
            .get_mut(&tid)?
            .value
            .downcast_mut::<R>()
    }

    /// Remove a resource, returning it.
    pub fn remove_resource<R: 'static>(&mut self) -> Option<R> {
        let tid = TypeId::of::<R>();
        self.resources
            .remove(&tid)?
            .value
            .downcast::<R>()
            .ok()
            .map(|b| *b)
    }

    /// Check if a resource exists.
    pub fn has_resource<R: 'static>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<R>())
    }

    // ── Frame timing ──

    /// Advance the frame counter and accumulate delta time.
    pub fn advance_frame(&mut self, delta_secs: f64) {
        self.timing.advance(delta_secs);
    }

    /// Current frame number.
    pub fn frame(&self) -> u64 {
        self.timing.frame
    }

    /// Delta time of the current frame.
    pub fn delta_secs(&self) -> f64 {
        self.timing.delta_secs
    }

    /// Total elapsed time.
    pub fn total_secs(&self) -> f64 {
        self.timing.total_secs
    }

    // ── Bulk operations ──

    /// Despawn all entities.
    pub fn clear_entities(&mut self) {
        let entities: Vec<Entity> = self.alive_entities();
        for entity in entities {
            self.despawn(entity);
        }
    }

    /// Clear all resources.
    pub fn clear_resources(&mut self) {
        self.resources.clear();
    }

    /// Reset the entire world.
    pub fn reset(&mut self) {
        self.clear_entities();
        self.clear_resources();
        self.timing = FrameTiming::new();
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Pos {
        x: f64,
        y: f64,
    }

    #[derive(Debug, PartialEq)]
    struct Vel {
        dx: f64,
        dy: f64,
    }

    #[derive(Debug, PartialEq)]
    struct Hp(u32);

    #[derive(Debug, PartialEq)]
    struct GameConfig {
        gravity: f64,
    }

    #[test]
    fn spawn_entity() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.is_alive(e));
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn despawn_entity() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.despawn(e));
        assert!(!world.is_alive(e));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn double_despawn() {
        let mut world = World::new();
        let e = world.spawn();
        world.despawn(e);
        assert!(!world.despawn(e));
    }

    #[test]
    fn insert_and_get_component() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Pos { x: 1.0, y: 2.0 });
        let pos = world.get_component::<Pos>(e).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-9);
        assert!((pos.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn insert_replaces_previous() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Hp(100));
        let prev = world.insert_component(e, Hp(50));
        assert_eq!(prev, Some(Hp(100)));
        assert_eq!(world.get_component::<Hp>(e), Some(&Hp(50)));
    }

    #[test]
    fn get_component_mut() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Pos { x: 0.0, y: 0.0 });
        if let Some(pos) = world.get_component_mut::<Pos>(e) {
            pos.x = 42.0;
        }
        assert!((world.get_component::<Pos>(e).unwrap().x - 42.0).abs() < 1e-9);
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Hp(100));
        let removed = world.remove_component::<Hp>(e);
        assert_eq!(removed, Some(Hp(100)));
        assert!(!world.has_component::<Hp>(e));
    }

    #[test]
    fn has_component() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(!world.has_component::<Pos>(e));
        world.insert_component(e, Pos { x: 0.0, y: 0.0 });
        assert!(world.has_component::<Pos>(e));
    }

    #[test]
    fn component_on_dead_entity() {
        let mut world = World::new();
        let e = world.spawn();
        world.despawn(e);
        assert!(world.insert_component(e, Hp(100)).is_none());
        assert!(world.get_component::<Hp>(e).is_none());
    }

    #[test]
    fn despawn_removes_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Pos { x: 1.0, y: 2.0 });
        world.insert_component(e, Hp(100));
        world.despawn(e);
        // Components should be gone. Spawn a new entity to verify.
        let e2 = world.spawn();
        assert!(!world.has_component::<Pos>(e2));
    }

    #[test]
    fn insert_and_get_resource() {
        let mut world = World::new();
        world.insert_resource(GameConfig { gravity: 9.81 });
        let cfg = world.get_resource::<GameConfig>().unwrap();
        assert!((cfg.gravity - 9.81).abs() < 1e-9);
    }

    #[test]
    fn resource_mut() {
        let mut world = World::new();
        world.insert_resource(GameConfig { gravity: 9.81 });
        if let Some(cfg) = world.get_resource_mut::<GameConfig>() {
            cfg.gravity = 1.62;
        }
        assert!((world.get_resource::<GameConfig>().unwrap().gravity - 1.62).abs() < 1e-9);
    }

    #[test]
    fn remove_resource() {
        let mut world = World::new();
        world.insert_resource(GameConfig { gravity: 9.81 });
        let removed = world.remove_resource::<GameConfig>().unwrap();
        assert!((removed.gravity - 9.81).abs() < 1e-9);
        assert!(!world.has_resource::<GameConfig>());
    }

    #[test]
    fn has_resource() {
        let mut world = World::new();
        assert!(!world.has_resource::<GameConfig>());
        world.insert_resource(GameConfig { gravity: 9.81 });
        assert!(world.has_resource::<GameConfig>());
    }

    #[test]
    fn frame_timing() {
        let mut world = World::new();
        assert_eq!(world.frame(), 0);
        world.advance_frame(0.016);
        assert_eq!(world.frame(), 1);
        assert!((world.delta_secs() - 0.016).abs() < 1e-9);
        assert!((world.total_secs() - 0.016).abs() < 1e-9);
        world.advance_frame(0.033);
        assert_eq!(world.frame(), 2);
        assert!((world.delta_secs() - 0.033).abs() < 1e-9);
        assert!((world.total_secs() - 0.049).abs() < 1e-9);
    }

    #[test]
    fn alive_entities() {
        let mut world = World::new();
        let a = world.spawn();
        let _b = world.spawn();
        let c = world.spawn();
        world.despawn(a);
        let alive = world.alive_entities();
        assert_eq!(alive.len(), 2);
        let ids: Vec<u64> = alive.iter().map(|e| e.id).collect();
        assert!(!ids.contains(&a.id));
        assert!(ids.contains(&c.id));
    }

    #[test]
    fn entities_with_component() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let _c = world.spawn();
        world.insert_component(a, Hp(100));
        world.insert_component(b, Hp(200));
        let mut ids = world.entities_with_component::<Hp>();
        ids.sort();
        assert_eq!(ids, vec![a.id, b.id]);
    }

    #[test]
    fn clear_entities() {
        let mut world = World::new();
        world.spawn();
        world.spawn();
        world.clear_entities();
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn clear_resources() {
        let mut world = World::new();
        world.insert_resource(GameConfig { gravity: 9.81 });
        world.clear_resources();
        assert!(!world.has_resource::<GameConfig>());
    }

    #[test]
    fn reset_world() {
        let mut world = World::new();
        world.spawn();
        world.insert_resource(GameConfig { gravity: 9.81 });
        world.advance_frame(0.016);
        world.reset();
        assert_eq!(world.entity_count(), 0);
        assert!(!world.has_resource::<GameConfig>());
        assert_eq!(world.frame(), 0);
    }

    #[test]
    fn multiple_component_types() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_component(e, Pos { x: 1.0, y: 2.0 });
        world.insert_component(e, Vel { dx: 3.0, dy: 4.0 });
        world.insert_component(e, Hp(100));
        assert!(world.has_component::<Pos>(e));
        assert!(world.has_component::<Vel>(e));
        assert!(world.has_component::<Hp>(e));
    }

    #[test]
    fn frame_timing_standalone() {
        let mut ft = FrameTiming::new();
        ft.advance(0.016);
        assert_eq!(ft.frame, 1);
        ft.advance(0.032);
        assert_eq!(ft.frame, 2);
        assert!((ft.total_secs - 0.048).abs() < 1e-9);
    }

    #[test]
    fn get_missing_resource() {
        let world = World::new();
        assert!(world.get_resource::<GameConfig>().is_none());
    }

    #[test]
    fn many_entities() {
        let mut world = World::new();
        let mut entities = Vec::new();
        for _ in 0..100 {
            entities.push(world.spawn());
        }
        assert_eq!(world.entity_count(), 100);
        for e in &entities[..50] {
            world.despawn(*e);
        }
        assert_eq!(world.entity_count(), 50);
    }
}
