//! Entity Component System — sparse-set storage, system registration, queries,
//! entity builder, archetype tracking, and world tick.
//!
//! Replaces JavaScript ECS libraries (bitecs, ECSY, Geotic) with a pure-Rust
//! implementation suitable for browser-based games via WASM.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

/// ECS domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcsError {
    /// Entity does not exist.
    EntityNotFound(u64),
    /// Component not found on entity.
    ComponentNotFound { entity: u64, component: &'static str },
    /// Entity already destroyed.
    EntityDestroyed(u64),
    /// System already registered.
    SystemAlreadyRegistered(String),
}

impl std::fmt::Display for EcsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntityNotFound(id) => write!(f, "entity {id} not found"),
            Self::ComponentNotFound { entity, component } => {
                write!(f, "component {component} not found on entity {entity}")
            }
            Self::EntityDestroyed(id) => write!(f, "entity {id} already destroyed"),
            Self::SystemAlreadyRegistered(name) => {
                write!(f, "system already registered: {name}")
            }
        }
    }
}

impl std::error::Error for EcsError {}

// ── Component trait ─────────────────────────────────────────────

/// Marker trait for components stored in the ECS.
pub trait Component: Any + 'static {
    /// Human-readable name for error messages.
    fn component_name() -> &'static str;
}

// ── Sparse set storage ──────────────────────────────────────────

/// Sparse-set storage for a single component type.
struct SparseSet {
    /// entity id → index into `dense`
    sparse: HashMap<u64, usize>,
    /// packed entity ids
    dense_entities: Vec<u64>,
    /// packed component data (boxed Any)
    dense_data: Vec<Box<dyn Any>>,
}

impl SparseSet {
    fn new() -> Self {
        Self {
            sparse: HashMap::new(),
            dense_entities: Vec::new(),
            dense_data: Vec::new(),
        }
    }

    fn insert(&mut self, entity: u64, data: Box<dyn Any>) {
        if let Some(&idx) = self.sparse.get(&entity) {
            self.dense_data[idx] = data;
        } else {
            let idx = self.dense_entities.len();
            self.dense_entities.push(entity);
            self.dense_data.push(data);
            self.sparse.insert(entity, idx);
        }
    }

    fn remove(&mut self, entity: u64) -> Option<Box<dyn Any>> {
        let idx = self.sparse.remove(&entity)?;
        let last = self.dense_entities.len() - 1;
        let removed = if idx == last {
            self.dense_entities.pop();
            self.dense_data.pop().unwrap()
        } else {
            let swapped_entity = self.dense_entities[last];
            self.dense_entities.swap(idx, last);
            self.dense_data.swap(idx, last);
            self.dense_entities.pop();
            let removed = self.dense_data.pop().unwrap();
            self.sparse.insert(swapped_entity, idx);
            removed
        };
        Some(removed)
    }

    fn get(&self, entity: u64) -> Option<&dyn Any> {
        let &idx = self.sparse.get(&entity)?;
        Some(self.dense_data[idx].as_ref())
    }

    fn get_mut(&mut self, entity: u64) -> Option<&mut dyn Any> {
        let &idx = self.sparse.get(&entity)?;
        Some(self.dense_data[idx].as_mut())
    }

    fn contains(&self, entity: u64) -> bool {
        self.sparse.contains_key(&entity)
    }

    fn entities(&self) -> &[u64] {
        &self.dense_entities
    }

    fn len(&self) -> usize {
        self.dense_entities.len()
    }
}

// ── Archetype ───────────────────────────────────────────────────

/// An archetype is a unique set of component types an entity has.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Archetype {
    /// Sorted list of TypeId for the components.
    pub component_types: Vec<TypeId>,
}

impl Archetype {
    fn new(mut types: Vec<TypeId>) -> Self {
        types.sort();
        types.dedup();
        Self { component_types: types }
    }
}

// ── System ──────────────────────────────────────────────────────

/// Trait for systems that operate on the world each tick.
pub trait System {
    /// Human-readable name.
    fn name(&self) -> &str;
    /// Run the system. Receives mutable access to the world.
    fn run(&mut self, world: &mut World);
}

/// Boxed system wrapper for storage.
struct RegisteredSystem {
    name: String,
    system: Box<dyn System>,
    priority: i32,
}

// ── Entity Builder ──────────────────────────────────────────────

/// Fluent builder for creating entities with components.
pub struct EntityBuilder<'w> {
    world: &'w mut World,
    entity: u64,
}

impl<'w> EntityBuilder<'w> {
    /// Attach a component to the entity being built.
    pub fn with<C: Component>(self, component: C) -> Self {
        let _ = self.world.add_component(self.entity, component);
        self
    }

    /// Finish building and return the entity id.
    pub fn build(self) -> u64 {
        self.entity
    }
}

// ── World ───────────────────────────────────────────────────────

/// The ECS world: manages entities, component storage, systems, and ticks.
pub struct World {
    next_entity: u64,
    alive: HashSet<u64>,
    destroyed: HashSet<u64>,
    storages: HashMap<TypeId, SparseSet>,
    /// entity → set of component TypeIds
    entity_components: HashMap<u64, HashSet<TypeId>>,
    systems: Vec<RegisteredSystem>,
    tick_count: u64,
}

impl World {
    /// Create an empty world.
    pub fn new() -> Self {
        Self {
            next_entity: 1,
            alive: HashSet::new(),
            destroyed: HashSet::new(),
            storages: HashMap::new(),
            entity_components: HashMap::new(),
            systems: Vec::new(),
            tick_count: 0,
        }
    }

    /// Spawn a new entity, returning its id.
    pub fn spawn(&mut self) -> u64 {
        let id = self.next_entity;
        self.next_entity += 1;
        self.alive.insert(id);
        self.entity_components.insert(id, HashSet::new());
        id
    }

    /// Spawn via the fluent entity builder.
    pub fn build_entity(&mut self) -> EntityBuilder<'_> {
        let entity = self.spawn();
        EntityBuilder { world: self, entity }
    }

    /// Destroy an entity and remove all its components.
    pub fn destroy(&mut self, entity: u64) -> Result<(), EcsError> {
        if self.destroyed.contains(&entity) {
            return Err(EcsError::EntityDestroyed(entity));
        }
        if !self.alive.remove(&entity) {
            return Err(EcsError::EntityNotFound(entity));
        }
        self.destroyed.insert(entity);
        if let Some(type_ids) = self.entity_components.remove(&entity) {
            for tid in type_ids {
                if let Some(storage) = self.storages.get_mut(&tid) {
                    storage.remove(entity);
                }
            }
        }
        Ok(())
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: u64) -> bool {
        self.alive.contains(&entity)
    }

    /// Add a component to an entity.
    pub fn add_component<C: Component>(
        &mut self,
        entity: u64,
        component: C,
    ) -> Result<(), EcsError> {
        if !self.alive.contains(&entity) {
            return Err(EcsError::EntityNotFound(entity));
        }
        let tid = TypeId::of::<C>();
        let storage = self.storages.entry(tid).or_insert_with(SparseSet::new);
        storage.insert(entity, Box::new(component));
        self.entity_components.entry(entity).or_default().insert(tid);
        Ok(())
    }

    /// Remove a component from an entity.
    pub fn remove_component<C: Component>(&mut self, entity: u64) -> Result<C, EcsError> {
        if !self.alive.contains(&entity) {
            return Err(EcsError::EntityNotFound(entity));
        }
        let tid = TypeId::of::<C>();
        let storage = self.storages.get_mut(&tid).ok_or(EcsError::ComponentNotFound {
            entity,
            component: C::component_name(),
        })?;
        let boxed = storage.remove(entity).ok_or(EcsError::ComponentNotFound {
            entity,
            component: C::component_name(),
        })?;
        if let Some(set) = self.entity_components.get_mut(&entity) {
            set.remove(&tid);
        }
        Ok(*boxed.downcast::<C>().unwrap())
    }

    /// Get an immutable reference to a component.
    pub fn get_component<C: Component>(&self, entity: u64) -> Result<&C, EcsError> {
        if !self.alive.contains(&entity) {
            return Err(EcsError::EntityNotFound(entity));
        }
        let tid = TypeId::of::<C>();
        let storage = self.storages.get(&tid).ok_or(EcsError::ComponentNotFound {
            entity,
            component: C::component_name(),
        })?;
        storage
            .get(entity)
            .and_then(|any| any.downcast_ref::<C>())
            .ok_or(EcsError::ComponentNotFound {
                entity,
                component: C::component_name(),
            })
    }

    /// Get a mutable reference to a component.
    pub fn get_component_mut<C: Component>(&mut self, entity: u64) -> Result<&mut C, EcsError> {
        if !self.alive.contains(&entity) {
            return Err(EcsError::EntityNotFound(entity));
        }
        let tid = TypeId::of::<C>();
        let storage = self.storages.get_mut(&tid).ok_or(EcsError::ComponentNotFound {
            entity,
            component: C::component_name(),
        })?;
        storage
            .get_mut(entity)
            .and_then(|any| any.downcast_mut::<C>())
            .ok_or(EcsError::ComponentNotFound {
                entity,
                component: C::component_name(),
            })
    }

    /// Check if entity has a component.
    pub fn has_component<C: Component>(&self, entity: u64) -> bool {
        let tid = TypeId::of::<C>();
        self.storages
            .get(&tid)
            .map(|s| s.contains(entity))
            .unwrap_or(false)
    }

    /// Query all entities that have a given component type.
    pub fn query<C: Component>(&self) -> Vec<u64> {
        let tid = TypeId::of::<C>();
        match self.storages.get(&tid) {
            Some(storage) => storage.entities().to_vec(),
            None => Vec::new(),
        }
    }

    /// Query entities that have ALL of the specified component types.
    pub fn query_with(&self, required: &[TypeId]) -> Vec<u64> {
        if required.is_empty() {
            return self.alive.iter().copied().collect();
        }
        // Start with the smallest storage for efficiency.
        let mut smallest: Option<(&SparseSet, usize)> = None;
        for tid in required {
            match self.storages.get(tid) {
                Some(s) => {
                    if smallest.is_none() || s.len() < smallest.unwrap().1 {
                        smallest = Some((s, s.len()));
                    }
                }
                None => return Vec::new(),
            }
        }
        let base = smallest.unwrap().0;
        base.entities()
            .iter()
            .copied()
            .filter(|e| {
                required.iter().all(|tid| {
                    self.storages
                        .get(tid)
                        .map(|s| s.contains(*e))
                        .unwrap_or(false)
                })
            })
            .collect()
    }

    /// Query entities that have all `required` but none of `excluded`.
    pub fn query_filter(&self, required: &[TypeId], excluded: &[TypeId]) -> Vec<u64> {
        let candidates = self.query_with(required);
        candidates
            .into_iter()
            .filter(|e| {
                !excluded.iter().any(|tid| {
                    self.storages
                        .get(tid)
                        .map(|s| s.contains(*e))
                        .unwrap_or(false)
                })
            })
            .collect()
    }

    /// Get the archetype (set of component types) for an entity.
    pub fn archetype_of(&self, entity: u64) -> Option<Archetype> {
        self.entity_components
            .get(&entity)
            .map(|set| Archetype::new(set.iter().copied().collect()))
    }

    /// Count of component types registered in a storage.
    pub fn component_type_count<C: Component>(&self) -> usize {
        let tid = TypeId::of::<C>();
        self.storages.get(&tid).map(|s| s.len()).unwrap_or(0)
    }

    /// Register a system with a given priority. Lower priority runs first.
    pub fn register_system<S: System + 'static>(
        &mut self,
        system: S,
        priority: i32,
    ) -> Result<(), EcsError> {
        let name = system.name().to_string();
        if self.systems.iter().any(|s| s.name == name) {
            return Err(EcsError::SystemAlreadyRegistered(name));
        }
        self.systems.push(RegisteredSystem {
            name,
            system: Box::new(system),
            priority,
        });
        self.systems.sort_by_key(|s| s.priority);
        Ok(())
    }

    /// Run all registered systems in priority order.
    pub fn tick(&mut self) {
        self.tick_count += 1;
        // Take systems out so we can pass &mut self.
        let mut systems = std::mem::take(&mut self.systems);
        for rs in &mut systems {
            rs.system.run(self);
        }
        self.systems = systems;
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Number of alive entities.
    pub fn entity_count(&self) -> usize {
        self.alive.len()
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Test components.
    #[derive(Debug, Clone, PartialEq)]
    struct Position { x: f64, y: f64 }
    impl Component for Position { fn component_name() -> &'static str { "Position" } }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity { dx: f64, dy: f64 }
    impl Component for Velocity { fn component_name() -> &'static str { "Velocity" } }

    #[derive(Debug, Clone, PartialEq)]
    struct Health(i32);
    impl Component for Health { fn component_name() -> &'static str { "Health" } }

    #[derive(Debug, Clone, PartialEq)]
    struct Dead;
    impl Component for Dead { fn component_name() -> &'static str { "Dead" } }

    #[test]
    fn spawn_and_destroy() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert_eq!(w.entity_count(), 1);
        w.destroy(e).unwrap();
        assert!(!w.is_alive(e));
        assert_eq!(w.entity_count(), 0);
    }

    #[test]
    fn add_get_remove_component() {
        let mut w = World::new();
        let e = w.spawn();
        w.add_component(e, Position { x: 1.0, y: 2.0 }).unwrap();
        assert_eq!(w.get_component::<Position>(e).unwrap(), &Position { x: 1.0, y: 2.0 });
        let removed = w.remove_component::<Position>(e).unwrap();
        assert_eq!(removed, Position { x: 1.0, y: 2.0 });
        assert!(!w.has_component::<Position>(e));
    }

    #[test]
    fn component_mutation() {
        let mut w = World::new();
        let e = w.spawn();
        w.add_component(e, Health(100)).unwrap();
        w.get_component_mut::<Health>(e).unwrap().0 -= 25;
        assert_eq!(w.get_component::<Health>(e).unwrap().0, 75);
    }

    #[test]
    fn entity_builder() {
        let mut w = World::new();
        let e = w.build_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .with(Velocity { dx: 1.0, dy: -1.0 })
            .with(Health(50))
            .build();
        assert!(w.has_component::<Position>(e));
        assert!(w.has_component::<Velocity>(e));
        assert!(w.has_component::<Health>(e));
    }

    #[test]
    fn query_single_type() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        let _e3 = w.spawn();
        w.add_component(e1, Position { x: 0.0, y: 0.0 }).unwrap();
        w.add_component(e2, Position { x: 1.0, y: 1.0 }).unwrap();
        let mut result = w.query::<Position>();
        result.sort();
        assert_eq!(result, vec![e1, e2]);
    }

    #[test]
    fn query_with_multiple() {
        let mut w = World::new();
        let e1 = w.build_entity().with(Position { x: 0.0, y: 0.0 }).with(Velocity { dx: 1.0, dy: 0.0 }).build();
        let _e2 = w.build_entity().with(Position { x: 0.0, y: 0.0 }).build();
        let required = vec![TypeId::of::<Position>(), TypeId::of::<Velocity>()];
        let result = w.query_with(&required);
        assert_eq!(result, vec![e1]);
    }

    #[test]
    fn query_filter_excludes() {
        let mut w = World::new();
        let e1 = w.build_entity().with(Health(100)).build();
        let e2 = w.build_entity().with(Health(0)).with(Dead).build();
        let required = vec![TypeId::of::<Health>()];
        let excluded = vec![TypeId::of::<Dead>()];
        let result = w.query_filter(&required, &excluded);
        assert_eq!(result, vec![e1]);
        assert!(!result.contains(&e2));
    }

    #[test]
    fn archetype_tracking() {
        let mut w = World::new();
        let e = w.build_entity().with(Position { x: 0.0, y: 0.0 }).with(Health(10)).build();
        let arch = w.archetype_of(e).unwrap();
        assert_eq!(arch.component_types.len(), 2);
    }

    #[test]
    fn destroy_removes_components() {
        let mut w = World::new();
        let e = w.build_entity().with(Position { x: 0.0, y: 0.0 }).build();
        w.destroy(e).unwrap();
        assert!(w.query::<Position>().is_empty());
    }

    #[test]
    fn system_registration_and_tick() {
        struct MoveSystem;
        impl System for MoveSystem {
            fn name(&self) -> &str { "MoveSystem" }
            fn run(&mut self, world: &mut World) {
                let entities = world.query_with(&[
                    TypeId::of::<Position>(),
                    TypeId::of::<Velocity>(),
                ]);
                for e in entities {
                    let vel = world.get_component::<Velocity>(e).unwrap().clone();
                    let pos = world.get_component_mut::<Position>(e).unwrap();
                    pos.x += vel.dx;
                    pos.y += vel.dy;
                }
            }
        }

        let mut w = World::new();
        let e = w.build_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .with(Velocity { dx: 5.0, dy: 3.0 })
            .build();
        w.register_system(MoveSystem, 0).unwrap();
        w.tick();
        assert_eq!(w.tick_count(), 1);
        let pos = w.get_component::<Position>(e).unwrap();
        assert!((pos.x - 5.0).abs() < 1e-9);
        assert!((pos.y - 3.0).abs() < 1e-9);
    }

    #[test]
    fn error_on_destroyed_entity() {
        let mut w = World::new();
        let e = w.spawn();
        w.destroy(e).unwrap();
        assert_eq!(w.destroy(e), Err(EcsError::EntityDestroyed(e)));
    }

    #[test]
    fn sparse_set_swap_remove() {
        let mut w = World::new();
        let e1 = w.build_entity().with(Health(1)).build();
        let e2 = w.build_entity().with(Health(2)).build();
        let e3 = w.build_entity().with(Health(3)).build();
        w.destroy(e1).unwrap();
        // e2, e3 still queryable
        assert_eq!(w.get_component::<Health>(e2).unwrap().0, 2);
        assert_eq!(w.get_component::<Health>(e3).unwrap().0, 3);
        assert_eq!(w.component_type_count::<Health>(), 2);
    }
}
