//! Type-erased component storage for a game-engine ECS.
//!
//! Component types are registered via `TypeId`-based dispatch and stored as
//! boxed `Any` values. Components can be attached to, detached from, and
//! queried by entity. Supports iteration over all entities that have a given
//! component type.

use std::any::{Any, TypeId};
use std::collections::HashMap;

// ── Component trait ──

/// Marker trait for ECS components. Requires `'static` for `TypeId`.
pub trait Component: Any + 'static {
    /// Human-readable name for debug output.
    fn component_name() -> &'static str
    where
        Self: Sized;
}

// ── ComponentTypeInfo ──

/// Metadata about a registered component type.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentTypeInfo {
    pub type_id: TypeId,
    pub name: String,
    pub registered_order: usize,
}

// ── Storage for one component type ──

struct ComponentColumn {
    info: ComponentTypeInfo,
    /// entity_id → boxed component value.
    data: HashMap<u64, Box<dyn Any>>,
}

impl ComponentColumn {
    fn new(info: ComponentTypeInfo) -> Self {
        Self {
            info,
            data: HashMap::new(),
        }
    }
}

// ── ComponentRegistry ──

/// Central registry for all component types and their per-entity data.
pub struct ComponentRegistry {
    columns: HashMap<TypeId, ComponentColumn>,
    next_order: usize,
}

impl ComponentRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            next_order: 0,
        }
    }

    /// Register a component type. Idempotent — returns false if already registered.
    pub fn register<C: Component>(&mut self) -> bool {
        let tid = TypeId::of::<C>();
        if self.columns.contains_key(&tid) {
            return false;
        }
        let info = ComponentTypeInfo {
            type_id: tid,
            name: C::component_name().to_string(),
            registered_order: self.next_order,
        };
        self.next_order += 1;
        self.columns.insert(tid, ComponentColumn::new(info));
        true
    }

    /// Check whether a component type is registered.
    pub fn is_registered<C: Component>(&self) -> bool {
        self.columns.contains_key(&TypeId::of::<C>())
    }

    /// Get info about a registered component type.
    pub fn type_info<C: Component>(&self) -> Option<&ComponentTypeInfo> {
        self.columns.get(&TypeId::of::<C>()).map(|c| &c.info)
    }

    /// Number of registered component types.
    pub fn type_count(&self) -> usize {
        self.columns.len()
    }

    /// Attach a component value to an entity. Auto-registers the type if needed.
    /// Returns the previous value if one existed.
    pub fn insert<C: Component>(&mut self, entity_id: u64, component: C) -> Option<C> {
        let tid = TypeId::of::<C>();
        if !self.columns.contains_key(&tid) {
            self.register::<C>();
        }
        let col = self.columns.get_mut(&tid).unwrap();
        let prev = col.data.remove(&entity_id);
        col.data.insert(entity_id, Box::new(component));
        prev.and_then(|b| b.downcast::<C>().ok().map(|b| *b))
    }

    /// Get a shared reference to an entity's component.
    pub fn get<C: Component>(&self, entity_id: u64) -> Option<&C> {
        let tid = TypeId::of::<C>();
        self.columns
            .get(&tid)
            .and_then(|col| col.data.get(&entity_id))
            .and_then(|b| b.downcast_ref::<C>())
    }

    /// Get a mutable reference to an entity's component.
    pub fn get_mut<C: Component>(&mut self, entity_id: u64) -> Option<&mut C> {
        let tid = TypeId::of::<C>();
        self.columns
            .get_mut(&tid)
            .and_then(|col| col.data.get_mut(&entity_id))
            .and_then(|b| b.downcast_mut::<C>())
    }

    /// Remove a component from an entity, returning it.
    pub fn remove<C: Component>(&mut self, entity_id: u64) -> Option<C> {
        let tid = TypeId::of::<C>();
        self.columns
            .get_mut(&tid)
            .and_then(|col| col.data.remove(&entity_id))
            .and_then(|b| b.downcast::<C>().ok().map(|b| *b))
    }

    /// Check whether an entity has a specific component.
    pub fn has<C: Component>(&self, entity_id: u64) -> bool {
        let tid = TypeId::of::<C>();
        self.columns
            .get(&tid)
            .map(|col| col.data.contains_key(&entity_id))
            .unwrap_or(false)
    }

    /// Number of entities that have a specific component type.
    pub fn count<C: Component>(&self) -> usize {
        let tid = TypeId::of::<C>();
        self.columns.get(&tid).map(|col| col.data.len()).unwrap_or(0)
    }

    /// Get all entity IDs that have a specific component type.
    /// Order is not guaranteed.
    pub fn entities_with<C: Component>(&self) -> Vec<u64> {
        let tid = TypeId::of::<C>();
        self.columns
            .get(&tid)
            .map(|col| col.data.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Remove all components of all types for a given entity.
    /// Returns the number of component types removed.
    pub fn remove_all_for_entity(&mut self, entity_id: u64) -> usize {
        let mut removed = 0;
        for col in self.columns.values_mut() {
            if col.data.remove(&entity_id).is_some() {
                removed += 1;
            }
        }
        removed
    }

    /// Remove all component data for a specific type across all entities.
    pub fn clear_type<C: Component>(&mut self) {
        let tid = TypeId::of::<C>();
        if let Some(col) = self.columns.get_mut(&tid) {
            col.data.clear();
        }
    }

    /// Unregister a component type entirely, dropping all stored data.
    pub fn unregister<C: Component>(&mut self) -> bool {
        let tid = TypeId::of::<C>();
        self.columns.remove(&tid).is_some()
    }

    /// Get entities that have ALL of the specified component types (by TypeId).
    pub fn entities_with_all(&self, type_ids: &[TypeId]) -> Vec<u64> {
        if type_ids.is_empty() {
            return Vec::new();
        }
        // Start with the smallest set for efficiency.
        let mut sets: Vec<&HashMap<u64, Box<dyn Any>>> = Vec::new();
        for tid in type_ids {
            match self.columns.get(tid) {
                Some(col) => sets.push(&col.data),
                None => return Vec::new(),
            }
        }
        sets.sort_by_key(|s| s.len());
        let first = sets[0];
        first
            .keys()
            .filter(|eid| sets[1..].iter().all(|s| s.contains_key(eid)))
            .copied()
            .collect()
    }

    /// Get entities that have ANY of the specified component types.
    pub fn entities_with_any(&self, type_ids: &[TypeId]) -> Vec<u64> {
        let mut result: Vec<u64> = Vec::new();
        let mut seen: HashMap<u64, bool> = HashMap::new();
        for tid in type_ids {
            if let Some(col) = self.columns.get(tid) {
                for eid in col.data.keys() {
                    if !seen.contains_key(eid) {
                        seen.insert(*eid, true);
                        result.push(*eid);
                    }
                }
            }
        }
        result
    }

    /// Bulk insert — attach the same component value (cloned) to multiple entities.
    pub fn bulk_insert<C: Component + Clone>(&mut self, entity_ids: &[u64], component: &C) {
        let tid = TypeId::of::<C>();
        if !self.columns.contains_key(&tid) {
            self.register::<C>();
        }
        let col = self.columns.get_mut(&tid).unwrap();
        for &eid in entity_ids {
            col.data.insert(eid, Box::new(component.clone()));
        }
    }

    /// Bulk remove — detach a component type from multiple entities.
    pub fn bulk_remove<C: Component>(&mut self, entity_ids: &[u64]) -> usize {
        let tid = TypeId::of::<C>();
        let col = match self.columns.get_mut(&tid) {
            Some(c) => c,
            None => return 0,
        };
        let mut removed = 0;
        for &eid in entity_ids {
            if col.data.remove(&eid).is_some() {
                removed += 1;
            }
        }
        removed
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // Test component types.
    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f64,
        y: f64,
    }
    impl Component for Position {
        fn component_name() -> &'static str {
            "Position"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        dx: f64,
        dy: f64,
    }
    impl Component for Velocity {
        fn component_name() -> &'static str {
            "Velocity"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Health(u32);
    impl Component for Health {
        fn component_name() -> &'static str {
            "Health"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Name(String);
    impl Component for Name {
        fn component_name() -> &'static str {
            "Name"
        }
    }

    #[test]
    fn register_and_info() {
        let mut reg = ComponentRegistry::new();
        assert!(reg.register::<Position>());
        assert!(!reg.register::<Position>()); // idempotent
        assert!(reg.is_registered::<Position>());
        let info = reg.type_info::<Position>().unwrap();
        assert_eq!(info.name, "Position");
        assert_eq!(info.registered_order, 0);
    }

    #[test]
    fn type_count() {
        let mut reg = ComponentRegistry::new();
        reg.register::<Position>();
        reg.register::<Velocity>();
        assert_eq!(reg.type_count(), 2);
    }

    #[test]
    fn insert_and_get() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 10.0, y: 20.0 });
        let pos = reg.get::<Position>(1).unwrap();
        assert!((pos.x - 10.0).abs() < 1e-9);
        assert!((pos.y - 20.0).abs() < 1e-9);
    }

    #[test]
    fn insert_replaces() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Health(100));
        let prev = reg.insert(1, Health(50));
        assert_eq!(prev, Some(Health(100)));
        assert_eq!(reg.get::<Health>(1), Some(&Health(50)));
    }

    #[test]
    fn get_mut() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        if let Some(pos) = reg.get_mut::<Position>(1) {
            pos.x = 42.0;
        }
        assert!((reg.get::<Position>(1).unwrap().x - 42.0).abs() < 1e-9);
    }

    #[test]
    fn remove_component() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Health(100));
        let removed = reg.remove::<Health>(1);
        assert_eq!(removed, Some(Health(100)));
        assert!(!reg.has::<Health>(1));
    }

    #[test]
    fn has_component() {
        let mut reg = ComponentRegistry::new();
        assert!(!reg.has::<Position>(1));
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        assert!(reg.has::<Position>(1));
    }

    #[test]
    fn count_entities() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Health(100));
        reg.insert(2, Health(200));
        reg.insert(3, Health(50));
        assert_eq!(reg.count::<Health>(), 3);
    }

    #[test]
    fn entities_with_type() {
        let mut reg = ComponentRegistry::new();
        reg.insert(10, Health(100));
        reg.insert(20, Health(200));
        let mut ids = reg.entities_with::<Health>();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn remove_all_for_entity() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        reg.insert(1, Health(100));
        reg.insert(1, Velocity { dx: 1.0, dy: 2.0 });
        let removed = reg.remove_all_for_entity(1);
        assert_eq!(removed, 3);
        assert!(!reg.has::<Position>(1));
        assert!(!reg.has::<Health>(1));
        assert!(!reg.has::<Velocity>(1));
    }

    #[test]
    fn clear_type() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Health(100));
        reg.insert(2, Health(200));
        reg.clear_type::<Health>();
        assert_eq!(reg.count::<Health>(), 0);
        assert!(reg.is_registered::<Health>());
    }

    #[test]
    fn unregister() {
        let mut reg = ComponentRegistry::new();
        reg.register::<Health>();
        assert!(reg.unregister::<Health>());
        assert!(!reg.is_registered::<Health>());
        assert!(!reg.unregister::<Health>()); // not registered
    }

    #[test]
    fn entities_with_all() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        reg.insert(1, Velocity { dx: 1.0, dy: 0.0 });
        reg.insert(2, Position { x: 5.0, y: 5.0 });
        // Only entity 1 has both.
        let types = [TypeId::of::<Position>(), TypeId::of::<Velocity>()];
        let result = reg.entities_with_all(&types);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn entities_with_all_empty_types() {
        let reg = ComponentRegistry::new();
        assert!(reg.entities_with_all(&[]).is_empty());
    }

    #[test]
    fn entities_with_all_missing_type() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        let types = [TypeId::of::<Position>(), TypeId::of::<Health>()];
        assert!(reg.entities_with_all(&types).is_empty());
    }

    #[test]
    fn entities_with_any() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 0.0, y: 0.0 });
        reg.insert(2, Velocity { dx: 1.0, dy: 0.0 });
        reg.insert(3, Position { x: 5.0, y: 5.0 });
        let types = [TypeId::of::<Position>(), TypeId::of::<Velocity>()];
        let mut result = reg.entities_with_any(&types);
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn bulk_insert_and_remove() {
        let mut reg = ComponentRegistry::new();
        let entities = vec![1, 2, 3, 4, 5];
        reg.bulk_insert(&entities, &Health(100));
        assert_eq!(reg.count::<Health>(), 5);
        let removed = reg.bulk_remove::<Health>(&[2, 4]);
        assert_eq!(removed, 2);
        assert_eq!(reg.count::<Health>(), 3);
    }

    #[test]
    fn get_missing_entity() {
        let reg = ComponentRegistry::new();
        assert!(reg.get::<Position>(999).is_none());
    }

    #[test]
    fn auto_register_on_insert() {
        let mut reg = ComponentRegistry::new();
        assert!(!reg.is_registered::<Name>());
        reg.insert(1, Name("Alice".into()));
        assert!(reg.is_registered::<Name>());
        assert_eq!(reg.get::<Name>(1), Some(&Name("Alice".into())));
    }

    #[test]
    fn multiple_components_per_entity() {
        let mut reg = ComponentRegistry::new();
        reg.insert(1, Position { x: 1.0, y: 2.0 });
        reg.insert(1, Velocity { dx: 3.0, dy: 4.0 });
        reg.insert(1, Health(100));
        assert!(reg.has::<Position>(1));
        assert!(reg.has::<Velocity>(1));
        assert!(reg.has::<Health>(1));
    }

    #[test]
    fn unregistered_type_count_zero() {
        let reg = ComponentRegistry::new();
        assert_eq!(reg.count::<Health>(), 0);
    }

    #[test]
    fn bulk_remove_unregistered_type() {
        let mut reg = ComponentRegistry::new();
        let removed = reg.bulk_remove::<Health>(&[1, 2, 3]);
        assert_eq!(removed, 0);
    }
}
