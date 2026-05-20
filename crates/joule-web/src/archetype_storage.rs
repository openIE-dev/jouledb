//! Archetype-based ECS storage.
//!
//! Entities with the same set of component types share an archetype. Each
//! archetype stores components in column-oriented layout for cache-friendly
//! iteration. Adding or removing a component from an entity migrates it
//! between archetypes.

use std::any::{Any, TypeId};
use std::collections::{BTreeSet, HashMap};

// ── ArchetypeId ──

/// Unique archetype identifier derived from the sorted set of component TypeIds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArchetypeId(Vec<TypeId>);

impl ArchetypeId {
    /// Create an archetype ID from a set of component type IDs.
    pub fn from_types(types: &BTreeSet<TypeId>) -> Self {
        Self(types.iter().copied().collect())
    }

    /// The component type IDs in this archetype.
    pub fn types(&self) -> &[TypeId] {
        &self.0
    }

    /// Number of component types in this archetype.
    pub fn component_count(&self) -> usize {
        self.0.len()
    }

    /// Check if this archetype contains a specific component type.
    pub fn contains(&self, tid: &TypeId) -> bool {
        self.0.contains(tid)
    }

    /// Create a new archetype ID with an additional type.
    pub fn with_type(&self, tid: TypeId) -> Self {
        let mut set: BTreeSet<TypeId> = self.0.iter().copied().collect();
        set.insert(tid);
        Self::from_types(&set)
    }

    /// Create a new archetype ID without a specific type.
    pub fn without_type(&self, tid: &TypeId) -> Self {
        let mut set: BTreeSet<TypeId> = self.0.iter().copied().collect();
        set.remove(tid);
        Self::from_types(&set)
    }
}

// ── Column ──

/// A column of component data within an archetype.
struct Column {
    /// Parallel to the entity list — one boxed value per entity row.
    data: Vec<Box<dyn Any>>,
}

impl Column {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn push(&mut self, val: Box<dyn Any>) {
        self.data.push(val);
    }

    fn swap_remove(&mut self, index: usize) -> Box<dyn Any> {
        self.data.swap_remove(index)
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<&dyn Any> {
        self.data.get(index).map(|b| b.as_ref())
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Any> {
        self.data.get_mut(index).map(|b| b.as_mut())
    }
}

// ── Archetype ──

/// A single archetype — stores all entities that have exactly a given set of
/// component types. Components are in column-oriented layout.
struct Archetype {
    id: ArchetypeId,
    /// entity IDs in insertion order (parallel to columns).
    entities: Vec<u64>,
    /// entity_id → row index within this archetype.
    entity_to_row: HashMap<u64, usize>,
    /// TypeId → column of component data.
    columns: HashMap<TypeId, Column>,
}

impl Archetype {
    fn new(id: ArchetypeId) -> Self {
        let mut columns = HashMap::new();
        for tid in id.types() {
            columns.insert(*tid, Column::new());
        }
        Self {
            id,
            entities: Vec::new(),
            entity_to_row: HashMap::new(),
            columns,
        }
    }

    fn len(&self) -> usize {
        self.entities.len()
    }

    fn contains_entity(&self, entity_id: u64) -> bool {
        self.entity_to_row.contains_key(&entity_id)
    }

    /// Push a new entity with its component data. `components` must have
    /// exactly one entry per type in the archetype ID.
    fn push_entity(
        &mut self,
        entity_id: u64,
        components: HashMap<TypeId, Box<dyn Any>>,
    ) -> Result<(), String> {
        if self.entity_to_row.contains_key(&entity_id) {
            return Err(format!("entity {entity_id} already in archetype"));
        }
        let row = self.entities.len();
        for tid in self.id.types() {
            match components.get(tid) {
                Some(_) => {}
                None => return Err(format!("missing component for {:?}", tid)),
            }
        }
        for (tid, val) in components {
            if let Some(col) = self.columns.get_mut(&tid) {
                col.push(val);
            }
        }
        self.entities.push(entity_id);
        self.entity_to_row.insert(entity_id, row);
        Ok(())
    }

    /// Remove an entity via swap-remove. Returns the removed components.
    fn remove_entity(&mut self, entity_id: u64) -> Option<HashMap<TypeId, Box<dyn Any>>> {
        let row = *self.entity_to_row.get(&entity_id)?;
        let last_row = self.entities.len() - 1;

        let mut result = HashMap::new();
        for (tid, col) in &mut self.columns {
            let val = col.swap_remove(row);
            result.insert(*tid, val);
        }

        // Fix up the swapped entity's index.
        if row != last_row {
            let swapped_entity = self.entities[last_row];
            self.entities[row] = swapped_entity;
            self.entity_to_row.insert(swapped_entity, row);
        }
        self.entities.pop();
        self.entity_to_row.remove(&entity_id);
        Some(result)
    }

    fn get_component(&self, entity_id: u64, tid: &TypeId) -> Option<&dyn Any> {
        let row = *self.entity_to_row.get(&entity_id)?;
        self.columns.get(tid)?.get(row)
    }

    fn get_component_mut(&mut self, entity_id: u64, tid: &TypeId) -> Option<&mut dyn Any> {
        let row = *self.entity_to_row.get(&entity_id)?;
        self.columns.get_mut(tid)?.get_mut(row)
    }
}

// ── ArchetypeStorage ──

/// Top-level archetype-based component storage.
pub struct ArchetypeStorage {
    /// ArchetypeId → Archetype.
    archetypes: HashMap<ArchetypeId, Archetype>,
    /// entity_id → which archetype it currently lives in.
    entity_archetype: HashMap<u64, ArchetypeId>,
}

impl ArchetypeStorage {
    pub fn new() -> Self {
        Self {
            archetypes: HashMap::new(),
            entity_archetype: HashMap::new(),
        }
    }

    /// Spawn an entity with a set of typed components.
    pub fn spawn(
        &mut self,
        entity_id: u64,
        components: HashMap<TypeId, Box<dyn Any>>,
    ) -> Result<(), String> {
        if self.entity_archetype.contains_key(&entity_id) {
            return Err(format!("entity {entity_id} already exists"));
        }
        let types: BTreeSet<TypeId> = components.keys().copied().collect();
        let arch_id = ArchetypeId::from_types(&types);

        let archetype = self
            .archetypes
            .entry(arch_id.clone())
            .or_insert_with(|| Archetype::new(arch_id.clone()));

        archetype.push_entity(entity_id, components)?;
        self.entity_archetype.insert(entity_id, arch_id);
        Ok(())
    }

    /// Despawn an entity, removing it from its archetype.
    pub fn despawn(&mut self, entity_id: u64) -> bool {
        let arch_id = match self.entity_archetype.remove(&entity_id) {
            Some(id) => id,
            None => return false,
        };
        if let Some(arch) = self.archetypes.get_mut(&arch_id) {
            arch.remove_entity(entity_id);
        }
        true
    }

    /// Add a component to an entity, migrating it to a new archetype.
    pub fn add_component(
        &mut self,
        entity_id: u64,
        tid: TypeId,
        value: Box<dyn Any>,
    ) -> Result<(), String> {
        let old_arch_id = self
            .entity_archetype
            .get(&entity_id)
            .cloned()
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        if old_arch_id.contains(&tid) {
            // Already has this component type — replace in place.
            if let Some(arch) = self.archetypes.get_mut(&old_arch_id) {
                let row = *arch
                    .entity_to_row
                    .get(&entity_id)
                    .ok_or("entity not found in archetype")?;
                if let Some(col) = arch.columns.get_mut(&tid) {
                    col.data[row] = value;
                }
            }
            return Ok(());
        }

        // Migrate to a new archetype.
        let new_arch_id = old_arch_id.with_type(tid);

        let mut components = self
            .archetypes
            .get_mut(&old_arch_id)
            .and_then(|a| a.remove_entity(entity_id))
            .ok_or_else(|| "failed to remove from old archetype".to_string())?;

        components.insert(tid, value);

        let archetype = self
            .archetypes
            .entry(new_arch_id.clone())
            .or_insert_with(|| Archetype::new(new_arch_id.clone()));

        archetype.push_entity(entity_id, components)?;
        self.entity_archetype.insert(entity_id, new_arch_id);
        Ok(())
    }

    /// Remove a component from an entity, migrating it to a new archetype.
    pub fn remove_component(
        &mut self,
        entity_id: u64,
        tid: &TypeId,
    ) -> Result<Box<dyn Any>, String> {
        let old_arch_id = self
            .entity_archetype
            .get(&entity_id)
            .cloned()
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        if !old_arch_id.contains(tid) {
            return Err("entity does not have this component".to_string());
        }

        let new_arch_id = old_arch_id.without_type(tid);

        let mut components = self
            .archetypes
            .get_mut(&old_arch_id)
            .and_then(|a| a.remove_entity(entity_id))
            .ok_or_else(|| "failed to remove from old archetype".to_string())?;

        let removed = components
            .remove(tid)
            .ok_or("component not found in removed data")?;

        if components.is_empty() {
            // Entity has no components — still track it in an empty archetype.
            let archetype = self
                .archetypes
                .entry(new_arch_id.clone())
                .or_insert_with(|| Archetype::new(new_arch_id.clone()));
            archetype.push_entity(entity_id, components)?;
        } else {
            let archetype = self
                .archetypes
                .entry(new_arch_id.clone())
                .or_insert_with(|| Archetype::new(new_arch_id.clone()));
            archetype.push_entity(entity_id, components)?;
        }
        self.entity_archetype.insert(entity_id, new_arch_id);
        Ok(removed)
    }

    /// Get a reference to a component on an entity.
    pub fn get_component<T: 'static>(&self, entity_id: u64) -> Option<&T> {
        let arch_id = self.entity_archetype.get(&entity_id)?;
        let arch = self.archetypes.get(arch_id)?;
        arch.get_component(entity_id, &TypeId::of::<T>())?
            .downcast_ref::<T>()
    }

    /// Get a mutable reference to a component on an entity.
    pub fn get_component_mut<T: 'static>(&mut self, entity_id: u64) -> Option<&mut T> {
        let arch_id = self.entity_archetype.get(&entity_id)?.clone();
        let arch = self.archetypes.get_mut(&arch_id)?;
        arch.get_component_mut(entity_id, &TypeId::of::<T>())?
            .downcast_mut::<T>()
    }

    /// Check if an entity has a specific component type.
    pub fn has_component<T: 'static>(&self, entity_id: u64) -> bool {
        self.entity_archetype
            .get(&entity_id)
            .map(|id| id.contains(&TypeId::of::<T>()))
            .unwrap_or(false)
    }

    /// Number of distinct archetypes.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Number of entities in a specific archetype.
    pub fn archetype_entity_count(&self, arch_id: &ArchetypeId) -> usize {
        self.archetypes.get(arch_id).map(|a| a.len()).unwrap_or(0)
    }

    /// All entity IDs currently stored.
    pub fn all_entities(&self) -> Vec<u64> {
        self.entity_archetype.keys().copied().collect()
    }

    /// Total entity count.
    pub fn entity_count(&self) -> usize {
        self.entity_archetype.len()
    }

    /// Get all entities that have a specific set of component types.
    pub fn query_entities(&self, required: &BTreeSet<TypeId>) -> Vec<u64> {
        let mut result = Vec::new();
        for (arch_id, arch) in &self.archetypes {
            if required.iter().all(|t| arch_id.contains(t)) {
                result.extend(arch.entities.iter().copied());
            }
        }
        result
    }

    /// Get all entities that have a specific set but NOT another set.
    pub fn query_entities_filtered(
        &self,
        required: &BTreeSet<TypeId>,
        excluded: &BTreeSet<TypeId>,
    ) -> Vec<u64> {
        let mut result = Vec::new();
        for (arch_id, arch) in &self.archetypes {
            let has_required = required.iter().all(|t| arch_id.contains(t));
            let has_excluded = excluded.iter().any(|t| arch_id.contains(t));
            if has_required && !has_excluded {
                result.extend(arch.entities.iter().copied());
            }
        }
        result
    }
}

impl Default for ArchetypeStorage {
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

    fn make_components(vals: Vec<(TypeId, Box<dyn Any>)>) -> HashMap<TypeId, Box<dyn Any>> {
        vals.into_iter().collect()
    }

    #[test]
    fn spawn_single_entity() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 2.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        assert_eq!(store.entity_count(), 1);
    }

    #[test]
    fn spawn_duplicate_error() {
        let mut store = ArchetypeStorage::new();
        let comps1 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        let comps2 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 1.0 })),
        ]);
        store.spawn(1, comps1).unwrap();
        assert!(store.spawn(1, comps2).is_err());
    }

    #[test]
    fn despawn_entity() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        assert!(store.despawn(1));
        assert!(!store.despawn(1));
        assert_eq!(store.entity_count(), 0);
    }

    #[test]
    fn get_component() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 3.0, y: 4.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        let pos = store.get_component::<Pos>(1).unwrap();
        assert!((pos.x - 3.0).abs() < 1e-9);
        assert!((pos.y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn get_component_mut() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        let pos = store.get_component_mut::<Pos>(1).unwrap();
        pos.x = 99.0;
        assert!((store.get_component::<Pos>(1).unwrap().x - 99.0).abs() < 1e-9);
    }

    #[test]
    fn has_component() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        assert!(store.has_component::<Pos>(1));
        assert!(!store.has_component::<Vel>(1));
    }

    #[test]
    fn add_component_migration() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 2.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        store
            .add_component(1, TypeId::of::<Vel>(), Box::new(Vel { dx: 3.0, dy: 4.0 }))
            .unwrap();
        assert!(store.has_component::<Pos>(1));
        assert!(store.has_component::<Vel>(1));
        let vel = store.get_component::<Vel>(1).unwrap();
        assert!((vel.dx - 3.0).abs() < 1e-9);
        // Original Pos preserved.
        let pos = store.get_component::<Pos>(1).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn add_existing_component_replaces() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Hp>(), Box::new(Hp(100))),
        ]);
        store.spawn(1, comps).unwrap();
        store
            .add_component(1, TypeId::of::<Hp>(), Box::new(Hp(50)))
            .unwrap();
        assert_eq!(store.get_component::<Hp>(1), Some(&Hp(50)));
    }

    #[test]
    fn remove_component_migration() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 2.0 })),
            (TypeId::of::<Vel>(), Box::new(Vel { dx: 3.0, dy: 4.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        let removed = store.remove_component(1, &TypeId::of::<Vel>()).unwrap();
        assert!(removed.downcast_ref::<Vel>().is_some());
        assert!(store.has_component::<Pos>(1));
        assert!(!store.has_component::<Vel>(1));
    }

    #[test]
    fn remove_component_entity_not_found() {
        let mut store = ArchetypeStorage::new();
        assert!(store.remove_component(999, &TypeId::of::<Pos>()).is_err());
    }

    #[test]
    fn remove_component_not_present() {
        let mut store = ArchetypeStorage::new();
        let comps = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps).unwrap();
        assert!(store.remove_component(1, &TypeId::of::<Vel>()).is_err());
    }

    #[test]
    fn same_archetype_shared() {
        let mut store = ArchetypeStorage::new();
        let comps1 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 1.0 })),
            (TypeId::of::<Vel>(), Box::new(Vel { dx: 0.0, dy: 0.0 })),
        ]);
        let comps2 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 2.0, y: 2.0 })),
            (TypeId::of::<Vel>(), Box::new(Vel { dx: 1.0, dy: 1.0 })),
        ]);
        store.spawn(1, comps1).unwrap();
        store.spawn(2, comps2).unwrap();
        // Both share the same archetype.
        assert_eq!(store.archetype_count(), 1);
    }

    #[test]
    fn different_archetypes() {
        let mut store = ArchetypeStorage::new();
        let comps1 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        let comps2 = make_components(vec![
            (TypeId::of::<Vel>(), Box::new(Vel { dx: 0.0, dy: 0.0 })),
        ]);
        store.spawn(1, comps1).unwrap();
        store.spawn(2, comps2).unwrap();
        assert_eq!(store.archetype_count(), 2);
    }

    #[test]
    fn query_entities_basic() {
        let mut store = ArchetypeStorage::new();
        let comps1 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
            (TypeId::of::<Vel>(), Box::new(Vel { dx: 0.0, dy: 0.0 })),
        ]);
        let comps2 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps1).unwrap();
        store.spawn(2, comps2).unwrap();
        let required: BTreeSet<TypeId> = [TypeId::of::<Pos>(), TypeId::of::<Vel>()]
            .into_iter()
            .collect();
        let result = store.query_entities(&required);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn query_with_exclusion() {
        let mut store = ArchetypeStorage::new();
        let comps1 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
            (TypeId::of::<Hp>(), Box::new(Hp(100))),
        ]);
        let comps2 = make_components(vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 })),
        ]);
        store.spawn(1, comps1).unwrap();
        store.spawn(2, comps2).unwrap();
        let required: BTreeSet<TypeId> = [TypeId::of::<Pos>()].into_iter().collect();
        let excluded: BTreeSet<TypeId> = [TypeId::of::<Hp>()].into_iter().collect();
        let result = store.query_entities_filtered(&required, &excluded);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn archetype_id_with_without() {
        let types: BTreeSet<TypeId> =
            [TypeId::of::<Pos>(), TypeId::of::<Vel>()].into_iter().collect();
        let id = ArchetypeId::from_types(&types);
        assert_eq!(id.component_count(), 2);

        let with_hp = id.with_type(TypeId::of::<Hp>());
        assert_eq!(with_hp.component_count(), 3);
        assert!(with_hp.contains(&TypeId::of::<Hp>()));

        let without_vel = id.without_type(&TypeId::of::<Vel>());
        assert_eq!(without_vel.component_count(), 1);
        assert!(!without_vel.contains(&TypeId::of::<Vel>()));
    }

    #[test]
    fn add_component_to_nonexistent_entity() {
        let mut store = ArchetypeStorage::new();
        assert!(store
            .add_component(999, TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 }))
            .is_err());
    }

    #[test]
    fn get_component_missing_entity() {
        let store = ArchetypeStorage::new();
        assert!(store.get_component::<Pos>(999).is_none());
    }

    #[test]
    fn swap_remove_preserves_other_entities() {
        let mut store = ArchetypeStorage::new();
        for i in 0..5 {
            let comps = make_components(vec![
                (TypeId::of::<Hp>(), Box::new(Hp(i as u32))),
            ]);
            store.spawn(i, comps).unwrap();
        }
        // Remove entity in the middle.
        store.despawn(2);
        assert_eq!(store.entity_count(), 4);
        // All remaining entities still accessible.
        for i in [0u64, 1, 3, 4] {
            assert!(store.get_component::<Hp>(i).is_some());
        }
    }

    #[test]
    fn all_entities_list() {
        let mut store = ArchetypeStorage::new();
        for i in 0..3 {
            let comps = make_components(vec![
                (TypeId::of::<Hp>(), Box::new(Hp(i))),
            ]);
            store.spawn(i as u64, comps).unwrap();
        }
        let mut all = store.all_entities();
        all.sort();
        assert_eq!(all, vec![0, 1, 2]);
    }
}
