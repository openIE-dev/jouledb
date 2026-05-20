//! Deferred command buffer for ECS.
//!
//! Commands (Spawn, Despawn, InsertComponent, RemoveComponent, etc.) are
//! queued during system execution and applied after the system pass completes.
//! This prevents borrow conflicts during iteration. Batch apply with rollback
//! on error.

use std::any::{Any, TypeId};
use std::collections::HashMap;

// ── Command types ──

/// A deferred ECS command.
pub enum Command {
    /// Spawn an entity with given ID and initial components.
    Spawn {
        entity_id: u64,
        components: Vec<(TypeId, Box<dyn Any>)>,
    },
    /// Despawn an entity.
    Despawn {
        entity_id: u64,
    },
    /// Insert or replace a component on an entity.
    InsertComponent {
        entity_id: u64,
        type_id: TypeId,
        value: Box<dyn Any>,
    },
    /// Remove a component from an entity.
    RemoveComponent {
        entity_id: u64,
        type_id: TypeId,
    },
    /// Remove all components from an entity (but keep it alive).
    ClearComponents {
        entity_id: u64,
    },
    /// Custom command with a boxed closure.
    Custom(Box<dyn FnOnce(&mut CommandWorld) -> Result<(), CommandError>>),
}

// ── CommandError ──

/// Errors that can occur when applying commands.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandError {
    /// Entity already exists (on Spawn).
    EntityAlreadyExists(u64),
    /// Entity not found (on Despawn/Insert/Remove).
    EntityNotFound(u64),
    /// Component not found on entity.
    ComponentNotFound {
        entity_id: u64,
        type_name: String,
    },
    /// Custom error message.
    Custom(String),
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntityAlreadyExists(id) => write!(f, "entity {id} already exists"),
            Self::EntityNotFound(id) => write!(f, "entity {id} not found"),
            Self::ComponentNotFound { entity_id, type_name } => {
                write!(f, "component {type_name} not found on entity {entity_id}")
            }
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

// ── CommandWorld ──

/// Minimal world interface for applying commands. Real ECS worlds should
/// implement `From`/`Into` or provide an adapter.
pub struct CommandWorld {
    /// entity_id → set of (TypeId → boxed component).
    entities: HashMap<u64, HashMap<TypeId, Box<dyn Any>>>,
}

impl CommandWorld {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    pub fn has_entity(&self, id: u64) -> bool {
        self.entities.contains_key(&id)
    }

    pub fn spawn_entity(&mut self, id: u64) -> Result<(), CommandError> {
        if self.entities.contains_key(&id) {
            return Err(CommandError::EntityAlreadyExists(id));
        }
        self.entities.insert(id, HashMap::new());
        Ok(())
    }

    pub fn despawn_entity(&mut self, id: u64) -> Result<(), CommandError> {
        if self.entities.remove(&id).is_none() {
            return Err(CommandError::EntityNotFound(id));
        }
        Ok(())
    }

    pub fn insert_component(
        &mut self,
        entity_id: u64,
        type_id: TypeId,
        value: Box<dyn Any>,
    ) -> Result<Option<Box<dyn Any>>, CommandError> {
        let comps = self
            .entities
            .get_mut(&entity_id)
            .ok_or(CommandError::EntityNotFound(entity_id))?;
        let prev = comps.insert(type_id, value);
        Ok(prev)
    }

    pub fn remove_component(
        &mut self,
        entity_id: u64,
        type_id: TypeId,
    ) -> Result<Box<dyn Any>, CommandError> {
        let comps = self
            .entities
            .get_mut(&entity_id)
            .ok_or(CommandError::EntityNotFound(entity_id))?;
        comps.remove(&type_id).ok_or(CommandError::ComponentNotFound {
            entity_id,
            type_name: format!("{:?}", type_id),
        })
    }

    pub fn clear_components(&mut self, entity_id: u64) -> Result<(), CommandError> {
        let comps = self
            .entities
            .get_mut(&entity_id)
            .ok_or(CommandError::EntityNotFound(entity_id))?;
        comps.clear();
        Ok(())
    }

    pub fn get_component<C: 'static>(&self, entity_id: u64) -> Option<&C> {
        self.entities
            .get(&entity_id)?
            .get(&TypeId::of::<C>())?
            .downcast_ref::<C>()
    }

    pub fn has_component(&self, entity_id: u64, type_id: &TypeId) -> bool {
        self.entities
            .get(&entity_id)
            .map(|c| c.contains_key(type_id))
            .unwrap_or(false)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn component_count(&self, entity_id: u64) -> usize {
        self.entities
            .get(&entity_id)
            .map(|c| c.len())
            .unwrap_or(0)
    }
}

impl Default for CommandWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot for rollback ──

/// Snapshot of the world state before a batch for rollback purposes.
struct WorldSnapshot {
    /// Full clone of entity component maps (type-erased; we store the keys).
    entity_ids: Vec<u64>,
    /// Per-entity: which TypeIds were present.
    entity_types: HashMap<u64, Vec<TypeId>>,
}

impl WorldSnapshot {
    fn capture(world: &CommandWorld) -> Self {
        let entity_ids: Vec<u64> = world.entities.keys().copied().collect();
        let entity_types: HashMap<u64, Vec<TypeId>> = world
            .entities
            .iter()
            .map(|(&eid, comps)| (eid, comps.keys().copied().collect()))
            .collect();
        Self {
            entity_ids,
            entity_types,
        }
    }
}

// ── CommandBuffer ──

/// Queues deferred commands and applies them as a batch.
pub struct CommandBuffer {
    commands: Vec<Command>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Queue a spawn command.
    pub fn spawn(&mut self, entity_id: u64) {
        self.commands.push(Command::Spawn {
            entity_id,
            components: Vec::new(),
        });
    }

    /// Queue a spawn command with initial components.
    pub fn spawn_with(
        &mut self,
        entity_id: u64,
        components: Vec<(TypeId, Box<dyn Any>)>,
    ) {
        self.commands.push(Command::Spawn {
            entity_id,
            components,
        });
    }

    /// Queue a despawn command.
    pub fn despawn(&mut self, entity_id: u64) {
        self.commands.push(Command::Despawn { entity_id });
    }

    /// Queue an insert-component command.
    pub fn insert_component<C: 'static>(&mut self, entity_id: u64, component: C) {
        self.commands.push(Command::InsertComponent {
            entity_id,
            type_id: TypeId::of::<C>(),
            value: Box::new(component),
        });
    }

    /// Queue a remove-component command.
    pub fn remove_component<C: 'static>(&mut self, entity_id: u64) {
        self.commands.push(Command::RemoveComponent {
            entity_id,
            type_id: TypeId::of::<C>(),
        });
    }

    /// Queue a clear-components command.
    pub fn clear_components(&mut self, entity_id: u64) {
        self.commands.push(Command::ClearComponents { entity_id });
    }

    /// Queue a custom command.
    pub fn push_custom<F>(&mut self, f: F)
    where
        F: FnOnce(&mut CommandWorld) -> Result<(), CommandError> + 'static,
    {
        self.commands.push(Command::Custom(Box::new(f)));
    }

    /// Push a raw command.
    pub fn push(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    /// Apply all queued commands to the world. On error, the world state
    /// is partially modified (commands before the error are applied).
    /// Returns the number of commands successfully applied and the error.
    pub fn apply(&mut self, world: &mut CommandWorld) -> Result<usize, (usize, CommandError)> {
        let commands = std::mem::take(&mut self.commands);
        let mut applied = 0;
        for cmd in commands {
            match Self::apply_one(cmd, world) {
                Ok(()) => applied += 1,
                Err(e) => return Err((applied, e)),
            }
        }
        Ok(applied)
    }

    /// Apply all commands, rolling back to original state on first error.
    /// Rollback restores entity existence but not component values (which
    /// are moved out of the command buffer).
    ///
    /// Returns the number of applied commands, or (applied, error).
    pub fn apply_with_rollback(
        &mut self,
        world: &mut CommandWorld,
    ) -> Result<usize, (usize, CommandError)> {
        let snapshot = WorldSnapshot::capture(world);
        let result = self.apply(world);
        if let Err((applied, ref _err)) = result {
            // Rollback: restore entity set to pre-apply state.
            // Remove entities that were spawned by the buffer.
            let current_ids: Vec<u64> = world.entities.keys().copied().collect();
            for eid in &current_ids {
                if !snapshot.entity_ids.contains(eid) {
                    world.entities.remove(eid);
                }
            }
            // Re-add entities that were despawned by the buffer.
            for eid in &snapshot.entity_ids {
                if !world.entities.contains_key(eid) {
                    // Restore with empty component set (values were consumed).
                    world.entities.insert(*eid, HashMap::new());
                }
            }
            return Err((applied, _err.clone()));
        }
        result
    }

    fn apply_one(cmd: Command, world: &mut CommandWorld) -> Result<(), CommandError> {
        match cmd {
            Command::Spawn {
                entity_id,
                components,
            } => {
                world.spawn_entity(entity_id)?;
                for (tid, val) in components {
                    world.insert_component(entity_id, tid, val)?;
                }
                Ok(())
            }
            Command::Despawn { entity_id } => {
                world.despawn_entity(entity_id)
            }
            Command::InsertComponent {
                entity_id,
                type_id,
                value,
            } => {
                world.insert_component(entity_id, type_id, value)?;
                Ok(())
            }
            Command::RemoveComponent { entity_id, type_id } => {
                world.remove_component(entity_id, type_id)?;
                Ok(())
            }
            Command::ClearComponents { entity_id } => {
                world.clear_components(entity_id)
            }
            Command::Custom(f) => f(world),
        }
    }

    /// Drain and return all queued commands without applying.
    pub fn drain(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.commands)
    }

    /// Clear all queued commands.
    pub fn clear(&mut self) {
        self.commands.clear();
    }
}

impl Default for CommandBuffer {
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

    #[test]
    fn spawn_command() {
        let mut buf = CommandBuffer::new();
        let mut world = CommandWorld::new();
        buf.spawn(1);
        let applied = buf.apply(&mut world).unwrap();
        assert_eq!(applied, 1);
        assert!(world.has_entity(1));
    }

    #[test]
    fn spawn_with_components() {
        let mut buf = CommandBuffer::new();
        let mut world = CommandWorld::new();
        buf.spawn_with(1, vec![
            (TypeId::of::<Pos>(), Box::new(Pos { x: 1.0, y: 2.0 })),
            (TypeId::of::<Hp>(), Box::new(Hp(100))),
        ]);
        buf.apply(&mut world).unwrap();
        assert!(world.has_entity(1));
        assert_eq!(world.component_count(1), 2);
        let pos = world.get_component::<Pos>(1).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn despawn_command() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        let mut buf = CommandBuffer::new();
        buf.despawn(1);
        buf.apply(&mut world).unwrap();
        assert!(!world.has_entity(1));
    }

    #[test]
    fn insert_component_command() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        let mut buf = CommandBuffer::new();
        buf.insert_component(1, Pos { x: 3.0, y: 4.0 });
        buf.apply(&mut world).unwrap();
        let pos = world.get_component::<Pos>(1).unwrap();
        assert!((pos.x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn remove_component_command() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        world
            .insert_component(1, TypeId::of::<Hp>(), Box::new(Hp(100)))
            .unwrap();
        let mut buf = CommandBuffer::new();
        buf.remove_component::<Hp>(1);
        buf.apply(&mut world).unwrap();
        assert!(!world.has_component(1, &TypeId::of::<Hp>()));
    }

    #[test]
    fn clear_components_command() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        world
            .insert_component(1, TypeId::of::<Hp>(), Box::new(Hp(100)))
            .unwrap();
        world
            .insert_component(1, TypeId::of::<Pos>(), Box::new(Pos { x: 0.0, y: 0.0 }))
            .unwrap();
        let mut buf = CommandBuffer::new();
        buf.clear_components(1);
        buf.apply(&mut world).unwrap();
        assert_eq!(world.component_count(1), 0);
        assert!(world.has_entity(1));
    }

    #[test]
    fn custom_command() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        let mut buf = CommandBuffer::new();
        buf.push_custom(|w| {
            w.insert_component(1, TypeId::of::<Hp>(), Box::new(Hp(42)))?;
            Ok(())
        });
        buf.apply(&mut world).unwrap();
        assert_eq!(world.get_component::<Hp>(1), Some(&Hp(42)));
    }

    #[test]
    fn error_on_duplicate_spawn() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        let err = buf.apply(&mut world).unwrap_err();
        assert_eq!(err.0, 0);
        assert_eq!(err.1, CommandError::EntityAlreadyExists(1));
    }

    #[test]
    fn error_on_despawn_missing() {
        let mut world = CommandWorld::new();
        let mut buf = CommandBuffer::new();
        buf.despawn(999);
        let err = buf.apply(&mut world).unwrap_err();
        assert_eq!(err.1, CommandError::EntityNotFound(999));
    }

    #[test]
    fn error_on_insert_missing_entity() {
        let mut world = CommandWorld::new();
        let mut buf = CommandBuffer::new();
        buf.insert_component(999, Hp(100));
        let err = buf.apply(&mut world).unwrap_err();
        assert_eq!(err.1, CommandError::EntityNotFound(999));
    }

    #[test]
    fn partial_apply_reports_count() {
        let mut world = CommandWorld::new();
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        buf.spawn(2);
        buf.spawn(2); // duplicate — error
        buf.spawn(3);
        let err = buf.apply(&mut world).unwrap_err();
        assert_eq!(err.0, 2); // 2 succeeded before error
    }

    #[test]
    fn rollback_restores_entities() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        let mut buf = CommandBuffer::new();
        buf.spawn(2); // succeeds
        buf.despawn(999); // fails — entity doesn't exist
        let err = buf.apply_with_rollback(&mut world).unwrap_err();
        assert_eq!(err.0, 1);
        // Rollback: entity 2 should be removed, entity 1 should remain.
        assert!(world.has_entity(1));
        assert!(!world.has_entity(2));
    }

    #[test]
    fn rollback_restores_despawned() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        world.spawn_entity(2).unwrap();
        let mut buf = CommandBuffer::new();
        buf.despawn(1); // succeeds
        buf.despawn(999); // fails
        let err = buf.apply_with_rollback(&mut world).unwrap_err();
        assert_eq!(err.0, 1);
        // Entity 1 should be restored.
        assert!(world.has_entity(1));
        assert!(world.has_entity(2));
    }

    #[test]
    fn buffer_len_and_empty() {
        let mut buf = CommandBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        buf.spawn(1);
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn drain_commands() {
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        buf.despawn(2);
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn clear_commands() {
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        buf.spawn(2);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn multiple_batches() {
        let mut world = CommandWorld::new();
        // First batch.
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        buf.spawn(2);
        buf.apply(&mut world).unwrap();
        assert_eq!(world.entity_count(), 2);
        // Second batch.
        let mut buf2 = CommandBuffer::new();
        buf2.insert_component(1, Hp(100));
        buf2.despawn(2);
        buf2.apply(&mut world).unwrap();
        assert_eq!(world.entity_count(), 1);
        assert_eq!(world.get_component::<Hp>(1), Some(&Hp(100)));
    }

    #[test]
    fn error_display() {
        let err = CommandError::Custom("test error".into());
        assert_eq!(err.to_string(), "test error");
        let err2 = CommandError::EntityNotFound(42);
        assert!(err2.to_string().contains("42"));
    }

    #[test]
    fn successful_rollback_returns_count() {
        let mut world = CommandWorld::new();
        let mut buf = CommandBuffer::new();
        buf.spawn(1);
        buf.spawn(2);
        let count = buf.apply_with_rollback(&mut world).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn command_world_has_component() {
        let mut world = CommandWorld::new();
        world.spawn_entity(1).unwrap();
        assert!(!world.has_component(1, &TypeId::of::<Hp>()));
        world
            .insert_component(1, TypeId::of::<Hp>(), Box::new(Hp(50)))
            .unwrap();
        assert!(world.has_component(1, &TypeId::of::<Hp>()));
    }

    #[test]
    fn command_world_missing_entity_component() {
        let world = CommandWorld::new();
        assert!(!world.has_component(999, &TypeId::of::<Hp>()));
        assert_eq!(world.component_count(999), 0);
    }

    #[test]
    fn push_raw_command() {
        let mut buf = CommandBuffer::new();
        buf.push(Command::Spawn {
            entity_id: 42,
            components: Vec::new(),
        });
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn custom_command_error() {
        let mut world = CommandWorld::new();
        let mut buf = CommandBuffer::new();
        buf.push_custom(|_w| Err(CommandError::Custom("custom fail".into())));
        let err = buf.apply(&mut world).unwrap_err();
        assert_eq!(err.1, CommandError::Custom("custom fail".into()));
    }
}
