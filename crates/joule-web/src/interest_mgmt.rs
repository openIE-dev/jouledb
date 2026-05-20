//! Area of interest management for MMO-style games — spatial partitioning & events.
//!
//! Tracks entities with position and relevance radius, provides grid-based spatial
//! partitioning for efficient nearby-entity queries, maintains per-player interest
//! sets, generates enter/exit events when entities cross interest boundaries,
//! supports configurable cell size and dynamic interest radius.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Interest management domain errors.
#[derive(Debug, Clone, PartialEq)]
pub enum InterestError {
    /// Entity not found.
    EntityNotFound(u64),
    /// Duplicate entity ID.
    DuplicateEntity(u64),
    /// Invalid radius (must be > 0).
    InvalidRadius(f64),
    /// Invalid cell size (must be > 0).
    InvalidCellSize(f64),
}

impl fmt::Display for InterestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntityNotFound(id) => write!(f, "entity not found: {id}"),
            Self::DuplicateEntity(id) => write!(f, "duplicate entity: {id}"),
            Self::InvalidRadius(r) => write!(f, "invalid radius: {r}"),
            Self::InvalidCellSize(s) => write!(f, "invalid cell size: {s}"),
        }
    }
}

impl std::error::Error for InterestError {}

// ── Position ────────────────────────────────────────────────────

/// 2D position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Squared Euclidean distance to another position.
    pub fn distance_sq(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another position.
    pub fn distance(&self, other: &Position) -> f64 {
        self.distance_sq(other).sqrt()
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.1}, {:.1})", self.x, self.y)
    }
}

// ── Entity ──────────────────────────────────────────────────────

/// An entity with a position and an interest (relevance) radius.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: u64,
    pub position: Position,
    pub relevance_radius: f64,
    /// Whether this entity is a "player" that receives interest events.
    pub is_observer: bool,
}

impl Entity {
    pub fn new(id: u64, x: f64, y: f64, radius: f64) -> Self {
        Self {
            id,
            position: Position::new(x, y),
            relevance_radius: radius,
            is_observer: false,
        }
    }

    pub fn observer(mut self) -> Self {
        self.is_observer = true;
        self
    }

    pub fn with_radius(mut self, r: f64) -> Self {
        self.relevance_radius = r;
        self
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}, pos={}, r={:.1})", self.id, self.position, self.relevance_radius)
    }
}

// ── Interest Event ──────────────────────────────────────────────

/// Events generated when entities enter or exit an observer's interest area.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterestEvent {
    /// Entity entered the observer's interest area.
    Enter { observer_id: u64, entity_id: u64 },
    /// Entity exited the observer's interest area.
    Exit { observer_id: u64, entity_id: u64 },
}

impl fmt::Display for InterestEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enter { observer_id, entity_id } => {
                write!(f, "Enter(observer={observer_id}, entity={entity_id})")
            }
            Self::Exit { observer_id, entity_id } => {
                write!(f, "Exit(observer={observer_id}, entity={entity_id})")
            }
        }
    }
}

// ── Grid Cell ───────────────────────────────────────────────────

/// Cell coordinate in the spatial grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellCoord {
    cx: i64,
    cy: i64,
}

// ── Interest Manager ────────────────────────────────────────────

/// Manages entity positions and interest sets using grid-based spatial partitioning.
pub struct InterestManager {
    cell_size: f64,
    entities: HashMap<u64, Entity>,
    /// Grid: cell -> set of entity IDs.
    grid: HashMap<CellCoord, HashSet<u64>>,
    /// Per-observer: current interest set.
    interest_sets: HashMap<u64, HashSet<u64>>,
}

impl InterestManager {
    pub fn new(cell_size: f64) -> Result<Self, InterestError> {
        if cell_size <= 0.0 {
            return Err(InterestError::InvalidCellSize(cell_size));
        }
        Ok(Self {
            cell_size,
            entities: HashMap::new(),
            grid: HashMap::new(),
            interest_sets: HashMap::new(),
        })
    }

    /// Cell coordinate for a world position.
    fn cell_for(&self, pos: &Position) -> CellCoord {
        Self::cell_for_pos(pos, self.cell_size)
    }

    fn cell_for_pos(pos: &Position, cell_size: f64) -> CellCoord {
        CellCoord {
            cx: (pos.x / cell_size).floor() as i64,
            cy: (pos.y / cell_size).floor() as i64,
        }
    }

    /// Cells that a circle (center + radius) overlaps.
    fn cells_in_radius(&self, center: &Position, radius: f64) -> Vec<CellCoord> {
        let min_cx = ((center.x - radius) / self.cell_size).floor() as i64;
        let max_cx = ((center.x + radius) / self.cell_size).floor() as i64;
        let min_cy = ((center.y - radius) / self.cell_size).floor() as i64;
        let max_cy = ((center.y + radius) / self.cell_size).floor() as i64;
        let mut cells = Vec::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                cells.push(CellCoord { cx, cy });
            }
        }
        cells
    }

    /// Add an entity to the manager.
    pub fn add_entity(&mut self, entity: Entity) -> Result<(), InterestError> {
        if self.entities.contains_key(&entity.id) {
            return Err(InterestError::DuplicateEntity(entity.id));
        }
        let cell = self.cell_for(&entity.position);
        self.grid.entry(cell).or_default().insert(entity.id);
        if entity.is_observer {
            self.interest_sets.insert(entity.id, HashSet::new());
        }
        self.entities.insert(entity.id, entity);
        Ok(())
    }

    /// Remove an entity.
    pub fn remove_entity(&mut self, id: u64) -> Result<Entity, InterestError> {
        let entity = self.entities.remove(&id).ok_or(InterestError::EntityNotFound(id))?;
        let cell = self.cell_for(&entity.position);
        if let Some(set) = self.grid.get_mut(&cell) {
            set.remove(&id);
            if set.is_empty() {
                self.grid.remove(&cell);
            }
        }
        self.interest_sets.remove(&id);
        Ok(entity)
    }

    /// Move an entity to a new position. Returns generated events.
    pub fn move_entity(&mut self, id: u64, new_x: f64, new_y: f64) -> Result<(), InterestError> {
        let entity = self.entities.get_mut(&id).ok_or(InterestError::EntityNotFound(id))?;
        let old_pos = entity.position.clone();
        entity.position = Position::new(new_x, new_y);
        let new_pos = entity.position.clone();
        let old_cell = Self::cell_for_pos(&old_pos, self.cell_size);
        let new_cell = Self::cell_for_pos(&new_pos, self.cell_size);

        if old_cell != new_cell {
            if let Some(set) = self.grid.get_mut(&old_cell) {
                set.remove(&id);
                if set.is_empty() {
                    self.grid.remove(&old_cell);
                }
            }
            self.grid.entry(new_cell).or_default().insert(id);
        }
        Ok(())
    }

    /// Update an entity's interest radius.
    pub fn set_radius(&mut self, id: u64, radius: f64) -> Result<(), InterestError> {
        if radius <= 0.0 {
            return Err(InterestError::InvalidRadius(radius));
        }
        let entity = self.entities.get_mut(&id).ok_or(InterestError::EntityNotFound(id))?;
        entity.relevance_radius = radius;
        Ok(())
    }

    /// Query all entities within `radius` of a given position.
    pub fn query_nearby(&self, center: &Position, radius: f64) -> Vec<u64> {
        let r_sq = radius * radius;
        let cells = self.cells_in_radius(center, radius);
        let mut result = Vec::new();
        for cell in cells {
            if let Some(ids) = self.grid.get(&cell) {
                for &eid in ids {
                    if let Some(e) = self.entities.get(&eid) {
                        if e.position.distance_sq(center) <= r_sq {
                            result.push(eid);
                        }
                    }
                }
            }
        }
        result
    }

    /// Update all interest sets and return enter/exit events.
    pub fn update_interests(&mut self) -> Vec<InterestEvent> {
        let mut events = Vec::new();

        // Collect observer info to avoid borrow issues.
        let observers: Vec<(u64, Position, f64)> = self.entities.values()
            .filter(|e| e.is_observer)
            .map(|e| (e.id, e.position, e.relevance_radius))
            .collect();

        for (obs_id, obs_pos, obs_radius) in &observers {
            let nearby: HashSet<u64> = self.query_nearby(obs_pos, *obs_radius)
                .into_iter()
                .filter(|eid| *eid != *obs_id)
                .collect();

            let old_set = self.interest_sets.entry(*obs_id).or_default();

            // Enter events.
            for &eid in &nearby {
                if !old_set.contains(&eid) {
                    events.push(InterestEvent::Enter { observer_id: *obs_id, entity_id: eid });
                }
            }
            // Exit events.
            for &eid in old_set.iter() {
                if !nearby.contains(&eid) {
                    events.push(InterestEvent::Exit { observer_id: *obs_id, entity_id: eid });
                }
            }
            *old_set = nearby;
        }

        events
    }

    /// Get the current interest set for an observer.
    pub fn interest_set(&self, observer_id: u64) -> Option<&HashSet<u64>> {
        self.interest_sets.get(&observer_id)
    }

    /// Total entities tracked.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of non-empty grid cells.
    pub fn active_cells(&self) -> usize {
        self.grid.len()
    }

    /// Cell size.
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Get entity by ID.
    pub fn get_entity(&self, id: u64) -> Option<&Entity> {
        self.entities.get(&id)
    }
}

impl fmt::Display for InterestManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InterestManager(entities={}, cells={}, cell_size={:.1})",
            self.entities.len(),
            self.grid.len(),
            self.cell_size,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_distance() {
        let a = Position::new(0.0, 0.0);
        let b = Position::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn position_display() {
        let p = Position::new(1.5, 2.5);
        assert_eq!(format!("{p}"), "(1.5, 2.5)");
    }

    #[test]
    fn entity_builder() {
        let e = Entity::new(1, 10.0, 20.0, 50.0).observer().with_radius(100.0);
        assert!(e.is_observer);
        assert_eq!(e.relevance_radius, 100.0);
    }

    #[test]
    fn invalid_cell_size() {
        assert!(InterestManager::new(0.0).is_err());
        assert!(InterestManager::new(-5.0).is_err());
    }

    #[test]
    fn add_and_remove_entity() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 50.0, 50.0, 100.0)).unwrap();
        assert_eq!(mgr.entity_count(), 1);
        mgr.remove_entity(1).unwrap();
        assert_eq!(mgr.entity_count(), 0);
    }

    #[test]
    fn duplicate_entity_error() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 50.0)).unwrap();
        assert!(matches!(
            mgr.add_entity(Entity::new(1, 10.0, 10.0, 50.0)),
            Err(InterestError::DuplicateEntity(1))
        ));
    }

    #[test]
    fn remove_nonexistent_error() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        assert!(matches!(mgr.remove_entity(999), Err(InterestError::EntityNotFound(999))));
    }

    #[test]
    fn query_nearby_finds_entities() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 10.0, 10.0, 50.0)).unwrap();
        mgr.add_entity(Entity::new(2, 20.0, 20.0, 50.0)).unwrap();
        mgr.add_entity(Entity::new(3, 500.0, 500.0, 50.0)).unwrap();
        let nearby = mgr.query_nearby(&Position::new(0.0, 0.0), 100.0);
        assert!(nearby.contains(&1));
        assert!(nearby.contains(&2));
        assert!(!nearby.contains(&3));
    }

    #[test]
    fn move_entity_updates_grid() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 10.0, 10.0, 50.0)).unwrap();
        mgr.move_entity(1, 500.0, 500.0).unwrap();
        let nearby = mgr.query_nearby(&Position::new(0.0, 0.0), 50.0);
        assert!(!nearby.contains(&1));
        let nearby2 = mgr.query_nearby(&Position::new(500.0, 500.0), 50.0);
        assert!(nearby2.contains(&1));
    }

    #[test]
    fn move_nonexistent_entity_error() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        assert!(matches!(mgr.move_entity(999, 0.0, 0.0), Err(InterestError::EntityNotFound(999))));
    }

    #[test]
    fn set_radius() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 50.0)).unwrap();
        mgr.set_radius(1, 200.0).unwrap();
        assert_eq!(mgr.get_entity(1).unwrap().relevance_radius, 200.0);
    }

    #[test]
    fn set_radius_invalid() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 50.0)).unwrap();
        assert!(matches!(mgr.set_radius(1, -1.0), Err(InterestError::InvalidRadius(_))));
    }

    #[test]
    fn interest_enter_events() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 100.0).observer()).unwrap();
        mgr.add_entity(Entity::new(2, 50.0, 50.0, 10.0)).unwrap();
        let events = mgr.update_interests();
        assert!(events.iter().any(|e| matches!(e, InterestEvent::Enter { observer_id: 1, entity_id: 2 })));
    }

    #[test]
    fn interest_exit_events() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 100.0).observer()).unwrap();
        mgr.add_entity(Entity::new(2, 50.0, 0.0, 10.0)).unwrap();
        mgr.update_interests(); // entity 2 enters
        mgr.move_entity(2, 5000.0, 5000.0).unwrap();
        let events = mgr.update_interests();
        assert!(events.iter().any(|e| matches!(e, InterestEvent::Exit { observer_id: 1, entity_id: 2 })));
    }

    #[test]
    fn interest_set_tracking() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 100.0).observer()).unwrap();
        mgr.add_entity(Entity::new(2, 10.0, 10.0, 10.0)).unwrap();
        mgr.update_interests();
        let set = mgr.interest_set(1).unwrap();
        assert!(set.contains(&2));
    }

    #[test]
    fn active_cells_count() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 10.0, 10.0, 50.0)).unwrap();
        mgr.add_entity(Entity::new(2, 10.0, 10.0, 50.0)).unwrap();
        assert_eq!(mgr.active_cells(), 1); // same cell
        mgr.add_entity(Entity::new(3, 500.0, 500.0, 50.0)).unwrap();
        assert_eq!(mgr.active_cells(), 2);
    }

    #[test]
    fn no_self_interest() {
        let mut mgr = InterestManager::new(100.0).unwrap();
        mgr.add_entity(Entity::new(1, 0.0, 0.0, 1000.0).observer()).unwrap();
        mgr.update_interests();
        let set = mgr.interest_set(1).unwrap();
        assert!(!set.contains(&1));
    }

    #[test]
    fn interest_event_display() {
        let e = InterestEvent::Enter { observer_id: 1, entity_id: 2 };
        assert!(format!("{e}").contains("Enter"));
    }

    #[test]
    fn manager_display() {
        let mgr = InterestManager::new(50.0).unwrap();
        let d = format!("{mgr}");
        assert!(d.contains("InterestManager"));
    }
}
