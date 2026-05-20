//! Netcode snapshot interpolation — snapshot buffering and interpolation for
//! smooth networked entity rendering.
//!
//! Replaces Unity/Unreal snapshot interpolation with a pure-Rust system.
//! Stores snapshots in a ring buffer, interpolates between two snapshots at
//! the render time, extrapolates when no future snapshot exists, computes
//! snapshot deltas, and tolerates network jitter via configurable delay.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Snapshot interpolation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    /// Buffer is empty, cannot interpolate.
    BufferEmpty,
    /// Not enough snapshots for interpolation (need at least 2).
    InsufficientSnapshots { have: usize },
    /// Entity not found in snapshot.
    EntityNotFound { entity_id: u64, tick: u64 },
    /// Render time is before all buffered snapshots.
    RenderTimeTooEarly { render_time: f64, earliest: f64 },
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferEmpty => write!(f, "snapshot buffer is empty"),
            Self::InsufficientSnapshots { have } => {
                write!(f, "need 2 snapshots for interpolation, have {have}")
            }
            Self::EntityNotFound { entity_id, tick } => {
                write!(f, "entity {entity_id} not found at tick {tick}")
            }
            Self::RenderTimeTooEarly { render_time, earliest } => {
                write!(f, "render time {render_time:.3} before earliest {earliest:.3}")
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

// ── Entity State ────────────────────────────────────────────────

/// State of a single entity in a snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityState {
    pub entity_id: u64,
    pub pos_x: f64,
    pub pos_y: f64,
    pub pos_z: f64,
    pub vel_x: f64,
    pub vel_y: f64,
    pub vel_z: f64,
    pub rotation: f64,
}

impl EntityState {
    pub fn new(id: u64, x: f64, y: f64, z: f64) -> Self {
        Self {
            entity_id: id,
            pos_x: x,
            pos_y: y,
            pos_z: z,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 0.0,
            rotation: 0.0,
        }
    }

    pub fn with_velocity(mut self, vx: f64, vy: f64, vz: f64) -> Self {
        self.vel_x = vx;
        self.vel_y = vy;
        self.vel_z = vz;
        self
    }

    pub fn with_rotation(mut self, rot: f64) -> Self {
        self.rotation = rot;
        self
    }

    /// Linearly interpolate between self and other at factor t in [0,1].
    pub fn lerp(&self, other: &EntityState, t: f64) -> EntityState {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        EntityState {
            entity_id: self.entity_id,
            pos_x: self.pos_x * inv + other.pos_x * t,
            pos_y: self.pos_y * inv + other.pos_y * t,
            pos_z: self.pos_z * inv + other.pos_z * t,
            vel_x: self.vel_x * inv + other.vel_x * t,
            vel_y: self.vel_y * inv + other.vel_y * t,
            vel_z: self.vel_z * inv + other.vel_z * t,
            rotation: self.rotation * inv + other.rotation * t,
        }
    }

    /// Extrapolate from self using velocity for a given duration.
    pub fn extrapolate(&self, dt: f64) -> EntityState {
        EntityState {
            entity_id: self.entity_id,
            pos_x: self.pos_x + self.vel_x * dt,
            pos_y: self.pos_y + self.vel_y * dt,
            pos_z: self.pos_z + self.vel_z * dt,
            vel_x: self.vel_x,
            vel_y: self.vel_y,
            vel_z: self.vel_z,
            rotation: self.rotation,
        }
    }
}

impl fmt::Display for EntityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Entity({}, pos=({:.2},{:.2},{:.2}), rot={:.2})",
            self.entity_id, self.pos_x, self.pos_y, self.pos_z, self.rotation
        )
    }
}

// ── Snapshot ────────────────────────────────────────────────────

/// A full snapshot of entity states at a specific tick.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub tick: u64,
    pub server_time: f64,
    pub entities: Vec<EntityState>,
}

impl Snapshot {
    pub fn new(tick: u64, server_time: f64) -> Self {
        Self { tick, server_time, entities: Vec::new() }
    }

    pub fn with_entity(mut self, entity: EntityState) -> Self {
        self.entities.push(entity);
        self
    }

    pub fn add_entity(&mut self, entity: EntityState) {
        self.entities.push(entity);
    }

    pub fn find_entity(&self, entity_id: u64) -> Option<&EntityState> {
        self.entities.iter().find(|e| e.entity_id == entity_id)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

// ── Snapshot Delta ──────────────────────────────────────────────

/// Difference between two snapshots for a specific entity.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityDelta {
    pub entity_id: u64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub d_rotation: f64,
}

/// Delta between two full snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDelta {
    pub from_tick: u64,
    pub to_tick: u64,
    pub entity_deltas: Vec<EntityDelta>,
    pub added_entities: Vec<u64>,
    pub removed_entities: Vec<u64>,
}

impl SnapshotDelta {
    /// Compute the delta from `from` to `to`.
    pub fn compute(from: &Snapshot, to: &Snapshot) -> Self {
        let mut entity_deltas = Vec::new();
        let mut added_entities = Vec::new();

        for to_ent in &to.entities {
            if let Some(from_ent) = from.find_entity(to_ent.entity_id) {
                entity_deltas.push(EntityDelta {
                    entity_id: to_ent.entity_id,
                    dx: to_ent.pos_x - from_ent.pos_x,
                    dy: to_ent.pos_y - from_ent.pos_y,
                    dz: to_ent.pos_z - from_ent.pos_z,
                    d_rotation: to_ent.rotation - from_ent.rotation,
                });
            } else {
                added_entities.push(to_ent.entity_id);
            }
        }

        let removed_entities: Vec<u64> = from
            .entities
            .iter()
            .filter(|e| to.find_entity(e.entity_id).is_none())
            .map(|e| e.entity_id)
            .collect();

        Self { from_tick: from.tick, to_tick: to.tick, entity_deltas, added_entities, removed_entities }
    }

    pub fn changed_count(&self) -> usize {
        self.entity_deltas.len()
    }
}

// ── Snapshot Buffer ─────────────────────────────────────────────

/// Ring buffer of network snapshots, sorted by server time.
#[derive(Debug)]
pub struct SnapshotBuffer {
    snapshots: VecDeque<Snapshot>,
    capacity: usize,
}

impl SnapshotBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { snapshots: VecDeque::with_capacity(capacity), capacity }
    }

    /// Insert a snapshot, maintaining time order.
    pub fn insert(&mut self, snapshot: Snapshot) {
        if self.snapshots.len() >= self.capacity {
            self.snapshots.pop_front();
        }
        // Insert in order by server_time.
        let pos = self
            .snapshots
            .iter()
            .position(|s| s.server_time > snapshot.server_time)
            .unwrap_or(self.snapshots.len());
        self.snapshots.insert(pos, snapshot);
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.back()
    }

    pub fn earliest(&self) -> Option<&Snapshot> {
        self.snapshots.front()
    }

    /// Find the two snapshots that bracket the given render time.
    pub fn find_bracket(&self, render_time: f64) -> Option<(&Snapshot, &Snapshot)> {
        if self.snapshots.len() < 2 {
            return None;
        }
        for i in 0..self.snapshots.len() - 1 {
            let a = &self.snapshots[i];
            let b = &self.snapshots[i + 1];
            if a.server_time <= render_time && b.server_time >= render_time {
                return Some((a, b));
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

// ── Interpolation Config ────────────────────────────────────────

/// Configuration for the snapshot interpolation system.
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Fixed delay behind latest snapshot (seconds) for jitter tolerance.
    pub interpolation_delay: f64,
    /// Maximum extrapolation time when no future snapshot exists.
    pub max_extrapolation: f64,
    /// Snapshot buffer capacity.
    pub buffer_capacity: usize,
}

impl InterpolationConfig {
    pub fn new() -> Self {
        Self { interpolation_delay: 0.1, max_extrapolation: 0.2, buffer_capacity: 32 }
    }

    pub fn with_delay(mut self, delay: f64) -> Self {
        self.interpolation_delay = delay;
        self
    }

    pub fn with_max_extrapolation(mut self, max: f64) -> Self {
        self.max_extrapolation = max;
        self
    }

    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.buffer_capacity = cap;
        self
    }
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Interpolation Result ────────────────────────────────────────

/// How a state was computed.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMode {
    Interpolated { from_tick: u64, to_tick: u64, t: f64 },
    Extrapolated { from_tick: u64, dt: f64 },
    Exact { tick: u64 },
}

/// Result of interpolating an entity.
#[derive(Debug, Clone)]
pub struct InterpolatedEntity {
    pub state: EntityState,
    pub mode: InterpolationMode,
}

// ── Snapshot Interpolator ───────────────────────────────────────

/// Main interpolation system: buffer snapshots, interpolate at render time.
#[derive(Debug)]
pub struct SnapshotInterpolator {
    config: InterpolationConfig,
    buffer: SnapshotBuffer,
    current_time: f64,
}

impl SnapshotInterpolator {
    pub fn new(config: InterpolationConfig) -> Self {
        let cap = config.buffer_capacity;
        Self { config, buffer: SnapshotBuffer::new(cap), current_time: 0.0 }
    }

    /// Receive a snapshot from the server.
    pub fn receive_snapshot(&mut self, snapshot: Snapshot) {
        self.buffer.insert(snapshot);
    }

    /// Advance the local clock.
    pub fn advance_time(&mut self, dt: f64) {
        self.current_time += dt;
    }

    pub fn set_time(&mut self, t: f64) {
        self.current_time = t;
    }

    /// Compute the render time (current time minus interpolation delay).
    pub fn render_time(&self) -> f64 {
        self.current_time - self.config.interpolation_delay
    }

    /// Interpolate a specific entity at the current render time.
    pub fn interpolate_entity(&self, entity_id: u64) -> Result<InterpolatedEntity, SnapshotError> {
        if self.buffer.is_empty() {
            return Err(SnapshotError::BufferEmpty);
        }

        let render_time = self.render_time();

        // Try bracketed interpolation.
        if let Some((from, to)) = self.buffer.find_bracket(render_time) {
            let dt = to.server_time - from.server_time;
            let t = if dt > 0.0 { (render_time - from.server_time) / dt } else { 0.0 };

            let from_ent = from.find_entity(entity_id).ok_or(SnapshotError::EntityNotFound {
                entity_id,
                tick: from.tick,
            })?;
            let to_ent = to.find_entity(entity_id).ok_or(SnapshotError::EntityNotFound {
                entity_id,
                tick: to.tick,
            })?;

            return Ok(InterpolatedEntity {
                state: from_ent.lerp(to_ent, t),
                mode: InterpolationMode::Interpolated { from_tick: from.tick, to_tick: to.tick, t },
            });
        }

        // Extrapolate from latest snapshot.
        let latest = self.buffer.latest().ok_or(SnapshotError::BufferEmpty)?;
        let dt = render_time - latest.server_time;

        if dt > self.config.max_extrapolation {
            // Still extrapolate but cap at max.
            let entity = latest.find_entity(entity_id).ok_or(SnapshotError::EntityNotFound {
                entity_id,
                tick: latest.tick,
            })?;
            return Ok(InterpolatedEntity {
                state: entity.extrapolate(self.config.max_extrapolation),
                mode: InterpolationMode::Extrapolated {
                    from_tick: latest.tick,
                    dt: self.config.max_extrapolation,
                },
            });
        }

        if dt >= 0.0 {
            let entity = latest.find_entity(entity_id).ok_or(SnapshotError::EntityNotFound {
                entity_id,
                tick: latest.tick,
            })?;
            return Ok(InterpolatedEntity {
                state: entity.extrapolate(dt),
                mode: InterpolationMode::Extrapolated { from_tick: latest.tick, dt },
            });
        }

        // Render time is before all snapshots.
        let earliest = self.buffer.earliest().unwrap();
        Err(SnapshotError::RenderTimeTooEarly {
            render_time,
            earliest: earliest.server_time,
        })
    }

    /// Interpolate all entities present in the latest snapshot.
    pub fn interpolate_all(&self) -> Vec<Result<InterpolatedEntity, SnapshotError>> {
        let Some(latest) = self.buffer.latest() else {
            return vec![Err(SnapshotError::BufferEmpty)];
        };
        latest.entities.iter().map(|e| self.interpolate_entity(e.entity_id)).collect()
    }

    pub fn buffered_snapshot_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn config(&self) -> &InterpolationConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u64, x: f64, y: f64) -> EntityState {
        EntityState::new(id, x, y, 0.0)
    }

    #[test]
    fn entity_lerp_midpoint() {
        let a = make_entity(1, 0.0, 0.0);
        let b = make_entity(1, 10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.pos_x - 5.0).abs() < 1e-9);
        assert!((mid.pos_y - 10.0).abs() < 1e-9);
    }

    #[test]
    fn entity_lerp_clamped() {
        let a = make_entity(1, 0.0, 0.0);
        let b = make_entity(1, 10.0, 0.0);
        let over = a.lerp(&b, 1.5);
        assert!((over.pos_x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn entity_extrapolate() {
        let e = make_entity(1, 0.0, 0.0).with_velocity(10.0, 5.0, 0.0);
        let ext = e.extrapolate(0.5);
        assert!((ext.pos_x - 5.0).abs() < 1e-9);
        assert!((ext.pos_y - 2.5).abs() < 1e-9);
    }

    #[test]
    fn snapshot_find_entity() {
        let snap = Snapshot::new(1, 0.0).with_entity(make_entity(42, 1.0, 2.0));
        assert!(snap.find_entity(42).is_some());
        assert!(snap.find_entity(99).is_none());
    }

    #[test]
    fn snapshot_delta_basic() {
        let s1 = Snapshot::new(1, 0.0).with_entity(make_entity(1, 0.0, 0.0));
        let s2 = Snapshot::new(2, 0.1).with_entity(make_entity(1, 5.0, 3.0));
        let delta = SnapshotDelta::compute(&s1, &s2);
        assert_eq!(delta.changed_count(), 1);
        assert!((delta.entity_deltas[0].dx - 5.0).abs() < 1e-9);
    }

    #[test]
    fn snapshot_delta_added_removed() {
        let s1 = Snapshot::new(1, 0.0).with_entity(make_entity(1, 0.0, 0.0));
        let s2 = Snapshot::new(2, 0.1).with_entity(make_entity(2, 1.0, 1.0));
        let delta = SnapshotDelta::compute(&s1, &s2);
        assert_eq!(delta.added_entities, vec![2]);
        assert_eq!(delta.removed_entities, vec![1]);
    }

    #[test]
    fn buffer_insert_order() {
        let mut buf = SnapshotBuffer::new(8);
        buf.insert(Snapshot::new(2, 0.2));
        buf.insert(Snapshot::new(1, 0.1));
        buf.insert(Snapshot::new(3, 0.3));
        assert_eq!(buf.earliest().unwrap().tick, 1);
        assert_eq!(buf.latest().unwrap().tick, 3);
    }

    #[test]
    fn buffer_capacity_eviction() {
        let mut buf = SnapshotBuffer::new(2);
        buf.insert(Snapshot::new(1, 0.1));
        buf.insert(Snapshot::new(2, 0.2));
        buf.insert(Snapshot::new(3, 0.3));
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.earliest().unwrap().tick, 2);
    }

    #[test]
    fn buffer_find_bracket() {
        let mut buf = SnapshotBuffer::new(8);
        buf.insert(Snapshot::new(1, 0.0));
        buf.insert(Snapshot::new(2, 1.0));
        let (a, b) = buf.find_bracket(0.5).unwrap();
        assert_eq!(a.tick, 1);
        assert_eq!(b.tick, 2);
    }

    #[test]
    fn interpolator_basic() {
        let config = InterpolationConfig::new().with_delay(0.0);
        let mut interp = SnapshotInterpolator::new(config);

        let s1 = Snapshot::new(1, 0.0).with_entity(make_entity(1, 0.0, 0.0));
        let s2 = Snapshot::new(2, 1.0).with_entity(make_entity(1, 10.0, 0.0));
        interp.receive_snapshot(s1);
        interp.receive_snapshot(s2);
        interp.set_time(0.5);

        let result = interp.interpolate_entity(1).unwrap();
        assert!((result.state.pos_x - 5.0).abs() < 1e-9);
        matches!(result.mode, InterpolationMode::Interpolated { .. });
    }

    #[test]
    fn interpolator_with_delay() {
        let config = InterpolationConfig::new().with_delay(0.1);
        let mut interp = SnapshotInterpolator::new(config);

        let s1 = Snapshot::new(1, 0.0).with_entity(make_entity(1, 0.0, 0.0));
        let s2 = Snapshot::new(2, 1.0).with_entity(make_entity(1, 10.0, 0.0));
        interp.receive_snapshot(s1);
        interp.receive_snapshot(s2);
        interp.set_time(0.6);

        let result = interp.interpolate_entity(1).unwrap();
        // render_time = 0.6 - 0.1 = 0.5 => lerp t=0.5
        assert!((result.state.pos_x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn interpolator_extrapolation() {
        let config = InterpolationConfig::new().with_delay(0.0).with_max_extrapolation(1.0);
        let mut interp = SnapshotInterpolator::new(config);

        let s1 = Snapshot::new(1, 0.0)
            .with_entity(make_entity(1, 0.0, 0.0).with_velocity(10.0, 0.0, 0.0));
        interp.receive_snapshot(s1);
        interp.set_time(0.5);

        let result = interp.interpolate_entity(1).unwrap();
        assert!((result.state.pos_x - 5.0).abs() < 1e-9);
        matches!(result.mode, InterpolationMode::Extrapolated { .. });
    }

    #[test]
    fn interpolator_empty_buffer_error() {
        let config = InterpolationConfig::new();
        let interp = SnapshotInterpolator::new(config);
        let err = interp.interpolate_entity(1).unwrap_err();
        assert_eq!(err, SnapshotError::BufferEmpty);
    }

    #[test]
    fn interpolator_entity_not_found() {
        let config = InterpolationConfig::new().with_delay(0.0);
        let mut interp = SnapshotInterpolator::new(config);
        let s1 = Snapshot::new(1, 0.0).with_entity(make_entity(1, 0.0, 0.0));
        let s2 = Snapshot::new(2, 1.0).with_entity(make_entity(1, 1.0, 0.0));
        interp.receive_snapshot(s1);
        interp.receive_snapshot(s2);
        interp.set_time(0.5);
        let err = interp.interpolate_entity(99).unwrap_err();
        matches!(err, SnapshotError::EntityNotFound { .. });
    }

    #[test]
    fn interpolate_all_entities() {
        let config = InterpolationConfig::new().with_delay(0.0);
        let mut interp = SnapshotInterpolator::new(config);

        let s1 = Snapshot::new(1, 0.0)
            .with_entity(make_entity(1, 0.0, 0.0))
            .with_entity(make_entity(2, 0.0, 0.0));
        let s2 = Snapshot::new(2, 1.0)
            .with_entity(make_entity(1, 10.0, 0.0))
            .with_entity(make_entity(2, 20.0, 0.0));
        interp.receive_snapshot(s1);
        interp.receive_snapshot(s2);
        interp.set_time(0.5);

        let results = interp.interpolate_all();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn entity_display() {
        let e = make_entity(7, 1.0, 2.0).with_rotation(3.14);
        let s = format!("{e}");
        assert!(s.contains("Entity(7"));
    }

    #[test]
    fn config_defaults() {
        let cfg = InterpolationConfig::default();
        assert!((cfg.interpolation_delay - 0.1).abs() < 1e-9);
        assert_eq!(cfg.buffer_capacity, 32);
    }

    #[test]
    fn snapshot_buffer_clear() {
        let mut buf = SnapshotBuffer::new(8);
        buf.insert(Snapshot::new(1, 0.0));
        buf.clear();
        assert!(buf.is_empty());
    }
}
