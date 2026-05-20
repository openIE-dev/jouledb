//! Lag compensation — server-side hit detection with historical entity rewind.
//!
//! Replaces custom lag compensation in Unreal/Source engine with a pure-Rust
//! system. Stores per-entity position history, rewinds to a past tick for
//! hit-scan validation, interpolates positions at fractional ticks, performs
//! bounding-box checks at historical positions, and enforces a maximum rewind
//! window.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Lag compensation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum LagCompError {
    /// Entity has no recorded history.
    NoHistory { entity_id: u64 },
    /// Requested tick is older than the earliest recorded.
    TickTooOld { entity_id: u64, requested: u64, oldest: u64 },
    /// Requested tick exceeds max rewind window.
    RewindExceeded { requested_ticks: u32, max: u32 },
    /// Entity not registered.
    EntityNotFound { entity_id: u64 },
}

impl fmt::Display for LagCompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoHistory { entity_id } => write!(f, "no history for entity {entity_id}"),
            Self::TickTooOld { entity_id, requested, oldest } => {
                write!(f, "entity {entity_id}: tick {requested} too old (oldest={oldest})")
            }
            Self::RewindExceeded { requested_ticks, max } => {
                write!(f, "rewind {requested_ticks} ticks exceeds max {max}")
            }
            Self::EntityNotFound { entity_id } => write!(f, "entity {entity_id} not found"),
        }
    }
}

impl std::error::Error for LagCompError {}

// ── Position Snapshot ───────────────────────────────────────────

/// A positional snapshot of an entity at a specific tick.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionSnapshot {
    pub tick: u64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl PositionSnapshot {
    pub fn new(tick: u64, x: f64, y: f64, z: f64) -> Self {
        Self { tick, x, y, z }
    }

    /// Lerp between two snapshots at fractional tick t.
    pub fn lerp(&self, other: &PositionSnapshot, t: f64) -> PositionSnapshot {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        PositionSnapshot {
            tick: self.tick,
            x: self.x * inv + other.x * t,
            y: self.y * inv + other.y * t,
            z: self.z * inv + other.z * t,
        }
    }

    /// Squared distance to a point.
    pub fn distance_sq_to(&self, px: f64, py: f64, pz: f64) -> f64 {
        let dx = self.x - px;
        let dy = self.y - py;
        let dz = self.z - pz;
        dx * dx + dy * dy + dz * dz
    }
}

impl fmt::Display for PositionSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pos(tick={}, ({:.2},{:.2},{:.2}))", self.tick, self.x, self.y, self.z)
    }
}

// ── Bounding Box ────────────────────────────────────────────────

/// Axis-aligned bounding box for hit detection.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    pub half_width: f64,
    pub half_height: f64,
    pub half_depth: f64,
}

impl BoundingBox {
    pub fn new(hw: f64, hh: f64, hd: f64) -> Self {
        Self { half_width: hw, half_height: hh, half_depth: hd }
    }

    pub fn cube(half: f64) -> Self {
        Self { half_width: half, half_height: half, half_depth: half }
    }

    /// Check if a point (px, py, pz) is inside this box centered at (cx, cy, cz).
    pub fn contains(&self, cx: f64, cy: f64, cz: f64, px: f64, py: f64, pz: f64) -> bool {
        (px - cx).abs() <= self.half_width
            && (py - cy).abs() <= self.half_height
            && (pz - cz).abs() <= self.half_depth
    }

    /// Check if a ray from origin in direction intersects this AABB centered at pos.
    /// Returns true if there is an intersection.
    pub fn ray_intersects(
        &self,
        cx: f64,
        cy: f64,
        cz: f64,
        ray_ox: f64,
        ray_oy: f64,
        ray_oz: f64,
        ray_dx: f64,
        ray_dy: f64,
        ray_dz: f64,
    ) -> bool {
        let inv_dx = if ray_dx.abs() > 1e-12 { 1.0 / ray_dx } else { f64::MAX };
        let inv_dy = if ray_dy.abs() > 1e-12 { 1.0 / ray_dy } else { f64::MAX };
        let inv_dz = if ray_dz.abs() > 1e-12 { 1.0 / ray_dz } else { f64::MAX };

        let t1 = ((cx - self.half_width) - ray_ox) * inv_dx;
        let t2 = ((cx + self.half_width) - ray_ox) * inv_dx;
        let t3 = ((cy - self.half_height) - ray_oy) * inv_dy;
        let t4 = ((cy + self.half_height) - ray_oy) * inv_dy;
        let t5 = ((cz - self.half_depth) - ray_oz) * inv_dz;
        let t6 = ((cz + self.half_depth) - ray_oz) * inv_dz;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        tmax >= 0.0 && tmin <= tmax
    }
}

// ── Entity History ──────────────────────────────────────────────

/// Position history for a single entity.
#[derive(Debug)]
pub struct EntityHistory {
    pub entity_id: u64,
    pub bounding_box: BoundingBox,
    snapshots: Vec<PositionSnapshot>,
    max_history: usize,
}

impl EntityHistory {
    pub fn new(entity_id: u64, bbox: BoundingBox, max_history: usize) -> Self {
        Self { entity_id, bounding_box: bbox, snapshots: Vec::with_capacity(max_history), max_history }
    }

    /// Record a position at a given tick.
    pub fn record(&mut self, snapshot: PositionSnapshot) {
        if self.snapshots.len() >= self.max_history {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    /// Get the position at an exact tick, or None.
    pub fn at_tick(&self, tick: u64) -> Option<&PositionSnapshot> {
        self.snapshots.iter().find(|s| s.tick == tick)
    }

    /// Get interpolated position at a fractional tick.
    pub fn at_fractional_tick(&self, tick: f64) -> Option<PositionSnapshot> {
        let floor_tick = tick.floor() as u64;
        let ceil_tick = tick.ceil() as u64;
        let frac = tick - tick.floor();

        if floor_tick == ceil_tick {
            return self.at_tick(floor_tick).cloned();
        }

        let a = self.at_tick(floor_tick)?;
        let b = self.at_tick(ceil_tick)?;
        Some(a.lerp(b, frac))
    }

    pub fn oldest_tick(&self) -> Option<u64> {
        self.snapshots.first().map(|s| s.tick)
    }

    pub fn newest_tick(&self) -> Option<u64> {
        self.snapshots.last().map(|s| s.tick)
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

// ── Hit Result ──────────────────────────────────────────────────

/// Result of a lag-compensated hit check.
#[derive(Debug, Clone, PartialEq)]
pub enum HitResult {
    /// Hit confirmed at the rewound position.
    Hit { entity_id: u64, tick: u64, distance: f64 },
    /// Shot missed the entity.
    Miss { entity_id: u64, tick: u64, closest_distance: f64 },
    /// Target tick was out of rewind range.
    OutOfRange { entity_id: u64, reason: String },
}

impl fmt::Display for HitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hit { entity_id, tick, distance } => {
                write!(f, "HIT entity {entity_id} at tick {tick} (dist={distance:.3})")
            }
            Self::Miss { entity_id, tick, closest_distance } => {
                write!(f, "MISS entity {entity_id} at tick {tick} (closest={closest_distance:.3})")
            }
            Self::OutOfRange { entity_id, reason } => {
                write!(f, "OUT_OF_RANGE entity {entity_id}: {reason}")
            }
        }
    }
}

// ── Lag Compensator Config ──────────────────────────────────────

/// Configuration for the lag compensator.
#[derive(Debug, Clone)]
pub struct LagCompConfig {
    pub max_rewind_ticks: u32,
    pub history_size: usize,
    pub current_tick: u64,
}

impl LagCompConfig {
    pub fn new() -> Self {
        Self { max_rewind_ticks: 30, history_size: 128, current_tick: 0 }
    }

    pub fn with_max_rewind(mut self, ticks: u32) -> Self {
        self.max_rewind_ticks = ticks;
        self
    }

    pub fn with_history(mut self, size: usize) -> Self {
        self.history_size = size;
        self
    }
}

impl Default for LagCompConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Lag Compensator ─────────────────────────────────────────────

/// Server-side lag compensator: rewinds entity positions for hit validation.
#[derive(Debug)]
pub struct LagCompensator {
    config: LagCompConfig,
    entities: HashMap<u64, EntityHistory>,
    current_tick: u64,
    total_checks: u64,
    total_hits: u64,
}

impl LagCompensator {
    pub fn new(config: LagCompConfig) -> Self {
        Self {
            current_tick: config.current_tick,
            config,
            entities: HashMap::new(),
            total_checks: 0,
            total_hits: 0,
        }
    }

    /// Register an entity with a bounding box.
    pub fn register_entity(&mut self, entity_id: u64, bbox: BoundingBox) {
        self.entities.insert(
            entity_id,
            EntityHistory::new(entity_id, bbox, self.config.history_size),
        );
    }

    /// Record a position for an entity at the current tick.
    pub fn record_position(&mut self, entity_id: u64, x: f64, y: f64, z: f64) -> Result<(), LagCompError> {
        let history = self.entities.get_mut(&entity_id).ok_or(LagCompError::EntityNotFound { entity_id })?;
        history.record(PositionSnapshot::new(self.current_tick, x, y, z));
        Ok(())
    }

    /// Advance to the next tick.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    pub fn set_tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Perform a point-based hit check at a historical tick.
    pub fn check_hit_point(
        &mut self,
        entity_id: u64,
        target_tick: u64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
    ) -> Result<HitResult, LagCompError> {
        self.total_checks += 1;

        let rewind = self.current_tick.saturating_sub(target_tick) as u32;
        if rewind > self.config.max_rewind_ticks {
            return Ok(HitResult::OutOfRange {
                entity_id,
                reason: format!("rewind {rewind} exceeds max {}", self.config.max_rewind_ticks),
            });
        }

        let history = self.entities.get(&entity_id).ok_or(LagCompError::EntityNotFound { entity_id })?;
        let pos = history.at_tick(target_tick).ok_or(LagCompError::TickTooOld {
            entity_id,
            requested: target_tick,
            oldest: history.oldest_tick().unwrap_or(0),
        })?;

        let hit = history.bounding_box.contains(pos.x, pos.y, pos.z, point_x, point_y, point_z);
        let dist = pos.distance_sq_to(point_x, point_y, point_z).sqrt();

        if hit {
            self.total_hits += 1;
            Ok(HitResult::Hit { entity_id, tick: target_tick, distance: dist })
        } else {
            Ok(HitResult::Miss { entity_id, tick: target_tick, closest_distance: dist })
        }
    }

    /// Perform a ray-based hit check at a historical tick.
    pub fn check_hit_ray(
        &mut self,
        entity_id: u64,
        target_tick: u64,
        ray_ox: f64,
        ray_oy: f64,
        ray_oz: f64,
        ray_dx: f64,
        ray_dy: f64,
        ray_dz: f64,
    ) -> Result<HitResult, LagCompError> {
        self.total_checks += 1;

        let rewind = self.current_tick.saturating_sub(target_tick) as u32;
        if rewind > self.config.max_rewind_ticks {
            return Ok(HitResult::OutOfRange {
                entity_id,
                reason: format!("rewind {rewind} exceeds max {}", self.config.max_rewind_ticks),
            });
        }

        let history = self.entities.get(&entity_id).ok_or(LagCompError::EntityNotFound { entity_id })?;
        let pos = history.at_tick(target_tick).ok_or(LagCompError::TickTooOld {
            entity_id,
            requested: target_tick,
            oldest: history.oldest_tick().unwrap_or(0),
        })?;

        let hit = history.bounding_box.ray_intersects(
            pos.x, pos.y, pos.z, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        );
        let dist = pos.distance_sq_to(ray_ox, ray_oy, ray_oz).sqrt();

        if hit {
            self.total_hits += 1;
            Ok(HitResult::Hit { entity_id, tick: target_tick, distance: dist })
        } else {
            Ok(HitResult::Miss { entity_id, tick: target_tick, closest_distance: dist })
        }
    }

    /// Perform a fractional-tick hit check (interpolated position).
    pub fn check_hit_fractional(
        &mut self,
        entity_id: u64,
        fractional_tick: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
    ) -> Result<HitResult, LagCompError> {
        self.total_checks += 1;

        let history = self.entities.get(&entity_id).ok_or(LagCompError::EntityNotFound { entity_id })?;
        let pos = history.at_fractional_tick(fractional_tick).ok_or(LagCompError::NoHistory { entity_id })?;
        let tick = fractional_tick.round() as u64;

        let hit = history.bounding_box.contains(pos.x, pos.y, pos.z, point_x, point_y, point_z);
        let dist = pos.distance_sq_to(point_x, point_y, point_z).sqrt();

        if hit {
            self.total_hits += 1;
            Ok(HitResult::Hit { entity_id, tick, distance: dist })
        } else {
            Ok(HitResult::Miss { entity_id, tick, closest_distance: dist })
        }
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    pub fn hit_rate(&self) -> f64 {
        if self.total_checks == 0 { 0.0 } else { self.total_hits as f64 / self.total_checks as f64 }
    }

    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_compensator() -> LagCompensator {
        let config = LagCompConfig::new().with_max_rewind(10);
        let mut comp = LagCompensator::new(config);
        comp.register_entity(1, BoundingBox::cube(1.0));
        comp
    }

    #[test]
    fn position_snapshot_lerp() {
        let a = PositionSnapshot::new(0, 0.0, 0.0, 0.0);
        let b = PositionSnapshot::new(1, 10.0, 20.0, 30.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-9);
        assert!((mid.y - 10.0).abs() < 1e-9);
        assert!((mid.z - 15.0).abs() < 1e-9);
    }

    #[test]
    fn position_snapshot_distance() {
        let p = PositionSnapshot::new(0, 3.0, 4.0, 0.0);
        let d = p.distance_sq_to(0.0, 0.0, 0.0).sqrt();
        assert!((d - 5.0).abs() < 1e-9);
    }

    #[test]
    fn bounding_box_contains() {
        let bb = BoundingBox::cube(1.0);
        assert!(bb.contains(5.0, 5.0, 5.0, 5.5, 5.5, 5.5));
        assert!(!bb.contains(5.0, 5.0, 5.0, 7.0, 5.0, 5.0));
    }

    #[test]
    fn bounding_box_ray_hit() {
        let bb = BoundingBox::cube(1.0);
        assert!(bb.ray_intersects(5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn bounding_box_ray_miss() {
        let bb = BoundingBox::cube(1.0);
        assert!(!bb.ray_intersects(5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn entity_history_record_and_retrieve() {
        let mut hist = EntityHistory::new(1, BoundingBox::cube(1.0), 64);
        hist.record(PositionSnapshot::new(0, 1.0, 2.0, 3.0));
        hist.record(PositionSnapshot::new(1, 4.0, 5.0, 6.0));
        assert_eq!(hist.len(), 2);
        assert!((hist.at_tick(0).unwrap().x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn entity_history_fractional_tick() {
        let mut hist = EntityHistory::new(1, BoundingBox::cube(1.0), 64);
        hist.record(PositionSnapshot::new(0, 0.0, 0.0, 0.0));
        hist.record(PositionSnapshot::new(1, 10.0, 0.0, 0.0));
        let pos = hist.at_fractional_tick(0.5).unwrap();
        assert!((pos.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn entity_history_eviction() {
        let mut hist = EntityHistory::new(1, BoundingBox::cube(1.0), 3);
        for i in 0..5 {
            hist.record(PositionSnapshot::new(i, i as f64, 0.0, 0.0));
        }
        assert_eq!(hist.len(), 3);
        assert_eq!(hist.oldest_tick(), Some(2));
    }

    #[test]
    fn compensator_register_and_record() {
        let mut comp = setup_compensator();
        comp.record_position(1, 5.0, 5.0, 5.0).unwrap();
        assert_eq!(comp.entity_count(), 1);
    }

    #[test]
    fn compensator_point_hit() {
        let mut comp = setup_compensator();
        comp.record_position(1, 5.0, 5.0, 5.0).unwrap();
        let result = comp.check_hit_point(1, 0, 5.5, 5.5, 5.5).unwrap();
        matches!(result, HitResult::Hit { .. });
    }

    #[test]
    fn compensator_point_miss() {
        let mut comp = setup_compensator();
        comp.record_position(1, 5.0, 5.0, 5.0).unwrap();
        let result = comp.check_hit_point(1, 0, 20.0, 20.0, 20.0).unwrap();
        matches!(result, HitResult::Miss { .. });
    }

    #[test]
    fn compensator_rewind_exceeded() {
        let mut comp = setup_compensator();
        comp.record_position(1, 0.0, 0.0, 0.0).unwrap();
        comp.set_tick(20);
        let result = comp.check_hit_point(1, 0, 0.0, 0.0, 0.0).unwrap();
        matches!(result, HitResult::OutOfRange { .. });
    }

    #[test]
    fn compensator_entity_not_found() {
        let mut comp = setup_compensator();
        let err = comp.check_hit_point(99, 0, 0.0, 0.0, 0.0).unwrap_err();
        assert_eq!(err, LagCompError::EntityNotFound { entity_id: 99 });
    }

    #[test]
    fn compensator_ray_hit() {
        let mut comp = setup_compensator();
        comp.record_position(1, 5.0, 0.0, 0.0).unwrap();
        let result = comp.check_hit_ray(1, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        matches!(result, HitResult::Hit { .. });
    }

    #[test]
    fn compensator_fractional_hit() {
        let mut comp = setup_compensator();
        comp.record_position(1, 0.0, 0.0, 0.0).unwrap();
        comp.advance_tick();
        comp.record_position(1, 10.0, 0.0, 0.0).unwrap();
        let result = comp.check_hit_fractional(1, 0.5, 5.0, 0.0, 0.0).unwrap();
        matches!(result, HitResult::Hit { .. });
    }

    #[test]
    fn compensator_hit_rate() {
        let mut comp = setup_compensator();
        comp.record_position(1, 5.0, 5.0, 5.0).unwrap();
        comp.check_hit_point(1, 0, 5.0, 5.0, 5.0).unwrap(); // hit
        comp.check_hit_point(1, 0, 50.0, 50.0, 50.0).unwrap(); // miss
        assert!((comp.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hit_result_display() {
        let hit = HitResult::Hit { entity_id: 1, tick: 5, distance: 0.5 };
        let s = format!("{hit}");
        assert!(s.contains("HIT"));
        assert!(s.contains("entity 1"));
    }

    #[test]
    fn position_snapshot_display() {
        let p = PositionSnapshot::new(42, 1.0, 2.0, 3.0);
        let s = format!("{p}");
        assert!(s.contains("tick=42"));
    }
}
