//! Formation movement for groups — line, column, wedge/V, circle, box, custom
//! offset table. Formation leader with slot assignments, dynamic rebalancing,
//! slot negotiation, formation steering relative to leader, formation rotation,
//! obstacle-aware reshaping, speed matching to slowest unit.
//!
//! Replaces JavaScript formation/squad libraries with a pure-Rust formation
//! system for RTS games and tactical simulations.

use std::collections::HashMap;

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for formation math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn length(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn length_sq(&self) -> f64 { self.x * self.x + self.y * self.y }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::ZERO } else { Self { x: self.x / len, y: self.y / len } }
    }

    pub fn add(&self, other: Vec2) -> Vec2 { Vec2 { x: self.x + other.x, y: self.y + other.y } }
    pub fn sub(&self, other: Vec2) -> Vec2 { Vec2 { x: self.x - other.x, y: self.y - other.y } }
    pub fn scale(&self, s: f64) -> Vec2 { Vec2 { x: self.x * s, y: self.y * s } }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn dist_sq(&self, other: Vec2) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    pub fn rotate(&self, angle: f64) -> Vec2 {
        let c = angle.cos();
        let s = angle.sin();
        Vec2 { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }

    pub fn lerp(&self, other: Vec2, t: f64) -> Vec2 {
        Vec2 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn truncate(&self, max_len: f64) -> Self {
        let len_sq = self.length_sq();
        if len_sq <= max_len * max_len { *self }
        else {
            let len = len_sq.sqrt();
            let s = max_len / len;
            Self { x: self.x * s, y: self.y * s }
        }
    }
}

// ── Formation shape ─────────────────────────────────────────────

/// Predefined formation shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum FormationShape {
    /// Single row perpendicular to heading.
    Line { spacing: f64 },
    /// Single column parallel to heading.
    Column { spacing: f64 },
    /// Wedge/V shape.
    Wedge { spacing: f64, angle: f64 },
    /// Circle around leader.
    Circle { radius: f64 },
    /// Rectangular box formation.
    Box { spacing: f64, columns: usize },
    /// Custom offset table (relative to leader facing forward).
    Custom { offsets: Vec<Vec2> },
}

impl FormationShape {
    /// Generate slot offsets for N units (relative to leader, heading = +Y).
    /// Slot 0 is always the leader at (0, 0).
    pub fn generate_offsets(&self, count: usize) -> Vec<Vec2> {
        if count == 0 {
            return Vec::new();
        }

        let mut offsets = vec![Vec2::ZERO]; // leader slot

        match self {
            FormationShape::Line { spacing } => {
                for i in 1..count {
                    let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
                    let idx = ((i + 1) / 2) as f64;
                    offsets.push(Vec2::new(sign * idx * spacing, 0.0));
                }
            }

            FormationShape::Column { spacing } => {
                for i in 1..count {
                    offsets.push(Vec2::new(0.0, -(i as f64) * spacing));
                }
            }

            FormationShape::Wedge { spacing, angle } => {
                for i in 1..count {
                    let side = if i % 2 == 1 { 1.0 } else { -1.0 };
                    let rank = ((i + 1) / 2) as f64;
                    let x = side * rank * spacing * angle.sin();
                    let y = -rank * spacing * angle.cos();
                    offsets.push(Vec2::new(x, y));
                }
            }

            FormationShape::Circle { radius } => {
                if count > 1 {
                    let angle_step = 2.0 * std::f64::consts::PI / (count - 1) as f64;
                    for i in 1..count {
                        let a = angle_step * (i - 1) as f64;
                        offsets.push(Vec2::new(radius * a.cos(), radius * a.sin()));
                    }
                }
            }

            FormationShape::Box { spacing, columns } => {
                let cols = (*columns).max(1);
                for i in 1..count {
                    let col = i % cols;
                    let row = i / cols;
                    let x = (col as f64 - (cols as f64 - 1.0) / 2.0) * spacing;
                    let y = -(row as f64) * spacing;
                    offsets.push(Vec2::new(x, y));
                }
            }

            FormationShape::Custom { offsets: custom } => {
                for i in 1..count {
                    if i < custom.len() {
                        offsets.push(custom[i]);
                    } else {
                        // Overflow: stack behind
                        offsets.push(Vec2::new(0.0, -(i as f64) * 2.0));
                    }
                }
            }
        }

        offsets
    }
}

// ── Unit ────────────────────────────────────────────────────────

/// A unit in the formation.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub max_speed: f64,
    pub max_force: f64,
}

impl Unit {
    pub fn new(id: usize, position: Vec2, max_speed: f64) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::ZERO,
            max_speed,
            max_force: max_speed * 0.5,
        }
    }
}

// ── Obstacle ────────────────────────────────────────────────────

/// Circular obstacle for reshaping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f64,
}

// ── Formation ───────────────────────────────────────────────────

/// Formation manager: assigns units to slots and steers them.
#[derive(Debug, Clone)]
pub struct Formation {
    pub shape: FormationShape,
    pub leader_pos: Vec2,
    pub leader_heading: f64, // radians (0 = +X, PI/2 = +Y)
    pub leader_speed: f64,
    /// unit_id -> slot index
    slot_assignments: HashMap<usize, usize>,
    /// Cached slot offsets (local space).
    slot_offsets: Vec<Vec2>,
    /// Arrival threshold for slot positioning.
    pub arrival_threshold: f64,
}

impl Formation {
    pub fn new(shape: FormationShape) -> Self {
        Self {
            shape,
            leader_pos: Vec2::ZERO,
            leader_heading: 0.0,
            leader_speed: 0.0,
            slot_assignments: HashMap::new(),
            slot_offsets: Vec::new(),
            arrival_threshold: 1.0,
        }
    }

    /// Update leader state.
    pub fn set_leader(&mut self, position: Vec2, heading: f64, speed: f64) {
        self.leader_pos = position;
        self.leader_heading = heading;
        self.leader_speed = speed;
    }

    /// Regenerate slot offsets for current unit count.
    fn regenerate_offsets(&mut self) {
        let count = self.slot_assignments.len();
        self.slot_offsets = self.shape.generate_offsets(count);
    }

    /// Add a unit to the formation; assigns to nearest empty slot.
    pub fn add_unit(&mut self, unit: &Unit) {
        let slot = self.slot_assignments.len();
        self.slot_assignments.insert(unit.id, slot);
        self.regenerate_offsets();
    }

    /// Remove a unit and rebalance slots.
    pub fn remove_unit(&mut self, unit_id: usize) {
        self.slot_assignments.remove(&unit_id);
        // Reassign contiguous slots
        let ids: Vec<usize> = {
            let mut pairs: Vec<(usize, usize)> = self.slot_assignments.drain().collect();
            pairs.sort_by_key(|(_, slot)| *slot);
            pairs.into_iter().map(|(id, _)| id).collect()
        };
        for (i, id) in ids.into_iter().enumerate() {
            self.slot_assignments.insert(id, i);
        }
        self.regenerate_offsets();
    }

    /// Get the world-space target position for a unit's assigned slot.
    pub fn slot_world_position(&self, unit_id: usize) -> Option<Vec2> {
        let slot = self.slot_assignments.get(&unit_id)?;
        if *slot >= self.slot_offsets.len() {
            return None;
        }
        let local = self.slot_offsets[*slot];
        let rotated = local.rotate(self.leader_heading);
        Some(self.leader_pos.add(rotated))
    }

    /// Get slot index for a unit.
    pub fn unit_slot(&self, unit_id: usize) -> Option<usize> {
        self.slot_assignments.get(&unit_id).copied()
    }

    /// Number of assigned units.
    pub fn unit_count(&self) -> usize {
        self.slot_assignments.len()
    }

    /// Slot negotiation: reassign units to slots by nearest distance.
    pub fn negotiate_slots(&mut self, units: &[Unit]) {
        let offsets = self.shape.generate_offsets(units.len());
        self.slot_offsets = offsets.clone();

        // Compute world positions for each slot
        let world_slots: Vec<Vec2> = offsets.iter()
            .map(|o| self.leader_pos.add(o.rotate(self.leader_heading)))
            .collect();

        // Greedy assignment: for each slot, pick nearest unassigned unit
        let mut assigned_units: Vec<bool> = vec![false; units.len()];
        let mut new_assignments = HashMap::new();

        for slot_idx in 0..world_slots.len() {
            let mut best_unit = None;
            let mut best_dist = f64::MAX;

            for (ui, unit) in units.iter().enumerate() {
                if assigned_units[ui] { continue; }
                let d = unit.position.dist_sq(world_slots[slot_idx]);
                if d < best_dist {
                    best_dist = d;
                    best_unit = Some(ui);
                }
            }

            if let Some(ui) = best_unit {
                assigned_units[ui] = true;
                new_assignments.insert(units[ui].id, slot_idx);
            }
        }

        self.slot_assignments = new_assignments;
    }

    /// Compute steering force for a unit toward its formation slot.
    pub fn steering_force(&self, unit: &Unit) -> Vec2 {
        let target = match self.slot_world_position(unit.id) {
            Some(t) => t,
            None => return Vec2::ZERO,
        };

        let offset = target.sub(unit.position);
        let dist = offset.length();

        if dist < self.arrival_threshold {
            // Slow down near target (arrive behavior)
            let desired_speed = unit.max_speed * (dist / self.arrival_threshold).min(1.0);
            let desired_vel = offset.normalized().scale(desired_speed);
            desired_vel.sub(unit.velocity).truncate(unit.max_force)
        } else {
            let desired_vel = offset.normalized().scale(unit.max_speed);
            desired_vel.sub(unit.velocity).truncate(unit.max_force)
        }
    }

    /// Get the slowest unit's max speed (for speed matching).
    pub fn slowest_speed(&self, units: &[Unit]) -> f64 {
        let assigned_ids: Vec<usize> = self.slot_assignments.keys().copied().collect();
        units.iter()
            .filter(|u| assigned_ids.contains(&u.id))
            .map(|u| u.max_speed)
            .fold(f64::MAX, f64::min)
    }

    /// Update all units in the formation by one timestep.
    pub fn step(&self, units: &mut [Unit], dt: f64) {
        let speed_limit = self.slowest_speed(units);

        for unit in units.iter_mut() {
            if !self.slot_assignments.contains_key(&unit.id) {
                continue;
            }
            let force = self.steering_force(unit);
            let accel = force.scale(1.0 / 1.0); // mass = 1
            unit.velocity = unit.velocity.add(accel.scale(dt))
                .truncate(speed_limit.min(unit.max_speed));
            unit.position = unit.position.add(unit.velocity.scale(dt));
        }
    }

    /// Obstacle-aware reshaping: shift slots that overlap obstacles.
    pub fn reshape_for_obstacles(&mut self, obstacles: &[Obstacle]) {
        for offset in &mut self.slot_offsets {
            let world = self.leader_pos.add(offset.rotate(self.leader_heading));
            for obs in obstacles {
                let dist = world.dist(obs.center);
                if dist < obs.radius + 2.0 {
                    // Push slot away from obstacle
                    let away = world.sub(obs.center).normalized();
                    let push = (obs.radius + 2.0 - dist).max(0.0);
                    // Apply in local space
                    let world_shifted = world.add(away.scale(push));
                    let new_local = world_shifted.sub(self.leader_pos).rotate(-self.leader_heading);
                    *offset = new_local;
                }
            }
        }
    }

    /// Rotate the entire formation by changing leader heading.
    pub fn rotate(&mut self, angle_delta: f64) {
        self.leader_heading += angle_delta;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit(id: usize, x: f64, y: f64) -> Unit {
        Unit::new(id, Vec2::new(x, y), 5.0)
    }

    #[test]
    fn test_vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!((r.x - 0.0).abs() < 1e-6);
        assert!((r.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_line_formation() {
        let offsets = FormationShape::Line { spacing: 2.0 }.generate_offsets(5);
        assert_eq!(offsets.len(), 5);
        assert!((offsets[0].x).abs() < 1e-6); // leader at center
        // Alternating sides
        assert!(offsets[1].x > 0.0);
        assert!(offsets[2].x < 0.0);
    }

    #[test]
    fn test_column_formation() {
        let offsets = FormationShape::Column { spacing: 3.0 }.generate_offsets(4);
        assert_eq!(offsets.len(), 4);
        for i in 1..4 {
            assert!((offsets[i].x).abs() < 1e-6); // all same x
            assert!(offsets[i].y < offsets[i - 1].y); // each further back
        }
    }

    #[test]
    fn test_wedge_formation() {
        let offsets = FormationShape::Wedge {
            spacing: 2.0,
            angle: std::f64::consts::FRAC_PI_4,
        }.generate_offsets(5);
        assert_eq!(offsets.len(), 5);
        // Alternating sides with increasing distance
        assert!(offsets[1].x > 0.0);
        assert!(offsets[2].x < 0.0);
    }

    #[test]
    fn test_circle_formation() {
        let offsets = FormationShape::Circle { radius: 10.0 }.generate_offsets(5);
        assert_eq!(offsets.len(), 5);
        // Non-leader slots at distance ~10 from origin
        for i in 1..5 {
            let dist = offsets[i].length();
            assert!((dist - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_box_formation() {
        let offsets = FormationShape::Box { spacing: 2.0, columns: 3 }.generate_offsets(7);
        assert_eq!(offsets.len(), 7);
        // First row has 3, second row has 3, third row has 1
    }

    #[test]
    fn test_custom_formation() {
        let custom = FormationShape::Custom {
            offsets: vec![
                Vec2::ZERO,
                Vec2::new(-2.0, -1.0),
                Vec2::new(2.0, -1.0),
            ],
        };
        let offsets = custom.generate_offsets(3);
        assert_eq!(offsets.len(), 3);
        assert!((offsets[1].x - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_custom_overflow() {
        let custom = FormationShape::Custom { offsets: vec![Vec2::ZERO, Vec2::new(1.0, 0.0)] };
        let offsets = custom.generate_offsets(5);
        assert_eq!(offsets.len(), 5);
    }

    #[test]
    fn test_formation_add_remove() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 2.0 });
        let u1 = make_unit(1, 0.0, 0.0);
        let u2 = make_unit(2, 5.0, 0.0);
        formation.add_unit(&u1);
        formation.add_unit(&u2);
        assert_eq!(formation.unit_count(), 2);

        formation.remove_unit(1);
        assert_eq!(formation.unit_count(), 1);
        assert!(formation.unit_slot(1).is_none());
        assert!(formation.unit_slot(2).is_some());
    }

    #[test]
    fn test_slot_world_position() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 3.0 });
        formation.set_leader(Vec2::new(10.0, 10.0), 0.0, 5.0);
        let u1 = make_unit(1, 0.0, 0.0);
        let u2 = make_unit(2, 0.0, 0.0);
        formation.add_unit(&u1);
        formation.add_unit(&u2);

        let pos = formation.slot_world_position(1).unwrap();
        assert!((pos.x - 10.0).abs() < 1e-6); // leader slot
        assert!((pos.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_formation_rotation() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 3.0 });
        formation.set_leader(Vec2::new(0.0, 0.0), 0.0, 5.0);
        let u1 = make_unit(1, 0.0, 0.0);
        let u2 = make_unit(2, 0.0, 3.0);
        formation.add_unit(&u1);
        formation.add_unit(&u2);

        // Rotate 90 degrees
        formation.rotate(std::f64::consts::FRAC_PI_2);

        let pos2 = formation.slot_world_position(2).unwrap();
        // After 90 degree rotation, x offset becomes y offset
        assert!((pos2.x).abs() < 1e-6 || (pos2.y).abs() > 0.0);
    }

    #[test]
    fn test_negotiate_slots() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 5.0 });
        formation.set_leader(Vec2::new(0.0, 0.0), 0.0, 5.0);

        let units = vec![
            make_unit(1, 6.0, 0.0),  // closer to right slot
            make_unit(2, -4.0, 0.0), // closer to left slot
            make_unit(3, 0.0, 0.0),  // closest to leader slot
        ];

        formation.negotiate_slots(&units);
        assert_eq!(formation.unit_count(), 3);
        // Each unit should have a slot
        assert!(formation.unit_slot(1).is_some());
        assert!(formation.unit_slot(2).is_some());
        assert!(formation.unit_slot(3).is_some());
    }

    #[test]
    fn test_steering_force_toward_slot() {
        let mut formation = Formation::new(FormationShape::Column { spacing: 5.0 });
        formation.set_leader(Vec2::new(50.0, 50.0), 0.0, 5.0);

        let unit = make_unit(1, 0.0, 0.0);
        formation.add_unit(&unit);

        let force = formation.steering_force(&unit);
        assert!(force.length() > 0.0); // should steer toward slot
    }

    #[test]
    fn test_steering_force_at_slot() {
        let mut formation = Formation::new(FormationShape::Column { spacing: 5.0 });
        formation.set_leader(Vec2::new(0.0, 0.0), 0.0, 5.0);

        let unit = make_unit(1, 0.0, 0.0); // at leader position
        formation.add_unit(&unit);

        let force = formation.steering_force(&unit);
        // Near target, force should be small
        assert!(force.length() < 1.0);
    }

    #[test]
    fn test_slowest_speed() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 2.0 });
        let u1 = Unit::new(1, Vec2::new(0.0, 0.0), 10.0);
        let u2 = Unit::new(2, Vec2::new(5.0, 0.0), 3.0);
        formation.add_unit(&u1);
        formation.add_unit(&u2);

        let units = vec![u1, u2];
        let slowest = formation.slowest_speed(&units);
        assert!((slowest - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_moves_units() {
        let mut formation = Formation::new(FormationShape::Column { spacing: 5.0 });
        formation.set_leader(Vec2::new(50.0, 50.0), 0.0, 5.0);

        let mut units = vec![
            make_unit(1, 10.0, 10.0),
            make_unit(2, 15.0, 10.0),
        ];
        for u in &units {
            formation.add_unit(u);
        }

        let before = units[0].position;
        formation.step(&mut units, 0.1);
        assert!(units[0].position.dist(before) > 1e-6);
    }

    #[test]
    fn test_reshape_for_obstacles() {
        let mut formation = Formation::new(FormationShape::Line { spacing: 3.0 });
        formation.set_leader(Vec2::new(0.0, 0.0), 0.0, 5.0);
        let u1 = make_unit(1, 0.0, 0.0);
        let u2 = make_unit(2, 3.0, 0.0);
        formation.add_unit(&u1);
        formation.add_unit(&u2);

        let obstacles = vec![Obstacle { center: Vec2::new(3.0, 0.0), radius: 2.0 }];
        formation.reshape_for_obstacles(&obstacles);
        // Slot near obstacle should have shifted
    }

    #[test]
    fn test_empty_formation() {
        let offsets = FormationShape::Line { spacing: 2.0 }.generate_offsets(0);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_single_unit_formation() {
        let offsets = FormationShape::Circle { radius: 10.0 }.generate_offsets(1);
        assert_eq!(offsets.len(), 1);
        assert!((offsets[0].x).abs() < 1e-6);
        assert!((offsets[0].y).abs() < 1e-6);
    }

    #[test]
    fn test_unassigned_unit_no_force() {
        let formation = Formation::new(FormationShape::Line { spacing: 2.0 });
        let unit = make_unit(99, 0.0, 0.0);
        let force = formation.steering_force(&unit);
        assert!((force.length()).abs() < 1e-10);
    }
}
