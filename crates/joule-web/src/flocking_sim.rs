//! Boid flocking simulation — separation, alignment, cohesion, configurable
//! weights and radii, spatial grid neighbor lookup, obstacle avoidance,
//! boundary containment, leader following, predator evasion, visual range
//! with blind spot, energy cost per boid per step.
//!
//! Replaces JavaScript boid/flocking libraries with a pure-Rust flocking
//! simulation for games and visualizations.

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for flocking math.
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

    pub fn truncate(&self, max_len: f64) -> Self {
        let len_sq = self.length_sq();
        if len_sq <= max_len * max_len { *self }
        else {
            let len = len_sq.sqrt();
            let s = max_len / len;
            Self { x: self.x * s, y: self.y * s }
        }
    }

    pub fn add(&self, other: Vec2) -> Vec2 { Vec2 { x: self.x + other.x, y: self.y + other.y } }
    pub fn sub(&self, other: Vec2) -> Vec2 { Vec2 { x: self.x - other.x, y: self.y - other.y } }
    pub fn scale(&self, s: f64) -> Vec2 { Vec2 { x: self.x * s, y: self.y * s } }
    pub fn dot(&self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn dist_sq(&self, other: Vec2) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    pub fn angle(&self) -> f64 { self.y.atan2(self.x) }
}

// ── Boid ────────────────────────────────────────────────────────

/// A single boid agent.
#[derive(Debug, Clone, PartialEq)]
pub struct Boid {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub max_speed: f64,
    pub max_force: f64,
    pub energy: f64,
}

impl Boid {
    pub fn new(id: usize, position: Vec2, velocity: Vec2) -> Self {
        Self {
            id,
            position,
            velocity,
            max_speed: 4.0,
            max_force: 0.3,
            energy: 100.0,
        }
    }

    pub fn heading(&self) -> Vec2 { self.velocity.normalized() }

    pub fn speed(&self) -> f64 { self.velocity.length() }
}

// ── Obstacle ────────────────────────────────────────────────────

/// Circular obstacle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f64,
}

// ── Flocking config ─────────────────────────────────────────────

/// Configuration for flocking behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct FlockConfig {
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
    pub separation_radius: f64,
    pub alignment_radius: f64,
    pub cohesion_radius: f64,
    pub obstacle_avoidance_weight: f64,
    pub obstacle_detection_radius: f64,
    /// Visual range (max distance to perceive neighbors).
    pub visual_range: f64,
    /// Blind spot angle behind boid (radians, 0 = 360 vision).
    pub blind_spot_angle: f64,
    /// Boundary containment.
    pub boundary: Option<Boundary>,
    pub boundary_weight: f64,
    /// Leader to follow (position + velocity).
    pub leader: Option<(Vec2, Vec2)>,
    pub leader_weight: f64,
    pub leader_follow_dist: f64,
    /// Predator positions to evade.
    pub predators: Vec<Vec2>,
    pub predator_weight: f64,
    pub predator_radius: f64,
    /// Energy cost per unit speed per step.
    pub energy_cost_per_speed: f64,
}

impl Default for FlockConfig {
    fn default() -> Self {
        Self {
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            separation_radius: 25.0,
            alignment_radius: 50.0,
            cohesion_radius: 50.0,
            obstacle_avoidance_weight: 2.0,
            obstacle_detection_radius: 30.0,
            visual_range: 75.0,
            blind_spot_angle: 0.5, // ~29 degrees
            boundary: None,
            boundary_weight: 1.0,
            leader: None,
            leader_weight: 1.0,
            leader_follow_dist: 40.0,
            predators: Vec::new(),
            predator_weight: 3.0,
            predator_radius: 60.0,
            energy_cost_per_speed: 0.01,
        }
    }
}

/// Rectangular boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boundary {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

// ── Spatial grid ────────────────────────────────────────────────

/// Grid-based spatial index for O(1) neighbor lookup.
pub struct SpatialGrid {
    cell_size: f64,
    cells: std::collections::HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    fn cell_key(&self, pos: Vec2) -> (i32, i32) {
        ((pos.x / self.cell_size).floor() as i32,
         (pos.y / self.cell_size).floor() as i32)
    }

    /// Build the grid from boid positions.
    pub fn build(&mut self, boids: &[Boid]) {
        self.cells.clear();
        for (i, boid) in boids.iter().enumerate() {
            let key = self.cell_key(boid.position);
            self.cells.entry(key).or_default().push(i);
        }
    }

    /// Find potential neighbor indices within a radius.
    pub fn query_radius(&self, pos: Vec2, radius: f64) -> Vec<usize> {
        let cells_to_check = (radius / self.cell_size).ceil() as i32 + 1;
        let center = self.cell_key(pos);
        let mut result = Vec::new();

        for dx in -cells_to_check..=cells_to_check {
            for dy in -cells_to_check..=cells_to_check {
                let key = (center.0 + dx, center.1 + dy);
                if let Some(indices) = self.cells.get(&key) {
                    result.extend_from_slice(indices);
                }
            }
        }

        result
    }
}

// ── Visibility check ────────────────────────────────────────────

/// Check if a neighbor is visible (within visual range and not in blind spot).
fn is_visible(boid: &Boid, neighbor_pos: Vec2, config: &FlockConfig) -> bool {
    let dist_sq = boid.position.dist_sq(neighbor_pos);
    if dist_sq > config.visual_range * config.visual_range {
        return false;
    }
    if config.blind_spot_angle <= 0.0 {
        return true;
    }
    let to_neighbor = neighbor_pos.sub(boid.position).normalized();
    let heading = boid.heading();
    let dot = heading.dot(to_neighbor);
    // dot = cos(angle); blind spot is behind boid
    dot > -config.blind_spot_angle.cos()
}

// ── Core flocking forces ────────────────────────────────────────

/// Separation: steer away from nearby boids.
fn separation(boid: &Boid, neighbors: &[(usize, Vec2)], radius: f64) -> Vec2 {
    let mut steer = Vec2::ZERO;
    let mut count = 0;

    for (_, pos) in neighbors {
        let dist = boid.position.dist(*pos);
        if dist > 1e-10 && dist < radius {
            let away = boid.position.sub(*pos).normalized().scale(1.0 / dist);
            steer = steer.add(away);
            count += 1;
        }
    }

    if count > 0 {
        steer = steer.scale(1.0 / count as f64);
    }
    steer
}

/// Alignment: match average heading of neighbors.
fn alignment(boid: &Boid, neighbors: &[(usize, Vec2, Vec2)], radius: f64) -> Vec2 {
    let mut avg_vel = Vec2::ZERO;
    let mut count = 0;

    for (_, pos, vel) in neighbors {
        let dist = boid.position.dist(*pos);
        if dist > 1e-10 && dist < radius {
            avg_vel = avg_vel.add(*vel);
            count += 1;
        }
    }

    if count > 0 {
        avg_vel = avg_vel.scale(1.0 / count as f64);
        let desired = avg_vel.normalized().scale(boid.max_speed);
        desired.sub(boid.velocity)
    } else {
        Vec2::ZERO
    }
}

/// Cohesion: steer toward center of mass of neighbors.
fn cohesion(boid: &Boid, neighbors: &[(usize, Vec2)], radius: f64) -> Vec2 {
    let mut center = Vec2::ZERO;
    let mut count = 0;

    for (_, pos) in neighbors {
        let dist = boid.position.dist(*pos);
        if dist > 1e-10 && dist < radius {
            center = center.add(*pos);
            count += 1;
        }
    }

    if count > 0 {
        center = center.scale(1.0 / count as f64);
        let desired = center.sub(boid.position).normalized().scale(boid.max_speed);
        desired.sub(boid.velocity)
    } else {
        Vec2::ZERO
    }
}

/// Obstacle avoidance force.
fn avoid_obstacles(boid: &Boid, obstacles: &[Obstacle], detect_radius: f64) -> Vec2 {
    let mut steer = Vec2::ZERO;
    for obs in obstacles {
        let dist = boid.position.dist(obs.center) - obs.radius;
        if dist < detect_radius && dist > 1e-10 {
            let away = boid.position.sub(obs.center).normalized();
            steer = steer.add(away.scale(1.0 / dist));
        }
    }
    steer
}

/// Boundary containment force.
fn contain(boid: &Boid, boundary: &Boundary) -> Vec2 {
    let margin = 20.0;
    let mut steer = Vec2::ZERO;

    if boid.position.x < boundary.min_x + margin {
        steer.x += (boundary.min_x + margin - boid.position.x) / margin;
    }
    if boid.position.x > boundary.max_x - margin {
        steer.x -= (boid.position.x - (boundary.max_x - margin)) / margin;
    }
    if boid.position.y < boundary.min_y + margin {
        steer.y += (boundary.min_y + margin - boid.position.y) / margin;
    }
    if boid.position.y > boundary.max_y - margin {
        steer.y -= (boid.position.y - (boundary.max_y - margin)) / margin;
    }

    steer
}

/// Leader following force.
fn follow_leader(boid: &Boid, leader_pos: Vec2, leader_vel: Vec2, follow_dist: f64) -> Vec2 {
    let behind = leader_pos.sub(leader_vel.normalized().scale(follow_dist));
    let desired = behind.sub(boid.position).normalized().scale(boid.max_speed);
    desired.sub(boid.velocity)
}

/// Predator evasion force.
fn evade_predators(boid: &Boid, predators: &[Vec2], radius: f64) -> Vec2 {
    let mut steer = Vec2::ZERO;
    for pred in predators {
        let dist = boid.position.dist(*pred);
        if dist < radius && dist > 1e-10 {
            let away = boid.position.sub(*pred).normalized().scale(1.0 / dist);
            steer = steer.add(away);
        }
    }
    steer
}

// ── Simulation step ─────────────────────────────────────────────

/// Run one simulation step for the flock.
pub fn flock_step(
    boids: &mut Vec<Boid>,
    config: &FlockConfig,
    obstacles: &[Obstacle],
    dt: f64,
) -> f64 {
    let mut grid = SpatialGrid::new(config.visual_range.max(1.0));
    grid.build(boids);

    // Snapshot positions and velocities for neighbor queries
    let snapshot: Vec<(Vec2, Vec2)> = boids.iter()
        .map(|b| (b.position, b.velocity))
        .collect();

    let mut total_energy = 0.0;

    for i in 0..boids.len() {
        let boid_pos = snapshot[i].0;
        let boid_vel = snapshot[i].1;

        // Find visible neighbors
        let candidates = grid.query_radius(boid_pos, config.visual_range);

        let mut sep_neighbors: Vec<(usize, Vec2)> = Vec::new();
        let mut align_neighbors: Vec<(usize, Vec2, Vec2)> = Vec::new();
        let mut coh_neighbors: Vec<(usize, Vec2)> = Vec::new();

        for &j in &candidates {
            if j == i { continue; }
            if !is_visible(&boids[i], snapshot[j].0, config) { continue; }
            sep_neighbors.push((j, snapshot[j].0));
            align_neighbors.push((j, snapshot[j].0, snapshot[j].1));
            coh_neighbors.push((j, snapshot[j].0));
        }

        let sep = separation(&boids[i], &sep_neighbors, config.separation_radius);
        let ali = alignment(&boids[i], &align_neighbors, config.alignment_radius);
        let coh = cohesion(&boids[i], &coh_neighbors, config.cohesion_radius);
        let obs = avoid_obstacles(&boids[i], obstacles, config.obstacle_detection_radius);

        let mut force = Vec2::ZERO;
        force = force.add(sep.scale(config.separation_weight));
        force = force.add(ali.scale(config.alignment_weight));
        force = force.add(coh.scale(config.cohesion_weight));
        force = force.add(obs.scale(config.obstacle_avoidance_weight));

        if let Some(boundary) = &config.boundary {
            force = force.add(contain(&boids[i], boundary).scale(config.boundary_weight));
        }

        if let Some((lpos, lvel)) = config.leader {
            let lf = follow_leader(&boids[i], lpos, lvel, config.leader_follow_dist);
            force = force.add(lf.scale(config.leader_weight));
        }

        if !config.predators.is_empty() {
            let ef = evade_predators(&boids[i], &config.predators, config.predator_radius);
            force = force.add(ef.scale(config.predator_weight));
        }

        let clamped = force.truncate(boids[i].max_force);
        let new_vel = boid_vel.add(clamped.scale(dt)).truncate(boids[i].max_speed);
        let new_pos = boid_pos.add(new_vel.scale(dt));

        let energy_used = new_vel.length() * config.energy_cost_per_speed * dt;
        total_energy += energy_used;

        boids[i].velocity = new_vel;
        boids[i].position = new_pos;
        boids[i].energy -= energy_used;
    }

    total_energy
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_boid(id: usize, x: f64, y: f64, vx: f64, vy: f64) -> Boid {
        Boid::new(id, Vec2::new(x, y), Vec2::new(vx, vy))
    }

    #[test]
    fn test_vec2_dist() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dist(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!((v.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_boid_heading() {
        let b = make_boid(0, 0.0, 0.0, 3.0, 4.0);
        let h = b.heading();
        assert!((h.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_grid_build() {
        let boids = vec![
            make_boid(0, 10.0, 10.0, 1.0, 0.0),
            make_boid(1, 15.0, 10.0, 1.0, 0.0),
            make_boid(2, 100.0, 100.0, 1.0, 0.0),
        ];
        let mut grid = SpatialGrid::new(50.0);
        grid.build(&boids);
        let near = grid.query_radius(Vec2::new(10.0, 10.0), 20.0);
        assert!(near.contains(&0));
        assert!(near.contains(&1));
    }

    #[test]
    fn test_separation_force() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let neighbors = vec![(1, Vec2::new(5.0, 0.0))];
        let force = separation(&boid, &neighbors, 10.0);
        assert!(force.x < 0.0); // pushes away
    }

    #[test]
    fn test_separation_no_neighbors() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let force = separation(&boid, &[], 10.0);
        assert!((force.length()).abs() < 1e-10);
    }

    #[test]
    fn test_cohesion_toward_center() {
        let boid = make_boid(0, 0.0, 0.0, 0.0, 0.0);
        let neighbors = vec![(1, Vec2::new(10.0, 0.0)), (2, Vec2::new(10.0, 10.0))];
        let force = cohesion(&boid, &neighbors, 50.0);
        assert!(force.x > 0.0); // toward neighbors
    }

    #[test]
    fn test_alignment_matches_direction() {
        let boid = make_boid(0, 0.0, 0.0, 0.0, 0.0);
        let neighbors = vec![
            (1, Vec2::new(5.0, 0.0), Vec2::new(1.0, 0.0)),
            (2, Vec2::new(5.0, 5.0), Vec2::new(1.0, 0.0)),
        ];
        let force = alignment(&boid, &neighbors, 50.0);
        assert!(force.x > 0.0); // tries to match rightward velocity
    }

    #[test]
    fn test_visibility_front() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let config = FlockConfig { visual_range: 100.0, blind_spot_angle: 0.5, ..Default::default() };
        assert!(is_visible(&boid, Vec2::new(10.0, 0.0), &config));
    }

    #[test]
    fn test_visibility_behind_blind_spot() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let config = FlockConfig { visual_range: 100.0, blind_spot_angle: 1.5, ..Default::default() };
        // Behind the boid
        assert!(!is_visible(&boid, Vec2::new(-10.0, 0.0), &config));
    }

    #[test]
    fn test_visibility_out_of_range() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let config = FlockConfig { visual_range: 5.0, ..Default::default() };
        assert!(!is_visible(&boid, Vec2::new(100.0, 0.0), &config));
    }

    #[test]
    fn test_obstacle_avoidance() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let obstacles = vec![Obstacle { center: Vec2::new(10.0, 0.0), radius: 5.0 }];
        let force = avoid_obstacles(&boid, &obstacles, 20.0);
        assert!(force.x < 0.0); // pushes away
    }

    #[test]
    fn test_boundary_containment() {
        let boid = make_boid(0, 5.0, 50.0, 0.0, 0.0);
        let boundary = Boundary { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
        let force = contain(&boid, &boundary);
        assert!(force.x > 0.0); // pushes away from left boundary
    }

    #[test]
    fn test_leader_following() {
        let boid = make_boid(0, 0.0, 0.0, 0.0, 0.0);
        let force = follow_leader(&boid, Vec2::new(50.0, 0.0), Vec2::new(1.0, 0.0), 10.0);
        assert!(force.x > 0.0); // follows leader
    }

    #[test]
    fn test_predator_evasion() {
        let boid = make_boid(0, 0.0, 0.0, 1.0, 0.0);
        let force = evade_predators(&boid, &[Vec2::new(10.0, 0.0)], 20.0);
        assert!(force.x < 0.0); // evades
    }

    #[test]
    fn test_flock_step_basic() {
        let mut boids = vec![
            make_boid(0, 50.0, 50.0, 1.0, 0.0),
            make_boid(1, 55.0, 50.0, 1.0, 0.0),
            make_boid(2, 50.0, 55.0, 1.0, 0.0),
        ];
        let config = FlockConfig::default();
        let energy = flock_step(&mut boids, &config, &[], 0.1);
        assert!(energy >= 0.0);
        // Boids should have moved
        assert!((boids[0].position.x - 50.0).abs() > 1e-6
             || (boids[0].position.y - 50.0).abs() > 1e-6);
    }

    #[test]
    fn test_energy_decreases() {
        let mut boids = vec![make_boid(0, 0.0, 0.0, 3.0, 0.0)];
        let config = FlockConfig { energy_cost_per_speed: 0.1, ..Default::default() };
        let initial_energy = boids[0].energy;
        flock_step(&mut boids, &config, &[], 1.0);
        assert!(boids[0].energy < initial_energy);
    }

    #[test]
    fn test_flock_with_boundary() {
        let mut boids = vec![make_boid(0, 5.0, 5.0, -2.0, -2.0)];
        let config = FlockConfig {
            boundary: Some(Boundary { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }),
            boundary_weight: 5.0,
            ..Default::default()
        };
        flock_step(&mut boids, &config, &[], 0.5);
        // Boid near edge should be pushed inward
    }

    #[test]
    fn test_default_config() {
        let config = FlockConfig::default();
        assert!(config.separation_weight > 0.0);
        assert!(config.alignment_weight > 0.0);
        assert!(config.cohesion_weight > 0.0);
    }

    #[test]
    fn test_empty_flock() {
        let mut boids: Vec<Boid> = Vec::new();
        let config = FlockConfig::default();
        let energy = flock_step(&mut boids, &config, &[], 0.1);
        assert!((energy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_boid() {
        let mut boids = vec![make_boid(0, 50.0, 50.0, 2.0, 0.0)];
        let config = FlockConfig::default();
        flock_step(&mut boids, &config, &[], 0.1);
        // No neighbors, should just move forward
        assert!(boids[0].position.x > 50.0);
    }
}
