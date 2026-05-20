//! Advanced boid/flocking simulation — 3D, obstacles, predators, energy, species.
//!
//! Replaces three.js / p5.js / Processing boid/flocking libraries. Classic separation,
//! alignment, cohesion rules plus obstacle avoidance (SDF), predator evasion, food
//! seeking, leader following, grid-based spatial partitioning, 3D banking/pitching,
//! per-boid energy model, flock splitting/merging, multi-species, and trail history.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BoidError {
    InvalidParameter(String),
    NoBoids,
}

impl fmt::Display for BoidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoBoids => write!(f, "no boids in simulation"),
        }
    }
}

impl std::error::Error for BoidError {}

// ── Vec3 ───────────────────────────────────────────────────────

/// Simple 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn length(&self) -> f64 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }

    pub fn length_sq(&self) -> f64 { self.x * self.x + self.y * self.y + self.z * self.z }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn distance(&self, other: &Self) -> f64 { self.sub(other).length() }

    pub fn clamp_length(&self, max: f64) -> Self {
        let len = self.length();
        if len <= max { *self } else { self.normalize().scale(max) }
    }

    /// Banking angle: angle between velocity direction and up vector projected.
    pub fn banking_angle(&self, up: &Self) -> f64 {
        let proj = self.dot(up);
        let len = self.length();
        if len < 1e-12 { return 0.0; }
        (proj / len).asin()
    }
}

// ── Obstacle (SDF-based) ───────────────────────────────────────

/// An obstacle with a signed distance function.
#[derive(Debug, Clone)]
pub enum Obstacle {
    /// Sphere obstacle at position with radius.
    Sphere { center: Vec3, radius: f64 },
    /// Infinite plane with normal and offset (distance from origin).
    Plane { normal: Vec3, offset: f64 },
}

impl Obstacle {
    /// Signed distance from point to obstacle surface (negative = inside).
    pub fn sdf(&self, point: &Vec3) -> f64 {
        match self {
            Self::Sphere { center, radius } => point.distance(center) - radius,
            Self::Plane { normal, offset } => point.dot(&normal.normalize()) - offset,
        }
    }

    /// Gradient of the SDF (outward normal direction).
    pub fn gradient(&self, point: &Vec3) -> Vec3 {
        match self {
            Self::Sphere { center, .. } => point.sub(center).normalize(),
            Self::Plane { normal, .. } => normal.normalize(),
        }
    }
}

// ── Food source ────────────────────────────────────────────────

/// A food source in the environment.
#[derive(Debug, Clone)]
pub struct FoodSource {
    pub position: Vec3,
    pub energy: f64,
    pub max_energy: f64,
    pub regen_rate: f64,
}

impl FoodSource {
    pub fn new(position: Vec3, energy: f64, regen_rate: f64) -> Self {
        Self { position, energy, max_energy: energy, regen_rate }
    }

    /// Consume some energy from this source.
    pub fn consume(&mut self, amount: f64) -> f64 {
        let taken = amount.min(self.energy);
        self.energy -= taken;
        taken
    }

    /// Regenerate energy per timestep.
    pub fn regenerate(&mut self) {
        self.energy = (self.energy + self.regen_rate).min(self.max_energy);
    }
}

// ── Species parameters ─────────────────────────────────────────

/// Parameters for a boid species.
#[derive(Debug, Clone)]
pub struct SpeciesParams {
    pub name: String,
    pub max_speed: f64,
    pub max_force: f64,
    pub perception_radius: f64,
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
    pub obstacle_weight: f64,
    pub predator_weight: f64,
    pub food_weight: f64,
    pub energy_cost_per_speed: f64,
    pub eat_radius: f64,
}

impl Default for SpeciesParams {
    fn default() -> Self {
        Self {
            name: "default".into(),
            max_speed: 4.0,
            max_force: 0.2,
            perception_radius: 50.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            obstacle_weight: 3.0,
            predator_weight: 5.0,
            food_weight: 2.0,
            energy_cost_per_speed: 0.01,
            eat_radius: 10.0,
        }
    }
}

// ── Boid ───────────────────────────────────────────────────────

/// A single boid agent.
#[derive(Debug, Clone)]
pub struct Boid {
    pub id: u32,
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub species: u8,
    pub energy: f64,
    pub max_energy: f64,
    pub is_predator: bool,
    pub is_leader: bool,
    pub trail: Vec<Vec3>,
    pub max_trail: usize,
    pub flock_id: u32,
}

impl Boid {
    pub fn new(id: u32, position: Vec3, velocity: Vec3, species: u8) -> Self {
        Self {
            id,
            position,
            velocity,
            acceleration: Vec3::ZERO,
            species,
            energy: 100.0,
            max_energy: 100.0,
            is_predator: false,
            is_leader: false,
            trail: Vec::new(),
            max_trail: 50,
            flock_id: 0,
        }
    }

    pub fn speed(&self) -> f64 { self.velocity.length() }

    /// Pitch angle (angle from horizontal).
    pub fn pitch(&self) -> f64 {
        let len = self.velocity.length();
        if len < 1e-12 { return 0.0; }
        (self.velocity.z / len).asin()
    }

    /// Heading angle in XY plane (radians from +X).
    pub fn heading(&self) -> f64 {
        self.velocity.y.atan2(self.velocity.x)
    }
}

// ── Spatial Grid ───────────────────────────────────────────────

/// Grid-based spatial partitioning for neighbor lookup.
#[derive(Debug)]
struct SpatialGrid {
    cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(cell_size: f64) -> Self {
        Self { cell_size, cells: HashMap::new() }
    }

    fn clear(&mut self) {
        self.cells.clear();
    }

    fn cell_key(&self, pos: &Vec3) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    fn insert(&mut self, idx: usize, pos: &Vec3) {
        let key = self.cell_key(pos);
        self.cells.entry(key).or_default().push(idx);
    }

    fn query_neighbors(&self, pos: &Vec3, radius: f64) -> Vec<usize> {
        let r2 = radius * radius;
        let min_key = self.cell_key(&Vec3::new(pos.x - radius, pos.y - radius, pos.z - radius));
        let max_key = self.cell_key(&Vec3::new(pos.x + radius, pos.y + radius, pos.z + radius));

        let mut result = Vec::new();
        for cx in min_key.0..=max_key.0 {
            for cy in min_key.1..=max_key.1 {
                for cz in min_key.2..=max_key.2 {
                    if let Some(indices) = self.cells.get(&(cx, cy, cz)) {
                        for &idx in indices {
                            // We'll do actual distance check outside
                            let _ = r2; // marker for compiler
                            result.push(idx);
                        }
                    }
                }
            }
        }
        result
    }
}

// ── Simulation ─────────────────────────────────────────────────

/// Advanced boid simulation.
#[derive(Debug)]
pub struct BoidSimulation {
    pub boids: Vec<Boid>,
    pub species_params: Vec<SpeciesParams>,
    pub obstacles: Vec<Obstacle>,
    pub food_sources: Vec<FoodSource>,
    grid: SpatialGrid,
    step_count: u64,
    next_id: u32,
    next_flock_id: u32,
}

impl BoidSimulation {
    /// Create a new simulation with default species.
    pub fn new() -> Self {
        Self {
            boids: Vec::new(),
            species_params: vec![SpeciesParams::default()],
            obstacles: Vec::new(),
            food_sources: Vec::new(),
            grid: SpatialGrid::new(50.0),
            step_count: 0,
            next_id: 0,
            next_flock_id: 1,
        }
    }

    /// Add a species and return its index.
    pub fn add_species(&mut self, params: SpeciesParams) -> u8 {
        let idx = self.species_params.len() as u8;
        self.species_params.push(params);
        idx
    }

    /// Add a boid.
    pub fn add_boid(&mut self, position: Vec3, velocity: Vec3, species: u8) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.boids.push(Boid::new(id, position, velocity, species));
        id
    }

    /// Add a predator boid.
    pub fn add_predator(&mut self, position: Vec3, velocity: Vec3, species: u8) -> u32 {
        let id = self.add_boid(position, velocity, species);
        if let Some(b) = self.boids.iter_mut().find(|b| b.id == id) {
            b.is_predator = true;
        }
        id
    }

    /// Add a leader boid.
    pub fn add_leader(&mut self, position: Vec3, velocity: Vec3, species: u8) -> u32 {
        let id = self.add_boid(position, velocity, species);
        if let Some(b) = self.boids.iter_mut().find(|b| b.id == id) {
            b.is_leader = true;
        }
        id
    }

    pub fn boid_count(&self) -> usize { self.boids.len() }
    pub fn step_count(&self) -> u64 { self.step_count }

    /// Get species parameters, defaulting to index 0 if out of range.
    fn get_params(&self, species: u8) -> &SpeciesParams {
        self.species_params.get(species as usize).unwrap_or(&self.species_params[0])
    }

    /// Advance one timestep.
    pub fn step(&mut self) {
        // Build spatial grid
        self.grid.clear();
        for (i, boid) in self.boids.iter().enumerate() {
            self.grid.insert(i, &boid.position);
        }

        let n = self.boids.len();
        let mut accelerations = vec![Vec3::ZERO; n];

        // Compute forces for each boid
        for i in 0..n {
            let params = self.get_params(self.boids[i].species);
            let radius = params.perception_radius;
            let candidates = self.grid.query_neighbors(&self.boids[i].position, radius);

            let mut sep = Vec3::ZERO;
            let mut ali = Vec3::ZERO;
            let mut coh = Vec3::ZERO;
            let mut sep_count = 0u32;
            let mut ali_count = 0u32;
            let mut coh_count = 0u32;

            for &j in &candidates {
                if i == j { continue; }
                let dist = self.boids[i].position.distance(&self.boids[j].position);
                if dist > radius || dist < 1e-12 { continue; }

                // Predator evasion
                if self.boids[j].is_predator && !self.boids[i].is_predator {
                    let flee = self.boids[i].position.sub(&self.boids[j].position).normalize();
                    let w = params.predator_weight;
                    accelerations[i] = accelerations[i].add(&flee.scale(w / dist.max(1.0)));
                    continue;
                }

                // Only flock with same species
                if self.boids[j].species != self.boids[i].species { continue; }

                // Separation: only active within close range (half the perception radius)
                if dist < radius * 0.5 {
                    let away = self.boids[i].position.sub(&self.boids[j].position);
                    sep = sep.add(&away.scale(1.0 / dist.max(1.0)));
                    sep_count += 1;
                }

                // Alignment
                ali = ali.add(&self.boids[j].velocity);
                ali_count += 1;

                // Cohesion
                coh = coh.add(&self.boids[j].position);
                coh_count += 1;

                // Leader following
                if self.boids[j].is_leader {
                    let toward = self.boids[j].position.sub(&self.boids[i].position).normalize();
                    accelerations[i] = accelerations[i].add(&toward.scale(2.0));
                }
            }

            if sep_count > 0 {
                let force = sep.scale(1.0 / sep_count as f64).normalize().scale(params.max_speed)
                    .sub(&self.boids[i].velocity).clamp_length(params.max_force);
                accelerations[i] = accelerations[i].add(&force.scale(params.separation_weight));
            }
            if ali_count > 0 {
                let force = ali.scale(1.0 / ali_count as f64).normalize().scale(params.max_speed)
                    .sub(&self.boids[i].velocity).clamp_length(params.max_force);
                accelerations[i] = accelerations[i].add(&force.scale(params.alignment_weight));
            }
            if coh_count > 0 {
                let center = coh.scale(1.0 / coh_count as f64);
                let force = center.sub(&self.boids[i].position).normalize().scale(params.max_speed)
                    .sub(&self.boids[i].velocity).clamp_length(params.max_force);
                accelerations[i] = accelerations[i].add(&force.scale(params.cohesion_weight));
            }

            // Obstacle avoidance
            for obs in &self.obstacles {
                let dist = obs.sdf(&self.boids[i].position);
                if dist < radius * 0.5 {
                    let grad = obs.gradient(&self.boids[i].position);
                    let strength = params.obstacle_weight / dist.abs().max(0.1);
                    accelerations[i] = accelerations[i].add(&grad.scale(strength));
                }
            }

            // Food seeking
            if !self.boids[i].is_predator {
                let mut closest_food: Option<(usize, f64)> = None;
                for (fi, food) in self.food_sources.iter().enumerate() {
                    if food.energy < 1e-6 { continue; }
                    let d = self.boids[i].position.distance(&food.position);
                    if d < radius {
                        if closest_food.is_none() || d < closest_food.unwrap().1 {
                            closest_food = Some((fi, d));
                        }
                    }
                }
                if let Some((_, d)) = closest_food {
                    let fi = closest_food.unwrap().0;
                    let toward = self.food_sources[fi].position.sub(&self.boids[i].position).normalize();
                    accelerations[i] = accelerations[i].add(&toward.scale(params.food_weight / d.max(1.0)));
                }
            }
        }

        // Apply forces, update positions
        let species_params_snapshot = self.species_params.clone();
        for i in 0..n {
            let species = self.boids[i].species;
            let params = species_params_snapshot.get(species as usize)
                .unwrap_or(&species_params_snapshot[0]);

            // Record trail
            let max_trail = self.boids[i].max_trail;
            if self.boids[i].trail.len() >= max_trail {
                self.boids[i].trail.remove(0);
            }
            let position = self.boids[i].position;
            self.boids[i].trail.push(position);

            let velocity = self.boids[i].velocity;
            self.boids[i].velocity = velocity
                .add(&accelerations[i])
                .clamp_length(params.max_speed);
            let new_velocity = self.boids[i].velocity;
            self.boids[i].position = self.boids[i].position.add(&new_velocity);

            // Energy cost
            let speed = self.boids[i].velocity.length();
            self.boids[i].energy -= speed * params.energy_cost_per_speed;

            // Eating
            let boid_pos = self.boids[i].position;
            let eat_radius = params.eat_radius;
            let boid_energy = self.boids[i].energy;
            let boid_max_energy = self.boids[i].max_energy;
            let mut new_energy = boid_energy;
            for food in &mut self.food_sources {
                if boid_pos.distance(&food.position) < eat_radius {
                    let gained = food.consume(1.0);
                    new_energy = (new_energy + gained).min(boid_max_energy);
                }
            }
            self.boids[i].energy = new_energy;
        }

        // Food regeneration
        for food in &mut self.food_sources {
            food.regenerate();
        }

        // Assign flock IDs via simple connected-components on neighbor graph
        self.assign_flocks();

        self.step_count += 1;
    }

    /// Assign flock IDs based on proximity.
    fn assign_flocks(&mut self) {
        let n = self.boids.len();
        let mut flock_ids = vec![0u32; n];
        let mut current_flock = 0u32;
        let mut visited = vec![false; n];

        for start in 0..n {
            if visited[start] { continue; }
            current_flock += 1;
            let mut queue = vec![start];
            visited[start] = true;
            flock_ids[start] = current_flock;

            while let Some(idx) = queue.pop() {
                let params = self.get_params(self.boids[idx].species);
                let radius = params.perception_radius;
                for j in 0..n {
                    if visited[j] || j == idx { continue; }
                    if self.boids[j].species != self.boids[idx].species { continue; }
                    if self.boids[idx].position.distance(&self.boids[j].position) < radius {
                        visited[j] = true;
                        flock_ids[j] = current_flock;
                        queue.push(j);
                    }
                }
            }
        }

        for (i, fid) in flock_ids.into_iter().enumerate() {
            self.boids[i].flock_id = fid;
        }
    }

    /// Count distinct flocks.
    pub fn flock_count(&self) -> usize {
        let mut ids: Vec<u32> = self.boids.iter().map(|b| b.flock_id).collect();
        ids.sort();
        ids.dedup();
        ids.len()
    }

    /// Average position of all boids.
    pub fn center_of_mass(&self) -> Vec3 {
        if self.boids.is_empty() { return Vec3::ZERO; }
        let mut sum = Vec3::ZERO;
        for b in &self.boids {
            sum = sum.add(&b.position);
        }
        sum.scale(1.0 / self.boids.len() as f64)
    }

    /// Average speed.
    pub fn average_speed(&self) -> f64 {
        if self.boids.is_empty() { return 0.0; }
        let total: f64 = self.boids.iter().map(|b| b.speed()).sum();
        total / self.boids.len() as f64
    }

    /// Average energy.
    pub fn average_energy(&self) -> f64 {
        if self.boids.is_empty() { return 0.0; }
        let total: f64 = self.boids.iter().map(|b| b.energy).sum();
        total / self.boids.len() as f64
    }
}

impl fmt::Display for BoidSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoidSim(boids={}, flocks={}, step={})",
            self.boids.len(), self.flock_count(), self.step_count)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_vec3_basics() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.length(), 5.0));
        let n = v.normalize();
        assert!(approx(n.length(), 1.0));
    }

    #[test]
    fn test_vec3_zero_normalize() {
        let v = Vec3::ZERO;
        let n = v.normalize();
        assert!(approx(n.length(), 0.0));
    }

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a.add(&b);
        assert!(approx(sum.x, 5.0));
        assert!(approx(sum.y, 7.0));
        assert!(approx(sum.z, 9.0));
        assert!(approx(a.dot(&b), 32.0));
    }

    #[test]
    fn test_vec3_clamp_length() {
        let v = Vec3::new(10.0, 0.0, 0.0);
        let c = v.clamp_length(5.0);
        assert!(approx(c.length(), 5.0));
    }

    #[test]
    fn test_obstacle_sphere_sdf() {
        let obs = Obstacle::Sphere { center: Vec3::ZERO, radius: 5.0 };
        let outside = Vec3::new(10.0, 0.0, 0.0);
        assert!(obs.sdf(&outside) > 0.0);
        let inside = Vec3::new(2.0, 0.0, 0.0);
        assert!(obs.sdf(&inside) < 0.0);
    }

    #[test]
    fn test_obstacle_plane_sdf() {
        let obs = Obstacle::Plane { normal: Vec3::new(0.0, 1.0, 0.0), offset: 0.0 };
        let above = Vec3::new(0.0, 5.0, 0.0);
        assert!(obs.sdf(&above) > 0.0);
        let below = Vec3::new(0.0, -5.0, 0.0);
        assert!(obs.sdf(&below) < 0.0);
    }

    #[test]
    fn test_food_source() {
        let mut food = FoodSource::new(Vec3::ZERO, 10.0, 0.5);
        let taken = food.consume(3.0);
        assert!(approx(taken, 3.0));
        assert!(approx(food.energy, 7.0));
        food.regenerate();
        assert!(approx(food.energy, 7.5));
    }

    #[test]
    fn test_food_consume_clamp() {
        let mut food = FoodSource::new(Vec3::ZERO, 2.0, 0.0);
        let taken = food.consume(5.0);
        assert!(approx(taken, 2.0));
        assert!(approx(food.energy, 0.0));
    }

    #[test]
    fn test_simulation_creation() {
        let sim = BoidSimulation::new();
        assert_eq!(sim.boid_count(), 0);
        assert_eq!(sim.step_count(), 0);
    }

    #[test]
    fn test_add_boid() {
        let mut sim = BoidSimulation::new();
        let id = sim.add_boid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        assert_eq!(id, 0);
        assert_eq!(sim.boid_count(), 1);
    }

    #[test]
    fn test_add_predator() {
        let mut sim = BoidSimulation::new();
        sim.add_predator(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0);
        assert!(sim.boids[0].is_predator);
    }

    #[test]
    fn test_add_leader() {
        let mut sim = BoidSimulation::new();
        sim.add_leader(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0);
        assert!(sim.boids[0].is_leader);
    }

    #[test]
    fn test_step_moves_boids() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.step();
        assert!(sim.boids[0].position.x > 0.0);
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_separation_force() {
        let mut sim = BoidSimulation::new();
        // Two boids very close together — should separate
        sim.add_boid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.add_boid(Vec3::new(0.1, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.step();
        // They should have moved apart
        let d = sim.boids[0].position.distance(&sim.boids[1].position);
        assert!(d > 0.1);
    }

    #[test]
    fn test_cohesion_force() {
        let mut sim = BoidSimulation::new();
        // Boids spread out — should converge
        sim.add_boid(Vec3::new(-20.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0);
        sim.add_boid(Vec3::new(20.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0);
        let d0 = sim.boids[0].position.distance(&sim.boids[1].position);
        for _ in 0..10 {
            sim.step();
        }
        let d1 = sim.boids[0].position.distance(&sim.boids[1].position);
        assert!(d1 < d0, "boids should converge: d0={d0}, d1={d1}");
    }

    #[test]
    fn test_flock_assignment() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.add_boid(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.add_boid(Vec3::new(1000.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0);
        sim.step();
        // First two should be in same flock, third in different
        assert_eq!(sim.boids[0].flock_id, sim.boids[1].flock_id);
        assert_ne!(sim.boids[0].flock_id, sim.boids[2].flock_id);
    }

    #[test]
    fn test_flock_count() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 0);
        sim.add_boid(Vec3::new(1000.0, 0.0, 0.0), Vec3::ZERO, 0);
        sim.step();
        assert_eq!(sim.flock_count(), 2);
    }

    #[test]
    fn test_multi_species() {
        let mut sim = BoidSimulation::new();
        let sp1 = sim.add_species(SpeciesParams { name: "fast".into(), max_speed: 8.0, ..Default::default() });
        sim.add_boid(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0);
        sim.add_boid(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), sp1);
        sim.step();
        // Different species — they shouldn't be in the same flock
        assert_ne!(sim.boids[0].flock_id, sim.boids[1].flock_id);
    }

    #[test]
    fn test_energy_decreases() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), 0);
        let e0 = sim.boids[0].energy;
        sim.step();
        assert!(sim.boids[0].energy < e0);
    }

    #[test]
    fn test_trail_recorded() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0);
        sim.step();
        assert_eq!(sim.boids[0].trail.len(), 1);
        sim.step();
        assert_eq!(sim.boids[0].trail.len(), 2);
    }

    #[test]
    fn test_center_of_mass() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO, 0);
        sim.add_boid(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, 0);
        let com = sim.center_of_mass();
        assert!(approx(com.x, 0.0));
    }

    #[test]
    fn test_average_speed() {
        let mut sim = BoidSimulation::new();
        sim.add_boid(Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0), 0);
        assert!(approx(sim.average_speed(), 5.0));
    }

    #[test]
    fn test_boid_heading() {
        let b = Boid::new(0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0);
        assert!(approx(b.heading(), 0.0));
    }

    #[test]
    fn test_boid_pitch() {
        let b = Boid::new(0, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 0);
        assert!(approx(b.pitch(), std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn test_display() {
        let sim = BoidSimulation::new();
        let s = format!("{sim}");
        assert!(s.contains("BoidSim"));
    }
}
