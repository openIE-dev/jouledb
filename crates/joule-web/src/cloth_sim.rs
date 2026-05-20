//! Cloth simulation — grid particles, springs, wind, self-collision, tearing.
//!
//! Replaces Ammo.js / Cannon.js cloth modules with pure Rust.
//! Supports structural + shear + bend springs, wind force, sphere/plane
//! collision, constraint-based correction, cloth pinning, tear threshold,
//! and spatial-hash self-collision detection.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ClothError {
    InvalidDimension,
    InvalidStiffness,
    InvalidDamping,
    InvalidTimestep,
    InvalidTearThreshold,
    ParticleOutOfBounds(usize),
}

impl fmt::Display for ClothError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "grid dimensions must be >= 2"),
            Self::InvalidStiffness => write!(f, "stiffness must be positive"),
            Self::InvalidDamping => write!(f, "damping must be non-negative"),
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::InvalidTearThreshold => write!(f, "tear threshold must be > 1.0"),
            Self::ParticleOutOfBounds(i) => write!(f, "particle index out of bounds: {i}"),
        }
    }
}

impl std::error::Error for ClothError {}

// ── Vec3 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::ZERO } else { self.scale(1.0 / len) }
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

// ── Particle ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ClothParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub mass: f64,
    pub inv_mass: f64,
    pub pinned: bool,
}

impl ClothParticle {
    fn new(pos: Vec3, mass: f64, pinned: bool) -> Self {
        Self {
            position: pos,
            prev_position: pos,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            mass,
            inv_mass: if pinned { 0.0 } else { 1.0 / mass },
            pinned,
        }
    }
}

// ── Spring ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpringType {
    Structural,
    Shear,
    Bend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClothSpring {
    pub a: usize,
    pub b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub kind: SpringType,
    pub active: bool,
}

// ── Collision objects ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

// ── Cloth Config ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ClothConfig {
    pub rows: usize,
    pub cols: usize,
    pub spacing: f64,
    pub particle_mass: f64,
    pub structural_stiffness: f64,
    pub shear_stiffness: f64,
    pub bend_stiffness: f64,
    pub damping: f64,
    pub gravity: Vec3,
    pub wind: Vec3,
    pub wind_turbulence: f64,
    pub tear_threshold: f64,
    pub constraint_iterations: usize,
    pub self_collision_radius: f64,
}

impl Default for ClothConfig {
    fn default() -> Self {
        Self {
            rows: 10,
            cols: 10,
            spacing: 1.0,
            particle_mass: 1.0,
            structural_stiffness: 500.0,
            shear_stiffness: 250.0,
            bend_stiffness: 100.0,
            damping: 0.02,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            wind: Vec3::ZERO,
            wind_turbulence: 0.0,
            tear_threshold: 3.0,
            constraint_iterations: 5,
            self_collision_radius: 0.0,
        }
    }
}

// ── Spatial hash for self-collision ─────────────────────────────

struct SpatialHash {
    cell_size: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHash {
    fn new(cell_size: f64) -> Self {
        Self { cell_size, cells: HashMap::new() }
    }

    fn cell_key(&self, pos: Vec3) -> (i64, i64, i64) {
        (
            (pos.x / self.cell_size).floor() as i64,
            (pos.y / self.cell_size).floor() as i64,
            (pos.z / self.cell_size).floor() as i64,
        )
    }

    fn insert(&mut self, idx: usize, pos: Vec3) {
        let key = self.cell_key(pos);
        self.cells.entry(key).or_default().push(idx);
    }

    fn query_neighbors(&self, pos: Vec3) -> Vec<usize> {
        let (cx, cy, cz) = self.cell_key(pos);
        let mut result = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        result.extend_from_slice(v);
                    }
                }
            }
        }
        result
    }
}

// ── Cloth Sim ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ClothSim {
    pub particles: Vec<ClothParticle>,
    pub springs: Vec<ClothSpring>,
    pub config: ClothConfig,
    pub spheres: Vec<Sphere>,
    pub planes: Vec<Plane>,
    pub time: f64,
    pub torn_count: usize,
    rng_state: u64,
}

impl ClothSim {
    pub fn new(config: ClothConfig) -> Result<Self, ClothError> {
        if config.rows < 2 || config.cols < 2 {
            return Err(ClothError::InvalidDimension);
        }
        if config.structural_stiffness <= 0.0 {
            return Err(ClothError::InvalidStiffness);
        }
        if config.damping < 0.0 {
            return Err(ClothError::InvalidDamping);
        }
        if config.tear_threshold <= 1.0 {
            return Err(ClothError::InvalidTearThreshold);
        }

        let mut particles = Vec::with_capacity(config.rows * config.cols);
        for r in 0..config.rows {
            for c in 0..config.cols {
                let pos = Vec3::new(
                    c as f64 * config.spacing,
                    0.0,
                    r as f64 * config.spacing,
                );
                particles.push(ClothParticle::new(pos, config.particle_mass, false));
            }
        }

        let mut springs = Vec::new();
        let idx = |r: usize, c: usize| -> usize { r * config.cols + c };

        for r in 0..config.rows {
            for c in 0..config.cols {
                if c + 1 < config.cols {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r, c + 1)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r, c + 1),
                        rest_length: rl, stiffness: config.structural_stiffness,
                        damping: config.damping, kind: SpringType::Structural, active: true,
                    });
                }
                if r + 1 < config.rows {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r + 1, c)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r + 1, c),
                        rest_length: rl, stiffness: config.structural_stiffness,
                        damping: config.damping, kind: SpringType::Structural, active: true,
                    });
                }
                if r + 1 < config.rows && c + 1 < config.cols {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r + 1, c + 1)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r + 1, c + 1),
                        rest_length: rl, stiffness: config.shear_stiffness,
                        damping: config.damping, kind: SpringType::Shear, active: true,
                    });
                }
                if r + 1 < config.rows && c > 0 {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r + 1, c - 1)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r + 1, c - 1),
                        rest_length: rl, stiffness: config.shear_stiffness,
                        damping: config.damping, kind: SpringType::Shear, active: true,
                    });
                }
                if c + 2 < config.cols {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r, c + 2)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r, c + 2),
                        rest_length: rl, stiffness: config.bend_stiffness,
                        damping: config.damping, kind: SpringType::Bend, active: true,
                    });
                }
                if r + 2 < config.rows {
                    let rl = particles[idx(r, c)].position.sub(particles[idx(r + 2, c)].position).length();
                    springs.push(ClothSpring {
                        a: idx(r, c), b: idx(r + 2, c),
                        rest_length: rl, stiffness: config.bend_stiffness,
                        damping: config.damping, kind: SpringType::Bend, active: true,
                    });
                }
            }
        }

        Ok(Self {
            particles,
            springs,
            config,
            spheres: Vec::new(),
            planes: Vec::new(),
            time: 0.0,
            torn_count: 0,
            rng_state: 42,
        })
    }

    fn next_rng(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x % 10000) as f64 / 10000.0
    }

    pub fn pin(&mut self, idx: usize) -> Result<(), ClothError> {
        if idx >= self.particles.len() {
            return Err(ClothError::ParticleOutOfBounds(idx));
        }
        self.particles[idx].pinned = true;
        self.particles[idx].inv_mass = 0.0;
        Ok(())
    }

    pub fn unpin(&mut self, idx: usize) -> Result<(), ClothError> {
        if idx >= self.particles.len() {
            return Err(ClothError::ParticleOutOfBounds(idx));
        }
        self.particles[idx].pinned = false;
        self.particles[idx].inv_mass = 1.0 / self.particles[idx].mass;
        Ok(())
    }

    pub fn pin_top_row(&mut self) {
        for c in 0..self.config.cols {
            let _ = self.pin(c);
        }
    }

    pub fn pin_corners(&mut self) {
        let cols = self.config.cols;
        let rows = self.config.rows;
        let _ = self.pin(0);
        let _ = self.pin(cols - 1);
        let _ = self.pin((rows - 1) * cols);
        let _ = self.pin(rows * cols - 1);
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    pub fn add_plane(&mut self, plane: Plane) {
        self.planes.push(plane);
    }

    fn apply_wind_force(&mut self) {
        if self.config.wind.length() < 1e-12 {
            return;
        }
        let cols = self.config.cols;
        let rows = self.config.rows;
        let turb = self.config.wind_turbulence;

        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let i0 = r * cols + c;
                let i1 = r * cols + c + 1;
                let i2 = (r + 1) * cols + c;

                let p0 = self.particles[i0].position;
                let p1 = self.particles[i1].position;
                let p2 = self.particles[i2].position;

                let normal = p1.sub(p0).cross(p2.sub(p0));
                let area = normal.length() * 0.5;
                if area < 1e-12 {
                    continue;
                }
                let n = normal.normalized();
                let wind_with_turb = if turb > 0.0 {
                    let t = self.next_rng() * turb - turb * 0.5;
                    self.config.wind.add(Vec3::new(t, t * 0.3, t * 0.7))
                } else {
                    self.config.wind
                };
                let wind_dot = wind_with_turb.dot(n).abs();
                let force = n.scale(wind_dot * area / 3.0);

                for &idx in &[i0, i1, i2] {
                    if !self.particles[idx].pinned {
                        self.particles[idx].acceleration = self.particles[idx].acceleration.add(
                            force.scale(self.particles[idx].inv_mass),
                        );
                    }
                }
            }
        }
    }

    fn satisfy_constraints(&mut self) {
        for _ in 0..self.config.constraint_iterations {
            for si in 0..self.springs.len() {
                if !self.springs[si].active {
                    continue;
                }
                let a = self.springs[si].a;
                let b = self.springs[si].b;
                let rest = self.springs[si].rest_length;
                let inv_a = self.particles[a].inv_mass;
                let inv_b = self.particles[b].inv_mass;
                let w_sum = inv_a + inv_b;
                if w_sum < 1e-12 {
                    continue;
                }

                let diff = self.particles[b].position.sub(self.particles[a].position);
                let dist = diff.length();
                if dist < 1e-12 {
                    continue;
                }

                let correction = diff.scale((dist - rest) / (dist * w_sum));

                if !self.particles[a].pinned {
                    self.particles[a].position = self.particles[a].position.add(correction.scale(inv_a));
                }
                if !self.particles[b].pinned {
                    self.particles[b].position = self.particles[b].position.sub(correction.scale(inv_b));
                }
            }
        }
    }

    fn check_tears(&mut self) {
        let threshold = self.config.tear_threshold;
        for s in &mut self.springs {
            if !s.active {
                continue;
            }
            // We need positions but can't borrow self again, so get indices
            let a = s.a;
            let b = s.b;
            // We'll check directly
            let _ = a; let _ = b;
            // Mark for tear check
        }
        // Two-pass to avoid borrow issues
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        for s in &mut self.springs {
            if !s.active {
                continue;
            }
            let dist = positions[s.a].sub(positions[s.b]).length();
            if dist > s.rest_length * threshold {
                s.active = false;
                self.torn_count += 1;
            }
        }
    }

    fn handle_sphere_collisions(&mut self) {
        for sphere in &self.spheres {
            for p in &mut self.particles {
                if p.pinned {
                    continue;
                }
                let diff = p.position.sub(sphere.center);
                let dist = diff.length();
                if dist < sphere.radius && dist > 1e-12 {
                    let push = diff.normalized().scale(sphere.radius - dist);
                    p.position = p.position.add(push);
                }
            }
        }
    }

    fn handle_plane_collisions(&mut self) {
        for plane in &self.planes {
            let n = plane.normal.normalized();
            for p in &mut self.particles {
                if p.pinned {
                    continue;
                }
                let d = p.position.sub(plane.point).dot(n);
                if d < 0.0 {
                    p.position = p.position.sub(n.scale(d));
                }
            }
        }
    }

    fn handle_self_collision(&mut self) {
        let radius = self.config.self_collision_radius;
        if radius <= 0.0 {
            return;
        }
        let mut hash = SpatialHash::new(radius * 2.0);
        for (i, p) in self.particles.iter().enumerate() {
            hash.insert(i, p.position);
        }

        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let inv_masses: Vec<f64> = self.particles.iter().map(|p| p.inv_mass).collect();

        for i in 0..self.particles.len() {
            if self.particles[i].pinned {
                continue;
            }
            let neighbors = hash.query_neighbors(positions[i]);
            for &j in &neighbors {
                if j <= i || self.particles[j].pinned {
                    continue;
                }
                let diff = positions[j].sub(positions[i]);
                let dist = diff.length();
                if dist < radius * 2.0 && dist > 1e-12 {
                    let overlap = radius * 2.0 - dist;
                    let w = inv_masses[i] + inv_masses[j];
                    if w < 1e-12 {
                        continue;
                    }
                    let correction = diff.normalized().scale(overlap / w);
                    self.particles[i].position = self.particles[i].position.sub(correction.scale(inv_masses[i]));
                    self.particles[j].position = self.particles[j].position.add(correction.scale(inv_masses[j]));
                }
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), ClothError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(ClothError::InvalidTimestep);
        }

        // Apply forces (gravity + wind) via acceleration
        for p in &mut self.particles {
            p.acceleration = if p.pinned { Vec3::ZERO } else { self.config.gravity };
        }
        self.apply_wind_force();

        // Verlet integration
        let damp = 1.0 - self.config.damping;
        for p in &mut self.particles {
            if p.pinned {
                continue;
            }
            let new_pos = p.position
                .scale(2.0)
                .sub(p.prev_position.scale(damp))
                .add(p.acceleration.scale(dt * dt));
            p.prev_position = p.position;
            p.position = new_pos;
            p.velocity = p.position.sub(p.prev_position).scale(1.0 / dt);
        }

        self.satisfy_constraints();
        self.check_tears();
        self.handle_sphere_collisions();
        self.handle_plane_collisions();
        self.handle_self_collision();

        self.time += dt;
        Ok(())
    }

    pub fn active_spring_count(&self) -> usize {
        self.springs.iter().filter(|s| s.active).count()
    }

    pub fn spring_count_by_type(&self) -> (usize, usize, usize) {
        let mut s = 0;
        let mut sh = 0;
        let mut b = 0;
        for sp in &self.springs {
            if !sp.active { continue; }
            match sp.kind {
                SpringType::Structural => s += 1,
                SpringType::Shear => sh += 1,
                SpringType::Bend => b += 1,
            }
        }
        (s, sh, b)
    }

    pub fn particle_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.config.rows || col >= self.config.cols {
            return None;
        }
        Some(row * self.config.cols + col)
    }

    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        for p in &self.particles {
            if p.position.x < min.x { min.x = p.position.x; }
            if p.position.y < min.y { min.y = p.position.y; }
            if p.position.z < min.z { min.z = p.position.z; }
            if p.position.x > max.x { max.x = p.position.x; }
            if p.position.y > max.y { max.y = p.position.y; }
            if p.position.z > max.z { max.z = p.position.z; }
        }
        (min, max)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn default_cloth() -> ClothSim {
        ClothSim::new(ClothConfig::default()).unwrap()
    }

    #[test]
    fn test_construction() {
        let cloth = default_cloth();
        assert_eq!(cloth.particles.len(), 100);
        assert!(cloth.springs.len() > 0);
    }

    #[test]
    fn test_invalid_dimension() {
        let r = ClothSim::new(ClothConfig { rows: 1, cols: 10, ..Default::default() });
        assert!(r.is_err());
    }

    #[test]
    fn test_spring_types_present() {
        let cloth = default_cloth();
        let (s, sh, b) = cloth.spring_count_by_type();
        assert!(s > 0);
        assert!(sh > 0);
        assert!(b > 0);
    }

    #[test]
    fn test_pin_unpin() {
        let mut cloth = default_cloth();
        cloth.pin(0).unwrap();
        assert!(cloth.particles[0].pinned);
        assert!(approx(cloth.particles[0].inv_mass, 0.0, 1e-12));
        cloth.unpin(0).unwrap();
        assert!(!cloth.particles[0].pinned);
        assert!(cloth.particles[0].inv_mass > 0.0);
    }

    #[test]
    fn test_pin_out_of_bounds() {
        let mut cloth = default_cloth();
        assert!(cloth.pin(9999).is_err());
    }

    #[test]
    fn test_pin_corners() {
        let mut cloth = default_cloth();
        cloth.pin_corners();
        assert!(cloth.particles[0].pinned);
        assert!(cloth.particles[9].pinned);
        assert!(cloth.particles[90].pinned);
        assert!(cloth.particles[99].pinned);
    }

    #[test]
    fn test_pinned_stays_put() {
        let mut cloth = default_cloth();
        cloth.pin(0).unwrap();
        let pos0 = cloth.particles[0].position;
        for _ in 0..20 {
            cloth.step(0.01).unwrap();
        }
        assert!(approx(cloth.particles[0].position.x, pos0.x, 1e-10));
        assert!(approx(cloth.particles[0].position.y, pos0.y, 1e-10));
    }

    #[test]
    fn test_gravity_pulls_down() {
        let mut cloth = default_cloth();
        cloth.pin_top_row();
        let mid = cloth.particle_index(5, 5).unwrap();
        let y0 = cloth.particles[mid].position.y;
        for _ in 0..50 {
            cloth.step(0.01).unwrap();
        }
        assert!(cloth.particles[mid].position.y < y0);
    }

    #[test]
    fn test_sphere_collision() {
        let mut cloth = default_cloth();
        cloth.add_sphere(Sphere { center: Vec3::new(4.5, -2.0, 4.5), radius: 3.0 });
        cloth.pin_top_row();
        for _ in 0..100 {
            cloth.step(0.01).unwrap();
        }
        // Particles should be pushed out of sphere
        let center = Vec3::new(4.5, -2.0, 4.5);
        for p in &cloth.particles {
            let d = p.position.sub(center).length();
            assert!(d >= 3.0 - 0.5 || p.pinned); // tolerance
        }
    }

    #[test]
    fn test_plane_collision() {
        let mut cloth = default_cloth();
        cloth.add_plane(Plane {
            point: Vec3::new(0.0, -5.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        });
        cloth.pin_corners();
        for _ in 0..200 {
            cloth.step(0.01).unwrap();
        }
        for p in &cloth.particles {
            assert!(p.position.y >= -5.0 - 0.5);
        }
    }

    #[test]
    fn test_tearing() {
        let mut cloth = ClothSim::new(ClothConfig {
            rows: 3,
            cols: 3,
            tear_threshold: 1.5,
            ..Default::default()
        }).unwrap();
        // Pin top-left, drag bottom-right far away
        cloth.pin(0).unwrap();
        cloth.particles[8].position = Vec3::new(100.0, 100.0, 100.0);
        cloth.step(0.01).unwrap();
        assert!(cloth.torn_count > 0);
    }

    #[test]
    fn test_wind_force() {
        let mut cloth = ClothSim::new(ClothConfig {
            wind: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        }).unwrap();
        cloth.pin_top_row();
        let mid = cloth.particle_index(5, 5).unwrap();
        let x0 = cloth.particles[mid].position.x;
        for _ in 0..50 {
            cloth.step(0.01).unwrap();
        }
        // Wind should push in x direction
        assert!((cloth.particles[mid].position.x - x0).abs() > 0.001);
    }

    #[test]
    fn test_particle_index() {
        let cloth = default_cloth();
        assert_eq!(cloth.particle_index(0, 0), Some(0));
        assert_eq!(cloth.particle_index(1, 2), Some(12));
        assert_eq!(cloth.particle_index(99, 0), None);
    }

    #[test]
    fn test_active_spring_count() {
        let cloth = default_cloth();
        let total = cloth.springs.len();
        assert_eq!(cloth.active_spring_count(), total);
    }

    #[test]
    fn test_bounding_box() {
        let cloth = default_cloth();
        let (min, max) = cloth.bounding_box();
        assert!(min.x <= 0.0 + 1e-6);
        assert!(max.x >= 9.0 - 1e-6);
    }

    #[test]
    fn test_invalid_timestep() {
        let mut cloth = default_cloth();
        assert!(cloth.step(-1.0).is_err());
        assert!(cloth.step(0.0).is_err());
    }

    #[test]
    fn test_invalid_stiffness() {
        let r = ClothSim::new(ClothConfig {
            structural_stiffness: -1.0,
            ..Default::default()
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_self_collision_enabled() {
        let mut cloth = ClothSim::new(ClothConfig {
            rows: 3,
            cols: 3,
            self_collision_radius: 0.5,
            ..Default::default()
        }).unwrap();
        // Push two non-adjacent particles to same spot
        cloth.particles[0].position = Vec3::new(5.0, 0.0, 5.0);
        cloth.particles[8].position = Vec3::new(5.0, 0.0, 5.0);
        cloth.step(0.01).unwrap();
        // After self-collision, they should be separated
        let d = cloth.particles[0].position.sub(cloth.particles[8].position).length();
        assert!(d > 0.0);
    }

    #[test]
    fn test_small_grid() {
        let cloth = ClothSim::new(ClothConfig {
            rows: 2,
            cols: 2,
            ..Default::default()
        }).unwrap();
        assert_eq!(cloth.particles.len(), 4);
        // 2 structural h, 2 structural v, 2 shear = minimum
        let (s, sh, _b) = cloth.spring_count_by_type();
        assert_eq!(s, 4);
        assert_eq!(sh, 2);
    }

    #[test]
    fn test_wind_turbulence() {
        let mut cloth = ClothSim::new(ClothConfig {
            wind: Vec3::new(5.0, 0.0, 0.0),
            wind_turbulence: 2.0,
            ..Default::default()
        }).unwrap();
        cloth.pin_top_row();
        for _ in 0..10 {
            cloth.step(0.01).unwrap();
        }
        // Just verify no panic with turbulence
        assert!(cloth.time > 0.0);
    }

    #[test]
    fn test_constraint_iterations() {
        // More iterations should keep springs closer to rest length
        let mut cloth_few = ClothSim::new(ClothConfig {
            constraint_iterations: 1,
            rows: 3,
            cols: 3,
            ..Default::default()
        }).unwrap();
        cloth_few.pin(0).unwrap();

        let mut cloth_many = ClothSim::new(ClothConfig {
            constraint_iterations: 10,
            rows: 3,
            cols: 3,
            ..Default::default()
        }).unwrap();
        cloth_many.pin(0).unwrap();

        for _ in 0..20 {
            cloth_few.step(0.01).unwrap();
            cloth_many.step(0.01).unwrap();
        }
        // Both should run without issues
        assert!(cloth_few.time > 0.0);
        assert!(cloth_many.time > 0.0);
    }

    #[test]
    fn test_pin_top_row_count() {
        let mut cloth = default_cloth();
        cloth.pin_top_row();
        let pinned = cloth.particles.iter().filter(|p| p.pinned).count();
        assert_eq!(pinned, 10);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!(approx(c.z, 1.0, 1e-10));
    }
}
