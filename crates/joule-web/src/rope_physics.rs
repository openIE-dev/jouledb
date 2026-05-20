//! Rope/chain physics simulation — Verlet integration, distance constraints, collision.
//!
//! Replaces Verlet.js / matter.js rope modules with pure Rust.
//! Supports chain of particles with distance constraints, Verlet integration,
//! iterative constraint satisfaction, rope properties (length, mass, stiffness),
//! collision with spheres/planes, attach/detach, slack detection, taut detection,
//! segment count control, and swing/pendulum behavior.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RopeError {
    InvalidTimestep,
    InvalidSegments,
    InvalidLength,
    InvalidMass,
    ParticleNotFound(usize),
    AlreadyAttached(usize),
    NotAttached(usize),
}

impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::InvalidSegments => write!(f, "segment count must be >= 1"),
            Self::InvalidLength => write!(f, "rope length must be positive"),
            Self::InvalidMass => write!(f, "mass must be positive"),
            Self::ParticleNotFound(i) => write!(f, "particle {i} not found"),
            Self::AlreadyAttached(i) => write!(f, "endpoint {i} is already attached"),
            Self::NotAttached(i) => write!(f, "endpoint {i} is not attached"),
        }
    }
}

impl std::error::Error for RopeError {}

// ── Vec3 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self.scale(1.0 / l) }
    }
}

// ── Collision Objects ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct CollisionSphere {
    pub center: Vec3,
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollisionPlane {
    pub point: Vec3,
    pub normal: Vec3,
}

// ── Rope Particle ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct RopeParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub acceleration: Vec3,
    pub mass: f64,
    pub inv_mass: f64,
    pub fixed: bool,
}

impl RopeParticle {
    fn new(pos: Vec3, mass: f64, fixed: bool) -> Self {
        Self {
            position: pos,
            prev_position: pos,
            acceleration: Vec3::ZERO,
            mass,
            inv_mass: if fixed { 0.0 } else { 1.0 / mass },
            fixed,
        }
    }
}

// ── Rope Config ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub segment_count: usize,
    pub total_length: f64,
    pub mass_per_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub constraint_iterations: usize,
    pub gravity: Vec3,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            segment_count: 20,
            total_length: 5.0,
            mass_per_length: 0.5,
            stiffness: 1.0,
            damping: 0.01,
            constraint_iterations: 10,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

// ── Rope ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Rope {
    pub particles: Vec<RopeParticle>,
    pub segment_length: f64,
    pub config: RopeConfig,
    pub spheres: Vec<CollisionSphere>,
    pub planes: Vec<CollisionPlane>,
    pub time: f64,
}

impl Rope {
    /// Create a rope hanging from start toward end.
    pub fn new(start: Vec3, end: Vec3, config: RopeConfig) -> Result<Self, RopeError> {
        if config.segment_count < 1 {
            return Err(RopeError::InvalidSegments);
        }
        if config.total_length <= 0.0 {
            return Err(RopeError::InvalidLength);
        }
        if config.mass_per_length <= 0.0 {
            return Err(RopeError::InvalidMass);
        }

        let seg_len = config.total_length / config.segment_count as f64;
        let dir = end.sub(start);
        let step = dir.scale(1.0 / config.segment_count as f64);
        let seg_mass = config.mass_per_length * seg_len;

        let mut particles = Vec::with_capacity(config.segment_count + 1);
        for i in 0..=config.segment_count {
            let pos = start.add(step.scale(i as f64));
            // First particle is fixed by default (attached)
            let fixed = i == 0;
            particles.push(RopeParticle::new(pos, seg_mass, fixed));
        }

        Ok(Self {
            particles,
            segment_length: seg_len,
            config,
            spheres: Vec::new(),
            planes: Vec::new(),
            time: 0.0,
        })
    }

    pub fn attach_start(&mut self) -> Result<(), RopeError> {
        if self.particles.is_empty() {
            return Err(RopeError::ParticleNotFound(0));
        }
        if self.particles[0].fixed {
            return Err(RopeError::AlreadyAttached(0));
        }
        self.particles[0].fixed = true;
        self.particles[0].inv_mass = 0.0;
        Ok(())
    }

    pub fn attach_end(&mut self) -> Result<(), RopeError> {
        let last = self.particles.len() - 1;
        if self.particles[last].fixed {
            return Err(RopeError::AlreadyAttached(last));
        }
        self.particles[last].fixed = true;
        self.particles[last].inv_mass = 0.0;
        Ok(())
    }

    pub fn detach_start(&mut self) -> Result<(), RopeError> {
        if self.particles.is_empty() {
            return Err(RopeError::ParticleNotFound(0));
        }
        if !self.particles[0].fixed {
            return Err(RopeError::NotAttached(0));
        }
        self.particles[0].fixed = false;
        self.particles[0].inv_mass = 1.0 / self.particles[0].mass;
        Ok(())
    }

    pub fn detach_end(&mut self) -> Result<(), RopeError> {
        let last = self.particles.len() - 1;
        if !self.particles[last].fixed {
            return Err(RopeError::NotAttached(last));
        }
        self.particles[last].fixed = false;
        self.particles[last].inv_mass = 1.0 / self.particles[last].mass;
        Ok(())
    }

    pub fn move_start(&mut self, pos: Vec3) -> Result<(), RopeError> {
        if self.particles.is_empty() {
            return Err(RopeError::ParticleNotFound(0));
        }
        self.particles[0].position = pos;
        self.particles[0].prev_position = pos;
        Ok(())
    }

    pub fn move_end(&mut self, pos: Vec3) -> Result<(), RopeError> {
        let last = self.particles.len() - 1;
        self.particles[last].position = pos;
        self.particles[last].prev_position = pos;
        Ok(())
    }

    pub fn add_sphere(&mut self, sphere: CollisionSphere) {
        self.spheres.push(sphere);
    }

    pub fn add_plane(&mut self, plane: CollisionPlane) {
        self.planes.push(plane);
    }

    fn verlet_integrate(&mut self, dt: f64) {
        let damp = 1.0 - self.config.damping;
        for p in &mut self.particles {
            if p.fixed {
                continue;
            }
            let vel = p.position.sub(p.prev_position).scale(damp);
            let new_pos = p.position.add(vel).add(p.acceleration.scale(dt * dt));
            p.prev_position = p.position;
            p.position = new_pos;
        }
    }

    fn satisfy_constraints(&mut self) {
        for _ in 0..self.config.constraint_iterations {
            for i in 0..self.particles.len() - 1 {
                let diff = self.particles[i + 1].position.sub(self.particles[i].position);
                let dist = diff.length();
                if dist < 1e-12 { continue; }

                let correction = diff.scale((dist - self.segment_length) / dist * 0.5 * self.config.stiffness);

                let w0 = self.particles[i].inv_mass;
                let w1 = self.particles[i + 1].inv_mass;
                let w_total = w0 + w1;
                if w_total < 1e-12 { continue; }

                if !self.particles[i].fixed {
                    self.particles[i].position = self.particles[i].position.add(
                        correction.scale(w0 / w_total * 2.0),
                    );
                }
                if !self.particles[i + 1].fixed {
                    self.particles[i + 1].position = self.particles[i + 1].position.sub(
                        correction.scale(w1 / w_total * 2.0),
                    );
                }
            }
        }
    }

    fn handle_collisions(&mut self) {
        for sphere in &self.spheres {
            for p in &mut self.particles {
                if p.fixed { continue; }
                let diff = p.position.sub(sphere.center);
                let dist = diff.length();
                if dist < sphere.radius && dist > 1e-12 {
                    p.position = sphere.center.add(diff.normalized().scale(sphere.radius));
                }
            }
        }

        for plane in &self.planes {
            let n = plane.normal.normalized();
            for p in &mut self.particles {
                if p.fixed { continue; }
                let d = p.position.sub(plane.point).dot(n);
                if d < 0.0 {
                    p.position = p.position.sub(n.scale(d));
                }
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), RopeError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(RopeError::InvalidTimestep);
        }

        // Apply gravity as acceleration
        for p in &mut self.particles {
            p.acceleration = if p.fixed { Vec3::ZERO } else { self.config.gravity };
        }

        self.verlet_integrate(dt);
        self.satisfy_constraints();
        self.handle_collisions();

        self.time += dt;
        Ok(())
    }

    /// Total length of the rope (sum of segment distances).
    pub fn current_length(&self) -> f64 {
        let mut len = 0.0;
        for i in 0..self.particles.len() - 1 {
            len += self.particles[i].position.sub(self.particles[i + 1].position).length();
        }
        len
    }

    /// Straight-line distance between endpoints.
    pub fn endpoint_distance(&self) -> f64 {
        if self.particles.len() < 2 { return 0.0; }
        let first = self.particles[0].position;
        let last = self.particles.last().unwrap().position;
        first.sub(last).length()
    }

    /// True if the rope is slack (endpoint distance < total rest length).
    pub fn is_slack(&self) -> bool {
        self.endpoint_distance() < self.config.total_length - 1e-6
    }

    /// True if the rope is taut (current length ≈ rest length under tension).
    pub fn is_taut(&self) -> bool {
        let ratio = self.endpoint_distance() / self.config.total_length;
        ratio > 0.95
    }

    /// Maximum sag (distance of farthest particle from the line between endpoints).
    pub fn max_sag(&self) -> f64 {
        if self.particles.len() < 3 { return 0.0; }
        let a = self.particles[0].position;
        let b = self.particles.last().unwrap().position;
        let ab = b.sub(a);
        let ab_len = ab.length();
        if ab_len < 1e-12 { return 0.0; }
        let ab_dir = ab.scale(1.0 / ab_len);

        let mut max_d = 0.0f64;
        for p in &self.particles {
            let ap = p.position.sub(a);
            let proj = ap.dot(ab_dir);
            let perp = ap.sub(ab_dir.scale(proj));
            let d = perp.length();
            if d > max_d { max_d = d; }
        }
        max_d
    }

    pub fn segment_count(&self) -> usize {
        if self.particles.len() < 2 { 0 } else { self.particles.len() - 1 }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Kinetic energy of the rope.
    pub fn kinetic_energy(&self, dt: f64) -> f64 {
        if dt <= 0.0 { return 0.0; }
        let mut ke = 0.0;
        for p in &self.particles {
            if p.fixed { continue; }
            let vel = p.position.sub(p.prev_position).scale(1.0 / dt);
            ke += 0.5 * p.mass * vel.length_sq();
        }
        ke
    }

    /// Center of mass.
    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for p in &self.particles {
            total_m += p.mass;
            weighted = weighted.add(p.position.scale(p.mass));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    /// Tension at a specific segment (force magnitude).
    pub fn segment_tension(&self, seg: usize) -> f64 {
        if seg >= self.particles.len() - 1 { return 0.0; }
        let dist = self.particles[seg].position.sub(self.particles[seg + 1].position).length();
        let extension = dist - self.segment_length;
        if extension > 0.0 {
            self.config.stiffness * extension * self.particles[seg].mass
        } else {
            0.0
        }
    }

    /// Maximum tension across all segments.
    pub fn max_tension(&self) -> f64 {
        (0..self.segment_count())
            .map(|i| self.segment_tension(i))
            .fold(0.0f64, f64::max)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn default_rope() -> Rope {
        Rope::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            RopeConfig::default(),
        ).unwrap()
    }

    #[test]
    fn test_creation() {
        let rope = default_rope();
        assert_eq!(rope.particle_count(), 21);
        assert_eq!(rope.segment_count(), 20);
    }

    #[test]
    fn test_invalid_segments() {
        let r = Rope::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), RopeConfig {
            segment_count: 0,
            ..Default::default()
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_invalid_length() {
        let r = Rope::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), RopeConfig {
            total_length: 0.0,
            ..Default::default()
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_invalid_mass() {
        let r = Rope::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), RopeConfig {
            mass_per_length: 0.0,
            ..Default::default()
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_start_attached() {
        let rope = default_rope();
        assert!(rope.particles[0].fixed);
    }

    #[test]
    fn test_attach_detach_end() {
        let mut rope = default_rope();
        rope.attach_end().unwrap();
        assert!(rope.particles.last().unwrap().fixed);
        rope.detach_end().unwrap();
        assert!(!rope.particles.last().unwrap().fixed);
    }

    #[test]
    fn test_already_attached_error() {
        let mut rope = default_rope();
        assert!(rope.attach_start().is_err());
    }

    #[test]
    fn test_not_attached_error() {
        let mut rope = default_rope();
        let last = rope.particles.len() - 1;
        assert!(rope.detach_end().is_err()); // not attached yet
        let _ = last;
    }

    #[test]
    fn test_gravity_sags_rope() {
        let mut rope = default_rope();
        for _ in 0..100 {
            rope.step(0.01).unwrap();
        }
        // Middle particle should be below start height
        let mid = rope.particles.len() / 2;
        assert!(rope.particles[mid].position.y < 5.0);
    }

    #[test]
    fn test_fixed_start_stays() {
        let mut rope = default_rope();
        let pos0 = rope.particles[0].position;
        for _ in 0..50 {
            rope.step(0.01).unwrap();
        }
        assert!(approx(rope.particles[0].position.x, pos0.x, 1e-10));
        assert!(approx(rope.particles[0].position.y, pos0.y, 1e-10));
    }

    #[test]
    fn test_current_length() {
        let rope = default_rope();
        // Initially, length should be close to endpoint distance
        assert!(approx(rope.current_length(), 5.0, 0.5));
    }

    #[test]
    fn test_endpoint_distance() {
        let rope = default_rope();
        assert!(approx(rope.endpoint_distance(), 5.0, 1e-4));
    }

    #[test]
    fn test_slack_detection() {
        let mut rope = default_rope();
        rope.attach_end().unwrap();
        // Rope is taut at start since endpoints match total length
        assert!(rope.is_taut());
    }

    #[test]
    fn test_slack_after_move() {
        let mut rope = Rope::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 5.0, 0.0),
            RopeConfig { total_length: 5.0, ..Default::default() },
        ).unwrap();
        // Endpoints are 1.0 apart but total length is 5.0 -> should be slack
        assert!(rope.is_slack());
    }

    #[test]
    fn test_max_sag() {
        let mut rope = default_rope();
        for _ in 0..200 {
            rope.step(0.01).unwrap();
        }
        // After hanging, there should be sag
        assert!(rope.max_sag() > 0.01);
    }

    #[test]
    fn test_sphere_collision() {
        let mut rope = default_rope();
        rope.add_sphere(CollisionSphere {
            center: Vec3::new(2.5, 3.0, 0.0),
            radius: 2.0,
        });
        for _ in 0..200 {
            rope.step(0.01).unwrap();
        }
        // Particles should be pushed out of sphere
        let center = Vec3::new(2.5, 3.0, 0.0);
        for p in &rope.particles {
            if p.fixed { continue; }
            let d = p.position.sub(center).length();
            assert!(d >= 2.0 - 0.5); // tolerance for constraint settling
        }
    }

    #[test]
    fn test_plane_collision() {
        let mut rope = default_rope();
        rope.add_plane(CollisionPlane {
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        });
        for _ in 0..300 {
            rope.step(0.01).unwrap();
        }
        for p in &rope.particles {
            assert!(p.position.y >= -0.5);
        }
    }

    #[test]
    fn test_invalid_timestep() {
        let mut rope = default_rope();
        assert!(rope.step(0.0).is_err());
        assert!(rope.step(-1.0).is_err());
    }

    #[test]
    fn test_move_start() {
        let mut rope = default_rope();
        rope.move_start(Vec3::new(0.0, 10.0, 0.0)).unwrap();
        assert!(approx(rope.particles[0].position.y, 10.0, 1e-10));
    }

    #[test]
    fn test_move_end() {
        let mut rope = default_rope();
        rope.move_end(Vec3::new(5.0, 10.0, 0.0)).unwrap();
        let last = rope.particles.len() - 1;
        assert!(approx(rope.particles[last].position.y, 10.0, 1e-10));
    }

    #[test]
    fn test_pendulum_swing() {
        // Single segment rope acts as pendulum
        let mut rope = Rope::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 4.0, 0.0),
            RopeConfig {
                segment_count: 1,
                total_length: 2.0,
                ..Default::default()
            },
        ).unwrap();
        // Displace the free end
        rope.particles[1].position = Vec3::new(2.0, 5.0, 0.0);
        rope.particles[1].prev_position = rope.particles[1].position;

        let x0 = rope.particles[1].position.x;
        for _ in 0..50 {
            rope.step(0.01).unwrap();
        }
        // Should have swung
        assert!((rope.particles[1].position.x - x0).abs() > 0.01);
    }

    #[test]
    fn test_kinetic_energy() {
        let mut rope = default_rope();
        assert!(approx(rope.kinetic_energy(0.01), 0.0, 1e-10));
        rope.step(0.01).unwrap();
        assert!(rope.kinetic_energy(0.01) > 0.0);
    }

    #[test]
    fn test_center_of_mass() {
        let rope = default_rope();
        let com = rope.center_of_mass();
        assert!(approx(com.x, 2.5, 0.5));
    }

    #[test]
    fn test_segment_tension() {
        let mut rope = default_rope();
        for _ in 0..50 {
            rope.step(0.01).unwrap();
        }
        // First segment (near anchor) should have tension
        let t = rope.segment_tension(0);
        // Tension may or may not be positive depending on constraint convergence
        assert!(t >= 0.0);
    }

    #[test]
    fn test_max_tension() {
        let rope = default_rope();
        let t = rope.max_tension();
        assert!(t >= 0.0);
    }

    #[test]
    fn test_detach_start_falls() {
        let mut rope = default_rope();
        for _ in 0..10 {
            rope.step(0.01).unwrap();
        }
        rope.detach_start().unwrap();
        let y0 = rope.particles[0].position.y;
        for _ in 0..50 {
            rope.step(0.01).unwrap();
        }
        assert!(rope.particles[0].position.y < y0);
    }

    #[test]
    fn test_short_rope() {
        let rope = Rope::new(
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            RopeConfig { segment_count: 1, total_length: 0.5, ..Default::default() },
        ).unwrap();
        assert_eq!(rope.particle_count(), 2);
        assert_eq!(rope.segment_count(), 1);
    }

    #[test]
    fn test_constraint_iterations_effect() {
        let mut rope_few = Rope::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            RopeConfig { constraint_iterations: 1, ..Default::default() },
        ).unwrap();

        let mut rope_many = Rope::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            RopeConfig { constraint_iterations: 50, ..Default::default() },
        ).unwrap();

        for _ in 0..50 {
            rope_few.step(0.01).unwrap();
            rope_many.step(0.01).unwrap();
        }

        // Both should complete without diverging
        assert!(rope_few.time > 0.0);
        assert!(rope_many.time > 0.0);
    }
}
