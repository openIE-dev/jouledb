//! N-body gravitational simulation — direct O(N^2) force computation.
//!
//! Replaces AstroPy N-body / rebound.js with pure Rust.
//! Bodies with position, velocity, mass. Integrators: Euler, Velocity Verlet,
//! Leapfrog. Gravitational softening, energy conservation tracking,
//! center-of-mass computation, collision detection with merging.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for N-body simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum NBodyError {
    /// Body mass must be positive.
    NonPositiveMass(f64),
    /// Timestep must be positive.
    NonPositiveTimestep(f64),
    /// Softening must be non-negative.
    NegativeSoftening(f64),
    /// No bodies in simulation.
    EmptySimulation,
    /// Body not found by id.
    BodyNotFound(u64),
    /// Collision radius must be non-negative.
    NegativeCollisionRadius(f64),
}

impl fmt::Display for NBodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveMass(m) => write!(f, "mass must be positive, got {m}"),
            Self::NonPositiveTimestep(dt) => write!(f, "timestep must be positive, got {dt}"),
            Self::NegativeSoftening(s) => write!(f, "softening must be non-negative, got {s}"),
            Self::EmptySimulation => write!(f, "simulation has no bodies"),
            Self::BodyNotFound(id) => write!(f, "body not found: {id}"),
            Self::NegativeCollisionRadius(r) => {
                write!(f, "collision radius must be non-negative, got {r}")
            }
        }
    }
}

impl std::error::Error for NBodyError {}

// ── 3D Vector ───────────────────────────────────────────────────

/// Simple 3D vector for positions, velocities, accelerations.
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

    pub fn magnitude_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn magnitude(self) -> f64 {
        self.magnitude_sq().sqrt()
    }

    pub fn normalized(self) -> Self {
        let m = self.magnitude();
        if m < 1e-30 {
            Self::ZERO
        } else {
            self * (1.0 / m)
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn distance_sq(self, other: Self) -> f64 {
        (self - other).magnitude_sq()
    }

    pub fn distance(self, other: Self) -> f64 {
        self.distance_sq(other).sqrt()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// ── Body ────────────────────────────────────────────────────────

/// A gravitating body with mass, position, and velocity.
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub id: u64,
    pub mass: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    /// Radius for collision detection. 0 disables.
    pub radius: f64,
}

impl Body {
    pub fn new(id: u64, mass: f64, position: Vec3, velocity: Vec3) -> Result<Self, NBodyError> {
        if mass <= 0.0 {
            return Err(NBodyError::NonPositiveMass(mass));
        }
        Ok(Self { id, mass, position, velocity, radius: 0.0 })
    }

    pub fn with_radius(mut self, r: f64) -> Result<Self, NBodyError> {
        if r < 0.0 {
            return Err(NBodyError::NegativeCollisionRadius(r));
        }
        self.radius = r;
        Ok(self)
    }

    /// Kinetic energy: 0.5 * m * v^2.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.magnitude_sq()
    }

    /// Momentum vector.
    pub fn momentum(&self) -> Vec3 {
        self.velocity * self.mass
    }
}

// ── Integrator ──────────────────────────────────────────────────

/// Integration method for advancing the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    /// First-order Euler (not symplectic).
    Euler,
    /// Velocity Verlet (symplectic, second-order).
    VelocityVerlet,
    /// Leapfrog (symplectic, second-order).
    Leapfrog,
}

// ── Simulation Config ───────────────────────────────────────────

/// Configuration for the N-body simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct NBodyConfig {
    /// Gravitational constant. Default: 1.0 (natural units).
    pub g_constant: f64,
    /// Softening parameter to avoid singularity. Default: 0.01.
    pub softening: f64,
    /// Timestep. Default: 0.001.
    pub dt: f64,
    /// Integration method. Default: VelocityVerlet.
    pub integrator: Integrator,
    /// Enable collision detection and merging.
    pub collisions_enabled: bool,
}

impl Default for NBodyConfig {
    fn default() -> Self {
        Self {
            g_constant: 1.0,
            softening: 0.01,
            dt: 0.001,
            integrator: Integrator::VelocityVerlet,
            collisions_enabled: false,
        }
    }
}

// ── Energy Stats ────────────────────────────────────────────────

/// Energy and conservation tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyStats {
    pub kinetic: f64,
    pub potential: f64,
    pub total: f64,
}

// ── Simulation ──────────────────────────────────────────────────

/// N-body gravitational simulation.
#[derive(Debug, Clone)]
pub struct NBodySim {
    pub bodies: Vec<Body>,
    pub config: NBodyConfig,
    pub time: f64,
    pub step_count: u64,
    next_id: u64,
    /// Cached accelerations for Verlet/Leapfrog integrators.
    accelerations: Vec<Vec3>,
}

impl NBodySim {
    /// Create a simulation from bodies and config.
    pub fn new(bodies: Vec<Body>, config: NBodyConfig) -> Result<Self, NBodyError> {
        if config.dt <= 0.0 {
            return Err(NBodyError::NonPositiveTimestep(config.dt));
        }
        if config.softening < 0.0 {
            return Err(NBodyError::NegativeSoftening(config.softening));
        }
        let max_id = bodies.iter().map(|b| b.id).max().unwrap_or(0);
        let n = bodies.len();
        let mut sim = Self {
            bodies,
            config,
            time: 0.0,
            step_count: 0,
            next_id: max_id + 1,
            accelerations: vec![Vec3::ZERO; n],
        };
        sim.compute_accelerations();
        Ok(sim)
    }

    /// Add a body to the simulation.
    pub fn add_body(&mut self, mut body: Body) -> u64 {
        body.id = self.next_id;
        self.next_id += 1;
        let id = body.id;
        self.bodies.push(body);
        self.accelerations.push(Vec3::ZERO);
        self.compute_accelerations();
        id
    }

    /// Remove a body by id.
    pub fn remove_body(&mut self, id: u64) -> Result<Body, NBodyError> {
        let idx = self.bodies.iter().position(|b| b.id == id)
            .ok_or(NBodyError::BodyNotFound(id))?;
        self.accelerations.remove(idx);
        let body = self.bodies.remove(idx);
        Ok(body)
    }

    /// Number of bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Total mass of the system.
    pub fn total_mass(&self) -> f64 {
        self.bodies.iter().map(|b| b.mass).sum()
    }

    /// Center of mass position.
    pub fn center_of_mass(&self) -> Vec3 {
        let tm = self.total_mass();
        if tm < 1e-30 {
            return Vec3::ZERO;
        }
        let mut com = Vec3::ZERO;
        for b in &self.bodies {
            com += b.position * b.mass;
        }
        com * (1.0 / tm)
    }

    /// Center of mass velocity.
    pub fn center_of_mass_velocity(&self) -> Vec3 {
        let tm = self.total_mass();
        if tm < 1e-30 {
            return Vec3::ZERO;
        }
        let mut v = Vec3::ZERO;
        for b in &self.bodies {
            v += b.velocity * b.mass;
        }
        v * (1.0 / tm)
    }

    /// Shift all bodies to the center of mass frame.
    pub fn shift_to_com_frame(&mut self) {
        let com = self.center_of_mass();
        let comv = self.center_of_mass_velocity();
        for b in &mut self.bodies {
            b.position = b.position - com;
            b.velocity = b.velocity - comv;
        }
    }

    /// Total momentum of the system.
    pub fn total_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for b in &self.bodies {
            p += b.momentum();
        }
        p
    }

    /// Total angular momentum about the origin.
    pub fn total_angular_momentum(&self) -> Vec3 {
        let mut l = Vec3::ZERO;
        for b in &self.bodies {
            l += b.position.cross(b.momentum());
        }
        l
    }

    /// Compute energy statistics.
    pub fn energy(&self) -> EnergyStats {
        let kinetic: f64 = self.bodies.iter().map(|b| b.kinetic_energy()).sum();
        let mut potential = 0.0;
        let eps2 = self.config.softening * self.config.softening;
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                let r2 = self.bodies[i].position.distance_sq(self.bodies[j].position) + eps2;
                let r = r2.sqrt();
                potential -= self.config.g_constant * self.bodies[i].mass
                    * self.bodies[j].mass / r;
            }
        }
        EnergyStats { kinetic, potential, total: kinetic + potential }
    }

    /// Compute O(N^2) gravitational accelerations with softening.
    fn compute_accelerations(&mut self) {
        let n = self.bodies.len();
        self.accelerations.resize(n, Vec3::ZERO);
        for a in &mut self.accelerations {
            *a = Vec3::ZERO;
        }
        let eps2 = self.config.softening * self.config.softening;
        let g = self.config.g_constant;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].position - self.bodies[i].position;
                let r2 = dx.magnitude_sq() + eps2;
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                let f = dx * (g * inv_r3);
                self.accelerations[i] += f * self.bodies[j].mass;
                self.accelerations[j] = self.accelerations[j] - f * self.bodies[i].mass;
            }
        }
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self) {
        match self.config.integrator {
            Integrator::Euler => self.step_euler(),
            Integrator::VelocityVerlet => self.step_verlet(),
            Integrator::Leapfrog => self.step_leapfrog(),
        }
        if self.config.collisions_enabled {
            self.detect_and_merge_collisions();
        }
        self.time += self.config.dt;
        self.step_count += 1;
    }

    /// Advance simulation by multiple timesteps.
    pub fn advance(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    fn step_euler(&mut self) {
        let dt = self.config.dt;
        self.compute_accelerations();
        for i in 0..self.bodies.len() {
            let vel = self.bodies[i].velocity;
            self.bodies[i].position += vel * dt;
            self.bodies[i].velocity += self.accelerations[i] * dt;
        }
    }

    fn step_verlet(&mut self) {
        let dt = self.config.dt;
        let half_dt = 0.5 * dt;
        // Half-step velocity and full-step position.
        for i in 0..self.bodies.len() {
            self.bodies[i].velocity += self.accelerations[i] * half_dt;
            let vel = self.bodies[i].velocity;
            self.bodies[i].position += vel * dt;
        }
        // Recompute accelerations at new positions.
        self.compute_accelerations();
        // Second half-step velocity.
        for i in 0..self.bodies.len() {
            self.bodies[i].velocity += self.accelerations[i] * half_dt;
        }
    }

    fn step_leapfrog(&mut self) {
        let dt = self.config.dt;
        // Kick
        for i in 0..self.bodies.len() {
            self.bodies[i].velocity += self.accelerations[i] * (0.5 * dt);
        }
        // Drift
        for i in 0..self.bodies.len() {
            let vel = self.bodies[i].velocity;
            self.bodies[i].position += vel * dt;
        }
        // Recompute accelerations
        self.compute_accelerations();
        // Kick
        for i in 0..self.bodies.len() {
            self.bodies[i].velocity += self.accelerations[i] * (0.5 * dt);
        }
    }

    /// Detect collisions and merge overlapping bodies (inelastic).
    fn detect_and_merge_collisions(&mut self) {
        let mut merged_away: Vec<bool> = vec![false; self.bodies.len()];
        let mut merges: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.bodies.len() {
            if merged_away[i] {
                continue;
            }
            for j in (i + 1)..self.bodies.len() {
                if merged_away[j] {
                    continue;
                }
                let dist = self.bodies[i].position.distance(self.bodies[j].position);
                let overlap = self.bodies[i].radius + self.bodies[j].radius;
                if overlap > 0.0 && dist < overlap {
                    merges.push((i, j));
                    merged_away[j] = true;
                }
            }
        }
        // Apply merges: absorb j into i (conservation of momentum).
        for &(i, j) in &merges {
            let total_m = self.bodies[i].mass + self.bodies[j].mass;
            let new_v = (self.bodies[i].momentum() + self.bodies[j].momentum()) * (1.0 / total_m);
            let new_pos = (self.bodies[i].position * self.bodies[i].mass
                + self.bodies[j].position * self.bodies[j].mass)
                * (1.0 / total_m);
            self.bodies[i].mass = total_m;
            self.bodies[i].position = new_pos;
            self.bodies[i].velocity = new_v;
            // Grow radius as cube-root of mass ratio (volume conservation).
            let r_i = self.bodies[i].radius;
            let r_j = self.bodies[j].radius;
            self.bodies[i].radius = (r_i * r_i * r_i + r_j * r_j * r_j).cbrt();
        }
        // Remove merged bodies in reverse index order.
        let mut to_remove: Vec<usize> =
            merges.iter().map(|&(_, j)| j).collect();
        to_remove.sort_unstable();
        to_remove.dedup();
        for &idx in to_remove.iter().rev() {
            self.bodies.remove(idx);
            self.accelerations.remove(idx);
        }
    }

    /// Set the timestep.
    pub fn set_dt(&mut self, dt: f64) -> Result<(), NBodyError> {
        if dt <= 0.0 {
            return Err(NBodyError::NonPositiveTimestep(dt));
        }
        self.config.dt = dt;
        Ok(())
    }

    /// Gravitational force between body i and body j.
    pub fn force_between(&self, i: usize, j: usize) -> Vec3 {
        let dx = self.bodies[j].position - self.bodies[i].position;
        let eps2 = self.config.softening * self.config.softening;
        let r2 = dx.magnitude_sq() + eps2;
        let inv_r3 = 1.0 / (r2 * r2.sqrt());
        dx * (self.config.g_constant * self.bodies[i].mass * self.bodies[j].mass * inv_r3)
    }
}

// ── Helper: create two-body orbit ───────────────────────────────

/// Create a simple two-body system (star + planet) in a circular orbit.
pub fn circular_two_body(
    star_mass: f64,
    planet_mass: f64,
    orbital_radius: f64,
    g_constant: f64,
) -> Result<Vec<Body>, NBodyError> {
    let mu = g_constant * (star_mass + planet_mass);
    let v_orbit = (mu / orbital_radius).sqrt();
    let factor_star = planet_mass / (star_mass + planet_mass);
    let factor_planet = star_mass / (star_mass + planet_mass);
    let star = Body::new(
        0,
        star_mass,
        Vec3::new(-orbital_radius * factor_star, 0.0, 0.0),
        Vec3::new(0.0, -v_orbit * factor_star, 0.0),
    )?;
    let planet = Body::new(
        1,
        planet_mass,
        Vec3::new(orbital_radius * factor_planet, 0.0, 0.0),
        Vec3::new(0.0, v_orbit * factor_planet, 0.0),
    )?;
    Ok(vec![star, planet])
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a + b, Vec3::new(5.0, 7.0, 9.0), 1e-10));
        assert!(vec3_approx_eq(a - b, Vec3::new(-3.0, -3.0, -3.0), 1e-10));
        assert!(approx_eq(a.dot(b), 32.0, 1e-10));
        assert!(vec3_approx_eq(a.cross(b), Vec3::new(-3.0, 6.0, -3.0), 1e-10));
    }

    #[test]
    fn vec3_magnitude() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.magnitude(), 5.0, 1e-10));
        let n = v.normalized();
        assert!(approx_eq(n.magnitude(), 1.0, 1e-10));
    }

    #[test]
    fn body_creation() {
        let b = Body::new(0, 1.0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!(b.is_ok());
        assert!(Body::new(0, -1.0, Vec3::ZERO, Vec3::ZERO).is_err());
        assert!(Body::new(0, 0.0, Vec3::ZERO, Vec3::ZERO).is_err());
    }

    #[test]
    fn body_kinetic_energy() {
        let b = Body::new(0, 2.0, Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0)).unwrap();
        // KE = 0.5 * 2 * 25 = 25
        assert!(approx_eq(b.kinetic_energy(), 25.0, 1e-10));
    }

    #[test]
    fn body_with_radius() {
        let b = Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap().with_radius(0.5);
        assert!(b.is_ok());
        assert!(Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap().with_radius(-1.0).is_err());
    }

    #[test]
    fn sim_creation() {
        let bodies = vec![
            Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap(),
            Body::new(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let sim = NBodySim::new(bodies, NBodyConfig::default());
        assert!(sim.is_ok());
    }

    #[test]
    fn sim_invalid_dt() {
        let bodies = vec![Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap()];
        let mut cfg = NBodyConfig::default();
        cfg.dt = -0.1;
        assert!(NBodySim::new(bodies, cfg).is_err());
    }

    #[test]
    fn sim_center_of_mass() {
        let bodies = vec![
            Body::new(0, 1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
            Body::new(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        assert!(vec3_approx_eq(sim.center_of_mass(), Vec3::ZERO, 1e-10));
    }

    #[test]
    fn sim_total_mass() {
        let bodies = vec![
            Body::new(0, 3.0, Vec3::ZERO, Vec3::ZERO).unwrap(),
            Body::new(1, 7.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        assert!(approx_eq(sim.total_mass(), 10.0, 1e-10));
    }

    #[test]
    fn energy_conservation_verlet() {
        let bodies = circular_two_body(100.0, 1.0, 10.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.001;
        cfg.softening = 0.0;
        cfg.integrator = Integrator::VelocityVerlet;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let e0 = sim.energy().total;
        sim.advance(1000);
        let e1 = sim.energy().total;
        let rel_err = ((e1 - e0) / e0.abs()).abs();
        assert!(rel_err < 1e-4, "Verlet energy drift {rel_err}");
    }

    #[test]
    fn energy_conservation_leapfrog() {
        let bodies = circular_two_body(100.0, 1.0, 10.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.001;
        cfg.softening = 0.0;
        cfg.integrator = Integrator::Leapfrog;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let e0 = sim.energy().total;
        sim.advance(1000);
        let e1 = sim.energy().total;
        let rel_err = ((e1 - e0) / e0.abs()).abs();
        assert!(rel_err < 1e-4, "Leapfrog energy drift {rel_err}");
    }

    #[test]
    fn euler_less_accurate() {
        let bodies = circular_two_body(100.0, 1.0, 10.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.001;
        cfg.softening = 0.0;
        cfg.integrator = Integrator::Euler;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let e0 = sim.energy().total;
        sim.advance(1000);
        let e1 = sim.energy().total;
        // Euler drifts more but should not explode in 1000 steps with small dt.
        let rel_err = ((e1 - e0) / e0.abs()).abs();
        assert!(rel_err < 0.1, "Euler energy drift too large: {rel_err}");
    }

    #[test]
    fn momentum_conservation() {
        let bodies = circular_two_body(100.0, 1.0, 10.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.001;
        cfg.integrator = Integrator::VelocityVerlet;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let p0 = sim.total_momentum();
        sim.advance(500);
        let p1 = sim.total_momentum();
        assert!(vec3_approx_eq(p0, p1, 1e-8));
    }

    #[test]
    fn add_and_remove_body() {
        let bodies = vec![Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap()];
        let mut sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        assert_eq!(sim.body_count(), 1);
        let new_id = sim.add_body(Body::new(99, 2.0, Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO).unwrap());
        assert_eq!(sim.body_count(), 2);
        sim.remove_body(new_id).unwrap();
        assert_eq!(sim.body_count(), 1);
        assert!(sim.remove_body(999).is_err());
    }

    #[test]
    fn collision_merging() {
        let b1 = Body::new(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
            .unwrap()
            .with_radius(1.0)
            .unwrap();
        let b2 = Body::new(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0))
            .unwrap()
            .with_radius(1.0)
            .unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.collisions_enabled = true;
        let mut sim = NBodySim::new(vec![b1, b2], cfg).unwrap();
        sim.step();
        // Bodies overlap initially (dist=0.5 < 2.0), so merge happens.
        assert_eq!(sim.body_count(), 1);
        assert!(approx_eq(sim.bodies[0].mass, 2.0, 1e-10));
        // Momentum conservation: net momentum was 0.
        assert!(approx_eq(sim.bodies[0].velocity.x, 0.0, 1e-10));
    }

    #[test]
    fn com_frame_shift() {
        let bodies = vec![
            Body::new(0, 3.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)).unwrap(),
            Body::new(1, 1.0, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(-6.0, 0.0, 0.0)).unwrap(),
        ];
        let mut sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        sim.shift_to_com_frame();
        assert!(vec3_approx_eq(sim.center_of_mass(), Vec3::ZERO, 1e-10));
        assert!(vec3_approx_eq(sim.center_of_mass_velocity(), Vec3::ZERO, 1e-10));
    }

    #[test]
    fn softening_prevents_divergence() {
        // Two bodies very close together.
        let bodies = vec![
            Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap(),
            Body::new(1, 1.0, Vec3::new(1e-6, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let mut cfg = NBodyConfig::default();
        cfg.softening = 0.1;
        cfg.dt = 0.001;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        sim.advance(100);
        // Should not produce NaN or Inf.
        for b in &sim.bodies {
            assert!(b.position.x.is_finite());
            assert!(b.velocity.x.is_finite());
        }
    }

    #[test]
    fn force_between_bodies() {
        let bodies = vec![
            Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap(),
            Body::new(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let mut cfg = NBodyConfig::default();
        cfg.softening = 0.0;
        let sim = NBodySim::new(bodies, cfg).unwrap();
        let f = sim.force_between(0, 1);
        // G*m1*m2/r^2 = 1.0, directed along +x
        assert!(approx_eq(f.x, 1.0, 1e-10));
        assert!(approx_eq(f.y, 0.0, 1e-10));
    }

    #[test]
    fn set_timestep() {
        let bodies = vec![Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap()];
        let mut sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        assert!(sim.set_dt(0.01).is_ok());
        assert!(sim.set_dt(-0.01).is_err());
    }

    #[test]
    fn angular_momentum_conservation() {
        let bodies = circular_two_body(100.0, 1.0, 10.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.001;
        cfg.softening = 0.0;
        cfg.integrator = Integrator::VelocityVerlet;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let l0 = sim.total_angular_momentum();
        sim.advance(500);
        let l1 = sim.total_angular_momentum();
        assert!(vec3_approx_eq(l0, l1, 1e-6));
    }

    #[test]
    fn three_body_runs() {
        let bodies = vec![
            Body::new(0, 1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -0.5, 0.0)).unwrap(),
            Body::new(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0)).unwrap(),
            Body::new(2, 1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.3, 0.0, 0.0)).unwrap(),
        ];
        let mut sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        sim.advance(100);
        assert_eq!(sim.step_count, 100);
        for b in &sim.bodies {
            assert!(b.position.x.is_finite());
        }
    }

    #[test]
    fn circular_orbit_stability() {
        let bodies = circular_two_body(1000.0, 1.0, 5.0, 1.0).unwrap();
        let mut cfg = NBodyConfig::default();
        cfg.dt = 0.0005;
        cfg.softening = 0.0;
        cfg.integrator = Integrator::VelocityVerlet;
        let mut sim = NBodySim::new(bodies, cfg).unwrap();
        let r0 = sim.bodies[1].position.distance(sim.bodies[0].position);
        // Run for ~1 orbit period: T = 2*pi*sqrt(a^3/(G*M))
        let period = 2.0 * std::f64::consts::PI * (5.0_f64.powi(3) / 1000.0).sqrt();
        let steps = (period / 0.0005) as u64;
        sim.advance(steps);
        let r1 = sim.bodies[1].position.distance(sim.bodies[0].position);
        // Radius should be close to initial.
        assert!(approx_eq(r0, r1, 0.1));
    }

    #[test]
    fn negative_softening_error() {
        let bodies = vec![Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap()];
        let mut cfg = NBodyConfig::default();
        cfg.softening = -0.1;
        assert!(NBodySim::new(bodies, cfg).is_err());
    }

    #[test]
    fn step_count_increments() {
        let bodies = vec![Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap()];
        let mut sim = NBodySim::new(bodies, NBodyConfig::default()).unwrap();
        assert_eq!(sim.step_count, 0);
        sim.step();
        assert_eq!(sim.step_count, 1);
        sim.advance(9);
        assert_eq!(sim.step_count, 10);
    }

    #[test]
    fn g_constant_scales_force() {
        let bodies = vec![
            Body::new(0, 1.0, Vec3::ZERO, Vec3::ZERO).unwrap(),
            Body::new(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO).unwrap(),
        ];
        let mut cfg1 = NBodyConfig::default();
        cfg1.softening = 0.0;
        cfg1.g_constant = 1.0;
        let sim1 = NBodySim::new(bodies.clone(), cfg1).unwrap();
        let f1 = sim1.force_between(0, 1);

        let mut cfg2 = NBodyConfig::default();
        cfg2.softening = 0.0;
        cfg2.g_constant = 2.0;
        let sim2 = NBodySim::new(bodies, cfg2).unwrap();
        let f2 = sim2.force_between(0, 1);
        assert!(approx_eq(f2.x, 2.0 * f1.x, 1e-10));
    }
}
