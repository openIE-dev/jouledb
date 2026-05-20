//! Mass-spring system simulation — particles, springs, forces, integration.
//!
//! Replaces matter.js / Cannon.js spring systems with pure Rust.
//! Supports Hooke's law + damping, semi-implicit Euler integration,
//! gravity, ground collision with bounce/friction, structural/shear/bend
//! springs for cloth-like behavior, and energy tracking.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MassSpringError {
    InvalidMass,
    InvalidStiffness,
    InvalidDamping,
    ParticleNotFound(usize),
    DuplicateSpring { a: usize, b: usize },
    SelfSpring(usize),
    InvalidTimestep,
}

impl fmt::Display for MassSpringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMass => write!(f, "mass must be positive"),
            Self::InvalidStiffness => write!(f, "stiffness must be non-negative"),
            Self::InvalidDamping => write!(f, "damping must be non-negative"),
            Self::ParticleNotFound(id) => write!(f, "particle not found: {id}"),
            Self::DuplicateSpring { a, b } => write!(f, "spring already exists between {a} and {b}"),
            Self::SelfSpring(id) => write!(f, "cannot connect particle {id} to itself"),
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
        }
    }
}

impl std::error::Error for MassSpringError {}

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

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::ZERO
        } else {
            Self { x: self.x / len, y: self.y / len, z: self.z / len }
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

// ── Spring Type ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpringKind {
    Structural,
    Shear,
    Bend,
}

// ── Particle ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub force: Vec3,
    pub mass: f64,
    pub pinned: bool,
    pub id: usize,
}

// ── Spring ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Spring {
    pub a: usize,
    pub b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub kind: SpringKind,
}

// ── Energy ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyState {
    pub kinetic: f64,
    pub potential_spring: f64,
    pub potential_gravity: f64,
    pub total: f64,
}

// ── System Config ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct MassSpringConfig {
    pub gravity: Vec3,
    pub ground_y: f64,
    pub ground_enabled: bool,
    pub bounce_factor: f64,
    pub friction_factor: f64,
    pub global_damping: f64,
}

impl Default for MassSpringConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ground_y: 0.0,
            ground_enabled: true,
            bounce_factor: 0.5,
            friction_factor: 0.3,
            global_damping: 0.01,
        }
    }
}

// ── Mass-Spring System ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MassSpringSystem {
    pub particles: Vec<Particle>,
    pub springs: Vec<Spring>,
    pub config: MassSpringConfig,
    pub time: f64,
    pub step_count: u64,
}

impl MassSpringSystem {
    pub fn new(config: MassSpringConfig) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            config,
            time: 0.0,
            step_count: 0,
        }
    }

    pub fn add_particle(&mut self, position: Vec3, mass: f64, pinned: bool) -> Result<usize, MassSpringError> {
        if mass <= 0.0 || !mass.is_finite() {
            return Err(MassSpringError::InvalidMass);
        }
        let id = self.particles.len();
        self.particles.push(Particle {
            position,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            mass,
            pinned,
            id,
        });
        Ok(id)
    }

    pub fn add_spring(
        &mut self,
        a: usize,
        b: usize,
        stiffness: f64,
        damping: f64,
        kind: SpringKind,
    ) -> Result<usize, MassSpringError> {
        if a == b {
            return Err(MassSpringError::SelfSpring(a));
        }
        if a >= self.particles.len() {
            return Err(MassSpringError::ParticleNotFound(a));
        }
        if b >= self.particles.len() {
            return Err(MassSpringError::ParticleNotFound(b));
        }
        if stiffness < 0.0 || !stiffness.is_finite() {
            return Err(MassSpringError::InvalidStiffness);
        }
        if damping < 0.0 || !damping.is_finite() {
            return Err(MassSpringError::InvalidDamping);
        }
        for s in &self.springs {
            if (s.a == a && s.b == b) || (s.a == b && s.b == a) {
                return Err(MassSpringError::DuplicateSpring { a, b });
            }
        }
        let rest_length = self.particles[a].position.sub(self.particles[b].position).length();
        let idx = self.springs.len();
        self.springs.push(Spring { a, b, rest_length, stiffness, damping, kind });
        Ok(idx)
    }

    pub fn add_spring_with_rest(
        &mut self,
        a: usize,
        b: usize,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
        kind: SpringKind,
    ) -> Result<usize, MassSpringError> {
        if a == b {
            return Err(MassSpringError::SelfSpring(a));
        }
        if a >= self.particles.len() {
            return Err(MassSpringError::ParticleNotFound(a));
        }
        if b >= self.particles.len() {
            return Err(MassSpringError::ParticleNotFound(b));
        }
        if stiffness < 0.0 || !stiffness.is_finite() {
            return Err(MassSpringError::InvalidStiffness);
        }
        if damping < 0.0 || !damping.is_finite() {
            return Err(MassSpringError::InvalidDamping);
        }
        let idx = self.springs.len();
        self.springs.push(Spring { a, b, rest_length: rest_length.abs(), stiffness, damping, kind });
        Ok(idx)
    }

    fn compute_spring_forces(&mut self) {
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let velocities: Vec<Vec3> = self.particles.iter().map(|p| p.velocity).collect();

        for spring in &self.springs {
            let diff = positions[spring.b].sub(positions[spring.a]);
            let dist = diff.length();
            if dist < 1e-12 {
                continue;
            }
            let dir = diff.scale(1.0 / dist);
            let stretch = dist - spring.rest_length;
            let rel_vel = velocities[spring.b].sub(velocities[spring.a]);
            let vel_along = rel_vel.dot(dir);

            let force_mag = spring.stiffness * stretch + spring.damping * vel_along;
            let force = dir.scale(force_mag);

            if !self.particles[spring.a].pinned {
                self.particles[spring.a].force = self.particles[spring.a].force.add(force);
            }
            if !self.particles[spring.b].pinned {
                self.particles[spring.b].force = self.particles[spring.b].force.sub(force);
            }
        }
    }

    fn apply_gravity(&mut self) {
        for p in &mut self.particles {
            if !p.pinned {
                p.force = p.force.add(self.config.gravity.scale(p.mass));
            }
        }
    }

    fn apply_global_damping(&mut self) {
        let d = self.config.global_damping;
        for p in &mut self.particles {
            if !p.pinned {
                p.force = p.force.sub(p.velocity.scale(d));
            }
        }
    }

    fn integrate_semi_implicit_euler(&mut self, dt: f64) {
        for p in &mut self.particles {
            if p.pinned {
                p.velocity = Vec3::ZERO;
                continue;
            }
            let accel = p.force.scale(1.0 / p.mass);
            p.velocity = p.velocity.add(accel.scale(dt));
            p.position = p.position.add(p.velocity.scale(dt));
        }
    }

    fn handle_ground_collision(&mut self) {
        if !self.config.ground_enabled {
            return;
        }
        let gy = self.config.ground_y;
        let bounce = self.config.bounce_factor;
        let friction = self.config.friction_factor;

        for p in &mut self.particles {
            if p.pinned {
                continue;
            }
            if p.position.y < gy {
                p.position.y = gy;
                if p.velocity.y < 0.0 {
                    p.velocity.y = -p.velocity.y * bounce;
                }
                p.velocity.x *= 1.0 - friction;
                p.velocity.z *= 1.0 - friction;
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), MassSpringError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(MassSpringError::InvalidTimestep);
        }
        // Preserve any externally applied forces, then add internal forces on top.
        self.apply_gravity();
        self.compute_spring_forces();
        self.apply_global_damping();
        self.integrate_semi_implicit_euler(dt);
        self.handle_ground_collision();
        // Clear forces after integration so the next step starts fresh
        // (external forces must be re-applied each step if desired).
        for p in &mut self.particles {
            p.force = Vec3::ZERO;
        }
        self.time += dt;
        self.step_count += 1;
        Ok(())
    }

    pub fn energy(&self) -> EnergyState {
        let mut kinetic = 0.0;
        let mut pot_gravity = 0.0;
        for p in &self.particles {
            kinetic += 0.5 * p.mass * p.velocity.length_sq();
            pot_gravity += p.mass * (-self.config.gravity.y) * (p.position.y - self.config.ground_y);
        }

        let mut pot_spring = 0.0;
        for s in &self.springs {
            let dist = self.particles[s.a].position.sub(self.particles[s.b].position).length();
            let stretch = dist - s.rest_length;
            pot_spring += 0.5 * s.stiffness * stretch * stretch;
        }

        EnergyState {
            kinetic,
            potential_spring: pot_spring,
            potential_gravity: pot_gravity,
            total: kinetic + pot_spring + pot_gravity,
        }
    }

    pub fn apply_force(&mut self, particle_id: usize, force: Vec3) -> Result<(), MassSpringError> {
        if particle_id >= self.particles.len() {
            return Err(MassSpringError::ParticleNotFound(particle_id));
        }
        self.particles[particle_id].force = self.particles[particle_id].force.add(force);
        Ok(())
    }

    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_mass = 0.0;
        let mut weighted = Vec3::ZERO;
        for p in &self.particles {
            total_mass += p.mass;
            weighted = weighted.add(p.position.scale(p.mass));
        }
        if total_mass < 1e-12 {
            return Vec3::ZERO;
        }
        weighted.scale(1.0 / total_mass)
    }

    pub fn total_momentum(&self) -> Vec3 {
        let mut mom = Vec3::ZERO;
        for p in &self.particles {
            if !p.pinned {
                mom = mom.add(p.velocity.scale(p.mass));
            }
        }
        mom
    }

    /// Build a cloth-like grid of particles with structural, shear, and bend springs.
    pub fn build_cloth_grid(
        rows: usize,
        cols: usize,
        spacing: f64,
        mass: f64,
        stiffness: f64,
        damping: f64,
        config: MassSpringConfig,
    ) -> Result<Self, MassSpringError> {
        let mut sys = Self::new(config);
        // Create particles in a grid on the XZ plane
        for r in 0..rows {
            for c in 0..cols {
                let pos = Vec3::new(c as f64 * spacing, 0.0, r as f64 * spacing);
                sys.add_particle(pos, mass, false)?;
            }
        }

        let idx = |r: usize, c: usize| -> usize { r * cols + c };

        for r in 0..rows {
            for c in 0..cols {
                // Structural — horizontal
                if c + 1 < cols {
                    sys.add_spring(idx(r, c), idx(r, c + 1), stiffness, damping, SpringKind::Structural)?;
                }
                // Structural — vertical
                if r + 1 < rows {
                    sys.add_spring(idx(r, c), idx(r + 1, c), stiffness, damping, SpringKind::Structural)?;
                }
                // Shear — diagonals
                if r + 1 < rows && c + 1 < cols {
                    sys.add_spring(idx(r, c), idx(r + 1, c + 1), stiffness * 0.5, damping, SpringKind::Shear)?;
                }
                if r + 1 < rows && c > 0 {
                    sys.add_spring(idx(r, c), idx(r + 1, c - 1), stiffness * 0.5, damping, SpringKind::Shear)?;
                }
                // Bend — skip one
                if c + 2 < cols {
                    sys.add_spring(idx(r, c), idx(r, c + 2), stiffness * 0.25, damping, SpringKind::Bend)?;
                }
                if r + 2 < rows {
                    sys.add_spring(idx(r, c), idx(r + 2, c), stiffness * 0.25, damping, SpringKind::Bend)?;
                }
            }
        }
        Ok(sys)
    }

    pub fn spring_count_by_kind(&self) -> (usize, usize, usize) {
        let mut structural = 0;
        let mut shear = 0;
        let mut bend = 0;
        for s in &self.springs {
            match s.kind {
                SpringKind::Structural => structural += 1,
                SpringKind::Shear => shear += 1,
                SpringKind::Bend => bend += 1,
            }
        }
        (structural, shear, bend)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn default_system() -> MassSpringSystem {
        MassSpringSystem::new(MassSpringConfig::default())
    }

    #[test]
    fn test_vec3_basics() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.length(), 5.0, 1e-10));
        let n = v.normalized();
        assert!(approx(n.length(), 1.0, 1e-10));
        assert!(approx(Vec3::ZERO.length(), 0.0, 1e-10));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(b), 32.0, 1e-10));
    }

    #[test]
    fn test_add_particle() {
        let mut sys = default_system();
        let id = sys.add_particle(Vec3::new(1.0, 2.0, 3.0), 1.0, false).unwrap();
        assert_eq!(id, 0);
        assert_eq!(sys.particles.len(), 1);
        assert!(approx(sys.particles[0].mass, 1.0, 1e-10));
    }

    #[test]
    fn test_invalid_mass() {
        let mut sys = default_system();
        assert_eq!(sys.add_particle(Vec3::ZERO, 0.0, false), Err(MassSpringError::InvalidMass));
        assert_eq!(sys.add_particle(Vec3::ZERO, -1.0, false), Err(MassSpringError::InvalidMass));
    }

    #[test]
    fn test_add_spring() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0, false).unwrap();
        let idx = sys.add_spring(0, 1, 100.0, 1.0, SpringKind::Structural).unwrap();
        assert_eq!(idx, 0);
        assert!(approx(sys.springs[0].rest_length, 1.0, 1e-10));
    }

    #[test]
    fn test_self_spring_rejected() {
        let mut sys = default_system();
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        assert_eq!(sys.add_spring(0, 0, 100.0, 1.0, SpringKind::Structural), Err(MassSpringError::SelfSpring(0)));
    }

    #[test]
    fn test_duplicate_spring_rejected() {
        let mut sys = default_system();
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        sys.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_spring(0, 1, 100.0, 1.0, SpringKind::Structural).unwrap();
        assert_eq!(
            sys.add_spring(1, 0, 100.0, 1.0, SpringKind::Structural),
            Err(MassSpringError::DuplicateSpring { a: 1, b: 0 })
        );
    }

    #[test]
    fn test_gravity_freefall() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            ground_enabled: false,
            ..Default::default()
        });
        sys.add_particle(Vec3::new(0.0, 10.0, 0.0), 1.0, false).unwrap();
        for _ in 0..100 {
            sys.step(0.01).unwrap();
        }
        // After 1 second of freefall: y ≈ 10 - 0.5*9.81*1^2 ≈ 5.095
        assert!(sys.particles[0].position.y < 10.0);
        assert!(sys.particles[0].velocity.y < 0.0);
    }

    #[test]
    fn test_pinned_particle_stays() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 5.0, 0.0), 1.0, true).unwrap();
        for _ in 0..50 {
            sys.step(0.01).unwrap();
        }
        assert!(approx(sys.particles[0].position.y, 5.0, 1e-10));
    }

    #[test]
    fn test_spring_oscillation() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::ZERO, 1.0, true).unwrap();
        sys.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_spring(0, 1, 100.0, 0.0, SpringKind::Structural).unwrap();
        // Rest length is 2.0; particle at 2.0. Displace:
        sys.particles[1].position.x = 3.0; // stretched by 1.0

        let initial_x = sys.particles[1].position.x;
        for _ in 0..200 {
            sys.step(0.001).unwrap();
        }
        // With no damping, it should oscillate — x should change from initial
        assert!((sys.particles[1].position.x - initial_x).abs() > 0.01);
    }

    #[test]
    fn test_ground_collision() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 0.5, 0.0), 1.0, false).unwrap();
        for _ in 0..500 {
            sys.step(0.01).unwrap();
        }
        // Particle should never go below ground
        assert!(sys.particles[0].position.y >= sys.config.ground_y - 1e-6);
    }

    #[test]
    fn test_energy_tracking() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::new(0.0, 10.0, 0.0), 1.0, false).unwrap();
        let e0 = sys.energy();
        assert!(e0.kinetic < 1e-10);
        assert!(e0.potential_gravity > 0.0);
        sys.step(0.01).unwrap();
        let e1 = sys.energy();
        assert!(e1.kinetic > 0.0);
    }

    #[test]
    fn test_cloth_grid_construction() {
        let sys = MassSpringSystem::build_cloth_grid(
            4, 4, 1.0, 1.0, 100.0, 1.0, MassSpringConfig::default(),
        ).unwrap();
        assert_eq!(sys.particles.len(), 16);
        let (structural, shear, bend) = sys.spring_count_by_kind();
        assert!(structural > 0);
        assert!(shear > 0);
        assert!(bend > 0);
    }

    #[test]
    fn test_cloth_grid_spring_counts() {
        // 3x3 grid
        let sys = MassSpringSystem::build_cloth_grid(
            3, 3, 1.0, 1.0, 100.0, 1.0, MassSpringConfig::default(),
        ).unwrap();
        // Structural: horizontal=6, vertical=6 = 12
        // Shear: diags = 2*2*2 = 8
        // Bend: horiz skip = 3, vert skip = 3 = 6
        let (s, sh, b) = sys.spring_count_by_kind();
        assert_eq!(s, 12);
        assert_eq!(sh, 8);
        assert_eq!(b, 6);
    }

    #[test]
    fn test_center_of_mass() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0, false).unwrap();
        let com = sys.center_of_mass();
        assert!(approx(com.x, 1.0, 1e-10));
    }

    #[test]
    fn test_total_momentum() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        sys.particles[0].velocity = Vec3::new(3.0, 0.0, 0.0);
        let mom = sys.total_momentum();
        assert!(approx(mom.x, 3.0, 1e-10));
    }

    #[test]
    fn test_invalid_timestep() {
        let mut sys = default_system();
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        assert_eq!(sys.step(0.0), Err(MassSpringError::InvalidTimestep));
        assert_eq!(sys.step(-1.0), Err(MassSpringError::InvalidTimestep));
    }

    #[test]
    fn test_spring_with_damping_settles() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::ZERO, 1.0, true).unwrap();
        sys.add_particle(Vec3::new(3.0, 0.0, 0.0), 1.0, false).unwrap();
        // Rest length = 3, start at 3, push away
        sys.add_spring(0, 1, 50.0, 10.0, SpringKind::Structural).unwrap();
        sys.particles[1].position.x = 5.0;

        for _ in 0..3000 {
            sys.step(0.005).unwrap();
        }
        // Should settle near rest length of 3
        let dist = sys.particles[0].position.sub(sys.particles[1].position).length();
        assert!(approx(dist, 3.0, 0.1));
    }

    #[test]
    fn test_apply_external_force() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        sys.apply_force(0, Vec3::new(10.0, 0.0, 0.0)).unwrap();
        // Force is accumulated, step will use it
        sys.step(0.1).unwrap();
        assert!(sys.particles[0].velocity.x > 0.0);
    }

    #[test]
    fn test_step_count_and_time() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 5.0, 0.0), 1.0, false).unwrap();
        sys.step(0.01).unwrap();
        sys.step(0.01).unwrap();
        assert_eq!(sys.step_count, 2);
        assert!(approx(sys.time, 0.02, 1e-10));
    }

    #[test]
    fn test_spring_rest_length_custom() {
        let mut sys = default_system();
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        sys.add_particle(Vec3::new(5.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_spring_with_rest(0, 1, 2.0, 100.0, 1.0, SpringKind::Structural).unwrap();
        assert!(approx(sys.springs[0].rest_length, 2.0, 1e-10));
    }

    #[test]
    fn test_particle_not_found_for_spring() {
        let mut sys = default_system();
        sys.add_particle(Vec3::ZERO, 1.0, false).unwrap();
        assert_eq!(
            sys.add_spring(0, 99, 100.0, 1.0, SpringKind::Structural),
            Err(MassSpringError::ParticleNotFound(99))
        );
    }

    #[test]
    fn test_energy_conservation_approx() {
        // With no damping and no ground, energy should be approximately conserved
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        sys.add_particle(Vec3::ZERO, 1.0, true).unwrap();
        sys.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.add_spring(0, 1, 100.0, 0.0, SpringKind::Structural).unwrap();
        sys.particles[1].position.x = 2.5;

        let e_initial = sys.energy().total;
        for _ in 0..100 {
            sys.step(0.001).unwrap();
        }
        let e_final = sys.energy().total;
        // Semi-implicit Euler doesn't perfectly conserve, but should be close
        assert!(approx(e_initial, e_final, 0.5));
    }

    #[test]
    fn test_multiple_springs_chain() {
        let mut sys = MassSpringSystem::new(MassSpringConfig {
            gravity: Vec3::ZERO,
            ground_enabled: false,
            global_damping: 0.0,
            ..Default::default()
        });
        for i in 0..5 {
            sys.add_particle(Vec3::new(i as f64, 0.0, 0.0), 1.0, i == 0).unwrap();
        }
        for i in 0..4 {
            sys.add_spring(i, i + 1, 200.0, 5.0, SpringKind::Structural).unwrap();
        }
        // Displace last particle
        sys.particles[4].position.x = 6.0;
        for _ in 0..500 {
            sys.step(0.002).unwrap();
        }
        // Chain should be somewhat stretched but connected
        for i in 0..4 {
            let d = sys.particles[i].position.sub(sys.particles[i + 1].position).length();
            assert!(d < 5.0); // not wildly diverged
        }
    }

    #[test]
    fn test_friction_slows_horizontal() {
        let mut sys = default_system();
        sys.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0, false).unwrap();
        sys.particles[0].velocity = Vec3::new(10.0, 0.0, 0.0);
        sys.step(0.01).unwrap();
        // On ground, friction should slow x velocity
        assert!(sys.particles[0].velocity.x.abs() < 10.0);
    }

    #[test]
    fn test_zero_normalized() {
        let n = Vec3::ZERO.normalized();
        assert!(approx(n.x, 0.0, 1e-10));
        assert!(approx(n.y, 0.0, 1e-10));
        assert!(approx(n.z, 0.0, 1e-10));
    }
}
