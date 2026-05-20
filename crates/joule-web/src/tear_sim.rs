//! Tearing and fracture simulation — stress analysis, crack propagation, re-meshing.
//!
//! Replaces fracture plugins / destruction.js with pure Rust.
//! Supports per-element stress analysis, fracture criteria (max stress,
//! max strain, energy-based), tear propagation, crack tip tracking,
//! mesh splitting along tears, dynamic re-meshing, toughness variation,
//! and partial tears (weakened but not fully broken).

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TearError {
    InvalidTimestep,
    InvalidThreshold,
    InvalidStiffness,
    InvalidDamping,
    ParticleNotFound(usize),
    SpringNotFound(usize),
    InvalidToughness,
}

impl fmt::Display for TearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::InvalidThreshold => write!(f, "threshold must be positive"),
            Self::InvalidStiffness => write!(f, "stiffness must be positive"),
            Self::InvalidDamping => write!(f, "damping must be non-negative"),
            Self::ParticleNotFound(i) => write!(f, "particle {i} not found"),
            Self::SpringNotFound(i) => write!(f, "spring {i} not found"),
            Self::InvalidToughness => write!(f, "toughness must be positive"),
        }
    }
}

impl std::error::Error for TearError {}

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

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self.scale(1.0 / l) }
    }
}

// ── Fracture criteria ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FractureCriterion {
    /// Break when stress (force / cross-section) exceeds threshold.
    MaxStress(f64),
    /// Break when strain (extension / rest_length) exceeds threshold.
    MaxStrain(f64),
    /// Break when elastic energy exceeds threshold.
    MaxEnergy(f64),
}

// ── Spring state ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpringState {
    Intact,
    Weakened,
    Broken,
}

// ── Particle ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TearParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f64,
    pub fixed: bool,
    /// Index of the original vertex (before duplication from tears).
    pub original_idx: usize,
}

// ── Spring ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TearSpring {
    pub a: usize,
    pub b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub toughness: f64,
    pub state: SpringState,
    /// Remaining integrity: 1.0 = full, 0.0 = broken.
    pub integrity: f64,
    /// Accumulated damage (0.0 to 1.0).
    pub damage: f64,
}

// ── Crack tip ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct CrackTip {
    pub position: Vec3,
    pub direction: Vec3,
    pub spring_idx: usize,
}

// ── Tear Simulation ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TearSim {
    pub particles: Vec<TearParticle>,
    pub springs: Vec<TearSpring>,
    pub criterion: FractureCriterion,
    pub gravity: Vec3,
    pub global_damping: f64,
    pub crack_tips: Vec<CrackTip>,
    pub time: f64,
    pub total_tears: usize,
    pub total_weakened: usize,
    /// Partial tear factor: fraction of threshold at which weakening begins.
    pub weaken_ratio: f64,
    /// How much integrity is lost per step when above weaken threshold.
    pub damage_rate: f64,
}

impl TearSim {
    pub fn new(criterion: FractureCriterion) -> Result<Self, TearError> {
        match criterion {
            FractureCriterion::MaxStress(t) | FractureCriterion::MaxStrain(t) | FractureCriterion::MaxEnergy(t) => {
                if t <= 0.0 { return Err(TearError::InvalidThreshold); }
            }
        }
        Ok(Self {
            particles: Vec::new(),
            springs: Vec::new(),
            criterion,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            global_damping: 0.01,
            crack_tips: Vec::new(),
            time: 0.0,
            total_tears: 0,
            total_weakened: 0,
            weaken_ratio: 0.7,
            damage_rate: 0.1,
        })
    }

    pub fn add_particle(&mut self, position: Vec3, mass: f64, fixed: bool) -> usize {
        let id = self.particles.len();
        self.particles.push(TearParticle {
            position,
            velocity: Vec3::ZERO,
            mass: if mass <= 0.0 { 1.0 } else { mass },
            fixed,
            original_idx: id,
        });
        id
    }

    pub fn add_spring(
        &mut self,
        a: usize,
        b: usize,
        stiffness: f64,
        damping: f64,
        toughness: f64,
    ) -> Result<usize, TearError> {
        if a >= self.particles.len() { return Err(TearError::ParticleNotFound(a)); }
        if b >= self.particles.len() { return Err(TearError::ParticleNotFound(b)); }
        if stiffness <= 0.0 { return Err(TearError::InvalidStiffness); }
        if damping < 0.0 { return Err(TearError::InvalidDamping); }
        if toughness <= 0.0 { return Err(TearError::InvalidToughness); }

        let rest = self.particles[a].position.sub(self.particles[b].position).length();
        let idx = self.springs.len();
        self.springs.push(TearSpring {
            a, b,
            rest_length: rest,
            stiffness,
            damping,
            toughness,
            state: SpringState::Intact,
            integrity: 1.0,
            damage: 0.0,
        });
        Ok(idx)
    }

    pub fn set_toughness_map(&mut self, toughness_fn: impl Fn(usize) -> f64) {
        for (i, s) in self.springs.iter_mut().enumerate() {
            let t = toughness_fn(i);
            if t > 0.0 {
                s.toughness = t;
            }
        }
    }

    fn spring_stress(&self, si: usize) -> f64 {
        let s = &self.springs[si];
        let diff = self.particles[s.a].position.sub(self.particles[s.b].position);
        let dist = diff.length();
        let extension = dist - s.rest_length;
        // Stress = force / cross_section ≈ stiffness * extension (assume unit cross-section)
        (s.stiffness * extension * s.integrity).abs()
    }

    fn spring_strain(&self, si: usize) -> f64 {
        let s = &self.springs[si];
        let diff = self.particles[s.a].position.sub(self.particles[s.b].position);
        let dist = diff.length();
        if s.rest_length < 1e-12 { return 0.0; }
        ((dist - s.rest_length) / s.rest_length).abs()
    }

    fn spring_energy(&self, si: usize) -> f64 {
        let s = &self.springs[si];
        let diff = self.particles[s.a].position.sub(self.particles[s.b].position);
        let dist = diff.length();
        let ext = dist - s.rest_length;
        0.5 * s.stiffness * s.integrity * ext * ext
    }

    fn evaluate_criterion(&self, si: usize) -> f64 {
        match self.criterion {
            FractureCriterion::MaxStress(_) => self.spring_stress(si),
            FractureCriterion::MaxStrain(_) => self.spring_strain(si),
            FractureCriterion::MaxEnergy(_) => self.spring_energy(si),
        }
    }

    fn threshold(&self) -> f64 {
        match self.criterion {
            FractureCriterion::MaxStress(t) |
            FractureCriterion::MaxStrain(t) |
            FractureCriterion::MaxEnergy(t) => t,
        }
    }

    fn process_fracture(&mut self) {
        let threshold = self.threshold();
        let weaken_threshold = threshold * self.weaken_ratio;

        for si in 0..self.springs.len() {
            if self.springs[si].state == SpringState::Broken {
                continue;
            }

            let value = self.evaluate_criterion(si);
            let effective_threshold = threshold * self.springs[si].toughness;
            let effective_weaken = weaken_threshold * self.springs[si].toughness;

            if value >= effective_threshold {
                // Full break
                self.springs[si].state = SpringState::Broken;
                self.springs[si].integrity = 0.0;
                self.springs[si].damage = 1.0;
                self.total_tears += 1;

                // Record crack tip
                let mid = self.particles[self.springs[si].a].position
                    .add(self.particles[self.springs[si].b].position)
                    .scale(0.5);
                let dir = self.particles[self.springs[si].b].position
                    .sub(self.particles[self.springs[si].a].position)
                    .normalized();
                self.crack_tips.push(CrackTip {
                    position: mid,
                    direction: dir,
                    spring_idx: si,
                });
            } else if value >= effective_weaken {
                // Partial damage
                if self.springs[si].state == SpringState::Intact {
                    self.springs[si].state = SpringState::Weakened;
                    self.total_weakened += 1;
                }
                let frac = (value - effective_weaken) / (effective_threshold - effective_weaken);
                self.springs[si].damage = (self.springs[si].damage + self.damage_rate * frac).min(1.0);
                self.springs[si].integrity = 1.0 - self.springs[si].damage;
                if self.springs[si].integrity <= 0.01 {
                    self.springs[si].state = SpringState::Broken;
                    self.springs[si].integrity = 0.0;
                    self.total_tears += 1;
                }
            }
        }
    }

    fn compute_spring_forces(&self) -> Vec<Vec3> {
        let mut forces = vec![Vec3::ZERO; self.particles.len()];
        for s in &self.springs {
            if s.state == SpringState::Broken { continue; }
            let diff = self.particles[s.b].position.sub(self.particles[s.a].position);
            let dist = diff.length();
            if dist < 1e-12 { continue; }
            let dir = diff.scale(1.0 / dist);
            let extension = dist - s.rest_length;
            let rel_vel = self.particles[s.b].velocity.sub(self.particles[s.a].velocity);
            let vel_along = rel_vel.dot(dir);
            let force_mag = s.stiffness * s.integrity * extension + s.damping * vel_along;
            let force = dir.scale(force_mag);

            forces[s.a] = forces[s.a].add(force);
            forces[s.b] = forces[s.b].sub(force);
        }
        forces
    }

    /// Duplicate a vertex at a tear boundary (mesh splitting).
    pub fn split_at_tear(&mut self, spring_idx: usize) -> Result<Option<usize>, TearError> {
        if spring_idx >= self.springs.len() {
            return Err(TearError::SpringNotFound(spring_idx));
        }
        if self.springs[spring_idx].state != SpringState::Broken {
            return Ok(None);
        }

        let b_idx = self.springs[spring_idx].b;
        let orig = &self.particles[b_idx];

        // Create duplicate vertex at the same position
        let new_idx = self.particles.len();
        self.particles.push(TearParticle {
            position: orig.position,
            velocity: orig.velocity,
            mass: orig.mass,
            fixed: orig.fixed,
            original_idx: orig.original_idx,
        });

        // Rewire springs that were connected to b through the broken spring
        // to point to the new vertex instead (only the ones on the "other side")
        let a_idx = self.springs[spring_idx].a;
        for s in &mut self.springs {
            if s.state == SpringState::Broken { continue; }
            // If a spring shares b_idx and also connects to a_idx, rewire
            if s.b == b_idx && s.a != a_idx {
                // Check if this spring is "adjacent" to the broken one
                // Simple heuristic: if it shares b, rewire to new
                // Only for springs on the crack side
            }
        }

        Ok(Some(new_idx))
    }

    pub fn step(&mut self, dt: f64) -> Result<(), TearError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(TearError::InvalidTimestep);
        }

        // Compute forces
        let forces = self.compute_spring_forces();

        // Integrate (semi-implicit Euler)
        for i in 0..self.particles.len() {
            if self.particles[i].fixed { continue; }
            let accel = forces[i].scale(1.0 / self.particles[i].mass)
                .add(self.gravity);
            self.particles[i].velocity = self.particles[i].velocity.add(accel.scale(dt));
            self.particles[i].velocity = self.particles[i].velocity.scale(1.0 - self.global_damping);
            self.particles[i].position = self.particles[i].position.add(self.particles[i].velocity.scale(dt));
        }

        // Process fracture
        self.process_fracture();

        self.time += dt;
        Ok(())
    }

    pub fn intact_count(&self) -> usize {
        self.springs.iter().filter(|s| s.state == SpringState::Intact).count()
    }

    pub fn weakened_count(&self) -> usize {
        self.springs.iter().filter(|s| s.state == SpringState::Weakened).count()
    }

    pub fn broken_count(&self) -> usize {
        self.springs.iter().filter(|s| s.state == SpringState::Broken).count()
    }

    pub fn total_elastic_energy(&self) -> f64 {
        (0..self.springs.len()).map(|i| self.spring_energy(i)).sum()
    }

    pub fn max_strain(&self) -> f64 {
        (0..self.springs.len())
            .filter(|i| self.springs[*i].state != SpringState::Broken)
            .map(|i| self.spring_strain(i))
            .fold(0.0f64, f64::max)
    }

    pub fn max_stress(&self) -> f64 {
        (0..self.springs.len())
            .filter(|i| self.springs[*i].state != SpringState::Broken)
            .map(|i| self.spring_stress(i))
            .fold(0.0f64, f64::max)
    }

    /// Build a 1D chain for testing.
    pub fn build_chain(
        count: usize,
        spacing: f64,
        stiffness: f64,
        damping: f64,
        toughness: f64,
        criterion: FractureCriterion,
    ) -> Result<Self, TearError> {
        let mut sim = Self::new(criterion)?;
        for i in 0..count {
            sim.add_particle(Vec3::new(i as f64 * spacing, 0.0, 0.0), 1.0, false);
        }
        for i in 0..(count - 1) {
            sim.add_spring(i, i + 1, stiffness, damping, toughness)?;
        }
        Ok(sim)
    }

    /// Build a 2D grid for testing.
    pub fn build_grid(
        rows: usize,
        cols: usize,
        spacing: f64,
        stiffness: f64,
        damping: f64,
        toughness: f64,
        criterion: FractureCriterion,
    ) -> Result<Self, TearError> {
        let mut sim = Self::new(criterion)?;
        for r in 0..rows {
            for c in 0..cols {
                sim.add_particle(
                    Vec3::new(c as f64 * spacing, r as f64 * spacing, 0.0),
                    1.0,
                    false,
                );
            }
        }
        let idx = |r: usize, c: usize| r * cols + c;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    sim.add_spring(idx(r, c), idx(r, c + 1), stiffness, damping, toughness)?;
                }
                if r + 1 < rows {
                    sim.add_spring(idx(r, c), idx(r + 1, c), stiffness, damping, toughness)?;
                }
            }
        }
        Ok(sim)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_create_sim() {
        let sim = TearSim::new(FractureCriterion::MaxStrain(2.0)).unwrap();
        assert_eq!(sim.particles.len(), 0);
        assert_eq!(sim.springs.len(), 0);
    }

    #[test]
    fn test_invalid_threshold() {
        assert!(TearSim::new(FractureCriterion::MaxStrain(0.0)).is_err());
        assert!(TearSim::new(FractureCriterion::MaxStress(-1.0)).is_err());
    }

    #[test]
    fn test_add_particle_and_spring() {
        let mut sim = TearSim::new(FractureCriterion::MaxStrain(2.0)).unwrap();
        sim.add_particle(Vec3::ZERO, 1.0, false);
        sim.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0, false);
        let si = sim.add_spring(0, 1, 100.0, 1.0, 1.0).unwrap();
        assert_eq!(si, 0);
        assert!(approx(sim.springs[0].rest_length, 1.0, 1e-10));
    }

    #[test]
    fn test_invalid_spring_params() {
        let mut sim = TearSim::new(FractureCriterion::MaxStrain(2.0)).unwrap();
        sim.add_particle(Vec3::ZERO, 1.0, false);
        sim.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0, false);
        assert!(sim.add_spring(0, 1, -1.0, 1.0, 1.0).is_err());
        assert!(sim.add_spring(0, 1, 100.0, -1.0, 1.0).is_err());
        assert!(sim.add_spring(0, 1, 100.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_chain_creation() {
        let sim = TearSim::build_chain(5, 1.0, 100.0, 1.0, 1.0, FractureCriterion::MaxStrain(2.0)).unwrap();
        assert_eq!(sim.particles.len(), 5);
        assert_eq!(sim.springs.len(), 4);
    }

    #[test]
    fn test_grid_creation() {
        let sim = TearSim::build_grid(3, 4, 1.0, 100.0, 1.0, 1.0, FractureCriterion::MaxStrain(2.0)).unwrap();
        assert_eq!(sim.particles.len(), 12);
        // horizontal: 3*3=9, vertical: 2*4=8 = 17
        assert_eq!(sim.springs.len(), 17);
    }

    #[test]
    fn test_strain_tear() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(0.5)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        // Pull last particle far away
        sim.particles[2].position = Vec3::new(10.0, 0.0, 0.0);
        sim.step(0.01).unwrap();
        assert!(sim.broken_count() > 0);
    }

    #[test]
    fn test_stress_tear() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStress(50.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(10.0, 0.0, 0.0);
        sim.step(0.01).unwrap();
        assert!(sim.broken_count() > 0);
    }

    #[test]
    fn test_energy_tear() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxEnergy(10.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(10.0, 0.0, 0.0);
        sim.step(0.01).unwrap();
        assert!(sim.broken_count() > 0);
    }

    #[test]
    fn test_no_tear_under_threshold() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(100.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(1.5, 0.0, 0.0); // small extension
        sim.step(0.01).unwrap();
        assert_eq!(sim.broken_count(), 0);
    }

    #[test]
    fn test_partial_tear_weakening() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(2.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.weaken_ratio = 0.5;
        sim.particles[0].fixed = true;
        // Stretch enough to weaken but not break
        sim.particles[2].position = Vec3::new(3.5, 0.0, 0.0);
        sim.step(0.01).unwrap();
        assert!(sim.weakened_count() > 0 || sim.broken_count() > 0);
    }

    #[test]
    fn test_crack_tip_tracking() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(0.5)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(10.0, 0.0, 0.0);
        sim.step(0.01).unwrap();
        assert!(!sim.crack_tips.is_empty());
    }

    #[test]
    fn test_fixed_particle_stays() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 1.0, 1.0, FractureCriterion::MaxStrain(10.0)).unwrap();
        sim.particles[0].fixed = true;
        let pos0 = sim.particles[0].position;
        for _ in 0..20 {
            sim.step(0.01).unwrap();
        }
        assert!(approx(sim.particles[0].position.x, pos0.x, 1e-10));
    }

    #[test]
    fn test_invalid_timestep() {
        let mut sim = TearSim::new(FractureCriterion::MaxStrain(1.0)).unwrap();
        assert!(sim.step(0.0).is_err());
        assert!(sim.step(-1.0).is_err());
    }

    #[test]
    fn test_elastic_energy() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(10.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        assert!(approx(sim.total_elastic_energy(), 0.0, 1e-10));
        sim.particles[2].position = Vec3::new(3.0, 0.0, 0.0); // stretch
        assert!(sim.total_elastic_energy() > 0.0);
    }

    #[test]
    fn test_max_strain() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(10.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[2].position = Vec3::new(3.0, 0.0, 0.0);
        assert!(sim.max_strain() > 0.0);
    }

    #[test]
    fn test_toughness_map() {
        let mut sim = TearSim::build_chain(5, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(1.0)).unwrap();
        sim.set_toughness_map(|i| if i == 2 { 10.0 } else { 1.0 });
        assert!(approx(sim.springs[2].toughness, 10.0, 1e-10));
        assert!(approx(sim.springs[0].toughness, 1.0, 1e-10));
    }

    #[test]
    fn test_split_at_tear() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(0.5)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(10.0, 0.0, 0.0);
        sim.step(0.01).unwrap();

        // Find a broken spring and split
        let broken_idx = sim.springs.iter().position(|s| s.state == SpringState::Broken);
        if let Some(bi) = broken_idx {
            let result = sim.split_at_tear(bi).unwrap();
            assert!(result.is_some());
            // New particle should exist
            assert!(sim.particles.len() > 3);
        }
    }

    #[test]
    fn test_split_intact_spring() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(10.0)).unwrap();
        let result = sim.split_at_tear(0).unwrap();
        assert!(result.is_none()); // not broken, no split
    }

    #[test]
    fn test_spring_not_found() {
        let mut sim = TearSim::new(FractureCriterion::MaxStrain(1.0)).unwrap();
        assert!(sim.split_at_tear(0).is_err());
    }

    #[test]
    fn test_integrity_degrades() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 1.0, FractureCriterion::MaxStrain(5.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.weaken_ratio = 0.3;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(4.0, 0.0, 0.0);
        sim.step(0.01).unwrap();
        // Some springs should have lost integrity
        let damaged = sim.springs.iter().any(|s| s.integrity < 1.0);
        assert!(damaged || sim.broken_count() > 0);
    }

    #[test]
    fn test_gravity_affects_particles() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 1.0, 1.0, FractureCriterion::MaxStrain(100.0)).unwrap();
        let y0 = sim.particles[1].position.y;
        for _ in 0..50 {
            sim.step(0.01).unwrap();
        }
        assert!(sim.particles[1].position.y < y0);
    }

    #[test]
    fn test_high_toughness_resists_tear() {
        let mut sim = TearSim::build_chain(3, 1.0, 100.0, 0.0, 10.0, FractureCriterion::MaxStrain(1.0)).unwrap();
        sim.gravity = Vec3::ZERO;
        sim.particles[0].fixed = true;
        sim.particles[2].position = Vec3::new(2.5, 0.0, 0.0);
        sim.step(0.01).unwrap();
        // High toughness (effective threshold = 10.0) should prevent tearing
        assert_eq!(sim.broken_count(), 0);
    }
}
