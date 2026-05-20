//! Smoothed Particle Hydrodynamics (SPH) fluid simulation.
//!
//! Particles carry position, velocity, density, and pressure. Kernel functions
//! (poly6 for density, spiky for pressure gradient, viscosity kernel) enable
//! meshfree fluid dynamics. A spatial hash grid accelerates neighbor search.
//! Density is computed from kernel sums, pressure from the Tait equation of state,
//! and forces include pressure gradient, viscosity, and boundary repulsion.
//! Semi-implicit Euler integration advances the simulation.

use std::collections::HashMap;
use std::fmt;

// ── Constants ─────────────────────────────────────────────────

const PI: f64 = std::f64::consts::PI;

// ── Errors ────────────────────────────────────────────────────

/// SPH simulation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SphError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Particle index out of range.
    ParticleOutOfRange(usize),
    /// Simulation diverged (NaN or Inf detected).
    Diverged(String),
    /// Empty particle set.
    NoParticles,
}

impl fmt::Display for SphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ParticleOutOfRange(idx) => write!(f, "particle index out of range: {idx}"),
            Self::Diverged(msg) => write!(f, "simulation diverged: {msg}"),
            Self::NoParticles => write!(f, "no particles in simulation"),
        }
    }
}

impl std::error::Error for SphError {}

// ── 2D Vector ─────────────────────────────────────────────────

/// Simple 2D vector for particle positions and velocities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::zero()
        } else {
            Self { x: self.x / len, y: self.y / len }
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }
}

// ── Particle ──────────────────────────────────────────────────

/// A single SPH particle carrying all fluid quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub density: f64,
    pub pressure: f64,
    pub force: Vec2,
    pub mass: f64,
    pub id: usize,
}

impl Particle {
    pub fn new(id: usize, position: Vec2, mass: f64) -> Self {
        Self {
            position,
            velocity: Vec2::zero(),
            density: 0.0,
            pressure: 0.0,
            force: Vec2::zero(),
            mass,
            id,
        }
    }

    /// Kinetic energy: 0.5 * m * v^2.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.length_sq()
    }
}

// ── SPH Configuration ─────────────────────────────────────────

/// Configuration for the SPH simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct SphConfig {
    /// Smoothing radius (kernel support).
    pub smoothing_radius: f64,
    /// Rest density of the fluid (kg/m^3).
    pub rest_density: f64,
    /// Gas constant for Tait equation of state.
    pub gas_constant: f64,
    /// Viscosity coefficient (mu).
    pub viscosity: f64,
    /// Simulation timestep (seconds).
    pub timestep: f64,
    /// Gravity vector.
    pub gravity: Vec2,
    /// Domain bounds: (min, max).
    pub domain_min: Vec2,
    pub domain_max: Vec2,
    /// Boundary repulsion stiffness.
    pub boundary_stiffness: f64,
    /// Boundary damping factor.
    pub boundary_damping: f64,
    /// Tait equation exponent (gamma, typically 7 for water).
    pub tait_exponent: f64,
}

impl SphConfig {
    pub fn validate(&self) -> Result<(), SphError> {
        if self.smoothing_radius <= 0.0 {
            return Err(SphError::InvalidConfig("smoothing_radius must be > 0".into()));
        }
        if self.rest_density <= 0.0 {
            return Err(SphError::InvalidConfig("rest_density must be > 0".into()));
        }
        if self.gas_constant <= 0.0 {
            return Err(SphError::InvalidConfig("gas_constant must be > 0".into()));
        }
        if self.timestep <= 0.0 {
            return Err(SphError::InvalidConfig("timestep must be > 0".into()));
        }
        if self.domain_min.x >= self.domain_max.x || self.domain_min.y >= self.domain_max.y {
            return Err(SphError::InvalidConfig("domain_min must be < domain_max".into()));
        }
        Ok(())
    }
}

impl Default for SphConfig {
    fn default() -> Self {
        Self {
            smoothing_radius: 0.1,
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 1.0,
            timestep: 0.001,
            gravity: Vec2::new(0.0, -9.81),
            domain_min: Vec2::new(0.0, 0.0),
            domain_max: Vec2::new(1.0, 1.0),
            boundary_stiffness: 10000.0,
            boundary_damping: 0.5,
            tait_exponent: 7.0,
        }
    }
}

// ── Kernel Functions ──────────────────────────────────────────

/// SPH kernel functions in 2D.
pub struct Kernels;

impl Kernels {
    /// Poly6 kernel for density estimation.
    /// W(r, h) = (4 / (pi * h^8)) * (h^2 - r^2)^3  for r <= h
    pub fn poly6(r_sq: f64, h: f64) -> f64 {
        let h_sq = h * h;
        if r_sq >= h_sq {
            return 0.0;
        }
        let diff = h_sq - r_sq;
        let coeff = 4.0 / (PI * h.powi(8));
        coeff * diff * diff * diff
    }

    /// Gradient of the spiky kernel for pressure forces.
    /// grad W_spiky = -(10 / (pi * h^5)) * (h - r)^2 * (r_vec / r)  for r <= h
    /// Returns the scalar multiplier; caller applies direction.
    pub fn spiky_gradient(r: f64, h: f64) -> f64 {
        if r >= h || r < 1e-12 {
            return 0.0;
        }
        let diff = h - r;
        let coeff = -10.0 / (PI * h.powi(5));
        coeff * diff * diff / r
    }

    /// Laplacian of the viscosity kernel.
    /// lap W_visc = (40 / (pi * h^5)) * (1 - r/h)  for r <= h
    pub fn viscosity_laplacian(r: f64, h: f64) -> f64 {
        if r >= h {
            return 0.0;
        }
        let coeff = 40.0 / (PI * h.powi(5));
        coeff * (1.0 - r / h)
    }
}

// ── Spatial Hash Grid ─────────────────────────────────────────

/// Grid cell key for spatial hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey(i64, i64);

/// Spatial hash grid for fast neighbor queries.
pub struct SpatialGrid {
    cell_size: f64,
    cells: HashMap<CellKey, Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn key_for(&self, pos: &Vec2) -> CellKey {
        CellKey(
            (pos.x / self.cell_size).floor() as i64,
            (pos.y / self.cell_size).floor() as i64,
        )
    }

    /// Clear and rebuild the grid from particle positions.
    pub fn rebuild(&mut self, particles: &[Particle]) {
        self.cells.clear();
        for (i, p) in particles.iter().enumerate() {
            let key = self.key_for(&p.position);
            self.cells.entry(key).or_default().push(i);
        }
    }

    /// Find all particle indices within `radius` of `pos`.
    pub fn neighbors(&self, pos: &Vec2, radius: f64) -> Vec<usize> {
        let radius_sq = radius * radius;
        let min_cx = ((pos.x - radius) / self.cell_size).floor() as i64;
        let max_cx = ((pos.x + radius) / self.cell_size).floor() as i64;
        let min_cy = ((pos.y - radius) / self.cell_size).floor() as i64;
        let max_cy = ((pos.y + radius) / self.cell_size).floor() as i64;

        let mut result = Vec::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                if let Some(indices) = self.cells.get(&CellKey(cx, cy)) {
                    result.extend(indices.iter().copied());
                }
            }
        }
        result.retain(|idx| {
            // We don't have particles here, so caller must filter by actual distance.
            // For grid, return all in neighboring cells.
            let _ = idx;
            true
        });
        // Remove duplicates
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

// ── SPH Simulation ────────────────────────────────────────────

/// Simulation statistics for a single step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepStats {
    pub total_kinetic_energy: f64,
    pub avg_density: f64,
    pub max_velocity: f64,
    pub particle_count: usize,
    pub step_number: u64,
}

/// The SPH fluid simulation engine.
pub struct SphSimulation {
    pub config: SphConfig,
    pub particles: Vec<Particle>,
    grid: SpatialGrid,
    step_count: u64,
}

impl SphSimulation {
    /// Create a new simulation with the given configuration.
    pub fn new(config: SphConfig) -> Result<Self, SphError> {
        config.validate()?;
        let grid = SpatialGrid::new(config.smoothing_radius);
        Ok(Self {
            config,
            particles: Vec::new(),
            grid,
            step_count: 0,
        })
    }

    /// Add a single particle at the given position.
    pub fn add_particle(&mut self, position: Vec2, mass: f64) -> usize {
        let id = self.particles.len();
        self.particles.push(Particle::new(id, position, mass));
        id
    }

    /// Create a block of particles in a rectangular region.
    pub fn create_block(
        &mut self,
        min: Vec2,
        max: Vec2,
        spacing: f64,
        mass: f64,
    ) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut y = min.y;
        while y <= max.y {
            let mut x = min.x;
            while x <= max.x {
                let id = self.add_particle(Vec2::new(x, y), mass);
                ids.push(id);
                x += spacing;
            }
            y += spacing;
        }
        ids
    }

    /// Number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Current simulation step.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Compute density for all particles using Poly6 kernel.
    fn compute_densities(&mut self) {
        let h = self.config.smoothing_radius;
        let positions: Vec<Vec2> = self.particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();
        let n = self.particles.len();

        let mut densities = vec![0.0_f64; n];

        for i in 0..n {
            let neighbors = self.grid.neighbors(&positions[i], h);
            let mut density = 0.0;
            for &j in &neighbors {
                let r_sq = positions[i].sub(&positions[j]).length_sq();
                density += masses[j] * Kernels::poly6(r_sq, h);
            }
            // Ensure density is at least some minimum to avoid division by zero
            densities[i] = density.max(1.0);
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.density = densities[i];
        }
    }

    /// Compute pressure from density using the Tait equation of state.
    /// P = k * ((rho / rho0)^gamma - 1)
    fn compute_pressures(&mut self) {
        let rho0 = self.config.rest_density;
        let k = self.config.gas_constant;
        let gamma = self.config.tait_exponent;

        for p in &mut self.particles {
            let ratio = p.density / rho0;
            p.pressure = k * (ratio.powf(gamma) - 1.0);
            if p.pressure < 0.0 {
                p.pressure = 0.0; // Clamp negative pressures
            }
        }
    }

    /// Compute pressure and viscosity forces on all particles.
    fn compute_forces(&mut self) {
        let h = self.config.smoothing_radius;
        let mu = self.config.viscosity;
        let gravity = self.config.gravity;

        let n = self.particles.len();
        let positions: Vec<Vec2> = self.particles.iter().map(|p| p.position).collect();
        let velocities: Vec<Vec2> = self.particles.iter().map(|p| p.velocity).collect();
        let densities: Vec<f64> = self.particles.iter().map(|p| p.density).collect();
        let pressures: Vec<f64> = self.particles.iter().map(|p| p.pressure).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();

        let mut forces = vec![Vec2::zero(); n];

        for i in 0..n {
            let neighbors = self.grid.neighbors(&positions[i], h);
            let mut f_pressure = Vec2::zero();
            let mut f_viscosity = Vec2::zero();

            for &j in &neighbors {
                if i == j {
                    continue;
                }

                let r_vec = positions[i].sub(&positions[j]);
                let r = r_vec.length();
                if r < 1e-12 || r >= h {
                    continue;
                }

                // Pressure force: -m_j * (P_i + P_j) / (2 * rho_j) * grad W_spiky
                let spiky_grad = Kernels::spiky_gradient(r, h);
                let pressure_term = (pressures[i] + pressures[j]) / (2.0 * densities[j]);
                let fp = r_vec.scale(masses[j] * pressure_term * spiky_grad);
                f_pressure = f_pressure.add(&fp);

                // Viscosity force: mu * m_j * (v_j - v_i) / rho_j * lap W_visc
                let visc_lap = Kernels::viscosity_laplacian(r, h);
                let vel_diff = velocities[j].sub(&velocities[i]);
                let fv = vel_diff.scale(mu * masses[j] * visc_lap / densities[j]);
                f_viscosity = f_viscosity.add(&fv);
            }

            // Total force = pressure + viscosity + gravity * density
            let f_gravity = gravity.scale(densities[i]);
            forces[i] = f_pressure.add(&f_viscosity).add(&f_gravity);
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.force = forces[i];
        }
    }

    /// Apply boundary repulsion forces to keep particles inside the domain.
    fn apply_boundary_forces(&mut self) {
        let stiffness = self.config.boundary_stiffness;
        let damping = self.config.boundary_damping;
        let dmin = self.config.domain_min;
        let dmax = self.config.domain_max;
        let h = self.config.smoothing_radius;

        for p in &mut self.particles {
            // Left wall
            let d = p.position.x - dmin.x;
            if d < h {
                let pen = h - d;
                p.force.x += stiffness * pen - damping * p.velocity.x;
            }
            // Right wall
            let d = dmax.x - p.position.x;
            if d < h {
                let pen = h - d;
                p.force.x -= stiffness * pen + damping * p.velocity.x;
            }
            // Bottom wall
            let d = p.position.y - dmin.y;
            if d < h {
                let pen = h - d;
                p.force.y += stiffness * pen - damping * p.velocity.y;
            }
            // Top wall
            let d = dmax.y - p.position.y;
            if d < h {
                let pen = h - d;
                p.force.y -= stiffness * pen + damping * p.velocity.y;
            }
        }
    }

    /// Semi-implicit Euler integration: update velocity then position.
    fn integrate(&mut self) {
        let dt = self.config.timestep;
        let dmin = self.config.domain_min;
        let dmax = self.config.domain_max;

        for p in &mut self.particles {
            // Acceleration = force / density
            let accel = if p.density > 1e-8 {
                p.force.scale(1.0 / p.density)
            } else {
                Vec2::zero()
            };

            // Semi-implicit: update velocity first
            p.velocity = p.velocity.add(&accel.scale(dt));

            // Update position
            p.position = p.position.add(&p.velocity.scale(dt));

            // Hard clamp to domain
            if p.position.x < dmin.x {
                p.position.x = dmin.x;
                p.velocity.x *= -0.3;
            }
            if p.position.x > dmax.x {
                p.position.x = dmax.x;
                p.velocity.x *= -0.3;
            }
            if p.position.y < dmin.y {
                p.position.y = dmin.y;
                p.velocity.y *= -0.3;
            }
            if p.position.y > dmax.y {
                p.position.y = dmax.y;
                p.velocity.y *= -0.3;
            }
        }
    }

    /// Check for NaN/Inf in any particle.
    fn check_divergence(&self) -> Result<(), SphError> {
        for p in &self.particles {
            if !p.position.x.is_finite() || !p.position.y.is_finite() {
                return Err(SphError::Diverged(format!("particle {} position NaN/Inf", p.id)));
            }
            if !p.velocity.x.is_finite() || !p.velocity.y.is_finite() {
                return Err(SphError::Diverged(format!("particle {} velocity NaN/Inf", p.id)));
            }
        }
        Ok(())
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self) -> Result<StepStats, SphError> {
        if self.particles.is_empty() {
            return Err(SphError::NoParticles);
        }

        // 1. Build spatial hash grid
        self.grid.rebuild(&self.particles);

        // 2. Compute density from kernel sums
        self.compute_densities();

        // 3. Compute pressure from equation of state
        self.compute_pressures();

        // 4. Compute pressure + viscosity forces
        self.compute_forces();

        // 5. Apply boundary repulsion
        self.apply_boundary_forces();

        // 6. Integrate (semi-implicit Euler)
        self.integrate();

        // 7. Check for divergence
        self.check_divergence()?;

        self.step_count += 1;

        Ok(self.compute_stats())
    }

    /// Run multiple simulation steps.
    pub fn run(&mut self, steps: u64) -> Result<Vec<StepStats>, SphError> {
        let mut stats = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            stats.push(self.step()?);
        }
        Ok(stats)
    }

    /// Compute current simulation statistics.
    pub fn compute_stats(&self) -> StepStats {
        let n = self.particles.len();
        let total_ke: f64 = self.particles.iter().map(|p| p.kinetic_energy()).sum();
        let avg_density = if n > 0 {
            self.particles.iter().map(|p| p.density).sum::<f64>() / n as f64
        } else {
            0.0
        };
        let max_vel = self.particles.iter()
            .map(|p| p.velocity.length())
            .fold(0.0_f64, f64::max);

        StepStats {
            total_kinetic_energy: total_ke,
            avg_density,
            max_velocity: max_vel,
            particle_count: n,
            step_number: self.step_count,
        }
    }

    /// Get a particle by index.
    pub fn get_particle(&self, idx: usize) -> Result<&Particle, SphError> {
        self.particles.get(idx).ok_or(SphError::ParticleOutOfRange(idx))
    }

    /// Total kinetic energy of the system.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.particles.iter().map(|p| p.kinetic_energy()).sum()
    }

    /// Center of mass of the fluid.
    pub fn center_of_mass(&self) -> Vec2 {
        if self.particles.is_empty() {
            return Vec2::zero();
        }
        let total_mass: f64 = self.particles.iter().map(|p| p.mass).sum();
        let mut com = Vec2::zero();
        for p in &self.particles {
            com = com.add(&p.position.scale(p.mass));
        }
        com.scale(1.0 / total_mass)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sim() -> SphSimulation {
        SphSimulation::new(SphConfig::default()).unwrap()
    }

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.length() - 5.0).abs() < 1e-10);
        let n = a.normalized();
        assert!((n.length() - 1.0).abs() < 1e-10);
        let b = Vec2::new(1.0, 2.0);
        assert!((a.dot(&b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_zero_normalize() {
        let z = Vec2::zero();
        let n = z.normalized();
        assert!((n.length()).abs() < 1e-10);
    }

    #[test]
    fn test_config_validation_ok() {
        assert!(SphConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_radius() {
        let mut cfg = SphConfig::default();
        cfg.smoothing_radius = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_density() {
        let mut cfg = SphConfig::default();
        cfg.rest_density = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_domain() {
        let mut cfg = SphConfig::default();
        cfg.domain_min = Vec2::new(1.0, 0.0);
        cfg.domain_max = Vec2::new(0.0, 1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_poly6_kernel_within_radius() {
        let val = Kernels::poly6(0.0, 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_poly6_kernel_outside_radius() {
        let val = Kernels::poly6(1.01, 1.0);
        assert!((val).abs() < 1e-12);
    }

    #[test]
    fn test_poly6_monotonic_decrease() {
        let v1 = Kernels::poly6(0.0, 1.0);
        let v2 = Kernels::poly6(0.25, 1.0);
        let v3 = Kernels::poly6(0.5, 1.0);
        assert!(v1 > v2);
        assert!(v2 > v3);
    }

    #[test]
    fn test_spiky_gradient_at_zero() {
        let val = Kernels::spiky_gradient(0.0, 1.0);
        assert!((val).abs() < 1e-12);
    }

    #[test]
    fn test_spiky_gradient_inside() {
        let val = Kernels::spiky_gradient(0.5, 1.0);
        assert!(val < 0.0); // Negative (pointing inward)
    }

    #[test]
    fn test_viscosity_laplacian_inside() {
        let val = Kernels::viscosity_laplacian(0.5, 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_viscosity_laplacian_outside() {
        let val = Kernels::viscosity_laplacian(1.5, 1.0);
        assert!((val).abs() < 1e-12);
    }

    #[test]
    fn test_add_particle() {
        let mut sim = default_sim();
        let id = sim.add_particle(Vec2::new(0.5, 0.5), 1.0);
        assert_eq!(id, 0);
        assert_eq!(sim.particle_count(), 1);
    }

    #[test]
    fn test_create_block() {
        let mut sim = default_sim();
        let ids = sim.create_block(Vec2::new(0.2, 0.2), Vec2::new(0.4, 0.4), 0.1, 1.0);
        assert_eq!(ids.len(), 9); // 3x3 grid
    }

    #[test]
    fn test_step_no_particles() {
        let mut sim = default_sim();
        let result = sim.step();
        assert!(matches!(result, Err(SphError::NoParticles)));
    }

    #[test]
    fn test_single_step() {
        let mut sim = default_sim();
        sim.create_block(Vec2::new(0.3, 0.3), Vec2::new(0.5, 0.5), 0.05, 0.01);
        let stats = sim.step().unwrap();
        assert!(stats.particle_count > 0);
        assert_eq!(stats.step_number, 1);
    }

    #[test]
    fn test_multiple_steps_no_divergence() {
        let mut sim = default_sim();
        sim.create_block(Vec2::new(0.3, 0.3), Vec2::new(0.5, 0.5), 0.05, 0.01);
        let result = sim.run(10);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_particles_stay_in_domain() {
        let mut sim = default_sim();
        sim.create_block(Vec2::new(0.3, 0.3), Vec2::new(0.5, 0.5), 0.05, 0.01);
        sim.run(20).unwrap();
        let dmin = sim.config.domain_min;
        let dmax = sim.config.domain_max;
        for p in &sim.particles {
            assert!(p.position.x >= dmin.x);
            assert!(p.position.x <= dmax.x);
            assert!(p.position.y >= dmin.y);
            assert!(p.position.y <= dmax.y);
        }
    }

    #[test]
    fn test_gravity_pulls_down() {
        let mut sim = default_sim();
        sim.add_particle(Vec2::new(0.5, 0.8), 0.01);
        let initial_y = sim.particles[0].position.y;
        sim.run(5).unwrap();
        let final_y = sim.particles[0].position.y;
        assert!(final_y < initial_y);
    }

    #[test]
    fn test_spatial_grid_rebuild() {
        let mut grid = SpatialGrid::new(0.1);
        let particles = vec![
            Particle::new(0, Vec2::new(0.05, 0.05), 1.0),
            Particle::new(1, Vec2::new(0.15, 0.05), 1.0),
            Particle::new(2, Vec2::new(0.95, 0.95), 1.0),
        ];
        grid.rebuild(&particles);
        assert!(grid.cell_count() >= 2);
    }

    #[test]
    fn test_spatial_grid_neighbors() {
        let mut grid = SpatialGrid::new(0.1);
        let particles = vec![
            Particle::new(0, Vec2::new(0.05, 0.05), 1.0),
            Particle::new(1, Vec2::new(0.06, 0.05), 1.0),
            Particle::new(2, Vec2::new(0.95, 0.95), 1.0),
        ];
        grid.rebuild(&particles);
        let nbrs = grid.neighbors(&Vec2::new(0.05, 0.05), 0.1);
        assert!(nbrs.contains(&0));
        assert!(nbrs.contains(&1));
    }

    #[test]
    fn test_center_of_mass() {
        let mut sim = default_sim();
        sim.add_particle(Vec2::new(0.0, 0.0), 1.0);
        sim.add_particle(Vec2::new(1.0, 0.0), 1.0);
        let com = sim.center_of_mass();
        assert!((com.x - 0.5).abs() < 1e-10);
        assert!((com.y).abs() < 1e-10);
    }

    #[test]
    fn test_kinetic_energy_at_rest() {
        let mut sim = default_sim();
        sim.add_particle(Vec2::new(0.5, 0.5), 1.0);
        assert!((sim.total_kinetic_energy()).abs() < 1e-12);
    }

    #[test]
    fn test_get_particle_out_of_range() {
        let sim = default_sim();
        assert!(matches!(sim.get_particle(0), Err(SphError::ParticleOutOfRange(0))));
    }

    #[test]
    fn test_step_count_increments() {
        let mut sim = default_sim();
        sim.add_particle(Vec2::new(0.5, 0.5), 0.01);
        assert_eq!(sim.step_count(), 0);
        sim.step().unwrap();
        assert_eq!(sim.step_count(), 1);
        sim.step().unwrap();
        assert_eq!(sim.step_count(), 2);
    }
}
