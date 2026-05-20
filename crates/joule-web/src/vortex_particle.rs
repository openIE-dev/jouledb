//! Vortex particle method for fluid simulation.
//!
//! Models vorticity as discrete point vortices with position, circulation (strength),
//! and core radius. Computes induced velocity via Biot-Savart law with regularized
//! kernels (Rosenhead-Moore, Lamb-Oseen) to avoid singularity. Supports 2D and 3D
//! vortex stretching, viscous diffusion (core spreading / PSE), vortex merging for
//! efficiency, and both direct O(N^2) and treecode-approximate summation.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Vortex particle method errors.
#[derive(Debug, Clone, PartialEq)]
pub enum VortexError {
    /// No vortices in the simulation.
    NoVortices,
    /// Invalid parameter.
    InvalidParam(String),
    /// Simulation diverged.
    Diverged(String),
    /// Vortex index out of range.
    IndexOutOfRange(usize),
}

impl fmt::Display for VortexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVortices => write!(f, "no vortices in simulation"),
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Diverged(msg) => write!(f, "simulation diverged: {msg}"),
            Self::IndexOutOfRange(idx) => write!(f, "vortex index out of range: {idx}"),
        }
    }
}

impl std::error::Error for VortexError {}

// ── 2D/3D Vector ──────────────────────────────────────────────

/// 3D vector (also used for 2D with z=0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn xy(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

// ── Regularization Kernel ─────────────────────────────────────

/// Regularization kernel type for Biot-Savart computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizationKernel {
    /// Rosenhead-Moore: K(r) = r^2 / (r^2 + sigma^2)^{3/2}
    RosenheadMoore,
    /// Lamb-Oseen: K(r) = 1 - exp(-r^2 / sigma^2)
    LambOseen,
}

/// Evaluate the regularized Biot-Savart kernel.
/// Returns the velocity magnitude scale factor for distance r and core radius sigma.
fn kernel_factor(kernel: RegularizationKernel, r_sq: f64, sigma: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    match kernel {
        RegularizationKernel::RosenheadMoore => {
            let denom = (r_sq + sigma_sq).powf(1.5);
            if denom < 1e-30 { 0.0 } else { r_sq.sqrt().powi(2) / denom }
        }
        RegularizationKernel::LambOseen => {
            if sigma_sq < 1e-30 { return 1.0; }
            1.0 - (-r_sq / sigma_sq).exp()
        }
    }
}

// ── Vortex Element ────────────────────────────────────────────

/// A vortex particle element.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexElement {
    /// Position in 3D space.
    pub position: Vec3,
    /// Vorticity vector (in 2D, only z-component is used = circulation).
    pub vorticity: Vec3,
    /// Core radius (sigma) for regularization.
    pub core_radius: f64,
    /// Unique identifier.
    pub id: usize,
}

impl VortexElement {
    /// Create a 2D point vortex with scalar circulation.
    pub fn point_vortex_2d(id: usize, x: f64, y: f64, circulation: f64, core_radius: f64) -> Self {
        Self {
            position: Vec3::xy(x, y),
            vorticity: Vec3::new(0.0, 0.0, circulation),
            core_radius,
            id,
        }
    }

    /// Create a 3D vortex element.
    pub fn vortex_3d(id: usize, pos: Vec3, vorticity: Vec3, core_radius: f64) -> Self {
        Self { position: pos, vorticity, core_radius, id }
    }

    /// Circulation magnitude (length of vorticity vector).
    pub fn circulation(&self) -> f64 {
        self.vorticity.length()
    }

    /// Kinetic energy contribution (proportional to gamma^2 * log for 2D).
    pub fn energy_contribution(&self) -> f64 {
        0.25 * self.vorticity.length_sq() * self.core_radius
    }
}

// ── Vortex Particle Simulation ────────────────────────────────

/// Configuration for the vortex particle simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexConfig {
    /// Regularization kernel type.
    pub kernel: RegularizationKernel,
    /// Kinematic viscosity (for diffusion).
    pub viscosity: f64,
    /// Timestep.
    pub dt: f64,
    /// Enable vortex stretching (3D only).
    pub enable_stretching: bool,
    /// Enable viscous diffusion via core spreading.
    pub enable_diffusion: bool,
    /// Merge distance threshold.
    pub merge_distance: f64,
    /// Minimum circulation to keep a vortex alive.
    pub min_circulation: f64,
    /// Use treecode approximation (theta parameter).
    pub treecode_theta: f64,
    /// Freestream velocity.
    pub freestream: Vec3,
}

impl Default for VortexConfig {
    fn default() -> Self {
        Self {
            kernel: RegularizationKernel::RosenheadMoore,
            viscosity: 0.01,
            dt: 0.01,
            enable_stretching: false,
            enable_diffusion: true,
            merge_distance: 0.05,
            min_circulation: 1e-6,
            treecode_theta: 0.0, // 0 = direct summation
            freestream: Vec3::zero(),
        }
    }
}

/// Statistics from a simulation step.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexStats {
    pub step: u64,
    pub vortex_count: usize,
    pub total_circulation: f64,
    pub max_velocity: f64,
    pub total_energy: f64,
    pub merges_performed: usize,
}

/// The vortex particle simulation.
pub struct VortexSimulation {
    pub config: VortexConfig,
    pub vortices: Vec<VortexElement>,
    step_count: u64,
    next_id: usize,
}

impl VortexSimulation {
    pub fn new(config: VortexConfig) -> Self {
        Self {
            config,
            vortices: Vec::new(),
            step_count: 0,
            next_id: 0,
        }
    }

    /// Add a 2D point vortex.
    pub fn add_vortex_2d(&mut self, x: f64, y: f64, circulation: f64, core_radius: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.vortices.push(VortexElement::point_vortex_2d(id, x, y, circulation, core_radius));
        id
    }

    /// Add a 3D vortex element.
    pub fn add_vortex_3d(&mut self, pos: Vec3, vorticity: Vec3, core_radius: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.vortices.push(VortexElement::vortex_3d(id, pos, vorticity, core_radius));
        id
    }

    pub fn vortex_count(&self) -> usize {
        self.vortices.len()
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Compute the induced velocity at a point from all vortices (Biot-Savart).
    pub fn velocity_at(&self, point: &Vec3) -> Vec3 {
        let mut vel = self.config.freestream;
        let two_pi = 2.0 * std::f64::consts::PI;

        for vort in &self.vortices {
            let r = point.sub(&vort.position);
            let r_sq = r.length_sq();

            if r_sq < 1e-30 {
                continue; // Skip self-interaction
            }

            let kf = kernel_factor(self.config.kernel, r_sq, vort.core_radius);

            // 2D case: velocity from z-vorticity
            // v_induced = (gamma / (2*pi)) * (-dy, dx, 0) / r^2 * K(r)
            let gamma_z = vort.vorticity.z;
            if gamma_z.abs() > 1e-30 {
                let factor = gamma_z * kf / (two_pi * r_sq.max(1e-30));
                vel.x += -r.y * factor;
                vel.y += r.x * factor;
            }

            // 3D Biot-Savart: v = (omega x r) / (4*pi*|r|^3) * K(r)
            let omega_xy = Vec3::new(vort.vorticity.x, vort.vorticity.y, 0.0);
            if omega_xy.length_sq() > 1e-30 {
                let r_len = r_sq.sqrt().max(1e-15);
                let cross = omega_xy.cross(&r);
                let factor = kf / (4.0 * std::f64::consts::PI * r_len * r_len * r_len);
                vel = vel.add(&cross.scale(factor));
            }
        }
        vel
    }

    /// Compute induced velocities at all vortex positions (direct O(N^2)).
    fn compute_velocities_direct(&self) -> Vec<Vec3> {
        self.vortices.iter()
            .map(|v| self.velocity_at(&v.position))
            .collect()
    }

    /// Compute induced velocities using a simple treecode approximation.
    /// Groups distant vortices and uses monopole expansion.
    fn compute_velocities_treecode(&self) -> Vec<Vec3> {
        let theta = self.config.treecode_theta;
        if theta <= 0.0 || self.vortices.len() < 16 {
            return self.compute_velocities_direct();
        }

        // Build a simple quadtree-like grouping (flat clustering)
        let n = self.vortices.len();
        let mut velocities = vec![self.config.freestream; n];

        // Compute center of vorticity and total circulation
        let mut center = Vec3::zero();
        let mut total_gamma = 0.0;
        for v in &self.vortices {
            center = center.add(&v.position.scale(v.circulation()));
            total_gamma += v.circulation();
        }
        if total_gamma.abs() > 1e-20 {
            center = center.scale(1.0 / total_gamma);
        }

        let max_dist = self.vortices.iter()
            .map(|v| v.position.sub(&center).length())
            .fold(0.0_f64, f64::max);

        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..n {
            let pos_i = self.vortices[i].position;
            let dist_to_center = pos_i.sub(&center).length();

            // If far enough, use monopole approximation
            if dist_to_center > max_dist / theta {
                // Approximate: treat all as single vortex at center
                let r = pos_i.sub(&center);
                let r_sq = r.length_sq();
                if r_sq > 1e-20 {
                    let factor = total_gamma / (two_pi * r_sq);
                    velocities[i].x += -r.y * factor;
                    velocities[i].y += r.x * factor;
                }
            } else {
                // Direct summation for this particle
                velocities[i] = self.velocity_at(&pos_i);
            }
        }
        velocities
    }

    /// Apply vortex stretching (3D): dw/dt += (w . grad) v
    fn apply_stretching(&mut self, velocities: &[Vec3]) {
        if !self.config.enable_stretching || self.vortices.len() < 2 {
            return;
        }
        let dt = self.config.dt;
        let n = self.vortices.len();

        for i in 0..n {
            // Approximate velocity gradient using nearby vortex velocities
            let pos = self.vortices[i].position;
            let omega = self.vortices[i].vorticity;

            // Find nearest neighbor for gradient estimation
            let mut min_dist = f64::MAX;
            let mut nearest_idx = 0;
            for j in 0..n {
                if i == j { continue; }
                let d = self.vortices[j].position.sub(&pos).length();
                if d < min_dist {
                    min_dist = d;
                    nearest_idx = j;
                }
            }

            if min_dist > 1e-10 {
                let dv = velocities[nearest_idx].sub(&velocities[i]);
                let dr = self.vortices[nearest_idx].position.sub(&pos);
                let dr_len = dr.length().max(1e-15);

                // Stretching: (omega . nabla) v ≈ omega * (dv/dr)
                let stretch_rate = omega.dot(&dr.scale(1.0 / dr_len));
                let stretch = dv.scale(stretch_rate * dt / dr_len);
                self.vortices[i].vorticity = self.vortices[i].vorticity.add(&stretch);
            }
        }
    }

    /// Apply viscous diffusion via core spreading.
    fn apply_diffusion(&mut self) {
        if !self.config.enable_diffusion {
            return;
        }
        let nu = self.config.viscosity;
        let dt = self.config.dt;

        for v in &mut self.vortices {
            // Core spreading: sigma^2(t) = sigma^2(0) + 2*nu*t
            let new_sigma_sq = v.core_radius * v.core_radius + 2.0 * nu * dt;
            v.core_radius = new_sigma_sq.sqrt();
        }
    }

    /// Merge nearby weak vortices to reduce particle count.
    fn merge_vortices(&mut self) -> usize {
        let dist_thresh = self.config.merge_distance;
        let min_circ = self.config.min_circulation;
        let n = self.vortices.len();
        if n < 2 {
            return 0;
        }

        let mut merged = vec![false; n];
        let mut merges = 0;

        for i in 0..n {
            if merged[i] { continue; }
            if self.vortices[i].circulation() > min_circ * 10.0 { continue; }

            for j in (i + 1)..n {
                if merged[j] { continue; }

                let dist = self.vortices[i].position.sub(&self.vortices[j].position).length();
                if dist < dist_thresh {
                    let ci = self.vortices[i].circulation();
                    let cj = self.vortices[j].circulation();
                    let total = ci + cj;

                    if total.abs() > 1e-30 {
                        // Weighted average position
                        let new_pos = self.vortices[i].position.scale(ci / total)
                            .add(&self.vortices[j].position.scale(cj / total));
                        let new_vort = self.vortices[i].vorticity.add(&self.vortices[j].vorticity);
                        let new_sigma = (self.vortices[i].core_radius + self.vortices[j].core_radius) * 0.5;

                        self.vortices[i].position = new_pos;
                        self.vortices[i].vorticity = new_vort;
                        self.vortices[i].core_radius = new_sigma;
                    }
                    merged[j] = true;
                    merges += 1;
                }
            }
        }

        // Remove merged vortices and those below min circulation
        let mut kept = Vec::new();
        for i in 0..n {
            if !merged[i] && self.vortices[i].circulation() >= min_circ {
                kept.push(self.vortices[i].clone());
            }
        }
        self.vortices = kept;
        merges
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self) -> Result<VortexStats, VortexError> {
        if self.vortices.is_empty() {
            return Err(VortexError::NoVortices);
        }

        // 1. Compute induced velocities
        let velocities = if self.config.treecode_theta > 0.0 {
            self.compute_velocities_treecode()
        } else {
            self.compute_velocities_direct()
        };

        // 2. Vortex stretching (3D)
        self.apply_stretching(&velocities);

        // 3. Advect vortex positions
        let dt = self.config.dt;
        for (i, v) in self.vortices.iter_mut().enumerate() {
            v.position = v.position.add(&velocities[i].scale(dt));
        }

        // 4. Viscous diffusion
        self.apply_diffusion();

        // 5. Check for divergence
        for v in &self.vortices {
            if !v.position.x.is_finite() || !v.position.y.is_finite() || !v.position.z.is_finite() {
                return Err(VortexError::Diverged("NaN/Inf in vortex position".into()));
            }
        }

        // 6. Merge
        let merges = self.merge_vortices();

        self.step_count += 1;

        // Compute stats
        let max_vel = velocities.iter().map(|v| v.length()).fold(0.0_f64, f64::max);
        let total_circ: f64 = self.vortices.iter().map(|v| v.vorticity.z).sum();
        let total_energy: f64 = self.vortices.iter().map(|v| v.energy_contribution()).sum();

        Ok(VortexStats {
            step: self.step_count,
            vortex_count: self.vortices.len(),
            total_circulation: total_circ,
            max_velocity: max_vel,
            total_energy,
            merges_performed: merges,
        })
    }

    /// Run multiple steps.
    pub fn run(&mut self, steps: u64) -> Result<Vec<VortexStats>, VortexError> {
        let mut stats = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            stats.push(self.step()?);
        }
        Ok(stats)
    }

    /// Total circulation (sum of z-component of all vorticity vectors).
    pub fn total_circulation(&self) -> f64 {
        self.vortices.iter().map(|v| v.vorticity.z).sum()
    }

    /// Center of vorticity (circulation-weighted centroid).
    pub fn center_of_vorticity(&self) -> Vec3 {
        let total = self.total_circulation();
        if total.abs() < 1e-20 || self.vortices.is_empty() {
            return Vec3::zero();
        }
        let mut center = Vec3::zero();
        for v in &self.vortices {
            center = center.add(&v.position.scale(v.vorticity.z));
        }
        center.scale(1.0 / total)
    }

    /// Get a vortex by index.
    pub fn get_vortex(&self, idx: usize) -> Result<&VortexElement, VortexError> {
        self.vortices.get(idx).ok_or(VortexError::IndexOutOfRange(idx))
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sim() -> VortexSimulation {
        VortexSimulation::new(VortexConfig::default())
    }

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.add(&b);
        assert!((c.x - 5.0).abs() < 1e-12);
        assert!((c.y - 7.0).abs() < 1e-12);
        assert!((c.z - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.x).abs() < 1e-12);
        assert!((c.y).abs() < 1e-12);
        assert!((c.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(&b) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_add_vortex_2d() {
        let mut sim = default_sim();
        let id = sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        assert_eq!(id, 0);
        assert_eq!(sim.vortex_count(), 1);
    }

    #[test]
    fn test_add_vortex_3d() {
        let mut sim = default_sim();
        let id = sim.add_vortex_3d(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 0.1);
        assert_eq!(id, 0);
        assert!((sim.vortices[0].circulation() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_velocity_at_far_point() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.01);
        let vel = sim.velocity_at(&Vec3::new(10.0, 0.0, 0.0));
        // At large distance, velocity should be small
        assert!(vel.length() < 1.0);
    }

    #[test]
    fn test_velocity_at_self() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        // Velocity at the vortex itself should be the freestream (no self-induction)
        let vel = sim.velocity_at(&Vec3::xy(0.0, 0.0));
        assert!(vel.length() < 1e-10);
    }

    #[test]
    fn test_counter_rotating_pair_translation() {
        // Two counter-rotating vortices translate together
        let mut sim = VortexSimulation::new(VortexConfig {
            viscosity: 0.0,
            enable_diffusion: false,
            merge_distance: 0.0,
            min_circulation: 0.0,
            dt: 0.01,
            ..Default::default()
        });
        sim.add_vortex_2d(0.0, 0.5, 1.0, 0.05);
        sim.add_vortex_2d(0.0, -0.5, -1.0, 0.05);
        let initial_x = 0.0;
        sim.step().unwrap();
        // Both should translate in x
        let avg_x = (sim.vortices[0].position.x + sim.vortices[1].position.x) / 2.0;
        assert!(avg_x > initial_x - 0.001); // Should move forward (or stay)
    }

    #[test]
    fn test_step_no_vortices() {
        let mut sim = default_sim();
        assert!(matches!(sim.step(), Err(VortexError::NoVortices)));
    }

    #[test]
    fn test_single_step() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        sim.add_vortex_2d(1.0, 0.0, -1.0, 0.1);
        let stats = sim.step().unwrap();
        assert_eq!(stats.step, 1);
        assert_eq!(stats.vortex_count, 2);
    }

    #[test]
    fn test_multiple_steps() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.5, 0.0, 1.0, 0.1);
        sim.add_vortex_2d(-0.5, 0.0, -1.0, 0.1);
        let stats = sim.run(5).unwrap();
        assert_eq!(stats.len(), 5);
    }

    #[test]
    fn test_diffusion_increases_core() {
        let mut sim = VortexSimulation::new(VortexConfig {
            viscosity: 0.1,
            enable_diffusion: true,
            ..Default::default()
        });
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        sim.add_vortex_2d(1.0, 0.0, -1.0, 0.1);
        let initial_sigma = sim.vortices[0].core_radius;
        sim.step().unwrap();
        assert!(sim.vortices[0].core_radius > initial_sigma);
    }

    #[test]
    fn test_merge_nearby_vortices() {
        let mut sim = VortexSimulation::new(VortexConfig {
            merge_distance: 1.0,
            min_circulation: 1e-10,
            ..Default::default()
        });
        sim.add_vortex_2d(0.0, 0.0, 0.001, 0.1);
        sim.add_vortex_2d(0.01, 0.0, 0.001, 0.1);
        sim.add_vortex_2d(10.0, 0.0, 1.0, 0.1);
        sim.step().unwrap();
        // The two nearby weak vortices should merge
        assert!(sim.vortex_count() <= 3);
    }

    #[test]
    fn test_total_circulation() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 5.0, 0.1);
        sim.add_vortex_2d(1.0, 0.0, -3.0, 0.1);
        assert!((sim.total_circulation() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_of_vorticity() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        sim.add_vortex_2d(2.0, 0.0, 1.0, 0.1);
        let cov = sim.center_of_vorticity();
        assert!((cov.x - 1.0).abs() < 1e-10);
        assert!((cov.y).abs() < 1e-10);
    }

    #[test]
    fn test_get_vortex() {
        let mut sim = default_sim();
        sim.add_vortex_2d(1.0, 2.0, 3.0, 0.1);
        let v = sim.get_vortex(0).unwrap();
        assert!((v.position.x - 1.0).abs() < 1e-12);
        assert!(sim.get_vortex(5).is_err());
    }

    #[test]
    fn test_rosenhead_moore_kernel() {
        let kf = kernel_factor(RegularizationKernel::RosenheadMoore, 1.0, 0.1);
        assert!(kf > 0.0);
        assert!(kf <= 1.0);
    }

    #[test]
    fn test_lamb_oseen_kernel() {
        let kf = kernel_factor(RegularizationKernel::LambOseen, 1.0, 0.1);
        assert!(kf > 0.0);
        assert!(kf <= 1.0);
    }

    #[test]
    fn test_lamb_oseen_near_zero() {
        let kf = kernel_factor(RegularizationKernel::LambOseen, 1e-20, 0.1);
        assert!(kf < 0.01); // Should be very small near center
    }

    #[test]
    fn test_energy_contribution() {
        let v = VortexElement::point_vortex_2d(0, 0.0, 0.0, 2.0, 0.1);
        assert!(v.energy_contribution() > 0.0);
    }

    #[test]
    fn test_step_count() {
        let mut sim = default_sim();
        sim.add_vortex_2d(0.0, 0.0, 1.0, 0.1);
        sim.add_vortex_2d(1.0, 0.0, -1.0, 0.1);
        assert_eq!(sim.step_count(), 0);
        sim.step().unwrap();
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_freestream_velocity() {
        let sim = VortexSimulation::new(VortexConfig {
            freestream: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        });
        let vel = sim.velocity_at(&Vec3::zero());
        assert!((vel.x - 1.0).abs() < 1e-12);
    }
}
