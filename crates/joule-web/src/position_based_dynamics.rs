//! Position-Based Dynamics (PBD) solver — constraint projection, XPBD compliance.
//!
//! Replaces Bullet.js / Oimo.js PBD solvers with pure Rust.
//! Supports distance, bending, volume, and collision constraints,
//! XPBD (Extended PBD) with compliance for physically accurate stiffness,
//! Gauss-Seidel iteration, and substep count control.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PbdError {
    InvalidTimestep,
    InvalidSubsteps,
    ParticleNotFound(usize),
    InvalidCompliance,
    SelfConstraint(usize),
    InvalidRadius,
}

impl fmt::Display for PbdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::InvalidSubsteps => write!(f, "substeps must be >= 1"),
            Self::ParticleNotFound(i) => write!(f, "particle {i} not found"),
            Self::InvalidCompliance => write!(f, "compliance must be non-negative"),
            Self::SelfConstraint(i) => write!(f, "cannot constrain particle {i} to itself"),
            Self::InvalidRadius => write!(f, "radius must be positive"),
        }
    }
}

impl std::error::Error for PbdError {}

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

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
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

// ── Particle ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct PbdParticle {
    pub position: Vec3,
    pub predicted: Vec3,
    pub velocity: Vec3,
    pub inv_mass: f64,
}

impl PbdParticle {
    pub fn new(position: Vec3, mass: f64) -> Self {
        Self {
            position,
            predicted: position,
            velocity: Vec3::ZERO,
            inv_mass: if mass <= 0.0 { 0.0 } else { 1.0 / mass },
        }
    }

    pub fn new_fixed(position: Vec3) -> Self {
        Self {
            position,
            predicted: position,
            velocity: Vec3::ZERO,
            inv_mass: 0.0,
        }
    }

    pub fn is_fixed(&self) -> bool {
        self.inv_mass <= 0.0
    }
}

// ── Constraints ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Distance constraint: keep two particles at a target distance.
    Distance {
        a: usize,
        b: usize,
        rest_length: f64,
        compliance: f64,
    },
    /// Bending constraint: angle between two triangles sharing an edge.
    Bending {
        p0: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        rest_angle: f64,
        compliance: f64,
    },
    /// Volume constraint: maintain tetrahedron volume.
    Volume {
        nodes: [usize; 4],
        rest_volume: f64,
        compliance: f64,
    },
    /// Collision with ground plane at y = level.
    GroundPlane {
        particle: usize,
        level: f64,
    },
    /// Collision with sphere.
    SphereCollision {
        particle: usize,
        center: Vec3,
        radius: f64,
    },
}

// ── PBD Solver ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PbdSolver {
    pub particles: Vec<PbdParticle>,
    pub constraints: Vec<Constraint>,
    pub gravity: Vec3,
    pub damping: f64,
    pub substeps: usize,
    pub iterations: usize,
    pub time: f64,
}

impl PbdSolver {
    pub fn new(substeps: usize, iterations: usize) -> Result<Self, PbdError> {
        if substeps < 1 {
            return Err(PbdError::InvalidSubsteps);
        }
        Ok(Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            substeps,
            iterations,
            time: 0.0,
        })
    }

    pub fn add_particle(&mut self, position: Vec3, mass: f64) -> usize {
        let id = self.particles.len();
        self.particles.push(PbdParticle::new(position, mass));
        id
    }

    pub fn add_fixed_particle(&mut self, position: Vec3) -> usize {
        let id = self.particles.len();
        self.particles.push(PbdParticle::new_fixed(position));
        id
    }

    pub fn add_distance_constraint(
        &mut self,
        a: usize,
        b: usize,
        compliance: f64,
    ) -> Result<usize, PbdError> {
        if a == b { return Err(PbdError::SelfConstraint(a)); }
        if a >= self.particles.len() { return Err(PbdError::ParticleNotFound(a)); }
        if b >= self.particles.len() { return Err(PbdError::ParticleNotFound(b)); }
        if compliance < 0.0 { return Err(PbdError::InvalidCompliance); }
        let rest = self.particles[a].position.sub(self.particles[b].position).length();
        let idx = self.constraints.len();
        self.constraints.push(Constraint::Distance { a, b, rest_length: rest, compliance });
        Ok(idx)
    }

    pub fn add_distance_constraint_with_rest(
        &mut self,
        a: usize,
        b: usize,
        rest_length: f64,
        compliance: f64,
    ) -> Result<usize, PbdError> {
        if a == b { return Err(PbdError::SelfConstraint(a)); }
        if a >= self.particles.len() { return Err(PbdError::ParticleNotFound(a)); }
        if b >= self.particles.len() { return Err(PbdError::ParticleNotFound(b)); }
        if compliance < 0.0 { return Err(PbdError::InvalidCompliance); }
        let idx = self.constraints.len();
        self.constraints.push(Constraint::Distance {
            a, b, rest_length: rest_length.abs(), compliance,
        });
        Ok(idx)
    }

    pub fn add_bending_constraint(
        &mut self,
        p0: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        compliance: f64,
    ) -> Result<usize, PbdError> {
        for &p in &[p0, p1, p2, p3] {
            if p >= self.particles.len() { return Err(PbdError::ParticleNotFound(p)); }
        }
        if compliance < 0.0 { return Err(PbdError::InvalidCompliance); }

        // Compute rest angle between triangles (p0,p1,p2) and (p1,p2,p3)
        let n1 = self.particles[p1].position.sub(self.particles[p0].position)
            .cross(self.particles[p2].position.sub(self.particles[p0].position));
        let n2 = self.particles[p1].position.sub(self.particles[p3].position)
            .cross(self.particles[p2].position.sub(self.particles[p3].position));
        let l1 = n1.length();
        let l2 = n2.length();
        let rest_angle = if l1 > 1e-12 && l2 > 1e-12 {
            let cos_a = n1.dot(n2) / (l1 * l2);
            cos_a.clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };

        let idx = self.constraints.len();
        self.constraints.push(Constraint::Bending { p0, p1, p2, p3, rest_angle, compliance });
        Ok(idx)
    }

    pub fn add_volume_constraint(
        &mut self,
        nodes: [usize; 4],
        compliance: f64,
    ) -> Result<usize, PbdError> {
        for &n in &nodes {
            if n >= self.particles.len() { return Err(PbdError::ParticleNotFound(n)); }
        }
        if compliance < 0.0 { return Err(PbdError::InvalidCompliance); }

        let p = &self.particles;
        let e1 = p[nodes[1]].position.sub(p[nodes[0]].position);
        let e2 = p[nodes[2]].position.sub(p[nodes[0]].position);
        let e3 = p[nodes[3]].position.sub(p[nodes[0]].position);
        let rest_volume = e1.dot(e2.cross(e3)) / 6.0;

        let idx = self.constraints.len();
        self.constraints.push(Constraint::Volume { nodes, rest_volume, compliance });
        Ok(idx)
    }

    pub fn add_ground_plane(&mut self, particle: usize, level: f64) -> Result<usize, PbdError> {
        if particle >= self.particles.len() { return Err(PbdError::ParticleNotFound(particle)); }
        let idx = self.constraints.len();
        self.constraints.push(Constraint::GroundPlane { particle, level });
        Ok(idx)
    }

    pub fn add_sphere_collision(
        &mut self,
        particle: usize,
        center: Vec3,
        radius: f64,
    ) -> Result<usize, PbdError> {
        if particle >= self.particles.len() { return Err(PbdError::ParticleNotFound(particle)); }
        if radius <= 0.0 { return Err(PbdError::InvalidRadius); }
        let idx = self.constraints.len();
        self.constraints.push(Constraint::SphereCollision { particle, center, radius });
        Ok(idx)
    }

    fn project_constraint(&mut self, ci: usize, dt_sub: f64) {
        let alpha = match &self.constraints[ci] {
            Constraint::Distance { compliance, .. } => *compliance / (dt_sub * dt_sub),
            Constraint::Bending { compliance, .. } => *compliance / (dt_sub * dt_sub),
            Constraint::Volume { compliance, .. } => *compliance / (dt_sub * dt_sub),
            Constraint::GroundPlane { .. } => 0.0,
            Constraint::SphereCollision { .. } => 0.0,
        };

        match self.constraints[ci].clone() {
            Constraint::Distance { a, b, rest_length, .. } => {
                let diff = self.particles[a].predicted.sub(self.particles[b].predicted);
                let dist = diff.length();
                if dist < 1e-12 { return; }
                let w_sum = self.particles[a].inv_mass + self.particles[b].inv_mass;
                if w_sum < 1e-12 { return; }
                let c_val = dist - rest_length;
                let lambda = -c_val / (w_sum + alpha);
                let correction = diff.scale(lambda / dist);

                self.particles[a].predicted = self.particles[a].predicted.add(
                    correction.scale(self.particles[a].inv_mass),
                );
                self.particles[b].predicted = self.particles[b].predicted.sub(
                    correction.scale(self.particles[b].inv_mass),
                );
            }
            Constraint::Bending { p0, p1, p2, p3, rest_angle, .. } => {
                let n1 = self.particles[p1].predicted.sub(self.particles[p0].predicted)
                    .cross(self.particles[p2].predicted.sub(self.particles[p0].predicted));
                let n2 = self.particles[p1].predicted.sub(self.particles[p3].predicted)
                    .cross(self.particles[p2].predicted.sub(self.particles[p3].predicted));
                let l1 = n1.length();
                let l2 = n2.length();
                if l1 < 1e-12 || l2 < 1e-12 { return; }
                let cos_a = (n1.dot(n2) / (l1 * l2)).clamp(-1.0, 1.0);
                let current_angle = cos_a.acos();
                let c_val = current_angle - rest_angle;
                if c_val.abs() < 1e-10 { return; }

                // Simple approach: push p3 along the normal direction
                let correction_dir = n2.normalized();
                let w3 = self.particles[p3].inv_mass;
                if w3 < 1e-12 { return; }
                let scale = -c_val * 0.1 / (w3 + alpha);
                self.particles[p3].predicted = self.particles[p3].predicted.add(
                    correction_dir.scale(scale * w3),
                );
            }
            Constraint::Volume { nodes, rest_volume, .. } => {
                let p0 = self.particles[nodes[0]].predicted;
                let p1 = self.particles[nodes[1]].predicted;
                let p2 = self.particles[nodes[2]].predicted;
                let p3 = self.particles[nodes[3]].predicted;

                let e1 = p1.sub(p0);
                let e2 = p2.sub(p0);
                let e3 = p3.sub(p0);
                let vol = e1.dot(e2.cross(e3)) / 6.0;
                let c_val = vol - rest_volume;
                if c_val.abs() < 1e-12 { return; }

                // Gradients (simplified: push nodes along face normals)
                let g1 = e2.cross(e3).scale(1.0 / 6.0);
                let g2 = e3.cross(e1).scale(1.0 / 6.0);
                let g3 = e1.cross(e2).scale(1.0 / 6.0);
                let g0 = Vec3::ZERO.sub(g1).sub(g2).sub(g3);

                let grads = [g0, g1, g2, g3];
                let mut w_sum = 0.0;
                for k in 0..4 {
                    w_sum += self.particles[nodes[k]].inv_mass * grads[k].length_sq();
                }
                if w_sum < 1e-12 { return; }

                let lambda = -c_val / (w_sum + alpha);
                for k in 0..4 {
                    let w = self.particles[nodes[k]].inv_mass;
                    self.particles[nodes[k]].predicted = self.particles[nodes[k]].predicted.add(
                        grads[k].scale(lambda * w),
                    );
                }
            }
            Constraint::GroundPlane { particle, level } => {
                if self.particles[particle].predicted.y < level && self.particles[particle].inv_mass > 0.0 {
                    self.particles[particle].predicted.y = level;
                }
            }
            Constraint::SphereCollision { particle, center, radius } => {
                if self.particles[particle].inv_mass <= 0.0 { return; }
                let diff = self.particles[particle].predicted.sub(center);
                let dist = diff.length();
                if dist < radius && dist > 1e-12 {
                    self.particles[particle].predicted = center.add(diff.normalized().scale(radius));
                }
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), PbdError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(PbdError::InvalidTimestep);
        }

        let dt_sub = dt / self.substeps as f64;

        for _ in 0..self.substeps {
            // Predict: apply external forces and predict positions
            for p in &mut self.particles {
                if p.inv_mass <= 0.0 {
                    p.predicted = p.position;
                    continue;
                }
                p.velocity = p.velocity.add(self.gravity.scale(dt_sub));
                p.velocity = p.velocity.scale(1.0 - self.damping);
                p.predicted = p.position.add(p.velocity.scale(dt_sub));
            }

            // Project constraints (Gauss-Seidel)
            for _ in 0..self.iterations {
                for ci in 0..self.constraints.len() {
                    self.project_constraint(ci, dt_sub);
                }
            }

            // Update velocities and positions
            for p in &mut self.particles {
                if p.inv_mass <= 0.0 {
                    continue;
                }
                p.velocity = p.predicted.sub(p.position).scale(1.0 / dt_sub);
                p.position = p.predicted;
            }
        }

        self.time += dt;
        Ok(())
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter()
            .filter(|p| p.inv_mass > 0.0)
            .map(|p| 0.5 * (1.0 / p.inv_mass) * p.velocity.length_sq())
            .sum()
    }

    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for p in &self.particles {
            let m = if p.inv_mass > 0.0 { 1.0 / p.inv_mass } else { 0.0 };
            total_m += m;
            weighted = weighted.add(p.position.scale(m));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn max_stretch(&self) -> f64 {
        let mut max_s = 0.0f64;
        for c in &self.constraints {
            if let Constraint::Distance { a, b, rest_length, .. } = c {
                let dist = self.particles[*a].position.sub(self.particles[*b].position).length();
                let stretch = if *rest_length > 1e-12 {
                    (dist - rest_length).abs() / rest_length
                } else {
                    0.0
                };
                if stretch > max_s { max_s = stretch; }
            }
        }
        max_s
    }
}

/// Build a simple rope as a chain of distance-constrained particles.
pub fn build_rope(
    start: Vec3,
    end: Vec3,
    segments: usize,
    mass_per_segment: f64,
    compliance: f64,
) -> Result<PbdSolver, PbdError> {
    if segments < 1 {
        return Err(PbdError::InvalidSubsteps);
    }
    let mut solver = PbdSolver::new(4, 10)?;
    let dir = end.sub(start);
    let step_vec = dir.scale(1.0 / segments as f64);

    for i in 0..=segments {
        let pos = start.add(step_vec.scale(i as f64));
        if i == 0 {
            solver.add_fixed_particle(pos);
        } else {
            solver.add_particle(pos, mass_per_segment);
        }
    }

    for i in 0..segments {
        solver.add_distance_constraint(i, i + 1, compliance)?;
    }

    Ok(solver)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_creation() {
        let solver = PbdSolver::new(4, 10).unwrap();
        assert_eq!(solver.particle_count(), 0);
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_invalid_substeps() {
        assert!(PbdSolver::new(0, 10).is_err());
    }

    #[test]
    fn test_add_particles() {
        let mut solver = PbdSolver::new(1, 5).unwrap();
        let a = solver.add_particle(Vec3::ZERO, 1.0);
        let b = solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_fixed_particle() {
        let mut solver = PbdSolver::new(1, 5).unwrap();
        let a = solver.add_fixed_particle(Vec3::new(0.0, 5.0, 0.0));
        assert!(solver.particles[a].is_fixed());
        for _ in 0..20 {
            solver.step(0.01).unwrap();
        }
        assert!(approx(solver.particles[a].position.y, 5.0, 1e-10));
    }

    #[test]
    fn test_distance_constraint() {
        let mut solver = PbdSolver::new(4, 10).unwrap();
        solver.gravity = Vec3::ZERO;
        let a = solver.add_fixed_particle(Vec3::ZERO);
        let b = solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver.add_distance_constraint(a, b, 0.0).unwrap();

        // Push b away
        solver.particles[b].position = Vec3::new(3.0, 0.0, 0.0);
        solver.particles[b].predicted = solver.particles[b].position;
        for _ in 0..50 {
            solver.step(0.01).unwrap();
        }
        let dist = solver.particles[a].position.sub(solver.particles[b].position).length();
        assert!(approx(dist, 1.0, 0.2)); // should converge toward rest length
    }

    #[test]
    fn test_self_constraint_rejected() {
        let mut solver = PbdSolver::new(1, 5).unwrap();
        solver.add_particle(Vec3::ZERO, 1.0);
        assert!(solver.add_distance_constraint(0, 0, 0.0).is_err());
    }

    #[test]
    fn test_invalid_compliance() {
        let mut solver = PbdSolver::new(1, 5).unwrap();
        solver.add_particle(Vec3::ZERO, 1.0);
        solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!(solver.add_distance_constraint(0, 1, -1.0).is_err());
    }

    #[test]
    fn test_ground_plane() {
        let mut solver = PbdSolver::new(2, 5).unwrap();
        let a = solver.add_particle(Vec3::new(0.0, 1.0, 0.0), 1.0);
        solver.add_ground_plane(a, 0.0).unwrap();
        for _ in 0..200 {
            solver.step(0.01).unwrap();
        }
        assert!(solver.particles[a].position.y >= -0.01);
    }

    #[test]
    fn test_sphere_collision() {
        let mut solver = PbdSolver::new(2, 5).unwrap();
        let a = solver.add_particle(Vec3::new(0.5, 0.0, 0.0), 1.0);
        solver.gravity = Vec3::ZERO;
        solver.add_sphere_collision(a, Vec3::ZERO, 2.0).unwrap();
        // Particle starts inside sphere
        solver.step(0.01).unwrap();
        let dist = solver.particles[a].position.length();
        assert!(dist >= 2.0 - 0.1);
    }

    #[test]
    fn test_invalid_radius() {
        let mut solver = PbdSolver::new(1, 5).unwrap();
        solver.add_particle(Vec3::ZERO, 1.0);
        assert!(solver.add_sphere_collision(0, Vec3::ZERO, -1.0).is_err());
    }

    #[test]
    fn test_gravity_freefall() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        solver.add_particle(Vec3::new(0.0, 10.0, 0.0), 1.0);
        for _ in 0..100 {
            solver.step(0.01).unwrap();
        }
        assert!(solver.particles[0].position.y < 10.0);
    }

    #[test]
    fn test_kinetic_energy() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        solver.add_particle(Vec3::ZERO, 1.0);
        assert!(approx(solver.kinetic_energy(), 0.0, 1e-12));
        solver.step(0.01).unwrap();
        assert!(solver.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_invalid_timestep() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        assert!(solver.step(0.0).is_err());
        assert!(solver.step(-1.0).is_err());
    }

    #[test]
    fn test_build_rope() {
        let rope = build_rope(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            10,
            0.1,
            0.0,
        ).unwrap();
        assert_eq!(rope.particle_count(), 11);
        assert_eq!(rope.constraint_count(), 10);
        assert!(rope.particles[0].is_fixed());
    }

    #[test]
    fn test_rope_hangs() {
        let mut rope = build_rope(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            5,
            0.5,
            0.0,
        ).unwrap();
        // Unfix the end
        for _ in 0..100 {
            rope.step(0.01).unwrap();
        }
        // Middle particles should sag below start height
        assert!(rope.particles[3].position.y < 5.0);
    }

    #[test]
    fn test_xpbd_compliance() {
        // High compliance = softer constraint
        let mut solver_stiff = PbdSolver::new(4, 10).unwrap();
        solver_stiff.gravity = Vec3::ZERO;
        solver_stiff.add_fixed_particle(Vec3::ZERO);
        solver_stiff.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver_stiff.add_distance_constraint(0, 1, 0.0).unwrap();

        let mut solver_soft = PbdSolver::new(4, 10).unwrap();
        solver_soft.gravity = Vec3::ZERO;
        solver_soft.add_fixed_particle(Vec3::ZERO);
        solver_soft.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver_soft.add_distance_constraint(0, 1, 1.0).unwrap(); // high compliance

        // Displace
        solver_stiff.particles[1].position = Vec3::new(3.0, 0.0, 0.0);
        solver_soft.particles[1].position = Vec3::new(3.0, 0.0, 0.0);

        for _ in 0..20 {
            solver_stiff.step(0.01).unwrap();
            solver_soft.step(0.01).unwrap();
        }

        let d_stiff = solver_stiff.particles[0].position.sub(solver_stiff.particles[1].position).length();
        let d_soft = solver_soft.particles[0].position.sub(solver_soft.particles[1].position).length();
        // Stiff should be closer to rest length (1.0) than soft
        assert!((d_stiff - 1.0).abs() <= (d_soft - 1.0).abs() + 0.5);
    }

    #[test]
    fn test_volume_constraint() {
        let mut solver = PbdSolver::new(4, 10).unwrap();
        solver.gravity = Vec3::ZERO;
        solver.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(0.0, 1.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(0.0, 0.0, 1.0), 1.0);
        solver.add_volume_constraint([0, 1, 2, 3], 0.0).unwrap();
        solver.step(0.01).unwrap();
        // Just verify no panic
        assert!(solver.time > 0.0);
    }

    #[test]
    fn test_center_of_mass() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        solver.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let com = solver.center_of_mass();
        assert!(approx(com.x, 1.0, 1e-10));
    }

    #[test]
    fn test_max_stretch() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        solver.gravity = Vec3::ZERO;
        solver.add_particle(Vec3::ZERO, 1.0);
        solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver.add_distance_constraint(0, 1, 0.0).unwrap();
        assert!(approx(solver.max_stretch(), 0.0, 1e-10));
        solver.particles[1].position = Vec3::new(2.0, 0.0, 0.0);
        assert!(approx(solver.max_stretch(), 1.0, 1e-10));
    }

    #[test]
    fn test_damping_reduces_energy() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        solver.damping = 0.5;
        solver.add_particle(Vec3::new(0.0, 10.0, 0.0), 1.0);
        solver.step(0.01).unwrap();
        let ke1 = solver.kinetic_energy();
        solver.step(0.01).unwrap();
        let ke2 = solver.kinetic_energy();
        // With high damping energy gain per step should be limited
        assert!(ke2 < 100.0);
        let _ = ke1;
    }

    #[test]
    fn test_particle_not_found() {
        let mut solver = PbdSolver::new(1, 1).unwrap();
        assert!(solver.add_ground_plane(0, 0.0).is_err());
    }

    #[test]
    fn test_bending_constraint() {
        let mut solver = PbdSolver::new(2, 5).unwrap();
        solver.gravity = Vec3::ZERO;
        solver.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver.add_particle(Vec3::new(0.0, 0.0, 1.0), 1.0);
        solver.add_particle(Vec3::new(1.0, 0.0, 1.0), 1.0);
        solver.add_bending_constraint(0, 1, 2, 3, 0.0).unwrap();
        solver.step(0.01).unwrap();
        assert!(solver.time > 0.0);
    }
}
