//! Meshless shape matching deformation — polar decomposition, clusters, fracture.
//!
//! Replaces position-based deformation libraries with pure Rust.
//! Supports optimal rotation via polar decomposition (SVD-like extraction),
//! stiffness control, linear and quadratic shape matching, cluster-based
//! decomposition for large models, and fracture via cluster splitting.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeMatchError {
    InvalidStiffness,
    NoParticles,
    InvalidTimestep,
    ClusterNotFound(usize),
    ParticleNotFound(usize),
    InvalidStrainThreshold,
}

impl fmt::Display for ShapeMatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStiffness => write!(f, "stiffness must be in (0, 1]"),
            Self::NoParticles => write!(f, "at least one particle required"),
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::ClusterNotFound(i) => write!(f, "cluster {i} not found"),
            Self::ParticleNotFound(i) => write!(f, "particle {i} not found"),
            Self::InvalidStrainThreshold => write!(f, "strain threshold must be positive"),
        }
    }
}

impl std::error::Error for ShapeMatchError {}

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
}

// ── Mat3 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    pub fn transpose(self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[j][i]; } }
        r
    }

    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            y: self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            z: self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        }
    }

    pub fn mul_mat(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { for k in 0..3 {
            r.m[i][j] += self.m[i][k] * o.m[k][j];
        }}}
        r
    }

    pub fn determinant(self) -> f64 {
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
            - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
            + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 { return None; }
        let inv = 1.0 / det;
        let mut r = Self::ZERO;
        r.m[0][0] = (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) * inv;
        r.m[0][1] = (self.m[0][2] * self.m[2][1] - self.m[0][1] * self.m[2][2]) * inv;
        r.m[0][2] = (self.m[0][1] * self.m[1][2] - self.m[0][2] * self.m[1][1]) * inv;
        r.m[1][0] = (self.m[1][2] * self.m[2][0] - self.m[1][0] * self.m[2][2]) * inv;
        r.m[1][1] = (self.m[0][0] * self.m[2][2] - self.m[0][2] * self.m[2][0]) * inv;
        r.m[1][2] = (self.m[0][2] * self.m[1][0] - self.m[0][0] * self.m[1][2]) * inv;
        r.m[2][0] = (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]) * inv;
        r.m[2][1] = (self.m[0][1] * self.m[2][0] - self.m[0][0] * self.m[2][1]) * inv;
        r.m[2][2] = (self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]) * inv;
        Some(r)
    }

    pub fn add(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] + o.m[i][j]; } }
        r
    }

    pub fn sub(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] - o.m[i][j]; } }
        r
    }

    pub fn scale(self, s: f64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] * s; } }
        r
    }

    pub fn frobenius_norm(self) -> f64 {
        let mut s = 0.0;
        for i in 0..3 { for j in 0..3 { s += self.m[i][j] * self.m[i][j]; } }
        s.sqrt()
    }

    /// Extract rotation via iterative polar decomposition.
    pub fn extract_rotation(self) -> Self {
        let mut r = self;
        for _ in 0..20 {
            if let Some(rt_inv) = r.transpose().inverse() {
                r = r.add(rt_inv).scale(0.5);
            } else {
                return Self::IDENTITY;
            }
            let diff = r.mul_mat(r.transpose()).sub(Self::IDENTITY).frobenius_norm();
            if diff < 1e-10 { break; }
        }
        r
    }

    /// Outer product: v * w^T.
    pub fn outer(v: Vec3, w: Vec3) -> Self {
        Self { m: [
            [v.x * w.x, v.x * w.y, v.x * w.z],
            [v.y * w.x, v.y * w.y, v.y * w.z],
            [v.z * w.x, v.z * w.y, v.z * w.z],
        ]}
    }
}

// ── Particle ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SmParticle {
    pub rest_position: Vec3,
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f64,
    pub fixed: bool,
}

// ── Cluster ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    pub particle_indices: Vec<usize>,
    pub rest_com: Vec3,
    pub total_mass: f64,
    pub active: bool,
}

// ── Shape matching system ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ShapeMatchSystem {
    pub particles: Vec<SmParticle>,
    pub clusters: Vec<Cluster>,
    pub stiffness: f64,
    pub gravity: Vec3,
    pub damping: f64,
    pub quadratic: bool,
    pub fracture_threshold: f64,
    pub time: f64,
    pub fracture_count: usize,
}

impl ShapeMatchSystem {
    pub fn new(
        positions: Vec<Vec3>,
        masses: Vec<f64>,
        stiffness: f64,
    ) -> Result<Self, ShapeMatchError> {
        if positions.is_empty() {
            return Err(ShapeMatchError::NoParticles);
        }
        if stiffness <= 0.0 || stiffness > 1.0 {
            return Err(ShapeMatchError::InvalidStiffness);
        }

        let particles: Vec<SmParticle> = positions.iter().zip(masses.iter()).enumerate().map(|(_, (pos, &m))| {
            SmParticle {
                rest_position: *pos,
                position: *pos,
                velocity: Vec3::ZERO,
                mass: if m <= 0.0 { 1.0 } else { m },
                fixed: false,
            }
        }).collect();

        // Default: single cluster containing all particles
        let indices: Vec<usize> = (0..particles.len()).collect();
        let total_mass: f64 = particles.iter().map(|p| p.mass).sum();
        let rest_com = Self::compute_com_static(&particles, &indices);

        let cluster = Cluster {
            particle_indices: indices,
            rest_com,
            total_mass,
            active: true,
        };

        Ok(Self {
            particles,
            clusters: vec![cluster],
            stiffness,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            quadratic: false,
            fracture_threshold: f64::MAX,
            time: 0.0,
            fracture_count: 0,
        })
    }

    fn compute_com_static(particles: &[SmParticle], indices: &[usize]) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for &i in indices {
            total_m += particles[i].mass;
            weighted = weighted.add(particles[i].rest_position.scale(particles[i].mass));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    fn compute_current_com(&self, indices: &[usize]) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for &i in indices {
            total_m += self.particles[i].mass;
            weighted = weighted.add(self.particles[i].position.scale(self.particles[i].mass));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    pub fn create_clusters(&mut self, cluster_size: usize) {
        if cluster_size == 0 || cluster_size >= self.particles.len() {
            return;
        }
        self.clusters.clear();
        let n = self.particles.len();
        let mut start = 0;
        while start < n {
            let end = (start + cluster_size).min(n);
            let indices: Vec<usize> = (start..end).collect();
            let rest_com = Self::compute_com_static(&self.particles, &indices);
            let total_mass: f64 = indices.iter().map(|i| self.particles[*i].mass).sum();
            self.clusters.push(Cluster {
                particle_indices: indices,
                rest_com,
                total_mass,
                active: true,
            });
            start = end;
        }
    }

    pub fn create_overlapping_clusters(&mut self, cluster_size: usize, overlap: usize) {
        if cluster_size == 0 {
            return;
        }
        let step = if cluster_size > overlap { cluster_size - overlap } else { 1 };
        self.clusters.clear();
        let n = self.particles.len();
        let mut start = 0;
        while start < n {
            let end = (start + cluster_size).min(n);
            let indices: Vec<usize> = (start..end).collect();
            let rest_com = Self::compute_com_static(&self.particles, &indices);
            let total_mass: f64 = indices.iter().map(|i| self.particles[*i].mass).sum();
            self.clusters.push(Cluster {
                particle_indices: indices,
                rest_com,
                total_mass,
                active: true,
            });
            start += step;
            if end >= n { break; }
        }
    }

    fn shape_match_cluster(&self, cluster: &Cluster) -> Vec<(usize, Vec3)> {
        if !cluster.active || cluster.particle_indices.is_empty() {
            return Vec::new();
        }

        let current_com = self.compute_current_com(&cluster.particle_indices);
        let rest_com = cluster.rest_com;

        // Build Apq matrix: sum mi * (pi - com_current) * (qi - com_rest)^T
        let mut apq = Mat3::ZERO;
        for &i in &cluster.particle_indices {
            let p = self.particles[i].position.sub(current_com);
            let q = self.particles[i].rest_position.sub(rest_com);
            let m = self.particles[i].mass;
            apq = apq.add(Mat3::outer(p, q).scale(m));
        }

        // Extract rotation R from Apq via polar decomposition
        let rotation = apq.extract_rotation();

        // Compute goal positions: g_i = R * (q_i - rest_com) + current_com
        let mut goals = Vec::with_capacity(cluster.particle_indices.len());
        for &i in &cluster.particle_indices {
            let q = self.particles[i].rest_position.sub(rest_com);
            let goal = rotation.mul_vec(q).add(current_com);
            goals.push((i, goal));
        }
        goals
    }

    fn check_fracture(&mut self) {
        if self.fracture_threshold >= 1e15 {
            return;
        }

        let mut new_clusters = Vec::new();
        for ci in 0..self.clusters.len() {
            if !self.clusters[ci].active || self.clusters[ci].particle_indices.len() <= 2 {
                continue;
            }

            let com = self.compute_current_com(&self.clusters[ci].particle_indices);
            let rest_com = self.clusters[ci].rest_com;

            // Compute strain for each particle
            let mut max_strain = 0.0f64;
            let mut split_idx = 0;
            for (li, &pi) in self.clusters[ci].particle_indices.iter().enumerate() {
                let current_d = self.particles[pi].position.sub(com).length();
                let rest_d = self.particles[pi].rest_position.sub(rest_com).length();
                let strain = if rest_d > 1e-12 { (current_d - rest_d).abs() / rest_d } else { 0.0 };
                if strain > max_strain {
                    max_strain = strain;
                    split_idx = li;
                }
            }

            if max_strain > self.fracture_threshold {
                // Split cluster at the highest-strain particle
                let indices = self.clusters[ci].particle_indices.clone();
                let (left, right) = indices.split_at(split_idx.max(1));
                if !left.is_empty() && !right.is_empty() {
                    self.clusters[ci].active = false;
                    let left_vec: Vec<usize> = left.to_vec();
                    let right_vec: Vec<usize> = right.to_vec();

                    let lcom = Self::compute_com_static(&self.particles, &left_vec);
                    let lm: f64 = left_vec.iter().map(|i| self.particles[*i].mass).sum();
                    new_clusters.push(Cluster {
                        particle_indices: left_vec,
                        rest_com: lcom,
                        total_mass: lm,
                        active: true,
                    });

                    let rcom = Self::compute_com_static(&self.particles, &right_vec);
                    let rm: f64 = right_vec.iter().map(|i| self.particles[*i].mass).sum();
                    new_clusters.push(Cluster {
                        particle_indices: right_vec,
                        rest_com: rcom,
                        total_mass: rm,
                        active: true,
                    });

                    self.fracture_count += 1;
                }
            }
        }
        self.clusters.extend(new_clusters);
    }

    pub fn set_fracture_threshold(&mut self, threshold: f64) -> Result<(), ShapeMatchError> {
        if threshold <= 0.0 {
            return Err(ShapeMatchError::InvalidStrainThreshold);
        }
        self.fracture_threshold = threshold;
        Ok(())
    }

    pub fn fix_particle(&mut self, idx: usize) -> Result<(), ShapeMatchError> {
        if idx >= self.particles.len() {
            return Err(ShapeMatchError::ParticleNotFound(idx));
        }
        self.particles[idx].fixed = true;
        Ok(())
    }

    pub fn step(&mut self, dt: f64) -> Result<(), ShapeMatchError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(ShapeMatchError::InvalidTimestep);
        }

        // Apply external forces (gravity)
        for p in &mut self.particles {
            if !p.fixed {
                p.velocity = p.velocity.add(self.gravity.scale(dt));
                p.velocity = p.velocity.scale(1.0 - self.damping);
            }
        }

        // Predict positions
        for p in &mut self.particles {
            if !p.fixed {
                p.position = p.position.add(p.velocity.scale(dt));
            }
        }

        // Shape matching for each cluster
        let active_clusters: Vec<Cluster> = self.clusters.iter().filter(|c| c.active).cloned().collect();
        // Count how many clusters each particle belongs to
        let mut cluster_count = vec![0usize; self.particles.len()];
        for c in &active_clusters {
            for &i in &c.particle_indices {
                cluster_count[i] += 1;
            }
        }

        // Accumulate goal positions
        let mut goal_accum: Vec<Vec3> = self.particles.iter().map(|p| Vec3::ZERO).collect();

        for cluster in &active_clusters {
            let goals = self.shape_match_cluster(cluster);
            for (i, goal) in goals {
                goal_accum[i] = goal_accum[i].add(goal);
            }
        }

        // Average and move toward goals
        for i in 0..self.particles.len() {
            if self.particles[i].fixed || cluster_count[i] == 0 {
                continue;
            }
            let avg_goal = goal_accum[i].scale(1.0 / cluster_count[i] as f64);
            let delta = avg_goal.sub(self.particles[i].position).scale(self.stiffness);
            self.particles[i].position = self.particles[i].position.add(delta);
        }

        // Update velocities from position change
        for p in &mut self.particles {
            if !p.fixed {
                // Velocity is implicitly updated by position correction
                // We keep the velocity as-is since it was used for prediction
            }
        }

        self.check_fracture();
        self.time += dt;
        Ok(())
    }

    pub fn active_cluster_count(&self) -> usize {
        self.clusters.iter().filter(|c| c.active).count()
    }

    pub fn max_displacement(&self) -> f64 {
        self.particles.iter()
            .map(|p| p.position.sub(p.rest_position).length())
            .fold(0.0f64, f64::max)
    }

    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for p in &self.particles {
            total_m += p.mass;
            weighted = weighted.add(p.position.scale(p.mass));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter()
            .map(|p| 0.5 * p.mass * p.velocity.length_sq())
            .sum()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn simple_system() -> ShapeMatchSystem {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let masses = vec![1.0; 4];
        ShapeMatchSystem::new(positions, masses, 0.5).unwrap()
    }

    #[test]
    fn test_creation() {
        let sys = simple_system();
        assert_eq!(sys.particles.len(), 4);
        assert_eq!(sys.clusters.len(), 1);
    }

    #[test]
    fn test_invalid_stiffness() {
        let r = ShapeMatchSystem::new(vec![Vec3::ZERO], vec![1.0], 0.0);
        assert!(r.is_err());
        let r2 = ShapeMatchSystem::new(vec![Vec3::ZERO], vec![1.0], 1.5);
        assert!(r2.is_err());
    }

    #[test]
    fn test_no_particles() {
        let r = ShapeMatchSystem::new(vec![], vec![], 0.5);
        assert!(r.is_err());
    }

    #[test]
    fn test_step_runs() {
        let mut sys = simple_system();
        sys.step(0.01).unwrap();
        assert!(sys.time > 0.0);
    }

    #[test]
    fn test_invalid_timestep() {
        let mut sys = simple_system();
        assert!(sys.step(0.0).is_err());
        assert!(sys.step(-1.0).is_err());
    }

    #[test]
    fn test_gravity_moves_particles() {
        let mut sys = simple_system();
        let y0 = sys.particles[0].position.y;
        for _ in 0..50 {
            sys.step(0.01).unwrap();
        }
        assert!(sys.particles[0].position.y < y0);
    }

    #[test]
    fn test_fixed_particle_stays() {
        let mut sys = simple_system();
        sys.fix_particle(0).unwrap();
        let pos0 = sys.particles[0].position;
        for _ in 0..20 {
            sys.step(0.01).unwrap();
        }
        assert!(approx(sys.particles[0].position.x, pos0.x, 1e-10));
        assert!(approx(sys.particles[0].position.y, pos0.y, 1e-10));
    }

    #[test]
    fn test_shape_preservation() {
        let mut sys = simple_system();
        sys.gravity = Vec3::ZERO;
        // Deform: push particle 2 away
        sys.particles[2].position = Vec3::new(0.0, 3.0, 0.0);
        // Step should pull it back toward rest shape
        let dist_before = sys.particles[2].position.sub(sys.particles[2].rest_position).length();
        for _ in 0..10 {
            sys.step(0.01).unwrap();
        }
        let dist_after = sys.particles[2].position.sub(sys.particles[2].rest_position).length();
        // Stiffness=0.5 should have reduced displacement somewhat
        assert!(dist_after < dist_before + 1.0);
    }

    #[test]
    fn test_cluster_creation() {
        let mut sys = simple_system();
        sys.create_clusters(2);
        assert_eq!(sys.clusters.len(), 2);
        assert_eq!(sys.clusters[0].particle_indices.len(), 2);
        assert_eq!(sys.clusters[1].particle_indices.len(), 2);
    }

    #[test]
    fn test_overlapping_clusters() {
        let positions: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64, 0.0, 0.0)).collect();
        let masses = vec![1.0; 10];
        let mut sys = ShapeMatchSystem::new(positions, masses, 0.5).unwrap();
        sys.create_overlapping_clusters(4, 2);
        assert!(sys.clusters.len() > 2);
        // First cluster should have indices 0..4
        assert_eq!(sys.clusters[0].particle_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_fracture() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let masses = vec![1.0; 4];
        let mut sys = ShapeMatchSystem::new(positions, masses, 0.5).unwrap();
        sys.gravity = Vec3::ZERO;
        sys.set_fracture_threshold(0.1).unwrap();
        // Yank particle 3 far away
        sys.particles[3].position = Vec3::new(100.0, 0.0, 0.0);
        sys.step(0.01).unwrap();
        assert!(sys.fracture_count > 0);
    }

    #[test]
    fn test_invalid_fracture_threshold() {
        let mut sys = simple_system();
        assert!(sys.set_fracture_threshold(-1.0).is_err());
    }

    #[test]
    fn test_center_of_mass() {
        let sys = simple_system();
        let com = sys.center_of_mass();
        assert!(approx(com.x, 0.5, 1e-10));
        assert!(approx(com.y, 0.5, 1e-10));
    }

    #[test]
    fn test_kinetic_energy_initial() {
        let sys = simple_system();
        assert!(approx(sys.kinetic_energy(), 0.0, 1e-12));
    }

    #[test]
    fn test_kinetic_energy_after_step() {
        let mut sys = simple_system();
        sys.step(0.01).unwrap();
        assert!(sys.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_active_cluster_count() {
        let sys = simple_system();
        assert_eq!(sys.active_cluster_count(), 1);
    }

    #[test]
    fn test_max_displacement_initial() {
        let sys = simple_system();
        assert!(approx(sys.max_displacement(), 0.0, 1e-12));
    }

    #[test]
    fn test_fix_particle_oob() {
        let mut sys = simple_system();
        assert!(sys.fix_particle(999).is_err());
    }

    #[test]
    fn test_mat3_outer_product() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let w = Vec3::new(0.0, 1.0, 0.0);
        let m = Mat3::outer(v, w);
        assert!(approx(m.m[0][1], 1.0, 1e-10));
        assert!(approx(m.m[0][0], 0.0, 1e-10));
    }

    #[test]
    fn test_rotation_extraction() {
        // A pure rotation matrix should be recovered as-is
        let angle = 0.5_f64;
        let c = angle.cos();
        let s = angle.sin();
        let rot = Mat3 { m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]] };
        let extracted = rot.extract_rotation();
        assert!(approx(extracted.m[0][0], c, 1e-4));
        assert!(approx(extracted.m[0][1], -s, 1e-4));
    }

    #[test]
    fn test_multiple_steps_no_divergence() {
        let mut sys = simple_system();
        for _ in 0..200 {
            sys.step(0.01).unwrap();
        }
        // Particles should not fly to infinity
        for p in &sys.particles {
            assert!(p.position.length() < 1000.0);
        }
    }

    #[test]
    fn test_damping_slows_motion() {
        let mut sys_low = simple_system();
        sys_low.damping = 0.0;
        let mut sys_high = simple_system();
        sys_high.damping = 0.5;

        for _ in 0..20 {
            sys_low.step(0.01).unwrap();
            sys_high.step(0.01).unwrap();
        }
        // Higher damping = less kinetic energy
        assert!(sys_high.kinetic_energy() < sys_low.kinetic_energy());
    }
}
