//! Barnes-Hut tree for O(N log N) N-body force computation.
//!
//! Replaces heavy C/Fortran tree codes with pure Rust.
//! Octree (3D) spatial decomposition with center-of-mass aggregation.
//! Force computation via tree traversal with multipole acceptance criterion.
//! Configurable opening angle theta (default 0.5).

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for Barnes-Hut tree.
#[derive(Debug, Clone, PartialEq)]
pub enum BarnesHutError {
    /// Opening angle must be positive.
    InvalidTheta(f64),
    /// No particles to build tree from.
    EmptyParticles,
    /// Softening must be non-negative.
    NegativeSoftening(f64),
    /// Bounding box has zero volume.
    ZeroBounds,
}

impl fmt::Display for BarnesHutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTheta(t) => write!(f, "opening angle must be positive, got {t}"),
            Self::EmptyParticles => write!(f, "no particles to build tree"),
            Self::NegativeSoftening(s) => write!(f, "softening must be non-negative, got {s}"),
            Self::ZeroBounds => write!(f, "bounding box has zero volume"),
        }
    }
}

impl std::error::Error for BarnesHutError {}

// ── Vec3 ────────────────────────────────────────────────────────

/// Simple 3D vector.
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

// ── Particle ────────────────────────────────────────────────────

/// A point mass in the simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub mass: f64,
}

// ── Bounding Box ────────────────────────────────────────────────

/// Axis-aligned bounding box for octree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub center: Vec3,
    pub half_size: f64,
}

impl BBox {
    pub fn new(center: Vec3, half_size: f64) -> Self {
        Self { center, half_size }
    }

    /// Which octant (0-7) does the point fall into?
    pub fn octant(&self, pos: Vec3) -> usize {
        let mut idx = 0;
        if pos.x >= self.center.x { idx |= 1; }
        if pos.y >= self.center.y { idx |= 2; }
        if pos.z >= self.center.z { idx |= 4; }
        idx
    }

    /// Child bounding box for a given octant.
    pub fn child(&self, octant: usize) -> Self {
        let qs = self.half_size * 0.5;
        let dx = if octant & 1 != 0 { qs } else { -qs };
        let dy = if octant & 2 != 0 { qs } else { -qs };
        let dz = if octant & 4 != 0 { qs } else { -qs };
        Self {
            center: Vec3::new(self.center.x + dx, self.center.y + dy, self.center.z + dz),
            half_size: qs,
        }
    }

    /// Does this box contain the point?
    pub fn contains(&self, pos: Vec3) -> bool {
        let hs = self.half_size;
        (pos.x - self.center.x).abs() <= hs
            && (pos.y - self.center.y).abs() <= hs
            && (pos.z - self.center.z).abs() <= hs
    }
}

// ── Octree Node ─────────────────────────────────────────────────

/// An octree node: either empty, a leaf with one particle, or internal with children.
#[derive(Debug, Clone)]
enum OctreeNode {
    Empty,
    Leaf(Particle),
    Internal {
        children: Box<[OctreeNode; 8]>,
        total_mass: f64,
        center_of_mass: Vec3,
    },
}

impl OctreeNode {
    fn mass(&self) -> f64 {
        match self {
            OctreeNode::Empty => 0.0,
            OctreeNode::Leaf(p) => p.mass,
            OctreeNode::Internal { total_mass, .. } => *total_mass,
        }
    }

    fn com(&self) -> Vec3 {
        match self {
            OctreeNode::Empty => Vec3::ZERO,
            OctreeNode::Leaf(p) => p.position,
            OctreeNode::Internal { center_of_mass, .. } => *center_of_mass,
        }
    }
}

// ── Octree ──────────────────────────────────────────────────────

/// Barnes-Hut octree.
#[derive(Debug, Clone)]
pub struct Octree {
    root: OctreeNode,
    bounds: BBox,
    node_count: usize,
    max_depth: usize,
}

impl Octree {
    /// Build an octree from a set of particles.
    pub fn build(particles: &[Particle]) -> Result<Self, BarnesHutError> {
        if particles.is_empty() {
            return Err(BarnesHutError::EmptyParticles);
        }
        let bounds = Self::compute_bounds(particles);
        if bounds.half_size < 1e-30 {
            // All particles at same point; give minimal box.
            let b = BBox::new(bounds.center, 1.0);
            let mut tree = Self { root: OctreeNode::Empty, bounds: b, node_count: 0, max_depth: 0 };
            for p in particles {
                tree.insert(*p, 0);
            }
            tree.compute_mass_distribution();
            return Ok(tree);
        }
        let mut tree = Self { root: OctreeNode::Empty, bounds, node_count: 0, max_depth: 0 };
        for p in particles {
            tree.insert(*p, 0);
        }
        tree.compute_mass_distribution();
        Ok(tree)
    }

    fn compute_bounds(particles: &[Particle]) -> BBox {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let mut min_z = f64::MAX;
        let mut max_z = f64::MIN;
        for p in particles {
            min_x = min_x.min(p.position.x);
            max_x = max_x.max(p.position.x);
            min_y = min_y.min(p.position.y);
            max_y = max_y.max(p.position.y);
            min_z = min_z.min(p.position.z);
            max_z = max_z.max(p.position.z);
        }
        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        let cz = 0.5 * (min_z + max_z);
        let half = ((max_x - min_x).max(max_y - min_y).max(max_z - min_z)) * 0.5 * 1.01;
        BBox::new(Vec3::new(cx, cy, cz), half.max(1e-10))
    }

    fn insert(&mut self, particle: Particle, depth: usize) {
        self.max_depth = self.max_depth.max(depth);
        Self::insert_into(&mut self.root, &self.bounds.clone(), particle, depth, &mut self.node_count);
    }

    fn insert_into(
        node: &mut OctreeNode,
        bbox: &BBox,
        particle: Particle,
        depth: usize,
        node_count: &mut usize,
    ) {
        const MAX_DEPTH: usize = 40;
        match node {
            OctreeNode::Empty => {
                *node = OctreeNode::Leaf(particle);
                *node_count += 1;
            }
            OctreeNode::Leaf(existing) => {
                if depth >= MAX_DEPTH {
                    // Just merge at this depth.
                    let total = existing.mass + particle.mass;
                    let com = (existing.position * existing.mass + particle.position * particle.mass)
                        * (1.0 / total);
                    *existing = Particle { position: com, mass: total };
                    return;
                }
                let old = *existing;
                let children: [OctreeNode; 8] = std::array::from_fn(|_| OctreeNode::Empty);
                *node = OctreeNode::Internal {
                    children: Box::new(children),
                    total_mass: 0.0,
                    center_of_mass: Vec3::ZERO,
                };
                *node_count += 1;
                // Re-insert old particle and new particle.
                Self::insert_into(node, bbox, old, depth + 1, node_count);
                Self::insert_into(node, bbox, particle, depth + 1, node_count);
            }
            OctreeNode::Internal { children, .. } => {
                let oct = bbox.octant(particle.position);
                let child_bbox = bbox.child(oct);
                Self::insert_into(&mut children[oct], &child_bbox, particle, depth + 1, node_count);
            }
        }
    }

    fn compute_mass_distribution(&mut self) {
        Self::update_node(&mut self.root);
    }

    fn update_node(node: &mut OctreeNode) -> (f64, Vec3) {
        match node {
            OctreeNode::Empty => (0.0, Vec3::ZERO),
            OctreeNode::Leaf(p) => (p.mass, p.position * p.mass),
            OctreeNode::Internal { children, total_mass, center_of_mass } => {
                let mut mass_sum = 0.0;
                let mut weighted_pos = Vec3::ZERO;
                for child in children.iter_mut() {
                    let (m, wp) = Self::update_node(child);
                    mass_sum += m;
                    weighted_pos += wp;
                }
                *total_mass = mass_sum;
                if mass_sum > 1e-30 {
                    *center_of_mass = weighted_pos * (1.0 / mass_sum);
                } else {
                    *center_of_mass = Vec3::ZERO;
                }
                (mass_sum, weighted_pos)
            }
        }
    }

    /// Total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Total mass in the tree.
    pub fn total_mass(&self) -> f64 {
        self.root.mass()
    }

    /// Center of mass of the whole tree.
    pub fn center_of_mass(&self) -> Vec3 {
        self.root.com()
    }
}

// ── Barnes-Hut Force Solver ─────────────────────────────────────

/// Configuration for the Barnes-Hut force solver.
#[derive(Debug, Clone, PartialEq)]
pub struct BarnesHutConfig {
    /// Opening angle: nodes with s/d < theta are treated as point masses.
    pub theta: f64,
    /// Gravitational constant.
    pub g_constant: f64,
    /// Gravitational softening to avoid singularity.
    pub softening: f64,
}

impl Default for BarnesHutConfig {
    fn default() -> Self {
        Self { theta: 0.5, g_constant: 1.0, softening: 0.01 }
    }
}

/// Statistics from a force computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForceStats {
    /// Number of direct (particle-particle) evaluations.
    pub direct_evals: u64,
    /// Number of approximate (node) evaluations.
    pub approx_evals: u64,
}

/// Barnes-Hut force solver.
#[derive(Debug, Clone)]
pub struct BarnesHutSolver {
    pub config: BarnesHutConfig,
}

impl BarnesHutSolver {
    pub fn new(config: BarnesHutConfig) -> Result<Self, BarnesHutError> {
        if config.theta <= 0.0 {
            return Err(BarnesHutError::InvalidTheta(config.theta));
        }
        if config.softening < 0.0 {
            return Err(BarnesHutError::NegativeSoftening(config.softening));
        }
        Ok(Self { config })
    }

    /// Compute gravitational accelerations on all particles using the tree.
    pub fn compute_forces(
        &self,
        particles: &[Particle],
    ) -> Result<(Vec<Vec3>, ForceStats), BarnesHutError> {
        let tree = Octree::build(particles)?;
        let mut accels = vec![Vec3::ZERO; particles.len()];
        let mut stats = ForceStats { direct_evals: 0, approx_evals: 0 };
        for (i, p) in particles.iter().enumerate() {
            accels[i] = self.compute_accel_for(
                p,
                &tree.root,
                &tree.bounds,
                &mut stats,
            );
        }
        Ok((accels, stats))
    }

    fn compute_accel_for(
        &self,
        target: &Particle,
        node: &OctreeNode,
        bbox: &BBox,
        stats: &mut ForceStats,
    ) -> Vec3 {
        match node {
            OctreeNode::Empty => Vec3::ZERO,
            OctreeNode::Leaf(p) => {
                let dx = p.position - target.position;
                let dist2 = dx.magnitude_sq();
                if dist2 < 1e-30 {
                    // Same particle.
                    return Vec3::ZERO;
                }
                stats.direct_evals += 1;
                let eps2 = self.config.softening * self.config.softening;
                let r2 = dist2 + eps2;
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                dx * (self.config.g_constant * p.mass * inv_r3)
            }
            OctreeNode::Internal { children, total_mass, center_of_mass } => {
                let dx = *center_of_mass - target.position;
                let dist = dx.magnitude();
                let cell_size = bbox.half_size * 2.0;

                // Multipole acceptance criterion: s/d < theta.
                if cell_size / (dist + 1e-30) < self.config.theta {
                    stats.approx_evals += 1;
                    let eps2 = self.config.softening * self.config.softening;
                    let r2 = dx.magnitude_sq() + eps2;
                    let inv_r3 = 1.0 / (r2 * r2.sqrt());
                    return dx * (self.config.g_constant * total_mass * inv_r3);
                }

                // Recurse into children.
                let mut accel = Vec3::ZERO;
                for (i, child) in children.iter().enumerate() {
                    let child_bbox = bbox.child(i);
                    accel += self.compute_accel_for(target, child, &child_bbox, stats);
                }
                accel
            }
        }
    }

    /// Compute potential energy between all particles using the tree.
    pub fn potential_energy(&self, particles: &[Particle]) -> Result<f64, BarnesHutError> {
        let tree = Octree::build(particles)?;
        let mut energy = 0.0;
        for p in particles {
            energy += self.potential_at(p, &tree.root, &tree.bounds);
        }
        // Each pair counted twice.
        Ok(energy * 0.5)
    }

    fn potential_at(&self, target: &Particle, node: &OctreeNode, bbox: &BBox) -> f64 {
        match node {
            OctreeNode::Empty => 0.0,
            OctreeNode::Leaf(p) => {
                let dx = p.position - target.position;
                let dist2 = dx.magnitude_sq();
                if dist2 < 1e-30 {
                    return 0.0;
                }
                let eps2 = self.config.softening * self.config.softening;
                let r = (dist2 + eps2).sqrt();
                -self.config.g_constant * target.mass * p.mass / r
            }
            OctreeNode::Internal { children, total_mass, center_of_mass } => {
                let dx = *center_of_mass - target.position;
                let dist = dx.magnitude();
                let cell_size = bbox.half_size * 2.0;
                if cell_size / (dist + 1e-30) < self.config.theta {
                    let eps2 = self.config.softening * self.config.softening;
                    let r = (dx.magnitude_sq() + eps2).sqrt();
                    return -self.config.g_constant * target.mass * total_mass / r;
                }
                let mut pot = 0.0;
                for (i, child) in children.iter().enumerate() {
                    pot += self.potential_at(target, child, &bbox.child(i));
                }
                pot
            }
        }
    }
}

// ── Direct (brute force) for comparison ─────────────────────────

/// Direct O(N^2) force for accuracy comparison.
pub fn direct_forces(particles: &[Particle], g: f64, softening: f64) -> Vec<Vec3> {
    let n = particles.len();
    let eps2 = softening * softening;
    let mut accels = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = particles[j].position - particles[i].position;
            let r2 = dx.magnitude_sq() + eps2;
            let inv_r3 = 1.0 / (r2 * r2.sqrt());
            let f = dx * (g * inv_r3);
            accels[i] += f * particles[j].mass;
            accels[j] = accels[j] - f * particles[i].mass;
        }
    }
    accels
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

    fn make_particles(n: usize) -> Vec<Particle> {
        // Deterministic layout in a cube.
        let mut particles = Vec::new();
        let side = (n as f64).cbrt().ceil() as usize;
        let mut count = 0;
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    if count >= n { break; }
                    particles.push(Particle {
                        position: Vec3::new(i as f64, j as f64, k as f64),
                        mass: 1.0,
                    });
                    count += 1;
                }
            }
        }
        particles
    }

    #[test]
    fn tree_build_single() {
        let ps = vec![Particle { position: Vec3::new(1.0, 2.0, 3.0), mass: 5.0 }];
        let tree = Octree::build(&ps).unwrap();
        assert!(approx_eq(tree.total_mass(), 5.0, 1e-10));
    }

    #[test]
    fn tree_build_two() {
        let ps = vec![
            Particle { position: Vec3::new(0.0, 0.0, 0.0), mass: 1.0 },
            Particle { position: Vec3::new(10.0, 0.0, 0.0), mass: 1.0 },
        ];
        let tree = Octree::build(&ps).unwrap();
        assert!(approx_eq(tree.total_mass(), 2.0, 1e-10));
        assert!(approx_eq(tree.center_of_mass().x, 5.0, 1e-10));
    }

    #[test]
    fn tree_build_empty() {
        let ps: Vec<Particle> = vec![];
        assert!(Octree::build(&ps).is_err());
    }

    #[test]
    fn tree_many_particles() {
        let ps = make_particles(64);
        let tree = Octree::build(&ps).unwrap();
        assert!(approx_eq(tree.total_mass(), 64.0, 1e-10));
        assert!(tree.node_count() > 0);
    }

    #[test]
    fn solver_invalid_theta() {
        let cfg = BarnesHutConfig { theta: -1.0, ..Default::default() };
        assert!(BarnesHutSolver::new(cfg).is_err());
    }

    #[test]
    fn solver_negative_softening() {
        let cfg = BarnesHutConfig { softening: -0.01, ..Default::default() };
        assert!(BarnesHutSolver::new(cfg).is_err());
    }

    #[test]
    fn two_body_force_direction() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::new(10.0, 0.0, 0.0), mass: 1.0 },
        ];
        let cfg = BarnesHutConfig { theta: 0.5, g_constant: 1.0, softening: 0.0 };
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let (accels, _stats) = solver.compute_forces(&ps).unwrap();
        // Particle 0 should be accelerated in +x.
        assert!(accels[0].x > 0.0);
        // Particle 1 should be accelerated in -x.
        assert!(accels[1].x < 0.0);
    }

    #[test]
    fn force_accuracy_vs_direct() {
        let ps = make_particles(27);
        let cfg = BarnesHutConfig { theta: 0.3, g_constant: 1.0, softening: 0.01 };
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let (bh_accels, _stats) = solver.compute_forces(&ps).unwrap();
        let direct = direct_forces(&ps, 1.0, 0.01);
        // With small theta, BH should be close to direct.
        for i in 0..ps.len() {
            let err = (bh_accels[i] - direct[i]).magnitude();
            let mag = direct[i].magnitude().max(1e-10);
            let rel = err / mag;
            assert!(rel < 0.15, "particle {i} relative force error {rel}");
        }
    }

    #[test]
    fn large_theta_fewer_evals() {
        let ps = make_particles(64);
        let cfg_small = BarnesHutConfig { theta: 0.3, ..Default::default() };
        let cfg_large = BarnesHutConfig { theta: 1.5, ..Default::default() };
        let s1 = BarnesHutSolver::new(cfg_small).unwrap();
        let s2 = BarnesHutSolver::new(cfg_large).unwrap();
        let (_a1, stats1) = s1.compute_forces(&ps).unwrap();
        let (_a2, stats2) = s2.compute_forces(&ps).unwrap();
        let total1 = stats1.direct_evals + stats1.approx_evals;
        let total2 = stats2.direct_evals + stats2.approx_evals;
        assert!(total2 <= total1, "larger theta should mean fewer evaluations");
    }

    #[test]
    fn potential_energy_two_body() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::new(1.0, 0.0, 0.0), mass: 1.0 },
        ];
        let cfg = BarnesHutConfig { theta: 0.5, g_constant: 1.0, softening: 0.0 };
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let pe = solver.potential_energy(&ps).unwrap();
        // -G*m1*m2/r = -1.0
        assert!(approx_eq(pe, -1.0, 1e-6));
    }

    #[test]
    fn bbox_octant_assignment() {
        let bb = BBox::new(Vec3::ZERO, 1.0);
        assert_eq!(bb.octant(Vec3::new(-0.5, -0.5, -0.5)), 0);
        assert_eq!(bb.octant(Vec3::new(0.5, -0.5, -0.5)), 1);
        assert_eq!(bb.octant(Vec3::new(-0.5, 0.5, -0.5)), 2);
        assert_eq!(bb.octant(Vec3::new(0.5, 0.5, 0.5)), 7);
    }

    #[test]
    fn bbox_child_contains_point() {
        let bb = BBox::new(Vec3::ZERO, 2.0);
        let child = bb.child(0); // (-1, -1, -1) center, half_size 1.0
        assert!(child.contains(Vec3::new(-1.5, -1.5, -1.5)));
        assert!(!child.contains(Vec3::new(1.5, 1.5, 1.5)));
    }

    #[test]
    fn coincident_particles() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::ZERO, mass: 2.0 },
            Particle { position: Vec3::ZERO, mass: 3.0 },
        ];
        let tree = Octree::build(&ps).unwrap();
        assert!(approx_eq(tree.total_mass(), 6.0, 1e-10));
    }

    #[test]
    fn tree_depth_reasonable() {
        let ps = make_particles(125);
        let tree = Octree::build(&ps).unwrap();
        assert!(tree.max_depth() < 30);
    }

    #[test]
    fn direct_force_symmetric() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::new(1.0, 0.0, 0.0), mass: 1.0 },
        ];
        let accels = direct_forces(&ps, 1.0, 0.0);
        // Equal and opposite.
        assert!(vec3_approx_eq(
            accels[0],
            Vec3::new(-accels[1].x, -accels[1].y, -accels[1].z),
            1e-10
        ));
    }

    #[test]
    fn g_constant_scaling() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::new(5.0, 0.0, 0.0), mass: 1.0 },
        ];
        let cfg1 = BarnesHutConfig { g_constant: 1.0, theta: 0.5, softening: 0.0 };
        let cfg2 = BarnesHutConfig { g_constant: 2.0, theta: 0.5, softening: 0.0 };
        let s1 = BarnesHutSolver::new(cfg1).unwrap();
        let s2 = BarnesHutSolver::new(cfg2).unwrap();
        let (a1, _) = s1.compute_forces(&ps).unwrap();
        let (a2, _) = s2.compute_forces(&ps).unwrap();
        assert!(approx_eq(a2[0].x, 2.0 * a1[0].x, 1e-10));
    }

    #[test]
    fn softening_reduces_peak_force() {
        let ps = vec![
            Particle { position: Vec3::ZERO, mass: 1.0 },
            Particle { position: Vec3::new(0.01, 0.0, 0.0), mass: 1.0 },
        ];
        let cfg_hard = BarnesHutConfig { softening: 0.0, theta: 0.5, g_constant: 1.0 };
        let cfg_soft = BarnesHutConfig { softening: 1.0, theta: 0.5, g_constant: 1.0 };
        let s1 = BarnesHutSolver::new(cfg_hard).unwrap();
        let s2 = BarnesHutSolver::new(cfg_soft).unwrap();
        let (a1, _) = s1.compute_forces(&ps).unwrap();
        let (a2, _) = s2.compute_forces(&ps).unwrap();
        assert!(a2[0].magnitude() < a1[0].magnitude());
    }

    #[test]
    fn stats_nonzero_for_many_particles() {
        let ps = make_particles(64);
        let cfg = BarnesHutConfig { theta: 0.5, ..Default::default() };
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let (_accels, stats) = solver.compute_forces(&ps).unwrap();
        assert!(stats.direct_evals > 0 || stats.approx_evals > 0);
    }

    #[test]
    fn force_finite_all_particles() {
        let ps = make_particles(27);
        let cfg = BarnesHutConfig::default();
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let (accels, _) = solver.compute_forces(&ps).unwrap();
        for a in &accels {
            assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite());
        }
    }

    #[test]
    fn single_particle_zero_force() {
        let ps = vec![Particle { position: Vec3::new(5.0, 3.0, 1.0), mass: 10.0 }];
        let cfg = BarnesHutConfig::default();
        let solver = BarnesHutSolver::new(cfg).unwrap();
        let (accels, _) = solver.compute_forces(&ps).unwrap();
        assert!(vec3_approx_eq(accels[0], Vec3::ZERO, 1e-10));
    }

    #[test]
    fn center_of_mass_weighted() {
        let ps = vec![
            Particle { position: Vec3::new(0.0, 0.0, 0.0), mass: 3.0 },
            Particle { position: Vec3::new(10.0, 0.0, 0.0), mass: 1.0 },
        ];
        let tree = Octree::build(&ps).unwrap();
        // CoM at 7.5 from origin? No: (0*3 + 10*1)/(3+1) = 2.5
        assert!(approx_eq(tree.center_of_mass().x, 2.5, 1e-10));
    }
}
