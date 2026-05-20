//! Grasp Planner — Grasp pose generation, force closure analysis, grasp quality
//! metrics, and antipodal grasp computation for robotic manipulation.
//!
//! Provides geometric and force-analytic methods for planning stable grasps
//! on rigid objects. All algorithms are std-only, using `f64` throughout.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Grasp planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum GraspError {
    /// Invalid contact geometry.
    InvalidContact(String),
    /// Grasp is not force-closed.
    NotForceClosed,
    /// Insufficient contacts for closure.
    InsufficientContacts(usize),
    /// Numeric computation failed.
    NumericFailure(String),
    /// Configuration out of bounds.
    OutOfBounds(String),
}

impl fmt::Display for GraspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContact(msg) => write!(f, "invalid contact: {msg}"),
            Self::NotForceClosed => write!(f, "grasp is not force closed"),
            Self::InsufficientContacts(n) => {
                write!(f, "insufficient contacts: {n} (need >= 2)")
            }
            Self::NumericFailure(msg) => write!(f, "numeric failure: {msg}"),
            Self::OutOfBounds(msg) => write!(f, "out of bounds: {msg}"),
        }
    }
}

impl std::error::Error for GraspError {}

// ── 3D Vector ───────────────────────────────────────────────────

/// Minimal 3D vector for grasp geometry.
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

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
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

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-12 {
            None
        } else {
            Some(Self { x: self.x / n, y: self.y / n, z: self.z / n })
        }
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

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── Contact Point ───────────────────────────────────────────────

/// A contact point on the object surface with position and inward normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// Position on the object surface.
    pub position: Vec3,
    /// Inward-pointing surface normal at the contact.
    pub normal: Vec3,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

impl ContactPoint {
    pub fn new(position: Vec3, normal: Vec3, friction: f64) -> Result<Self, GraspError> {
        let normal = normal
            .normalized()
            .ok_or_else(|| GraspError::InvalidContact("zero normal".into()))?;
        if friction < 0.0 {
            return Err(GraspError::InvalidContact("negative friction".into()));
        }
        Ok(Self { position, normal, friction })
    }
}

impl fmt::Display for ContactPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Contact(pos={}, n={}, mu={:.3})", self.position, self.normal, self.friction)
    }
}

// ── Wrench ──────────────────────────────────────────────────────

/// A 6D wrench (force + torque).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    pub force: Vec3,
    pub torque: Vec3,
}

impl Wrench {
    pub fn new(force: Vec3, torque: Vec3) -> Self {
        Self { force, torque }
    }

    pub fn zero() -> Self {
        Self { force: Vec3::zero(), torque: Vec3::zero() }
    }

    /// Magnitude of the wrench in R6.
    pub fn magnitude(&self) -> f64 {
        (self.force.dot(self.force) + self.torque.dot(self.torque)).sqrt()
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            force: self.force.add(other.force),
            torque: self.torque.add(other.torque),
        }
    }
}

impl fmt::Display for Wrench {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wrench(f={}, tau={})", self.force, self.torque)
    }
}

// ── Friction Cone ───────────────────────────────────────────────

/// Linearised friction cone as a set of wrench primitives.
#[derive(Debug, Clone)]
pub struct FrictionCone {
    /// Number of edges in the polyhedral approximation.
    pub num_edges: usize,
    /// Wrenches at the boundary of the linearised cone.
    pub edge_wrenches: Vec<Wrench>,
}

impl FrictionCone {
    /// Build a linearised friction cone for a contact point around the object
    /// centroid. `num_edges` must be >= 3.
    pub fn from_contact(
        contact: &ContactPoint,
        centroid: Vec3,
        num_edges: usize,
    ) -> Result<Self, GraspError> {
        if num_edges < 3 {
            return Err(GraspError::InvalidContact(
                "friction cone needs >= 3 edges".into(),
            ));
        }
        // Build a local tangent frame
        let n = contact.normal;
        let arbitrary = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let t1 = n.cross(arbitrary).normalized().unwrap();
        let t2 = n.cross(t1).normalized().unwrap();

        let mu = contact.friction;
        let r = contact.position.sub(centroid);
        let mut edge_wrenches = Vec::with_capacity(num_edges);

        for i in 0..num_edges {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_edges as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            // Force on the friction cone boundary
            let f_dir = n.add(t1.scale(mu * cos_a)).add(t2.scale(mu * sin_a));
            let force = f_dir.normalized().unwrap_or(n);
            let torque = r.cross(force);
            edge_wrenches.push(Wrench::new(force, torque));
        }

        Ok(Self { num_edges, edge_wrenches })
    }
}

// ── Grasp Quality Metrics ───────────────────────────────────────

/// Quality metrics for a grasp configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraspQuality {
    /// Minimum singular value of the grasp matrix (epsilon quality).
    pub epsilon: f64,
    /// Volume of the grasp wrench space (proportional to det(G G^T)).
    pub volume: f64,
    /// Force closure flag.
    pub force_closed: bool,
    /// Largest contact force needed to resist a unit disturbance.
    pub max_contact_force: f64,
}

impl fmt::Display for GraspQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraspQuality(eps={:.4}, vol={:.4}, fc={}, max_cf={:.4})",
            self.epsilon, self.volume, self.force_closed, self.max_contact_force
        )
    }
}

// ── Grasp Pose ──────────────────────────────────────────────────

/// A candidate grasp pose with approach direction and width.
#[derive(Debug, Clone, PartialEq)]
pub struct GraspPose {
    /// Centre position of the gripper.
    pub center: Vec3,
    /// Approach direction (unit vector).
    pub approach: Vec3,
    /// Gripper opening width.
    pub width: f64,
    /// Quality score (higher is better).
    pub score: f64,
}

impl GraspPose {
    pub fn new(center: Vec3, approach: Vec3, width: f64) -> Result<Self, GraspError> {
        let approach = approach
            .normalized()
            .ok_or_else(|| GraspError::InvalidContact("zero approach".into()))?;
        if width <= 0.0 {
            return Err(GraspError::OutOfBounds("width must be positive".into()));
        }
        Ok(Self { center, approach, width, score: 0.0 })
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }
}

impl fmt::Display for GraspPose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraspPose(c={}, app={}, w={:.4}, s={:.4})",
            self.center, self.approach, self.width, self.score
        )
    }
}

// ── Antipodal Analysis ──────────────────────────────────────────

/// Check whether two contacts form an antipodal pair within the given
/// friction cones. Returns `true` if the line connecting the contacts
/// lies inside both friction cones.
pub fn is_antipodal(c1: &ContactPoint, c2: &ContactPoint) -> bool {
    let d = c2.position.sub(c1.position);
    let d_norm = match d.normalized() {
        Some(v) => v,
        None => return false,
    };
    // d must lie inside friction cone of c1 (angle with n1 <= atan(mu1))
    let cos1 = d_norm.dot(c1.normal);
    let threshold1 = (1.0 + c1.friction * c1.friction).sqrt().recip();
    // -d must lie inside friction cone of c2
    let cos2 = d_norm.scale(-1.0).dot(c2.normal);
    let threshold2 = (1.0 + c2.friction * c2.friction).sqrt().recip();

    cos1 >= threshold1 && cos2 >= threshold2
}

/// Compute the antipodal angle margin between two contacts (radians).
/// A positive value means the contacts are antipodal with that angular margin.
pub fn antipodal_margin(c1: &ContactPoint, c2: &ContactPoint) -> f64 {
    let d = c2.position.sub(c1.position);
    let d_norm = match d.normalized() {
        Some(v) => v,
        None => return -std::f64::consts::PI,
    };
    let angle1 = d_norm.dot(c1.normal).acos();
    let half_cone1 = c1.friction.atan();
    let angle2 = d_norm.scale(-1.0).dot(c2.normal).acos();
    let half_cone2 = c2.friction.atan();

    let margin1 = half_cone1 - angle1;
    let margin2 = half_cone2 - angle2;
    margin1.min(margin2)
}

// ── Grasp Matrix ────────────────────────────────────────────────

/// Build a 6xN grasp matrix from contact wrenches.
/// Each column is a primitive wrench (force + torque) from the friction cone.
fn build_grasp_matrix(cones: &[FrictionCone]) -> Vec<[f64; 6]> {
    let mut columns = Vec::new();
    for cone in cones {
        for w in &cone.edge_wrenches {
            columns.push([
                w.force.x, w.force.y, w.force.z,
                w.torque.x, w.torque.y, w.torque.z,
            ]);
        }
    }
    columns
}

/// Compute the singular values of a 6xN matrix using a one-sided Jacobi SVD.
/// Returns singular values in descending order.
fn singular_values_6xn(columns: &[[f64; 6]]) -> Vec<f64> {
    let n = columns.len();
    if n == 0 {
        return vec![];
    }
    // Form the 6x6 G*G^T matrix
    let mut gg = [[0.0f64; 6]; 6];
    for col in columns {
        for i in 0..6 {
            for j in 0..6 {
                gg[i][j] += col[i] * col[j];
            }
        }
    }
    // Power iteration to find eigenvalues of gg (symmetric positive semi-definite)
    eigenvalues_symmetric_6x6(&gg)
        .into_iter()
        .map(|ev| ev.max(0.0).sqrt())
        .collect()
}

/// Eigenvalues of a 6x6 symmetric matrix via Jacobi iteration.
fn eigenvalues_symmetric_6x6(a: &[[f64; 6]; 6]) -> Vec<f64> {
    let mut m = *a;
    for _ in 0..200 {
        let mut off_diag = 0.0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                off_diag += m[i][j] * m[i][j];
            }
        }
        if off_diag < 1e-24 {
            break;
        }
        // Find largest off-diagonal
        let (mut p, mut q) = (0, 1);
        let mut max_val = m[0][1].abs();
        for i in 0..6 {
            for j in (i + 1)..6 {
                if m[i][j].abs() > max_val {
                    max_val = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        // Jacobi rotation
        let theta = if (m[p][p] - m[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * m[p][q] / (m[p][p] - m[q][q])).atan()
        };
        let (s, c) = theta.sin_cos();
        let mut new_m = m;
        for i in 0..6 {
            new_m[i][p] = c * m[i][p] + s * m[i][q];
            new_m[i][q] = -s * m[i][p] + c * m[i][q];
        }
        let row_p = new_m.map(|r| r[p]);
        let row_q = new_m.map(|r| r[q]);
        for j in 0..6 {
            new_m[p][j] = c * row_p[j] + s * row_q[j];
            new_m[q][j] = -s * row_p[j] + c * row_q[j];
        }
        m = new_m;
    }
    let mut evals: Vec<f64> = (0..6).map(|i| m[i][i]).collect();
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    evals
}

// ── Grasp Planner ───────────────────────────────────────────────

/// Configuration for the grasp planner.
#[derive(Debug, Clone)]
pub struct GraspPlannerConfig {
    /// Number of edges for friction cone linearisation.
    pub cone_edges: usize,
    /// Minimum quality threshold for accepting a grasp.
    pub min_quality: f64,
    /// Maximum number of candidate grasps to generate.
    pub max_candidates: usize,
    /// Object centroid for torque computation.
    pub centroid: Vec3,
}

impl Default for GraspPlannerConfig {
    fn default() -> Self {
        Self {
            cone_edges: 8,
            min_quality: 0.01,
            max_candidates: 100,
            centroid: Vec3::zero(),
        }
    }
}

impl GraspPlannerConfig {
    pub fn with_cone_edges(mut self, n: usize) -> Self {
        self.cone_edges = n;
        self
    }

    pub fn with_min_quality(mut self, q: f64) -> Self {
        self.min_quality = q;
        self
    }

    pub fn with_max_candidates(mut self, n: usize) -> Self {
        self.max_candidates = n;
        self
    }

    pub fn with_centroid(mut self, c: Vec3) -> Self {
        self.centroid = c;
        self
    }
}

/// The grasp planner evaluates contact sets for force closure and quality.
#[derive(Debug, Clone)]
pub struct GraspPlanner {
    config: GraspPlannerConfig,
    contacts: Vec<ContactPoint>,
}

impl GraspPlanner {
    pub fn new(config: GraspPlannerConfig) -> Self {
        Self { config, contacts: Vec::new() }
    }

    pub fn with_contacts(mut self, contacts: Vec<ContactPoint>) -> Self {
        self.contacts = contacts;
        self
    }

    /// Add a single contact.
    pub fn add_contact(&mut self, c: ContactPoint) {
        self.contacts.push(c);
    }

    /// Evaluate force closure for the current contact set.
    pub fn evaluate_force_closure(&self) -> Result<bool, GraspError> {
        if self.contacts.len() < 2 {
            return Err(GraspError::InsufficientContacts(self.contacts.len()));
        }
        let quality = self.compute_quality()?;
        Ok(quality.force_closed)
    }

    /// Compute grasp quality metrics for the current contact set.
    pub fn compute_quality(&self) -> Result<GraspQuality, GraspError> {
        if self.contacts.len() < 2 {
            return Err(GraspError::InsufficientContacts(self.contacts.len()));
        }
        let mut cones = Vec::with_capacity(self.contacts.len());
        for c in &self.contacts {
            cones.push(FrictionCone::from_contact(c, self.config.centroid, self.config.cone_edges)?);
        }
        let columns = build_grasp_matrix(&cones);
        let svs = singular_values_6xn(&columns);
        let epsilon = svs.last().copied().unwrap_or(0.0);
        let volume: f64 = svs.iter().product();
        let force_closed = epsilon > 1e-6;
        let max_contact_force = if epsilon > 1e-12 { 1.0 / epsilon } else { f64::INFINITY };

        Ok(GraspQuality { epsilon, volume, force_closed, max_contact_force })
    }

    /// Generate antipodal grasp candidates from a set of surface samples.
    pub fn generate_antipodal_candidates(
        &self,
        samples: &[ContactPoint],
    ) -> Vec<(usize, usize, f64)> {
        let mut candidates = Vec::new();
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                let margin = antipodal_margin(&samples[i], &samples[j]);
                if margin > 0.0 {
                    candidates.push((i, j, margin));
                }
                if candidates.len() >= self.config.max_candidates {
                    break;
                }
            }
            if candidates.len() >= self.config.max_candidates {
                break;
            }
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Rank a set of grasp poses by quality.
    pub fn rank_grasps(&self, grasps: &mut [GraspPose]) {
        grasps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Number of contacts currently stored.
    pub fn num_contacts(&self) -> usize {
        self.contacts.len()
    }
}

impl fmt::Display for GraspPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraspPlanner(contacts={}, cone_edges={}, min_q={:.4})",
            self.contacts.len(),
            self.config.cone_edges,
            self.config.min_quality
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn contact(px: f64, py: f64, pz: f64, nx: f64, ny: f64, nz: f64, mu: f64) -> ContactPoint {
        ContactPoint::new(Vec3::new(px, py, pz), Vec3::new(nx, ny, nz), mu).unwrap()
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_norm() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(0.0, 3.0, 4.0);
        let n = v.normalized().unwrap();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_zero_normalized() {
        assert!(Vec3::zero().normalized().is_none());
    }

    #[test]
    fn test_contact_negative_friction() {
        let r = ContactPoint::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0), -0.1);
        assert!(r.is_err());
    }

    #[test]
    fn test_contact_zero_normal() {
        let r = ContactPoint::new(Vec3::zero(), Vec3::zero(), 0.5);
        assert!(r.is_err());
    }

    #[test]
    fn test_contact_display() {
        let c = contact(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
        let s = format!("{c}");
        assert!(s.contains("Contact"));
    }

    #[test]
    fn test_wrench_magnitude() {
        let w = Wrench::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        assert!((w.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_antipodal_opposing() {
        let c1 = contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
        let c2 = contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5);
        assert!(is_antipodal(&c1, &c2));
    }

    #[test]
    fn test_antipodal_parallel_normals_fails() {
        let c1 = contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3);
        let c2 = contact(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3);
        assert!(!is_antipodal(&c1, &c2));
    }

    #[test]
    fn test_antipodal_margin_positive() {
        let c1 = contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
        let c2 = contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5);
        assert!(antipodal_margin(&c1, &c2) > 0.0);
    }

    #[test]
    fn test_friction_cone_edges() {
        let c = contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5);
        let cone = FrictionCone::from_contact(&c, Vec3::zero(), 6).unwrap();
        assert_eq!(cone.edge_wrenches.len(), 6);
    }

    #[test]
    fn test_friction_cone_too_few_edges() {
        let c = contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5);
        assert!(FrictionCone::from_contact(&c, Vec3::zero(), 2).is_err());
    }

    #[test]
    fn test_force_closure_opposing() {
        let planner = GraspPlanner::new(GraspPlannerConfig::default())
            .with_contacts(vec![
                contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
                contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
            ]);
        // Two opposing contacts alone cannot span the full 6D wrench space
        assert!(!planner.evaluate_force_closure().unwrap());
    }

    #[test]
    fn test_insufficient_contacts() {
        let planner = GraspPlanner::new(GraspPlannerConfig::default())
            .with_contacts(vec![contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5)]);
        assert!(planner.evaluate_force_closure().is_err());
    }

    #[test]
    fn test_quality_epsilon_positive() {
        let planner = GraspPlanner::new(GraspPlannerConfig::default())
            .with_contacts(vec![
                contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
                contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
                contact(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5),
                contact(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.5),
            ]);
        let q = planner.compute_quality().unwrap();
        assert!(q.epsilon > 0.0);
    }

    #[test]
    fn test_grasp_pose_creation() {
        let gp = GraspPose::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), 0.08).unwrap();
        assert!((gp.width - 0.08).abs() < 1e-10);
    }

    #[test]
    fn test_grasp_pose_zero_width() {
        assert!(GraspPose::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), 0.0).is_err());
    }

    #[test]
    fn test_generate_candidates() {
        let planner = GraspPlanner::new(
            GraspPlannerConfig::default().with_max_candidates(10),
        );
        let samples = vec![
            contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.6),
            contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.6),
            contact(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.6),
            contact(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.6),
        ];
        let cands = planner.generate_antipodal_candidates(&samples);
        assert!(!cands.is_empty());
    }

    #[test]
    fn test_planner_display() {
        let planner = GraspPlanner::new(GraspPlannerConfig::default());
        let s = format!("{planner}");
        assert!(s.contains("GraspPlanner"));
    }
}
