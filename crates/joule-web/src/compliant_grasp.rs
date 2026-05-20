//! Compliant Grasping — Soft finger contact models, contact wrench space
//! analysis, form/force closure verification, and enveloping grasp planning.
//!
//! Implements soft-finger contact models with torsional friction, wrench space
//! construction, and closure tests for compliant/underactuated grippers.
//! All algorithms are std-only, using `f64` throughout.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Compliant grasping errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CompliantError {
    /// Invalid contact parameter.
    InvalidContact(String),
    /// Grasp is not form-closed.
    NotFormClosed,
    /// Grasp is not force-closed.
    NotForceClosed,
    /// Insufficient contacts.
    InsufficientContacts(usize),
    /// Numeric failure.
    NumericFailure(String),
}

impl fmt::Display for CompliantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContact(m) => write!(f, "invalid contact: {m}"),
            Self::NotFormClosed => write!(f, "grasp is not form-closed"),
            Self::NotForceClosed => write!(f, "grasp is not force-closed"),
            Self::InsufficientContacts(n) => write!(f, "need >= 2 contacts, got {n}"),
            Self::NumericFailure(m) => write!(f, "numeric failure: {m}"),
        }
    }
}

impl std::error::Error for CompliantError {}

// ── 3D Vector ───────────────────────────────────────────────────

/// Minimal 3D vector.
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

// ── Wrench ──────────────────────────────────────────────────────

/// A 6D wrench.
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

    pub fn as_array(&self) -> [f64; 6] {
        [self.force.x, self.force.y, self.force.z, self.torque.x, self.torque.y, self.torque.z]
    }

    pub fn magnitude(&self) -> f64 {
        (self.force.dot(self.force) + self.torque.dot(self.torque)).sqrt()
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            force: self.force.add(other.force),
            torque: self.torque.add(other.torque),
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            force: self.force.scale(s),
            torque: self.torque.scale(s),
        }
    }
}

impl fmt::Display for Wrench {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "W(f={}, tau={})", self.force, self.torque)
    }
}

// ── Soft Finger Contact ─────────────────────────────────────────

/// Contact type for compliance modelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactType {
    /// Point contact with friction (3D friction cone).
    PointContactFriction,
    /// Soft finger contact (friction cone + torsional friction).
    SoftFinger,
    /// Rigid contact (no compliance).
    Rigid,
}

impl fmt::Display for ContactType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PointContactFriction => write!(f, "PointFriction"),
            Self::SoftFinger => write!(f, "SoftFinger"),
            Self::Rigid => write!(f, "Rigid"),
        }
    }
}

/// A soft-finger contact with position, normal, and friction properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftContact {
    /// Contact position on the object surface.
    pub position: Vec3,
    /// Inward-pointing surface normal.
    pub normal: Vec3,
    /// Coulomb friction coefficient (translational).
    pub mu_t: f64,
    /// Torsional friction coefficient.
    pub mu_r: f64,
    /// Contact stiffness (N/m).
    pub stiffness: f64,
    /// Contact type.
    pub contact_type: ContactType,
}

impl SoftContact {
    pub fn new(
        position: Vec3,
        normal: Vec3,
        mu_t: f64,
        mu_r: f64,
        stiffness: f64,
        contact_type: ContactType,
    ) -> Result<Self, CompliantError> {
        let normal = normal
            .normalized()
            .ok_or_else(|| CompliantError::InvalidContact("zero normal".into()))?;
        if mu_t < 0.0 || mu_r < 0.0 {
            return Err(CompliantError::InvalidContact("negative friction".into()));
        }
        if stiffness <= 0.0 {
            return Err(CompliantError::InvalidContact("stiffness must be positive".into()));
        }
        Ok(Self { position, normal, mu_t, mu_r, stiffness, contact_type })
    }

    /// Number of wrench components this contact contributes.
    pub fn wrench_dimension(&self) -> usize {
        match self.contact_type {
            ContactType::PointContactFriction => 3, // fx, fy, fz in friction cone
            ContactType::SoftFinger => 4,           // fx, fy, fz + torsion
            ContactType::Rigid => 6,                // full wrench
        }
    }

    /// Build the linearised wrench set for this contact around a centroid.
    pub fn wrench_set(&self, centroid: Vec3, num_edges: usize) -> Vec<Wrench> {
        let n = self.normal;
        // Build tangent frame
        let arbitrary = if n.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let t1 = n.cross(arbitrary).normalized().unwrap();
        let t2 = n.cross(t1).normalized().unwrap();
        let r = self.position.sub(centroid);

        let mut wrenches = Vec::with_capacity(num_edges);
        for i in 0..num_edges {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_edges as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            let force = n.add(t1.scale(self.mu_t * cos_a)).add(t2.scale(self.mu_t * sin_a));
            let force = force.normalized().unwrap_or(n);
            let mut torque = r.cross(force);
            // Add torsional friction component for soft fingers
            if self.contact_type == ContactType::SoftFinger {
                torque = torque.add(n.scale(self.mu_r * if i % 2 == 0 { 1.0 } else { -1.0 }));
            }
            wrenches.push(Wrench::new(force, torque));
        }
        wrenches
    }

    /// Compute the contact force for a given deformation.
    pub fn contact_force(&self, deformation: f64) -> f64 {
        if deformation <= 0.0 {
            0.0
        } else {
            self.stiffness * deformation
        }
    }
}

impl fmt::Display for SoftContact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SoftContact({}, pos={}, mu_t={:.3}, mu_r={:.3}, k={:.1})",
            self.contact_type, self.position, self.mu_t, self.mu_r, self.stiffness
        )
    }
}

// ── Contact Wrench Space ────────────────────────────────────────

/// The contact wrench space for a multi-contact grasp.
#[derive(Debug, Clone)]
pub struct ContactWrenchSpace {
    /// All primitive wrenches from all contacts.
    pub wrenches: Vec<Wrench>,
    /// Number of contacts.
    pub num_contacts: usize,
}

impl ContactWrenchSpace {
    /// Build from a set of soft contacts around a centroid.
    pub fn from_contacts(
        contacts: &[SoftContact],
        centroid: Vec3,
        cone_edges: usize,
    ) -> Result<Self, CompliantError> {
        if contacts.len() < 2 {
            return Err(CompliantError::InsufficientContacts(contacts.len()));
        }
        let mut wrenches = Vec::new();
        for c in contacts {
            wrenches.extend(c.wrench_set(centroid, cone_edges));
        }
        Ok(Self { wrenches, num_contacts: contacts.len() })
    }

    /// Check if the origin is inside the convex hull of wrenches (force closure).
    /// Uses a simplified check: for each axis, verify that both positive and
    /// negative wrenches exist.
    pub fn check_force_closure(&self) -> bool {
        // Check that for each of the 6 wrench dimensions, there exist primitives
        // on both sides of zero.
        for dim in 0..6 {
            let mut has_positive = false;
            let mut has_negative = false;
            for w in &self.wrenches {
                let val = w.as_array()[dim];
                if val > 1e-8 {
                    has_positive = true;
                }
                if val < -1e-8 {
                    has_negative = true;
                }
            }
            if !has_positive || !has_negative {
                return false;
            }
        }
        true
    }

    /// Compute the epsilon quality (minimum distance from origin to convex hull
    /// boundary). Approximated by the smallest maximum wrench component across
    /// axes.
    pub fn epsilon_quality(&self) -> f64 {
        let mut min_extent = f64::INFINITY;
        for dim in 0..6 {
            let mut max_pos = 0.0_f64;
            let mut max_neg = 0.0_f64;
            for w in &self.wrenches {
                let val = w.as_array()[dim];
                if val > 0.0 {
                    max_pos = max_pos.max(val);
                } else {
                    max_neg = max_neg.max(-val);
                }
            }
            let extent = max_pos.min(max_neg);
            min_extent = min_extent.min(extent);
        }
        if min_extent == f64::INFINITY { 0.0 } else { min_extent }
    }

    /// Number of primitive wrenches.
    pub fn num_primitives(&self) -> usize {
        self.wrenches.len()
    }
}

impl fmt::Display for ContactWrenchSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CWS(contacts={}, primitives={})",
            self.num_contacts,
            self.wrenches.len()
        )
    }
}

// ── Enveloping Grasp ────────────────────────────────────────────

/// Configuration for enveloping grasp analysis.
#[derive(Debug, Clone)]
pub struct EnvelopingGraspConfig {
    /// Number of finger phalanges.
    pub num_phalanges: usize,
    /// Link lengths for each phalanx (metres).
    pub link_lengths: Vec<f64>,
    /// Joint stiffnesses (N·m/rad).
    pub joint_stiffnesses: Vec<f64>,
    /// Friction at each phalanx contact.
    pub friction: f64,
    /// Cone linearisation edges.
    pub cone_edges: usize,
}

impl Default for EnvelopingGraspConfig {
    fn default() -> Self {
        Self {
            num_phalanges: 3,
            link_lengths: vec![0.04, 0.03, 0.02],
            joint_stiffnesses: vec![0.5, 0.3, 0.2],
            friction: 0.5,
            cone_edges: 8,
        }
    }
}

impl EnvelopingGraspConfig {
    pub fn with_num_phalanges(mut self, n: usize) -> Self {
        self.num_phalanges = n;
        self
    }

    pub fn with_link_lengths(mut self, lengths: Vec<f64>) -> Self {
        self.link_lengths = lengths;
        self
    }

    pub fn with_joint_stiffnesses(mut self, stiffnesses: Vec<f64>) -> Self {
        self.joint_stiffnesses = stiffnesses;
        self
    }

    pub fn with_friction(mut self, mu: f64) -> Self {
        self.friction = mu;
        self
    }

    pub fn with_cone_edges(mut self, n: usize) -> Self {
        self.cone_edges = n;
        self
    }
}

/// Enveloping grasp analyser for underactuated fingers.
#[derive(Debug, Clone)]
pub struct EnvelopingGrasp {
    config: EnvelopingGraspConfig,
    contacts: Vec<SoftContact>,
}

impl EnvelopingGrasp {
    pub fn new(config: EnvelopingGraspConfig) -> Self {
        Self { config, contacts: Vec::new() }
    }

    pub fn with_contacts(mut self, contacts: Vec<SoftContact>) -> Self {
        self.contacts = contacts;
        self
    }

    pub fn add_contact(&mut self, contact: SoftContact) {
        self.contacts.push(contact);
    }

    /// Compute total contact stiffness.
    pub fn total_stiffness(&self) -> f64 {
        self.contacts.iter().map(|c| c.stiffness).sum()
    }

    /// Compute the joint torques from contact forces via Jacobian transpose.
    /// Uses simplified 2D planar finger kinematics projected to 3D.
    pub fn joint_torques_from_contacts(&self, contact_forces: &[f64]) -> Vec<f64> {
        let n = self.config.num_phalanges.min(contact_forces.len());
        let mut torques = vec![0.0; n];
        for i in 0..n {
            // Jacobian column for joint i affects contacts i..n
            for j in i..n {
                let lever = self.config.link_lengths[i..=j].iter().sum::<f64>();
                torques[i] += contact_forces[j] * lever;
            }
        }
        torques
    }

    /// Compute the grasp wrench space and check force closure.
    pub fn analyse(&self, centroid: Vec3) -> Result<GraspAnalysis, CompliantError> {
        if self.contacts.len() < 2 {
            return Err(CompliantError::InsufficientContacts(self.contacts.len()));
        }
        let cws = ContactWrenchSpace::from_contacts(
            &self.contacts,
            centroid,
            self.config.cone_edges,
        )?;
        let force_closed = cws.check_force_closure();
        let epsilon = cws.epsilon_quality();
        let form_closed = self.contacts.len() >= 7; // form closure needs >= 7 contacts in 3D

        Ok(GraspAnalysis {
            force_closed,
            form_closed,
            epsilon,
            num_contacts: self.contacts.len(),
            total_stiffness: self.total_stiffness(),
        })
    }
}

impl fmt::Display for EnvelopingGrasp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EnvelopingGrasp(phalanges={}, contacts={})",
            self.config.num_phalanges,
            self.contacts.len()
        )
    }
}

/// Result of grasp analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraspAnalysis {
    pub force_closed: bool,
    pub form_closed: bool,
    pub epsilon: f64,
    pub num_contacts: usize,
    pub total_stiffness: f64,
}

impl fmt::Display for GraspAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraspAnalysis(fc={}, form={}, eps={:.4}, contacts={}, k={:.1})",
            self.force_closed, self.form_closed, self.epsilon,
            self.num_contacts, self.total_stiffness
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn soft_contact(
        px: f64, py: f64, pz: f64,
        nx: f64, ny: f64, nz: f64,
        mu_t: f64,
    ) -> SoftContact {
        SoftContact::new(
            Vec3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
            mu_t, 0.1, 100.0,
            ContactType::SoftFinger,
        )
        .unwrap()
    }

    #[test]
    fn test_soft_contact_creation() {
        let c = soft_contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
        assert_eq!(c.contact_type, ContactType::SoftFinger);
    }

    #[test]
    fn test_soft_contact_zero_normal() {
        let r = SoftContact::new(
            Vec3::zero(), Vec3::zero(), 0.5, 0.1, 100.0,
            ContactType::SoftFinger,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_soft_contact_negative_friction() {
        let r = SoftContact::new(
            Vec3::zero(), Vec3::new(0.0, 0.0, 1.0), -0.1, 0.1, 100.0,
            ContactType::SoftFinger,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_wrench_dimension() {
        let point = SoftContact::new(
            Vec3::zero(), Vec3::new(0.0, 0.0, 1.0), 0.5, 0.0, 100.0,
            ContactType::PointContactFriction,
        )
        .unwrap();
        assert_eq!(point.wrench_dimension(), 3);

        let soft = soft_contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
        assert_eq!(soft.wrench_dimension(), 4);
    }

    #[test]
    fn test_contact_force_zero_deformation() {
        let c = soft_contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
        assert!((c.contact_force(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_contact_force_positive_deformation() {
        let c = soft_contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
        assert!((c.contact_force(0.001) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_wrench_set_count() {
        let c = soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5);
        let ws = c.wrench_set(Vec3::zero(), 8);
        assert_eq!(ws.len(), 8);
    }

    #[test]
    fn test_cws_creation() {
        let contacts = vec![
            soft_contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
            soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
        ];
        let cws = ContactWrenchSpace::from_contacts(&contacts, Vec3::zero(), 8).unwrap();
        assert_eq!(cws.num_contacts, 2);
    }

    #[test]
    fn test_cws_insufficient_contacts() {
        let contacts = vec![soft_contact(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5)];
        assert!(ContactWrenchSpace::from_contacts(&contacts, Vec3::zero(), 8).is_err());
    }

    #[test]
    fn test_cws_force_closure_opposing() {
        let contacts = vec![
            soft_contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
            soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
            soft_contact(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5),
            soft_contact(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.5),
            soft_contact(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.5),
            soft_contact(0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.5),
        ];
        let cws = ContactWrenchSpace::from_contacts(&contacts, Vec3::zero(), 8).unwrap();
        assert!(cws.check_force_closure());
    }

    #[test]
    fn test_epsilon_quality_positive() {
        let contacts = vec![
            soft_contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
            soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
            soft_contact(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5),
            soft_contact(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.5),
            soft_contact(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.5),
            soft_contact(0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.5),
        ];
        let cws = ContactWrenchSpace::from_contacts(&contacts, Vec3::zero(), 8).unwrap();
        assert!(cws.epsilon_quality() > 0.0);
    }

    #[test]
    fn test_enveloping_grasp_creation() {
        let eg = EnvelopingGrasp::new(EnvelopingGraspConfig::default());
        assert_eq!(eg.config.num_phalanges, 3);
    }

    #[test]
    fn test_joint_torques() {
        let eg = EnvelopingGrasp::new(EnvelopingGraspConfig::default());
        let forces = vec![1.0, 1.0, 1.0];
        let torques = eg.joint_torques_from_contacts(&forces);
        assert_eq!(torques.len(), 3);
        // Joint 0 should have highest torque (longest lever)
        assert!(torques[0] >= torques[1]);
    }

    #[test]
    fn test_total_stiffness() {
        let eg = EnvelopingGrasp::new(EnvelopingGraspConfig::default())
            .with_contacts(vec![
                soft_contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
                soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
            ]);
        assert!((eg.total_stiffness() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_enveloping_analyse() {
        let eg = EnvelopingGrasp::new(EnvelopingGraspConfig::default())
            .with_contacts(vec![
                soft_contact(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5),
                soft_contact(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5),
                soft_contact(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5),
                soft_contact(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.5),
            ]);
        let analysis = eg.analyse(Vec3::zero()).unwrap();
        assert!(analysis.num_contacts == 4);
    }

    #[test]
    fn test_grasp_analysis_display() {
        let analysis = GraspAnalysis {
            force_closed: true,
            form_closed: false,
            epsilon: 0.5,
            num_contacts: 4,
            total_stiffness: 400.0,
        };
        let s = format!("{analysis}");
        assert!(s.contains("GraspAnalysis"));
    }

    #[test]
    fn test_wrench_add() {
        let w1 = Wrench::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        let w2 = Wrench::new(Vec3::new(0.0, 1.0, 0.0), Vec3::zero());
        let sum = w1.add(w2);
        assert!((sum.force.x - 1.0).abs() < 1e-10);
        assert!((sum.force.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let cfg = EnvelopingGraspConfig::default()
            .with_num_phalanges(4)
            .with_friction(0.6)
            .with_cone_edges(12);
        assert_eq!(cfg.num_phalanges, 4);
        assert!((cfg.friction - 0.6).abs() < 1e-10);
        assert_eq!(cfg.cone_edges, 12);
    }

    #[test]
    fn test_enveloping_display() {
        let eg = EnvelopingGrasp::new(EnvelopingGraspConfig::default());
        let s = format!("{eg}");
        assert!(s.contains("EnvelopingGrasp"));
    }
}
