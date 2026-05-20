//! 3D Sequential Impulse Constraint Solver — contact constraints with normal
//! + friction (2 tangent directions), position correction via split impulse,
//! warm-starting, configurable velocity/position iterations, box friction
//! cone approximation, and restitution with velocity threshold.

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }
    pub fn cross(self, r: Self) -> Self {
        Self {
            x: self.y * r.z - self.z * r.y,
            y: self.z * r.x - self.x * r.z,
            z: self.x * r.y - self.y * r.x,
        }
    }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; }
}
impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, r: Self) { self.x -= r.x; self.y -= r.y; self.z -= r.z; }
}

// ── Mat3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };
    pub const IDENTITY: Self = Self { m: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] };

    pub fn diagonal(a: f64, b: f64, c: f64) -> Self {
        Self { m: [[a,0.0,0.0],[0.0,b,0.0],[0.0,0.0,c]] }
    }

    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z,
            y: self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z,
            z: self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z,
        }
    }
}

// ── Solver Configuration ─────────────────────────────────────

/// Configuration for the constraint solver.
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    pub velocity_iterations: usize,
    pub position_iterations: usize,
    /// Baumgarte stabilization factor (0..1).
    pub baumgarte: f64,
    /// Slop: allowed penetration before correction.
    pub slop: f64,
    /// Restitution velocity threshold: below this, no bounce.
    pub restitution_threshold: f64,
    /// Whether to use split impulse for position correction.
    pub use_split_impulse: bool,
    /// Warm-starting factor (0 = no warmstart, 1 = full).
    pub warmstart_factor: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            baumgarte: 0.2,
            slop: 0.005,
            restitution_threshold: 1.0,
            use_split_impulse: true,
            warmstart_factor: 0.85,
        }
    }
}

// ── Solver Body ──────────────────────────────────────────────

/// A body as seen by the solver (velocity-level data).
#[derive(Debug, Clone)]
pub struct SolverBody {
    pub inv_mass: f64,
    pub inv_inertia: Mat3,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Split impulse pseudo-velocities for position correction.
    pub pseudo_linear: Vec3,
    pub pseudo_angular: Vec3,
    pub position: Vec3,
}

impl SolverBody {
    pub fn new(inv_mass: f64, inv_inertia: Mat3, pos: Vec3) -> Self {
        Self {
            inv_mass,
            inv_inertia,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            pseudo_linear: Vec3::ZERO,
            pseudo_angular: Vec3::ZERO,
            position: pos,
        }
    }

    pub fn is_static(&self) -> bool {
        self.inv_mass < 1e-12
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, r: Vec3) {
        self.linear_velocity += impulse * self.inv_mass;
        self.angular_velocity += self.inv_inertia.mul_vec(r.cross(impulse));
    }

    pub fn apply_pseudo_impulse(&mut self, impulse: Vec3, r: Vec3) {
        self.pseudo_linear += impulse * self.inv_mass;
        self.pseudo_angular += self.inv_inertia.mul_vec(r.cross(impulse));
    }

    /// Velocity of a point on this body.
    pub fn velocity_at(&self, r: Vec3) -> Vec3 {
        self.linear_velocity + self.angular_velocity.cross(r)
    }
}

// ── Contact Constraint ──────────────────────────────────────

/// Precomputed data for a single contact constraint.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub contact_point: Vec3,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub depth: f64,
    pub restitution: f64,
    pub friction: f64,

    // Precomputed
    pub r_a: Vec3,
    pub r_b: Vec3,
    pub effective_mass_normal: f64,
    pub effective_mass_tangent1: f64,
    pub effective_mass_tangent2: f64,
    pub velocity_bias: f64,

    // Accumulated impulses
    pub accumulated_normal: f64,
    pub accumulated_tangent1: f64,
    pub accumulated_tangent2: f64,
}

impl ContactConstraint {
    pub fn new(
        body_a: usize,
        body_b: usize,
        contact_point: Vec3,
        normal: Vec3,
        depth: f64,
        restitution: f64,
        friction: f64,
    ) -> Self {
        // Compute tangent frame
        let (t1, t2) = compute_tangent_frame(normal);
        Self {
            body_a,
            body_b,
            contact_point,
            normal,
            tangent1: t1,
            tangent2: t2,
            depth,
            restitution,
            friction,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            effective_mass_normal: 0.0,
            effective_mass_tangent1: 0.0,
            effective_mass_tangent2: 0.0,
            velocity_bias: 0.0,
            accumulated_normal: 0.0,
            accumulated_tangent1: 0.0,
            accumulated_tangent2: 0.0,
        }
    }
}

fn compute_tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let t1 = normal.cross(up).normalized();
    let t2 = normal.cross(t1).normalized();
    (t1, t2)
}

fn compute_effective_mass(
    inv_mass_a: f64, inv_inertia_a: &Mat3, r_a: Vec3,
    inv_mass_b: f64, inv_inertia_b: &Mat3, r_b: Vec3,
    direction: Vec3,
) -> f64 {
    let rn_a = r_a.cross(direction);
    let rn_b = r_b.cross(direction);
    let k = inv_mass_a + inv_mass_b
        + rn_a.dot(inv_inertia_a.mul_vec(rn_a))
        + rn_b.dot(inv_inertia_b.mul_vec(rn_b));
    if k > 1e-12 { 1.0 / k } else { 0.0 }
}

// ── Solver ───────────────────────────────────────────────────

/// 3D sequential impulse constraint solver.
pub struct ConstraintSolver3D {
    pub config: SolverConfig,
    constraints: Vec<ContactConstraint>,
}

impl ConstraintSolver3D {
    pub fn new(config: SolverConfig) -> Self {
        Self { config, constraints: Vec::new() }
    }

    pub fn with_default_config() -> Self {
        Self::new(SolverConfig::default())
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    pub fn add_constraint(&mut self, constraint: ContactConstraint) {
        self.constraints.push(constraint);
    }

    /// Pre-solve: compute effective masses, velocity biases, and apply warm-starting.
    pub fn prepare(&mut self, bodies: &mut [SolverBody], dt: f64) {
        let inv_dt = if dt > 1e-12 { 1.0 / dt } else { 0.0 };

        for c in &mut self.constraints {
            let pos_a = bodies[c.body_a].position;
            let pos_b = bodies[c.body_b].position;
            c.r_a = c.contact_point - pos_a;
            c.r_b = c.contact_point - pos_b;

            let ba = &bodies[c.body_a];
            let bb = &bodies[c.body_b];

            c.effective_mass_normal = compute_effective_mass(
                ba.inv_mass, &ba.inv_inertia, c.r_a,
                bb.inv_mass, &bb.inv_inertia, c.r_b,
                c.normal,
            );
            c.effective_mass_tangent1 = compute_effective_mass(
                ba.inv_mass, &ba.inv_inertia, c.r_a,
                bb.inv_mass, &bb.inv_inertia, c.r_b,
                c.tangent1,
            );
            c.effective_mass_tangent2 = compute_effective_mass(
                ba.inv_mass, &ba.inv_inertia, c.r_a,
                bb.inv_mass, &bb.inv_inertia, c.r_b,
                c.tangent2,
            );

            // Velocity bias for restitution
            // rel_vel convention: vA - vB (positive vn = separating, negative = approaching)
            let rel_vel = ba.velocity_at(c.r_a) - bb.velocity_at(c.r_b);
            let vn = rel_vel.dot(c.normal);
            c.velocity_bias = 0.0;
            if vn < -self.config.restitution_threshold {
                c.velocity_bias = -c.restitution * vn; // bounce
            }
            if !self.config.use_split_impulse {
                // Baumgarte position correction via velocity
                let penetration_correction = (c.depth - self.config.slop).max(0.0);
                c.velocity_bias += self.config.baumgarte * inv_dt * penetration_correction;
            }

            // Warm-starting
            let warm = self.config.warmstart_factor;
            let impulse = c.normal * (c.accumulated_normal * warm)
                + c.tangent1 * (c.accumulated_tangent1 * warm)
                + c.tangent2 * (c.accumulated_tangent2 * warm);
            c.accumulated_normal *= warm;
            c.accumulated_tangent1 *= warm;
            c.accumulated_tangent2 *= warm;

            bodies[c.body_a].apply_impulse(impulse, c.r_a);
            bodies[c.body_b].apply_impulse(-impulse, c.r_b);
        }
    }

    /// Solve velocity constraints.
    pub fn solve_velocities(&mut self, bodies: &mut [SolverBody]) {
        for _ in 0..self.config.velocity_iterations {
            for ci in 0..self.constraints.len() {
                // Normal constraint
                {
                    let c = &self.constraints[ci];
                    let ba_idx = c.body_a;
                    let bb_idx = c.body_b;
                    let r_a = c.r_a;
                    let r_b = c.r_b;
                    let normal = c.normal;
                    let eff_mass = c.effective_mass_normal;
                    let bias = c.velocity_bias;

                    let rel_vel = bodies[ba_idx].velocity_at(r_a) - bodies[bb_idx].velocity_at(r_b);
                    let vn = rel_vel.dot(normal);
                    let lambda = eff_mass * (-vn + bias);

                    let old_acc = self.constraints[ci].accumulated_normal;
                    self.constraints[ci].accumulated_normal = (old_acc + lambda).max(0.0);
                    let applied = self.constraints[ci].accumulated_normal - old_acc;

                    let impulse = normal * applied;
                    bodies[ba_idx].apply_impulse(impulse, r_a);
                    bodies[bb_idx].apply_impulse(-impulse, r_b);
                }

                // Tangent 1 (friction)
                {
                    let c = &self.constraints[ci];
                    let ba_idx = c.body_a;
                    let bb_idx = c.body_b;
                    let r_a = c.r_a;
                    let r_b = c.r_b;
                    let tangent = c.tangent1;
                    let eff_mass = c.effective_mass_tangent1;
                    let friction = c.friction;
                    let max_friction = friction * self.constraints[ci].accumulated_normal;

                    let rel_vel = bodies[ba_idx].velocity_at(r_a) - bodies[bb_idx].velocity_at(r_b);
                    let vt = rel_vel.dot(tangent);
                    let lambda = eff_mass * (-vt);

                    let old_acc = self.constraints[ci].accumulated_tangent1;
                    self.constraints[ci].accumulated_tangent1 =
                        (old_acc + lambda).clamp(-max_friction, max_friction);
                    let applied = self.constraints[ci].accumulated_tangent1 - old_acc;

                    let impulse = tangent * applied;
                    bodies[ba_idx].apply_impulse(impulse, r_a);
                    bodies[bb_idx].apply_impulse(-impulse, r_b);
                }

                // Tangent 2 (friction)
                {
                    let c = &self.constraints[ci];
                    let ba_idx = c.body_a;
                    let bb_idx = c.body_b;
                    let r_a = c.r_a;
                    let r_b = c.r_b;
                    let tangent = c.tangent2;
                    let eff_mass = c.effective_mass_tangent2;
                    let friction = c.friction;
                    let max_friction = friction * self.constraints[ci].accumulated_normal;

                    let rel_vel = bodies[ba_idx].velocity_at(r_a) - bodies[bb_idx].velocity_at(r_b);
                    let vt = rel_vel.dot(tangent);
                    let lambda = eff_mass * (-vt);

                    let old_acc = self.constraints[ci].accumulated_tangent2;
                    self.constraints[ci].accumulated_tangent2 =
                        (old_acc + lambda).clamp(-max_friction, max_friction);
                    let applied = self.constraints[ci].accumulated_tangent2 - old_acc;

                    let impulse = tangent * applied;
                    bodies[ba_idx].apply_impulse(impulse, r_a);
                    bodies[bb_idx].apply_impulse(-impulse, r_b);
                }
            }
        }
    }

    /// Solve position constraints using split impulse / pseudo-velocities.
    pub fn solve_positions(&mut self, bodies: &mut [SolverBody], dt: f64) {
        if !self.config.use_split_impulse { return; }
        let inv_dt = if dt > 1e-12 { 1.0 / dt } else { 0.0 };

        for _ in 0..self.config.position_iterations {
            for ci in 0..self.constraints.len() {
                let c = &self.constraints[ci];
                let penetration = (c.depth - self.config.slop).max(0.0);
                if penetration < 1e-8 { continue; }

                let ba_idx = c.body_a;
                let bb_idx = c.body_b;
                let r_a = c.r_a;
                let r_b = c.r_b;
                let normal = c.normal;
                let eff_mass = c.effective_mass_normal;

                let bias = self.config.baumgarte * inv_dt * penetration;
                let pseudo_rel = bodies[ba_idx].pseudo_linear + bodies[ba_idx].pseudo_angular.cross(r_a)
                    - bodies[bb_idx].pseudo_linear - bodies[bb_idx].pseudo_angular.cross(r_b);
                let pseudo_vn = pseudo_rel.dot(normal);
                let lambda = eff_mass * (bias - pseudo_vn);
                let lambda = lambda.max(0.0);

                let impulse = normal * lambda;
                bodies[ba_idx].apply_pseudo_impulse(impulse, r_a);
                bodies[bb_idx].apply_pseudo_impulse(-impulse, r_b);
            }
        }
    }

    /// Full solve step: prepare, solve velocities, solve positions.
    pub fn solve(&mut self, bodies: &mut [SolverBody], dt: f64) {
        self.prepare(bodies, dt);
        self.solve_velocities(bodies);
        self.solve_positions(bodies, dt);
    }
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_dynamic_body(mass: f64) -> SolverBody {
        let inv = 1.0 / mass;
        SolverBody::new(inv, Mat3::diagonal(inv, inv, inv), Vec3::ZERO)
    }

    fn make_static_body() -> SolverBody {
        SolverBody::new(0.0, Mat3::ZERO, Vec3::ZERO)
    }

    #[test]
    fn test_solver_config_default() {
        let c = SolverConfig::default();
        assert_eq!(c.velocity_iterations, 8);
        assert_eq!(c.position_iterations, 3);
        assert!(approx(c.baumgarte, 0.2));
    }

    #[test]
    fn test_solver_body_static() {
        let b = make_static_body();
        assert!(b.is_static());
    }

    #[test]
    fn test_solver_body_dynamic() {
        let b = make_dynamic_body(1.0);
        assert!(!b.is_static());
    }

    #[test]
    fn test_apply_impulse() {
        let mut b = make_dynamic_body(2.0);
        b.apply_impulse(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        // dv = impulse * inv_mass = 4 * 0.5 = 2
        assert!(approx(b.linear_velocity.x, 2.0));
    }

    #[test]
    fn test_apply_impulse_with_angular() {
        let mut b = make_dynamic_body(1.0);
        b.apply_impulse(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // angular = inv_I * (r x impulse) = inv_I * (1,0,0) x (0,1,0) = inv_I * (0,0,1)
        assert!(b.angular_velocity.z.abs() > EPS);
    }

    #[test]
    fn test_velocity_at_point() {
        let mut b = make_dynamic_body(1.0);
        b.angular_velocity = Vec3::new(0.0, 0.0, 1.0);
        let v = b.velocity_at(Vec3::new(1.0, 0.0, 0.0));
        // omega x r = (0,0,1) x (1,0,0) = (0,1,0)
        assert!(approx(v.y, 1.0));
    }

    #[test]
    fn test_compute_tangent_frame_orthogonal() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let (t1, t2) = compute_tangent_frame(n);
        assert!(approx(n.dot(t1), 0.0));
        assert!(approx(n.dot(t2), 0.0));
        assert!(approx(t1.dot(t2), 0.0));
        assert!(approx(t1.length(), 1.0));
        assert!(approx(t2.length(), 1.0));
    }

    #[test]
    fn test_compute_tangent_frame_x_aligned() {
        let n = Vec3::new(1.0, 0.0, 0.0);
        let (t1, t2) = compute_tangent_frame(n);
        assert!(approx(n.dot(t1), 0.0));
        assert!(approx(n.dot(t2), 0.0));
    }

    #[test]
    fn test_contact_constraint_creation() {
        let c = ContactConstraint::new(
            0, 1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.01, 0.5, 0.3,
        );
        assert_eq!(c.body_a, 0);
        assert_eq!(c.body_b, 1);
        assert!(approx(c.restitution, 0.5));
        assert!(approx(c.friction, 0.3));
    }

    #[test]
    fn test_effective_mass_two_dynamic() {
        let inv_m = 1.0;
        let inv_i = Mat3::IDENTITY;
        let r_a = Vec3::new(0.0, 0.5, 0.0);
        let r_b = Vec3::new(0.0, -0.5, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let em = compute_effective_mass(inv_m, &inv_i, r_a, inv_m, &inv_i, r_b, n);
        assert!(em > 0.0);
    }

    #[test]
    fn test_effective_mass_one_static() {
        let inv_i = Mat3::IDENTITY;
        let r = Vec3::ZERO;
        let n = Vec3::new(0.0, 1.0, 0.0);
        let em = compute_effective_mass(1.0, &inv_i, r, 0.0, &Mat3::ZERO, r, n);
        assert!(em > 0.0);
    }

    #[test]
    fn test_solver_add_constraints() {
        let mut solver = ConstraintSolver3D::with_default_config();
        solver.add_constraint(ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.01, 0.0, 0.3,
        ));
        assert_eq!(solver.constraint_count(), 1);
    }

    #[test]
    fn test_solver_clear_constraints() {
        let mut solver = ConstraintSolver3D::with_default_config();
        solver.add_constraint(ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.01, 0.0, 0.3,
        ));
        solver.clear_constraints();
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_solver_normal_impulse_separates() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];
        bodies[0].linear_velocity = Vec3::new(0.0, -5.0, 0.0);

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.config.use_split_impulse = false;
        solver.add_constraint(ContactConstraint::new(
            0, 1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.01, 0.0, 0.0,
        ));

        solver.solve(&mut bodies, 1.0 / 60.0);
        // Normal impulse should stop the downward velocity
        assert!(bodies[0].linear_velocity.y >= -EPS);
    }

    #[test]
    fn test_solver_restitution() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];
        bodies[0].linear_velocity = Vec3::new(0.0, -10.0, 0.0);

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.config.use_split_impulse = false;
        solver.config.restitution_threshold = 0.5;
        solver.add_constraint(ContactConstraint::new(
            0, 1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.01, 1.0, 0.0, // full restitution
        ));

        solver.solve(&mut bodies, 1.0 / 60.0);
        // With full restitution, body should bounce up
        assert!(bodies[0].linear_velocity.y > 0.0);
    }

    #[test]
    fn test_solver_friction_slows_tangent() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];
        // Body sliding along x, resting on y=0
        bodies[0].linear_velocity = Vec3::new(10.0, 0.0, 0.0);

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.config.use_split_impulse = false;
        // Give it a penetrating contact so there's a normal force for friction
        solver.add_constraint(ContactConstraint::new(
            0, 1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.02, 0.0, 0.5, // friction = 0.5
        ));

        solver.solve(&mut bodies, 1.0 / 60.0);
        // Friction should have reduced tangential velocity
        assert!(bodies[0].linear_velocity.x < 10.0);
    }

    #[test]
    fn test_solver_warm_starting() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];
        bodies[0].linear_velocity = Vec3::new(0.0, -5.0, 0.0);

        let mut solver = ConstraintSolver3D::with_default_config();
        let mut c = ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.01, 0.0, 0.0,
        );
        c.accumulated_normal = 3.0; // warm-start hint
        solver.add_constraint(c);

        solver.solve(&mut bodies, 1.0 / 60.0);
        // Should converge faster with warm-starting
        assert!(bodies[0].linear_velocity.y >= -EPS);
    }

    #[test]
    fn test_solver_position_correction() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.config.use_split_impulse = true;
        solver.add_constraint(ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.05, 0.0, 0.0,
        ));

        solver.solve(&mut bodies, 1.0 / 60.0);
        // Pseudo velocities should push body upward
        assert!(bodies[0].pseudo_linear.y > 0.0);
    }

    #[test]
    fn test_solver_two_dynamic_bodies() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_dynamic_body(1.0),
        ];
        bodies[0].linear_velocity = Vec3::new(5.0, 0.0, 0.0);
        bodies[0].position = Vec3::new(-0.5, 0.0, 0.0);
        bodies[1].position = Vec3::new(0.5, 0.0, 0.0);

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.config.use_split_impulse = false;
        solver.add_constraint(ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.01, 0.0, 0.0,
        ));

        let v_before = bodies[0].linear_velocity.x + bodies[1].linear_velocity.x;
        solver.solve(&mut bodies, 1.0 / 60.0);
        let v_after = bodies[0].linear_velocity.x + bodies[1].linear_velocity.x;

        // Momentum should be approximately conserved
        assert!(approx(v_before, v_after));
    }

    #[test]
    fn test_no_penetration_no_correction() {
        let mut bodies = vec![
            make_dynamic_body(1.0),
            make_static_body(),
        ];

        let mut solver = ConstraintSolver3D::with_default_config();
        solver.add_constraint(ContactConstraint::new(
            0, 1, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0, 0.0,
        ));

        solver.solve(&mut bodies, 1.0 / 60.0);
        // No penetration, so no position correction
        assert!(approx(bodies[0].pseudo_linear.y, 0.0));
    }
}
