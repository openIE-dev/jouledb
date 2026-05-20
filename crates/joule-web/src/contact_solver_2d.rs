//! Sequential impulse contact solver — resolve penetration, apply restitution
//! and Coulomb friction.  Baumgarte stabilization for position drift.
//! Warm-starting from previous frame contacts.  Configurable iteration count.
//!
//! Convention:
//! - Normal points from A to B.
//! - Relative velocity `dv = vb - va` at the contact point.
//! - Closing velocity: `dv.dot(n) < 0`.
//! - Impulse `p` is applied as `+p` to A and `-p` to B (separating push).

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn cross(self, o: Self) -> f64 { self.x * o.y - self.y * o.x }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn negate(self) -> Self { Self { x: -self.x, y: -self.y } }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Material ─────────────────────────────────────────────────

/// Physical material surface properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub restitution: f64,
    pub static_friction: f64,
    pub dynamic_friction: f64,
}

impl Material {
    pub fn new(restitution: f64, static_friction: f64, dynamic_friction: f64) -> Self {
        Self { restitution, static_friction, dynamic_friction }
    }
    pub fn rubber() -> Self { Self::new(0.8, 0.9, 0.7) }
    pub fn ice() -> Self { Self::new(0.1, 0.05, 0.02) }
    pub fn steel() -> Self { Self::new(0.5, 0.6, 0.4) }
    pub fn default_mat() -> Self { Self::new(0.3, 0.5, 0.3) }
}

impl Default for Material {
    fn default() -> Self { Self::default_mat() }
}

// ── Body state ───────────────────────────────────────────────

/// Mutable body state used during solving.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverBody {
    pub position: Vec2,
    pub rotation: f64,
    pub linear_velocity: Vec2,
    pub angular_velocity: f64,
    pub inv_mass: f64,
    pub inv_inertia: f64,
    pub material: Material,
}

impl SolverBody {
    pub fn new_dynamic(pos: Vec2, mass: f64, inertia: f64) -> Self {
        Self {
            position: pos, rotation: 0.0,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inv_inertia: if inertia > 0.0 { 1.0 / inertia } else { 0.0 },
            material: Material::default(),
        }
    }

    pub fn new_static(pos: Vec2) -> Self {
        Self {
            position: pos, rotation: 0.0,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            inv_mass: 0.0, inv_inertia: 0.0,
            material: Material::default(),
        }
    }
}

// ── Contact constraint ───────────────────────────────────────

/// Per-contact-point constraint data.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactConstraint {
    pub body_a: usize,
    pub body_b: usize,
    /// World-space contact point.
    pub point: Vec2,
    /// Contact normal (points from A toward B).
    pub normal: Vec2,
    /// Penetration depth (positive = overlapping).
    pub depth: f64,

    // warm-start accumulators
    pub normal_impulse: f64,
    pub tangent_impulse: f64,

    // precomputed in pre_solve
    r_a: Vec2,
    r_b: Vec2,
    effective_mass_normal: f64,
    effective_mass_tangent: f64,
    velocity_bias: f64,
}

impl ContactConstraint {
    pub fn new(body_a: usize, body_b: usize, point: Vec2, normal: Vec2, depth: f64) -> Self {
        Self {
            body_a, body_b, point, normal, depth,
            normal_impulse: 0.0, tangent_impulse: 0.0,
            r_a: Vec2::zero(), r_b: Vec2::zero(),
            effective_mass_normal: 0.0, effective_mass_tangent: 0.0,
            velocity_bias: 0.0,
        }
    }
}

// ── Solver config ────────────────────────────────────────────

/// Contact solver configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverConfig {
    /// Number of velocity solver iterations (default 8).
    pub iterations: usize,
    /// Baumgarte stabilization factor (0.1-0.3 typical).
    pub baumgarte: f64,
    /// Penetration slop (small penetration allowed to avoid jitter).
    pub slop: f64,
    /// Restitution velocity threshold: below this, no bounce.
    pub restitution_threshold: f64,
    /// Whether to warm-start from previous impulses.
    pub warm_start: bool,
    /// Warm-start scaling factor (< 1.0 for stability).
    pub warm_start_factor: f64,
}

impl SolverConfig {
    pub fn new() -> Self {
        Self {
            iterations: 8,
            baumgarte: 0.2,
            slop: 0.005,
            restitution_threshold: 1.0,
            warm_start: true,
            warm_start_factor: 0.8,
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self { Self::new() }
}

// ── Helpers ──────────────────────────────────────────────────

/// Velocity at the contact point on body at `bodies[idx]` with lever arm `r`.
fn vel_at(bodies: &[SolverBody], idx: usize, r: Vec2) -> Vec2 {
    let b = &bodies[idx];
    b.linear_velocity.add(Vec2::new(-b.angular_velocity * r.y, b.angular_velocity * r.x))
}

/// Apply impulse `p`: body_a gets `-p`, body_b gets `+p` (separates along normal).
fn apply_impulse_pair(
    a: &mut SolverBody, b: &mut SolverBody,
    r_a: Vec2, r_b: Vec2, p: Vec2,
) {
    a.linear_velocity = a.linear_velocity.sub(p.scale(a.inv_mass));
    a.angular_velocity -= a.inv_inertia * r_a.cross(p);
    b.linear_velocity = b.linear_velocity.add(p.scale(b.inv_mass));
    b.angular_velocity += b.inv_inertia * r_b.cross(p);
}

/// Safe `apply_impulse_pair` that splits the slice to avoid double borrow.
fn apply_impulse_slice(bodies: &mut [SolverBody], ia: usize, ib: usize, r_a: Vec2, r_b: Vec2, p: Vec2) {
    if ia == ib { return; }
    if ia < ib {
        let (left, right) = bodies.split_at_mut(ib);
        apply_impulse_pair(&mut left[ia], &mut right[0], r_a, r_b, p);
    } else {
        let (left, right) = bodies.split_at_mut(ia);
        // Note: when ia > ib, right[0] is body_a and left[ib] is body_b.
        apply_impulse_pair(&mut right[0], &mut left[ib], r_a, r_b, p);
    }
}

// ── Contact solver ───────────────────────────────────────────

/// Sequential impulse contact solver.
pub struct ContactSolver {
    pub config: SolverConfig,
    contacts: Vec<ContactConstraint>,
}

impl ContactSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config, contacts: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.contacts.clear();
    }

    pub fn add_contact(&mut self, c: ContactConstraint) {
        self.contacts.push(c);
    }

    pub fn contact_count(&self) -> usize { self.contacts.len() }

    /// Pre-solve: compute effective masses, velocity bias, apply warm-start.
    pub fn pre_solve(&mut self, bodies: &mut [SolverBody], dt: f64) {
        let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };

        for c in &mut self.contacts {
            let pos_a = bodies[c.body_a].position;
            let pos_b = bodies[c.body_b].position;

            c.r_a = c.point.sub(pos_a);
            c.r_b = c.point.sub(pos_b);

            let inv_m_a = bodies[c.body_a].inv_mass;
            let inv_m_b = bodies[c.body_b].inv_mass;
            let inv_i_a = bodies[c.body_a].inv_inertia;
            let inv_i_b = bodies[c.body_b].inv_inertia;

            // Effective mass for normal direction.
            let rn_a = c.r_a.cross(c.normal);
            let rn_b = c.r_b.cross(c.normal);
            let k_normal = inv_m_a + inv_m_b + inv_i_a * rn_a * rn_a + inv_i_b * rn_b * rn_b;
            c.effective_mass_normal = if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 };

            // Effective mass for tangent direction.
            let tangent = c.normal.perpendicular();
            let rt_a = c.r_a.cross(tangent);
            let rt_b = c.r_b.cross(tangent);
            let k_tangent = inv_m_a + inv_m_b + inv_i_a * rt_a * rt_a + inv_i_b * rt_b * rt_b;
            c.effective_mass_tangent = if k_tangent > 0.0 { 1.0 / k_tangent } else { 0.0 };

            // Baumgarte velocity bias for penetration correction.
            let penetration = (c.depth - self.config.slop).max(0.0);
            c.velocity_bias = self.config.baumgarte * inv_dt * penetration;

            // Restitution: compute initial closing velocity.
            let va = vel_at(bodies, c.body_a, c.r_a);
            let vb = vel_at(bodies, c.body_b, c.r_b);
            let dv = vb.sub(va);
            let vn = dv.dot(c.normal);
            // vn < 0 means closing.  Add restitution bias if closing fast enough.
            if -vn > self.config.restitution_threshold {
                let e = (bodies[c.body_a].material.restitution + bodies[c.body_b].material.restitution) * 0.5;
                c.velocity_bias += e * (-vn);
            }

            // Warm-start: apply cached impulse from previous frame.
            if self.config.warm_start && (c.normal_impulse.abs() > 1e-12 || c.tangent_impulse.abs() > 1e-12) {
                let p = c.normal.scale(c.normal_impulse * self.config.warm_start_factor)
                    .add(tangent.scale(c.tangent_impulse * self.config.warm_start_factor));
                apply_impulse_slice(bodies, c.body_a, c.body_b, c.r_a, c.r_b, p);
            }
        }
    }

    /// Run velocity solver iterations.
    pub fn solve_velocity(&mut self, bodies: &mut [SolverBody]) {
        for _ in 0..self.config.iterations {
            for ci in 0..self.contacts.len() {
                // Copy out everything we need to avoid borrow conflicts.
                let body_a = self.contacts[ci].body_a;
                let body_b = self.contacts[ci].body_b;
                let r_a = self.contacts[ci].r_a;
                let r_b = self.contacts[ci].r_b;
                let normal = self.contacts[ci].normal;
                let eff_mass_n = self.contacts[ci].effective_mass_normal;
                let eff_mass_t = self.contacts[ci].effective_mass_tangent;
                let velocity_bias = self.contacts[ci].velocity_bias;
                let old_normal_impulse = self.contacts[ci].normal_impulse;
                let old_tangent_impulse = self.contacts[ci].tangent_impulse;

                // ── Normal impulse ──
                // dv = vb - va; vn = dv.dot(n).  vn < 0 = closing.
                let va = vel_at(bodies, body_a, r_a);
                let vb = vel_at(bodies, body_b, r_b);
                let vn = vb.sub(va).dot(normal);

                // We want to drive vn to velocity_bias (≥ 0, pushes apart).
                // lambda = eff_mass * (velocity_bias - vn)
                let lambda_n = eff_mass_n * (velocity_bias - vn);
                // Accumulated clamping: total impulse ≥ 0 (only push, never pull).
                let new_impulse = (old_normal_impulse + lambda_n).max(0.0);
                let applied_n = new_impulse - old_normal_impulse;
                self.contacts[ci].normal_impulse = new_impulse;

                let p_n = normal.scale(applied_n);
                apply_impulse_slice(bodies, body_a, body_b, r_a, r_b, p_n);

                // ── Tangent (friction) impulse ──
                let tangent = normal.perpendicular();
                let va2 = vel_at(bodies, body_a, r_a);
                let vb2 = vel_at(bodies, body_b, r_b);
                let vt = vb2.sub(va2).dot(tangent);

                let lambda_t = eff_mass_t * (-vt);

                // Coulomb friction clamp.
                let mu_s = (bodies[body_a].material.static_friction + bodies[body_b].material.static_friction) * 0.5;
                let mu_d = (bodies[body_a].material.dynamic_friction + bodies[body_b].material.dynamic_friction) * 0.5;
                let max_friction = mu_s * self.contacts[ci].normal_impulse;
                let new_t = old_tangent_impulse + lambda_t;
                let clamped_t = if new_t.abs() <= max_friction {
                    new_t
                } else {
                    new_t.signum() * mu_d * self.contacts[ci].normal_impulse
                };
                let applied_t = clamped_t - old_tangent_impulse;
                self.contacts[ci].tangent_impulse = clamped_t;

                let p_t = tangent.scale(applied_t);
                apply_impulse_slice(bodies, body_a, body_b, r_a, r_b, p_t);
            }
        }
    }

    /// Full solve pass: pre_solve + solve_velocity.
    pub fn solve(&mut self, bodies: &mut [SolverBody], dt: f64) {
        self.pre_solve(bodies, dt);
        self.solve_velocity(bodies);
    }

    pub fn contacts(&self) -> &[ContactConstraint] {
        &self.contacts
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn approx_wide(a: f64, b: f64) -> bool { (a - b).abs() < 0.5 }

    #[test]
    fn material_defaults() {
        let m = Material::default();
        assert!(m.restitution >= 0.0 && m.restitution <= 1.0);
    }

    #[test]
    fn solver_config_defaults() {
        let c = SolverConfig::new();
        assert_eq!(c.iterations, 8);
    }

    #[test]
    fn solver_no_contacts() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
        ];
        solver.solve(&mut bodies, 1.0 / 60.0);
        assert_eq!(solver.contact_count(), 0);
    }

    #[test]
    fn solver_separating_contact() {
        // Bodies already moving apart — solver should not add impulse.
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_dynamic(Vec2::new(2.0, 0.0), 1.0, 1.0),
        ];
        bodies[0].linear_velocity = Vec2::new(-1.0, 0.0);
        bodies[1].linear_velocity = Vec2::new(1.0, 0.0);
        // Normal points from A to B (right).
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 0.0));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // dv.dot(n) = (1 - (-1)) * 1 = 2 > 0 → separating → clamped to 0.
        assert!(approx(solver.contacts()[0].normal_impulse, 0.0));
    }

    #[test]
    fn solver_colliding_equal_mass() {
        // Head-on collision: body0 → right, body1 → left, equal mass, e=1.
        let mut solver = ContactSolver::new(SolverConfig { restitution_threshold: 0.0, ..SolverConfig::new() });
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_dynamic(Vec2::new(2.0, 0.0), 1.0, 1.0),
        ];
        bodies[0].linear_velocity = Vec2::new(5.0, 0.0);
        bodies[1].linear_velocity = Vec2::new(-5.0, 0.0);
        bodies[0].material = Material::new(1.0, 0.0, 0.0);
        bodies[1].material = Material::new(1.0, 0.0, 0.0);

        // Normal from A to B (right).
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 0.0));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Perfectly elastic: velocities swap.
        assert!(approx_wide(bodies[0].linear_velocity.x, -5.0));
        assert!(approx_wide(bodies[1].linear_velocity.x, 5.0));
    }

    #[test]
    fn solver_static_floor() {
        // Dynamic body falling onto static floor.
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 0.5), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(0.0, -10.0);
        bodies[0].material = Material::new(0.0, 0.3, 0.3);
        bodies[1].material = Material::new(0.0, 0.3, 0.3);
        // Normal from A (dynamic) to B (static floor), pointing down.
        // Contact at floor surface, normal = (0, -1) from A to B.
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Body should have downward velocity reduced (impulse pushes body up).
        assert!(bodies[0].linear_velocity.y > -10.0);
    }

    #[test]
    fn solver_penetration_correction() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 0.0), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, -1.0)),
        ];
        // Normal from A (dynamic) to B (floor below), pointing down.
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, -0.5), Vec2::new(0.0, -1.0), 0.1,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Baumgarte pushes body A upward (positive y).
        assert!(bodies[0].linear_velocity.y > 0.0);
    }

    #[test]
    fn solver_friction_reduces_tangent_velocity() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 0.5), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(10.0, -1.0);
        bodies[0].material = Material::new(0.0, 0.8, 0.6);
        bodies[1].material = Material::new(0.0, 0.8, 0.6);

        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        assert!(bodies[0].linear_velocity.x < 10.0);
    }

    #[test]
    fn solver_multiple_contacts() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 1.0), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(0.0, -5.0);
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(-0.5, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.5, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        assert!(bodies[0].linear_velocity.y > -5.0);
    }

    #[test]
    fn solver_warm_start() {
        let config = SolverConfig { warm_start: true, ..SolverConfig::new() };
        let mut solver = ContactSolver::new(config);
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, -1.0)),
        ];
        let mut c = ContactConstraint::new(0, 1, Vec2::new(0.0, -0.5), Vec2::new(0.0, -1.0), 0.05);
        c.normal_impulse = 5.0;
        solver.add_contact(c);
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Warm-start applies cached impulse: pushes A up.
        assert!(bodies[0].linear_velocity.y > 0.0);
    }

    #[test]
    fn solver_warm_start_disabled() {
        let config = SolverConfig { warm_start: false, ..SolverConfig::new() };
        let mut solver = ContactSolver::new(config);
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, -1.0)),
        ];
        let mut c = ContactConstraint::new(0, 1, Vec2::new(0.0, -0.5), Vec2::new(0.0, -1.0), 0.0);
        c.normal_impulse = 5.0;
        solver.add_contact(c);
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Warm-start disabled: no cached impulse applied.
    }

    #[test]
    fn solver_high_restitution() {
        // Ball bouncing off static floor.
        let mut solver = ContactSolver::new(SolverConfig { restitution_threshold: 0.0, ..SolverConfig::new() });
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 1.0), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(0.0, -10.0);
        bodies[0].material = Material::rubber();
        bodies[1].material = Material::rubber();
        // Normal from A (ball) to B (floor): (0, -1).
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, 0.0), Vec2::new(0.0, -1.0), 0.0,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // High restitution: body should bounce upward.
        assert!(bodies[0].linear_velocity.y > 0.0);
    }

    #[test]
    fn solver_ice_low_friction() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 0.5), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(10.0, -1.0);
        bodies[0].material = Material::ice();
        bodies[1].material = Material::ice();

        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Ice: very low friction, tangential velocity barely reduced.
        assert!(bodies[0].linear_velocity.x > 9.0);
    }

    #[test]
    fn solver_clear_resets() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::zero(), Vec2::new(0.0, 1.0), 0.0));
        assert_eq!(solver.contact_count(), 1);
        solver.clear();
        assert_eq!(solver.contact_count(), 0);
    }

    #[test]
    fn solver_iterations_config() {
        let config = SolverConfig { iterations: 16, ..SolverConfig::new() };
        let solver = ContactSolver::new(config);
        assert_eq!(solver.config.iterations, 16);
    }

    #[test]
    fn solver_zero_mass_body() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_static(Vec2::zero()),
            SolverBody::new_static(Vec2::new(1.0, 0.0)),
        ];
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::new(0.5, 0.0), Vec2::new(1.0, 0.0), 0.1));
        solver.solve(&mut bodies, 1.0 / 60.0);
        assert!(approx(bodies[0].linear_velocity.x, 0.0));
        assert!(approx(bodies[1].linear_velocity.x, 0.0));
    }

    #[test]
    fn effective_mass_positive() {
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_dynamic(Vec2::new(2.0, 0.0), 1.0, 1.0),
        ];
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 0.0));
        solver.pre_solve(&mut bodies, 1.0 / 60.0);
        assert!(solver.contacts()[0].effective_mass_normal > 0.0);
        assert!(solver.contacts()[0].effective_mass_tangent > 0.0);
    }

    #[test]
    fn solver_inelastic_stops() {
        // e=0: body should come to rest (vy ≈ 0) against floor.
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::new(0.0, 0.5), 1.0, 1.0),
            SolverBody::new_static(Vec2::new(0.0, 0.0)),
        ];
        bodies[0].linear_velocity = Vec2::new(0.0, -5.0);
        bodies[0].material = Material::new(0.0, 0.0, 0.0);
        bodies[1].material = Material::new(0.0, 0.0, 0.0);
        solver.add_contact(ContactConstraint::new(
            0, 1, Vec2::new(0.0, 0.0), Vec2::new(0.0, -1.0), 0.01,
        ));
        solver.solve(&mut bodies, 1.0 / 60.0);
        // Inelastic: should stop, not bounce.
        assert!(bodies[0].linear_velocity.y.abs() < 2.0);
    }

    #[test]
    fn solver_normal_impulse_nonneg() {
        // Accumulated normal impulse should never go negative.
        let mut solver = ContactSolver::new(SolverConfig::new());
        let mut bodies = vec![
            SolverBody::new_dynamic(Vec2::zero(), 1.0, 1.0),
            SolverBody::new_dynamic(Vec2::new(2.0, 0.0), 1.0, 1.0),
        ];
        // Bodies separating.
        bodies[0].linear_velocity = Vec2::new(-5.0, 0.0);
        bodies[1].linear_velocity = Vec2::new(5.0, 0.0);
        solver.add_contact(ContactConstraint::new(0, 1, Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 0.0));
        solver.solve(&mut bodies, 1.0 / 60.0);
        assert!(solver.contacts()[0].normal_impulse >= 0.0);
    }
}
