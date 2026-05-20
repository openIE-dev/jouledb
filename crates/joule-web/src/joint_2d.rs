//! 2D physics joints / constraints — revolute (hinge), distance (rod),
//! prismatic (slider), spring (Hooke + damping), rope (max distance),
//! weld (fixed relative transform).  Joint limits, motors, break force.

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
    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Body snapshot ────────────────────────────────────────────

/// Minimal body data needed by joint solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodySnapshot {
    pub position: Vec2,
    pub rotation: f64,
    pub linear_velocity: Vec2,
    pub angular_velocity: f64,
    pub inv_mass: f64,
    pub inv_inertia: f64,
}

impl BodySnapshot {
    pub fn new_static() -> Self {
        Self {
            position: Vec2::zero(), rotation: 0.0,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            inv_mass: 0.0, inv_inertia: 0.0,
        }
    }
    pub fn new_dynamic(pos: Vec2, rot: f64, mass: f64, inertia: f64) -> Self {
        Self {
            position: pos, rotation: rot,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inv_inertia: if inertia > 0.0 { 1.0 / inertia } else { 0.0 },
        }
    }
}

// ── Joint limits ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointLimits {
    pub enabled: bool,
    pub lower: f64,
    pub upper: f64,
}

impl JointLimits {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { enabled: true, lower, upper }
    }
    pub fn disabled() -> Self {
        Self { enabled: false, lower: 0.0, upper: 0.0 }
    }
    pub fn clamp(&self, value: f64) -> f64 {
        if !self.enabled { return value; }
        value.clamp(self.lower, self.upper)
    }
}

impl Default for JointLimits {
    fn default() -> Self { Self::disabled() }
}

// ── Motor ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMotor {
    pub enabled: bool,
    pub target_velocity: f64,
    pub max_torque: f64,
}

impl JointMotor {
    pub fn new(target_velocity: f64, max_torque: f64) -> Self {
        Self { enabled: true, target_velocity, max_torque }
    }
    pub fn disabled() -> Self {
        Self { enabled: false, target_velocity: 0.0, max_torque: 0.0 }
    }
}

impl Default for JointMotor {
    fn default() -> Self { Self::disabled() }
}

// ── Joint result ─────────────────────────────────────────────

/// Impulse/correction produced by a joint step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointImpulse {
    pub linear_a: Vec2,
    pub angular_a: f64,
    pub linear_b: Vec2,
    pub angular_b: f64,
    /// Magnitude of the constraint force this step.
    pub force_magnitude: f64,
}

impl JointImpulse {
    pub fn zero() -> Self {
        Self { linear_a: Vec2::zero(), angular_a: 0.0, linear_b: Vec2::zero(), angular_b: 0.0, force_magnitude: 0.0 }
    }
}

// ── Joint types ──────────────────────────────────────────────

pub type JointId = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum Joint2D {
    /// Revolute (hinge): rotation around an anchor point.
    Revolute {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        limits: JointLimits,
        motor: JointMotor,
        break_force: f64,
        broken: bool,
    },
    /// Distance: fixed-length rod between two anchor points.
    Distance {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        target_distance: f64,
        break_force: f64,
        broken: bool,
    },
    /// Prismatic: slider along an axis.
    Prismatic {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        axis: Vec2,
        limits: JointLimits,
        motor: JointMotor,
        break_force: f64,
        broken: bool,
    },
    /// Spring: Hooke's law with damping coefficient.
    Spring {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
        break_force: f64,
        broken: bool,
    },
    /// Rope: max distance constraint (no min).
    Rope {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        max_length: f64,
        break_force: f64,
        broken: bool,
    },
    /// Weld: fixed relative transform.
    Weld {
        id: JointId,
        body_a: u64,
        body_b: u64,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        reference_angle: f64,
        break_force: f64,
        broken: bool,
    },
}

impl Joint2D {
    pub fn id(&self) -> JointId {
        match self {
            Self::Revolute { id, .. } | Self::Distance { id, .. } |
            Self::Prismatic { id, .. } | Self::Spring { id, .. } |
            Self::Rope { id, .. } | Self::Weld { id, .. } => *id,
        }
    }

    pub fn body_a(&self) -> u64 {
        match self {
            Self::Revolute { body_a, .. } | Self::Distance { body_a, .. } |
            Self::Prismatic { body_a, .. } | Self::Spring { body_a, .. } |
            Self::Rope { body_a, .. } | Self::Weld { body_a, .. } => *body_a,
        }
    }

    pub fn body_b(&self) -> u64 {
        match self {
            Self::Revolute { body_b, .. } | Self::Distance { body_b, .. } |
            Self::Prismatic { body_b, .. } | Self::Spring { body_b, .. } |
            Self::Rope { body_b, .. } | Self::Weld { body_b, .. } => *body_b,
        }
    }

    pub fn is_broken(&self) -> bool {
        match self {
            Self::Revolute { broken, .. } | Self::Distance { broken, .. } |
            Self::Prismatic { broken, .. } | Self::Spring { broken, .. } |
            Self::Rope { broken, .. } | Self::Weld { broken, .. } => *broken,
        }
    }

    fn break_force(&self) -> f64 {
        match self {
            Self::Revolute { break_force, .. } | Self::Distance { break_force, .. } |
            Self::Prismatic { break_force, .. } | Self::Spring { break_force, .. } |
            Self::Rope { break_force, .. } | Self::Weld { break_force, .. } => *break_force,
        }
    }

    fn set_broken(&mut self) {
        match self {
            Self::Revolute { broken, .. } | Self::Distance { broken, .. } |
            Self::Prismatic { broken, .. } | Self::Spring { broken, .. } |
            Self::Rope { broken, .. } | Self::Weld { broken, .. } => *broken = true,
        }
    }

    /// Solve the joint constraint for one iteration.
    pub fn solve(&mut self, a: &BodySnapshot, b: &BodySnapshot, dt: f64) -> JointImpulse {
        if self.is_broken() { return JointImpulse::zero(); }
        let imp = match self {
            Self::Revolute { local_anchor_a, local_anchor_b, limits, motor, .. } => {
                solve_revolute(a, b, *local_anchor_a, *local_anchor_b, limits, motor, dt)
            }
            Self::Distance { local_anchor_a, local_anchor_b, target_distance, .. } => {
                solve_distance(a, b, *local_anchor_a, *local_anchor_b, *target_distance, dt)
            }
            Self::Prismatic { local_anchor_a, axis, limits, motor, .. } => {
                solve_prismatic(a, b, *local_anchor_a, *axis, limits, motor, dt)
            }
            Self::Spring { local_anchor_a, local_anchor_b, rest_length, stiffness, damping, .. } => {
                solve_spring(a, b, *local_anchor_a, *local_anchor_b, *rest_length, *stiffness, *damping, dt)
            }
            Self::Rope { local_anchor_a, local_anchor_b, max_length, .. } => {
                solve_rope(a, b, *local_anchor_a, *local_anchor_b, *max_length, dt)
            }
            Self::Weld { local_anchor_a, local_anchor_b, reference_angle, .. } => {
                solve_weld(a, b, *local_anchor_a, *local_anchor_b, *reference_angle, dt)
            }
        };

        // Check break force.
        let bf = self.break_force();
        if bf > 0.0 && imp.force_magnitude > bf {
            self.set_broken();
            return JointImpulse::zero();
        }
        imp
    }
}

// ── Solvers ──────────────────────────────────────────────────

fn world_anchor(body: &BodySnapshot, local: Vec2) -> Vec2 {
    body.position.add(local.rotate(body.rotation))
}

fn solve_revolute(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, lb: Vec2,
    limits: &JointLimits, motor: &JointMotor,
    dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let wb = world_anchor(b, lb);
    let error = wb.sub(wa);
    let dist = error.length();

    // Position correction impulse.
    let bias = 0.2 / dt;
    let correction = error.scale(bias);
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }
    let impulse = correction.scale(1.0 / inv_mass_sum);

    let mut result = JointImpulse {
        linear_a: impulse.scale(a.inv_mass),
        angular_a: 0.0,
        linear_b: impulse.negate().scale(b.inv_mass),
        angular_b: 0.0,
        force_magnitude: impulse.length() / dt.max(1e-12),
    };

    // Angular limits.
    if limits.enabled {
        let angle_diff = b.rotation - a.rotation;
        if angle_diff < limits.lower {
            result.angular_a = -(limits.lower - angle_diff) * bias * a.inv_inertia;
            result.angular_b = (limits.lower - angle_diff) * bias * b.inv_inertia;
        } else if angle_diff > limits.upper {
            result.angular_a = (angle_diff - limits.upper) * bias * a.inv_inertia;
            result.angular_b = -(angle_diff - limits.upper) * bias * b.inv_inertia;
        }
    }

    // Motor.
    if motor.enabled {
        let rel_omega = b.angular_velocity - a.angular_velocity;
        let motor_impulse = (motor.target_velocity - rel_omega).clamp(-motor.max_torque * dt, motor.max_torque * dt);
        result.angular_a -= motor_impulse * a.inv_inertia;
        result.angular_b += motor_impulse * b.inv_inertia;
    }

    result
}

fn solve_distance(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, lb: Vec2,
    target: f64, dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let wb = world_anchor(b, lb);
    let d = wb.sub(wa);
    let dist = d.length();
    if dist < 1e-12 { return JointImpulse::zero(); }
    let dir = d.scale(1.0 / dist);
    let error = dist - target;
    let bias = 0.2 / dt;
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }
    let lambda = -error * bias / inv_mass_sum;
    let impulse = dir.scale(lambda);
    JointImpulse {
        linear_a: impulse.negate().scale(a.inv_mass),
        angular_a: 0.0,
        linear_b: impulse.scale(b.inv_mass),
        angular_b: 0.0,
        force_magnitude: lambda.abs() / dt.max(1e-12),
    }
}

fn solve_prismatic(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, axis: Vec2,
    limits: &JointLimits, motor: &JointMotor,
    dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let d = b.position.sub(wa);
    let world_axis = axis.rotate(a.rotation).normalized();
    let lateral = d.sub(world_axis.scale(d.dot(world_axis)));
    let lateral_len = lateral.length();

    let bias = 0.2 / dt;
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }

    // Correct lateral drift.
    let correction = if lateral_len > 1e-12 {
        lateral.scale(-bias / inv_mass_sum)
    } else {
        Vec2::zero()
    };

    let mut result = JointImpulse {
        linear_a: correction.negate().scale(a.inv_mass),
        angular_a: 0.0,
        linear_b: correction.scale(b.inv_mass),
        angular_b: 0.0,
        force_magnitude: correction.length() / dt.max(1e-12),
    };

    // Limits along axis.
    if limits.enabled {
        let translation = d.dot(world_axis);
        if translation < limits.lower {
            let push = world_axis.scale((limits.lower - translation) * bias / inv_mass_sum);
            result.linear_b = result.linear_b.add(push.scale(b.inv_mass));
            result.linear_a = result.linear_a.sub(push.scale(a.inv_mass));
        } else if translation > limits.upper {
            let push = world_axis.scale((limits.upper - translation) * bias / inv_mass_sum);
            result.linear_b = result.linear_b.add(push.scale(b.inv_mass));
            result.linear_a = result.linear_a.sub(push.scale(a.inv_mass));
        }
    }

    // Motor along axis.
    if motor.enabled {
        let rel_vel = b.linear_velocity.sub(a.linear_velocity).dot(world_axis);
        let motor_impulse = (motor.target_velocity - rel_vel).clamp(-motor.max_torque * dt, motor.max_torque * dt);
        let motor_vec = world_axis.scale(motor_impulse);
        result.linear_a = result.linear_a.sub(motor_vec.scale(a.inv_mass));
        result.linear_b = result.linear_b.add(motor_vec.scale(b.inv_mass));
    }

    result
}

fn solve_spring(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, lb: Vec2,
    rest: f64, stiffness: f64, damp: f64, dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let wb = world_anchor(b, lb);
    let d = wb.sub(wa);
    let dist = d.length();
    if dist < 1e-12 { return JointImpulse::zero(); }
    let dir = d.scale(1.0 / dist);
    let extension = dist - rest;

    // Hooke's law: F = -k * x
    let spring_force = stiffness * extension;

    // Damping: F_d = -c * v_rel along spring axis
    let rel_vel = b.linear_velocity.sub(a.linear_velocity).dot(dir);
    let damp_force = damp * rel_vel;

    let total = (spring_force + damp_force) * dt;
    let impulse = dir.scale(total);

    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }

    JointImpulse {
        linear_a: impulse.scale(a.inv_mass),
        angular_a: 0.0,
        linear_b: impulse.negate().scale(b.inv_mass),
        angular_b: 0.0,
        force_magnitude: total.abs() / dt.max(1e-12),
    }
}

fn solve_rope(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, lb: Vec2,
    max_len: f64, dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let wb = world_anchor(b, lb);
    let d = wb.sub(wa);
    let dist = d.length();
    if dist <= max_len || dist < 1e-12 { return JointImpulse::zero(); }
    let dir = d.scale(1.0 / dist);
    let error = dist - max_len;
    let bias = 0.2 / dt;
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }
    let lambda = -error * bias / inv_mass_sum;
    let impulse = dir.scale(lambda);
    JointImpulse {
        linear_a: impulse.negate().scale(a.inv_mass),
        angular_a: 0.0,
        linear_b: impulse.scale(b.inv_mass),
        angular_b: 0.0,
        force_magnitude: lambda.abs() / dt.max(1e-12),
    }
}

fn solve_weld(
    a: &BodySnapshot, b: &BodySnapshot,
    la: Vec2, lb: Vec2,
    ref_angle: f64, dt: f64,
) -> JointImpulse {
    let wa = world_anchor(a, la);
    let wb = world_anchor(b, lb);
    let error = wb.sub(wa);
    let bias = 0.2 / dt;
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return JointImpulse::zero(); }
    let impulse = error.scale(bias / inv_mass_sum);
    let angle_error = (b.rotation - a.rotation) - ref_angle;
    let inv_i_sum = a.inv_inertia + b.inv_inertia;
    let ang_impulse = if inv_i_sum > 1e-12 { angle_error * bias / inv_i_sum } else { 0.0 };
    JointImpulse {
        linear_a: impulse.scale(a.inv_mass),
        angular_a: ang_impulse * a.inv_inertia,
        linear_b: impulse.negate().scale(b.inv_mass),
        angular_b: -ang_impulse * b.inv_inertia,
        force_magnitude: impulse.length() / dt.max(1e-12),
    }
}

// ── Joint storage ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct JointSet {
    joints: Vec<Joint2D>,
    next_id: JointId,
}

impl JointSet {
    pub fn new() -> Self { Self { joints: Vec::new(), next_id: 1 } }

    pub fn add(&mut self, mut joint: Joint2D) -> JointId {
        let id = self.next_id;
        self.next_id += 1;
        // Patch the id.
        match &mut joint {
            Joint2D::Revolute { id: jid, .. } | Joint2D::Distance { id: jid, .. } |
            Joint2D::Prismatic { id: jid, .. } | Joint2D::Spring { id: jid, .. } |
            Joint2D::Rope { id: jid, .. } | Joint2D::Weld { id: jid, .. } => *jid = id,
        }
        self.joints.push(joint);
        id
    }

    pub fn remove(&mut self, id: JointId) -> Option<Joint2D> {
        let idx = self.joints.iter().position(|j| j.id() == id)?;
        Some(self.joints.swap_remove(idx))
    }

    pub fn get(&self, id: JointId) -> Option<&Joint2D> {
        self.joints.iter().find(|j| j.id() == id)
    }

    pub fn len(&self) -> usize { self.joints.len() }
    pub fn is_empty(&self) -> bool { self.joints.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = &Joint2D> { self.joints.iter() }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Joint2D> { self.joints.iter_mut() }
}

impl Default for JointSet {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v2_approx(a: Vec2, b: Vec2) -> bool { approx(a.x, b.x) && approx(a.y, b.y) }

    fn body_at(x: f64, y: f64) -> BodySnapshot {
        BodySnapshot::new_dynamic(Vec2::new(x, y), 0.0, 1.0, 1.0)
    }

    // ── Revolute ──

    #[test]
    fn revolute_no_error() {
        let a = body_at(0.0, 0.0);
        let b = body_at(0.0, 0.0);
        let mut j = Joint2D::Revolute {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            limits: JointLimits::disabled(), motor: JointMotor::disabled(),
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(v2_approx(imp.linear_a, Vec2::zero()));
    }

    #[test]
    fn revolute_with_error() {
        let a = body_at(0.0, 0.0);
        let b = body_at(2.0, 0.0);
        let mut j = Joint2D::Revolute {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            limits: JointLimits::disabled(), motor: JointMotor::disabled(),
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude > 0.0);
    }

    #[test]
    fn revolute_with_limits() {
        let mut a = body_at(0.0, 0.0);
        let mut b = body_at(0.0, 0.0);
        b.rotation = 2.0; // exceeds upper
        let mut j = Joint2D::Revolute {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            limits: JointLimits::new(-1.0, 1.0), motor: JointMotor::disabled(),
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        // Should apply angular correction.
        assert!(imp.angular_a != 0.0 || imp.angular_b != 0.0);
    }

    #[test]
    fn revolute_motor() {
        let a = body_at(0.0, 0.0);
        let b = body_at(0.0, 0.0);
        let mut j = Joint2D::Revolute {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            limits: JointLimits::disabled(),
            motor: JointMotor::new(5.0, 100.0),
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.angular_b.abs() > 0.0);
    }

    // ── Distance ──

    #[test]
    fn distance_at_rest() {
        let a = body_at(0.0, 0.0);
        let b = body_at(5.0, 0.0);
        let mut j = Joint2D::Distance {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            target_distance: 5.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        // At target distance, error ~0.
        assert!(imp.force_magnitude < 1.0);
    }

    #[test]
    fn distance_stretched() {
        let a = body_at(0.0, 0.0);
        let b = body_at(10.0, 0.0);
        let mut j = Joint2D::Distance {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            target_distance: 5.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude > 0.0);
    }

    // ── Spring ──

    #[test]
    fn spring_compressed() {
        let a = body_at(0.0, 0.0);
        let b = body_at(1.0, 0.0);
        let mut j = Joint2D::Spring {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            rest_length: 5.0, stiffness: 100.0, damping: 0.0,
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        // Spring compressed: should push b away.
        assert!(imp.force_magnitude > 0.0);
    }

    #[test]
    fn spring_at_rest() {
        let a = body_at(0.0, 0.0);
        let b = body_at(5.0, 0.0);
        let mut j = Joint2D::Spring {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            rest_length: 5.0, stiffness: 100.0, damping: 0.0,
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude < 1.0);
    }

    // ── Rope ──

    #[test]
    fn rope_within_length() {
        let a = body_at(0.0, 0.0);
        let b = body_at(3.0, 0.0);
        let mut j = Joint2D::Rope {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            max_length: 5.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude < EPS);
    }

    #[test]
    fn rope_exceeded() {
        let a = body_at(0.0, 0.0);
        let b = body_at(10.0, 0.0);
        let mut j = Joint2D::Rope {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            max_length: 5.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude > 0.0);
    }

    // ── Weld ──

    #[test]
    fn weld_aligned() {
        let a = body_at(0.0, 0.0);
        let b = body_at(0.0, 0.0);
        let mut j = Joint2D::Weld {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            reference_angle: 0.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude < EPS);
    }

    #[test]
    fn weld_misaligned() {
        let a = body_at(0.0, 0.0);
        let mut b = body_at(1.0, 1.0);
        b.rotation = 0.5;
        let mut j = Joint2D::Weld {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            reference_angle: 0.0, break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude > 0.0);
    }

    // ── Break force ──

    #[test]
    fn joint_breaks() {
        let a = body_at(0.0, 0.0);
        let b = body_at(100.0, 0.0);
        let mut j = Joint2D::Distance {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            target_distance: 1.0, break_force: 1.0, broken: false,
        };
        let _imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(j.is_broken());
    }

    #[test]
    fn broken_joint_returns_zero() {
        let a = body_at(0.0, 0.0);
        let b = body_at(10.0, 0.0);
        let mut j = Joint2D::Distance {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            target_distance: 1.0, break_force: 0.0, broken: true,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        assert!(imp.force_magnitude < EPS);
    }

    // ── JointSet ──

    #[test]
    fn joint_set_add_remove() {
        let mut set = JointSet::new();
        let j = Joint2D::Distance {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), local_anchor_b: Vec2::zero(),
            target_distance: 5.0, break_force: 0.0, broken: false,
        };
        let id = set.add(j);
        assert_eq!(set.len(), 1);
        set.remove(id);
        assert!(set.is_empty());
    }

    #[test]
    fn joint_limits_clamp() {
        let lim = JointLimits::new(-1.0, 1.0);
        assert!(approx(lim.clamp(5.0), 1.0));
        assert!(approx(lim.clamp(-5.0), -1.0));
        assert!(approx(lim.clamp(0.5), 0.5));
    }

    #[test]
    fn joint_limits_disabled_passthrough() {
        let lim = JointLimits::disabled();
        assert!(approx(lim.clamp(100.0), 100.0));
    }

    // ── Prismatic ──

    #[test]
    fn prismatic_along_axis() {
        let a = body_at(0.0, 0.0);
        let b = body_at(3.0, 0.0);
        let mut j = Joint2D::Prismatic {
            id: 0, body_a: 1, body_b: 2,
            local_anchor_a: Vec2::zero(), axis: Vec2::new(1.0, 0.0),
            limits: JointLimits::disabled(), motor: JointMotor::disabled(),
            break_force: 0.0, broken: false,
        };
        let imp = j.solve(&a, &b, 1.0 / 60.0);
        // On axis, no lateral error.
        assert!(imp.linear_a.y.abs() < EPS);
    }
}
