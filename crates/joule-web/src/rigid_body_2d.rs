//! 2D rigid body dynamics — position, rotation, linear/angular velocity, mass,
//! moment of inertia, force/torque accumulators, semi-implicit Euler integration,
//! body types (dynamic/static/kinematic), damping, gravity scale, sleep/wake.

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

// ── Body type ────────────────────────────────────────────────

/// Classification that determines how integration and forces apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Fully simulated — forces, velocity, gravity.
    Dynamic,
    /// Never moves, infinite mass.  Other bodies collide with it.
    Static,
    /// Moved by user code; has velocity but ignores forces.
    Kinematic,
}

// ── Sleep state ──────────────────────────────────────────────

/// Tracks whether a body is sleeping (optimisation to skip integration).
#[derive(Debug, Clone, PartialEq)]
pub struct SleepState {
    /// `true` when the body is sleeping.
    pub sleeping: bool,
    /// Accumulated low-energy time in seconds.
    pub idle_time: f64,
    /// Velocity magnitude below which idle_time accumulates.
    pub sleep_threshold: f64,
    /// Seconds of idleness before entering sleep.
    pub time_to_sleep: f64,
}

impl SleepState {
    pub fn new() -> Self {
        Self {
            sleeping: false,
            idle_time: 0.0,
            sleep_threshold: 0.05,
            time_to_sleep: 0.5,
        }
    }

    /// Update the sleep timer.  Returns `true` if the body just fell asleep.
    pub fn update(&mut self, speed: f64, ang_speed: f64, dt: f64) -> bool {
        if speed < self.sleep_threshold && ang_speed < self.sleep_threshold {
            self.idle_time += dt;
            if self.idle_time >= self.time_to_sleep && !self.sleeping {
                self.sleeping = true;
                return true;
            }
        } else {
            self.idle_time = 0.0;
            self.sleeping = false;
        }
        false
    }

    pub fn wake(&mut self) {
        self.sleeping = false;
        self.idle_time = 0.0;
    }
}

impl Default for SleepState {
    fn default() -> Self { Self::new() }
}

// ── Rigid body ───────────────────────────────────────────────

/// Unique body identifier.
pub type BodyId = u64;

/// A 2D rigid body with transform, velocities, mass properties, and force accumulators.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody2D {
    pub id: BodyId,
    pub body_type: BodyType,

    // transform
    pub position: Vec2,
    pub rotation: f64,

    // velocities
    pub linear_velocity: Vec2,
    pub angular_velocity: f64,

    // mass
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia: f64,
    pub inv_inertia: f64,

    // accumulators (reset after each step)
    pub force: Vec2,
    pub torque: f64,

    // damping  (0 = none, 1 = full)
    pub linear_damping: f64,
    pub angular_damping: f64,

    /// Per-body gravity multiplier (0 = no gravity, 1 = normal, −1 = anti-gravity).
    pub gravity_scale: f64,

    pub sleep: SleepState,
}

impl RigidBody2D {
    /// Create a dynamic body with the given mass and moment of inertia.
    pub fn new_dynamic(id: BodyId, mass: f64, inertia: f64) -> Self {
        assert!(mass > 0.0, "dynamic body must have positive mass");
        assert!(inertia > 0.0, "dynamic body must have positive inertia");
        Self {
            id,
            body_type: BodyType::Dynamic,
            position: Vec2::zero(),
            rotation: 0.0,
            linear_velocity: Vec2::zero(),
            angular_velocity: 0.0,
            mass,
            inv_mass: 1.0 / mass,
            inertia,
            inv_inertia: 1.0 / inertia,
            force: Vec2::zero(),
            torque: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            sleep: SleepState::new(),
        }
    }

    /// Create a static (immovable) body.
    pub fn new_static(id: BodyId) -> Self {
        Self {
            id,
            body_type: BodyType::Static,
            position: Vec2::zero(),
            rotation: 0.0,
            linear_velocity: Vec2::zero(),
            angular_velocity: 0.0,
            mass: 0.0,
            inv_mass: 0.0,
            inertia: 0.0,
            inv_inertia: 0.0,
            force: Vec2::zero(),
            torque: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 0.0,
            sleep: SleepState::new(),
        }
    }

    /// Create a kinematic body (moved by user code).
    pub fn new_kinematic(id: BodyId) -> Self {
        Self {
            id,
            body_type: BodyType::Kinematic,
            position: Vec2::zero(),
            rotation: 0.0,
            linear_velocity: Vec2::zero(),
            angular_velocity: 0.0,
            mass: 0.0,
            inv_mass: 0.0,
            inertia: 0.0,
            inv_inertia: 0.0,
            force: Vec2::zero(),
            torque: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 0.0,
            sleep: SleepState::new(),
        }
    }

    // ── force / impulse API ──

    /// Accumulate a force (applied at centre of mass).
    pub fn apply_force(&mut self, f: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.force = self.force.add(f);
        self.sleep.wake();
    }

    /// Accumulate a force applied at `point` (world-space), generating torque.
    pub fn apply_force_at(&mut self, f: Vec2, point: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.force = self.force.add(f);
        let r = point.sub(self.position);
        self.torque += r.cross(f);
        self.sleep.wake();
    }

    /// Accumulate pure torque.
    pub fn apply_torque(&mut self, t: f64) {
        if self.body_type != BodyType::Dynamic { return; }
        self.torque += t;
        self.sleep.wake();
    }

    /// Apply an instantaneous linear impulse.
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.linear_velocity = self.linear_velocity.add(impulse.scale(self.inv_mass));
        self.sleep.wake();
    }

    /// Apply impulse at a world-space point, affecting both linear and angular velocity.
    pub fn apply_impulse_at(&mut self, impulse: Vec2, point: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.linear_velocity = self.linear_velocity.add(impulse.scale(self.inv_mass));
        let r = point.sub(self.position);
        self.angular_velocity += self.inv_inertia * r.cross(impulse);
        self.sleep.wake();
    }

    /// Apply an instantaneous angular impulse.
    pub fn apply_angular_impulse(&mut self, impulse: f64) {
        if self.body_type != BodyType::Dynamic { return; }
        self.angular_velocity += self.inv_inertia * impulse;
        self.sleep.wake();
    }

    // ── integration ──

    /// Semi-implicit Euler step.  `gravity` is the global gravity vector.
    pub fn integrate(&mut self, dt: f64, gravity: Vec2) {
        if self.body_type == BodyType::Static {
            return;
        }

        if self.body_type == BodyType::Kinematic {
            // Kinematic: just integrate position from velocity, ignore forces.
            self.position = self.position.add(self.linear_velocity.scale(dt));
            self.rotation += self.angular_velocity * dt;
            return;
        }

        // Dynamic — check sleep
        if self.sleep.sleeping {
            self.force = Vec2::zero();
            self.torque = 0.0;
            return;
        }

        // acceleration = force / mass + gravity * gravity_scale
        let accel = self.force.scale(self.inv_mass).add(gravity.scale(self.gravity_scale));
        let ang_accel = self.torque * self.inv_inertia;

        // semi-implicit: update velocity first, then position
        self.linear_velocity = self.linear_velocity.add(accel.scale(dt));
        self.angular_velocity += ang_accel * dt;

        // damping  (exponential decay approximation)
        self.linear_velocity = self.linear_velocity.scale((1.0 - self.linear_damping).max(0.0));
        self.angular_velocity *= (1.0 - self.angular_damping).max(0.0);

        // position / rotation
        self.position = self.position.add(self.linear_velocity.scale(dt));
        self.rotation += self.angular_velocity * dt;

        // clear accumulators
        self.force = Vec2::zero();
        self.torque = 0.0;

        // sleep check
        let speed = self.linear_velocity.length();
        let ang_speed = self.angular_velocity.abs();
        self.sleep.update(speed, ang_speed, dt);
    }

    /// Set the mass (and update inverse).
    pub fn set_mass(&mut self, m: f64) {
        if m <= 0.0 {
            self.mass = 0.0;
            self.inv_mass = 0.0;
        } else {
            self.mass = m;
            self.inv_mass = 1.0 / m;
        }
    }

    /// Set the inertia (and update inverse).
    pub fn set_inertia(&mut self, i: f64) {
        if i <= 0.0 {
            self.inertia = 0.0;
            self.inv_inertia = 0.0;
        } else {
            self.inertia = i;
            self.inv_inertia = 1.0 / i;
        }
    }

    /// Wake the body.
    pub fn wake(&mut self) {
        self.sleep.wake();
    }

    /// Put the body to sleep (zero velocities).
    pub fn put_to_sleep(&mut self) {
        self.sleep.sleeping = true;
        self.sleep.idle_time = self.sleep.time_to_sleep;
        self.linear_velocity = Vec2::zero();
        self.angular_velocity = 0.0;
    }

    /// Velocity of a world-space point on the body (linear + angular contribution).
    pub fn velocity_at(&self, world_point: Vec2) -> Vec2 {
        let r = world_point.sub(self.position);
        // v = linear_velocity + angular_velocity × r  (2D: ω × r = (-ωry, ωrx))
        let tang = Vec2::new(-self.angular_velocity * r.y, self.angular_velocity * r.x);
        self.linear_velocity.add(tang)
    }

    /// Kinetic energy: 0.5 * m * v² + 0.5 * I * ω².
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.linear_velocity.length_sq()
            + 0.5 * self.inertia * self.angular_velocity * self.angular_velocity
    }

    /// Transform a local-space point into world-space.
    pub fn local_to_world(&self, local: Vec2) -> Vec2 {
        self.position.add(local.rotate(self.rotation))
    }

    /// Transform a world-space point into local-space.
    pub fn world_to_local(&self, world: Vec2) -> Vec2 {
        world.sub(self.position).rotate(-self.rotation)
    }
}

// ── Body storage ─────────────────────────────────────────────

/// Simple container for multiple rigid bodies with ID-based access.
#[derive(Debug, Clone)]
pub struct BodySet {
    bodies: Vec<RigidBody2D>,
    next_id: BodyId,
}

impl BodySet {
    pub fn new() -> Self {
        Self { bodies: Vec::new(), next_id: 1 }
    }

    /// Insert a body (id is assigned automatically).  Returns the assigned id.
    pub fn insert_dynamic(&mut self, mass: f64, inertia: f64) -> BodyId {
        let id = self.next_id;
        self.next_id += 1;
        self.bodies.push(RigidBody2D::new_dynamic(id, mass, inertia));
        id
    }

    pub fn insert_static(&mut self) -> BodyId {
        let id = self.next_id;
        self.next_id += 1;
        self.bodies.push(RigidBody2D::new_static(id));
        id
    }

    pub fn insert_kinematic(&mut self) -> BodyId {
        let id = self.next_id;
        self.next_id += 1;
        self.bodies.push(RigidBody2D::new_kinematic(id));
        id
    }

    pub fn get(&self, id: BodyId) -> Option<&RigidBody2D> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn get_mut(&mut self, id: BodyId) -> Option<&mut RigidBody2D> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    pub fn remove(&mut self, id: BodyId) -> Option<RigidBody2D> {
        let idx = self.bodies.iter().position(|b| b.id == id)?;
        Some(self.bodies.swap_remove(idx))
    }

    pub fn len(&self) -> usize { self.bodies.len() }
    pub fn is_empty(&self) -> bool { self.bodies.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = &RigidBody2D> {
        self.bodies.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut RigidBody2D> {
        self.bodies.iter_mut()
    }

    /// Integrate all dynamic bodies.
    pub fn integrate_all(&mut self, dt: f64, gravity: Vec2) {
        for b in &mut self.bodies {
            b.integrate(dt, gravity);
        }
    }
}

impl Default for BodySet {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v2_approx(a: Vec2, b: Vec2) -> bool { approx(a.x, b.x) && approx(a.y, b.y) }

    #[test]
    fn vec2_basics() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx(a.length(), 5.0));
        assert!(approx(a.length_sq(), 25.0));
        let n = a.normalized();
        assert!(approx(n.length(), 1.0));
    }

    #[test]
    fn vec2_arithmetic() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(v2_approx(a.add(b), Vec2::new(4.0, 6.0)));
        assert!(v2_approx(a.sub(b), Vec2::new(-2.0, -2.0)));
        assert!(v2_approx(a.scale(3.0), Vec2::new(3.0, 6.0)));
        assert!(v2_approx(a.negate(), Vec2::new(-1.0, -2.0)));
    }

    #[test]
    fn vec2_dot_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!(approx(a.dot(b), 0.0));
        assert!(approx(a.cross(b), 1.0));
    }

    #[test]
    fn vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(v2_approx(r, Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn vec2_perpendicular() {
        let v = Vec2::new(1.0, 0.0);
        let p = v.perpendicular();
        assert!(v2_approx(p, Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn dynamic_body_creation() {
        let b = RigidBody2D::new_dynamic(1, 10.0, 5.0);
        assert_eq!(b.body_type, BodyType::Dynamic);
        assert!(approx(b.inv_mass, 0.1));
        assert!(approx(b.inv_inertia, 0.2));
        assert!(!b.sleep.sleeping);
    }

    #[test]
    fn static_body_creation() {
        let b = RigidBody2D::new_static(2);
        assert_eq!(b.body_type, BodyType::Static);
        assert!(approx(b.inv_mass, 0.0));
        assert!(approx(b.inv_inertia, 0.0));
    }

    #[test]
    fn kinematic_body_creation() {
        let b = RigidBody2D::new_kinematic(3);
        assert_eq!(b.body_type, BodyType::Kinematic);
        assert!(approx(b.inv_mass, 0.0));
    }

    #[test]
    fn apply_force_accumulates() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.apply_force(Vec2::new(10.0, 0.0));
        b.apply_force(Vec2::new(0.0, 5.0));
        assert!(v2_approx(b.force, Vec2::new(10.0, 5.0)));
    }

    #[test]
    fn apply_force_at_generates_torque() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.position = Vec2::new(0.0, 0.0);
        // force (0,10) at point (1,0) → r=(1,0), torque = r×f = 1*10 - 0*0 = 10
        b.apply_force_at(Vec2::new(0.0, 10.0), Vec2::new(1.0, 0.0));
        assert!(approx(b.torque, 10.0));
    }

    #[test]
    fn static_body_ignores_forces() {
        let mut b = RigidBody2D::new_static(1);
        b.apply_force(Vec2::new(100.0, 100.0));
        assert!(v2_approx(b.force, Vec2::zero()));
    }

    #[test]
    fn integration_no_gravity() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.apply_force(Vec2::new(10.0, 0.0));
        b.integrate(1.0, Vec2::zero());
        // after 1s: v = 10, pos = 10
        assert!(approx(b.linear_velocity.x, 10.0));
        assert!(approx(b.position.x, 10.0));
    }

    #[test]
    fn integration_with_gravity() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        let gravity = Vec2::new(0.0, -9.81);
        b.integrate(1.0, gravity);
        assert!(approx(b.linear_velocity.y, -9.81));
        assert!(approx(b.position.y, -9.81));
    }

    #[test]
    fn gravity_scale_zero_means_no_gravity() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.gravity_scale = 0.0;
        let gravity = Vec2::new(0.0, -9.81);
        b.integrate(1.0, gravity);
        assert!(approx(b.linear_velocity.y, 0.0));
    }

    #[test]
    fn angular_integration() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.apply_torque(2.0);
        b.integrate(1.0, Vec2::zero());
        assert!(approx(b.angular_velocity, 2.0));
        assert!(approx(b.rotation, 2.0));
    }

    #[test]
    fn impulse_changes_velocity() {
        let mut b = RigidBody2D::new_dynamic(1, 2.0, 1.0);
        b.apply_impulse(Vec2::new(10.0, 0.0));
        // Δv = impulse / mass = 10 / 2 = 5
        assert!(approx(b.linear_velocity.x, 5.0));
    }

    #[test]
    fn impulse_at_point() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.position = Vec2::zero();
        b.apply_impulse_at(Vec2::new(0.0, 10.0), Vec2::new(1.0, 0.0));
        assert!(approx(b.linear_velocity.y, 10.0));
        // angular: inv_inertia * r×J = 1 * (1*10) = 10
        assert!(approx(b.angular_velocity, 10.0));
    }

    #[test]
    fn damping_reduces_velocity() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.linear_velocity = Vec2::new(10.0, 0.0);
        b.angular_velocity = 5.0;
        b.linear_damping = 0.1;
        b.angular_damping = 0.1;
        b.integrate(1.0, Vec2::zero());
        assert!(b.linear_velocity.x < 10.0);
        assert!(b.angular_velocity < 5.0);
    }

    #[test]
    fn kinematic_integration() {
        let mut b = RigidBody2D::new_kinematic(1);
        b.linear_velocity = Vec2::new(5.0, 0.0);
        b.angular_velocity = 1.0;
        b.integrate(2.0, Vec2::new(0.0, -9.81));
        // Kinematic: position updates from velocity, gravity ignored.
        assert!(approx(b.position.x, 10.0));
        assert!(approx(b.rotation, 2.0));
    }

    #[test]
    fn sleep_after_idle_time() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.sleep.time_to_sleep = 0.5;
        b.sleep.sleep_threshold = 1.0;
        // Velocity zero → should accumulate idle time.
        for _ in 0..10 {
            b.integrate(0.1, Vec2::zero());
        }
        assert!(b.sleep.sleeping);
    }

    #[test]
    fn wake_on_force() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.put_to_sleep();
        assert!(b.sleep.sleeping);
        b.apply_force(Vec2::new(1.0, 0.0));
        assert!(!b.sleep.sleeping);
    }

    #[test]
    fn kinetic_energy() {
        let mut b = RigidBody2D::new_dynamic(1, 2.0, 3.0);
        b.linear_velocity = Vec2::new(4.0, 3.0); // speed = 5, KE_lin = 0.5*2*25 = 25
        b.angular_velocity = 2.0; // KE_rot = 0.5*3*4 = 6
        assert!(approx(b.kinetic_energy(), 31.0));
    }

    #[test]
    fn velocity_at_point() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.linear_velocity = Vec2::new(1.0, 0.0);
        b.angular_velocity = 2.0;
        b.position = Vec2::zero();
        let v = b.velocity_at(Vec2::new(0.0, 1.0));
        // v = (1,0) + ω×r = (1,0) + (-2*1, 2*0) = (-1, 0)
        assert!(v2_approx(v, Vec2::new(-1.0, 0.0)));
    }

    #[test]
    fn local_world_roundtrip() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.position = Vec2::new(5.0, 3.0);
        b.rotation = 0.7;
        let local = Vec2::new(1.0, 2.0);
        let world = b.local_to_world(local);
        let back = b.world_to_local(world);
        assert!(v2_approx(back, local));
    }

    #[test]
    fn set_mass_updates_inverse() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.set_mass(4.0);
        assert!(approx(b.inv_mass, 0.25));
        b.set_mass(0.0);
        assert!(approx(b.inv_mass, 0.0));
    }

    #[test]
    fn set_inertia_updates_inverse() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.set_inertia(8.0);
        assert!(approx(b.inv_inertia, 0.125));
    }

    #[test]
    fn body_set_insert_and_get() {
        let mut set = BodySet::new();
        let id = set.insert_dynamic(1.0, 1.0);
        assert_eq!(set.len(), 1);
        let b = set.get(id).unwrap();
        assert_eq!(b.body_type, BodyType::Dynamic);
    }

    #[test]
    fn body_set_remove() {
        let mut set = BodySet::new();
        let id = set.insert_static();
        set.remove(id);
        assert!(set.is_empty());
        assert!(set.get(id).is_none());
    }

    #[test]
    fn body_set_integrate_all() {
        let mut set = BodySet::new();
        let id = set.insert_dynamic(1.0, 1.0);
        set.get_mut(id).unwrap().apply_force(Vec2::new(5.0, 0.0));
        set.integrate_all(1.0, Vec2::zero());
        let b = set.get(id).unwrap();
        assert!(approx(b.position.x, 5.0));
    }

    #[test]
    fn accumulators_cleared_after_step() {
        let mut b = RigidBody2D::new_dynamic(1, 1.0, 1.0);
        b.apply_force(Vec2::new(100.0, 0.0));
        b.apply_torque(50.0);
        b.integrate(0.01, Vec2::zero());
        assert!(v2_approx(b.force, Vec2::zero()));
        assert!(approx(b.torque, 0.0));
    }
}
