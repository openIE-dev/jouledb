//! 2D rigid body physics — position, velocity, acceleration, mass, moment of inertia,
//! force/torque application, Euler integration, angular dynamics, impulse resolution, friction.

use std::f64::consts::PI;

// ── Vec2 (local) ─────────────────────────────────────────────

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

// ── Material ─────────────────────────────────────────────────

/// Physical material properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Coefficient of restitution (bounciness) in [0, 1].
    pub restitution: f64,
    /// Static friction coefficient.
    pub static_friction: f64,
    /// Dynamic (kinetic) friction coefficient.
    pub dynamic_friction: f64,
}

impl Material {
    pub fn new(restitution: f64, static_friction: f64, dynamic_friction: f64) -> Self {
        Self { restitution, static_friction, dynamic_friction }
    }

    pub fn rubber() -> Self { Self::new(0.8, 0.9, 0.7) }
    pub fn ice() -> Self { Self::new(0.05, 0.05, 0.02) }
    pub fn steel() -> Self { Self::new(0.4, 0.6, 0.4) }
    pub fn wood() -> Self { Self::new(0.3, 0.5, 0.3) }
    pub fn bouncy() -> Self { Self::new(1.0, 0.0, 0.0) }
}

impl Default for Material {
    fn default() -> Self { Self::new(0.3, 0.5, 0.3) }
}

// ── RigidBody ────────────────────────────────────────────────

/// 2D rigid body with linear and angular dynamics.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    /// Position of center of mass.
    pub position: Vec2,
    /// Linear velocity.
    pub velocity: Vec2,
    /// Linear acceleration (accumulated).
    pub acceleration: Vec2,
    /// Rotation angle in radians.
    pub angle: f64,
    /// Angular velocity (rad/s).
    pub angular_velocity: f64,
    /// Angular acceleration (rad/s^2).
    pub angular_acceleration: f64,
    /// Mass (kg). 0 = infinite mass (static body).
    mass: f64,
    /// Inverse mass (cached). 0 for static bodies.
    inv_mass: f64,
    /// Moment of inertia.
    inertia: f64,
    /// Inverse moment of inertia. 0 for static bodies.
    inv_inertia: f64,
    /// Accumulated force for the current step.
    force: Vec2,
    /// Accumulated torque for the current step.
    torque: f64,
    /// Physical material.
    pub material: Material,
    /// Linear damping (0 = none, 1 = full stop each frame).
    pub linear_damping: f64,
    /// Angular damping.
    pub angular_damping: f64,
    /// Whether the body is static (immovable).
    pub is_static: bool,
}

impl RigidBody {
    /// Create a dynamic body with given mass and moment of inertia.
    pub fn new(mass: f64, inertia: f64) -> Self {
        let (m, im) = if mass <= 0.0 { (0.0, 0.0) } else { (mass, 1.0 / mass) };
        let (i, ii) = if inertia <= 0.0 { (0.0, 0.0) } else { (inertia, 1.0 / inertia) };
        Self {
            position: Vec2::zero(),
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            angle: 0.0,
            angular_velocity: 0.0,
            angular_acceleration: 0.0,
            mass: m,
            inv_mass: im,
            inertia: i,
            inv_inertia: ii,
            force: Vec2::zero(),
            torque: 0.0,
            material: Material::default(),
            linear_damping: 0.01,
            angular_damping: 0.01,
            is_static: false,
        }
    }

    /// Create a static (immovable) body.
    pub fn new_static() -> Self {
        let mut b = Self::new(0.0, 0.0);
        b.is_static = true;
        b
    }

    /// Create a body for a circle with given mass and radius.
    pub fn circle(mass: f64, radius: f64) -> Self {
        // I = 0.5 * m * r^2
        let inertia = 0.5 * mass * radius * radius;
        Self::new(mass, inertia)
    }

    /// Create a body for a rectangle with given mass, width, height.
    pub fn rectangle(mass: f64, width: f64, height: f64) -> Self {
        // I = (1/12) * m * (w^2 + h^2)
        let inertia = mass * (width * width + height * height) / 12.0;
        Self::new(mass, inertia)
    }

    pub fn mass(&self) -> f64 { self.mass }
    pub fn inv_mass(&self) -> f64 { self.inv_mass }
    pub fn inertia(&self) -> f64 { self.inertia }
    pub fn inv_inertia(&self) -> f64 { self.inv_inertia }

    pub fn set_mass(&mut self, mass: f64) {
        if mass <= 0.0 {
            self.mass = 0.0;
            self.inv_mass = 0.0;
        } else {
            self.mass = mass;
            self.inv_mass = 1.0 / mass;
        }
    }

    pub fn set_inertia(&mut self, inertia: f64) {
        if inertia <= 0.0 {
            self.inertia = 0.0;
            self.inv_inertia = 0.0;
        } else {
            self.inertia = inertia;
            self.inv_inertia = 1.0 / inertia;
        }
    }

    /// Set position.
    pub fn set_position(&mut self, x: f64, y: f64) {
        self.position = Vec2::new(x, y);
    }

    /// Set velocity.
    pub fn set_velocity(&mut self, vx: f64, vy: f64) {
        self.velocity = Vec2::new(vx, vy);
    }

    /// Apply a force at center of mass (accumulated until next step).
    pub fn apply_force(&mut self, force: Vec2) {
        if self.is_static { return; }
        self.force = self.force.add(force);
    }

    /// Apply a force at a world-space point, generating both force and torque.
    pub fn apply_force_at_point(&mut self, force: Vec2, point: Vec2) {
        if self.is_static { return; }
        self.force = self.force.add(force);
        let r = point.sub(self.position);
        self.torque += r.cross(force);
    }

    /// Apply a torque (accumulated until next step).
    pub fn apply_torque(&mut self, torque: f64) {
        if self.is_static { return; }
        self.torque += torque;
    }

    /// Apply an instantaneous impulse at center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        if self.is_static { return; }
        self.velocity = self.velocity.add(impulse.scale(self.inv_mass));
    }

    /// Apply an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec2, point: Vec2) {
        if self.is_static { return; }
        self.velocity = self.velocity.add(impulse.scale(self.inv_mass));
        let r = point.sub(self.position);
        self.angular_velocity += r.cross(impulse) * self.inv_inertia;
    }

    /// Semi-implicit Euler integration step.
    pub fn integrate(&mut self, dt: f64) {
        if self.is_static { return; }

        // Linear: F = ma => a = F/m
        let linear_accel = self.force.scale(self.inv_mass).add(self.acceleration);
        self.velocity = self.velocity.add(linear_accel.scale(dt));

        // Apply linear damping
        self.velocity = self.velocity.scale(1.0 - self.linear_damping);

        self.position = self.position.add(self.velocity.scale(dt));

        // Angular: T = I*alpha => alpha = T/I
        let angular_accel = self.torque * self.inv_inertia + self.angular_acceleration;
        self.angular_velocity += angular_accel * dt;

        // Apply angular damping
        self.angular_velocity *= 1.0 - self.angular_damping;

        self.angle += self.angular_velocity * dt;
        // Normalize angle to [-PI, PI]
        while self.angle > PI { self.angle -= 2.0 * PI; }
        while self.angle < -PI { self.angle += 2.0 * PI; }

        // Clear accumulated forces
        self.force = Vec2::zero();
        self.torque = 0.0;
    }

    /// Kinetic energy (linear + rotational).
    pub fn kinetic_energy(&self) -> f64 {
        let linear = 0.5 * self.mass * self.velocity.length_sq();
        let angular = 0.5 * self.inertia * self.angular_velocity * self.angular_velocity;
        linear + angular
    }

    /// Momentum vector.
    pub fn momentum(&self) -> Vec2 {
        self.velocity.scale(self.mass)
    }

    /// Velocity at a world-space point on the body.
    pub fn velocity_at_point(&self, point: Vec2) -> Vec2 {
        let r = point.sub(self.position);
        // v_point = v_cm + omega x r (in 2D: omega x r = (-omega*ry, omega*rx))
        let angular_contrib = Vec2::new(-self.angular_velocity * r.y, self.angular_velocity * r.x);
        self.velocity.add(angular_contrib)
    }

    /// Transform a local-space point to world space.
    pub fn local_to_world(&self, local: Vec2) -> Vec2 {
        self.position.add(local.rotate(self.angle))
    }

    /// Transform a world-space point to local space.
    pub fn world_to_local(&self, world: Vec2) -> Vec2 {
        world.sub(self.position).rotate(-self.angle)
    }
}

// ── Collision Resolution ─────────────────────────────────────

/// Resolve a collision between two rigid bodies.
///
/// `normal`: collision normal (from a toward b).
/// `penetration`: overlap depth.
/// `contact`: contact point in world space.
pub fn resolve_collision(
    a: &mut RigidBody,
    b: &mut RigidBody,
    normal: Vec2,
    penetration: f64,
    contact: Vec2,
) {
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum < 1e-12 { return; }

    // Positional correction (push apart)
    let correction_pct = 0.4;
    let slop = 0.01;
    let correction_mag = ((penetration - slop).max(0.0) / inv_mass_sum) * correction_pct;
    let correction = normal.scale(correction_mag);
    a.position = a.position.sub(correction.scale(a.inv_mass));
    b.position = b.position.add(correction.scale(b.inv_mass));

    // Relative velocity at contact point
    let ra = contact.sub(a.position);
    let rb = contact.sub(b.position);
    let va = a.velocity.add(Vec2::new(-a.angular_velocity * ra.y, a.angular_velocity * ra.x));
    let vb = b.velocity.add(Vec2::new(-b.angular_velocity * rb.y, b.angular_velocity * rb.x));
    let relative_vel = vb.sub(va);

    let vel_along_normal = relative_vel.dot(normal);
    // Don't resolve if separating
    if vel_along_normal > 0.0 { return; }

    // Restitution (use minimum)
    let e = a.material.restitution.min(b.material.restitution);

    let ra_cross_n = ra.cross(normal);
    let rb_cross_n = rb.cross(normal);
    let denom = inv_mass_sum
        + ra_cross_n * ra_cross_n * a.inv_inertia
        + rb_cross_n * rb_cross_n * b.inv_inertia;

    let j = -(1.0 + e) * vel_along_normal / denom;
    let impulse = normal.scale(j);

    a.velocity = a.velocity.sub(impulse.scale(a.inv_mass));
    b.velocity = b.velocity.add(impulse.scale(b.inv_mass));
    a.angular_velocity -= ra.cross(impulse) * a.inv_inertia;
    b.angular_velocity += rb.cross(impulse) * b.inv_inertia;

    // Friction impulse
    let tangent_vel = relative_vel.sub(normal.scale(vel_along_normal));
    let tangent_len = tangent_vel.length();
    if tangent_len < 1e-12 { return; }
    let tangent = tangent_vel.scale(-1.0 / tangent_len);

    let ra_cross_t = ra.cross(tangent);
    let rb_cross_t = rb.cross(tangent);
    let friction_denom = inv_mass_sum
        + ra_cross_t * ra_cross_t * a.inv_inertia
        + rb_cross_t * rb_cross_t * b.inv_inertia;

    let jt = -relative_vel.dot(tangent) / friction_denom;

    // Coulomb friction: clamp tangential impulse
    let mu_s = (a.material.static_friction * a.material.static_friction
        + b.material.static_friction * b.material.static_friction).sqrt();
    let mu_d = (a.material.dynamic_friction * a.material.dynamic_friction
        + b.material.dynamic_friction * b.material.dynamic_friction).sqrt();

    let friction_impulse = if jt.abs() < j * mu_s {
        tangent.scale(jt)
    } else {
        tangent.scale(-j * mu_d)
    };

    a.velocity = a.velocity.sub(friction_impulse.scale(a.inv_mass));
    b.velocity = b.velocity.add(friction_impulse.scale(b.inv_mass));
    a.angular_velocity -= ra.cross(friction_impulse) * a.inv_inertia;
    b.angular_velocity += rb.cross(friction_impulse) * b.inv_inertia;
}

// ── PhysicsWorld ─────────────────────────────────────────────

/// Simple physics world that holds bodies and steps the simulation.
#[derive(Debug, Clone)]
pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    /// Global gravity (applied each step).
    pub gravity: Vec2,
}

impl PhysicsWorld {
    pub fn new(gravity: Vec2) -> Self {
        Self { bodies: Vec::new(), gravity }
    }

    /// Add a body, returns its index.
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Step the simulation by dt seconds (integration only, no collision).
    pub fn step(&mut self, dt: f64) {
        let g = self.gravity;
        for body in &mut self.bodies {
            if !body.is_static {
                // Apply gravity as a force: F = m * g
                let gf = g.scale(body.mass);
                body.force = body.force.add(gf);
            }
            body.integrate(dt);
        }
    }

    /// Total kinetic energy of all bodies.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(|b| b.kinetic_energy()).sum()
    }

    /// Total momentum of all bodies.
    pub fn total_momentum(&self) -> Vec2 {
        self.bodies.iter().fold(Vec2::zero(), |acc, b| acc.add(b.momentum()))
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 0.05 }

    #[test]
    fn create_dynamic_body() {
        let b = RigidBody::new(10.0, 5.0);
        assert!(approx(b.mass(), 10.0));
        assert!(approx(b.inv_mass(), 0.1));
        assert!(approx(b.inertia(), 5.0));
        assert!(!b.is_static);
    }

    #[test]
    fn create_static_body() {
        let b = RigidBody::new_static();
        assert!(approx(b.mass(), 0.0));
        assert!(approx(b.inv_mass(), 0.0));
        assert!(b.is_static);
    }

    #[test]
    fn circle_body_inertia() {
        let b = RigidBody::circle(4.0, 2.0);
        // I = 0.5 * 4 * 4 = 8
        assert!(approx(b.inertia(), 8.0));
    }

    #[test]
    fn rectangle_body_inertia() {
        let b = RigidBody::rectangle(12.0, 3.0, 4.0);
        // I = 12 * (9+16) / 12 = 25
        assert!(approx(b.inertia(), 25.0));
    }

    #[test]
    fn integration_no_force() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.set_velocity(10.0, 0.0);
        b.linear_damping = 0.0;
        b.integrate(1.0);
        // After 1s at 10 m/s, should be at x=10
        assert!(approx(b.position.x, 10.0));
    }

    #[test]
    fn integration_with_force() {
        let mut b = RigidBody::new(2.0, 1.0);
        b.linear_damping = 0.0;
        b.apply_force(Vec2::new(10.0, 0.0));
        b.integrate(1.0);
        // a = F/m = 5, v = 5, x = 5
        assert!(approx(b.velocity.x, 5.0));
        assert!(approx(b.position.x, 5.0));
    }

    #[test]
    fn integration_with_gravity() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.linear_damping = 0.0;
        b.acceleration = Vec2::new(0.0, -9.81);
        b.integrate(1.0);
        assert!(approx(b.velocity.y, -9.81));
    }

    #[test]
    fn apply_impulse_changes_velocity() {
        let mut b = RigidBody::new(2.0, 1.0);
        b.apply_impulse(Vec2::new(10.0, 0.0));
        // dv = impulse / mass = 5
        assert!(approx(b.velocity.x, 5.0));
    }

    #[test]
    fn apply_torque_changes_angular() {
        let mut b = RigidBody::new(1.0, 2.0);
        b.angular_damping = 0.0;
        b.apply_torque(4.0);
        b.integrate(1.0);
        // alpha = T/I = 2, omega = 2
        assert!(approx(b.angular_velocity, 2.0));
    }

    #[test]
    fn force_at_point_generates_torque() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.angular_damping = 0.0;
        b.linear_damping = 0.0;
        b.set_position(0.0, 0.0);
        // Force at (1, 0) perpendicular: should generate torque
        b.apply_force_at_point(Vec2::new(0.0, 10.0), Vec2::new(1.0, 0.0));
        b.integrate(1.0);
        // torque = r x F = (1,0) x (0,10) = 10
        assert!(approx(b.angular_velocity, 10.0));
    }

    #[test]
    fn static_body_ignores_forces() {
        let mut b = RigidBody::new_static();
        b.apply_force(Vec2::new(100.0, 0.0));
        b.apply_impulse(Vec2::new(100.0, 0.0));
        b.integrate(1.0);
        assert!(approx(b.velocity.x, 0.0));
        assert!(approx(b.position.x, 0.0));
    }

    #[test]
    fn kinetic_energy() {
        let mut b = RigidBody::new(2.0, 1.0);
        b.set_velocity(3.0, 4.0);
        // KE = 0.5 * 2 * 25 = 25
        assert!(approx(b.kinetic_energy(), 25.0));
    }

    #[test]
    fn momentum() {
        let mut b = RigidBody::new(3.0, 1.0);
        b.set_velocity(4.0, 0.0);
        let p = b.momentum();
        assert!(approx(p.x, 12.0));
    }

    #[test]
    fn resolve_head_on_elastic() {
        let mut a = RigidBody::new(1.0, 1.0);
        a.set_velocity(5.0, 0.0);
        a.material = Material::bouncy();
        a.linear_damping = 0.0;

        let mut b = RigidBody::new(1.0, 1.0);
        b.set_position(10.0, 0.0);
        b.material = Material::bouncy();
        b.linear_damping = 0.0;

        let normal = Vec2::new(1.0, 0.0);
        let contact = Vec2::new(5.0, 0.0);
        resolve_collision(&mut a, &mut b, normal, 0.1, contact);

        // Perfectly elastic equal-mass: a stops, b gets a's velocity
        assert!(approx(a.velocity.x, 0.0));
        assert!(approx(b.velocity.x, 5.0));
    }

    #[test]
    fn resolve_static_collision() {
        let mut ball = RigidBody::new(1.0, 1.0);
        ball.set_velocity(0.0, -10.0);
        ball.material = Material::bouncy();
        ball.linear_damping = 0.0;

        let mut wall = RigidBody::new_static();
        wall.set_position(0.0, -5.0);
        wall.material = Material::bouncy();

        // Normal points from a (ball) toward b (wall), i.e. downward
        let normal = Vec2::new(0.0, -1.0);
        let contact = Vec2::new(0.0, -5.0);
        resolve_collision(&mut ball, &mut wall, normal, 0.01, contact);

        // Ball should bounce back upward
        assert!(ball.velocity.y > 0.0);
        // Wall shouldn't move
        assert!(approx(wall.velocity.y, 0.0));
    }

    #[test]
    fn physics_world_gravity() {
        let mut world = PhysicsWorld::new(Vec2::new(0.0, -10.0));
        let mut b = RigidBody::new(1.0, 1.0);
        b.linear_damping = 0.0;
        world.add_body(b);
        world.step(1.0);
        assert!(world.bodies[0].velocity.y < 0.0);
    }

    #[test]
    fn local_to_world_transform() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.set_position(5.0, 5.0);
        b.angle = std::f64::consts::FRAC_PI_2;
        let world = b.local_to_world(Vec2::new(1.0, 0.0));
        // Rotated 90 degrees: (1,0) -> (0,1), then translate
        assert!(approx(world.x, 5.0));
        assert!(approx(world.y, 6.0));
    }

    #[test]
    fn world_to_local_transform() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.set_position(5.0, 5.0);
        let local = b.world_to_local(Vec2::new(7.0, 5.0));
        assert!(approx(local.x, 2.0));
        assert!(approx(local.y, 0.0));
    }

    #[test]
    fn velocity_at_point() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.set_velocity(5.0, 0.0);
        b.angular_velocity = 2.0;
        // Point 1 unit above center
        let v = b.velocity_at_point(Vec2::new(0.0, 1.0));
        // v = (5, 0) + (-2*1, 2*0) = (3, 0)
        assert!(approx(v.x, 3.0));
        assert!(approx(v.y, 0.0));
    }

    #[test]
    fn material_presets() {
        let r = Material::rubber();
        assert!(r.restitution > 0.5);
        let i = Material::ice();
        assert!(i.static_friction < 0.1);
    }

    #[test]
    fn set_mass_and_inertia() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.set_mass(5.0);
        assert!(approx(b.mass(), 5.0));
        assert!(approx(b.inv_mass(), 0.2));
        b.set_inertia(10.0);
        assert!(approx(b.inertia(), 10.0));
    }

    #[test]
    fn world_total_energy() {
        let mut world = PhysicsWorld::new(Vec2::zero());
        let mut b1 = RigidBody::new(1.0, 1.0);
        b1.set_velocity(3.0, 4.0); // KE = 0.5 * 25 = 12.5
        world.add_body(b1);
        let mut b2 = RigidBody::new(2.0, 1.0);
        b2.set_velocity(1.0, 0.0); // KE = 0.5 * 2 * 1 = 1
        world.add_body(b2);
        assert!(approx(world.total_kinetic_energy(), 13.5));
    }

    #[test]
    fn angle_normalization() {
        let mut b = RigidBody::new(1.0, 1.0);
        b.angular_damping = 0.0;
        b.angle = 3.0 * PI;
        b.integrate(0.0);
        assert!(b.angle >= -PI && b.angle <= PI);
    }
}
