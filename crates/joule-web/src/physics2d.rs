//! 2D physics engine — rigid body dynamics, collision resolution, constraints, and sleep.
//!
//! Pure Rust replacement for matter.js, p2.js, planck.js physics engines.
//! Semi-implicit Euler integration, impulse-based collision response,
//! distance and pin joints, and body sleep for resting objects.

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }

    pub fn perpendicular(self) -> Self {
        Self { x: -self.y, y: self.x }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }

    pub fn negate(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Shape ────────────────────────────────────────────────────

/// 2D collision shape for physics bodies.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Circle { radius: f64 },
    AABB { half_width: f64, half_height: f64 },
}

impl Shape {
    pub fn circle(radius: f64) -> Self {
        Self::Circle { radius }
    }

    pub fn aabb(half_width: f64, half_height: f64) -> Self {
        Self::AABB { half_width, half_height }
    }

    /// Compute the moment of inertia for a given mass.
    pub fn moment_of_inertia(&self, mass: f64) -> f64 {
        match self {
            Shape::Circle { radius } => 0.5 * mass * radius * radius,
            Shape::AABB { half_width, half_height } => {
                let w = half_width * 2.0;
                let h = half_height * 2.0;
                mass * (w * w + h * h) / 12.0
            }
        }
    }
}

// ── BodyType ─────────────────────────────────────────────────

/// Type of physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Affected by forces and collisions.
    Dynamic,
    /// Not affected by forces, immovable.
    Static,
    /// Moved by user code, not by forces, but does affect dynamic bodies.
    Kinematic,
}

// ── Body2D ───────────────────────────────────────────────────

/// A 2D rigid body.
#[derive(Debug, Clone)]
pub struct Body2D {
    pub id: u64,
    pub body_type: BodyType,
    pub position: Vec2,
    pub velocity: Vec2,
    pub rotation: f64,
    pub angular_velocity: f64,
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia: f64,
    pub inv_inertia: f64,
    pub restitution: f64,
    pub friction: f64,
    pub shape: Shape,
    pub force: Vec2,
    pub torque: f64,

    // Sleep state
    pub sleeping: bool,
    sleep_timer: f64,
    velocity_threshold: f64,
    sleep_delay: f64,
}

impl Body2D {
    pub fn new(id: u64, shape: Shape, body_type: BodyType) -> Self {
        let mass = match body_type {
            BodyType::Dynamic => 1.0,
            BodyType::Static | BodyType::Kinematic => 0.0,
        };
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inertia = if mass > 0.0 { shape.moment_of_inertia(mass) } else { 0.0 };
        let inv_inertia = if inertia > 0.0 { 1.0 / inertia } else { 0.0 };

        Self {
            id,
            body_type,
            position: Vec2::zero(),
            velocity: Vec2::zero(),
            rotation: 0.0,
            angular_velocity: 0.0,
            mass,
            inv_mass,
            inertia,
            inv_inertia,
            restitution: 0.3,
            friction: 0.5,
            shape,
            force: Vec2::zero(),
            torque: 0.0,
            sleeping: false,
            sleep_timer: 0.0,
            velocity_threshold: 0.5,
            sleep_delay: 0.5,
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = Vec2::new(x, y);
        self
    }

    pub fn with_velocity(mut self, vx: f64, vy: f64) -> Self {
        self.velocity = Vec2::new(vx, vy);
        self
    }

    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self.inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.inertia = if mass > 0.0 { self.shape.moment_of_inertia(mass) } else { 0.0 };
        self.inv_inertia = if self.inertia > 0.0 { 1.0 / self.inertia } else { 0.0 };
        self
    }

    pub fn with_restitution(mut self, e: f64) -> Self {
        self.restitution = e;
        self
    }

    pub fn with_friction(mut self, f: f64) -> Self {
        self.friction = f;
        self
    }

    /// Apply a force at the center of mass.
    pub fn apply_force(&mut self, force: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.force = self.force.add(force);
        self.wake();
    }

    /// Apply an instantaneous impulse at the center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        if self.body_type != BodyType::Dynamic { return; }
        self.velocity = self.velocity.add(impulse.scale(self.inv_mass));
        self.wake();
    }

    /// Apply torque.
    pub fn apply_torque(&mut self, torque: f64) {
        if self.body_type != BodyType::Dynamic { return; }
        self.torque += torque;
        self.wake();
    }

    /// Wake the body from sleep.
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    /// Get kinetic energy of the body.
    pub fn kinetic_energy(&self) -> f64 {
        let linear = 0.5 * self.mass * self.velocity.length_sq();
        let angular = 0.5 * self.inertia * self.angular_velocity * self.angular_velocity;
        linear + angular
    }

    fn try_sleep(&mut self, dt: f64) {
        if self.body_type != BodyType::Dynamic { return; }
        let speed = self.velocity.length() + self.angular_velocity.abs();
        if speed < self.velocity_threshold {
            self.sleep_timer += dt;
            if self.sleep_timer >= self.sleep_delay {
                self.sleeping = true;
                self.velocity = Vec2::zero();
                self.angular_velocity = 0.0;
            }
        } else {
            self.sleep_timer = 0.0;
        }
    }
}

// ── Contact ──────────────────────────────────────────────────

/// Contact information for collision resolution.
#[derive(Debug, Clone)]
struct PhysicsContact {
    body_a: usize,
    body_b: usize,
    normal: Vec2,
    penetration: f64,
    point: Vec2,
}

// ── Constraint ───────────────────────────────────────────────

/// A constraint between two bodies.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Distance joint: maintains a fixed distance between two bodies.
    Distance {
        body_a: u64,
        body_b: u64,
        anchor_a: Vec2,
        anchor_b: Vec2,
        target_distance: f64,
        stiffness: f64,
        damping: f64,
    },
    /// Pin joint: pins a body to a world point.
    Pin {
        body: u64,
        anchor: Vec2,
        world_point: Vec2,
        stiffness: f64,
        damping: f64,
    },
}

impl Constraint {
    pub fn distance(body_a: u64, body_b: u64, distance: f64) -> Self {
        Self::Distance {
            body_a,
            body_b,
            anchor_a: Vec2::zero(),
            anchor_b: Vec2::zero(),
            target_distance: distance,
            stiffness: 1.0,
            damping: 0.1,
        }
    }

    pub fn pin(body: u64, world_point: Vec2) -> Self {
        Self::Pin {
            body,
            anchor: Vec2::zero(),
            world_point,
            stiffness: 1.0,
            damping: 0.1,
        }
    }
}

// ── World2D ──────────────────────────────────────────────────

/// 2D physics world.
#[derive(Debug, Clone)]
pub struct World2D {
    pub gravity: Vec2,
    pub bodies: Vec<Body2D>,
    pub constraints: Vec<Constraint>,
    next_id: u64,
    iterations: u32,
}

impl World2D {
    pub fn new(gravity_x: f64, gravity_y: f64) -> Self {
        Self {
            gravity: Vec2::new(gravity_x, gravity_y),
            bodies: Vec::new(),
            constraints: Vec::new(),
            next_id: 1,
            iterations: 4,
        }
    }

    pub fn with_iterations(mut self, n: u32) -> Self {
        self.iterations = n;
        self
    }

    /// Add a body and return its ID.
    pub fn add_body(&mut self, body: Body2D) -> u64 {
        let id = body.id;
        self.bodies.push(body);
        id
    }

    /// Create and add a dynamic body, returning its ID.
    pub fn create_body(&mut self, shape: Shape, x: f64, y: f64, mass: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let body = Body2D::new(id, shape, BodyType::Dynamic)
            .with_position(x, y)
            .with_mass(mass);
        self.add_body(body)
    }

    /// Create and add a static body, returning its ID.
    pub fn create_static(&mut self, shape: Shape, x: f64, y: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let body = Body2D::new(id, shape, BodyType::Static)
            .with_position(x, y);
        self.add_body(body)
    }

    /// Add a constraint.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Find body index by ID.
    fn body_index(&self, id: u64) -> Option<usize> {
        self.bodies.iter().position(|b| b.id == id)
    }

    /// Get a body by ID.
    pub fn get_body(&self, id: u64) -> Option<&Body2D> {
        self.bodies.iter().find(|b| b.id == id)
    }

    /// Get a mutable body by ID.
    pub fn get_body_mut(&mut self, id: u64) -> Option<&mut Body2D> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    /// Step the physics simulation forward by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        // 1. Apply gravity and integrate forces (semi-implicit Euler)
        for body in &mut self.bodies {
            if body.body_type != BodyType::Dynamic || body.sleeping {
                continue;
            }

            // Apply gravity
            let gravity_force = self.gravity.scale(body.mass);
            body.force = body.force.add(gravity_force);

            // Semi-implicit Euler: update velocity first, then position
            body.velocity = body.velocity.add(body.force.scale(body.inv_mass * dt));
            body.angular_velocity += body.torque * body.inv_inertia * dt;

            body.position = body.position.add(body.velocity.scale(dt));
            body.rotation += body.angular_velocity * dt;

            // Clear forces
            body.force = Vec2::zero();
            body.torque = 0.0;
        }

        // 2. Detect collisions
        let contacts = self.detect_collisions();

        // 3. Resolve collisions (iterate for stability)
        for _ in 0..self.iterations {
            for contact in &contacts {
                self.resolve_collision(contact);
            }
        }

        // 4. Positional correction (prevent sinking)
        for contact in &contacts {
            self.positional_correction(contact);
        }

        // 5. Solve constraints
        let constraints = self.constraints.clone();
        for constraint in &constraints {
            self.solve_constraint(constraint, dt);
        }

        // 6. Sleep check
        for body in &mut self.bodies {
            body.try_sleep(dt);
        }
    }

    fn detect_collisions(&self) -> Vec<PhysicsContact> {
        let mut contacts = Vec::new();
        let n = self.bodies.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.bodies[i];
                let b = &self.bodies[j];

                // Skip if both static or both sleeping
                if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
                    continue;
                }
                if a.sleeping && b.sleeping {
                    continue;
                }

                if let Some(contact) = self.test_pair(a, b, i, j) {
                    contacts.push(contact);
                }
            }
        }

        contacts
    }

    fn test_pair(&self, a: &Body2D, b: &Body2D, idx_a: usize, idx_b: usize) -> Option<PhysicsContact> {
        match (&a.shape, &b.shape) {
            (Shape::Circle { radius: ra }, Shape::Circle { radius: rb }) => {
                let diff = b.position.sub(a.position);
                let dist = diff.length();
                let sum_r = ra + rb;

                if dist >= sum_r {
                    return None;
                }

                let normal = if dist < 1e-10 { Vec2::new(1.0, 0.0) } else { diff.normalized() };
                let penetration = sum_r - dist;
                let point = a.position.add(normal.scale(*ra));

                Some(PhysicsContact {
                    body_a: idx_a,
                    body_b: idx_b,
                    normal,
                    penetration,
                    point,
                })
            }
            (Shape::AABB { half_width: hw_a, half_height: hh_a },
             Shape::AABB { half_width: hw_b, half_height: hh_b }) => {
                let diff = b.position.sub(a.position);
                let overlap_x = hw_a + hw_b - diff.x.abs();
                let overlap_y = hh_a + hh_b - diff.y.abs();

                if overlap_x <= 0.0 || overlap_y <= 0.0 {
                    return None;
                }

                let (normal, penetration) = if overlap_x < overlap_y {
                    (Vec2::new(if diff.x > 0.0 { 1.0 } else { -1.0 }, 0.0), overlap_x)
                } else {
                    (Vec2::new(0.0, if diff.y > 0.0 { 1.0 } else { -1.0 }), overlap_y)
                };

                let point = a.position.add(b.position).scale(0.5);

                Some(PhysicsContact {
                    body_a: idx_a,
                    body_b: idx_b,
                    normal,
                    penetration,
                    point,
                })
            }
            (Shape::Circle { radius }, Shape::AABB { half_width, half_height }) => {
                let diff = a.position.sub(b.position);
                let closest = Vec2::new(
                    diff.x.clamp(-half_width, *half_width),
                    diff.y.clamp(-half_height, *half_height),
                );
                let delta = diff.sub(closest);
                let dist = delta.length();

                if dist >= *radius {
                    return None;
                }

                let normal = if dist < 1e-10 { Vec2::new(1.0, 0.0) } else { delta.normalized() };

                Some(PhysicsContact {
                    body_a: idx_a,
                    body_b: idx_b,
                    normal,
                    penetration: radius - dist,
                    point: a.position.sub(normal.scale(*radius)),
                })
            }
            (Shape::AABB { .. }, Shape::Circle { .. }) => {
                self.test_pair(b, a, idx_b, idx_a).map(|c| PhysicsContact {
                    body_a: idx_a,
                    body_b: idx_b,
                    normal: c.normal.negate(),
                    ..c
                })
            }
        }
    }

    fn resolve_collision(&mut self, contact: &PhysicsContact) {
        let (inv_mass_a, inv_mass_b, vel_a, vel_b, restitution) = {
            let a = &self.bodies[contact.body_a];
            let b = &self.bodies[contact.body_b];
            let e = a.restitution.min(b.restitution);
            (a.inv_mass, b.inv_mass, a.velocity, b.velocity, e)
        };

        let inv_mass_sum = inv_mass_a + inv_mass_b;
        if inv_mass_sum < 1e-10 {
            return; // Both static
        }

        let relative_vel = vel_b.sub(vel_a);
        let vel_along_normal = relative_vel.dot(contact.normal);

        // Don't resolve if separating
        if vel_along_normal > 0.0 {
            return;
        }

        let j = -(1.0 + restitution) * vel_along_normal / inv_mass_sum;
        let impulse = contact.normal.scale(j);

        // Apply impulses
        {
            let a = &mut self.bodies[contact.body_a];
            if a.body_type == BodyType::Dynamic {
                a.velocity = a.velocity.sub(impulse.scale(a.inv_mass));
                a.wake();
            }
        }
        {
            let b = &mut self.bodies[contact.body_b];
            if b.body_type == BodyType::Dynamic {
                b.velocity = b.velocity.add(impulse.scale(b.inv_mass));
                b.wake();
            }
        }

        // Friction impulse
        let (vel_a2, vel_b2, friction) = {
            let a = &self.bodies[contact.body_a];
            let b = &self.bodies[contact.body_b];
            (a.velocity, b.velocity, (a.friction * b.friction).sqrt())
        };

        let relative_vel2 = vel_b2.sub(vel_a2);
        let tangent_vel = relative_vel2.sub(contact.normal.scale(relative_vel2.dot(contact.normal)));
        let tangent_len = tangent_vel.length();

        if tangent_len > 1e-10 {
            let tangent = tangent_vel.normalized();
            let jt = -tangent_vel.dot(tangent) / inv_mass_sum;
            let jt_clamped = jt.clamp(-j * friction, j * friction);
            let friction_impulse = tangent.scale(jt_clamped);

            {
                let a = &mut self.bodies[contact.body_a];
                if a.body_type == BodyType::Dynamic {
                    a.velocity = a.velocity.sub(friction_impulse.scale(a.inv_mass));
                }
            }
            {
                let b = &mut self.bodies[contact.body_b];
                if b.body_type == BodyType::Dynamic {
                    b.velocity = b.velocity.add(friction_impulse.scale(b.inv_mass));
                }
            }
        }
    }

    fn positional_correction(&mut self, contact: &PhysicsContact) {
        let slop = 0.01;
        let percent = 0.4;

        let inv_mass_a = self.bodies[contact.body_a].inv_mass;
        let inv_mass_b = self.bodies[contact.body_b].inv_mass;
        let inv_mass_sum = inv_mass_a + inv_mass_b;
        if inv_mass_sum < 1e-10 {
            return;
        }

        let correction = contact.normal.scale(
            ((contact.penetration - slop).max(0.0) / inv_mass_sum) * percent
        );

        if self.bodies[contact.body_a].body_type == BodyType::Dynamic {
            let corr_a = correction.scale(inv_mass_a);
            self.bodies[contact.body_a].position = self.bodies[contact.body_a].position.sub(corr_a);
        }
        if self.bodies[contact.body_b].body_type == BodyType::Dynamic {
            let corr_b = correction.scale(inv_mass_b);
            self.bodies[contact.body_b].position = self.bodies[contact.body_b].position.add(corr_b);
        }
    }

    fn solve_constraint(&mut self, constraint: &Constraint, dt: f64) {
        match constraint {
            Constraint::Distance { body_a, body_b, anchor_a, anchor_b, target_distance, stiffness, damping } => {
                let (idx_a, idx_b) = match (self.body_index(*body_a), self.body_index(*body_b)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return,
                };

                let pos_a = self.bodies[idx_a].position.add(*anchor_a);
                let pos_b = self.bodies[idx_b].position.add(*anchor_b);
                let diff = pos_b.sub(pos_a);
                let dist = diff.length();

                if dist < 1e-10 {
                    return;
                }

                let normal = diff.normalized();
                let error = dist - target_distance;

                let inv_mass_a = self.bodies[idx_a].inv_mass;
                let inv_mass_b = self.bodies[idx_b].inv_mass;
                let inv_sum = inv_mass_a + inv_mass_b;
                if inv_sum < 1e-10 {
                    return;
                }

                // Spring force
                let spring_force = error * stiffness;

                // Damping force
                let rel_vel = self.bodies[idx_b].velocity.sub(self.bodies[idx_a].velocity);
                let damp_force = rel_vel.dot(normal) * damping;

                let force_mag = spring_force + damp_force;
                let force = normal.scale(force_mag * dt);

                if self.bodies[idx_a].body_type == BodyType::Dynamic {
                    self.bodies[idx_a].velocity = self.bodies[idx_a].velocity.add(force.scale(inv_mass_a));
                }
                if self.bodies[idx_b].body_type == BodyType::Dynamic {
                    self.bodies[idx_b].velocity = self.bodies[idx_b].velocity.sub(force.scale(inv_mass_b));
                }
            }
            Constraint::Pin { body, anchor, world_point, stiffness, damping } => {
                let idx = match self.body_index(*body) {
                    Some(i) => i,
                    None => return,
                };

                let body_ref = &self.bodies[idx];
                if body_ref.body_type != BodyType::Dynamic {
                    return;
                }

                let pos = body_ref.position.add(*anchor);
                let diff = world_point.sub(pos);
                let spring_force = diff.scale(*stiffness);
                let damp_force = body_ref.velocity.scale(-damping);

                let total = spring_force.add(damp_force).scale(dt);
                self.bodies[idx].velocity = self.bodies[idx].velocity.add(total.scale(self.bodies[idx].inv_mass));
            }
        }
    }

    /// Get total kinetic energy of all bodies.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(|b| b.kinetic_energy()).sum()
    }

    /// Number of sleeping bodies.
    pub fn sleeping_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.sleeping).count()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn body_creation() {
        let body = Body2D::new(1, Shape::circle(1.0), BodyType::Dynamic)
            .with_position(10.0, 20.0)
            .with_mass(5.0);
        assert_eq!(body.position.x, 10.0);
        assert_eq!(body.mass, 5.0);
        assert!(approx(body.inv_mass, 0.2, 0.001));
    }

    #[test]
    fn static_body_has_zero_inv_mass() {
        let body = Body2D::new(1, Shape::circle(1.0), BodyType::Static);
        assert_eq!(body.inv_mass, 0.0);
        assert_eq!(body.inv_inertia, 0.0);
    }

    #[test]
    fn apply_impulse() {
        let mut body = Body2D::new(1, Shape::circle(1.0), BodyType::Dynamic)
            .with_mass(2.0);
        body.apply_impulse(Vec2::new(10.0, 0.0));
        assert!(approx(body.velocity.x, 5.0, 0.001)); // 10 / 2
    }

    #[test]
    fn apply_impulse_static_ignored() {
        let mut body = Body2D::new(1, Shape::circle(1.0), BodyType::Static);
        body.apply_impulse(Vec2::new(10.0, 0.0));
        assert_eq!(body.velocity.x, 0.0);
    }

    #[test]
    fn gravity_integration() {
        let mut world = World2D::new(0.0, -9.81);
        let id = world.create_body(Shape::circle(1.0), 0.0, 10.0, 1.0);

        world.step(1.0 / 60.0);

        let body = world.get_body(id).unwrap();
        assert!(body.velocity.y < 0.0); // Falling down
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn circle_circle_collision_response() {
        let mut world = World2D::new(0.0, 0.0);
        let a = world.create_body(Shape::circle(5.0), 0.0, 0.0, 1.0);
        let b = world.create_body(Shape::circle(5.0), 8.0, 0.0, 1.0);

        // Give them approaching velocities
        world.get_body_mut(a).unwrap().velocity = Vec2::new(5.0, 0.0);
        world.get_body_mut(b).unwrap().velocity = Vec2::new(-5.0, 0.0);

        world.step(1.0 / 60.0);

        // After collision, they should be moving apart
        let va = world.get_body(a).unwrap().velocity.x;
        let vb = world.get_body(b).unwrap().velocity.x;
        assert!(va < 0.0, "body a should bounce back, got {va}");
        assert!(vb > 0.0, "body b should bounce back, got {vb}");
    }

    #[test]
    fn aabb_aabb_collision() {
        let mut world = World2D::new(0.0, 0.0);
        let a = world.create_body(Shape::aabb(5.0, 5.0), 0.0, 0.0, 1.0);
        let b = world.create_body(Shape::aabb(5.0, 5.0), 8.0, 0.0, 1.0);

        world.get_body_mut(a).unwrap().velocity = Vec2::new(5.0, 0.0);
        world.get_body_mut(b).unwrap().velocity = Vec2::new(-5.0, 0.0);

        world.step(1.0 / 60.0);

        let va = world.get_body(a).unwrap().velocity.x;
        let vb = world.get_body(b).unwrap().velocity.x;
        assert!(va < 5.0, "aabb collision should slow/reverse body a");
        assert!(vb > -5.0, "aabb collision should slow/reverse body b");
    }

    #[test]
    fn static_body_immovable() {
        let mut world = World2D::new(0.0, 0.0);
        let wall = world.create_static(Shape::aabb(100.0, 5.0), 0.0, -10.0);
        let ball = world.create_body(Shape::circle(3.0), 0.0, -3.0, 1.0);

        world.get_body_mut(ball).unwrap().velocity = Vec2::new(0.0, -10.0);

        for _ in 0..10 {
            world.step(1.0 / 60.0);
        }

        let wall_body = world.get_body(wall).unwrap();
        assert_eq!(wall_body.velocity.x, 0.0);
        assert_eq!(wall_body.velocity.y, 0.0);
    }

    #[test]
    fn kinetic_energy() {
        let body = Body2D::new(1, Shape::circle(1.0), BodyType::Dynamic)
            .with_mass(2.0)
            .with_velocity(3.0, 4.0);
        // KE = 0.5 * 2 * (9+16) = 25
        assert!(approx(body.kinetic_energy(), 25.0, 0.01));
    }

    #[test]
    fn sleep_resting_body() {
        let mut world = World2D::new(0.0, 0.0);
        let id = world.create_body(Shape::circle(1.0), 0.0, 0.0, 1.0);
        // Body with zero velocity should sleep after delay
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        assert!(world.get_body(id).unwrap().sleeping);
    }

    #[test]
    fn wake_sleeping_body() {
        let mut world = World2D::new(0.0, 0.0);
        let id = world.create_body(Shape::circle(1.0), 0.0, 0.0, 1.0);
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        assert!(world.get_body(id).unwrap().sleeping);

        world.get_body_mut(id).unwrap().apply_impulse(Vec2::new(10.0, 0.0));
        assert!(!world.get_body(id).unwrap().sleeping);
    }

    #[test]
    fn distance_constraint() {
        let mut world = World2D::new(0.0, 0.0);
        let a = world.create_body(Shape::circle(1.0), 0.0, 0.0, 1.0);
        let b = world.create_body(Shape::circle(1.0), 20.0, 0.0, 1.0);

        world.add_constraint(Constraint::distance(a, b, 10.0));

        // Run simulation — bodies should be pulled toward 10 unit distance
        for _ in 0..200 {
            world.step(1.0 / 60.0);
        }

        let pos_a = world.get_body(a).unwrap().position;
        let pos_b = world.get_body(b).unwrap().position;
        let dist = pos_b.sub(pos_a).length();
        // Should be close to target distance (within tolerance due to spring behavior)
        assert!(dist < 15.0, "distance should converge toward 10, got {dist}");
    }

    #[test]
    fn pin_constraint() {
        let mut world = World2D::new(0.0, 0.0);
        let id = world.create_body(Shape::circle(1.0), 10.0, 0.0, 1.0);

        world.add_constraint(Constraint::Pin {
            body: id,
            anchor: Vec2::zero(),
            world_point: Vec2::new(0.0, 0.0),
            stiffness: 50.0,
            damping: 5.0,
        });

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }

        let pos = world.get_body(id).unwrap().position;
        // Should be pulled toward (0,0)
        assert!(pos.length() < 5.0, "body should approach pin point, distance is {}", pos.length());
    }

    #[test]
    fn total_kinetic_energy() {
        let mut world = World2D::new(0.0, 0.0);
        let a = world.create_body(Shape::circle(1.0), 0.0, 0.0, 1.0);
        world.get_body_mut(a).unwrap().velocity = Vec2::new(3.0, 4.0);
        // KE = 0.5 * 1 * 25 = 12.5
        assert!(approx(world.total_kinetic_energy(), 12.5, 0.01));
    }

    #[test]
    fn sleeping_count() {
        let mut world = World2D::new(0.0, 0.0);
        world.create_body(Shape::circle(1.0), 0.0, 0.0, 1.0);
        world.create_body(Shape::circle(1.0), 100.0, 0.0, 1.0);

        assert_eq!(world.sleeping_count(), 0);

        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }

        assert_eq!(world.sleeping_count(), 2);
    }

    #[test]
    fn moment_of_inertia_circle() {
        let shape = Shape::circle(2.0);
        let moi = shape.moment_of_inertia(3.0);
        // I = 0.5 * m * r^2 = 0.5 * 3 * 4 = 6
        assert!(approx(moi, 6.0, 0.01));
    }

    #[test]
    fn moment_of_inertia_aabb() {
        let shape = Shape::aabb(3.0, 2.0);
        let moi = shape.moment_of_inertia(1.0);
        // I = m * (w^2 + h^2) / 12 = 1 * (36 + 16) / 12 = 52/12
        assert!(approx(moi, 52.0 / 12.0, 0.01));
    }

    #[test]
    fn apply_force_accumulates() {
        let mut body = Body2D::new(1, Shape::circle(1.0), BodyType::Dynamic).with_mass(1.0);
        body.apply_force(Vec2::new(5.0, 0.0));
        body.apply_force(Vec2::new(3.0, 0.0));
        assert!(approx(body.force.x, 8.0, 0.001));
    }
}
