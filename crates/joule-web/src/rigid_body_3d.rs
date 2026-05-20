//! 3D Rigid Body Dynamics — position, orientation, linear/angular velocity,
//! mass, inertia tensor, force/torque accumulators, semi-implicit Euler
//! integration with quaternion normalization. Supports Dynamic, Static,
//! and Kinematic body types with damping, gravity scale, and sleep.

use std::collections::HashMap;

// ── Vec3 ─────────────────────────────────────────────────────

/// 3-component vector for positions, velocities, forces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const RIGHT: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
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

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// ── Quaternion ───────────────────────────────────────────────

/// Unit quaternion for 3D orientation (w + xi + yj + zk).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Build from axis (must be unit) and angle in radians.
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        Self { w: half.cos(), x: axis.x * s, y: axis.y * s, z: axis.z * s }
    }

    pub fn length_sq(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Self {
        let len = self.length_sq().sqrt();
        if len < 1e-12 {
            Self::IDENTITY
        } else {
            let inv = 1.0 / len;
            Self { w: self.w * inv, x: self.x * inv, y: self.y * inv, z: self.z * inv }
        }
    }

    pub fn conjugate(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Rotate a vector by this quaternion: q * v * q^-1.
    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v + (uv * (2.0 * self.w)) + (uuv * 2.0)
    }

    /// Hamilton product.
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Convert to 3x3 rotation matrix.
    pub fn to_mat3(self) -> Mat3 {
        let (xx, yy, zz) = (self.x * self.x, self.y * self.y, self.z * self.z);
        let (xy, xz, yz) = (self.x * self.y, self.x * self.z, self.y * self.z);
        let (wx, wy, wz) = (self.w * self.x, self.w * self.y, self.w * self.z);
        Mat3 {
            m: [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
                [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
            ],
        }
    }
}

// ── Mat3 ─────────────────────────────────────────────────────

/// 3x3 matrix in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    pub fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// Diagonal matrix from 3 values (for principal inertia tensor).
    pub fn diagonal(a: f64, b: f64, c: f64) -> Self {
        Self { m: [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]] }
    }

    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            y: self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            z: self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        }
    }

    pub fn transpose(self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    /// M * N matrix multiply.
    pub fn mul_mat(self, rhs: Self) -> Self {
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = self.m[i][0] * rhs.m[0][j]
                         + self.m[i][1] * rhs.m[1][j]
                         + self.m[i][2] * rhs.m[2][j];
            }
        }
        Self { m: r }
    }

    /// Inverse of a 3x3 matrix. Returns ZERO if singular.
    pub fn inverse(self) -> Self {
        let det = self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
                - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
                + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]);
        if det.abs() < 1e-12 {
            return Self::ZERO;
        }
        let inv_det = 1.0 / det;
        Self {
            m: [
                [
                    (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) * inv_det,
                    (self.m[0][2] * self.m[2][1] - self.m[0][1] * self.m[2][2]) * inv_det,
                    (self.m[0][1] * self.m[1][2] - self.m[0][2] * self.m[1][1]) * inv_det,
                ],
                [
                    (self.m[1][2] * self.m[2][0] - self.m[1][0] * self.m[2][2]) * inv_det,
                    (self.m[0][0] * self.m[2][2] - self.m[0][2] * self.m[2][0]) * inv_det,
                    (self.m[0][2] * self.m[1][0] - self.m[0][0] * self.m[1][2]) * inv_det,
                ],
                [
                    (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]) * inv_det,
                    (self.m[0][1] * self.m[2][0] - self.m[0][0] * self.m[2][1]) * inv_det,
                    (self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]) * inv_det,
                ],
            ],
        }
    }

    /// Scale every element.
    pub fn scale(self, s: f64) -> Self {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] *= s;
            }
        }
        r
    }
}

// ── Body Type ────────────────────────────────────────────────

/// Rigid body type controls how physics affects the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Full physics simulation.
    Dynamic,
    /// Immovable — infinite mass, ignores forces.
    Static,
    /// Moved by user code — responds to velocity but not forces.
    Kinematic,
}

// ── Rigid Body ───────────────────────────────────────────────

/// Unique identifier for a rigid body.
pub type BodyId = u64;

/// Full 3D rigid body with position, orientation, velocities, mass properties,
/// damping, gravity scale, and sleep management.
#[derive(Debug, Clone)]
pub struct RigidBody3D {
    pub id: BodyId,
    pub body_type: BodyType,

    // Transform
    pub position: Vec3,
    pub orientation: Quaternion,

    // Velocities
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,

    // Mass properties
    pub mass: f64,
    pub inv_mass: f64,
    pub local_inertia: Mat3,
    pub inv_inertia_world: Mat3,

    // Accumulators
    pub force: Vec3,
    pub torque: Vec3,

    // Damping
    pub linear_damping: f64,
    pub angular_damping: f64,

    // Gravity
    pub gravity_scale: f64,

    // Sleep
    pub sleep_threshold: f64,
    pub sleeping: bool,
    pub sleep_timer: f64,
    pub sleep_delay: f64,
}

impl RigidBody3D {
    /// Create a new dynamic body at the given position with mass and inertia.
    pub fn new_dynamic(id: BodyId, position: Vec3, mass: f64, inertia: Mat3) -> Self {
        let inv_mass = if mass > 1e-12 { 1.0 / mass } else { 0.0 };
        Self {
            id,
            body_type: BodyType::Dynamic,
            position,
            orientation: Quaternion::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            local_inertia: inertia,
            inv_inertia_world: inertia.inverse(),
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            linear_damping: 0.01,
            angular_damping: 0.01,
            gravity_scale: 1.0,
            sleep_threshold: 0.01,
            sleeping: false,
            sleep_timer: 0.0,
            sleep_delay: 0.5,
        }
    }

    pub fn new_static(id: BodyId, position: Vec3) -> Self {
        Self {
            id,
            body_type: BodyType::Static,
            position,
            orientation: Quaternion::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            local_inertia: Mat3::ZERO,
            inv_inertia_world: Mat3::ZERO,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 0.0,
            sleep_threshold: 0.0,
            sleeping: false,
            sleep_timer: 0.0,
            sleep_delay: 0.0,
        }
    }

    pub fn new_kinematic(id: BodyId, position: Vec3) -> Self {
        let mut body = Self::new_static(id, position);
        body.body_type = BodyType::Kinematic;
        body
    }

    /// Apply a force at the center of mass (world frame).
    pub fn apply_force(&mut self, f: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += f;
        self.wake();
    }

    /// Apply a force at a world-space point (generates torque).
    pub fn apply_force_at_point(&mut self, f: Vec3, point: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += f;
        let r = point - self.position;
        self.torque += r.cross(f);
        self.wake();
    }

    /// Apply a torque (world frame).
    pub fn apply_torque(&mut self, t: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.torque += t;
        self.wake();
    }

    /// Apply an instantaneous impulse at the center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        self.wake();
    }

    /// Apply an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec3, point: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        let r = point - self.position;
        self.angular_velocity += self.inv_inertia_world.mul_vec(r.cross(impulse));
        self.wake();
    }

    /// Clear force and torque accumulators.
    pub fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }

    /// Update world-space inverse inertia from orientation.
    pub fn update_inertia_world(&mut self) {
        let rot = self.orientation.to_mat3();
        let rot_t = rot.transpose();
        self.inv_inertia_world = rot.mul_mat(self.local_inertia.inverse()).mul_mat(rot_t);
    }

    /// Kinetic energy = 0.5 * m * v^2 + 0.5 * omega^T * I * omega.
    pub fn kinetic_energy(&self) -> f64 {
        let lin = 0.5 * self.mass * self.linear_velocity.length_sq();
        let ang = 0.5 * self.angular_velocity.dot(
            self.local_inertia.mul_vec(self.angular_velocity),
        );
        lin + ang
    }

    /// Check if the body should be put to sleep.
    pub fn should_sleep(&self) -> bool {
        self.kinetic_energy() < self.sleep_threshold
    }

    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    /// Semi-implicit Euler integration step.
    pub fn integrate(&mut self, dt: f64, gravity: Vec3) {
        if self.body_type == BodyType::Static || self.sleeping {
            self.clear_forces();
            return;
        }

        if self.body_type == BodyType::Dynamic {
            // Linear: v += (F/m + g * gravScale) * dt
            let accel = self.force * self.inv_mass + gravity * self.gravity_scale;
            self.linear_velocity += accel * dt;
            self.linear_velocity = self.linear_velocity * (1.0 - self.linear_damping).max(0.0);

            // Angular: omega += I^-1 * (torque - omega x (I * omega)) * dt  (Euler equation)
            let i_omega = self.local_inertia.mul_vec(self.angular_velocity);
            let gyroscopic = self.angular_velocity.cross(i_omega);
            let angular_accel = self.inv_inertia_world.mul_vec(self.torque - gyroscopic);
            self.angular_velocity += angular_accel * dt;
            self.angular_velocity = self.angular_velocity * (1.0 - self.angular_damping).max(0.0);
        }

        // Position: x += v * dt
        self.position += self.linear_velocity * dt;

        // Orientation: q += 0.5 * omega_quat * q * dt
        let omega_q = Quaternion::new(
            0.0,
            self.angular_velocity.x,
            self.angular_velocity.y,
            self.angular_velocity.z,
        );
        let dq = omega_q.mul(self.orientation);
        self.orientation = Quaternion {
            w: self.orientation.w + dq.w * 0.5 * dt,
            x: self.orientation.x + dq.x * 0.5 * dt,
            y: self.orientation.y + dq.y * 0.5 * dt,
            z: self.orientation.z + dq.z * 0.5 * dt,
        }.normalized();

        self.update_inertia_world();
        self.clear_forces();

        // Sleep management
        if self.body_type == BodyType::Dynamic {
            if self.should_sleep() {
                self.sleep_timer += dt;
                if self.sleep_timer >= self.sleep_delay {
                    self.sleeping = true;
                    self.linear_velocity = Vec3::ZERO;
                    self.angular_velocity = Vec3::ZERO;
                }
            } else {
                self.sleep_timer = 0.0;
            }
        }
    }

    /// Get the velocity of a world-space point on this body.
    pub fn velocity_at_point(&self, world_point: Vec3) -> Vec3 {
        let r = world_point - self.position;
        self.linear_velocity + self.angular_velocity.cross(r)
    }

    /// Transform a local-space point to world space.
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        self.position + self.orientation.rotate_vec(local)
    }

    /// Transform a world-space point to local space.
    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        self.orientation.conjugate().rotate_vec(world - self.position)
    }
}

// ── Physics World ────────────────────────────────────────────

/// Container that manages a set of rigid bodies.
pub struct PhysicsWorld3D {
    bodies: HashMap<BodyId, RigidBody3D>,
    gravity: Vec3,
    next_id: BodyId,
}

impl PhysicsWorld3D {
    pub fn new(gravity: Vec3) -> Self {
        Self { bodies: HashMap::new(), gravity, next_id: 1 }
    }

    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn set_gravity(&mut self, g: Vec3) {
        self.gravity = g;
    }

    pub fn add_body(&mut self, mut body: RigidBody3D) -> BodyId {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;
        self.bodies.insert(id, body);
        id
    }

    pub fn remove_body(&mut self, id: BodyId) -> Option<RigidBody3D> {
        self.bodies.remove(&id)
    }

    pub fn get_body(&self, id: BodyId) -> Option<&RigidBody3D> {
        self.bodies.get(&id)
    }

    pub fn get_body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody3D> {
        self.bodies.get_mut(&id)
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Step the simulation forward.
    pub fn step(&mut self, dt: f64) {
        let g = self.gravity;
        for body in self.bodies.values_mut() {
            body.integrate(dt, g);
        }
    }

    /// Iterate over all bodies.
    pub fn bodies(&self) -> impl Iterator<Item = &RigidBody3D> {
        self.bodies.values()
    }

    /// Collect all body IDs.
    pub fn body_ids(&self) -> Vec<BodyId> {
        self.bodies.keys().copied().collect()
    }

    /// Wake all sleeping bodies.
    pub fn wake_all(&mut self) {
        for body in self.bodies.values_mut() {
            body.wake();
        }
    }

    /// Count sleeping bodies.
    pub fn sleeping_count(&self) -> usize {
        self.bodies.values().filter(|b| b.sleeping).count()
    }
}

// ── Inertia helpers ──────────────────────────────────────────

/// Compute the inertia tensor of a solid sphere.
pub fn sphere_inertia(mass: f64, radius: f64) -> Mat3 {
    let i = 0.4 * mass * radius * radius;
    Mat3::diagonal(i, i, i)
}

/// Compute the inertia tensor of a solid box (half-extents).
pub fn box_inertia(mass: f64, half_x: f64, half_y: f64, half_z: f64) -> Mat3 {
    let sx = (2.0 * half_x) * (2.0 * half_x);
    let sy = (2.0 * half_y) * (2.0 * half_y);
    let sz = (2.0 * half_z) * (2.0 * half_z);
    let factor = mass / 12.0;
    Mat3::diagonal(factor * (sy + sz), factor * (sx + sz), factor * (sx + sy))
}

/// Compute the inertia tensor of a solid cylinder aligned along Y.
pub fn cylinder_inertia(mass: f64, radius: f64, height: f64) -> Mat3 {
    let r2 = radius * radius;
    let h2 = height * height;
    let iy = 0.5 * mass * r2;
    let ixz = mass * (3.0 * r2 + h2) / 12.0;
    Mat3::diagonal(ixz, iy, ixz)
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn test_vec3_basics() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert!(approx_eq(sum.x, 5.0));
        assert!(approx_eq(sum.y, 7.0));
        assert!(approx_eq(sum.z, 9.0));
    }

    #[test]
    fn test_vec3_dot_and_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dot(b), 0.0));
        let c = a.cross(b);
        assert!(vec3_approx(c, Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalized();
        assert!(approx_eq(n.length(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));
    }

    #[test]
    fn test_vec3_zero_normalize() {
        let n = Vec3::ZERO.normalized();
        assert!(vec3_approx(n, Vec3::ZERO));
    }

    #[test]
    fn test_quaternion_identity_rotation() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = Quaternion::IDENTITY.rotate_vec(v);
        assert!(vec3_approx(rotated, v));
    }

    #[test]
    fn test_quaternion_90_deg_rotation() {
        let q = Quaternion::from_axis_angle(Vec3::UP, std::f64::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec(v);
        // 90 deg about Y: (1,0,0) -> (0,0,-1)
        assert!(vec3_approx(rotated, Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn test_quaternion_mul_inverse_is_identity() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalized(), 1.2);
        let prod = q.mul(q.conjugate());
        assert!(approx_eq(prod.w, 1.0));
        assert!(approx_eq(prod.x, 0.0));
        assert!(approx_eq(prod.y, 0.0));
        assert!(approx_eq(prod.z, 0.0));
    }

    #[test]
    fn test_quaternion_to_mat3_identity() {
        let m = Quaternion::IDENTITY.to_mat3();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(m.m[i][j], expected));
            }
        }
    }

    #[test]
    fn test_mat3_mul_vec_identity() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = Mat3::IDENTITY.mul_vec(v);
        assert!(vec3_approx(r, v));
    }

    #[test]
    fn test_mat3_inverse() {
        let m = Mat3::diagonal(2.0, 4.0, 8.0);
        let inv = m.inverse();
        assert!(approx_eq(inv.m[0][0], 0.5));
        assert!(approx_eq(inv.m[1][1], 0.25));
        assert!(approx_eq(inv.m[2][2], 0.125));
    }

    #[test]
    fn test_mat3_mul_mat_identity() {
        let m = Mat3::diagonal(3.0, 5.0, 7.0);
        let r = m.mul_mat(Mat3::IDENTITY);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(r.m[i][j], m.m[i][j]));
            }
        }
    }

    #[test]
    fn test_body_creation_dynamic() {
        let body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 10.0, Mat3::IDENTITY);
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert!(approx_eq(body.inv_mass, 0.1));
        assert!(!body.sleeping);
    }

    #[test]
    fn test_body_creation_static() {
        let body = RigidBody3D::new_static(2, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(body.body_type, BodyType::Static);
        assert!(approx_eq(body.inv_mass, 0.0));
    }

    #[test]
    fn test_body_creation_kinematic() {
        let body = RigidBody3D::new_kinematic(3, Vec3::ZERO);
        assert_eq!(body.body_type, BodyType::Kinematic);
        assert!(approx_eq(body.inv_mass, 0.0));
    }

    #[test]
    fn test_apply_force_dynamic() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 2.0, Mat3::IDENTITY);
        body.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert!(approx_eq(body.force.x, 10.0));
    }

    #[test]
    fn test_apply_force_static_ignored() {
        let mut body = RigidBody3D::new_static(1, Vec3::ZERO);
        body.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert!(approx_eq(body.force.x, 0.0));
    }

    #[test]
    fn test_apply_impulse() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 2.0, Mat3::IDENTITY);
        body.apply_impulse(Vec3::new(4.0, 0.0, 0.0));
        // dv = impulse / mass = 4/2 = 2
        assert!(approx_eq(body.linear_velocity.x, 2.0));
    }

    #[test]
    fn test_integrate_free_fall() {
        let inertia = sphere_inertia(1.0, 0.5);
        let mut body = RigidBody3D::new_dynamic(1, Vec3::new(0.0, 10.0, 0.0), 1.0, inertia);
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        body.sleep_threshold = 0.0; // prevent sleep

        let g = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            body.integrate(dt, g);
        }
        // After ~1 second of free fall, y should be around 10 - 0.5*g*t^2 ≈ 5.1
        assert!(body.position.y < 10.0);
        assert!(body.position.y > 0.0);
    }

    #[test]
    fn test_orientation_integration() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        body.angular_velocity = Vec3::new(0.0, std::f64::consts::PI, 0.0); // 180 deg/s about Y
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        body.sleep_threshold = 0.0;

        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            body.integrate(dt, Vec3::ZERO);
        }
        // After 1s at PI rad/s, should have rotated ~180 deg
        // The quaternion w should be close to 0 for a 180-deg rotation
        assert!(body.orientation.w.abs() < 0.15);
    }

    #[test]
    fn test_sleep_mechanism() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        body.linear_velocity = Vec3::ZERO;
        body.angular_velocity = Vec3::ZERO;
        body.sleep_threshold = 1.0;
        body.sleep_delay = 0.1;
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;

        let dt = 1.0 / 60.0;
        for _ in 0..20 {
            body.integrate(dt, Vec3::ZERO);
        }
        assert!(body.sleeping);
    }

    #[test]
    fn test_wake_from_sleep() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        body.sleeping = true;
        body.apply_force(Vec3::new(1.0, 0.0, 0.0));
        assert!(!body.sleeping);
    }

    #[test]
    fn test_velocity_at_point() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        body.angular_velocity = Vec3::new(0.0, 0.0, 1.0);
        // Point at (1,0,0), omega=(0,0,1) → v = omega x r = (0,0,1) x (1,0,0) = (0,1,0)
        let vp = body.velocity_at_point(Vec3::new(1.0, 0.0, 0.0));
        assert!(vec3_approx(vp, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_local_to_world() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::new(5.0, 0.0, 0.0), 1.0, Mat3::IDENTITY);
        body.orientation = Quaternion::IDENTITY;
        let wp = body.local_to_world(Vec3::new(1.0, 0.0, 0.0));
        assert!(vec3_approx(wp, Vec3::new(6.0, 0.0, 0.0)));
    }

    #[test]
    fn test_world_to_local_round_trip() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::new(2.0, 3.0, 4.0), 1.0, Mat3::IDENTITY);
        body.orientation = Quaternion::from_axis_angle(Vec3::UP, 0.7);
        let world_pt = Vec3::new(5.0, 6.0, 7.0);
        let local_pt = body.world_to_local(world_pt);
        let back = body.local_to_world(local_pt);
        assert!(vec3_approx(back, world_pt));
    }

    #[test]
    fn test_physics_world_add_remove() {
        let mut world = PhysicsWorld3D::new(Vec3::new(0.0, -9.81, 0.0));
        let b = RigidBody3D::new_dynamic(0, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        let id = world.add_body(b);
        assert_eq!(world.body_count(), 1);
        world.remove_body(id);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_physics_world_step() {
        let mut world = PhysicsWorld3D::new(Vec3::new(0.0, -9.81, 0.0));
        let mut b = RigidBody3D::new_dynamic(0, Vec3::new(0.0, 10.0, 0.0), 1.0, sphere_inertia(1.0, 0.5));
        b.linear_damping = 0.0;
        b.sleep_threshold = 0.0;
        let id = world.add_body(b);
        world.step(1.0 / 60.0);
        let body = world.get_body(id).unwrap();
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn test_sphere_inertia_values() {
        let i = sphere_inertia(5.0, 2.0);
        // I = 2/5 * m * r^2 = 0.4 * 5 * 4 = 8
        assert!(approx_eq(i.m[0][0], 8.0));
        assert!(approx_eq(i.m[1][1], 8.0));
        assert!(approx_eq(i.m[2][2], 8.0));
    }

    #[test]
    fn test_box_inertia_values() {
        let i = box_inertia(12.0, 1.0, 2.0, 3.0);
        // Ix = m/12 * (h^2 + d^2) = 12/12 * (16+36) = 52
        let sx = 4.0; let sy = 16.0; let sz = 36.0;
        assert!(approx_eq(i.m[0][0], 1.0 * (sy + sz)));
        assert!(approx_eq(i.m[1][1], 1.0 * (sx + sz)));
    }

    #[test]
    fn test_kinetic_energy_linear() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 2.0, Mat3::ZERO);
        body.linear_velocity = Vec3::new(3.0, 0.0, 0.0);
        // KE = 0.5 * 2 * 9 = 9
        assert!(approx_eq(body.kinetic_energy(), 9.0));
    }

    #[test]
    fn test_force_at_point_generates_torque() {
        let mut body = RigidBody3D::new_dynamic(1, Vec3::ZERO, 1.0, Mat3::IDENTITY);
        body.apply_force_at_point(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // torque = r x F = (1,0,0) x (0,1,0) = (0,0,1)
        assert!(vec3_approx(body.torque, Vec3::new(0.0, 0.0, 1.0)));
    }
}
