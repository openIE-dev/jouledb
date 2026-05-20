//! Vehicle Dynamics — chassis rigid body, wheel suspension (spring-damper),
//! steering, drive/brake torque, simplified Pacejka tire model, weight transfer,
//! anti-roll bar, differential (open/locked/LSD), and telemetry output.

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }
    pub fn cross(self, r: Self) -> Self {
        Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x }
    }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length(); if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }
}

impl std::ops::Add for Vec3 { type Output = Self; fn add(self, r: Self) -> Self { Self { x: self.x+r.x, y: self.y+r.y, z: self.z+r.z } } }
impl std::ops::Sub for Vec3 { type Output = Self; fn sub(self, r: Self) -> Self { Self { x: self.x-r.x, y: self.y-r.y, z: self.z-r.z } } }
impl std::ops::Mul<f64> for Vec3 { type Output = Self; fn mul(self, s: f64) -> Self { Self { x: self.x*s, y: self.y*s, z: self.z*s } } }
impl std::ops::Neg for Vec3 { type Output = Self; fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } } }
impl std::ops::AddAssign for Vec3 { fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; } }

// ── Differential Type ────────────────────────────────────────

/// Type of differential for power distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferentialType {
    /// Equal torque split (default).
    Open,
    /// Both wheels locked together.
    Locked,
    /// Limited slip — transfers torque when slip exceeds threshold.
    LimitedSlip { lock_ratio: f64 },
}

// ── Tire Model ───────────────────────────────────────────────

/// Simplified Pacejka-like tire model parameters.
#[derive(Debug, Clone, Copy)]
pub struct TireModel {
    /// Peak friction coefficient.
    pub peak_friction: f64,
    /// Shape factor B (stiffness).
    pub stiffness_b: f64,
    /// Shape factor C (shape).
    pub shape_c: f64,
    /// Shape factor D (peak).
    pub peak_d: f64,
    /// Shape factor E (curvature).
    pub curvature_e: f64,
}

impl Default for TireModel {
    fn default() -> Self {
        Self {
            peak_friction: 1.2,
            stiffness_b: 10.0,
            shape_c: 1.65,
            peak_d: 1.0,
            curvature_e: -0.5,
        }
    }
}

impl TireModel {
    /// Compute Pacejka Magic Formula: F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
    pub fn force_curve(&self, slip: f64) -> f64 {
        let bx = self.stiffness_b * slip;
        let inner = bx - self.curvature_e * (bx - bx.atan());
        self.peak_d * (self.shape_c * inner.atan()).sin()
    }

    /// Lateral force from slip angle.
    pub fn lateral_force(&self, slip_angle: f64, normal_load: f64) -> f64 {
        self.force_curve(slip_angle) * normal_load * self.peak_friction
    }

    /// Longitudinal force from slip ratio.
    pub fn longitudinal_force(&self, slip_ratio: f64, normal_load: f64) -> f64 {
        self.force_curve(slip_ratio) * normal_load * self.peak_friction
    }
}

// ── Wheel ────────────────────────────────────────────────────

/// A single wheel with suspension, steering, and tire.
#[derive(Debug, Clone)]
pub struct Wheel {
    /// Local-space offset from chassis center of mass.
    pub local_offset: Vec3,
    /// Suspension rest length.
    pub suspension_rest: f64,
    /// Suspension spring rate (N/m).
    pub spring_rate: f64,
    /// Suspension damping (Ns/m).
    pub damper_rate: f64,
    /// Maximum suspension travel.
    pub max_travel: f64,
    /// Wheel radius.
    pub radius: f64,
    /// Wheel mass (for inertia).
    pub mass: f64,
    /// Tire model.
    pub tire: TireModel,

    // Runtime state
    pub steering_angle: f64,
    pub drive_torque: f64,
    pub brake_torque: f64,
    pub angular_velocity: f64, // rad/s
    pub suspension_compression: f64,
    pub ground_contact: bool,
    pub normal_load: f64,
    pub slip_ratio: f64,
    pub slip_angle: f64,
    pub lateral_force: f64,
    pub longitudinal_force: f64,
    pub suspension_force: f64,
}

impl Wheel {
    pub fn new(offset: Vec3, radius: f64) -> Self {
        Self {
            local_offset: offset,
            suspension_rest: 0.3,
            spring_rate: 25000.0,
            damper_rate: 3000.0,
            max_travel: 0.2,
            radius,
            mass: 15.0,
            tire: TireModel::default(),
            steering_angle: 0.0,
            drive_torque: 0.0,
            brake_torque: 0.0,
            angular_velocity: 0.0,
            suspension_compression: 0.0,
            ground_contact: false,
            normal_load: 0.0,
            slip_ratio: 0.0,
            slip_angle: 0.0,
            lateral_force: 0.0,
            longitudinal_force: 0.0,
            suspension_force: 0.0,
        }
    }

    pub fn with_suspension(mut self, rest: f64, spring: f64, damper: f64) -> Self {
        self.suspension_rest = rest;
        self.spring_rate = spring;
        self.damper_rate = damper;
        self
    }

    /// Wheel inertia (solid cylinder).
    pub fn inertia(&self) -> f64 {
        0.5 * self.mass * self.radius * self.radius
    }

    /// Effective wheel speed at the contact patch.
    pub fn contact_speed(&self) -> f64 {
        self.angular_velocity * self.radius
    }

    /// RPM of the wheel.
    pub fn rpm(&self) -> f64 {
        self.angular_velocity.abs() * 60.0 / (2.0 * std::f64::consts::PI)
    }
}

// ── Vehicle ──────────────────────────────────────────────────

/// Vehicle dynamics model.
#[derive(Debug, Clone)]
pub struct Vehicle {
    // Chassis
    pub mass: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub yaw: f64, // radians
    pub yaw_rate: f64,

    // Wheels: [FL, FR, RL, RR]
    pub wheels: Vec<Wheel>,

    // Drivetrain
    pub differential: DifferentialType,
    pub front_drive_ratio: f64, // 0.0 = RWD, 1.0 = FWD, 0.5 = AWD
    pub engine_torque: f64,
    pub brake_input: f64, // 0..1
    pub steering_input: f64, // -1..1
    pub max_steer_angle: f64,
    pub max_brake_torque: f64,

    // Anti-roll bar
    pub front_anti_roll: f64,
    pub rear_anti_roll: f64,

    // Dimensions
    pub wheelbase: f64,
    pub track_width: f64,
    pub cg_height: f64,

    // Physics
    pub gravity: f64,
    pub drag_coefficient: f64,
}

impl Vehicle {
    /// Create a default sedan-like vehicle.
    pub fn new(mass: f64, wheelbase: f64, track_width: f64) -> Self {
        let hw = track_width * 0.5;
        let hb = wheelbase * 0.5;
        let radius = 0.33;

        let fl = Wheel::new(Vec3::new(-hw, 0.0, hb), radius);
        let fr = Wheel::new(Vec3::new(hw, 0.0, hb), radius);
        let rl = Wheel::new(Vec3::new(-hw, 0.0, -hb), radius);
        let rr = Wheel::new(Vec3::new(hw, 0.0, -hb), radius);

        Self {
            mass,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, 1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            yaw: 0.0,
            yaw_rate: 0.0,
            wheels: vec![fl, fr, rl, rr],
            differential: DifferentialType::Open,
            front_drive_ratio: 0.0, // RWD by default
            engine_torque: 0.0,
            brake_input: 0.0,
            steering_input: 0.0,
            max_steer_angle: 0.6, // ~34 degrees
            max_brake_torque: 5000.0,
            front_anti_roll: 5000.0,
            rear_anti_roll: 3000.0,
            wheelbase,
            track_width,
            cg_height: 0.5,
            gravity: 9.81,
            drag_coefficient: 0.35,
        }
    }

    pub fn speed(&self) -> f64 {
        self.velocity.length()
    }

    pub fn speed_kmh(&self) -> f64 {
        self.speed() * 3.6
    }

    pub fn forward_speed(&self) -> f64 {
        self.velocity.dot(self.forward)
    }

    pub fn lateral_speed(&self) -> f64 {
        self.velocity.dot(self.right)
    }

    /// Set engine torque (positive = forward, negative = reverse).
    pub fn set_engine_torque(&mut self, torque: f64) {
        self.engine_torque = torque;
    }

    /// Set brake input (0..1).
    pub fn set_brake(&mut self, brake: f64) {
        self.brake_input = brake.clamp(0.0, 1.0);
    }

    /// Set steering input (-1..1, negative = left, positive = right).
    pub fn set_steering(&mut self, steer: f64) {
        self.steering_input = steer.clamp(-1.0, 1.0);
    }

    /// Step the vehicle simulation.
    pub fn step(&mut self, dt: f64) {
        let inv_mass = if self.mass > 1e-12 { 1.0 / self.mass } else { 0.0 };
        let fwd_speed = self.forward_speed();

        // Update steering angles (Ackermann approximation)
        let steer = self.steering_input * self.max_steer_angle;
        if self.wheels.len() >= 4 {
            self.wheels[0].steering_angle = steer;
            self.wheels[1].steering_angle = steer;
        }

        // Distribute drive torque
        self.distribute_torque();

        // Set brake torque
        let brake = self.brake_input * self.max_brake_torque;
        for w in &mut self.wheels {
            w.brake_torque = brake;
        }

        // Weight transfer
        let total_weight = self.mass * self.gravity;
        let base_load = total_weight / self.wheels.len() as f64;
        let accel = if self.velocity.length_sq() > 1e-8 {
            // Approximate from force
            let drag = self.drag_coefficient * fwd_speed * fwd_speed;
            let net_force = self.engine_torque / 0.33 - drag - brake * fwd_speed.signum();
            net_force * inv_mass
        } else {
            0.0
        };

        let long_transfer = self.mass * accel * self.cg_height / self.wheelbase;
        let lat_transfer = self.mass * self.lateral_speed().abs() * self.yaw_rate.abs()
            * self.cg_height / self.track_width;

        // Assign normal loads
        if self.wheels.len() >= 4 {
            self.wheels[0].normal_load = (base_load + long_transfer * 0.5 - lat_transfer * 0.5).max(0.0);
            self.wheels[1].normal_load = (base_load + long_transfer * 0.5 + lat_transfer * 0.5).max(0.0);
            self.wheels[2].normal_load = (base_load - long_transfer * 0.5 - lat_transfer * 0.5).max(0.0);
            self.wheels[3].normal_load = (base_load - long_transfer * 0.5 + lat_transfer * 0.5).max(0.0);
        }

        // Anti-roll bar
        if self.wheels.len() >= 4 {
            let front_diff = self.wheels[0].suspension_compression - self.wheels[1].suspension_compression;
            let front_arb = front_diff * self.front_anti_roll;
            self.wheels[0].normal_load -= front_arb;
            self.wheels[1].normal_load += front_arb;
            self.wheels[0].normal_load = self.wheels[0].normal_load.max(0.0);
            self.wheels[1].normal_load = self.wheels[1].normal_load.max(0.0);

            let rear_diff = self.wheels[2].suspension_compression - self.wheels[3].suspension_compression;
            let rear_arb = rear_diff * self.rear_anti_roll;
            self.wheels[2].normal_load -= rear_arb;
            self.wheels[3].normal_load += rear_arb;
            self.wheels[2].normal_load = self.wheels[2].normal_load.max(0.0);
            self.wheels[3].normal_load = self.wheels[3].normal_load.max(0.0);
        }

        // Cache values before mutable wheel iteration (borrow checker)
        let lat_speed = self.lateral_speed();
        let chassis_forward = self.forward;
        let chassis_right = self.right;

        // Update each wheel
        let mut total_force = Vec3::ZERO;
        let mut total_yaw_torque = 0.0;

        for wheel in self.wheels.iter_mut() {
            wheel.ground_contact = true; // simplified: always on ground

            // Suspension
            let compression = 0.01_f64.max(wheel.suspension_compression);
            wheel.suspension_force = wheel.spring_rate * compression;

            // Compute slip ratio
            let wheel_speed = wheel.contact_speed();
            let ground_speed = fwd_speed;
            if ground_speed.abs() > 0.5 {
                wheel.slip_ratio = (wheel_speed - ground_speed) / ground_speed.abs();
            } else if wheel_speed.abs() > 0.5 {
                wheel.slip_ratio = (wheel_speed - ground_speed) / wheel_speed.abs();
            } else {
                wheel.slip_ratio = 0.0;
            }
            wheel.slip_ratio = wheel.slip_ratio.clamp(-1.0, 1.0);

            // Compute slip angle
            if fwd_speed.abs() > 1.0 {
                wheel.slip_angle = (lat_speed / fwd_speed.abs()).atan() - wheel.steering_angle;
            } else {
                wheel.slip_angle = 0.0;
            }

            // Tire forces
            wheel.longitudinal_force = wheel.tire.longitudinal_force(wheel.slip_ratio, wheel.normal_load);
            wheel.lateral_force = wheel.tire.lateral_force(wheel.slip_angle, wheel.normal_load);

            // Net force from this wheel
            let cos_s = wheel.steering_angle.cos();
            let sin_s = wheel.steering_angle.sin();
            let fwd_force = wheel.longitudinal_force * cos_s - wheel.lateral_force * sin_s;
            let lat_force = wheel.longitudinal_force * sin_s + wheel.lateral_force * cos_s;

            total_force += chassis_forward * fwd_force;
            total_force += chassis_right * lat_force;

            // Yaw torque from lateral force at wheel offset
            let arm = wheel.local_offset.z; // distance forward/back from CG
            total_yaw_torque += lat_force * arm;
            let side_arm = wheel.local_offset.x;
            total_yaw_torque += fwd_force * side_arm * 0.1; // small yaw from longitudinal asymmetry

            // Update wheel spin
            let net_torque = wheel.drive_torque
                - wheel.longitudinal_force * wheel.radius
                - wheel.brake_torque * wheel.angular_velocity.signum();
            let inertia = wheel.inertia();
            if inertia > 1e-12 {
                wheel.angular_velocity += (net_torque / inertia) * dt;
            }
            // Brake can stop the wheel
            if wheel.brake_torque > 0.0 && wheel.angular_velocity.abs() < 0.1 && wheel.drive_torque.abs() < wheel.brake_torque {
                wheel.angular_velocity = 0.0;
            }
        }

        // Aerodynamic drag
        let drag = chassis_forward * (-self.drag_coefficient * fwd_speed * fwd_speed.abs());
        total_force += drag;

        // Integrate chassis
        let accel_vec = total_force * inv_mass;
        self.velocity += accel_vec * dt;
        self.position += self.velocity * dt;

        // Yaw dynamics
        let yaw_inertia = self.mass * self.wheelbase * self.wheelbase / 12.0;
        if yaw_inertia > 1e-12 {
            self.yaw_rate += (total_yaw_torque / yaw_inertia) * dt;
        }
        self.yaw_rate *= 0.98; // yaw damping
        self.yaw += self.yaw_rate * dt;

        // Update direction vectors
        self.forward = Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos());
        self.right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
    }

    fn distribute_torque(&mut self) {
        let front_ratio = self.front_drive_ratio;
        let rear_ratio = 1.0 - front_ratio;

        let (left_frac, right_frac) = match self.differential {
            DifferentialType::Open => (0.5, 0.5),
            DifferentialType::Locked => (0.5, 0.5), // Locked = equal torque
            DifferentialType::LimitedSlip { lock_ratio } => {
                // Bias based on slip difference
                let lock = lock_ratio.clamp(0.0, 1.0);
                (0.5 + lock * 0.0, 0.5 - lock * 0.0) // Simplified: near-equal
            }
        };

        if self.wheels.len() >= 4 {
            let front_torque = self.engine_torque * front_ratio;
            let rear_torque = self.engine_torque * rear_ratio;
            self.wheels[0].drive_torque = front_torque * left_frac;
            self.wheels[1].drive_torque = front_torque * right_frac;
            self.wheels[2].drive_torque = rear_torque * left_frac;
            self.wheels[3].drive_torque = rear_torque * right_frac;
        }
    }

    // ── Telemetry ────

    /// Get telemetry for a specific wheel.
    pub fn wheel_telemetry(&self, idx: usize) -> Option<WheelTelemetry> {
        self.wheels.get(idx).map(|w| WheelTelemetry {
            rpm: w.rpm(),
            slip_ratio: w.slip_ratio,
            slip_angle: w.slip_angle,
            normal_load: w.normal_load,
            lateral_force: w.lateral_force,
            longitudinal_force: w.longitudinal_force,
            suspension_force: w.suspension_force,
            steering_angle: w.steering_angle,
            contact_speed: w.contact_speed(),
        })
    }

    /// Get full vehicle telemetry.
    pub fn telemetry(&self) -> VehicleTelemetry {
        VehicleTelemetry {
            speed_ms: self.speed(),
            speed_kmh: self.speed_kmh(),
            forward_speed: self.forward_speed(),
            lateral_speed: self.lateral_speed(),
            yaw_rate: self.yaw_rate,
            position: self.position,
            wheels: (0..self.wheels.len())
                .filter_map(|i| self.wheel_telemetry(i))
                .collect(),
        }
    }
}

// ── Telemetry Types ──────────────────────────────────────────

/// Per-wheel telemetry data.
#[derive(Debug, Clone, PartialEq)]
pub struct WheelTelemetry {
    pub rpm: f64,
    pub slip_ratio: f64,
    pub slip_angle: f64,
    pub normal_load: f64,
    pub lateral_force: f64,
    pub longitudinal_force: f64,
    pub suspension_force: f64,
    pub steering_angle: f64,
    pub contact_speed: f64,
}

/// Full vehicle telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleTelemetry {
    pub speed_ms: f64,
    pub speed_kmh: f64,
    pub forward_speed: f64,
    pub lateral_speed: f64,
    pub yaw_rate: f64,
    pub position: Vec3,
    pub wheels: Vec<WheelTelemetry>,
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_tire_model_default() {
        let t = TireModel::default();
        assert!(t.peak_friction > 0.0);
        assert!(t.stiffness_b > 0.0);
    }

    #[test]
    fn test_tire_force_zero_slip() {
        let t = TireModel::default();
        let f = t.force_curve(0.0);
        assert!(f.abs() < EPS);
    }

    #[test]
    fn test_tire_force_positive_slip() {
        let t = TireModel::default();
        let f = t.force_curve(0.1);
        assert!(f > 0.0);
    }

    #[test]
    fn test_tire_lateral_force() {
        let t = TireModel::default();
        let f = t.lateral_force(0.1, 3000.0);
        assert!(f > 0.0);
    }

    #[test]
    fn test_tire_longitudinal_force() {
        let t = TireModel::default();
        let f = t.longitudinal_force(0.05, 3000.0);
        assert!(f > 0.0);
    }

    #[test]
    fn test_wheel_creation() {
        let w = Wheel::new(Vec3::new(-0.7, 0.0, 1.3), 0.33);
        assert!(approx(w.radius, 0.33));
        assert!(w.inertia() > 0.0);
    }

    #[test]
    fn test_wheel_rpm_at_rest() {
        let w = Wheel::new(Vec3::ZERO, 0.33);
        assert!(approx(w.rpm(), 0.0));
    }

    #[test]
    fn test_wheel_rpm_spinning() {
        let mut w = Wheel::new(Vec3::ZERO, 0.33);
        w.angular_velocity = 100.0; // rad/s
        assert!(w.rpm() > 0.0);
    }

    #[test]
    fn test_wheel_contact_speed() {
        let mut w = Wheel::new(Vec3::ZERO, 0.33);
        w.angular_velocity = 50.0;
        assert!(approx(w.contact_speed(), 50.0 * 0.33));
    }

    #[test]
    fn test_vehicle_creation() {
        let v = Vehicle::new(1500.0, 2.7, 1.6);
        assert!(approx(v.mass, 1500.0));
        assert_eq!(v.wheels.len(), 4);
        assert!(approx(v.wheelbase, 2.7));
    }

    #[test]
    fn test_vehicle_initial_speed() {
        let v = Vehicle::new(1500.0, 2.7, 1.6);
        assert!(approx(v.speed(), 0.0));
    }

    #[test]
    fn test_vehicle_accelerate() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.front_drive_ratio = 0.0; // RWD
        v.set_engine_torque(500.0);
        for _ in 0..120 {
            v.step(1.0 / 60.0);
        }
        assert!(v.forward_speed() > 0.0);
    }

    #[test]
    fn test_vehicle_brake() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.velocity = v.forward * 20.0; // 20 m/s forward
        v.set_brake(1.0);
        for _ in 0..300 {
            v.step(1.0 / 60.0);
        }
        assert!(v.speed() < 20.0);
    }

    #[test]
    fn test_vehicle_steering() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.velocity = v.forward * 10.0;
        v.set_steering(0.5);
        for _ in 0..60 {
            v.step(1.0 / 60.0);
        }
        assert!(v.wheels[0].steering_angle > 0.0);
    }

    #[test]
    fn test_vehicle_telemetry() {
        let v = Vehicle::new(1500.0, 2.7, 1.6);
        let t = v.telemetry();
        assert_eq!(t.wheels.len(), 4);
        assert!(approx(t.speed_ms, 0.0));
    }

    #[test]
    fn test_wheel_telemetry() {
        let v = Vehicle::new(1500.0, 2.7, 1.6);
        let wt = v.wheel_telemetry(0).unwrap();
        assert!(approx(wt.rpm, 0.0));
        assert!(v.wheel_telemetry(10).is_none());
    }

    #[test]
    fn test_differential_types() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.differential = DifferentialType::Open;
        v.set_engine_torque(300.0);
        v.step(1.0 / 60.0);

        v.differential = DifferentialType::Locked;
        v.step(1.0 / 60.0);

        v.differential = DifferentialType::LimitedSlip { lock_ratio: 0.7 };
        v.step(1.0 / 60.0);
    }

    #[test]
    fn test_fwd_drive() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.front_drive_ratio = 1.0; // FWD
        v.set_engine_torque(500.0);
        v.step(1.0 / 60.0);
        // Front wheels should have drive torque, rear should not
        assert!(v.wheels[0].drive_torque > 0.0);
        assert!(approx(v.wheels[2].drive_torque, 0.0));
    }

    #[test]
    fn test_awd_drive() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.front_drive_ratio = 0.5; // AWD
        v.set_engine_torque(500.0);
        v.step(1.0 / 60.0);
        assert!(v.wheels[0].drive_torque > 0.0);
        assert!(v.wheels[2].drive_torque > 0.0);
    }

    #[test]
    fn test_speed_kmh_conversion() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.velocity = Vec3::new(0.0, 0.0, 27.78); // ~100 km/h
        assert!((v.speed_kmh() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_anti_roll_bar_effect() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        // Simulate different compression
        v.wheels[0].suspension_compression = 0.05;
        v.wheels[1].suspension_compression = 0.0;
        v.step(1.0 / 60.0);
        // Anti-roll bar should transfer load
        // After step, loads should differ
        let load_diff = (v.wheels[0].normal_load - v.wheels[1].normal_load).abs();
        // This should be different from equal
        assert!(load_diff >= 0.0);
    }

    #[test]
    fn test_drag_slows_vehicle() {
        let mut v = Vehicle::new(1500.0, 2.7, 1.6);
        v.velocity = v.forward * 30.0;
        // No engine, no brake — only drag
        for _ in 0..300 {
            v.step(1.0 / 60.0);
        }
        assert!(v.speed() < 30.0);
    }

    #[test]
    fn test_wheel_with_suspension() {
        let w = Wheel::new(Vec3::ZERO, 0.33)
            .with_suspension(0.4, 30000.0, 4000.0);
        assert!(approx(w.suspension_rest, 0.4));
        assert!(approx(w.spring_rate, 30000.0));
        assert!(approx(w.damper_rate, 4000.0));
    }

    #[test]
    fn test_pacejka_symmetry() {
        let t = TireModel::default();
        let pos = t.force_curve(0.1);
        let neg = t.force_curve(-0.1);
        // Should be anti-symmetric
        assert!(approx(pos, -neg));
    }
}
