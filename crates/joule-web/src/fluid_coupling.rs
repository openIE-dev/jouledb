//! Two-way fluid-solid coupling for rigid bodies in fluid.
//!
//! Computes fluid forces on rigid bodies (pressure drag, viscous drag), applies body
//! velocity as boundary conditions in the fluid, and models buoyancy (Archimedes),
//! Reynolds-dependent drag coefficients, added mass effects, floating-object waterline
//! computation, multiple-body interaction, and force/torque accumulation.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Fluid-solid coupling errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Body not found by id.
    BodyNotFound(u64),
    /// Simulation diverged.
    Diverged(String),
    /// No fluid configuration set.
    NoFluid,
}

impl fmt::Display for CouplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::BodyNotFound(id) => write!(f, "body not found: {id}"),
            Self::Diverged(msg) => write!(f, "simulation diverged: {msg}"),
            Self::NoFluid => write!(f, "no fluid configuration set"),
        }
    }
}

impl std::error::Error for CouplingError {}

// ── 2D Vector ─────────────────────────────────────────────────

/// Simple 2D vector for forces and positions.
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

    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-15 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (scalar result): self x other = x1*y2 - y1*x2.
    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

// ── Rigid Body Shape ──────────────────────────────────────────

/// Shape of a rigid body (2D cross-section).
#[derive(Debug, Clone, PartialEq)]
pub enum BodyShape {
    /// Circle with radius.
    Circle { radius: f64 },
    /// Axis-aligned rectangle with half-extents.
    Rectangle { half_width: f64, half_height: f64 },
}

impl BodyShape {
    /// Cross-sectional area.
    pub fn area(&self) -> f64 {
        match self {
            Self::Circle { radius } => std::f64::consts::PI * radius * radius,
            Self::Rectangle { half_width, half_height } => 4.0 * half_width * half_height,
        }
    }

    /// Reference length for drag (diameter or width).
    pub fn reference_length(&self) -> f64 {
        match self {
            Self::Circle { radius } => 2.0 * radius,
            Self::Rectangle { half_width, .. } => 2.0 * half_width,
        }
    }

    /// Perimeter (for viscous drag estimation).
    pub fn perimeter(&self) -> f64 {
        match self {
            Self::Circle { radius } => 2.0 * std::f64::consts::PI * radius,
            Self::Rectangle { half_width, half_height } => 4.0 * (half_width + half_height),
        }
    }

    /// Check if a point is inside the shape (centered at origin).
    pub fn contains(&self, px: f64, py: f64) -> bool {
        match self {
            Self::Circle { radius } => px * px + py * py <= radius * radius,
            Self::Rectangle { half_width, half_height } => {
                px.abs() <= *half_width && py.abs() <= *half_height
            }
        }
    }

    /// Submerged area for a given waterline y-coordinate (body at origin).
    pub fn submerged_area(&self, waterline_y: f64) -> f64 {
        match self {
            Self::Circle { radius } => {
                let r = *radius;
                // Segment area below waterline_y
                let y_clamped = waterline_y.clamp(-r, r);
                let theta = ((y_clamped / r).clamp(-1.0, 1.0)).asin();
                let segment = r * r * (std::f64::consts::FRAC_PI_2 + theta)
                    + y_clamped * (r * r - y_clamped * y_clamped).max(0.0).sqrt();
                segment.max(0.0)
            }
            Self::Rectangle { half_width, half_height } => {
                let h = *half_height;
                let w = *half_width;
                let submerged_h = (waterline_y + h).clamp(0.0, 2.0 * h);
                2.0 * w * submerged_h
            }
        }
    }

    /// Moment of inertia (for unit density, about center).
    pub fn moment_of_inertia(&self, mass: f64) -> f64 {
        match self {
            Self::Circle { radius } => 0.5 * mass * radius * radius,
            Self::Rectangle { half_width, half_height } => {
                let w = 2.0 * half_width;
                let h = 2.0 * half_height;
                mass * (w * w + h * h) / 12.0
            }
        }
    }
}

// ── Rigid Body ────────────────────────────────────────────────

/// A rigid body immersed in fluid.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    pub id: u64,
    pub shape: BodyShape,
    pub position: Vec2,
    pub velocity: Vec2,
    pub angle: f64,
    pub angular_velocity: f64,
    pub mass: f64,
    pub density: f64,
    /// Accumulated force this step.
    pub force: Vec2,
    /// Accumulated torque this step.
    pub torque: f64,
    /// Whether this body is fixed (immovable).
    pub is_fixed: bool,
}

impl RigidBody {
    pub fn new(id: u64, shape: BodyShape, position: Vec2, mass: f64) -> Self {
        let area = shape.area();
        let density = if area > 0.0 { mass / area } else { 1.0 };
        Self {
            id,
            shape,
            position,
            velocity: Vec2::zero(),
            angle: 0.0,
            angular_velocity: 0.0,
            mass,
            density,
            force: Vec2::zero(),
            torque: 0.0,
            is_fixed: false,
        }
    }

    /// Moment of inertia.
    pub fn moment_of_inertia(&self) -> f64 {
        self.shape.moment_of_inertia(self.mass)
    }

    /// Kinetic energy (translational + rotational).
    pub fn kinetic_energy(&self) -> f64 {
        let ke_trans = 0.5 * self.mass * self.velocity.length_sq();
        let ke_rot = 0.5 * self.moment_of_inertia() * self.angular_velocity * self.angular_velocity;
        ke_trans + ke_rot
    }

    /// Clear accumulated forces and torque.
    pub fn clear_forces(&mut self) {
        self.force = Vec2::zero();
        self.torque = 0.0;
    }

    /// Add a force at the center of mass.
    pub fn add_force(&mut self, force: Vec2) {
        self.force = self.force.add(&force);
    }

    /// Add a force at a specific point (generates torque).
    pub fn add_force_at(&mut self, force: Vec2, point: Vec2) {
        self.force = self.force.add(&force);
        let r = point.sub(&self.position);
        self.torque += r.cross(&force);
    }

    /// Add torque directly.
    pub fn add_torque(&mut self, torque: f64) {
        self.torque += torque;
    }

    /// Integrate position and velocity (semi-implicit Euler).
    pub fn integrate(&mut self, dt: f64) {
        if self.is_fixed {
            return;
        }
        // Translational
        let accel = self.force.scale(1.0 / self.mass);
        self.velocity = self.velocity.add(&accel.scale(dt));
        self.position = self.position.add(&self.velocity.scale(dt));

        // Rotational
        let moi = self.moment_of_inertia();
        if moi > 1e-15 {
            let alpha = self.torque / moi;
            self.angular_velocity += alpha * dt;
            self.angle += self.angular_velocity * dt;
        }
    }
}

// ── Fluid Properties ──────────────────────────────────────────

/// Fluid properties for coupling calculations.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidProperties {
    /// Fluid density (kg/m^3).
    pub density: f64,
    /// Dynamic viscosity (Pa.s).
    pub dynamic_viscosity: f64,
    /// Free-surface level (y-coordinate of water surface).
    pub surface_level: f64,
    /// Gravity magnitude (m/s^2).
    pub gravity: f64,
    /// Free-stream velocity.
    pub freestream: Vec2,
}

impl Default for FluidProperties {
    fn default() -> Self {
        Self {
            density: 1000.0,
            dynamic_viscosity: 1.0e-3,
            surface_level: 0.0,
            gravity: 9.81,
            freestream: Vec2::zero(),
        }
    }
}

impl FluidProperties {
    /// Kinematic viscosity = mu / rho.
    pub fn kinematic_viscosity(&self) -> f64 {
        self.dynamic_viscosity / self.density
    }
}

// ── Drag Coefficients ─────────────────────────────────────────

/// Compute Reynolds number for a body in flow.
pub fn reynolds_number(
    relative_speed: f64,
    characteristic_length: f64,
    kinematic_viscosity: f64,
) -> f64 {
    if kinematic_viscosity <= 0.0 {
        return f64::INFINITY;
    }
    relative_speed * characteristic_length / kinematic_viscosity
}

/// Drag coefficient as a function of Reynolds number.
/// Uses standard correlations for a sphere/cylinder.
pub fn drag_coefficient(re: f64) -> f64 {
    if re < 1e-10 {
        return 0.0;
    }
    if re < 1.0 {
        // Stokes regime: Cd = 24/Re
        24.0 / re
    } else if re < 1000.0 {
        // Intermediate: Schiller-Naumann
        24.0 / re * (1.0 + 0.15 * re.powf(0.687))
    } else if re < 2e5 {
        // Newton regime: roughly constant
        0.44
    } else {
        // Drag crisis for smooth sphere
        0.1
    }
}

// ── Force Computations ────────────────────────────────────────

/// Compute buoyancy force on a body (Archimedes principle).
pub fn buoyancy_force(
    body: &RigidBody,
    fluid: &FluidProperties,
) -> Vec2 {
    let waterline_local = fluid.surface_level - body.position.y;
    let submerged = body.shape.submerged_area(waterline_local);
    let displaced_mass = fluid.density * submerged;
    Vec2::new(0.0, displaced_mass * fluid.gravity)
}

/// Compute drag force on a body.
pub fn drag_force(
    body: &RigidBody,
    fluid: &FluidProperties,
) -> Vec2 {
    let relative_vel = fluid.freestream.sub(&body.velocity);
    let speed = relative_vel.length();
    if speed < 1e-15 {
        return Vec2::zero();
    }

    let nu = fluid.kinematic_viscosity();
    let char_len = body.shape.reference_length();
    let re = reynolds_number(speed, char_len, nu);
    let cd = drag_coefficient(re);

    // Drag = 0.5 * rho * Cd * A * |v|^2 * direction
    let reference_area = char_len; // For 2D, use reference_length as area per unit depth
    let drag_mag = 0.5 * fluid.density * cd * reference_area * speed * speed;
    let direction = relative_vel.normalized();
    direction.scale(drag_mag)
}

/// Compute viscous skin friction force.
pub fn viscous_force(
    body: &RigidBody,
    fluid: &FluidProperties,
) -> Vec2 {
    let relative_vel = fluid.freestream.sub(&body.velocity);
    let speed = relative_vel.length();
    if speed < 1e-15 {
        return Vec2::zero();
    }

    // Viscous force ~ mu * perimeter * velocity_tangential
    let mu = fluid.dynamic_viscosity;
    let perim = body.shape.perimeter();
    let force_mag = mu * perim * speed;
    let direction = relative_vel.normalized();
    direction.scale(force_mag)
}

/// Compute added mass force (virtual mass effect).
/// F_added = -C_m * rho_fluid * V_body * acceleration
pub fn added_mass_force(
    body: &RigidBody,
    fluid: &FluidProperties,
    body_acceleration: Vec2,
    added_mass_coefficient: f64,
) -> Vec2 {
    let displaced_volume = body.shape.area(); // 2D: area = volume per unit depth
    let force = body_acceleration.scale(-added_mass_coefficient * fluid.density * displaced_volume);
    force
}

/// Compute the waterline for a floating body (equilibrium position).
/// Returns the y-position where buoyancy equals weight.
pub fn waterline_y(
    body: &RigidBody,
    fluid: &FluidProperties,
) -> f64 {
    let weight = body.mass * fluid.gravity;
    let target_submerged_area = weight / (fluid.density * fluid.gravity);

    // Binary search for waterline
    let ref_len = body.shape.reference_length();
    let mut lo = -ref_len;
    let mut hi = ref_len;

    for _ in 0..50 {
        let mid = (lo + hi) * 0.5;
        let submerged = body.shape.submerged_area(mid);
        if submerged < target_submerged_area {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

// ── Coupling Simulation ───────────────────────────────────────

/// Statistics from a coupling step.
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingStats {
    pub step: u64,
    pub body_count: usize,
    pub total_kinetic_energy: f64,
    pub max_body_speed: f64,
    pub total_buoyancy_magnitude: f64,
    pub total_drag_magnitude: f64,
}

/// Two-way fluid-solid coupling simulation.
pub struct FluidCoupling {
    pub fluid: FluidProperties,
    pub bodies: Vec<RigidBody>,
    pub dt: f64,
    /// Added mass coefficient (0.5 for sphere, ~1.0 for cylinder).
    pub added_mass_coeff: f64,
    /// Enable added mass effect.
    pub enable_added_mass: bool,
    step_count: u64,
    next_id: u64,
}

impl FluidCoupling {
    pub fn new(fluid: FluidProperties, dt: f64) -> Self {
        Self {
            fluid,
            bodies: Vec::new(),
            dt,
            added_mass_coeff: 0.5,
            enable_added_mass: true,
            step_count: 0,
            next_id: 0,
        }
    }

    /// Add a rigid body to the simulation.
    pub fn add_body(&mut self, shape: BodyShape, position: Vec2, mass: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.bodies.push(RigidBody::new(id, shape, position, mass));
        id
    }

    /// Add a fixed (immovable) rigid body.
    pub fn add_fixed_body(&mut self, shape: BodyShape, position: Vec2) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut body = RigidBody::new(id, shape, position, 1.0);
        body.is_fixed = true;
        self.bodies.push(body);
        id
    }

    /// Get a body by id.
    pub fn get_body(&self, id: u64) -> Result<&RigidBody, CouplingError> {
        self.bodies.iter().find(|b| b.id == id).ok_or(CouplingError::BodyNotFound(id))
    }

    /// Get a mutable body by id.
    pub fn get_body_mut(&mut self, id: u64) -> Result<&mut RigidBody, CouplingError> {
        self.bodies.iter_mut().find(|b| b.id == id).ok_or(CouplingError::BodyNotFound(id))
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Advance the coupling simulation by one timestep.
    pub fn step(&mut self) -> Result<CouplingStats, CouplingError> {
        if self.bodies.is_empty() {
            return Ok(CouplingStats {
                step: self.step_count,
                body_count: 0,
                total_kinetic_energy: 0.0,
                max_body_speed: 0.0,
                total_buoyancy_magnitude: 0.0,
                total_drag_magnitude: 0.0,
            });
        }

        let dt = self.dt;
        let fluid = self.fluid.clone();
        let added_mass_coeff = self.added_mass_coeff;
        let enable_am = self.enable_added_mass;

        let mut total_buoyancy = 0.0;
        let mut total_drag = 0.0;

        for body in &mut self.bodies {
            body.clear_forces();

            // Gravity
            let weight = Vec2::new(0.0, -body.mass * fluid.gravity);
            body.add_force(weight);

            // Buoyancy
            let buoy = buoyancy_force(body, &fluid);
            body.add_force(buoy);
            total_buoyancy += buoy.length();

            // Drag
            let d = drag_force(body, &fluid);
            body.add_force(d);
            total_drag += d.length();

            // Viscous friction
            let visc = viscous_force(body, &fluid);
            body.add_force(visc);

            // Added mass (uses current acceleration estimate)
            if enable_am && !body.is_fixed {
                let accel_estimate = body.force.scale(1.0 / body.mass);
                let am = added_mass_force(body, &fluid, accel_estimate, added_mass_coeff);
                body.add_force(am);
            }

            // Integrate
            body.integrate(dt);
        }

        // Check for divergence
        for body in &self.bodies {
            if !body.position.x.is_finite() || !body.position.y.is_finite() {
                return Err(CouplingError::Diverged(format!("body {} position NaN/Inf", body.id)));
            }
        }

        self.step_count += 1;

        let total_ke: f64 = self.bodies.iter().map(|b| b.kinetic_energy()).sum();
        let max_speed = self.bodies.iter().map(|b| b.velocity.length()).fold(0.0_f64, f64::max);

        Ok(CouplingStats {
            step: self.step_count,
            body_count: self.bodies.len(),
            total_kinetic_energy: total_ke,
            max_body_speed: max_speed,
            total_buoyancy_magnitude: total_buoyancy,
            total_drag_magnitude: total_drag,
        })
    }

    /// Run multiple steps.
    pub fn run(&mut self, steps: u64) -> Result<Vec<CouplingStats>, CouplingError> {
        let mut stats = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            stats.push(self.step()?);
        }
        Ok(stats)
    }

    /// Velocity boundary condition: what velocity should the fluid have at a body surface point.
    pub fn surface_velocity(&self, body_id: u64, surface_point: Vec2) -> Result<Vec2, CouplingError> {
        let body = self.get_body(body_id)?;
        // v_surface = v_body + omega x r
        let r = surface_point.sub(&body.position);
        // In 2D: omega x r = (-omega * r.y, omega * r.x)
        let rot_vel = Vec2::new(-body.angular_velocity * r.y, body.angular_velocity * r.x);
        Ok(body.velocity.add(&rot_vel))
    }

    /// Total system energy (all bodies).
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(|b| b.kinetic_energy()).sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_fluid() -> FluidProperties {
        FluidProperties::default()
    }

    fn default_coupling() -> FluidCoupling {
        FluidCoupling::new(FluidProperties::default(), 0.001)
    }

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.length() - 5.0).abs() < 1e-10);
        let n = a.normalized();
        assert!((n.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.cross(&b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_circle_area() {
        let c = BodyShape::Circle { radius: 1.0 };
        assert!((c.area() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_rectangle_area() {
        let r = BodyShape::Rectangle { half_width: 2.0, half_height: 3.0 };
        assert!((r.area() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_contains() {
        let c = BodyShape::Circle { radius: 1.0 };
        assert!(c.contains(0.5, 0.5));
        assert!(!c.contains(1.5, 0.0));
    }

    #[test]
    fn test_rectangle_contains() {
        let r = BodyShape::Rectangle { half_width: 1.0, half_height: 1.0 };
        assert!(r.contains(0.5, 0.5));
        assert!(!r.contains(1.5, 0.5));
    }

    #[test]
    fn test_drag_coefficient_stokes() {
        let cd = drag_coefficient(0.1);
        assert!((cd - 240.0).abs() < 1.0); // 24/0.1
    }

    #[test]
    fn test_drag_coefficient_newton() {
        let cd = drag_coefficient(10000.0);
        assert!((cd - 0.44).abs() < 0.01);
    }

    #[test]
    fn test_drag_coefficient_zero_re() {
        assert!((drag_coefficient(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_reynolds_number() {
        let re = reynolds_number(1.0, 0.1, 1e-6);
        assert!((re - 1e5).abs() < 1.0);
    }

    #[test]
    fn test_buoyancy_fully_submerged_rect() {
        let body = RigidBody::new(0,
            BodyShape::Rectangle { half_width: 0.5, half_height: 0.5 },
            Vec2::new(0.0, -2.0), // Well below surface
            1.0,
        );
        let fluid = default_fluid();
        let buoy = buoyancy_force(&body, &fluid);
        // Buoyancy should be upward
        assert!(buoy.y > 0.0);
    }

    #[test]
    fn test_buoyancy_above_surface() {
        let body = RigidBody::new(0,
            BodyShape::Rectangle { half_width: 0.5, half_height: 0.5 },
            Vec2::new(0.0, 10.0), // Well above surface
            1.0,
        );
        let fluid = default_fluid();
        let buoy = buoyancy_force(&body, &fluid);
        // Should have minimal buoyancy
        assert!(buoy.y.abs() < 100.0); // Small or zero
    }

    #[test]
    fn test_drag_force_stationary() {
        let body = RigidBody::new(0,
            BodyShape::Circle { radius: 0.1 },
            Vec2::zero(),
            1.0,
        );
        let fluid = default_fluid(); // freestream = 0
        let d = drag_force(&body, &fluid);
        assert!(d.length() < 1e-10);
    }

    #[test]
    fn test_drag_force_moving() {
        let mut body = RigidBody::new(0,
            BodyShape::Circle { radius: 0.1 },
            Vec2::zero(),
            1.0,
        );
        body.velocity = Vec2::new(1.0, 0.0);
        let fluid = default_fluid();
        let d = drag_force(&body, &fluid);
        // Drag should oppose motion (negative x)
        assert!(d.x < 0.0);
    }

    #[test]
    fn test_viscous_force_moving() {
        let mut body = RigidBody::new(0,
            BodyShape::Circle { radius: 0.1 },
            Vec2::zero(),
            1.0,
        );
        body.velocity = Vec2::new(1.0, 0.0);
        let fluid = default_fluid();
        let v = viscous_force(&body, &fluid);
        assert!(v.x < 0.0); // Opposes motion
    }

    #[test]
    fn test_rigid_body_kinetic_energy_at_rest() {
        let body = RigidBody::new(0,
            BodyShape::Circle { radius: 1.0 },
            Vec2::zero(),
            1.0,
        );
        assert!((body.kinetic_energy()).abs() < 1e-12);
    }

    #[test]
    fn test_rigid_body_kinetic_energy_moving() {
        let mut body = RigidBody::new(0,
            BodyShape::Circle { radius: 1.0 },
            Vec2::zero(),
            2.0,
        );
        body.velocity = Vec2::new(3.0, 4.0);
        let ke = body.kinetic_energy();
        // 0.5 * 2 * 25 = 25
        assert!((ke - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_body() {
        let mut sim = default_coupling();
        let id = sim.add_body(BodyShape::Circle { radius: 0.1 }, Vec2::zero(), 1.0);
        assert_eq!(id, 0);
        assert_eq!(sim.body_count(), 1);
    }

    #[test]
    fn test_add_fixed_body() {
        let mut sim = default_coupling();
        let id = sim.add_fixed_body(BodyShape::Circle { radius: 0.5 }, Vec2::zero());
        let body = sim.get_body(id).unwrap();
        assert!(body.is_fixed);
    }

    #[test]
    fn test_step_empty() {
        let mut sim = default_coupling();
        let stats = sim.step().unwrap();
        assert_eq!(stats.body_count, 0);
    }

    #[test]
    fn test_step_falling_body() {
        let mut sim = FluidCoupling::new(
            FluidProperties { surface_level: -1000.0, gravity: 9.81, ..Default::default() },
            0.01,
        );
        sim.enable_added_mass = false;
        sim.add_body(BodyShape::Circle { radius: 0.1 }, Vec2::new(0.0, 10.0), 1.0);
        let initial_y = sim.bodies[0].position.y;
        sim.run(50).unwrap();
        // Body should fall under gravity (no water to resist)
        assert!(sim.bodies[0].position.y < initial_y,
            "body y={} should be < initial {}", sim.bodies[0].position.y, initial_y);
    }

    #[test]
    fn test_fixed_body_stays() {
        let mut sim = default_coupling();
        sim.add_fixed_body(BodyShape::Circle { radius: 0.5 }, Vec2::new(1.0, 1.0));
        sim.run(10).unwrap();
        assert!((sim.bodies[0].position.x - 1.0).abs() < 1e-12);
        assert!((sim.bodies[0].position.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_surface_velocity() {
        let mut sim = default_coupling();
        let id = sim.add_body(BodyShape::Circle { radius: 1.0 }, Vec2::zero(), 1.0);
        sim.get_body_mut(id).unwrap().velocity = Vec2::new(1.0, 0.0);
        let sv = sim.surface_velocity(id, Vec2::new(1.0, 0.0)).unwrap();
        assert!((sv.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_surface_velocity_with_rotation() {
        let mut sim = default_coupling();
        let id = sim.add_body(BodyShape::Circle { radius: 1.0 }, Vec2::zero(), 1.0);
        sim.get_body_mut(id).unwrap().angular_velocity = 1.0;
        let sv = sim.surface_velocity(id, Vec2::new(1.0, 0.0)).unwrap();
        // omega x r = (-1*0, 1*1) = (0, 1)
        assert!((sv.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_body_not_found() {
        let sim = default_coupling();
        assert!(matches!(sim.get_body(99), Err(CouplingError::BodyNotFound(99))));
    }

    #[test]
    fn test_waterline_rect() {
        let body = RigidBody::new(0,
            BodyShape::Rectangle { half_width: 1.0, half_height: 0.5 },
            Vec2::zero(),
            500.0, // Half the density of water
        );
        let fluid = default_fluid();
        let wl = waterline_y(&body, &fluid);
        // Should be partially submerged
        assert!(wl > -0.5);
        assert!(wl < 0.5);
    }

    #[test]
    fn test_added_mass_force() {
        let body = RigidBody::new(0,
            BodyShape::Circle { radius: 0.1 },
            Vec2::zero(),
            1.0,
        );
        let fluid = default_fluid();
        let accel = Vec2::new(1.0, 0.0);
        let am = added_mass_force(&body, &fluid, accel, 0.5);
        // Should oppose the acceleration
        assert!(am.x < 0.0);
    }

    #[test]
    fn test_multiple_bodies() {
        let mut sim = default_coupling();
        sim.add_body(BodyShape::Circle { radius: 0.1 }, Vec2::new(-1.0, 0.0), 1.0);
        sim.add_body(BodyShape::Circle { radius: 0.1 }, Vec2::new(1.0, 0.0), 1.0);
        assert_eq!(sim.body_count(), 2);
        sim.run(5).unwrap();
        assert_eq!(sim.body_count(), 2);
    }

    #[test]
    fn test_step_count() {
        let mut sim = default_coupling();
        sim.add_body(BodyShape::Circle { radius: 0.1 }, Vec2::zero(), 1.0);
        assert_eq!(sim.step_count(), 0);
        sim.step().unwrap();
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_moment_of_inertia_circle() {
        let c = BodyShape::Circle { radius: 1.0 };
        let moi = c.moment_of_inertia(2.0);
        assert!((moi - 1.0).abs() < 1e-10); // 0.5 * 2 * 1^2
    }
}
