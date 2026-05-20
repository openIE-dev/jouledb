//! Push Manipulation — Quasi-static pushing models, contact mechanics, push
//! planning, and slider dynamics for non-prehensile manipulation.
//!
//! Implements Mason's voting theorem for stable push direction analysis,
//! limit surface models, and quasi-static slider dynamics.
//! All algorithms are std-only, using `f64` throughout.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Push manipulation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PushError {
    /// Invalid physical parameter.
    InvalidParameter(String),
    /// Push direction infeasible.
    Infeasible(String),
    /// Simulation diverged.
    Diverged(String),
    /// Contact lost during push.
    ContactLost,
}

impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(m) => write!(f, "invalid parameter: {m}"),
            Self::Infeasible(m) => write!(f, "infeasible push: {m}"),
            Self::Diverged(m) => write!(f, "simulation diverged: {m}"),
            Self::ContactLost => write!(f, "contact lost during push"),
        }
    }
}

impl std::error::Error for PushError {}

// ── 2D Vector ───────────────────────────────────────────────────

/// 2D vector for planar push analysis.
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

    /// 2D cross product (scalar).
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-12 {
            None
        } else {
            Some(Self { x: self.x / n, y: self.y / n })
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn rotate(self, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self { x: c * self.x - s * self.y, y: s * self.x + c * self.y }
    }

    /// Perpendicular vector (90 deg counter-clockwise).
    pub fn perp(self) -> Self {
        Self { x: -self.y, y: self.x }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4})", self.x, self.y)
    }
}

// ── Slider State ────────────────────────────────────────────────

/// State of a planar rigid-body slider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderState {
    /// Centre of mass position.
    pub position: Vec2,
    /// Orientation angle (radians).
    pub theta: f64,
    /// Linear velocity.
    pub velocity: Vec2,
    /// Angular velocity (rad/s).
    pub omega: f64,
}

impl SliderState {
    pub fn new(position: Vec2, theta: f64) -> Self {
        Self { position, theta, velocity: Vec2::zero(), omega: 0.0 }
    }

    pub fn with_velocity(mut self, vel: Vec2, omega: f64) -> Self {
        self.velocity = vel;
        self.omega = omega;
        self
    }

    /// Transform a body-frame point to world frame.
    pub fn body_to_world(&self, local: Vec2) -> Vec2 {
        self.position.add(local.rotate(self.theta))
    }

    /// World-frame normal at a contact point (outward from slider).
    pub fn outward_normal(&self, contact_local: Vec2) -> Vec2 {
        contact_local.rotate(self.theta).normalized().unwrap_or(Vec2::new(1.0, 0.0))
    }
}

impl fmt::Display for SliderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Slider(pos={}, theta={:.4}, vel={}, omega={:.4})",
            self.position, self.theta, self.velocity, self.omega
        )
    }
}

// ── Limit Surface ───────────────────────────────────────────────

/// Ellipsoidal limit surface model for quasi-static pushing.
/// The limit surface maps applied wrenches to slider twist velocities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimitSurface {
    /// Maximum friction force (N).
    pub f_max: f64,
    /// Maximum friction torque (N·m).
    pub tau_max: f64,
    /// Mass of the slider (kg).
    pub mass: f64,
    /// Friction coefficient.
    pub mu: f64,
}

impl LimitSurface {
    pub fn new(mass: f64, mu: f64, support_radius: f64) -> Result<Self, PushError> {
        if mass <= 0.0 {
            return Err(PushError::InvalidParameter("mass must be positive".into()));
        }
        if mu <= 0.0 {
            return Err(PushError::InvalidParameter("friction must be positive".into()));
        }
        if support_radius <= 0.0 {
            return Err(PushError::InvalidParameter("support radius must be positive".into()));
        }
        let f_max = mu * mass * 9.81;
        // For a uniform disk, tau_max = (2/3) * mu * m * g * R
        let tau_max = (2.0 / 3.0) * mu * mass * 9.81 * support_radius;
        Ok(Self { f_max, tau_max, mass, mu })
    }

    /// Map an applied wrench (fx, fy, tau) to the generalised velocity (vx, vy, omega).
    /// Uses the ellipsoidal limit surface: v_i = f_i / (f_max^2), omega = tau / (tau_max^2).
    pub fn wrench_to_twist(&self, fx: f64, fy: f64, tau: f64) -> (f64, f64, f64) {
        let vx = fx / (self.f_max * self.f_max);
        let vy = fy / (self.f_max * self.f_max);
        let omega = tau / (self.tau_max * self.tau_max);
        (vx, vy, omega)
    }

    /// Check if a wrench is within the limit surface (feasible).
    pub fn is_feasible(&self, fx: f64, fy: f64, tau: f64) -> bool {
        let nf = (fx * fx + fy * fy) / (self.f_max * self.f_max);
        let nt = (tau * tau) / (self.tau_max * self.tau_max);
        nf + nt <= 1.0
    }

    /// Characteristic length c = tau_max / f_max.
    pub fn characteristic_length(&self) -> f64 {
        self.tau_max / self.f_max
    }
}

impl fmt::Display for LimitSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LimitSurface(f_max={:.3}, tau_max={:.3}, mu={:.3})",
            self.f_max, self.tau_max, self.mu
        )
    }
}

// ── Contact Mechanics ───────────────────────────────────────────

/// Point contact model for pushing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PushContact {
    /// Contact position in slider body frame.
    pub local_position: Vec2,
    /// Outward surface normal in body frame.
    pub local_normal: Vec2,
    /// Friction coefficient at contact.
    pub mu: f64,
}

impl PushContact {
    pub fn new(local_position: Vec2, local_normal: Vec2, mu: f64) -> Result<Self, PushError> {
        let local_normal = local_normal
            .normalized()
            .ok_or_else(|| PushError::InvalidParameter("zero normal".into()))?;
        if mu < 0.0 {
            return Err(PushError::InvalidParameter("negative friction".into()));
        }
        Ok(Self { local_position, local_normal, mu })
    }

    /// Compute the friction cone edges in the body frame.
    /// Returns (left_edge, right_edge) directions.
    pub fn friction_cone_edges(&self) -> (Vec2, Vec2) {
        let push_dir = self.local_normal.scale(-1.0);
        let half_angle = self.mu.atan();
        let left = push_dir.rotate(half_angle);
        let right = push_dir.rotate(-half_angle);
        (left, right)
    }

    /// Wrench produced by a unit force in direction `push_dir` at this contact.
    pub fn contact_wrench(&self, push_dir: Vec2) -> (f64, f64, f64) {
        let fx = push_dir.x;
        let fy = push_dir.y;
        let tau = self.local_position.cross(push_dir);
        (fx, fy, tau)
    }
}

impl fmt::Display for PushContact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PushContact(pos={}, n={}, mu={:.3})",
            self.local_position, self.local_normal, self.mu
        )
    }
}

// ── Voting Theorem ──────────────────────────────────────────────

/// Determine the push mode (stable left, stable right, or unstable)
/// using Mason's voting theorem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushMode {
    /// Slider rotates clockwise (CW) — stable.
    StableRight,
    /// Slider rotates counter-clockwise (CCW) — stable.
    StableLeft,
    /// Pure translation, no rotation.
    PureTranslation,
}

impl fmt::Display for PushMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StableRight => write!(f, "StableRight (CW)"),
            Self::StableLeft => write!(f, "StableLeft (CCW)"),
            Self::PureTranslation => write!(f, "PureTranslation"),
        }
    }
}

/// Classify the push mode from the centre of rotation position relative
/// to the contact point. Uses the voting theorem: all support points
/// vote for CW or CCW depending on which side of the push line they lie.
pub fn classify_push_mode(
    contact: &PushContact,
    push_direction: Vec2,
    support_points: &[Vec2],
) -> PushMode {
    let push_dir = match push_direction.normalized() {
        Some(d) => d,
        None => return PushMode::PureTranslation,
    };
    let push_line_normal = push_dir.perp();
    let mut cw_votes: i64 = 0;
    let mut ccw_votes: i64 = 0;

    for sp in support_points {
        let relative = sp.sub(contact.local_position);
        let side = relative.dot(push_line_normal);
        if side > 1e-10 {
            ccw_votes += 1;
        } else if side < -1e-10 {
            cw_votes += 1;
        }
    }

    if cw_votes > ccw_votes {
        PushMode::StableRight
    } else if ccw_votes > cw_votes {
        PushMode::StableLeft
    } else {
        PushMode::PureTranslation
    }
}

// ── Quasi-Static Push Simulator ─────────────────────────────────

/// Configuration for the push simulator.
#[derive(Debug, Clone)]
pub struct PushSimConfig {
    /// Time step (seconds).
    pub dt: f64,
    /// Maximum simulation steps.
    pub max_steps: usize,
    /// Velocity damping factor (0..1).
    pub damping: f64,
    /// Object support radius for limit surface.
    pub support_radius: f64,
}

impl Default for PushSimConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            max_steps: 1000,
            damping: 0.95,
            support_radius: 0.05,
        }
    }
}

impl PushSimConfig {
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d;
        self
    }

    pub fn with_support_radius(mut self, r: f64) -> Self {
        self.support_radius = r;
        self
    }
}

/// Quasi-static push simulator.
#[derive(Debug, Clone)]
pub struct PushSimulator {
    config: PushSimConfig,
    limit_surface: LimitSurface,
    state: SliderState,
    trajectory: Vec<SliderState>,
}

impl PushSimulator {
    pub fn new(
        config: PushSimConfig,
        mass: f64,
        mu: f64,
        initial_state: SliderState,
    ) -> Result<Self, PushError> {
        let ls = LimitSurface::new(mass, mu, config.support_radius)?;
        Ok(Self {
            config,
            limit_surface: ls,
            state: initial_state,
            trajectory: vec![initial_state],
        })
    }

    pub fn state(&self) -> &SliderState {
        &self.state
    }

    pub fn trajectory(&self) -> &[SliderState] {
        &self.trajectory
    }

    /// Step the simulation with a given push force at a contact point.
    pub fn step(
        &mut self,
        contact: &PushContact,
        push_force: f64,
    ) -> Result<&SliderState, PushError> {
        if push_force < 0.0 {
            return Err(PushError::InvalidParameter("push force must be non-negative".into()));
        }
        // Push direction is opposite to the outward normal
        let push_dir = contact.local_normal.scale(-1.0);
        let (fx, fy, tau) = contact.contact_wrench(push_dir.scale(push_force));

        // Quasi-static: velocity proportional to wrench via limit surface
        let (vx, vy, omega) = self.limit_surface.wrench_to_twist(fx, fy, tau);

        // Scale velocities for the time step
        let scaled_vx = vx * self.config.dt;
        let scaled_vy = vy * self.config.dt;
        let scaled_omega = omega * self.config.dt;

        // Integrate position
        let new_pos = self.state.position.add(
            Vec2::new(scaled_vx, scaled_vy).rotate(self.state.theta),
        );
        let new_theta = self.state.theta + scaled_omega;

        // Check for divergence
        if new_pos.norm() > 1e6 || new_theta.abs() > 1e6 {
            return Err(PushError::Diverged("state values too large".into()));
        }

        self.state = SliderState {
            position: new_pos,
            theta: new_theta,
            velocity: Vec2::new(vx, vy).scale(self.config.damping),
            omega: omega * self.config.damping,
        };
        self.trajectory.push(self.state);
        Ok(&self.state)
    }

    /// Run the simulation for a fixed number of steps with constant force.
    pub fn run(
        &mut self,
        contact: &PushContact,
        push_force: f64,
        steps: usize,
    ) -> Result<(), PushError> {
        let max = steps.min(self.config.max_steps);
        for _ in 0..max {
            self.step(contact, push_force)?;
        }
        Ok(())
    }

    /// Total displacement from initial position.
    pub fn total_displacement(&self) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }
        self.trajectory[0].position.sub(self.state.position).norm()
    }

    /// Total rotation from initial orientation.
    pub fn total_rotation(&self) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }
        (self.state.theta - self.trajectory[0].theta).abs()
    }
}

impl fmt::Display for PushSimulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PushSim(state={}, steps={})",
            self.state,
            self.trajectory.len()
        )
    }
}

// ── Push Planner ────────────────────────────────────────────────

/// A simple push plan: a sequence of (contact, force, steps) actions.
#[derive(Debug, Clone)]
pub struct PushPlan {
    pub actions: Vec<(PushContact, f64, usize)>,
}

impl PushPlan {
    pub fn new() -> Self {
        Self { actions: Vec::new() }
    }

    pub fn add_action(&mut self, contact: PushContact, force: f64, steps: usize) {
        self.actions.push((contact, force, steps));
    }

    pub fn total_steps(&self) -> usize {
        self.actions.iter().map(|(_, _, s)| *s).sum()
    }

    pub fn num_actions(&self) -> usize {
        self.actions.len()
    }
}

impl fmt::Display for PushPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PushPlan(actions={}, total_steps={})", self.num_actions(), self.total_steps())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dot(b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(r.x.abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_perp() {
        let v = Vec2::new(1.0, 0.0);
        let p = v.perp();
        assert!((p.x - 0.0).abs() < 1e-10);
        assert!((p.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_limit_surface_creation() {
        let ls = LimitSurface::new(1.0, 0.3, 0.05).unwrap();
        assert!(ls.f_max > 0.0);
        assert!(ls.tau_max > 0.0);
    }

    #[test]
    fn test_limit_surface_invalid_mass() {
        assert!(LimitSurface::new(0.0, 0.3, 0.05).is_err());
    }

    #[test]
    fn test_limit_surface_feasibility() {
        let ls = LimitSurface::new(1.0, 0.3, 0.05).unwrap();
        assert!(ls.is_feasible(0.0, 0.0, 0.0));
        assert!(!ls.is_feasible(ls.f_max * 2.0, 0.0, 0.0));
    }

    #[test]
    fn test_limit_surface_characteristic_length() {
        let ls = LimitSurface::new(1.0, 0.3, 0.05).unwrap();
        let c = ls.characteristic_length();
        assert!(c > 0.0);
    }

    #[test]
    fn test_push_contact_creation() {
        let pc = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        assert!((pc.local_normal.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_push_contact_friction_cone() {
        let pc = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        let (left, right) = pc.friction_cone_edges();
        assert!((left.norm() - 1.0).abs() < 0.01);
        assert!((right.norm() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_push_mode_classification() {
        let contact = PushContact::new(
            Vec2::new(0.05, 0.0),
            Vec2::new(1.0, 0.0),
            0.3,
        )
        .unwrap();
        let push_dir = Vec2::new(-1.0, 0.0);
        // Support points symmetric about push line => PureTranslation
        let supports = vec![
            Vec2::new(0.0, 0.05),
            Vec2::new(0.0, -0.05),
        ];
        let mode = classify_push_mode(&contact, push_dir, &supports);
        assert_eq!(mode, PushMode::PureTranslation);
    }

    #[test]
    fn test_push_mode_asymmetric() {
        let contact = PushContact::new(
            Vec2::new(0.05, 0.0),
            Vec2::new(1.0, 0.0),
            0.3,
        )
        .unwrap();
        let push_dir = Vec2::new(-1.0, 0.0);
        // All support points on one side => rotation
        let supports = vec![
            Vec2::new(0.0, 0.05),
            Vec2::new(-0.05, 0.05),
            Vec2::new(0.03, 0.05),
        ];
        let mode = classify_push_mode(&contact, push_dir, &supports);
        assert_ne!(mode, PushMode::PureTranslation);
    }

    #[test]
    fn test_slider_body_to_world() {
        let state = SliderState::new(Vec2::new(1.0, 0.0), 0.0);
        let world = state.body_to_world(Vec2::new(0.5, 0.0));
        assert!((world.x - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_simulator_creation() {
        let state = SliderState::new(Vec2::zero(), 0.0);
        let sim = PushSimulator::new(PushSimConfig::default(), 0.5, 0.3, state).unwrap();
        assert_eq!(sim.trajectory().len(), 1);
    }

    #[test]
    fn test_simulator_step() {
        let state = SliderState::new(Vec2::zero(), 0.0);
        let mut sim = PushSimulator::new(PushSimConfig::default(), 0.5, 0.3, state).unwrap();
        let contact = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        sim.step(&contact, 1.0).unwrap();
        assert_eq!(sim.trajectory().len(), 2);
    }

    #[test]
    fn test_simulator_run() {
        let state = SliderState::new(Vec2::zero(), 0.0);
        let mut sim = PushSimulator::new(PushSimConfig::default(), 0.5, 0.3, state).unwrap();
        let contact = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        sim.run(&contact, 1.0, 10).unwrap();
        assert!(sim.total_displacement() > 0.0);
    }

    #[test]
    fn test_simulator_negative_force() {
        let state = SliderState::new(Vec2::zero(), 0.0);
        let mut sim = PushSimulator::new(PushSimConfig::default(), 0.5, 0.3, state).unwrap();
        let contact = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        assert!(sim.step(&contact, -1.0).is_err());
    }

    #[test]
    fn test_push_plan() {
        let mut plan = PushPlan::new();
        let contact = PushContact::new(Vec2::new(0.05, 0.0), Vec2::new(1.0, 0.0), 0.3).unwrap();
        plan.add_action(contact, 1.0, 50);
        assert_eq!(plan.total_steps(), 50);
        assert_eq!(plan.num_actions(), 1);
    }

    #[test]
    fn test_push_sim_display() {
        let state = SliderState::new(Vec2::zero(), 0.0);
        let sim = PushSimulator::new(PushSimConfig::default(), 0.5, 0.3, state).unwrap();
        let s = format!("{sim}");
        assert!(s.contains("PushSim"));
    }
}
