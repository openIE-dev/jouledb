//! Dynamic Window Approach — velocity space search, admissible velocities,
//! objective function evaluation, and trajectory scoring for local navigation.
//!
//! Pure-Rust DWA planner for differential-drive and holonomic robots with
//! configurable cost weights, obstacle clearance, and trajectory simulation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DwaError {
    InvalidParameter(String),
    NoAdmissibleVelocity,
}

impl fmt::Display for DwaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoAdmissibleVelocity => write!(f, "no admissible velocity in dynamic window"),
        }
    }
}

impl std::error::Error for DwaError {}

// ── Vec2 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }

    pub fn magnitude(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn distance_to(self, o: Vec2) -> f64 {
        ((self.x - o.x).powi(2) + (self.y - o.y).powi(2)).sqrt()
    }

    pub fn add(self, o: Vec2) -> Vec2 { Vec2 { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Vec2) -> Vec2 { Vec2 { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Vec2 { Vec2 { x: self.x * s, y: self.y * s } }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Robot State ─────────────────────────────────────────────────

/// Robot state: position, heading, linear/angular velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotState {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub v: f64,
    pub omega: f64,
}

impl RobotState {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta, v: 0.0, omega: 0.0 }
    }

    pub fn with_velocity(mut self, v: f64, omega: f64) -> Self {
        self.v = v;
        self.omega = omega;
        self
    }

    pub fn position(&self) -> Vec2 { Vec2::new(self.x, self.y) }

    /// Simulate forward one timestep with given velocity commands.
    pub fn simulate_step(&self, v: f64, omega: f64, dt: f64) -> Self {
        let theta_new = self.theta + omega * dt;
        let x_new;
        let y_new;
        if omega.abs() < 1e-10 {
            x_new = self.x + v * self.theta.cos() * dt;
            y_new = self.y + v * self.theta.sin() * dt;
        } else {
            let r = v / omega;
            x_new = self.x + r * ((self.theta + omega * dt).sin() - self.theta.sin());
            y_new = self.y - r * ((self.theta + omega * dt).cos() - self.theta.cos());
        }
        Self { x: x_new, y: y_new, theta: theta_new, v, omega }
    }
}

impl fmt::Display for RobotState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Robot(pos=({:.3}, {:.3}), theta={:.3}, v={:.3}, omega={:.3})",
            self.x, self.y, self.theta, self.v, self.omega,
        )
    }
}

// ── Obstacle ────────────────────────────────────────────────────

/// Circular obstacle for DWA clearance checking.
#[derive(Debug, Clone, Copy)]
pub struct CircleObstacle {
    pub center: Vec2,
    pub radius: f64,
}

impl CircleObstacle {
    pub fn new(x: f64, y: f64, r: f64) -> Self {
        Self { center: Vec2::new(x, y), radius: r }
    }

    pub fn distance_to_point(&self, p: Vec2) -> f64 {
        (p.distance_to(self.center) - self.radius).max(0.0)
    }
}

impl fmt::Display for CircleObstacle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CircleObs(center={}, r={:.2})", self.center, self.radius)
    }
}

// ── DWA Config ──────────────────────────────────────────────────

/// Robot kinematic and planner configuration.
#[derive(Debug, Clone)]
pub struct DwaConfig {
    // Kinematic limits
    pub max_v: f64,
    pub min_v: f64,
    pub max_omega: f64,
    pub max_accel: f64,
    pub max_alpha: f64,  // max angular acceleration

    // Simulation
    pub dt: f64,
    pub predict_time: f64,
    pub v_resolution: f64,
    pub omega_resolution: f64,

    // Costs
    pub heading_weight: f64,
    pub distance_weight: f64,
    pub velocity_weight: f64,
    pub clearance_weight: f64,

    // Safety
    pub robot_radius: f64,
    pub min_clearance: f64,
}

impl DwaConfig {
    pub fn new() -> Self {
        Self {
            max_v: 1.0,
            min_v: 0.0,
            max_omega: 1.0,
            max_accel: 0.5,
            max_alpha: 1.0,
            dt: 0.1,
            predict_time: 3.0,
            v_resolution: 0.05,
            omega_resolution: 0.05,
            heading_weight: 1.0,
            distance_weight: 1.0,
            velocity_weight: 1.0,
            clearance_weight: 1.0,
            robot_radius: 0.3,
            min_clearance: 0.1,
        }
    }

    pub fn with_max_v(mut self, v: f64) -> Self { self.max_v = v; self }
    pub fn with_min_v(mut self, v: f64) -> Self { self.min_v = v; self }
    pub fn with_max_omega(mut self, o: f64) -> Self { self.max_omega = o; self }
    pub fn with_max_accel(mut self, a: f64) -> Self { self.max_accel = a; self }
    pub fn with_max_alpha(mut self, a: f64) -> Self { self.max_alpha = a; self }
    pub fn with_dt(mut self, dt: f64) -> Self { self.dt = dt; self }
    pub fn with_predict_time(mut self, t: f64) -> Self { self.predict_time = t; self }
    pub fn with_v_resolution(mut self, r: f64) -> Self { self.v_resolution = r; self }
    pub fn with_omega_resolution(mut self, r: f64) -> Self { self.omega_resolution = r; self }
    pub fn with_heading_weight(mut self, w: f64) -> Self { self.heading_weight = w; self }
    pub fn with_distance_weight(mut self, w: f64) -> Self { self.distance_weight = w; self }
    pub fn with_velocity_weight(mut self, w: f64) -> Self { self.velocity_weight = w; self }
    pub fn with_clearance_weight(mut self, w: f64) -> Self { self.clearance_weight = w; self }
    pub fn with_robot_radius(mut self, r: f64) -> Self { self.robot_radius = r; self }
    pub fn with_min_clearance(mut self, c: f64) -> Self { self.min_clearance = c; self }
}

impl fmt::Display for DwaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DwaConfig(v=[{:.2},{:.2}], omega={:.2}, accel={:.2}, dt={:.2})",
            self.min_v, self.max_v, self.max_omega, self.max_accel, self.dt,
        )
    }
}

// ── Trajectory ──────────────────────────────────────────────────

/// A simulated trajectory with associated score.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub states: Vec<RobotState>,
    pub v_cmd: f64,
    pub omega_cmd: f64,
    pub heading_score: f64,
    pub distance_score: f64,
    pub velocity_score: f64,
    pub clearance_score: f64,
    pub total_score: f64,
}

impl Trajectory {
    pub fn end_state(&self) -> Option<&RobotState> { self.states.last() }
    pub fn length(&self) -> usize { self.states.len() }
}

impl fmt::Display for Trajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trajectory(v={:.3}, omega={:.3}, score={:.3}, len={})",
            self.v_cmd, self.omega_cmd, self.total_score, self.states.len(),
        )
    }
}

// ── DWA Result ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DwaResult {
    pub best_v: f64,
    pub best_omega: f64,
    pub best_score: f64,
    pub best_trajectory: Trajectory,
    pub candidates_evaluated: usize,
    pub candidates_admissible: usize,
}

impl fmt::Display for DwaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DwaResult(v={:.3}, omega={:.3}, score={:.3}, eval={}, admissible={})",
            self.best_v, self.best_omega, self.best_score,
            self.candidates_evaluated, self.candidates_admissible,
        )
    }
}

// ── DWA Planner ─────────────────────────────────────────────────

/// Dynamic Window Approach local planner.
pub struct DwaPlanner {
    config: DwaConfig,
    obstacles: Vec<CircleObstacle>,
}

impl DwaPlanner {
    pub fn new(config: DwaConfig) -> Self {
        Self { config, obstacles: Vec::new() }
    }

    pub fn with_obstacles(mut self, obs: Vec<CircleObstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    pub fn set_obstacles(&mut self, obs: Vec<CircleObstacle>) {
        self.obstacles = obs;
    }

    pub fn add_obstacle(&mut self, obs: CircleObstacle) {
        self.obstacles.push(obs);
    }

    /// Compute the dynamic window: range of reachable (v, omega) from current state.
    fn dynamic_window(&self, state: &RobotState) -> (f64, f64, f64, f64) {
        let v_min = (state.v - self.config.max_accel * self.config.dt).max(self.config.min_v);
        let v_max = (state.v + self.config.max_accel * self.config.dt).min(self.config.max_v);
        let omega_min = (state.omega - self.config.max_alpha * self.config.dt).max(-self.config.max_omega);
        let omega_max = (state.omega + self.config.max_alpha * self.config.dt).min(self.config.max_omega);
        (v_min, v_max, omega_min, omega_max)
    }

    /// Simulate a trajectory with constant (v, omega) for predict_time.
    fn simulate_trajectory(&self, state: &RobotState, v: f64, omega: f64) -> Vec<RobotState> {
        let steps = (self.config.predict_time / self.config.dt).ceil() as usize;
        let mut traj = Vec::with_capacity(steps + 1);
        let mut current = *state;
        traj.push(current);
        for _ in 0..steps {
            current = current.simulate_step(v, omega, self.config.dt);
            traj.push(current);
        }
        traj
    }

    /// Minimum distance from trajectory to any obstacle.
    fn trajectory_clearance(&self, traj: &[RobotState]) -> f64 {
        let mut min_d = f64::MAX;
        for st in traj {
            let pos = st.position();
            for obs in &self.obstacles {
                let d = obs.distance_to_point(pos) - self.config.robot_radius;
                if d < min_d { min_d = d; }
            }
        }
        min_d
    }

    /// Heading score: alignment of trajectory end heading with goal direction.
    fn heading_score(&self, traj: &[RobotState], goal: Vec2) -> f64 {
        let end = traj.last().unwrap();
        let goal_angle = (goal.y - end.y).atan2(goal.x - end.x);
        let diff = normalize_angle(goal_angle - end.theta).abs();
        1.0 - diff / std::f64::consts::PI
    }

    /// Distance score: how close the trajectory end is to the goal.
    fn distance_score(&self, traj: &[RobotState], goal: Vec2) -> f64 {
        let end = traj.last().unwrap();
        let d = end.position().distance_to(goal);
        1.0 / (1.0 + d)
    }

    /// Velocity score: prefer faster forward motion.
    fn velocity_score(&self, v: f64) -> f64 {
        if self.config.max_v > 0.0 { v / self.config.max_v } else { 0.0 }
    }

    /// Clearance score: distance to nearest obstacle.
    fn clearance_score(&self, traj: &[RobotState]) -> f64 {
        let c = self.trajectory_clearance(traj);
        if c < self.config.min_clearance { return 0.0; }
        (c / (c + 1.0)).min(1.0)
    }

    /// Plan the best (v, omega) from the current state toward the goal.
    pub fn plan(&self, state: &RobotState, goal: Vec2) -> Result<DwaResult, DwaError> {
        let (v_min, v_max, omega_min, omega_max) = self.dynamic_window(state);
        let mut best: Option<Trajectory> = None;
        let mut candidates_evaluated = 0usize;
        let mut candidates_admissible = 0usize;

        let mut v = v_min;
        while v <= v_max + 1e-10 {
            let mut omega = omega_min;
            while omega <= omega_max + 1e-10 {
                candidates_evaluated += 1;
                let traj = self.simulate_trajectory(state, v, omega);

                // Admissibility: trajectory must not collide
                let clearance = self.trajectory_clearance(&traj);
                if clearance < self.config.min_clearance {
                    omega += self.config.omega_resolution;
                    continue;
                }
                candidates_admissible += 1;

                let hs = self.heading_score(&traj, goal);
                let ds = self.distance_score(&traj, goal);
                let vs = self.velocity_score(v);
                let cs = self.clearance_score(&traj);

                let total = self.config.heading_weight * hs
                    + self.config.distance_weight * ds
                    + self.config.velocity_weight * vs
                    + self.config.clearance_weight * cs;

                let candidate = Trajectory {
                    states: traj,
                    v_cmd: v,
                    omega_cmd: omega,
                    heading_score: hs,
                    distance_score: ds,
                    velocity_score: vs,
                    clearance_score: cs,
                    total_score: total,
                };

                match &best {
                    Some(b) if b.total_score >= total => {}
                    _ => best = Some(candidate),
                }

                omega += self.config.omega_resolution;
            }
            v += self.config.v_resolution;
        }

        match best {
            Some(t) => Ok(DwaResult {
                best_v: t.v_cmd,
                best_omega: t.omega_cmd,
                best_score: t.total_score,
                candidates_evaluated,
                candidates_admissible,
                best_trajectory: t,
            }),
            None => Err(DwaError::NoAdmissibleVelocity),
        }
    }

    pub fn obstacle_count(&self) -> usize { self.obstacles.len() }
}

impl fmt::Display for DwaPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "DwaPlanner(obstacles={}, {})",
            self.obstacles.len(), self.config,
        )
    }
}

// ── Utility ─────────────────────────────────────────────────────

fn normalize_angle(a: f64) -> f64 {
    let mut angle = a % (2.0 * std::f64::consts::PI);
    if angle > std::f64::consts::PI { angle -= 2.0 * std::f64::consts::PI; }
    if angle < -std::f64::consts::PI { angle += 2.0 * std::f64::consts::PI; }
    angle
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn test_vec2_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(approx(a.distance_to(b), 5.0));
    }

    #[test]
    fn test_vec2_display() {
        let v = Vec2::new(1.5, 2.5);
        assert!(format!("{v}").contains("1.500"));
    }

    #[test]
    fn test_robot_state_straight() {
        let s = RobotState::new(0.0, 0.0, 0.0);
        let s2 = s.simulate_step(1.0, 0.0, 1.0);
        assert!(approx(s2.x, 1.0));
        assert!(approx(s2.y, 0.0));
    }

    #[test]
    fn test_robot_state_turn() {
        let s = RobotState::new(0.0, 0.0, 0.0);
        let s2 = s.simulate_step(0.0, 1.0, 1.0);
        assert!(approx(s2.x, 0.0));
        assert!(approx(s2.y, 0.0));
        assert!(approx(s2.theta, 1.0));
    }

    #[test]
    fn test_robot_state_display() {
        let s = RobotState::new(1.0, 2.0, 0.5);
        let text = format!("{s}");
        assert!(text.contains("Robot"));
    }

    #[test]
    fn test_circle_obstacle_distance() {
        let obs = CircleObstacle::new(5.0, 0.0, 1.0);
        let d = obs.distance_to_point(Vec2::new(0.0, 0.0));
        assert!(approx(d, 4.0));
    }

    #[test]
    fn test_circle_obstacle_display() {
        let obs = CircleObstacle::new(1.0, 2.0, 0.5);
        assert!(format!("{obs}").contains("CircleObs"));
    }

    #[test]
    fn test_dynamic_window_range() {
        let cfg = DwaConfig::new().with_max_accel(0.5).with_dt(0.1);
        let planner = DwaPlanner::new(cfg);
        let state = RobotState::new(0.0, 0.0, 0.0).with_velocity(0.5, 0.0);
        let (v_min, v_max, omega_min, omega_max) = planner.dynamic_window(&state);
        assert!(v_min >= 0.0);
        assert!(v_max <= 1.0);
        assert!(omega_min >= -1.0);
        assert!(omega_max <= 1.0);
        assert!(v_max >= v_min);
    }

    #[test]
    fn test_plan_toward_goal() {
        let cfg = DwaConfig::new()
            .with_max_v(1.0)
            .with_v_resolution(0.1)
            .with_omega_resolution(0.1);
        let planner = DwaPlanner::new(cfg);
        let state = RobotState::new(0.0, 0.0, 0.0);
        let goal = Vec2::new(10.0, 0.0);
        let result = planner.plan(&state, goal).unwrap();
        assert!(result.best_v >= 0.0, "should move forward");
        assert!(result.best_score > 0.0);
    }

    #[test]
    fn test_plan_avoid_obstacle() {
        let cfg = DwaConfig::new()
            .with_v_resolution(0.1)
            .with_omega_resolution(0.1)
            .with_predict_time(2.0)
            .with_clearance_weight(5.0);
        let obs = vec![CircleObstacle::new(2.0, 0.0, 0.5)];
        let planner = DwaPlanner::new(cfg).with_obstacles(obs);
        let state = RobotState::new(0.0, 0.0, 0.0).with_velocity(0.5, 0.0);
        let result = planner.plan(&state, Vec2::new(5.0, 0.0)).unwrap();
        // Should steer away from the obstacle
        assert!(result.best_omega.abs() > 0.0 || result.best_v < 0.5);
    }

    #[test]
    fn test_heading_score() {
        let cfg = DwaConfig::new();
        let planner = DwaPlanner::new(cfg);
        let traj = vec![RobotState::new(0.0, 0.0, 0.0)];
        let score = planner.heading_score(&traj, Vec2::new(10.0, 0.0));
        assert!(score > 0.9, "straight ahead should score high: {score}");
    }

    #[test]
    fn test_velocity_score() {
        let cfg = DwaConfig::new().with_max_v(2.0);
        let planner = DwaPlanner::new(cfg);
        assert!(approx(planner.velocity_score(2.0), 1.0));
        assert!(approx(planner.velocity_score(1.0), 0.5));
    }

    #[test]
    fn test_normalize_angle() {
        assert!(approx(normalize_angle(0.0), 0.0));
        assert!(approx(normalize_angle(4.0 * std::f64::consts::PI), 0.0));
        assert!(approx(normalize_angle(std::f64::consts::PI + 0.1), -std::f64::consts::PI + 0.1));
    }

    #[test]
    fn test_config_display() {
        let cfg = DwaConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DwaConfig"));
    }

    #[test]
    fn test_trajectory_display() {
        let t = Trajectory {
            states: vec![RobotState::new(0.0, 0.0, 0.0)],
            v_cmd: 0.5,
            omega_cmd: 0.1,
            heading_score: 0.9,
            distance_score: 0.5,
            velocity_score: 0.5,
            clearance_score: 1.0,
            total_score: 2.9,
        };
        assert!(format!("{t}").contains("Trajectory"));
    }

    #[test]
    fn test_result_display() {
        let r = DwaResult {
            best_v: 0.5,
            best_omega: 0.1,
            best_score: 3.0,
            best_trajectory: Trajectory {
                states: vec![RobotState::new(0.0, 0.0, 0.0)],
                v_cmd: 0.5, omega_cmd: 0.1,
                heading_score: 1.0, distance_score: 1.0,
                velocity_score: 0.5, clearance_score: 0.5,
                total_score: 3.0,
            },
            candidates_evaluated: 100,
            candidates_admissible: 80,
        };
        assert!(format!("{r}").contains("admissible=80"));
    }

    #[test]
    fn test_planner_display() {
        let planner = DwaPlanner::new(DwaConfig::new());
        assert!(format!("{planner}").contains("DwaPlanner"));
    }

    #[test]
    fn test_error_display() {
        let e = DwaError::NoAdmissibleVelocity;
        assert_eq!(format!("{e}"), "no admissible velocity in dynamic window");
    }
}
