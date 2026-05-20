//! Waypoint Following — Pure pursuit, Stanley controller, carrot-chasing,
//! and lookahead distance tuning for path-tracking control.
//!
//! These controllers take a reference path (sequence of waypoints) and the
//! robot's current pose and velocity, then output steering commands (curvature
//! or angular velocity) to track the path. All controllers are suitable for
//! differential-drive and Ackermann-steered vehicles.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Waypoint following errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FollowError {
    /// Path is empty or has too few points.
    EmptyPath,
    /// Could not find a valid lookahead point.
    NoLookahead,
    /// Invalid parameter.
    InvalidParam(String),
    /// Robot has reached the end of the path.
    PathComplete,
}

impl fmt::Display for FollowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPath => write!(f, "path is empty"),
            Self::NoLookahead => write!(f, "no valid lookahead point"),
            Self::InvalidParam(m) => write!(f, "invalid parameter: {m}"),
            Self::PathComplete => write!(f, "path complete"),
        }
    }
}

impl std::error::Error for FollowError {}

// ── Waypoint ────────────────────────────────────────────────────

/// A 2-D waypoint with optional speed target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub x: f64,
    pub y: f64,
    pub target_speed: Option<f64>,
}

impl Waypoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, target_speed: None }
    }

    pub fn with_speed(mut self, speed: f64) -> Self {
        self.target_speed = Some(speed);
        self
    }

    pub fn distance_to(&self, other: &Waypoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for Waypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target_speed {
            Some(s) => write!(f, "WP({:.2}, {:.2}, v={:.2})", self.x, self.y, s),
            None => write!(f, "WP({:.2}, {:.2})", self.x, self.y),
        }
    }
}

// ── Vehicle Pose ────────────────────────────────────────────────

/// Vehicle state: position, heading, and speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleState {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub speed: f64,
}

impl VehicleState {
    pub fn new(x: f64, y: f64, theta: f64, speed: f64) -> Self {
        Self { x, y, theta, speed }
    }

    pub fn distance_to_wp(&self, wp: &Waypoint) -> f64 {
        let dx = self.x - wp.x;
        let dy = self.y - wp.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for VehicleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vehicle({:.2}, {:.2}, θ={:.3}, v={:.2})",
            self.x, self.y, self.theta, self.speed
        )
    }
}

// ── Steering Output ─────────────────────────────────────────────

/// Output from a path-tracking controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteeringOutput {
    /// Desired curvature (1/m). Positive = left turn.
    pub curvature: f64,
    /// Cross-track error (m). Positive = right of path.
    pub cross_track_error: f64,
    /// Index of the closest waypoint on the path.
    pub closest_idx: usize,
}

impl SteeringOutput {
    /// Convert curvature to angular velocity given current speed.
    pub fn angular_velocity(&self, speed: f64) -> f64 {
        self.curvature * speed
    }

    /// Convert curvature to Ackermann steering angle given wheelbase.
    pub fn steering_angle(&self, wheelbase: f64) -> f64 {
        (self.curvature * wheelbase).atan()
    }
}

impl fmt::Display for SteeringOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Steer(κ={:.4}, cte={:.3}, idx={})",
            self.curvature, self.cross_track_error, self.closest_idx
        )
    }
}

// ── Lookahead Tuner ─────────────────────────────────────────────

/// Adaptive lookahead distance based on vehicle speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookaheadTuner {
    pub min_dist: f64,
    pub max_dist: f64,
    pub gain: f64,
}

impl LookaheadTuner {
    pub fn new(min_dist: f64, max_dist: f64, gain: f64) -> Self {
        Self {
            min_dist: min_dist.max(0.1),
            max_dist: max_dist.max(min_dist),
            gain: gain.max(0.0),
        }
    }

    /// Compute lookahead distance for a given speed.
    pub fn compute(&self, speed: f64) -> f64 {
        (self.min_dist + self.gain * speed.abs()).min(self.max_dist)
    }
}

impl fmt::Display for LookaheadTuner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lookahead(min={:.2}, max={:.2}, k={:.2})",
            self.min_dist, self.max_dist, self.gain
        )
    }
}

// ── Pure Pursuit ────────────────────────────────────────────────

/// Pure Pursuit path-tracking controller.
///
/// Computes curvature to drive the vehicle towards a lookahead point on the
/// path. The lookahead distance is optionally speed-adaptive.
#[derive(Debug, Clone)]
pub struct PurePursuit {
    lookahead: LookaheadTuner,
    goal_tolerance: f64,
}

impl PurePursuit {
    pub fn new(lookahead_dist: f64) -> Self {
        Self {
            lookahead: LookaheadTuner::new(lookahead_dist, lookahead_dist * 3.0, 0.5),
            goal_tolerance: 0.5,
        }
    }

    pub fn with_lookahead_tuner(mut self, tuner: LookaheadTuner) -> Self {
        self.lookahead = tuner;
        self
    }

    pub fn with_goal_tolerance(mut self, tol: f64) -> Self {
        self.goal_tolerance = tol.max(0.01);
        self
    }

    /// Compute steering for the given vehicle state and path.
    pub fn compute(
        &self,
        state: VehicleState,
        path: &[Waypoint],
    ) -> Result<SteeringOutput, FollowError> {
        if path.is_empty() {
            return Err(FollowError::EmptyPath);
        }

        let closest = self.find_closest(state, path);
        let last_wp = path.last().unwrap();
        if state.distance_to_wp(last_wp) < self.goal_tolerance {
            return Err(FollowError::PathComplete);
        }

        let ld = self.lookahead.compute(state.speed);
        let lookahead_wp = self.find_lookahead(state, path, closest, ld)?;

        let cte = self.cross_track_error(state, path, closest);

        // Transform lookahead point to vehicle frame.
        let dx = lookahead_wp.x - state.x;
        let dy = lookahead_wp.y - state.y;
        let local_x = dx * state.theta.cos() + dy * state.theta.sin();
        let local_y = -dx * state.theta.sin() + dy * state.theta.cos();

        let l_sq = local_x * local_x + local_y * local_y;
        let curvature = if l_sq > 1e-9 { 2.0 * local_y / l_sq } else { 0.0 };

        Ok(SteeringOutput { curvature, cross_track_error: cte, closest_idx: closest })
    }

    fn find_closest(&self, state: VehicleState, path: &[Waypoint]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, wp) in path.iter().enumerate() {
            let d = state.distance_to_wp(wp);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn find_lookahead(
        &self,
        state: VehicleState,
        path: &[Waypoint],
        start_idx: usize,
        ld: f64,
    ) -> Result<Waypoint, FollowError> {
        // Walk forward from closest, find first point beyond lookahead distance.
        for i in start_idx..path.len() {
            if state.distance_to_wp(&path[i]) >= ld {
                return Ok(path[i]);
            }
        }
        // If nothing is far enough, use the last waypoint.
        if path.len() > start_idx {
            Ok(*path.last().unwrap())
        } else {
            Err(FollowError::NoLookahead)
        }
    }

    fn cross_track_error(&self, state: VehicleState, path: &[Waypoint], idx: usize) -> f64 {
        if path.len() < 2 {
            return state.distance_to_wp(&path[0]);
        }
        let i = if idx == 0 { 0 } else { idx - 1 };
        let j = (i + 1).min(path.len() - 1);
        let ax = path[i].x;
        let ay = path[i].y;
        let bx = path[j].x;
        let by = path[j].y;
        let seg_len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
        if seg_len < 1e-9 {
            return state.distance_to_wp(&path[i]);
        }
        let cross = (state.x - ax) * (by - ay) - (state.y - ay) * (bx - ax);
        cross / seg_len
    }
}

impl fmt::Display for PurePursuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PurePursuit({})", self.lookahead)
    }
}

// ── Stanley Controller ──────────────────────────────────────────

/// Stanley path-tracking controller.
///
/// Combines heading error correction with cross-track error correction via
/// the arctan(k * cte / speed) term. Commonly used on Ackermann vehicles.
#[derive(Debug, Clone)]
pub struct StanleyController {
    gain_k: f64,
    soft_speed: f64,
    max_steer: f64,
    goal_tolerance: f64,
}

impl StanleyController {
    pub fn new(gain_k: f64) -> Self {
        Self {
            gain_k,
            soft_speed: 0.5,
            max_steer: 1.0,
            goal_tolerance: 0.5,
        }
    }

    pub fn with_soft_speed(mut self, s: f64) -> Self {
        self.soft_speed = s.max(0.01);
        self
    }

    pub fn with_max_steer(mut self, m: f64) -> Self {
        self.max_steer = m.max(0.01);
        self
    }

    pub fn with_goal_tolerance(mut self, tol: f64) -> Self {
        self.goal_tolerance = tol.max(0.01);
        self
    }

    /// Compute steering output.
    pub fn compute(
        &self,
        state: VehicleState,
        path: &[Waypoint],
    ) -> Result<SteeringOutput, FollowError> {
        if path.len() < 2 {
            return Err(FollowError::EmptyPath);
        }

        let last_wp = path.last().unwrap();
        if state.distance_to_wp(last_wp) < self.goal_tolerance {
            return Err(FollowError::PathComplete);
        }

        let closest = self.find_closest(state, path);
        let (cte, path_heading) = self.compute_cte_and_heading(state, path, closest);

        // Heading error.
        let heading_err = Self::normalize_angle(path_heading - state.theta);

        // Cross-track correction.
        let cte_correction = (self.gain_k * cte / (state.speed.abs() + self.soft_speed)).atan();

        let steer = (heading_err + cte_correction).clamp(-self.max_steer, self.max_steer);

        // Approximate curvature from steering angle assuming unit wheelbase.
        let curvature = steer.tan();

        Ok(SteeringOutput { curvature, cross_track_error: cte, closest_idx: closest })
    }

    fn find_closest(&self, state: VehicleState, path: &[Waypoint]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, wp) in path.iter().enumerate() {
            let d = state.distance_to_wp(wp);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn compute_cte_and_heading(
        &self,
        state: VehicleState,
        path: &[Waypoint],
        idx: usize,
    ) -> (f64, f64) {
        let j = (idx + 1).min(path.len() - 1);
        let i = if j > 0 { j - 1 } else { 0 };
        let dx = path[j].x - path[i].x;
        let dy = path[j].y - path[i].y;
        let seg_heading = dy.atan2(dx);
        let seg_len = (dx * dx + dy * dy).sqrt();
        let cte = if seg_len > 1e-9 {
            let cross = (state.x - path[i].x) * dy - (state.y - path[i].y) * dx;
            cross / seg_len
        } else {
            state.distance_to_wp(&path[i])
        };
        (cte, seg_heading)
    }

    fn normalize_angle(a: f64) -> f64 {
        let mut a = a % (2.0 * std::f64::consts::PI);
        if a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        } else if a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        a
    }
}

impl fmt::Display for StanleyController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stanley(k={:.2}, max_steer={:.2})", self.gain_k, self.max_steer)
    }
}

// ── Carrot Chasing ──────────────────────────────────────────────

/// Simple carrot-chasing controller.
///
/// Projects a "carrot" point ahead on the path at a fixed distance and
/// steers toward it. Simpler than pure pursuit but less accurate on curves.
#[derive(Debug, Clone)]
pub struct CarrotChaser {
    carrot_dist: f64,
    goal_tolerance: f64,
    angular_gain: f64,
}

impl CarrotChaser {
    pub fn new(carrot_dist: f64) -> Self {
        Self { carrot_dist: carrot_dist.max(0.1), goal_tolerance: 0.5, angular_gain: 1.0 }
    }

    pub fn with_goal_tolerance(mut self, tol: f64) -> Self {
        self.goal_tolerance = tol.max(0.01);
        self
    }

    pub fn with_angular_gain(mut self, k: f64) -> Self {
        self.angular_gain = k.max(0.0);
        self
    }

    /// Compute angular velocity to chase the carrot point.
    pub fn compute(&self, state: VehicleState, path: &[Waypoint]) -> Result<f64, FollowError> {
        if path.is_empty() {
            return Err(FollowError::EmptyPath);
        }
        let last = path.last().unwrap();
        if state.distance_to_wp(last) < self.goal_tolerance {
            return Err(FollowError::PathComplete);
        }

        // Find carrot along accumulated path distance from closest point.
        let closest = self.find_closest(state, path);
        let carrot = self.find_carrot(state, path, closest);

        let dx = carrot.x - state.x;
        let dy = carrot.y - state.y;
        let angle_to_carrot = dy.atan2(dx);
        let err = Self::normalize_angle(angle_to_carrot - state.theta);

        Ok(self.angular_gain * err)
    }

    fn find_closest(&self, state: VehicleState, path: &[Waypoint]) -> usize {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, wp) in path.iter().enumerate() {
            let d = state.distance_to_wp(wp);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    fn find_carrot(&self, _state: VehicleState, path: &[Waypoint], start: usize) -> Waypoint {
        let mut accum = 0.0;
        for i in start..path.len().saturating_sub(1) {
            let seg = path[i].distance_to(&path[i + 1]);
            accum += seg;
            if accum >= self.carrot_dist {
                return path[i + 1];
            }
        }
        *path.last().unwrap()
    }

    fn normalize_angle(a: f64) -> f64 {
        let mut a = a % (2.0 * std::f64::consts::PI);
        if a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        } else if a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        a
    }
}

impl fmt::Display for CarrotChaser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CarrotChaser(d={:.2}, k={:.2})", self.carrot_dist, self.angular_gain)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_path() -> Vec<Waypoint> {
        (0..20).map(|i| Waypoint::new(i as f64, 0.0)).collect()
    }

    fn curved_path() -> Vec<Waypoint> {
        (0..20)
            .map(|i| {
                let t = i as f64 * 0.2;
                Waypoint::new(t.cos() * 5.0, t.sin() * 5.0)
            })
            .collect()
    }

    #[test]
    fn test_waypoint_display() {
        let wp = Waypoint::new(1.5, 2.5).with_speed(3.0);
        let s = format!("{wp}");
        assert!(s.contains("v=3.00"));
    }

    #[test]
    fn test_waypoint_distance() {
        let a = Waypoint::new(0.0, 0.0);
        let b = Waypoint::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_vehicle_state_display() {
        let v = VehicleState::new(1.0, 2.0, 0.5, 3.0);
        assert!(format!("{v}").contains("Vehicle"));
    }

    #[test]
    fn test_lookahead_tuner() {
        let tuner = LookaheadTuner::new(1.0, 5.0, 0.5);
        assert_eq!(tuner.compute(0.0), 1.0);
        assert_eq!(tuner.compute(8.0), 5.0);
        assert!((tuner.compute(2.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_pure_pursuit_straight() {
        let pp = PurePursuit::new(2.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        let path = straight_path();
        let out = pp.compute(state, &path).unwrap();
        assert!(out.curvature.abs() < 0.1);
    }

    #[test]
    fn test_pure_pursuit_off_path() {
        let pp = PurePursuit::new(2.0);
        let state = VehicleState::new(0.0, 2.0, 0.0, 1.0);
        let path = straight_path();
        let out = pp.compute(state, &path).unwrap();
        assert!(out.cross_track_error.abs() > 0.1);
    }

    #[test]
    fn test_pure_pursuit_empty() {
        let pp = PurePursuit::new(2.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        assert!(pp.compute(state, &[]).is_err());
    }

    #[test]
    fn test_pure_pursuit_at_goal() {
        let pp = PurePursuit::new(2.0).with_goal_tolerance(1.0);
        let path = vec![Waypoint::new(0.0, 0.0), Waypoint::new(1.0, 0.0)];
        let state = VehicleState::new(1.0, 0.0, 0.0, 0.0);
        let result = pp.compute(state, &path);
        assert_eq!(result, Err(FollowError::PathComplete));
    }

    #[test]
    fn test_stanley_straight() {
        let sc = StanleyController::new(1.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        let path = straight_path();
        let out = sc.compute(state, &path).unwrap();
        assert!(out.curvature.abs() < 0.5);
    }

    #[test]
    fn test_stanley_cte() {
        let sc = StanleyController::new(2.0);
        let state = VehicleState::new(5.0, 3.0, 0.0, 1.0);
        let path = straight_path();
        let out = sc.compute(state, &path).unwrap();
        assert!(out.cross_track_error.abs() > 0.5);
    }

    #[test]
    fn test_stanley_empty() {
        let sc = StanleyController::new(1.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        assert!(sc.compute(state, &[Waypoint::new(0.0, 0.0)]).is_err());
    }

    #[test]
    fn test_carrot_chaser_straight() {
        let cc = CarrotChaser::new(3.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        let path = straight_path();
        let omega = cc.compute(state, &path).unwrap();
        assert!(omega.abs() < 0.1);
    }

    #[test]
    fn test_carrot_chaser_curve() {
        let cc = CarrotChaser::new(2.0);
        let state = VehicleState::new(5.0, 0.0, std::f64::consts::FRAC_PI_2, 1.0);
        let path = curved_path();
        let omega = cc.compute(state, &path).unwrap();
        assert!(omega.abs() > 0.0);
    }

    #[test]
    fn test_carrot_chaser_empty() {
        let cc = CarrotChaser::new(2.0);
        let state = VehicleState::new(0.0, 0.0, 0.0, 1.0);
        assert!(cc.compute(state, &[]).is_err());
    }

    #[test]
    fn test_steering_output_angular_vel() {
        let out = SteeringOutput { curvature: 0.5, cross_track_error: 0.0, closest_idx: 0 };
        assert!((out.angular_velocity(2.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_steering_output_ackermann() {
        let out = SteeringOutput { curvature: 0.0, cross_track_error: 0.0, closest_idx: 0 };
        assert!((out.steering_angle(2.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_steering_display() {
        let out = SteeringOutput { curvature: 0.123, cross_track_error: 0.456, closest_idx: 5 };
        let s = format!("{out}");
        assert!(s.contains("0.1230"));
    }

    #[test]
    fn test_pure_pursuit_display() {
        let pp = PurePursuit::new(2.0);
        assert!(format!("{pp}").contains("PurePursuit"));
    }

    #[test]
    fn test_stanley_display() {
        let sc = StanleyController::new(1.5);
        assert!(format!("{sc}").contains("Stanley"));
    }
}
