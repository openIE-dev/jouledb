//! Pick and Place — Approach, grasp, lift, and place phase management for
//! robotic manipulation, including collision-free waypoint generation and
//! place pose optimisation.
//!
//! Implements a finite-state-machine for pick-and-place operations with
//! configurable approach offsets, lift heights, and place-pose scoring.
//! All algorithms are std-only, using `f64` throughout.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Pick-and-place errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PickPlaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Collision detected.
    Collision(String),
    /// Phase transition error.
    InvalidTransition(String),
    /// No valid place pose found.
    NoValidPlacePose,
    /// Kinematic limit exceeded.
    KinematicLimit(String),
}

impl fmt::Display for PickPlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            Self::Collision(m) => write!(f, "collision: {m}"),
            Self::InvalidTransition(m) => write!(f, "invalid transition: {m}"),
            Self::NoValidPlacePose => write!(f, "no valid place pose found"),
            Self::KinematicLimit(m) => write!(f, "kinematic limit: {m}"),
        }
    }
}

impl std::error::Error for PickPlaceError {}

// ── 3D Pose ─────────────────────────────────────────────────────

/// 3D position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn offset(self, dx: f64, dy: f64, dz: f64) -> Self {
        Self { x: self.x + dx, y: self.y + dy, z: self.z + dz }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// Orientation as a unit quaternion (w, x, y, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Orientation {
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from_axis_angle(ax: f64, ay: f64, az: f64, angle: f64) -> Self {
        let norm = (ax * ax + ay * ay + az * az).sqrt();
        if norm < 1e-12 {
            return Self::identity();
        }
        let half = angle * 0.5;
        let s = half.sin() / norm;
        Self { w: half.cos(), x: ax * s, y: ay * s, z: az * s }
    }

    pub fn normalize(self) -> Self {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-12 {
            return Self::identity();
        }
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    /// Spherical linear interpolation.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let other = if dot < 0.0 {
            dot = -dot;
            Self { w: -other.w, x: -other.x, y: -other.y, z: -other.z }
        } else {
            other
        };
        if dot > 0.9995 {
            // Linear interpolation for nearly identical orientations
            return Self {
                w: self.w + t * (other.w - self.w),
                x: self.x + t * (other.x - self.x),
                y: self.y + t * (other.y - self.y),
                z: self.z + t * (other.z - self.z),
            }
            .normalize();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        Self {
            w: a * self.w + b * other.w,
            x: a * self.x + b * other.x,
            y: a * self.y + b * other.y,
            z: a * self.z + b * other.z,
        }
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q({:.4}, {:.4}, {:.4}, {:.4})", self.w, self.x, self.y, self.z)
    }
}

/// A full 6-DOF pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: Position,
    pub orientation: Orientation,
}

impl Pose {
    pub fn new(position: Position, orientation: Orientation) -> Self {
        Self { position, orientation }
    }

    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        Self { position: Position::new(x, y, z), orientation: Orientation::identity() }
    }

    /// Linear interpolation of position with slerp of orientation.
    pub fn interpolate(self, other: Self, t: f64) -> Self {
        let px = self.position.x + t * (other.position.x - self.position.x);
        let py = self.position.y + t * (other.position.y - self.position.y);
        let pz = self.position.z + t * (other.position.z - self.position.z);
        Self {
            position: Position::new(px, py, pz),
            orientation: self.orientation.slerp(other.orientation, t),
        }
    }
}

impl fmt::Display for Pose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pose(pos={}, ori={})", self.position, self.orientation)
    }
}

// ── Phases ──────────────────────────────────────────────────────

/// State machine phases for a pick-and-place operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Approach,
    PreGrasp,
    Grasp,
    Lift,
    Transit,
    PrePlace,
    Place,
    Release,
    Retreat,
    Done,
}

impl Phase {
    /// Valid next phases.
    pub fn successors(self) -> &'static [Phase] {
        match self {
            Self::Idle => &[Self::Approach],
            Self::Approach => &[Self::PreGrasp],
            Self::PreGrasp => &[Self::Grasp],
            Self::Grasp => &[Self::Lift],
            Self::Lift => &[Self::Transit],
            Self::Transit => &[Self::PrePlace],
            Self::PrePlace => &[Self::Place],
            Self::Place => &[Self::Release],
            Self::Release => &[Self::Retreat],
            Self::Retreat => &[Self::Done],
            Self::Done => &[Self::Idle],
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Idle => "Idle",
            Self::Approach => "Approach",
            Self::PreGrasp => "PreGrasp",
            Self::Grasp => "Grasp",
            Self::Lift => "Lift",
            Self::Transit => "Transit",
            Self::PrePlace => "PrePlace",
            Self::Place => "Place",
            Self::Release => "Release",
            Self::Retreat => "Retreat",
            Self::Done => "Done",
        };
        write!(f, "{name}")
    }
}

// ── AABB for Collision Checking ─────────────────────────────────

/// Axis-aligned bounding box for simple collision checking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Position,
    pub max: Position,
}

impl AABB {
    pub fn new(min: Position, max: Position) -> Self {
        Self { min, max }
    }

    /// Check if a point is inside this AABB (with margin).
    pub fn contains(&self, p: Position, margin: f64) -> bool {
        p.x >= self.min.x - margin
            && p.x <= self.max.x + margin
            && p.y >= self.min.y - margin
            && p.y <= self.max.y + margin
            && p.z >= self.min.z - margin
            && p.z <= self.max.z + margin
    }

    /// Check if two AABBs overlap.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

impl fmt::Display for AABB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AABB(min={}, max={})", self.min, self.max)
    }
}

// ── Place Pose Scorer ───────────────────────────────────────────

/// Score a candidate place pose based on multiple criteria.
#[derive(Debug, Clone)]
pub struct PlacePoseScorer {
    /// Preferred placement height (z).
    pub target_height: f64,
    /// Weight for distance from preferred position.
    pub distance_weight: f64,
    /// Weight for height deviation.
    pub height_weight: f64,
    /// Weight for orientation alignment.
    pub orientation_weight: f64,
    /// Preferred placement position.
    pub preferred: Position,
}

impl PlacePoseScorer {
    pub fn new(preferred: Position, target_height: f64) -> Self {
        Self {
            target_height,
            distance_weight: 1.0,
            height_weight: 2.0,
            orientation_weight: 0.5,
            preferred,
        }
    }

    pub fn with_distance_weight(mut self, w: f64) -> Self {
        self.distance_weight = w;
        self
    }

    pub fn with_height_weight(mut self, w: f64) -> Self {
        self.height_weight = w;
        self
    }

    /// Score a pose (lower is better).
    pub fn score(&self, pose: &Pose) -> f64 {
        let xy_dist = ((pose.position.x - self.preferred.x).powi(2)
            + (pose.position.y - self.preferred.y).powi(2))
        .sqrt();
        let height_err = (pose.position.z - self.target_height).abs();
        // Orientation: penalise deviation from identity (upright)
        let ori_err = 1.0 - pose.orientation.w.abs();
        self.distance_weight * xy_dist
            + self.height_weight * height_err
            + self.orientation_weight * ori_err
    }
}

// ── Waypoint Generator ──────────────────────────────────────────

/// Generate collision-free waypoints between two poses.
pub fn generate_waypoints(
    start: Pose,
    end: Pose,
    lift_height: f64,
    num_segments: usize,
    obstacles: &[AABB],
) -> Result<Vec<Pose>, PickPlaceError> {
    if num_segments == 0 {
        return Err(PickPlaceError::InvalidConfig("num_segments must be > 0".into()));
    }
    let via_start = Pose::new(
        start.position.offset(0.0, 0.0, lift_height),
        start.orientation,
    );
    let via_end = Pose::new(
        end.position.offset(0.0, 0.0, lift_height),
        end.orientation,
    );

    let mut waypoints = Vec::with_capacity(num_segments + 3);
    waypoints.push(start);
    waypoints.push(via_start);
    for i in 1..num_segments {
        let t = i as f64 / num_segments as f64;
        waypoints.push(via_start.interpolate(via_end, t));
    }
    waypoints.push(via_end);
    waypoints.push(end);

    // Collision check
    for wp in &waypoints {
        for obs in obstacles {
            if obs.contains(wp.position, 0.01) {
                return Err(PickPlaceError::Collision(
                    format!("waypoint {} inside obstacle {obs}", wp.position),
                ));
            }
        }
    }
    Ok(waypoints)
}

// ── Pick-and-Place Controller ───────────────────────────────────

/// Configuration for a pick-and-place operation.
#[derive(Debug, Clone)]
pub struct PickPlaceConfig {
    /// Approach offset above the grasp pose (metres).
    pub approach_offset: f64,
    /// Lift height after grasping (metres).
    pub lift_height: f64,
    /// Gripper close force (N).
    pub grasp_force: f64,
    /// Retreat offset above place pose (metres).
    pub retreat_offset: f64,
    /// Maximum velocity during transit (m/s).
    pub max_velocity: f64,
}

impl Default for PickPlaceConfig {
    fn default() -> Self {
        Self {
            approach_offset: 0.10,
            lift_height: 0.15,
            grasp_force: 20.0,
            retreat_offset: 0.10,
            max_velocity: 0.5,
        }
    }
}

impl PickPlaceConfig {
    pub fn with_approach_offset(mut self, v: f64) -> Self {
        self.approach_offset = v;
        self
    }

    pub fn with_lift_height(mut self, v: f64) -> Self {
        self.lift_height = v;
        self
    }

    pub fn with_grasp_force(mut self, v: f64) -> Self {
        self.grasp_force = v;
        self
    }

    pub fn with_retreat_offset(mut self, v: f64) -> Self {
        self.retreat_offset = v;
        self
    }

    pub fn with_max_velocity(mut self, v: f64) -> Self {
        self.max_velocity = v;
        self
    }
}

/// Pick-and-place state machine.
#[derive(Debug, Clone)]
pub struct PickPlaceController {
    config: PickPlaceConfig,
    phase: Phase,
    pick_pose: Pose,
    place_pose: Pose,
    current_pose: Pose,
    gripper_closed: bool,
    elapsed: f64,
}

impl PickPlaceController {
    pub fn new(config: PickPlaceConfig, pick_pose: Pose, place_pose: Pose) -> Self {
        Self {
            config,
            phase: Phase::Idle,
            pick_pose,
            place_pose,
            current_pose: pick_pose,
            gripper_closed: false,
            elapsed: 0.0,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn gripper_closed(&self) -> bool {
        self.gripper_closed
    }

    pub fn current_pose(&self) -> Pose {
        self.current_pose
    }

    /// Advance to the next phase.
    pub fn advance(&mut self) -> Result<Phase, PickPlaceError> {
        let successors = self.phase.successors();
        if successors.is_empty() {
            return Err(PickPlaceError::InvalidTransition(
                format!("no successor for {}", self.phase),
            ));
        }
        let next = successors[0];
        // Update state based on the phase transition
        match next {
            Phase::Approach => {
                self.current_pose = Pose::new(
                    self.pick_pose.position.offset(0.0, 0.0, self.config.approach_offset),
                    self.pick_pose.orientation,
                );
            }
            Phase::PreGrasp | Phase::Grasp => {
                self.current_pose = self.pick_pose;
                if next == Phase::Grasp {
                    self.gripper_closed = true;
                }
            }
            Phase::Lift => {
                self.current_pose = Pose::new(
                    self.pick_pose.position.offset(0.0, 0.0, self.config.lift_height),
                    self.pick_pose.orientation,
                );
            }
            Phase::Transit | Phase::PrePlace => {
                self.current_pose = Pose::new(
                    self.place_pose.position.offset(0.0, 0.0, self.config.lift_height),
                    self.place_pose.orientation,
                );
            }
            Phase::Place => {
                self.current_pose = self.place_pose;
            }
            Phase::Release => {
                self.gripper_closed = false;
            }
            Phase::Retreat => {
                self.current_pose = Pose::new(
                    self.place_pose.position.offset(0.0, 0.0, self.config.retreat_offset),
                    self.place_pose.orientation,
                );
            }
            Phase::Done | Phase::Idle => {}
        }
        self.phase = next;
        Ok(next)
    }

    /// Run all phases to completion.
    pub fn run_to_completion(&mut self) -> Result<Vec<Phase>, PickPlaceError> {
        let mut history = vec![self.phase];
        while self.phase != Phase::Done {
            let next = self.advance()?;
            history.push(next);
        }
        Ok(history)
    }

    /// Compute the estimated cycle time in seconds.
    pub fn estimated_cycle_time(&self) -> f64 {
        let pick_to_place = self.pick_pose.position.distance_to(self.place_pose.position);
        let approach_time = self.config.approach_offset / self.config.max_velocity;
        let lift_time = self.config.lift_height / self.config.max_velocity;
        let transit_time = pick_to_place / self.config.max_velocity;
        let place_time = self.config.lift_height / self.config.max_velocity;
        let retreat_time = self.config.retreat_offset / self.config.max_velocity;
        // Grasp/release settle time ~ 0.5s each
        2.0 * approach_time + 2.0 * lift_time + transit_time + place_time + retreat_time + 1.0
    }
}

impl fmt::Display for PickPlaceController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PickPlace(phase={}, gripper={}, pos={})",
            self.phase,
            if self.gripper_closed { "closed" } else { "open" },
            self.current_pose.position
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let a = Position::new(0.0, 0.0, 0.0);
        let b = Position::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_orientation_identity() {
        let q = Orientation::identity();
        assert!((q.w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_orientation_from_axis_angle() {
        let q = Orientation::from_axis_angle(0.0, 0.0, 1.0, std::f64::consts::PI);
        assert!(q.w.abs() < 1e-10); // cos(pi/2) = 0
        assert!((q.z - 1.0).abs() < 1e-10); // sin(pi/2) * (0,0,1) = (0,0,1)
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = Orientation::identity();
        let b = Orientation::from_axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let s0 = a.slerp(b, 0.0);
        assert!((s0.w - a.w).abs() < 1e-10);
        let s1 = a.slerp(b, 1.0);
        assert!((s1.w - b.w).abs() < 1e-6);
    }

    #[test]
    fn test_pose_interpolate() {
        let a = Pose::from_xyz(0.0, 0.0, 0.0);
        let b = Pose::from_xyz(1.0, 2.0, 3.0);
        let mid = a.interpolate(b, 0.5);
        assert!((mid.position.x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_phase_successors() {
        assert_eq!(Phase::Idle.successors(), &[Phase::Approach]);
        assert_eq!(Phase::Grasp.successors(), &[Phase::Lift]);
    }

    #[test]
    fn test_aabb_contains() {
        let aabb = AABB::new(Position::new(0.0, 0.0, 0.0), Position::new(1.0, 1.0, 1.0));
        assert!(aabb.contains(Position::new(0.5, 0.5, 0.5), 0.0));
        assert!(!aabb.contains(Position::new(2.0, 0.5, 0.5), 0.0));
    }

    #[test]
    fn test_aabb_overlaps() {
        let a = AABB::new(Position::new(0.0, 0.0, 0.0), Position::new(1.0, 1.0, 1.0));
        let b = AABB::new(Position::new(0.5, 0.5, 0.5), Position::new(1.5, 1.5, 1.5));
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_place_pose_scorer() {
        let scorer = PlacePoseScorer::new(Position::new(0.5, 0.5, 0.0), 0.1);
        let near = Pose::from_xyz(0.5, 0.5, 0.1);
        let far = Pose::from_xyz(2.0, 2.0, 0.5);
        assert!(scorer.score(&near) < scorer.score(&far));
    }

    #[test]
    fn test_generate_waypoints_basic() {
        let start = Pose::from_xyz(0.0, 0.0, 0.0);
        let end = Pose::from_xyz(1.0, 0.0, 0.0);
        let wps = generate_waypoints(start, end, 0.2, 3, &[]).unwrap();
        assert!(wps.len() >= 4);
    }

    #[test]
    fn test_generate_waypoints_collision() {
        let start = Pose::from_xyz(0.0, 0.0, 0.0);
        let end = Pose::from_xyz(1.0, 0.0, 0.0);
        let obs = vec![AABB::new(
            Position::new(-0.5, -0.5, 0.1),
            Position::new(1.5, 0.5, 0.3),
        )];
        let r = generate_waypoints(start, end, 0.2, 3, &obs);
        assert!(r.is_err());
    }

    #[test]
    fn test_controller_run_to_completion() {
        let pick = Pose::from_xyz(0.3, 0.0, 0.05);
        let place = Pose::from_xyz(0.3, 0.4, 0.05);
        let mut ctrl = PickPlaceController::new(PickPlaceConfig::default(), pick, place);
        let history = ctrl.run_to_completion().unwrap();
        assert_eq!(*history.last().unwrap(), Phase::Done);
    }

    #[test]
    fn test_controller_gripper_state() {
        let pick = Pose::from_xyz(0.3, 0.0, 0.05);
        let place = Pose::from_xyz(0.3, 0.4, 0.05);
        let mut ctrl = PickPlaceController::new(PickPlaceConfig::default(), pick, place);
        assert!(!ctrl.gripper_closed());
        // Advance to Grasp
        ctrl.advance().unwrap(); // Approach
        ctrl.advance().unwrap(); // PreGrasp
        ctrl.advance().unwrap(); // Grasp
        assert!(ctrl.gripper_closed());
    }

    #[test]
    fn test_controller_phases_count() {
        let pick = Pose::from_xyz(0.0, 0.0, 0.0);
        let place = Pose::from_xyz(1.0, 0.0, 0.0);
        let mut ctrl = PickPlaceController::new(PickPlaceConfig::default(), pick, place);
        let history = ctrl.run_to_completion().unwrap();
        // Idle -> Approach -> PreGrasp -> Grasp -> Lift -> Transit -> PrePlace -> Place -> Release -> Retreat -> Done
        assert_eq!(history.len(), 11);
    }

    #[test]
    fn test_estimated_cycle_time() {
        let pick = Pose::from_xyz(0.0, 0.0, 0.05);
        let place = Pose::from_xyz(0.5, 0.0, 0.05);
        let ctrl = PickPlaceController::new(PickPlaceConfig::default(), pick, place);
        let time = ctrl.estimated_cycle_time();
        assert!(time > 0.0);
    }

    #[test]
    fn test_config_builder() {
        let cfg = PickPlaceConfig::default()
            .with_approach_offset(0.15)
            .with_lift_height(0.2)
            .with_grasp_force(30.0);
        assert!((cfg.approach_offset - 0.15).abs() < 1e-10);
        assert!((cfg.lift_height - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_controller_display() {
        let ctrl = PickPlaceController::new(
            PickPlaceConfig::default(),
            Pose::from_xyz(0.0, 0.0, 0.0),
            Pose::from_xyz(1.0, 0.0, 0.0),
        );
        let s = format!("{ctrl}");
        assert!(s.contains("PickPlace"));
    }

    #[test]
    fn test_pose_display() {
        let p = Pose::from_xyz(1.0, 2.0, 3.0);
        let s = format!("{p}");
        assert!(s.contains("Pose"));
    }

    #[test]
    fn test_generate_waypoints_zero_segments() {
        let start = Pose::from_xyz(0.0, 0.0, 0.0);
        let end = Pose::from_xyz(1.0, 0.0, 0.0);
        assert!(generate_waypoints(start, end, 0.2, 0, &[]).is_err());
    }
}
