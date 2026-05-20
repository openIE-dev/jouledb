//! Motion Primitives — parameterized trajectories, primitive library management,
//! concatenation, feasibility checking for motion planning systems.
//!
//! Pure-Rust motion primitive infrastructure supporting polynomial, arc, and
//! clothoid trajectory segments with velocity/acceleration constraints,
//! concatenation validation, and path feasibility analysis.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveError {
    InvalidParameter(String),
    InfeasibleTrajectory(String),
    ConcatenationMismatch(String),
    EmptyLibrary,
    DuplicateId(String),
}

impl fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::InfeasibleTrajectory(s) => write!(f, "infeasible trajectory: {s}"),
            Self::ConcatenationMismatch(s) => write!(f, "concatenation mismatch: {s}"),
            Self::EmptyLibrary => write!(f, "primitive library is empty"),
            Self::DuplicateId(s) => write!(f, "duplicate primitive id: {s}"),
        }
    }
}

impl std::error::Error for PrimitiveError {}

// ── State ───────────────────────────────────────────────────────

/// A kinematic state: position, heading, velocity, curvature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicState {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub velocity: f64,
    pub curvature: f64,
}

impl KinematicState {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta, velocity: 0.0, curvature: 0.0 }
    }

    pub fn with_velocity(mut self, v: f64) -> Self { self.velocity = v; self }
    pub fn with_curvature(mut self, k: f64) -> Self { self.curvature = k; self }

    pub fn position_distance(&self, other: &KinematicState) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Angle difference, normalized to [-pi, pi].
    pub fn heading_difference(&self, other: &KinematicState) -> f64 {
        normalize_angle(self.theta - other.theta)
    }

    /// Check if two states are approximately equal within tolerances.
    pub fn approx_eq(&self, other: &KinematicState, pos_tol: f64, angle_tol: f64) -> bool {
        self.position_distance(other) <= pos_tol
            && self.heading_difference(other).abs() <= angle_tol
    }
}

impl fmt::Display for KinematicState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "(x={:.3}, y={:.3}, θ={:.3}, v={:.3}, κ={:.4})",
            self.x, self.y, self.theta, self.velocity, self.curvature,
        )
    }
}

// ── Trajectory Type ─────────────────────────────────────────────

/// The type of trajectory primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrajectoryType {
    /// Straight line segment.
    Straight,
    /// Circular arc with constant curvature.
    Arc,
    /// Clothoid (Euler spiral) with linearly varying curvature.
    Clothoid,
    /// Cubic polynomial in x/y.
    CubicPoly,
}

impl fmt::Display for TrajectoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Straight => write!(f, "Straight"),
            Self::Arc => write!(f, "Arc"),
            Self::Clothoid => write!(f, "Clothoid"),
            Self::CubicPoly => write!(f, "CubicPoly"),
        }
    }
}

// ── Kinematic Limits ────────────────────────────────────────────

/// Kinematic feasibility constraints.
#[derive(Debug, Clone, Copy)]
pub struct KinematicLimits {
    pub max_velocity: f64,
    pub max_acceleration: f64,
    pub max_curvature: f64,
    pub max_curvature_rate: f64,
}

impl KinematicLimits {
    pub fn new(max_vel: f64, max_acc: f64, max_curv: f64, max_curv_rate: f64) -> Self {
        Self {
            max_velocity: max_vel,
            max_acceleration: max_acc,
            max_curvature: max_curv,
            max_curvature_rate: max_curv_rate,
        }
    }
}

impl fmt::Display for KinematicLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Limits(v_max={:.2}, a_max={:.2}, κ_max={:.3})",
            self.max_velocity, self.max_acceleration, self.max_curvature,
        )
    }
}

// ── Motion Primitive ────────────────────────────────────────────

/// A single motion primitive: a parameterized trajectory segment.
#[derive(Debug, Clone)]
pub struct MotionPrimitive {
    pub id: String,
    pub traj_type: TrajectoryType,
    pub start_state: KinematicState,
    pub end_state: KinematicState,
    pub samples: Vec<KinematicState>,
    pub duration: f64,
    pub arc_length: f64,
    pub cost: f64,
}

impl MotionPrimitive {
    /// Create a straight-line primitive.
    pub fn straight(id: &str, start: KinematicState, length: f64, velocity: f64) -> Result<Self, PrimitiveError> {
        if length <= 0.0 {
            return Err(PrimitiveError::InvalidParameter("length must be > 0".into()));
        }
        if velocity.abs() < 1e-12 {
            return Err(PrimitiveError::InvalidParameter("velocity must be non-zero".into()));
        }
        let duration = length / velocity.abs();
        let n = (length * 10.0).ceil().max(5.0) as usize;
        let mut samples = Vec::with_capacity(n + 1);

        for i in 0..=n {
            let s = length * i as f64 / n as f64;
            samples.push(KinematicState {
                x: start.x + s * start.theta.cos(),
                y: start.y + s * start.theta.sin(),
                theta: start.theta,
                velocity,
                curvature: 0.0,
            });
        }

        let end_state = *samples.last().unwrap();
        Ok(Self {
            id: id.to_string(),
            traj_type: TrajectoryType::Straight,
            start_state: start,
            end_state,
            samples,
            duration,
            arc_length: length,
            cost: length,
        })
    }

    /// Create a circular arc primitive.
    pub fn arc(id: &str, start: KinematicState, curvature: f64, arc_len: f64, velocity: f64) -> Result<Self, PrimitiveError> {
        if arc_len <= 0.0 {
            return Err(PrimitiveError::InvalidParameter("arc_length must be > 0".into()));
        }
        if curvature.abs() < 1e-12 {
            return Err(PrimitiveError::InvalidParameter("curvature must be non-zero (use straight)".into()));
        }
        if velocity.abs() < 1e-12 {
            return Err(PrimitiveError::InvalidParameter("velocity must be non-zero".into()));
        }

        let radius = 1.0 / curvature.abs();
        let duration = arc_len / velocity.abs();
        let n = (arc_len * 10.0).ceil().max(5.0) as usize;
        let mut samples = Vec::with_capacity(n + 1);

        for i in 0..=n {
            let s = arc_len * i as f64 / n as f64;
            let delta_theta = s * curvature;
            let theta = start.theta + delta_theta;

            let (x, y) = if curvature.abs() > 1e-12 {
                let cx = start.x - (start.theta.sin()) / curvature;
                let cy = start.y + (start.theta.cos()) / curvature;
                (
                    cx + radius * (theta).sin() * curvature.signum(),
                    cy - radius * (theta).cos() * curvature.signum(),
                )
            } else {
                (start.x + s * start.theta.cos(), start.y + s * start.theta.sin())
            };

            samples.push(KinematicState { x, y, theta, velocity, curvature });
        }

        let end_state = *samples.last().unwrap();
        Ok(Self {
            id: id.to_string(),
            traj_type: TrajectoryType::Arc,
            start_state: start,
            end_state,
            samples,
            duration,
            arc_length: arc_len,
            cost: arc_len,
        })
    }

    /// Create a clothoid (Euler spiral) primitive using numerical integration.
    pub fn clothoid(
        id: &str,
        start: KinematicState,
        start_curvature: f64,
        end_curvature: f64,
        arc_len: f64,
        velocity: f64,
    ) -> Result<Self, PrimitiveError> {
        if arc_len <= 0.0 {
            return Err(PrimitiveError::InvalidParameter("arc_length must be > 0".into()));
        }
        if velocity.abs() < 1e-12 {
            return Err(PrimitiveError::InvalidParameter("velocity must be non-zero".into()));
        }

        let duration = arc_len / velocity.abs();
        let n = (arc_len * 20.0).ceil().max(10.0) as usize;
        let ds = arc_len / n as f64;
        let dk_ds = (end_curvature - start_curvature) / arc_len;

        let mut samples = Vec::with_capacity(n + 1);
        let mut x = start.x;
        let mut y = start.y;
        let mut theta = start.theta;
        let mut kappa = start_curvature;

        samples.push(KinematicState { x, y, theta, velocity, curvature: kappa });

        for _ in 0..n {
            theta += kappa * ds;
            kappa += dk_ds * ds;
            x += ds * theta.cos();
            y += ds * theta.sin();
            samples.push(KinematicState { x, y, theta, velocity, curvature: kappa });
        }

        let end_state = *samples.last().unwrap();
        Ok(Self {
            id: id.to_string(),
            traj_type: TrajectoryType::Clothoid,
            start_state: start,
            end_state,
            samples,
            duration,
            arc_length: arc_len,
            cost: arc_len * (1.0 + 0.1 * dk_ds.abs()), // slight cost for curvature change
        })
    }

    /// Check feasibility against kinematic limits.
    pub fn is_feasible(&self, limits: &KinematicLimits) -> bool {
        for s in &self.samples {
            if s.velocity.abs() > limits.max_velocity { return false; }
            if s.curvature.abs() > limits.max_curvature { return false; }
        }
        // Check acceleration (velocity change over time)
        if self.samples.len() >= 2 {
            let dt = self.duration / (self.samples.len() - 1) as f64;
            for w in self.samples.windows(2) {
                let accel = (w[1].velocity - w[0].velocity).abs() / dt;
                if accel > limits.max_acceleration { return false; }
                let krate = (w[1].curvature - w[0].curvature).abs() / dt;
                if krate > limits.max_curvature_rate { return false; }
            }
        }
        true
    }

    /// Check if this primitive can be concatenated after `prev`.
    pub fn can_follow(&self, prev: &MotionPrimitive, pos_tol: f64, angle_tol: f64) -> bool {
        prev.end_state.approx_eq(&self.start_state, pos_tol, angle_tol)
    }
}

impl fmt::Display for MotionPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Primitive(id={}, type={}, len={:.3}, dur={:.3}, pts={})",
            self.id, self.traj_type, self.arc_length, self.duration, self.samples.len(),
        )
    }
}

// ── Primitive Library ───────────────────────────────────────────

/// A collection of motion primitives indexed by id.
pub struct PrimitiveLibraryStore {
    primitives: Vec<MotionPrimitive>,
}

impl PrimitiveLibraryStore {
    pub fn new() -> Self { Self { primitives: Vec::new() } }

    pub fn add(&mut self, prim: MotionPrimitive) -> Result<(), PrimitiveError> {
        if self.primitives.iter().any(|p| p.id == prim.id) {
            return Err(PrimitiveError::DuplicateId(prim.id));
        }
        self.primitives.push(prim);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&MotionPrimitive> {
        self.primitives.iter().find(|p| p.id == id)
    }

    pub fn count(&self) -> usize { self.primitives.len() }

    pub fn all(&self) -> &[MotionPrimitive] { &self.primitives }

    /// Get primitives that are feasible under given limits.
    pub fn feasible(&self, limits: &KinematicLimits) -> Vec<&MotionPrimitive> {
        self.primitives.iter().filter(|p| p.is_feasible(limits)).collect()
    }

    /// Get primitives whose start state matches a query (within tolerance).
    pub fn matching_start(
        &self,
        state: &KinematicState,
        pos_tol: f64,
        angle_tol: f64,
    ) -> Vec<&MotionPrimitive> {
        self.primitives.iter()
            .filter(|p| p.start_state.approx_eq(state, pos_tol, angle_tol))
            .collect()
    }
}

impl fmt::Display for PrimitiveLibraryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrimitiveLibraryStore(count={})", self.primitives.len())
    }
}

// ── Trajectory Concatenation ────────────────────────────────────

/// A concatenated sequence of motion primitives forming a complete trajectory.
#[derive(Debug, Clone)]
pub struct ConcatenatedTrajectory {
    pub segments: Vec<MotionPrimitive>,
    pub total_length: f64,
    pub total_duration: f64,
    pub total_cost: f64,
}

impl ConcatenatedTrajectory {
    pub fn new() -> Self {
        Self { segments: Vec::new(), total_length: 0.0, total_duration: 0.0, total_cost: 0.0 }
    }

    /// Append a primitive, checking continuity.
    pub fn append(
        &mut self,
        prim: MotionPrimitive,
        pos_tol: f64,
        angle_tol: f64,
    ) -> Result<(), PrimitiveError> {
        if let Some(last) = self.segments.last() {
            if !prim.can_follow(last, pos_tol, angle_tol) {
                return Err(PrimitiveError::ConcatenationMismatch(format!(
                    "end {} vs start {}",
                    last.end_state, prim.start_state,
                )));
            }
        }
        self.total_length += prim.arc_length;
        self.total_duration += prim.duration;
        self.total_cost += prim.cost;
        self.segments.push(prim);
        Ok(())
    }

    /// Get all sample points across all segments.
    pub fn all_samples(&self) -> Vec<KinematicState> {
        let mut result = Vec::new();
        for (i, seg) in self.segments.iter().enumerate() {
            let start = if i == 0 { 0 } else { 1 }; // skip duplicate junction points
            for s in &seg.samples[start..] {
                result.push(*s);
            }
        }
        result
    }

    /// Check feasibility of the entire trajectory.
    pub fn is_feasible(&self, limits: &KinematicLimits) -> bool {
        self.segments.iter().all(|s| s.is_feasible(limits))
    }

    pub fn segment_count(&self) -> usize { self.segments.len() }

    pub fn start_state(&self) -> Option<KinematicState> {
        self.segments.first().map(|s| s.start_state)
    }

    pub fn end_state(&self) -> Option<KinematicState> {
        self.segments.last().map(|s| s.end_state)
    }
}

impl fmt::Display for ConcatenatedTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "ConcatTraj(segs={}, len={:.3}, dur={:.3}, cost={:.3})",
            self.segments.len(), self.total_length, self.total_duration, self.total_cost,
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

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-4 }
    fn origin() -> KinematicState { KinematicState::new(0.0, 0.0, 0.0).with_velocity(1.0) }

    #[test]
    fn test_state_display() {
        let s = KinematicState::new(1.0, 2.0, 0.5);
        let text = format!("{s}");
        assert!(text.contains("x=1.000"));
    }

    #[test]
    fn test_state_distance() {
        let a = KinematicState::new(0.0, 0.0, 0.0);
        let b = KinematicState::new(3.0, 4.0, 0.0);
        assert!(approx(a.position_distance(&b), 5.0));
    }

    #[test]
    fn test_state_approx_eq() {
        let a = KinematicState::new(0.0, 0.0, 0.0);
        let b = KinematicState::new(0.001, 0.0, 0.001);
        assert!(a.approx_eq(&b, 0.01, 0.01));
        assert!(!a.approx_eq(&b, 0.0001, 0.01));
    }

    #[test]
    fn test_straight_primitive() {
        let prim = MotionPrimitive::straight("s1", origin(), 5.0, 1.0).unwrap();
        assert!(approx(prim.arc_length, 5.0));
        assert!(approx(prim.duration, 5.0));
        assert!(approx(prim.end_state.x, 5.0));
        assert!(approx(prim.end_state.y, 0.0));
    }

    #[test]
    fn test_straight_invalid() {
        assert!(MotionPrimitive::straight("s1", origin(), 0.0, 1.0).is_err());
        assert!(MotionPrimitive::straight("s1", origin(), 5.0, 0.0).is_err());
    }

    #[test]
    fn test_arc_primitive() {
        let prim = MotionPrimitive::arc("a1", origin(), 0.5, 3.0, 1.0).unwrap();
        assert!(approx(prim.arc_length, 3.0));
        assert!(prim.end_state.theta > 0.0); // turned left
    }

    #[test]
    fn test_arc_invalid() {
        assert!(MotionPrimitive::arc("a1", origin(), 0.0, 3.0, 1.0).is_err());
    }

    #[test]
    fn test_clothoid_primitive() {
        let prim = MotionPrimitive::clothoid("c1", origin(), 0.0, 0.5, 4.0, 1.0).unwrap();
        assert!(approx(prim.arc_length, 4.0));
        // End curvature should be approximately 0.5
        assert!((prim.end_state.curvature - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_clothoid_invalid() {
        assert!(MotionPrimitive::clothoid("c1", origin(), 0.0, 0.5, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_feasibility() {
        let prim = MotionPrimitive::straight("s1", origin(), 5.0, 1.0).unwrap();
        let limits = KinematicLimits::new(2.0, 1.0, 1.0, 1.0);
        assert!(prim.is_feasible(&limits));

        let tight = KinematicLimits::new(0.5, 1.0, 1.0, 1.0); // speed too low
        assert!(!prim.is_feasible(&tight));
    }

    #[test]
    fn test_concatenation() {
        let p1 = MotionPrimitive::straight("s1", origin(), 2.0, 1.0).unwrap();
        let start2 = p1.end_state;
        let p2 = MotionPrimitive::straight("s2", start2, 3.0, 1.0).unwrap();

        let mut traj = ConcatenatedTrajectory::new();
        traj.append(p1, 0.01, 0.01).unwrap();
        traj.append(p2, 0.01, 0.01).unwrap();
        assert_eq!(traj.segment_count(), 2);
        assert!(approx(traj.total_length, 5.0));
    }

    #[test]
    fn test_concatenation_mismatch() {
        let p1 = MotionPrimitive::straight("s1", origin(), 2.0, 1.0).unwrap();
        let far_start = KinematicState::new(100.0, 100.0, 0.0).with_velocity(1.0);
        let p2 = MotionPrimitive::straight("s2", far_start, 3.0, 1.0).unwrap();

        let mut traj = ConcatenatedTrajectory::new();
        traj.append(p1, 0.01, 0.01).unwrap();
        assert!(traj.append(p2, 0.01, 0.01).is_err());
    }

    #[test]
    fn test_all_samples() {
        let p1 = MotionPrimitive::straight("s1", origin(), 1.0, 1.0).unwrap();
        let start2 = p1.end_state;
        let p2 = MotionPrimitive::straight("s2", start2, 1.0, 1.0).unwrap();

        let mut traj = ConcatenatedTrajectory::new();
        traj.append(p1, 0.1, 0.1).unwrap();
        traj.append(p2, 0.1, 0.1).unwrap();
        let samples = traj.all_samples();
        assert!(samples.len() >= 4);
    }

    #[test]
    fn test_library_store() {
        let mut lib = PrimitiveLibraryStore::new();
        let p = MotionPrimitive::straight("s1", origin(), 2.0, 1.0).unwrap();
        lib.add(p).unwrap();
        assert_eq!(lib.count(), 1);
        assert!(lib.get("s1").is_some());
        assert!(lib.get("missing").is_none());
    }

    #[test]
    fn test_library_duplicate() {
        let mut lib = PrimitiveLibraryStore::new();
        lib.add(MotionPrimitive::straight("s1", origin(), 2.0, 1.0).unwrap()).unwrap();
        assert!(lib.add(MotionPrimitive::straight("s1", origin(), 3.0, 1.0).unwrap()).is_err());
    }

    #[test]
    fn test_library_feasible_filter() {
        let mut lib = PrimitiveLibraryStore::new();
        lib.add(MotionPrimitive::straight("slow", origin().with_velocity(0.5), 2.0, 0.5).unwrap()).unwrap();
        lib.add(MotionPrimitive::straight("fast", origin().with_velocity(5.0), 2.0, 5.0).unwrap()).unwrap();
        let limits = KinematicLimits::new(2.0, 10.0, 1.0, 1.0);
        let feasible = lib.feasible(&limits);
        assert_eq!(feasible.len(), 1);
        assert_eq!(feasible[0].id, "slow");
    }

    #[test]
    fn test_library_display() {
        let lib = PrimitiveLibraryStore::new();
        let s = format!("{lib}");
        assert!(s.contains("PrimitiveLibraryStore"));
    }

    #[test]
    fn test_primitive_display() {
        let p = MotionPrimitive::straight("s1", origin(), 2.0, 1.0).unwrap();
        let s = format!("{p}");
        assert!(s.contains("Primitive"));
        assert!(s.contains("Straight"));
    }

    #[test]
    fn test_concat_display() {
        let traj = ConcatenatedTrajectory::new();
        let s = format!("{traj}");
        assert!(s.contains("ConcatTraj"));
    }

    #[test]
    fn test_trajectory_type_display() {
        assert_eq!(format!("{}", TrajectoryType::Clothoid), "Clothoid");
    }

    #[test]
    fn test_error_display() {
        let e = PrimitiveError::EmptyLibrary;
        assert_eq!(format!("{e}"), "primitive library is empty");
    }
}
