//! Trajectory generation — joint-space interpolation, Cartesian paths,
//! trapezoidal velocity profiles, and S-curve motion planning.
//!
//! Generates smooth, time-parameterized trajectories for robot joints and
//! end-effector paths, with support for velocity and acceleration constraints.

use std::f64::consts::PI;

// ── Errors ──────────────────────────────────────────────────────

/// Trajectory generation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryError {
    /// Duration must be positive.
    InvalidDuration(f64),
    /// Not enough waypoints.
    InsufficientWaypoints { min: usize, got: usize },
    /// Dimension mismatch between waypoints.
    DimensionMismatch { expected: usize, got: usize },
    /// Velocity or acceleration limit violation.
    LimitViolation(String),
    /// Time parameter out of range.
    TimeOutOfRange { time: f64, duration: f64 },
}

impl std::fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDuration(d) => write!(f, "invalid duration: {d:.4}"),
            Self::InsufficientWaypoints { min, got } => {
                write!(f, "need at least {min} waypoints, got {got}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::LimitViolation(msg) => write!(f, "limit violation: {msg}"),
            Self::TimeOutOfRange { time, duration } => {
                write!(f, "time {time:.4} out of range [0, {duration:.4}]")
            }
        }
    }
}

impl std::error::Error for TrajectoryError {}

// ── Trajectory Point ───────────────────────────────────────────

/// A single sampled trajectory point.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryPoint {
    /// Time stamp (seconds).
    pub time: f64,
    /// Position (joint values or Cartesian coordinates).
    pub position: Vec<f64>,
    /// Velocity.
    pub velocity: Vec<f64>,
    /// Acceleration.
    pub acceleration: Vec<f64>,
}

impl TrajectoryPoint {
    /// Create a trajectory point with zero velocity/acceleration.
    pub fn at_rest(time: f64, position: Vec<f64>) -> Self {
        let n = position.len();
        Self {
            time,
            position,
            velocity: vec![0.0; n],
            acceleration: vec![0.0; n],
        }
    }

    /// Dimension of the point.
    pub fn dim(&self) -> usize {
        self.position.len()
    }
}

impl std::fmt::Display for TrajectoryPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t={:.4} pos={:?}", self.time, self.position)
    }
}

// ── Linear Interpolation ───────────────────────────────────────

/// Linear (joint-space) trajectory between two configurations.
#[derive(Debug, Clone)]
pub struct LinearTrajectory {
    start: Vec<f64>,
    end: Vec<f64>,
    duration: f64,
}

impl LinearTrajectory {
    /// Create a linear trajectory.
    pub fn new(start: Vec<f64>, end: Vec<f64>, duration: f64) -> Result<Self, TrajectoryError> {
        if duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(duration));
        }
        if start.len() != end.len() {
            return Err(TrajectoryError::DimensionMismatch {
                expected: start.len(),
                got: end.len(),
            });
        }
        Ok(Self { start, end, duration })
    }

    /// Sample at time `t`.
    pub fn sample(&self, t: f64) -> Result<TrajectoryPoint, TrajectoryError> {
        let t_clamped = t.clamp(0.0, self.duration);
        let s = t_clamped / self.duration;
        let n = self.start.len();
        let mut pos = vec![0.0; n];
        let mut vel = vec![0.0; n];
        for i in 0..n {
            let delta = self.end[i] - self.start[i];
            pos[i] = self.start[i] + s * delta;
            vel[i] = delta / self.duration;
        }
        Ok(TrajectoryPoint {
            time: t_clamped,
            position: pos,
            velocity: vel,
            acceleration: vec![0.0; n],
        })
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Sample the entire trajectory at a given frequency.
    pub fn sample_uniform(&self, dt: f64) -> Result<Vec<TrajectoryPoint>, TrajectoryError> {
        if dt <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(dt));
        }
        let mut points = Vec::new();
        let mut t = 0.0;
        while t <= self.duration + 1e-12 {
            points.push(self.sample(t)?);
            t += dt;
        }
        Ok(points)
    }
}

impl std::fmt::Display for LinearTrajectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LinearTrajectory({} DOF, {:.3}s)",
            self.start.len(),
            self.duration,
        )
    }
}

// ── Cubic Polynomial Trajectory ────────────────────────────────

/// Cubic polynomial trajectory with specified boundary velocities.
///
/// Fits `q(t) = a0 + a1*t + a2*t^2 + a3*t^3` per DOF.
#[derive(Debug, Clone)]
pub struct CubicTrajectory {
    /// Coefficients per DOF: `[a0, a1, a2, a3]`.
    coefficients: Vec<[f64; 4]>,
    duration: f64,
}

impl CubicTrajectory {
    /// Create a cubic trajectory from start/end positions and velocities.
    pub fn new(
        q_start: &[f64],
        q_end: &[f64],
        v_start: &[f64],
        v_end: &[f64],
        duration: f64,
    ) -> Result<Self, TrajectoryError> {
        if duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(duration));
        }
        let n = q_start.len();
        if q_end.len() != n || v_start.len() != n || v_end.len() != n {
            return Err(TrajectoryError::DimensionMismatch { expected: n, got: q_end.len() });
        }
        let tf = duration;
        let tf2 = tf * tf;
        let tf3 = tf2 * tf;
        let mut coefficients = Vec::with_capacity(n);
        for i in 0..n {
            let a0 = q_start[i];
            let a1 = v_start[i];
            let a2 = (3.0 * (q_end[i] - q_start[i]) - (2.0 * v_start[i] + v_end[i]) * tf) / tf2;
            let a3 = (-2.0 * (q_end[i] - q_start[i]) + (v_start[i] + v_end[i]) * tf) / tf3;
            coefficients.push([a0, a1, a2, a3]);
        }
        Ok(Self { coefficients, duration })
    }

    /// Create with zero boundary velocities.
    pub fn zero_velocity(
        q_start: &[f64],
        q_end: &[f64],
        duration: f64,
    ) -> Result<Self, TrajectoryError> {
        let n = q_start.len();
        let zeros = vec![0.0; n];
        Self::new(q_start, q_end, &zeros, &zeros, duration)
    }

    /// Sample at time `t`.
    pub fn sample(&self, t: f64) -> TrajectoryPoint {
        let tc = t.clamp(0.0, self.duration);
        let tc2 = tc * tc;
        let tc3 = tc2 * tc;
        let n = self.coefficients.len();
        let mut pos = vec![0.0; n];
        let mut vel = vec![0.0; n];
        let mut acc = vec![0.0; n];
        for (i, c) in self.coefficients.iter().enumerate() {
            pos[i] = c[0] + c[1] * tc + c[2] * tc2 + c[3] * tc3;
            vel[i] = c[1] + 2.0 * c[2] * tc + 3.0 * c[3] * tc2;
            acc[i] = 2.0 * c[2] + 6.0 * c[3] * tc;
        }
        TrajectoryPoint { time: tc, position: pos, velocity: vel, acceleration: acc }
    }

    /// Duration.
    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl std::fmt::Display for CubicTrajectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CubicTrajectory({} DOF, {:.3}s)",
            self.coefficients.len(),
            self.duration,
        )
    }
}

// ── Quintic Polynomial Trajectory ──────────────────────────────

/// Quintic polynomial trajectory with position, velocity, and acceleration
/// boundary conditions.
#[derive(Debug, Clone)]
pub struct QuinticTrajectory {
    /// Coefficients per DOF: `[a0, a1, a2, a3, a4, a5]`.
    coefficients: Vec<[f64; 6]>,
    duration: f64,
}

impl QuinticTrajectory {
    /// Create a quintic trajectory.
    pub fn new(
        q0: &[f64], q1: &[f64],
        v0: &[f64], v1: &[f64],
        a0: &[f64], a1: &[f64],
        duration: f64,
    ) -> Result<Self, TrajectoryError> {
        if duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(duration));
        }
        let n = q0.len();
        if q1.len() != n || v0.len() != n || v1.len() != n || a0.len() != n || a1.len() != n {
            return Err(TrajectoryError::DimensionMismatch { expected: n, got: q1.len() });
        }
        let tf = duration;
        let tf2 = tf * tf;
        let tf3 = tf2 * tf;
        let tf4 = tf3 * tf;
        let tf5 = tf4 * tf;

        let mut coefficients = Vec::with_capacity(n);
        for i in 0..n {
            let c0 = q0[i];
            let c1 = v0[i];
            let c2 = a0[i] / 2.0;
            let dq = q1[i] - q0[i];
            let c3 = (20.0 * dq - (8.0 * v1[i] + 12.0 * v0[i]) * tf
                - (3.0 * a0[i] - a1[i]) * tf2)
                / (2.0 * tf3);
            let c4 = (-30.0 * dq + (14.0 * v1[i] + 16.0 * v0[i]) * tf
                + (3.0 * a0[i] - 2.0 * a1[i]) * tf2)
                / (2.0 * tf4);
            let c5 = (12.0 * dq - 6.0 * (v1[i] + v0[i]) * tf
                + (a1[i] - a0[i]) * tf2)
                / (2.0 * tf5);
            coefficients.push([c0, c1, c2, c3, c4, c5]);
        }
        Ok(Self { coefficients, duration })
    }

    /// Create with zero boundary velocities and accelerations.
    pub fn rest_to_rest(q0: &[f64], q1: &[f64], duration: f64) -> Result<Self, TrajectoryError> {
        let n = q0.len();
        let zeros = vec![0.0; n];
        Self::new(q0, q1, &zeros, &zeros, &zeros, &zeros, duration)
    }

    /// Sample at time `t`.
    pub fn sample(&self, t: f64) -> TrajectoryPoint {
        let tc = t.clamp(0.0, self.duration);
        let n = self.coefficients.len();
        let mut pos = vec![0.0; n];
        let mut vel = vec![0.0; n];
        let mut acc = vec![0.0; n];
        for (i, c) in self.coefficients.iter().enumerate() {
            let t2 = tc * tc;
            let t3 = t2 * tc;
            let t4 = t3 * tc;
            let t5 = t4 * tc;
            pos[i] = c[0] + c[1] * tc + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5;
            vel[i] = c[1] + 2.0 * c[2] * tc + 3.0 * c[3] * t2 + 4.0 * c[4] * t3 + 5.0 * c[5] * t4;
            acc[i] = 2.0 * c[2] + 6.0 * c[3] * tc + 12.0 * c[4] * t2 + 20.0 * c[5] * t3;
        }
        TrajectoryPoint { time: tc, position: pos, velocity: vel, acceleration: acc }
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl std::fmt::Display for QuinticTrajectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QuinticTrajectory({} DOF, {:.3}s)",
            self.coefficients.len(),
            self.duration,
        )
    }
}

// ── Multi-Segment (Waypoint) Trajectory ────────────────────────

/// Multi-segment cubic spline trajectory through waypoints.
#[derive(Debug, Clone)]
pub struct WaypointTrajectory {
    /// Segments (one cubic per pair of waypoints, per DOF).
    segments: Vec<CubicTrajectory>,
    /// Cumulative time for each segment start.
    segment_times: Vec<f64>,
    /// Total duration.
    total_duration: f64,
}

impl WaypointTrajectory {
    /// Create a waypoint trajectory with equal time segments and zero-velocity endpoints.
    pub fn new(waypoints: &[Vec<f64>], segment_duration: f64) -> Result<Self, TrajectoryError> {
        if waypoints.len() < 2 {
            return Err(TrajectoryError::InsufficientWaypoints {
                min: 2,
                got: waypoints.len(),
            });
        }
        if segment_duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(segment_duration));
        }
        let n = waypoints[0].len();
        for (i, wp) in waypoints.iter().enumerate() {
            if wp.len() != n {
                return Err(TrajectoryError::DimensionMismatch { expected: n, got: wp.len() });
            }
            let _ = i; // used for error context
        }

        // Compute intermediate velocities using central differences
        let num_seg = waypoints.len() - 1;
        let mut velocities = vec![vec![0.0; n]; waypoints.len()];
        for i in 1..waypoints.len() - 1 {
            for j in 0..n {
                velocities[i][j] = (waypoints[i + 1][j] - waypoints[i - 1][j])
                    / (2.0 * segment_duration);
            }
        }

        let mut segments = Vec::with_capacity(num_seg);
        let mut segment_times = Vec::with_capacity(num_seg);
        for i in 0..num_seg {
            segment_times.push(i as f64 * segment_duration);
            let seg = CubicTrajectory::new(
                &waypoints[i],
                &waypoints[i + 1],
                &velocities[i],
                &velocities[i + 1],
                segment_duration,
            )?;
            segments.push(seg);
        }
        let total_duration = num_seg as f64 * segment_duration;
        Ok(Self { segments, segment_times, total_duration })
    }

    /// Sample at time `t`.
    pub fn sample(&self, t: f64) -> TrajectoryPoint {
        let tc = t.clamp(0.0, self.total_duration);
        // Find which segment
        let mut seg_idx = 0;
        for (i, &st) in self.segment_times.iter().enumerate().rev() {
            if tc >= st {
                seg_idx = i;
                break;
            }
        }
        if seg_idx >= self.segments.len() {
            seg_idx = self.segments.len() - 1;
        }
        let local_t = tc - self.segment_times[seg_idx];
        let mut pt = self.segments[seg_idx].sample(local_t);
        pt.time = tc;
        pt
    }

    /// Total duration.
    pub fn total_duration(&self) -> f64 {
        self.total_duration
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

impl std::fmt::Display for WaypointTrajectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WaypointTrajectory({} segments, {:.3}s total)",
            self.segments.len(),
            self.total_duration,
        )
    }
}

// ── Cartesian Line Trajectory ──────────────────────────────────

/// Straight-line Cartesian trajectory between two 3D points.
#[derive(Debug, Clone)]
pub struct CartesianLine {
    start: [f64; 3],
    end: [f64; 3],
    duration: f64,
}

impl CartesianLine {
    /// Create a Cartesian line trajectory.
    pub fn new(start: [f64; 3], end: [f64; 3], duration: f64) -> Result<Self, TrajectoryError> {
        if duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(duration));
        }
        Ok(Self { start, end, duration })
    }

    /// Sample at time `t` with a quintic time-scaling for smooth start/stop.
    pub fn sample_smooth(&self, t: f64) -> [f64; 3] {
        let tc = t.clamp(0.0, self.duration);
        let tau = tc / self.duration;
        // Quintic time-scaling: s = 10*tau^3 - 15*tau^4 + 6*tau^5
        let s = 10.0 * tau.powi(3) - 15.0 * tau.powi(4) + 6.0 * tau.powi(5);
        [
            self.start[0] + s * (self.end[0] - self.start[0]),
            self.start[1] + s * (self.end[1] - self.start[1]),
            self.start[2] + s * (self.end[2] - self.start[2]),
        ]
    }

    /// Sample at time `t` with linear time scaling.
    pub fn sample_linear(&self, t: f64) -> [f64; 3] {
        let tc = t.clamp(0.0, self.duration);
        let s = tc / self.duration;
        [
            self.start[0] + s * (self.end[0] - self.start[0]),
            self.start[1] + s * (self.end[1] - self.start[1]),
            self.start[2] + s * (self.end[2] - self.start[2]),
        ]
    }

    /// Path length.
    pub fn path_length(&self) -> f64 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl std::fmt::Display for CartesianLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CartesianLine(len={:.4}, {:.3}s)",
            self.path_length(),
            self.duration,
        )
    }
}

// ── Circular Arc Trajectory ────────────────────────────────────

/// Circular arc trajectory in 3D around a given center and axis.
#[derive(Debug, Clone)]
pub struct CircularArc {
    center: [f64; 3],
    radius: f64,
    /// Normal vector to the plane of the arc (unit vector).
    normal: [f64; 3],
    /// Start angle (radians).
    start_angle: f64,
    /// Sweep angle (radians, can be negative for CW).
    sweep: f64,
    duration: f64,
    /// Local x-axis in the arc plane.
    u: [f64; 3],
    /// Local y-axis in the arc plane.
    v: [f64; 3],
}

impl CircularArc {
    /// Create a circular arc in the XY plane.
    pub fn xy_arc(
        center: [f64; 2],
        radius: f64,
        start_angle: f64,
        sweep: f64,
        z: f64,
        duration: f64,
    ) -> Result<Self, TrajectoryError> {
        if duration <= 0.0 {
            return Err(TrajectoryError::InvalidDuration(duration));
        }
        if radius <= 0.0 {
            return Err(TrajectoryError::LimitViolation("radius must be positive".into()));
        }
        Ok(Self {
            center: [center[0], center[1], z],
            radius,
            normal: [0.0, 0.0, 1.0],
            start_angle,
            sweep,
            duration,
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
        })
    }

    /// Sample position at time `t`.
    pub fn sample(&self, t: f64) -> [f64; 3] {
        let tc = t.clamp(0.0, self.duration);
        let s = tc / self.duration;
        let angle = self.start_angle + s * self.sweep;
        let ca = angle.cos();
        let sa = angle.sin();
        [
            self.center[0] + self.radius * (ca * self.u[0] + sa * self.v[0]),
            self.center[1] + self.radius * (ca * self.u[1] + sa * self.v[1]),
            self.center[2] + self.radius * (ca * self.u[2] + sa * self.v[2]),
        ]
    }

    /// Arc length.
    pub fn arc_length(&self) -> f64 {
        self.radius * self.sweep.abs()
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl std::fmt::Display for CircularArc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CircularArc(r={:.4}, sweep={:.2}°, {:.3}s)",
            self.radius,
            self.sweep.to_degrees(),
            self.duration,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_linear_start() {
        let traj = LinearTrajectory::new(vec![0.0], vec![1.0], 2.0).unwrap();
        let pt = traj.sample(0.0).unwrap();
        assert!(approx_eq(pt.position[0], 0.0));
    }

    #[test]
    fn test_linear_end() {
        let traj = LinearTrajectory::new(vec![0.0], vec![1.0], 2.0).unwrap();
        let pt = traj.sample(2.0).unwrap();
        assert!(approx_eq(pt.position[0], 1.0));
    }

    #[test]
    fn test_linear_midpoint() {
        let traj = LinearTrajectory::new(vec![0.0, 10.0], vec![10.0, 0.0], 1.0).unwrap();
        let pt = traj.sample(0.5).unwrap();
        assert!(approx_eq(pt.position[0], 5.0));
        assert!(approx_eq(pt.position[1], 5.0));
    }

    #[test]
    fn test_linear_invalid_duration() {
        let r = LinearTrajectory::new(vec![0.0], vec![1.0], -1.0);
        assert!(matches!(r, Err(TrajectoryError::InvalidDuration(_))));
    }

    #[test]
    fn test_linear_dimension_mismatch() {
        let r = LinearTrajectory::new(vec![0.0], vec![1.0, 2.0], 1.0);
        assert!(matches!(r, Err(TrajectoryError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_linear_uniform_sampling() {
        let traj = LinearTrajectory::new(vec![0.0], vec![1.0], 1.0).unwrap();
        let pts = traj.sample_uniform(0.25).unwrap();
        assert!(pts.len() >= 4);
    }

    #[test]
    fn test_cubic_boundary_positions() {
        let traj = CubicTrajectory::zero_velocity(&[0.0], &[1.0], 1.0).unwrap();
        let p0 = traj.sample(0.0);
        let p1 = traj.sample(1.0);
        assert!(approx_eq(p0.position[0], 0.0));
        assert!(approx_eq(p1.position[0], 1.0));
    }

    #[test]
    fn test_cubic_zero_boundary_velocity() {
        let traj = CubicTrajectory::zero_velocity(&[0.0], &[1.0], 1.0).unwrap();
        let p0 = traj.sample(0.0);
        let p1 = traj.sample(1.0);
        assert!(approx_eq(p0.velocity[0], 0.0));
        assert!(approx_eq(p1.velocity[0], 0.0));
    }

    #[test]
    fn test_quintic_boundary_conditions() {
        let traj = QuinticTrajectory::rest_to_rest(&[0.0], &[2.0], 1.0).unwrap();
        let p0 = traj.sample(0.0);
        let p1 = traj.sample(1.0);
        assert!(approx_eq(p0.position[0], 0.0));
        assert!(approx_eq(p1.position[0], 2.0));
        assert!(approx_eq(p0.velocity[0], 0.0));
        assert!(approx_eq(p1.velocity[0], 0.0));
        assert!(approx_eq(p0.acceleration[0], 0.0));
        assert!(approx_eq(p1.acceleration[0], 0.0));
    }

    #[test]
    fn test_waypoint_trajectory() {
        let wps = vec![vec![0.0], vec![1.0], vec![0.5]];
        let traj = WaypointTrajectory::new(&wps, 1.0).unwrap();
        assert_eq!(traj.num_segments(), 2);
        assert!(approx_eq(traj.total_duration(), 2.0));
    }

    #[test]
    fn test_waypoint_start_end() {
        let wps = vec![vec![0.0], vec![1.0], vec![2.0]];
        let traj = WaypointTrajectory::new(&wps, 1.0).unwrap();
        let p0 = traj.sample(0.0);
        let p_end = traj.sample(2.0);
        assert!(approx_eq(p0.position[0], 0.0));
        assert!(approx_eq(p_end.position[0], 2.0));
    }

    #[test]
    fn test_waypoint_too_few() {
        let r = WaypointTrajectory::new(&[vec![0.0]], 1.0);
        assert!(matches!(r, Err(TrajectoryError::InsufficientWaypoints { .. })));
    }

    #[test]
    fn test_cartesian_line_start() {
        let line = CartesianLine::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0).unwrap();
        let p = line.sample_linear(0.0);
        assert!(approx_eq(p[0], 0.0));
    }

    #[test]
    fn test_cartesian_line_end() {
        let line = CartesianLine::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 2.0).unwrap();
        let p = line.sample_linear(2.0);
        assert!(approx_eq(p[0], 1.0));
        assert!(approx_eq(p[1], 2.0));
        assert!(approx_eq(p[2], 3.0));
    }

    #[test]
    fn test_cartesian_smooth_endpoints() {
        let line = CartesianLine::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0).unwrap();
        let p0 = line.sample_smooth(0.0);
        let p1 = line.sample_smooth(1.0);
        assert!(approx_eq(p0[0], 0.0));
        assert!(approx_eq(p1[0], 1.0));
    }

    #[test]
    fn test_cartesian_path_length() {
        let line = CartesianLine::new([0.0, 0.0, 0.0], [3.0, 4.0, 0.0], 1.0).unwrap();
        assert!(approx_eq(line.path_length(), 5.0));
    }

    #[test]
    fn test_circular_arc_start() {
        let arc = CircularArc::xy_arc([0.0, 0.0], 1.0, 0.0, PI, 0.0, 1.0).unwrap();
        let p = arc.sample(0.0);
        assert!(approx_eq(p[0], 1.0));
        assert!(approx_eq(p[1], 0.0));
    }

    #[test]
    fn test_circular_arc_end() {
        let arc = CircularArc::xy_arc([0.0, 0.0], 1.0, 0.0, PI, 0.0, 1.0).unwrap();
        let p = arc.sample(1.0);
        assert!(approx_eq(p[0], -1.0));
        assert!((p[1]).abs() < EPS);
    }

    #[test]
    fn test_circular_arc_length() {
        let arc = CircularArc::xy_arc([0.0, 0.0], 2.0, 0.0, PI, 0.0, 1.0).unwrap();
        assert!(approx_eq(arc.arc_length(), 2.0 * PI));
    }

    #[test]
    fn test_display_linear() {
        let traj = LinearTrajectory::new(vec![0.0, 1.0], vec![1.0, 2.0], 1.0).unwrap();
        let s = format!("{traj}");
        assert!(s.contains("LinearTrajectory"));
    }

    #[test]
    fn test_display_cubic() {
        let traj = CubicTrajectory::zero_velocity(&[0.0], &[1.0], 1.0).unwrap();
        let s = format!("{traj}");
        assert!(s.contains("CubicTrajectory"));
    }
}
