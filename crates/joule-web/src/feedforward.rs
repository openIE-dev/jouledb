//! Feedforward Control — inverse model feedforward, gravity compensation,
//! friction compensation (Coulomb + viscous), velocity/acceleration feedforward,
//! combined feedforward+feedback architecture, lookup-table feedforward,
//! and measured disturbance rejection.
//!
//! Pure-Rust feedforward controllers for robotics, motion control,
//! and industrial automation workloads.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Feedforward control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedforwardError {
    /// Invalid parameter.
    InvalidParameter(String),
    /// Lookup table error.
    LookupError(String),
    /// Dimension mismatch.
    DimensionMismatch(String),
    /// Division by zero or singular condition.
    Singular(String),
}

impl std::fmt::Display for FeedforwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::LookupError(msg) => write!(f, "lookup error: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::Singular(msg) => write!(f, "singular: {msg}"),
        }
    }
}

impl std::error::Error for FeedforwardError {}

// ── Trajectory Point ────────────────────────────────────────────

/// Desired trajectory point with position, velocity, acceleration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Desired position for each DOF.
    pub position: Vec<f64>,
    /// Desired velocity for each DOF.
    pub velocity: Vec<f64>,
    /// Desired acceleration for each DOF.
    pub acceleration: Vec<f64>,
}

impl TrajectoryPoint {
    /// Create a single-DOF trajectory point.
    pub fn single(pos: f64, vel: f64, acc: f64) -> Self {
        Self {
            position: vec![pos],
            velocity: vec![vel],
            acceleration: vec![acc],
        }
    }

    /// Number of degrees of freedom.
    pub fn dof(&self) -> usize {
        self.position.len()
    }
}

// ── Friction Model ──────────────────────────────────────────────

/// Coulomb + viscous friction model: F_friction = F_c * sign(v) + b * v.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrictionModel {
    /// Coulomb friction force (positive).
    pub coulomb: f64,
    /// Viscous friction coefficient.
    pub viscous: f64,
    /// Stiction force (static friction, >= coulomb).
    pub stiction: f64,
    /// Velocity threshold for stiction breakaway.
    pub stiction_velocity: f64,
}

impl FrictionModel {
    /// Create a simple Coulomb + viscous model.
    pub fn new(coulomb: f64, viscous: f64) -> Result<Self, FeedforwardError> {
        if coulomb < 0.0 {
            return Err(FeedforwardError::InvalidParameter("coulomb must be >= 0".into()));
        }
        if viscous < 0.0 {
            return Err(FeedforwardError::InvalidParameter("viscous must be >= 0".into()));
        }
        Ok(Self {
            coulomb,
            viscous,
            stiction: coulomb,
            stiction_velocity: 0.01,
        })
    }

    /// Create with stiction.
    pub fn with_stiction(
        coulomb: f64,
        viscous: f64,
        stiction: f64,
        stiction_velocity: f64,
    ) -> Result<Self, FeedforwardError> {
        if stiction < coulomb {
            return Err(FeedforwardError::InvalidParameter(
                "stiction must be >= coulomb".into(),
            ));
        }
        Ok(Self { coulomb, viscous, stiction, stiction_velocity })
    }

    /// Compute friction compensation torque/force for a given velocity.
    pub fn compensate(&self, velocity: f64) -> f64 {
        let abs_v = velocity.abs();
        if abs_v < 1e-12 {
            return 0.0;
        }

        let coulomb_component = if abs_v < self.stiction_velocity {
            // Smooth transition from stiction to Coulomb.
            let t = abs_v / self.stiction_velocity;
            let smooth_force = self.stiction * (2.0 * t - t * t);
            smooth_force * velocity.signum()
        } else {
            self.coulomb * velocity.signum()
        };

        let viscous_component = self.viscous * velocity;

        coulomb_component + viscous_component
    }
}

// ── Gravity Compensation ────────────────────────────────────────

/// Gravity compensation for a robot arm joint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GravityCompensator {
    /// Mass of the link (kg).
    pub mass: f64,
    /// Distance from joint to center of mass (m).
    pub com_distance: f64,
    /// Gravitational acceleration (m/s^2).
    pub gravity: f64,
}

impl GravityCompensator {
    /// Create with standard gravity.
    pub fn new(mass: f64, com_distance: f64) -> Result<Self, FeedforwardError> {
        if mass < 0.0 {
            return Err(FeedforwardError::InvalidParameter("mass must be >= 0".into()));
        }
        Ok(Self {
            mass,
            com_distance,
            gravity: 9.81,
        })
    }

    /// Compute gravity torque at a given joint angle (radians from vertical).
    /// tau = m * g * d * sin(theta)
    pub fn torque(&self, angle: f64) -> f64 {
        self.mass * self.gravity * self.com_distance * angle.sin()
    }

    /// Multi-link gravity: sum of gravity torques.
    pub fn multi_link_torque(links: &[GravityCompensator], angles: &[f64]) -> Vec<f64> {
        // Simplified: each joint compensates its own link plus downstream links.
        let n = links.len().min(angles.len());
        let mut torques = vec![0.0; n];
        for i in 0..n {
            // Joint i supports links i..n.
            let mut cumulative_torque = 0.0;
            let mut angle_sum = 0.0;
            for j in i..n {
                angle_sum += angles[j];
                cumulative_torque += links[j].mass * links[j].gravity
                    * links[j].com_distance * angle_sum.sin();
            }
            torques[i] = cumulative_torque;
        }
        torques
    }
}

// ── Velocity / Acceleration Feedforward ─────────────────────────

/// Velocity and acceleration feedforward controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VelocityAccelFeedforward {
    /// Velocity feedforward gain (kv).
    pub kv: f64,
    /// Acceleration feedforward gain (ka).
    pub ka: f64,
    /// Static offset (e.g., gravity at a specific operating point).
    pub static_offset: f64,
}

impl VelocityAccelFeedforward {
    pub fn new(kv: f64, ka: f64) -> Self {
        Self { kv, ka, static_offset: 0.0 }
    }

    /// Compute feedforward from trajectory point.
    pub fn compute(&self, desired_vel: f64, desired_acc: f64) -> f64 {
        self.kv * desired_vel + self.ka * desired_acc + self.static_offset
    }

    /// Compute for multi-DOF.
    pub fn compute_multi(&self, trajectory: &TrajectoryPoint) -> Vec<f64> {
        trajectory.velocity.iter()
            .zip(&trajectory.acceleration)
            .map(|(v, a)| self.kv * v + self.ka * a + self.static_offset)
            .collect()
    }
}

// ── Lookup Table Feedforward ────────────────────────────────────

/// Breakpoint-based feedforward lookup table with linear interpolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookupFeedforward {
    /// Input breakpoints (must be sorted ascending).
    pub breakpoints: Vec<f64>,
    /// Output values at each breakpoint.
    pub values: Vec<f64>,
}

impl LookupFeedforward {
    /// Create from breakpoints and values.
    pub fn new(breakpoints: Vec<f64>, values: Vec<f64>) -> Result<Self, FeedforwardError> {
        if breakpoints.len() != values.len() {
            return Err(FeedforwardError::LookupError(
                "breakpoints and values must have same length".into(),
            ));
        }
        if breakpoints.len() < 2 {
            return Err(FeedforwardError::LookupError(
                "need at least 2 breakpoints".into(),
            ));
        }
        // Verify sorted.
        for i in 1..breakpoints.len() {
            if breakpoints[i] <= breakpoints[i - 1] {
                return Err(FeedforwardError::LookupError(
                    "breakpoints must be strictly ascending".into(),
                ));
            }
        }
        Ok(Self { breakpoints, values })
    }

    /// Interpolate/extrapolate to get feedforward value.
    pub fn lookup(&self, input: f64) -> f64 {
        let n = self.breakpoints.len();

        // Below range: extrapolate from first segment.
        if input <= self.breakpoints[0] {
            let slope = (self.values[1] - self.values[0])
                / (self.breakpoints[1] - self.breakpoints[0]);
            return self.values[0] + slope * (input - self.breakpoints[0]);
        }

        // Above range: extrapolate from last segment.
        if input >= self.breakpoints[n - 1] {
            let slope = (self.values[n - 1] - self.values[n - 2])
                / (self.breakpoints[n - 1] - self.breakpoints[n - 2]);
            return self.values[n - 1] + slope * (input - self.breakpoints[n - 1]);
        }

        // Binary search for segment.
        let mut lo = 0;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if input < self.breakpoints[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        // Linear interpolation.
        let t = (input - self.breakpoints[lo])
            / (self.breakpoints[hi] - self.breakpoints[lo]);
        self.values[lo] + t * (self.values[hi] - self.values[lo])
    }
}

// ── Inverse Model Feedforward ───────────────────────────────────

/// Inverse dynamics feedforward for a simple mass-spring-damper system.
/// F = m*a + c*v + k*x
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InverseDynamics {
    /// Mass.
    pub mass: f64,
    /// Damping coefficient.
    pub damping: f64,
    /// Spring constant.
    pub stiffness: f64,
}

impl InverseDynamics {
    pub fn new(mass: f64, damping: f64, stiffness: f64) -> Result<Self, FeedforwardError> {
        if mass <= 0.0 {
            return Err(FeedforwardError::InvalidParameter("mass must be > 0".into()));
        }
        Ok(Self { mass, damping, stiffness })
    }

    /// Compute feedforward force from desired trajectory.
    pub fn compute(&self, trajectory: &TrajectoryPoint) -> Result<Vec<f64>, FeedforwardError> {
        let n = trajectory.dof();
        if trajectory.velocity.len() != n || trajectory.acceleration.len() != n {
            return Err(FeedforwardError::DimensionMismatch(
                "trajectory dimensions inconsistent".into(),
            ));
        }
        Ok((0..n).map(|i| {
            self.mass * trajectory.acceleration[i]
                + self.damping * trajectory.velocity[i]
                + self.stiffness * trajectory.position[i]
        }).collect())
    }
}

// ── Combined Feedforward + Feedback ─────────────────────────────

/// Architecture combining feedforward and feedback (PID-like).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedforwardFeedback {
    /// Feedforward gains: [kff_pos, kff_vel, kff_acc].
    pub kff: [f64; 3],
    /// Feedback gains: [kp, ki, kd].
    pub kfb: [f64; 3],
    /// Sample time.
    pub dt: f64,
    /// Accumulated integral error.
    pub integral: f64,
    /// Previous error.
    pub prev_error: f64,
}

impl FeedforwardFeedback {
    /// Create with gains.
    pub fn new(kff: [f64; 3], kfb: [f64; 3], dt: f64) -> Result<Self, FeedforwardError> {
        if dt <= 0.0 {
            return Err(FeedforwardError::InvalidParameter("dt must be > 0".into()));
        }
        Ok(Self {
            kff, kfb, dt,
            integral: 0.0,
            prev_error: 0.0,
        })
    }

    /// Compute combined output.
    pub fn compute(
        &mut self,
        desired_pos: f64,
        desired_vel: f64,
        desired_acc: f64,
        actual_pos: f64,
    ) -> f64 {
        // Feedforward.
        let ff = self.kff[0] * desired_pos
            + self.kff[1] * desired_vel
            + self.kff[2] * desired_acc;

        // Feedback.
        let error = desired_pos - actual_pos;
        self.integral += error * self.dt;
        let derivative = (error - self.prev_error) / self.dt;
        let fb = self.kfb[0] * error
            + self.kfb[1] * self.integral
            + self.kfb[2] * derivative;
        self.prev_error = error;

        ff + fb
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ── Disturbance Rejection ───────────────────────────────────────

/// Feedforward disturbance rejection using measured disturbance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisturbanceRejection {
    /// Disturbance-to-output gain (transfer function gain at DC).
    pub disturbance_gain: f64,
    /// Plant input-to-output gain at DC.
    pub plant_gain: f64,
    /// Low-pass filter coefficient for disturbance measurement.
    pub filter_alpha: f64,
    /// Filtered disturbance.
    pub filtered_disturbance: f64,
}

impl DisturbanceRejection {
    pub fn new(
        disturbance_gain: f64,
        plant_gain: f64,
        filter_alpha: f64,
    ) -> Result<Self, FeedforwardError> {
        if plant_gain.abs() < 1e-14 {
            return Err(FeedforwardError::Singular("plant gain is zero".into()));
        }
        if !(0.0..=1.0).contains(&filter_alpha) {
            return Err(FeedforwardError::InvalidParameter(
                "filter_alpha must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            disturbance_gain,
            plant_gain,
            filter_alpha,
            filtered_disturbance: 0.0,
        })
    }

    /// Compute compensation signal from measured disturbance.
    pub fn compensate(&mut self, measured_disturbance: f64) -> f64 {
        // Low-pass filter the disturbance measurement.
        self.filtered_disturbance = self.filter_alpha * self.filtered_disturbance
            + (1.0 - self.filter_alpha) * measured_disturbance;

        // Feedforward compensation: u_ff = -(Gd / Gp) * d
        -(self.disturbance_gain / self.plant_gain) * self.filtered_disturbance
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.filtered_disturbance = 0.0;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_friction_zero_velocity() {
        let fm = FrictionModel::new(1.0, 0.5).unwrap();
        assert!(approx(fm.compensate(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_friction_positive_velocity() {
        let fm = FrictionModel::new(1.0, 0.5).unwrap();
        let f = fm.compensate(2.0);
        // Coulomb + viscous = 1.0 + 0.5*2 = 2.0
        assert!(approx(f, 2.0, 1e-4));
    }

    #[test]
    fn test_friction_negative_velocity() {
        let fm = FrictionModel::new(1.0, 0.5).unwrap();
        let f = fm.compensate(-2.0);
        // -1.0 + 0.5*(-2) = -2.0
        assert!(approx(f, -2.0, 1e-4));
    }

    #[test]
    fn test_friction_stiction() {
        let fm = FrictionModel::with_stiction(1.0, 0.5, 2.0, 0.1).unwrap();
        let f = fm.compensate(0.005); // Below stiction_velocity
        // Should be between 0 and stiction force.
        assert!(f > 0.0);
        assert!(f < 2.5);
    }

    #[test]
    fn test_friction_invalid_params() {
        assert!(FrictionModel::new(-1.0, 0.5).is_err());
        assert!(FrictionModel::new(1.0, -0.5).is_err());
        assert!(FrictionModel::with_stiction(2.0, 0.5, 1.0, 0.1).is_err());
    }

    #[test]
    fn test_gravity_compensation_vertical() {
        let gc = GravityCompensator::new(1.0, 0.5).unwrap();
        // At vertical (0 degrees), sin(0) = 0 => no torque.
        assert!(approx(gc.torque(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_gravity_compensation_horizontal() {
        let gc = GravityCompensator::new(1.0, 0.5).unwrap();
        // At 90 degrees, sin(pi/2) = 1 => max torque.
        let t = gc.torque(std::f64::consts::FRAC_PI_2);
        assert!(approx(t, 1.0 * 9.81 * 0.5, 1e-4));
    }

    #[test]
    fn test_gravity_multi_link() {
        let links = vec![
            GravityCompensator::new(1.0, 0.3).unwrap(),
            GravityCompensator::new(0.5, 0.2).unwrap(),
        ];
        let angles = vec![0.0, std::f64::consts::FRAC_PI_2];
        let torques = GravityCompensator::multi_link_torque(&links, &angles);
        assert_eq!(torques.len(), 2);
        // Joint 0 supports both links.
        assert!(torques[0].abs() > 0.0);
    }

    #[test]
    fn test_velocity_accel_feedforward() {
        let ff = VelocityAccelFeedforward::new(1.0, 0.5);
        let output = ff.compute(2.0, 3.0);
        assert!(approx(output, 1.0 * 2.0 + 0.5 * 3.0, 1e-10));
    }

    #[test]
    fn test_velocity_accel_multi_dof() {
        let ff = VelocityAccelFeedforward::new(1.0, 0.5);
        let tp = TrajectoryPoint {
            position: vec![0.0, 0.0],
            velocity: vec![2.0, 3.0],
            acceleration: vec![1.0, 2.0],
        };
        let output = ff.compute_multi(&tp);
        assert!(approx(output[0], 2.5, 1e-10));
        assert!(approx(output[1], 4.0, 1e-10));
    }

    #[test]
    fn test_lookup_feedforward_interpolation() {
        let lut = LookupFeedforward::new(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 10.0, 30.0, 60.0],
        ).unwrap();
        // At breakpoint.
        assert!(approx(lut.lookup(1.0), 10.0, 1e-10));
        // Midpoint interpolation.
        assert!(approx(lut.lookup(0.5), 5.0, 1e-10));
        assert!(approx(lut.lookup(1.5), 20.0, 1e-10));
    }

    #[test]
    fn test_lookup_feedforward_extrapolation() {
        let lut = LookupFeedforward::new(
            vec![0.0, 1.0],
            vec![0.0, 10.0],
        ).unwrap();
        // Below range.
        assert!(approx(lut.lookup(-1.0), -10.0, 1e-10));
        // Above range.
        assert!(approx(lut.lookup(2.0), 20.0, 1e-10));
    }

    #[test]
    fn test_lookup_feedforward_invalid() {
        // Too few breakpoints.
        assert!(LookupFeedforward::new(vec![0.0], vec![0.0]).is_err());
        // Unsorted.
        assert!(LookupFeedforward::new(vec![2.0, 1.0], vec![0.0, 1.0]).is_err());
        // Mismatched length.
        assert!(LookupFeedforward::new(vec![0.0, 1.0], vec![0.0]).is_err());
    }

    #[test]
    fn test_inverse_dynamics() {
        let id = InverseDynamics::new(2.0, 0.5, 1.0).unwrap();
        let tp = TrajectoryPoint::single(1.0, 2.0, 3.0);
        let forces = id.compute(&tp).unwrap();
        // F = 2*3 + 0.5*2 + 1*1 = 6 + 1 + 1 = 8
        assert!(approx(forces[0], 8.0, 1e-10));
    }

    #[test]
    fn test_inverse_dynamics_invalid_mass() {
        assert!(InverseDynamics::new(0.0, 0.5, 1.0).is_err());
        assert!(InverseDynamics::new(-1.0, 0.5, 1.0).is_err());
    }

    #[test]
    fn test_feedforward_feedback_combined() {
        let mut ffb = FeedforwardFeedback::new(
            [0.0, 1.0, 0.0],  // velocity feedforward
            [10.0, 0.0, 0.0], // proportional feedback
            0.01,
        ).unwrap();
        let output = ffb.compute(10.0, 5.0, 0.0, 8.0);
        // ff = 0 + 1*5 + 0 = 5, fb = 10*(10-8) = 20, total = 25
        assert!(approx(output, 25.0, 1e-4));
    }

    #[test]
    fn test_feedforward_feedback_reset() {
        let mut ffb = FeedforwardFeedback::new(
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.01,
        ).unwrap();
        ffb.compute(10.0, 0.0, 0.0, 5.0);
        ffb.reset();
        assert!(approx(ffb.integral, 0.0, 1e-10));
        assert!(approx(ffb.prev_error, 0.0, 1e-10));
    }

    #[test]
    fn test_disturbance_rejection() {
        let mut dr = DisturbanceRejection::new(1.0, 2.0, 0.0).unwrap();
        let comp = dr.compensate(10.0);
        // u_ff = -(1/2)*10 = -5
        assert!(approx(comp, -5.0, 1e-4));
    }

    #[test]
    fn test_disturbance_rejection_filter() {
        let mut dr = DisturbanceRejection::new(1.0, 1.0, 0.9).unwrap();
        // Step disturbance.
        for _ in 0..100 {
            dr.compensate(10.0);
        }
        // Filtered should converge to 10.
        assert!(approx(dr.filtered_disturbance, 10.0, 0.5));
    }

    #[test]
    fn test_disturbance_rejection_invalid() {
        assert!(DisturbanceRejection::new(1.0, 0.0, 0.5).is_err()); // zero plant gain
        assert!(DisturbanceRejection::new(1.0, 1.0, 1.5).is_err()); // bad alpha
    }

    #[test]
    fn test_trajectory_point_dof() {
        let tp = TrajectoryPoint::single(1.0, 2.0, 3.0);
        assert_eq!(tp.dof(), 1);
    }

    #[test]
    fn test_gravity_compensator_invalid_mass() {
        assert!(GravityCompensator::new(-1.0, 0.5).is_err());
    }

    #[test]
    fn test_feedforward_feedback_invalid_dt() {
        assert!(FeedforwardFeedback::new([0.0; 3], [0.0; 3], 0.0).is_err());
    }

    #[test]
    fn test_disturbance_rejection_reset() {
        let mut dr = DisturbanceRejection::new(1.0, 1.0, 0.5).unwrap();
        dr.compensate(10.0);
        dr.reset();
        assert!(approx(dr.filtered_disturbance, 0.0, 1e-10));
    }
}
