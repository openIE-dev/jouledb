//! Joint space representation — joint limits, velocity/acceleration bounds,
//! configuration space, and singularity detection.
//!
//! Models the configuration space of serial manipulators including joint
//! constraints, distance metrics, interpolation, and singularity analysis.

use std::f64::consts::PI;

// ── Errors ──────────────────────────────────────────────────────

/// Joint-space errors.
#[derive(Debug, Clone, PartialEq)]
pub enum JointSpaceError {
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid limits (min > max).
    InvalidLimits { joint: usize, min: f64, max: f64 },
    /// Configuration violates one or more joint limits.
    LimitViolation { joint: usize, value: f64, min: f64, max: f64 },
    /// Empty configuration space.
    Empty,
    /// Singularity detected.
    Singularity(String),
}

impl std::fmt::Display for JointSpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidLimits { joint, min, max } => {
                write!(f, "invalid limits for joint {joint}: [{min:.4}, {max:.4}]")
            }
            Self::LimitViolation { joint, value, min, max } => {
                write!(f, "joint {joint} value {value:.4} outside [{min:.4}, {max:.4}]")
            }
            Self::Empty => write!(f, "empty configuration space"),
            Self::Singularity(msg) => write!(f, "singularity: {msg}"),
        }
    }
}

impl std::error::Error for JointSpaceError {}

// ── Joint Limits ───────────────────────────────────────────────

/// Limits for a single joint axis.
#[derive(Debug, Clone, PartialEq)]
pub struct JointLimits {
    /// Position lower bound.
    pub pos_min: f64,
    /// Position upper bound.
    pub pos_max: f64,
    /// Maximum absolute velocity.
    pub vel_max: f64,
    /// Maximum absolute acceleration.
    pub acc_max: f64,
    /// Maximum absolute jerk (0 = unlimited).
    pub jerk_max: f64,
    /// Whether the joint wraps (continuous rotation).
    pub continuous: bool,
}

impl JointLimits {
    /// Create limits for a revolute joint.
    pub fn revolute(pos_min: f64, pos_max: f64) -> Self {
        Self {
            pos_min,
            pos_max,
            vel_max: 3.0,     // rad/s default
            acc_max: 10.0,    // rad/s^2 default
            jerk_max: 0.0,
            continuous: false,
        }
    }

    /// Create limits for a prismatic joint.
    pub fn prismatic(pos_min: f64, pos_max: f64) -> Self {
        Self {
            pos_min,
            pos_max,
            vel_max: 1.0,     // m/s default
            acc_max: 5.0,     // m/s^2 default
            jerk_max: 0.0,
            continuous: false,
        }
    }

    /// Create limits for a continuous (wrapping) joint.
    pub fn continuous_revolute() -> Self {
        Self {
            pos_min: -PI,
            pos_max: PI,
            vel_max: 6.0,
            acc_max: 20.0,
            jerk_max: 0.0,
            continuous: true,
        }
    }

    /// Set velocity limit.
    pub fn with_vel_max(mut self, v: f64) -> Self {
        self.vel_max = v;
        self
    }

    /// Set acceleration limit.
    pub fn with_acc_max(mut self, a: f64) -> Self {
        self.acc_max = a;
        self
    }

    /// Set jerk limit.
    pub fn with_jerk_max(mut self, j: f64) -> Self {
        self.jerk_max = j;
        self
    }

    /// Range of the joint.
    pub fn range(&self) -> f64 {
        self.pos_max - self.pos_min
    }

    /// Center of the joint range.
    pub fn center(&self) -> f64 {
        (self.pos_min + self.pos_max) / 2.0
    }

    /// Check whether `q` is within position limits.
    pub fn in_position_limits(&self, q: f64) -> bool {
        if self.continuous {
            true
        } else {
            q >= self.pos_min && q <= self.pos_max
        }
    }

    /// Clamp `q` to position limits.
    pub fn clamp(&self, q: f64) -> f64 {
        if self.continuous {
            wrap_angle(q)
        } else {
            q.clamp(self.pos_min, self.pos_max)
        }
    }

    /// Normalized position in [0, 1] within the range.
    pub fn normalize(&self, q: f64) -> f64 {
        let range = self.range();
        if range.abs() < 1e-15 {
            0.5
        } else {
            (q - self.pos_min) / range
        }
    }

    /// How close is `q` to the nearest limit, as a fraction of range (0 = at limit, 1 = center).
    pub fn limit_proximity(&self, q: f64) -> f64 {
        let range = self.range();
        if range.abs() < 1e-15 {
            return 0.0;
        }
        let d_lo = (q - self.pos_min).abs();
        let d_hi = (self.pos_max - q).abs();
        let d_min = d_lo.min(d_hi);
        (d_min / (range / 2.0)).min(1.0)
    }
}

impl std::fmt::Display for JointLimits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.continuous { "continuous" } else { "bounded" };
        write!(
            f,
            "JointLimits({kind}, [{:.3}, {:.3}], v≤{:.2}, a≤{:.2})",
            self.pos_min, self.pos_max, self.vel_max, self.acc_max,
        )
    }
}

// ── Configuration ──────────────────────────────────────────────

/// A point in joint space (a configuration).
#[derive(Debug, Clone, PartialEq)]
pub struct JointConfig {
    /// Joint values.
    pub values: Vec<f64>,
}

impl JointConfig {
    /// Create from a slice.
    pub fn new(values: &[f64]) -> Self {
        Self { values: values.to_vec() }
    }

    /// Zero configuration of `n` DOFs.
    pub fn zeros(n: usize) -> Self {
        Self { values: vec![0.0; n] }
    }

    /// Number of DOFs.
    pub fn ndof(&self) -> usize {
        self.values.len()
    }

    /// Euclidean distance to another configuration.
    pub fn distance(&self, other: &JointConfig) -> f64 {
        assert_eq!(self.values.len(), other.values.len());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Weighted distance (each DOF has a weight).
    pub fn weighted_distance(&self, other: &JointConfig, weights: &[f64]) -> f64 {
        assert_eq!(self.values.len(), other.values.len());
        assert_eq!(self.values.len(), weights.len());
        self.values
            .iter()
            .zip(other.values.iter())
            .zip(weights.iter())
            .map(|((a, b), w)| w * (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Linear interpolation toward `other` at parameter `t` in [0, 1].
    pub fn lerp(&self, other: &JointConfig, t: f64) -> JointConfig {
        assert_eq!(self.values.len(), other.values.len());
        let t_clamp = t.clamp(0.0, 1.0);
        let vals: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + t_clamp * (b - a))
            .collect();
        JointConfig { values: vals }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &JointConfig) -> JointConfig {
        assert_eq!(self.values.len(), other.values.len());
        let vals: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();
        JointConfig { values: vals }
    }

    /// Scale all values.
    pub fn scale(&self, s: f64) -> JointConfig {
        let vals: Vec<f64> = self.values.iter().map(|v| v * s).collect();
        JointConfig { values: vals }
    }

    /// Infinity-norm (max absolute value).
    pub fn linf_norm(&self) -> f64 {
        self.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }
}

impl std::fmt::Display for JointConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JointConfig[")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        write!(f, "]")
    }
}

// ── Configuration Space ────────────────────────────────────────

/// Configuration space defined by per-joint limits.
#[derive(Debug, Clone)]
pub struct ConfigurationSpace {
    limits: Vec<JointLimits>,
}

impl ConfigurationSpace {
    /// Create from joint limits.
    pub fn new(limits: Vec<JointLimits>) -> Result<Self, JointSpaceError> {
        if limits.is_empty() {
            return Err(JointSpaceError::Empty);
        }
        for (i, lim) in limits.iter().enumerate() {
            if !lim.continuous && lim.pos_min > lim.pos_max {
                return Err(JointSpaceError::InvalidLimits {
                    joint: i,
                    min: lim.pos_min,
                    max: lim.pos_max,
                });
            }
        }
        Ok(Self { limits })
    }

    /// Number of DOFs.
    pub fn ndof(&self) -> usize {
        self.limits.len()
    }

    /// Reference to limits.
    pub fn limits(&self) -> &[JointLimits] {
        &self.limits
    }

    /// Check whether a configuration is within bounds.
    pub fn is_valid(&self, config: &JointConfig) -> bool {
        if config.ndof() != self.ndof() {
            return false;
        }
        config
            .values
            .iter()
            .zip(self.limits.iter())
            .all(|(q, lim)| lim.in_position_limits(*q))
    }

    /// Validate a configuration, returning detailed error if invalid.
    pub fn validate(&self, config: &JointConfig) -> Result<(), JointSpaceError> {
        if config.ndof() != self.ndof() {
            return Err(JointSpaceError::DimensionMismatch {
                expected: self.ndof(),
                got: config.ndof(),
            });
        }
        for (i, (q, lim)) in config.values.iter().zip(self.limits.iter()).enumerate() {
            if !lim.in_position_limits(*q) {
                return Err(JointSpaceError::LimitViolation {
                    joint: i,
                    value: *q,
                    min: lim.pos_min,
                    max: lim.pos_max,
                });
            }
        }
        Ok(())
    }

    /// Clamp a configuration to be within bounds.
    pub fn clamp(&self, config: &JointConfig) -> JointConfig {
        let vals: Vec<f64> = config
            .values
            .iter()
            .zip(self.limits.iter())
            .map(|(q, lim)| lim.clamp(*q))
            .collect();
        JointConfig { values: vals }
    }

    /// Center configuration.
    pub fn center(&self) -> JointConfig {
        let vals: Vec<f64> = self.limits.iter().map(|l| l.center()).collect();
        JointConfig { values: vals }
    }

    /// Random configuration (deterministic via a simple hash-based approach).
    pub fn sample_config(&self, seed: u64) -> JointConfig {
        let mut state = seed;
        let vals: Vec<f64> = self
            .limits
            .iter()
            .map(|lim| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (state >> 33) as f64 / (1u64 << 31) as f64;
                lim.pos_min + u * lim.range()
            })
            .collect();
        JointConfig { values: vals }
    }

    /// Maximum velocity vector.
    pub fn max_velocities(&self) -> Vec<f64> {
        self.limits.iter().map(|l| l.vel_max).collect()
    }

    /// Maximum acceleration vector.
    pub fn max_accelerations(&self) -> Vec<f64> {
        self.limits.iter().map(|l| l.acc_max).collect()
    }

    /// Volume of the configuration space (product of ranges).
    pub fn volume(&self) -> f64 {
        self.limits.iter().map(|l| l.range()).product()
    }
}

impl std::fmt::Display for ConfigurationSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConfigurationSpace({} DOF, vol={:.4})", self.ndof(), self.volume())
    }
}

// ── Singularity Detection ──────────────────────────────────────

/// Singularity analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct SingularityInfo {
    /// Manipulability index (determinant of J J^T).
    pub manipulability: f64,
    /// Condition number of the Jacobian.
    pub condition_number: f64,
    /// Whether the configuration is near a singularity.
    pub is_singular: bool,
    /// Threshold used.
    pub threshold: f64,
}

impl SingularityInfo {
    /// Create from a 3-DOF Jacobian (3x3 row-major).
    pub fn from_jacobian_3x3(jac: &[f64; 9], threshold: f64) -> Self {
        let det = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7])
            - jac[1] * (jac[3] * jac[8] - jac[5] * jac[6])
            + jac[2] * (jac[3] * jac[7] - jac[4] * jac[6]);

        let manipulability = det.abs();

        // Frobenius norm as proxy for condition number
        let frob: f64 = jac.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cond = if manipulability > 1e-15 {
            frob / manipulability
        } else {
            f64::INFINITY
        };

        Self {
            manipulability,
            condition_number: cond,
            is_singular: manipulability < threshold,
            threshold,
        }
    }

    /// Check from a flat m x n Jacobian (row-major) using a simplified measure.
    ///
    /// Computes J J^T, then estimates manipulability from diagonal product.
    pub fn from_jacobian(jac: &[f64], rows: usize, cols: usize, threshold: f64) -> Self {
        // J J^T (rows x rows)
        let mut jjt = vec![0.0; rows * rows];
        for r in 0..rows {
            for c in 0..rows {
                let mut s = 0.0;
                for k in 0..cols {
                    s += jac[r * cols + k] * jac[c * cols + k];
                }
                jjt[r * rows + c] = s;
            }
        }
        // Manipulability: sqrt(det(J J^T)) — approximate via diagonal product for small matrices
        let diag_product: f64 = (0..rows).map(|i| jjt[i * rows + i]).product();
        let manipulability = diag_product.abs().sqrt();

        let frob: f64 = jac.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cond = if manipulability > 1e-15 {
            frob / manipulability
        } else {
            f64::INFINITY
        };

        Self {
            manipulability,
            condition_number: cond,
            is_singular: manipulability < threshold,
            threshold,
        }
    }
}

impl std::fmt::Display for SingularityInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_singular { "SINGULAR" } else { "OK" };
        write!(
            f,
            "Singularity({status}, μ={:.6}, κ={:.2})",
            self.manipulability, self.condition_number,
        )
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Wrap angle to `[-PI, PI)`.
fn wrap_angle(a: f64) -> f64 {
    let mut v = a % (2.0 * PI);
    if v >= PI {
        v -= 2.0 * PI;
    } else if v < -PI {
        v += 2.0 * PI;
    }
    v
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_revolute_limits() {
        let lim = JointLimits::revolute(-PI, PI);
        assert!(lim.in_position_limits(0.0));
        assert!(lim.in_position_limits(-PI));
        assert!(!lim.in_position_limits(4.0));
    }

    #[test]
    fn test_prismatic_limits() {
        let lim = JointLimits::prismatic(0.0, 0.5);
        assert!(lim.in_position_limits(0.25));
        assert!(!lim.in_position_limits(-0.1));
    }

    #[test]
    fn test_continuous_limits() {
        let lim = JointLimits::continuous_revolute();
        assert!(lim.in_position_limits(100.0)); // always true
    }

    #[test]
    fn test_limit_clamp() {
        let lim = JointLimits::revolute(-1.0, 1.0);
        assert!(approx_eq(lim.clamp(5.0), 1.0));
        assert!(approx_eq(lim.clamp(-5.0), -1.0));
        assert!(approx_eq(lim.clamp(0.5), 0.5));
    }

    #[test]
    fn test_limit_normalize() {
        let lim = JointLimits::revolute(0.0, 10.0);
        assert!(approx_eq(lim.normalize(5.0), 0.5));
        assert!(approx_eq(lim.normalize(0.0), 0.0));
        assert!(approx_eq(lim.normalize(10.0), 1.0));
    }

    #[test]
    fn test_limit_proximity() {
        let lim = JointLimits::revolute(0.0, 10.0);
        assert!(approx_eq(lim.limit_proximity(5.0), 1.0)); // center
        assert!(approx_eq(lim.limit_proximity(0.0), 0.0)); // at limit
    }

    #[test]
    fn test_joint_config_distance() {
        let a = JointConfig::new(&[0.0, 0.0]);
        let b = JointConfig::new(&[3.0, 4.0]);
        assert!(approx_eq(a.distance(&b), 5.0));
    }

    #[test]
    fn test_joint_config_lerp() {
        let a = JointConfig::new(&[0.0, 0.0]);
        let b = JointConfig::new(&[10.0, 20.0]);
        let mid = a.lerp(&b, 0.5);
        assert!(approx_eq(mid.values[0], 5.0));
        assert!(approx_eq(mid.values[1], 10.0));
    }

    #[test]
    fn test_joint_config_add() {
        let a = JointConfig::new(&[1.0, 2.0]);
        let b = JointConfig::new(&[3.0, 4.0]);
        let c = a.add(&b);
        assert!(approx_eq(c.values[0], 4.0));
        assert!(approx_eq(c.values[1], 6.0));
    }

    #[test]
    fn test_joint_config_scale() {
        let a = JointConfig::new(&[1.0, 2.0]);
        let b = a.scale(3.0);
        assert!(approx_eq(b.values[0], 3.0));
        assert!(approx_eq(b.values[1], 6.0));
    }

    #[test]
    fn test_config_space_valid() {
        let cs = ConfigurationSpace::new(vec![
            JointLimits::revolute(-1.0, 1.0),
            JointLimits::revolute(-2.0, 2.0),
        ])
        .unwrap();
        let cfg = JointConfig::new(&[0.0, 0.0]);
        assert!(cs.is_valid(&cfg));
    }

    #[test]
    fn test_config_space_invalid() {
        let cs = ConfigurationSpace::new(vec![JointLimits::revolute(-1.0, 1.0)]).unwrap();
        let cfg = JointConfig::new(&[5.0]);
        assert!(!cs.is_valid(&cfg));
    }

    #[test]
    fn test_config_space_clamp() {
        let cs = ConfigurationSpace::new(vec![JointLimits::revolute(-1.0, 1.0)]).unwrap();
        let cfg = JointConfig::new(&[5.0]);
        let clamped = cs.clamp(&cfg);
        assert!(approx_eq(clamped.values[0], 1.0));
    }

    #[test]
    fn test_config_space_center() {
        let cs = ConfigurationSpace::new(vec![
            JointLimits::revolute(0.0, 2.0),
            JointLimits::revolute(-4.0, 4.0),
        ])
        .unwrap();
        let center = cs.center();
        assert!(approx_eq(center.values[0], 1.0));
        assert!(approx_eq(center.values[1], 0.0));
    }

    #[test]
    fn test_config_space_volume() {
        let cs = ConfigurationSpace::new(vec![
            JointLimits::revolute(0.0, 2.0),
            JointLimits::revolute(0.0, 3.0),
        ])
        .unwrap();
        assert!(approx_eq(cs.volume(), 6.0));
    }

    #[test]
    fn test_empty_config_space() {
        let r = ConfigurationSpace::new(vec![]);
        assert!(matches!(r, Err(JointSpaceError::Empty)));
    }

    #[test]
    fn test_singularity_3x3_identity() {
        let jac = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let info = SingularityInfo::from_jacobian_3x3(&jac, 0.01);
        assert!(!info.is_singular);
        assert!(approx_eq(info.manipulability, 1.0));
    }

    #[test]
    fn test_singularity_3x3_singular() {
        let jac = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let info = SingularityInfo::from_jacobian_3x3(&jac, 0.01);
        assert!(info.is_singular);
    }

    #[test]
    fn test_display_config() {
        let c = JointConfig::new(&[1.0, 2.0, 3.0]);
        let s = format!("{c}");
        assert!(s.contains("JointConfig"));
    }

    #[test]
    fn test_display_config_space() {
        let cs = ConfigurationSpace::new(vec![JointLimits::revolute(-1.0, 1.0)]).unwrap();
        let s = format!("{cs}");
        assert!(s.contains("ConfigurationSpace"));
    }

    #[test]
    fn test_sample_config() {
        let cs = ConfigurationSpace::new(vec![
            JointLimits::revolute(-1.0, 1.0),
            JointLimits::revolute(-2.0, 2.0),
        ])
        .unwrap();
        let cfg = cs.sample_config(42);
        assert!(cs.is_valid(&cfg));
    }
}
