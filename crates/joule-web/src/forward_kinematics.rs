//! Forward kinematics — DH parameter chain, homogeneous transforms,
//! end-effector pose computation, and multi-joint chain evaluation.
//!
//! Computes the position and orientation of each link frame in a serial
//! manipulator using the Denavit-Hartenberg convention.  All transforms
//! are 4x4 homogeneous matrices stored in column-major order.

use std::f64::consts::PI;

// ── Errors ──────────────────────────────────────────────────────

/// Errors produced by forward-kinematics operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FkError {
    /// Chain has zero joints.
    EmptyChain,
    /// Joint index out of range.
    JointIndexOutOfRange { index: usize, count: usize },
    /// Joint value violates configured limits.
    JointLimitViolation { index: usize, value: f64, min: f64, max: f64 },
    /// Singular or degenerate frame.
    DegenerateFrame(String),
}

impl std::fmt::Display for FkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "forward kinematics chain is empty"),
            Self::JointIndexOutOfRange { index, count } => {
                write!(f, "joint index {index} out of range (chain has {count} joints)")
            }
            Self::JointLimitViolation { index, value, min, max } => {
                write!(f, "joint {index} value {value:.4} outside [{min:.4}, {max:.4}]")
            }
            Self::DegenerateFrame(msg) => write!(f, "degenerate frame: {msg}"),
        }
    }
}

impl std::error::Error for FkError {}

// ── 4x4 Homogeneous Transform ──────────────────────────────────

/// A 4x4 homogeneous transformation matrix stored in row-major order.
///
/// Layout: `[r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, 0, 0, 0, 1]`
#[derive(Debug, Clone, PartialEq)]
pub struct Transform4 {
    /// Row-major 4x4 elements.
    pub m: [f64; 16],
}

impl Transform4 {
    /// Identity transform.
    pub fn identity() -> Self {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self { m }
    }

    /// Build from rotation matrix (3x3, row-major) and translation.
    pub fn from_rotation_translation(rot: &[f64; 9], tx: f64, ty: f64, tz: f64) -> Self {
        let mut m = [0.0; 16];
        m[0] = rot[0]; m[1] = rot[1]; m[2] = rot[2]; m[3] = tx;
        m[4] = rot[3]; m[5] = rot[4]; m[6] = rot[5]; m[7] = ty;
        m[8] = rot[6]; m[9] = rot[7]; m[10] = rot[8]; m[11] = tz;
        m[15] = 1.0;
        Self { m }
    }

    /// Pure translation transform.
    pub fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        let mut t = Self::identity();
        t.m[3] = tx;
        t.m[7] = ty;
        t.m[11] = tz;
        t
    }

    /// Rotation about the X axis.
    pub fn rot_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let mut t = Self::identity();
        t.m[5] = c;  t.m[6] = -s;
        t.m[9] = s;  t.m[10] = c;
        t
    }

    /// Rotation about the Z axis.
    pub fn rot_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let mut t = Self::identity();
        t.m[0] = c;  t.m[1] = -s;
        t.m[4] = s;  t.m[5] = c;
        t
    }

    /// Multiply two 4x4 transforms: `self * rhs`.
    pub fn mul(&self, rhs: &Transform4) -> Transform4 {
        let a = &self.m;
        let b = &rhs.m;
        let mut o = [0.0; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += a[row * 4 + k] * b[k * 4 + col];
                }
                o[row * 4 + col] = sum;
            }
        }
        Transform4 { m: o }
    }

    /// Extract the translation component.
    pub fn position(&self) -> [f64; 3] {
        [self.m[3], self.m[7], self.m[11]]
    }

    /// Extract the 3x3 rotation matrix (row-major).
    pub fn rotation(&self) -> [f64; 9] {
        [
            self.m[0], self.m[1], self.m[2],
            self.m[4], self.m[5], self.m[6],
            self.m[8], self.m[9], self.m[10],
        ]
    }

    /// Compute the inverse transform (assumes valid rigid-body transform).
    pub fn inverse(&self) -> Self {
        let r = self.rotation();
        // R^T
        let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
        let p = self.position();
        // -R^T * p
        let tx = -(rt[0] * p[0] + rt[1] * p[1] + rt[2] * p[2]);
        let ty = -(rt[3] * p[0] + rt[4] * p[1] + rt[5] * p[2]);
        let tz = -(rt[6] * p[0] + rt[7] * p[1] + rt[8] * p[2]);
        Self::from_rotation_translation(&rt, tx, ty, tz)
    }

    /// Convert rotation to ZYX Euler angles (roll, pitch, yaw) in radians.
    pub fn to_euler_zyx(&self) -> [f64; 3] {
        let r = self.rotation();
        let pitch = (-r[6]).asin();
        if pitch.cos().abs() > 1e-10 {
            let roll = r[7].atan2(r[8]);
            let yaw = r[3].atan2(r[0]);
            [roll, pitch, yaw]
        } else {
            // Gimbal lock
            let roll = 0.0;
            let yaw = (-r[1]).atan2(r[4]);
            [roll, pitch, yaw]
        }
    }

    /// Build the standard DH transform for one joint.
    ///
    /// T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    pub fn from_dh(theta: f64, d: f64, a: f64, alpha: f64) -> Self {
        Self::rot_z(theta)
            .mul(&Self::translation(0.0, 0.0, d))
            .mul(&Self::translation(a, 0.0, 0.0))
            .mul(&Self::rot_x(alpha))
    }
}

impl std::fmt::Display for Transform4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..4 {
            let i = row * 4;
            writeln!(
                f,
                "[{:8.4} {:8.4} {:8.4} {:8.4}]",
                self.m[i], self.m[i + 1], self.m[i + 2], self.m[i + 3],
            )?;
        }
        Ok(())
    }
}

// ── DH Joint Definition ────────────────────────────────────────

/// Type of joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    /// Revolute (rotational) joint — theta is the variable.
    Revolute,
    /// Prismatic (translational) joint — d is the variable.
    Prismatic,
}

/// A single DH joint definition.
#[derive(Debug, Clone, PartialEq)]
pub struct DhJoint {
    /// Fixed theta offset (radians).
    pub theta_offset: f64,
    /// Fixed d offset.
    pub d_offset: f64,
    /// Link length a.
    pub a: f64,
    /// Link twist alpha (radians).
    pub alpha: f64,
    /// Joint type.
    pub joint_type: JointType,
    /// Minimum joint value.
    pub min_val: f64,
    /// Maximum joint value.
    pub max_val: f64,
}

impl DhJoint {
    /// Create a revolute joint with the given DH constants.
    pub fn revolute(theta_offset: f64, d: f64, a: f64, alpha: f64) -> Self {
        Self {
            theta_offset,
            d_offset: d,
            a,
            alpha,
            joint_type: JointType::Revolute,
            min_val: -PI,
            max_val: PI,
        }
    }

    /// Create a prismatic joint with the given DH constants.
    pub fn prismatic(theta: f64, d_offset: f64, a: f64, alpha: f64) -> Self {
        Self {
            theta_offset: theta,
            d_offset,
            a,
            alpha,
            joint_type: JointType::Prismatic,
            min_val: 0.0,
            max_val: 1.0,
        }
    }

    /// Set joint limits.
    pub fn with_limits(mut self, min_val: f64, max_val: f64) -> Self {
        self.min_val = min_val;
        self.max_val = max_val;
        self
    }

    /// Compute the transform for this joint at the given joint value.
    pub fn transform(&self, q: f64) -> Transform4 {
        match self.joint_type {
            JointType::Revolute => {
                Transform4::from_dh(self.theta_offset + q, self.d_offset, self.a, self.alpha)
            }
            JointType::Prismatic => {
                Transform4::from_dh(self.theta_offset, self.d_offset + q, self.a, self.alpha)
            }
        }
    }

    /// Check whether `q` is within the configured limits.
    pub fn in_limits(&self, q: f64) -> bool {
        q >= self.min_val && q <= self.max_val
    }
}

impl std::fmt::Display for DhJoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.joint_type {
            JointType::Revolute => "R",
            JointType::Prismatic => "P",
        };
        write!(
            f,
            "{kind}(θ_off={:.3}, d_off={:.3}, a={:.3}, α={:.3})",
            self.theta_offset, self.d_offset, self.a, self.alpha,
        )
    }
}

// ── Forward Kinematics Chain ───────────────────────────────────

/// Forward kinematics solver for a serial chain described by DH parameters.
#[derive(Debug, Clone)]
pub struct ForwardKinematics {
    joints: Vec<DhJoint>,
    base_frame: Transform4,
    tool_frame: Transform4,
    enforce_limits: bool,
}

impl ForwardKinematics {
    /// Create an FK solver from a list of DH joints.
    pub fn new(joints: Vec<DhJoint>) -> Result<Self, FkError> {
        if joints.is_empty() {
            return Err(FkError::EmptyChain);
        }
        Ok(Self {
            joints,
            base_frame: Transform4::identity(),
            tool_frame: Transform4::identity(),
            enforce_limits: false,
        })
    }

    /// Set a custom base frame.
    pub fn with_base_frame(mut self, base: Transform4) -> Self {
        self.base_frame = base;
        self
    }

    /// Set a custom tool frame (attached beyond the last joint).
    pub fn with_tool_frame(mut self, tool: Transform4) -> Self {
        self.tool_frame = tool;
        self
    }

    /// Enable joint-limit enforcement.
    pub fn with_limit_enforcement(mut self, enforce: bool) -> Self {
        self.enforce_limits = enforce;
        self
    }

    /// Number of joints in the chain.
    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }

    /// Reference to the i-th joint definition.
    pub fn joint(&self, i: usize) -> Option<&DhJoint> {
        self.joints.get(i)
    }

    /// Compute the end-effector transform for joint values `q`.
    ///
    /// Returns `T_base * T_0 * T_1 * ... * T_{n-1} * T_tool`.
    pub fn end_effector(&self, q: &[f64]) -> Result<Transform4, FkError> {
        if q.len() != self.joints.len() {
            return Err(FkError::JointIndexOutOfRange {
                index: q.len(),
                count: self.joints.len(),
            });
        }
        if self.enforce_limits {
            for (i, (joint, &val)) in self.joints.iter().zip(q.iter()).enumerate() {
                if !joint.in_limits(val) {
                    return Err(FkError::JointLimitViolation {
                        index: i,
                        value: val,
                        min: joint.min_val,
                        max: joint.max_val,
                    });
                }
            }
        }
        let mut t = self.base_frame.clone();
        for (joint, &val) in self.joints.iter().zip(q.iter()) {
            t = t.mul(&joint.transform(val));
        }
        t = t.mul(&self.tool_frame);
        Ok(t)
    }

    /// Compute all intermediate frame transforms (including base and tool).
    ///
    /// Returns `n + 2` transforms: base, joint0, joint1, ..., tool.
    pub fn all_frames(&self, q: &[f64]) -> Result<Vec<Transform4>, FkError> {
        if q.len() != self.joints.len() {
            return Err(FkError::JointIndexOutOfRange {
                index: q.len(),
                count: self.joints.len(),
            });
        }
        let mut frames = Vec::with_capacity(self.joints.len() + 2);
        let mut t = self.base_frame.clone();
        frames.push(t.clone());
        for (joint, &val) in self.joints.iter().zip(q.iter()) {
            t = t.mul(&joint.transform(val));
            frames.push(t.clone());
        }
        t = t.mul(&self.tool_frame);
        frames.push(t);
        Ok(frames)
    }

    /// Compute the end-effector position only (no orientation).
    pub fn end_effector_position(&self, q: &[f64]) -> Result<[f64; 3], FkError> {
        Ok(self.end_effector(q)?.position())
    }

    /// Compute a numerical Jacobian (6 x n) by finite differencing.
    ///
    /// Top 3 rows = linear velocity, bottom 3 rows = angular velocity.
    /// The Jacobian is returned as a flat row-major vector of length `6 * n`.
    pub fn numerical_jacobian(&self, q: &[f64], delta: f64) -> Result<Vec<f64>, FkError> {
        let n = self.joints.len();
        let t0 = self.end_effector(q)?;
        let p0 = t0.position();
        let euler0 = t0.to_euler_zyx();

        let mut jac = vec![0.0; 6 * n];
        let mut q_perturbed = q.to_vec();

        for j in 0..n {
            let original = q_perturbed[j];
            q_perturbed[j] = original + delta;
            let t_plus = self.end_effector(&q_perturbed)?;
            let p_plus = t_plus.position();
            let euler_plus = t_plus.to_euler_zyx();

            for r in 0..3 {
                jac[r * n + j] = (p_plus[r] - p0[r]) / delta;
            }
            for r in 0..3 {
                jac[(r + 3) * n + j] = (euler_plus[r] - euler0[r]) / delta;
            }
            q_perturbed[j] = original;
        }
        Ok(jac)
    }

    /// Compute the geometric Jacobian at joint `q` using frame-by-frame z-axes.
    ///
    /// Returns a flat row-major 6 x n matrix.
    pub fn geometric_jacobian(&self, q: &[f64]) -> Result<Vec<f64>, FkError> {
        let frames = self.all_frames(q)?;
        let n = self.joints.len();
        let ee_pos = frames.last().unwrap().position();
        let mut jac = vec![0.0; 6 * n];

        for i in 0..n {
            let frame = &frames[i];
            let z = [frame.m[2], frame.m[6], frame.m[10]]; // z-axis of frame i
            let p = frame.position();

            match self.joints[i].joint_type {
                JointType::Revolute => {
                    // Jv = z_i x (p_ee - p_i)
                    let d = [ee_pos[0] - p[0], ee_pos[1] - p[1], ee_pos[2] - p[2]];
                    let cross = cross3(&z, &d);
                    for r in 0..3 {
                        jac[r * n + i] = cross[r];
                    }
                    // Jw = z_i
                    for r in 0..3 {
                        jac[(r + 3) * n + i] = z[r];
                    }
                }
                JointType::Prismatic => {
                    // Jv = z_i
                    for r in 0..3 {
                        jac[r * n + i] = z[r];
                    }
                    // Jw = 0 (already zero)
                }
            }
        }
        Ok(jac)
    }
}

impl std::fmt::Display for ForwardKinematics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ForwardKinematics({} joints):", self.joints.len())?;
        for (i, j) in self.joints.iter().enumerate() {
            writeln!(f, "  [{i}] {j}")?;
        }
        Ok(())
    }
}

// ── Pose ───────────────────────────────────────────────────────

/// Compact representation of a 6-DOF pose (position + ZYX Euler angles).
#[derive(Debug, Clone, PartialEq)]
pub struct Pose6D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

impl Pose6D {
    /// Create from a homogeneous transform.
    pub fn from_transform(t: &Transform4) -> Self {
        let p = t.position();
        let e = t.to_euler_zyx();
        Self { x: p[0], y: p[1], z: p[2], roll: e[0], pitch: e[1], yaw: e[2] }
    }

    /// Euclidean distance to another pose (position only).
    pub fn position_distance(&self, other: &Pose6D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Angular distance (sum of absolute Euler-angle differences).
    pub fn orientation_distance(&self, other: &Pose6D) -> f64 {
        (self.roll - other.roll).abs()
            + (self.pitch - other.pitch).abs()
            + (self.yaw - other.yaw).abs()
    }
}

impl std::fmt::Display for Pose6D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pose6D(x={:.4}, y={:.4}, z={:.4}, R={:.3}, P={:.3}, Y={:.3})",
            self.x, self.y, self.z, self.roll, self.pitch, self.yaw,
        )
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Cross product of two 3-vectors.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Wrap an angle into `[-PI, PI)`.
pub fn wrap_angle(a: f64) -> f64 {
    let mut v = a % (2.0 * PI);
    if v >= PI {
        v -= 2.0 * PI;
    } else if v < -PI {
        v += 2.0 * PI;
    }
    v
}

// ── Standard Robot Configurations ──────────────────────────────

/// Build DH parameters for a classic 6-DOF industrial manipulator
/// (similar to a PUMA 560 layout).
pub fn puma560_joints() -> Vec<DhJoint> {
    vec![
        DhJoint::revolute(0.0, 0.6718, 0.0, -PI / 2.0),
        DhJoint::revolute(0.0, 0.0, 0.4318, 0.0),
        DhJoint::revolute(0.0, 0.15005, 0.0203, -PI / 2.0),
        DhJoint::revolute(0.0, 0.4318, 0.0, PI / 2.0),
        DhJoint::revolute(0.0, 0.0, 0.0, -PI / 2.0),
        DhJoint::revolute(0.0, 0.0, 0.0, 0.0),
    ]
}

/// Build DH parameters for a 3-DOF planar RRR arm.
pub fn planar_3r(l1: f64, l2: f64, l3: f64) -> Vec<DhJoint> {
    vec![
        DhJoint::revolute(0.0, 0.0, l1, 0.0),
        DhJoint::revolute(0.0, 0.0, l2, 0.0),
        DhJoint::revolute(0.0, 0.0, l3, 0.0),
    ]
}

/// Build DH parameters for a SCARA robot (RRP + wrist rotation).
pub fn scara_joints(l1: f64, l2: f64, d_max: f64) -> Vec<DhJoint> {
    vec![
        DhJoint::revolute(0.0, 0.0, l1, 0.0),
        DhJoint::revolute(0.0, 0.0, l2, PI),
        DhJoint::prismatic(0.0, 0.0, 0.0, 0.0).with_limits(0.0, d_max),
        DhJoint::revolute(0.0, 0.0, 0.0, 0.0),
    ]
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
    fn test_identity_transform() {
        let t = Transform4::identity();
        assert_eq!(t.position(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_translation_transform() {
        let t = Transform4::translation(1.0, 2.0, 3.0);
        assert_eq!(t.position(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_identity_mul() {
        let id = Transform4::identity();
        let t = Transform4::translation(5.0, 6.0, 7.0);
        let r = id.mul(&t);
        assert_eq!(r.position(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_rot_z_90() {
        let t = Transform4::rot_z(PI / 2.0);
        assert!(approx_eq(t.m[0], 0.0));
        assert!(approx_eq(t.m[1], -1.0));
        assert!(approx_eq(t.m[4], 1.0));
        assert!(approx_eq(t.m[5], 0.0));
    }

    #[test]
    fn test_rot_x_90() {
        let t = Transform4::rot_x(PI / 2.0);
        assert!(approx_eq(t.m[5], 0.0));
        assert!(approx_eq(t.m[6], -1.0));
        assert!(approx_eq(t.m[9], 1.0));
        assert!(approx_eq(t.m[10], 0.0));
    }

    #[test]
    fn test_inverse_identity() {
        let t = Transform4::identity();
        let inv = t.inverse();
        assert_eq!(inv.position(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_inverse_translation() {
        let t = Transform4::translation(3.0, -2.0, 1.5);
        let inv = t.inverse();
        let p = inv.position();
        assert!(approx_eq(p[0], -3.0));
        assert!(approx_eq(p[1], 2.0));
        assert!(approx_eq(p[2], -1.5));
    }

    #[test]
    fn test_mul_inverse_gives_identity() {
        let t = Transform4::from_dh(0.3, 0.5, 1.0, -0.7);
        let inv = t.inverse();
        let result = t.mul(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(result.m[i * 4 + j], expected));
            }
        }
    }

    #[test]
    fn test_dh_transform_zero() {
        let t = Transform4::from_dh(0.0, 0.0, 0.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(t.m[i * 4 + j], expected));
            }
        }
    }

    #[test]
    fn test_empty_chain_error() {
        let result = ForwardKinematics::new(vec![]);
        assert!(matches!(result, Err(FkError::EmptyChain)));
    }

    #[test]
    fn test_single_revolute_zero() {
        let joints = vec![DhJoint::revolute(0.0, 0.0, 1.0, 0.0)];
        let fk = ForwardKinematics::new(joints).unwrap();
        let t = fk.end_effector(&[0.0]).unwrap();
        let p = t.position();
        assert!(approx_eq(p[0], 1.0));
        assert!(approx_eq(p[1], 0.0));
        assert!(approx_eq(p[2], 0.0));
    }

    #[test]
    fn test_single_revolute_90() {
        let joints = vec![DhJoint::revolute(0.0, 0.0, 1.0, 0.0)];
        let fk = ForwardKinematics::new(joints).unwrap();
        let t = fk.end_effector(&[PI / 2.0]).unwrap();
        let p = t.position();
        assert!(approx_eq(p[0], 0.0));
        assert!(approx_eq(p[1], 1.0));
    }

    #[test]
    fn test_planar_2r_straight() {
        let joints = planar_3r(1.0, 1.0, 0.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let t = fk.end_effector(&[0.0, 0.0, 0.0]).unwrap();
        let p = t.position();
        assert!(approx_eq(p[0], 2.0));
        assert!(approx_eq(p[1], 0.0));
    }

    #[test]
    fn test_planar_2r_folded() {
        let joints = planar_3r(1.0, 1.0, 0.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let t = fk.end_effector(&[0.0, PI, 0.0]).unwrap();
        let p = t.position();
        assert!(approx_eq(p[0], 0.0));
        assert!((p[1]).abs() < 1e-6);
    }

    #[test]
    fn test_all_frames_count() {
        let joints = planar_3r(1.0, 1.0, 1.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let frames = fk.all_frames(&[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(frames.len(), 5); // base + 3 joints + tool
    }

    #[test]
    fn test_joint_limit_violation() {
        let joints = vec![DhJoint::revolute(0.0, 0.0, 1.0, 0.0).with_limits(-1.0, 1.0)];
        let fk = ForwardKinematics::new(joints).unwrap().with_limit_enforcement(true);
        let result = fk.end_effector(&[2.0]);
        assert!(matches!(result, Err(FkError::JointLimitViolation { .. })));
    }

    #[test]
    fn test_wrong_q_length() {
        let joints = vec![DhJoint::revolute(0.0, 0.0, 1.0, 0.0)];
        let fk = ForwardKinematics::new(joints).unwrap();
        let result = fk.end_effector(&[0.0, 0.0]);
        assert!(matches!(result, Err(FkError::JointIndexOutOfRange { .. })));
    }

    #[test]
    fn test_tool_frame_offset() {
        let joints = vec![DhJoint::revolute(0.0, 0.0, 1.0, 0.0)];
        let tool = Transform4::translation(0.5, 0.0, 0.0);
        let fk = ForwardKinematics::new(joints).unwrap().with_tool_frame(tool);
        let t = fk.end_effector(&[0.0]).unwrap();
        let p = t.position();
        assert!(approx_eq(p[0], 1.5));
    }

    #[test]
    fn test_numerical_jacobian_shape() {
        let joints = planar_3r(1.0, 1.0, 1.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let jac = fk.numerical_jacobian(&[0.0, 0.0, 0.0], 1e-6).unwrap();
        assert_eq!(jac.len(), 6 * 3);
    }

    #[test]
    fn test_geometric_jacobian_shape() {
        let joints = planar_3r(1.0, 1.0, 1.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let jac = fk.geometric_jacobian(&[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(jac.len(), 6 * 3);
    }

    #[test]
    fn test_pose6d_from_identity() {
        let t = Transform4::identity();
        let pose = Pose6D::from_transform(&t);
        assert!(approx_eq(pose.x, 0.0));
        assert!(approx_eq(pose.y, 0.0));
        assert!(approx_eq(pose.z, 0.0));
    }

    #[test]
    fn test_wrap_angle() {
        assert!(approx_eq(wrap_angle(0.0), 0.0));
        assert!(approx_eq(wrap_angle(2.0 * PI), 0.0));
        assert!(approx_eq(wrap_angle(-2.0 * PI), 0.0));
        assert!(wrap_angle(PI + 0.1) < 0.0);
    }

    #[test]
    fn test_puma560_has_6_joints() {
        let joints = puma560_joints();
        assert_eq!(joints.len(), 6);
    }

    #[test]
    fn test_scara_has_4_joints() {
        let joints = scara_joints(0.4, 0.3, 0.2);
        assert_eq!(joints.len(), 4);
        assert_eq!(joints[2].joint_type, JointType::Prismatic);
    }

    #[test]
    fn test_display_fk() {
        let joints = planar_3r(1.0, 1.0, 1.0);
        let fk = ForwardKinematics::new(joints).unwrap();
        let s = format!("{fk}");
        assert!(s.contains("ForwardKinematics(3 joints)"));
    }
}
