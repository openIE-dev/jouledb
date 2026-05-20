//! # Serial Kinematic Chain
//!
//! Represents a serial chain of links and joints for robotic manipulators.
//! Supports chain composition, tool frame attachment, link-joint structure
//! definition, and kinematic computations along the chain.

use std::fmt;

// ── Core Types ──

/// 4x4 homogeneous transformation matrix stored in row-major order.
#[derive(Clone, Debug)]
pub struct Transform4 {
    pub data: [f64; 16],
}

impl Transform4 {
    pub fn identity() -> Self {
        let mut data = [0.0; 16];
        data[0] = 1.0;
        data[5] = 1.0;
        data[10] = 1.0;
        data[15] = 1.0;
        Self { data }
    }

    pub fn from_dh(a: f64, alpha: f64, d: f64, theta: f64) -> Self {
        let ct = theta.cos();
        let st = theta.sin();
        let ca = alpha.cos();
        let sa = alpha.sin();
        Self {
            data: [
                ct,     -st * ca,  st * sa,  a * ct,
                st,      ct * ca, -ct * sa,  a * st,
                0.0,     sa,       ca,       d,
                0.0,     0.0,      0.0,      1.0,
            ],
        }
    }

    pub fn from_translation(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity();
        t.data[3] = x;
        t.data[7] = y;
        t.data[11] = z;
        t
    }

    pub fn from_rotation_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut t = Self::identity();
        t.data[0] = c;
        t.data[1] = -s;
        t.data[4] = s;
        t.data[5] = c;
        t
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.data[i * 4 + k] * other.data[k * 4 + j];
                }
                result[i * 4 + j] = sum;
            }
        }
        Self { data: result }
    }

    pub fn translation(&self) -> [f64; 3] {
        [self.data[3], self.data[7], self.data[11]]
    }

    pub fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.data[0], self.data[1], self.data[2]],
            [self.data[4], self.data[5], self.data[6]],
            [self.data[8], self.data[9], self.data[10]],
        ]
    }

    pub fn inverse(&self) -> Self {
        let r = self.rotation_matrix();
        let t = self.translation();
        // R^T
        let rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        // -R^T * t
        let nt = [
            -(rt[0][0] * t[0] + rt[0][1] * t[1] + rt[0][2] * t[2]),
            -(rt[1][0] * t[0] + rt[1][1] * t[1] + rt[1][2] * t[2]),
            -(rt[2][0] * t[0] + rt[2][1] * t[1] + rt[2][2] * t[2]),
        ];
        Self {
            data: [
                rt[0][0], rt[0][1], rt[0][2], nt[0],
                rt[1][0], rt[1][1], rt[1][2], nt[1],
                rt[2][0], rt[2][1], rt[2][2], nt[2],
                0.0,      0.0,      0.0,      1.0,
            ],
        }
    }
}

impl fmt::Display for Transform4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        write!(f, "T([{:.3}, {:.3}, {:.3}])", t[0], t[1], t[2])
    }
}

// ── Joint Types ──

/// Type of joint in the kinematic chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointType {
    Revolute,
    Prismatic,
    Fixed,
}

impl fmt::Display for JointType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Revolute => write!(f, "Revolute"),
            Self::Prismatic => write!(f, "Prismatic"),
            Self::Fixed => write!(f, "Fixed"),
        }
    }
}

/// Joint with DH parameters and limits.
#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    pub joint_type: JointType,
    pub a: f64,
    pub alpha: f64,
    pub d: f64,
    pub theta_offset: f64,
    pub min_limit: f64,
    pub max_limit: f64,
}

impl Joint {
    pub fn revolute(name: &str, a: f64, alpha: f64, d: f64) -> Self {
        Self {
            name: name.to_string(),
            joint_type: JointType::Revolute,
            a, alpha, d,
            theta_offset: 0.0,
            min_limit: -std::f64::consts::PI,
            max_limit: std::f64::consts::PI,
        }
    }

    pub fn prismatic(name: &str, a: f64, alpha: f64, theta: f64) -> Self {
        Self {
            name: name.to_string(),
            joint_type: JointType::Prismatic,
            a, alpha,
            d: 0.0,
            theta_offset: theta,
            min_limit: 0.0,
            max_limit: 1.0,
        }
    }

    pub fn fixed(name: &str, a: f64, alpha: f64, d: f64, theta: f64) -> Self {
        Self {
            name: name.to_string(),
            joint_type: JointType::Fixed,
            a, alpha, d,
            theta_offset: theta,
            min_limit: 0.0,
            max_limit: 0.0,
        }
    }

    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.min_limit = min;
        self.max_limit = max;
        self
    }

    pub fn with_offset(mut self, offset: f64) -> Self {
        self.theta_offset = offset;
        self
    }

    /// Compute the transform for this joint at the given variable value.
    pub fn transform(&self, q: f64) -> Transform4 {
        match self.joint_type {
            JointType::Revolute => Transform4::from_dh(self.a, self.alpha, self.d, q + self.theta_offset),
            JointType::Prismatic => Transform4::from_dh(self.a, self.alpha, self.d + q, self.theta_offset),
            JointType::Fixed => Transform4::from_dh(self.a, self.alpha, self.d, self.theta_offset),
        }
    }

    pub fn is_within_limits(&self, q: f64) -> bool {
        q >= self.min_limit && q <= self.max_limit
    }

    pub fn clamp(&self, q: f64) -> f64 {
        q.max(self.min_limit).min(self.max_limit)
    }
}

impl fmt::Display for Joint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Joint({}, {})", self.name, self.joint_type)
    }
}

// ── Serial Chain ──

/// A serial kinematic chain composed of joints with an optional tool frame.
#[derive(Clone, Debug)]
pub struct SerialChain {
    pub name: String,
    joints: Vec<Joint>,
    base_frame: Transform4,
    tool_frame: Transform4,
}

impl SerialChain {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            joints: Vec::new(),
            base_frame: Transform4::identity(),
            tool_frame: Transform4::identity(),
        }
    }

    pub fn with_base_frame(mut self, frame: Transform4) -> Self {
        self.base_frame = frame;
        self
    }

    pub fn with_tool_frame(mut self, frame: Transform4) -> Self {
        self.tool_frame = frame;
        self
    }

    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    pub fn with_joint(mut self, joint: Joint) -> Self {
        self.joints.push(joint);
        self
    }

    pub fn num_joints(&self) -> usize {
        self.joints.iter().filter(|j| j.joint_type != JointType::Fixed).count()
    }

    pub fn num_links(&self) -> usize {
        self.joints.len()
    }

    pub fn joint(&self, idx: usize) -> Option<&Joint> {
        self.joints.get(idx)
    }

    /// Compute forward kinematics up to and including the specified joint.
    pub fn fk_to_joint(&self, q: &[f64], joint_idx: usize) -> Transform4 {
        let mut tf = self.base_frame.clone();
        let mut q_idx = 0;

        for (i, joint) in self.joints.iter().enumerate() {
            if i > joint_idx { break; }
            let val = if joint.joint_type == JointType::Fixed {
                0.0
            } else {
                let v = q.get(q_idx).copied().unwrap_or(0.0);
                q_idx += 1;
                v
            };
            tf = tf.multiply(&joint.transform(val));
        }
        tf
    }

    /// Compute full forward kinematics including tool frame.
    pub fn fk(&self, q: &[f64]) -> Transform4 {
        let last = if self.joints.is_empty() { 0 } else { self.joints.len() - 1 };
        let tf = self.fk_to_joint(q, last);
        tf.multiply(&self.tool_frame)
    }

    /// Get all intermediate transforms along the chain.
    pub fn fk_all(&self, q: &[f64]) -> Vec<Transform4> {
        let mut transforms = Vec::with_capacity(self.joints.len() + 1);
        let mut tf = self.base_frame.clone();
        transforms.push(tf.clone());

        let mut q_idx = 0;
        for joint in &self.joints {
            let val = if joint.joint_type == JointType::Fixed {
                0.0
            } else {
                let v = q.get(q_idx).copied().unwrap_or(0.0);
                q_idx += 1;
                v
            };
            tf = tf.multiply(&joint.transform(val));
            transforms.push(tf.clone());
        }
        transforms
    }

    /// Check if all joint values are within limits.
    pub fn within_limits(&self, q: &[f64]) -> bool {
        let mut q_idx = 0;
        for joint in &self.joints {
            if joint.joint_type == JointType::Fixed { continue; }
            let val = q.get(q_idx).copied().unwrap_or(0.0);
            if !joint.is_within_limits(val) { return false; }
            q_idx += 1;
        }
        true
    }

    /// Clamp joint values to limits.
    pub fn clamp_joints(&self, q: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        let mut q_idx = 0;
        for joint in &self.joints {
            if joint.joint_type == JointType::Fixed { continue; }
            let val = q.get(q_idx).copied().unwrap_or(0.0);
            result.push(joint.clamp(val));
            q_idx += 1;
        }
        result
    }

    /// Compose two chains by appending the second chain's joints.
    pub fn compose(&self, other: &SerialChain) -> SerialChain {
        let mut composed = self.clone();
        composed.name = format!("{}+{}", self.name, other.name);
        // The tool frame of self becomes part of the chain
        let connecting = self.tool_frame.clone();
        composed.tool_frame = Transform4::identity();

        // Add a fixed joint for the connection
        composed.joints.push(Joint::fixed("connection", 0.0, 0.0, 0.0, 0.0));

        for joint in &other.joints {
            composed.joints.push(joint.clone());
        }
        composed.tool_frame = other.tool_frame.clone();
        let _ = connecting; // connection transform absorbed
        composed
    }

    /// Total chain length (sum of link lengths).
    pub fn total_length(&self) -> f64 {
        self.joints.iter().map(|j| (j.a * j.a + j.d * j.d).sqrt()).sum()
    }
}

impl fmt::Display for SerialChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SerialChain({}, {} joints, {} DOF)",
            self.name, self.joints.len(), self.num_joints())
    }
}

// ── Common Robot Configurations ──

/// Create a 2-DOF planar arm.
pub fn planar_2r(l1: f64, l2: f64) -> SerialChain {
    SerialChain::new("Planar2R")
        .with_joint(Joint::revolute("j1", l1, 0.0, 0.0))
        .with_joint(Joint::revolute("j2", l2, 0.0, 0.0))
}

/// Create a 3-DOF planar arm.
pub fn planar_3r(l1: f64, l2: f64, l3: f64) -> SerialChain {
    SerialChain::new("Planar3R")
        .with_joint(Joint::revolute("j1", l1, 0.0, 0.0))
        .with_joint(Joint::revolute("j2", l2, 0.0, 0.0))
        .with_joint(Joint::revolute("j3", l3, 0.0, 0.0))
}

/// Create a SCARA-like 4-DOF robot.
pub fn scara_4dof(l1: f64, l2: f64, d_range: f64) -> SerialChain {
    SerialChain::new("SCARA4")
        .with_joint(Joint::revolute("j1", l1, 0.0, 0.0))
        .with_joint(Joint::revolute("j2", l2, std::f64::consts::PI, 0.0))
        .with_joint(Joint::prismatic("j3", 0.0, 0.0, 0.0).with_limits(0.0, d_range))
        .with_joint(Joint::revolute("j4", 0.0, 0.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_transform_identity() {
        let t = Transform4::identity();
        assert!((t.data[0] - 1.0).abs() < 1e-10);
        assert_eq!(t.translation(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transform_translation() {
        let t = Transform4::from_translation(1.0, 2.0, 3.0);
        let pos = t.translation();
        assert!((pos[0] - 1.0).abs() < 1e-10);
        assert!((pos[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_multiply_identity() {
        let a = Transform4::from_translation(1.0, 0.0, 0.0);
        let id = Transform4::identity();
        let result = a.multiply(&id);
        assert!((result.translation()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_inverse() {
        let t = Transform4::from_translation(3.0, 4.0, 5.0);
        let inv = t.inverse();
        let result = t.multiply(&inv);
        for i in 0..3 {
            assert!((result.translation()[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dh_transform() {
        let t = Transform4::from_dh(1.0, 0.0, 0.0, 0.0);
        assert!((t.translation()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_revolute() {
        let j = Joint::revolute("j1", 1.0, 0.0, 0.0);
        assert_eq!(j.joint_type, JointType::Revolute);
        assert!(j.is_within_limits(0.0));
    }

    #[test]
    fn test_joint_limits() {
        let j = Joint::revolute("j1", 1.0, 0.0, 0.0).with_limits(-1.0, 1.0);
        assert!(j.is_within_limits(0.5));
        assert!(!j.is_within_limits(1.5));
        assert!((j.clamp(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_serial_chain_creation() {
        let chain = planar_2r(1.0, 1.0);
        assert_eq!(chain.num_joints(), 2);
        assert_eq!(chain.num_links(), 2);
    }

    #[test]
    fn test_fk_straight() {
        let chain = planar_2r(1.0, 1.0);
        let pos = chain.fk(&[0.0, 0.0]).translation();
        assert!((pos[0] - 2.0).abs() < 1e-10);
        assert!(pos[1].abs() < 1e-10);
    }

    #[test]
    fn test_fk_90_degrees() {
        let chain = planar_2r(1.0, 1.0);
        let pos = chain.fk(&[PI / 2.0, 0.0]).translation();
        assert!(pos[0].abs() < 1e-10);
        assert!((pos[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fk_folded() {
        let chain = planar_2r(1.0, 1.0);
        let pos = chain.fk(&[0.0, PI]).translation();
        assert!(pos[0].abs() < 1e-10);
    }

    #[test]
    fn test_fk_all_intermediate() {
        let chain = planar_2r(1.0, 1.0);
        let all = chain.fk_all(&[0.0, 0.0]);
        assert_eq!(all.len(), 3); // base + 2 joints
    }

    #[test]
    fn test_tool_frame() {
        let chain = planar_2r(1.0, 1.0)
            .with_tool_frame(Transform4::from_translation(0.5, 0.0, 0.0));
        let pos = chain.fk(&[0.0, 0.0]).translation();
        assert!((pos[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_within_limits() {
        let chain = SerialChain::new("test")
            .with_joint(Joint::revolute("j1", 1.0, 0.0, 0.0).with_limits(-1.0, 1.0));
        assert!(chain.within_limits(&[0.5]));
        assert!(!chain.within_limits(&[2.0]));
    }

    #[test]
    fn test_clamp_joints() {
        let chain = SerialChain::new("test")
            .with_joint(Joint::revolute("j1", 1.0, 0.0, 0.0).with_limits(-1.0, 1.0));
        let clamped = chain.clamp_joints(&[5.0]);
        assert!((clamped[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_compose() {
        let a = planar_2r(1.0, 1.0);
        let b = planar_2r(0.5, 0.5);
        let composed = a.compose(&b);
        assert!(composed.num_joints() >= 4);
    }

    #[test]
    fn test_planar_3r() {
        let chain = planar_3r(1.0, 1.0, 0.5);
        assert_eq!(chain.num_joints(), 3);
        let pos = chain.fk(&[0.0, 0.0, 0.0]).translation();
        assert!((pos[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_scara() {
        let chain = scara_4dof(1.0, 1.0, 0.5);
        assert_eq!(chain.num_joints(), 4);
    }

    #[test]
    fn test_total_length() {
        let chain = planar_2r(1.5, 1.0);
        assert!((chain.total_length() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let chain = planar_2r(1.0, 1.0);
        let s = format!("{chain}");
        assert!(s.contains("Planar2R"));
        assert!(s.contains("2 DOF"));
    }
}
