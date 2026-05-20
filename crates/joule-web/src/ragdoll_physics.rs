//! Ragdoll Physics — ragdoll system built on rigid bodies and joints.
//! Bone hierarchy with capsule bodies, hinge/cone-twist joints, pose matching
//! via spring-damper muscle forces, partial ragdoll mode, and impact activation.

use std::collections::HashMap;

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }
    pub fn cross(self, r: Self) -> Self {
        Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x }
    }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length(); if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }
}

impl std::ops::Add for Vec3 { type Output = Self; fn add(self, r: Self) -> Self { Self { x: self.x+r.x, y: self.y+r.y, z: self.z+r.z } } }
impl std::ops::Sub for Vec3 { type Output = Self; fn sub(self, r: Self) -> Self { Self { x: self.x-r.x, y: self.y-r.y, z: self.z-r.z } } }
impl std::ops::Mul<f64> for Vec3 { type Output = Self; fn mul(self, s: f64) -> Self { Self { x: self.x*s, y: self.y*s, z: self.z*s } } }
impl std::ops::Neg for Vec3 { type Output = Self; fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } } }
impl std::ops::AddAssign for Vec3 { fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; } }
impl std::ops::SubAssign for Vec3 { fn sub_assign(&mut self, r: Self) { self.x -= r.x; self.y -= r.y; self.z -= r.z; } }

// ── Quaternion ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion { pub w: f64, pub x: f64, pub y: f64, pub z: f64 }

impl Quaternion {
    pub const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let h = angle * 0.5; let s = h.sin();
        Self { w: h.cos(), x: axis.x * s, y: axis.y * s, z: axis.z * s }
    }

    pub fn normalized(self) -> Self {
        let l = (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt();
        if l < 1e-12 { Self::IDENTITY } else {
            let inv = 1.0 / l;
            Self { w: self.w*inv, x: self.x*inv, y: self.y*inv, z: self.z*inv }
        }
    }

    pub fn conjugate(self) -> Self { Self { w: self.w, x: -self.x, y: -self.y, z: -self.z } }

    pub fn mul(self, r: Self) -> Self {
        Self {
            w: self.w*r.w - self.x*r.x - self.y*r.y - self.z*r.z,
            x: self.w*r.x + self.x*r.w + self.y*r.z - self.z*r.y,
            y: self.w*r.y - self.x*r.z + self.y*r.w + self.z*r.x,
            z: self.w*r.z + self.x*r.y - self.y*r.x + self.z*r.w,
        }
    }

    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v + uv * (2.0 * self.w) + uuv * 2.0
    }

    /// Slerp between self and other.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z;
        let other = if dot < 0.0 {
            dot = -dot;
            Quaternion { w: -other.w, x: -other.x, y: -other.y, z: -other.z }
        } else {
            other
        };

        if dot > 0.9995 {
            return Quaternion {
                w: self.w + (other.w - self.w) * t,
                x: self.x + (other.x - self.x) * t,
                y: self.y + (other.y - self.y) * t,
                z: self.z + (other.z - self.z) * t,
            }.normalized();
        }

        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-12 { return self; }
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        Quaternion {
            w: self.w * a + other.w * b,
            x: self.x * a + other.x * b,
            y: self.y * a + other.y * b,
            z: self.z * a + other.z * b,
        }.normalized()
    }

    /// Angular difference as axis-angle vector.
    pub fn angular_diff(self, target: Self) -> Vec3 {
        let diff = target.mul(self.conjugate()).normalized();
        let half_angle = diff.w.clamp(-1.0, 1.0).acos();
        let axis = Vec3::new(diff.x, diff.y, diff.z);
        let axis_len = axis.length();
        if axis_len < 1e-12 || half_angle < 1e-12 {
            Vec3::ZERO
        } else {
            axis * ((2.0 * half_angle) / axis_len)
        }
    }
}

// ── Bone ─────────────────────────────────────────────────────

pub type BoneId = u32;

/// A single bone in the ragdoll skeleton.
#[derive(Debug, Clone)]
pub struct Bone {
    pub id: BoneId,
    pub name: String,
    pub parent: Option<BoneId>,
    pub children: Vec<BoneId>,

    // Capsule shape
    pub length: f64,
    pub radius: f64,
    pub mass: f64,

    // Current state
    pub position: Vec3,
    pub orientation: Quaternion,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,

    // Ragdoll mode
    pub physics_enabled: bool,
}

impl Bone {
    pub fn new(id: BoneId, name: &str, length: f64, radius: f64, mass: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            parent: None,
            children: Vec::new(),
            length,
            radius,
            mass,
            position: Vec3::ZERO,
            orientation: Quaternion::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            physics_enabled: false,
        }
    }

    /// Capsule endpoint A (bottom).
    pub fn endpoint_a(&self) -> Vec3 {
        self.position - self.orientation.rotate_vec(Vec3::UP) * (self.length * 0.5)
    }

    /// Capsule endpoint B (top).
    pub fn endpoint_b(&self) -> Vec3 {
        self.position + self.orientation.rotate_vec(Vec3::UP) * (self.length * 0.5)
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.linear_velocity.length_sq()
            + 0.5 * self.mass * self.angular_velocity.length_sq() * self.radius * self.radius
    }
}

// ── Joint Types ──────────────────────────────────────────────

/// Joint constraint type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Single-axis rotation (e.g., elbow, knee).
    Hinge {
        axis: Vec3,
        min_angle: f64,
        max_angle: f64,
    },
    /// Cone-twist (e.g., shoulder, hip). Cone angle + twist limit.
    ConeTwist {
        cone_angle: f64,
        twist_angle: f64,
    },
    /// Fixed (no relative rotation allowed).
    Fixed,
}

/// A joint connecting two bones.
#[derive(Debug, Clone)]
pub struct Joint {
    pub bone_a: BoneId,
    pub bone_b: BoneId,
    pub joint_type: JointType,
    pub anchor_a: Vec3, // Local space offset on bone A
    pub anchor_b: Vec3, // Local space offset on bone B
    pub stiffness: f64,
    pub damping: f64,
}

impl Joint {
    pub fn new(bone_a: BoneId, bone_b: BoneId, joint_type: JointType) -> Self {
        Self {
            bone_a,
            bone_b,
            joint_type,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            stiffness: 500.0,
            damping: 50.0,
        }
    }

    pub fn with_anchors(mut self, a: Vec3, b: Vec3) -> Self {
        self.anchor_a = a;
        self.anchor_b = b;
        self
    }

    pub fn with_spring(mut self, stiffness: f64, damping: f64) -> Self {
        self.stiffness = stiffness;
        self.damping = damping;
        self
    }
}

// ── Muscle ───────────────────────────────────────────────────

/// Spring-damper muscle that drives a bone toward a target pose.
#[derive(Debug, Clone)]
pub struct Muscle {
    pub bone_id: BoneId,
    pub target_orientation: Quaternion,
    pub spring_constant: f64,
    pub damping_constant: f64,
    pub max_torque: f64,
}

impl Muscle {
    pub fn new(bone_id: BoneId, spring: f64, damping: f64, max_torque: f64) -> Self {
        Self {
            bone_id,
            target_orientation: Quaternion::IDENTITY,
            spring_constant: spring,
            damping_constant: damping,
            max_torque,
        }
    }

    /// Compute muscle torque given current orientation and angular velocity.
    pub fn compute_torque(&self, current_orientation: Quaternion, angular_velocity: Vec3) -> Vec3 {
        let error = current_orientation.angular_diff(self.target_orientation);
        let spring_torque = error * self.spring_constant;
        let damp_torque = angular_velocity * (-self.damping_constant);
        let total = spring_torque + damp_torque;
        let len = total.length();
        if len > self.max_torque && len > 1e-12 {
            total * (self.max_torque / len)
        } else {
            total
        }
    }
}

// ── Ragdoll ──────────────────────────────────────────────────

/// Ragdoll state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagdollState {
    /// Fully animated (no physics).
    Animated,
    /// Partially ragdolled (some bones physics, some animated).
    Partial,
    /// Fully ragdolled.
    FullRagdoll,
    /// Blending from ragdoll back to animation.
    BlendToAnim,
}

/// Ragdoll physics system.
#[derive(Debug, Clone)]
pub struct Ragdoll {
    pub bones: HashMap<BoneId, Bone>,
    pub joints: Vec<Joint>,
    pub muscles: Vec<Muscle>,
    pub state: RagdollState,
    pub blend_factor: f64,
    pub blend_speed: f64,
    pub impact_threshold: f64,
    pub gravity: Vec3,
    pub linear_damping: f64,
    pub angular_damping: f64,
    root_bone: Option<BoneId>,
    next_bone_id: BoneId,
}

impl Ragdoll {
    pub fn new() -> Self {
        Self {
            bones: HashMap::new(),
            joints: Vec::new(),
            muscles: Vec::new(),
            state: RagdollState::Animated,
            blend_factor: 0.0,
            blend_speed: 2.0,
            impact_threshold: 50.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            linear_damping: 0.05,
            angular_damping: 0.1,
            root_bone: None,
            next_bone_id: 0,
        }
    }

    /// Add a bone. Returns its BoneId.
    pub fn add_bone(&mut self, name: &str, length: f64, radius: f64, mass: f64, parent: Option<BoneId>) -> BoneId {
        let id = self.next_bone_id;
        self.next_bone_id += 1;
        let mut bone = Bone::new(id, name, length, radius, mass);
        bone.parent = parent;

        if parent.is_none() && self.root_bone.is_none() {
            self.root_bone = Some(id);
        }

        if let Some(pid) = parent {
            if let Some(p) = self.bones.get_mut(&pid) {
                p.children.push(id);
            }
        }
        self.bones.insert(id, bone);
        id
    }

    /// Add a joint between two bones.
    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    /// Add a muscle to a bone.
    pub fn add_muscle(&mut self, muscle: Muscle) {
        self.muscles.push(muscle);
    }

    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn root_bone(&self) -> Option<BoneId> {
        self.root_bone
    }

    /// Activate full ragdoll mode.
    pub fn activate_ragdoll(&mut self) {
        self.state = RagdollState::FullRagdoll;
        self.blend_factor = 1.0;
        for bone in self.bones.values_mut() {
            bone.physics_enabled = true;
        }
    }

    /// Activate partial ragdoll (only specified bones).
    pub fn activate_partial(&mut self, bone_ids: &[BoneId]) {
        self.state = RagdollState::Partial;
        for bone in self.bones.values_mut() {
            bone.physics_enabled = false;
        }
        for id in bone_ids {
            if let Some(bone) = self.bones.get_mut(id) {
                bone.physics_enabled = true;
            }
        }
    }

    /// Deactivate ragdoll (back to animated).
    pub fn deactivate_ragdoll(&mut self) {
        self.state = RagdollState::Animated;
        self.blend_factor = 0.0;
        for bone in self.bones.values_mut() {
            bone.physics_enabled = false;
            bone.linear_velocity = Vec3::ZERO;
            bone.angular_velocity = Vec3::ZERO;
        }
    }

    /// Start blending from ragdoll back to animation.
    pub fn start_blend_to_anim(&mut self) {
        self.state = RagdollState::BlendToAnim;
    }

    /// Check if an impact should trigger ragdoll.
    pub fn check_impact(&mut self, impulse_magnitude: f64) -> bool {
        if impulse_magnitude > self.impact_threshold {
            self.activate_ragdoll();
            true
        } else {
            false
        }
    }

    /// Set target poses for all muscles (animation pose).
    pub fn set_animation_pose(&mut self, poses: &[(BoneId, Quaternion)]) {
        for (bone_id, target) in poses {
            for muscle in &mut self.muscles {
                if muscle.bone_id == *bone_id {
                    muscle.target_orientation = *target;
                }
            }
        }
    }

    /// Step the ragdoll simulation.
    pub fn step(&mut self, dt: f64) {
        if self.state == RagdollState::Animated {
            return;
        }

        // Handle blend-to-anim transition
        if self.state == RagdollState::BlendToAnim {
            self.blend_factor -= self.blend_speed * dt;
            if self.blend_factor <= 0.0 {
                self.deactivate_ragdoll();
                return;
            }
        }

        // Apply gravity and muscle forces, then integrate
        let bone_ids: Vec<BoneId> = self.bones.keys().copied().collect();
        for id in &bone_ids {
            let physics_enabled = self.bones[id].physics_enabled;
            if !physics_enabled { continue; }

            // Find muscle torque for this bone
            let current_ori = self.bones[id].orientation;
            let current_omega = self.bones[id].angular_velocity;
            let mut muscle_torque = Vec3::ZERO;
            for m in &self.muscles {
                if m.bone_id == *id {
                    muscle_torque = m.compute_torque(current_ori, current_omega);
                }
            }

            let bone = self.bones.get_mut(id).unwrap();
            let inv_mass = if bone.mass > 1e-12 { 1.0 / bone.mass } else { 0.0 };

            // Linear: gravity + damping
            bone.linear_velocity += self.gravity * dt;
            bone.linear_velocity = bone.linear_velocity * (1.0 - self.linear_damping);
            bone.position += bone.linear_velocity * dt;

            // Angular: muscle torque + damping
            let inertia = 0.5 * bone.mass * bone.radius * bone.radius;
            let inv_inertia = if inertia > 1e-12 { 1.0 / inertia } else { 0.0 };
            bone.angular_velocity += muscle_torque * (inv_inertia * dt);
            bone.angular_velocity = bone.angular_velocity * (1.0 - self.angular_damping);

            // Integrate orientation
            let omega_q = Quaternion {
                w: 0.0, x: bone.angular_velocity.x,
                y: bone.angular_velocity.y, z: bone.angular_velocity.z,
            };
            let dq = omega_q.mul(bone.orientation);
            bone.orientation = Quaternion {
                w: bone.orientation.w + dq.w * 0.5 * dt,
                x: bone.orientation.x + dq.x * 0.5 * dt,
                y: bone.orientation.y + dq.y * 0.5 * dt,
                z: bone.orientation.z + dq.z * 0.5 * dt,
            }.normalized();
        }

        // Enforce joint constraints (position-based)
        for joint in &self.joints {
            let (pos_a, ori_a, pos_b, ori_b) = {
                let ba = match self.bones.get(&joint.bone_a) { Some(b) => b, None => continue };
                let bb = match self.bones.get(&joint.bone_b) { Some(b) => b, None => continue };
                (ba.position, ba.orientation, bb.position, bb.orientation)
            };

            let world_anchor_a = pos_a + ori_a.rotate_vec(joint.anchor_a);
            let world_anchor_b = pos_b + ori_b.rotate_vec(joint.anchor_b);
            let correction = (world_anchor_a - world_anchor_b) * 0.5;

            if let Some(bone_b) = self.bones.get_mut(&joint.bone_b) {
                if bone_b.physics_enabled {
                    bone_b.position += correction;
                }
            }
            if let Some(bone_a) = self.bones.get_mut(&joint.bone_a) {
                if bone_a.physics_enabled {
                    bone_a.position -= correction;
                }
            }

            // Angular limits (simplified)
            match joint.joint_type {
                JointType::Hinge { axis, min_angle, max_angle } => {
                    let rel_q = ori_b.mul(ori_a.conjugate());
                    let angle_vec = Quaternion::IDENTITY.angular_diff(rel_q);
                    let angle = angle_vec.dot(axis);
                    if angle < min_angle || angle > max_angle {
                        let clamped = angle.clamp(min_angle, max_angle);
                        let correction_torque = axis * (clamped - angle) * joint.stiffness * 0.01;
                        if let Some(bone_b) = self.bones.get_mut(&joint.bone_b) {
                            if bone_b.physics_enabled {
                                bone_b.angular_velocity += correction_torque;
                            }
                        }
                    }
                }
                JointType::ConeTwist { cone_angle, twist_angle } => {
                    let rel_q = ori_b.mul(ori_a.conjugate());
                    let angle_vec = Quaternion::IDENTITY.angular_diff(rel_q);
                    let total_angle = angle_vec.length();
                    if total_angle > cone_angle {
                        let correction = angle_vec.normalized() * ((cone_angle - total_angle) * joint.stiffness * 0.01);
                        if let Some(bone_b) = self.bones.get_mut(&joint.bone_b) {
                            if bone_b.physics_enabled {
                                bone_b.angular_velocity += correction;
                            }
                        }
                    }
                    let _ = twist_angle; // Twist limit is more complex, simplified here
                }
                JointType::Fixed => {
                    // Lock relative orientation
                    let rel_q = ori_b.mul(ori_a.conjugate());
                    let angle_vec = Quaternion::IDENTITY.angular_diff(rel_q);
                    let correction = angle_vec * (-joint.stiffness * 0.01);
                    if let Some(bone_b) = self.bones.get_mut(&joint.bone_b) {
                        if bone_b.physics_enabled {
                            bone_b.angular_velocity += correction;
                        }
                    }
                }
            }
        }
    }

    /// Get total kinetic energy of all physics-enabled bones.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bones.values()
            .filter(|b| b.physics_enabled)
            .map(|b| b.kinetic_energy())
            .sum()
    }

    /// Apply an impulse to a specific bone.
    pub fn apply_impulse(&mut self, bone_id: BoneId, impulse: Vec3) {
        if let Some(bone) = self.bones.get_mut(&bone_id) {
            if bone.mass > 1e-12 {
                bone.linear_velocity += impulse * (1.0 / bone.mass);
            }
        }
    }

    /// Get bone by ID.
    pub fn get_bone(&self, id: BoneId) -> Option<&Bone> {
        self.bones.get(&id)
    }

    /// Get bone by name.
    pub fn find_bone_by_name(&self, name: &str) -> Option<&Bone> {
        self.bones.values().find(|b| b.name == name)
    }
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v3_approx(a: Vec3, b: Vec3) -> bool { approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) }

    fn make_simple_ragdoll() -> Ragdoll {
        let mut r = Ragdoll::new();
        let torso = r.add_bone("torso", 0.5, 0.15, 10.0, None);
        let head = r.add_bone("head", 0.2, 0.1, 4.0, Some(torso));
        let l_arm = r.add_bone("left_arm", 0.4, 0.05, 3.0, Some(torso));
        let r_arm = r.add_bone("right_arm", 0.4, 0.05, 3.0, Some(torso));
        let l_leg = r.add_bone("left_leg", 0.5, 0.07, 5.0, Some(torso));
        let r_leg = r.add_bone("right_leg", 0.5, 0.07, 5.0, Some(torso));

        r.add_joint(Joint::new(torso, head, JointType::ConeTwist { cone_angle: 0.5, twist_angle: 0.3 }));
        r.add_joint(Joint::new(torso, l_arm, JointType::ConeTwist { cone_angle: 1.5, twist_angle: 1.0 }));
        r.add_joint(Joint::new(torso, r_arm, JointType::ConeTwist { cone_angle: 1.5, twist_angle: 1.0 }));
        r.add_joint(Joint::new(torso, l_leg, JointType::Hinge {
            axis: Vec3::new(1.0, 0.0, 0.0), min_angle: -2.0, max_angle: 0.1,
        }));
        r.add_joint(Joint::new(torso, r_leg, JointType::Hinge {
            axis: Vec3::new(1.0, 0.0, 0.0), min_angle: -2.0, max_angle: 0.1,
        }));

        r
    }

    #[test]
    fn test_bone_creation() {
        let bone = Bone::new(0, "test", 0.5, 0.1, 3.0);
        assert_eq!(bone.id, 0);
        assert_eq!(bone.name, "test");
        assert!(approx(bone.length, 0.5));
    }

    #[test]
    fn test_bone_endpoints() {
        let mut bone = Bone::new(0, "test", 2.0, 0.1, 1.0);
        bone.position = Vec3::new(0.0, 1.0, 0.0);
        let a = bone.endpoint_a();
        let b = bone.endpoint_b();
        assert!(v3_approx(a, Vec3::new(0.0, 0.0, 0.0)));
        assert!(v3_approx(b, Vec3::new(0.0, 2.0, 0.0)));
    }

    #[test]
    fn test_ragdoll_creation() {
        let r = make_simple_ragdoll();
        assert_eq!(r.bone_count(), 6);
        assert_eq!(r.joint_count(), 5);
    }

    #[test]
    fn test_ragdoll_hierarchy() {
        let r = make_simple_ragdoll();
        let root = r.root_bone().unwrap();
        let torso = r.get_bone(root).unwrap();
        assert_eq!(torso.children.len(), 5); // head, 2 arms, 2 legs
    }

    #[test]
    fn test_ragdoll_initial_state() {
        let r = make_simple_ragdoll();
        assert_eq!(r.state, RagdollState::Animated);
    }

    #[test]
    fn test_activate_ragdoll() {
        let mut r = make_simple_ragdoll();
        r.activate_ragdoll();
        assert_eq!(r.state, RagdollState::FullRagdoll);
        for bone in r.bones.values() {
            assert!(bone.physics_enabled);
        }
    }

    #[test]
    fn test_deactivate_ragdoll() {
        let mut r = make_simple_ragdoll();
        r.activate_ragdoll();
        r.deactivate_ragdoll();
        assert_eq!(r.state, RagdollState::Animated);
        for bone in r.bones.values() {
            assert!(!bone.physics_enabled);
        }
    }

    #[test]
    fn test_partial_ragdoll() {
        let mut r = make_simple_ragdoll();
        let head = r.find_bone_by_name("head").unwrap().id;
        r.activate_partial(&[head]);
        assert_eq!(r.state, RagdollState::Partial);
        assert!(r.get_bone(head).unwrap().physics_enabled);
        let torso = r.root_bone().unwrap();
        assert!(!r.get_bone(torso).unwrap().physics_enabled);
    }

    #[test]
    fn test_impact_activation() {
        let mut r = make_simple_ragdoll();
        assert!(!r.check_impact(10.0));
        assert_eq!(r.state, RagdollState::Animated);
        assert!(r.check_impact(100.0));
        assert_eq!(r.state, RagdollState::FullRagdoll);
    }

    #[test]
    fn test_step_no_op_when_animated() {
        let mut r = make_simple_ragdoll();
        let pos_before: Vec<Vec3> = r.bones.values().map(|b| b.position).collect();
        r.step(1.0 / 60.0);
        let pos_after: Vec<Vec3> = r.bones.values().map(|b| b.position).collect();
        for (a, b) in pos_before.iter().zip(pos_after.iter()) {
            assert!(v3_approx(*a, *b));
        }
    }

    #[test]
    fn test_step_gravity_when_ragdoll() {
        let mut r = make_simple_ragdoll();
        let root = r.root_bone().unwrap();
        r.bones.get_mut(&root).unwrap().position = Vec3::new(0.0, 5.0, 0.0);
        r.activate_ragdoll();
        for _ in 0..10 {
            r.step(1.0 / 60.0);
        }
        assert!(r.get_bone(root).unwrap().position.y < 5.0);
    }

    #[test]
    fn test_muscle_torque_at_rest() {
        let m = Muscle::new(0, 100.0, 10.0, 50.0);
        let torque = m.compute_torque(Quaternion::IDENTITY, Vec3::ZERO);
        assert!(v3_approx(torque, Vec3::ZERO));
    }

    #[test]
    fn test_muscle_torque_with_error() {
        let mut m = Muscle::new(0, 100.0, 10.0, 500.0);
        m.target_orientation = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0).normalized(), 0.5);
        let torque = m.compute_torque(Quaternion::IDENTITY, Vec3::ZERO);
        assert!(torque.length() > 0.0);
    }

    #[test]
    fn test_muscle_max_torque_clamp() {
        let mut m = Muscle::new(0, 10000.0, 0.0, 5.0);
        m.target_orientation = Quaternion::from_axis_angle(Vec3::UP, 1.0);
        let torque = m.compute_torque(Quaternion::IDENTITY, Vec3::ZERO);
        assert!(torque.length() <= 5.0 + EPS);
    }

    #[test]
    fn test_apply_impulse() {
        let mut r = make_simple_ragdoll();
        let root = r.root_bone().unwrap();
        r.apply_impulse(root, Vec3::new(100.0, 0.0, 0.0));
        assert!(r.get_bone(root).unwrap().linear_velocity.x > 0.0);
    }

    #[test]
    fn test_find_bone_by_name() {
        let r = make_simple_ragdoll();
        assert!(r.find_bone_by_name("head").is_some());
        assert!(r.find_bone_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_blend_to_anim() {
        let mut r = make_simple_ragdoll();
        r.activate_ragdoll();
        r.start_blend_to_anim();
        assert_eq!(r.state, RagdollState::BlendToAnim);
        // Step enough to blend back
        for _ in 0..100 {
            r.step(1.0 / 60.0);
        }
        assert_eq!(r.state, RagdollState::Animated);
    }

    #[test]
    fn test_total_kinetic_energy() {
        let mut r = make_simple_ragdoll();
        r.activate_ragdoll();
        r.apply_impulse(r.root_bone().unwrap(), Vec3::new(10.0, 0.0, 0.0));
        assert!(r.total_kinetic_energy() > 0.0);
    }

    #[test]
    fn test_joint_creation() {
        let j = Joint::new(0, 1, JointType::Fixed)
            .with_anchors(Vec3::new(0.0, 0.25, 0.0), Vec3::new(0.0, -0.25, 0.0))
            .with_spring(1000.0, 100.0);
        assert!(approx(j.stiffness, 1000.0));
        assert!(approx(j.damping, 100.0));
    }

    #[test]
    fn test_quaternion_slerp_identity() {
        let q = Quaternion::IDENTITY;
        let r = Quaternion::from_axis_angle(Vec3::UP, 1.0);
        let mid = q.slerp(r, 0.0);
        assert!(approx(mid.w, q.w) && approx(mid.x, q.x));
    }

    #[test]
    fn test_set_animation_pose() {
        let mut r = make_simple_ragdoll();
        let head = r.find_bone_by_name("head").unwrap().id;
        r.add_muscle(Muscle::new(head, 100.0, 10.0, 50.0));
        let target = Quaternion::from_axis_angle(Vec3::UP, 0.3);
        r.set_animation_pose(&[(head, target)]);
        let m = r.muscles.iter().find(|m| m.bone_id == head).unwrap();
        assert!(approx(m.target_orientation.w, target.w));
    }
}
