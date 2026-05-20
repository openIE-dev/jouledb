//! 3D transform hierarchy (parent-child). Each node: local position (Vec3),
//! rotation (quaternion), scale (Vec3). Compose local to world via parent chain.
//! Dirty flags for lazy recalculation. Operations: translate, rotate, scale,
//! look_at. Decompose world matrix back to TRS. Depth-first and breadth-first
//! traversal.

use std::collections::VecDeque;

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn one() -> Self { Self { x: 1.0, y: 1.0, z: 1.0 } }
    pub fn up() -> Self { Self { x: 0.0, y: 1.0, z: 0.0 } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn mul_comp(self, o: Self) -> Self { Self { x: self.x * o.x, y: self.y * o.y, z: self.z * o.z } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { self.scale(1.0 / len) }
    }
}

// ── Quaternion ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quaternion {
    pub fn identity() -> Self { Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } }

    pub fn from_axis_angle(axis: Vec3, angle_rad: f64) -> Self {
        let half = angle_rad * 0.5;
        let s = half.sin();
        let a = axis.normalized();
        Self { x: a.x * s, y: a.y * s, z: a.z * s, w: half.cos() }
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::identity() }
        else { Self { x: self.x / len, y: self.y / len, z: self.z / len, w: self.w / len } }
    }

    pub fn conjugate(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z, w: self.w } }

    pub fn mul(self, o: Self) -> Self {
        Self {
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        }
    }

    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let q_vec = Quaternion { x: v.x, y: v.y, z: v.z, w: 0.0 };
        let result = self.mul(q_vec).mul(self.conjugate());
        Vec3::new(result.x, result.y, result.z)
    }

    pub fn to_matrix(self) -> Mat4 {
        let q = self.normalized();
        let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
        let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
        let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;
        Mat4 { m: [
            [1.0 - 2.0*(yy+zz), 2.0*(xy-wz),       2.0*(xz+wy),       0.0],
            [2.0*(xy+wz),       1.0 - 2.0*(xx+zz), 2.0*(yz-wx),       0.0],
            [2.0*(xz-wy),       2.0*(yz+wx),       1.0 - 2.0*(xx+yy), 0.0],
            [0.0,               0.0,               0.0,               1.0],
        ]}
    }

    /// Extract Euler angles (XYZ order) from quaternion.
    pub fn to_euler(self) -> Vec3 {
        let q = self.normalized();
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp);
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp);
        Vec3::new(roll, pitch, yaw)
    }
}

// ── Mat4 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        Self { m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]}
    }

    pub fn translation(v: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][3] = v.x;
        m.m[1][3] = v.y;
        m.m[2][3] = v.z;
        m
    }

    pub fn scaling(v: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][0] = v.x;
        m.m[1][1] = v.y;
        m.m[2][2] = v.z;
        m
    }

    pub fn mul(self, o: Self) -> Self {
        let mut r = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    r[i][j] += self.m[i][k] * o.m[k][j];
                }
            }
        }
        Self { m: r }
    }

    pub fn transform_point(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z + self.m[0][3],
            self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z + self.m[1][3],
            self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z + self.m[2][3],
        )
    }

    /// Decompose a TRS matrix into translation, rotation (quaternion), and scale.
    pub fn decompose(&self) -> (Vec3, Quaternion, Vec3) {
        let translation = Vec3::new(self.m[0][3], self.m[1][3], self.m[2][3]);
        let sx = Vec3::new(self.m[0][0], self.m[1][0], self.m[2][0]).length();
        let sy = Vec3::new(self.m[0][1], self.m[1][1], self.m[2][1]).length();
        let sz = Vec3::new(self.m[0][2], self.m[1][2], self.m[2][2]).length();
        let scale = Vec3::new(sx, sy, sz);
        // Extract rotation matrix (divide out scale)
        let isx = if sx > 1e-12 { 1.0 / sx } else { 0.0 };
        let isy = if sy > 1e-12 { 1.0 / sy } else { 0.0 };
        let isz = if sz > 1e-12 { 1.0 / sz } else { 0.0 };
        let r00 = self.m[0][0] * isx; let r10 = self.m[1][0] * isx; let r20 = self.m[2][0] * isx;
        let r01 = self.m[0][1] * isy; let r11 = self.m[1][1] * isy; let r21 = self.m[2][1] * isy;
        let r02 = self.m[0][2] * isz; let r12 = self.m[1][2] * isz; let r22 = self.m[2][2] * isz;
        // Shepperd's method for quaternion from rotation matrix
        let trace = r00 + r11 + r22;
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quaternion { w: 0.25 * s, x: (r21 - r12) / s, y: (r02 - r20) / s, z: (r10 - r01) / s }
        } else if r00 > r11 && r00 > r22 {
            let s = (1.0 + r00 - r11 - r22).sqrt() * 2.0;
            Quaternion { w: (r21 - r12) / s, x: 0.25 * s, y: (r01 + r10) / s, z: (r02 + r20) / s }
        } else if r11 > r22 {
            let s = (1.0 + r11 - r00 - r22).sqrt() * 2.0;
            Quaternion { w: (r02 - r20) / s, x: (r01 + r10) / s, y: 0.25 * s, z: (r12 + r21) / s }
        } else {
            let s = (1.0 + r22 - r00 - r11).sqrt() * 2.0;
            Quaternion { w: (r10 - r01) / s, x: (r02 + r20) / s, y: (r12 + r21) / s, z: 0.25 * s }
        };
        (translation, q.normalized(), scale)
    }

    /// Build a TRS matrix from components.
    pub fn from_trs(t: Vec3, r: Quaternion, s: Vec3) -> Self {
        Mat4::translation(t).mul(r.to_matrix()).mul(Mat4::scaling(s))
    }

    /// Build a look-at matrix (right-handed).
    pub fn look_at(eye: Vec3, target: Vec3, up_hint: Vec3) -> Self {
        let forward = target.sub(eye).normalized();
        let right = forward.cross(up_hint).normalized();
        let up = right.cross(forward);
        let mut m = Self::identity();
        m.m[0][0] = right.x;   m.m[0][1] = right.y;   m.m[0][2] = right.z;
        m.m[1][0] = up.x;      m.m[1][1] = up.y;      m.m[1][2] = up.z;
        m.m[2][0] = -forward.x; m.m[2][1] = -forward.y; m.m[2][2] = -forward.z;
        m.m[0][3] = -right.dot(eye);
        m.m[1][3] = -up.dot(eye);
        m.m[2][3] = forward.dot(eye);
        m
    }
}

// ── Transform ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quaternion,
    pub local_scale: Vec3,
}

impl Transform {
    pub fn new() -> Self {
        Self { position: Vec3::zero(), rotation: Quaternion::identity(), local_scale: Vec3::one() }
    }

    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_trs(self.position, self.rotation, self.local_scale)
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.position = self.position.add(delta);
    }

    pub fn rotate(&mut self, axis: Vec3, angle_rad: f64) {
        let q = Quaternion::from_axis_angle(axis, angle_rad);
        self.rotation = q.mul(self.rotation).normalized();
    }

    pub fn scale_by(&mut self, factor: Vec3) {
        self.local_scale = self.local_scale.mul_comp(factor);
    }

    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = target.sub(self.position).normalized();
        if forward.length() < 1e-9 { return; }
        let right = forward.cross(up).normalized();
        if right.length() < 1e-9 { return; }
        let corrected_up = right.cross(forward);
        // Build rotation matrix and extract quaternion
        let mat = Mat4 { m: [
            [right.x,    right.y,    right.z,    0.0],
            [corrected_up.x, corrected_up.y, corrected_up.z, 0.0],
            [-forward.x, -forward.y, -forward.z, 0.0],
            [0.0,        0.0,        0.0,        1.0],
        ]};
        let (_, q, _) = mat.decompose();
        self.rotation = q;
    }
}

// ── TransformNode / Hierarchy ────────────────────────────────

/// A uniquely identified node in the hierarchy.
pub type NodeId = usize;

#[derive(Debug, Clone)]
struct HierarchyNode {
    transform: Transform,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    world_matrix: Mat4,
    dirty: bool,
    name: String,
}

/// The full scene transform hierarchy.
#[derive(Debug, Clone)]
pub struct TransformHierarchy {
    nodes: Vec<HierarchyNode>,
}

impl TransformHierarchy {
    pub fn new() -> Self { Self { nodes: Vec::new() } }

    pub fn create_node(&mut self, name: &str) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(HierarchyNode {
            transform: Transform::new(),
            parent: None,
            children: Vec::new(),
            world_matrix: Mat4::identity(),
            dirty: true,
            name: name.to_string(),
        });
        id
    }

    pub fn set_parent(&mut self, child: NodeId, parent: NodeId) {
        // Remove from old parent
        if let Some(old_parent) = self.nodes[child].parent {
            self.nodes[old_parent].children.retain(|c| *c != child);
        }
        self.nodes[child].parent = Some(parent);
        self.nodes[parent].children.push(child);
        self.mark_dirty(child);
    }

    pub fn detach(&mut self, node: NodeId) {
        if let Some(parent) = self.nodes[node].parent {
            self.nodes[parent].children.retain(|c| *c != node);
        }
        self.nodes[node].parent = None;
        self.mark_dirty(node);
    }

    pub fn transform(&self, node: NodeId) -> &Transform { &self.nodes[node].transform }

    pub fn transform_mut(&mut self, node: NodeId) -> &mut Transform {
        self.mark_dirty(node);
        &mut self.nodes[node].transform
    }

    pub fn name(&self, node: NodeId) -> &str { &self.nodes[node].name }
    pub fn parent(&self, node: NodeId) -> Option<NodeId> { self.nodes[node].parent }
    pub fn children(&self, node: NodeId) -> &[NodeId] { &self.nodes[node].children }
    pub fn node_count(&self) -> usize { self.nodes.len() }

    fn mark_dirty(&mut self, node: NodeId) {
        if self.nodes[node].dirty { return; }
        self.nodes[node].dirty = true;
        let children: Vec<NodeId> = self.nodes[node].children.clone();
        for child in children {
            self.mark_dirty(child);
        }
    }

    /// Recompute world matrices for all dirty nodes.
    pub fn update(&mut self) {
        let roots: Vec<NodeId> = (0..self.nodes.len())
            .filter(|i| self.nodes[*i].parent.is_none())
            .collect();
        for root in roots {
            self.update_node(root, Mat4::identity());
        }
    }

    fn update_node(&mut self, node: NodeId, parent_world: Mat4) {
        if self.nodes[node].dirty {
            let local = self.nodes[node].transform.local_matrix();
            self.nodes[node].world_matrix = parent_world.mul(local);
            self.nodes[node].dirty = false;
        }
        let world = self.nodes[node].world_matrix;
        let children: Vec<NodeId> = self.nodes[node].children.clone();
        for child in children {
            self.update_node(child, world);
        }
    }

    pub fn world_matrix(&self, node: NodeId) -> Mat4 { self.nodes[node].world_matrix }

    /// Get world-space position of a node.
    pub fn world_position(&self, node: NodeId) -> Vec3 {
        let m = &self.nodes[node].world_matrix;
        Vec3::new(m.m[0][3], m.m[1][3], m.m[2][3])
    }

    /// Decompose world matrix into TRS.
    pub fn world_trs(&self, node: NodeId) -> (Vec3, Quaternion, Vec3) {
        self.nodes[node].world_matrix.decompose()
    }

    /// Depth-first traversal, returning node IDs in visit order.
    pub fn traverse_depth_first(&self, root: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            result.push(node);
            for &child in self.nodes[node].children.iter().rev() {
                stack.push(child);
            }
        }
        result
    }

    /// Breadth-first traversal, returning node IDs in visit order.
    pub fn traverse_breadth_first(&self, root: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            result.push(node);
            for &child in &self.nodes[node].children {
                queue.push_back(child);
            }
        }
        result
    }

    /// Find all root nodes (nodes with no parent).
    pub fn roots(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).filter(|i| self.nodes[*i].parent.is_none()).collect()
    }

    /// Depth of a node in the hierarchy.
    pub fn depth(&self, node: NodeId) -> usize {
        let mut d = 0;
        let mut current = node;
        while let Some(p) = self.nodes[current].parent {
            d += 1;
            current = p;
        }
        d
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn approx(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }
    fn v3_approx(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    #[test]
    fn test_identity_transform() {
        let t = Transform::new();
        let m = t.local_matrix();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(m.m[i][j], expected, 1e-9));
            }
        }
    }

    #[test]
    fn test_translation() {
        let mut t = Transform::new();
        t.translate(Vec3::new(3.0, 4.0, 5.0));
        let m = t.local_matrix();
        assert!(approx(m.m[0][3], 3.0, 1e-9));
        assert!(approx(m.m[1][3], 4.0, 1e-9));
        assert!(approx(m.m[2][3], 5.0, 1e-9));
    }

    #[test]
    fn test_rotation_90_y() {
        let q = Quaternion::from_axis_angle(Vec3::up(), FRAC_PI_2);
        let v = q.rotate_vec3(Vec3::new(1.0, 0.0, 0.0));
        assert!(v3_approx(v, Vec3::new(0.0, 0.0, -1.0), 1e-9));
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        let v = q.rotate_vec3(Vec3::new(1.0, 2.0, 3.0));
        assert!(v3_approx(v, Vec3::new(1.0, 2.0, 3.0), 1e-9));
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::from_axis_angle(Vec3::up(), 1.0);
        let qc = q.conjugate();
        let product = q.mul(qc);
        assert!(approx(product.w, 1.0, 1e-9));
        assert!(approx(product.x, 0.0, 1e-9));
    }

    #[test]
    fn test_scale_matrix() {
        let mut t = Transform::new();
        t.local_scale = Vec3::new(2.0, 3.0, 4.0);
        let m = t.local_matrix();
        let p = m.transform_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(v3_approx(p, Vec3::new(2.0, 3.0, 4.0), 1e-9));
    }

    #[test]
    fn test_hierarchy_parent_child() {
        let mut h = TransformHierarchy::new();
        let root = h.create_node("root");
        let child = h.create_node("child");
        h.set_parent(child, root);
        assert_eq!(h.parent(child), Some(root));
        assert_eq!(h.children(root), &[child]);
    }

    #[test]
    fn test_hierarchy_world_position() {
        let mut h = TransformHierarchy::new();
        let parent = h.create_node("parent");
        let child = h.create_node("child");
        h.set_parent(child, parent);
        h.transform_mut(parent).translate(Vec3::new(10.0, 0.0, 0.0));
        h.transform_mut(child).translate(Vec3::new(0.0, 5.0, 0.0));
        h.update();
        let wp = h.world_position(child);
        assert!(v3_approx(wp, Vec3::new(10.0, 5.0, 0.0), 1e-9));
    }

    #[test]
    fn test_hierarchy_scale_propagation() {
        let mut h = TransformHierarchy::new();
        let parent = h.create_node("parent");
        let child = h.create_node("child");
        h.set_parent(child, parent);
        h.transform_mut(parent).local_scale = Vec3::new(2.0, 2.0, 2.0);
        h.transform_mut(child).translate(Vec3::new(1.0, 0.0, 0.0));
        h.update();
        let wp = h.world_position(child);
        assert!(v3_approx(wp, Vec3::new(2.0, 0.0, 0.0), 1e-9));
    }

    #[test]
    fn test_detach() {
        let mut h = TransformHierarchy::new();
        let parent = h.create_node("parent");
        let child = h.create_node("child");
        h.set_parent(child, parent);
        h.detach(child);
        assert_eq!(h.parent(child), None);
        assert!(h.children(parent).is_empty());
    }

    #[test]
    fn test_depth_first_traversal() {
        let mut h = TransformHierarchy::new();
        let root = h.create_node("root");
        let a = h.create_node("a");
        let b = h.create_node("b");
        let aa = h.create_node("aa");
        h.set_parent(a, root);
        h.set_parent(b, root);
        h.set_parent(aa, a);
        let order = h.traverse_depth_first(root);
        assert_eq!(order, vec![root, a, aa, b]);
    }

    #[test]
    fn test_breadth_first_traversal() {
        let mut h = TransformHierarchy::new();
        let root = h.create_node("root");
        let a = h.create_node("a");
        let b = h.create_node("b");
        let aa = h.create_node("aa");
        h.set_parent(a, root);
        h.set_parent(b, root);
        h.set_parent(aa, a);
        let order = h.traverse_breadth_first(root);
        assert_eq!(order, vec![root, a, b, aa]);
    }

    #[test]
    fn test_decompose_trs() {
        let t = Vec3::new(1.0, 2.0, 3.0);
        let r = Quaternion::from_axis_angle(Vec3::up(), 0.5);
        let s = Vec3::new(1.0, 2.0, 3.0);
        let m = Mat4::from_trs(t, r, s);
        let (dt, dr, ds) = m.decompose();
        assert!(v3_approx(dt, t, 1e-9));
        assert!(v3_approx(ds, s, 1e-9));
        // Quaternion might be negated but represent same rotation
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rv1 = r.rotate_vec3(v);
        let rv2 = dr.rotate_vec3(v);
        assert!(v3_approx(rv1, rv2, 1e-9));
    }

    #[test]
    fn test_node_depth() {
        let mut h = TransformHierarchy::new();
        let root = h.create_node("root");
        let child = h.create_node("child");
        let grandchild = h.create_node("grandchild");
        h.set_parent(child, root);
        h.set_parent(grandchild, child);
        assert_eq!(h.depth(root), 0);
        assert_eq!(h.depth(child), 1);
        assert_eq!(h.depth(grandchild), 2);
    }

    #[test]
    fn test_roots() {
        let mut h = TransformHierarchy::new();
        let r1 = h.create_node("r1");
        let r2 = h.create_node("r2");
        let c = h.create_node("c");
        h.set_parent(c, r1);
        let roots = h.roots();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&r1));
        assert!(roots.contains(&r2));
    }

    #[test]
    fn test_look_at_matrix() {
        let m = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::zero(),
            Vec3::up(),
        );
        let p = m.transform_point(Vec3::zero());
        // RH view: objects in front of camera have negative z in view space
        assert!(approx(p.z, -5.0, 1e-9));
    }

    #[test]
    fn test_transform_look_at() {
        let mut t = Transform::new();
        t.position = Vec3::new(0.0, 0.0, 5.0);
        t.look_at(Vec3::zero(), Vec3::up());
        let forward = t.rotation.rotate_vec3(Vec3::new(0.0, 0.0, -1.0));
        let expected = Vec3::new(0.0, 0.0, -1.0);
        assert!(v3_approx(forward, expected, 1e-9));
    }

    #[test]
    fn test_reparent() {
        let mut h = TransformHierarchy::new();
        let a = h.create_node("a");
        let b = h.create_node("b");
        let child = h.create_node("child");
        h.set_parent(child, a);
        assert_eq!(h.parent(child), Some(a));
        h.set_parent(child, b);
        assert_eq!(h.parent(child), Some(b));
        assert!(h.children(a).is_empty());
        assert_eq!(h.children(b), &[child]);
    }

    #[test]
    fn test_euler_roundtrip() {
        let q = Quaternion::from_axis_angle(Vec3::up(), 0.7);
        let euler = q.to_euler();
        // For a pure Y rotation, pitch should be ~0.7
        assert!(approx(euler.y, 0.7, 1e-6));
    }

    #[test]
    fn test_mat4_mul_identity() {
        let m = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        let result = m.mul(Mat4::identity());
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx(result.m[i][j], m.m[i][j], 1e-9));
            }
        }
    }

    #[test]
    fn test_transform_scale_by() {
        let mut t = Transform::new();
        t.local_scale = Vec3::new(2.0, 2.0, 2.0);
        t.scale_by(Vec3::new(3.0, 3.0, 3.0));
        assert!(approx(t.local_scale.x, 6.0, 1e-9));
    }

    #[test]
    fn test_node_name() {
        let mut h = TransformHierarchy::new();
        let n = h.create_node("my_node");
        assert_eq!(h.name(n), "my_node");
    }
}
