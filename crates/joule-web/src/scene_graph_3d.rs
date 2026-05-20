//! 3D scene graph with parent-child transform hierarchy.
//!
//! Each node has a local transform (position, rotation as quaternion, scale)
//! and a computed world transform. Dirty-flag propagation ensures world
//! transforms are recomputed only when needed. Supports add, remove,
//! reparent, depth-first traversal, and find-by-name.

use std::collections::HashMap;

// ── Quaternion ──────────────────────────────────────────────────

/// Unit quaternion for 3D rotation (w + xi + yj + zk).
#[derive(Debug, Clone, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Construct from axis-angle (axis must be unit length, angle in radians).
    pub fn from_axis_angle(ax: f64, ay: f64, az: f64, angle: f64) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        Self { w: half.cos(), x: ax * s, y: ay * s, z: az * s }
    }

    pub fn length_sq(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Self {
        let len = self.length_sq().sqrt();
        if len < 1e-12 {
            return Self::identity();
        }
        Self {
            w: self.w / len,
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    /// Hamilton product: self * other.
    pub fn mul(&self, o: &Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    /// Rotate a 3D vector by this quaternion.
    pub fn rotate_vec(&self, vx: f64, vy: f64, vz: f64) -> (f64, f64, f64) {
        let qv = Quat { w: 0.0, x: vx, y: vy, z: vz };
        let conj = Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z };
        let result = self.mul(&qv).mul(&conj);
        (result.x, result.y, result.z)
    }

    /// Convert to a 4x4 column-major rotation matrix.
    pub fn to_mat4(&self) -> [f64; 16] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let x2 = x + x; let y2 = y + y; let z2 = z + z;
        let xx = x * x2; let xy = x * y2; let xz = x * z2;
        let yy = y * y2; let yz = y * z2; let zz = z * z2;
        let wx = w * x2; let wy = w * y2; let wz = w * z2;
        [
            1.0 - (yy + zz), xy + wz,         xz - wy,         0.0,
            xy - wz,         1.0 - (xx + zz),  yz + wx,         0.0,
            xz + wy,         yz - wx,          1.0 - (xx + yy), 0.0,
            0.0,             0.0,              0.0,              1.0,
        ]
    }
}

// ── Vec3 helper ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn one() -> Self { Self { x: 1.0, y: 1.0, z: 1.0 } }
    pub fn add(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn scale(&self, s: f64) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) }
    pub fn mul_comp(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z) }
}

// ── Transform ───────────────────────────────────────────────────

/// Local transform: position, rotation, and non-uniform scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            position: Vec3::zero(),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        }
    }

    /// Compose a parent * child transform.
    pub fn compose(&self, child: &Transform) -> Transform {
        let scaled_pos = Vec3::new(
            child.position.x * self.scale.x,
            child.position.y * self.scale.y,
            child.position.z * self.scale.z,
        );
        let (rx, ry, rz) = self.rotation.rotate_vec(scaled_pos.x, scaled_pos.y, scaled_pos.z);
        Transform {
            position: Vec3::new(self.position.x + rx, self.position.y + ry, self.position.z + rz),
            rotation: self.rotation.mul(&child.rotation).normalize(),
            scale: self.scale.mul_comp(&child.scale),
        }
    }

    /// Build a 4x4 column-major TRS matrix.
    pub fn to_mat4(&self) -> [f64; 16] {
        let mut m = self.rotation.to_mat4();
        // Apply scale to rotation columns.
        m[0] *= self.scale.x; m[1] *= self.scale.x; m[2] *= self.scale.x;
        m[4] *= self.scale.y; m[5] *= self.scale.y; m[6] *= self.scale.y;
        m[8] *= self.scale.z; m[9] *= self.scale.z; m[10] *= self.scale.z;
        // Translation in column 3.
        m[12] = self.position.x;
        m[13] = self.position.y;
        m[14] = self.position.z;
        m
    }
}

// ── Scene node ──────────────────────────────────────────────────

type NodeId = u64;

#[derive(Debug, Clone)]
struct SceneNode {
    id: NodeId,
    name: String,
    local: Transform,
    world: Transform,
    dirty: bool,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
}

// ── Scene graph ─────────────────────────────────────────────────

/// A 3D scene graph with hierarchical transforms.
#[derive(Debug)]
pub struct SceneGraph3D {
    nodes: HashMap<NodeId, SceneNode>,
    roots: Vec<NodeId>,
    next_id: NodeId,
}

impl SceneGraph3D {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), roots: Vec::new(), next_id: 1 }
    }

    /// Add a root node. Returns its ID.
    pub fn add_root(&mut self, name: &str, local: Transform) -> NodeId {
        let id = self.alloc_id();
        let node = SceneNode {
            id,
            name: name.to_string(),
            local,
            world: Transform::identity(),
            dirty: true,
            parent: None,
            children: Vec::new(),
        };
        self.nodes.insert(id, node);
        self.roots.push(id);
        id
    }

    /// Add a child node under `parent_id`. Returns its ID or None if parent missing.
    pub fn add_child(&mut self, parent_id: NodeId, name: &str, local: Transform) -> Option<NodeId> {
        if !self.nodes.contains_key(&parent_id) {
            return None;
        }
        let id = self.alloc_id();
        let node = SceneNode {
            id,
            name: name.to_string(),
            local,
            world: Transform::identity(),
            dirty: true,
            parent: Some(parent_id),
            children: Vec::new(),
        };
        self.nodes.insert(id, node);
        self.nodes.get_mut(&parent_id).unwrap().children.push(id);
        Some(id)
    }

    /// Remove a node and all its descendants. Returns number of nodes removed.
    pub fn remove(&mut self, id: NodeId) -> usize {
        let ids = self.collect_subtree(id);
        if ids.is_empty() {
            return 0;
        }
        // Detach from parent.
        if let Some(node) = self.nodes.get(&id) {
            let parent = node.parent;
            if let Some(pid) = parent {
                if let Some(p) = self.nodes.get_mut(&pid) {
                    p.children.retain(|c| *c != id);
                }
            }
        }
        self.roots.retain(|r| *r != id);
        for &nid in &ids {
            self.nodes.remove(&nid);
        }
        ids.len()
    }

    /// Reparent a node under a new parent (or make it a root if new_parent is None).
    pub fn reparent(&mut self, id: NodeId, new_parent: Option<NodeId>) -> bool {
        if !self.nodes.contains_key(&id) {
            return false;
        }
        if let Some(np) = new_parent {
            if !self.nodes.contains_key(&np) || np == id {
                return false;
            }
            // Prevent reparenting under own descendant.
            let subtree = self.collect_subtree(id);
            if subtree.contains(&np) {
                return false;
            }
        }

        // Detach from old parent.
        let old_parent = self.nodes.get(&id).unwrap().parent;
        if let Some(op) = old_parent {
            if let Some(p) = self.nodes.get_mut(&op) {
                p.children.retain(|c| *c != id);
            }
        }
        self.roots.retain(|r| *r != id);

        // Attach to new parent.
        match new_parent {
            Some(np) => {
                self.nodes.get_mut(&np).unwrap().children.push(id);
                self.nodes.get_mut(&id).unwrap().parent = Some(np);
            }
            None => {
                self.nodes.get_mut(&id).unwrap().parent = None;
                self.roots.push(id);
            }
        }
        self.mark_dirty(id);
        true
    }

    /// Set the local transform of a node, marking it (and descendants) dirty.
    pub fn set_local(&mut self, id: NodeId, local: Transform) -> bool {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.local = local;
            node.dirty = true;
            let children: Vec<NodeId> = node.children.clone();
            for cid in children {
                self.mark_dirty(cid);
            }
            true
        } else {
            false
        }
    }

    /// Update all dirty world transforms.
    pub fn update_transforms(&mut self) {
        let root_ids: Vec<NodeId> = self.roots.clone();
        for rid in root_ids {
            self.update_node(rid, &Transform::identity());
        }
    }

    /// Get the world transform of a node (call after `update_transforms`).
    pub fn world_transform(&self, id: NodeId) -> Option<&Transform> {
        self.nodes.get(&id).map(|n| &n.world)
    }

    /// Get the local transform of a node.
    pub fn local_transform(&self, id: NodeId) -> Option<&Transform> {
        self.nodes.get(&id).map(|n| &n.local)
    }

    /// Find a node by name (first match, depth-first from roots).
    pub fn find_by_name(&self, name: &str) -> Option<NodeId> {
        let root_ids: Vec<NodeId> = self.roots.clone();
        for rid in root_ids {
            if let Some(id) = self.find_name_dfs(rid, name) {
                return Some(id);
            }
        }
        None
    }

    /// Depth-first traversal, calling `visitor(id, &name, &world_transform)`.
    pub fn traverse_depth_first<F>(&self, mut visitor: F)
    where
        F: FnMut(NodeId, &str, &Transform),
    {
        let root_ids: Vec<NodeId> = self.roots.clone();
        for rid in root_ids {
            self.visit_dfs(rid, &mut visitor);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes.get(&id).map(|n| n.children.clone()).unwrap_or_default()
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes.get(&id).and_then(|n| n.parent)
    }

    pub fn name(&self, id: NodeId) -> Option<&str> {
        self.nodes.get(&id).map(|n| n.name.as_str())
    }

    // ── Private helpers ─────────────────────────────────────────

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn mark_dirty(&mut self, id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.dirty = true;
            let children: Vec<NodeId> = node.children.clone();
            for cid in children {
                self.mark_dirty(cid);
            }
        }
    }

    fn collect_subtree(&self, id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        self.collect_subtree_inner(id, &mut result);
        result
    }

    fn collect_subtree_inner(&self, id: NodeId, out: &mut Vec<NodeId>) {
        if let Some(node) = self.nodes.get(&id) {
            out.push(id);
            for &cid in &node.children {
                self.collect_subtree_inner(cid, out);
            }
        }
    }

    fn update_node(&mut self, id: NodeId, parent_world: &Transform) {
        let (dirty, local, children) = {
            let node = match self.nodes.get(&id) {
                Some(n) => n,
                None => return,
            };
            (node.dirty, node.local.clone(), node.children.clone())
        };

        let world = if dirty {
            let w = parent_world.compose(&local);
            if let Some(node) = self.nodes.get_mut(&id) {
                node.world = w.clone();
                node.dirty = false;
            }
            w
        } else {
            self.nodes.get(&id).unwrap().world.clone()
        };

        for cid in children {
            self.update_node(cid, &world);
        }
    }

    fn find_name_dfs(&self, id: NodeId, name: &str) -> Option<NodeId> {
        let node = self.nodes.get(&id)?;
        if node.name == name {
            return Some(id);
        }
        for &cid in &node.children {
            if let Some(found) = self.find_name_dfs(cid, name) {
                return Some(found);
            }
        }
        None
    }

    fn visit_dfs<F>(&self, id: NodeId, visitor: &mut F)
    where
        F: FnMut(NodeId, &str, &Transform),
    {
        if let Some(node) = self.nodes.get(&id) {
            visitor(node.id, &node.name, &node.world);
            for &cid in &node.children {
                self.visit_dfs(cid, visitor);
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn assert_vec3_eq(a: &Vec3, b: &Vec3) {
        assert!((a.x - b.x).abs() < EPS, "x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < EPS, "y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < EPS, "z: {} vs {}", a.z, b.z);
    }

    #[test]
    fn test_quat_identity() {
        let q = Quat::identity();
        let (rx, ry, rz) = q.rotate_vec(1.0, 0.0, 0.0);
        assert!((rx - 1.0).abs() < EPS);
        assert!(ry.abs() < EPS);
        assert!(rz.abs() < EPS);
    }

    #[test]
    fn test_quat_rotate_90_around_z() {
        let q = Quat::from_axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let (rx, ry, rz) = q.rotate_vec(1.0, 0.0, 0.0);
        assert!(rx.abs() < EPS);
        assert!((ry - 1.0).abs() < EPS);
        assert!(rz.abs() < EPS);
    }

    #[test]
    fn test_quat_normalize() {
        let q = Quat { w: 2.0, x: 0.0, y: 0.0, z: 0.0 };
        let n = q.normalize();
        assert!((n.length_sq() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_transform_identity_compose() {
        let parent = Transform::identity();
        let child = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        };
        let result = parent.compose(&child);
        assert_vec3_eq(&result.position, &Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_transform_scaled_compose() {
        let parent = Transform {
            position: Vec3::zero(),
            rotation: Quat::identity(),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let child = Transform {
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        };
        let result = parent.compose(&child);
        assert_vec3_eq(&result.position, &Vec3::new(2.0, 0.0, 0.0));
        assert_vec3_eq(&result.scale, &Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_add_root() {
        let mut sg = SceneGraph3D::new();
        let id = sg.add_root("root", Transform::identity());
        assert_eq!(sg.node_count(), 1);
        assert_eq!(sg.name(id), Some("root"));
    }

    #[test]
    fn test_add_child() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        let child = sg.add_child(root, "child", Transform::identity());
        assert!(child.is_some());
        assert_eq!(sg.node_count(), 2);
        assert_eq!(sg.parent(child.unwrap()), Some(root));
    }

    #[test]
    fn test_add_child_invalid_parent() {
        let mut sg = SceneGraph3D::new();
        let result = sg.add_child(999, "orphan", Transform::identity());
        assert!(result.is_none());
    }

    #[test]
    fn test_remove_subtree() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        let c1 = sg.add_child(root, "c1", Transform::identity()).unwrap();
        sg.add_child(c1, "gc1", Transform::identity());
        let removed = sg.remove(c1);
        assert_eq!(removed, 2);
        assert_eq!(sg.node_count(), 1);
    }

    #[test]
    fn test_remove_root() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        sg.remove(root);
        assert_eq!(sg.node_count(), 0);
    }

    #[test]
    fn test_reparent() {
        let mut sg = SceneGraph3D::new();
        let a = sg.add_root("a", Transform::identity());
        let b = sg.add_root("b", Transform::identity());
        let c = sg.add_child(a, "c", Transform::identity()).unwrap();
        assert!(sg.reparent(c, Some(b)));
        assert_eq!(sg.parent(c), Some(b));
        assert!(sg.children(a).is_empty());
        assert_eq!(sg.children(b), vec![c]);
    }

    #[test]
    fn test_reparent_to_root() {
        let mut sg = SceneGraph3D::new();
        let a = sg.add_root("a", Transform::identity());
        let c = sg.add_child(a, "c", Transform::identity()).unwrap();
        assert!(sg.reparent(c, None));
        assert_eq!(sg.parent(c), None);
    }

    #[test]
    fn test_reparent_under_descendant_fails() {
        let mut sg = SceneGraph3D::new();
        let a = sg.add_root("a", Transform::identity());
        let b = sg.add_child(a, "b", Transform::identity()).unwrap();
        assert!(!sg.reparent(a, Some(b)));
    }

    #[test]
    fn test_world_transform_propagation() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform {
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        });
        sg.add_child(root, "child", Transform {
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        });
        sg.update_transforms();
        let child_id = sg.find_by_name("child").unwrap();
        let wt = sg.world_transform(child_id).unwrap();
        assert_vec3_eq(&wt.position, &Vec3::new(15.0, 0.0, 0.0));
    }

    #[test]
    fn test_dirty_flag_set_local() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        sg.update_transforms();
        sg.set_local(root, Transform {
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        });
        sg.update_transforms();
        let wt = sg.world_transform(root).unwrap();
        assert_vec3_eq(&wt.position, &Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_find_by_name() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        sg.add_child(root, "target", Transform::identity());
        assert!(sg.find_by_name("target").is_some());
        assert!(sg.find_by_name("missing").is_none());
    }

    #[test]
    fn test_depth_first_traversal() {
        let mut sg = SceneGraph3D::new();
        let a = sg.add_root("a", Transform::identity());
        let b = sg.add_child(a, "b", Transform::identity()).unwrap();
        sg.add_child(b, "c", Transform::identity());
        sg.add_child(a, "d", Transform::identity());
        sg.update_transforms();

        let mut order = Vec::new();
        sg.traverse_depth_first(|_id, name, _t| {
            order.push(name.to_string());
        });
        assert_eq!(order, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_transform_to_mat4_identity() {
        let t = Transform::identity();
        let m = t.to_mat4();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((m[j * 4 + i] - expected).abs() < EPS);
            }
        }
    }

    #[test]
    fn test_multiple_roots() {
        let mut sg = SceneGraph3D::new();
        sg.add_root("r1", Transform { position: Vec3::new(1.0, 0.0, 0.0), rotation: Quat::identity(), scale: Vec3::one() });
        sg.add_root("r2", Transform { position: Vec3::new(2.0, 0.0, 0.0), rotation: Quat::identity(), scale: Vec3::one() });
        sg.update_transforms();
        let id1 = sg.find_by_name("r1").unwrap();
        let id2 = sg.find_by_name("r2").unwrap();
        assert_vec3_eq(&sg.world_transform(id1).unwrap().position, &Vec3::new(1.0, 0.0, 0.0));
        assert_vec3_eq(&sg.world_transform(id2).unwrap().position, &Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut sg = SceneGraph3D::new();
        assert_eq!(sg.remove(999), 0);
    }

    #[test]
    fn test_children_accessor() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform::identity());
        let c1 = sg.add_child(root, "c1", Transform::identity()).unwrap();
        let c2 = sg.add_child(root, "c2", Transform::identity()).unwrap();
        let kids = sg.children(root);
        assert_eq!(kids.len(), 2);
        assert!(kids.contains(&c1));
        assert!(kids.contains(&c2));
    }

    #[test]
    fn test_scale_chain_propagation() {
        let mut sg = SceneGraph3D::new();
        let root = sg.add_root("root", Transform {
            position: Vec3::zero(),
            rotation: Quat::identity(),
            scale: Vec3::new(2.0, 2.0, 2.0),
        });
        let child = sg.add_child(root, "child", Transform {
            position: Vec3::zero(),
            rotation: Quat::identity(),
            scale: Vec3::new(3.0, 3.0, 3.0),
        }).unwrap();
        sg.update_transforms();
        let wt = sg.world_transform(child).unwrap();
        assert_vec3_eq(&wt.scale, &Vec3::new(6.0, 6.0, 6.0));
    }
}
