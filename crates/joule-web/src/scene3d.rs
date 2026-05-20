//! Scene Graph — transform hierarchy with depth-first/breadth-first traversal,
//! world-transform computation, node reparenting, and frustum culling.

use std::collections::{HashMap, VecDeque};

use crate::webgl::{Frustum, Mat4, Quaternion, Vec3, Vec4};

// ── AABB ──────────────────────────────────────────────────────

/// Axis-Aligned Bounding Box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn half_extents(&self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// Test AABB against 6-plane frustum. Returns true if at least partially inside.
    pub fn intersects_frustum(&self, frustum: &Frustum) -> bool {
        let center = self.center();
        let half = self.half_extents();
        for plane in &frustum.planes {
            let d = center.x * plane.x + center.y * plane.y + center.z * plane.z + plane.w;
            let r = half.x * plane.x.abs() + half.y * plane.y.abs() + half.z * plane.z.abs();
            if d + r < 0.0 {
                return false;
            }
        }
        true
    }
}

// ── Transform ─────────────────────────────────────────────────

/// Local transform with position, rotation (quaternion), and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quaternion,
    pub scale: Vec3,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            position: Vec3::zero(),
            rotation: Quaternion::identity(),
            scale: Vec3::one(),
        }
    }

    pub fn from_position(pos: Vec3) -> Self {
        Self {
            position: pos,
            ..Self::identity()
        }
    }

    /// Convert this transform into a 4x4 column-major matrix: T * R * S.
    pub fn to_mat4(&self) -> Mat4 {
        let t = Mat4::translation(self.position.x, self.position.y, self.position.z);
        let r = self.rotation.to_mat4();
        let s = Mat4::scaling(self.scale.x, self.scale.y, self.scale.z);
        t.multiply(&r).multiply(&s)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

// ── SceneNode ─────────────────────────────────────────────────

/// Unique node identifier.
pub type NodeId = u64;

/// A node in the scene graph.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub transform: Transform,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub aabb: Option<Aabb>,
    pub visible: bool,
}

// ── SceneGraph ────────────────────────────────────────────────

/// Hierarchical scene graph.
pub struct SceneGraph {
    nodes: HashMap<NodeId, SceneNode>,
    root: NodeId,
    next_id: NodeId,
}

impl SceneGraph {
    pub fn new() -> Self {
        let root_id = 0;
        let root = SceneNode {
            id: root_id,
            name: "root".into(),
            transform: Transform::identity(),
            parent: None,
            children: Vec::new(),
            aabb: None,
            visible: true,
        };
        let mut nodes = HashMap::new();
        nodes.insert(root_id, root);
        Self {
            nodes,
            root: root_id,
            next_id: 1,
        }
    }

    pub fn root_id(&self) -> NodeId {
        self.root
    }

    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add a child node under the given parent. Returns the new node's id.
    pub fn add_node(&mut self, parent: NodeId, name: &str, transform: Transform) -> Option<NodeId> {
        if !self.nodes.contains_key(&parent) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let node = SceneNode {
            id,
            name: name.into(),
            transform,
            parent: Some(parent),
            children: Vec::new(),
            aabb: None,
            visible: true,
        };
        self.nodes.insert(id, node);
        self.nodes.get_mut(&parent).unwrap().children.push(id);
        Some(id)
    }

    /// Remove a node and all its descendants. Cannot remove root.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if id == self.root {
            return false;
        }
        // Collect descendants depth-first.
        let mut to_remove = Vec::new();
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            to_remove.push(current);
            if let Some(node) = self.nodes.get(&current) {
                for &child in &node.children {
                    stack.push(child);
                }
            }
        }
        // Remove from parent's children list.
        if let Some(node) = self.nodes.get(&id) {
            if let Some(parent_id) = node.parent {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.children.retain(|c| *c != id);
                }
            }
        }
        for nid in &to_remove {
            self.nodes.remove(nid);
        }
        true
    }

    /// Reparent a node to a new parent.
    pub fn reparent(&mut self, node_id: NodeId, new_parent: NodeId) -> bool {
        if node_id == self.root || !self.nodes.contains_key(&new_parent) {
            return false;
        }
        // Prevent reparenting under a descendant.
        if self.is_descendant_of(new_parent, node_id) {
            return false;
        }
        // Remove from old parent.
        if let Some(node) = self.nodes.get(&node_id) {
            if let Some(old_parent_id) = node.parent {
                if let Some(old_parent) = self.nodes.get_mut(&old_parent_id) {
                    old_parent.children.retain(|c| *c != node_id);
                }
            }
        }
        // Add to new parent.
        self.nodes.get_mut(&new_parent).unwrap().children.push(node_id);
        self.nodes.get_mut(&node_id).unwrap().parent = Some(new_parent);
        true
    }

    /// Check if `candidate` is a descendant of `ancestor`.
    fn is_descendant_of(&self, candidate: NodeId, ancestor: NodeId) -> bool {
        let mut current = candidate;
        loop {
            if current == ancestor {
                return true;
            }
            match self.nodes.get(&current).and_then(|n| n.parent) {
                Some(p) => current = p,
                None => return false,
            }
        }
    }

    /// Compute the world transform for a given node by walking up the parent chain.
    pub fn world_transform(&self, id: NodeId) -> Mat4 {
        let mut chain = Vec::new();
        let mut current = id;
        loop {
            if let Some(node) = self.nodes.get(&current) {
                chain.push(node.transform.to_mat4());
                match node.parent {
                    Some(p) => current = p,
                    None => break,
                }
            } else {
                break;
            }
        }
        // Multiply from root down: root * ... * child.
        let mut result = Mat4::identity();
        for m in chain.iter().rev() {
            result = result.multiply(m);
        }
        result
    }

    /// Depth-first traversal starting from `start`. Calls `f` with (node_id, depth).
    pub fn traverse_depth_first<F: FnMut(NodeId, usize)>(&self, start: NodeId, f: &mut F) {
        let mut stack = vec![(start, 0usize)];
        while let Some((id, depth)) = stack.pop() {
            f(id, depth);
            if let Some(node) = self.nodes.get(&id) {
                // Push in reverse so left-most child is visited first.
                for &child in node.children.iter().rev() {
                    stack.push((child, depth + 1));
                }
            }
        }
    }

    /// Breadth-first traversal starting from `start`. Calls `f` with (node_id, depth).
    pub fn traverse_breadth_first<F: FnMut(NodeId, usize)>(&self, start: NodeId, f: &mut F) {
        let mut queue = VecDeque::new();
        queue.push_back((start, 0usize));
        while let Some((id, depth)) = queue.pop_front() {
            f(id, depth);
            if let Some(node) = self.nodes.get(&id) {
                for &child in &node.children {
                    queue.push_back((child, depth + 1));
                }
            }
        }
    }

    /// Return IDs of visible nodes whose AABB is inside the frustum.
    pub fn cull(&self, frustum: &Frustum) -> Vec<NodeId> {
        let mut visible = Vec::new();
        self.traverse_depth_first(self.root, &mut |id, _depth| {
            if let Some(node) = self.nodes.get(&id) {
                if !node.visible {
                    return;
                }
                if let Some(aabb) = &node.aabb {
                    // Transform AABB center by world transform (approximation).
                    let world = self.world_transform(id);
                    let world_center = world.transform_vec3(&aabb.center());
                    let he = aabb.half_extents();
                    // Conservative: use max scale axis for radius.
                    let sx = (world.data[0] * world.data[0]
                        + world.data[1] * world.data[1]
                        + world.data[2] * world.data[2])
                        .sqrt();
                    let sy = (world.data[4] * world.data[4]
                        + world.data[5] * world.data[5]
                        + world.data[6] * world.data[6])
                        .sqrt();
                    let sz = (world.data[8] * world.data[8]
                        + world.data[9] * world.data[9]
                        + world.data[10] * world.data[10])
                        .sqrt();
                    let max_scale = sx.max(sy).max(sz);
                    let radius = he.length() * max_scale;
                    if frustum.intersects_sphere(&world_center, radius) {
                        visible.push(id);
                    }
                } else {
                    // No AABB — always include.
                    visible.push(id);
                }
            }
        });
        visible
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgl::{Camera, Frustum, Quaternion, Vec3};

    const EPS: f64 = 1e-6;

    #[test]
    fn new_graph_has_root() {
        let g = SceneGraph::new();
        assert!(g.node(g.root_id()).is_some());
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn add_and_remove_nodes() {
        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "a", Transform::identity()).unwrap();
        let b = g.add_node(a, "b", Transform::identity()).unwrap();
        assert_eq!(g.node_count(), 3);
        g.remove_node(a);
        assert_eq!(g.node_count(), 1); // only root remains
        assert!(g.node(b).is_none());
    }

    #[test]
    fn cannot_remove_root() {
        let mut g = SceneGraph::new();
        assert!(!g.remove_node(g.root_id()));
    }

    #[test]
    fn reparent_node() {
        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "a", Transform::identity()).unwrap();
        let b = g.add_node(g.root_id(), "b", Transform::identity()).unwrap();
        let c = g.add_node(a, "c", Transform::identity()).unwrap();
        assert!(g.reparent(c, b));
        assert_eq!(g.node(b).unwrap().children.len(), 1);
        assert_eq!(g.node(a).unwrap().children.len(), 0);
        assert_eq!(g.node(c).unwrap().parent, Some(b));
    }

    #[test]
    fn reparent_prevents_cycle() {
        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "a", Transform::identity()).unwrap();
        let b = g.add_node(a, "b", Transform::identity()).unwrap();
        // Cannot reparent `a` under its own descendant `b`.
        assert!(!g.reparent(a, b));
    }

    #[test]
    fn world_transform_multiplies_chain() {
        let mut g = SceneGraph::new();
        let a = g
            .add_node(
                g.root_id(),
                "a",
                Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        let b = g
            .add_node(
                a,
                "b",
                Transform::from_position(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap();

        let wt = g.world_transform(b);
        let origin = wt.transform_vec3(&Vec3::zero());
        assert!((origin.x - 1.0).abs() < EPS);
        assert!((origin.y - 2.0).abs() < EPS);
        assert!((origin.z).abs() < EPS);
    }

    #[test]
    fn depth_first_traversal() {
        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "a", Transform::identity()).unwrap();
        let b = g.add_node(g.root_id(), "b", Transform::identity()).unwrap();
        let c = g.add_node(a, "c", Transform::identity()).unwrap();

        let mut visited = Vec::new();
        g.traverse_depth_first(g.root_id(), &mut |id, depth| {
            visited.push((id, depth));
        });
        // Root(0), a(1), c(2), b(1)
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0].0, g.root_id());
        assert_eq!(visited[1].0, a);
        assert_eq!(visited[2].0, c);
        assert_eq!(visited[3].0, b);
    }

    #[test]
    fn breadth_first_traversal() {
        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "a", Transform::identity()).unwrap();
        let b = g.add_node(g.root_id(), "b", Transform::identity()).unwrap();
        let _c = g.add_node(a, "c", Transform::identity()).unwrap();

        let mut visited = Vec::new();
        g.traverse_breadth_first(g.root_id(), &mut |id, depth| {
            visited.push((id, depth));
        });
        // Root(depth 0), a(1), b(1), c(2)
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0].1, 0);
        assert_eq!(visited[1].1, 1);
        assert_eq!(visited[2].1, 1);
        assert_eq!(visited[3].1, 2);
    }

    #[test]
    fn aabb_basics() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let c = aabb.center();
        assert!((c.x).abs() < EPS);
        let he = aabb.half_extents();
        assert!((he.x - 1.0).abs() < EPS);
    }

    #[test]
    fn frustum_culling() {
        let cam = Camera::new();
        let vp = cam.view_projection();
        let frustum = Frustum::from_view_projection(&vp);

        let mut g = SceneGraph::new();
        // Node at origin — visible.
        let a = g.add_node(g.root_id(), "visible", Transform::identity()).unwrap();
        g.node_mut(a).unwrap().aabb =
            Some(Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)));

        // Node far behind camera — invisible.
        let b = g
            .add_node(
                g.root_id(),
                "hidden",
                Transform::from_position(Vec3::new(0.0, 0.0, 1000.0)),
            )
            .unwrap();
        g.node_mut(b).unwrap().aabb =
            Some(Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1)));

        let visible = g.cull(&frustum);
        assert!(visible.contains(&a));
        assert!(!visible.contains(&b));
    }

    #[test]
    fn transform_to_mat4_roundtrip() {
        let t = Transform {
            position: Vec3::new(3.0, -1.0, 2.0),
            rotation: Quaternion::from_axis_angle(&Vec3::up(), std::f64::consts::FRAC_PI_4),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let m = t.to_mat4();
        // Origin in local space should map to position in world.
        let origin = m.transform_vec3(&Vec3::zero());
        assert!((origin.x - 3.0).abs() < EPS);
        assert!((origin.y - (-1.0)).abs() < EPS);
        assert!((origin.z - 2.0).abs() < EPS);
    }

    #[test]
    fn invisible_node_excluded_from_cull() {
        let cam = Camera::new();
        let vp = cam.view_projection();
        let frustum = Frustum::from_view_projection(&vp);

        let mut g = SceneGraph::new();
        let a = g.add_node(g.root_id(), "hidden", Transform::identity()).unwrap();
        g.node_mut(a).unwrap().visible = false;
        let visible = g.cull(&frustum);
        assert!(!visible.contains(&a));
    }
}
