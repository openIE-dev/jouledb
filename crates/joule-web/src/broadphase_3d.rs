//! 3D Broadphase — Dynamic AABB Tree (DBVT) with fat AABBs for movement
//! prediction, overlap-pair queries, ray cast with early termination,
//! frustum query, tree balancing via rotation, and surface-area cost heuristic.

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn min_comp(self, r: Self) -> Self {
        Self { x: self.x.min(r.x), y: self.y.min(r.y), z: self.z.min(r.z) }
    }
    pub fn max_comp(self, r: Self) -> Self {
        Self { x: self.x.max(r.x), y: self.y.max(r.y), z: self.z.max(r.z) }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}

// ── AABB ─────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Surface area (used as cost heuristic).
    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.x * d.z)
    }

    /// Volume.
    pub fn volume(&self) -> f64 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    /// Expand by a uniform margin (fat AABB).
    pub fn expanded(&self, margin: f64) -> Self {
        let m = Vec3::new(margin, margin, margin);
        Self { min: self.min - m, max: self.max + m }
    }

    /// Merge two AABBs.
    pub fn merged(&self, other: &Aabb3) -> Self {
        Self {
            min: self.min.min_comp(other.min),
            max: self.max.max_comp(other.max),
        }
    }

    /// Test overlap.
    pub fn overlaps(&self, other: &Aabb3) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Contains another AABB entirely.
    pub fn contains(&self, other: &Aabb3) -> bool {
        self.min.x <= other.min.x && self.max.x >= other.max.x
            && self.min.y <= other.min.y && self.max.y >= other.max.y
            && self.min.z <= other.min.z && self.max.z >= other.max.z
    }

    /// Ray-AABB intersection: returns Some(t) for the nearest hit or None.
    pub fn ray_intersect(&self, origin: &Vec3, inv_dir: &Vec3) -> Option<f64> {
        let t1 = (self.min.x - origin.x) * inv_dir.x;
        let t2 = (self.max.x - origin.x) * inv_dir.x;
        let t3 = (self.min.y - origin.y) * inv_dir.y;
        let t4 = (self.max.y - origin.y) * inv_dir.y;
        let t5 = (self.min.z - origin.z) * inv_dir.z;
        let t6 = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax { None } else { Some(if tmin >= 0.0 { tmin } else { tmax }) }
    }
}

// ── Frustum (6-plane) ────────────────────────────────────────

/// A frustum defined by 6 planes. Each plane is (normal_x, normal_y, normal_z, distance).
/// Points are "inside" if dot(normal, point) + d >= 0 for all planes.
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [(f64, f64, f64, f64); 6],
}

impl Frustum {
    pub fn new(planes: [(f64, f64, f64, f64); 6]) -> Self { Self { planes } }

    /// Test if an AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb3) -> bool {
        for &(nx, ny, nz, d) in &self.planes {
            // Use the p-vertex (the corner most in the direction of the normal)
            let px = if nx >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if ny >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if nz >= 0.0 { aabb.max.z } else { aabb.min.z };
            if nx * px + ny * py + nz * pz + d < 0.0 {
                return false;
            }
        }
        true
    }
}

// ── DBVT Node ────────────────────────────────────────────────

pub type NodeId = u32;
pub type ProxyId = u32;

const NULL_NODE: NodeId = u32::MAX;

#[derive(Debug, Clone)]
struct Node {
    aabb: Aabb3,
    parent: NodeId,
    left: NodeId,
    right: NodeId,
    height: i32,
    proxy_id: Option<ProxyId>, // Some for leaf, None for internal
}

impl Node {
    fn is_leaf(&self) -> bool { self.left == NULL_NODE }
}

// ── Dynamic AABB Tree ────────────────────────────────────────

/// Dynamic bounding-volume tree for broadphase collision detection.
/// Uses surface-area heuristic (SAH) for insertion and tree rotations
/// for balancing.
pub struct DynamicAabbTree {
    nodes: Vec<Node>,
    root: NodeId,
    free_list: Vec<NodeId>,
    margin: f64,
    proxy_to_node: Vec<(ProxyId, NodeId)>,
    next_proxy: ProxyId,
}

impl DynamicAabbTree {
    pub fn new(margin: f64) -> Self {
        Self {
            nodes: Vec::new(),
            root: NULL_NODE,
            free_list: Vec::new(),
            margin,
            proxy_to_node: Vec::new(),
            next_proxy: 0,
        }
    }

    fn alloc_node(&mut self) -> NodeId {
        if let Some(id) = self.free_list.pop() {
            id
        } else {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Node {
                aabb: Aabb3::new(Vec3::ZERO, Vec3::ZERO),
                parent: NULL_NODE,
                left: NULL_NODE,
                right: NULL_NODE,
                height: 0,
                proxy_id: None,
            });
            id
        }
    }

    fn free_node(&mut self, id: NodeId) {
        self.nodes[id as usize].parent = NULL_NODE;
        self.nodes[id as usize].left = NULL_NODE;
        self.nodes[id as usize].right = NULL_NODE;
        self.nodes[id as usize].height = -1;
        self.nodes[id as usize].proxy_id = None;
        self.free_list.push(id);
    }

    /// Insert a new proxy with the given tight AABB. Returns a proxy id.
    pub fn insert(&mut self, tight_aabb: Aabb3) -> ProxyId {
        let fat = tight_aabb.expanded(self.margin);
        let leaf = self.alloc_node();
        self.nodes[leaf as usize].aabb = fat;
        self.nodes[leaf as usize].height = 0;
        let pid = self.next_proxy;
        self.next_proxy += 1;
        self.nodes[leaf as usize].proxy_id = Some(pid);

        self.insert_leaf(leaf);
        self.proxy_to_node.push((pid, leaf));
        pid
    }

    /// Remove a proxy by id.
    pub fn remove(&mut self, proxy_id: ProxyId) {
        if let Some(pos) = self.proxy_to_node.iter().position(|(p, _)| *p == proxy_id) {
            let (_, node_id) = self.proxy_to_node.swap_remove(pos);
            self.remove_leaf(node_id);
            self.free_node(node_id);
        }
    }

    /// Update a proxy's AABB. Returns true if the tree was actually updated.
    pub fn update(&mut self, proxy_id: ProxyId, new_aabb: Aabb3) -> bool {
        if let Some(pos) = self.proxy_to_node.iter().position(|(p, _)| *p == proxy_id) {
            let node_id = self.proxy_to_node[pos].1;
            let fat = self.nodes[node_id as usize].aabb;
            if fat.contains(&new_aabb) {
                return false; // still fits in fat AABB
            }
            self.remove_leaf(node_id);
            self.nodes[node_id as usize].aabb = new_aabb.expanded(self.margin);
            self.insert_leaf(node_id);
            true
        } else {
            false
        }
    }

    /// Get all overlapping proxy pairs.
    pub fn query_pairs(&self) -> Vec<(ProxyId, ProxyId)> {
        let mut pairs = Vec::new();
        let leaves: Vec<(ProxyId, Aabb3)> = self.proxy_to_node.iter()
            .map(|(pid, nid)| (*pid, self.nodes[*nid as usize].aabb))
            .collect();

        for i in 0..leaves.len() {
            for j in (i + 1)..leaves.len() {
                if leaves[i].1.overlaps(&leaves[j].1) {
                    pairs.push((leaves[i].0, leaves[j].0));
                }
            }
        }
        pairs
    }

    /// Query all proxies whose fat AABB overlaps the given AABB.
    pub fn query_aabb(&self, aabb: &Aabb3) -> Vec<ProxyId> {
        let mut result = Vec::new();
        if self.root == NULL_NODE { return result; }
        let mut stack = vec![self.root];
        while let Some(nid) = stack.pop() {
            let node = &self.nodes[nid as usize];
            if !node.aabb.overlaps(aabb) { continue; }
            if node.is_leaf() {
                if let Some(pid) = node.proxy_id {
                    result.push(pid);
                }
            } else {
                if node.left != NULL_NODE { stack.push(node.left); }
                if node.right != NULL_NODE { stack.push(node.right); }
            }
        }
        result
    }

    /// Ray cast through the tree. Returns all proxy IDs hit, sorted by t.
    pub fn ray_cast(&self, origin: Vec3, direction: Vec3) -> Vec<(ProxyId, f64)> {
        let inv_dir = Vec3::new(
            if direction.x.abs() > 1e-12 { 1.0 / direction.x } else { f64::MAX.copysign(direction.x) },
            if direction.y.abs() > 1e-12 { 1.0 / direction.y } else { f64::MAX.copysign(direction.y) },
            if direction.z.abs() > 1e-12 { 1.0 / direction.z } else { f64::MAX.copysign(direction.z) },
        );
        let mut hits = Vec::new();
        if self.root == NULL_NODE { return hits; }

        let mut stack = vec![self.root];
        while let Some(nid) = stack.pop() {
            let node = &self.nodes[nid as usize];
            if node.aabb.ray_intersect(&origin, &inv_dir).is_none() {
                continue;
            }
            if node.is_leaf() {
                if let Some(pid) = node.proxy_id {
                    if let Some(t) = node.aabb.ray_intersect(&origin, &inv_dir) {
                        hits.push((pid, t));
                    }
                }
            } else {
                if node.left != NULL_NODE { stack.push(node.left); }
                if node.right != NULL_NODE { stack.push(node.right); }
            }
        }
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Frustum query: return all proxy IDs whose AABB overlaps the frustum.
    pub fn query_frustum(&self, frustum: &Frustum) -> Vec<ProxyId> {
        let mut result = Vec::new();
        if self.root == NULL_NODE { return result; }
        let mut stack = vec![self.root];
        while let Some(nid) = stack.pop() {
            let node = &self.nodes[nid as usize];
            if !frustum.intersects_aabb(&node.aabb) { continue; }
            if node.is_leaf() {
                if let Some(pid) = node.proxy_id {
                    result.push(pid);
                }
            } else {
                if node.left != NULL_NODE { stack.push(node.left); }
                if node.right != NULL_NODE { stack.push(node.right); }
            }
        }
        result
    }

    /// Number of leaf proxies.
    pub fn proxy_count(&self) -> usize {
        self.proxy_to_node.len()
    }

    /// Tree height (0 for empty, 1 for single leaf).
    pub fn height(&self) -> i32 {
        if self.root == NULL_NODE { 0 } else { self.nodes[self.root as usize].height + 1 }
    }

    /// Get the fat AABB for a proxy.
    pub fn get_fat_aabb(&self, proxy_id: ProxyId) -> Option<Aabb3> {
        self.proxy_to_node.iter()
            .find(|(p, _)| *p == proxy_id)
            .map(|(_, nid)| self.nodes[*nid as usize].aabb)
    }

    // ── Internal: insert a leaf into the tree using SAH ──

    fn insert_leaf(&mut self, leaf: NodeId) {
        if self.root == NULL_NODE {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NULL_NODE;
            return;
        }

        // Walk the tree using SAH to find the best sibling
        let leaf_aabb = self.nodes[leaf as usize].aabb;
        let mut sibling = self.root;

        while !self.nodes[sibling as usize].is_leaf() {
            let node = &self.nodes[sibling as usize];
            let combined = node.aabb.merged(&leaf_aabb);
            let combined_sa = combined.surface_area();
            let cost = 2.0 * combined_sa;
            let inherit_cost = 2.0 * (combined_sa - node.aabb.surface_area());

            let left = node.left;
            let right = node.right;

            let left_cost = if self.nodes[left as usize].is_leaf() {
                leaf_aabb.merged(&self.nodes[left as usize].aabb).surface_area() + inherit_cost
            } else {
                let new_sa = leaf_aabb.merged(&self.nodes[left as usize].aabb).surface_area();
                let old_sa = self.nodes[left as usize].aabb.surface_area();
                new_sa - old_sa + inherit_cost
            };

            let right_cost = if self.nodes[right as usize].is_leaf() {
                leaf_aabb.merged(&self.nodes[right as usize].aabb).surface_area() + inherit_cost
            } else {
                let new_sa = leaf_aabb.merged(&self.nodes[right as usize].aabb).surface_area();
                let old_sa = self.nodes[right as usize].aabb.surface_area();
                new_sa - old_sa + inherit_cost
            };

            if cost < left_cost && cost < right_cost {
                break;
            }
            sibling = if left_cost < right_cost { left } else { right };
        }

        // Create a new internal node
        let old_parent = self.nodes[sibling as usize].parent;
        let new_parent = self.alloc_node();
        self.nodes[new_parent as usize].parent = old_parent;
        self.nodes[new_parent as usize].aabb = leaf_aabb.merged(&self.nodes[sibling as usize].aabb);
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;
        self.nodes[new_parent as usize].left = sibling;
        self.nodes[new_parent as usize].right = leaf;

        self.nodes[sibling as usize].parent = new_parent;
        self.nodes[leaf as usize].parent = new_parent;

        if old_parent != NULL_NODE {
            if self.nodes[old_parent as usize].left == sibling {
                self.nodes[old_parent as usize].left = new_parent;
            } else {
                self.nodes[old_parent as usize].right = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        // Walk back up, fixing heights and AABBs
        self.fix_upward(new_parent);
    }

    fn remove_leaf(&mut self, leaf: NodeId) {
        if leaf == self.root {
            self.root = NULL_NODE;
            return;
        }

        let parent = self.nodes[leaf as usize].parent;
        let grandparent = self.nodes[parent as usize].parent;
        let sibling = if self.nodes[parent as usize].left == leaf {
            self.nodes[parent as usize].right
        } else {
            self.nodes[parent as usize].left
        };

        if grandparent != NULL_NODE {
            if self.nodes[grandparent as usize].left == parent {
                self.nodes[grandparent as usize].left = sibling;
            } else {
                self.nodes[grandparent as usize].right = sibling;
            }
            self.nodes[sibling as usize].parent = grandparent;
            self.free_node(parent);
            self.fix_upward(grandparent);
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].parent = NULL_NODE;
            self.free_node(parent);
        }
    }

    fn fix_upward(&mut self, mut idx: NodeId) {
        while idx != NULL_NODE {
            idx = self.balance(idx);
            let left = self.nodes[idx as usize].left;
            let right = self.nodes[idx as usize].right;
            if left != NULL_NODE && right != NULL_NODE {
                self.nodes[idx as usize].height = 1 + self.nodes[left as usize].height
                    .max(self.nodes[right as usize].height);
                self.nodes[idx as usize].aabb = self.nodes[left as usize].aabb
                    .merged(&self.nodes[right as usize].aabb);
            }
            idx = self.nodes[idx as usize].parent;
        }
    }

    /// Tree rotation for balancing. Returns the (possibly new) root of subtree.
    fn balance(&mut self, a: NodeId) -> NodeId {
        if self.nodes[a as usize].is_leaf() || self.nodes[a as usize].height < 2 {
            return a;
        }
        let b = self.nodes[a as usize].left;
        let c = self.nodes[a as usize].right;
        let balance_factor = self.nodes[c as usize].height - self.nodes[b as usize].height;

        // Rotate C up
        if balance_factor > 1 {
            let f = self.nodes[c as usize].left;
            let g = self.nodes[c as usize].right;

            // Swap A and C
            self.nodes[c as usize].left = a;
            self.nodes[c as usize].parent = self.nodes[a as usize].parent;
            self.nodes[a as usize].parent = c;

            if self.nodes[c as usize].parent != NULL_NODE {
                let cp = self.nodes[c as usize].parent;
                if self.nodes[cp as usize].left == a {
                    self.nodes[cp as usize].left = c;
                } else {
                    self.nodes[cp as usize].right = c;
                }
            } else {
                self.root = c;
            }

            // Rotate: pick which child of C becomes child of A
            if f != NULL_NODE && g != NULL_NODE
                && self.nodes[f as usize].height > self.nodes[g as usize].height
            {
                self.nodes[c as usize].right = f;
                self.nodes[a as usize].right = g;
                self.nodes[g as usize].parent = a;
                self.nodes[a as usize].aabb = self.nodes[b as usize].aabb
                    .merged(&self.nodes[g as usize].aabb);
                self.nodes[c as usize].aabb = self.nodes[a as usize].aabb
                    .merged(&self.nodes[f as usize].aabb);
                self.nodes[a as usize].height = 1 + self.nodes[b as usize].height
                    .max(self.nodes[g as usize].height);
                self.nodes[c as usize].height = 1 + self.nodes[a as usize].height
                    .max(self.nodes[f as usize].height);
            } else {
                self.nodes[c as usize].right = g;
                self.nodes[a as usize].right = f;
                if f != NULL_NODE { self.nodes[f as usize].parent = a; }
                let b_aabb = self.nodes[b as usize].aabb;
                let f_aabb = if f != NULL_NODE { self.nodes[f as usize].aabb } else { b_aabb };
                self.nodes[a as usize].aabb = b_aabb.merged(&f_aabb);
                let g_aabb = if g != NULL_NODE { self.nodes[g as usize].aabb } else { self.nodes[a as usize].aabb };
                self.nodes[c as usize].aabb = self.nodes[a as usize].aabb.merged(&g_aabb);
                let bh = self.nodes[b as usize].height;
                let fh = if f != NULL_NODE { self.nodes[f as usize].height } else { 0 };
                self.nodes[a as usize].height = 1 + bh.max(fh);
                let ah = self.nodes[a as usize].height;
                let gh = if g != NULL_NODE { self.nodes[g as usize].height } else { 0 };
                self.nodes[c as usize].height = 1 + ah.max(gh);
            }
            return c;
        }

        // Rotate B up
        if balance_factor < -1 {
            let d = self.nodes[b as usize].left;
            let e = self.nodes[b as usize].right;

            self.nodes[b as usize].left = a;
            self.nodes[b as usize].parent = self.nodes[a as usize].parent;
            self.nodes[a as usize].parent = b;

            if self.nodes[b as usize].parent != NULL_NODE {
                let bp = self.nodes[b as usize].parent;
                if self.nodes[bp as usize].left == a {
                    self.nodes[bp as usize].left = b;
                } else {
                    self.nodes[bp as usize].right = b;
                }
            } else {
                self.root = b;
            }

            if d != NULL_NODE && e != NULL_NODE
                && self.nodes[d as usize].height > self.nodes[e as usize].height
            {
                self.nodes[b as usize].right = d;
                self.nodes[a as usize].left = e;
                self.nodes[e as usize].parent = a;
                self.nodes[a as usize].aabb = self.nodes[c as usize].aabb
                    .merged(&self.nodes[e as usize].aabb);
                self.nodes[b as usize].aabb = self.nodes[a as usize].aabb
                    .merged(&self.nodes[d as usize].aabb);
                self.nodes[a as usize].height = 1 + self.nodes[c as usize].height
                    .max(self.nodes[e as usize].height);
                self.nodes[b as usize].height = 1 + self.nodes[a as usize].height
                    .max(self.nodes[d as usize].height);
            } else {
                self.nodes[b as usize].right = e;
                self.nodes[a as usize].left = d;
                if d != NULL_NODE { self.nodes[d as usize].parent = a; }
                let c_aabb = self.nodes[c as usize].aabb;
                let d_aabb = if d != NULL_NODE { self.nodes[d as usize].aabb } else { c_aabb };
                self.nodes[a as usize].aabb = c_aabb.merged(&d_aabb);
                let e_aabb = if e != NULL_NODE { self.nodes[e as usize].aabb } else { self.nodes[a as usize].aabb };
                self.nodes[b as usize].aabb = self.nodes[a as usize].aabb.merged(&e_aabb);
                let ch = self.nodes[c as usize].height;
                let dh = if d != NULL_NODE { self.nodes[d as usize].height } else { 0 };
                self.nodes[a as usize].height = 1 + ch.max(dh);
                let ah = self.nodes[a as usize].height;
                let eh = if e != NULL_NODE { self.nodes[e as usize].height } else { 0 };
                self.nodes[b as usize].height = 1 + ah.max(eh);
            }
            return b;
        }

        a
    }
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_aabb(cx: f64, cy: f64, cz: f64, half: f64) -> Aabb3 {
        Aabb3::new(
            Vec3::new(cx - half, cy - half, cz - half),
            Vec3::new(cx + half, cy + half, cz + half),
        )
    }

    #[test]
    fn test_aabb_surface_area() {
        let a = Aabb3::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        // SA = 2*(2*3 + 3*4 + 2*4) = 2*(6+12+8) = 52
        assert!(approx(a.surface_area(), 52.0));
    }

    #[test]
    fn test_aabb_volume() {
        let a = Aabb3::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        assert!(approx(a.volume(), 24.0));
    }

    #[test]
    fn test_aabb_overlap() {
        let a = make_aabb(0.0, 0.0, 0.0, 1.0);
        let b = make_aabb(1.5, 0.0, 0.0, 1.0);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_aabb_no_overlap() {
        let a = make_aabb(0.0, 0.0, 0.0, 1.0);
        let b = make_aabb(5.0, 0.0, 0.0, 1.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_aabb_contains() {
        let outer = Aabb3::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let inner = make_aabb(0.0, 0.0, 0.0, 1.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_aabb_expanded() {
        let a = make_aabb(0.0, 0.0, 0.0, 1.0);
        let fat = a.expanded(0.5);
        assert!(approx(fat.min.x, -1.5));
        assert!(approx(fat.max.x, 1.5));
    }

    #[test]
    fn test_aabb_merged() {
        let a = make_aabb(0.0, 0.0, 0.0, 1.0);
        let b = make_aabb(3.0, 0.0, 0.0, 1.0);
        let m = a.merged(&b);
        assert!(approx(m.min.x, -1.0));
        assert!(approx(m.max.x, 4.0));
    }

    #[test]
    fn test_aabb_ray_hit() {
        let a = make_aabb(5.0, 0.0, 0.0, 1.0);
        let origin = Vec3::ZERO;
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let inv = Vec3::new(1.0, f64::MAX, f64::MAX);
        let t = a.ray_intersect(&origin, &inv);
        assert!(t.is_some());
        assert!(t.unwrap() > 0.0);
    }

    #[test]
    fn test_aabb_ray_miss() {
        let a = make_aabb(5.0, 5.0, 0.0, 1.0);
        let origin = Vec3::ZERO;
        let inv = Vec3::new(1.0, f64::MAX, f64::MAX);
        let t = a.ray_intersect(&origin, &inv);
        assert!(t.is_none());
    }

    #[test]
    fn test_tree_insert_single() {
        let mut tree = DynamicAabbTree::new(0.1);
        let pid = tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        assert_eq!(tree.proxy_count(), 1);
        assert!(tree.get_fat_aabb(pid).is_some());
    }

    #[test]
    fn test_tree_insert_multiple() {
        let mut tree = DynamicAabbTree::new(0.1);
        for i in 0..10 {
            tree.insert(make_aabb(i as f64 * 3.0, 0.0, 0.0, 1.0));
        }
        assert_eq!(tree.proxy_count(), 10);
        assert!(tree.height() > 0);
    }

    #[test]
    fn test_tree_remove() {
        let mut tree = DynamicAabbTree::new(0.1);
        let p1 = tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        let _p2 = tree.insert(make_aabb(5.0, 0.0, 0.0, 1.0));
        tree.remove(p1);
        assert_eq!(tree.proxy_count(), 1);
    }

    #[test]
    fn test_tree_update_no_change() {
        let mut tree = DynamicAabbTree::new(0.5);
        let pid = tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        // Small movement within fat AABB margin
        let changed = tree.update(pid, make_aabb(0.1, 0.0, 0.0, 1.0));
        assert!(!changed);
    }

    #[test]
    fn test_tree_update_requires_refit() {
        let mut tree = DynamicAabbTree::new(0.1);
        let pid = tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        let changed = tree.update(pid, make_aabb(5.0, 0.0, 0.0, 1.0));
        assert!(changed);
    }

    #[test]
    fn test_query_pairs_overlapping() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(1.5, 0.0, 0.0, 1.0));
        let pairs = tree.query_pairs();
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_query_pairs_no_overlap() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(10.0, 0.0, 0.0, 1.0));
        let pairs = tree.query_pairs();
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_query_aabb() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(5.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(10.0, 0.0, 0.0, 1.0));

        let q = make_aabb(0.0, 0.0, 0.0, 2.0);
        let hits = tree.query_aabb(&q);
        assert!(hits.len() >= 1);
    }

    #[test]
    fn test_ray_cast() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(5.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(10.0, 0.0, 0.0, 1.0));

        let hits = tree.ray_cast(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(hits.len(), 2);
        assert!(hits[0].1 < hits[1].1); // sorted by t
    }

    #[test]
    fn test_ray_cast_miss() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(5.0, 5.0, 0.0, 1.0));
        let hits = tree.ray_cast(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn test_frustum_query() {
        let mut tree = DynamicAabbTree::new(0.0);
        tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        tree.insert(make_aabb(100.0, 100.0, 100.0, 1.0));

        // Frustum that contains origin area
        let frustum = Frustum::new([
            ( 1.0, 0.0, 0.0, 5.0),  // x >= -5
            (-1.0, 0.0, 0.0, 5.0),  // x <= 5
            ( 0.0, 1.0, 0.0, 5.0),
            ( 0.0,-1.0, 0.0, 5.0),
            ( 0.0, 0.0, 1.0, 5.0),
            ( 0.0, 0.0,-1.0, 5.0),
        ]);
        let hits = tree.query_frustum(&frustum);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_frustum_intersects_aabb() {
        let frustum = Frustum::new([
            ( 1.0, 0.0, 0.0, 10.0),
            (-1.0, 0.0, 0.0, 10.0),
            ( 0.0, 1.0, 0.0, 10.0),
            ( 0.0,-1.0, 0.0, 10.0),
            ( 0.0, 0.0, 1.0, 10.0),
            ( 0.0, 0.0,-1.0, 10.0),
        ]);
        let inside = make_aabb(0.0, 0.0, 0.0, 1.0);
        let outside = make_aabb(50.0, 50.0, 50.0, 1.0);
        assert!(frustum.intersects_aabb(&inside));
        assert!(!frustum.intersects_aabb(&outside));
    }

    #[test]
    fn test_tree_height_grows() {
        let mut tree = DynamicAabbTree::new(0.0);
        assert_eq!(tree.height(), 0);
        tree.insert(make_aabb(0.0, 0.0, 0.0, 1.0));
        assert_eq!(tree.height(), 1);
        tree.insert(make_aabb(5.0, 0.0, 0.0, 1.0));
        assert!(tree.height() >= 2);
    }

    #[test]
    fn test_fat_aabb_larger_than_tight() {
        let mut tree = DynamicAabbTree::new(0.5);
        let tight = make_aabb(0.0, 0.0, 0.0, 1.0);
        let pid = tree.insert(tight);
        let fat = tree.get_fat_aabb(pid).unwrap();
        assert!(fat.min.x < tight.min.x);
        assert!(fat.max.x > tight.max.x);
    }
}
