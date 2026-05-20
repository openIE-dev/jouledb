//! View frustum culling — extract 6 frustum planes from view-projection matrix.
//! Test AABB, sphere, and point against frustum (inside/outside/intersecting).
//! Hierarchical frustum culling with BVH. Culling statistics. Masking
//! optimization (skip already-known-inside planes).

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
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn min_comp(self, o: Self) -> Self { Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) } }
    pub fn max_comp(self, o: Self) -> Self { Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) } }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
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

    /// Build a perspective projection matrix (right-handed, depth [0, 1]).
    pub fn perspective(fov_y_rad: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = 1.0 / (near - far);
        Self { m: [
            [f / aspect, 0.0,  0.0,           0.0],
            [0.0,        f,    0.0,           0.0],
            [0.0,        0.0,  far * nf,      near * far * nf],
            [0.0,        0.0, -1.0,           0.0],
        ]}
    }

    /// Build a look-at view matrix.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = {
            let d = target.sub(eye);
            let l = d.length();
            if l < 1e-12 { Vec3::new(0.0, 0.0, -1.0) } else { d.scale(1.0 / l) }
        };
        let r = {
            let c = Vec3::new(
                f.y * up.z - f.z * up.y,
                f.z * up.x - f.x * up.z,
                f.x * up.y - f.y * up.x,
            );
            let l = c.length();
            if l < 1e-12 { Vec3::new(1.0, 0.0, 0.0) } else { c.scale(1.0 / l) }
        };
        let u = Vec3::new(
            r.y * f.z - r.z * f.y,
            r.z * f.x - r.x * f.z,
            r.x * f.y - r.y * f.x,
        );
        Self { m: [
            [r.x,  r.y,  r.z,  -r.dot(eye)],
            [u.x,  u.y,  u.z,  -u.dot(eye)],
            [-f.x, -f.y, -f.z, f.dot(eye)],
            [0.0,  0.0,  0.0,  1.0],
        ]}
    }
}

// ── AABB ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }
    pub fn center(&self) -> Vec3 { self.min.add(self.max).scale(0.5) }
    pub fn half_extents(&self) -> Vec3 { self.max.sub(self.min).scale(0.5) }
}

// ── BoundingSphere ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f64,
}

// ── Plane ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f64,
}

impl Plane {
    pub fn new(normal: Vec3, distance: f64) -> Self { Self { normal, distance } }

    pub fn normalize(&mut self) {
        let len = self.normal.length();
        if len > 1e-12 {
            self.normal = self.normal.scale(1.0 / len);
            self.distance /= len;
        }
    }

    pub fn distance_to_point(&self, p: Vec3) -> f64 {
        self.normal.dot(p) + self.distance
    }
}

// ── CullResult ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullResult {
    Inside,
    Outside,
    Intersecting,
}

// ── CullStats ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CullStats {
    pub tested: usize,
    pub visible: usize,
    pub culled: usize,
    pub intersecting: usize,
}

impl CullStats {
    pub fn new() -> Self { Self { tested: 0, visible: 0, culled: 0, intersecting: 0 } }

    pub fn record(&mut self, result: CullResult) {
        self.tested += 1;
        match result {
            CullResult::Inside => self.visible += 1,
            CullResult::Outside => self.culled += 1,
            CullResult::Intersecting => { self.visible += 1; self.intersecting += 1; }
        }
    }

    pub fn cull_ratio(&self) -> f64 {
        if self.tested == 0 { 0.0 } else { self.culled as f64 / self.tested as f64 }
    }
}

// ── Frustum ──────────────────────────────────────────────────

/// The six planes of a view frustum: left, right, bottom, top, near, far.
#[derive(Debug, Clone, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

/// Named indices for frustum planes.
pub const PLANE_LEFT: usize = 0;
pub const PLANE_RIGHT: usize = 1;
pub const PLANE_BOTTOM: usize = 2;
pub const PLANE_TOP: usize = 3;
pub const PLANE_NEAR: usize = 4;
pub const PLANE_FAR: usize = 5;

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix (row-major).
    /// Griggs-Hartmann method.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let row = |r: usize| -> [f64; 4] {
            [vp.m[r][0], vp.m[r][1], vp.m[r][2], vp.m[r][3]]
        };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let mut planes = [Plane::new(Vec3::zero(), 0.0); 6];

        // Left:   row3 + row0
        planes[PLANE_LEFT] = Plane::new(
            Vec3::new(r3[0]+r0[0], r3[1]+r0[1], r3[2]+r0[2]),
            r3[3]+r0[3],
        );
        // Right:  row3 - row0
        planes[PLANE_RIGHT] = Plane::new(
            Vec3::new(r3[0]-r0[0], r3[1]-r0[1], r3[2]-r0[2]),
            r3[3]-r0[3],
        );
        // Bottom: row3 + row1
        planes[PLANE_BOTTOM] = Plane::new(
            Vec3::new(r3[0]+r1[0], r3[1]+r1[1], r3[2]+r1[2]),
            r3[3]+r1[3],
        );
        // Top:    row3 - row1
        planes[PLANE_TOP] = Plane::new(
            Vec3::new(r3[0]-r1[0], r3[1]-r1[1], r3[2]-r1[2]),
            r3[3]-r1[3],
        );
        // Near:   row3 + row2
        planes[PLANE_NEAR] = Plane::new(
            Vec3::new(r3[0]+r2[0], r3[1]+r2[1], r3[2]+r2[2]),
            r3[3]+r2[3],
        );
        // Far:    row3 - row2
        planes[PLANE_FAR] = Plane::new(
            Vec3::new(r3[0]-r2[0], r3[1]-r2[1], r3[2]-r2[2]),
            r3[3]-r2[3],
        );

        for p in &mut planes { p.normalize(); }
        Self { planes }
    }

    /// Test a point against the frustum.
    pub fn test_point(&self, p: Vec3) -> CullResult {
        for plane in &self.planes {
            if plane.distance_to_point(p) < 0.0 {
                return CullResult::Outside;
            }
        }
        CullResult::Inside
    }

    /// Test a sphere against the frustum.
    pub fn test_sphere(&self, sphere: &BoundingSphere) -> CullResult {
        let mut all_inside = true;
        for plane in &self.planes {
            let dist = plane.distance_to_point(sphere.center);
            if dist < -sphere.radius {
                return CullResult::Outside;
            }
            if dist < sphere.radius {
                all_inside = false;
            }
        }
        if all_inside { CullResult::Inside } else { CullResult::Intersecting }
    }

    /// Test an AABB against the frustum.
    pub fn test_aabb(&self, aabb: &Aabb) -> CullResult {
        let mut all_inside = true;
        for plane in &self.planes {
            // Find p-vertex (most positive along normal) and n-vertex
            let p_vertex = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            let n_vertex = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.min.x } else { aabb.max.x },
                if plane.normal.y >= 0.0 { aabb.min.y } else { aabb.max.y },
                if plane.normal.z >= 0.0 { aabb.min.z } else { aabb.max.z },
            );
            if plane.distance_to_point(p_vertex) < 0.0 {
                return CullResult::Outside;
            }
            if plane.distance_to_point(n_vertex) < 0.0 {
                all_inside = false;
            }
        }
        if all_inside { CullResult::Inside } else { CullResult::Intersecting }
    }

    /// Test AABB with a plane mask to skip already-known-inside planes.
    /// Mask bits: if bit i is set, plane i needs testing.
    /// Returns (result, updated_mask) — updated_mask clears bits for
    /// planes that the AABB is fully inside of.
    pub fn test_aabb_masked(&self, aabb: &Aabb, in_mask: u8) -> (CullResult, u8) {
        let mut out_mask = in_mask;
        let mut all_inside = true;
        for i in 0..6 {
            if in_mask & (1 << i) == 0 { continue; }
            let plane = &self.planes[i];
            let p_vertex = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            let n_vertex = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.min.x } else { aabb.max.x },
                if plane.normal.y >= 0.0 { aabb.min.y } else { aabb.max.y },
                if plane.normal.z >= 0.0 { aabb.min.z } else { aabb.max.z },
            );
            if plane.distance_to_point(p_vertex) < 0.0 {
                return (CullResult::Outside, in_mask);
            }
            if plane.distance_to_point(n_vertex) >= 0.0 {
                // Fully inside this plane — clear the bit
                out_mask &= !(1 << i);
            } else {
                all_inside = false;
            }
        }
        let result = if all_inside { CullResult::Inside } else { CullResult::Intersecting };
        (result, out_mask)
    }
}

// ── BVH Node for hierarchical culling ────────────────────────

#[derive(Debug, Clone)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub children: BvhChildren,
}

#[derive(Debug, Clone)]
pub enum BvhChildren {
    Leaf(Vec<usize>),           // indices into objects array
    Interior(Vec<BvhNode>),
}

impl BvhNode {
    pub fn leaf(aabb: Aabb, objects: Vec<usize>) -> Self {
        Self { aabb, children: BvhChildren::Leaf(objects) }
    }

    pub fn interior(aabb: Aabb, children: Vec<BvhNode>) -> Self {
        Self { aabb, children: BvhChildren::Interior(children) }
    }
}

/// Hierarchical frustum culling on a BVH tree.
pub fn cull_bvh(frustum: &Frustum, node: &BvhNode, stats: &mut CullStats) -> Vec<usize> {
    cull_bvh_masked(frustum, node, 0x3F, stats)
}

fn cull_bvh_masked(frustum: &Frustum, node: &BvhNode, mask: u8, stats: &mut CullStats) -> Vec<usize> {
    let (result, child_mask) = frustum.test_aabb_masked(&node.aabb, mask);
    stats.record(result);

    match result {
        CullResult::Outside => Vec::new(),
        CullResult::Inside => {
            // Everything inside — collect all leaves
            collect_all_leaves(node)
        }
        CullResult::Intersecting => {
            match &node.children {
                BvhChildren::Leaf(objects) => objects.clone(),
                BvhChildren::Interior(children) => {
                    let mut visible = Vec::new();
                    for child in children {
                        visible.extend(cull_bvh_masked(frustum, child, child_mask, stats));
                    }
                    visible
                }
            }
        }
    }
}

fn collect_all_leaves(node: &BvhNode) -> Vec<usize> {
    match &node.children {
        BvhChildren::Leaf(objects) => objects.clone(),
        BvhChildren::Interior(children) => {
            let mut result = Vec::new();
            for child in children {
                result.extend(collect_all_leaves(child));
            }
            result
        }
    }
}

/// Build a simple BVH from a list of AABBs (median split on longest axis).
pub fn build_bvh(aabbs: &[Aabb], max_leaf_size: usize) -> Option<BvhNode> {
    if aabbs.is_empty() { return None; }
    let indices: Vec<usize> = (0..aabbs.len()).collect();
    Some(build_bvh_recursive(aabbs, &indices, max_leaf_size))
}

fn build_bvh_recursive(aabbs: &[Aabb], indices: &[usize], max_leaf_size: usize) -> BvhNode {
    let mut enclosing = Aabb::new(
        Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &i in indices {
        enclosing.min = enclosing.min.min_comp(aabbs[i].min);
        enclosing.max = enclosing.max.max_comp(aabbs[i].max);
    }

    if indices.len() <= max_leaf_size {
        return BvhNode::leaf(enclosing, indices.to_vec());
    }

    // Split on longest axis
    let extent = enclosing.max.sub(enclosing.min);
    let axis = if extent.x >= extent.y && extent.x >= extent.z { 0 }
               else if extent.y >= extent.z { 1 }
               else { 2 };

    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        let ca = aabbs[a].center();
        let cb = aabbs[b].center();
        let va = match axis { 0 => ca.x, 1 => ca.y, _ => ca.z };
        let vb = match axis { 0 => cb.x, 1 => cb.y, _ => cb.z };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = sorted.len() / 2;
    let left = build_bvh_recursive(aabbs, &sorted[..mid], max_leaf_size);
    let right = build_bvh_recursive(aabbs, &sorted[mid..], max_leaf_size);
    BvhNode::interior(enclosing, vec![left, right])
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    fn make_test_frustum() -> Frustum {
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::zero(),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let proj = Mat4::perspective(FRAC_PI_4 * 2.0, 1.0, 0.1, 100.0);
        Frustum::from_view_projection(&proj.mul(view))
    }

    #[test]
    fn test_frustum_planes_count() {
        let f = make_test_frustum();
        assert_eq!(f.planes.len(), 6);
    }

    #[test]
    fn test_frustum_planes_normalized() {
        let f = make_test_frustum();
        for p in &f.planes {
            let len = p.normal.length();
            assert!((len - 1.0).abs() < 1e-6, "plane normal should be unit, got {}", len);
        }
    }

    #[test]
    fn test_point_inside_frustum() {
        let f = make_test_frustum();
        let result = f.test_point(Vec3::zero());
        assert_ne!(result, CullResult::Outside);
    }

    #[test]
    fn test_point_outside_frustum() {
        let f = make_test_frustum();
        let result = f.test_point(Vec3::new(0.0, 0.0, 200.0));
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_point_behind_camera() {
        let f = make_test_frustum();
        let result = f.test_point(Vec3::new(0.0, 0.0, 10.0));
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_sphere_inside() {
        let f = make_test_frustum();
        let s = BoundingSphere { center: Vec3::zero(), radius: 0.1 };
        let result = f.test_sphere(&s);
        assert_ne!(result, CullResult::Outside);
    }

    #[test]
    fn test_sphere_outside() {
        let f = make_test_frustum();
        let s = BoundingSphere { center: Vec3::new(0.0, 0.0, 200.0), radius: 1.0 };
        assert_eq!(f.test_sphere(&s), CullResult::Outside);
    }

    #[test]
    fn test_aabb_inside() {
        let f = make_test_frustum();
        let aabb = Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1));
        let result = f.test_aabb(&aabb);
        assert_ne!(result, CullResult::Outside);
    }

    #[test]
    fn test_aabb_outside() {
        let f = make_test_frustum();
        let aabb = Aabb::new(Vec3::new(100.0, 100.0, 100.0), Vec3::new(101.0, 101.0, 101.0));
        assert_eq!(f.test_aabb(&aabb), CullResult::Outside);
    }

    #[test]
    fn test_aabb_intersecting() {
        let f = make_test_frustum();
        // Large AABB that partially overlaps the frustum
        let aabb = Aabb::new(Vec3::new(-50.0, -50.0, -110.0), Vec3::new(50.0, 50.0, 10.0));
        let result = f.test_aabb(&aabb);
        assert_ne!(result, CullResult::Outside);
    }

    #[test]
    fn test_cull_stats() {
        let f = make_test_frustum();
        let mut stats = CullStats::new();
        let r1 = f.test_aabb(&Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1)));
        stats.record(r1);
        let r2 = f.test_aabb(&Aabb::new(Vec3::new(200.0, 200.0, 200.0), Vec3::new(201.0, 201.0, 201.0)));
        stats.record(r2);
        assert_eq!(stats.tested, 2);
        assert_eq!(stats.culled, 1);
    }

    #[test]
    fn test_cull_ratio() {
        let mut stats = CullStats::new();
        stats.record(CullResult::Inside);
        stats.record(CullResult::Outside);
        stats.record(CullResult::Outside);
        assert!((stats.cull_ratio() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_masked_cull_all_planes() {
        let f = make_test_frustum();
        let aabb = Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1));
        let (result, _mask) = f.test_aabb_masked(&aabb, 0x3F);
        assert_ne!(result, CullResult::Outside);
    }

    #[test]
    fn test_masked_cull_outside() {
        let f = make_test_frustum();
        let aabb = Aabb::new(Vec3::new(200.0, 200.0, 200.0), Vec3::new(201.0, 201.0, 201.0));
        let (result, _) = f.test_aabb_masked(&aabb, 0x3F);
        assert_eq!(result, CullResult::Outside);
    }

    #[test]
    fn test_build_bvh() {
        let aabbs = vec![
            Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(4.0, 0.0, 0.0), Vec3::new(5.0, 1.0, 1.0)),
        ];
        let bvh = build_bvh(&aabbs, 1).unwrap();
        match &bvh.children {
            BvhChildren::Interior(_) => {} // expected
            BvhChildren::Leaf(_) => panic!("expected interior node"),
        }
    }

    #[test]
    fn test_bvh_leaf() {
        let aabbs = vec![
            Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        ];
        let bvh = build_bvh(&aabbs, 2).unwrap();
        match &bvh.children {
            BvhChildren::Leaf(objs) => assert_eq!(objs.len(), 1),
            _ => panic!("expected leaf"),
        }
    }

    #[test]
    fn test_hierarchical_cull() {
        let f = make_test_frustum();
        let aabbs = vec![
            Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1)),     // inside
            Aabb::new(Vec3::new(200.0, 200.0, 200.0), Vec3::new(201.0, 201.0, 201.0)), // outside
        ];
        let bvh = build_bvh(&aabbs, 1).unwrap();
        let mut stats = CullStats::new();
        let visible = cull_bvh(&f, &bvh, &mut stats);
        assert!(visible.contains(&0), "object 0 should be visible");
        assert!(!visible.contains(&1), "object 1 should be culled");
    }

    #[test]
    fn test_empty_bvh() {
        let result = build_bvh(&[], 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_collect_all_leaves() {
        let node = BvhNode::interior(
            Aabb::new(Vec3::zero(), Vec3::new(1.0, 1.0, 1.0)),
            vec![
                BvhNode::leaf(Aabb::new(Vec3::zero(), Vec3::new(0.5, 0.5, 0.5)), vec![0, 1]),
                BvhNode::leaf(Aabb::new(Vec3::zero(), Vec3::new(1.0, 1.0, 1.0)), vec![2]),
            ],
        );
        let all = collect_all_leaves(&node);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_plane_distance() {
        let p = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        assert!((p.distance_to_point(Vec3::new(0.0, 5.0, 0.0)) - 5.0).abs() < 1e-9);
        assert!((p.distance_to_point(Vec3::new(0.0, -3.0, 0.0)) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_stats_initial() {
        let stats = CullStats::new();
        assert_eq!(stats.tested, 0);
        assert_eq!(stats.visible, 0);
        assert_eq!(stats.culled, 0);
    }

    #[test]
    fn test_sphere_large_radius_intersecting() {
        let f = make_test_frustum();
        let s = BoundingSphere { center: Vec3::new(0.0, 100.0, 0.0), radius: 200.0 };
        let result = f.test_sphere(&s);
        assert_ne!(result, CullResult::Outside);
    }
}
