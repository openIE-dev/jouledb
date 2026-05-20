//! Octree for 3D space — insert, remove, range query (AABB), ray intersection,
//! nearest neighbor, frustum culling, level-of-detail, bounding sphere queries.
//!
//! Replaces JavaScript 3D spatial libraries with a pure-Rust octree for WebGL
//! and WASM-based 3D applications.

// ── 3D primitives ───────────────────────────────────────────────

/// A 3D point with an identifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(id: u64, x: f64, y: f64, z: f64) -> Self { Self { id, x, y, z } }
}

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl AABB {
    pub fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
        Self { min_x, min_y, min_z, max_x, max_y, max_z }
    }

    pub fn cube(cx: f64, cy: f64, cz: f64, half: f64) -> Self {
        Self::new(cx - half, cy - half, cz - half, cx + half, cy + half, cz + half)
    }

    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.min_x && x < self.max_x
            && y >= self.min_y && y < self.max_y
            && z >= self.min_z && z < self.max_z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        !(self.max_x <= other.min_x || other.max_x <= self.min_x
            || self.max_y <= other.min_y || other.max_y <= self.min_y
            || self.max_z <= other.min_z || other.max_z <= self.min_z)
    }

    pub fn center(&self) -> (f64, f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )
    }

    pub fn half_size(&self) -> (f64, f64, f64) {
        (
            (self.max_x - self.min_x) / 2.0,
            (self.max_y - self.min_y) / 2.0,
            (self.max_z - self.min_z) / 2.0,
        )
    }

    fn octants(&self) -> [AABB; 8] {
        let (cx, cy, cz) = self.center();
        [
            AABB::new(self.min_x, self.min_y, self.min_z, cx, cy, cz),
            AABB::new(cx, self.min_y, self.min_z, self.max_x, cy, cz),
            AABB::new(self.min_x, cy, self.min_z, cx, self.max_y, cz),
            AABB::new(cx, cy, self.min_z, self.max_x, self.max_y, cz),
            AABB::new(self.min_x, self.min_y, cz, cx, cy, self.max_z),
            AABB::new(cx, self.min_y, cz, self.max_x, cy, self.max_z),
            AABB::new(self.min_x, cy, cz, cx, self.max_y, self.max_z),
            AABB::new(cx, cy, cz, self.max_x, self.max_y, self.max_z),
        ]
    }

    /// Bounding sphere radius from center.
    fn bounding_radius(&self) -> f64 {
        let (hx, hy, hz) = self.half_size();
        (hx * hx + hy * hy + hz * hz).sqrt()
    }
}

/// A ray in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin_x: f64,
    pub origin_y: f64,
    pub origin_z: f64,
    pub dir_x: f64,
    pub dir_y: f64,
    pub dir_z: f64,
}

impl Ray {
    pub fn new(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) -> Self {
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        Self {
            origin_x: ox, origin_y: oy, origin_z: oz,
            dir_x: dx / len, dir_y: dy / len, dir_z: dz / len,
        }
    }

    /// Slab-based ray-AABB intersection. Returns (t_min, t_max) or None.
    fn intersect_aabb(&self, aabb: &AABB) -> Option<(f64, f64)> {
        let inv_x = if self.dir_x.abs() > 1e-12 { 1.0 / self.dir_x } else { f64::INFINITY.copysign(self.dir_x) };
        let inv_y = if self.dir_y.abs() > 1e-12 { 1.0 / self.dir_y } else { f64::INFINITY.copysign(self.dir_y) };
        let inv_z = if self.dir_z.abs() > 1e-12 { 1.0 / self.dir_z } else { f64::INFINITY.copysign(self.dir_z) };

        let t1 = (aabb.min_x - self.origin_x) * inv_x;
        let t2 = (aabb.max_x - self.origin_x) * inv_x;
        let t3 = (aabb.min_y - self.origin_y) * inv_y;
        let t4 = (aabb.max_y - self.origin_y) * inv_y;
        let t5 = (aabb.min_z - self.origin_z) * inv_z;
        let t6 = (aabb.max_z - self.origin_z) * inv_z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax { None } else { Some((tmin.max(0.0), tmax)) }
    }
}

/// Frustum defined by 6 planes (Ax + By + Cz + D >= 0 for inside).
#[derive(Debug, Clone)]
pub struct Frustum {
    /// Six planes as (A, B, C, D).
    pub planes: [(f64, f64, f64, f64); 6],
}

impl Frustum {
    /// Check if an AABB is (potentially) inside the frustum.
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let (cx, cy, cz) = aabb.center();
        let (hx, hy, hz) = aabb.half_size();
        for &(a, b, c, d) in &self.planes {
            let extent = hx * a.abs() + hy * b.abs() + hz * c.abs();
            let dist = a * cx + b * cy + c * cz + d;
            if dist + extent < 0.0 {
                return false;
            }
        }
        true
    }
}

// ── Octree ──────────────────────────────────────────────────────

/// An octree node for 3D spatial indexing.
#[derive(Debug, Clone)]
pub struct Octree {
    boundary: AABB,
    capacity: usize,
    points: Vec<Point3>,
    children: Option<Box<[Octree; 8]>>,
    total_count: usize,
    max_depth: usize,
    depth: usize,
}

impl Octree {
    pub fn new(boundary: AABB, capacity: usize) -> Self {
        Self {
            boundary,
            capacity: capacity.max(1),
            points: Vec::new(),
            children: None,
            total_count: 0,
            max_depth: 10,
            depth: 0,
        }
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn insert(&mut self, point: Point3) -> bool {
        if !self.boundary.contains(point.x, point.y, point.z) {
            return false;
        }
        self.total_count += 1;

        if self.children.is_none() && (self.points.len() < self.capacity || self.depth >= self.max_depth) {
            self.points.push(point);
            return true;
        }

        if self.children.is_none() {
            self.subdivide();
        }

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if child.insert(point) {
                    return true;
                }
            }
        }
        // Boundary edge case — store here.
        self.points.push(point);
        true
    }

    fn subdivide(&mut self) {
        let octs = self.boundary.octants();
        let cap = self.capacity;
        let d = self.depth + 1;
        let md = self.max_depth;
        let mut children = Box::new([
            Octree { boundary: octs[0], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[1], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[2], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[3], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[4], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[5], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[6], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
            Octree { boundary: octs[7], capacity: cap, points: Vec::new(), children: None, total_count: 0, max_depth: md, depth: d },
        ]);
        let old_points: Vec<Point3> = self.points.drain(..).collect();
        for p in old_points {
            let mut inserted = false;
            for child in children.iter_mut() {
                if child.insert(p) {
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                self.points.push(p);
            }
        }
        self.children = Some(children);
    }

    /// Remove a point by id.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(idx) = self.points.iter().position(|p| p.id == id) {
            self.points.swap_remove(idx);
            self.total_count -= 1;
            return true;
        }
        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if child.remove(id) {
                    self.total_count -= 1;
                    return true;
                }
            }
        }
        false
    }

    /// Range query: all points within an AABB.
    pub fn query_range(&self, range: &AABB) -> Vec<Point3> {
        let mut result = Vec::new();
        self.query_range_inner(range, &mut result);
        result
    }

    fn query_range_inner(&self, range: &AABB, result: &mut Vec<Point3>) {
        if !self.boundary.intersects(range) { return; }
        for p in &self.points {
            if range.contains(p.x, p.y, p.z) {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query_range_inner(range, result);
            }
        }
    }

    /// Ray intersection: find all points within `radius` of the ray.
    pub fn ray_query(&self, ray: &Ray, radius: f64) -> Vec<Point3> {
        let mut result = Vec::new();
        self.ray_query_inner(ray, radius, &mut result);
        result
    }

    fn ray_query_inner(&self, ray: &Ray, radius: f64, result: &mut Vec<Point3>) {
        // Expand AABB by radius for ray test.
        let expanded = AABB::new(
            self.boundary.min_x - radius, self.boundary.min_y - radius, self.boundary.min_z - radius,
            self.boundary.max_x + radius, self.boundary.max_y + radius, self.boundary.max_z + radius,
        );
        if ray.intersect_aabb(&expanded).is_none() { return; }

        for p in &self.points {
            if point_ray_dist_sq(p, ray) <= radius * radius {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.ray_query_inner(ray, radius, result);
            }
        }
    }

    /// Nearest neighbor in 3D.
    pub fn nearest(&self, qx: f64, qy: f64, qz: f64) -> Option<Point3> {
        let mut best: Option<(Point3, f64)> = None;
        self.nearest_inner(qx, qy, qz, &mut best);
        best.map(|(p, _)| p)
    }

    fn nearest_inner(&self, qx: f64, qy: f64, qz: f64, best: &mut Option<(Point3, f64)>) {
        if let Some((_, bd)) = best {
            let closest_x = qx.clamp(self.boundary.min_x, self.boundary.max_x);
            let closest_y = qy.clamp(self.boundary.min_y, self.boundary.max_y);
            let closest_z = qz.clamp(self.boundary.min_z, self.boundary.max_z);
            let d = dist3_sq(qx, qy, qz, closest_x, closest_y, closest_z);
            if d > *bd { return; }
        }
        for p in &self.points {
            let d = dist3_sq(qx, qy, qz, p.x, p.y, p.z);
            if best.is_none() || d < best.unwrap().1 {
                *best = Some((*p, d));
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.nearest_inner(qx, qy, qz, best);
            }
        }
    }

    /// Frustum culling: return all points visible in the frustum.
    pub fn query_frustum(&self, frustum: &Frustum) -> Vec<Point3> {
        let mut result = Vec::new();
        self.frustum_inner(frustum, &mut result);
        result
    }

    fn frustum_inner(&self, frustum: &Frustum, result: &mut Vec<Point3>) {
        if !frustum.intersects_aabb(&self.boundary) { return; }
        for p in &self.points {
            if point_in_frustum(p, frustum) {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.frustum_inner(frustum, result);
            }
        }
    }

    /// Bounding sphere query: all points within `radius` of center.
    pub fn query_sphere(&self, cx: f64, cy: f64, cz: f64, radius: f64) -> Vec<Point3> {
        let mut result = Vec::new();
        self.sphere_inner(cx, cy, cz, radius * radius, &mut result);
        result
    }

    fn sphere_inner(&self, cx: f64, cy: f64, cz: f64, r_sq: f64, result: &mut Vec<Point3>) {
        // Quick reject: if closest point on AABB is farther than radius.
        let closest_x = cx.clamp(self.boundary.min_x, self.boundary.max_x);
        let closest_y = cy.clamp(self.boundary.min_y, self.boundary.max_y);
        let closest_z = cz.clamp(self.boundary.min_z, self.boundary.max_z);
        if dist3_sq(cx, cy, cz, closest_x, closest_y, closest_z) > r_sq { return; }

        for p in &self.points {
            if dist3_sq(cx, cy, cz, p.x, p.y, p.z) <= r_sq {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.sphere_inner(cx, cy, cz, r_sq, result);
            }
        }
    }

    /// Level-of-detail: return points from nodes whose projected size exceeds
    /// `min_screen_size`. Nodes too far or too small are culled.
    pub fn query_lod(&self, eye_x: f64, eye_y: f64, eye_z: f64, min_screen_size: f64) -> Vec<Point3> {
        let mut result = Vec::new();
        self.lod_inner(eye_x, eye_y, eye_z, min_screen_size, &mut result);
        result
    }

    fn lod_inner(&self, ex: f64, ey: f64, ez: f64, min_size: f64, result: &mut Vec<Point3>) {
        let (cx, cy, cz) = self.boundary.center();
        let dist = dist3_sq(ex, ey, ez, cx, cy, cz).sqrt();
        if dist < 1e-6 {
            result.extend_from_slice(&self.points);
            if let Some(children) = &self.children {
                for child in children.iter() {
                    child.lod_inner(ex, ey, ez, min_size, result);
                }
            }
            return;
        }
        let projected_size = self.boundary.bounding_radius() / dist;
        if projected_size < min_size {
            // Too small — skip or just add representative point.
            if let Some(p) = self.points.first() {
                result.push(*p);
            }
            return;
        }
        result.extend_from_slice(&self.points);
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.lod_inner(ex, ey, ez, min_size, result);
            }
        }
    }

    pub fn len(&self) -> usize { self.total_count }
    pub fn is_empty(&self) -> bool { self.total_count == 0 }
}

fn dist3_sq(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz)
}

fn point_ray_dist_sq(p: &Point3, ray: &Ray) -> f64 {
    let dx = p.x - ray.origin_x;
    let dy = p.y - ray.origin_y;
    let dz = p.z - ray.origin_z;
    let t = dx * ray.dir_x + dy * ray.dir_y + dz * ray.dir_z;
    let t = t.max(0.0);
    let proj_x = ray.origin_x + t * ray.dir_x;
    let proj_y = ray.origin_y + t * ray.dir_y;
    let proj_z = ray.origin_z + t * ray.dir_z;
    dist3_sq(p.x, p.y, p.z, proj_x, proj_y, proj_z)
}

fn point_in_frustum(p: &Point3, frustum: &Frustum) -> bool {
    for &(a, b, c, d) in &frustum.planes {
        if a * p.x + b * p.y + c * p.z + d < 0.0 {
            return false;
        }
    }
    true
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_bounds() -> AABB {
        AABB::cube(0.0, 0.0, 0.0, 50.0)
    }

    #[test]
    fn insert_and_count() {
        let mut ot = Octree::new(cube_bounds(), 4);
        for i in 0..10 {
            let v = (i as f64) * 5.0 - 20.0;
            assert!(ot.insert(Point3::new(i, v, v, v)));
        }
        assert_eq!(ot.len(), 10);
    }

    #[test]
    fn out_of_bounds() {
        let mut ot = Octree::new(cube_bounds(), 4);
        assert!(!ot.insert(Point3::new(1, 100.0, 0.0, 0.0)));
    }

    #[test]
    fn remove_point() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 0.0, 0.0, 0.0));
        ot.insert(Point3::new(2, 10.0, 10.0, 10.0));
        assert!(ot.remove(1));
        assert_eq!(ot.len(), 1);
        assert!(!ot.remove(99));
    }

    #[test]
    fn range_query() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 5.0, 5.0, 5.0));
        ot.insert(Point3::new(2, -40.0, -40.0, -40.0));
        let result = ot.query_range(&AABB::cube(5.0, 5.0, 5.0, 2.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn nearest_neighbor() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 0.0, 0.0, 0.0));
        ot.insert(Point3::new(2, 30.0, 30.0, 30.0));
        let nearest = ot.nearest(1.0, 1.0, 1.0).unwrap();
        assert_eq!(nearest.id, 1);
    }

    #[test]
    fn ray_intersection() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 10.0, 0.0, 0.0));
        ot.insert(Point3::new(2, 0.0, 30.0, 0.0));
        let ray = Ray::new(-50.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let hits = ot.ray_query(&ray, 2.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn sphere_query() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 0.0, 0.0, 0.0));
        ot.insert(Point3::new(2, 40.0, 40.0, 40.0));
        let result = ot.query_sphere(0.0, 0.0, 0.0, 5.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn frustum_query() {
        let mut ot = Octree::new(cube_bounds(), 4);
        ot.insert(Point3::new(1, 5.0, 5.0, 5.0));
        ot.insert(Point3::new(2, -45.0, -45.0, -45.0));
        // Frustum that includes positive octant only.
        let frustum = Frustum {
            planes: [
                (1.0, 0.0, 0.0, 0.0),  // x >= 0
                (0.0, 1.0, 0.0, 0.0),  // y >= 0
                (0.0, 0.0, 1.0, 0.0),  // z >= 0
                (-1.0, 0.0, 0.0, 50.0), // x <= 50
                (0.0, -1.0, 0.0, 50.0), // y <= 50
                (0.0, 0.0, -1.0, 50.0), // z <= 50
            ],
        };
        let result = ot.query_frustum(&frustum);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn lod_query() {
        let mut ot = Octree::new(cube_bounds(), 4);
        for i in 0..5 {
            ot.insert(Point3::new(i, (i as f64) * 5.0, 0.0, 0.0));
        }
        // Close eye — everything visible.
        let result = ot.query_lod(0.0, 0.0, 0.0, 0.001);
        assert!(result.len() >= 5);
    }

    #[test]
    fn aabb_operations() {
        let a = AABB::cube(0.0, 0.0, 0.0, 10.0);
        assert!(a.contains(5.0, 5.0, 5.0));
        assert!(!a.contains(10.0, 0.0, 0.0)); // exclusive
        let b = AABB::cube(5.0, 5.0, 5.0, 10.0);
        assert!(a.intersects(&b));
        let c = AABB::cube(100.0, 100.0, 100.0, 5.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn empty_tree() {
        let ot = Octree::new(cube_bounds(), 4);
        assert!(ot.is_empty());
        assert!(ot.nearest(0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn subdivision_with_many_points() {
        let mut ot = Octree::new(cube_bounds(), 2);
        for i in 0..20 {
            let v = (i as f64) * 2.0 - 20.0;
            ot.insert(Point3::new(i, v, v, v));
        }
        assert_eq!(ot.len(), 20);
        let result = ot.query_range(&cube_bounds());
        assert_eq!(result.len(), 20);
    }
}
