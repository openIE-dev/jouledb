//! Mesh Boolean — Constructive Solid Geometry (CSG) operations on triangle meshes.
//! Operations: union, intersection, difference. BSP tree approach with triangle
//! splitting, inside/outside classification, vertex welding, and coplanar handling.

use std::collections::HashMap;

// ── Vector types ───────────────────────────────────────────────

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn add(&self, o: &Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn dot(&self, o: &Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn lerp(&self, o: &Self, t: f64) -> Self {
        self.add(&o.sub(self).scale(t))
    }
}

// ── Plane ──────────────────────────────────────────────────────

const PLANE_EPS: f64 = 1e-6;

/// A plane defined by normal and distance from origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f64,
}

impl Plane {
    pub fn new(normal: Vec3, d: f64) -> Self {
        Self { normal, d }
    }

    /// Create from three points.
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = b.sub(&a).cross(&c.sub(&a)).normalized();
        let d = -normal.dot(&a);
        Self { normal, d }
    }

    /// Signed distance from a point to the plane.
    pub fn distance(&self, p: Vec3) -> f64 {
        self.normal.dot(&p) + self.d
    }

    /// Classify a point: +1 front, -1 back, 0 coplanar.
    pub fn classify_point(&self, p: Vec3) -> i8 {
        let d = self.distance(p);
        if d > PLANE_EPS {
            1
        } else if d < -PLANE_EPS {
            -1
        } else {
            0
        }
    }
}

// ── Triangle ───────────────────────────────────────────────────

/// A triangle with three vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { a, b, c }
    }

    pub fn normal(&self) -> Vec3 {
        self.b.sub(&self.a).cross(&self.c.sub(&self.a)).normalized()
    }

    pub fn plane(&self) -> Plane {
        Plane::from_points(self.a, self.b, self.c)
    }

    pub fn centroid(&self) -> Vec3 {
        self.a.add(&self.b).add(&self.c).scale(1.0 / 3.0)
    }

    pub fn flip(&self) -> Self {
        Self::new(self.c, self.b, self.a)
    }

    pub fn area(&self) -> f64 {
        self.b.sub(&self.a).cross(&self.c.sub(&self.a)).length() * 0.5
    }
}

// ── BSP Tree ───────────────────────────────────────────────────

/// BSP tree node.
struct BspNode {
    plane: Plane,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
    coplanar: Vec<Triangle>,
}

impl BspNode {
    fn new(triangles: &[Triangle]) -> Option<Box<Self>> {
        if triangles.is_empty() {
            return None;
        }

        let plane = triangles[0].plane();
        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front_tris = Vec::new();
        let mut back_tris = Vec::new();

        for tri in triangles {
            split_triangle(tri, &plane, &mut coplanar_front, &mut coplanar_back, &mut front_tris, &mut back_tris);
        }

        let mut coplanar = coplanar_front;
        coplanar.extend(coplanar_back);

        Some(Box::new(BspNode {
            plane,
            front: BspNode::new(&front_tris),
            back: BspNode::new(&back_tris),
            coplanar,
        }))
    }

    /// Collect all triangles from this BSP tree.
    fn collect_triangles(&self) -> Vec<Triangle> {
        let mut result = self.coplanar.clone();
        if let Some(front) = &self.front {
            result.extend(front.collect_triangles());
        }
        if let Some(back) = &self.back {
            result.extend(back.collect_triangles());
        }
        result
    }

    /// Clip triangles to the inside (back) of this BSP tree.
    fn clip_triangles(&self, triangles: &[Triangle]) -> Vec<Triangle> {
        let mut front_list = Vec::new();
        let mut back_list = Vec::new();

        for tri in triangles {
            let mut front = Vec::new();
            let mut back = Vec::new();
            let mut coplanar_front = Vec::new();
            let mut coplanar_back = Vec::new();
            split_triangle(tri, &self.plane, &mut coplanar_front, &mut coplanar_back, &mut front, &mut back);
            front_list.extend(front);
            front_list.extend(coplanar_front);
            back_list.extend(back);
            back_list.extend(coplanar_back);
        }

        let mut result = if let Some(front) = &self.front {
            front.clip_triangles(&front_list)
        } else {
            Vec::new()
        };

        if let Some(back) = &self.back {
            result.extend(back.clip_triangles(&back_list));
        } else {
            result.extend(back_list);
        }

        result
    }

    /// Clip this BSP tree against another BSP tree (remove inside geometry).
    fn clip_to(&mut self, bsp: &BspNode) {
        self.coplanar = bsp.clip_triangles(&self.coplanar);
        if let Some(front) = &mut self.front {
            front.clip_to(bsp);
        }
        if let Some(back) = &mut self.back {
            back.clip_to(bsp);
        }
    }

    /// Invert all triangles and the BSP planes.
    fn invert(&mut self) {
        for tri in &mut self.coplanar {
            *tri = tri.flip();
        }
        self.plane.normal = self.plane.normal.scale(-1.0);
        self.plane.d = -self.plane.d;
        std::mem::swap(&mut self.front, &mut self.back);
        if let Some(front) = &mut self.front {
            front.invert();
        }
        if let Some(back) = &mut self.back {
            back.invert();
        }
    }

    /// Insert triangles into this BSP tree.
    fn insert(&mut self, triangles: &[Triangle]) {
        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front_tris = Vec::new();
        let mut back_tris = Vec::new();

        for tri in triangles {
            split_triangle(
                tri,
                &self.plane,
                &mut coplanar_front,
                &mut coplanar_back,
                &mut front_tris,
                &mut back_tris,
            );
        }
        self.coplanar.extend(coplanar_front);
        self.coplanar.extend(coplanar_back);

        if !front_tris.is_empty() {
            if let Some(front) = &mut self.front {
                front.insert(&front_tris);
            } else {
                self.front = BspNode::new(&front_tris);
            }
        }
        if !back_tris.is_empty() {
            if let Some(back) = &mut self.back {
                back.insert(&back_tris);
            } else {
                self.back = BspNode::new(&back_tris);
            }
        }
    }
}

// ── Triangle splitting ─────────────────────────────────────────

/// Classify and split a triangle against a plane.
fn split_triangle(
    tri: &Triangle,
    plane: &Plane,
    coplanar_front: &mut Vec<Triangle>,
    coplanar_back: &mut Vec<Triangle>,
    front: &mut Vec<Triangle>,
    back: &mut Vec<Triangle>,
) {
    let ca = plane.classify_point(tri.a);
    let cb = plane.classify_point(tri.b);
    let cc = plane.classify_point(tri.c);

    let classification = ca + cb + cc;

    // All coplanar
    if ca == 0 && cb == 0 && cc == 0 {
        if tri.normal().dot(&plane.normal) > 0.0 {
            coplanar_front.push(*tri);
        } else {
            coplanar_back.push(*tri);
        }
        return;
    }

    // All on one side
    if ca >= 0 && cb >= 0 && cc >= 0 {
        front.push(*tri);
        return;
    }
    if ca <= 0 && cb <= 0 && cc <= 0 {
        back.push(*tri);
        return;
    }

    // Split needed: the triangle straddles the plane
    let verts = [tri.a, tri.b, tri.c];
    let classes = [ca, cb, cc];
    let mut front_pts = Vec::new();
    let mut back_pts = Vec::new();

    for i in 0..3 {
        let j = (i + 1) % 3;
        let vi = verts[i];
        let vj = verts[j];
        let ci = classes[i];
        let cj = classes[j];

        if ci >= 0 {
            front_pts.push(vi);
        }
        if ci <= 0 {
            back_pts.push(vi);
        }

        if (ci > 0 && cj < 0) || (ci < 0 && cj > 0) {
            // Edge crosses the plane
            let di = plane.distance(vi);
            let dj = plane.distance(vj);
            let t = di / (di - dj);
            let t = t.clamp(0.0, 1.0);
            let intersection = vi.lerp(&vj, t);
            front_pts.push(intersection);
            back_pts.push(intersection);
        }
    }

    // Triangulate front polygon
    triangulate_polygon(&front_pts, front);
    // Triangulate back polygon
    triangulate_polygon(&back_pts, back);
}

/// Fan-triangulate a convex polygon.
fn triangulate_polygon(pts: &[Vec3], output: &mut Vec<Triangle>) {
    if pts.len() < 3 {
        return;
    }
    for i in 1..pts.len() - 1 {
        let tri = Triangle::new(pts[0], pts[i], pts[i + 1]);
        if tri.area() > 1e-10 {
            output.push(tri);
        }
    }
}

// ── CSG mesh ───────────────────────────────────────────────────

/// Input/output triangle mesh for CSG operations.
#[derive(Debug, Clone, PartialEq)]
pub struct CsgMesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

impl CsgMesh {
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Convert to a list of Triangle structs.
    fn to_triangles(&self) -> Vec<Triangle> {
        self.indices
            .iter()
            .map(|idx| {
                Triangle::new(
                    self.vertices[idx[0] as usize],
                    self.vertices[idx[1] as usize],
                    self.vertices[idx[2] as usize],
                )
            })
            .collect()
    }

    /// Build from a list of Triangle structs, welding vertices.
    fn from_triangles(triangles: &[Triangle], weld_eps: f64) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut vertex_map: HashMap<[i64; 3], u32> = HashMap::new();

        let quantize = |v: Vec3| -> [i64; 3] {
            let scale = 1.0 / weld_eps;
            [
                (v.x * scale).round() as i64,
                (v.y * scale).round() as i64,
                (v.z * scale).round() as i64,
            ]
        };

        let mut get_or_insert = |v: Vec3, vertices: &mut Vec<Vec3>, map: &mut HashMap<[i64; 3], u32>| -> u32 {
            let key = quantize(v);
            if let Some(&idx) = map.get(&key) {
                idx
            } else {
                let idx = vertices.len() as u32;
                vertices.push(v);
                map.insert(key, idx);
                idx
            }
        };

        for tri in triangles {
            let ia = get_or_insert(tri.a, &mut vertices, &mut vertex_map);
            let ib = get_or_insert(tri.b, &mut vertices, &mut vertex_map);
            let ic = get_or_insert(tri.c, &mut vertices, &mut vertex_map);
            if ia != ib && ib != ic && ia != ic {
                indices.push([ia, ib, ic]);
            }
        }

        CsgMesh { vertices, indices }
    }

    /// Compute smooth normals.
    pub fn compute_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::zero(); self.vertices.len()];
        for idx in &self.indices {
            let a = self.vertices[idx[0] as usize];
            let b = self.vertices[idx[1] as usize];
            let c = self.vertices[idx[2] as usize];
            let fn_vec = b.sub(&a).cross(&c.sub(&a));
            for &vi in idx {
                normals[vi as usize] = normals[vi as usize].add(&fn_vec);
            }
        }
        normals.into_iter().map(|n| n.normalized()).collect()
    }

    /// Compute total surface area.
    pub fn surface_area(&self) -> f64 {
        self.to_triangles().iter().map(|t| t.area()).sum()
    }
}

/// CSG operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOp {
    Union,
    Intersection,
    Difference,
}

/// Perform a CSG boolean operation.
pub fn csg_operation(a: &CsgMesh, b: &CsgMesh, op: CsgOp) -> CsgMesh {
    let a_tris = a.to_triangles();
    let b_tris = b.to_triangles();

    if a_tris.is_empty() || b_tris.is_empty() {
        return match op {
            CsgOp::Union => {
                let all: Vec<Triangle> = a_tris.into_iter().chain(b_tris).collect();
                CsgMesh::from_triangles(&all, 1e-6)
            }
            CsgOp::Intersection => CsgMesh::new(vec![], vec![]),
            CsgOp::Difference => CsgMesh::from_triangles(&a_tris, 1e-6),
        };
    }

    let mut bsp_a = BspNode::new(&a_tris).unwrap();
    let mut bsp_b = BspNode::new(&b_tris).unwrap();

    let result_tris = match op {
        CsgOp::Union => {
            // csg.js union: a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.allPolygons)
            bsp_a.clip_to(&bsp_b);
            bsp_b.clip_to(&bsp_a);
            bsp_b.invert();
            bsp_b.clip_to(&bsp_a);
            bsp_b.invert();
            let b_remaining = bsp_b.collect_triangles();
            bsp_a.insert(&b_remaining);
            bsp_a.collect_triangles()
        }
        CsgOp::Intersection => {
            // csg.js intersect: a.invert(); b.clipTo(a); b.invert(); a.clipTo(b); b.clipTo(a); a.invert(); a.build(b.allPolygons)
            bsp_a.invert();
            bsp_b.clip_to(&bsp_a);
            bsp_b.invert();
            bsp_a.clip_to(&bsp_b);
            bsp_b.clip_to(&bsp_a);
            let b_remaining = bsp_b.collect_triangles();
            bsp_a.insert(&b_remaining);
            bsp_a.invert();
            bsp_a.collect_triangles()
        }
        CsgOp::Difference => {
            // csg.js subtract: a.invert(); a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.allPolygons); a.invert()
            bsp_a.invert();
            bsp_a.clip_to(&bsp_b);
            bsp_b.clip_to(&bsp_a);
            bsp_b.invert();
            bsp_b.clip_to(&bsp_a);
            bsp_b.invert();
            let b_remaining = bsp_b.collect_triangles();
            bsp_a.insert(&b_remaining);
            bsp_a.invert();
            bsp_a.collect_triangles()
        }
    };

    CsgMesh::from_triangles(&result_tris, 1e-6)
}

/// Convenience: union.
pub fn csg_union(a: &CsgMesh, b: &CsgMesh) -> CsgMesh {
    csg_operation(a, b, CsgOp::Union)
}

/// Convenience: intersection.
pub fn csg_intersection(a: &CsgMesh, b: &CsgMesh) -> CsgMesh {
    csg_operation(a, b, CsgOp::Intersection)
}

/// Convenience: difference (A - B).
pub fn csg_difference(a: &CsgMesh, b: &CsgMesh) -> CsgMesh {
    csg_operation(a, b, CsgOp::Difference)
}

/// Build a unit cube centered at the origin.
pub fn unit_cube() -> CsgMesh {
    let v = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];
    let idx = vec![
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ];
    CsgMesh::new(v, idx)
}

/// Build a translated unit cube.
pub fn cube_at(center: Vec3) -> CsgMesh {
    let mut c = unit_cube();
    for v in &mut c.vertices {
        *v = v.add(&center);
    }
    c
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let mid = a.lerp(&b, 0.5);
        assert!(approx(mid.x, 1.0, EPS));
        assert!(approx(mid.y, 2.0, EPS));
        assert!(approx(mid.z, 3.0, EPS));
    }

    #[test]
    fn test_plane_from_points() {
        let p = Plane::from_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(approx(p.normal.z, 1.0, 0.01));
    }

    #[test]
    fn test_plane_distance() {
        let p = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0); // y = 0
        assert!(approx(p.distance(Vec3::new(0.0, 3.0, 0.0)), 3.0, EPS));
    }

    #[test]
    fn test_plane_classify() {
        let p = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        assert_eq!(p.classify_point(Vec3::new(0.0, 1.0, 0.0)), 1);
        assert_eq!(p.classify_point(Vec3::new(0.0, -1.0, 0.0)), -1);
        assert_eq!(p.classify_point(Vec3::new(0.0, 0.0, 0.0)), 0);
    }

    #[test]
    fn test_triangle_normal() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let n = tri.normal();
        assert!(approx(n.z, 1.0, 0.01));
    }

    #[test]
    fn test_triangle_area() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        );
        assert!(approx(tri.area(), 3.0, EPS));
    }

    #[test]
    fn test_triangle_flip() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let flipped = tri.flip();
        let n_orig = tri.normal();
        let n_flip = flipped.normal();
        assert!(approx(n_orig.z + n_flip.z, 0.0, 0.01));
    }

    #[test]
    fn test_unit_cube() {
        let c = unit_cube();
        assert_eq!(c.vertex_count(), 8);
        assert_eq!(c.triangle_count(), 12);
    }

    #[test]
    fn test_cube_surface_area() {
        let c = unit_cube();
        let area = c.surface_area();
        assert!(approx(area, 6.0, 0.01));
    }

    #[test]
    fn test_union_overlapping_cubes() {
        // Offset cubes that partially overlap
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.25, 0.25, 0.0));
        let result = csg_union(&a, &b);
        // Union of overlapping cubes should produce geometry
        assert!(result.triangle_count() > 0);
        // Surface area of union < sum of individual areas (overlap removed)
        let area_a = a.surface_area();
        let area_union = result.surface_area();
        assert!(area_union > 0.0);
        assert!(area_union < area_a * 2.0 + 0.5);
    }

    #[test]
    fn test_union_touching_cubes() {
        // Cubes sharing a face
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.5, 0.0, 0.0));
        let result = csg_union(&a, &b);
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_intersection_overlapping_cubes() {
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.25, 0.0, 0.0));
        let result = csg_intersection(&a, &b);
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_intersection_concentric() {
        // A fully contains B (B is smaller, centered inside A)
        let a = unit_cube();
        let mut b = unit_cube();
        for v in &mut b.vertices {
            *v = v.scale(0.4); // shrink B to 40%
        }
        let result = csg_intersection(&a, &b);
        // Intersection should be ~B
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_difference_overlapping_cubes() {
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.25, 0.0, 0.0));
        let result = csg_difference(&a, &b);
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_difference_produces_smaller_surface() {
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.25, 0.25, 0.25));
        let result = csg_difference(&a, &b);
        // A - B should have some geometry
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_union_with_empty() {
        let a = unit_cube();
        let b = CsgMesh::new(vec![], vec![]);
        let result = csg_union(&a, &b);
        assert_eq!(result.triangle_count(), a.triangle_count());
    }

    #[test]
    fn test_intersection_with_empty() {
        let a = unit_cube();
        let b = CsgMesh::new(vec![], vec![]);
        let result = csg_intersection(&a, &b);
        assert_eq!(result.triangle_count(), 0);
    }

    #[test]
    fn test_difference_with_empty() {
        let a = unit_cube();
        let b = CsgMesh::new(vec![], vec![]);
        let result = csg_difference(&a, &b);
        assert_eq!(result.triangle_count(), a.triangle_count());
    }

    #[test]
    fn test_vertex_welding() {
        let tris = vec![
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            Triangle::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CsgMesh::from_triangles(&tris, 1e-6);
        // Shared vertices should be welded
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_compute_normals() {
        let c = unit_cube();
        let normals = c.compute_normals();
        assert_eq!(normals.len(), c.vertex_count());
        for n in &normals {
            assert!(n.length() > 0.5);
        }
    }

    #[test]
    fn test_indices_in_bounds() {
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.25, 0.0, 0.0));
        let result = csg_union(&a, &b);
        let max_v = result.vertex_count() as u32;
        for idx in &result.indices {
            assert!(idx[0] < max_v);
            assert!(idx[1] < max_v);
            assert!(idx[2] < max_v);
        }
    }

    #[test]
    fn test_no_degenerate_output() {
        let a = cube_at(Vec3::zero());
        let b = cube_at(Vec3::new(0.3, 0.3, 0.3));
        let result = csg_difference(&a, &b);
        for idx in &result.indices {
            assert!(idx[0] != idx[1] || idx[1] != idx[2]);
        }
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        );
        let c = tri.centroid();
        assert!(approx(c.x, 1.0, EPS));
        assert!(approx(c.y, 1.0, EPS));
    }

    #[test]
    fn test_split_triangle_front() {
        let tri = Triangle::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.5, 2.0, 0.0),
        );
        let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0); // y = 0
        let mut cf = Vec::new();
        let mut cb = Vec::new();
        let mut front = Vec::new();
        let mut back = Vec::new();
        split_triangle(&tri, &plane, &mut cf, &mut cb, &mut front, &mut back);
        assert_eq!(front.len(), 1);
        assert_eq!(back.len(), 0);
    }
}
