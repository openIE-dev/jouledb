//! 3D Collision Shapes — sphere, AABB, OBB, capsule, convex hull, triangle mesh.
//! Contact generation with normal/depth/contact points, shape-vs-shape dispatch,
//! point-in-shape queries, closest point, volume and centroid calculation.

// ── Vec3 ─────────────────────────────────────────────────────

/// 3-component vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }

    pub fn cross(self, r: Self) -> Self {
        Self {
            x: self.y * r.z - self.z * r.y,
            y: self.z * r.x - self.x * r.z,
            z: self.x * r.y - self.y * r.x,
        }
    }

    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }

    pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }

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
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

// ── Quaternion (for OBB orientation) ─────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion { pub w: f64, pub x: f64, pub y: f64, pub z: f64 }

impl Quaternion {
    pub const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let h = angle * 0.5;
        let s = h.sin();
        Self { w: h.cos(), x: axis.x * s, y: axis.y * s, z: axis.z * s }
    }

    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v + uv * (2.0 * self.w) + uuv * 2.0
    }

    pub fn conjugate(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Get the local axes of this orientation.
    pub fn axes(self) -> [Vec3; 3] {
        [
            self.rotate_vec(Vec3::new(1.0, 0.0, 0.0)),
            self.rotate_vec(Vec3::new(0.0, 1.0, 0.0)),
            self.rotate_vec(Vec3::new(0.0, 0.0, 1.0)),
        ]
    }
}

// ── Contact ──────────────────────────────────────────────────

/// A single contact point from a collision test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f64,
}

// ── Collision Shapes ─────────────────────────────────────────

/// 3D collision shape variants.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere { center: Vec3, radius: f64 },
    Aabb { min: Vec3, max: Vec3 },
    Obb { center: Vec3, half_extents: Vec3, orientation: Quaternion },
    Capsule { a: Vec3, b: Vec3, radius: f64 },
    ConvexHull { vertices: Vec<Vec3> },
    TriangleMesh { vertices: Vec<Vec3>, indices: Vec<[usize; 3]> },
}

impl CollisionShape {
    // ── Constructors ────

    pub fn sphere(center: Vec3, radius: f64) -> Self {
        CollisionShape::Sphere { center, radius }
    }

    pub fn aabb(min: Vec3, max: Vec3) -> Self {
        CollisionShape::Aabb { min, max }
    }

    pub fn obb(center: Vec3, half_extents: Vec3, orientation: Quaternion) -> Self {
        CollisionShape::Obb { center, half_extents, orientation }
    }

    pub fn capsule(a: Vec3, b: Vec3, radius: f64) -> Self {
        CollisionShape::Capsule { a, b, radius }
    }

    pub fn convex_hull(vertices: Vec<Vec3>) -> Self {
        CollisionShape::ConvexHull { vertices }
    }

    pub fn triangle_mesh(vertices: Vec<Vec3>, indices: Vec<[usize; 3]>) -> Self {
        CollisionShape::TriangleMesh { vertices, indices }
    }

    // ── AABB envelope ────

    pub fn bounding_aabb(&self) -> (Vec3, Vec3) {
        match self {
            CollisionShape::Sphere { center, radius } => {
                let r = Vec3::new(*radius, *radius, *radius);
                (*center - r, *center + r)
            }
            CollisionShape::Aabb { min, max } => (*min, *max),
            CollisionShape::Obb { center, half_extents, orientation } => {
                let axes = orientation.axes();
                let mut extent = Vec3::ZERO;
                for i in 0..3 {
                    let he = match i {
                        0 => half_extents.x,
                        1 => half_extents.y,
                        _ => half_extents.z,
                    };
                    extent = Vec3::new(
                        extent.x + axes[i].x.abs() * he,
                        extent.y + axes[i].y.abs() * he,
                        extent.z + axes[i].z.abs() * he,
                    );
                }
                (*center - extent, *center + extent)
            }
            CollisionShape::Capsule { a, b, radius } => {
                let r = Vec3::new(*radius, *radius, *radius);
                (a.min_comp(*b) - r, a.max_comp(*b) + r)
            }
            CollisionShape::ConvexHull { vertices } => {
                if vertices.is_empty() {
                    return (Vec3::ZERO, Vec3::ZERO);
                }
                let mut lo = vertices[0];
                let mut hi = vertices[0];
                for v in &vertices[1..] {
                    lo = lo.min_comp(*v);
                    hi = hi.max_comp(*v);
                }
                (lo, hi)
            }
            CollisionShape::TriangleMesh { vertices, .. } => {
                if vertices.is_empty() {
                    return (Vec3::ZERO, Vec3::ZERO);
                }
                let mut lo = vertices[0];
                let mut hi = vertices[0];
                for v in &vertices[1..] {
                    lo = lo.min_comp(*v);
                    hi = hi.max_comp(*v);
                }
                (lo, hi)
            }
        }
    }

    // ── Volume ────

    pub fn volume(&self) -> f64 {
        match self {
            CollisionShape::Sphere { radius, .. } => {
                (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius
            }
            CollisionShape::Aabb { min, max } => {
                let d = *max - *min;
                d.x * d.y * d.z
            }
            CollisionShape::Obb { half_extents, .. } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            CollisionShape::Capsule { a, b, radius } => {
                let h = (*b - *a).length();
                let r = *radius;
                std::f64::consts::PI * r * r * h + (4.0 / 3.0) * std::f64::consts::PI * r * r * r
            }
            CollisionShape::ConvexHull { vertices } => {
                convex_hull_volume(vertices)
            }
            CollisionShape::TriangleMesh { vertices, indices } => {
                mesh_volume(vertices, indices)
            }
        }
    }

    // ── Centroid ────

    pub fn centroid(&self) -> Vec3 {
        match self {
            CollisionShape::Sphere { center, .. } => *center,
            CollisionShape::Aabb { min, max } => (*min + *max) * 0.5,
            CollisionShape::Obb { center, .. } => *center,
            CollisionShape::Capsule { a, b, .. } => (*a + *b) * 0.5,
            CollisionShape::ConvexHull { vertices } => {
                if vertices.is_empty() { return Vec3::ZERO; }
                let mut sum = Vec3::ZERO;
                for v in vertices { sum = sum + *v; }
                sum * (1.0 / vertices.len() as f64)
            }
            CollisionShape::TriangleMesh { vertices, .. } => {
                if vertices.is_empty() { return Vec3::ZERO; }
                let mut sum = Vec3::ZERO;
                for v in vertices { sum = sum + *v; }
                sum * (1.0 / vertices.len() as f64)
            }
        }
    }

    // ── Point-in-shape ────

    pub fn contains_point(&self, p: Vec3) -> bool {
        match self {
            CollisionShape::Sphere { center, radius } => {
                (p - *center).length_sq() <= radius * radius
            }
            CollisionShape::Aabb { min, max } => {
                p.x >= min.x && p.x <= max.x
                    && p.y >= min.y && p.y <= max.y
                    && p.z >= min.z && p.z <= max.z
            }
            CollisionShape::Obb { center, half_extents, orientation } => {
                let local = orientation.conjugate().rotate_vec(p - *center);
                local.x.abs() <= half_extents.x
                    && local.y.abs() <= half_extents.y
                    && local.z.abs() <= half_extents.z
            }
            CollisionShape::Capsule { a, b, radius } => {
                let d = closest_point_on_segment(p, *a, *b);
                (p - d).length_sq() <= radius * radius
            }
            CollisionShape::ConvexHull { vertices } => {
                point_in_convex_hull(p, vertices)
            }
            CollisionShape::TriangleMesh { .. } => {
                // Point-in-mesh is expensive; use raycast parity
                false // conservative: not supported for mesh
            }
        }
    }

    // ── Closest point ────

    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        match self {
            CollisionShape::Sphere { center, radius } => {
                let d = p - *center;
                let len = d.length();
                if len < 1e-12 {
                    *center + Vec3::new(*radius, 0.0, 0.0)
                } else {
                    *center + d * (*radius / len)
                }
            }
            CollisionShape::Aabb { min, max } => {
                Vec3::new(
                    p.x.max(min.x).min(max.x),
                    p.y.max(min.y).min(max.y),
                    p.z.max(min.z).min(max.z),
                )
            }
            CollisionShape::Obb { center, half_extents, orientation } => {
                let local = orientation.conjugate().rotate_vec(p - *center);
                let clamped = Vec3::new(
                    local.x.max(-half_extents.x).min(half_extents.x),
                    local.y.max(-half_extents.y).min(half_extents.y),
                    local.z.max(-half_extents.z).min(half_extents.z),
                );
                *center + orientation.rotate_vec(clamped)
            }
            CollisionShape::Capsule { a, b, radius } => {
                let seg_pt = closest_point_on_segment(p, *a, *b);
                let d = p - seg_pt;
                let len = d.length();
                if len < 1e-12 {
                    seg_pt + Vec3::new(*radius, 0.0, 0.0)
                } else {
                    seg_pt + d * (*radius / len)
                }
            }
            CollisionShape::ConvexHull { vertices } => {
                closest_point_on_convex_hull(p, vertices)
            }
            CollisionShape::TriangleMesh { vertices, indices } => {
                closest_point_on_mesh(p, vertices, indices)
            }
        }
    }
}

// ── Helper: closest point on segment ─────────────────────────

fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-12 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

// ── Helper: convex hull volume (tetrahedra from origin) ──────

fn convex_hull_volume(verts: &[Vec3]) -> f64 {
    if verts.len() < 4 { return 0.0; }
    // Use first vertex as pivot, treat each consecutive triple as face
    let mut vol = 0.0;
    let o = verts[0];
    for i in 1..verts.len() - 1 {
        let a = verts[i] - o;
        let b = verts[(i + 1) % verts.len()] - o;
        let c = if i + 2 < verts.len() { verts[i + 2] - o } else { verts[1] - o };
        vol += a.dot(b.cross(c)).abs();
    }
    vol / 6.0
}

fn mesh_volume(verts: &[Vec3], indices: &[[usize; 3]]) -> f64 {
    let mut vol = 0.0;
    for tri in indices {
        if tri[0] >= verts.len() || tri[1] >= verts.len() || tri[2] >= verts.len() {
            continue;
        }
        let a = verts[tri[0]];
        let b = verts[tri[1]];
        let c = verts[tri[2]];
        vol += a.dot(b.cross(c));
    }
    (vol / 6.0).abs()
}

// ── Helper: point in convex hull (approximate) ───────────────

fn point_in_convex_hull(p: Vec3, verts: &[Vec3]) -> bool {
    if verts.len() < 4 { return false; }
    let centroid = {
        let mut s = Vec3::ZERO;
        for v in verts { s = s + *v; }
        s * (1.0 / verts.len() as f64)
    };
    // Check if p is "inside" relative to centroid, using distance heuristic
    let max_dist_sq = verts.iter()
        .map(|v| (*v - centroid).length_sq())
        .fold(0.0f64, f64::max);
    (p - centroid).length_sq() <= max_dist_sq
}

fn closest_point_on_convex_hull(p: Vec3, verts: &[Vec3]) -> Vec3 {
    if verts.is_empty() { return Vec3::ZERO; }
    let mut best = verts[0];
    let mut best_dist = (p - best).length_sq();
    for v in &verts[1..] {
        let d = (p - *v).length_sq();
        if d < best_dist {
            best_dist = d;
            best = *v;
        }
    }
    best
}

fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

fn closest_point_on_mesh(p: Vec3, verts: &[Vec3], indices: &[[usize; 3]]) -> Vec3 {
    let mut best = Vec3::ZERO;
    let mut best_dist = f64::MAX;
    for tri in indices {
        if tri[0] >= verts.len() || tri[1] >= verts.len() || tri[2] >= verts.len() {
            continue;
        }
        let cp = closest_point_on_triangle(p, verts[tri[0]], verts[tri[1]], verts[tri[2]]);
        let d = (p - cp).length_sq();
        if d < best_dist {
            best_dist = d;
            best = cp;
        }
    }
    best
}

// ── Collision dispatch ───────────────────────────────────────

/// Test two shapes for collision, returning a contact if penetrating.
pub fn test_collision(a: &CollisionShape, b: &CollisionShape) -> Option<Contact> {
    match (a, b) {
        (CollisionShape::Sphere { center: ca, radius: ra },
         CollisionShape::Sphere { center: cb, radius: rb }) => {
            sphere_vs_sphere(*ca, *ra, *cb, *rb)
        }
        (CollisionShape::Sphere { center, radius },
         CollisionShape::Aabb { min, max }) |
        (CollisionShape::Aabb { min, max },
         CollisionShape::Sphere { center, radius }) => {
            sphere_vs_aabb(*center, *radius, *min, *max)
        }
        (CollisionShape::Aabb { min: amin, max: amax },
         CollisionShape::Aabb { min: bmin, max: bmax }) => {
            aabb_vs_aabb(*amin, *amax, *bmin, *bmax)
        }
        (CollisionShape::Sphere { center, radius },
         CollisionShape::Capsule { a: ca, b: cb, radius: cr }) |
        (CollisionShape::Capsule { a: ca, b: cb, radius: cr },
         CollisionShape::Sphere { center, radius }) => {
            let seg_pt = closest_point_on_segment(*center, *ca, *cb);
            sphere_vs_sphere(*center, *radius, seg_pt, *cr)
        }
        (CollisionShape::Capsule { a: a1, b: b1, radius: r1 },
         CollisionShape::Capsule { a: a2, b: b2, radius: r2 }) => {
            capsule_vs_capsule(*a1, *b1, *r1, *a2, *b2, *r2)
        }
        _ => {
            // For OBB, convex hull, and mesh: use bounding sphere fallback
            let (amin, amax) = a.bounding_aabb();
            let (bmin, bmax) = b.bounding_aabb();
            aabb_vs_aabb(amin, amax, bmin, bmax)
        }
    }
}

fn sphere_vs_sphere(ca: Vec3, ra: f64, cb: Vec3, rb: f64) -> Option<Contact> {
    let d = cb - ca;
    let dist_sq = d.length_sq();
    let sum_r = ra + rb;
    if dist_sq > sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-12 { d * (1.0 / dist) } else { Vec3::new(1.0, 0.0, 0.0) };
    let depth = sum_r - dist;
    let point = ca + normal * ra;
    Some(Contact { point, normal, depth })
}

fn sphere_vs_aabb(center: Vec3, radius: f64, min: Vec3, max: Vec3) -> Option<Contact> {
    let closest = Vec3::new(
        center.x.max(min.x).min(max.x),
        center.y.max(min.y).min(max.y),
        center.z.max(min.z).min(max.z),
    );
    let d = center - closest;
    let dist_sq = d.length_sq();
    if dist_sq > radius * radius {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-12 { d * (1.0 / dist) } else { Vec3::new(1.0, 0.0, 0.0) };
    let depth = radius - dist;
    Some(Contact { point: closest, normal, depth })
}

fn aabb_vs_aabb(amin: Vec3, amax: Vec3, bmin: Vec3, bmax: Vec3) -> Option<Contact> {
    let overlap_x = (amax.x.min(bmax.x)) - (amin.x.max(bmin.x));
    let overlap_y = (amax.y.min(bmax.y)) - (amin.y.max(bmin.y));
    let overlap_z = (amax.z.min(bmax.z)) - (amin.z.max(bmin.z));

    if overlap_x < 0.0 || overlap_y < 0.0 || overlap_z < 0.0 {
        return None;
    }

    let (normal, depth) = if overlap_x <= overlap_y && overlap_x <= overlap_z {
        let ca = (amin.x + amax.x) * 0.5;
        let cb = (bmin.x + bmax.x) * 0.5;
        let sign = if ca < cb { -1.0 } else { 1.0 };
        (Vec3::new(sign, 0.0, 0.0), overlap_x)
    } else if overlap_y <= overlap_z {
        let ca = (amin.y + amax.y) * 0.5;
        let cb = (bmin.y + bmax.y) * 0.5;
        let sign = if ca < cb { -1.0 } else { 1.0 };
        (Vec3::new(0.0, sign, 0.0), overlap_y)
    } else {
        let ca = (amin.z + amax.z) * 0.5;
        let cb = (bmin.z + bmax.z) * 0.5;
        let sign = if ca < cb { -1.0 } else { 1.0 };
        (Vec3::new(0.0, 0.0, sign), overlap_z)
    };

    let center_a = (amin + amax) * 0.5;
    let center_b = (bmin + bmax) * 0.5;
    let point = (center_a + center_b) * 0.5;

    Some(Contact { point, normal, depth })
}

fn capsule_vs_capsule(a1: Vec3, b1: Vec3, r1: f64, a2: Vec3, b2: Vec3, r2: f64) -> Option<Contact> {
    let (c1, c2) = closest_points_between_segments(a1, b1, a2, b2);
    sphere_vs_sphere(c1, r1, c2, r2)
}

fn closest_points_between_segments(a1: Vec3, b1: Vec3, a2: Vec3, b2: Vec3) -> (Vec3, Vec3) {
    let d1 = b1 - a1;
    let d2 = b2 - a2;
    let r = a1 - a2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a < 1e-12 && e < 1e-12 {
        return (a1, a2);
    }

    let (s, t);
    if a < 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e < 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = d1.dot(d2);
            let denom = a * e - b_val * b_val;
            s = if denom.abs() > 1e-12 {
                ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = ((b_val * s + f) / e).clamp(0.0, 1.0);
        }
    }

    (a1 + d1 * s, a2 + d2 * t)
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v3_approx(a: Vec3, b: Vec3) -> bool { approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) }

    #[test]
    fn test_sphere_vs_sphere_hit() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(approx(c.depth, 0.5));
    }

    #[test]
    fn test_sphere_vs_sphere_miss() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(c.is_none());
    }

    #[test]
    fn test_aabb_vs_aabb_hit() {
        let c = aabb_vs_aabb(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.5, -1.0, -1.0), Vec3::new(2.5, 1.0, 1.0),
        );
        assert!(c.is_some());
        assert!(approx(c.unwrap().depth, 0.5));
    }

    #[test]
    fn test_aabb_vs_aabb_miss() {
        let c = aabb_vs_aabb(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, -1.0, -1.0), Vec3::new(4.0, 1.0, 1.0),
        );
        assert!(c.is_none());
    }

    #[test]
    fn test_sphere_vs_aabb_hit() {
        let c = sphere_vs_aabb(
            Vec3::new(2.0, 0.0, 0.0), 1.5,
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(c.is_some());
        assert!(c.unwrap().depth > 0.0);
    }

    #[test]
    fn test_sphere_vs_aabb_miss() {
        let c = sphere_vs_aabb(
            Vec3::new(5.0, 0.0, 0.0), 1.0,
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(c.is_none());
    }

    #[test]
    fn test_capsule_vs_capsule_hit() {
        let c = capsule_vs_capsule(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 0.5,
            Vec3::new(0.8, 0.0, 0.0), Vec3::new(0.8, 2.0, 0.0), 0.5,
        );
        assert!(c.is_some());
    }

    #[test]
    fn test_capsule_vs_capsule_miss() {
        let c = capsule_vs_capsule(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 0.3,
            Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.3,
        );
        assert!(c.is_none());
    }

    #[test]
    fn test_sphere_bounding_aabb() {
        let s = CollisionShape::sphere(Vec3::new(1.0, 2.0, 3.0), 0.5);
        let (lo, hi) = s.bounding_aabb();
        assert!(v3_approx(lo, Vec3::new(0.5, 1.5, 2.5)));
        assert!(v3_approx(hi, Vec3::new(1.5, 2.5, 3.5)));
    }

    #[test]
    fn test_aabb_volume() {
        let s = CollisionShape::aabb(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert!(approx(s.volume(), 2.0 * 4.0 * 6.0));
    }

    #[test]
    fn test_sphere_volume() {
        let s = CollisionShape::sphere(Vec3::ZERO, 1.0);
        let expected = (4.0 / 3.0) * std::f64::consts::PI;
        assert!(approx(s.volume(), expected));
    }

    #[test]
    fn test_capsule_volume() {
        let s = CollisionShape::capsule(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 1.0);
        let cyl = std::f64::consts::PI * 1.0 * 2.0;
        let sph = (4.0 / 3.0) * std::f64::consts::PI;
        assert!(approx(s.volume(), cyl + sph));
    }

    #[test]
    fn test_point_in_sphere() {
        let s = CollisionShape::sphere(Vec3::ZERO, 2.0);
        assert!(s.contains_point(Vec3::new(1.0, 0.0, 0.0)));
        assert!(!s.contains_point(Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_point_in_aabb() {
        let s = CollisionShape::aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(s.contains_point(Vec3::ZERO));
        assert!(!s.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_point_in_obb() {
        let s = CollisionShape::obb(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), Quaternion::IDENTITY);
        assert!(s.contains_point(Vec3::new(0.5, 0.5, 0.5)));
        assert!(!s.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_point_in_capsule() {
        let s = CollisionShape::capsule(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 4.0, 0.0), 1.0);
        assert!(s.contains_point(Vec3::new(0.5, 2.0, 0.0)));
        assert!(!s.contains_point(Vec3::new(3.0, 2.0, 0.0)));
    }

    #[test]
    fn test_closest_point_aabb() {
        let s = CollisionShape::aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let cp = s.closest_point(Vec3::new(5.0, 0.0, 0.0));
        assert!(v3_approx(cp, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_closest_point_sphere() {
        let s = CollisionShape::sphere(Vec3::ZERO, 2.0);
        let cp = s.closest_point(Vec3::new(5.0, 0.0, 0.0));
        assert!(v3_approx(cp, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_dispatch_sphere_sphere() {
        let a = CollisionShape::sphere(Vec3::ZERO, 1.0);
        let b = CollisionShape::sphere(Vec3::new(1.5, 0.0, 0.0), 1.0);
        let c = test_collision(&a, &b);
        assert!(c.is_some());
    }

    #[test]
    fn test_dispatch_sphere_aabb() {
        let a = CollisionShape::sphere(Vec3::new(0.5, 0.0, 0.0), 1.0);
        let b = CollisionShape::aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let c = test_collision(&a, &b);
        assert!(c.is_some());
    }

    #[test]
    fn test_obb_centroid() {
        let s = CollisionShape::obb(
            Vec3::new(3.0, 4.0, 5.0),
            Vec3::new(1.0, 1.0, 1.0),
            Quaternion::IDENTITY,
        );
        assert!(v3_approx(s.centroid(), Vec3::new(3.0, 4.0, 5.0)));
    }

    #[test]
    fn test_obb_volume() {
        let s = CollisionShape::obb(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            Quaternion::IDENTITY,
        );
        assert!(approx(s.volume(), 48.0)); // 8 * 1 * 2 * 3
    }

    #[test]
    fn test_closest_point_on_segment_midpoint() {
        let cp = closest_point_on_segment(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert!(v3_approx(cp, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_closest_point_on_segment_endpoint() {
        let cp = closest_point_on_segment(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert!(v3_approx(cp, Vec3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_triangle_mesh_closest_point() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let indices = vec![[0, 1, 2]];
        let s = CollisionShape::triangle_mesh(verts, indices);
        let cp = s.closest_point(Vec3::new(0.0, 0.0, 5.0));
        assert!(v3_approx(cp, Vec3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_obb_bounding_aabb_aligned() {
        let s = CollisionShape::obb(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            Quaternion::IDENTITY,
        );
        let (lo, hi) = s.bounding_aabb();
        assert!(v3_approx(lo, Vec3::new(-1.0, -2.0, -3.0)));
        assert!(v3_approx(hi, Vec3::new(1.0, 2.0, 3.0)));
    }
}
