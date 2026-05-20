//! 3D geometry primitives — pure-Rust replacement for three.js math, gl-matrix.
//!
//! Vec3, Plane, Sphere, AABB, Ray. Ray-sphere/ray-plane/ray-AABB intersection,
//! frustum culling, distance functions, mesh normal calculation.

use std::fmt;

const EPS: f64 = 1e-10;

// ── Vec3 ──────────────────────────────────────────────────────

/// A 3D vector / point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const RIGHT: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: -1.0 };

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < EPS { return Self::ZERO; }
        self.scale(1.0 / len)
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn negate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    pub fn distance_to(self, other: Self) -> f64 {
        self.sub(other).length()
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self.scale(1.0 - t).add(other.scale(t))
    }

    pub fn reflect(self, normal: Self) -> Self {
        self.sub(normal.scale(2.0 * self.dot(normal)))
    }

    pub fn project_onto(self, other: Self) -> Self {
        let d = other.dot(other);
        if d < EPS { return Self::ZERO; }
        other.scale(self.dot(other) / d)
    }

    pub fn angle_between(self, other: Self) -> f64 {
        let d = self.dot(other);
        let len_prod = self.length() * other.length();
        if len_prod < EPS { return 0.0; }
        (d / len_prod).clamp(-1.0, 1.0).acos()
    }

    pub fn component_min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
    }

    pub fn component_max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
    }

    pub fn approx_eq(self, other: Self, tol: f64) -> bool {
        (self.x - other.x).abs() < tol
            && (self.y - other.y).abs() < tol
            && (self.z - other.z).abs() < tol
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Ray ───────────────────────────────────────────────────────

/// A ray defined by an origin and direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }

    /// Point along the ray at parameter t.
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin.add(self.direction.scale(t))
    }

    /// Closest point on the ray to a given point (t >= 0).
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        let t = p.sub(self.origin).dot(self.direction).max(0.0);
        self.at(t)
    }

    /// Distance from the ray to a point.
    pub fn distance_to_point(&self, p: Vec3) -> f64 {
        p.distance_to(self.closest_point(p))
    }
}

// ── Plane ─────────────────────────────────────────────────────

/// A plane defined by normal . point = d, or equivalently normal.x*x + normal.y*y + normal.z*z = d.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f64,
}

impl Plane {
    /// Create from a normal and a point on the plane.
    pub fn from_normal_and_point(normal: Vec3, point: Vec3) -> Self {
        let n = normal.normalize();
        Self { normal: n, d: n.dot(point) }
    }

    /// Create from three points (CCW winding = normal direction).
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = b.sub(a).cross(c.sub(a)).normalize();
        Self { normal, d: normal.dot(a) }
    }

    /// Signed distance from a point to the plane.
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        self.normal.dot(p) - self.d
    }

    /// Unsigned distance from a point to the plane.
    pub fn distance(&self, p: Vec3) -> f64 {
        self.signed_distance(p).abs()
    }

    /// Project a point onto the plane.
    pub fn project_point(&self, p: Vec3) -> Vec3 {
        p.sub(self.normal.scale(self.signed_distance(p)))
    }

    /// Ray-plane intersection. Returns parameter t (ray.at(t)) or None if parallel.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < EPS {
            return None;
        }
        let t = (self.d - self.normal.dot(ray.origin)) / denom;
        if t >= -EPS {
            Some(t)
        } else {
            None
        }
    }
}

// ── Sphere ────────────────────────────────────────────────────

/// A sphere defined by center and radius.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64) -> Self {
        Self { center, radius: radius.abs() }
    }

    pub fn contains(&self, p: Vec3) -> bool {
        self.center.distance_to(p) <= self.radius + EPS
    }

    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius * self.radius
    }

    /// Ray-sphere intersection. Returns (t_near, t_far) or None.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f64, f64)> {
        let oc = ray.origin.sub(self.center);
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);
        Some((t1, t2))
    }

    /// Whether this sphere intersects another sphere.
    pub fn intersects_sphere(&self, other: &Sphere) -> bool {
        self.center.distance_to(other.center) <= self.radius + other.radius + EPS
    }
}

// ── AABB ──────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.component_min(max),
            max: min.component_max(max),
        }
    }

    /// Create from center and half-extents.
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self::new(center.sub(half_extents), center.add(half_extents))
    }

    pub fn center(&self) -> Vec3 {
        self.min.lerp(self.max, 0.5)
    }

    pub fn extents(&self) -> Vec3 {
        self.max.sub(self.min)
    }

    pub fn half_extents(&self) -> Vec3 {
        self.extents().scale(0.5)
    }

    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    pub fn surface_area(&self) -> f64 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.x * e.z)
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x - EPS && p.x <= self.max.x + EPS
            && p.y >= self.min.y - EPS && p.y <= self.max.y + EPS
            && p.z >= self.min.z - EPS && p.z <= self.max.z + EPS
    }

    /// Whether two AABBs overlap.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x + EPS && self.max.x >= other.min.x - EPS
            && self.min.y <= other.max.y + EPS && self.max.y >= other.min.y - EPS
            && self.min.z <= other.max.z + EPS && self.max.z >= other.min.z - EPS
    }

    /// Merge two AABBs.
    pub fn merge(&self, other: &Aabb) -> Self {
        Self::new(self.min.component_min(other.min), self.max.component_max(other.max))
    }

    /// Expand to include a point.
    pub fn expand_to_include(&self, p: Vec3) -> Self {
        Self::new(self.min.component_min(p), self.max.component_max(p))
    }

    /// Ray-AABB intersection (slab method). Returns (t_min, t_max) or None.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f64, f64)> {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;

        let axes = [
            (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
            (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
            (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
        ];

        for (orig, dir, bmin, bmax) in axes {
            if dir.abs() < EPS {
                if orig < bmin || orig > bmax {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir;
                let mut t1 = (bmin - orig) * inv_d;
                let mut t2 = (bmax - orig) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                tmin = tmin.max(t1);
                tmax = tmax.min(t2);
                if tmin > tmax {
                    return None;
                }
            }
        }

        if tmax < 0.0 {
            return None;
        }
        Some((tmin, tmax))
    }

    /// Closest point on the AABB surface to a given point.
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
            p.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Distance from the AABB to a point.
    pub fn distance_to_point(&self, p: Vec3) -> f64 {
        p.distance_to(self.closest_point(p))
    }
}

// ── Frustum ───────────────────────────────────────────────────

/// A view frustum defined by 6 planes (near, far, left, right, top, bottom).
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    pub fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Build a frustum from a combined view-projection matrix (column-major 4x4).
    pub fn from_matrix(m: &[f64; 16]) -> Self {
        // Gribb/Hartmann method
        let planes = [
            // Near
            Plane {
                normal: Vec3::new(m[3] + m[2], m[7] + m[6], m[11] + m[10]).normalize(),
                d: -(m[15] + m[14]) / Vec3::new(m[3] + m[2], m[7] + m[6], m[11] + m[10]).length(),
            },
            // Far
            Plane {
                normal: Vec3::new(m[3] - m[2], m[7] - m[6], m[11] - m[10]).normalize(),
                d: -(m[15] - m[14]) / Vec3::new(m[3] - m[2], m[7] - m[6], m[11] - m[10]).length(),
            },
            // Left
            Plane {
                normal: Vec3::new(m[3] + m[0], m[7] + m[4], m[11] + m[8]).normalize(),
                d: -(m[15] + m[12]) / Vec3::new(m[3] + m[0], m[7] + m[4], m[11] + m[8]).length(),
            },
            // Right
            Plane {
                normal: Vec3::new(m[3] - m[0], m[7] - m[4], m[11] - m[8]).normalize(),
                d: -(m[15] - m[12]) / Vec3::new(m[3] - m[0], m[7] - m[4], m[11] - m[8]).length(),
            },
            // Top
            Plane {
                normal: Vec3::new(m[3] - m[1], m[7] - m[5], m[11] - m[9]).normalize(),
                d: -(m[15] - m[13]) / Vec3::new(m[3] - m[1], m[7] - m[5], m[11] - m[9]).length(),
            },
            // Bottom
            Plane {
                normal: Vec3::new(m[3] + m[1], m[7] + m[5], m[11] + m[9]).normalize(),
                d: -(m[15] + m[13]) / Vec3::new(m[3] + m[1], m[7] + m[5], m[11] + m[9]).length(),
            },
        ];
        Self { planes }
    }

    /// Whether a point is inside the frustum.
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(p) >= -EPS)
    }

    /// Whether a sphere is at least partially inside the frustum.
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(sphere.center) >= -sphere.radius - EPS)
    }

    /// Whether an AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            // Test the positive vertex (farthest in normal direction)
            let px = if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z };
            let p_vertex = Vec3::new(px, py, pz);
            if plane.signed_distance(p_vertex) < -EPS {
                return false;
            }
        }
        true
    }
}

// ── Mesh normals ──────────────────────────────────────────────

/// Compute the face normal for a triangle.
pub fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    b.sub(a).cross(c.sub(a)).normalize()
}

/// Compute smooth vertex normals from triangle mesh data.
/// `positions` is vertex positions; `indices` is triplets of triangle indices.
/// Returns a normal for each vertex (area-weighted average of adjacent face normals).
pub fn compute_vertex_normals(positions: &[Vec3], indices: &[usize]) -> Vec<Vec3> {
    let n = positions.len();
    let mut normals = vec![Vec3::ZERO; n];

    let tri_count = indices.len() / 3;
    for t in 0..tri_count {
        let i0 = indices[t * 3];
        let i1 = indices[t * 3 + 1];
        let i2 = indices[t * 3 + 2];
        let face_normal = positions[i1].sub(positions[i0])
            .cross(positions[i2].sub(positions[i0]));
        // Not normalized — magnitude encodes area weight
        normals[i0] = normals[i0].add(face_normal);
        normals[i1] = normals[i1].add(face_normal);
        normals[i2] = normals[i2].add(face_normal);
    }

    for normal in &mut normals {
        *normal = normal.normalize();
    }
    normals
}

/// Compute distance from a point to a line segment in 3D.
pub fn point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b.sub(a);
    let ap = p.sub(a);
    let len_sq = ab.dot(ab);
    if len_sq < EPS {
        return p.distance_to(a);
    }
    let t = ap.dot(ab) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest = a.add(ab.scale(t));
    p.distance_to(closest)
}

/// Compute distance from a point to a triangle in 3D.
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ap = p.sub(a);

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return p.distance_to(a);
    }

    let bp = p.sub(b);
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return p.distance_to(b);
    }

    let cp_vec = p.sub(c);
    let d5 = ab.dot(cp_vec);
    let d6 = ac.dot(cp_vec);
    if d6 >= 0.0 && d5 <= d6 {
        return p.distance_to(c);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a.add(ab.scale(v));
        return p.distance_to(closest);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a.add(ac.scale(w));
        return p.distance_to(closest);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b.add(c.sub(b).scale(w));
        return p.distance_to(closest);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a.add(ab.scale(v)).add(ac.scale(w));
    p.distance_to(closest)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < EPS);
    }

    #[test]
    fn vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!(c.approx_eq(Vec3::new(0.0, 0.0, 1.0), EPS));
    }

    #[test]
    fn vec3_length_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < EPS);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < EPS);
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let reflected = v.reflect(normal);
        assert!(reflected.approx_eq(Vec3::new(1.0, 1.0, 0.0), EPS));
    }

    #[test]
    fn vec3_lerp() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert!(mid.approx_eq(Vec3::new(5.0, 5.0, 5.0), EPS));
    }

    #[test]
    fn vec3_angle_between() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let angle = a.angle_between(b);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn ray_sphere_intersection() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        let (t1, t2) = sphere.intersect_ray(&ray).unwrap();
        assert!((t1 - 4.0).abs() < EPS);
        assert!((t2 - 6.0).abs() < EPS);
    }

    #[test]
    fn ray_sphere_miss() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new(Vec3::ZERO, 1.0);
        assert!(sphere.intersect_ray(&ray).is_none());
    }

    #[test]
    fn ray_plane_intersection() {
        let plane = Plane::from_normal_and_point(Vec3::UP, Vec3::ZERO);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let t = plane.intersect_ray(&ray).unwrap();
        assert!((t - 5.0).abs() < EPS);
    }

    #[test]
    fn ray_aabb_intersection() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let (tmin, tmax) = aabb.intersect_ray(&ray).unwrap();
        assert!((tmin - 4.0).abs() < EPS);
        assert!((tmax - 6.0).abs() < EPS);
    }

    #[test]
    fn ray_aabb_miss() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(aabb.intersect_ray(&ray).is_none());
    }

    #[test]
    fn aabb_contains() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert!(aabb.contains(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!aabb.contains(Vec3::new(3.0, 1.0, 1.0)));
    }

    #[test]
    fn aabb_intersection() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let b = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(a.intersects(&b));
        let c = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_merge() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let merged = a.merge(&b);
        assert!(merged.min.approx_eq(Vec3::ZERO, EPS));
        assert!(merged.max.approx_eq(Vec3::new(3.0, 3.0, 3.0), EPS));
    }

    #[test]
    fn plane_distance() {
        let plane = Plane::from_normal_and_point(Vec3::UP, Vec3::ZERO);
        assert!((plane.distance(Vec3::new(0.0, 5.0, 0.0)) - 5.0).abs() < EPS);
        assert!((plane.signed_distance(Vec3::new(0.0, -3.0, 0.0)) - (-3.0)).abs() < EPS);
    }

    #[test]
    fn triangle_normal_calc() {
        let n = triangle_normal(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(n.approx_eq(Vec3::new(0.0, 0.0, 1.0), EPS));
    }

    #[test]
    fn vertex_normals() {
        // A simple quad made of two triangles
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        let normals = compute_vertex_normals(&positions, &indices);
        for normal in &normals {
            assert!(normal.approx_eq(Vec3::new(0.0, 0.0, 1.0), EPS));
        }
    }

    #[test]
    fn sphere_volume_area() {
        let s = Sphere::new(Vec3::ZERO, 1.0);
        assert!((s.volume() - 4.0 / 3.0 * std::f64::consts::PI).abs() < EPS);
        assert!((s.surface_area() - 4.0 * std::f64::consts::PI).abs() < EPS);
    }

    #[test]
    fn point_segment_dist() {
        let d = point_segment_distance(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert!((d - 1.0).abs() < EPS);
    }

    #[test]
    fn aabb_closest_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let p = Vec3::new(3.0, 1.0, 1.0);
        let closest = aabb.closest_point(p);
        assert!(closest.approx_eq(Vec3::new(2.0, 1.0, 1.0), EPS));
    }

    #[test]
    fn sphere_contains() {
        let s = Sphere::new(Vec3::ZERO, 5.0);
        assert!(s.contains(Vec3::new(3.0, 4.0, 0.0)));
        assert!(!s.contains(Vec3::new(3.0, 4.0, 1.0)));
    }

    #[test]
    fn sphere_sphere_intersect() {
        let a = Sphere::new(Vec3::ZERO, 2.0);
        let b = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 2.0);
        assert!(a.intersects_sphere(&b));
        let c = Sphere::new(Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!(!a.intersects_sphere(&c));
    }

    #[test]
    fn plane_from_three_points() {
        let plane = Plane::from_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(plane.normal.approx_eq(Vec3::new(0.0, 0.0, 1.0), EPS));
    }

    #[test]
    fn plane_project() {
        let plane = Plane::from_normal_and_point(Vec3::UP, Vec3::ZERO);
        let projected = plane.project_point(Vec3::new(3.0, 5.0, 7.0));
        assert!(projected.approx_eq(Vec3::new(3.0, 0.0, 7.0), EPS));
    }
}
