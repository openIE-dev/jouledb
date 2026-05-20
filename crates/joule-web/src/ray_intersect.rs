// Ray-primitive intersection tests for ray tracing pipeline.
// Supports sphere, AABB (slab), triangle (Möller-Trumbore), plane, disc, cylinder, capsule.

use std::fmt;

const RAY_EPSILON: f64 = 1e-6;

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
        Self { x: 0.0, y: 0.0, z: 0.0 }
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

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            return Self::zero();
        }
        self * (1.0 / len)
    }

    pub fn component_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    pub fn component_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// A ray defined by origin + direction (direction should be normalized).
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalized() }
    }

    pub fn at(self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Offset origin along normal to avoid self-intersection.
    pub fn offset_origin(point: Vec3, normal: Vec3) -> Vec3 {
        point + normal * RAY_EPSILON
    }
}

/// Hit record for an intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitRecord {
    pub t: f64,
    pub point: Vec3,
    pub normal: Vec3,
}

impl HitRecord {
    pub fn new(t: f64, point: Vec3, normal: Vec3) -> Self {
        Self { t, point, normal: normal.normalized() }
    }
}

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

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.component_min(other.min),
            max: self.max.component_max(other.max),
        }
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// Triangle defined by three vertices.
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        Self { v0, v1, v2 }
    }

    pub fn normal(&self) -> Vec3 {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2).normalized()
    }

    pub fn bounding_box(&self) -> Aabb {
        Aabb {
            min: self.v0.component_min(self.v1).component_min(self.v2),
            max: self.v0.component_max(self.v1).component_max(self.v2),
        }
    }
}

// ─── Intersection routines ───

/// Ray-sphere intersection. Returns the closest hit in [t_min, t_max].
pub fn intersect_sphere(
    ray: &Ray,
    center: Vec3,
    radius: f64,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let half_b = oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();
    let mut t = (-half_b - sqrt_d) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sqrt_d) / a;
        if t < t_min || t > t_max {
            return None;
        }
    }
    let point = ray.at(t);
    let normal = (point - center) * (1.0 / radius);
    Some(HitRecord::new(t, point, normal))
}

/// Any-hit ray-sphere: returns true if there is any intersection in [t_min, t_max].
pub fn any_hit_sphere(
    ray: &Ray,
    center: Vec3,
    radius: f64,
    t_min: f64,
    t_max: f64,
) -> bool {
    intersect_sphere(ray, center, radius, t_min, t_max).is_some()
}

/// Ray-AABB intersection via slab method. Returns (t_enter, t_exit) or None.
pub fn intersect_aabb(ray: &Ray, aabb: &Aabb, t_min: f64, t_max: f64) -> Option<(f64, f64)> {
    let inv_d = Vec3::new(
        if ray.direction.x.abs() > 1e-15 { 1.0 / ray.direction.x } else { f64::INFINITY.copysign(ray.direction.x) },
        if ray.direction.y.abs() > 1e-15 { 1.0 / ray.direction.y } else { f64::INFINITY.copysign(ray.direction.y) },
        if ray.direction.z.abs() > 1e-15 { 1.0 / ray.direction.z } else { f64::INFINITY.copysign(ray.direction.z) },
    );

    let t0x = (aabb.min.x - ray.origin.x) * inv_d.x;
    let t1x = (aabb.max.x - ray.origin.x) * inv_d.x;
    let (t0x, t1x) = if inv_d.x < 0.0 { (t1x, t0x) } else { (t0x, t1x) };

    let t0y = (aabb.min.y - ray.origin.y) * inv_d.y;
    let t1y = (aabb.max.y - ray.origin.y) * inv_d.y;
    let (t0y, t1y) = if inv_d.y < 0.0 { (t1y, t0y) } else { (t0y, t1y) };

    let t0z = (aabb.min.z - ray.origin.z) * inv_d.z;
    let t1z = (aabb.max.z - ray.origin.z) * inv_d.z;
    let (t0z, t1z) = if inv_d.z < 0.0 { (t1z, t0z) } else { (t0z, t1z) };

    let t_enter = t0x.max(t0y).max(t0z).max(t_min);
    let t_exit = t1x.min(t1y).min(t1z).min(t_max);

    if t_enter <= t_exit {
        Some((t_enter, t_exit))
    } else {
        None
    }
}

/// Ray-triangle intersection using Möller-Trumbore algorithm.
pub fn intersect_triangle(
    ray: &Ray,
    tri: &Triangle,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = ray.direction.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = ray.origin - tri.v0;
    let u = s.dot(h) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    if t < t_min || t > t_max {
        return None;
    }
    let point = ray.at(t);
    let normal = e1.cross(e2).normalized();
    // Ensure normal faces the ray
    let normal = if normal.dot(ray.direction) > 0.0 { -normal } else { normal };
    Some(HitRecord::new(t, point, normal))
}

/// Ray-plane intersection. Plane defined by point + normal.
pub fn intersect_plane(
    ray: &Ray,
    plane_point: Vec3,
    plane_normal: Vec3,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    let denom = plane_normal.dot(ray.direction);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = (plane_point - ray.origin).dot(plane_normal) / denom;
    if t < t_min || t > t_max {
        return None;
    }
    let point = ray.at(t);
    let normal = if denom > 0.0 { -plane_normal } else { plane_normal };
    Some(HitRecord::new(t, point, normal.normalized()))
}

/// Ray-disc intersection. Disc = plane + radius constraint.
pub fn intersect_disc(
    ray: &Ray,
    center: Vec3,
    normal: Vec3,
    radius: f64,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    let hit = intersect_plane(ray, center, normal, t_min, t_max)?;
    let dist_sq = (hit.point - center).length_squared();
    if dist_sq > radius * radius {
        return None;
    }
    Some(hit)
}

/// Ray-cylinder intersection (finite cylinder along axis from base to base+axis_dir*height).
pub fn intersect_cylinder(
    ray: &Ray,
    base: Vec3,
    axis: Vec3,
    radius: f64,
    height: f64,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    let axis_n = axis.normalized();
    let d = ray.direction;
    let oc = ray.origin - base;

    let d_par = axis_n * d.dot(axis_n);
    let d_perp = d - d_par;
    let oc_par = axis_n * oc.dot(axis_n);
    let oc_perp = oc - oc_par;

    let a = d_perp.dot(d_perp);
    let b = 2.0 * d_perp.dot(oc_perp);
    let c = oc_perp.dot(oc_perp) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();

    let mut best: Option<HitRecord> = None;

    for &t_cand in &[(-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)] {
        if t_cand < t_min || t_cand > t_max {
            continue;
        }
        let p = ray.at(t_cand);
        let h = (p - base).dot(axis_n);
        if h >= 0.0 && h <= height {
            let center_on_axis = base + axis_n * h;
            let normal = (p - center_on_axis).normalized();
            if best.is_none() || t_cand < best.as_ref().unwrap().t {
                best = Some(HitRecord::new(t_cand, p, normal));
            }
        }
    }
    best
}

/// Ray-capsule intersection (sphere-swept line segment).
pub fn intersect_capsule(
    ray: &Ray,
    a: Vec3,
    b: Vec3,
    radius: f64,
    t_min: f64,
    t_max: f64,
) -> Option<HitRecord> {
    // Try cylinder body
    let axis = b - a;
    let height = axis.length();
    if height < 1e-12 {
        return intersect_sphere(ray, a, radius, t_min, t_max);
    }
    let axis_n = axis * (1.0 / height);

    let mut best: Option<HitRecord> = None;

    // Cylinder part
    if let Some(hit) = intersect_cylinder(ray, a, axis_n, radius, height, t_min, t_max) {
        best = Some(hit);
    }

    // Bottom sphere cap
    if let Some(hit) = intersect_sphere(ray, a, radius, t_min, t_max) {
        let h = (hit.point - a).dot(axis_n);
        if h <= 0.0 {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    // Top sphere cap
    if let Some(hit) = intersect_sphere(ray, b, radius, t_min, t_max) {
        let h = (hit.point - b).dot(axis_n * (-1.0));
        if h <= 0.0 {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    best
}

/// Batch ray-triangle intersection for a mesh. Returns closest hit across all triangles.
pub fn intersect_mesh(
    ray: &Ray,
    triangles: &[Triangle],
    t_min: f64,
    t_max: f64,
) -> Option<(usize, HitRecord)> {
    let mut closest_t = t_max;
    let mut result: Option<(usize, HitRecord)> = None;
    for (i, tri) in triangles.iter().enumerate() {
        if let Some(hit) = intersect_triangle(ray, tri, t_min, closest_t) {
            closest_t = hit.t;
            result = Some((i, hit));
        }
    }
    result
}

/// Any-hit for mesh: returns true if any triangle hit in range.
pub fn any_hit_mesh(
    ray: &Ray,
    triangles: &[Triangle],
    t_min: f64,
    t_max: f64,
) -> bool {
    for tri in triangles {
        if intersect_triangle(ray, tri, t_min, t_max).is_some() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn test_vec3_basic_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert!(approx_eq(sum.x, 5.0, 1e-9));
        assert!(approx_eq(sum.y, 7.0, 1e-9));
        assert!(approx_eq(sum.z, 9.0, 1e-9));

        let diff = b - a;
        assert!(approx_eq(diff.x, 3.0, 1e-9));
    }

    #[test]
    fn test_vec3_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dot(b), 0.0, 1e-9));
        let c = a.cross(b);
        assert!(vec3_approx_eq(c, Vec3::new(0.0, 0.0, 1.0), 1e-9));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalized();
        assert!(approx_eq(n.length(), 1.0, 1e-9));
        assert!(approx_eq(n.x, 0.6, 1e-9));
        assert!(approx_eq(n.y, 0.8, 1e-9));
    }

    #[test]
    fn test_ray_at() {
        let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let p = r.at(5.0);
        assert!(vec3_approx_eq(p, Vec3::new(5.0, 0.0, 0.0), 1e-9));
    }

    #[test]
    fn test_sphere_hit_front() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_sphere(&ray, Vec3::zero(), 1.0, 0.0, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 4.0, 1e-6));
        assert!(vec3_approx_eq(hit.point, Vec3::new(0.0, 0.0, -1.0), 1e-6));
        assert!(vec3_approx_eq(hit.normal, Vec3::new(0.0, 0.0, -1.0), 1e-6));
    }

    #[test]
    fn test_sphere_miss() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_sphere(&ray, Vec3::zero(), 1.0, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_sphere_inside() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_sphere(&ray, Vec3::zero(), 2.0, RAY_EPSILON, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 2.0, 1e-6));
    }

    #[test]
    fn test_any_hit_sphere() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(any_hit_sphere(&ray, Vec3::zero(), 1.0, 0.0, f64::INFINITY));
        assert!(!any_hit_sphere(&ray, Vec3::new(10.0, 0.0, 0.0), 1.0, 0.0, f64::INFINITY));
    }

    #[test]
    fn test_aabb_hit() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let (t_enter, t_exit) = intersect_aabb(&ray, &aabb, 0.0, f64::INFINITY).unwrap();
        assert!(approx_eq(t_enter, 4.0, 1e-6));
        assert!(approx_eq(t_exit, 6.0, 1e-6));
    }

    #[test]
    fn test_aabb_miss() {
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(intersect_aabb(&ray, &aabb, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_aabb_inside() {
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        let aabb = Aabb::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
        let result = intersect_aabb(&ray, &aabb, 0.0, f64::INFINITY);
        assert!(result.is_some());
    }

    #[test]
    fn test_triangle_hit() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let ray = Ray::new(Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_triangle(&ray, &tri, 0.0, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 2.0, 1e-6));
        assert!(vec3_approx_eq(hit.point, Vec3::new(0.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_triangle_miss_outside() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let ray = Ray::new(Vec3::new(5.0, 5.0, -2.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_triangle(&ray, &tri, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_triangle_backface_normal() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // Shoot from +z side
        let ray = Ray::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect_triangle(&ray, &tri, 0.0, f64::INFINITY).unwrap();
        // Normal should face the ray
        assert!(hit.normal.dot(ray.direction) < 0.0);
    }

    #[test]
    fn test_plane_hit() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = intersect_plane(&ray, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0), 0.0, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 5.0, 1e-6));
        assert!(vec3_approx_eq(hit.point, Vec3::zero(), 1e-6));
    }

    #[test]
    fn test_plane_parallel() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(intersect_plane(&ray, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0), 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_disc_hit() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = intersect_disc(&ray, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0), 2.0, 0.0, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 5.0, 1e-6));
    }

    #[test]
    fn test_disc_miss_outside_radius() {
        let ray = Ray::new(Vec3::new(3.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(intersect_disc(&ray, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0), 2.0, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_cylinder_hit() {
        let ray = Ray::new(Vec3::new(-5.0, 0.5, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = intersect_cylinder(
            &ray,
            Vec3::zero(),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            2.0,
            0.0,
            f64::INFINITY,
        ).unwrap();
        assert!(approx_eq(hit.t, 4.0, 1e-5));
    }

    #[test]
    fn test_cylinder_miss_above() {
        let ray = Ray::new(Vec3::new(-5.0, 10.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(intersect_cylinder(
            &ray,
            Vec3::zero(),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            2.0,
            0.0,
            f64::INFINITY,
        ).is_none());
    }

    #[test]
    fn test_capsule_hit_body() {
        let ray = Ray::new(Vec3::new(-5.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = intersect_capsule(
            &ray,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert!(hit.is_some());
    }

    #[test]
    fn test_capsule_hit_cap() {
        // Aim at bottom cap
        let ray = Ray::new(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = intersect_capsule(
            &ray,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert!(hit.is_some());
        assert!(approx_eq(hit.unwrap().t, 4.0, 1e-5));
    }

    #[test]
    fn test_mesh_intersection() {
        let tris = vec![
            Triangle::new(
                Vec3::new(-10.0, -1.0, 5.0),
                Vec3::new(10.0, -1.0, 5.0),
                Vec3::new(0.0, 10.0, 5.0),
            ),
            Triangle::new(
                Vec3::new(-1.0, -1.0, 2.0),
                Vec3::new(1.0, -1.0, 2.0),
                Vec3::new(0.0, 1.0, 2.0),
            ),
        ];
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0));
        let (idx, hit) = intersect_mesh(&ray, &tris, 0.0, f64::INFINITY).unwrap();
        // Closer triangle is index 1 at z=2
        assert_eq!(idx, 1);
        assert!(approx_eq(hit.t, 2.0, 1e-6));
    }

    #[test]
    fn test_any_hit_mesh() {
        let tris = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 2.0),
                Vec3::new(1.0, -1.0, 2.0),
                Vec3::new(0.0, 1.0, 2.0),
            ),
        ];
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0));
        assert!(any_hit_mesh(&ray, &tris, 0.0, f64::INFINITY));
        let ray2 = Ray::new(Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(!any_hit_mesh(&ray2, &tris, 0.0, f64::INFINITY));
    }

    #[test]
    fn test_offset_origin() {
        let p = Vec3::zero();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let offset = Ray::offset_origin(p, n);
        assert!(offset.y > 0.0);
        assert!(offset.y < 1e-4);
    }

    #[test]
    fn test_aabb_surface_area() {
        let aabb = Aabb::new(Vec3::zero(), Vec3::new(1.0, 2.0, 3.0));
        // 2*(1*2 + 2*3 + 3*1) = 2*(2+6+3) = 22
        assert!(approx_eq(aabb.surface_area(), 22.0, 1e-9));
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0));
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let u = a.union(&b);
        assert!(vec3_approx_eq(u.min, Vec3::new(-1.0, -1.0, -1.0), 1e-9));
        assert!(vec3_approx_eq(u.max, Vec3::new(1.0, 1.0, 1.0), 1e-9));
    }

    #[test]
    fn test_sphere_t_range() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        // Sphere at t=4, but we restrict t_min=5 so the front hit is out of range
        // and the back hit at t=6 is in range
        let hit = intersect_sphere(&ray, Vec3::zero(), 1.0, 5.0, f64::INFINITY).unwrap();
        assert!(approx_eq(hit.t, 6.0, 1e-6));
    }
}
