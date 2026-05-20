//! Bounding volume types — AABB, OBB, bounding sphere. Construction from point
//! clouds. Intersection tests: AABB-AABB, sphere-sphere, AABB-sphere,
//! ray-AABB, ray-sphere. Merge (union of two volumes). Transform bounding
//! volumes. Enclosing volume computation.

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
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::zero() } else { self.scale(1.0 / l) }
    }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
    pub fn min_comp(self, o: Self) -> Self {
        Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }
    pub fn max_comp(self, o: Self) -> Self {
        Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }
    pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }
}

// ── Mat4 (minimal, for transforms) ──────────────────────────

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

    pub fn transform_point(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z + self.m[0][3],
            self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z + self.m[1][3],
            self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z + self.m[2][3],
        )
    }

    pub fn transform_direction(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z,
            self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z,
            self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z,
        )
    }

    pub fn translation(t: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][3] = t.x; m.m[1][3] = t.y; m.m[2][3] = t.z;
        m
    }

    pub fn scaling(s: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][0] = s.x; m.m[1][1] = s.y; m.m[2][2] = s.z;
        m
    }
}

// ── Ray ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalized() }
    }
    pub fn point_at(&self, t: f64) -> Vec3 { self.origin.add(self.direction.scale(t)) }
}

// ── AABB ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut aabb = Self::empty();
        for &p in points {
            aabb.expand_point(p);
        }
        aabb
    }

    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min_comp(p);
        self.max = self.max.max_comp(p);
    }

    pub fn center(&self) -> Vec3 { self.min.add(self.max).scale(0.5) }
    pub fn extents(&self) -> Vec3 { self.max.sub(self.min) }
    pub fn half_extents(&self) -> Vec3 { self.extents().scale(0.5) }

    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    pub fn surface_area(&self) -> f64 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects_aabb(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn intersects_sphere(&self, sphere: &BoundingSphere) -> bool {
        // Find closest point on AABB to sphere center
        let closest = Vec3::new(
            sphere.center.x.max(self.min.x).min(self.max.x),
            sphere.center.y.max(self.min.y).min(self.max.y),
            sphere.center.z.max(self.min.z).min(self.max.z),
        );
        closest.distance(sphere.center) <= sphere.radius
    }

    /// Slab test for ray-AABB intersection. Returns (t_enter, t_exit) or None.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f64, f64)> {
        let mut t_min = f64::NEG_INFINITY;
        let mut t_max = f64::INFINITY;

        let axes = [
            (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
            (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
            (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
        ];

        for &(orig, dir, bmin, bmax) in &axes {
            if dir.abs() < 1e-12 {
                if orig < bmin || orig > bmax { return None; }
            } else {
                let inv_d = 1.0 / dir;
                let mut t0 = (bmin - orig) * inv_d;
                let mut t1 = (bmax - orig) * inv_d;
                if t0 > t1 { std::mem::swap(&mut t0, &mut t1); }
                t_min = t_min.max(t0);
                t_max = t_max.min(t1);
                if t_min > t_max { return None; }
            }
        }
        if t_max < 0.0 { return None; }
        Some((t_min.max(0.0), t_max))
    }

    /// Merge two AABBs into an enclosing AABB.
    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min_comp(other.min),
            max: self.max.max_comp(other.max),
        }
    }

    /// Transform AABB by a matrix, producing a new enclosing AABB.
    pub fn transform(&self, mat: &Mat4) -> Aabb {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut result = Aabb::empty();
        for c in &corners {
            result.expand_point(mat.transform_point(*c));
        }
        result
    }
}

// ── BoundingSphere ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f64,
}

impl BoundingSphere {
    pub fn new(center: Vec3, radius: f64) -> Self { Self { center, radius } }

    /// Ritter's bounding sphere from point cloud.
    pub fn from_points(points: &[Vec3]) -> Self {
        if points.is_empty() {
            return Self::new(Vec3::zero(), 0.0);
        }
        if points.len() == 1 {
            return Self::new(points[0], 0.0);
        }
        // Find two most distant points (approximate via axis extremes)
        let mut min_x = 0usize;
        let mut max_x = 0usize;
        for i in 1..points.len() {
            if points[i].x < points[min_x].x { min_x = i; }
            if points[i].x > points[max_x].x { max_x = i; }
        }
        let mut min_y = 0; let mut max_y = 0;
        let mut min_z = 0; let mut max_z = 0;
        for i in 1..points.len() {
            if points[i].y < points[min_y].y { min_y = i; }
            if points[i].y > points[max_y].y { max_y = i; }
            if points[i].z < points[min_z].z { min_z = i; }
            if points[i].z > points[max_z].z { max_z = i; }
        }
        let dx = points[max_x].distance(points[min_x]);
        let dy = points[max_y].distance(points[min_y]);
        let dz = points[max_z].distance(points[min_z]);
        let (p1, p2) = if dx >= dy && dx >= dz {
            (points[min_x], points[max_x])
        } else if dy >= dz {
            (points[min_y], points[max_y])
        } else {
            (points[min_z], points[max_z])
        };
        let mut center = p1.add(p2).scale(0.5);
        let mut radius = p1.distance(p2) * 0.5;
        // Grow to encompass all points
        for &p in points {
            let dist = center.distance(p);
            if dist > radius {
                let new_radius = (radius + dist) * 0.5;
                let k = (new_radius - radius) / dist;
                center = center.add(p.sub(center).scale(k));
                radius = new_radius;
            }
        }
        Self { center, radius }
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        self.center.distance(p) <= self.radius + 1e-9
    }

    pub fn intersects_sphere(&self, other: &BoundingSphere) -> bool {
        self.center.distance(other.center) <= self.radius + other.radius
    }

    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        aabb.intersects_sphere(self)
    }

    /// Ray-sphere intersection. Returns closest positive t or None.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        let oc = ray.origin.sub(self.center);
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { return None; }
        let sqrt_disc = disc.sqrt();
        let t0 = (-b - sqrt_disc) / (2.0 * a);
        let t1 = (-b + sqrt_disc) / (2.0 * a);
        if t0 >= 0.0 { Some(t0) }
        else if t1 >= 0.0 { Some(t1) }
        else { None }
    }

    /// Merge two spheres into an enclosing sphere.
    pub fn merge(&self, other: &BoundingSphere) -> BoundingSphere {
        let d = self.center.distance(other.center);
        if d + other.radius <= self.radius { return *self; }
        if d + self.radius <= other.radius { return *other; }
        let new_radius = (d + self.radius + other.radius) * 0.5;
        let t = (new_radius - self.radius) / d;
        let new_center = self.center.add(other.center.sub(self.center).scale(t));
        BoundingSphere::new(new_center, new_radius)
    }

    /// Transform sphere by a matrix (applies translation + uniform scale).
    pub fn transform(&self, mat: &Mat4) -> BoundingSphere {
        let new_center = mat.transform_point(self.center);
        let sx = mat.transform_direction(Vec3::new(1.0, 0.0, 0.0)).length();
        let sy = mat.transform_direction(Vec3::new(0.0, 1.0, 0.0)).length();
        let sz = mat.transform_direction(Vec3::new(0.0, 0.0, 1.0)).length();
        let max_scale = sx.max(sy).max(sz);
        BoundingSphere::new(new_center, self.radius * max_scale)
    }
}

// ── OBB ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obb {
    pub center: Vec3,
    /// Half-extents along each local axis.
    pub half_extents: Vec3,
    /// Local axes (column vectors of rotation matrix).
    pub axes: [Vec3; 3],
}

impl Obb {
    pub fn new(center: Vec3, half_extents: Vec3, axes: [Vec3; 3]) -> Self {
        Self { center, half_extents, axes }
    }

    /// Create an OBB from an AABB (axis-aligned axes).
    pub fn from_aabb(aabb: &Aabb) -> Self {
        Self {
            center: aabb.center(),
            half_extents: aabb.half_extents(),
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    /// Get the 8 corner points.
    pub fn corners(&self) -> [Vec3; 8] {
        let mut corners = [Vec3::zero(); 8];
        for i in 0..8 {
            let sx = if i & 1 != 0 { 1.0 } else { -1.0 };
            let sy = if i & 2 != 0 { 1.0 } else { -1.0 };
            let sz = if i & 4 != 0 { 1.0 } else { -1.0 };
            corners[i] = self.center
                .add(self.axes[0].scale(self.half_extents.x * sx))
                .add(self.axes[1].scale(self.half_extents.y * sy))
                .add(self.axes[2].scale(self.half_extents.z * sz));
        }
        corners
    }

    /// Convert to an enclosing AABB.
    pub fn to_aabb(&self) -> Aabb {
        Aabb::from_points(&self.corners())
    }

    /// Test if a point is inside the OBB.
    pub fn contains_point(&self, p: Vec3) -> bool {
        let d = p.sub(self.center);
        for i in 0..3 {
            let proj = d.dot(self.axes[i]).abs();
            let extent = [self.half_extents.x, self.half_extents.y, self.half_extents.z][i];
            if proj > extent + 1e-9 { return false; }
        }
        true
    }

    /// SAT-based OBB-OBB intersection test.
    pub fn intersects_obb(&self, other: &Obb) -> bool {
        let t = other.center.sub(self.center);
        let he_a = [self.half_extents.x, self.half_extents.y, self.half_extents.z];
        let he_b = [other.half_extents.x, other.half_extents.y, other.half_extents.z];

        // Test 15 separating axes: 3 from A, 3 from B, 9 cross products
        // Precompute rotation and absolute rotation matrices
        let mut r_mat = [[0.0f64; 3]; 3];
        let mut abs_r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_mat[i][j] = self.axes[i].dot(other.axes[j]);
                abs_r[i][j] = r_mat[i][j].abs() + 1e-9;
            }
        }

        let t_proj = [t.dot(self.axes[0]), t.dot(self.axes[1]), t.dot(self.axes[2])];

        // Test axes from A
        for i in 0..3 {
            let ra = he_a[i];
            let rb = he_b[0]*abs_r[i][0] + he_b[1]*abs_r[i][1] + he_b[2]*abs_r[i][2];
            if t_proj[i].abs() > ra + rb { return false; }
        }

        // Test axes from B
        for j in 0..3 {
            let ra = he_a[0]*abs_r[0][j] + he_a[1]*abs_r[1][j] + he_a[2]*abs_r[2][j];
            let rb = he_b[j];
            let proj = t_proj[0]*r_mat[0][j] + t_proj[1]*r_mat[1][j] + t_proj[2]*r_mat[2][j];
            if proj.abs() > ra + rb { return false; }
        }

        // Test 9 cross-product axes
        let cross_tests: [(usize, usize); 9] = [
            (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),
        ];
        for &(i, j) in &cross_tests {
            let i1 = (i + 1) % 3;
            let i2 = (i + 2) % 3;
            let j1 = (j + 1) % 3;
            let j2 = (j + 2) % 3;
            let ra = he_a[i1]*abs_r[i2][j] + he_a[i2]*abs_r[i1][j];
            let rb = he_b[j1]*abs_r[i][j2] + he_b[j2]*abs_r[i][j1];
            let proj = (t_proj[i2]*r_mat[i1][j] - t_proj[i1]*r_mat[i2][j]).abs();
            if proj > ra + rb { return false; }
        }

        true
    }

    /// Transform OBB by a matrix.
    pub fn transform(&self, mat: &Mat4) -> Obb {
        let new_center = mat.transform_point(self.center);
        let new_axes = [
            mat.transform_direction(self.axes[0]).normalized(),
            mat.transform_direction(self.axes[1]).normalized(),
            mat.transform_direction(self.axes[2]).normalized(),
        ];
        let sx = mat.transform_direction(self.axes[0].scale(self.half_extents.x)).length();
        let sy = mat.transform_direction(self.axes[1].scale(self.half_extents.y)).length();
        let sz = mat.transform_direction(self.axes[2].scale(self.half_extents.z)).length();
        Obb::new(new_center, Vec3::new(sx, sy, sz), new_axes)
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_aabb_from_points() {
        let pts = [Vec3::new(-1.0, -2.0, -3.0), Vec3::new(4.0, 5.0, 6.0)];
        let aabb = Aabb::from_points(&pts);
        assert!(approx(aabb.min.x, -1.0, 1e-9));
        assert!(approx(aabb.max.z, 6.0, 1e-9));
    }

    #[test]
    fn test_aabb_center() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let c = aabb.center();
        assert!(approx(c.x, 0.0, 1e-9));
        assert!(approx(c.y, 0.0, 1e-9));
    }

    #[test]
    fn test_aabb_volume() {
        let aabb = Aabb::new(Vec3::zero(), Vec3::new(2.0, 3.0, 4.0));
        assert!(approx(aabb.volume(), 24.0, 1e-9));
    }

    #[test]
    fn test_aabb_surface_area() {
        let aabb = Aabb::new(Vec3::zero(), Vec3::new(1.0, 2.0, 3.0));
        assert!(approx(aabb.surface_area(), 22.0, 1e-9));
    }

    #[test]
    fn test_aabb_contains_point() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(aabb.contains_point(Vec3::zero()));
        assert!(!aabb.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_aabb_aabb_intersection() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        assert!(a.intersects_aabb(&b));
        let c = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects_aabb(&c));
    }

    #[test]
    fn test_sphere_from_points() {
        let pts = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        let s = BoundingSphere::from_points(&pts);
        for &p in &pts {
            assert!(s.contains_point(p), "sphere should contain all input points");
        }
    }

    #[test]
    fn test_sphere_sphere_intersection() {
        let a = BoundingSphere::new(Vec3::zero(), 1.0);
        let b = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(a.intersects_sphere(&b));
        let c = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(!a.intersects_sphere(&c));
    }

    #[test]
    fn test_aabb_sphere_intersection() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let s = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(aabb.intersects_sphere(&s));
        let far = BoundingSphere::new(Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!(!aabb.intersects_sphere(&far));
    }

    #[test]
    fn test_ray_aabb_hit() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = aabb.intersect_ray(&ray);
        assert!(hit.is_some());
        let (t_enter, _) = hit.unwrap();
        assert!(approx(t_enter, 4.0, 1e-9));
    }

    #[test]
    fn test_ray_aabb_miss() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(aabb.intersect_ray(&ray).is_none());
    }

    #[test]
    fn test_ray_aabb_inside() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        let hit = aabb.intersect_ray(&ray);
        assert!(hit.is_some());
        let (t_enter, _) = hit.unwrap();
        assert!(approx(t_enter, 0.0, 1e-9));
    }

    #[test]
    fn test_ray_sphere_hit() {
        let s = BoundingSphere::new(Vec3::zero(), 1.0);
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let t = s.intersect_ray(&ray).unwrap();
        assert!(approx(t, 4.0, 1e-9));
    }

    #[test]
    fn test_ray_sphere_miss() {
        let s = BoundingSphere::new(Vec3::zero(), 1.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(s.intersect_ray(&ray).is_none());
    }

    #[test]
    fn test_aabb_merge() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0));
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let merged = a.merge(&b);
        assert!(approx(merged.min.x, -1.0, 1e-9));
        assert!(approx(merged.max.x, 2.0, 1e-9));
    }

    #[test]
    fn test_sphere_merge() {
        let a = BoundingSphere::new(Vec3::new(-1.0, 0.0, 0.0), 1.0);
        let b = BoundingSphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let merged = a.merge(&b);
        assert!(merged.contains_point(Vec3::new(-2.0, 0.0, 0.0)));
        assert!(merged.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_sphere_merge_contained() {
        let big = BoundingSphere::new(Vec3::zero(), 10.0);
        let small = BoundingSphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let merged = big.merge(&small);
        assert!(approx(merged.radius, 10.0, 1e-9));
    }

    #[test]
    fn test_aabb_transform() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let mat = Mat4::translation(Vec3::new(5.0, 0.0, 0.0));
        let transformed = aabb.transform(&mat);
        assert!(approx(transformed.center().x, 5.0, 1e-9));
    }

    #[test]
    fn test_aabb_transform_scale() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let mat = Mat4::scaling(Vec3::new(2.0, 2.0, 2.0));
        let transformed = aabb.transform(&mat);
        assert!(approx(transformed.extents().x, 4.0, 1e-9));
    }

    #[test]
    fn test_sphere_transform() {
        let s = BoundingSphere::new(Vec3::zero(), 1.0);
        let mat = Mat4::translation(Vec3::new(3.0, 0.0, 0.0));
        let ts = s.transform(&mat);
        assert!(approx(ts.center.x, 3.0, 1e-9));
        assert!(approx(ts.radius, 1.0, 1e-9));
    }

    #[test]
    fn test_obb_from_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let obb = Obb::from_aabb(&aabb);
        assert!(approx(obb.half_extents.x, 1.0, 1e-9));
        assert!(approx(obb.half_extents.y, 2.0, 1e-9));
    }

    #[test]
    fn test_obb_contains_point() {
        let obb = Obb::from_aabb(&Aabb::new(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        ));
        assert!(obb.contains_point(Vec3::zero()));
        assert!(!obb.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_obb_obb_intersection() {
        let a = Obb::from_aabb(&Aabb::new(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        ));
        let b = Obb::from_aabb(&Aabb::new(
            Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.0, 2.0, 2.0),
        ));
        assert!(a.intersects_obb(&b));
    }

    #[test]
    fn test_obb_obb_no_intersection() {
        let a = Obb::from_aabb(&Aabb::new(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        ));
        let b = Obb::from_aabb(&Aabb::new(
            Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0),
        ));
        assert!(!a.intersects_obb(&b));
    }

    #[test]
    fn test_obb_corners_count() {
        let obb = Obb::from_aabb(&Aabb::new(
            Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        ));
        assert_eq!(obb.corners().len(), 8);
    }

    #[test]
    fn test_obb_to_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let obb = Obb::from_aabb(&aabb);
        let back = obb.to_aabb();
        assert!(approx(back.min.x, -1.0, 1e-9));
        assert!(approx(back.max.x, 1.0, 1e-9));
    }

    #[test]
    fn test_empty_sphere_from_points() {
        let s = BoundingSphere::from_points(&[]);
        assert!(approx(s.radius, 0.0, 1e-9));
    }

    #[test]
    fn test_single_point_sphere() {
        let s = BoundingSphere::from_points(&[Vec3::new(1.0, 2.0, 3.0)]);
        assert!(approx(s.center.x, 1.0, 1e-9));
        assert!(approx(s.radius, 0.0, 1e-9));
    }
}
