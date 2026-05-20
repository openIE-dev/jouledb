//! Ray casting engine — 2D ray cast against segments/circles/AABBs/polygons,
//! closest hit, all hits sorted by distance, ray marching, field-of-view cone,
//! shadow casting, line-of-sight.

use std::f64::consts::PI;

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn cross(self, o: Self) -> f64 { self.x * o.y - self.y * o.x }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn negate(self) -> Self { Self { x: -self.x, y: -self.y } }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
    pub fn angle(self) -> f64 { self.y.atan2(self.x) }

    pub fn from_angle(angle: f64) -> Self {
        Self { x: angle.cos(), y: angle.sin() }
    }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Ray ──────────────────────────────────────────────────────

/// A ray defined by origin and direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec2,
    pub direction: Vec2,
}

impl Ray {
    pub fn new(origin: Vec2, direction: Vec2) -> Self {
        Self { origin, direction: direction.normalized() }
    }

    pub fn from_points(from: Vec2, to: Vec2) -> Self {
        Self::new(from, to.sub(from))
    }

    pub fn point_at(&self, t: f64) -> Vec2 {
        self.origin.add(self.direction.scale(t))
    }
}

// ── Hit Result ───────────────────────────────────────────────

/// Result of a ray intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along ray.
    pub t: f64,
    /// Hit point in world space.
    pub point: Vec2,
    /// Surface normal at hit point (pointing away from surface).
    pub normal: Vec2,
    /// Index of the shape that was hit (in multi-shape queries).
    pub shape_index: usize,
}

impl RayHit {
    pub fn new(t: f64, point: Vec2, normal: Vec2, shape_index: usize) -> Self {
        Self { t, point, normal, shape_index }
    }
}

// ── Shapes ───────────────────────────────────────────────────

/// A line segment from `a` to `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub a: Vec2,
    pub b: Vec2,
}

impl Segment {
    pub fn new(a: Vec2, b: Vec2) -> Self { Self { a, b } }

    pub fn length(&self) -> f64 { self.b.sub(self.a).length() }

    pub fn direction(&self) -> Vec2 { self.b.sub(self.a).normalized() }

    pub fn normal(&self) -> Vec2 { self.direction().perpendicular() }
}

/// A circle shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f64,
}

impl Circle {
    pub fn new(cx: f64, cy: f64, r: f64) -> Self {
        Self { center: Vec2::new(cx, cy), radius: r }
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min: Vec2::new(min_x, min_y), max: Vec2::new(max_x, max_y) }
    }
}

/// A convex polygon.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    pub vertices: Vec<Vec2>,
}

impl Polygon {
    pub fn new(vertices: Vec<Vec2>) -> Self { Self { vertices } }

    pub fn segments(&self) -> Vec<Segment> {
        let n = self.vertices.len();
        (0..n).map(|i| {
            Segment::new(self.vertices[i], self.vertices[(i + 1) % n])
        }).collect()
    }
}

// ── Ray vs Segment ───────────────────────────────────────────

/// Cast a ray against a line segment.
pub fn ray_vs_segment(ray: &Ray, seg: &Segment) -> Option<RayHit> {
    let d = seg.b.sub(seg.a);
    let denom = ray.direction.cross(d);
    if denom.abs() < 1e-12 {
        return None; // Parallel
    }

    let t = seg.a.sub(ray.origin).cross(d) / denom;
    let u = seg.a.sub(ray.origin).cross(ray.direction) / denom;

    if t < 0.0 || u < 0.0 || u > 1.0 {
        return None;
    }

    let point = ray.point_at(t);
    let normal = d.perpendicular().normalized();
    // Ensure normal faces toward ray origin
    let facing_normal = if normal.dot(ray.direction) > 0.0 {
        normal.negate()
    } else {
        normal
    };

    Some(RayHit::new(t, point, facing_normal, 0))
}

// ── Ray vs Circle ────────────────────────────────────────────

/// Cast a ray against a circle.
pub fn ray_vs_circle(ray: &Ray, circle: &Circle) -> Option<RayHit> {
    let oc = ray.origin.sub(circle.center);
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - circle.radius * circle.radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);
    let t = if t1 >= 0.0 { t1 } else if t2 >= 0.0 { t2 } else { return None };
    let point = ray.point_at(t);
    let normal = point.sub(circle.center).normalized();
    Some(RayHit::new(t, point, normal, 0))
}

// ── Ray vs AABB ──────────────────────────────────────────────

/// Cast a ray against an axis-aligned bounding box.
pub fn ray_vs_aabb(ray: &Ray, aabb: &AABB) -> Option<RayHit> {
    let inv_dx = if ray.direction.x.abs() < 1e-12 { f64::INFINITY.copysign(ray.direction.x) } else { 1.0 / ray.direction.x };
    let inv_dy = if ray.direction.y.abs() < 1e-12 { f64::INFINITY.copysign(ray.direction.y) } else { 1.0 / ray.direction.y };

    let mut t_min_x = (aabb.min.x - ray.origin.x) * inv_dx;
    let mut t_max_x = (aabb.max.x - ray.origin.x) * inv_dx;
    if t_min_x > t_max_x { std::mem::swap(&mut t_min_x, &mut t_max_x); }

    let mut t_min_y = (aabb.min.y - ray.origin.y) * inv_dy;
    let mut t_max_y = (aabb.max.y - ray.origin.y) * inv_dy;
    if t_min_y > t_max_y { std::mem::swap(&mut t_min_y, &mut t_max_y); }

    if t_min_x > t_max_y || t_min_y > t_max_x {
        return None;
    }

    let t_enter = t_min_x.max(t_min_y);
    let t_exit = t_max_x.min(t_max_y);
    if t_exit < 0.0 { return None; }

    let t = if t_enter >= 0.0 { t_enter } else { t_exit };
    let point = ray.point_at(t);

    let normal = if (t - t_min_x).abs() < 1e-10 {
        Vec2::new(if ray.direction.x > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        Vec2::new(0.0, if ray.direction.y > 0.0 { -1.0 } else { 1.0 })
    };

    Some(RayHit::new(t, point, normal, 0))
}

// ── Ray vs Polygon ───────────────────────────────────────────

/// Cast a ray against a convex polygon.
pub fn ray_vs_polygon(ray: &Ray, polygon: &Polygon) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    for seg in &polygon.segments() {
        if let Some(hit) = ray_vs_segment(ray, seg) {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }
    best
}

// ── Multi-Shape Query ────────────────────────────────────────

/// A scene obstacle for ray casting.
#[derive(Debug, Clone)]
pub enum Obstacle {
    Seg(Segment),
    Circ(Circle),
    Rect(AABB),
    Poly(Polygon),
}

/// Cast a ray against all obstacles and return the closest hit.
pub fn closest_hit(ray: &Ray, obstacles: &[Obstacle]) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    for (i, obs) in obstacles.iter().enumerate() {
        let hit = match obs {
            Obstacle::Seg(s) => ray_vs_segment(ray, s),
            Obstacle::Circ(c) => ray_vs_circle(ray, c),
            Obstacle::Rect(a) => ray_vs_aabb(ray, a),
            Obstacle::Poly(p) => ray_vs_polygon(ray, p),
        };
        if let Some(mut h) = hit {
            h.shape_index = i;
            if best.is_none() || h.t < best.as_ref().unwrap().t {
                best = Some(h);
            }
        }
    }
    best
}

/// Cast a ray against all obstacles and return all hits, sorted by distance.
pub fn all_hits(ray: &Ray, obstacles: &[Obstacle]) -> Vec<RayHit> {
    let mut hits = Vec::new();
    for (i, obs) in obstacles.iter().enumerate() {
        let hit = match obs {
            Obstacle::Seg(s) => ray_vs_segment(ray, s),
            Obstacle::Circ(c) => ray_vs_circle(ray, c),
            Obstacle::Rect(a) => ray_vs_aabb(ray, a),
            Obstacle::Poly(p) => ray_vs_polygon(ray, p),
        };
        if let Some(mut h) = hit {
            h.shape_index = i;
            hits.push(h);
        }
    }
    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

// ── Ray Marching ─────────────────────────────────────────────

/// Signed distance function type.
pub type SdfFn = dyn Fn(Vec2) -> f64;

/// Ray march using a signed distance function (SDF).
///
/// Returns the hit point if the ray reaches a surface (distance < epsilon),
/// or None if max_steps or max_dist is exceeded.
pub fn ray_march(
    ray: &Ray,
    sdf: &SdfFn,
    max_steps: u32,
    max_dist: f64,
    epsilon: f64,
) -> Option<Vec2> {
    let mut t = 0.0;
    for _ in 0..max_steps {
        let p = ray.point_at(t);
        let d = sdf(p);
        if d < epsilon {
            return Some(p);
        }
        t += d;
        if t > max_dist {
            return None;
        }
    }
    None
}

// ── Field of View Cone ───────────────────────────────────────

/// A field-of-view cone for visibility testing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovCone {
    /// Origin of the cone (observer position).
    pub origin: Vec2,
    /// Forward direction of the cone.
    pub direction: Vec2,
    /// Half-angle of the cone in radians.
    pub half_angle: f64,
    /// Maximum range of the cone.
    pub range: f64,
}

impl FovCone {
    pub fn new(origin: Vec2, direction: Vec2, fov_angle: f64, range: f64) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            half_angle: fov_angle * 0.5,
            range,
        }
    }

    /// Check if a point is within the FOV cone (ignoring obstacles).
    pub fn contains(&self, point: Vec2) -> bool {
        let to_point = point.sub(self.origin);
        let dist = to_point.length();
        if dist > self.range || dist < 1e-12 {
            return dist < 1e-12;
        }
        let normalized = to_point.normalized();
        // Must be in the forward hemisphere
        if self.direction.dot(normalized) <= 0.0 {
            return false;
        }
        let cross = self.direction.cross(normalized).abs();
        let angle = cross.min(1.0).asin();
        angle <= self.half_angle
    }

    /// Cast rays to determine visibility within the cone.
    /// Returns the visible boundary polygon (approximation using `ray_count` rays).
    pub fn cast_visibility(
        &self,
        obstacles: &[Obstacle],
        ray_count: u32,
    ) -> Vec<Vec2> {
        let base_angle = self.direction.angle();
        let start = base_angle - self.half_angle;
        let end = base_angle + self.half_angle;
        let step = (end - start) / (ray_count.max(1) - 1).max(1) as f64;

        let mut points = vec![self.origin];
        for i in 0..ray_count {
            let angle = start + step * i as f64;
            let dir = Vec2::from_angle(angle);
            let ray = Ray::new(self.origin, dir);
            let t = match closest_hit(&ray, obstacles) {
                Some(hit) if hit.t <= self.range => hit.t,
                _ => self.range,
            };
            points.push(ray.point_at(t));
        }
        points
    }
}

// ── Shadow Casting ───────────────────────────────────────────

/// Cast shadows from a point light source.
///
/// Given a light position and obstacles, returns a visibility polygon
/// (the lit area). Uses `ray_count` evenly-spaced rays + extra rays
/// toward obstacle vertices.
pub fn shadow_cast(
    light: Vec2,
    obstacles: &[Obstacle],
    range: f64,
    ray_count: u32,
) -> Vec<Vec2> {
    // Collect all obstacle vertex angles
    let mut angles: Vec<f64> = Vec::new();

    for obs in obstacles {
        let verts = match obs {
            Obstacle::Seg(s) => vec![s.a, s.b],
            Obstacle::Circ(c) => {
                // Sample circle edge points
                let mut pts = Vec::new();
                for i in 0..8 {
                    let a = (i as f64 / 8.0) * 2.0 * PI;
                    pts.push(c.center.add(Vec2::from_angle(a).scale(c.radius)));
                }
                pts
            }
            Obstacle::Rect(r) => vec![
                r.min,
                Vec2::new(r.max.x, r.min.y),
                r.max,
                Vec2::new(r.min.x, r.max.y),
            ],
            Obstacle::Poly(p) => p.vertices.clone(),
        };

        for v in verts {
            let a = v.sub(light).angle();
            angles.push(a);
            // Slight offsets to peek around corners
            angles.push(a - 0.001);
            angles.push(a + 0.001);
        }
    }

    // Add uniform sweep angles
    let base_count = ray_count.max(8);
    for i in 0..base_count {
        let a = (i as f64 / base_count as f64) * 2.0 * PI - PI;
        angles.push(a);
    }

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    angles.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    let mut boundary = Vec::new();
    for angle in &angles {
        let dir = Vec2::from_angle(*angle);
        let ray = Ray::new(light, dir);
        let t = match closest_hit(&ray, obstacles) {
            Some(hit) if hit.t <= range => hit.t,
            _ => range,
        };
        boundary.push(ray.point_at(t));
    }

    boundary
}

// ── Line of Sight ────────────────────────────────────────────

/// Check line-of-sight between two points (no obstacles blocking).
pub fn line_of_sight(from: Vec2, to: Vec2, obstacles: &[Obstacle]) -> bool {
    let diff = to.sub(from);
    let dist = diff.length();
    if dist < 1e-12 { return true; }

    let ray = Ray::new(from, diff);
    match closest_hit(&ray, obstacles) {
        Some(hit) => hit.t > dist - 1e-6,
        None => true,
    }
}

/// Check line-of-sight with a maximum range.
pub fn line_of_sight_range(from: Vec2, to: Vec2, obstacles: &[Obstacle], max_range: f64) -> bool {
    let dist = from.distance(to);
    if dist > max_range { return false; }
    line_of_sight(from, to, obstacles)
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 0.05 }

    #[test]
    fn ray_segment_hit() {
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let seg = Segment::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0));
        let hit = ray_vs_segment(&ray, &seg).unwrap();
        assert!(approx(hit.t, 5.0));
        assert!(approx(hit.point.x, 5.0));
    }

    #[test]
    fn ray_segment_miss() {
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let seg = Segment::new(Vec2::new(5.0, 10.0), Vec2::new(5.0, 20.0));
        assert!(ray_vs_segment(&ray, &seg).is_none());
    }

    #[test]
    fn ray_segment_behind() {
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let seg = Segment::new(Vec2::new(-5.0, -5.0), Vec2::new(-5.0, 5.0));
        assert!(ray_vs_segment(&ray, &seg).is_none());
    }

    #[test]
    fn ray_circle_hit() {
        let ray = Ray::new(Vec2::new(-10.0, 0.0), Vec2::new(1.0, 0.0));
        let circle = Circle::new(0.0, 0.0, 5.0);
        let hit = ray_vs_circle(&ray, &circle).unwrap();
        assert!(approx(hit.point.x, -5.0));
        assert!(approx(hit.normal.x, -1.0));
    }

    #[test]
    fn ray_circle_miss() {
        let ray = Ray::new(Vec2::new(-10.0, 10.0), Vec2::new(1.0, 0.0));
        let circle = Circle::new(0.0, 0.0, 5.0);
        assert!(ray_vs_circle(&ray, &circle).is_none());
    }

    #[test]
    fn ray_aabb_hit() {
        let ray = Ray::new(Vec2::new(-5.0, 5.0), Vec2::new(1.0, 0.0));
        let aabb = AABB::new(0.0, 0.0, 10.0, 10.0);
        let hit = ray_vs_aabb(&ray, &aabb).unwrap();
        assert!(approx(hit.point.x, 0.0));
        assert!(approx(hit.normal.x, -1.0));
    }

    #[test]
    fn ray_aabb_miss() {
        let ray = Ray::new(Vec2::new(-5.0, 15.0), Vec2::new(1.0, 0.0));
        let aabb = AABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(ray_vs_aabb(&ray, &aabb).is_none());
    }

    #[test]
    fn ray_polygon_hit() {
        let poly = Polygon::new(vec![
            Vec2::new(5.0, -5.0), Vec2::new(10.0, 0.0),
            Vec2::new(5.0, 5.0), Vec2::new(0.0, 0.0),
        ]);
        let ray = Ray::new(Vec2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let hit = ray_vs_polygon(&ray, &poly).unwrap();
        assert!(hit.t > 0.0);
    }

    #[test]
    fn closest_hit_test() {
        let obstacles = vec![
            Obstacle::Seg(Segment::new(Vec2::new(10.0, -5.0), Vec2::new(10.0, 5.0))),
            Obstacle::Seg(Segment::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0))),
        ];
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hit = closest_hit(&ray, &obstacles).unwrap();
        // Closer segment is at x=5
        assert!(approx(hit.t, 5.0));
        assert_eq!(hit.shape_index, 1);
    }

    #[test]
    fn all_hits_sorted() {
        let obstacles = vec![
            Obstacle::Seg(Segment::new(Vec2::new(10.0, -5.0), Vec2::new(10.0, 5.0))),
            Obstacle::Seg(Segment::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0))),
        ];
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = all_hits(&ray, &obstacles);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].t < hits[1].t);
    }

    #[test]
    fn ray_march_circle_sdf() {
        let sdf = |p: Vec2| -> f64 {
            p.length() - 5.0 // Circle of radius 5 at origin
        };
        let ray = Ray::new(Vec2::new(-20.0, 0.0), Vec2::new(1.0, 0.0));
        let hit = ray_march(&ray, &sdf, 100, 50.0, 0.01).unwrap();
        assert!(approx(hit.x, -5.0));
    }

    #[test]
    fn ray_march_miss() {
        let sdf = |p: Vec2| -> f64 {
            p.length() - 5.0
        };
        let ray = Ray::new(Vec2::new(-20.0, 20.0), Vec2::new(1.0, 0.0));
        assert!(ray_march(&ray, &sdf, 50, 100.0, 0.01).is_none());
    }

    #[test]
    fn fov_cone_contains() {
        let fov = FovCone::new(
            Vec2::zero(),
            Vec2::new(1.0, 0.0),
            PI * 0.5, // 90 degrees total
            100.0,
        );
        // Directly ahead should be visible
        assert!(fov.contains(Vec2::new(10.0, 0.0)));
        // Slightly off-axis within 45 degrees
        assert!(fov.contains(Vec2::new(10.0, 3.0)));
        // Behind should not be visible
        assert!(!fov.contains(Vec2::new(-10.0, 0.0)));
        // Beyond range
        assert!(!fov.contains(Vec2::new(200.0, 0.0)));
    }

    #[test]
    fn fov_cast_visibility() {
        let fov = FovCone::new(
            Vec2::zero(),
            Vec2::new(1.0, 0.0),
            PI * 0.25, // 45 degrees total
            50.0,
        );
        let obstacles = vec![
            Obstacle::Seg(Segment::new(Vec2::new(20.0, -10.0), Vec2::new(20.0, 10.0))),
        ];
        let vis = fov.cast_visibility(&obstacles, 16);
        // Should include origin as first point
        assert!(approx(vis[0].x, 0.0));
        assert!(approx(vis[0].y, 0.0));
        // Boundary points should not exceed wall at x=20
        for p in &vis[1..] {
            assert!(p.x <= 20.1);
        }
    }

    #[test]
    fn shadow_cast_basic() {
        let light = Vec2::new(5.0, 5.0);
        let obstacles = vec![
            Obstacle::Seg(Segment::new(Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0))),
        ];
        let boundary = shadow_cast(light, &obstacles, 50.0, 32);
        assert!(!boundary.is_empty());
        // All boundary points should be within range of light
        for p in &boundary {
            assert!(light.distance(*p) <= 50.1);
        }
    }

    #[test]
    fn line_of_sight_clear() {
        let obstacles: Vec<Obstacle> = vec![];
        assert!(line_of_sight(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles,
        ));
    }

    #[test]
    fn line_of_sight_blocked() {
        let obstacles = vec![
            Obstacle::Seg(Segment::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0))),
        ];
        assert!(!line_of_sight(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles,
        ));
    }

    #[test]
    fn line_of_sight_range_exceeded() {
        let obstacles: Vec<Obstacle> = vec![];
        assert!(!line_of_sight_range(
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            &obstacles,
            50.0,
        ));
    }

    #[test]
    fn line_of_sight_range_ok() {
        let obstacles: Vec<Obstacle> = vec![];
        assert!(line_of_sight_range(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles,
            50.0,
        ));
    }

    #[test]
    fn ray_from_points() {
        let ray = Ray::from_points(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        assert!(approx(ray.direction.x, 1.0));
        assert!(approx(ray.direction.y, 0.0));
    }

    #[test]
    fn segment_properties() {
        let seg = Segment::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        assert!(approx(seg.length(), 10.0));
        assert!(approx(seg.direction().x, 1.0));
    }

    #[test]
    fn shadow_cast_with_circle() {
        let light = Vec2::new(0.0, 0.0);
        let obstacles = vec![
            Obstacle::Circ(Circle::new(10.0, 0.0, 3.0)),
        ];
        let boundary = shadow_cast(light, &obstacles, 50.0, 32);
        assert!(!boundary.is_empty());
    }

    #[test]
    fn all_hits_empty_scene() {
        let ray = Ray::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let hits = all_hits(&ray, &[]);
        assert!(hits.is_empty());
    }
}
