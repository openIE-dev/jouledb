//! 2D collision detection — AABB, circle, polygon (SAT), point-in-shape,
//! sweep test (continuous collision), collision manifold with contact points.
//!
//! Pure Rust replacement for matter.js collision, p2.js, planck.js narrow/broad phase.

use std::collections::HashSet;

// ── Vec2 ─────────────────────────────────────────────────────

/// 2D vector / point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y
    }

    pub fn cross(self, o: Self) -> f64 {
        self.x * o.y - self.y * o.x
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            return Self::zero();
        }
        Self { x: self.x / len, y: self.y / len }
    }

    pub fn perpendicular(self) -> Self {
        Self { x: -self.y, y: self.x }
    }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }

    pub fn negate(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self.add(other.sub(self).scale(t))
    }

    pub fn distance(self, other: Self) -> f64 {
        self.sub(other).length()
    }

    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Shape2D ──────────────────────────────────────────────────

/// 2D collision shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape2D {
    Circle { center: Vec2, radius: f64 },
    AABB { min: Vec2, max: Vec2 },
    OBB { center: Vec2, half_extents: Vec2, angle: f64 },
    Polygon { vertices: Vec<Vec2> },
}

impl Shape2D {
    pub fn circle(cx: f64, cy: f64, radius: f64) -> Self {
        Self::Circle { center: Vec2::new(cx, cy), radius }
    }

    pub fn aabb(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self::AABB {
            min: Vec2::new(min_x, min_y),
            max: Vec2::new(max_x, max_y),
        }
    }

    pub fn obb(cx: f64, cy: f64, hw: f64, hh: f64, angle: f64) -> Self {
        Self::OBB {
            center: Vec2::new(cx, cy),
            half_extents: Vec2::new(hw, hh),
            angle,
        }
    }

    pub fn polygon(vertices: Vec<Vec2>) -> Self {
        Self::Polygon { vertices }
    }

    /// Compute the axis-aligned bounding box.
    pub fn bounding_box(&self) -> (Vec2, Vec2) {
        match self {
            Shape2D::Circle { center, radius } => (
                Vec2::new(center.x - radius, center.y - radius),
                Vec2::new(center.x + radius, center.y + radius),
            ),
            Shape2D::AABB { min, max } => (*min, *max),
            Shape2D::OBB { center, half_extents, angle } => {
                let cos_a = angle.cos().abs();
                let sin_a = angle.sin().abs();
                let hx = half_extents.x * cos_a + half_extents.y * sin_a;
                let hy = half_extents.x * sin_a + half_extents.y * cos_a;
                (
                    Vec2::new(center.x - hx, center.y - hy),
                    Vec2::new(center.x + hx, center.y + hy),
                )
            }
            Shape2D::Polygon { vertices } => {
                let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
                let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
                for v in vertices {
                    lo.x = lo.x.min(v.x);
                    lo.y = lo.y.min(v.y);
                    hi.x = hi.x.max(v.x);
                    hi.y = hi.y.max(v.y);
                }
                (lo, hi)
            }
        }
    }

    /// Translate the shape by a vector.
    pub fn translate(&self, offset: Vec2) -> Self {
        match self {
            Shape2D::Circle { center, radius } => Shape2D::Circle {
                center: center.add(offset),
                radius: *radius,
            },
            Shape2D::AABB { min, max } => Shape2D::AABB {
                min: min.add(offset),
                max: max.add(offset),
            },
            Shape2D::OBB { center, half_extents, angle } => Shape2D::OBB {
                center: center.add(offset),
                half_extents: *half_extents,
                angle: *angle,
            },
            Shape2D::Polygon { vertices } => Shape2D::Polygon {
                vertices: vertices.iter().map(|v| v.add(offset)).collect(),
            },
        }
    }
}

// ── Contact ──────────────────────────────────────────────────

/// Single contact point from a collision test.
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    pub point: Vec2,
    pub normal: Vec2,
    pub penetration: f64,
}

// ── Collision Manifold ───────────────────────────────────────

/// Collision manifold containing all contact points, normal, and depth.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionManifold {
    /// Contact points on the collision boundary.
    pub contacts: Vec<Vec2>,
    /// Collision normal (from shape A toward shape B).
    pub normal: Vec2,
    /// Penetration depth (positive when overlapping).
    pub depth: f64,
}

impl CollisionManifold {
    pub fn new(contacts: Vec<Vec2>, normal: Vec2, depth: f64) -> Self {
        Self { contacts, normal, depth }
    }

    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }
}

// ── Sweep Result ─────────────────────────────────────────────

/// Result of a continuous collision (sweep) test.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepResult {
    /// Time of impact in \[0, 1\] range.
    pub toi: f64,
    /// Contact normal at time of impact.
    pub normal: Vec2,
    /// Contact point at time of impact.
    pub point: Vec2,
}

// ── Ray ──────────────────────────────────────────────────────

/// A ray for intersection tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec2,
    pub direction: Vec2,
}

impl Ray {
    pub fn new(origin: Vec2, direction: Vec2) -> Self {
        Self { origin, direction: direction.normalized() }
    }

    pub fn point_at(&self, t: f64) -> Vec2 {
        self.origin.add(self.direction.scale(t))
    }
}

/// Ray intersection result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub t: f64,
    pub point: Vec2,
    pub normal: Vec2,
}

// ── Collision Tests ──────────────────────────────────────────

/// Test collision between two shapes. Returns contact info if colliding.
pub fn test_collision(a: &Shape2D, b: &Shape2D) -> Option<Contact> {
    match (a, b) {
        (Shape2D::Circle { .. }, Shape2D::Circle { .. }) => circle_vs_circle(a, b),
        (Shape2D::AABB { .. }, Shape2D::AABB { .. }) => aabb_vs_aabb(a, b),
        (Shape2D::Circle { .. }, Shape2D::AABB { .. }) => circle_vs_aabb(a, b),
        (Shape2D::AABB { .. }, Shape2D::Circle { .. }) => {
            circle_vs_aabb(b, a).map(|c| Contact {
                normal: c.normal.negate(),
                ..c
            })
        }
        (Shape2D::Polygon { .. }, Shape2D::Polygon { .. }) => sat_polygons(a, b),
        (Shape2D::Circle { .. }, Shape2D::Polygon { .. }) => circle_vs_polygon(a, b),
        (Shape2D::Polygon { .. }, Shape2D::Circle { .. }) => {
            circle_vs_polygon(b, a).map(|c| Contact {
                normal: c.normal.negate(),
                ..c
            })
        }
        (Shape2D::AABB { min, max }, other) => {
            let poly = aabb_to_polygon(*min, *max);
            let ps = Shape2D::Polygon { vertices: poly };
            test_collision(&ps, other)
        }
        (other, Shape2D::AABB { min, max }) => {
            let poly = aabb_to_polygon(*min, *max);
            let ps = Shape2D::Polygon { vertices: poly };
            test_collision(other, &ps)
        }
        _ => {
            let pa = shape_to_polygon(a);
            let pb = shape_to_polygon(b);
            if let (Some(va), Some(vb)) = (pa, pb) {
                let sa = Shape2D::Polygon { vertices: va };
                let sb = Shape2D::Polygon { vertices: vb };
                sat_polygons(&sa, &sb)
            } else {
                None
            }
        }
    }
}

/// Compute a full collision manifold between two shapes.
pub fn compute_manifold(a: &Shape2D, b: &Shape2D) -> Option<CollisionManifold> {
    match (a, b) {
        (Shape2D::Circle { center: ca, radius: ra },
         Shape2D::Circle { center: cb, radius: rb }) => {
            let diff = cb.sub(*ca);
            let dist = diff.length();
            let sum_r = ra + rb;
            if dist >= sum_r {
                return None;
            }
            let normal = if dist < 1e-12 {
                Vec2::new(1.0, 0.0)
            } else {
                diff.normalized()
            };
            let depth = sum_r - dist;
            let contact = ca.add(normal.scale(*ra - depth * 0.5));
            Some(CollisionManifold::new(vec![contact], normal, depth))
        }
        (Shape2D::AABB { min: a_min, max: a_max },
         Shape2D::AABB { min: b_min, max: b_max }) => {
            let overlap_x = a_max.x.min(b_max.x) - a_min.x.max(b_min.x);
            let overlap_y = a_max.y.min(b_max.y) - a_min.y.max(b_min.y);
            if overlap_x <= 0.0 || overlap_y <= 0.0 {
                return None;
            }
            let (normal, depth) = if overlap_x < overlap_y {
                let ca_x = (a_min.x + a_max.x) * 0.5;
                let cb_x = (b_min.x + b_max.x) * 0.5;
                let nx = if cb_x > ca_x { 1.0 } else { -1.0 };
                (Vec2::new(nx, 0.0), overlap_x)
            } else {
                let ca_y = (a_min.y + a_max.y) * 0.5;
                let cb_y = (b_min.y + b_max.y) * 0.5;
                let ny = if cb_y > ca_y { 1.0 } else { -1.0 };
                (Vec2::new(0.0, ny), overlap_y)
            };
            // Two contact points at the overlap edge corners
            let ix_min = a_min.x.max(b_min.x);
            let ix_max = a_max.x.min(b_max.x);
            let iy_min = a_min.y.max(b_min.y);
            let iy_max = a_max.y.min(b_max.y);
            let contacts = if overlap_x < overlap_y {
                vec![
                    Vec2::new(ix_min + (ix_max - ix_min) * 0.5, iy_min),
                    Vec2::new(ix_min + (ix_max - ix_min) * 0.5, iy_max),
                ]
            } else {
                vec![
                    Vec2::new(ix_min, iy_min + (iy_max - iy_min) * 0.5),
                    Vec2::new(ix_max, iy_min + (iy_max - iy_min) * 0.5),
                ]
            };
            Some(CollisionManifold::new(contacts, normal, depth))
        }
        _ => {
            // Fall back to single-contact test
            test_collision(a, b).map(|c| {
                CollisionManifold::new(vec![c.point], c.normal, c.penetration)
            })
        }
    }
}

fn circle_vs_circle(a: &Shape2D, b: &Shape2D) -> Option<Contact> {
    if let (
        Shape2D::Circle { center: ca, radius: ra },
        Shape2D::Circle { center: cb, radius: rb },
    ) = (a, b) {
        let diff = cb.sub(*ca);
        let dist = diff.length();
        let sum_r = ra + rb;
        if dist >= sum_r {
            return None;
        }
        let normal = if dist < 1e-12 { Vec2::new(1.0, 0.0) } else { diff.normalized() };
        let pen = sum_r - dist;
        let point = ca.add(normal.scale(*ra - pen * 0.5));
        Some(Contact { point, normal, penetration: pen })
    } else {
        None
    }
}

fn aabb_vs_aabb(a: &Shape2D, b: &Shape2D) -> Option<Contact> {
    if let (
        Shape2D::AABB { min: a_min, max: a_max },
        Shape2D::AABB { min: b_min, max: b_max },
    ) = (a, b) {
        let overlap_x = a_max.x.min(b_max.x) - a_min.x.max(b_min.x);
        let overlap_y = a_max.y.min(b_max.y) - a_min.y.max(b_min.y);
        if overlap_x <= 0.0 || overlap_y <= 0.0 {
            return None;
        }
        let (normal, penetration) = if overlap_x < overlap_y {
            let ca_x = (a_min.x + a_max.x) * 0.5;
            let cb_x = (b_min.x + b_max.x) * 0.5;
            let nx = if cb_x > ca_x { 1.0 } else { -1.0 };
            (Vec2::new(nx, 0.0), overlap_x)
        } else {
            let ca_y = (a_min.y + a_max.y) * 0.5;
            let cb_y = (b_min.y + b_max.y) * 0.5;
            let ny = if cb_y > ca_y { 1.0 } else { -1.0 };
            (Vec2::new(0.0, ny), overlap_y)
        };
        let point = Vec2::new(
            a_min.x.max(b_min.x) + overlap_x * 0.5,
            a_min.y.max(b_min.y) + overlap_y * 0.5,
        );
        Some(Contact { point, normal, penetration })
    } else {
        None
    }
}

fn circle_vs_aabb(circle: &Shape2D, aabb: &Shape2D) -> Option<Contact> {
    if let (
        Shape2D::Circle { center, radius },
        Shape2D::AABB { min, max },
    ) = (circle, aabb) {
        let closest = Vec2::new(
            center.x.clamp(min.x, max.x),
            center.y.clamp(min.y, max.y),
        );
        let diff = center.sub(closest);
        let dist_sq = diff.length_sq();
        if dist_sq >= radius * radius {
            return None;
        }
        let dist = dist_sq.sqrt();
        let normal = if dist < 1e-12 {
            let dl = center.x - min.x;
            let dr = max.x - center.x;
            let dt = center.y - min.y;
            let db = max.y - center.y;
            let m = dl.min(dr).min(dt).min(db);
            if (m - dl).abs() < 1e-12 { Vec2::new(-1.0, 0.0) }
            else if (m - dr).abs() < 1e-12 { Vec2::new(1.0, 0.0) }
            else if (m - dt).abs() < 1e-12 { Vec2::new(0.0, -1.0) }
            else { Vec2::new(0.0, 1.0) }
        } else {
            diff.normalized()
        };
        let pen = radius - dist;
        let point = center.sub(normal.scale(*radius));
        Some(Contact { point, normal, penetration: pen })
    } else {
        None
    }
}

fn circle_vs_polygon(circle: &Shape2D, polygon: &Shape2D) -> Option<Contact> {
    if let (
        Shape2D::Circle { center, radius },
        Shape2D::Polygon { vertices },
    ) = (circle, polygon) {
        if vertices.len() < 3 {
            return None;
        }
        let mut min_dist = f64::INFINITY;
        let mut closest = *center;
        let mut best_normal = Vec2::zero();

        for i in 0..vertices.len() {
            let j = (i + 1) % vertices.len();
            let a = vertices[i];
            let b = vertices[j];
            let edge = b.sub(a);
            let to_c = center.sub(a);
            let t = (to_c.dot(edge) / edge.length_sq()).clamp(0.0, 1.0);
            let pt = a.add(edge.scale(t));
            let d = center.sub(pt).length();
            if d < min_dist {
                min_dist = d;
                closest = pt;
                best_normal = edge.perpendicular().normalized();
                if best_normal.dot(center.sub(a)) < 0.0 {
                    best_normal = best_normal.negate();
                }
            }
        }
        if min_dist >= *radius {
            return None;
        }
        Some(Contact {
            point: closest,
            normal: best_normal,
            penetration: radius - min_dist,
        })
    } else {
        None
    }
}

/// SAT collision test for convex polygons.
fn sat_polygons(a: &Shape2D, b: &Shape2D) -> Option<Contact> {
    if let (
        Shape2D::Polygon { vertices: va },
        Shape2D::Polygon { vertices: vb },
    ) = (a, b) {
        if va.len() < 3 || vb.len() < 3 {
            return None;
        }
        let mut min_pen = f64::INFINITY;
        let mut best_normal = Vec2::zero();

        for verts in [va, vb] {
            for i in 0..verts.len() {
                let j = (i + 1) % verts.len();
                let edge = verts[j].sub(verts[i]);
                let axis = edge.perpendicular().normalized();
                let (a_lo, a_hi) = project_polygon(va, axis);
                let (b_lo, b_hi) = project_polygon(vb, axis);
                let overlap = a_hi.min(b_hi) - a_lo.max(b_lo);
                if overlap <= 0.0 {
                    return None;
                }
                if overlap < min_pen {
                    min_pen = overlap;
                    best_normal = axis;
                }
            }
        }

        let ca = polygon_center(va);
        let cb = polygon_center(vb);
        if best_normal.dot(cb.sub(ca)) < 0.0 {
            best_normal = best_normal.negate();
        }
        let point = ca.add(cb).scale(0.5);
        Some(Contact { point, normal: best_normal, penetration: min_pen })
    } else {
        None
    }
}

fn project_polygon(vertices: &[Vec2], axis: Vec2) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in vertices {
        let p = v.dot(axis);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

fn polygon_center(vertices: &[Vec2]) -> Vec2 {
    let n = vertices.len() as f64;
    let sx: f64 = vertices.iter().map(|v| v.x).sum();
    let sy: f64 = vertices.iter().map(|v| v.y).sum();
    Vec2::new(sx / n, sy / n)
}

fn aabb_to_polygon(min: Vec2, max: Vec2) -> Vec<Vec2> {
    vec![
        Vec2::new(min.x, min.y),
        Vec2::new(max.x, min.y),
        Vec2::new(max.x, max.y),
        Vec2::new(min.x, max.y),
    ]
}

fn shape_to_polygon(shape: &Shape2D) -> Option<Vec<Vec2>> {
    match shape {
        Shape2D::AABB { min, max } => Some(aabb_to_polygon(*min, *max)),
        Shape2D::OBB { center, half_extents, angle } => {
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let hx = half_extents.x;
            let hy = half_extents.y;
            let corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)];
            Some(corners.iter().map(|(lx, ly)| {
                Vec2::new(
                    center.x + lx * cos_a - ly * sin_a,
                    center.y + lx * sin_a + ly * cos_a,
                )
            }).collect())
        }
        Shape2D::Polygon { vertices } => Some(vertices.clone()),
        _ => None,
    }
}

// ── Point-in-shape ───────────────────────────────────────────

/// Test if a point is inside a shape.
pub fn point_in_shape(point: Vec2, shape: &Shape2D) -> bool {
    match shape {
        Shape2D::Circle { center, radius } => {
            point.sub(*center).length_sq() <= radius * radius
        }
        Shape2D::AABB { min, max } => {
            point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y
        }
        Shape2D::OBB { center, half_extents, angle } => {
            let d = point.sub(*center);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let lx = d.x * cos_a + d.y * sin_a;
            let ly = -d.x * sin_a + d.y * cos_a;
            lx.abs() <= half_extents.x && ly.abs() <= half_extents.y
        }
        Shape2D::Polygon { vertices } => point_in_polygon(point, vertices),
    }
}

/// Ray-casting point-in-polygon test.
pub fn point_in_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    let mut inside = false;
    let n = vertices.len();
    let mut j = n - 1;
    for i in 0..n {
        let vi = &vertices[i];
        let vj = &vertices[j];
        if ((vi.y > point.y) != (vj.y > point.y))
            && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ── Sweep Test (Continuous Collision) ────────────────────────

/// Linear sweep / continuous collision detection between two moving shapes.
///
/// `vel_a` and `vel_b` are the velocities of shapes a and b over the time step.
/// Returns the earliest time of impact in \[0, 1\] if they collide during the step.
pub fn sweep_test(
    a: &Shape2D,
    vel_a: Vec2,
    b: &Shape2D,
    vel_b: Vec2,
    steps: u32,
) -> Option<SweepResult> {
    let n = steps.max(8);
    let dt = 1.0 / n as f64;

    // Binary-search refinement after coarse pass
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;

    // Coarse pass: find the interval where collision begins
    let mut found_collision = false;
    for i in 0..n {
        let t = (i + 1) as f64 * dt;
        let offset_a = vel_a.scale(t);
        let offset_b = vel_b.scale(t);
        let sa = a.translate(offset_a);
        let sb = b.translate(offset_b);
        if test_collision(&sa, &sb).is_some() {
            hi = t;
            lo = t - dt;
            found_collision = true;
            break;
        }
    }

    if !found_collision {
        return None;
    }

    // Binary search refinement (8 iterations gives ~1/256 precision)
    for _ in 0..8 {
        let mid = (lo + hi) * 0.5;
        let offset_a = vel_a.scale(mid);
        let offset_b = vel_b.scale(mid);
        let sa = a.translate(offset_a);
        let sb = b.translate(offset_b);
        if test_collision(&sa, &sb).is_some() {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let toi = hi;
    let offset_a = vel_a.scale(toi);
    let offset_b = vel_b.scale(toi);
    let sa = a.translate(offset_a);
    let sb = b.translate(offset_b);
    test_collision(&sa, &sb).map(|c| SweepResult {
        toi,
        normal: c.normal,
        point: c.point,
    })
}

// ── Ray Intersection ─────────────────────────────────────────

/// Test ray intersection with a shape. Returns the nearest hit.
pub fn ray_intersect(ray: &Ray, shape: &Shape2D) -> Option<RayHit> {
    match shape {
        Shape2D::Circle { center, radius } => ray_vs_circle(ray, *center, *radius),
        Shape2D::AABB { min, max } => ray_vs_aabb(ray, *min, *max),
        Shape2D::Polygon { vertices } => ray_vs_polygon(ray, vertices),
        Shape2D::OBB { .. } => {
            if let Some(verts) = shape_to_polygon(shape) {
                ray_vs_polygon(ray, &verts)
            } else {
                None
            }
        }
    }
}

fn ray_vs_circle(ray: &Ray, center: Vec2, radius: f64) -> Option<RayHit> {
    let oc = ray.origin.sub(center);
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);
    let t = if t1 >= 0.0 { t1 } else if t2 >= 0.0 { t2 } else { return None };
    let point = ray.point_at(t);
    let normal = point.sub(center).normalized();
    Some(RayHit { t, point, normal })
}

fn ray_vs_aabb(ray: &Ray, min: Vec2, max: Vec2) -> Option<RayHit> {
    let inv_dx = if ray.direction.x.abs() < 1e-12 { f64::INFINITY } else { 1.0 / ray.direction.x };
    let inv_dy = if ray.direction.y.abs() < 1e-12 { f64::INFINITY } else { 1.0 / ray.direction.y };

    let mut t_min_x = (min.x - ray.origin.x) * inv_dx;
    let mut t_max_x = (max.x - ray.origin.x) * inv_dx;
    if t_min_x > t_max_x { std::mem::swap(&mut t_min_x, &mut t_max_x); }

    let mut t_min_y = (min.y - ray.origin.y) * inv_dy;
    let mut t_max_y = (max.y - ray.origin.y) * inv_dy;
    if t_min_y > t_max_y { std::mem::swap(&mut t_min_y, &mut t_max_y); }

    if t_min_x > t_max_y || t_min_y > t_max_x {
        return None;
    }

    let t_enter = t_min_x.max(t_min_y);
    let t_exit = t_max_x.min(t_max_y);

    if t_exit < 0.0 {
        return None;
    }

    let t = if t_enter >= 0.0 { t_enter } else { t_exit };
    let point = ray.point_at(t);

    let normal = if (t - t_min_x).abs() < 1e-10 {
        Vec2::new(if ray.direction.x > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        Vec2::new(0.0, if ray.direction.y > 0.0 { -1.0 } else { 1.0 })
    };

    Some(RayHit { t, point, normal })
}

fn ray_vs_polygon(ray: &Ray, vertices: &[Vec2]) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    let n = vertices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let a = vertices[i];
        let b = vertices[j];
        let edge = b.sub(a);
        let edge_n = edge.perpendicular().normalized();
        let denom = ray.direction.dot(edge_n);
        if denom.abs() < 1e-12 {
            continue;
        }
        let t = a.sub(ray.origin).dot(edge_n) / denom;
        if t < 0.0 {
            continue;
        }
        let hit = ray.point_at(t);
        let proj = hit.sub(a).dot(edge) / edge.length_sq();
        if proj < 0.0 || proj > 1.0 {
            continue;
        }
        let normal = if denom < 0.0 { edge_n } else { edge_n.negate() };
        if best.is_none() || t < best.as_ref().unwrap().t {
            best = Some(RayHit { t, point: hit, normal });
        }
    }
    best
}

// ── Broad Phase Grid ─────────────────────────────────────────

/// Spatial hash grid for broad-phase collision detection.
#[derive(Debug)]
pub struct SpatialGrid {
    cell_size: f64,
    cells: std::collections::HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert a shape with the given index.
    pub fn insert(&mut self, index: usize, shape: &Shape2D) {
        let (lo, hi) = shape.bounding_box();
        let min_cx = (lo.x / self.cell_size).floor() as i32;
        let min_cy = (lo.y / self.cell_size).floor() as i32;
        let max_cx = (hi.x / self.cell_size).floor() as i32;
        let max_cy = (hi.y / self.cell_size).floor() as i32;
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                self.cells.entry((cx, cy)).or_default().push(index);
            }
        }
    }

    /// Query potential collision pairs (broad phase).
    pub fn query_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = HashSet::new();
        for indices in self.cells.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let a = indices[i].min(indices[j]);
                    let b = indices[i].max(indices[j]);
                    pairs.insert((a, b));
                }
            }
        }
        pairs.into_iter().collect()
    }

    /// Query shapes near a given shape.
    pub fn query_shape(&self, shape: &Shape2D) -> Vec<usize> {
        let (lo, hi) = shape.bounding_box();
        let min_cx = (lo.x / self.cell_size).floor() as i32;
        let min_cy = (lo.y / self.cell_size).floor() as i32;
        let max_cx = (hi.x / self.cell_size).floor() as i32;
        let max_cy = (hi.y / self.cell_size).floor() as i32;
        let mut result = HashSet::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                if let Some(ids) = self.cells.get(&(cx, cy)) {
                    for id in ids {
                        result.insert(*id);
                    }
                }
            }
        }
        result.into_iter().collect()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.02
    }

    #[test]
    fn circle_circle_collision() {
        let a = Shape2D::circle(0.0, 0.0, 5.0);
        let b = Shape2D::circle(8.0, 0.0, 5.0);
        let c = test_collision(&a, &b).unwrap();
        assert!(approx(c.penetration, 2.0));
        assert!(approx(c.normal.x, 1.0));
    }

    #[test]
    fn circle_circle_no_collision() {
        let a = Shape2D::circle(0.0, 0.0, 5.0);
        let b = Shape2D::circle(20.0, 0.0, 5.0);
        assert!(test_collision(&a, &b).is_none());
    }

    #[test]
    fn aabb_aabb_collision() {
        let a = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        let b = Shape2D::aabb(8.0, 0.0, 18.0, 10.0);
        let c = test_collision(&a, &b).unwrap();
        assert!(approx(c.penetration, 2.0));
    }

    #[test]
    fn aabb_aabb_no_collision() {
        let a = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        let b = Shape2D::aabb(15.0, 0.0, 25.0, 10.0);
        assert!(test_collision(&a, &b).is_none());
    }

    #[test]
    fn circle_aabb_collision() {
        let c = Shape2D::circle(12.0, 5.0, 5.0);
        let a = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        let ct = test_collision(&c, &a).unwrap();
        assert!(ct.penetration > 0.0);
    }

    #[test]
    fn circle_aabb_no_collision() {
        let c = Shape2D::circle(20.0, 5.0, 5.0);
        let a = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        assert!(test_collision(&c, &a).is_none());
    }

    #[test]
    fn sat_polygon_collision() {
        let a = Shape2D::polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        let b = Shape2D::polygon(vec![
            Vec2::new(8.0, 0.0), Vec2::new(18.0, 0.0),
            Vec2::new(18.0, 10.0), Vec2::new(8.0, 10.0),
        ]);
        let c = test_collision(&a, &b).unwrap();
        assert!(approx(c.penetration, 2.0));
    }

    #[test]
    fn sat_polygon_no_collision() {
        let a = Shape2D::polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        let b = Shape2D::polygon(vec![
            Vec2::new(20.0, 0.0), Vec2::new(30.0, 0.0),
            Vec2::new(30.0, 10.0), Vec2::new(20.0, 10.0),
        ]);
        assert!(test_collision(&a, &b).is_none());
    }

    #[test]
    fn point_in_circle() {
        let s = Shape2D::circle(5.0, 5.0, 3.0);
        assert!(point_in_shape(Vec2::new(5.0, 5.0), &s));
        assert!(point_in_shape(Vec2::new(6.0, 5.0), &s));
        assert!(!point_in_shape(Vec2::new(9.0, 5.0), &s));
    }

    #[test]
    fn point_in_aabb() {
        let s = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        assert!(point_in_shape(Vec2::new(5.0, 5.0), &s));
        assert!(!point_in_shape(Vec2::new(11.0, 5.0), &s));
    }

    #[test]
    fn point_in_polygon_test() {
        let s = Shape2D::polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        assert!(point_in_shape(Vec2::new(5.0, 5.0), &s));
        assert!(!point_in_shape(Vec2::new(15.0, 5.0), &s));
    }

    #[test]
    fn point_in_obb() {
        let s = Shape2D::obb(5.0, 5.0, 5.0, 3.0, 0.0);
        assert!(point_in_shape(Vec2::new(5.0, 5.0), &s));
        assert!(!point_in_shape(Vec2::new(11.0, 5.0), &s));
    }

    #[test]
    fn ray_circle_hit() {
        let ray = Ray::new(Vec2::new(-10.0, 0.0), Vec2::new(1.0, 0.0));
        let s = Shape2D::circle(0.0, 0.0, 5.0);
        let hit = ray_intersect(&ray, &s).unwrap();
        assert!(approx(hit.point.x, -5.0));
        assert!(approx(hit.normal.x, -1.0));
    }

    #[test]
    fn ray_circle_miss() {
        let ray = Ray::new(Vec2::new(-10.0, 10.0), Vec2::new(1.0, 0.0));
        let s = Shape2D::circle(0.0, 0.0, 5.0);
        assert!(ray_intersect(&ray, &s).is_none());
    }

    #[test]
    fn ray_aabb_hit() {
        let ray = Ray::new(Vec2::new(-5.0, 5.0), Vec2::new(1.0, 0.0));
        let s = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        let hit = ray_intersect(&ray, &s).unwrap();
        assert!(approx(hit.point.x, 0.0));
        assert!(approx(hit.normal.x, -1.0));
    }

    #[test]
    fn sweep_test_moving_circles() {
        let a = Shape2D::circle(0.0, 0.0, 2.0);
        let b = Shape2D::circle(20.0, 0.0, 2.0);
        // Move a toward b
        let va = Vec2::new(20.0, 0.0);
        let vb = Vec2::zero();
        let result = sweep_test(&a, va, &b, vb, 32);
        assert!(result.is_some());
        let r = result.unwrap();
        // Should collide around t=0.8 (when gap closes: 20 - 4 = 16 units, vel = 20)
        assert!(r.toi > 0.5 && r.toi < 1.0);
    }

    #[test]
    fn sweep_test_no_collision() {
        let a = Shape2D::circle(0.0, 0.0, 2.0);
        let b = Shape2D::circle(100.0, 0.0, 2.0);
        let va = Vec2::new(5.0, 0.0);
        let vb = Vec2::zero();
        assert!(sweep_test(&a, va, &b, vb, 16).is_none());
    }

    #[test]
    fn sweep_test_parallel_motion() {
        let a = Shape2D::circle(0.0, 0.0, 2.0);
        let b = Shape2D::circle(0.0, 10.0, 2.0);
        let va = Vec2::new(10.0, 0.0);
        let vb = Vec2::new(10.0, 0.0);
        // Moving in parallel, same direction, never collide
        assert!(sweep_test(&a, va, &b, vb, 16).is_none());
    }

    #[test]
    fn manifold_circle_circle() {
        let a = Shape2D::circle(0.0, 0.0, 5.0);
        let b = Shape2D::circle(8.0, 0.0, 5.0);
        let m = compute_manifold(&a, &b).unwrap();
        assert_eq!(m.contacts.len(), 1);
        assert!(approx(m.depth, 2.0));
        assert!(approx(m.normal.x, 1.0));
    }

    #[test]
    fn manifold_aabb_aabb() {
        let a = Shape2D::aabb(0.0, 0.0, 10.0, 10.0);
        let b = Shape2D::aabb(8.0, 0.0, 18.0, 10.0);
        let m = compute_manifold(&a, &b).unwrap();
        assert_eq!(m.contacts.len(), 2);
        assert!(approx(m.depth, 2.0));
    }

    #[test]
    fn manifold_no_collision() {
        let a = Shape2D::circle(0.0, 0.0, 2.0);
        let b = Shape2D::circle(10.0, 0.0, 2.0);
        assert!(compute_manifold(&a, &b).is_none());
    }

    #[test]
    fn spatial_grid_pairs() {
        let mut grid = SpatialGrid::new(10.0);
        grid.insert(0, &Shape2D::circle(5.0, 5.0, 3.0));
        grid.insert(1, &Shape2D::circle(7.0, 5.0, 3.0));
        grid.insert(2, &Shape2D::circle(50.0, 50.0, 3.0));
        let pairs = grid.query_pairs();
        assert!(pairs.contains(&(0, 1)));
        assert!(!pairs.contains(&(0, 2)));
    }

    #[test]
    fn spatial_grid_query() {
        let mut grid = SpatialGrid::new(10.0);
        grid.insert(0, &Shape2D::circle(5.0, 5.0, 2.0));
        grid.insert(1, &Shape2D::circle(50.0, 50.0, 2.0));
        let nearby = grid.query_shape(&Shape2D::circle(6.0, 6.0, 1.0));
        assert!(nearby.contains(&0));
        assert!(!nearby.contains(&1));
    }

    #[test]
    fn bounding_box_shapes() {
        let c = Shape2D::circle(5.0, 5.0, 3.0);
        let (lo, hi) = c.bounding_box();
        assert!(approx(lo.x, 2.0));
        assert!(approx(hi.x, 8.0));
    }

    #[test]
    fn obb_collision_via_polygon() {
        let a = Shape2D::obb(5.0, 5.0, 5.0, 5.0, 0.0);
        let b = Shape2D::obb(12.0, 5.0, 5.0, 5.0, 0.0);
        let c = test_collision(&a, &b).unwrap();
        assert!(c.penetration > 0.0);
    }

    #[test]
    fn translate_circle() {
        let c = Shape2D::circle(0.0, 0.0, 5.0);
        let moved = c.translate(Vec2::new(10.0, 0.0));
        if let Shape2D::Circle { center, .. } = moved {
            assert!(approx(center.x, 10.0));
        } else {
            panic!("expected circle");
        }
    }

    #[test]
    fn vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx(a.length(), 5.0));
        let n = a.normalized();
        assert!(approx(n.length(), 1.0));
        let r = a.rotate(std::f64::consts::FRAC_PI_2);
        assert!(approx(r.x, -4.0));
        assert!(approx(r.y, 3.0));
    }

    #[test]
    fn circle_polygon_collision() {
        let c = Shape2D::circle(0.0, 5.0, 3.0);
        let p = Shape2D::polygon(vec![
            Vec2::new(2.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(2.0, 10.0),
        ]);
        let ct = test_collision(&c, &p).unwrap();
        assert!(ct.penetration > 0.0);
    }

    #[test]
    fn sweep_test_aabb_moving() {
        let a = Shape2D::aabb(0.0, 0.0, 4.0, 4.0);
        let b = Shape2D::aabb(10.0, 0.0, 14.0, 4.0);
        let va = Vec2::new(12.0, 0.0);
        let vb = Vec2::zero();
        let r = sweep_test(&a, va, &b, vb, 32).unwrap();
        assert!(r.toi > 0.3 && r.toi < 0.8);
    }

    #[test]
    fn manifold_polygon_fallback() {
        let a = Shape2D::polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        let b = Shape2D::polygon(vec![
            Vec2::new(8.0, 0.0), Vec2::new(18.0, 0.0),
            Vec2::new(18.0, 10.0), Vec2::new(8.0, 10.0),
        ]);
        let m = compute_manifold(&a, &b).unwrap();
        assert!(!m.is_empty());
        assert!(m.depth > 0.0);
    }
}
