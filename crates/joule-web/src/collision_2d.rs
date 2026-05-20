//! 2D collision shapes and detection — circle, AABB, OBB, convex polygon (≤8 verts),
//! capsule, line segment.  Contact generation, penetration depth, point-in-shape,
//! ray-vs-shape intersection with hit distance and normal.

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
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn negate(self) -> Self { Self { x: -self.x, y: -self.y } }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── AABB ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self { Self { min, max } }

    pub fn from_center_half(center: Vec2, half: Vec2) -> Self {
        Self { min: center.sub(half), max: center.add(half) }
    }

    pub fn center(&self) -> Vec2 { self.min.add(self.max).scale(0.5) }
    pub fn half_extents(&self) -> Vec2 { self.max.sub(self.min).scale(0.5) }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
    }

    pub fn contains_point(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn merge(&self, other: &AABB) -> AABB {
        AABB {
            min: Vec2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Vec2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }
}

// ── Shapes ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Shape2D {
    Circle { center: Vec2, radius: f64 },
    Aabb(AABB),
    Obb { center: Vec2, half_extents: Vec2, rotation: f64 },
    Polygon { vertices: Vec<Vec2> },          // convex, ≤ 8 verts
    Capsule { a: Vec2, b: Vec2, radius: f64 },
    Segment { a: Vec2, b: Vec2 },
}

impl Shape2D {
    /// Compute axis-aligned bounding box for any shape.
    pub fn aabb(&self) -> AABB {
        match self {
            Shape2D::Circle { center, radius } => AABB {
                min: Vec2::new(center.x - radius, center.y - radius),
                max: Vec2::new(center.x + radius, center.y + radius),
            },
            Shape2D::Aabb(a) => *a,
            Shape2D::Obb { center, half_extents, rotation } => {
                let cos_a = rotation.cos().abs();
                let sin_a = rotation.sin().abs();
                let hx = half_extents.x * cos_a + half_extents.y * sin_a;
                let hy = half_extents.x * sin_a + half_extents.y * cos_a;
                AABB {
                    min: Vec2::new(center.x - hx, center.y - hy),
                    max: Vec2::new(center.x + hx, center.y + hy),
                }
            }
            Shape2D::Polygon { vertices } => {
                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;
                for v in vertices {
                    min_x = min_x.min(v.x);
                    min_y = min_y.min(v.y);
                    max_x = max_x.max(v.x);
                    max_y = max_y.max(v.y);
                }
                AABB { min: Vec2::new(min_x, min_y), max: Vec2::new(max_x, max_y) }
            }
            Shape2D::Capsule { a, b, radius } => AABB {
                min: Vec2::new(a.x.min(b.x) - radius, a.y.min(b.y) - radius),
                max: Vec2::new(a.x.max(b.x) + radius, a.y.max(b.y) + radius),
            },
            Shape2D::Segment { a, b } => AABB {
                min: Vec2::new(a.x.min(b.x), a.y.min(b.y)),
                max: Vec2::new(a.x.max(b.x), a.y.max(b.y)),
            },
        }
    }
}

// ── Contact ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    /// World-space contact point.
    pub point: Vec2,
    /// Contact normal (points from shape B toward shape A).
    pub normal: Vec2,
    /// Penetration depth (positive when overlapping).
    pub depth: f64,
}

// ── Ray ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec2,
    pub direction: Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub distance: f64,
    pub point: Vec2,
    pub normal: Vec2,
}

// ── Point-in-shape ───────────────────────────────────────────

pub fn point_in_circle(p: Vec2, center: Vec2, radius: f64) -> bool {
    p.sub(center).length_sq() <= radius * radius
}

pub fn point_in_aabb(p: Vec2, aabb: &AABB) -> bool {
    aabb.contains_point(p)
}

pub fn point_in_polygon(p: Vec2, verts: &[Vec2]) -> bool {
    let n = verts.len();
    if n < 3 { return false; }
    // All cross products same sign → inside convex polygon.
    let mut sign = None;
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = verts[j].sub(verts[i]);
        let to_p = p.sub(verts[i]);
        let c = edge.cross(to_p);
        match sign {
            None => sign = Some(c >= 0.0),
            Some(s) => {
                if (c >= 0.0) != s { return false; }
            }
        }
    }
    true
}

pub fn point_in_shape(p: Vec2, shape: &Shape2D) -> bool {
    match shape {
        Shape2D::Circle { center, radius } => point_in_circle(p, *center, *radius),
        Shape2D::Aabb(aabb) => aabb.contains_point(p),
        Shape2D::Obb { center, half_extents, rotation } => {
            let local = p.sub(*center).rotate(-*rotation);
            local.x.abs() <= half_extents.x && local.y.abs() <= half_extents.y
        }
        Shape2D::Polygon { vertices } => point_in_polygon(p, vertices),
        Shape2D::Capsule { a, b, radius } => {
            closest_point_on_segment(p, *a, *b).sub(p).length_sq() <= radius * radius
        }
        Shape2D::Segment { a, b } => {
            let cp = closest_point_on_segment(p, *a, *b);
            cp.sub(p).length_sq() < 1e-10
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b.sub(a);
    let len_sq = ab.length_sq();
    if len_sq < 1e-12 { return a; }
    let t = p.sub(a).dot(ab) / len_sq;
    let t = t.clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

// ── Circle-Circle ────────────────────────────────────────────

pub fn circle_vs_circle(c1: Vec2, r1: f64, c2: Vec2, r2: f64) -> Option<Contact> {
    let d = c2.sub(c1);
    let dist_sq = d.length_sq();
    let sum_r = r1 + r2;
    if dist_sq > sum_r * sum_r { return None; }
    let dist = dist_sq.sqrt();
    let normal = if dist < 1e-12 { Vec2::new(1.0, 0.0) } else { d.scale(1.0 / dist) };
    let depth = sum_r - dist;
    let point = c1.add(normal.scale(r1 - depth * 0.5));
    Some(Contact { point, normal, depth })
}

// ── Circle-AABB ──────────────────────────────────────────────

pub fn circle_vs_aabb(center: Vec2, radius: f64, aabb: &AABB) -> Option<Contact> {
    let clamped = Vec2::new(
        center.x.clamp(aabb.min.x, aabb.max.x),
        center.y.clamp(aabb.min.y, aabb.max.y),
    );
    let d = center.sub(clamped);
    let dist_sq = d.length_sq();
    if dist_sq > radius * radius { return None; }
    let dist = dist_sq.sqrt();
    let normal = if dist < 1e-12 {
        // Circle centre inside AABB — pick smallest overlap axis.
        let dx_left = center.x - aabb.min.x;
        let dx_right = aabb.max.x - center.x;
        let dy_bottom = center.y - aabb.min.y;
        let dy_top = aabb.max.y - center.y;
        let min_d = dx_left.min(dx_right).min(dy_bottom).min(dy_top);
        if (min_d - dx_left).abs() < 1e-12 { Vec2::new(-1.0, 0.0) }
        else if (min_d - dx_right).abs() < 1e-12 { Vec2::new(1.0, 0.0) }
        else if (min_d - dy_bottom).abs() < 1e-12 { Vec2::new(0.0, -1.0) }
        else { Vec2::new(0.0, 1.0) }
    } else {
        d.scale(1.0 / dist)
    };
    let depth = radius - dist;
    let point = clamped;
    Some(Contact { point, normal, depth })
}

// ── AABB-AABB ────────────────────────────────────────────────

pub fn aabb_vs_aabb(a: &AABB, b: &AABB) -> Option<Contact> {
    let ac = a.center();
    let bc = b.center();
    let ah = a.half_extents();
    let bh = b.half_extents();

    let dx = bc.x - ac.x;
    let dy = bc.y - ac.y;
    let ox = ah.x + bh.x - dx.abs();
    let oy = ah.y + bh.y - dy.abs();
    if ox <= 0.0 || oy <= 0.0 { return None; }

    let (normal, depth) = if ox < oy {
        let nx = if dx > 0.0 { 1.0 } else { -1.0 };
        (Vec2::new(nx, 0.0), ox)
    } else {
        let ny = if dy > 0.0 { 1.0 } else { -1.0 };
        (Vec2::new(0.0, ny), oy)
    };

    let point = Vec2::new(
        if normal.x.abs() > 0.5 { ac.x + normal.x * ah.x } else { ac.x.max(bc.x).min(ac.x) },
        if normal.y.abs() > 0.5 { ac.y + normal.y * ah.y } else { ac.y.max(bc.y).min(ac.y) },
    );
    Some(Contact { point, normal, depth })
}

// ── Circle-Polygon ───────────────────────────────────────────

pub fn circle_vs_polygon(center: Vec2, radius: f64, verts: &[Vec2]) -> Option<Contact> {
    let n = verts.len();
    if n < 3 { return None; }

    let mut min_dist = f64::MAX;
    let mut best_cp = center;
    for i in 0..n {
        let j = (i + 1) % n;
        let cp = closest_point_on_segment(center, verts[i], verts[j]);
        let d = center.sub(cp).length_sq();
        if d < min_dist {
            min_dist = d;
            best_cp = cp;
        }
    }
    let dist = min_dist.sqrt();

    let inside = point_in_polygon(center, verts);
    if !inside && dist > radius { return None; }

    let diff = center.sub(best_cp);
    let normal = if diff.length_sq() < 1e-12 {
        Vec2::new(0.0, 1.0)
    } else if inside {
        diff.normalized().negate()
    } else {
        diff.normalized()
    };

    let depth = if inside { radius + dist } else { radius - dist };
    Some(Contact { point: best_cp, normal, depth })
}

// ── Shape dispatch ───────────────────────────────────────────

pub fn test_shapes(a: &Shape2D, b: &Shape2D) -> Option<Contact> {
    match (a, b) {
        (Shape2D::Circle { center: c1, radius: r1 }, Shape2D::Circle { center: c2, radius: r2 }) =>
            circle_vs_circle(*c1, *r1, *c2, *r2),
        (Shape2D::Aabb(a), Shape2D::Aabb(b)) =>
            aabb_vs_aabb(a, b),
        (Shape2D::Circle { center, radius }, Shape2D::Aabb(aabb)) =>
            circle_vs_aabb(*center, *radius, aabb),
        (Shape2D::Aabb(aabb), Shape2D::Circle { center, radius }) =>
            circle_vs_aabb(*center, *radius, aabb).map(|c| Contact {
                normal: c.normal.negate(), ..c
            }),
        (Shape2D::Circle { center, radius }, Shape2D::Polygon { vertices }) =>
            circle_vs_polygon(*center, *radius, vertices),
        (Shape2D::Polygon { vertices }, Shape2D::Circle { center, radius }) =>
            circle_vs_polygon(*center, *radius, vertices).map(|c| Contact {
                normal: c.normal.negate(), ..c
            }),
        (Shape2D::Aabb(a), Shape2D::Polygon { vertices }) => {
            let verts_a = vec![
                a.min, Vec2::new(a.max.x, a.min.y), a.max, Vec2::new(a.min.x, a.max.y),
            ];
            polygon_vs_polygon(&verts_a, vertices)
        }
        (Shape2D::Polygon { vertices }, Shape2D::Aabb(a)) => {
            let verts_b = vec![
                a.min, Vec2::new(a.max.x, a.min.y), a.max, Vec2::new(a.min.x, a.max.y),
            ];
            polygon_vs_polygon(vertices, &verts_b)
        }
        (Shape2D::Polygon { vertices: va }, Shape2D::Polygon { vertices: vb }) =>
            polygon_vs_polygon(va, vb),
        (Shape2D::Capsule { a: a1, b: b1, radius: r1 }, Shape2D::Circle { center, radius: r2 }) => {
            let cp = closest_point_on_segment(*center, *a1, *b1);
            circle_vs_circle(cp, *r1, *center, *r2)
        }
        (Shape2D::Circle { center, radius: r2 }, Shape2D::Capsule { a: a1, b: b1, radius: r1 }) => {
            let cp = closest_point_on_segment(*center, *a1, *b1);
            circle_vs_circle(*center, *r2, cp, *r1)
        }
        (Shape2D::Capsule { a: a1, b: b1, radius: r1 }, Shape2D::Capsule { a: a2, b: b2, radius: r2 }) => {
            let (p1, p2) = closest_points_segments(*a1, *b1, *a2, *b2);
            circle_vs_circle(p1, *r1, p2, *r2)
        }
        _ => None, // segment and OBB pairs: not all implemented
    }
}

// ── Polygon-Polygon (SAT) ────────────────────────────────────

fn polygon_vs_polygon(va: &[Vec2], vb: &[Vec2]) -> Option<Contact> {
    let na = va.len();
    let nb = vb.len();
    if na < 3 || nb < 3 { return None; }

    let mut min_depth = f64::MAX;
    let mut best_normal = Vec2::new(1.0, 0.0);

    // Check axes from polygon A
    for i in 0..na {
        let j = (i + 1) % na;
        let edge = va[j].sub(va[i]);
        let axis = edge.perpendicular().normalized();
        let (min_a, max_a) = project_polygon(va, axis);
        let (min_b, max_b) = project_polygon(vb, axis);
        let overlap = max_a.min(max_b) - min_a.max(min_b);
        if overlap <= 0.0 { return None; }
        if overlap < min_depth {
            min_depth = overlap;
            best_normal = axis;
        }
    }

    // Check axes from polygon B
    for i in 0..nb {
        let j = (i + 1) % nb;
        let edge = vb[j].sub(vb[i]);
        let axis = edge.perpendicular().normalized();
        let (min_a, max_a) = project_polygon(va, axis);
        let (min_b, max_b) = project_polygon(vb, axis);
        let overlap = max_a.min(max_b) - min_a.max(min_b);
        if overlap <= 0.0 { return None; }
        if overlap < min_depth {
            min_depth = overlap;
            best_normal = axis;
        }
    }

    // Ensure normal points from B towards A
    let center_a = polygon_centroid(va);
    let center_b = polygon_centroid(vb);
    let dir = center_a.sub(center_b);
    if dir.dot(best_normal) < 0.0 {
        best_normal = best_normal.negate();
    }

    // Contact point: deepest vertex of B in direction of normal
    let mut best_point = vb[0];
    let mut best_proj = vb[0].dot(best_normal.negate());
    for v in &vb[1..] {
        let proj = v.dot(best_normal.negate());
        if proj > best_proj {
            best_proj = proj;
            best_point = *v;
        }
    }

    Some(Contact { point: best_point, normal: best_normal, depth: min_depth })
}

fn project_polygon(verts: &[Vec2], axis: Vec2) -> (f64, f64) {
    let mut min_p = f64::MAX;
    let mut max_p = f64::MIN;
    for v in verts {
        let p = v.dot(axis);
        min_p = min_p.min(p);
        max_p = max_p.max(p);
    }
    (min_p, max_p)
}

fn polygon_centroid(verts: &[Vec2]) -> Vec2 {
    let mut c = Vec2::zero();
    for v in verts { c = c.add(*v); }
    c.scale(1.0 / verts.len() as f64)
}

fn closest_points_segments(a1: Vec2, b1: Vec2, a2: Vec2, b2: Vec2) -> (Vec2, Vec2) {
    let d1 = b1.sub(a1);
    let d2 = b2.sub(a2);
    let r = a1.sub(a2);
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
            s = if denom.abs() > 1e-12 { ((b_val * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            t = ((b_val * s + f) / e).clamp(0.0, 1.0);
        }
    }
    (a1.add(d1.scale(s)), a2.add(d2.scale(t)))
}

// ── Ray intersection ─────────────────────────────────────────

pub fn ray_vs_circle(ray: &Ray, center: Vec2, radius: f64) -> Option<RayHit> {
    let oc = ray.origin.sub(center);
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return None; }
    let sqrt_disc = disc.sqrt();
    let t = (-b - sqrt_disc) / (2.0 * a);
    let t = if t < 0.0 { (-b + sqrt_disc) / (2.0 * a) } else { t };
    if t < 0.0 { return None; }
    let point = ray.origin.add(ray.direction.scale(t));
    let normal = point.sub(center).normalized();
    Some(RayHit { distance: t, point, normal })
}

pub fn ray_vs_aabb(ray: &Ray, aabb: &AABB) -> Option<RayHit> {
    let inv_dx = if ray.direction.x.abs() > 1e-12 { 1.0 / ray.direction.x } else { f64::MAX.copysign(ray.direction.x) };
    let inv_dy = if ray.direction.y.abs() > 1e-12 { 1.0 / ray.direction.y } else { f64::MAX.copysign(ray.direction.y) };

    let tx1 = (aabb.min.x - ray.origin.x) * inv_dx;
    let tx2 = (aabb.max.x - ray.origin.x) * inv_dx;
    let ty1 = (aabb.min.y - ray.origin.y) * inv_dy;
    let ty2 = (aabb.max.y - ray.origin.y) * inv_dy;

    let tmin = tx1.min(tx2).max(ty1.min(ty2));
    let tmax = tx1.max(tx2).min(ty1.max(ty2));

    if tmax < 0.0 || tmin > tmax { return None; }
    let t = if tmin >= 0.0 { tmin } else { tmax };
    if t < 0.0 { return None; }

    let point = ray.origin.add(ray.direction.scale(t));
    // Determine normal from which face was hit.
    let normal = if (t - tx1.min(tx2)).abs() < 1e-6 {
        Vec2::new(if ray.direction.x > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        Vec2::new(0.0, if ray.direction.y > 0.0 { -1.0 } else { 1.0 })
    };
    Some(RayHit { distance: t, point, normal })
}

pub fn ray_vs_shape(ray: &Ray, shape: &Shape2D) -> Option<RayHit> {
    match shape {
        Shape2D::Circle { center, radius } => ray_vs_circle(ray, *center, *radius),
        Shape2D::Aabb(aabb) => ray_vs_aabb(ray, aabb),
        _ => None, // other shapes: callers can extend
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v2_approx(a: Vec2, b: Vec2) -> bool { approx(a.x, b.x) && approx(a.y, b.y) }

    // ── AABB ──

    #[test]
    fn aabb_overlap() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let b = AABB::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert!(a.overlaps(&b));
    }

    #[test]
    fn aabb_no_overlap() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let b = AABB::new(Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0));
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn aabb_contains_point() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(5.0, 5.0));
        assert!(a.contains_point(Vec2::new(2.5, 2.5)));
        assert!(!a.contains_point(Vec2::new(6.0, 2.5)));
    }

    #[test]
    fn aabb_merge() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let b = AABB::new(Vec2::new(2.0, -1.0), Vec2::new(3.0, 2.0));
        let m = a.merge(&b);
        assert!(v2_approx(m.min, Vec2::new(0.0, -1.0)));
        assert!(v2_approx(m.max, Vec2::new(3.0, 2.0)));
    }

    // ── Circle-Circle ──

    #[test]
    fn circles_overlapping() {
        let c = circle_vs_circle(Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(approx(c.depth, 0.5));
    }

    #[test]
    fn circles_separate() {
        assert!(circle_vs_circle(Vec2::zero(), 1.0, Vec2::new(5.0, 0.0), 1.0).is_none());
    }

    #[test]
    fn circles_coincident() {
        let c = circle_vs_circle(Vec2::zero(), 1.0, Vec2::zero(), 1.0).unwrap();
        assert!(approx(c.depth, 2.0));
    }

    // ── Circle-AABB ──

    #[test]
    fn circle_aabb_hit() {
        let aabb = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let c = circle_vs_aabb(Vec2::new(2.8, 1.0), 1.0, &aabb);
        assert!(c.is_some());
    }

    #[test]
    fn circle_aabb_miss() {
        let aabb = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        assert!(circle_vs_aabb(Vec2::new(10.0, 10.0), 1.0, &aabb).is_none());
    }

    // ── AABB-AABB ──

    #[test]
    fn aabb_vs_aabb_hit() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let b = AABB::new(Vec2::new(1.5, 0.5), Vec2::new(3.0, 1.5));
        let c = aabb_vs_aabb(&a, &b).unwrap();
        assert!(c.depth > 0.0);
    }

    #[test]
    fn aabb_vs_aabb_miss() {
        let a = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let b = AABB::new(Vec2::new(2.0, 0.0), Vec2::new(3.0, 1.0));
        assert!(aabb_vs_aabb(&a, &b).is_none());
    }

    // ── Point-in-shape ──

    #[test]
    fn point_in_circle_test() {
        assert!(point_in_circle(Vec2::new(0.5, 0.0), Vec2::zero(), 1.0));
        assert!(!point_in_circle(Vec2::new(2.0, 0.0), Vec2::zero(), 1.0));
    }

    #[test]
    fn point_in_polygon_test() {
        let sq = vec![
            Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0),
        ];
        assert!(point_in_polygon(Vec2::new(1.0, 1.0), &sq));
        assert!(!point_in_polygon(Vec2::new(3.0, 1.0), &sq));
    }

    #[test]
    fn point_in_obb() {
        let obb = Shape2D::Obb {
            center: Vec2::zero(),
            half_extents: Vec2::new(2.0, 1.0),
            rotation: 0.0,
        };
        assert!(point_in_shape(Vec2::new(1.0, 0.5), &obb));
        assert!(!point_in_shape(Vec2::new(3.0, 0.0), &obb));
    }

    // ── Shape2D::aabb ──

    #[test]
    fn shape_aabb_circle() {
        let s = Shape2D::Circle { center: Vec2::new(5.0, 5.0), radius: 2.0 };
        let bb = s.aabb();
        assert!(v2_approx(bb.min, Vec2::new(3.0, 3.0)));
        assert!(v2_approx(bb.max, Vec2::new(7.0, 7.0)));
    }

    #[test]
    fn shape_aabb_capsule() {
        let s = Shape2D::Capsule { a: Vec2::new(0.0, 0.0), b: Vec2::new(4.0, 0.0), radius: 1.0 };
        let bb = s.aabb();
        assert!(v2_approx(bb.min, Vec2::new(-1.0, -1.0)));
        assert!(v2_approx(bb.max, Vec2::new(5.0, 1.0)));
    }

    // ── Polygon vs Polygon ──

    #[test]
    fn polygon_vs_polygon_overlapping() {
        let sq1 = Shape2D::Polygon { vertices: vec![
            Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0),
        ]};
        let sq2 = Shape2D::Polygon { vertices: vec![
            Vec2::new(1.0, 1.0), Vec2::new(3.0, 1.0),
            Vec2::new(3.0, 3.0), Vec2::new(1.0, 3.0),
        ]};
        let c = test_shapes(&sq1, &sq2);
        assert!(c.is_some());
        assert!(c.unwrap().depth > 0.0);
    }

    #[test]
    fn polygon_vs_polygon_separated() {
        let sq1 = Shape2D::Polygon { vertices: vec![
            Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        ]};
        let sq2 = Shape2D::Polygon { vertices: vec![
            Vec2::new(5.0, 5.0), Vec2::new(6.0, 5.0),
            Vec2::new(6.0, 6.0), Vec2::new(5.0, 6.0),
        ]};
        assert!(test_shapes(&sq1, &sq2).is_none());
    }

    // ── Circle vs Polygon ──

    #[test]
    fn circle_vs_polygon_hit() {
        let sq = vec![
            Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0),
        ];
        let c = circle_vs_polygon(Vec2::new(2.5, 1.0), 1.0, &sq);
        assert!(c.is_some());
    }

    #[test]
    fn circle_vs_polygon_miss() {
        let sq = vec![
            Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0),
        ];
        assert!(circle_vs_polygon(Vec2::new(10.0, 10.0), 0.5, &sq).is_none());
    }

    // ── Ray-vs-Circle ──

    #[test]
    fn ray_hits_circle() {
        let ray = Ray { origin: Vec2::new(-5.0, 0.0), direction: Vec2::new(1.0, 0.0) };
        let hit = ray_vs_circle(&ray, Vec2::zero(), 1.0).unwrap();
        assert!(approx(hit.distance, 4.0));
        assert!(approx(hit.normal.x, -1.0));
    }

    #[test]
    fn ray_misses_circle() {
        let ray = Ray { origin: Vec2::new(-5.0, 5.0), direction: Vec2::new(1.0, 0.0) };
        assert!(ray_vs_circle(&ray, Vec2::zero(), 1.0).is_none());
    }

    // ── Ray-vs-AABB ──

    #[test]
    fn ray_hits_aabb() {
        let aabb = AABB::new(Vec2::new(2.0, -1.0), Vec2::new(4.0, 1.0));
        let ray = Ray { origin: Vec2::new(0.0, 0.0), direction: Vec2::new(1.0, 0.0) };
        let hit = ray_vs_aabb(&ray, &aabb).unwrap();
        assert!(approx(hit.distance, 2.0));
    }

    #[test]
    fn ray_misses_aabb() {
        let aabb = AABB::new(Vec2::new(2.0, 2.0), Vec2::new(4.0, 4.0));
        let ray = Ray { origin: Vec2::new(0.0, 0.0), direction: Vec2::new(1.0, 0.0) };
        assert!(ray_vs_aabb(&ray, &aabb).is_none());
    }

    // ── Capsule ──

    #[test]
    fn capsule_vs_circle() {
        let cap = Shape2D::Capsule { a: Vec2::new(0.0, 0.0), b: Vec2::new(4.0, 0.0), radius: 0.5 };
        let circ = Shape2D::Circle { center: Vec2::new(2.0, 0.8), radius: 0.5 };
        let c = test_shapes(&cap, &circ);
        assert!(c.is_some());
    }

    #[test]
    fn capsule_vs_capsule() {
        let c1 = Shape2D::Capsule { a: Vec2::new(0.0, 0.0), b: Vec2::new(4.0, 0.0), radius: 0.5 };
        let c2 = Shape2D::Capsule { a: Vec2::new(2.0, 0.0), b: Vec2::new(2.0, 4.0), radius: 0.5 };
        let c = test_shapes(&c1, &c2);
        assert!(c.is_some());
    }

    // ── Dispatch symmetry ──

    #[test]
    fn dispatch_symmetric() {
        let a = Shape2D::Circle { center: Vec2::zero(), radius: 1.0 };
        let b = Shape2D::Aabb(AABB::new(Vec2::new(0.5, -0.5), Vec2::new(2.0, 0.5)));
        let c1 = test_shapes(&a, &b);
        let c2 = test_shapes(&b, &a);
        assert!(c1.is_some());
        assert!(c2.is_some());
    }
}
