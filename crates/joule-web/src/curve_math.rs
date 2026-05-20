//! Parametric curves — cubic Bezier, B-spline, Catmull-Rom spline, curve evaluation,
//! tangent/normal at point, arc length estimation, subdivision, closest point on curve,
//! curve-curve intersection.

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn lerp(self, o: Self, t: f64) -> Self { self.add(o.sub(self).scale(t)) }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Curve Trait ──────────────────────────────────────────────

/// Trait for parametric curves evaluated at t in [0, 1].
pub trait Curve {
    /// Evaluate the curve position at parameter t.
    fn evaluate(&self, t: f64) -> Vec2;

    /// Evaluate the tangent (first derivative) at parameter t.
    fn tangent(&self, t: f64) -> Vec2;

    /// Evaluate the normal (perpendicular to tangent) at parameter t.
    fn normal(&self, t: f64) -> Vec2 {
        self.tangent(t).perpendicular().normalized()
    }

    /// Estimate arc length by sampling with `segments` line segments.
    fn arc_length(&self, segments: u32) -> f64 {
        let n = segments.max(1);
        let mut length = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let curr = self.evaluate(t);
            length += prev.distance(curr);
            prev = curr;
        }
        length
    }

    /// Arc length from t=0 to t=t_end.
    fn arc_length_to(&self, t_end: f64, segments: u32) -> f64 {
        let n = segments.max(1);
        let mut length = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=n {
            let t = t_end * i as f64 / n as f64;
            let curr = self.evaluate(t);
            length += prev.distance(curr);
            prev = curr;
        }
        length
    }

    /// Subdivide the curve into `n` evenly-spaced sample points.
    fn subdivide(&self, n: u32) -> Vec<Vec2> {
        let count = n.max(2);
        (0..count).map(|i| {
            let t = i as f64 / (count - 1) as f64;
            self.evaluate(t)
        }).collect()
    }

    /// Find the parameter t of the closest point on the curve to `point`.
    fn closest_point(&self, point: Vec2, samples: u32) -> (f64, Vec2) {
        let n = samples.max(8);
        let mut best_t = 0.0;
        let mut best_dist_sq = f64::INFINITY;
        let mut best_pt = self.evaluate(0.0);

        // Coarse pass
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let p = self.evaluate(t);
            let d = point.sub(p).length_sq();
            if d < best_dist_sq {
                best_dist_sq = d;
                best_t = t;
                best_pt = p;
            }
        }

        // Refine with binary-style search
        let mut lo = (best_t - 1.0 / n as f64).max(0.0);
        let mut hi = (best_t + 1.0 / n as f64).min(1.0);
        for _ in 0..16 {
            let t1 = lo + (hi - lo) * 0.333;
            let t2 = lo + (hi - lo) * 0.667;
            let d1 = point.sub(self.evaluate(t1)).length_sq();
            let d2 = point.sub(self.evaluate(t2)).length_sq();
            if d1 < d2 {
                hi = t2;
            } else {
                lo = t1;
            }
        }
        let t_final = (lo + hi) * 0.5;
        let p_final = self.evaluate(t_final);
        if point.sub(p_final).length_sq() < best_dist_sq {
            (t_final, p_final)
        } else {
            (best_t, best_pt)
        }
    }
}

// ── Cubic Bezier ─────────────────────────────────────────────

/// Cubic Bezier curve defined by four control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
}

impl CubicBezier {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Split the curve at parameter t using De Casteljau's algorithm.
    /// Returns (left_half, right_half).
    pub fn split(&self, t: f64) -> (CubicBezier, CubicBezier) {
        let a0 = self.p0.lerp(self.p1, t);
        let a1 = self.p1.lerp(self.p2, t);
        let a2 = self.p2.lerp(self.p3, t);

        let b0 = a0.lerp(a1, t);
        let b1 = a1.lerp(a2, t);

        let c0 = b0.lerp(b1, t);

        let left = CubicBezier::new(self.p0, a0, b0, c0);
        let right = CubicBezier::new(c0, b1, a2, self.p3);
        (left, right)
    }

    /// Bounding box of the curve (conservative, uses control point hull).
    pub fn bounding_box(&self) -> (Vec2, Vec2) {
        let mut lo = self.p0;
        let mut hi = self.p0;
        for p in [self.p1, self.p2, self.p3] {
            lo.x = lo.x.min(p.x);
            lo.y = lo.y.min(p.y);
            hi.x = hi.x.max(p.x);
            hi.y = hi.y.max(p.y);
        }
        (lo, hi)
    }
}

impl Curve for CubicBezier {
    fn evaluate(&self, t: f64) -> Vec2 {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;
        let uuu = uu * u;
        let ttt = tt * t;

        self.p0.scale(uuu)
            .add(self.p1.scale(3.0 * uu * t))
            .add(self.p2.scale(3.0 * u * tt))
            .add(self.p3.scale(ttt))
    }

    fn tangent(&self, t: f64) -> Vec2 {
        let u = 1.0 - t;
        let a = self.p1.sub(self.p0).scale(3.0 * u * u);
        let b = self.p2.sub(self.p1).scale(6.0 * u * t);
        let c = self.p3.sub(self.p2).scale(3.0 * t * t);
        a.add(b).add(c)
    }
}

// ── Quadratic Bezier ─────────────────────────────────────────

/// Quadratic Bezier curve defined by three control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticBezier {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
}

impl QuadraticBezier {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        Self { p0, p1, p2 }
    }

    /// Elevate to a cubic Bezier.
    pub fn to_cubic(&self) -> CubicBezier {
        let c1 = self.p0.add(self.p1.sub(self.p0).scale(2.0 / 3.0));
        let c2 = self.p2.add(self.p1.sub(self.p2).scale(2.0 / 3.0));
        CubicBezier::new(self.p0, c1, c2, self.p2)
    }
}

impl Curve for QuadraticBezier {
    fn evaluate(&self, t: f64) -> Vec2 {
        let u = 1.0 - t;
        self.p0.scale(u * u)
            .add(self.p1.scale(2.0 * u * t))
            .add(self.p2.scale(t * t))
    }

    fn tangent(&self, t: f64) -> Vec2 {
        let u = 1.0 - t;
        self.p1.sub(self.p0).scale(2.0 * u)
            .add(self.p2.sub(self.p1).scale(2.0 * t))
    }
}

// ── Catmull-Rom Spline ───────────────────────────────────────

/// Catmull-Rom spline passing through all control points.
#[derive(Debug, Clone, PartialEq)]
pub struct CatmullRomSpline {
    /// Control points (the curve passes through these).
    pub points: Vec<Vec2>,
    /// Tension parameter (0.0 = Catmull-Rom, higher = tighter).
    pub tension: f64,
}

impl CatmullRomSpline {
    pub fn new(points: Vec<Vec2>) -> Self {
        Self { points, tension: 0.0 }
    }

    pub fn with_tension(mut self, tension: f64) -> Self {
        self.tension = tension;
        self
    }

    /// Number of segments (one less than number of points, minus 2 for endpoints).
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 4 { return 0; }
        self.points.len() - 3
    }

    /// Evaluate a single Catmull-Rom segment.
    fn eval_segment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f64, tension: f64) -> Vec2 {
        let s = (1.0 - tension) * 0.5;
        let tt = t * t;
        let ttt = tt * t;

        let a = p1;
        let b = p2.sub(p0).scale(s);
        let c = p0.scale(2.0 * s)
            .add(p1.scale(s - 3.0))
            .add(p2.scale(3.0 - 2.0 * s))
            .add(p3.scale(-s));
        let d = p0.scale(-s)
            .add(p1.scale(2.0 - s))
            .add(p2.scale(s - 2.0))
            .add(p3.scale(s));

        a.add(b.scale(t)).add(c.scale(tt)).add(d.scale(ttt))
    }

    /// Evaluate the tangent of a Catmull-Rom segment.
    fn tangent_segment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f64, tension: f64) -> Vec2 {
        let s = (1.0 - tension) * 0.5;
        let tt = t * t;

        let b = p2.sub(p0).scale(s);
        let c = p0.scale(2.0 * s)
            .add(p1.scale(s - 3.0))
            .add(p2.scale(3.0 - 2.0 * s))
            .add(p3.scale(-s));
        let d = p0.scale(-s)
            .add(p1.scale(2.0 - s))
            .add(p2.scale(s - 2.0))
            .add(p3.scale(s));

        b.add(c.scale(2.0 * t)).add(d.scale(3.0 * tt))
    }

    /// Map a global parameter t in [0, 1] to (segment_index, local_t).
    fn map_parameter(&self, t: f64) -> (usize, f64) {
        let seg_count = self.segment_count();
        if seg_count == 0 { return (0, 0.0); }
        let scaled = t * seg_count as f64;
        let seg = (scaled.floor() as usize).min(seg_count - 1);
        let local_t = scaled - seg as f64;
        (seg, local_t.clamp(0.0, 1.0))
    }
}

impl Curve for CatmullRomSpline {
    fn evaluate(&self, t: f64) -> Vec2 {
        if self.points.len() < 4 {
            if self.points.is_empty() { return Vec2::zero(); }
            if self.points.len() == 1 { return self.points[0]; }
            let idx = ((self.points.len() - 1) as f64 * t).round() as usize;
            return self.points[idx.min(self.points.len() - 1)];
        }
        let (seg, lt) = self.map_parameter(t);
        let i = seg + 1;
        Self::eval_segment(
            self.points[i - 1],
            self.points[i],
            self.points[i + 1],
            self.points[i + 2],
            lt,
            self.tension,
        )
    }

    fn tangent(&self, t: f64) -> Vec2 {
        if self.points.len() < 4 { return Vec2::new(1.0, 0.0); }
        let (seg, lt) = self.map_parameter(t);
        let i = seg + 1;
        Self::tangent_segment(
            self.points[i - 1],
            self.points[i],
            self.points[i + 1],
            self.points[i + 2],
            lt,
            self.tension,
        )
    }
}

// ── B-Spline (Uniform Cubic) ─────────────────────────────────

/// Uniform cubic B-spline.
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    pub control_points: Vec<Vec2>,
}

impl BSpline {
    pub fn new(control_points: Vec<Vec2>) -> Self {
        Self { control_points }
    }

    pub fn segment_count(&self) -> usize {
        if self.control_points.len() < 4 { 0 }
        else { self.control_points.len() - 3 }
    }

    fn eval_segment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f64) -> Vec2 {
        let tt = t * t;
        let ttt = tt * t;
        let inv6 = 1.0 / 6.0;

        let a = p0.scale(-1.0).add(p1.scale(3.0)).add(p2.scale(-3.0)).add(p3).scale(inv6);
        let b = p0.scale(3.0).add(p1.scale(-6.0)).add(p2.scale(3.0)).scale(inv6);
        let c = p0.scale(-3.0).add(p2.scale(3.0)).scale(inv6);
        let d = p0.add(p1.scale(4.0)).add(p2).scale(inv6);

        a.scale(ttt).add(b.scale(tt)).add(c.scale(t)).add(d)
    }

    fn tangent_segment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f64) -> Vec2 {
        let tt = t * t;
        let inv6 = 1.0 / 6.0;

        let a = p0.scale(-1.0).add(p1.scale(3.0)).add(p2.scale(-3.0)).add(p3).scale(inv6);
        let b = p0.scale(3.0).add(p1.scale(-6.0)).add(p2.scale(3.0)).scale(inv6);
        let c = p0.scale(-3.0).add(p2.scale(3.0)).scale(inv6);

        a.scale(3.0 * tt).add(b.scale(2.0 * t)).add(c)
    }

    fn map_parameter(&self, t: f64) -> (usize, f64) {
        let seg_count = self.segment_count();
        if seg_count == 0 { return (0, 0.0); }
        let scaled = t * seg_count as f64;
        let seg = (scaled.floor() as usize).min(seg_count - 1);
        let local = scaled - seg as f64;
        (seg, local.clamp(0.0, 1.0))
    }
}

impl Curve for BSpline {
    fn evaluate(&self, t: f64) -> Vec2 {
        let n = self.control_points.len();
        if n < 4 {
            if n == 0 { return Vec2::zero(); }
            let idx = ((n - 1) as f64 * t).round() as usize;
            return self.control_points[idx.min(n - 1)];
        }
        let (seg, lt) = self.map_parameter(t);
        Self::eval_segment(
            self.control_points[seg],
            self.control_points[seg + 1],
            self.control_points[seg + 2],
            self.control_points[seg + 3],
            lt,
        )
    }

    fn tangent(&self, t: f64) -> Vec2 {
        if self.control_points.len() < 4 { return Vec2::new(1.0, 0.0); }
        let (seg, lt) = self.map_parameter(t);
        Self::tangent_segment(
            self.control_points[seg],
            self.control_points[seg + 1],
            self.control_points[seg + 2],
            self.control_points[seg + 3],
            lt,
        )
    }
}

// ── Polyline (Linear Curve) ──────────────────────────────────

/// A polyline (piecewise-linear curve) for reference/comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct Polyline {
    pub points: Vec<Vec2>,
}

impl Polyline {
    pub fn new(points: Vec<Vec2>) -> Self { Self { points } }

    pub fn total_length(&self) -> f64 {
        let mut len = 0.0;
        for i in 1..self.points.len() {
            len += self.points[i - 1].distance(self.points[i]);
        }
        len
    }
}

impl Curve for Polyline {
    fn evaluate(&self, t: f64) -> Vec2 {
        if self.points.is_empty() { return Vec2::zero(); }
        if self.points.len() == 1 { return self.points[0]; }
        let n = self.points.len() - 1;
        let scaled = t * n as f64;
        let seg = (scaled.floor() as usize).min(n - 1);
        let lt = scaled - seg as f64;
        self.points[seg].lerp(self.points[seg + 1], lt)
    }

    fn tangent(&self, t: f64) -> Vec2 {
        if self.points.len() < 2 { return Vec2::new(1.0, 0.0); }
        let n = self.points.len() - 1;
        let scaled = t * n as f64;
        let seg = (scaled.floor() as usize).min(n - 1);
        self.points[seg + 1].sub(self.points[seg]).normalized()
    }
}

// ── Curve-Curve Intersection ─────────────────────────────────

/// Find approximate intersection points between two curves using recursive subdivision.
pub fn curve_intersections(
    a: &dyn Curve,
    b: &dyn Curve,
    tolerance: f64,
    max_depth: u32,
) -> Vec<(f64, f64, Vec2)> {
    let mut results = Vec::new();
    intersect_recursive(a, 0.0, 1.0, b, 0.0, 1.0, tolerance, max_depth, 0, &mut results);
    results
}

fn intersect_recursive(
    a: &dyn Curve,
    a_lo: f64,
    a_hi: f64,
    b: &dyn Curve,
    b_lo: f64,
    b_hi: f64,
    tolerance: f64,
    max_depth: u32,
    depth: u32,
    results: &mut Vec<(f64, f64, Vec2)>,
) {
    // Check bounding box overlap
    let a_bb = curve_bbox(a, a_lo, a_hi, 8);
    let b_bb = curve_bbox(b, b_lo, b_hi, 8);

    if !bbox_overlap(&a_bb, &b_bb) {
        return;
    }

    let a_mid = (a_lo + a_hi) * 0.5;
    let b_mid = (b_lo + b_hi) * 0.5;

    // If small enough or max depth, check for intersection
    if depth >= max_depth || ((a_hi - a_lo) < tolerance && (b_hi - b_lo) < tolerance) {
        let pa = a.evaluate(a_mid);
        let pb = b.evaluate(b_mid);
        if pa.distance(pb) < tolerance * 10.0 {
            let midpoint = pa.lerp(pb, 0.5);
            results.push((a_mid, b_mid, midpoint));
        }
        return;
    }

    // Subdivide both curves
    intersect_recursive(a, a_lo, a_mid, b, b_lo, b_mid, tolerance, max_depth, depth + 1, results);
    intersect_recursive(a, a_lo, a_mid, b, b_mid, b_hi, tolerance, max_depth, depth + 1, results);
    intersect_recursive(a, a_mid, a_hi, b, b_lo, b_mid, tolerance, max_depth, depth + 1, results);
    intersect_recursive(a, a_mid, a_hi, b, b_mid, b_hi, tolerance, max_depth, depth + 1, results);
}

fn curve_bbox(c: &dyn Curve, lo: f64, hi: f64, samples: u32) -> (Vec2, Vec2) {
    let mut min_p = c.evaluate(lo);
    let mut max_p = min_p;
    for i in 1..=samples {
        let t = lo + (hi - lo) * i as f64 / samples as f64;
        let p = c.evaluate(t);
        min_p.x = min_p.x.min(p.x);
        min_p.y = min_p.y.min(p.y);
        max_p.x = max_p.x.max(p.x);
        max_p.y = max_p.y.max(p.y);
    }
    (min_p, max_p)
}

fn bbox_overlap(a: &(Vec2, Vec2), b: &(Vec2, Vec2)) -> bool {
    a.0.x <= b.1.x && a.1.x >= b.0.x &&
    a.0.y <= b.1.y && a.1.y >= b.0.y
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 0.1 }
    fn approx_v(a: Vec2, b: Vec2) -> bool { approx(a.x, b.x) && approx(a.y, b.y) }

    #[test]
    fn cubic_bezier_endpoints() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx_v(c.evaluate(0.0), Vec2::new(0.0, 0.0)));
        assert!(approx_v(c.evaluate(1.0), Vec2::new(4.0, 0.0)));
    }

    #[test]
    fn cubic_bezier_midpoint() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        let mid = c.evaluate(0.5);
        // Should be approximately at center
        assert!(mid.x > 2.0 && mid.x < 8.0);
        assert!(mid.y > 2.0);
    }

    #[test]
    fn cubic_bezier_tangent() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );
        // Straight line along x: tangent should be in +x direction
        let t = c.tangent(0.5);
        assert!(t.x > 0.0);
        assert!(t.y.abs() < 0.01);
    }

    #[test]
    fn cubic_bezier_normal() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );
        let n = c.normal(0.5);
        // Normal should be perpendicular to tangent
        let t = c.tangent(0.5);
        assert!(t.dot(n).abs() < 0.01);
    }

    #[test]
    fn cubic_bezier_arc_length() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );
        // Straight line from (0,0) to (3,0), length should be 3
        let len = c.arc_length(100);
        assert!(approx(len, 3.0));
    }

    #[test]
    fn cubic_bezier_split() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(4.0, 0.0),
        );
        let (left, right) = c.split(0.5);
        // Left starts at p0, right ends at p3
        assert!(approx_v(left.p0, c.p0));
        assert!(approx_v(right.p3, c.p3));
        // They should meet at the midpoint
        assert!(approx_v(left.p3, right.p0));
        // That point should match the original curve at t=0.5
        let mid = c.evaluate(0.5);
        assert!(approx_v(left.p3, mid));
    }

    #[test]
    fn cubic_bezier_subdivide() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(4.0, 0.0),
        );
        let pts = c.subdivide(10);
        assert_eq!(pts.len(), 10);
        assert!(approx_v(pts[0], c.p0));
        assert!(approx_v(pts[9], c.p3));
    }

    #[test]
    fn cubic_bezier_closest_point() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        // Point at (5, 10) should be closest to the curve near the top
        let (t, pt) = c.closest_point(Vec2::new(5.0, 12.0), 64);
        assert!(t > 0.3 && t < 0.7);
        assert!(pt.y > 5.0);
    }

    #[test]
    fn quadratic_bezier_endpoints() {
        let q = QuadraticBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        assert!(approx_v(q.evaluate(0.0), Vec2::new(0.0, 0.0)));
        assert!(approx_v(q.evaluate(1.0), Vec2::new(10.0, 0.0)));
    }

    #[test]
    fn quadratic_to_cubic() {
        let q = QuadraticBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        let c = q.to_cubic();
        // Should produce the same curve
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let pq = q.evaluate(t);
            let pc = c.evaluate(t);
            assert!(approx_v(pq, pc));
        }
    }

    #[test]
    fn catmull_rom_passes_through_points() {
        let spline = CatmullRomSpline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 3.0),
            Vec2::new(5.0, 1.0),
        ]);
        // t=0 should be at points[1], t=1 at points[2]
        let p0 = spline.evaluate(0.0);
        let p1 = spline.evaluate(1.0);
        assert!(approx_v(p0, Vec2::new(1.0, 2.0)));
        assert!(approx_v(p1, Vec2::new(3.0, 3.0)));
    }

    #[test]
    fn catmull_rom_tangent() {
        let spline = CatmullRomSpline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ]);
        // Straight line: tangent should be in +x
        let t = spline.tangent(0.5);
        assert!(t.x > 0.0);
        assert!(t.y.abs() < 0.01);
    }

    #[test]
    fn bspline_evaluate() {
        let bs = BSpline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(4.0, 0.0),
        ]);
        let p_start = bs.evaluate(0.0);
        let p_end = bs.evaluate(1.0);
        // B-spline doesn't pass through control points (except special cases)
        // but should be within the convex hull
        assert!(p_start.x >= 0.0 && p_start.x <= 4.0);
        assert!(p_end.x >= 0.0 && p_end.x <= 4.0);
    }

    #[test]
    fn bspline_tangent() {
        let bs = BSpline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ]);
        let t = bs.tangent(0.5);
        assert!(t.x > 0.0);
    }

    #[test]
    fn polyline_evaluate() {
        let pl = Polyline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            Vec2::new(10.0, 0.0),
        ]);
        assert!(approx_v(pl.evaluate(0.0), Vec2::new(0.0, 0.0)));
        assert!(approx_v(pl.evaluate(0.5), Vec2::new(5.0, 0.0)));
        assert!(approx_v(pl.evaluate(1.0), Vec2::new(10.0, 0.0)));
    }

    #[test]
    fn polyline_total_length() {
        let pl = Polyline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(3.0, 4.0),
        ]);
        assert!(approx(pl.total_length(), 7.0));
    }

    #[test]
    fn curve_intersection_crossing_lines() {
        // Two crossing straight-line Beziers
        let a = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(3.33, 3.33),
            Vec2::new(6.66, 6.66),
            Vec2::new(10.0, 10.0),
        );
        let b = CubicBezier::new(
            Vec2::new(0.0, 10.0),
            Vec2::new(3.33, 6.66),
            Vec2::new(6.66, 3.33),
            Vec2::new(10.0, 0.0),
        );
        let hits = curve_intersections(&a, &b, 0.01, 12);
        assert!(!hits.is_empty());
        // Should intersect near (5, 5)
        let (_, _, pt) = hits[0];
        assert!(pt.x > 3.0 && pt.x < 7.0);
        assert!(pt.y > 3.0 && pt.y < 7.0);
    }

    #[test]
    fn curve_intersection_no_cross() {
        let a = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );
        let b = CubicBezier::new(
            Vec2::new(0.0, 10.0),
            Vec2::new(1.0, 10.0),
            Vec2::new(2.0, 10.0),
            Vec2::new(3.0, 10.0),
        );
        let hits = curve_intersections(&a, &b, 0.01, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn arc_length_to_partial() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );
        let half_len = c.arc_length_to(0.5, 100);
        let full_len = c.arc_length(100);
        assert!(approx(half_len, full_len * 0.5));
    }

    #[test]
    fn bounding_box_cubic() {
        let c = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 10.0),
            Vec2::new(9.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        let (lo, hi) = c.bounding_box();
        assert!(lo.x <= 0.0);
        assert!(lo.y <= 0.0);
        assert!(hi.x >= 10.0);
        assert!(hi.y >= 10.0);
    }

    #[test]
    fn catmull_rom_with_tension() {
        let spline = CatmullRomSpline::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 5.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 5.0),
        ]).with_tension(0.5);
        // Should still pass through inner points
        let p = spline.evaluate(0.0);
        assert!(approx_v(p, Vec2::new(1.0, 5.0)));
    }
}
