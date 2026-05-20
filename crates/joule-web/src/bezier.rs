//! Bezier curves — quadratic, cubic, paths, and utilities.
//!
//! Pure-Rust replacement for bezier.js, Paper.js path operations, and similar
//! JS curve libraries.

use std::fmt;

// ── Point type ─────────────────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4})", self.x, self.y)
    }
}

// ── Bounding box ───────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BBox {
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
}

// ── Quadratic Bezier ───────────────────────────────────────────

/// Quadratic Bezier curve: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticBezier {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
}

impl QuadraticBezier {
    pub fn new(p0: Point, p1: Point, p2: Point) -> Self {
        Self { p0, p1, p2 }
    }

    /// Evaluate point at parameter t in [0, 1] using De Casteljau.
    pub fn point_at(&self, t: f64) -> Point {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t);
        a.lerp(b, t)
    }

    /// Derivative (tangent vector) at parameter t.
    pub fn derivative_at(&self, t: f64) -> Point {
        let one_minus_t = 1.0 - t;
        Point {
            x: 2.0 * one_minus_t * (self.p1.x - self.p0.x) + 2.0 * t * (self.p2.x - self.p1.x),
            y: 2.0 * one_minus_t * (self.p1.y - self.p0.y) + 2.0 * t * (self.p2.y - self.p1.y),
        }
    }

    /// Split into two sub-curves at parameter t.
    pub fn split_at(&self, t: f64) -> (QuadraticBezier, QuadraticBezier) {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t);
        let mid = a.lerp(b, t);
        (
            QuadraticBezier::new(self.p0, a, mid),
            QuadraticBezier::new(mid, b, self.p2),
        )
    }

    /// Axis-aligned bounding box (conservative: uses control points).
    pub fn bounding_box(&self) -> BBox {
        let min_x = self.p0.x.min(self.p1.x).min(self.p2.x);
        let min_y = self.p0.y.min(self.p1.y).min(self.p2.y);
        let max_x = self.p0.x.max(self.p1.x).max(self.p2.x);
        let max_y = self.p0.y.max(self.p1.y).max(self.p2.y);
        BBox {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Approximate arc length by subdivision.
    pub fn arc_length(&self, segments: usize) -> f64 {
        let n = segments.max(1);
        let mut length = 0.0;
        let mut prev = self.point_at(0.0);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let curr = self.point_at(t);
            length += prev.distance(curr);
            prev = curr;
        }
        length
    }
}

// ── Cubic Bezier ───────────────────────────────────────────────

/// Cubic Bezier curve: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3.
#[derive(Debug, Clone, Copy)]
pub struct CubicBezier {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
}

impl CubicBezier {
    pub fn new(p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluate point at parameter t using De Casteljau.
    pub fn point_at(&self, t: f64) -> Point {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t);
        let c = self.p2.lerp(self.p3, t);
        let d = a.lerp(b, t);
        let e = b.lerp(c, t);
        d.lerp(e, t)
    }

    /// Derivative (tangent vector) at parameter t.
    pub fn derivative_at(&self, t: f64) -> Point {
        let t2 = t * t;
        let one_t = 1.0 - t;
        let one_t2 = one_t * one_t;
        Point {
            x: 3.0 * one_t2 * (self.p1.x - self.p0.x)
                + 6.0 * one_t * t * (self.p2.x - self.p1.x)
                + 3.0 * t2 * (self.p3.x - self.p2.x),
            y: 3.0 * one_t2 * (self.p1.y - self.p0.y)
                + 6.0 * one_t * t * (self.p2.y - self.p1.y)
                + 3.0 * t2 * (self.p3.y - self.p2.y),
        }
    }

    /// Split into two cubic sub-curves at parameter t.
    pub fn split_at(&self, t: f64) -> (CubicBezier, CubicBezier) {
        let a = self.p0.lerp(self.p1, t);
        let b = self.p1.lerp(self.p2, t);
        let c = self.p2.lerp(self.p3, t);
        let d = a.lerp(b, t);
        let e = b.lerp(c, t);
        let mid = d.lerp(e, t);
        (
            CubicBezier::new(self.p0, a, d, mid),
            CubicBezier::new(mid, e, c, self.p3),
        )
    }

    /// Axis-aligned bounding box (conservative: control-point hull).
    pub fn bounding_box(&self) -> BBox {
        let min_x = self.p0.x.min(self.p1.x).min(self.p2.x).min(self.p3.x);
        let min_y = self.p0.y.min(self.p1.y).min(self.p2.y).min(self.p3.y);
        let max_x = self.p0.x.max(self.p1.x).max(self.p2.x).max(self.p3.x);
        let max_y = self.p0.y.max(self.p1.y).max(self.p2.y).max(self.p3.y);
        BBox {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Approximate arc length using adaptive subdivision.
    pub fn arc_length(&self, segments: usize) -> f64 {
        let n = segments.max(1);
        let mut length = 0.0;
        let mut prev = self.point_at(0.0);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let curr = self.point_at(t);
            length += prev.distance(curr);
            prev = curr;
        }
        length
    }

    /// Arc length via recursive subdivision until error < tolerance.
    pub fn arc_length_adaptive(&self, tolerance: f64) -> f64 {
        fn subdivide(curve: &CubicBezier, tolerance: f64, depth: usize) -> f64 {
            let chord = curve.p0.distance(curve.p3);
            let control = curve.p0.distance(curve.p1)
                + curve.p1.distance(curve.p2)
                + curve.p2.distance(curve.p3);
            if depth > 16 || (control - chord).abs() < tolerance {
                return (chord + control) / 2.0;
            }
            let (left, right) = curve.split_at(0.5);
            subdivide(&left, tolerance, depth + 1) + subdivide(&right, tolerance, depth + 1)
        }
        subdivide(self, tolerance, 0)
    }

    /// Find the closest point on the curve to a given point (brute force + refinement).
    pub fn closest_point(&self, target: Point, initial_samples: usize) -> (f64, Point) {
        let n = initial_samples.max(10);
        let mut best_t = 0.0;
        let mut best_dist = f64::INFINITY;

        // Coarse search
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let p = self.point_at(t);
            let d = target.distance(p);
            if d < best_dist {
                best_dist = d;
                best_t = t;
            }
        }

        // Refine with bisection-like search
        let mut lo = (best_t - 1.0 / n as f64).max(0.0);
        let mut hi = (best_t + 1.0 / n as f64).min(1.0);
        for _ in 0..20 {
            let t1 = lo + (hi - lo) / 3.0;
            let t2 = hi - (hi - lo) / 3.0;
            let d1 = target.distance(self.point_at(t1));
            let d2 = target.distance(self.point_at(t2));
            if d1 < d2 {
                hi = t2;
            } else {
                lo = t1;
            }
        }
        let t_final = (lo + hi) / 2.0;
        (t_final, self.point_at(t_final))
    }

    /// Flatten curve to a polyline using recursive subdivision.
    pub fn flatten(&self, tolerance: f64) -> Vec<Point> {
        let mut result = vec![self.p0];
        flatten_recursive(self, tolerance, 0, &mut result);
        result
    }
}

fn flatten_recursive(curve: &CubicBezier, tolerance: f64, depth: usize, output: &mut Vec<Point>) {
    if depth > 16 {
        output.push(curve.p3);
        return;
    }
    // Check if the control points are close enough to the chord
    let d1 = point_to_line_dist(curve.p1, curve.p0, curve.p3);
    let d2 = point_to_line_dist(curve.p2, curve.p0, curve.p3);
    if d1 <= tolerance && d2 <= tolerance {
        output.push(curve.p3);
    } else {
        let (left, right) = curve.split_at(0.5);
        flatten_recursive(&left, tolerance, depth + 1, output);
        flatten_recursive(&right, tolerance, depth + 1, output);
    }
}

/// Distance from point `p` to line segment `a`-`b`.
fn point_to_line_dist(p: Point, a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        return p.distance(a);
    }
    let cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
    cross.abs() / len_sq.sqrt()
}

// ── Bezier path ────────────────────────────────────────────────

/// A path composed of chained cubic Bezier segments.
#[derive(Debug, Clone)]
pub struct BezierPath {
    pub segments: Vec<CubicBezier>,
}

impl BezierPath {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn push(&mut self, segment: CubicBezier) {
        self.segments.push(segment);
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Total arc length.
    pub fn arc_length(&self, segments_per_curve: usize) -> f64 {
        self.segments
            .iter()
            .map(|s| s.arc_length(segments_per_curve))
            .sum()
    }

    /// Evaluate point at parameter t in [0, N] where N = number of segments.
    pub fn point_at(&self, t: f64) -> Option<Point> {
        if self.segments.is_empty() {
            return None;
        }
        let n = self.segments.len() as f64;
        let t_clamped = t.clamp(0.0, n);
        let idx = (t_clamped.floor() as usize).min(self.segments.len() - 1);
        let local_t = t_clamped - idx as f64;
        Some(self.segments[idx].point_at(local_t.min(1.0)))
    }

    /// Bounding box of entire path.
    pub fn bounding_box(&self) -> Option<BBox> {
        if self.segments.is_empty() {
            return None;
        }
        let first = self.segments[0].bounding_box();
        let mut bb = first;
        for seg in &self.segments[1..] {
            let b = seg.bounding_box();
            bb.min_x = bb.min_x.min(b.min_x);
            bb.min_y = bb.min_y.min(b.min_y);
            bb.max_x = bb.max_x.max(b.max_x);
            bb.max_y = bb.max_y.max(b.max_y);
        }
        Some(bb)
    }

    /// Flatten all segments to a polyline.
    pub fn flatten(&self, tolerance: f64) -> Vec<Point> {
        if self.segments.is_empty() {
            return vec![];
        }
        let mut pts = vec![self.segments[0].p0];
        for seg in &self.segments {
            flatten_recursive(seg, tolerance, 0, &mut pts);
        }
        pts
    }
}

impl Default for BezierPath {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn pt_approx(a: Point, b: Point) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    #[test]
    fn quadratic_endpoints() {
        let q = QuadraticBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 0.0),
        );
        assert!(pt_approx(q.point_at(0.0), Point::new(0.0, 0.0)));
        assert!(pt_approx(q.point_at(1.0), Point::new(2.0, 0.0)));
    }

    #[test]
    fn quadratic_midpoint() {
        let q = QuadraticBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.5, 1.0),
            Point::new(1.0, 0.0),
        );
        let mid = q.point_at(0.5);
        // At t=0.5: x = 0.5, y = 0.5
        assert!(approx(mid.x, 0.5));
        assert!(approx(mid.y, 0.5));
    }

    #[test]
    fn quadratic_split() {
        let q = QuadraticBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 0.0),
        );
        let (left, right) = q.split_at(0.5);
        let mid = q.point_at(0.5);
        assert!(pt_approx(left.point_at(1.0), mid));
        assert!(pt_approx(right.point_at(0.0), mid));
    }

    #[test]
    fn cubic_endpoints() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        );
        assert!(pt_approx(c.point_at(0.0), Point::new(0.0, 0.0)));
        assert!(pt_approx(c.point_at(1.0), Point::new(1.0, 0.0)));
    }

    #[test]
    fn cubic_derivative() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        );
        // Straight line along x: derivative at any t should have y=0
        let d = c.derivative_at(0.5);
        assert!(approx(d.y, 0.0));
        assert!(d.x > 0.0);
    }

    #[test]
    fn cubic_split_consistency() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        );
        let (left, right) = c.split_at(0.3);
        let original = c.point_at(0.3);
        assert!(pt_approx(left.point_at(1.0), original));
        assert!(pt_approx(right.point_at(0.0), original));
    }

    #[test]
    fn cubic_bounding_box() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(-1.0, 2.0),
            Point::new(3.0, 2.0),
            Point::new(2.0, 0.0),
        );
        let bb = c.bounding_box();
        assert!(bb.min_x <= 0.0);
        assert!(bb.max_x >= 2.0);
        assert!(bb.min_y <= 0.0);
        assert!(bb.max_y >= 2.0);
    }

    #[test]
    fn cubic_arc_length_straight_line() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        );
        let len = c.arc_length(100);
        assert!((len - 3.0).abs() < 0.01);
    }

    #[test]
    fn cubic_closest_point() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        );
        let (t, p) = c.closest_point(Point::new(1.5, 1.0), 100);
        assert!((p.x - 1.5).abs() < 0.05);
        assert!(p.y.abs() < 0.05);
        assert!(t > 0.0 && t < 1.0);
    }

    #[test]
    fn cubic_flatten() {
        let c = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        );
        let pts = c.flatten(0.01);
        assert!(pts.len() >= 3);
        assert!(pt_approx(pts[0], Point::new(0.0, 0.0)));
        assert!(pt_approx(*pts.last().unwrap(), Point::new(1.0, 0.0)));
    }

    #[test]
    fn bezier_path_basic() {
        let mut path = BezierPath::new();
        path.push(CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 1.0),
            Point::new(3.0, 0.0),
        ));
        path.push(CubicBezier::new(
            Point::new(3.0, 0.0),
            Point::new(4.0, -1.0),
            Point::new(5.0, -1.0),
            Point::new(6.0, 0.0),
        ));
        assert_eq!(path.len(), 2);
        let p = path.point_at(0.0).unwrap();
        assert!(pt_approx(p, Point::new(0.0, 0.0)));
        let p_end = path.point_at(2.0).unwrap();
        assert!(pt_approx(p_end, Point::new(6.0, 0.0)));
    }

    #[test]
    fn quadratic_arc_length() {
        // Straight line quadratic
        let q = QuadraticBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        );
        let len = q.arc_length(100);
        assert!((len - 2.0).abs() < 0.01);
    }
}
