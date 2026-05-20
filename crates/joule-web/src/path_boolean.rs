//! Boolean operations on 2D paths: union, intersection, difference, XOR of closed
//! paths. Represent paths as sequences of line/arc/bezier segments. Winding number
//! tests, segment intersection detection.
//!
//! Pure math — no browser dependency.

use std::fmt;

// ── Point ──────────────────────────────────────────────────────

/// 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn lerp(&self, other: &Point, t: f64) -> Point {
        Point::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Path Segment ───────────────────────────────────────────────

/// A single path segment.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// Line from start to end.
    Line { start: Point, end: Point },
    /// Quadratic bezier.
    QuadBezier { start: Point, control: Point, end: Point },
    /// Cubic bezier.
    CubicBezier { start: Point, ctrl1: Point, ctrl2: Point, end: Point },
    /// Circular arc.
    Arc {
        center: Point,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
    },
}

impl Segment {
    pub fn start_point(&self) -> Point {
        match self {
            Segment::Line { start, .. } => *start,
            Segment::QuadBezier { start, .. } => *start,
            Segment::CubicBezier { start, .. } => *start,
            Segment::Arc { center, radius, start_angle, .. } => {
                Point::new(
                    center.x + radius * start_angle.cos(),
                    center.y + radius * start_angle.sin(),
                )
            }
        }
    }

    pub fn end_point(&self) -> Point {
        match self {
            Segment::Line { end, .. } => *end,
            Segment::QuadBezier { end, .. } => *end,
            Segment::CubicBezier { end, .. } => *end,
            Segment::Arc { center, radius, end_angle, .. } => {
                Point::new(
                    center.x + radius * end_angle.cos(),
                    center.y + radius * end_angle.sin(),
                )
            }
        }
    }

    /// Evaluate the segment at parameter t (0..1).
    pub fn point_at(&self, t: f64) -> Point {
        match self {
            Segment::Line { start, end } => start.lerp(end, t),
            Segment::QuadBezier { start, control, end } => {
                let mt = 1.0 - t;
                Point::new(
                    mt * mt * start.x + 2.0 * mt * t * control.x + t * t * end.x,
                    mt * mt * start.y + 2.0 * mt * t * control.y + t * t * end.y,
                )
            }
            Segment::CubicBezier { start, ctrl1, ctrl2, end } => {
                let mt = 1.0 - t;
                Point::new(
                    mt * mt * mt * start.x + 3.0 * mt * mt * t * ctrl1.x
                        + 3.0 * mt * t * t * ctrl2.x + t * t * t * end.x,
                    mt * mt * mt * start.y + 3.0 * mt * mt * t * ctrl1.y
                        + 3.0 * mt * t * t * ctrl2.y + t * t * t * end.y,
                )
            }
            Segment::Arc { center, radius, start_angle, end_angle } => {
                let angle = start_angle + (end_angle - start_angle) * t;
                Point::new(
                    center.x + radius * angle.cos(),
                    center.y + radius * angle.sin(),
                )
            }
        }
    }

    /// Flatten to line segments for polygon operations.
    pub fn flatten(&self, tolerance: f64) -> Vec<Point> {
        match self {
            Segment::Line { start, end } => vec![*start, *end],
            _ => {
                let steps = (1.0 / tolerance).ceil() as usize;
                let steps = steps.max(4).min(64);
                (0..=steps)
                    .map(|i| self.point_at(i as f64 / steps as f64))
                    .collect()
            }
        }
    }
}

// ── Path ───────────────────────────────────────────────────────

/// A closed 2D path composed of segments.
#[derive(Debug, Clone)]
pub struct Path {
    pub segments: Vec<Segment>,
}

impl Path {
    pub fn new() -> Self {
        Self { segments: Vec::new() }
    }

    pub fn from_segments(segments: Vec<Segment>) -> Self {
        Self { segments }
    }

    /// Create a rectangular path.
    pub fn rect(x: f64, y: f64, w: f64, h: f64) -> Self {
        let tl = Point::new(x, y);
        let tr = Point::new(x + w, y);
        let br = Point::new(x + w, y + h);
        let bl = Point::new(x, y + h);
        Self::from_segments(vec![
            Segment::Line { start: tl, end: tr },
            Segment::Line { start: tr, end: br },
            Segment::Line { start: br, end: bl },
            Segment::Line { start: bl, end: tl },
        ])
    }

    /// Create a circular path.
    pub fn circle(cx: f64, cy: f64, r: f64) -> Self {
        let center = Point::new(cx, cy);
        Self::from_segments(vec![
            Segment::Arc {
                center,
                radius: r,
                start_angle: 0.0,
                end_angle: std::f64::consts::TAU,
            },
        ])
    }

    /// Flatten all segments into a polygon (list of points).
    pub fn to_polygon(&self, tolerance: f64) -> Vec<Point> {
        let mut points = Vec::new();
        for seg in &self.segments {
            let pts = seg.flatten(tolerance);
            if points.is_empty() {
                points.extend_from_slice(&pts);
            } else {
                // Skip first point (duplicate of previous end).
                points.extend_from_slice(&pts[1..]);
            }
        }
        // Remove duplicate closing point if present.
        if points.len() > 1 {
            let first = points[0];
            let last = points[points.len() - 1];
            if first.distance(&last) < 1e-10 {
                points.pop();
            }
        }
        points
    }

    /// Signed area (positive = CCW).
    pub fn signed_area(&self) -> f64 {
        let pts = self.to_polygon(0.1);
        polygon_signed_area(&pts)
    }

    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    pub fn is_clockwise(&self) -> bool {
        self.signed_area() < 0.0
    }
}

// ── Winding Number ─────────────────────────────────────────────

/// Compute the winding number of point `p` with respect to polygon `poly`.
pub fn winding_number(p: &Point, poly: &[Point]) -> i32 {
    let n = poly.len();
    if n < 3 {
        return 0;
    }
    let mut wn = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let (yi, yj) = (poly[i].y, poly[j].y);
        if yi <= p.y {
            if yj > p.y {
                if cross_2d(&poly[i], &poly[j], p) > 0.0 {
                    wn += 1;
                }
            }
        } else if yj <= p.y {
            if cross_2d(&poly[i], &poly[j], p) < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

/// Point-in-polygon test using winding number (nonzero rule).
pub fn point_in_polygon(p: &Point, poly: &[Point]) -> bool {
    winding_number(p, poly) != 0
}

fn cross_2d(a: &Point, b: &Point, p: &Point) -> f64 {
    (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y)
}

// ── Line-Line Intersection ─────────────────────────────────────

/// Intersection of two line segments. Returns parameter t for each segment.
pub fn line_line_intersection(
    a1: &Point, a2: &Point,
    b1: &Point, b2: &Point,
) -> Option<(f64, f64, Point)> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 {
        return None; // parallel
    }

    let dx = b1.x - a1.x;
    let dy = b1.y - a1.y;
    let t = (dx * d2y - dy * d2x) / denom;
    let u = (dx * d1y - dy * d1x) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        let p = Point::new(a1.x + t * d1x, a1.y + t * d1y);
        Some((t, u, p))
    } else {
        None
    }
}

/// Find all intersection points between two polygons.
pub fn polygon_intersections(poly_a: &[Point], poly_b: &[Point]) -> Vec<Point> {
    let mut intersections = Vec::new();
    let na = poly_a.len();
    let nb = poly_b.len();

    for i in 0..na {
        let i_next = (i + 1) % na;
        for j in 0..nb {
            let j_next = (j + 1) % nb;
            if let Some((_, _, pt)) = line_line_intersection(
                &poly_a[i], &poly_a[i_next],
                &poly_b[j], &poly_b[j_next],
            ) {
                intersections.push(pt);
            }
        }
    }
    intersections
}

fn polygon_signed_area(pts: &[Point]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    area / 2.0
}

// ── Boolean Operations ─────────────────────────────────────────

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
    Xor,
}

/// Result of a boolean operation — a set of polygons.
#[derive(Debug, Clone)]
pub struct BooleanResult {
    pub polygons: Vec<Vec<Point>>,
}

impl BooleanResult {
    pub fn total_area(&self) -> f64 {
        self.polygons
            .iter()
            .map(|p| polygon_signed_area(p).abs())
            .sum()
    }

    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }
}

/// Perform a boolean operation on two paths.
/// Uses Sutherland-Hodgman for convex polygons and a point-classification
/// approach for general polygons.
pub fn boolean_op(path_a: &Path, path_b: &Path, op: BooleanOp) -> BooleanResult {
    let poly_a = path_a.to_polygon(0.05);
    let poly_b = path_b.to_polygon(0.05);

    match op {
        BooleanOp::Intersection => {
            let clipped = sutherland_hodgman_clip(&poly_a, &poly_b);
            if clipped.len() >= 3 {
                BooleanResult { polygons: vec![clipped] }
            } else {
                BooleanResult { polygons: vec![] }
            }
        }
        BooleanOp::Union => {
            // Approximate: if they overlap, combine outlines; otherwise return both.
            let intersections = polygon_intersections(&poly_a, &poly_b);
            if intersections.is_empty() {
                // Check containment.
                if point_in_polygon(&poly_a[0], &poly_b) {
                    BooleanResult { polygons: vec![poly_b] }
                } else if point_in_polygon(&poly_b[0], &poly_a) {
                    BooleanResult { polygons: vec![poly_a] }
                } else {
                    BooleanResult { polygons: vec![poly_a, poly_b] }
                }
            } else {
                // Approximate union: merge outline points classified as outside the other.
                let mut result = Vec::new();
                for pt in &poly_a {
                    if !point_in_polygon(pt, &poly_b) {
                        result.push(*pt);
                    }
                }
                for pt in &poly_b {
                    if !point_in_polygon(pt, &poly_a) {
                        result.push(*pt);
                    }
                }
                result.extend(intersections);
                if result.len() >= 3 {
                    sort_points_ccw(&mut result);
                    BooleanResult { polygons: vec![result] }
                } else {
                    BooleanResult { polygons: vec![poly_a, poly_b] }
                }
            }
        }
        BooleanOp::Difference => {
            // A minus B: points of A not in B + intersection boundary.
            let mut result = Vec::new();
            for pt in &poly_a {
                if !point_in_polygon(pt, &poly_b) {
                    result.push(*pt);
                }
            }
            let ints = polygon_intersections(&poly_a, &poly_b);
            result.extend(ints);
            if result.len() >= 3 {
                sort_points_ccw(&mut result);
                BooleanResult { polygons: vec![result] }
            } else {
                BooleanResult { polygons: vec![poly_a] }
            }
        }
        BooleanOp::Xor => {
            // XOR = Union minus Intersection (approximate).
            let intersection = sutherland_hodgman_clip(&poly_a, &poly_b);
            let mut result_a = Vec::new();
            let mut result_b = Vec::new();
            for pt in &poly_a {
                if !point_in_polygon(pt, &poly_b) {
                    result_a.push(*pt);
                }
            }
            for pt in &poly_b {
                if !point_in_polygon(pt, &poly_a) {
                    result_b.push(*pt);
                }
            }
            let mut polys = Vec::new();
            if result_a.len() >= 3 {
                sort_points_ccw(&mut result_a);
                polys.push(result_a);
            }
            if result_b.len() >= 3 {
                sort_points_ccw(&mut result_b);
                polys.push(result_b);
            }
            BooleanResult { polygons: polys }
        }
    }
}

/// Sort points in CCW order around their centroid.
fn sort_points_ccw(pts: &mut Vec<Point>) {
    if pts.len() < 3 {
        return;
    }
    let cx: f64 = pts.iter().map(|p| p.x).sum::<f64>() / pts.len() as f64;
    let cy: f64 = pts.iter().map(|p| p.y).sum::<f64>() / pts.len() as f64;
    pts.sort_by(|a, b| {
        let angle_a = (a.y - cy).atan2(a.x - cx);
        let angle_b = (b.y - cy).atan2(b.x - cx);
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sutherland-Hodgman polygon clipping (clip `subject` against `clip` polygon).
pub fn sutherland_hodgman_clip(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    if subject.is_empty() || clip.is_empty() {
        return vec![];
    }
    let mut output = subject.to_vec();
    let n = clip.len();

    for i in 0..n {
        if output.is_empty() {
            break;
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % n];
        let input = output;
        output = Vec::new();
        let m = input.len();

        for j in 0..m {
            let current = input[j];
            let previous = input[(j + m - 1) % m];
            let curr_inside = is_inside(&current, &edge_start, &edge_end);
            let prev_inside = is_inside(&previous, &edge_start, &edge_end);

            if curr_inside {
                if !prev_inside {
                    if let Some(inter) = line_intersect_unbounded(&previous, &current, &edge_start, &edge_end) {
                        output.push(inter);
                    }
                }
                output.push(current);
            } else if prev_inside {
                if let Some(inter) = line_intersect_unbounded(&previous, &current, &edge_start, &edge_end) {
                    output.push(inter);
                }
            }
        }
    }
    output
}

fn is_inside(p: &Point, edge_start: &Point, edge_end: &Point) -> bool {
    cross_2d(edge_start, edge_end, p) >= 0.0
}

fn line_intersect_unbounded(a1: &Point, a2: &Point, b1: &Point, b2: &Point) -> Option<Point> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let dx = b1.x - a1.x;
    let dy = b1.y - a1.y;
    let t = (dx * d2y - dy * d2x) / denom;
    Some(Point::new(a1.x + t * d1x, a1.y + t * d1y))
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Path {
        Path::rect(x, y, s, s)
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_segment_at() {
        let seg = Segment::Line {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
        };
        let mid = seg.point_at(0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rect_area() {
        let r = Path::rect(0.0, 0.0, 10.0, 5.0);
        assert!((r.area() - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_winding_number_inside() {
        let sq = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        assert_ne!(winding_number(&Point::new(5.0, 5.0), &sq), 0);
    }

    #[test]
    fn test_winding_number_outside() {
        let sq = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        assert_eq!(winding_number(&Point::new(15.0, 5.0), &sq), 0);
    }

    #[test]
    fn test_line_intersection() {
        let r = line_line_intersection(
            &Point::new(0.0, 0.0),
            &Point::new(10.0, 10.0),
            &Point::new(0.0, 10.0),
            &Point::new(10.0, 0.0),
        );
        let (_, _, pt) = r.unwrap();
        assert!((pt.x - 5.0).abs() < 1e-10);
        assert!((pt.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_intersection_parallel() {
        let r = line_line_intersection(
            &Point::new(0.0, 0.0),
            &Point::new(10.0, 0.0),
            &Point::new(0.0, 1.0),
            &Point::new(10.0, 1.0),
        );
        assert!(r.is_none());
    }

    #[test]
    fn test_boolean_intersection_overlapping() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(5.0, 5.0, 10.0);
        let result = boolean_op(&a, &b, BooleanOp::Intersection);
        assert!(!result.polygons.is_empty());
        let area = result.total_area();
        assert!(area > 0.0);
        // Intersection of two overlapping 10×10 squares offset by 5 should be ~25.
        assert!((area - 25.0).abs() < 5.0);
    }

    #[test]
    fn test_boolean_intersection_disjoint() {
        let a = square(0.0, 0.0, 5.0);
        let b = square(20.0, 20.0, 5.0);
        let result = boolean_op(&a, &b, BooleanOp::Intersection);
        assert!(result.polygons.is_empty() || result.total_area() < 1.0);
    }

    #[test]
    fn test_boolean_union_disjoint() {
        let a = square(0.0, 0.0, 5.0);
        let b = square(20.0, 20.0, 5.0);
        let result = boolean_op(&a, &b, BooleanOp::Union);
        assert_eq!(result.polygon_count(), 2);
    }

    #[test]
    fn test_boolean_difference() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(5.0, 5.0, 10.0);
        let result = boolean_op(&a, &b, BooleanOp::Difference);
        assert!(!result.polygons.is_empty());
        // Difference area should be less than original.
        assert!(result.total_area() < 100.0);
    }

    #[test]
    fn test_quadratic_bezier() {
        let seg = Segment::QuadBezier {
            start: Point::new(0.0, 0.0),
            control: Point::new(5.0, 10.0),
            end: Point::new(10.0, 0.0),
        };
        let mid = seg.point_at(0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!(mid.y > 0.0); // should be above x-axis
    }

    #[test]
    fn test_flatten_line() {
        let seg = Segment::Line {
            start: Point::new(0.0, 0.0),
            end: Point::new(5.0, 5.0),
        };
        let pts = seg.flatten(0.1);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn test_polygon_intersections() {
        let sq_a = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let sq_b = vec![
            Point::new(5.0, 5.0),
            Point::new(15.0, 5.0),
            Point::new(15.0, 15.0),
            Point::new(5.0, 15.0),
        ];
        let ints = polygon_intersections(&sq_a, &sq_b);
        assert_eq!(ints.len(), 2);
    }
}
