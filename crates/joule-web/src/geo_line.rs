//! Line operations — line intersection, line-polygon intersection, line buffering,
//! line simplification, line splitting, polyline length, line offset (parallel),
//! LineString builder.
//!
//! All geometry uses `f64` coordinates. Lines are represented as ordered sequences
//! of `Point2` vertices forming polylines (open paths).

use std::fmt;

// ── Point2 ─────────────────────────────────────────────────────

/// A 2-D point in Cartesian space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dist_sq(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    pub fn dist(self, other: Self) -> f64 {
        self.dist_sq(other).sqrt()
    }
}

impl fmt::Display for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6})", self.x, self.y)
    }
}

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LineError {
    TooFewPoints(usize),
    InvalidEpsilon(f64),
    InvalidDistance(f64),
    ParallelLines,
    EmptyInput,
}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints(n) => write!(f, "line needs >=2 points, got {n}"),
            Self::InvalidEpsilon(e) => write!(f, "epsilon must be positive, got {e}"),
            Self::InvalidDistance(d) => write!(f, "distance must be positive, got {d}"),
            Self::ParallelLines => write!(f, "lines are parallel"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for LineError {}

// ── Segment intersection ───────────────────────────────────────

/// Result of a segment-segment intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentIntersection {
    /// Segments intersect at a single point.
    Point(Point2),
    /// Segments are collinear and overlap.
    Collinear,
    /// Segments do not intersect.
    None,
}

impl fmt::Display for SegmentIntersection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Point(p) => write!(f, "intersection at {p}"),
            Self::Collinear => write!(f, "collinear overlap"),
            Self::None => write!(f, "no intersection"),
        }
    }
}

/// Test two line segments `(a1–a2)` and `(b1–b2)` for intersection.
pub fn segment_intersect(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> SegmentIntersection {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;
    let denom = d1x * d2y - d1y * d2x;

    let dx = b1.x - a1.x;
    let dy = b1.y - a1.y;

    if denom.abs() < 1e-15 {
        // Check collinearity.
        let cross = dx * d1y - dy * d1x;
        if cross.abs() < 1e-15 {
            return SegmentIntersection::Collinear;
        }
        return SegmentIntersection::None;
    }

    let t = (dx * d2y - dy * d2x) / denom;
    let u = (dx * d1y - dy * d1x) / denom;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        SegmentIntersection::Point(Point2::new(a1.x + t * d1x, a1.y + t * d1y))
    } else {
        SegmentIntersection::None
    }
}

/// Infinite-line intersection of lines through `(a1–a2)` and `(b1–b2)`.
pub fn line_intersect(
    a1: Point2,
    a2: Point2,
    b1: Point2,
    b2: Point2,
) -> Result<Point2, LineError> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-15 {
        return Err(LineError::ParallelLines);
    }
    let t = ((b1.x - a1.x) * d2y - (b1.y - a1.y) * d2x) / denom;
    Ok(Point2::new(a1.x + t * d1x, a1.y + t * d1y))
}

// ── LineString ─────────────────────────────────────────────────

/// An open polyline defined by an ordered sequence of points.
#[derive(Debug, Clone, PartialEq)]
pub struct LineString {
    pub points: Vec<Point2>,
}

impl LineString {
    pub fn new(points: Vec<Point2>) -> Result<Self, LineError> {
        if points.len() < 2 {
            return Err(LineError::TooFewPoints(points.len()));
        }
        Ok(Self { points })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Total length along the polyline.
    pub fn length(&self) -> f64 {
        let mut total = 0.0;
        for i in 1..self.points.len() {
            total += self.points[i - 1].dist(self.points[i]);
        }
        total
    }

    /// Bounding box `(min, max)`.
    pub fn bbox(&self) -> (Point2, Point2) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for v in &self.points {
            if v.x < min_x { min_x = v.x; }
            if v.y < min_y { min_y = v.y; }
            if v.x > max_x { max_x = v.x; }
            if v.y > max_y { max_y = v.y; }
        }
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Douglas-Peucker simplification of the line.
    pub fn simplify(&self, epsilon: f64) -> Result<LineString, LineError> {
        if epsilon <= 0.0 {
            return Err(LineError::InvalidEpsilon(epsilon));
        }
        let mut out = Vec::new();
        dp_recurse(&self.points, epsilon, &mut out);
        out.push(*self.points.last().unwrap());
        if out.len() < 2 {
            return Ok(self.clone());
        }
        LineString::new(out)
    }

    /// Split the polyline at a given distance along it.
    pub fn split_at_distance(&self, dist: f64) -> Result<(LineString, LineString), LineError> {
        if dist <= 0.0 {
            return Err(LineError::InvalidDistance(dist));
        }
        let total = self.length();
        if dist >= total {
            return Err(LineError::InvalidDistance(dist));
        }
        let mut acc = 0.0;
        for i in 1..self.points.len() {
            let seg_len = self.points[i - 1].dist(self.points[i]);
            if acc + seg_len >= dist {
                let frac = (dist - acc) / seg_len;
                let split_pt = Point2::new(
                    self.points[i - 1].x + frac * (self.points[i].x - self.points[i - 1].x),
                    self.points[i - 1].y + frac * (self.points[i].y - self.points[i - 1].y),
                );
                let mut first = self.points[..i].to_vec();
                first.push(split_pt);
                let mut second = vec![split_pt];
                second.extend_from_slice(&self.points[i..]);
                return Ok((LineString::new(first)?, LineString::new(second)?));
            }
            acc += seg_len;
        }
        Err(LineError::InvalidDistance(dist))
    }

    /// Create an offset (parallel) line at the given distance.
    /// Positive distance offsets to the left, negative to the right.
    pub fn offset(&self, distance: f64) -> Result<LineString, LineError> {
        if distance.abs() < 1e-15 {
            return Ok(self.clone());
        }
        let n = self.points.len();
        let mut normals: Vec<(f64, f64)> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dx = self.points[i + 1].x - self.points[i].x;
            let dy = self.points[i + 1].y - self.points[i].y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-15 {
                normals.push((0.0, 0.0));
            } else {
                normals.push((-dy / len, dx / len));
            }
        }

        let mut result = Vec::with_capacity(n);
        // First point.
        result.push(Point2::new(
            self.points[0].x + normals[0].0 * distance,
            self.points[0].y + normals[0].1 * distance,
        ));
        // Interior points: average adjacent normals.
        for i in 1..n - 1 {
            let nx = (normals[i - 1].0 + normals[i].0) * 0.5;
            let ny = (normals[i - 1].1 + normals[i].1) * 0.5;
            let nlen = (nx * nx + ny * ny).sqrt();
            if nlen < 1e-15 {
                result.push(self.points[i]);
            } else {
                result.push(Point2::new(
                    self.points[i].x + (nx / nlen) * distance,
                    self.points[i].y + (ny / nlen) * distance,
                ));
            }
        }
        // Last point.
        let last_n = normals[n - 2];
        result.push(Point2::new(
            self.points[n - 1].x + last_n.0 * distance,
            self.points[n - 1].y + last_n.1 * distance,
        ));

        LineString::new(result)
    }

    /// Nearest point on the polyline to the given query point.
    pub fn nearest_point(&self, query: Point2) -> Point2 {
        let mut best = self.points[0];
        let mut best_d = f64::INFINITY;
        for i in 0..self.points.len() - 1 {
            let p = nearest_point_on_segment(query, self.points[i], self.points[i + 1]);
            let d = query.dist_sq(p);
            if d < best_d {
                best_d = d;
                best = p;
            }
        }
        best
    }

    /// Compute intersections of this polyline with a line segment `(a–b)`.
    pub fn intersect_segment(&self, a: Point2, b: Point2) -> Vec<Point2> {
        let mut hits = Vec::new();
        for i in 0..self.points.len() - 1 {
            if let SegmentIntersection::Point(p) =
                segment_intersect(self.points[i], self.points[i + 1], a, b)
            {
                hits.push(p);
            }
        }
        hits
    }
}

impl fmt::Display for LineString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LineString({} pts, len={:.4})", self.points.len(), self.length())
    }
}

// ── Nearest point on segment ───────────────────────────────────

/// Project `p` onto segment `a–b`, clamping to the segment extent.
pub fn nearest_point_on_segment(p: Point2, a: Point2, b: Point2) -> Point2 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-15 {
        return a;
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    Point2::new(a.x + t * dx, a.y + t * dy)
}

// ── Perpendicular distance ─────────────────────────────────────

fn point_line_dist(p: Point2, a: Point2, b: Point2) -> f64 {
    let proj = nearest_point_on_segment(p, a, b);
    p.dist(proj)
}

fn dp_recurse(pts: &[Point2], eps: f64, out: &mut Vec<Point2>) {
    if pts.len() < 2 {
        return;
    }
    let first = pts[0];
    let last = pts[pts.len() - 1];
    let mut max_d = 0.0_f64;
    let mut idx = 0;
    for i in 1..pts.len() - 1 {
        let d = point_line_dist(pts[i], first, last);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > eps {
        dp_recurse(&pts[..=idx], eps, out);
        dp_recurse(&pts[idx..], eps, out);
    } else {
        out.push(first);
    }
}

// ── LineStringBuilder ──────────────────────────────────────────

/// Builder for constructing polylines point by point.
#[derive(Debug, Clone)]
pub struct LineStringBuilder {
    points: Vec<Point2>,
}

impl LineStringBuilder {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn with_point(mut self, x: f64, y: f64) -> Self {
        self.points.push(Point2::new(x, y));
        self
    }

    pub fn with_points(mut self, pts: &[(f64, f64)]) -> Self {
        for &(x, y) in pts {
            self.points.push(Point2::new(x, y));
        }
        self
    }

    pub fn build(self) -> Result<LineString, LineError> {
        LineString::new(self.points)
    }
}

impl Default for LineStringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LineStringBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LineStringBuilder({} points queued)", self.points.len())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn sample_line() -> LineString {
        LineString::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        ])
        .unwrap()
    }

    #[test]
    fn test_point2_display() {
        let p = Point2::new(3.14, 2.71);
        assert!(format!("{p}").contains("3.14"));
    }

    #[test]
    fn test_linestring_length() {
        let ls = sample_line();
        assert!(approx(ls.length(), 3.0));
    }

    #[test]
    fn test_linestring_display() {
        let ls = sample_line();
        let s = format!("{ls}");
        assert!(s.contains("4 pts"));
    }

    #[test]
    fn test_segment_intersect_cross() {
        let r = segment_intersect(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
            Point2::new(2.0, 0.0),
        );
        if let SegmentIntersection::Point(p) = r {
            assert!(approx(p.x, 1.0));
            assert!(approx(p.y, 1.0));
        } else {
            panic!("expected intersection point");
        }
    }

    #[test]
    fn test_segment_no_intersect() {
        let r = segment_intersect(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        );
        assert_eq!(r, SegmentIntersection::None);
    }

    #[test]
    fn test_segment_collinear() {
        let r = segment_intersect(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(3.0, 0.0),
        );
        assert_eq!(r, SegmentIntersection::Collinear);
    }

    #[test]
    fn test_line_intersect_basic() {
        let p = line_intersect(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 0.0),
        )
        .unwrap();
        assert!(approx(p.x, 0.5));
    }

    #[test]
    fn test_line_intersect_parallel() {
        let r = line_intersect(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_nearest_point_on_segment() {
        let p = nearest_point_on_segment(
            Point2::new(0.5, 1.0),
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
        );
        assert!(approx(p.x, 0.5));
        assert!(approx(p.y, 0.0));
    }

    #[test]
    fn test_linestring_nearest_point() {
        let ls = sample_line();
        let np = ls.nearest_point(Point2::new(1.5, 1.0));
        assert!(approx(np.x, 1.5));
        assert!(approx(np.y, 0.0));
    }

    #[test]
    fn test_linestring_split() {
        let ls = sample_line();
        let (a, b) = ls.split_at_distance(1.5).unwrap();
        assert!(approx(a.length(), 1.5));
        assert!(approx(b.length(), 1.5));
    }

    #[test]
    fn test_linestring_split_invalid() {
        let ls = sample_line();
        assert!(ls.split_at_distance(0.0).is_err());
        assert!(ls.split_at_distance(5.0).is_err());
    }

    #[test]
    fn test_linestring_simplify() {
        let ls = sample_line();
        let simplified = ls.simplify(0.1).unwrap();
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_linestring_offset() {
        let ls = LineString::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
        ])
        .unwrap();
        let off = ls.offset(1.0).unwrap();
        assert!(approx(off.points[0].y, 1.0));
        assert!(approx(off.points[1].y, 1.0));
    }

    #[test]
    fn test_linestring_bbox() {
        let ls = sample_line();
        let (lo, hi) = ls.bbox();
        assert!(approx(lo.x, 0.0));
        assert!(approx(hi.x, 3.0));
    }

    #[test]
    fn test_linestring_too_few() {
        let r = LineString::new(vec![Point2::new(0.0, 0.0)]);
        assert!(r.is_err());
    }

    #[test]
    fn test_intersect_segment() {
        let ls = sample_line();
        let hits = ls.intersect_segment(
            Point2::new(1.5, -1.0),
            Point2::new(1.5, 1.0),
        );
        assert_eq!(hits.len(), 1);
        assert!(approx(hits[0].x, 1.5));
    }

    #[test]
    fn test_builder_basic() {
        let ls = LineStringBuilder::new()
            .with_point(0.0, 0.0)
            .with_point(1.0, 1.0)
            .build()
            .unwrap();
        assert_eq!(ls.len(), 2);
    }

    #[test]
    fn test_builder_with_points() {
        let ls = LineStringBuilder::new()
            .with_points(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
            .build()
            .unwrap();
        assert!(approx(ls.length(), 2.0));
    }

    #[test]
    fn test_builder_display() {
        let b = LineStringBuilder::new().with_point(0.0, 0.0);
        let s = format!("{b}");
        assert!(s.contains("1 points"));
    }
}
