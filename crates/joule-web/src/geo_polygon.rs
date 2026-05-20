//! Polygon operations — point-in-polygon (ray casting), polygon area (Shoelace),
//! centroid, convex hull (Graham scan), polygon simplification (Douglas-Peucker),
//! polygon clipping (Sutherland-Hodgman).
//!
//! All geometry uses `f64` coordinates. Polygons are represented as closed rings
//! where the first and last vertex are implicitly connected.

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

    /// Squared Euclidean distance to another point.
    pub fn dist_sq(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another point.
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
pub enum PolygonError {
    TooFewVertices(usize),
    InvalidEpsilon(f64),
    EmptyInput,
}

impl fmt::Display for PolygonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewVertices(n) => write!(f, "polygon needs >=3 vertices, got {n}"),
            Self::InvalidEpsilon(e) => write!(f, "epsilon must be positive, got {e}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for PolygonError {}

// ── Polygon ────────────────────────────────────────────────────

/// A simple polygon represented as an ordered ring of vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    /// Vertices in order; the ring is implicitly closed.
    pub vertices: Vec<Point2>,
}

impl Polygon {
    /// Create a polygon from at least 3 vertices.
    pub fn new(vertices: Vec<Point2>) -> Result<Self, PolygonError> {
        if vertices.len() < 3 {
            return Err(PolygonError::TooFewVertices(vertices.len()));
        }
        Ok(Self { vertices })
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Returns `true` when the vertex list is empty (should never happen after `new`).
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    // ── Point-in-polygon (ray casting) ─────────────────────────

    /// Test whether `pt` lies inside the polygon using the ray-casting algorithm.
    /// Points exactly on the boundary may be classified either way.
    pub fn contains(&self, pt: Point2) -> bool {
        let n = self.vertices.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];
            if ((vi.y > pt.y) != (vj.y > pt.y))
                && (pt.x < (vj.x - vi.x) * (pt.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    // ── Area (Shoelace) ────────────────────────────────────────

    /// Signed area using the Shoelace formula.  Positive when vertices are
    /// counter-clockwise.
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        let mut sum = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.vertices[i].x * self.vertices[j].y;
            sum -= self.vertices[j].x * self.vertices[i].y;
        }
        sum * 0.5
    }

    /// Unsigned (absolute) area.
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    // ── Centroid ───────────────────────────────────────────────

    /// Centroid of the polygon (center of mass assuming uniform density).
    pub fn centroid(&self) -> Point2 {
        let n = self.vertices.len();
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut a = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = self.vertices[i].x * self.vertices[j].y
                - self.vertices[j].x * self.vertices[i].y;
            cx += (self.vertices[i].x + self.vertices[j].x) * cross;
            cy += (self.vertices[i].y + self.vertices[j].y) * cross;
            a += cross;
        }
        a *= 0.5;
        let factor = 1.0 / (6.0 * a);
        Point2::new(cx * factor, cy * factor)
    }

    /// Returns `true` when vertices are ordered counter-clockwise.
    pub fn is_ccw(&self) -> bool {
        self.signed_area() > 0.0
    }

    /// Reverse vertex winding order in place.
    pub fn reverse(&mut self) {
        self.vertices.reverse();
    }

    /// Bounding box `(min, max)`.
    pub fn bbox(&self) -> (Point2, Point2) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for v in &self.vertices {
            if v.x < min_x { min_x = v.x; }
            if v.y < min_y { min_y = v.y; }
            if v.x > max_x { max_x = v.x; }
            if v.y > max_y { max_y = v.y; }
        }
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Perimeter of the polygon.
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        let mut total = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            total += self.vertices[i].dist(self.vertices[j]);
        }
        total
    }
}

impl fmt::Display for Polygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polygon({} vertices, area={:.4})", self.vertices.len(), self.area())
    }
}

// ── Convex Hull (Graham scan) ──────────────────────────────────

/// Cross product of vectors OA and OB where O is origin.
fn cross(o: Point2, a: Point2, b: Point2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Compute the convex hull of a set of points using the Graham-scan algorithm.
/// Returns a `Polygon` with vertices in counter-clockwise order.
pub fn convex_hull(points: &[Point2]) -> Result<Polygon, PolygonError> {
    if points.len() < 3 {
        return Err(PolygonError::TooFewVertices(points.len()));
    }
    let mut pts: Vec<Point2> = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));

    // Build lower hull.
    let mut lower: Vec<Point2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }

    // Build upper hull.
    let mut upper: Vec<Point2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }

    // Remove last point of each half because it repeats.
    lower.pop();
    upper.pop();
    lower.append(&mut upper);
    Polygon::new(lower)
}

// ── Douglas-Peucker simplification ────────────────────────────

/// Perpendicular distance from point `p` to line segment `a–b`.
fn point_line_dist(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-15 {
        return p.dist(a);
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = Point2::new(a.x + t * dx, a.y + t * dy);
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

/// Simplify a polygon using the Douglas-Peucker algorithm.
pub fn simplify_polygon(poly: &Polygon, epsilon: f64) -> Result<Polygon, PolygonError> {
    if epsilon <= 0.0 {
        return Err(PolygonError::InvalidEpsilon(epsilon));
    }
    let mut ring = poly.vertices.clone();
    // Close the ring for simplification.
    ring.push(ring[0]);

    let mut simplified = Vec::new();
    dp_recurse(&ring, epsilon, &mut simplified);
    // The last point from recursion may be missing; ensure closure.
    if simplified.len() >= 2 {
        let last_added = *simplified.last().unwrap();
        let first = simplified[0];
        if last_added.dist(first) > 1e-12 {
            simplified.push(*ring.last().unwrap());
        }
    }
    // Remove closing duplicate if present.
    if simplified.len() > 1 {
        let first = simplified[0];
        let last = *simplified.last().unwrap();
        if (first.x - last.x).abs() < 1e-12 && (first.y - last.y).abs() < 1e-12 {
            simplified.pop();
        }
    }
    if simplified.len() < 3 {
        return Ok(poly.clone());
    }
    Polygon::new(simplified)
}

// ── Sutherland-Hodgman clipping ────────────────────────────────

/// Clip `subject` polygon by the convex `clip` polygon using the
/// Sutherland-Hodgman algorithm. Returns the clipped polygon.
pub fn clip_polygon(
    subject: &Polygon,
    clip: &Polygon,
) -> Result<Option<Polygon>, PolygonError> {
    let mut output = subject.vertices.clone();
    let cn = clip.vertices.len();

    for i in 0..cn {
        if output.is_empty() {
            return Ok(None);
        }
        let edge_a = clip.vertices[i];
        let edge_b = clip.vertices[(i + 1) % cn];
        let mut input = output;
        output = Vec::new();

        let sn = input.len();
        if sn == 0 {
            return Ok(None);
        }
        let mut s = input[sn - 1];
        for &e in &input {
            if inside_edge(e, edge_a, edge_b) {
                if !inside_edge(s, edge_a, edge_b) {
                    if let Some(ix) = line_intersect_seg(s, e, edge_a, edge_b) {
                        output.push(ix);
                    }
                }
                output.push(e);
            } else if inside_edge(s, edge_a, edge_b) {
                if let Some(ix) = line_intersect_seg(s, e, edge_a, edge_b) {
                    output.push(ix);
                }
            }
            s = e;
        }
    }
    if output.len() < 3 {
        return Ok(None);
    }
    Polygon::new(output).map(Some)
}

/// Is `p` on the inside (left) of directed edge `a→b`?
fn inside_edge(p: Point2, a: Point2, b: Point2) -> bool {
    cross(a, b, p) >= 0.0
}

/// Intersection of line segments `p1–p2` and `p3–p4` (infinite line intersection).
fn line_intersect_seg(p1: Point2, p2: Point2, p3: Point2, p4: Point2) -> Option<Point2> {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
    Some(Point2::new(p1.x + t * d1x, p1.y + t * d1y))
}

// ── PolygonBuilder ─────────────────────────────────────────────

/// Builder for constructing polygons vertex by vertex.
#[derive(Debug, Clone)]
pub struct PolygonBuilder {
    vertices: Vec<Point2>,
}

impl PolygonBuilder {
    pub fn new() -> Self {
        Self { vertices: Vec::new() }
    }

    pub fn with_vertex(mut self, x: f64, y: f64) -> Self {
        self.vertices.push(Point2::new(x, y));
        self
    }

    pub fn with_vertices(mut self, pts: &[(f64, f64)]) -> Self {
        for &(x, y) in pts {
            self.vertices.push(Point2::new(x, y));
        }
        self
    }

    pub fn build(self) -> Result<Polygon, PolygonError> {
        Polygon::new(self.vertices)
    }
}

impl Default for PolygonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PolygonBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PolygonBuilder({} vertices queued)", self.vertices.len())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn unit_square() -> Polygon {
        Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
        .unwrap()
    }

    #[test]
    fn test_point2_display() {
        let p = Point2::new(1.5, 2.5);
        assert_eq!(format!("{p}"), "(1.500000, 2.500000)");
    }

    #[test]
    fn test_polygon_area_square() {
        let sq = unit_square();
        assert!(approx(sq.area(), 1.0));
    }

    #[test]
    fn test_polygon_signed_area_ccw() {
        let sq = unit_square();
        assert!(sq.signed_area() > 0.0);
    }

    #[test]
    fn test_polygon_centroid_square() {
        let sq = unit_square();
        let c = sq.centroid();
        assert!(approx(c.x, 0.5));
        assert!(approx(c.y, 0.5));
    }

    #[test]
    fn test_polygon_contains_inside() {
        let sq = unit_square();
        assert!(sq.contains(Point2::new(0.5, 0.5)));
    }

    #[test]
    fn test_polygon_contains_outside() {
        let sq = unit_square();
        assert!(!sq.contains(Point2::new(2.0, 2.0)));
    }

    #[test]
    fn test_polygon_perimeter() {
        let sq = unit_square();
        assert!(approx(sq.perimeter(), 4.0));
    }

    #[test]
    fn test_polygon_too_few_vertices() {
        let r = Polygon::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)]);
        assert!(r.is_err());
    }

    #[test]
    fn test_polygon_display() {
        let sq = unit_square();
        let s = format!("{sq}");
        assert!(s.contains("4 vertices"));
    }

    #[test]
    fn test_polygon_bbox() {
        let sq = unit_square();
        let (lo, hi) = sq.bbox();
        assert!(approx(lo.x, 0.0));
        assert!(approx(hi.x, 1.0));
    }

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 0.5),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&pts).unwrap();
        assert_eq!(hull.len(), 4);
        assert!(approx(hull.area(), 1.0));
    }

    #[test]
    fn test_convex_hull_triangle() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        ];
        let hull = convex_hull(&pts).unwrap();
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_too_few() {
        let pts = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        assert!(convex_hull(&pts).is_err());
    }

    #[test]
    fn test_simplify_square_large_eps() {
        let sq = unit_square();
        let simplified = simplify_polygon(&sq, 10.0).unwrap();
        assert!(simplified.len() >= 3);
    }

    #[test]
    fn test_simplify_invalid_epsilon() {
        let sq = unit_square();
        assert!(simplify_polygon(&sq, 0.0).is_err());
    }

    #[test]
    fn test_simplify_preserves_shape() {
        let sq = unit_square();
        let simplified = simplify_polygon(&sq, 0.001).unwrap();
        assert_eq!(simplified.len(), 4);
    }

    #[test]
    fn test_clip_polygon_overlap() {
        let subject = unit_square();
        let clip = Polygon::new(vec![
            Point2::new(0.5, 0.5),
            Point2::new(1.5, 0.5),
            Point2::new(1.5, 1.5),
            Point2::new(0.5, 1.5),
        ])
        .unwrap();
        let result = clip_polygon(&subject, &clip).unwrap();
        assert!(result.is_some());
        let clipped = result.unwrap();
        assert!(approx(clipped.area(), 0.25));
    }

    #[test]
    fn test_clip_polygon_no_overlap() {
        let subject = unit_square();
        let clip = Polygon::new(vec![
            Point2::new(5.0, 5.0),
            Point2::new(6.0, 5.0),
            Point2::new(6.0, 6.0),
            Point2::new(5.0, 6.0),
        ])
        .unwrap();
        let result = clip_polygon(&subject, &clip).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_polygon_builder_basic() {
        let poly = PolygonBuilder::new()
            .with_vertex(0.0, 0.0)
            .with_vertex(1.0, 0.0)
            .with_vertex(1.0, 1.0)
            .build()
            .unwrap();
        assert_eq!(poly.len(), 3);
    }

    #[test]
    fn test_polygon_builder_with_vertices() {
        let poly = PolygonBuilder::new()
            .with_vertices(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
            .build()
            .unwrap();
        assert!(approx(poly.area(), 1.0));
    }

    #[test]
    fn test_polygon_is_ccw() {
        let sq = unit_square();
        assert!(sq.is_ccw());
    }

    #[test]
    fn test_polygon_reverse() {
        let mut sq = unit_square();
        sq.reverse();
        assert!(!sq.is_ccw());
    }

    #[test]
    fn test_triangle_area() {
        let tri = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(0.0, 3.0),
        ])
        .unwrap();
        assert!(approx(tri.area(), 6.0));
    }
}
