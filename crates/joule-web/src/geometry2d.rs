//! 2D computational geometry — pure-Rust replacement for turf.js, clipper.js, d3-polygon.
//!
//! Point, Line, Segment, Circle, Polygon. Intersection tests, convex hull (Graham scan),
//! polygon area, point-in-polygon, line clipping (Cohen-Sutherland), Voronoi (simplified).

use std::fmt;

const EPS: f64 = 1e-10;

// ── Point ─────────────────────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn midpoint(self, other: Self) -> Self {
        Self::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (z-component of 3D cross product).
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(self) -> Self {
        let m = self.magnitude();
        if m < EPS { return Self::ORIGIN; }
        Self::new(self.x / m, self.y / m)
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    /// Angle from this point to other, in radians.
    pub fn angle_to(self, other: Self) -> f64 {
        (other.y - self.y).atan2(other.x - self.x)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ── Line ──────────────────────────────────────────────────────

/// An infinite line defined by ax + by + c = 0.
#[derive(Debug, Clone, Copy)]
pub struct Line {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Line {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    /// Line through two points.
    pub fn through(p1: Point, p2: Point) -> Self {
        let a = p2.y - p1.y;
        let b = p1.x - p2.x;
        let c = -(a * p1.x + b * p1.y);
        Self { a, b, c }
    }

    /// Distance from a point to this line.
    pub fn distance_to_point(&self, p: Point) -> f64 {
        (self.a * p.x + self.b * p.y + self.c).abs()
            / (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Intersection of two lines. Returns None if parallel.
    pub fn intersect(&self, other: &Self) -> Option<Point> {
        let det = self.a * other.b - other.a * self.b;
        if det.abs() < EPS {
            return None;
        }
        let x = (self.b * other.c - other.b * self.c) / det;
        let y = (other.a * self.c - self.a * other.c) / det;
        Some(Point::new(x, y))
    }

    /// Whether two lines are parallel.
    pub fn is_parallel(&self, other: &Self) -> bool {
        (self.a * other.b - other.a * self.b).abs() < EPS
    }

    /// Normal direction of the line.
    pub fn normal(&self) -> Point {
        Point::new(self.a, self.b).normalize()
    }
}

// ── Segment ───────────────────────────────────────────────────

/// A line segment from `start` to `end`.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: Point,
    pub end: Point,
}

impl Segment {
    pub fn new(start: Point, end: Point) -> Self {
        Self { start, end }
    }

    pub fn length(&self) -> f64 {
        self.start.distance_to(self.end)
    }

    pub fn midpoint(&self) -> Point {
        self.start.midpoint(self.end)
    }

    /// Direction vector (not normalized).
    pub fn direction(&self) -> Point {
        self.end.sub(self.start)
    }

    /// Closest point on the segment to a given point.
    pub fn closest_point(&self, p: Point) -> Point {
        let d = self.end.sub(self.start);
        let len_sq = d.dot(d);
        if len_sq < EPS {
            return self.start;
        }
        let t = p.sub(self.start).dot(d) / len_sq;
        let t = t.clamp(0.0, 1.0);
        self.start.add(d.scale(t))
    }

    /// Distance from a point to this segment.
    pub fn distance_to_point(&self, p: Point) -> f64 {
        p.distance_to(self.closest_point(p))
    }

    /// Whether two segments intersect. Returns the intersection point if they do.
    pub fn intersect(&self, other: &Segment) -> Option<Point> {
        let d1 = self.end.sub(self.start);
        let d2 = other.end.sub(other.start);
        let cross = d1.cross(d2);
        if cross.abs() < EPS {
            return None; // parallel
        }
        let d = other.start.sub(self.start);
        let t = d.cross(d2) / cross;
        let u = d.cross(d1) / cross;
        if t >= -EPS && t <= 1.0 + EPS && u >= -EPS && u <= 1.0 + EPS {
            Some(self.start.add(d1.scale(t)))
        } else {
            None
        }
    }
}

// ── Circle ────────────────────────────────────────────────────

/// A circle defined by center and radius.
#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub center: Point,
    pub radius: f64,
}

impl Circle {
    pub fn new(center: Point, radius: f64) -> Self {
        Self { center, radius: radius.abs() }
    }

    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    /// Whether a point lies inside (or on) the circle.
    pub fn contains(&self, p: Point) -> bool {
        self.center.distance_to(p) <= self.radius + EPS
    }

    /// Intersection points of two circles. Returns 0, 1, or 2 points.
    pub fn intersect_circle(&self, other: &Circle) -> Vec<Point> {
        let d = self.center.distance_to(other.center);
        if d > self.radius + other.radius + EPS {
            return Vec::new(); // too far apart
        }
        if d < (self.radius - other.radius).abs() - EPS {
            return Vec::new(); // one inside the other
        }
        if d < EPS {
            return Vec::new(); // concentric
        }

        let a = (self.radius * self.radius - other.radius * other.radius + d * d) / (2.0 * d);
        let h_sq = self.radius * self.radius - a * a;
        let h = if h_sq < 0.0 { 0.0 } else { h_sq.sqrt() };

        let dir = other.center.sub(self.center).scale(1.0 / d);
        let mid = self.center.add(dir.scale(a));

        if h < EPS {
            vec![mid] // tangent
        } else {
            let perp = Point::new(-dir.y, dir.x);
            vec![
                mid.add(perp.scale(h)),
                mid.sub(perp.scale(h)),
            ]
        }
    }

    /// Intersection points of circle with a line.
    pub fn intersect_line(&self, line: &Line) -> Vec<Point> {
        let dist = line.distance_to_point(self.center);
        if dist > self.radius + EPS {
            return Vec::new();
        }

        // Closest point on line to center
        let denom = line.a * line.a + line.b * line.b;
        let cx = (line.b * (line.b * self.center.x - line.a * self.center.y) - line.a * line.c) / denom;
        let cy = (line.a * (-(line.b * self.center.x) + line.a * self.center.y) - line.b * line.c) / denom;
        let closest = Point::new(cx, cy);

        if (dist - self.radius).abs() < EPS {
            return vec![closest]; // tangent
        }

        let half_chord = (self.radius * self.radius - dist * dist).sqrt();
        let dir_len = (line.a * line.a + line.b * line.b).sqrt();
        let dir = Point::new(-line.b / dir_len, line.a / dir_len);

        vec![
            closest.add(dir.scale(half_chord)),
            closest.sub(dir.scale(half_chord)),
        ]
    }
}

// ── Polygon ───────────────────────────────────────────────────

/// A polygon defined by its vertices in order.
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Point>,
}

impl Polygon {
    pub fn new(vertices: Vec<Point>) -> Self {
        Self { vertices }
    }

    /// Signed area (positive if CCW, negative if CW).
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 { return 0.0; }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        area / 2.0
    }

    /// Unsigned area.
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Perimeter.
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        if n < 2 { return 0.0; }
        let mut p = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            p += self.vertices[i].distance_to(self.vertices[j]);
        }
        p
    }

    /// Centroid.
    pub fn centroid(&self) -> Point {
        let n = self.vertices.len();
        if n == 0 { return Point::ORIGIN; }
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
        a /= 2.0;
        if a.abs() < EPS {
            // Degenerate — average of points
            let sx: f64 = self.vertices.iter().map(|p| p.x).sum();
            let sy: f64 = self.vertices.iter().map(|p| p.y).sum();
            return Point::new(sx / n as f64, sy / n as f64);
        }
        Point::new(cx / (6.0 * a), cy / (6.0 * a))
    }

    /// Point-in-polygon test (ray casting).
    pub fn contains(&self, p: Point) -> bool {
        let n = self.vertices.len();
        if n < 3 { return false; }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = self.vertices[i];
            let vj = self.vertices[j];
            if ((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Whether the polygon is convex.
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n < 3 { return false; }
        let mut sign = 0i32;
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];
            let c = self.vertices[(i + 2) % n];
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if cross.abs() < EPS { continue; }
            let s = if cross > 0.0 { 1 } else { -1 };
            if sign == 0 {
                sign = s;
            } else if sign != s {
                return false;
            }
        }
        true
    }
}

// ── Convex Hull (Graham scan) ─────────────────────────────────

/// Compute the convex hull of a set of points using Graham scan.
/// Returns vertices in counter-clockwise order.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Find the lowest point (and leftmost if tied)
    let mut pts: Vec<Point> = points.to_vec();
    let mut lowest = 0;
    for i in 1..n {
        if pts[i].y < pts[lowest].y || (pts[i].y == pts[lowest].y && pts[i].x < pts[lowest].x) {
            lowest = i;
        }
    }
    pts.swap(0, lowest);
    let pivot = pts[0];

    // Sort by polar angle from pivot
    pts[1..].sort_by(|a, b| {
        let angle_a = (a.y - pivot.y).atan2(a.x - pivot.x);
        let angle_b = (b.y - pivot.y).atan2(b.x - pivot.x);
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let da = pivot.distance_to(*a);
                let db = pivot.distance_to(*b);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut hull: Vec<Point> = Vec::new();
    for &p in &pts {
        while hull.len() >= 2 {
            let top = hull[hull.len() - 1];
            let second = hull[hull.len() - 2];
            let cross = (top.x - second.x) * (p.y - second.y) - (top.y - second.y) * (p.x - second.x);
            if cross <= EPS {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

// ── Cohen-Sutherland line clipping ────────────────────────────

const INSIDE: u8 = 0;
const LEFT: u8 = 1;
const RIGHT: u8 = 2;
const BOTTOM: u8 = 4;
const TOP: u8 = 8;

fn compute_outcode(p: Point, x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> u8 {
    let mut code = INSIDE;
    if p.x < x_min { code |= LEFT; }
    if p.x > x_max { code |= RIGHT; }
    if p.y < y_min { code |= BOTTOM; }
    if p.y > y_max { code |= TOP; }
    code
}

/// Clip a line segment to a rectangle using Cohen-Sutherland.
/// Returns the clipped segment, or None if entirely outside.
pub fn clip_segment(
    seg: &Segment,
    x_min: f64, y_min: f64, x_max: f64, y_max: f64,
) -> Option<Segment> {
    let mut x0 = seg.start.x;
    let mut y0 = seg.start.y;
    let mut x1 = seg.end.x;
    let mut y1 = seg.end.y;

    let mut outcode0 = compute_outcode(Point::new(x0, y0), x_min, y_min, x_max, y_max);
    let mut outcode1 = compute_outcode(Point::new(x1, y1), x_min, y_min, x_max, y_max);

    loop {
        if outcode0 | outcode1 == 0 {
            return Some(Segment::new(Point::new(x0, y0), Point::new(x1, y1)));
        }
        if outcode0 & outcode1 != 0 {
            return None;
        }

        let outcode_out = if outcode0 != 0 { outcode0 } else { outcode1 };
        let (x, y);
        if outcode_out & TOP != 0 {
            x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0);
            y = y_max;
        } else if outcode_out & BOTTOM != 0 {
            x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0);
            y = y_min;
        } else if outcode_out & RIGHT != 0 {
            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0);
            x = x_max;
        } else {
            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0);
            x = x_min;
        }

        if outcode_out == outcode0 {
            x0 = x;
            y0 = y;
            outcode0 = compute_outcode(Point::new(x0, y0), x_min, y_min, x_max, y_max);
        } else {
            x1 = x;
            y1 = y;
            outcode1 = compute_outcode(Point::new(x1, y1), x_min, y_min, x_max, y_max);
        }
    }
}

// ── Voronoi (simplified incremental) ──────────────────────────

/// A Voronoi cell: a site and the vertices of its cell polygon.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    pub site: Point,
    pub vertices: Vec<Point>,
}

/// Compute a bounded Voronoi diagram within a bounding box.
/// Uses a simplified approach: for each site, clip the half-planes formed by
/// perpendicular bisectors with all other sites against the bounding box.
pub fn voronoi_diagram(
    sites: &[Point],
    x_min: f64, y_min: f64, x_max: f64, y_max: f64,
) -> Vec<VoronoiCell> {
    let mut cells = Vec::new();

    for (i, &site) in sites.iter().enumerate() {
        // Start with bounding box polygon
        let mut poly = vec![
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ];

        for (j, &other) in sites.iter().enumerate() {
            if i == j { continue; }
            // Clip poly against the half-plane closer to `site` than `other`.
            let mid = site.midpoint(other);
            // Normal pointing from other toward site
            let normal = Point::new(site.x - other.x, site.y - other.y);
            poly = clip_polygon_by_halfplane(&poly, mid, normal);
            if poly.is_empty() { break; }
        }

        cells.push(VoronoiCell { site, vertices: poly });
    }

    cells
}

/// Clip a convex polygon to the half-plane defined by: (p - point) . normal >= 0.
fn clip_polygon_by_halfplane(poly: &[Point], point: Point, normal: Point) -> Vec<Point> {
    if poly.is_empty() { return Vec::new(); }
    let n = poly.len();
    let mut output = Vec::new();

    for i in 0..n {
        let current = poly[i];
        let next = poly[(i + 1) % n];
        let dc = (current.x - point.x) * normal.x + (current.y - point.y) * normal.y;
        let dn = (next.x - point.x) * normal.x + (next.y - point.y) * normal.y;

        if dc >= -EPS {
            output.push(current);
        }
        // If edge crosses the boundary, add intersection
        if (dc > EPS && dn < -EPS) || (dc < -EPS && dn > EPS) {
            let t = dc / (dc - dn);
            output.push(Point::new(
                current.x + t * (next.x - current.x),
                current.y + t * (next.y - current.y),
            ));
        }
    }

    output
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < EPS);
    }

    #[test]
    fn point_midpoint() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(4.0, 6.0);
        let m = a.midpoint(b);
        assert!((m.x - 2.0).abs() < EPS);
        assert!((m.y - 3.0).abs() < EPS);
    }

    #[test]
    fn point_cross_product() {
        let a = Point::new(1.0, 0.0);
        let b = Point::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < EPS);
    }

    #[test]
    fn point_rotate() {
        let p = Point::new(1.0, 0.0);
        let rotated = p.rotate(std::f64::consts::FRAC_PI_2);
        assert!((rotated.x).abs() < EPS);
        assert!((rotated.y - 1.0).abs() < EPS);
    }

    #[test]
    fn line_through_points() {
        let l = Line::through(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!((l.distance_to_point(Point::new(0.5, 0.5))).abs() < EPS);
    }

    #[test]
    fn line_intersection() {
        let l1 = Line::through(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let l2 = Line::through(Point::new(0.0, 1.0), Point::new(1.0, 0.0));
        let p = l1.intersect(&l2).unwrap();
        assert!((p.x - 0.5).abs() < EPS);
        assert!((p.y - 0.5).abs() < EPS);
    }

    #[test]
    fn parallel_lines() {
        let l1 = Line::new(1.0, 0.0, 0.0);
        let l2 = Line::new(1.0, 0.0, -5.0);
        assert!(l1.is_parallel(&l2));
        assert!(l1.intersect(&l2).is_none());
    }

    #[test]
    fn segment_intersection() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
        let s2 = Segment::new(Point::new(0.0, 2.0), Point::new(2.0, 0.0));
        let p = s1.intersect(&s2).unwrap();
        assert!((p.x - 1.0).abs() < EPS);
        assert!((p.y - 1.0).abs() < EPS);
    }

    #[test]
    fn segment_no_intersection() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let s2 = Segment::new(Point::new(0.0, 1.0), Point::new(1.0, 1.0));
        assert!(s1.intersect(&s2).is_none());
    }

    #[test]
    fn circle_area_circumference() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!((c.area() - std::f64::consts::PI).abs() < EPS);
        assert!((c.circumference() - 2.0 * std::f64::consts::PI).abs() < EPS);
    }

    #[test]
    fn circle_contains_point() {
        let c = Circle::new(Point::ORIGIN, 5.0);
        assert!(c.contains(Point::new(3.0, 4.0)));
        assert!(!c.contains(Point::new(4.0, 4.0)));
    }

    #[test]
    fn circle_circle_intersection() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(3.0, 0.0), 2.0);
        let pts = c1.intersect_circle(&c2);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn polygon_area() {
        // Unit square
        let poly = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);
        assert!((poly.area() - 1.0).abs() < EPS);
    }

    #[test]
    fn polygon_contains_point() {
        let poly = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ]);
        assert!(poly.contains(Point::new(2.0, 2.0)));
        assert!(!poly.contains(Point::new(5.0, 5.0)));
    }

    #[test]
    fn polygon_is_convex() {
        let square = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]);
        assert!(square.is_convex());

        let concave = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 0.5),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);
        assert!(!concave.is_convex());
    }

    #[test]
    fn polygon_centroid() {
        let poly = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]);
        let c = poly.centroid();
        assert!((c.x - 1.0).abs() < EPS);
        assert!((c.y - 1.0).abs() < EPS);
    }

    #[test]
    fn convex_hull_square() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 0.5), // interior point
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4); // interior point excluded
    }

    #[test]
    fn convex_hull_triangle() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn line_clipping_inside() {
        let seg = Segment::new(Point::new(1.0, 1.0), Point::new(3.0, 3.0));
        let clipped = clip_segment(&seg, 0.0, 0.0, 5.0, 5.0).unwrap();
        assert!((clipped.start.x - 1.0).abs() < EPS);
        assert!((clipped.end.x - 3.0).abs() < EPS);
    }

    #[test]
    fn line_clipping_partial() {
        let seg = Segment::new(Point::new(-1.0, -1.0), Point::new(5.0, 5.0));
        let clipped = clip_segment(&seg, 0.0, 0.0, 4.0, 4.0).unwrap();
        assert!((clipped.start.x).abs() < EPS);
        assert!((clipped.end.x - 4.0).abs() < EPS);
    }

    #[test]
    fn line_clipping_outside() {
        let seg = Segment::new(Point::new(-2.0, -2.0), Point::new(-1.0, -1.0));
        assert!(clip_segment(&seg, 0.0, 0.0, 4.0, 4.0).is_none());
    }

    #[test]
    fn voronoi_two_sites() {
        let sites = vec![Point::new(1.0, 1.0), Point::new(3.0, 1.0)];
        let cells = voronoi_diagram(&sites, 0.0, 0.0, 4.0, 4.0);
        assert_eq!(cells.len(), 2);
        // Both cells should have nonzero area
        for cell in &cells {
            let poly = Polygon::new(cell.vertices.clone());
            assert!(poly.area() > 0.1);
        }
    }

    #[test]
    fn segment_closest_point() {
        let seg = Segment::new(Point::new(0.0, 0.0), Point::new(4.0, 0.0));
        let p = Point::new(2.0, 3.0);
        let closest = seg.closest_point(p);
        assert!((closest.x - 2.0).abs() < EPS);
        assert!((closest.y).abs() < EPS);
    }

    #[test]
    fn segment_distance_to_point() {
        let seg = Segment::new(Point::new(0.0, 0.0), Point::new(4.0, 0.0));
        assert!((seg.distance_to_point(Point::new(2.0, 3.0)) - 3.0).abs() < EPS);
    }

    #[test]
    fn polygon_perimeter() {
        let poly = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(3.0, 4.0),
        ]);
        // 3 + 4 + 5 = 12
        assert!((poly.perimeter() - 12.0).abs() < EPS);
    }
}
