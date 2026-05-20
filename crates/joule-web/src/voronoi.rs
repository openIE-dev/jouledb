//! Voronoi diagram and Delaunay triangulation.
//!
//! Bowyer-Watson algorithm for Delaunay triangulation, dual construction for Voronoi,
//! Fortune's sweep line events, cell/edge/vertex queries, nearest site lookup,
//! bounded Voronoi (clipped to rect), Lloyd relaxation.

use std::collections::HashSet;

// ── Point ───────────────────────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dist_sq(&self, other: &Point) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    pub fn dist(&self, other: &Point) -> f64 {
        self.dist_sq(other).sqrt()
    }

    pub fn lerp(&self, other: &Point, t: f64) -> Point {
        Point::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

// ── Triangle ────────────────────────────────────────────────────

/// A triangle defined by three point indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

impl Triangle {
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    pub fn contains_vertex(&self, idx: usize) -> bool {
        self.a == idx || self.b == idx || self.c == idx
    }

    /// Shared edge with another triangle.
    pub fn shared_edge(&self, other: &Triangle) -> Option<(usize, usize)> {
        let self_verts = [self.a, self.b, self.c];
        let other_verts = [other.a, other.b, other.c];
        let mut shared = Vec::new();
        for &v in &self_verts {
            if other_verts.contains(&v) {
                shared.push(v);
            }
        }
        if shared.len() == 2 {
            Some((shared[0], shared[1]))
        } else {
            None
        }
    }

    /// Get edges as ordered pairs (smaller index first).
    pub fn edges(&self) -> [(usize, usize); 3] {
        [
            (self.a.min(self.b), self.a.max(self.b)),
            (self.b.min(self.c), self.b.max(self.c)),
            (self.a.min(self.c), self.a.max(self.c)),
        ]
    }
}

// ── Edge ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge {
    a: usize,
    b: usize,
}

impl Edge {
    fn new(a: usize, b: usize) -> Self {
        if a < b { Self { a, b } } else { Self { a: b, b: a } }
    }
}

// ── Voronoi Edge ────────────────────────────────────────────────

/// An edge in the Voronoi diagram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiEdge {
    /// Start vertex of the edge.
    pub start: Point,
    /// End vertex of the edge.
    pub end: Point,
    /// Index of the left site.
    pub left_site: usize,
    /// Index of the right site.
    pub right_site: usize,
}

// ── Circumcircle ────────────────────────────────────────────────

/// Circumcircle of a triangle.
#[derive(Debug, Clone, Copy)]
pub struct Circumcircle {
    pub center: Point,
    pub radius_sq: f64,
}

/// Compute the circumcircle of three points.
pub fn circumcircle(a: &Point, b: &Point, c: &Point) -> Circumcircle {
    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-15 {
        let center = Point::new(
            (ax + bx + cx) / 3.0,
            (ay + by + cy) / 3.0,
        );
        return Circumcircle { center, radius_sq: f64::MAX };
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let center = Point::new(ux, uy);
    let radius_sq = center.dist_sq(a);
    Circumcircle { center, radius_sq }
}

// ── Delaunay Triangulation (Bowyer-Watson) ──────────────────────

/// Delaunay triangulation result.
#[derive(Debug, Clone)]
pub struct Delaunay {
    /// The original points.
    pub points: Vec<Point>,
    /// Triangles referencing indices into `points`.
    pub triangles: Vec<Triangle>,
}

impl Delaunay {
    /// Build a Delaunay triangulation using Bowyer-Watson.
    pub fn from_points(points: &[Point]) -> Self {
        if points.len() < 3 {
            return Self {
                points: points.to_vec(),
                triangles: Vec::new(),
            };
        }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let dmax = dx.max(dy);
        let mid_x = (min_x + max_x) / 2.0;
        let mid_y = (min_y + max_y) / 2.0;

        let n = points.len();
        let mut all_points: Vec<Point> = points.to_vec();
        all_points.push(Point::new(mid_x - 20.0 * dmax, mid_y - dmax));
        all_points.push(Point::new(mid_x, mid_y + 20.0 * dmax));
        all_points.push(Point::new(mid_x + 20.0 * dmax, mid_y - dmax));

        let mut triangulation: Vec<Triangle> = vec![Triangle::new(n, n + 1, n + 2)];

        for i in 0..n {
            let p = &all_points[i];
            let mut bad_triangles = Vec::new();

            for (ti, tri) in triangulation.iter().enumerate() {
                let cc = circumcircle(
                    &all_points[tri.a],
                    &all_points[tri.b],
                    &all_points[tri.c],
                );
                if p.dist_sq(&cc.center) <= cc.radius_sq {
                    bad_triangles.push(ti);
                }
            }

            let mut boundary = Vec::new();
            for &ti in &bad_triangles {
                let tri = &triangulation[ti];
                let edges = [
                    Edge::new(tri.a, tri.b),
                    Edge::new(tri.b, tri.c),
                    Edge::new(tri.c, tri.a),
                ];
                for edge in &edges {
                    let shared = bad_triangles.iter().filter(|&&other_ti| {
                        if other_ti == ti { return false; }
                        let other = &triangulation[other_ti];
                        let other_edges = [
                            Edge::new(other.a, other.b),
                            Edge::new(other.b, other.c),
                            Edge::new(other.c, other.a),
                        ];
                        other_edges.contains(edge)
                    }).count();
                    if shared == 0 {
                        boundary.push(*edge);
                    }
                }
            }

            let mut bad_sorted = bad_triangles.clone();
            bad_sorted.sort_unstable_by(|a, b| b.cmp(a));
            for ti in bad_sorted {
                triangulation.swap_remove(ti);
            }

            for edge in &boundary {
                triangulation.push(Triangle::new(edge.a, edge.b, i));
            }
        }

        triangulation.retain(|tri| {
            !tri.contains_vertex(n) && !tri.contains_vertex(n + 1) && !tri.contains_vertex(n + 2)
        });

        Delaunay {
            points: points.to_vec(),
            triangles: triangulation,
        }
    }

    /// Find the nearest point to a query.
    pub fn nearest(&self, query: &Point) -> Option<usize> {
        if self.points.is_empty() { return None; }
        self.points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                query.dist_sq(a).partial_cmp(&query.dist_sq(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Get all unique edges in the triangulation.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edge_set = HashSet::new();
        for tri in &self.triangles {
            for e in &tri.edges() {
                edge_set.insert(*e);
            }
        }
        edge_set.into_iter().collect()
    }

    /// Find triangles adjacent to a given point index.
    pub fn adjacent_triangles(&self, point_idx: usize) -> Vec<usize> {
        self.triangles.iter().enumerate()
            .filter(|(_, t)| t.contains_vertex(point_idx))
            .map(|(i, _)| i)
            .collect()
    }
}

// ── Voronoi Diagram ─────────────────────────────────────────────

/// A Voronoi cell polygon.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the site (point) this cell belongs to.
    pub site: usize,
    /// Polygon vertices in order.
    pub vertices: Vec<Point>,
}

impl VoronoiCell {
    /// Compute the area of the cell polygon.
    pub fn area(&self) -> f64 {
        polygon_area(&self.vertices).abs()
    }

    /// Compute the centroid of the cell.
    pub fn centroid(&self) -> Point {
        polygon_centroid(&self.vertices)
    }

    /// Check if a point is inside the cell.
    pub fn contains(&self, point: &Point) -> bool {
        point_in_polygon(point, &self.vertices)
    }
}

/// Voronoi diagram.
#[derive(Debug, Clone)]
pub struct Voronoi {
    pub cells: Vec<VoronoiCell>,
    pub sites: Vec<Point>,
    pub edges: Vec<VoronoiEdge>,
}

impl Voronoi {
    /// Compute Voronoi diagram from points, clipped to a bounding box.
    pub fn from_points(points: &[Point], bbox: (f64, f64, f64, f64)) -> Self {
        if points.len() < 2 {
            let cells = points
                .iter()
                .enumerate()
                .map(|(i, _)| VoronoiCell {
                    site: i,
                    vertices: vec![
                        Point::new(bbox.0, bbox.1),
                        Point::new(bbox.2, bbox.1),
                        Point::new(bbox.2, bbox.3),
                        Point::new(bbox.0, bbox.3),
                    ],
                })
                .collect();
            return Self { cells, sites: points.to_vec(), edges: Vec::new() };
        }

        let delaunay = Delaunay::from_points(points);

        // Collect Voronoi edges from Delaunay dual.
        // Interior edges: connect circumcenters of adjacent triangles.
        // Boundary edges: extend from circumcenter outward to bbox.
        let mut voronoi_edges = Vec::new();

        // Build a map from Delaunay edge to the triangles that contain it.
        use std::collections::HashMap;
        let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (ti, tri) in delaunay.triangles.iter().enumerate() {
            for (ea, eb) in tri.edges() {
                edge_tris.entry((ea, eb)).or_default().push(ti);
            }
        }

        for (&(ea, eb), tris) in &edge_tris {
            if tris.len() == 2 {
                // Interior edge: connect circumcenters.
                let t0 = &delaunay.triangles[tris[0]];
                let t1 = &delaunay.triangles[tris[1]];
                let cc0 = circumcircle(
                    &delaunay.points[t0.a], &delaunay.points[t0.b], &delaunay.points[t0.c],
                );
                let cc1 = circumcircle(
                    &delaunay.points[t1.a], &delaunay.points[t1.b], &delaunay.points[t1.c],
                );
                voronoi_edges.push(VoronoiEdge {
                    start: cc0.center,
                    end: cc1.center,
                    left_site: ea,
                    right_site: eb,
                });
            } else if tris.len() == 1 {
                // Boundary edge: extend from circumcenter outward.
                let t0 = &delaunay.triangles[tris[0]];
                let cc = circumcircle(
                    &delaunay.points[t0.a], &delaunay.points[t0.b], &delaunay.points[t0.c],
                );
                let pa = &delaunay.points[ea];
                let pb = &delaunay.points[eb];
                let mx = (pa.x + pb.x) * 0.5;
                let my = (pa.y + pb.y) * 0.5;
                // Perpendicular direction to the edge
                let dx = pb.x - pa.x;
                let dy = pb.y - pa.y;
                // The Voronoi edge goes perpendicular: (-dy, dx)
                // Choose the direction away from the circumcenter's
                // opposite side by going from cc toward the midpoint and beyond.
                let dir_x = -(dy);
                let dir_y = dx;
                // Pick the direction that goes away from the triangle interior.
                // The third vertex of the triangle (not ea or eb):
                let third = if t0.a != ea && t0.a != eb { t0.a }
                            else if t0.b != ea && t0.b != eb { t0.b }
                            else { t0.c };
                let tp = &delaunay.points[third];
                // If the direction points toward the third vertex, flip it.
                let to_third_x = tp.x - mx;
                let to_third_y = tp.y - my;
                let (dir_x, dir_y) = if dir_x * to_third_x + dir_y * to_third_y > 0.0 {
                    (-dir_x, -dir_y)
                } else {
                    (dir_x, dir_y)
                };
                // Extend to a far point (well outside bbox)
                let scale = (bbox.2 - bbox.0 + bbox.3 - bbox.1) * 2.0;
                let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                let far = if len > 1e-15 {
                    Point::new(
                        cc.center.x + dir_x / len * scale,
                        cc.center.y + dir_y / len * scale,
                    )
                } else {
                    Point::new(mx + scale, my + scale)
                };
                voronoi_edges.push(VoronoiEdge {
                    start: cc.center,
                    end: far,
                    left_site: ea,
                    right_site: eb,
                });
            }
        }

        // Build cells: start with bbox, clip by perpendicular bisectors of neighbors.
        let mut cells = Vec::new();
        let bbox_poly = vec![
            Point::new(bbox.0, bbox.1),
            Point::new(bbox.2, bbox.1),
            Point::new(bbox.2, bbox.3),
            Point::new(bbox.0, bbox.3),
        ];
        for (i, site) in points.iter().enumerate() {
            let mut cell_poly = bbox_poly.clone();
            for (j, other) in points.iter().enumerate() {
                if i == j { continue; }
                // Clip cell_poly to the half-plane closer to site than other.
                let mx = (site.x + other.x) * 0.5;
                let my = (site.y + other.y) * 0.5;
                // Normal pointing from other toward site.
                let nx = site.x - other.x;
                let ny = site.y - other.y;
                cell_poly = clip_edge(&cell_poly,
                    |p| (p.x - mx) * nx + (p.y - my) * ny >= -1e-12,
                    |a, b| {
                        let da = (a.x - mx) * nx + (a.y - my) * ny;
                        let db = (b.x - mx) * nx + (b.y - my) * ny;
                        let t = da / (da - db);
                        Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
                    },
                );
            }
            cells.push(VoronoiCell { site: i, vertices: cell_poly });
        }

        Voronoi { cells, sites: points.to_vec(), edges: voronoi_edges }
    }

    /// Find which cell a query point falls in (nearest site).
    pub fn cell_for_point(&self, query: &Point) -> Option<usize> {
        self.sites
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                query.dist_sq(a).partial_cmp(&query.dist_sq(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Find the nearest site to a query point.
    pub fn nearest_site(&self, query: &Point) -> Option<(usize, f64)> {
        self.sites
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                query.dist_sq(a).partial_cmp(&query.dist_sq(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, p)| (i, query.dist(p)))
    }

    /// Get the neighbors of a site (sites sharing a Voronoi edge).
    pub fn neighbors(&self, site: usize) -> Vec<usize> {
        let mut result = HashSet::new();
        for edge in &self.edges {
            if edge.left_site == site {
                result.insert(edge.right_site);
            } else if edge.right_site == site {
                result.insert(edge.left_site);
            }
        }
        result.into_iter().collect()
    }

    /// Total number of Voronoi edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ── Sutherland-Hodgman Clipping ──────────────────────────────────

/// Clip a polygon to a bounding box using Sutherland-Hodgman algorithm.
fn clip_polygon_to_bbox(polygon: &[Point], bbox: (f64, f64, f64, f64)) -> Vec<Point> {
    if polygon.is_empty() { return Vec::new(); }
    let (x_min, y_min, x_max, y_max) = bbox;

    let mut output = polygon.to_vec();

    output = clip_edge(&output, |p| p.x >= x_min, |a, b| {
        let t = (x_min - a.x) / (b.x - a.x);
        Point::new(x_min, a.y + t * (b.y - a.y))
    });
    output = clip_edge(&output, |p| p.x <= x_max, |a, b| {
        let t = (x_max - a.x) / (b.x - a.x);
        Point::new(x_max, a.y + t * (b.y - a.y))
    });
    output = clip_edge(&output, |p| p.y >= y_min, |a, b| {
        let t = (y_min - a.y) / (b.y - a.y);
        Point::new(a.x + t * (b.x - a.x), y_min)
    });
    output = clip_edge(&output, |p| p.y <= y_max, |a, b| {
        let t = (y_max - a.y) / (b.y - a.y);
        Point::new(a.x + t * (b.x - a.x), y_max)
    });

    output
}

fn clip_edge(
    polygon: &[Point],
    inside: impl Fn(&Point) -> bool,
    intersect: impl Fn(&Point, &Point) -> Point,
) -> Vec<Point> {
    if polygon.is_empty() { return Vec::new(); }
    let mut output = Vec::new();
    let n = polygon.len();
    for i in 0..n {
        let current = &polygon[i];
        let next = &polygon[(i + 1) % n];
        let cur_in = inside(current);
        let next_in = inside(next);
        match (cur_in, next_in) {
            (true, true) => output.push(*next),
            (true, false) => output.push(intersect(current, next)),
            (false, true) => {
                output.push(intersect(current, next));
                output.push(*next);
            }
            (false, false) => {}
        }
    }
    output
}

// ── Lloyd Relaxation ────────────────────────────────────────────

/// Perform one step of Lloyd relaxation.
pub fn lloyd_relax(points: &mut [Point], bbox: (f64, f64, f64, f64)) {
    if points.len() < 2 { return; }
    let voronoi = Voronoi::from_points(points, bbox);
    for cell in &voronoi.cells {
        if cell.vertices.len() < 3 { continue; }
        let centroid = polygon_centroid(&cell.vertices);
        let x = centroid.x.clamp(bbox.0, bbox.2);
        let y = centroid.y.clamp(bbox.1, bbox.3);
        points[cell.site] = Point::new(x, y);
    }
}

/// Perform N iterations of Lloyd relaxation.
pub fn lloyd_relax_n(points: &mut [Point], bbox: (f64, f64, f64, f64), iterations: u32) {
    for _ in 0..iterations {
        lloyd_relax(points, bbox);
    }
}

// ── Fortune's Sweep Line Events ──────────────────────────────────

/// Event type for Fortune's sweep line algorithm.
#[derive(Debug, Clone)]
pub enum SweepEvent {
    /// A new site is encountered by the sweep line.
    Site { site_index: usize, y: f64 },
    /// A circle event where a parabolic arc vanishes.
    Circle { arc_index: usize, center: Point, radius: f64 },
}

impl SweepEvent {
    pub fn y(&self) -> f64 {
        match self {
            SweepEvent::Site { y, .. } => *y,
            SweepEvent::Circle { center, radius, .. } => center.y - radius,
        }
    }
}

/// Generate Fortune's sweep line events for a set of points.
///
/// Returns site events sorted by y-coordinate (for sweep line processing).
/// Circle events must be computed dynamically during the sweep.
pub fn fortune_site_events(points: &[Point]) -> Vec<SweepEvent> {
    let mut events: Vec<SweepEvent> = points.iter().enumerate()
        .map(|(i, p)| SweepEvent::Site { site_index: i, y: p.y })
        .collect();
    events.sort_by(|a, b| {
        b.y().partial_cmp(&a.y()).unwrap_or(std::cmp::Ordering::Equal)
    });
    events
}

/// Compute a circle event from three points (potential arc vanishing).
pub fn compute_circle_event(
    p1: &Point,
    p2: &Point,
    p3: &Point,
    arc_index: usize,
) -> Option<SweepEvent> {
    let cc = circumcircle(p1, p2, p3);
    if cc.radius_sq >= f64::MAX * 0.5 {
        return None; // Degenerate
    }
    let radius = cc.radius_sq.sqrt();
    Some(SweepEvent::Circle {
        arc_index,
        center: cc.center,
        radius,
    })
}

// ── Helper Functions ────────────────────────────────────────────

/// Signed area of a polygon (positive if CCW).
fn polygon_area(vertices: &[Point]) -> f64 {
    if vertices.is_empty() { return 0.0; }
    let n = vertices.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
    }
    area * 0.5
}

/// Centroid of a polygon.
fn polygon_centroid(vertices: &[Point]) -> Point {
    if vertices.is_empty() { return Point::new(0.0, 0.0); }
    let n = vertices.len();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        area += cross;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
    }
    area *= 0.5;
    if area.abs() < 1e-15 {
        let sx: f64 = vertices.iter().map(|p| p.x).sum();
        let sy: f64 = vertices.iter().map(|p| p.y).sum();
        return Point::new(sx / n as f64, sy / n as f64);
    }
    cx /= 6.0 * area;
    cy /= 6.0 * area;
    Point::new(cx, cy)
}

/// Point-in-polygon test (ray casting).
fn point_in_polygon(point: &Point, vertices: &[Point]) -> bool {
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

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square_points() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
            Point::new(5.0, 5.0),
        ]
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.dist(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_lerp() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 10.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!((mid.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_circumcircle_equilateral() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 0.0);
        let c = Point::new(0.5, 0.866);
        let cc = circumcircle(&a, &b, &c);
        assert!((cc.center.x - 0.5).abs() < 0.01);
        let da = cc.center.dist_sq(&a);
        let db = cc.center.dist_sq(&b);
        assert!((da - db).abs() < 0.01);
    }

    #[test]
    fn test_delaunay_basic() {
        let points = square_points();
        let d = Delaunay::from_points(&points);
        assert_eq!(d.points.len(), 5);
        assert!(!d.triangles.is_empty());
        assert!(d.triangles.len() >= 4);
    }

    #[test]
    fn test_delaunay_three_points() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 1.0),
        ];
        let d = Delaunay::from_points(&points);
        assert_eq!(d.triangles.len(), 1);
    }

    #[test]
    fn test_delaunay_too_few() {
        let d = Delaunay::from_points(&[Point::new(0.0, 0.0), Point::new(1.0, 1.0)]);
        assert!(d.triangles.is_empty());
    }

    #[test]
    fn test_nearest_neighbor() {
        let points = square_points();
        let d = Delaunay::from_points(&points);
        let nearest = d.nearest(&Point::new(4.9, 5.1)).unwrap();
        assert_eq!(nearest, 4);
    }

    #[test]
    fn test_delaunay_edges() {
        let points = square_points();
        let d = Delaunay::from_points(&points);
        let edges = d.edges();
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_adjacent_triangles() {
        let points = square_points();
        let d = Delaunay::from_points(&points);
        let adj = d.adjacent_triangles(4); // center point
        assert!(!adj.is_empty());
    }

    #[test]
    fn test_voronoi_basic() {
        let points = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(5.0, 8.0),
        ];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        assert_eq!(v.cells.len(), 3);
        for cell in &v.cells {
            assert!(!cell.vertices.is_empty());
        }
    }

    #[test]
    fn test_voronoi_cell_for_point() {
        let points = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(5.0, 8.0),
        ];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        assert_eq!(v.cell_for_point(&Point::new(1.0, 1.0)), Some(0));
        assert_eq!(v.cell_for_point(&Point::new(9.0, 1.0)), Some(1));
    }

    #[test]
    fn test_voronoi_single_point() {
        let points = vec![Point::new(5.0, 5.0)];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        assert_eq!(v.cells.len(), 1);
        assert_eq!(v.cells[0].vertices.len(), 4);
    }

    #[test]
    fn test_voronoi_nearest_site() {
        let points = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 8.0),
        ];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        let (idx, dist) = v.nearest_site(&Point::new(3.0, 3.0)).unwrap();
        assert_eq!(idx, 0);
        assert!(dist < 2.0);
    }

    #[test]
    fn test_voronoi_edges() {
        let points = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(5.0, 8.0),
        ];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        assert!(v.edge_count() > 0);
    }

    #[test]
    fn test_cell_area() {
        let cell = VoronoiCell {
            site: 0,
            vertices: vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
            ],
        };
        assert!((cell.area() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_centroid() {
        let cell = VoronoiCell {
            site: 0,
            vertices: vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
            ],
        };
        let c = cell.centroid();
        assert!((c.x - 5.0).abs() < 0.01);
        assert!((c.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_contains() {
        let cell = VoronoiCell {
            site: 0,
            vertices: vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
            ],
        };
        assert!(cell.contains(&Point::new(5.0, 5.0)));
        assert!(!cell.contains(&Point::new(15.0, 5.0)));
    }

    #[test]
    fn test_lloyd_relaxation() {
        let mut points = vec![
            Point::new(1.0, 1.0),
            Point::new(1.5, 1.2),
            Point::new(8.0, 8.0),
            Point::new(8.5, 8.2),
        ];
        let bbox = (0.0, 0.0, 10.0, 10.0);
        let original = points.clone();
        lloyd_relax(&mut points, bbox);
        let moved = points.iter().zip(original.iter())
            .any(|(a, b)| (a.x - b.x).abs() > 0.001 || (a.y - b.y).abs() > 0.001);
        assert!(moved);
    }

    #[test]
    fn test_lloyd_relax_n() {
        let mut points = vec![
            Point::new(1.0, 1.0),
            Point::new(1.5, 1.2),
            Point::new(8.0, 8.0),
        ];
        let bbox = (0.0, 0.0, 10.0, 10.0);
        let original = points.clone();
        lloyd_relax_n(&mut points, bbox, 5);
        let moved = points.iter().zip(original.iter())
            .any(|(a, b)| (a.x - b.x).abs() > 0.001 || (a.y - b.y).abs() > 0.001);
        assert!(moved);
    }

    #[test]
    fn test_clip_polygon() {
        let poly = vec![
            Point::new(-1.0, 0.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(-1.0, 5.0),
        ];
        let clipped = clip_polygon_to_bbox(&poly, (0.0, 0.0, 10.0, 10.0));
        for p in &clipped {
            assert!(p.x >= -0.001 && p.x <= 10.001);
            assert!(p.y >= -0.001 && p.y <= 10.001);
        }
    }

    #[test]
    fn test_triangle_shared_edge() {
        let t1 = Triangle::new(0, 1, 2);
        let t2 = Triangle::new(1, 2, 3);
        let edge = t1.shared_edge(&t2);
        assert!(edge.is_some());
        let (a, b) = edge.unwrap();
        assert!((a == 1 && b == 2) || (a == 2 && b == 1));
    }

    #[test]
    fn test_triangle_edges() {
        let t = Triangle::new(0, 3, 5);
        let edges = t.edges();
        assert_eq!(edges.len(), 3);
        // All edges should have smaller index first
        for (a, b) in &edges {
            assert!(a < b);
        }
    }

    #[test]
    fn test_fortune_site_events() {
        let points = vec![
            Point::new(1.0, 5.0),
            Point::new(3.0, 2.0),
            Point::new(5.0, 8.0),
        ];
        let events = fortune_site_events(&points);
        assert_eq!(events.len(), 3);
        // Should be sorted by descending y
        let y_vals: Vec<f64> = events.iter().map(|e| e.y()).collect();
        for i in 1..y_vals.len() {
            assert!(y_vals[i - 1] >= y_vals[i]);
        }
    }

    #[test]
    fn test_circle_event() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(1.0, 0.0);
        let p3 = Point::new(0.5, 0.866);
        let event = compute_circle_event(&p1, &p2, &p3, 1);
        assert!(event.is_some());
        if let Some(SweepEvent::Circle { arc_index, .. }) = event {
            assert_eq!(arc_index, 1);
        }
    }

    #[test]
    fn test_voronoi_neighbors() {
        let points = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(5.0, 8.0),
        ];
        let v = Voronoi::from_points(&points, (0.0, 0.0, 10.0, 10.0));
        // In a 3-point Voronoi, every site is a neighbor of every other
        let n0 = v.neighbors(0);
        assert!(n0.len() >= 1);
    }
}
