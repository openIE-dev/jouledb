//! Convex hull algorithms: Graham scan, gift wrapping (Jarvis march), convex hull
//! area/perimeter, point-in-hull test, minimum bounding rectangle, rotating calipers
//! for diameter, convex hull merge.
//!
//! Pure math — no browser dependency.

use std::cmp::Ordering;
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

    pub fn distance_sq(&self, other: &Point) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

/// Cross product of vectors OA and OB.
fn cross(o: &Point, a: &Point, b: &Point) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

// ── Graham Scan ────────────────────────────────────────────────

/// Compute the convex hull using Graham scan. Returns points in CCW order.
pub fn graham_scan(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Find the lowest point (then leftmost).
    let mut pts = points.to_vec();
    let pivot_idx = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.y.partial_cmp(&b.y)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal))
        })
        .map(|(i, _)| i)
        .unwrap();
    pts.swap(0, pivot_idx);
    let pivot = pts[0];

    // Sort by polar angle relative to pivot.
    pts[1..].sort_by(|a, b| {
        let c = cross(&pivot, a, b);
        if c.abs() < 1e-12 {
            // Collinear — closer first.
            pivot
                .distance_sq(a)
                .partial_cmp(&pivot.distance_sq(b))
                .unwrap_or(Ordering::Equal)
        } else if c > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });

    let mut hull: Vec<Point> = Vec::new();
    for pt in &pts {
        while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], pt) <= 0.0 {
            hull.pop();
        }
        hull.push(*pt);
    }

    hull
}

// ── Gift Wrapping (Jarvis March) ───────────────────────────────

/// Compute the convex hull using Jarvis march (gift wrapping).
pub fn jarvis_march(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Start with leftmost point.
    let start = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.x.partial_cmp(&b.x)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal))
        })
        .map(|(i, _)| i)
        .unwrap();

    let mut hull = Vec::new();
    let mut current = start;

    loop {
        hull.push(points[current]);
        let mut next = 0;

        for i in 1..n {
            if next == current {
                next = i;
                continue;
            }
            let c = cross(&points[current], &points[next], &points[i]);
            if c > 0.0 || (c.abs() < 1e-12 && points[current].distance_sq(&points[i]) > points[current].distance_sq(&points[next])) {
                next = i;
            }
        }

        current = next;
        if current == start {
            break;
        }
        // Safety guard.
        if hull.len() > n {
            break;
        }
    }

    hull
}

// ── Hull Properties ────────────────────────────────────────────

/// Area of a convex hull (shoelace formula).
pub fn hull_area(hull: &[Point]) -> f64 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i].x * hull[j].y;
        area -= hull[j].x * hull[i].y;
    }
    (area / 2.0).abs()
}

/// Perimeter of a convex hull.
pub fn hull_perimeter(hull: &[Point]) -> f64 {
    let n = hull.len();
    if n < 2 {
        return 0.0;
    }
    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        perimeter += hull[i].distance(&hull[j]);
    }
    perimeter
}

// ── Point-in-Hull Test ─────────────────────────────────────────

/// Test whether a point is inside a convex hull (CCW ordered).
/// Uses cross-product: point must be on the left of all edges.
pub fn point_in_hull(p: &Point, hull: &[Point]) -> bool {
    let n = hull.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let j = (i + 1) % n;
        if cross(&hull[i], &hull[j], p) < -1e-10 {
            return false;
        }
    }
    true
}

// ── Minimum Bounding Rectangle ─────────────────────────────────

/// An oriented bounding rectangle.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingRect {
    /// Corner points of the rectangle.
    pub corners: [Point; 4],
    pub area: f64,
    pub width: f64,
    pub height: f64,
    pub angle: f64, // rotation angle in radians
}

/// Compute the minimum-area bounding rectangle of a convex hull.
/// Uses rotating calipers approach.
pub fn minimum_bounding_rect(hull: &[Point]) -> BoundingRect {
    let n = hull.len();
    if n < 2 {
        let p = if n == 1 { hull[0] } else { Point::new(0.0, 0.0) };
        return BoundingRect {
            corners: [p; 4],
            area: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };
    }

    let mut best_area = f64::INFINITY;
    let mut best_rect = None;

    for i in 0..n {
        let j = (i + 1) % n;
        let edge_dx = hull[j].x - hull[i].x;
        let edge_dy = hull[j].y - hull[i].y;
        let edge_len = (edge_dx * edge_dx + edge_dy * edge_dy).sqrt();
        if edge_len < 1e-12 {
            continue;
        }

        let ux = edge_dx / edge_len;
        let uy = edge_dy / edge_len;
        let vx = -uy;
        let vy = ux;

        // Project all hull points onto (u, v).
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;

        for pt in hull {
            let du = (pt.x - hull[i].x) * ux + (pt.y - hull[i].y) * uy;
            let dv = (pt.x - hull[i].x) * vx + (pt.y - hull[i].y) * vy;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_v = min_v.min(dv);
            max_v = max_v.max(dv);
        }

        let width = max_u - min_u;
        let height = max_v - min_v;
        let area = width * height;

        if area < best_area {
            best_area = area;
            let origin_x = hull[i].x + min_u * ux + min_v * vx;
            let origin_y = hull[i].y + min_u * uy + min_v * vy;
            let c0 = Point::new(origin_x, origin_y);
            let c1 = Point::new(origin_x + width * ux, origin_y + width * uy);
            let c2 = Point::new(
                origin_x + width * ux + height * vx,
                origin_y + width * uy + height * vy,
            );
            let c3 = Point::new(origin_x + height * vx, origin_y + height * vy);
            let angle = uy.atan2(ux);
            best_rect = Some(BoundingRect {
                corners: [c0, c1, c2, c3],
                area,
                width,
                height,
                angle,
            });
        }
    }

    best_rect.unwrap_or(BoundingRect {
        corners: [hull[0]; 4],
        area: 0.0,
        width: 0.0,
        height: 0.0,
        angle: 0.0,
    })
}

// ── Rotating Calipers – Diameter ───────────────────────────────

/// Find the diameter (maximum distance between any two hull points)
/// using the rotating calipers algorithm.
pub fn hull_diameter(hull: &[Point]) -> (f64, Point, Point) {
    let n = hull.len();
    if n < 2 {
        let p = if n == 1 { hull[0] } else { Point::new(0.0, 0.0) };
        return (0.0, p, p);
    }
    if n == 2 {
        return (hull[0].distance(&hull[1]), hull[0], hull[1]);
    }

    // Antipodal pair search.
    let mut max_dist = 0.0f64;
    let mut best_a = hull[0];
    let mut best_b = hull[1];
    let mut j = 1;

    for i in 0..n {
        let i_next = (i + 1) % n;
        // Advance j while triangle area increases.
        loop {
            let j_next = (j + 1) % n;
            let area_j = cross(&hull[i], &hull[i_next], &hull[j]).abs();
            let area_j_next = cross(&hull[i], &hull[i_next], &hull[j_next]).abs();
            if area_j_next > area_j {
                j = j_next;
            } else {
                break;
            }
        }

        let d = hull[i].distance(&hull[j]);
        if d > max_dist {
            max_dist = d;
            best_a = hull[i];
            best_b = hull[j];
        }
    }

    (max_dist, best_a, best_b)
}

// ── Convex Hull Merge ──────────────────────────────────────────

/// Merge two convex hulls into a single convex hull.
pub fn merge_hulls(hull_a: &[Point], hull_b: &[Point]) -> Vec<Point> {
    let mut all = Vec::with_capacity(hull_a.len() + hull_b.len());
    all.extend_from_slice(hull_a);
    all.extend_from_slice(hull_b);
    graham_scan(&all)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square_points() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]
    }

    #[test]
    fn test_graham_scan_square() {
        let hull = graham_scan(&square_points());
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_graham_scan_with_interior() {
        let mut pts = square_points();
        pts.push(Point::new(5.0, 5.0)); // interior point
        pts.push(Point::new(3.0, 7.0)); // interior point
        let hull = graham_scan(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_jarvis_march_square() {
        let hull = jarvis_march(&square_points());
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_hull_area_square() {
        let hull = graham_scan(&square_points());
        let area = hull_area(&hull);
        assert!((area - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_hull_perimeter_square() {
        let hull = graham_scan(&square_points());
        let perim = hull_perimeter(&hull);
        assert!((perim - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_point_in_hull() {
        let hull = graham_scan(&square_points());
        assert!(point_in_hull(&Point::new(5.0, 5.0), &hull));
        assert!(!point_in_hull(&Point::new(15.0, 5.0), &hull));
    }

    #[test]
    fn test_hull_diameter() {
        let hull = graham_scan(&square_points());
        let (diam, _, _) = hull_diameter(&hull);
        let expected = (200.0f64).sqrt(); // diagonal of 10×10 square
        assert!((diam - expected).abs() < 1e-6);
    }

    #[test]
    fn test_minimum_bounding_rect() {
        let hull = graham_scan(&square_points());
        let rect = minimum_bounding_rect(&hull);
        assert!((rect.area - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_merge_hulls() {
        let a = vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(0.0, 5.0),
        ];
        let b = vec![
            Point::new(10.0, 0.0),
            Point::new(15.0, 0.0),
            Point::new(15.0, 5.0),
            Point::new(10.0, 5.0),
        ];
        let merged = merge_hulls(&a, &b);
        assert!(merged.len() >= 4);
        let area = hull_area(&merged);
        assert!(area >= 50.0); // at least as large as both squares
    }

    #[test]
    fn test_collinear_points() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.0),
            Point::new(10.0, 0.0),
        ];
        let hull = graham_scan(&pts);
        assert!(hull.len() >= 2);
    }

    #[test]
    fn test_triangle_hull() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 8.66),
        ];
        let hull = graham_scan(&pts);
        assert_eq!(hull.len(), 3);
        let area = hull_area(&hull);
        assert!((area - 43.3).abs() < 0.5);
    }

    #[test]
    fn test_random_cloud() {
        // Points around a circle + some interior.
        let mut pts = Vec::new();
        for i in 0..12 {
            let angle = i as f64 * std::f64::consts::TAU / 12.0;
            pts.push(Point::new(10.0 * angle.cos(), 10.0 * angle.sin()));
        }
        pts.push(Point::new(0.0, 0.0));
        pts.push(Point::new(3.0, 3.0));
        let hull = graham_scan(&pts);
        assert_eq!(hull.len(), 12);
        assert!(point_in_hull(&Point::new(0.0, 0.0), &hull));
    }
}
