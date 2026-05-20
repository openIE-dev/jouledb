//! Polygon clipping algorithms: Sutherland-Hodgman clipping against convex polygon,
//! Cohen-Sutherland line clipping, Liang-Barsky line clipping, polygon-polygon
//! clipping, clip to rectangle.
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
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Rectangle ──────────────────────────────────────────────────

/// Axis-aligned rectangle for clipping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipRect {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl ClipRect {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self {
        Self { x_min, y_min, x_max, y_max }
    }

    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn contains(&self, p: &Point) -> bool {
        p.x >= self.x_min && p.x <= self.x_max && p.y >= self.y_min && p.y <= self.y_max
    }

    /// Convert to polygon (CCW).
    pub fn to_polygon(&self) -> Vec<Point> {
        vec![
            Point::new(self.x_min, self.y_min),
            Point::new(self.x_max, self.y_min),
            Point::new(self.x_max, self.y_max),
            Point::new(self.x_min, self.y_max),
        ]
    }
}

// ── Cohen-Sutherland ───────────────────────────────────────────

/// Outcode bits for Cohen-Sutherland.
const INSIDE: u8 = 0b0000;
const LEFT: u8 = 0b0001;
const RIGHT: u8 = 0b0010;
const BOTTOM: u8 = 0b0100;
const TOP: u8 = 0b1000;

fn compute_outcode(p: &Point, rect: &ClipRect) -> u8 {
    let mut code = INSIDE;
    if p.x < rect.x_min {
        code |= LEFT;
    } else if p.x > rect.x_max {
        code |= RIGHT;
    }
    if p.y < rect.y_min {
        code |= BOTTOM;
    } else if p.y > rect.y_max {
        code |= TOP;
    }
    code
}

/// Cohen-Sutherland line clipping against an axis-aligned rectangle.
/// Returns the clipped endpoints, or None if fully outside.
pub fn cohen_sutherland_clip(
    p0: Point,
    p1: Point,
    rect: &ClipRect,
) -> Option<(Point, Point)> {
    let mut x0 = p0.x;
    let mut y0 = p0.y;
    let mut x1 = p1.x;
    let mut y1 = p1.y;
    let mut code0 = compute_outcode(&Point::new(x0, y0), rect);
    let mut code1 = compute_outcode(&Point::new(x1, y1), rect);

    loop {
        if (code0 | code1) == 0 {
            // Both inside.
            return Some((Point::new(x0, y0), Point::new(x1, y1)));
        }
        if (code0 & code1) != 0 {
            // Both on same outside side.
            return None;
        }

        let code_out = if code0 != 0 { code0 } else { code1 };
        let (x, y);

        if code_out & TOP != 0 {
            x = x0 + (x1 - x0) * (rect.y_max - y0) / (y1 - y0);
            y = rect.y_max;
        } else if code_out & BOTTOM != 0 {
            x = x0 + (x1 - x0) * (rect.y_min - y0) / (y1 - y0);
            y = rect.y_min;
        } else if code_out & RIGHT != 0 {
            y = y0 + (y1 - y0) * (rect.x_max - x0) / (x1 - x0);
            x = rect.x_max;
        } else {
            // LEFT
            y = y0 + (y1 - y0) * (rect.x_min - x0) / (x1 - x0);
            x = rect.x_min;
        }

        if code_out == code0 {
            x0 = x;
            y0 = y;
            code0 = compute_outcode(&Point::new(x0, y0), rect);
        } else {
            x1 = x;
            y1 = y;
            code1 = compute_outcode(&Point::new(x1, y1), rect);
        }
    }
}

// ── Liang-Barsky ───────────────────────────────────────────────

/// Liang-Barsky line clipping against an axis-aligned rectangle.
/// Returns the clipped endpoints, or None if fully outside.
pub fn liang_barsky_clip(
    p0: Point,
    p1: Point,
    rect: &ClipRect,
) -> Option<(Point, Point)> {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;

    let p = [-dx, dx, -dy, dy];
    let q = [
        p0.x - rect.x_min,
        rect.x_max - p0.x,
        p0.y - rect.y_min,
        rect.y_max - p0.y,
    ];

    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;

    for i in 0..4 {
        if p[i].abs() < 1e-12 {
            // Parallel to this edge.
            if q[i] < 0.0 {
                return None; // Outside
            }
            continue;
        }
        let t = q[i] / p[i];
        if p[i] < 0.0 {
            // Entering
            if t > t1 {
                return None;
            }
            t0 = t0.max(t);
        } else {
            // Leaving
            if t < t0 {
                return None;
            }
            t1 = t1.min(t);
        }
    }

    if t0 > t1 {
        return None;
    }

    Some((
        Point::new(p0.x + t0 * dx, p0.y + t0 * dy),
        Point::new(p0.x + t1 * dx, p0.y + t1 * dy),
    ))
}

// ── Sutherland-Hodgman ─────────────────────────────────────────

/// Cross product of edge vector and point vector (2D).
fn edge_cross(edge_start: &Point, edge_end: &Point, p: &Point) -> f64 {
    (edge_end.x - edge_start.x) * (p.y - edge_start.y)
        - (edge_end.y - edge_start.y) * (p.x - edge_start.x)
}

fn is_inside_edge(p: &Point, edge_start: &Point, edge_end: &Point) -> bool {
    edge_cross(edge_start, edge_end, p) >= 0.0
}

fn line_intersect(a1: &Point, a2: &Point, b1: &Point, b2: &Point) -> Option<Point> {
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

/// Sutherland-Hodgman polygon clipping against a convex clipping polygon.
pub fn sutherland_hodgman(subject: &[Point], clip: &[Point]) -> Vec<Point> {
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
            let curr_in = is_inside_edge(&current, &edge_start, &edge_end);
            let prev_in = is_inside_edge(&previous, &edge_start, &edge_end);

            if curr_in {
                if !prev_in {
                    if let Some(inter) = line_intersect(&previous, &current, &edge_start, &edge_end) {
                        output.push(inter);
                    }
                }
                output.push(current);
            } else if prev_in {
                if let Some(inter) = line_intersect(&previous, &current, &edge_start, &edge_end) {
                    output.push(inter);
                }
            }
        }
    }

    output
}

/// Clip a polygon against an axis-aligned rectangle (Sutherland-Hodgman).
pub fn clip_polygon_to_rect(polygon: &[Point], rect: &ClipRect) -> Vec<Point> {
    let clip = rect.to_polygon();
    sutherland_hodgman(polygon, &clip)
}

// ── Polygon-Polygon Clipping ───────────────────────────────────

/// Clip polygon A against convex polygon B (intersection).
pub fn clip_polygon_by_polygon(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    sutherland_hodgman(subject, clip)
}

/// Compute the area of a polygon (shoelace formula).
pub fn polygon_area(pts: &[Point]) -> f64 {
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
    (area / 2.0).abs()
}

/// Compute the centroid of a polygon.
pub fn polygon_centroid(pts: &[Point]) -> Point {
    let n = pts.len();
    if n == 0 {
        return Point::new(0.0, 0.0);
    }
    let cx: f64 = pts.iter().map(|p| p.x).sum::<f64>() / n as f64;
    let cy: f64 = pts.iter().map(|p| p.y).sum::<f64>() / n as f64;
    Point::new(cx, cy)
}

/// Clip multiple line segments against a rectangle, returning surviving segments.
pub fn clip_polyline_to_rect(
    polyline: &[Point],
    rect: &ClipRect,
) -> Vec<(Point, Point)> {
    let mut result = Vec::new();
    for i in 0..polyline.len().saturating_sub(1) {
        if let Some(clipped) = cohen_sutherland_clip(polyline[i], polyline[i + 1], rect) {
            result.push(clipped);
        }
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_rect() -> ClipRect {
        ClipRect::new(0.0, 0.0, 10.0, 10.0)
    }

    #[test]
    fn test_cohen_sutherland_fully_inside() {
        let r = unit_rect();
        let result = cohen_sutherland_clip(Point::new(2.0, 2.0), Point::new(8.0, 8.0), &r);
        let (a, b) = result.unwrap();
        assert!((a.x - 2.0).abs() < 1e-10);
        assert!((b.x - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_sutherland_fully_outside() {
        let r = unit_rect();
        let result = cohen_sutherland_clip(Point::new(-5.0, -5.0), Point::new(-1.0, -1.0), &r);
        assert!(result.is_none());
    }

    #[test]
    fn test_cohen_sutherland_partial_clip() {
        let r = unit_rect();
        let result = cohen_sutherland_clip(Point::new(-5.0, 5.0), Point::new(15.0, 5.0), &r);
        let (a, b) = result.unwrap();
        assert!((a.x - 0.0).abs() < 1e-10);
        assert!((b.x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_liang_barsky_fully_inside() {
        let r = unit_rect();
        let result = liang_barsky_clip(Point::new(3.0, 3.0), Point::new(7.0, 7.0), &r);
        let (a, b) = result.unwrap();
        assert!((a.x - 3.0).abs() < 1e-10);
        assert!((b.x - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_liang_barsky_clipped() {
        let r = unit_rect();
        let result = liang_barsky_clip(Point::new(5.0, -5.0), Point::new(5.0, 15.0), &r);
        let (a, b) = result.unwrap();
        assert!((a.y - 0.0).abs() < 1e-10);
        assert!((b.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_liang_barsky_outside() {
        let r = unit_rect();
        let result = liang_barsky_clip(Point::new(15.0, 15.0), Point::new(20.0, 20.0), &r);
        assert!(result.is_none());
    }

    #[test]
    fn test_sutherland_hodgman_square_clip() {
        let subject = vec![
            Point::new(-5.0, -5.0),
            Point::new(15.0, -5.0),
            Point::new(15.0, 15.0),
            Point::new(-5.0, 15.0),
        ];
        let clip = unit_rect().to_polygon();
        let result = sutherland_hodgman(&subject, &clip);
        assert_eq!(result.len(), 4);
        let area = polygon_area(&result);
        assert!((area - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_polygon_to_rect() {
        // Triangle that extends beyond the rect.
        let triangle = vec![
            Point::new(5.0, -5.0),
            Point::new(15.0, 5.0),
            Point::new(5.0, 15.0),
        ];
        let r = unit_rect();
        let clipped = clip_polygon_to_rect(&triangle, &r);
        assert!(!clipped.is_empty());
        let area = polygon_area(&clipped);
        assert!(area > 0.0);
        assert!(area < r.area());
    }

    #[test]
    fn test_polygon_area_square() {
        let sq = vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(0.0, 5.0),
        ];
        assert!((polygon_area(&sq) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_centroid() {
        let sq = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let c = polygon_centroid(&sq);
        assert!((c.x - 5.0).abs() < 1e-10);
        assert!((c.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_rect_contains() {
        let r = unit_rect();
        assert!(r.contains(&Point::new(5.0, 5.0)));
        assert!(!r.contains(&Point::new(-1.0, 5.0)));
    }

    #[test]
    fn test_clip_polyline() {
        let line = vec![
            Point::new(-5.0, 5.0),
            Point::new(5.0, 5.0),
            Point::new(15.0, 5.0),
        ];
        let r = unit_rect();
        let segments = clip_polyline_to_rect(&line, &r);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_polygon_by_polygon_clip() {
        let a = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let b = vec![
            Point::new(5.0, 5.0),
            Point::new(15.0, 5.0),
            Point::new(15.0, 15.0),
            Point::new(5.0, 15.0),
        ];
        let clipped = clip_polygon_by_polygon(&a, &b);
        let area = polygon_area(&clipped);
        assert!((area - 25.0).abs() < 1e-6);
    }
}
