//! Animate objects along paths with arc-length parameterization.
//!
//! Replaces CSS `offset-path` / GSAP MotionPathPlugin. Supports line,
//! quadratic Bézier, cubic Bézier, and arc segments. Provides uniform-speed
//! traversal via a precomputed arc-length lookup table.

use std::fmt;

// ── Geometry ───────────────────────────────────────────────────

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

    pub fn distance_to(self, other: Point) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(self, other: Point, t: f64) -> Point {
        Point {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

// ── Path Segment ───────────────────────────────────────────────

/// A segment of a motion path.
#[derive(Debug, Clone, PartialEq)]
pub enum PathSegment {
    /// Straight line to a point.
    LineTo(Point),
    /// Quadratic Bézier curve: control point, end point.
    QuadraticTo(Point, Point),
    /// Cubic Bézier curve: control1, control2, end point.
    CubicTo(Point, Point, Point),
    /// Circular arc: center, radius, start angle (rad), end angle (rad), clockwise.
    ArcTo {
        center: Point,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        clockwise: bool,
    },
}

// ── Motion Path ────────────────────────────────────────────────

/// Number of samples per segment for the arc-length lookup table.
const LUT_SAMPLES_PER_SEGMENT: usize = 100;

/// A complete motion path built from segments.
#[derive(Debug, Clone)]
pub struct MotionPath {
    start: Point,
    segments: Vec<PathSegment>,
    /// Lookup table: cumulative arc length at each sample point.
    lut_distances: Vec<f64>,
    /// Corresponding (segment_index, local_t) for each LUT entry.
    lut_params: Vec<(usize, f64)>,
    total_length: f64,
}

impl MotionPath {
    /// Build a motion path starting at `start` with the given segments.
    pub fn new(start: Point, segments: Vec<PathSegment>) -> Self {
        let mut path = Self {
            start,
            segments,
            lut_distances: Vec::new(),
            lut_params: Vec::new(),
            total_length: 0.0,
        };
        path.build_lut();
        path
    }

    fn build_lut(&mut self) {
        self.lut_distances.clear();
        self.lut_params.clear();

        let mut cumulative = 0.0;
        let mut prev = self.start;

        self.lut_distances.push(0.0);
        self.lut_params.push((0, 0.0));

        for (seg_idx, seg) in self.segments.iter().enumerate() {
            for i in 1..=LUT_SAMPLES_PER_SEGMENT {
                let t = i as f64 / LUT_SAMPLES_PER_SEGMENT as f64;
                let pt = self.evaluate_segment(seg_idx, t);
                cumulative += prev.distance_to(pt);
                self.lut_distances.push(cumulative);
                self.lut_params.push((seg_idx, t));
                prev = pt;
            }
        }

        self.total_length = cumulative;
    }

    /// Total arc length of the path.
    pub fn total_length(&self) -> f64 {
        self.total_length
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Start point.
    pub fn start(&self) -> Point {
        self.start
    }

    /// Get the starting point for a segment (end of previous, or path start).
    fn segment_start(&self, seg_idx: usize) -> Point {
        if seg_idx == 0 {
            self.start
        } else {
            self.segment_end(seg_idx - 1)
        }
    }

    /// Get the endpoint of a segment.
    fn segment_end(&self, seg_idx: usize) -> Point {
        match &self.segments[seg_idx] {
            PathSegment::LineTo(p) => *p,
            PathSegment::QuadraticTo(_, p) => *p,
            PathSegment::CubicTo(_, _, p) => *p,
            PathSegment::ArcTo { center, radius, end_angle, .. } => {
                Point::new(
                    center.x + radius * end_angle.cos(),
                    center.y + radius * end_angle.sin(),
                )
            }
        }
    }

    /// Evaluate a point on a segment at local parameter t [0, 1].
    fn evaluate_segment(&self, seg_idx: usize, t: f64) -> Point {
        let p0 = self.segment_start(seg_idx);
        match &self.segments[seg_idx] {
            PathSegment::LineTo(p1) => p0.lerp(*p1, t),
            PathSegment::QuadraticTo(cp, p2) => {
                let a = p0.lerp(*cp, t);
                let b = cp.lerp(*p2, t);
                a.lerp(b, t)
            }
            PathSegment::CubicTo(cp1, cp2, p3) => {
                let a = p0.lerp(*cp1, t);
                let b = cp1.lerp(*cp2, t);
                let c = cp2.lerp(*p3, t);
                let d = a.lerp(b, t);
                let e = b.lerp(c, t);
                d.lerp(e, t)
            }
            PathSegment::ArcTo { center, radius, start_angle, end_angle, clockwise } => {
                let sweep = if *clockwise {
                    if end_angle > start_angle {
                        end_angle - start_angle
                    } else {
                        end_angle - start_angle + std::f64::consts::TAU
                    }
                } else if start_angle > end_angle {
                    start_angle - end_angle
                } else {
                    start_angle - end_angle + std::f64::consts::TAU
                };
                let angle = if *clockwise {
                    start_angle + sweep * t
                } else {
                    start_angle - sweep * t
                };
                Point::new(center.x + radius * angle.cos(), center.y + radius * angle.sin())
            }
        }
    }

    /// Evaluate a tangent vector on a segment at local parameter t [0, 1].
    fn tangent_segment(&self, seg_idx: usize, t: f64) -> Point {
        let epsilon = 1e-6;
        let t0 = (t - epsilon).max(0.0);
        let t1 = (t + epsilon).min(1.0);
        let p0 = self.evaluate_segment(seg_idx, t0);
        let p1 = self.evaluate_segment(seg_idx, t1);
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            Point::new(1.0, 0.0)
        } else {
            Point::new(dx / len, dy / len)
        }
    }

    /// Map a distance along the path to (segment_index, local_t) using the LUT.
    fn distance_to_param(&self, distance: f64) -> (usize, f64) {
        if distance <= 0.0 {
            return (0, 0.0);
        }
        if distance >= self.total_length {
            return self.segments.len().checked_sub(1).map_or((0, 0.0), |i| (i, 1.0));
        }

        // Binary search in the LUT.
        let idx = match self.lut_distances.binary_search_by(|d| d.partial_cmp(&distance).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(self.lut_distances.len() - 1),
        };

        if idx == 0 {
            return self.lut_params[0];
        }

        let lo = idx - 1;
        let hi = idx;
        let range = self.lut_distances[hi] - self.lut_distances[lo];
        let frac = if range > 0.0 {
            (distance - self.lut_distances[lo]) / range
        } else {
            0.0
        };

        let (seg_lo, t_lo) = self.lut_params[lo];
        let (seg_hi, t_hi) = self.lut_params[hi];

        if seg_lo == seg_hi {
            (seg_lo, t_lo + (t_hi - t_lo) * frac)
        } else {
            // Crossing a segment boundary — pick the closer one.
            if frac < 0.5 {
                self.lut_params[lo]
            } else {
                self.lut_params[hi]
            }
        }
    }

    /// Point at a given distance along the path (uniform speed).
    pub fn point_at_distance(&self, distance: f64) -> Point {
        let (seg, t) = self.distance_to_param(distance);
        self.evaluate_segment(seg, t)
    }

    /// Point at a normalized progress [0, 1] (uniform speed).
    pub fn point_at_progress(&self, progress: f64) -> Point {
        let progress = progress.clamp(0.0, 1.0);
        self.point_at_distance(progress * self.total_length)
    }

    /// Tangent angle (radians) at a distance along the path.
    pub fn tangent_angle_at_distance(&self, distance: f64) -> f64 {
        let (seg, t) = self.distance_to_param(distance);
        let tan = self.tangent_segment(seg, t);
        tan.y.atan2(tan.x)
    }

    /// Tangent angle at a normalized progress [0, 1].
    pub fn tangent_angle_at_progress(&self, progress: f64) -> f64 {
        let progress = progress.clamp(0.0, 1.0);
        self.tangent_angle_at_distance(progress * self.total_length)
    }

    /// Animate: given progress [0, 1] return (position, rotation in radians).
    pub fn animate(&self, progress: f64) -> (Point, f64) {
        let progress = progress.clamp(0.0, 1.0);
        let dist = progress * self.total_length;
        (self.point_at_distance(dist), self.tangent_angle_at_distance(dist))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn line_path() -> MotionPath {
        MotionPath::new(
            Point::new(0.0, 0.0),
            vec![PathSegment::LineTo(Point::new(100.0, 0.0))],
        )
    }

    #[test]
    fn line_total_length() {
        let path = line_path();
        assert!((path.total_length() - 100.0).abs() < 0.1);
    }

    #[test]
    fn line_midpoint() {
        let path = line_path();
        let mid = path.point_at_progress(0.5);
        assert!((mid.x - 50.0).abs() < 0.5);
        assert!((mid.y - 0.0).abs() < 0.5);
    }

    #[test]
    fn line_endpoints() {
        let path = line_path();
        let start = path.point_at_progress(0.0);
        assert!((start.x - 0.0).abs() < 0.1);
        let end = path.point_at_progress(1.0);
        assert!((end.x - 100.0).abs() < 0.1);
    }

    #[test]
    fn line_tangent_is_horizontal() {
        let path = line_path();
        let angle = path.tangent_angle_at_progress(0.5);
        assert!(angle.abs() < 0.01, "Horizontal line tangent should be ~0 rad");
    }

    #[test]
    fn quadratic_bezier_length() {
        let path = MotionPath::new(
            Point::new(0.0, 0.0),
            vec![PathSegment::QuadraticTo(
                Point::new(50.0, 100.0),
                Point::new(100.0, 0.0),
            )],
        );
        // A quadratic with control point at (50,100) should be longer than 100.
        assert!(path.total_length() > 100.0);
    }

    #[test]
    fn cubic_bezier_path() {
        let path = MotionPath::new(
            Point::new(0.0, 0.0),
            vec![PathSegment::CubicTo(
                Point::new(30.0, 80.0),
                Point::new(70.0, 80.0),
                Point::new(100.0, 0.0),
            )],
        );
        let mid = path.point_at_progress(0.5);
        // Midpoint of this symmetric curve should be near x=50.
        assert!((mid.x - 50.0).abs() < 2.0);
    }

    #[test]
    fn arc_segment() {
        let path = MotionPath::new(
            Point::new(10.0, 0.0),
            vec![PathSegment::ArcTo {
                center: Point::new(0.0, 0.0),
                radius: 10.0,
                start_angle: 0.0,
                end_angle: PI,
                clockwise: true,
            }],
        );
        // Semicircle: length ≈ π * r = 31.4
        assert!((path.total_length() - PI * 10.0).abs() < 0.5);
    }

    #[test]
    fn multi_segment_path() {
        let path = MotionPath::new(
            Point::new(0.0, 0.0),
            vec![
                PathSegment::LineTo(Point::new(50.0, 0.0)),
                PathSegment::LineTo(Point::new(50.0, 50.0)),
            ],
        );
        // L-shaped: 50 + 50 = 100
        assert!((path.total_length() - 100.0).abs() < 0.5);
        assert_eq!(path.segment_count(), 2);
    }

    #[test]
    fn animate_returns_position_and_rotation() {
        let path = line_path();
        let (pos, rot) = path.animate(0.5);
        assert!((pos.x - 50.0).abs() < 0.5);
        assert!(rot.abs() < 0.01);
    }

    #[test]
    fn point_display() {
        let p = Point::new(1.5, 2.75);
        assert_eq!(format!("{p}"), "(1.50, 2.75)");
    }

    #[test]
    fn point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn uniform_speed() {
        let path = line_path();
        // Positions at 0.25, 0.5, 0.75 should be evenly spaced.
        let p1 = path.point_at_progress(0.25);
        let p2 = path.point_at_progress(0.50);
        let p3 = path.point_at_progress(0.75);
        let d1 = p1.distance_to(p2);
        let d2 = p2.distance_to(p3);
        assert!((d1 - d2).abs() < 1.0, "Should be uniform speed");
    }
}
