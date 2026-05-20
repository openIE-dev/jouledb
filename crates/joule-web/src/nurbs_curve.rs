//! NURBS Curve — Non-Uniform Rational B-Spline curves with weighted control
//! points, knot vectors, De Boor evaluation, derivative computation, knot
//! insertion, and curve splitting for CAD/CAM geometry.
//!
//! Pure-Rust NURBS implementation operating in homogeneous coordinates with
//! configurable degree, rational weight handling, and exact knot manipulation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NurbsError {
    InvalidDegree(String),
    InvalidKnotVector(String),
    InvalidWeight(String),
    InsufficientPoints(String),
    ParameterOutOfRange(String),
    SplitFailed(String),
}

impl fmt::Display for NurbsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDegree(s) => write!(f, "invalid degree: {s}"),
            Self::InvalidKnotVector(s) => write!(f, "invalid knot vector: {s}"),
            Self::InvalidWeight(s) => write!(f, "invalid weight: {s}"),
            Self::InsufficientPoints(s) => write!(f, "insufficient control points: {s}"),
            Self::ParameterOutOfRange(s) => write!(f, "parameter out of range: {s}"),
            Self::SplitFailed(s) => write!(f, "curve split failed: {s}"),
        }
    }
}

impl std::error::Error for NurbsError {}

// ── Point types ─────────────────────────────────────────────────

/// A weighted control point in 3D space.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl WeightedPoint {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn new2d(x: f64, y: f64, w: f64) -> Self {
        Self { x, y, z: 0.0, w }
    }

    /// Convert to homogeneous coordinates (wx, wy, wz, w).
    pub fn to_homogeneous(&self) -> [f64; 4] {
        [self.x * self.w, self.y * self.w, self.z * self.w, self.w]
    }

    /// Create from homogeneous coordinates.
    pub fn from_homogeneous(h: [f64; 4]) -> Self {
        let w = h[3];
        if w.abs() < 1e-15 {
            Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
        } else {
            Self { x: h[0] / w, y: h[1] / w, z: h[2] / w, w }
        }
    }

    /// Euclidean distance to another point (ignoring weights).
    pub fn distance_to(&self, other: &WeightedPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for WeightedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4}; w={:.4})", self.x, self.y, self.z, self.w)
    }
}

/// A 3D point result from curve evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

// ── Config builder ──────────────────────────────────────────────

/// Builder for NURBS curve configuration.
#[derive(Debug, Clone)]
pub struct NurbsCurveConfig {
    pub control_points: Vec<WeightedPoint>,
    pub knots: Vec<f64>,
    pub degree: usize,
}

impl NurbsCurveConfig {
    pub fn new() -> Self {
        Self {
            control_points: Vec::new(),
            knots: Vec::new(),
            degree: 3,
        }
    }

    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    pub fn with_control_points(mut self, points: Vec<WeightedPoint>) -> Self {
        self.control_points = points;
        self
    }

    pub fn with_knots(mut self, knots: Vec<f64>) -> Self {
        self.knots = knots;
        self
    }

    pub fn build(self) -> Result<NurbsCurve, NurbsError> {
        NurbsCurve::new(self.control_points, self.knots, self.degree)
    }
}

impl fmt::Display for NurbsCurveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NurbsCurveConfig(degree={}, points={}, knots={})",
            self.degree, self.control_points.len(), self.knots.len())
    }
}

// ── NURBS Curve ─────────────────────────────────────────────────

/// A Non-Uniform Rational B-Spline curve.
#[derive(Debug, Clone)]
pub struct NurbsCurve {
    control_points: Vec<WeightedPoint>,
    knots: Vec<f64>,
    degree: usize,
}

impl NurbsCurve {
    /// Create a NURBS curve from control points, knot vector, and degree.
    pub fn new(
        control_points: Vec<WeightedPoint>,
        knots: Vec<f64>,
        degree: usize,
    ) -> Result<Self, NurbsError> {
        if degree == 0 {
            return Err(NurbsError::InvalidDegree("degree must be >= 1".into()));
        }
        let n = control_points.len();
        if n < degree + 1 {
            return Err(NurbsError::InsufficientPoints(
                format!("need at least {} points for degree {}, got {}", degree + 1, degree, n),
            ));
        }
        let expected_knots = n + degree + 1;
        if knots.len() != expected_knots {
            return Err(NurbsError::InvalidKnotVector(
                format!("expected {} knots, got {}", expected_knots, knots.len()),
            ));
        }
        // Check non-decreasing
        for i in 1..knots.len() {
            if knots[i] < knots[i - 1] {
                return Err(NurbsError::InvalidKnotVector(
                    "knot vector must be non-decreasing".into(),
                ));
            }
        }
        // Check weights
        for (i, pt) in control_points.iter().enumerate() {
            if pt.w <= 0.0 {
                return Err(NurbsError::InvalidWeight(
                    format!("weight at index {} must be positive, got {}", i, pt.w),
                ));
            }
        }
        Ok(Self { control_points, knots, degree })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn control_points(&self) -> &[WeightedPoint] {
        &self.control_points
    }

    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Number of control points.
    pub fn num_points(&self) -> usize {
        self.control_points.len()
    }

    /// Domain of the curve [u_min, u_max].
    pub fn domain(&self) -> (f64, f64) {
        let p = self.degree;
        (self.knots[p], self.knots[self.knots.len() - p - 1])
    }

    /// Find the knot span index for parameter u (binary search).
    fn find_span(&self, u: f64) -> usize {
        let n = self.control_points.len() - 1;
        let p = self.degree;
        let (u_min, u_max) = self.domain();
        if u >= u_max {
            // Special case: at the end of the domain
            return n;
        }
        if u <= u_min {
            return p;
        }
        let mut lo = p;
        let mut hi = n + 1;
        let mut mid = (lo + hi) / 2;
        while u < self.knots[mid] || u >= self.knots[mid + 1] {
            if u < self.knots[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
        }
        mid
    }

    /// De Boor evaluation in homogeneous coordinates.
    pub fn evaluate(&self, u: f64) -> Result<Point3, NurbsError> {
        let (u_min, u_max) = self.domain();
        let u_clamped = u.clamp(u_min, u_max);
        let span = self.find_span(u_clamped);
        let p = self.degree;

        // Collect homogeneous control points for the relevant span
        let mut d: Vec<[f64; 4]> = Vec::with_capacity(p + 1);
        for j in 0..=p {
            let idx = span.saturating_sub(p) + j;
            let idx = idx.min(self.control_points.len() - 1);
            d.push(self.control_points[idx].to_homogeneous());
        }

        // De Boor recursion
        for r in 1..=p {
            for j in (r..=p).rev() {
                let knot_idx = span.saturating_sub(p) + j;
                let left = self.knots[knot_idx];
                let right_idx = knot_idx + p + 1 - r;
                let right = if right_idx < self.knots.len() {
                    self.knots[right_idx]
                } else {
                    self.knots[self.knots.len() - 1]
                };
                let denom = right - left;
                if denom.abs() < 1e-15 {
                    continue;
                }
                let alpha = (u_clamped - left) / denom;
                for k in 0..4 {
                    d[j][k] = (1.0 - alpha) * d[j - 1][k] + alpha * d[j][k];
                }
            }
        }

        let h = d[p];
        let pt = WeightedPoint::from_homogeneous(h);
        Ok(Point3::new(pt.x, pt.y, pt.z))
    }

    /// Evaluate multiple points along the curve.
    pub fn evaluate_many(&self, count: usize) -> Result<Vec<Point3>, NurbsError> {
        if count < 2 {
            return Err(NurbsError::ParameterOutOfRange("count must be >= 2".into()));
        }
        let (u_min, u_max) = self.domain();
        let mut points = Vec::with_capacity(count);
        for i in 0..count {
            let t = i as f64 / (count - 1) as f64;
            let u = u_min + t * (u_max - u_min);
            points.push(self.evaluate(u)?);
        }
        Ok(points)
    }

    /// First derivative at parameter u using finite differences in homogeneous space.
    pub fn derivative(&self, u: f64) -> Result<Point3, NurbsError> {
        let h = 1e-7;
        let (u_min, u_max) = self.domain();
        let u = u.clamp(u_min, u_max);
        let u_lo = (u - h).max(u_min);
        let u_hi = (u + h).min(u_max);
        let delta = u_hi - u_lo;
        if delta.abs() < 1e-15 {
            return Ok(Point3::new(0.0, 0.0, 0.0));
        }
        let p_lo = self.evaluate(u_lo)?;
        let p_hi = self.evaluate(u_hi)?;
        Ok(Point3::new(
            (p_hi.x - p_lo.x) / delta,
            (p_hi.y - p_lo.y) / delta,
            (p_hi.z - p_lo.z) / delta,
        ))
    }

    /// Insert a knot at parameter u, increasing multiplicity by 1.
    pub fn insert_knot(&mut self, u: f64) -> Result<(), NurbsError> {
        let (u_min, u_max) = self.domain();
        if u < u_min || u > u_max {
            return Err(NurbsError::ParameterOutOfRange(
                format!("u={} outside domain [{}, {}]", u, u_min, u_max),
            ));
        }
        let span = self.find_span(u);
        let p = self.degree;
        let n = self.control_points.len();

        // Build new control points in homogeneous space
        let mut new_pts: Vec<[f64; 4]> = Vec::with_capacity(n + 1);

        for i in 0..=n {
            if i <= span.saturating_sub(p) {
                new_pts.push(self.control_points[i].to_homogeneous());
            } else if i <= span {
                let ki = self.knots[i];
                let ki_p = self.knots[i + p];
                let denom = ki_p - ki;
                if denom.abs() < 1e-15 {
                    new_pts.push(self.control_points[i.min(n - 1)].to_homogeneous());
                } else {
                    let alpha = (u - ki) / denom;
                    let h0 = self.control_points[(i - 1).min(n - 1)].to_homogeneous();
                    let h1 = self.control_points[i.min(n - 1)].to_homogeneous();
                    let mut blended = [0.0; 4];
                    for k in 0..4 {
                        blended[k] = (1.0 - alpha) * h0[k] + alpha * h1[k];
                    }
                    new_pts.push(blended);
                }
            } else {
                new_pts.push(self.control_points[(i - 1).min(n - 1)].to_homogeneous());
            }
        }

        // Build new knot vector
        let mut new_knots = Vec::with_capacity(self.knots.len() + 1);
        for i in 0..=span {
            new_knots.push(self.knots[i]);
        }
        new_knots.push(u);
        for i in (span + 1)..self.knots.len() {
            new_knots.push(self.knots[i]);
        }

        self.control_points = new_pts
            .into_iter()
            .map(WeightedPoint::from_homogeneous)
            .collect();
        self.knots = new_knots;
        Ok(())
    }

    /// Split the curve at parameter u into two NURBS curves.
    pub fn split(&self, u: f64) -> Result<(NurbsCurve, NurbsCurve), NurbsError> {
        let (u_min, u_max) = self.domain();
        if u <= u_min || u >= u_max {
            return Err(NurbsError::SplitFailed(
                format!("split parameter {} must be strictly inside domain ({}, {})", u, u_min, u_max),
            ));
        }
        let p = self.degree;

        // Insert knot p times to get C^0 continuity at the split point
        let mut working = self.clone();
        for _ in 0..p {
            working.insert_knot(u)?;
        }

        // Find the split index in the knot vector
        let mut split_idx = 0;
        for (i, &k) in working.knots.iter().enumerate() {
            if (k - u).abs() < 1e-12 {
                split_idx = i;
                break;
            }
        }

        // Left curve
        let left_knots: Vec<f64> = working.knots[..=split_idx + p].to_vec();
        let left_n = left_knots.len() - p - 1;
        let left_pts: Vec<WeightedPoint> = working.control_points[..left_n].to_vec();

        // Right curve
        let right_knots: Vec<f64> = working.knots[split_idx..].to_vec();
        let right_n = right_knots.len() - p - 1;
        let right_start = working.control_points.len().saturating_sub(right_n);
        let right_pts: Vec<WeightedPoint> = working.control_points[right_start..].to_vec();

        let left = NurbsCurve::new(left_pts, left_knots, p)
            .map_err(|e| NurbsError::SplitFailed(format!("left: {e}")))?;
        let right = NurbsCurve::new(right_pts, right_knots, p)
            .map_err(|e| NurbsError::SplitFailed(format!("right: {e}")))?;
        Ok((left, right))
    }

    /// Create a clamped uniform knot vector for n control points and degree p.
    pub fn clamped_uniform_knots(n: usize, p: usize) -> Vec<f64> {
        let m = n + p + 1;
        let mut knots = Vec::with_capacity(m);
        for _ in 0..=p {
            knots.push(0.0);
        }
        let internal = m - 2 * (p + 1);
        for i in 1..=internal {
            knots.push(i as f64 / (internal + 1) as f64);
        }
        for _ in 0..=p {
            knots.push(1.0);
        }
        knots
    }

    /// Approximate arc length by evaluating many points.
    pub fn arc_length(&self, segments: usize) -> Result<f64, NurbsError> {
        let seg = segments.max(2);
        let pts = self.evaluate_many(seg + 1)?;
        let mut length = 0.0;
        for i in 1..pts.len() {
            length += pts[i - 1].distance_to(&pts[i]);
        }
        Ok(length)
    }
}

impl fmt::Display for NurbsCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (u_min, u_max) = self.domain();
        write!(f, "NurbsCurve(degree={}, points={}, domain=[{:.4}, {:.4}])",
            self.degree, self.control_points.len(), u_min, u_max)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_curve() -> NurbsCurve {
        let pts = vec![
            WeightedPoint::new2d(0.0, 0.0, 1.0),
            WeightedPoint::new2d(1.0, 0.0, 1.0),
        ];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        NurbsCurve::new(pts, knots, 1).unwrap()
    }

    fn quad_curve() -> NurbsCurve {
        let pts = vec![
            WeightedPoint::new2d(0.0, 0.0, 1.0),
            WeightedPoint::new2d(0.5, 1.0, 1.0),
            WeightedPoint::new2d(1.0, 0.0, 1.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        NurbsCurve::new(pts, knots, 2).unwrap()
    }

    #[test]
    fn test_line_endpoints() {
        let c = line_curve();
        let p0 = c.evaluate(0.0).unwrap();
        let p1 = c.evaluate(1.0).unwrap();
        assert!((p0.x).abs() < 1e-10);
        assert!((p1.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_midpoint() {
        let c = line_curve();
        let mid = c.evaluate(0.5).unwrap();
        assert!((mid.x - 0.5).abs() < 1e-10);
        assert!((mid.y).abs() < 1e-10);
    }

    #[test]
    fn test_quad_endpoints() {
        let c = quad_curve();
        let p0 = c.evaluate(0.0).unwrap();
        let p1 = c.evaluate(1.0).unwrap();
        assert!((p0.x).abs() < 1e-10);
        assert!((p1.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_midpoint() {
        let c = quad_curve();
        let mid = c.evaluate(0.5).unwrap();
        assert!((mid.x - 0.5).abs() < 1e-6);
        assert!(mid.y > 0.0); // parabolic arch
    }

    #[test]
    fn test_evaluate_many() {
        let c = line_curve();
        let pts = c.evaluate_many(11).unwrap();
        assert_eq!(pts.len(), 11);
        assert!((pts[0].x).abs() < 1e-10);
        assert!((pts[10].x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_line() {
        let c = line_curve();
        let d = c.derivative(0.5).unwrap();
        assert!((d.x - 1.0).abs() < 1e-4);
        assert!(d.y.abs() < 1e-4);
    }

    #[test]
    fn test_domain() {
        let c = quad_curve();
        let (lo, hi) = c.domain();
        assert!((lo).abs() < 1e-15);
        assert!((hi - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_clamped_uniform_knots() {
        let knots = NurbsCurve::clamped_uniform_knots(5, 3);
        assert_eq!(knots.len(), 9); // n + p + 1
        assert!((knots[0]).abs() < 1e-15);
        assert!((knots[8] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_insert_knot() {
        let mut c = quad_curve();
        let before = c.evaluate(0.5).unwrap();
        c.insert_knot(0.5).unwrap();
        let after = c.evaluate(0.5).unwrap();
        assert!(before.distance_to(&after) < 1e-8);
        assert_eq!(c.num_points(), 4);
    }

    #[test]
    fn test_knot_insertion_preserves_endpoints() {
        let mut c = quad_curve();
        let p0_before = c.evaluate(0.0).unwrap();
        let p1_before = c.evaluate(1.0).unwrap();
        c.insert_knot(0.3).unwrap();
        let p0_after = c.evaluate(0.0).unwrap();
        let p1_after = c.evaluate(1.0).unwrap();
        assert!(p0_before.distance_to(&p0_after) < 1e-8);
        assert!(p1_before.distance_to(&p1_after) < 1e-8);
    }

    #[test]
    fn test_arc_length_line() {
        let c = line_curve();
        let len = c.arc_length(100).unwrap();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_degree_zero() {
        let pts = vec![WeightedPoint::new2d(0.0, 0.0, 1.0)];
        let knots = vec![0.0, 1.0];
        assert!(NurbsCurve::new(pts, knots, 0).is_err());
    }

    #[test]
    fn test_invalid_knot_count() {
        let pts = vec![
            WeightedPoint::new2d(0.0, 0.0, 1.0),
            WeightedPoint::new2d(1.0, 0.0, 1.0),
        ];
        let knots = vec![0.0, 1.0]; // wrong count
        assert!(NurbsCurve::new(pts, knots, 1).is_err());
    }

    #[test]
    fn test_invalid_weight() {
        let pts = vec![
            WeightedPoint::new2d(0.0, 0.0, 0.0), // zero weight
            WeightedPoint::new2d(1.0, 0.0, 1.0),
        ];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        assert!(NurbsCurve::new(pts, knots, 1).is_err());
    }

    #[test]
    fn test_weighted_circle_arc() {
        // Quarter circle as rational quadratic
        let w = (2.0_f64).sqrt() / 2.0;
        let pts = vec![
            WeightedPoint::new2d(1.0, 0.0, 1.0),
            WeightedPoint::new2d(1.0, 1.0, w),
            WeightedPoint::new2d(0.0, 1.0, 1.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = NurbsCurve::new(pts, knots, 2).unwrap();
        let mid = c.evaluate(0.5).unwrap();
        let radius = (mid.x * mid.x + mid.y * mid.y).sqrt();
        assert!((radius - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_builder() {
        let c = NurbsCurveConfig::new()
            .with_degree(1)
            .with_control_points(vec![
                WeightedPoint::new2d(0.0, 0.0, 1.0),
                WeightedPoint::new2d(1.0, 1.0, 1.0),
            ])
            .with_knots(vec![0.0, 0.0, 1.0, 1.0])
            .build()
            .unwrap();
        assert_eq!(c.degree(), 1);
        assert_eq!(c.num_points(), 2);
    }

    #[test]
    fn test_display() {
        let c = line_curve();
        let s = format!("{c}");
        assert!(s.contains("NurbsCurve"));
        assert!(s.contains("degree=1"));
    }

    #[test]
    fn test_point3_display() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let s = format!("{p}");
        assert!(s.contains("1.0"));
    }

    #[test]
    fn test_weighted_point_display() {
        let wp = WeightedPoint::new(1.0, 2.0, 3.0, 0.5);
        let s = format!("{wp}");
        assert!(s.contains("w=0.5"));
    }
}
