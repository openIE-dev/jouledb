//! Lane Detection — Hough transform for line detection, sliding window lane
//! search, polynomial lane fitting, and curvature estimation.
//!
//! Processes a binary edge image (or raw intensity image with built-in
//! thresholding) to identify lane boundaries. The pipeline: edge detection →
//! region of interest masking → Hough lines / sliding window → polynomial
//! fit → curvature output.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Lane detection errors.
#[derive(Debug, Clone, PartialEq)]
pub enum LaneError {
    /// Image dimensions are invalid.
    InvalidImage(String),
    /// Not enough edge points to detect lanes.
    InsufficientEdges,
    /// Polynomial fit failed (singular system).
    FitFailed,
    /// Parameter out of range.
    InvalidParam(String),
}

impl fmt::Display for LaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImage(m) => write!(f, "invalid image: {m}"),
            Self::InsufficientEdges => write!(f, "insufficient edge points"),
            Self::FitFailed => write!(f, "polynomial fit failed"),
            Self::InvalidParam(m) => write!(f, "invalid parameter: {m}"),
        }
    }
}

impl std::error::Error for LaneError {}

// ── Grayscale Image ─────────────────────────────────────────────

/// Simple row-major grayscale image.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Result<Self, LaneError> {
        if width == 0 || height == 0 {
            return Err(LaneError::InvalidImage("dimensions must be > 0".into()));
        }
        Ok(Self { width, height, pixels: vec![0; width * height] })
    }

    pub fn from_data(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self, LaneError> {
        if width == 0 || height == 0 {
            return Err(LaneError::InvalidImage("dimensions must be > 0".into()));
        }
        if pixels.len() != width * height {
            return Err(LaneError::InvalidImage(format!(
                "expected {} pixels, got {}",
                width * height,
                pixels.len()
            )));
        }
        Ok(Self { width, height, pixels })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.pixels[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: u8) {
        self.pixels[y * self.width + x] = val;
    }
}

impl fmt::Display for GrayImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GrayImage({}x{})", self.width, self.height)
    }
}

// ── Hough Line ──────────────────────────────────────────────────

/// A line detected by the Hough transform in (rho, theta) form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLine {
    /// Distance from origin to the closest point on the line.
    pub rho: f64,
    /// Angle of the normal in radians [0, π).
    pub theta: f64,
    /// Number of votes in the accumulator.
    pub votes: usize,
}

impl HoughLine {
    /// Compute y for a given x (if the line is not vertical).
    pub fn y_at_x(&self, x: f64) -> Option<f64> {
        let sin_t = self.theta.sin();
        if sin_t.abs() < 1e-9 {
            return None; // vertical line
        }
        Some((self.rho - x * self.theta.cos()) / sin_t)
    }

    /// Compute x for a given y (if the line is not horizontal).
    pub fn x_at_y(&self, y: f64) -> Option<f64> {
        let cos_t = self.theta.cos();
        if cos_t.abs() < 1e-9 {
            return None;
        }
        Some((self.rho - y * self.theta.sin()) / cos_t)
    }

    /// Slope in Cartesian coordinates (rise/run), if defined.
    pub fn slope(&self) -> Option<f64> {
        let sin_t = self.theta.sin();
        if sin_t.abs() < 1e-9 {
            return None;
        }
        Some(-self.theta.cos() / sin_t)
    }
}

impl fmt::Display for HoughLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HoughLine(ρ={:.1}, θ={:.3}, votes={})", self.rho, self.theta, self.votes)
    }
}

// ── Hough Transform ─────────────────────────────────────────────

/// Classical Hough Transform for line detection on a binary edge image.
#[derive(Debug, Clone)]
pub struct HoughTransform {
    rho_resolution: f64,
    theta_resolution: f64,
    vote_threshold: usize,
    edge_threshold: u8,
}

impl HoughTransform {
    pub fn new() -> Self {
        Self {
            rho_resolution: 1.0,
            theta_resolution: std::f64::consts::PI / 180.0,
            vote_threshold: 50,
            edge_threshold: 128,
        }
    }

    pub fn with_rho_resolution(mut self, r: f64) -> Self {
        self.rho_resolution = r.max(0.1);
        self
    }

    pub fn with_theta_resolution(mut self, t: f64) -> Self {
        self.theta_resolution = t.max(0.001);
        self
    }

    pub fn with_vote_threshold(mut self, v: usize) -> Self {
        self.vote_threshold = v;
        self
    }

    pub fn with_edge_threshold(mut self, e: u8) -> Self {
        self.edge_threshold = e;
        self
    }

    /// Detect lines in the given image.
    pub fn detect(&self, image: &GrayImage) -> Vec<HoughLine> {
        let diag = ((image.width as f64).powi(2) + (image.height as f64).powi(2)).sqrt();
        let max_rho = diag;
        let rho_bins = (2.0 * max_rho / self.rho_resolution) as usize + 1;
        let theta_steps = (std::f64::consts::PI / self.theta_resolution) as usize;

        // Precompute sin/cos.
        let thetas: Vec<f64> = (0..theta_steps)
            .map(|i| i as f64 * self.theta_resolution)
            .collect();
        let cos_t: Vec<f64> = thetas.iter().map(|t| t.cos()).collect();
        let sin_t: Vec<f64> = thetas.iter().map(|t| t.sin()).collect();

        // Accumulator.
        let mut accum = vec![0usize; rho_bins * theta_steps];

        for y in 0..image.height {
            for x in 0..image.width {
                if image.get(x, y) < self.edge_threshold {
                    continue;
                }
                for ti in 0..theta_steps {
                    let rho = x as f64 * cos_t[ti] + y as f64 * sin_t[ti];
                    let ri = ((rho + max_rho) / self.rho_resolution) as usize;
                    if ri < rho_bins {
                        accum[ri * theta_steps + ti] += 1;
                    }
                }
            }
        }

        // Extract peaks above threshold.
        let mut lines = Vec::new();
        for ri in 0..rho_bins {
            for ti in 0..theta_steps {
                let votes = accum[ri * theta_steps + ti];
                if votes >= self.vote_threshold {
                    let rho = ri as f64 * self.rho_resolution - max_rho;
                    lines.push(HoughLine { rho, theta: thetas[ti], votes });
                }
            }
        }

        // Sort by votes descending.
        lines.sort_by(|a, b| b.votes.cmp(&a.votes));
        lines
    }
}

impl fmt::Display for HoughTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HoughTransform(Δρ={:.1}, Δθ={:.4}, thresh={})",
            self.rho_resolution, self.theta_resolution, self.vote_threshold
        )
    }
}

// ── Sliding Window Lane Search ──────────────────────────────────

/// Sliding window lane finder that starts from the bottom of an image
/// and walks upward, re-centring a search window on detected pixels.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    num_windows: usize,
    window_width: usize,
    min_pixels: usize,
    edge_threshold: u8,
}

impl SlidingWindow {
    pub fn new(num_windows: usize, window_width: usize) -> Self {
        Self {
            num_windows: num_windows.max(1),
            window_width: window_width.max(2),
            min_pixels: 10,
            edge_threshold: 128,
        }
    }

    pub fn with_min_pixels(mut self, m: usize) -> Self {
        self.min_pixels = m;
        self
    }

    pub fn with_edge_threshold(mut self, e: u8) -> Self {
        self.edge_threshold = e;
        self
    }

    /// Detect lane pixels starting from a base column `start_x`.
    /// Returns a list of (x, y) pixel coordinates belonging to the lane.
    pub fn detect(
        &self,
        image: &GrayImage,
        start_x: usize,
    ) -> Vec<(usize, usize)> {
        let win_h = image.height / self.num_windows;
        let half_w = self.window_width / 2;
        let mut current_x = start_x;
        let mut lane_points = Vec::new();

        for w in 0..self.num_windows {
            let y_top = image.height.saturating_sub((w + 1) * win_h);
            let y_bot = image.height.saturating_sub(w * win_h);
            let x_lo = current_x.saturating_sub(half_w);
            let x_hi = (current_x + half_w).min(image.width);

            let mut points_in_window = Vec::new();
            for y in y_top..y_bot {
                for x in x_lo..x_hi {
                    if image.get(x, y) >= self.edge_threshold {
                        points_in_window.push((x, y));
                    }
                }
            }

            if points_in_window.len() >= self.min_pixels {
                let mean_x: usize =
                    points_in_window.iter().map(|p| p.0).sum::<usize>() / points_in_window.len();
                current_x = mean_x;
            }

            lane_points.extend(points_in_window);
        }
        lane_points
    }
}

impl fmt::Display for SlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SlidingWindow(windows={}, width={})",
            self.num_windows, self.window_width
        )
    }
}

// ── Polynomial Lane Fit ─────────────────────────────────────────

/// Coefficients of a 2nd-degree polynomial: y = a*x^2 + b*x + c.
/// Here x is the image row (y-coord in image) and y is the column (x-coord).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanePoly {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl LanePoly {
    /// Evaluate the polynomial at parameter `t`.
    pub fn eval(&self, t: f64) -> f64 {
        self.a * t * t + self.b * t + self.c
    }

    /// Curvature at parameter `t`: κ = |2a| / (1 + (2at + b)^2)^(3/2).
    pub fn curvature(&self, t: f64) -> f64 {
        let dy = 2.0 * self.a * t + self.b;
        let ddy = 2.0 * self.a;
        ddy.abs() / (1.0 + dy * dy).powf(1.5)
    }

    /// Radius of curvature at parameter `t`.
    pub fn radius(&self, t: f64) -> f64 {
        let k = self.curvature(t);
        if k.abs() < 1e-12 {
            f64::INFINITY
        } else {
            1.0 / k
        }
    }
}

impl fmt::Display for LanePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lane({:.4}t² + {:.4}t + {:.4})", self.a, self.b, self.c)
    }
}

/// Fit a 2nd-order polynomial to a set of (t, value) points using
/// least-squares via the normal equations.
pub fn fit_poly2(points: &[(f64, f64)]) -> Result<LanePoly, LaneError> {
    let n = points.len();
    if n < 3 {
        return Err(LaneError::InsufficientEdges);
    }

    // Build normal equations for [a, b, c]:
    //   sum(t^4) a + sum(t^3) b + sum(t^2) c = sum(t^2 * y)
    //   sum(t^3) a + sum(t^2) b + sum(t)   c = sum(t * y)
    //   sum(t^2) a + sum(t)   b + n         c = sum(y)
    let mut s = [0.0f64; 5]; // s[k] = sum(t^k)
    let mut rhs = [0.0f64; 3]; // rhs[k] = sum(t^k * y)
    for &(t, y) in points {
        let mut tk = 1.0;
        for item in s.iter_mut() {
            *item += tk;
            tk *= t;
        }
        rhs[0] += y;
        rhs[1] += t * y;
        rhs[2] += t * t * y;
    }

    // Solve 3x3 system using Cramer's rule.
    let mat = [
        [s[4], s[3], s[2]],
        [s[3], s[2], s[1]],
        [s[2], s[1], s[0]],
    ];
    let det = determinant_3x3(&mat);
    if det.abs() < 1e-12 {
        return Err(LaneError::FitFailed);
    }

    let mut mat_a = mat;
    mat_a[0][0] = rhs[2];
    mat_a[1][0] = rhs[1];
    mat_a[2][0] = rhs[0];
    let a = determinant_3x3(&mat_a) / det;

    let mut mat_b = mat;
    mat_b[0][1] = rhs[2];
    mat_b[1][1] = rhs[1];
    mat_b[2][1] = rhs[0];
    let b = determinant_3x3(&mat_b) / det;

    let mut mat_c = mat;
    mat_c[0][2] = rhs[2];
    mat_c[1][2] = rhs[1];
    mat_c[2][2] = rhs[0];
    let c = determinant_3x3(&mat_c) / det;

    Ok(LanePoly { a, b, c })
}

fn determinant_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ── Lane Curvature Estimator ────────────────────────────────────

/// Estimate lane curvature from two polynomial lane boundaries.
#[derive(Debug, Clone)]
pub struct CurvatureEstimator {
    /// Metres per pixel in x.
    pub mx: f64,
    /// Metres per pixel in y.
    pub my: f64,
}

impl CurvatureEstimator {
    pub fn new(mx: f64, my: f64) -> Self {
        Self { mx, my }
    }

    /// Average curvature of a single lane polynomial at a given row.
    pub fn curvature_at(&self, lane: &LanePoly, row: f64) -> f64 {
        let a_scaled = lane.a * self.mx / (self.my * self.my);
        let b_scaled = lane.b * self.mx / self.my;
        let dy = 2.0 * a_scaled * row + b_scaled;
        let ddy = 2.0 * a_scaled;
        ddy.abs() / (1.0 + dy * dy).powf(1.5)
    }

    /// Lane centre offset at a given row, given left and right lane polys.
    pub fn centre_offset(
        &self,
        left: &LanePoly,
        right: &LanePoly,
        row: f64,
        image_centre_x: f64,
    ) -> f64 {
        let left_x = left.eval(row);
        let right_x = right.eval(row);
        let lane_centre = (left_x + right_x) / 2.0;
        (lane_centre - image_centre_x) * self.mx
    }
}

impl fmt::Display for CurvatureEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurvEst(mx={:.4}, my={:.4})", self.mx, self.my)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_image_create() {
        let img = GrayImage::new(10, 10).unwrap();
        assert_eq!(img.pixels.len(), 100);
    }

    #[test]
    fn test_gray_image_invalid() {
        assert!(GrayImage::new(0, 10).is_err());
    }

    #[test]
    fn test_gray_image_from_data() {
        let data = vec![128u8; 20];
        let img = GrayImage::from_data(5, 4, data).unwrap();
        assert_eq!(img.get(2, 2), 128);
    }

    #[test]
    fn test_gray_image_mismatch() {
        let data = vec![0u8; 10];
        assert!(GrayImage::from_data(5, 5, data).is_err());
    }

    #[test]
    fn test_hough_line_slope() {
        let line = HoughLine { rho: 100.0, theta: std::f64::consts::FRAC_PI_4, votes: 50 };
        let slope = line.slope().unwrap();
        assert!((slope - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_hough_line_vertical() {
        let line = HoughLine { rho: 50.0, theta: 0.0, votes: 30 };
        assert!(line.y_at_x(10.0).is_none());
        assert!(line.slope().is_none());
    }

    #[test]
    fn test_hough_line_display() {
        let line = HoughLine { rho: 10.0, theta: 1.5, votes: 42 };
        assert!(format!("{line}").contains("votes=42"));
    }

    #[test]
    fn test_hough_detect_horizontal_line() {
        let mut img = GrayImage::new(100, 100).unwrap();
        for x in 0..100 {
            img.set(x, 50, 255);
        }
        let ht = HoughTransform::new().with_vote_threshold(20);
        let lines = ht.detect(&img);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_hough_no_edges() {
        let img = GrayImage::new(50, 50).unwrap();
        let ht = HoughTransform::new();
        let lines = ht.detect(&img);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_sliding_window_basic() {
        let mut img = GrayImage::new(100, 100).unwrap();
        for y in 0..100 {
            img.set(50, y, 255);
        }
        let sw = SlidingWindow::new(10, 20).with_min_pixels(1);
        let pts = sw.detect(&img, 50);
        assert!(!pts.is_empty());
    }

    #[test]
    fn test_sliding_window_empty() {
        let img = GrayImage::new(50, 50).unwrap();
        let sw = SlidingWindow::new(5, 10).with_min_pixels(1);
        let pts = sw.detect(&img, 25);
        assert!(pts.is_empty());
    }

    #[test]
    fn test_fit_poly2_line() {
        let pts: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 2.0 * i as f64 + 3.0)).collect();
        let poly = fit_poly2(&pts).unwrap();
        assert!(poly.a.abs() < 1e-6);
        assert!((poly.b - 2.0).abs() < 1e-6);
        assert!((poly.c - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fit_poly2_quadratic() {
        let pts: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let t = i as f64;
                (t, 0.5 * t * t + 1.0 * t + 2.0)
            })
            .collect();
        let poly = fit_poly2(&pts).unwrap();
        assert!((poly.a - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_fit_poly2_insufficient() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        assert!(fit_poly2(&pts).is_err());
    }

    #[test]
    fn test_lane_poly_eval() {
        let poly = LanePoly { a: 1.0, b: 0.0, c: 0.0 };
        assert!((poly.eval(3.0) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_lane_poly_curvature() {
        let poly = LanePoly { a: 0.0, b: 1.0, c: 0.0 };
        let k = poly.curvature(0.0);
        assert!(k.abs() < 1e-9); // straight line
    }

    #[test]
    fn test_lane_poly_radius() {
        let poly = LanePoly { a: 0.5, b: 0.0, c: 0.0 };
        let r = poly.radius(0.0);
        assert!(r.is_finite());
        assert!(r > 0.0);
    }

    #[test]
    fn test_curvature_estimator() {
        let est = CurvatureEstimator::new(0.03, 0.03);
        let poly = LanePoly { a: 0.001, b: 0.0, c: 320.0 };
        let k = est.curvature_at(&poly, 500.0);
        assert!(k >= 0.0);
    }

    #[test]
    fn test_centre_offset() {
        let est = CurvatureEstimator::new(0.03, 0.03);
        let left = LanePoly { a: 0.0, b: 0.0, c: 200.0 };
        let right = LanePoly { a: 0.0, b: 0.0, c: 400.0 };
        let offset = est.centre_offset(&left, &right, 500.0, 300.0);
        assert!(offset.abs() < 1e-9); // centered
    }

    #[test]
    fn test_hough_transform_display() {
        let ht = HoughTransform::new();
        assert!(format!("{ht}").contains("HoughTransform"));
    }
}
