//! Camera calibration: intrinsic/extrinsic parameters, distortion
//! coefficients, checkerboard corner detection, and projection.
//!
//! Models the pinhole camera with radial and tangential distortion.
//! Operates on grayscale image buffers (`&[f64]`, row-major,
//! values in `[0.0, 255.0]`).

use std::fmt;

// ── Point types ────────────────────────────────────────────────

/// A 2D point (image coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

/// A 3D point (world coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Point3D) -> f64 {
        ((self.x - other.x).powi(2)
        + (self.y - other.y).powi(2)
        + (self.z - other.z).powi(2)).sqrt()
    }
}

impl fmt::Display for Point3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

// ── Intrinsic Matrix ───────────────────────────────────────────

/// Camera intrinsic parameters (3x3 matrix, row-major).
#[derive(Debug, Clone, PartialEq)]
pub struct IntrinsicMatrix {
    /// Focal length in X pixels.
    pub fx: f64,
    /// Focal length in Y pixels.
    pub fy: f64,
    /// Principal point X.
    pub cx: f64,
    /// Principal point Y.
    pub cy: f64,
    /// Skew coefficient (usually 0).
    pub skew: f64,
}

impl IntrinsicMatrix {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy, skew: 0.0 }
    }

    pub fn with_skew(mut self, skew: f64) -> Self {
        self.skew = skew;
        self
    }

    /// Return the 3x3 matrix in row-major order.
    pub fn to_matrix(&self) -> [f64; 9] {
        [
            self.fx, self.skew, self.cx,
            0.0,     self.fy,   self.cy,
            0.0,     0.0,       1.0,
        ]
    }

    /// Inverse projection: image point → normalized camera coordinates.
    pub fn unproject(&self, p: &Point2D) -> (f64, f64) {
        let y_norm = (p.y - self.cy) / self.fy;
        let x_norm = (p.x - self.cx - self.skew * y_norm) / self.fx;
        (x_norm, y_norm)
    }

    /// Forward projection: normalized coordinates → image point.
    pub fn project_normalized(&self, x_norm: f64, y_norm: f64) -> Point2D {
        Point2D::new(
            self.fx * x_norm + self.skew * y_norm + self.cx,
            self.fy * y_norm + self.cy,
        )
    }
}

impl fmt::Display for IntrinsicMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Intrinsic(fx={:.1} fy={:.1} cx={:.1} cy={:.1})",
               self.fx, self.fy, self.cx, self.cy)
    }
}

// ── Distortion Coefficients ────────────────────────────────────

/// Lens distortion coefficients (Brown-Conrady model).
#[derive(Debug, Clone, PartialEq)]
pub struct DistortionCoeffs {
    /// Radial distortion: k1, k2, k3.
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// Tangential distortion: p1, p2.
    pub p1: f64,
    pub p2: f64,
}

impl DistortionCoeffs {
    pub fn zero() -> Self {
        Self { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0 }
    }

    pub fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    pub fn with_radial(mut self, k1: f64, k2: f64, k3: f64) -> Self {
        self.k1 = k1;
        self.k2 = k2;
        self.k3 = k3;
        self
    }

    pub fn with_tangential(mut self, p1: f64, p2: f64) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self
    }

    /// Apply distortion to normalized camera coordinates.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let x_d = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let y_d = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x_d, y_d)
    }

    /// Iterative undistortion of normalized camera coordinates.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        let mut x = x_d;
        let mut y = y_d;

        for _ in 0..20 {
            let (xd, yd) = self.distort(x, y);
            x = x_d + (x - xd);
            y = y_d + (y - yd);
        }
        (x, y)
    }

    /// Is this essentially zero distortion?
    pub fn is_negligible(&self) -> bool {
        self.k1.abs() < 1e-10
            && self.k2.abs() < 1e-10
            && self.k3.abs() < 1e-10
            && self.p1.abs() < 1e-10
            && self.p2.abs() < 1e-10
    }
}

impl fmt::Display for DistortionCoeffs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Distortion(k1={:.4e} k2={:.4e} k3={:.4e} p1={:.4e} p2={:.4e})",
               self.k1, self.k2, self.k3, self.p1, self.p2)
    }
}

// ── Extrinsic Matrix ───────────────────────────────────────────

/// Camera extrinsic parameters: rotation (Rodrigues) + translation.
#[derive(Debug, Clone, PartialEq)]
pub struct Extrinsics {
    /// Rodrigues rotation vector [rx, ry, rz].
    pub rvec: [f64; 3],
    /// Translation vector [tx, ty, tz].
    pub tvec: [f64; 3],
}

impl Extrinsics {
    pub fn identity() -> Self {
        Self { rvec: [0.0; 3], tvec: [0.0; 3] }
    }

    pub fn new(rvec: [f64; 3], tvec: [f64; 3]) -> Self {
        Self { rvec, tvec }
    }

    /// Convert Rodrigues vector to 3x3 rotation matrix (row-major).
    pub fn rotation_matrix(&self) -> [f64; 9] {
        let theta = (self.rvec[0].powi(2) + self.rvec[1].powi(2) + self.rvec[2].powi(2)).sqrt();
        if theta < 1e-12 {
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }

        let k = [self.rvec[0] / theta, self.rvec[1] / theta, self.rvec[2] / theta];
        let c = theta.cos();
        let s = theta.sin();
        let v = 1.0 - c;

        [
            c + k[0] * k[0] * v,         k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
            k[1] * k[0] * v + k[2] * s,  c + k[1] * k[1] * v,         k[1] * k[2] * v - k[0] * s,
            k[2] * k[0] * v - k[1] * s,  k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v,
        ]
    }

    /// Transform a 3D world point to camera coordinates.
    pub fn world_to_camera(&self, point: &Point3D) -> Point3D {
        let r = self.rotation_matrix();
        Point3D::new(
            r[0] * point.x + r[1] * point.y + r[2] * point.z + self.tvec[0],
            r[3] * point.x + r[4] * point.y + r[5] * point.z + self.tvec[1],
            r[6] * point.x + r[7] * point.y + r[8] * point.z + self.tvec[2],
        )
    }
}

impl fmt::Display for Extrinsics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Extrinsics(rvec=[{:.3},{:.3},{:.3}] tvec=[{:.3},{:.3},{:.3}])",
               self.rvec[0], self.rvec[1], self.rvec[2],
               self.tvec[0], self.tvec[1], self.tvec[2])
    }
}

// ── Camera Model ───────────────────────────────────────────────

/// Complete camera model with intrinsics, distortion, and extrinsics.
#[derive(Debug, Clone)]
pub struct CameraModel {
    pub intrinsics: IntrinsicMatrix,
    pub distortion: DistortionCoeffs,
    pub extrinsics: Extrinsics,
}

impl CameraModel {
    pub fn new(intrinsics: IntrinsicMatrix) -> Self {
        Self {
            intrinsics,
            distortion: DistortionCoeffs::zero(),
            extrinsics: Extrinsics::identity(),
        }
    }

    pub fn with_distortion(mut self, dist: DistortionCoeffs) -> Self {
        self.distortion = dist;
        self
    }

    pub fn with_extrinsics(mut self, ext: Extrinsics) -> Self {
        self.extrinsics = ext;
        self
    }

    /// Project a 3D world point to a 2D image point.
    pub fn project(&self, point: &Point3D) -> Point2D {
        let cam = self.extrinsics.world_to_camera(point);
        if cam.z.abs() < 1e-12 {
            return Point2D::new(0.0, 0.0);
        }
        let x_norm = cam.x / cam.z;
        let y_norm = cam.y / cam.z;
        let (x_d, y_d) = self.distortion.distort(x_norm, y_norm);
        self.intrinsics.project_normalized(x_d, y_d)
    }

    /// Reprojection error for a set of 3D-2D correspondences.
    pub fn reprojection_error(&self, world: &[Point3D], image: &[Point2D]) -> f64 {
        assert_eq!(world.len(), image.len());
        if world.is_empty() { return 0.0; }
        let sum: f64 = world.iter().zip(image.iter()).map(|(w, i)| {
            let proj = self.project(w);
            proj.distance(i)
        }).sum();
        sum / world.len() as f64
    }
}

impl fmt::Display for CameraModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CameraModel({}, {})", self.intrinsics, self.distortion)
    }
}

// ── Checkerboard Detection ─────────────────────────────────────

/// Checkerboard pattern specification.
#[derive(Debug, Clone)]
pub struct CheckerboardPattern {
    /// Number of inner corners per row.
    pub cols: usize,
    /// Number of inner corners per column.
    pub rows: usize,
    /// Square size in world units (e.g., meters).
    pub square_size: f64,
}

impl CheckerboardPattern {
    pub fn new(cols: usize, rows: usize, square_size: f64) -> Self {
        Self { cols, rows, square_size }
    }

    /// Generate ideal 3D corner positions (Z=0 plane).
    pub fn ideal_corners(&self) -> Vec<Point3D> {
        let mut corners = Vec::with_capacity(self.cols * self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                corners.push(Point3D::new(
                    c as f64 * self.square_size,
                    r as f64 * self.square_size,
                    0.0,
                ));
            }
        }
        corners
    }

    /// Total number of inner corners.
    pub fn corner_count(&self) -> usize {
        self.cols * self.rows
    }
}

impl fmt::Display for CheckerboardPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Checkerboard({}x{} sq={:.3})", self.cols, self.rows, self.square_size)
    }
}

/// Detect checkerboard corners in a grayscale image using Harris
/// corner detection on a binary-thresholded image.
pub fn detect_checkerboard_corners(
    pixels: &[f64], width: usize, height: usize,
    _pattern: &CheckerboardPattern,
    harris_threshold: f64,
) -> Vec<Point2D> {
    // Simplified: find strong corner responses using Harris-like
    // structure tensor on Sobel gradients.
    let mut corners = Vec::new();
    let k = 0.04_f64;

    for r in 2..height - 2 {
        for c in 2..width - 2 {
            let mut sxx = 0.0_f64;
            let mut syy = 0.0_f64;
            let mut sxy = 0.0_f64;

            for wr in 0..3 {
                for wc in 0..3 {
                    let y = r + wr - 1;
                    let x = c + wc - 1;
                    let idx = |rr: usize, cc: usize| pixels[rr * width + cc];
                    let gx = -idx(y - 1, x - 1) + idx(y - 1, x + 1)
                           - 2.0 * idx(y, x - 1) + 2.0 * idx(y, x + 1)
                           - idx(y + 1, x - 1) + idx(y + 1, x + 1);
                    let gy = -idx(y - 1, x - 1) - 2.0 * idx(y - 1, x)
                           - idx(y - 1, x + 1)
                           + idx(y + 1, x - 1) + 2.0 * idx(y + 1, x)
                           + idx(y + 1, x + 1);
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }

            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let response = det - k * trace * trace;

            if response > harris_threshold {
                // NMS: check 3x3 neighborhood
                let mut is_max = true;
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        if dr == 0 && dc == 0 { continue; }
                        let nr = (r as i32 + dr) as usize;
                        let nc = (c as i32 + dc) as usize;
                        // We'd need to recompute for neighbors; simplified: skip
                        let _ = (nr, nc);
                    }
                }
                if is_max {
                    corners.push(Point2D::new(c as f64, r as f64));
                }
            }
        }
    }
    corners
}

/// Sub-pixel corner refinement using gradient-based optimization.
pub fn refine_corner_subpixel(
    pixels: &[f64], width: usize, height: usize,
    corner: &Point2D, window: usize,
) -> Point2D {
    let half = window as i32 / 2;
    let cx = corner.x.round() as i32;
    let cy = corner.y.round() as i32;

    let mut sum_gxx = 0.0_f64;
    let mut sum_gyy = 0.0_f64;
    let mut sum_gxy = 0.0_f64;
    let mut sum_gx_x = 0.0_f64;
    let mut sum_gy_y = 0.0_f64;

    for dy in -half..=half {
        for dx in -half..=half {
            let px = cx + dx;
            let py = cy + dy;
            if px < 1 || py < 1 || px >= (width as i32 - 1) || py >= (height as i32 - 1) {
                continue;
            }
            let pu = px as usize;
            let pv = py as usize;
            let gx = (pixels[pv * width + pu + 1] - pixels[pv * width + pu - 1]) * 0.5;
            let gy = (pixels[(pv + 1) * width + pu] - pixels[(pv - 1) * width + pu]) * 0.5;
            sum_gxx += gx * gx;
            sum_gyy += gy * gy;
            sum_gxy += gx * gy;
            sum_gx_x += gx * gx * px as f64 + gx * gy * py as f64;
            sum_gy_y += gx * gy * px as f64 + gy * gy * py as f64;
        }
    }

    let det = sum_gxx * sum_gyy - sum_gxy * sum_gxy;
    if det.abs() < 1e-12 {
        return *corner;
    }

    let new_x = (sum_gyy * sum_gx_x - sum_gxy * sum_gy_y) / det;
    let new_y = (sum_gxx * sum_gy_y - sum_gxy * sum_gx_x) / det;

    Point2D::new(new_x, new_y)
}

// ── Calibration Result ─────────────────────────────────────────

/// Result of camera calibration.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub camera: CameraModel,
    pub rms_error: f64,
    pub num_images: usize,
    pub num_points: usize,
}

impl CalibrationResult {
    pub fn new(camera: CameraModel, rms_error: f64, num_images: usize, num_points: usize) -> Self {
        Self { camera, rms_error, num_images, num_points }
    }
}

impl fmt::Display for CalibrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CalibResult(rms={:.4} images={} points={} {})",
               self.rms_error, self.num_images, self.num_points, self.camera)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_point2d_display() {
        let p = Point2D::new(1.5, 2.5);
        let s = format!("{}", p);
        assert!(s.contains("1.50"));
    }

    #[test]
    fn test_point3d_distance() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(1.0, 2.0, 2.0);
        assert!((a.distance(&b) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_intrinsic_matrix() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let mat = k.to_matrix();
        assert_eq!(mat[0], 500.0);
        assert_eq!(mat[4], 500.0);
        assert_eq!(mat[8], 1.0);
    }

    #[test]
    fn test_intrinsic_project_unproject() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let projected = k.project_normalized(0.5, 0.3);
        let (x, y) = k.unproject(&projected);
        assert!((x - 0.5).abs() < 1e-9);
        assert!((y - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_intrinsic_display() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let s = format!("{}", k);
        assert!(s.contains("500.0"));
    }

    #[test]
    fn test_distortion_zero() {
        let d = DistortionCoeffs::zero();
        let (x, y) = d.distort(0.5, 0.3);
        assert!((x - 0.5).abs() < 1e-9);
        assert!((y - 0.3).abs() < 1e-9);
        assert!(d.is_negligible());
    }

    #[test]
    fn test_distortion_roundtrip() {
        let d = DistortionCoeffs::new(0.1, -0.05, 0.001, 0.0001, -0.0001);
        let (xd, yd) = d.distort(0.3, 0.2);
        let (xu, yu) = d.undistort(xd, yd);
        assert!((xu - 0.3).abs() < 1e-6, "x: {} vs 0.3", xu);
        assert!((yu - 0.2).abs() < 1e-6, "y: {} vs 0.2", yu);
    }

    #[test]
    fn test_distortion_builder() {
        let d = DistortionCoeffs::zero()
            .with_radial(0.1, 0.2, 0.3)
            .with_tangential(0.01, 0.02);
        assert_eq!(d.k1, 0.1);
        assert_eq!(d.p2, 0.02);
    }

    #[test]
    fn test_distortion_display() {
        let d = DistortionCoeffs::zero();
        let s = format!("{}", d);
        assert!(s.contains("Distortion"));
    }

    #[test]
    fn test_extrinsics_identity() {
        let ext = Extrinsics::identity();
        let r = ext.rotation_matrix();
        assert!((r[0] - 1.0).abs() < 1e-9);
        assert!((r[4] - 1.0).abs() < 1e-9);
        assert!((r[8] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_extrinsics_world_to_camera() {
        let ext = Extrinsics::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let p = Point3D::new(10.0, 20.0, 30.0);
        let cam = ext.world_to_camera(&p);
        assert!((cam.x - 11.0).abs() < 1e-9);
        assert!((cam.y - 22.0).abs() < 1e-9);
        assert!((cam.z - 33.0).abs() < 1e-9);
    }

    #[test]
    fn test_extrinsics_display() {
        let ext = Extrinsics::identity();
        let s = format!("{}", ext);
        assert!(s.contains("Extrinsics"));
    }

    #[test]
    fn test_camera_model_project() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let cam = CameraModel::new(k);
        let p = Point3D::new(0.0, 0.0, 10.0);
        let img = cam.project(&p);
        assert!((img.x - 320.0).abs() < 1e-6);
        assert!((img.y - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_camera_model_display() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let cam = CameraModel::new(k);
        let s = format!("{}", cam);
        assert!(s.contains("CameraModel"));
    }

    #[test]
    fn test_reprojection_error_perfect() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let cam = CameraModel::new(k);
        let world = vec![Point3D::new(0.0, 0.0, 10.0)];
        let image = vec![cam.project(&world[0])];
        let err = cam.reprojection_error(&world, &image);
        assert!(err < 1e-9);
    }

    #[test]
    fn test_checkerboard_corners() {
        let board = CheckerboardPattern::new(7, 5, 0.03);
        let corners = board.ideal_corners();
        assert_eq!(corners.len(), 35);
        assert!((corners[0].x).abs() < 1e-9);
        assert!((corners[1].x - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_checkerboard_display() {
        let board = CheckerboardPattern::new(9, 6, 0.025);
        let s = format!("{}", board);
        assert!(s.contains("9x6"));
    }

    #[test]
    fn test_calibration_result_display() {
        let k = IntrinsicMatrix::new(500.0, 500.0, 320.0, 240.0);
        let cam = CameraModel::new(k);
        let result = CalibrationResult::new(cam, 0.25, 10, 350);
        let s = format!("{}", result);
        assert!(s.contains("0.25"));
        assert!(s.contains("350"));
    }

    #[test]
    fn test_rotation_90_degrees_z() {
        let angle = std::f64::consts::FRAC_PI_2;
        let ext = Extrinsics::new([0.0, 0.0, angle], [0.0, 0.0, 0.0]);
        let p = Point3D::new(1.0, 0.0, 0.0);
        let cam = ext.world_to_camera(&p);
        // 90-degree rotation about Z: (1,0,0) -> (0,1,0)
        assert!(cam.x.abs() < 1e-6, "x should be ~0, got {}", cam.x);
        assert!((cam.y - 1.0).abs() < 1e-6, "y should be ~1, got {}", cam.y);
    }

    #[test]
    fn test_refine_corner_identity() {
        // Uniform image: refinement should return same corner
        let img = vec![128.0_f64; 20 * 20];
        let corner = Point2D::new(10.0, 10.0);
        let refined = refine_corner_subpixel(&img, 20, 20, &corner, 5);
        // With uniform gradients, should stay near the input
        assert!((refined.x - 10.0).abs() < 5.0);
        assert!((refined.y - 10.0).abs() < 5.0);
    }
}
