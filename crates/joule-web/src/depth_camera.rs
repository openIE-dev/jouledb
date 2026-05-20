//! Depth camera processing — depth map to point cloud conversion, depth hole
//! filling via nearest-neighbor interpolation, bilateral filtering for edge-
//! preserving smoothing, and RGB-D alignment using known intrinsic parameters.
//!
//! Pure-Rust depth image pipeline for structured-light and time-of-flight
//! cameras, suitable for embedded 3D reconstruction workloads.

use std::fmt;

// ── Camera Intrinsics ───────────────────────────────────────────

/// Pinhole camera intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: usize,
    pub height: usize,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: usize, height: usize) -> Self {
        Self { fx, fy, cx, cy, width, height }
    }

    pub fn with_resolution(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Horizontal field of view in radians.
    pub fn hfov(&self) -> f64 {
        2.0 * (self.width as f64 / (2.0 * self.fx)).atan()
    }

    /// Vertical field of view in radians.
    pub fn vfov(&self) -> f64 {
        2.0 * (self.height as f64 / (2.0 * self.fy)).atan()
    }

    /// Back-project a pixel (u, v, depth) to a 3D point.
    pub fn backproject(&self, u: f64, v: f64, depth: f64) -> Point3D {
        Point3D {
            x: (u - self.cx) * depth / self.fx,
            y: (v - self.cy) * depth / self.fy,
            z: depth,
        }
    }

    /// Project a 3D point to pixel coordinates (u, v).
    pub fn project(&self, p: &Point3D) -> Option<(f64, f64)> {
        if p.z <= 0.0 {
            return None;
        }
        let u = self.fx * p.x / p.z + self.cx;
        let v = self.fy * p.y / p.z + self.cy;
        Some((u, v))
    }
}

impl fmt::Display for CameraIntrinsics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Intrinsics(fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, {}x{})",
            self.fx, self.fy, self.cx, self.cy, self.width, self.height,
        )
    }
}

// ── Point3D ─────────────────────────────────────────────────────

/// 3D point from depth camera.
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

    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl fmt::Display for Point3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// ── Depth Map ───────────────────────────────────────────────────

/// Dense depth map stored as a row-major f64 array.
#[derive(Debug, Clone)]
pub struct DepthMap {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl DepthMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize) -> Option<Self> {
        if data.len() != width * height {
            return None;
        }
        Some(Self { data, width, height })
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.height && col < self.width {
            self.data[row * self.width + col]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, row: usize, col: usize, depth: f64) {
        if row < self.height && col < self.width {
            self.data[row * self.width + col] = depth;
        }
    }

    /// Count valid (non-zero, finite) depth pixels.
    pub fn valid_count(&self) -> usize {
        self.data.iter().filter(|&&d| d > 0.0 && d.is_finite()).count()
    }

    /// Min and max valid depth values.
    pub fn depth_range(&self) -> Option<(f64, f64)> {
        let mut min_d = f64::MAX;
        let mut max_d = 0.0f64;
        let mut found = false;
        for &d in &self.data {
            if d > 0.0 && d.is_finite() {
                if d < min_d { min_d = d; }
                if d > max_d { max_d = d; }
                found = true;
            }
        }
        if found { Some((min_d, max_d)) } else { None }
    }

    /// Convert depth map to a 3D point cloud using camera intrinsics.
    pub fn to_point_cloud(&self, intrinsics: &CameraIntrinsics) -> Vec<Point3D> {
        let mut points = Vec::with_capacity(self.valid_count());
        for row in 0..self.height {
            for col in 0..self.width {
                let d = self.get(row, col);
                if d > 0.0 && d.is_finite() {
                    points.push(intrinsics.backproject(col as f64, row as f64, d));
                }
            }
        }
        points
    }
}

impl fmt::Display for DepthMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DepthMap({}x{}, valid={})",
            self.width, self.height, self.valid_count()
        )
    }
}

// ── Hole Filling ────────────────────────────────────────────────

/// Depth hole filling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleFillMethod {
    NearestNeighbor,
    MinNeighbor,
    MaxNeighbor,
    AverageNeighbor,
}

/// Depth map hole filler using neighborhood interpolation.
#[derive(Debug, Clone)]
pub struct HoleFiller {
    pub method: HoleFillMethod,
    pub kernel_size: usize,
    pub max_passes: usize,
}

impl HoleFiller {
    pub fn new(method: HoleFillMethod) -> Self {
        Self {
            method,
            kernel_size: 3,
            max_passes: 5,
        }
    }

    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = if size % 2 == 0 { size + 1 } else { size };
        self
    }

    pub fn with_max_passes(mut self, passes: usize) -> Self {
        self.max_passes = passes;
        self
    }

    /// Fill holes in the depth map. Returns a new DepthMap.
    pub fn fill(&self, depth: &DepthMap) -> DepthMap {
        let mut result = depth.clone();
        let half = self.kernel_size / 2;

        for _ in 0..self.max_passes {
            let mut filled_any = false;
            let prev = result.clone();
            for row in 0..depth.height {
                for col in 0..depth.width {
                    if prev.get(row, col) > 0.0 {
                        continue;
                    }
                    let mut neighbors = Vec::new();
                    let row_start = row.saturating_sub(half);
                    let row_end = (row + half + 1).min(depth.height);
                    let col_start = col.saturating_sub(half);
                    let col_end = (col + half + 1).min(depth.width);
                    for r in row_start..row_end {
                        for c in col_start..col_end {
                            let d = prev.get(r, c);
                            if d > 0.0 && d.is_finite() {
                                neighbors.push(d);
                            }
                        }
                    }
                    if neighbors.is_empty() {
                        continue;
                    }
                    let val = match self.method {
                        HoleFillMethod::NearestNeighbor => {
                            // Pick the neighbor closest in pixel distance
                            let mut best = neighbors[0];
                            let mut best_dist = f64::MAX;
                            let row_start = row.saturating_sub(half);
                            let col_start = col.saturating_sub(half);
                            for r in row_start..row_end {
                                for c in col_start..col_end {
                                    let d = prev.get(r, c);
                                    if d > 0.0 && d.is_finite() {
                                        let dr = r as f64 - row as f64;
                                        let dc = c as f64 - col as f64;
                                        let dist = dr * dr + dc * dc;
                                        if dist < best_dist {
                                            best_dist = dist;
                                            best = d;
                                        }
                                    }
                                }
                            }
                            best
                        }
                        HoleFillMethod::MinNeighbor => {
                            neighbors.iter().cloned().fold(f64::MAX, f64::min)
                        }
                        HoleFillMethod::MaxNeighbor => {
                            neighbors.iter().cloned().fold(0.0f64, f64::max)
                        }
                        HoleFillMethod::AverageNeighbor => {
                            let sum: f64 = neighbors.iter().sum();
                            sum / neighbors.len() as f64
                        }
                    };
                    result.set(row, col, val);
                    filled_any = true;
                }
            }
            if !filled_any {
                break;
            }
        }
        result
    }
}

impl fmt::Display for HoleFiller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HoleFiller({:?}, kernel={}, passes={})",
            self.method, self.kernel_size, self.max_passes
        )
    }
}

// ── Bilateral Filter ────────────────────────────────────────────

/// Edge-preserving bilateral filter for depth maps.
#[derive(Debug, Clone)]
pub struct BilateralFilter {
    pub kernel_size: usize,
    pub sigma_spatial: f64,
    pub sigma_depth: f64,
}

impl BilateralFilter {
    pub fn new() -> Self {
        Self {
            kernel_size: 5,
            sigma_spatial: 2.0,
            sigma_depth: 0.05,
        }
    }

    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = if size % 2 == 0 { size + 1 } else { size };
        self
    }

    pub fn with_sigma_spatial(mut self, sigma: f64) -> Self {
        self.sigma_spatial = sigma;
        self
    }

    pub fn with_sigma_depth(mut self, sigma: f64) -> Self {
        self.sigma_depth = sigma;
        self
    }

    /// Apply bilateral filter to a depth map.
    pub fn apply(&self, depth: &DepthMap) -> DepthMap {
        let mut result = DepthMap::new(depth.width, depth.height);
        let half = self.kernel_size / 2;
        let spatial_coeff = -0.5 / (self.sigma_spatial * self.sigma_spatial);
        let depth_coeff = -0.5 / (self.sigma_depth * self.sigma_depth);

        for row in 0..depth.height {
            for col in 0..depth.width {
                let center = depth.get(row, col);
                if center <= 0.0 || !center.is_finite() {
                    result.set(row, col, 0.0);
                    continue;
                }

                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;

                let r_start = row.saturating_sub(half);
                let r_end = (row + half + 1).min(depth.height);
                let c_start = col.saturating_sub(half);
                let c_end = (col + half + 1).min(depth.width);

                for r in r_start..r_end {
                    for c in c_start..c_end {
                        let d = depth.get(r, c);
                        if d <= 0.0 || !d.is_finite() {
                            continue;
                        }
                        let dr = r as f64 - row as f64;
                        let dc = c as f64 - col as f64;
                        let spatial_dist2 = dr * dr + dc * dc;
                        let depth_diff = d - center;
                        let depth_dist2 = depth_diff * depth_diff;

                        let weight = (spatial_coeff * spatial_dist2 + depth_coeff * depth_dist2).exp();
                        weighted_sum += weight * d;
                        weight_total += weight;
                    }
                }

                if weight_total > 0.0 {
                    result.set(row, col, weighted_sum / weight_total);
                } else {
                    result.set(row, col, center);
                }
            }
        }
        result
    }
}

impl fmt::Display for BilateralFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bilateral(kernel={}, sigma_s={:.2}, sigma_d={:.3})",
            self.kernel_size, self.sigma_spatial, self.sigma_depth
        )
    }
}

// ── RGB-D Alignment ─────────────────────────────────────────────

/// Extrinsic parameters for aligning RGB and depth cameras.
#[derive(Debug, Clone)]
pub struct RgbdExtrinsics {
    /// Rotation matrix (3x3 row-major).
    pub rotation: [[f64; 3]; 3],
    /// Translation vector [tx, ty, tz] in meters.
    pub translation: [f64; 3],
}

impl RgbdExtrinsics {
    pub fn identity() -> Self {
        Self {
            rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn with_translation(mut self, tx: f64, ty: f64, tz: f64) -> Self {
        self.translation = [tx, ty, tz];
        self
    }

    pub fn with_rotation(mut self, rot: [[f64; 3]; 3]) -> Self {
        self.rotation = rot;
        self
    }

    /// Transform a 3D point from depth camera frame to RGB camera frame.
    pub fn transform(&self, p: &Point3D) -> Point3D {
        let r = &self.rotation;
        let t = &self.translation;
        Point3D {
            x: r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
            y: r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
            z: r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
        }
    }
}

impl fmt::Display for RgbdExtrinsics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Extrinsics(t=[{:.4}, {:.4}, {:.4}])",
            self.translation[0], self.translation[1], self.translation[2]
        )
    }
}

/// RGB-D frame aligner that maps depth pixels to RGB pixel coordinates.
#[derive(Debug, Clone)]
pub struct RgbdAligner {
    pub depth_intrinsics: CameraIntrinsics,
    pub rgb_intrinsics: CameraIntrinsics,
    pub extrinsics: RgbdExtrinsics,
}

impl RgbdAligner {
    pub fn new(
        depth_intrinsics: CameraIntrinsics,
        rgb_intrinsics: CameraIntrinsics,
        extrinsics: RgbdExtrinsics,
    ) -> Self {
        Self { depth_intrinsics, rgb_intrinsics, extrinsics }
    }

    /// Align a depth map to the RGB camera frame. Returns an aligned depth map
    /// at the RGB camera resolution.
    pub fn align_depth_to_rgb(&self, depth: &DepthMap) -> DepthMap {
        let mut aligned = DepthMap::new(self.rgb_intrinsics.width, self.rgb_intrinsics.height);

        for row in 0..depth.height {
            for col in 0..depth.width {
                let d = depth.get(row, col);
                if d <= 0.0 || !d.is_finite() {
                    continue;
                }
                // Back-project from depth image to 3D
                let pt_depth = self.depth_intrinsics.backproject(col as f64, row as f64, d);
                // Transform to RGB camera frame
                let pt_rgb = self.extrinsics.transform(&pt_depth);
                // Project to RGB image
                if let Some((u, v)) = self.rgb_intrinsics.project(&pt_rgb) {
                    let ru = v.round() as usize;
                    let cu = u.round() as usize;
                    if ru < self.rgb_intrinsics.height && cu < self.rgb_intrinsics.width {
                        let existing = aligned.get(ru, cu);
                        // Keep closer depth (z-buffer)
                        if existing <= 0.0 || pt_rgb.z < existing {
                            aligned.set(ru, cu, pt_rgb.z);
                        }
                    }
                }
            }
        }
        aligned
    }

    /// Map a single depth pixel to the corresponding RGB pixel.
    pub fn map_pixel(&self, depth_row: usize, depth_col: usize, depth_val: f64) -> Option<(usize, usize)> {
        if depth_val <= 0.0 {
            return None;
        }
        let pt = self.depth_intrinsics.backproject(depth_col as f64, depth_row as f64, depth_val);
        let pt_rgb = self.extrinsics.transform(&pt);
        let (u, v) = self.rgb_intrinsics.project(&pt_rgb)?;
        let ru = v.round() as usize;
        let cu = u.round() as usize;
        if ru < self.rgb_intrinsics.height && cu < self.rgb_intrinsics.width {
            Some((ru, cu))
        } else {
            None
        }
    }
}

impl fmt::Display for RgbdAligner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RgbdAligner(depth={}x{}, rgb={}x{})",
            self.depth_intrinsics.width,
            self.depth_intrinsics.height,
            self.rgb_intrinsics.width,
            self.rgb_intrinsics.height,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
    }

    #[test]
    fn test_intrinsics_fov() {
        let intr = test_intrinsics();
        assert!(intr.hfov() > 0.0);
        assert!(intr.vfov() > 0.0);
        assert!(intr.hfov() > intr.vfov());
    }

    #[test]
    fn test_backproject_center() {
        let intr = test_intrinsics();
        let p = intr.backproject(320.0, 240.0, 1.0);
        assert!(p.x.abs() < 1e-10);
        assert!(p.y.abs() < 1e-10);
        assert!((p.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_roundtrip() {
        let intr = test_intrinsics();
        let p = Point3D::new(0.5, -0.3, 2.0);
        let (u, v) = intr.project(&p).unwrap();
        let p2 = intr.backproject(u, v, 2.0);
        assert!((p2.x - p.x).abs() < 1e-10);
        assert!((p2.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_project_behind_camera() {
        let intr = test_intrinsics();
        let p = Point3D::new(0.0, 0.0, -1.0);
        assert!(intr.project(&p).is_none());
    }

    #[test]
    fn test_intrinsics_display() {
        let intr = test_intrinsics();
        let s = format!("{intr}");
        assert!(s.contains("Intrinsics"));
    }

    #[test]
    fn test_depth_map_basic() {
        let mut dm = DepthMap::new(4, 3);
        dm.set(1, 2, 1.5);
        assert!((dm.get(1, 2) - 1.5).abs() < 1e-12);
        assert_eq!(dm.valid_count(), 1);
    }

    #[test]
    fn test_depth_map_range() {
        let data = vec![0.0, 1.0, 2.0, 0.0, 3.0, 0.5];
        let dm = DepthMap::from_data(data, 3, 2).unwrap();
        let (min, max) = dm.depth_range().unwrap();
        assert!((min - 0.5).abs() < 1e-12);
        assert!((max - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_depth_map_to_cloud() {
        let intr = test_intrinsics();
        let mut dm = DepthMap::new(640, 480);
        dm.set(240, 320, 2.0); // Center pixel
        let cloud = dm.to_point_cloud(&intr);
        assert_eq!(cloud.len(), 1);
        assert!(cloud[0].x.abs() < 1e-10);
    }

    #[test]
    fn test_depth_map_display() {
        let dm = DepthMap::new(10, 10);
        let s = format!("{dm}");
        assert!(s.contains("DepthMap"));
    }

    #[test]
    fn test_hole_fill_nearest() {
        let data = vec![
            1.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
        ];
        let dm = DepthMap::from_data(data, 3, 3).unwrap();
        let filler = HoleFiller::new(HoleFillMethod::NearestNeighbor);
        let filled = filler.fill(&dm);
        // All holes should be filled
        assert_eq!(filled.valid_count(), 9);
    }

    #[test]
    fn test_hole_fill_average() {
        let data = vec![
            2.0, 0.0, 4.0,
            0.0, 0.0, 0.0,
            2.0, 0.0, 4.0,
        ];
        let dm = DepthMap::from_data(data, 3, 3).unwrap();
        let filler = HoleFiller::new(HoleFillMethod::AverageNeighbor);
        let filled = filler.fill(&dm);
        assert!(filled.valid_count() > dm.valid_count());
    }

    #[test]
    fn test_hole_filler_display() {
        let filler = HoleFiller::new(HoleFillMethod::MinNeighbor);
        let s = format!("{filler}");
        assert!(s.contains("HoleFiller"));
    }

    #[test]
    fn test_bilateral_preserves_edges() {
        let mut dm = DepthMap::new(5, 5);
        // Left half at depth 1.0, right half at depth 5.0
        for row in 0..5 {
            for col in 0..5 {
                if col < 3 { dm.set(row, col, 1.0); }
                else { dm.set(row, col, 5.0); }
            }
        }
        let filter = BilateralFilter::new().with_sigma_depth(0.1);
        let filtered = filter.apply(&dm);
        // Edge should be preserved: left side stays ~1.0, right stays ~5.0
        assert!((filtered.get(2, 0) - 1.0).abs() < 0.5);
        assert!((filtered.get(2, 4) - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_bilateral_display() {
        let filter = BilateralFilter::new();
        let s = format!("{filter}");
        assert!(s.contains("Bilateral"));
    }

    #[test]
    fn test_extrinsics_identity() {
        let ext = RgbdExtrinsics::identity();
        let p = Point3D::new(1.0, 2.0, 3.0);
        let q = ext.transform(&p);
        assert!((q.x - 1.0).abs() < 1e-12);
        assert!((q.y - 2.0).abs() < 1e-12);
        assert!((q.z - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_extrinsics_translation() {
        let ext = RgbdExtrinsics::identity().with_translation(0.05, 0.0, 0.0);
        let p = Point3D::new(0.0, 0.0, 1.0);
        let q = ext.transform(&p);
        assert!((q.x - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_rgbd_aligner_identity() {
        let intr = test_intrinsics();
        let aligner = RgbdAligner::new(intr, intr, RgbdExtrinsics::identity());
        let mut dm = DepthMap::new(640, 480);
        dm.set(240, 320, 1.0);
        let aligned = aligner.align_depth_to_rgb(&dm);
        assert!((aligned.get(240, 320) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rgbd_map_pixel() {
        let intr = test_intrinsics();
        let aligner = RgbdAligner::new(intr, intr, RgbdExtrinsics::identity());
        let result = aligner.map_pixel(240, 320, 1.0);
        assert!(result.is_some());
        let (r, c) = result.unwrap();
        assert_eq!(r, 240);
        assert_eq!(c, 320);
    }

    #[test]
    fn test_rgbd_aligner_display() {
        let intr = test_intrinsics();
        let aligner = RgbdAligner::new(intr, intr, RgbdExtrinsics::identity());
        let s = format!("{aligner}");
        assert!(s.contains("RgbdAligner"));
    }
}
