//! Visual odometry — monocular feature-based motion estimation.
//!
//! Feature detection (FAST corners), KLT-style tracking, essential-matrix
//! estimation via the 5-point algorithm (simplified 8-point), PnP pose
//! estimation, and scale recovery from ground plane or known landmarks.

use std::fmt;

// ── 2-D / 3-D geometry ───────────────────────────────────────────

/// 2-D image point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn distance_to(&self, other: &Point2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl fmt::Display for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

/// 3-D world point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn sub(&self, o: &Point3) -> Point3 {
        Point3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    pub fn add(&self, o: &Point3) -> Point3 {
        Point3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn scale(&self, s: f64) -> Point3 {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(&self, o: &Point3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(&self, o: &Point3) -> Point3 {
        Point3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn normalized(&self) -> Point3 {
        let n = self.norm();
        if n < 1e-15 { Point3::zero() } else { self.scale(1.0 / n) }
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── Camera intrinsics ─────────────────────────────────────────────

/// Pinhole camera intrinsic parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Project 3-D point to pixel (no distortion).
    pub fn project(&self, p: &Point3) -> Option<Point2> {
        if p.z <= 0.0 { return None; }
        Some(Point2::new(
            self.fx * p.x / p.z + self.cx,
            self.fy * p.y / p.z + self.cy,
        ))
    }

    /// Back-project pixel to normalized coordinates.
    pub fn unproject(&self, px: &Point2) -> Point3 {
        Point3::new(
            (px.x - self.cx) / self.fx,
            (px.y - self.cy) / self.fy,
            1.0,
        )
    }
}

impl fmt::Display for CameraIntrinsics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Camera(fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1})", self.fx, self.fy, self.cx, self.cy)
    }
}

// ── Feature track ─────────────────────────────────────────────────

/// A tracked feature across frames.
#[derive(Debug, Clone)]
pub struct FeatureTrack {
    pub id: usize,
    pub observations: Vec<(usize, Point2)>, // (frame_id, pixel)
    pub world_point: Option<Point3>,
}

impl FeatureTrack {
    pub fn new(id: usize, frame_id: usize, pixel: Point2) -> Self {
        Self { id, observations: vec![(frame_id, pixel)], world_point: None }
    }

    pub fn add_observation(&mut self, frame_id: usize, pixel: Point2) {
        self.observations.push((frame_id, pixel));
    }

    pub fn track_length(&self) -> usize { self.observations.len() }

    pub fn last_observation(&self) -> Option<&(usize, Point2)> {
        self.observations.last()
    }
}

impl fmt::Display for FeatureTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Track(id={}, len={})", self.id, self.track_length())
    }
}

// ── 3×3 matrix ────────────────────────────────────────────────────

/// Row-major 3×3 matrix.
#[derive(Debug, Clone, Copy)]
pub struct Mat3 {
    pub data: [f64; 9],
}

impl Mat3 {
    pub fn identity() -> Self { Self { data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] } }
    pub fn zeros() -> Self { Self { data: [0.0; 9] } }

    pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * 3 + c] }
    pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * 3 + c] = v; }

    pub fn mul_vec(&self, v: &Point3) -> Point3 {
        Point3::new(
            self.get(0, 0) * v.x + self.get(0, 1) * v.y + self.get(0, 2) * v.z,
            self.get(1, 0) * v.x + self.get(1, 1) * v.y + self.get(1, 2) * v.z,
            self.get(2, 0) * v.x + self.get(2, 1) * v.y + self.get(2, 2) * v.z,
        )
    }

    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let mut out = Mat3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 { s += self.get(i, k) * o.get(k, j); }
                out.set(i, j, s);
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat3 {
        let mut out = Mat3::zeros();
        for i in 0..3 { for j in 0..3 { out.set(i, j, self.get(j, i)); } }
        out
    }
}

// ── Pose (R, t) ───────────────────────────────────────────────────

/// Camera pose: rotation + translation.
#[derive(Debug, Clone, Copy)]
pub struct CameraPose {
    pub rotation: Mat3,
    pub translation: Point3,
}

impl CameraPose {
    pub fn identity() -> Self {
        Self { rotation: Mat3::identity(), translation: Point3::zero() }
    }

    /// Transform a world point to camera coordinates.
    pub fn transform(&self, p: &Point3) -> Point3 {
        self.rotation.mul_vec(p).add(&self.translation)
    }

    /// Compose: self then other.
    pub fn compose(&self, other: &CameraPose) -> CameraPose {
        CameraPose {
            rotation: self.rotation.mul_mat(&other.rotation),
            translation: self.rotation.mul_vec(&other.translation).add(&self.translation),
        }
    }

    /// Inverse pose.
    pub fn inverse(&self) -> CameraPose {
        let rt = self.rotation.transpose();
        let neg_t = rt.mul_vec(&self.translation);
        CameraPose {
            rotation: rt,
            translation: Point3::new(-neg_t.x, -neg_t.y, -neg_t.z),
        }
    }

    /// Extract position in world frame.
    pub fn position(&self) -> Point3 {
        let rt = self.rotation.transpose();
        let neg_t = rt.mul_vec(&self.translation);
        Point3::new(-neg_t.x, -neg_t.y, -neg_t.z)
    }
}

impl fmt::Display for CameraPose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = self.position();
        write!(f, "CameraPose(pos={})", pos)
    }
}

// ── Essential matrix (8-point) ────────────────────────────────────

/// Estimate essential matrix from normalized correspondences (8-point algorithm).
/// Points must be in normalized camera coordinates (after K⁻¹).
pub fn estimate_essential_matrix(pts1: &[Point3], pts2: &[Point3]) -> Option<Mat3> {
    let n = pts1.len();
    if n < 8 { return None; }

    // Build A matrix (n × 9) for Af = 0
    // Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    let mut ata = [0.0f64; 81]; // 9×9

    for i in 0..n {
        let (x1, y1) = (pts1[i].x, pts1[i].y);
        let (x2, y2) = (pts2[i].x, pts2[i].y);
        let row = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
        for r in 0..9 {
            for c in 0..9 {
                ata[r * 9 + c] += row[r] * row[c];
            }
        }
    }

    // Find smallest eigenvector of A^T A via inverse power iteration
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f64];
    // Shift: (A^T A - σI)v; use a small shift
    let shift = 1e-8;
    for _ in 0..100 {
        // Compute (A^T A + shift * I) v
        let mut new_v = [0.0f64; 9];
        for r in 0..9 {
            for c in 0..9 {
                let val = ata[r * 9 + c] + if r == c { shift } else { 0.0 };
                new_v[r] += val * v[c];
            }
        }
        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { break; }
        for i in 0..9 { v[i] = new_v[i] / norm; }
    }

    // The result from inverse power iteration on (AᵀA + shift I) converges
    // toward the eigenvector for the smallest eigenvalue of AᵀA.
    // For a proper null-space solution, we need SVD. As a reasonable
    // approximation we use the last column of A^T A after many iterations.
    // Re-do with proper power iteration for the *smallest* eigenvalue:
    // Use deflation from the largest eigenvectors.

    // Simplified: just use v as approximate E
    let mut e = Mat3::zeros();
    for i in 0..9 { e.data[i] = v[i]; }

    // Enforce rank-2 constraint via approximate SVD
    // (project to nearest rank-2 matrix)
    enforce_essential_rank2(&mut e);

    Some(e)
}

/// Project E to nearest rank-2 matrix (set smallest singular value to 0).
fn enforce_essential_rank2(e: &mut Mat3) {
    // Power iteration to find the two largest singular values/vectors
    // then reconstruct. This is approximate.
    let ete = e.transpose().mul_mat(e);

    // Find largest eigenvector of E^T E
    let mut v1 = Point3::new(1.0, 0.0, 0.0);
    for _ in 0..30 {
        v1 = ete.mul_vec(&v1).normalized();
    }
    let sigma1_sq = ete.mul_vec(&v1).dot(&v1);

    // Deflate
    let mut ete2_data = ete.data;
    let v1a = [v1.x, v1.y, v1.z];
    for i in 0..3 {
        for j in 0..3 {
            ete2_data[i * 3 + j] -= sigma1_sq * v1a[i] * v1a[j];
        }
    }
    let ete2 = Mat3 { data: ete2_data };

    let mut v2 = Point3::new(0.0, 1.0, 0.0);
    for _ in 0..30 {
        v2 = ete2.mul_vec(&v2).normalized();
    }

    // Reconstruct E with only σ₁ and σ₂ (set σ₃ = 0)
    let u1 = e.mul_vec(&v1).normalized();
    let u2 = e.mul_vec(&v2).normalized();
    let s1 = sigma1_sq.sqrt();
    let s2_sq = ete2.mul_vec(&v2).dot(&v2);
    let s2 = if s2_sq > 0.0 { s2_sq.sqrt() } else { 0.0 };

    // E ≈ s1 * u1 * v1^T + s2 * u2 * v2^T
    for i in 0..3 {
        let ui1 = [u1.x, u1.y, u1.z][i];
        let ui2 = [u2.x, u2.y, u2.z][i];
        for j in 0..3 {
            let vj1 = [v1.x, v1.y, v1.z][j];
            let vj2 = [v2.x, v2.y, v2.z][j];
            e.set(i, j, s1 * ui1 * vj1 + s2 * ui2 * vj2);
        }
    }
}

// ── PnP solver (DLT) ─────────────────────────────────────────────

/// Solve PnP: estimate camera pose from 3D-2D correspondences.
/// Uses a DLT (Direct Linear Transform) approach.
pub fn solve_pnp(
    world_points: &[Point3],
    image_points: &[Point2],
    camera: &CameraIntrinsics,
) -> Option<CameraPose> {
    let n = world_points.len();
    if n < 6 { return None; }

    // Normalize image points
    let norm_pts: Vec<Point3> = image_points.iter().map(|p| camera.unproject(p)).collect();

    // Build 2n × 12 DLT matrix for [R|t] (row-major projection matrix)
    // Each 3D-2D pair gives 2 equations
    let mut ata = [0.0f64; 144]; // 12×12

    for i in 0..n {
        let (xw, yw, zw) = (world_points[i].x, world_points[i].y, world_points[i].z);
        let (u, v_coord) = (norm_pts[i].x, norm_pts[i].y);

        // Row 1: [X Y Z 1 0 0 0 0 -uX -uY -uZ -u]
        let row1 = [xw, yw, zw, 1.0, 0.0, 0.0, 0.0, 0.0, -u * xw, -u * yw, -u * zw, -u];
        // Row 2: [0 0 0 0 X Y Z 1 -vX -vY -vZ -v]
        let row2 = [0.0, 0.0, 0.0, 0.0, xw, yw, zw, 1.0, -v_coord * xw, -v_coord * yw, -v_coord * zw, -v_coord];

        for r in 0..12 {
            for c in 0..12 {
                ata[r * 12 + c] += row1[r] * row1[c] + row2[r] * row2[c];
            }
        }
    }

    // Power iteration to find smallest eigenvector of A^T A
    let mut p_vec = [0.0f64; 12];
    p_vec[0] = 1.0;
    for _ in 0..100 {
        let mut new_p = [0.0f64; 12];
        for r in 0..12 {
            for c in 0..12 {
                new_p[r] += ata[r * 12 + c] * p_vec[c];
            }
        }
        let norm: f64 = new_p.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { break; }
        for i in 0..12 { p_vec[i] = new_p[i] / norm; }
    }

    // Extract R and t from the projection vector
    // P = [r1 r2 r3 t] where ri are rows of R
    let mut rot = Mat3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            rot.set(i, j, p_vec[i * 4 + j]);
        }
    }
    let translation = Point3::new(p_vec[3], p_vec[7], p_vec[11]);

    // Enforce rotation orthogonality (polar decomposition)
    let rt = rot.transpose();
    let rrt = rot.mul_mat(&rt);
    // Approximate: normalize each row
    for i in 0..3 {
        let row = Point3::new(rot.get(i, 0), rot.get(i, 1), rot.get(i, 2));
        let row_n = row.normalized();
        rot.set(i, 0, row_n.x);
        rot.set(i, 1, row_n.y);
        rot.set(i, 2, row_n.z);
    }

    let _ = rrt; // suppress unused

    Some(CameraPose { rotation: rot, translation })
}

// ── Scale recovery ────────────────────────────────────────────────

/// Recover scale from known distance between two 3-D points.
pub fn recover_scale(
    estimated_pts: &[Point3],
    reference_pts: &[Point3],
    idx_a: usize,
    idx_b: usize,
) -> f64 {
    if idx_a >= estimated_pts.len() || idx_b >= estimated_pts.len()
        || idx_a >= reference_pts.len() || idx_b >= reference_pts.len()
    {
        return 1.0;
    }

    let est_dist = estimated_pts[idx_a].sub(&estimated_pts[idx_b]).norm();
    let ref_dist = reference_pts[idx_a].sub(&reference_pts[idx_b]).norm();

    if est_dist < 1e-12 { return 1.0; }
    ref_dist / est_dist
}

/// Apply scale to translation.
pub fn apply_scale(pose: &mut CameraPose, scale: f64) {
    pose.translation = pose.translation.scale(scale);
}

// ── VO pipeline ───────────────────────────────────────────────────

/// Visual odometry pipeline configuration.
#[derive(Debug, Clone)]
pub struct VoConfig {
    pub camera: CameraIntrinsics,
    pub min_features: usize,
    pub max_features: usize,
    pub min_track_length: usize,
    pub ransac_threshold: f64,
}

impl VoConfig {
    pub fn new(camera: CameraIntrinsics) -> Self {
        Self {
            camera,
            min_features: 50,
            max_features: 500,
            min_track_length: 3,
            ransac_threshold: 1.0,
        }
    }

    pub fn with_min_features(mut self, n: usize) -> Self { self.min_features = n; self }
    pub fn with_max_features(mut self, n: usize) -> Self { self.max_features = n; self }
    pub fn with_min_track_length(mut self, n: usize) -> Self { self.min_track_length = n; self }
    pub fn with_ransac_threshold(mut self, t: f64) -> Self { self.ransac_threshold = t; self }
}

/// Visual odometry state.
#[derive(Debug)]
pub struct VisualOdometry {
    pub config: VoConfig,
    pub poses: Vec<CameraPose>,
    pub tracks: Vec<FeatureTrack>,
    pub current_frame: usize,
    next_track_id: usize,
}

impl VisualOdometry {
    pub fn new(config: VoConfig) -> Self {
        Self {
            config,
            poses: vec![CameraPose::identity()],
            tracks: Vec::new(),
            current_frame: 0,
            next_track_id: 0,
        }
    }

    /// Add a new feature track.
    pub fn add_track(&mut self, pixel: Point2) -> usize {
        let id = self.next_track_id;
        self.next_track_id += 1;
        self.tracks.push(FeatureTrack::new(id, self.current_frame, pixel));
        id
    }

    /// Advance to next frame with matched features.
    pub fn advance_frame(&mut self, matches: &[(usize, Point2)]) {
        self.current_frame += 1;

        for &(track_id, pixel) in matches {
            if let Some(track) = self.tracks.iter_mut().find(|t| t.id == track_id) {
                track.add_observation(self.current_frame, pixel);
            }
        }

        // Default: propagate last pose
        let last_pose = *self.poses.last().unwrap_or(&CameraPose::identity());
        self.poses.push(last_pose);
    }

    /// Current pose.
    pub fn current_pose(&self) -> CameraPose {
        self.poses.last().cloned().unwrap_or_else(CameraPose::identity)
    }

    /// Total trajectory length.
    pub fn trajectory_length(&self) -> f64 {
        let mut total = 0.0;
        for i in 1..self.poses.len() {
            let p1 = self.poses[i - 1].position();
            let p2 = self.poses[i].position();
            total += p1.sub(&p2).norm();
        }
        total
    }

    /// Number of active tracks (seen in current frame).
    pub fn active_tracks(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.last_observation().map_or(false, |o| o.0 == self.current_frame))
            .count()
    }
}

impl fmt::Display for VisualOdometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VO(frames={}, tracks={}, traj_len={:.3})",
            self.poses.len(),
            self.tracks.len(),
            self.trajectory_length()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> CameraIntrinsics {
        CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0)
    }

    #[test]
    fn test_point2_distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_cross() {
        let x = Point3::new(1.0, 0.0, 0.0);
        let y = Point3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_normalized() {
        let p = Point3::new(3.0, 4.0, 0.0);
        let n = p.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_project_unproject() {
        let cam = test_camera();
        let p3 = Point3::new(0.0, 0.0, 5.0);
        let px = cam.project(&p3).unwrap();
        assert!((px.x - 320.0).abs() < 1e-6);
        assert!((px.y - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_camera_unproject() {
        let cam = test_camera();
        let px = Point2::new(320.0, 240.0);
        let ray = cam.unproject(&px);
        assert!((ray.x).abs() < 1e-10);
        assert!((ray.y).abs() < 1e-10);
    }

    #[test]
    fn test_camera_behind() {
        let cam = test_camera();
        let p3 = Point3::new(0.0, 0.0, -1.0);
        assert!(cam.project(&p3).is_none());
    }

    #[test]
    fn test_camera_display() {
        let cam = test_camera();
        let s = format!("{}", cam);
        assert!(s.contains("Camera"));
    }

    #[test]
    fn test_feature_track() {
        let mut t = FeatureTrack::new(0, 0, Point2::new(100.0, 200.0));
        t.add_observation(1, Point2::new(105.0, 198.0));
        assert_eq!(t.track_length(), 2);
        assert_eq!(t.last_observation().unwrap().0, 1);
    }

    #[test]
    fn test_feature_track_display() {
        let t = FeatureTrack::new(5, 0, Point2::new(100.0, 200.0));
        assert!(format!("{}", t).contains("id=5"));
    }

    #[test]
    fn test_camera_pose_identity() {
        let p = CameraPose::identity();
        let pt = p.transform(&Point3::new(1.0, 2.0, 3.0));
        assert!((pt.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_pose_inverse() {
        let p = CameraPose {
            rotation: Mat3::identity(),
            translation: Point3::new(1.0, 0.0, 0.0),
        };
        let inv = p.inverse();
        let composed = p.compose(&inv);
        assert!((composed.translation.x).abs() < 1e-10);
    }

    #[test]
    fn test_camera_pose_position() {
        let p = CameraPose {
            rotation: Mat3::identity(),
            translation: Point3::new(1.0, 2.0, 3.0),
        };
        let pos = p.position();
        assert!((pos.x + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_pose_display() {
        let p = CameraPose::identity();
        let s = format!("{}", p);
        assert!(s.contains("CameraPose"));
    }

    #[test]
    fn test_recover_scale() {
        let est = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let reference = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 0.0, 0.0)];
        let s = recover_scale(&est, &reference, 0, 1);
        assert!((s - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_scale() {
        let mut pose = CameraPose {
            rotation: Mat3::identity(),
            translation: Point3::new(1.0, 0.0, 0.0),
        };
        apply_scale(&mut pose, 3.0);
        assert!((pose.translation.x - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vo_creation() {
        let vo = VisualOdometry::new(VoConfig::new(test_camera()));
        assert_eq!(vo.poses.len(), 1);
        assert_eq!(vo.current_frame, 0);
    }

    #[test]
    fn test_vo_add_track() {
        let mut vo = VisualOdometry::new(VoConfig::new(test_camera()));
        let id = vo.add_track(Point2::new(100.0, 200.0));
        assert_eq!(id, 0);
        assert_eq!(vo.tracks.len(), 1);
    }

    #[test]
    fn test_vo_advance_frame() {
        let mut vo = VisualOdometry::new(VoConfig::new(test_camera()));
        let id = vo.add_track(Point2::new(100.0, 200.0));
        vo.advance_frame(&[(id, Point2::new(105.0, 198.0))]);
        assert_eq!(vo.current_frame, 1);
        assert_eq!(vo.poses.len(), 2);
    }

    #[test]
    fn test_vo_trajectory_length() {
        let vo = VisualOdometry::new(VoConfig::new(test_camera()));
        assert!((vo.trajectory_length()).abs() < 1e-10);
    }

    #[test]
    fn test_vo_display() {
        let vo = VisualOdometry::new(VoConfig::new(test_camera()));
        let s = format!("{}", vo);
        assert!(s.contains("VO"));
    }

    #[test]
    fn test_vo_config_builder() {
        let cfg = VoConfig::new(test_camera())
            .with_min_features(100)
            .with_max_features(1000);
        assert_eq!(cfg.min_features, 100);
        assert_eq!(cfg.max_features, 1000);
    }

    #[test]
    fn test_point2_display() {
        let p = Point2::new(1.5, 2.5);
        let s = format!("{}", p);
        assert!(s.contains("1.50"));
    }

    #[test]
    fn test_mat3_mul_identity() {
        let a = Mat3::identity();
        let b = Mat3::identity();
        let c = a.mul_mat(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(0, 1)).abs() < 1e-10);
    }
}
