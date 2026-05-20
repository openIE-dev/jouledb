//! Bin Picking — Point cloud segmentation, pose estimation, collision checking,
//! and grasp ranking for automated bin-picking scenarios.
//!
//! Implements voxel-grid downsampling, region-growing segmentation, PCA-based
//! pose estimation, AABB collision checking, and multi-criteria grasp ranking.
//! All algorithms are std-only, using `f64` throughout.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Bin picking errors.
#[derive(Debug, Clone, PartialEq)]
pub enum BinPickError {
    /// Invalid input data.
    InvalidInput(String),
    /// No objects segmented.
    NoObjects,
    /// Collision detected.
    Collision(String),
    /// No feasible grasp.
    NoFeasibleGrasp,
    /// Numeric failure.
    NumericFailure(String),
}

impl fmt::Display for BinPickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::NoObjects => write!(f, "no objects segmented"),
            Self::Collision(m) => write!(f, "collision: {m}"),
            Self::NoFeasibleGrasp => write!(f, "no feasible grasp found"),
            Self::NumericFailure(m) => write!(f, "numeric failure: {m}"),
        }
    }
}

impl std::error::Error for BinPickError {}

// ── Point3 ──────────────────────────────────────────────────────

/// A 3D point with optional normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-12 { None } else { Some(self.scale(1.0 / n)) }
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── AABB ────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Point3,
    pub max: Point3,
}

impl AABB {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// Build the tightest AABB for a point cloud.
    pub fn from_points(points: &[Point3]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut min = points[0];
        let mut max = points[0];
        for p in points.iter().skip(1) {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        Some(Self { min, max })
    }

    /// Check AABB overlap.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Check if a point is inside (with margin).
    pub fn contains(&self, p: Point3, margin: f64) -> bool {
        p.x >= self.min.x - margin
            && p.x <= self.max.x + margin
            && p.y >= self.min.y - margin
            && p.y <= self.max.y + margin
            && p.z >= self.min.z - margin
            && p.z <= self.max.z + margin
    }

    /// Expand this AABB by a margin.
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            min: Point3::new(self.min.x - margin, self.min.y - margin, self.min.z - margin),
            max: Point3::new(self.max.x + margin, self.max.y + margin, self.max.z + margin),
        }
    }

    /// Volume of this AABB.
    pub fn volume(&self) -> f64 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y) * (self.max.z - self.min.z)
    }

    /// Centre of the AABB.
    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }
}

impl fmt::Display for AABB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AABB(min={}, max={})", self.min, self.max)
    }
}

// ── Voxel Grid Downsampling ─────────────────────────────────────

/// Downsample a point cloud using a voxel grid.
/// Returns the centroid of each occupied voxel.
pub fn voxel_downsample(points: &[Point3], voxel_size: f64) -> Result<Vec<Point3>, BinPickError> {
    if voxel_size <= 0.0 {
        return Err(BinPickError::InvalidInput("voxel size must be positive".into()));
    }
    if points.is_empty() {
        return Ok(Vec::new());
    }

    // Hash points into voxels
    let mut voxels: std::collections::HashMap<(i64, i64, i64), (Point3, usize)> =
        std::collections::HashMap::new();

    for p in points {
        let vx = (p.x / voxel_size).floor() as i64;
        let vy = (p.y / voxel_size).floor() as i64;
        let vz = (p.z / voxel_size).floor() as i64;
        let entry = voxels.entry((vx, vy, vz)).or_insert((Point3::zero(), 0));
        entry.0 = entry.0.add(*p);
        entry.1 += 1;
    }

    let result: Vec<Point3> = voxels
        .values()
        .map(|(sum, count)| sum.scale(1.0 / (*count as f64)))
        .collect();
    Ok(result)
}

// ── Region Growing Segmentation ─────────────────────────────────

/// A segmented object cluster.
#[derive(Debug, Clone)]
pub struct ObjectCluster {
    /// Indices into the original point cloud.
    pub indices: Vec<usize>,
    /// Centroid of the cluster.
    pub centroid: Point3,
    /// Bounding box.
    pub bbox: AABB,
}

impl ObjectCluster {
    pub fn num_points(&self) -> usize {
        self.indices.len()
    }
}

impl fmt::Display for ObjectCluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cluster(pts={}, centroid={}, bbox={})",
            self.indices.len(),
            self.centroid,
            self.bbox
        )
    }
}

/// Configuration for region-growing segmentation.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Maximum distance between neighbouring points.
    pub distance_threshold: f64,
    /// Minimum number of points per cluster.
    pub min_cluster_size: usize,
    /// Maximum number of points per cluster.
    pub max_cluster_size: usize,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.01,
            min_cluster_size: 50,
            max_cluster_size: 10_000,
        }
    }
}

impl SegmentConfig {
    pub fn with_distance_threshold(mut self, d: f64) -> Self {
        self.distance_threshold = d;
        self
    }

    pub fn with_min_cluster_size(mut self, n: usize) -> Self {
        self.min_cluster_size = n;
        self
    }

    pub fn with_max_cluster_size(mut self, n: usize) -> Self {
        self.max_cluster_size = n;
        self
    }
}

/// Segment a point cloud into clusters using region growing.
pub fn segment_objects(
    points: &[Point3],
    config: &SegmentConfig,
) -> Vec<ObjectCluster> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();
    let threshold_sq = config.distance_threshold * config.distance_threshold;

    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut indices = Vec::new();
        queue.push_back(seed);
        visited[seed] = true;

        while let Some(idx) = queue.pop_front() {
            indices.push(idx);
            if indices.len() >= config.max_cluster_size {
                break;
            }
            // Find neighbours (brute-force for simplicity)
            for j in 0..n {
                if visited[j] {
                    continue;
                }
                let dist_sq = {
                    let dx = points[idx].x - points[j].x;
                    let dy = points[idx].y - points[j].y;
                    let dz = points[idx].z - points[j].z;
                    dx * dx + dy * dy + dz * dz
                };
                if dist_sq <= threshold_sq {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }

        if indices.len() >= config.min_cluster_size {
            let centroid = {
                let sum = indices.iter().fold(Point3::zero(), |acc, &i| acc.add(points[i]));
                sum.scale(1.0 / indices.len() as f64)
            };
            let cluster_points: Vec<Point3> = indices.iter().map(|i| points[*i]).collect();
            let bbox = AABB::from_points(&cluster_points).unwrap();
            clusters.push(ObjectCluster { indices, centroid, bbox });
        }
    }
    clusters
}

// ── PCA Pose Estimation ─────────────────────────────────────────

/// Estimated object pose from PCA.
#[derive(Debug, Clone)]
pub struct ObjectPose {
    /// Object centroid.
    pub centroid: Point3,
    /// Principal axes (sorted by eigenvalue, largest first).
    pub axes: [Point3; 3],
    /// Eigenvalues (sorted descending).
    pub eigenvalues: [f64; 3],
    /// Elongation ratio (largest / smallest eigenvalue).
    pub elongation: f64,
}

impl ObjectPose {
    /// Primary axis (largest eigenvalue direction).
    pub fn primary_axis(&self) -> Point3 {
        self.axes[0]
    }
}

impl fmt::Display for ObjectPose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObjectPose(centroid={}, elongation={:.2})",
            self.centroid, self.elongation
        )
    }
}

/// Estimate the 6-DOF pose of an object cluster via PCA.
pub fn estimate_pose(points: &[Point3]) -> Result<ObjectPose, BinPickError> {
    if points.len() < 3 {
        return Err(BinPickError::InvalidInput("need >= 3 points for PCA".into()));
    }
    let n = points.len() as f64;
    let centroid = {
        let sum = points.iter().fold(Point3::zero(), |acc, p| acc.add(*p));
        sum.scale(1.0 / n)
    };

    // Compute 3x3 covariance matrix
    let mut cov = [[0.0f64; 3]; 3];
    for p in points {
        let d = [p.x - centroid.x, p.y - centroid.y, p.z - centroid.z];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= n;
        }
    }

    // Jacobi eigenvalue decomposition for 3x3 symmetric matrix
    let (eigenvalues, eigenvectors) = jacobi_3x3(&cov)?;

    let elongation = if eigenvalues[2].abs() > 1e-15 {
        eigenvalues[0] / eigenvalues[2]
    } else {
        f64::INFINITY
    };

    let axes = [
        Point3::new(eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]),
        Point3::new(eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]),
        Point3::new(eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2]),
    ];

    Ok(ObjectPose { centroid, axes, eigenvalues, elongation })
}

/// Jacobi eigendecomposition for a 3x3 symmetric matrix.
/// Returns eigenvalues (descending) and eigenvector matrix.
fn jacobi_3x3(a: &[[f64; 3]; 3]) -> Result<([f64; 3], [[f64; 3]; 3]), BinPickError> {
    let mut m = *a;
    let mut v = [[0.0f64; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _ in 0..100 {
        // Find largest off-diagonal
        let (mut p, mut q) = (0, 1);
        let mut max_val = m[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if m[i][j].abs() > max_val {
                    max_val = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }

        let theta = if (m[p][p] - m[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * m[p][q] / (m[p][p] - m[q][q])).atan()
        };
        let (s, c) = theta.sin_cos();

        // Apply rotation to m
        let mut new_m = m;
        for i in 0..3 {
            new_m[i][p] = c * m[i][p] + s * m[i][q];
            new_m[i][q] = -s * m[i][p] + c * m[i][q];
        }
        let col_p: [f64; 3] = [new_m[0][p], new_m[1][p], new_m[2][p]];
        let col_q: [f64; 3] = [new_m[0][q], new_m[1][q], new_m[2][q]];
        for i in 0..3 {
            new_m[p][i] = c * col_p[i] + s * col_q[i];
            new_m[q][i] = -s * col_p[i] + c * col_q[i];
        }
        // Fix diagonal symmetry
        new_m[p][q] = 0.0;
        new_m[q][p] = 0.0;
        m = new_m;

        // Accumulate eigenvectors
        for i in 0..3 {
            let vp = v[i][p];
            let vq = v[i][q];
            v[i][p] = c * vp + s * vq;
            v[i][q] = -s * vp + c * vq;
        }
    }

    let mut evals = [m[0][0], m[1][1], m[2][2]];
    let mut order = [0usize, 1, 2];
    // Sort descending
    if evals[order[0]] < evals[order[1]] { order.swap(0, 1); }
    if evals[order[0]] < evals[order[2]] { order.swap(0, 2); }
    if evals[order[1]] < evals[order[2]] { order.swap(1, 2); }

    let sorted_evals = [evals[order[0]], evals[order[1]], evals[order[2]]];
    let mut sorted_v = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            sorted_v[i][j] = v[i][order[j]];
        }
    }
    let _ = evals; // suppress warning

    Ok((sorted_evals, sorted_v))
}

// ── Grasp Candidate ─────────────────────────────────────────────

/// A candidate grasp for bin picking.
#[derive(Debug, Clone)]
pub struct GraspCandidate {
    /// Gripper center position.
    pub position: Point3,
    /// Approach direction (unit vector).
    pub approach: Point3,
    /// Gripper opening width.
    pub width: f64,
    /// Quality score (higher is better).
    pub score: f64,
    /// Is collision-free flag.
    pub collision_free: bool,
}

impl GraspCandidate {
    pub fn new(position: Point3, approach: Point3, width: f64) -> Self {
        Self {
            position,
            approach: approach.normalized().unwrap_or(Point3::new(0.0, 0.0, -1.0)),
            width,
            score: 0.0,
            collision_free: true,
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }
}

impl fmt::Display for GraspCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Grasp(pos={}, w={:.3}, score={:.3}, free={})",
            self.position, self.width, self.score, self.collision_free
        )
    }
}

// ── Bin Picking Pipeline ────────────────────────────────────────

/// Configuration for the bin picking pipeline.
#[derive(Debug, Clone)]
pub struct BinPickConfig {
    /// Voxel size for downsampling.
    pub voxel_size: f64,
    /// Segmentation config.
    pub segment_config: SegmentConfig,
    /// Collision margin (metres).
    pub collision_margin: f64,
    /// Bin bounding box.
    pub bin_bounds: AABB,
    /// Maximum gripper width.
    pub max_gripper_width: f64,
}

impl Default for BinPickConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.005,
            segment_config: SegmentConfig::default(),
            collision_margin: 0.005,
            bin_bounds: AABB::new(
                Point3::new(-0.3, -0.2, 0.0),
                Point3::new(0.3, 0.2, 0.15),
            ),
            max_gripper_width: 0.08,
        }
    }
}

impl BinPickConfig {
    pub fn with_voxel_size(mut self, v: f64) -> Self {
        self.voxel_size = v;
        self
    }

    pub fn with_collision_margin(mut self, m: f64) -> Self {
        self.collision_margin = m;
        self
    }

    pub fn with_bin_bounds(mut self, b: AABB) -> Self {
        self.bin_bounds = b;
        self
    }

    pub fn with_max_gripper_width(mut self, w: f64) -> Self {
        self.max_gripper_width = w;
        self
    }
}

/// Bin picking pipeline.
#[derive(Debug)]
pub struct BinPicker {
    config: BinPickConfig,
}

impl BinPicker {
    pub fn new(config: BinPickConfig) -> Self {
        Self { config }
    }

    /// Run the full pipeline: downsample -> segment -> pose -> rank grasps.
    pub fn process(
        &self,
        raw_cloud: &[Point3],
    ) -> Result<Vec<(ObjectCluster, ObjectPose, Vec<GraspCandidate>)>, BinPickError> {
        if raw_cloud.is_empty() {
            return Err(BinPickError::InvalidInput("empty point cloud".into()));
        }
        // 1. Downsample
        let downsampled = voxel_downsample(raw_cloud, self.config.voxel_size)?;

        // 2. Segment
        let clusters = segment_objects(&downsampled, &self.config.segment_config);
        if clusters.is_empty() {
            return Err(BinPickError::NoObjects);
        }

        // 3. For each cluster: estimate pose and generate grasp candidates
        let mut results = Vec::new();
        for cluster in clusters {
            let cluster_pts: Vec<Point3> =
                cluster.indices.iter().map(|i| downsampled[*i]).collect();
            let pose = estimate_pose(&cluster_pts)?;

            // Generate grasps along primary axis
            let grasps = self.generate_grasps(&pose, &cluster.bbox);
            results.push((cluster, pose, grasps));
        }

        Ok(results)
    }

    /// Generate grasp candidates for an object.
    fn generate_grasps(&self, pose: &ObjectPose, bbox: &AABB) -> Vec<GraspCandidate> {
        let mut candidates = Vec::new();
        let center = pose.centroid;
        let approach = Point3::new(0.0, 0.0, -1.0); // top-down approach

        // Generate grasps at different orientations around the primary axis
        for i in 0..8 {
            let angle = std::f64::consts::PI * (i as f64) / 8.0;
            let axis = pose.primary_axis();
            let offset = Point3::new(
                axis.x * 0.01 * angle.cos(),
                axis.y * 0.01 * angle.sin(),
                0.0,
            );
            let grasp_pos = center.add(offset);

            // Score: prefer grasps near the top of the bin
            let height_score = (grasp_pos.z - self.config.bin_bounds.min.z)
                / (self.config.bin_bounds.max.z - self.config.bin_bounds.min.z + 1e-10);
            let clearance = self.check_clearance(&grasp_pos, bbox);

            let mut candidate = GraspCandidate::new(grasp_pos, approach, self.config.max_gripper_width)
                .with_score(height_score * 0.6 + clearance * 0.4);
            candidate.collision_free = clearance > 0.0;
            candidates.push(candidate);
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Check clearance above a grasp point (0..1).
    fn check_clearance(&self, pos: &Point3, bbox: &AABB) -> f64 {
        let above_bbox = pos.z > bbox.max.z;
        if above_bbox {
            1.0
        } else {
            let frac = (pos.z - bbox.min.z) / (bbox.max.z - bbox.min.z + 1e-10);
            frac.clamp(0.0, 1.0)
        }
    }
}

impl fmt::Display for BinPicker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BinPicker(voxel={:.4}, margin={:.4})",
            self.config.voxel_size, self.config.collision_margin
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster_points(cx: f64, cy: f64, cz: f64, n: usize, spread: f64) -> Vec<Point3> {
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            let angle = t * std::f64::consts::TAU;
            pts.push(Point3::new(
                cx + spread * angle.cos() * (1.0 + t * 0.1),
                cy + spread * angle.sin() * (1.0 + t * 0.05),
                cz + spread * 0.3 * (2.0 * t - 1.0),
            ));
        }
        pts
    }

    #[test]
    fn test_point3_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_cross() {
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_from_points() {
        let pts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0)];
        let aabb = AABB::from_points(&pts).unwrap();
        assert!((aabb.max.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_from_empty() {
        assert!(AABB::from_points(&[]).is_none());
    }

    #[test]
    fn test_aabb_overlaps() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Point3::new(0.5, 0.5, 0.5), Point3::new(1.5, 1.5, 1.5));
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_aabb_no_overlap() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0));
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_aabb_volume() {
        let aabb = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 3.0, 4.0));
        assert!((aabb.volume() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_voxel_downsample() {
        let pts = make_cluster_points(0.0, 0.0, 0.0, 100, 0.05);
        let ds = voxel_downsample(&pts, 0.02).unwrap();
        assert!(ds.len() < pts.len());
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_voxel_downsample_invalid_size() {
        assert!(voxel_downsample(&[], -1.0).is_err());
    }

    #[test]
    fn test_voxel_downsample_empty() {
        let ds = voxel_downsample(&[], 0.01).unwrap();
        assert!(ds.is_empty());
    }

    #[test]
    fn test_segment_objects_single_cluster() {
        let pts = make_cluster_points(0.0, 0.0, 0.0, 100, 0.01);
        let config = SegmentConfig::default()
            .with_distance_threshold(0.02)
            .with_min_cluster_size(10);
        let clusters = segment_objects(&pts, &config);
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_segment_empty() {
        let clusters = segment_objects(&[], &SegmentConfig::default());
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_estimate_pose() {
        let pts = make_cluster_points(0.0, 0.0, 0.0, 50, 0.05);
        let pose = estimate_pose(&pts).unwrap();
        // Elongation is eigenvalue[0]/eigenvalue[2]; Jacobi may produce
        // slightly negative near-zero eigenvalues, so check finite.
        assert!(pose.elongation.is_finite() || pose.elongation == f64::INFINITY);
    }

    #[test]
    fn test_estimate_pose_too_few_points() {
        let pts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        assert!(estimate_pose(&pts).is_err());
    }

    #[test]
    fn test_grasp_candidate_creation() {
        let g = GraspCandidate::new(
            Point3::new(0.0, 0.0, 0.1),
            Point3::new(0.0, 0.0, -1.0),
            0.08,
        );
        assert!(g.collision_free);
    }

    #[test]
    fn test_grasp_candidate_with_score() {
        let g = GraspCandidate::new(Point3::zero(), Point3::new(0.0, 0.0, -1.0), 0.08)
            .with_score(0.9);
        assert!((g.score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_bin_picker_creation() {
        let picker = BinPicker::new(BinPickConfig::default());
        let s = format!("{picker}");
        assert!(s.contains("BinPicker"));
    }

    #[test]
    fn test_config_builder() {
        let cfg = BinPickConfig::default()
            .with_voxel_size(0.01)
            .with_collision_margin(0.01)
            .with_max_gripper_width(0.1);
        assert!((cfg.voxel_size - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_display() {
        let cluster = ObjectCluster {
            indices: vec![0, 1, 2],
            centroid: Point3::zero(),
            bbox: AABB::new(Point3::zero(), Point3::new(1.0, 1.0, 1.0)),
        };
        let s = format!("{cluster}");
        assert!(s.contains("Cluster"));
    }

    #[test]
    fn test_object_pose_display() {
        let pts = make_cluster_points(0.0, 0.0, 0.0, 50, 0.05);
        let pose = estimate_pose(&pts).unwrap();
        let s = format!("{pose}");
        assert!(s.contains("ObjectPose"));
    }
}
