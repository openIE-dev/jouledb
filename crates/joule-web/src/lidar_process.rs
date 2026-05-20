//! LiDAR processing — point cloud construction from range data, scan matching
//! via iterative closest point (ICP), ground plane detection with RANSAC,
//! and obstacle clustering using Euclidean distance.
//!
//! Pure-Rust LiDAR pipeline operating on 2D/3D range scans, suitable for
//! embedded navigation and mapping workloads without external dependencies.

use std::f64::consts::PI;
use std::fmt;

// ── Point types ─────────────────────────────────────────────────

/// A 3D point with optional intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z, intensity: 0.0 }
    }

    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn dot(&self, other: &Point3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Point3) -> Point3 {
        Point3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
            intensity: 0.0,
        }
    }

    pub fn sub(&self, other: &Point3) -> Point3 {
        Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn add(&self, other: &Point3) -> Point3 {
        Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn scale(&self, s: f64) -> Point3 {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// ── Point Cloud ─────────────────────────────────────────────────

/// A collection of 3D points from a LiDAR scan.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { points: Vec::with_capacity(capacity) }
    }

    pub fn push(&mut self, pt: Point3) {
        self.points.push(pt);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute centroid of the point cloud.
    pub fn centroid(&self) -> Point3 {
        if self.points.is_empty() {
            return Point3::new(0.0, 0.0, 0.0);
        }
        let n = self.points.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for p in &self.points {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        Point3::new(cx / n, cy / n, cz / n)
    }

    /// Bounding box as (min_corner, max_corner).
    pub fn bounding_box(&self) -> Option<(Point3, Point3)> {
        if self.points.is_empty() {
            return None;
        }
        let mut min = self.points[0];
        let mut max = self.points[0];
        for p in &self.points[1..] {
            if p.x < min.x { min.x = p.x; }
            if p.y < min.y { min.y = p.y; }
            if p.z < min.z { min.z = p.z; }
            if p.x > max.x { max.x = p.x; }
            if p.y > max.y { max.y = p.y; }
            if p.z > max.z { max.z = p.z; }
        }
        Some((min, max))
    }

    /// Filter points by maximum range from origin.
    pub fn filter_range(&self, max_range: f64) -> PointCloud {
        let mut out = PointCloud::new();
        for p in &self.points {
            if p.norm() <= max_range {
                out.push(*p);
            }
        }
        out
    }

    /// Down-sample using a voxel grid with given resolution.
    pub fn voxel_downsample(&self, voxel_size: f64) -> PointCloud {
        if voxel_size <= 0.0 || self.points.is_empty() {
            return self.clone();
        }
        let inv = 1.0 / voxel_size;
        let mut voxels: std::collections::HashMap<(i64, i64, i64), (Point3, usize)> =
            std::collections::HashMap::new();
        for p in &self.points {
            let key = (
                (p.x * inv).floor() as i64,
                (p.y * inv).floor() as i64,
                (p.z * inv).floor() as i64,
            );
            let entry = voxels.entry(key).or_insert((Point3::new(0.0, 0.0, 0.0), 0));
            entry.0.x += p.x;
            entry.0.y += p.y;
            entry.0.z += p.z;
            entry.1 += 1;
        }
        let mut out = PointCloud::with_capacity(voxels.len());
        for (_, (sum, count)) in &voxels {
            let n = *count as f64;
            out.push(Point3::new(sum.x / n, sum.y / n, sum.z / n));
        }
        out
    }
}

impl fmt::Display for PointCloud {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PointCloud({} points)", self.points.len())
    }
}

// ── Range Scan to Point Cloud ───────────────────────────────────

/// Configuration for converting a 2D LiDAR scan into a point cloud.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_increment: f64,
    pub range_min: f64,
    pub range_max: f64,
    pub mount_height: f64,
    pub mount_pitch: f64,
}

impl ScanConfig {
    pub fn new(angle_min: f64, angle_max: f64, num_beams: usize) -> Self {
        let angle_increment = if num_beams > 1 {
            (angle_max - angle_min) / (num_beams - 1) as f64
        } else {
            0.0
        };
        Self {
            angle_min,
            angle_max,
            angle_increment,
            range_min: 0.1,
            range_max: 100.0,
            mount_height: 0.0,
            mount_pitch: 0.0,
        }
    }

    pub fn with_range_limits(mut self, min: f64, max: f64) -> Self {
        self.range_min = min;
        self.range_max = max;
        self
    }

    pub fn with_mount(mut self, height: f64, pitch: f64) -> Self {
        self.mount_height = height;
        self.mount_pitch = pitch;
        self
    }

    /// Convert an array of range values into a 3D point cloud.
    pub fn ranges_to_cloud(&self, ranges: &[f64]) -> PointCloud {
        let mut cloud = PointCloud::with_capacity(ranges.len());
        let cp = self.mount_pitch.cos();
        let sp = self.mount_pitch.sin();
        for (i, &r) in ranges.iter().enumerate() {
            if r < self.range_min || r > self.range_max || r.is_nan() {
                continue;
            }
            let angle = self.angle_min + i as f64 * self.angle_increment;
            let x = r * angle.cos() * cp;
            let y = r * angle.sin() * cp;
            let z = self.mount_height + r * sp;
            cloud.push(Point3::new(x, y, z));
        }
        cloud
    }
}

impl fmt::Display for ScanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ScanConfig(angle=[{:.1}..{:.1}] deg, range=[{:.1}..{:.1}] m)",
            self.angle_min.to_degrees(),
            self.angle_max.to_degrees(),
            self.range_min,
            self.range_max,
        )
    }
}

// ── Scan Matching (ICP) ────────────────────────────────────────

/// 2D rigid transform (rotation + translation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub tx: f64,
    pub ty: f64,
    pub theta: f64,
}

impl Transform2D {
    pub fn identity() -> Self {
        Self { tx: 0.0, ty: 0.0, theta: 0.0 }
    }

    pub fn apply(&self, p: &Point3) -> Point3 {
        let c = self.theta.cos();
        let s = self.theta.sin();
        Point3::new(
            c * p.x - s * p.y + self.tx,
            s * p.x + c * p.y + self.ty,
            p.z,
        )
    }

    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        let c = self.theta.cos();
        let s = self.theta.sin();
        Transform2D {
            tx: c * other.tx - s * other.ty + self.tx,
            ty: s * other.tx + c * other.ty + self.ty,
            theta: self.theta + other.theta,
        }
    }
}

impl fmt::Display for Transform2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "T2D(tx={:.4}, ty={:.4}, theta={:.4} deg)",
            self.tx, self.ty, self.theta.to_degrees()
        )
    }
}

/// Iterative closest point (ICP) scan matcher for 2D point clouds.
#[derive(Debug, Clone)]
pub struct IcpMatcher {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub max_correspondence_dist: f64,
}

impl IcpMatcher {
    pub fn new() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-6,
            max_correspondence_dist: 2.0,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_convergence_threshold(mut self, eps: f64) -> Self {
        self.convergence_threshold = eps;
        self
    }

    pub fn with_max_correspondence_dist(mut self, d: f64) -> Self {
        self.max_correspondence_dist = d;
        self
    }

    /// Find nearest neighbor in target for a query point (brute-force).
    fn nearest_neighbor(query: &Point3, target: &[Point3]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, t) in target.iter().enumerate() {
            let dx = query.x - t.x;
            let dy = query.y - t.y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist {
                best_dist = d2;
                best_idx = i;
            }
        }
        (best_idx, best_dist.sqrt())
    }

    /// Align source cloud to target cloud. Returns the estimated transform.
    pub fn align(&self, source: &PointCloud, target: &PointCloud) -> Transform2D {
        if source.is_empty() || target.is_empty() {
            return Transform2D::identity();
        }
        let mut cumulative = Transform2D::identity();
        let mut transformed: Vec<Point3> = source.points.clone();

        for _ in 0..self.max_iterations {
            // Find correspondences
            let mut src_matched = Vec::new();
            let mut tgt_matched = Vec::new();
            for s in &transformed {
                let (idx, dist) = Self::nearest_neighbor(s, &target.points);
                if dist < self.max_correspondence_dist {
                    src_matched.push(*s);
                    tgt_matched.push(target.points[idx]);
                }
            }
            if src_matched.len() < 3 {
                break;
            }
            // Compute centroids
            let n = src_matched.len() as f64;
            let (mut sx, mut sy) = (0.0, 0.0);
            let (mut tx_sum, mut ty_sum) = (0.0, 0.0);
            for (s, t) in src_matched.iter().zip(tgt_matched.iter()) {
                sx += s.x; sy += s.y;
                tx_sum += t.x; ty_sum += t.y;
            }
            let sc = (sx / n, sy / n);
            let tc = (tx_sum / n, ty_sum / n);

            // SVD-free 2D rotation via cross/dot sums
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for (s, t) in src_matched.iter().zip(tgt_matched.iter()) {
                let ds = (s.x - sc.0, s.y - sc.1);
                let dt = (t.x - tc.0, t.y - tc.1);
                sxx += ds.0 * dt.0 + ds.1 * dt.1;
                sxy += ds.0 * dt.1 - ds.1 * dt.0;
            }
            let theta = sxy.atan2(sxx);
            let c = theta.cos();
            let s_val = theta.sin();
            let tx = tc.0 - (c * sc.0 - s_val * sc.1);
            let ty = tc.1 - (s_val * sc.0 + c * sc.1);

            let delta = Transform2D { tx, ty, theta };
            // Apply delta
            for p in &mut transformed {
                *p = delta.apply(p);
            }
            cumulative = delta.compose(&cumulative);

            if tx.abs() < self.convergence_threshold
                && ty.abs() < self.convergence_threshold
                && theta.abs() < self.convergence_threshold
            {
                break;
            }
        }
        cumulative
    }
}

impl fmt::Display for IcpMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ICP(max_iter={}, eps={:.1e})", self.max_iterations, self.convergence_threshold)
    }
}

// ── Ground Plane Detection (RANSAC) ─────────────────────────────

/// RANSAC-based ground plane detector.
#[derive(Debug, Clone)]
pub struct GroundDetector {
    pub max_iterations: usize,
    pub distance_threshold: f64,
    pub min_inlier_ratio: f64,
    seed: u64,
}

impl GroundDetector {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            distance_threshold: 0.15,
            min_inlier_ratio: 0.3,
            seed: 42,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_distance_threshold(mut self, d: f64) -> Self {
        self.distance_threshold = d;
        self
    }

    pub fn with_min_inlier_ratio(mut self, r: f64) -> Self {
        self.min_inlier_ratio = r;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn pseudo_random(&self, iteration: usize, which: usize) -> usize {
        let mut h = self.seed.wrapping_add((iteration as u64).wrapping_mul(6364136223846793005));
        h = h.wrapping_add((which as u64).wrapping_mul(1442695040888963407));
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        h as usize
    }

    /// Plane equation: ax + by + cz + d = 0. Returns (a, b, c, d, inlier_indices).
    pub fn detect(&self, cloud: &PointCloud) -> Option<(f64, f64, f64, f64, Vec<usize>)> {
        let n = cloud.points.len();
        if n < 3 {
            return None;
        }
        let mut best_inliers: Vec<usize> = Vec::new();
        for iter in 0..self.max_iterations {
            let i0 = self.pseudo_random(iter, 0) % n;
            let i1 = self.pseudo_random(iter, 1) % n;
            let i2 = self.pseudo_random(iter, 2) % n;
            if i0 == i1 || i0 == i2 || i1 == i2 {
                continue;
            }
            let p0 = &cloud.points[i0];
            let p1 = &cloud.points[i1];
            let p2 = &cloud.points[i2];
            let v1 = p1.sub(p0);
            let v2 = p2.sub(p0);
            let normal = v1.cross(&v2);
            let len = normal.norm();
            if len < 1e-12 {
                continue;
            }
            let a = normal.x / len;
            let b = normal.y / len;
            let c = normal.z / len;
            let d = -(a * p0.x + b * p0.y + c * p0.z);

            let mut inliers = Vec::new();
            for (j, p) in cloud.points.iter().enumerate() {
                let dist = (a * p.x + b * p.y + c * p.z + d).abs();
                if dist < self.distance_threshold {
                    inliers.push(j);
                }
            }
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }
        if (best_inliers.len() as f64 / n as f64) < self.min_inlier_ratio {
            return None;
        }
        // Refit plane to all inliers via least-squares normal
        let centroid = {
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cz = 0.0;
            for &i in &best_inliers {
                cx += cloud.points[i].x;
                cy += cloud.points[i].y;
                cz += cloud.points[i].z;
            }
            let m = best_inliers.len() as f64;
            Point3::new(cx / m, cy / m, cz / m)
        };
        // Covariance matrix (3x3, symmetric)
        let mut cov = [[0.0f64; 3]; 3];
        for &i in &best_inliers {
            let dx = cloud.points[i].x - centroid.x;
            let dy = cloud.points[i].y - centroid.y;
            let dz = cloud.points[i].z - centroid.z;
            cov[0][0] += dx * dx;
            cov[0][1] += dx * dy;
            cov[0][2] += dx * dz;
            cov[1][1] += dy * dy;
            cov[1][2] += dy * dz;
            cov[2][2] += dz * dz;
        }
        cov[1][0] = cov[0][1];
        cov[2][0] = cov[0][2];
        cov[2][1] = cov[1][2];
        // Use smallest eigenvector via power iteration on inverse (approximate)
        // For ground plane, assume z-dominant normal
        let len = (cov[0][2] * cov[0][2] + cov[1][2] * cov[1][2] + cov[2][2] * cov[2][2]).sqrt();
        let (a, b, c) = if len > 1e-12 {
            (cov[0][2] / len, cov[1][2] / len, cov[2][2] / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        let d = -(a * centroid.x + b * centroid.y + c * centroid.z);
        Some((a, b, c, d, best_inliers))
    }

    /// Separate a point cloud into ground and non-ground points.
    pub fn segment(&self, cloud: &PointCloud) -> (PointCloud, PointCloud) {
        let mut ground = PointCloud::new();
        let mut obstacles = PointCloud::new();
        if let Some((_, _, _, _, inliers)) = self.detect(cloud) {
            let inlier_set: std::collections::HashSet<usize> = inliers.into_iter().collect();
            for (i, p) in cloud.points.iter().enumerate() {
                if inlier_set.contains(&i) {
                    ground.push(*p);
                } else {
                    obstacles.push(*p);
                }
            }
        } else {
            obstacles = cloud.clone();
        }
        (ground, obstacles)
    }
}

impl fmt::Display for GroundDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GroundDetector(iters={}, thresh={:.3})",
            self.max_iterations, self.distance_threshold
        )
    }
}

// ── Obstacle Clustering ─────────────────────────────────────────

/// Cluster of points representing a detected obstacle.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub points: Vec<Point3>,
    pub id: usize,
}

impl Cluster {
    pub fn centroid(&self) -> Point3 {
        let n = self.points.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for p in &self.points {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        Point3::new(cx / n, cy / n, cz / n)
    }

    pub fn bounding_radius(&self) -> f64 {
        let c = self.centroid();
        let mut max_r = 0.0f64;
        for p in &self.points {
            let r = p.distance_to(&c);
            if r > max_r {
                max_r = r;
            }
        }
        max_r
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cluster(id={}, points={})", self.id, self.points.len())
    }
}

/// Euclidean clustering for obstacle segmentation.
#[derive(Debug, Clone)]
pub struct EuclideanClusterer {
    pub cluster_tolerance: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
}

impl EuclideanClusterer {
    pub fn new(tolerance: f64) -> Self {
        Self {
            cluster_tolerance: tolerance,
            min_cluster_size: 3,
            max_cluster_size: 10000,
        }
    }

    pub fn with_min_size(mut self, s: usize) -> Self {
        self.min_cluster_size = s;
        self
    }

    pub fn with_max_size(mut self, s: usize) -> Self {
        self.max_cluster_size = s;
        self
    }

    /// Perform Euclidean clustering on a point cloud.
    pub fn cluster(&self, cloud: &PointCloud) -> Vec<Cluster> {
        let n = cloud.points.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();
        let mut cluster_id = 0;
        let tol2 = self.cluster_tolerance * self.cluster_tolerance;

        for i in 0..n {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let mut queue = vec![i];
            let mut cluster_pts = vec![cloud.points[i]];
            let mut head = 0;

            while head < queue.len() {
                let qi = queue[head];
                head += 1;
                let qp = &cloud.points[qi];
                for j in 0..n {
                    if visited[j] {
                        continue;
                    }
                    let dx = qp.x - cloud.points[j].x;
                    let dy = qp.y - cloud.points[j].y;
                    let dz = qp.z - cloud.points[j].z;
                    if dx * dx + dy * dy + dz * dz <= tol2 {
                        visited[j] = true;
                        queue.push(j);
                        cluster_pts.push(cloud.points[j]);
                    }
                }
                if cluster_pts.len() > self.max_cluster_size {
                    break;
                }
            }
            if cluster_pts.len() >= self.min_cluster_size
                && cluster_pts.len() <= self.max_cluster_size
            {
                clusters.push(Cluster { points: cluster_pts, id: cluster_id });
                cluster_id += 1;
            }
        }
        clusters
    }
}

impl fmt::Display for EuclideanClusterer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EuclideanClusterer(tol={:.3}, size=[{}..{}])",
            self.cluster_tolerance, self.min_cluster_size, self.max_cluster_size
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_point3_distance() {
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 0.0, 0.0);
        assert!((a.distance_to(&b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_point3_cross() {
        let x = Point3::new(1.0, 0.0, 0.0);
        let y = Point3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.x).abs() < 1e-12);
        assert!((z.y).abs() < 1e-12);
        assert!((z.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_point3_display() {
        let p = Point3::new(1.5, -2.3, 0.0);
        let s = format!("{p}");
        assert!(s.contains("1.500"));
    }

    #[test]
    fn test_cloud_centroid() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3::new(0.0, 0.0, 0.0));
        cloud.push(Point3::new(2.0, 0.0, 0.0));
        cloud.push(Point3::new(0.0, 2.0, 0.0));
        let c = cloud.centroid();
        assert!((c.x - 2.0 / 3.0).abs() < 1e-10);
        assert!((c.y - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cloud_bounding_box() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3::new(-1.0, 2.0, 3.0));
        cloud.push(Point3::new(4.0, -5.0, 6.0));
        let (min, max) = cloud.bounding_box().unwrap();
        assert!((min.x - (-1.0)).abs() < 1e-12);
        assert!((max.x - 4.0).abs() < 1e-12);
        assert!((min.y - (-5.0)).abs() < 1e-12);
    }

    #[test]
    fn test_filter_range() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3::new(1.0, 0.0, 0.0));
        cloud.push(Point3::new(10.0, 0.0, 0.0));
        cloud.push(Point3::new(2.0, 0.0, 0.0));
        let filtered = cloud.filter_range(5.0);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_voxel_downsample() {
        let mut cloud = PointCloud::new();
        // Four points in the same voxel
        cloud.push(Point3::new(0.1, 0.1, 0.1));
        cloud.push(Point3::new(0.2, 0.2, 0.2));
        cloud.push(Point3::new(0.3, 0.3, 0.3));
        cloud.push(Point3::new(0.4, 0.4, 0.4));
        // One point in a different voxel
        cloud.push(Point3::new(5.0, 5.0, 5.0));
        let ds = cloud.voxel_downsample(1.0);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_scan_config_basic() {
        let config = ScanConfig::new(-PI / 2.0, PI / 2.0, 5);
        let ranges = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let cloud = config.ranges_to_cloud(&ranges);
        assert_eq!(cloud.len(), 5);
    }

    #[test]
    fn test_scan_config_filters_invalid() {
        let config = ScanConfig::new(0.0, PI, 3).with_range_limits(0.5, 10.0);
        let ranges = vec![0.1, 5.0, 15.0]; // too close, ok, too far
        let cloud = config.ranges_to_cloud(&ranges);
        assert_eq!(cloud.len(), 1);
    }

    #[test]
    fn test_scan_config_display() {
        let config = ScanConfig::new(-PI, PI, 360);
        let s = format!("{config}");
        assert!(s.contains("ScanConfig"));
    }

    #[test]
    fn test_transform2d_identity() {
        let t = Transform2D::identity();
        let p = Point3::new(3.0, 4.0, 0.0);
        let q = t.apply(&p);
        assert!((q.x - 3.0).abs() < 1e-12);
        assert!((q.y - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_transform2d_rotation() {
        let t = Transform2D { tx: 0.0, ty: 0.0, theta: PI / 2.0 };
        let p = Point3::new(1.0, 0.0, 0.0);
        let q = t.apply(&p);
        assert!((q.x).abs() < 1e-12);
        assert!((q.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_icp_identical_clouds() {
        let mut cloud = PointCloud::new();
        for i in 0..20 {
            let angle = i as f64 * PI / 10.0;
            cloud.push(Point3::new(angle.cos() * 5.0, angle.sin() * 5.0, 0.0));
        }
        let matcher = IcpMatcher::new();
        let t = matcher.align(&cloud, &cloud);
        assert!(t.tx.abs() < 0.01);
        assert!(t.ty.abs() < 0.01);
        assert!(t.theta.abs() < 0.01);
    }

    #[test]
    fn test_icp_display() {
        let matcher = IcpMatcher::new().with_max_iterations(100);
        let s = format!("{matcher}");
        assert!(s.contains("ICP"));
    }

    #[test]
    fn test_ground_detector_flat() {
        let mut cloud = PointCloud::new();
        // Flat ground plane at z=0
        for i in 0..50 {
            let x = (i % 10) as f64 * 0.5;
            let y = (i / 10) as f64 * 0.5;
            cloud.push(Point3::new(x, y, 0.0));
        }
        // Some obstacles above
        cloud.push(Point3::new(1.0, 1.0, 2.0));
        cloud.push(Point3::new(2.0, 2.0, 3.0));
        let detector = GroundDetector::new().with_distance_threshold(0.1);
        let result = detector.detect(&cloud);
        assert!(result.is_some());
        let (_, _, _, _, inliers) = result.unwrap();
        assert!(inliers.len() >= 40);
    }

    #[test]
    fn test_ground_segment() {
        let mut cloud = PointCloud::new();
        // Create a 2D ground plane (non-collinear) so RANSAC can fit a plane
        for i in 0..30 {
            let x = (i % 6) as f64 * 0.5;
            let y = (i / 6) as f64 * 0.5;
            cloud.push(Point3::new(x, y, 0.0));
        }
        cloud.push(Point3::new(1.0, 1.0, 5.0));
        let detector = GroundDetector::new();
        let (ground, obstacles) = detector.segment(&cloud);
        assert!(ground.len() >= 20);
        assert!(obstacles.len() >= 1);
    }

    #[test]
    fn test_cluster_centroid() {
        let cluster = Cluster {
            points: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(0.0, 2.0, 0.0),
            ],
            id: 0,
        };
        let c = cluster.centroid();
        assert!((c.x - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_clustering() {
        let mut cloud = PointCloud::new();
        // Cluster A: near origin
        cloud.push(Point3::new(0.0, 0.0, 0.0));
        cloud.push(Point3::new(0.1, 0.0, 0.0));
        cloud.push(Point3::new(0.0, 0.1, 0.0));
        cloud.push(Point3::new(0.1, 0.1, 0.0));
        // Cluster B: far away
        cloud.push(Point3::new(10.0, 10.0, 0.0));
        cloud.push(Point3::new(10.1, 10.0, 0.0));
        cloud.push(Point3::new(10.0, 10.1, 0.0));
        let clusterer = EuclideanClusterer::new(0.5).with_min_size(3);
        let clusters = clusterer.cluster(&cloud);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_clustering_min_size() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3::new(0.0, 0.0, 0.0));
        cloud.push(Point3::new(0.1, 0.0, 0.0));
        let clusterer = EuclideanClusterer::new(0.5).with_min_size(5);
        let clusters = clusterer.cluster(&cloud);
        assert_eq!(clusters.len(), 0);
    }

    #[test]
    fn test_cloud_display() {
        let cloud = PointCloud::new();
        assert_eq!(format!("{cloud}"), "PointCloud(0 points)");
    }
}
