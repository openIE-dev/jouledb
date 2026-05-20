//! Point cloud processing — voxel downsampling, normal estimation,
//! plane segmentation (RANSAC), and DBSCAN clustering.
//!
//! All operations work on `Vec<Point3>` collections with no external
//! dependencies.

use std::collections::HashMap;
use std::fmt;

// ── 3-D point ─────────────────────────────────────────────────────

/// A 3-D point with optional normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }

    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn distance_sq(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    pub fn dot(&self, other: &Point3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Point3) -> Point3 {
        Point3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Point3 {
        let n = self.norm();
        if n < 1e-15 { return Point3::zero(); }
        Point3 { x: self.x / n, y: self.y / n, z: self.z / n }
    }

    pub fn add(&self, other: &Point3) -> Point3 {
        Point3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(&self, other: &Point3) -> Point3 {
        Point3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn scale(&self, s: f64) -> Point3 {
        Point3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── Point cloud ───────────────────────────────────────────────────

/// A collection of 3-D points.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3>,
}

impl PointCloud {
    pub fn new() -> Self { Self { points: Vec::new() } }

    pub fn from_points(points: Vec<Point3>) -> Self { Self { points } }

    pub fn len(&self) -> usize { self.points.len() }

    pub fn is_empty(&self) -> bool { self.points.is_empty() }

    pub fn push(&mut self, p: Point3) { self.points.push(p); }

    /// Centroid (arithmetic mean).
    pub fn centroid(&self) -> Point3 {
        if self.points.is_empty() { return Point3::zero(); }
        let n = self.points.len() as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sz = 0.0;
        for p in &self.points {
            sx += p.x;
            sy += p.y;
            sz += p.z;
        }
        Point3::new(sx / n, sy / n, sz / n)
    }

    /// Axis-aligned bounding box: (min, max).
    pub fn bounding_box(&self) -> (Point3, Point3) {
        if self.points.is_empty() {
            return (Point3::zero(), Point3::zero());
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
        (min, max)
    }
}

impl fmt::Display for PointCloud {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PointCloud(n={})", self.points.len())
    }
}

// ── Voxel downsampling ────────────────────────────────────────────

/// Voxel key for hashing.
fn voxel_key(p: &Point3, voxel_size: f64) -> (i64, i64, i64) {
    let kx = (p.x / voxel_size).floor() as i64;
    let ky = (p.y / voxel_size).floor() as i64;
    let kz = (p.z / voxel_size).floor() as i64;
    (kx, ky, kz)
}

/// Downsample a point cloud using voxel grid filtering.
/// Each occupied voxel emits the centroid of all points within it.
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f64) -> PointCloud {
    let mut buckets: HashMap<(i64, i64, i64), (f64, f64, f64, usize)> = HashMap::new();
    for p in &cloud.points {
        let key = voxel_key(p, voxel_size);
        let entry = buckets.entry(key).or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += p.x;
        entry.1 += p.y;
        entry.2 += p.z;
        entry.3 += 1;
    }
    let mut out = PointCloud::new();
    for (sx, sy, sz, count) in buckets.values() {
        let n = *count as f64;
        out.push(Point3::new(sx / n, sy / n, sz / n));
    }
    out
}

// ── Normal estimation ─────────────────────────────────────────────

/// Estimate surface normals using k nearest neighbors.
/// Returns a vector of unit normals parallel to `cloud.points`.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> Vec<Point3> {
    let n = cloud.len();
    let mut normals = Vec::with_capacity(n);

    for i in 0..n {
        let center = &cloud.points[i];
        // Find k nearest (brute force)
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|j| *j != i)
            .map(|j| (j, center.distance_sq(&cloud.points[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<usize> = dists.iter().take(k).map(|(idx, _)| *idx).collect();
        let normal = fit_plane_normal(center, &neighbors, &cloud.points);
        normals.push(normal);
    }
    normals
}

/// Fit a plane normal via covariance PCA (power iteration on 3×3 covariance).
fn fit_plane_normal(center: &Point3, neighbors: &[usize], points: &[Point3]) -> Point3 {
    if neighbors.len() < 2 { return Point3::new(0.0, 0.0, 1.0); }

    // Compute 3×3 covariance
    let mut cov = [0.0f64; 9];
    let mut count = 0.0;
    for &idx in neighbors {
        let d = points[idx].sub(center);
        cov[0] += d.x * d.x; cov[1] += d.x * d.y; cov[2] += d.x * d.z;
        cov[3] += d.y * d.x; cov[4] += d.y * d.y; cov[5] += d.y * d.z;
        cov[6] += d.z * d.x; cov[7] += d.z * d.y; cov[8] += d.z * d.z;
        count += 1.0;
    }
    for v in &mut cov { *v /= count; }

    // Power iteration to find smallest eigenvector (normal)
    // We find the largest eigenvector, then use cross products
    let mut v1 = Point3::new(1.0, 0.0, 0.0);
    for _ in 0..30 {
        let x = cov[0] * v1.x + cov[1] * v1.y + cov[2] * v1.z;
        let y = cov[3] * v1.x + cov[4] * v1.y + cov[5] * v1.z;
        let z = cov[6] * v1.x + cov[7] * v1.y + cov[8] * v1.z;
        v1 = Point3::new(x, y, z).normalized();
    }

    // Deflate and find second eigenvector
    let mut cov2 = cov;
    for i in 0..3 {
        let vi = [v1.x, v1.y, v1.z][i];
        for j in 0..3 {
            let vj = [v1.x, v1.y, v1.z][j];
            let lam = cov[0] * v1.x * v1.x + cov[4] * v1.y * v1.y + cov[8] * v1.z * v1.z
                + 2.0 * (cov[1] * v1.x * v1.y + cov[2] * v1.x * v1.z + cov[5] * v1.y * v1.z);
            cov2[i * 3 + j] = cov[i * 3 + j] - lam * vi * vj;
        }
    }

    let mut v2 = Point3::new(0.0, 1.0, 0.0);
    for _ in 0..30 {
        let x = cov2[0] * v2.x + cov2[1] * v2.y + cov2[2] * v2.z;
        let y = cov2[3] * v2.x + cov2[4] * v2.y + cov2[5] * v2.z;
        let z = cov2[6] * v2.x + cov2[7] * v2.y + cov2[8] * v2.z;
        v2 = Point3::new(x, y, z).normalized();
    }

    // Normal = cross product of the two principal directions
    v1.cross(&v2).normalized()
}

// ── Plane segmentation (RANSAC) ───────────────────────────────────

/// Plane model: ax + by + cz + d = 0.
#[derive(Debug, Clone, Copy)]
pub struct PlaneModel {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl PlaneModel {
    /// Distance from a point to this plane.
    pub fn distance(&self, p: &Point3) -> f64 {
        (self.a * p.x + self.b * p.y + self.c * p.z + self.d).abs()
            / (self.a * self.a + self.b * self.b + self.c * self.c).sqrt()
    }
}

impl fmt::Display for PlaneModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plane({:.4}x + {:.4}y + {:.4}z + {:.4} = 0)", self.a, self.b, self.c, self.d)
    }
}

/// Simple LCG RNG for RANSAC.
struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn range(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }
}

/// RANSAC plane segmentation. Returns (inlier indices, plane model).
pub fn ransac_plane(
    cloud: &PointCloud,
    max_iterations: usize,
    distance_threshold: f64,
    seed: u64,
) -> Option<(Vec<usize>, PlaneModel)> {
    let n = cloud.len();
    if n < 3 { return None; }

    let mut rng = SimpleRng::new(seed);
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_plane = PlaneModel { a: 0.0, b: 0.0, c: 0.0, d: 0.0 };

    for _ in 0..max_iterations {
        let i0 = rng.range(n);
        let i1 = rng.range(n);
        let i2 = rng.range(n);
        if i0 == i1 || i1 == i2 || i0 == i2 { continue; }

        let p0 = &cloud.points[i0];
        let p1 = &cloud.points[i1];
        let p2 = &cloud.points[i2];

        let v1 = p1.sub(p0);
        let v2 = p2.sub(p0);
        let normal = v1.cross(&v2);
        let norm_len = normal.norm();
        if norm_len < 1e-12 { continue; }

        let nn = normal.scale(1.0 / norm_len);
        let d = -(nn.x * p0.x + nn.y * p0.y + nn.z * p0.z);
        let plane = PlaneModel { a: nn.x, b: nn.y, c: nn.z, d };

        let inliers: Vec<usize> = (0..n)
            .filter(|j| plane.distance(&cloud.points[*j]) < distance_threshold)
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_plane = plane;
        }
    }

    if best_inliers.len() >= 3 {
        Some((best_inliers, best_plane))
    } else {
        None
    }
}

// ── DBSCAN clustering ─────────────────────────────────────────────

/// DBSCAN clustering on a point cloud.
/// Returns a vector of cluster labels (−1 = noise).
pub fn dbscan(cloud: &PointCloud, eps: f64, min_pts: usize) -> Vec<i32> {
    let n = cloud.len();
    let eps_sq = eps * eps;
    let mut labels = vec![-1i32; n];
    let mut cluster_id: i32 = 0;

    for i in 0..n {
        if labels[i] != -1 { continue; }

        // Region query
        let neighbors: Vec<usize> = (0..n)
            .filter(|j| cloud.points[i].distance_sq(&cloud.points[*j]) <= eps_sq)
            .collect();

        if neighbors.len() < min_pts { continue; }

        labels[i] = cluster_id;
        let mut seed_set: Vec<usize> = neighbors.into_iter().filter(|j| *j != i).collect();
        let mut si = 0;
        while si < seed_set.len() {
            let q = seed_set[si];
            if labels[q] == -1 || labels[q] == -2 {
                // -2 would be "visited noise" but we just use -1
            }
            if labels[q] != -1 && labels[q] != cluster_id {
                si += 1;
                continue;
            }

            labels[q] = cluster_id;

            let q_neighbors: Vec<usize> = (0..n)
                .filter(|j| cloud.points[q].distance_sq(&cloud.points[*j]) <= eps_sq)
                .collect();

            if q_neighbors.len() >= min_pts {
                for &nb in &q_neighbors {
                    if labels[nb] == -1 {
                        seed_set.push(nb);
                    }
                }
            }

            si += 1;
        }

        cluster_id += 1;
    }

    labels
}

/// Count distinct clusters (excluding noise).
pub fn cluster_count(labels: &[i32]) -> usize {
    let mut max_id = -1i32;
    for &l in labels {
        if l > max_id { max_id = l; }
    }
    if max_id < 0 { 0 } else { (max_id + 1) as usize }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cloud() -> PointCloud {
        let mut pc = PointCloud::new();
        for i in 0..100 {
            let t = i as f64 * 0.1;
            pc.push(Point3::new(t.cos(), t.sin(), (i as f64) * 0.01));
        }
        pc
    }

    #[test]
    fn test_point_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_cross() {
        let x = Point3::new(1.0, 0.0, 0.0);
        let y = Point3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.x).abs() < 1e-10);
        assert!((z.y).abs() < 1e-10);
        assert!((z.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_normalize() {
        let p = Point3::new(3.0, 4.0, 0.0);
        let n = p.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_display() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let s = format!("{}", p);
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_cloud_centroid() {
        let mut pc = PointCloud::new();
        pc.push(Point3::new(0.0, 0.0, 0.0));
        pc.push(Point3::new(2.0, 4.0, 6.0));
        let c = pc.centroid();
        assert!((c.x - 1.0).abs() < 1e-10);
        assert!((c.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cloud_bounding_box() {
        let pc = sample_cloud();
        let (mn, mx) = pc.bounding_box();
        assert!(mn.x < mx.x);
        assert!(mn.z < mx.z);
    }

    #[test]
    fn test_voxel_downsample() {
        let pc = sample_cloud();
        let ds = voxel_downsample(&pc, 0.5);
        assert!(ds.len() < pc.len());
        assert!(ds.len() > 0);
    }

    #[test]
    fn test_voxel_downsample_preserves_single() {
        let mut pc = PointCloud::new();
        pc.push(Point3::new(1.0, 2.0, 3.0));
        let ds = voxel_downsample(&pc, 1.0);
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_estimate_normals() {
        // Flat plane z=0
        let mut pc = PointCloud::new();
        for i in 0..20 {
            for j in 0..20 {
                pc.push(Point3::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
            }
        }
        let normals = estimate_normals(&pc, 5);
        assert_eq!(normals.len(), pc.len());
        // Normals should be roughly ±z
        let nz = normals[200].z.abs(); // middle point
        assert!(nz > 0.5, "Expected normal ~z, got {}", nz);
    }

    #[test]
    fn test_plane_distance() {
        let plane = PlaneModel { a: 0.0, b: 0.0, c: 1.0, d: 0.0 };
        let p = Point3::new(5.0, 3.0, 2.0);
        assert!((plane.distance(&p) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_plane_display() {
        let plane = PlaneModel { a: 0.0, b: 0.0, c: 1.0, d: -1.0 };
        let s = format!("{}", plane);
        assert!(s.contains("Plane"));
    }

    #[test]
    fn test_ransac_plane() {
        let mut pc = PointCloud::new();
        // Ground plane z=0
        for i in 0..200 {
            pc.push(Point3::new(i as f64 * 0.05, (i * 7 % 100) as f64 * 0.05, 0.0));
        }
        // Outliers
        for _ in 0..20 {
            pc.push(Point3::new(1.0, 1.0, 5.0));
        }
        let result = ransac_plane(&pc, 100, 0.01, 42);
        assert!(result.is_some());
        let (inliers, plane) = result.unwrap();
        assert!(inliers.len() >= 180);
        assert!(plane.c.abs() > 0.9);
    }

    #[test]
    fn test_dbscan_two_clusters() {
        let mut pc = PointCloud::new();
        // Cluster A around (0,0,0)
        for i in 0..20 {
            pc.push(Point3::new(i as f64 * 0.01, 0.0, 0.0));
        }
        // Cluster B around (10,0,0)
        for i in 0..20 {
            pc.push(Point3::new(10.0 + i as f64 * 0.01, 0.0, 0.0));
        }
        let labels = dbscan(&pc, 0.5, 3);
        let nc = cluster_count(&labels);
        assert_eq!(nc, 2);
    }

    #[test]
    fn test_dbscan_noise() {
        let mut pc = PointCloud::new();
        // Scattered points
        for i in 0..5 {
            pc.push(Point3::new(i as f64 * 100.0, 0.0, 0.0));
        }
        let labels = dbscan(&pc, 0.1, 3);
        assert!(labels.iter().all(|l| *l == -1));
    }

    #[test]
    fn test_cluster_count_empty() {
        assert_eq!(cluster_count(&[]), 0);
    }

    #[test]
    fn test_cloud_display() {
        let pc = sample_cloud();
        let s = format!("{}", pc);
        assert!(s.contains("PointCloud(n=100)"));
    }

    #[test]
    fn test_point_dot() {
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        assert!((a.dot(&b)).abs() < 1e-10);
    }
}
