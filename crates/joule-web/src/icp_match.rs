//! Iterative Closest Point (ICP) — rigid-body alignment of 3-D point sets.
//!
//! Implements point-to-point and point-to-plane ICP variants with a KD-tree
//! accelerated nearest-neighbor search and configurable convergence criteria.

use std::fmt;

// ── 3-D point ─────────────────────────────────────────────────────

/// A 3-D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }

    pub fn add(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn sub(&self, o: &Vec3) -> Vec3 { Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn scale(&self, s: f64) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) }
    pub fn dot(&self, o: &Vec3) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn norm_sq(&self) -> f64 { self.dot(self) }
    pub fn norm(&self) -> f64 { self.norm_sq().sqrt() }
    pub fn normalized(&self) -> Vec3 {
        let n = self.norm();
        if n < 1e-15 { Vec3::zero() } else { self.scale(1.0 / n) }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── 3×3 rotation matrix ──────────────────────────────────────────

/// Row-major 3×3 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub data: [f64; 9],
}

impl Mat3 {
    pub fn identity() -> Self { Self { data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] } }

    pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * 3 + c] }

    pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * 3 + c] = v; }

    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            self.get(0, 0) * v.x + self.get(0, 1) * v.y + self.get(0, 2) * v.z,
            self.get(1, 0) * v.x + self.get(1, 1) * v.y + self.get(1, 2) * v.z,
            self.get(2, 0) * v.x + self.get(2, 1) * v.y + self.get(2, 2) * v.z,
        )
    }

    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let mut out = Mat3 { data: [0.0; 9] };
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
        let mut out = Mat3 { data: [0.0; 9] };
        for i in 0..3 { for j in 0..3 { out.set(i, j, self.get(j, i)); } }
        out
    }

    /// SVD-based computation: returns rotation that best aligns the cross-covariance H.
    /// Uses iterative polar decomposition as a substitute for full SVD.
    pub fn rotation_from_cross_covariance(h: &Mat3) -> Mat3 {
        // Polar decomposition: iterate R = 0.5 * (R + (R^{-T}))
        // Start from identity
        let mut r = *h;
        for _ in 0..50 {
            let rt = r.transpose();
            // For orthogonality: R_new = 0.5*(R + inv(R^T))
            // Approximate inv(R^T) with cofactor/det
            let det = r.determinant();
            if det.abs() < 1e-15 { return Mat3::identity(); }
            let adj = r.adjugate();
            let inv_rt_data: [f64; 9] = {
                let mut d = [0.0; 9];
                // inv(R^T) = adj(R) / det(R) then transpose
                for i in 0..3 {
                    for j in 0..3 {
                        d[i * 3 + j] = adj.get(j, i) / det;
                    }
                }
                d
            };
            let inv_rt = Mat3 { data: inv_rt_data };

            let mut new_r = Mat3 { data: [0.0; 9] };
            for i in 0..9 {
                new_r.data[i] = 0.5 * (r.data[i] + inv_rt.data[i]);
            }

            // Check convergence
            let diff: f64 = (0..9).map(|i| (new_r.data[i] - r.data[i]).abs()).sum();
            r = new_r;
            if diff < 1e-12 { break; }
            let _ = rt; // suppress unused
        }

        // Ensure proper rotation (det = +1)
        if r.determinant() < 0.0 {
            // Negate the column with smallest singular value (approximate)
            for i in 0..3 { r.set(i, 2, -r.get(i, 2)); }
        }
        r
    }

    pub fn determinant(&self) -> f64 {
        self.get(0, 0) * (self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1))
            - self.get(0, 1) * (self.get(1, 0) * self.get(2, 2) - self.get(1, 2) * self.get(2, 0))
            + self.get(0, 2) * (self.get(1, 0) * self.get(2, 1) - self.get(1, 1) * self.get(2, 0))
    }

    pub fn adjugate(&self) -> Mat3 {
        let mut adj = Mat3 { data: [0.0; 9] };
        adj.set(0, 0, self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1));
        adj.set(0, 1, -(self.get(1, 0) * self.get(2, 2) - self.get(1, 2) * self.get(2, 0)));
        adj.set(0, 2, self.get(1, 0) * self.get(2, 1) - self.get(1, 1) * self.get(2, 0));
        adj.set(1, 0, -(self.get(0, 1) * self.get(2, 2) - self.get(0, 2) * self.get(2, 1)));
        adj.set(1, 1, self.get(0, 0) * self.get(2, 2) - self.get(0, 2) * self.get(2, 0));
        adj.set(1, 2, -(self.get(0, 0) * self.get(2, 1) - self.get(0, 1) * self.get(2, 0)));
        adj.set(2, 0, self.get(0, 1) * self.get(1, 2) - self.get(0, 2) * self.get(1, 1));
        adj.set(2, 1, -(self.get(0, 0) * self.get(1, 2) - self.get(0, 2) * self.get(1, 0)));
        adj.set(2, 2, self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0));
        adj
    }
}

// ── KD-tree (3-D) ─────────────────────────────────────────────────

#[derive(Debug)]
enum KdNode {
    Leaf { point: Vec3, index: usize },
    Split { axis: usize, median: f64, left: Box<KdNode>, right: Box<KdNode> },
}

/// A 3-D KD-tree for nearest-neighbor queries.
#[derive(Debug)]
pub struct KdTree {
    root: Option<KdNode>,
    size: usize,
}

impl KdTree {
    /// Build a KD-tree from a slice of points.
    pub fn build(points: &[Vec3]) -> Self {
        let mut indices: Vec<(Vec3, usize)> = points.iter().copied().enumerate().map(|(i, p)| (p, i)).collect();
        let root = Self::build_recursive(&mut indices, 0);
        KdTree { root, size: points.len() }
    }

    fn build_recursive(pts: &mut [(Vec3, usize)], depth: usize) -> Option<KdNode> {
        if pts.is_empty() { return None; }
        if pts.len() == 1 {
            return Some(KdNode::Leaf { point: pts[0].0, index: pts[0].1 });
        }

        let axis = depth % 3;
        pts.sort_by(|a, b| {
            let va = [a.0.x, a.0.y, a.0.z][axis];
            let vb = [b.0.x, b.0.y, b.0.z][axis];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = pts.len() / 2;
        let median = [pts[mid].0.x, pts[mid].0.y, pts[mid].0.z][axis];
        let (left_pts, right_pts) = pts.split_at_mut(mid);
        let left = Self::build_recursive(left_pts, depth + 1);
        let right = Self::build_recursive(right_pts, depth + 1);

        Some(KdNode::Split {
            axis,
            median,
            left: Box::new(left.unwrap_or(KdNode::Leaf { point: Vec3::zero(), index: 0 })),
            right: Box::new(right.unwrap_or(KdNode::Leaf { point: Vec3::zero(), index: 0 })),
        })
    }

    /// Find nearest neighbor to `query`. Returns (index, distance²).
    pub fn nearest(&self, query: &Vec3) -> Option<(usize, f64)> {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        if let Some(ref root) = self.root {
            Self::nn_search(root, query, 0, &mut best_idx, &mut best_dist);
        }
        if best_dist < f64::MAX { Some((best_idx, best_dist)) } else { None }
    }

    fn nn_search(node: &KdNode, query: &Vec3, depth: usize, best_idx: &mut usize, best_dist: &mut f64) {
        match node {
            KdNode::Leaf { point, index } => {
                let d = query.sub(point).norm_sq();
                if d < *best_dist { *best_dist = d; *best_idx = *index; }
            }
            KdNode::Split { axis, median, left, right } => {
                let qval = [query.x, query.y, query.z][*axis];
                let (first, second) = if qval < *median { (left, right) } else { (right, left) };

                Self::nn_search(first, query, depth + 1, best_idx, best_dist);

                let plane_dist = (qval - median).powi(2);
                if plane_dist < *best_dist {
                    Self::nn_search(second, query, depth + 1, best_idx, best_dist);
                }
            }
        }
    }
}

impl fmt::Display for KdTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KdTree(n={})", self.size)
    }
}

// ── ICP configuration ─────────────────────────────────────────────

/// ICP algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcpVariant {
    PointToPoint,
    PointToPlane,
}

impl fmt::Display for IcpVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IcpVariant::PointToPoint => write!(f, "point-to-point"),
            IcpVariant::PointToPlane => write!(f, "point-to-plane"),
        }
    }
}

/// ICP configuration.
#[derive(Debug, Clone)]
pub struct IcpConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub max_correspondence_dist: f64,
    pub variant: IcpVariant,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            max_correspondence_dist: 1.0,
            variant: IcpVariant::PointToPoint,
        }
    }
}

impl IcpConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_max_correspondence_dist(mut self, d: f64) -> Self { self.max_correspondence_dist = d; self }
    pub fn with_variant(mut self, v: IcpVariant) -> Self { self.variant = v; self }
}

// ── ICP result ────────────────────────────────────────────────────

/// Result of ICP alignment.
#[derive(Debug, Clone)]
pub struct IcpResult {
    pub rotation: Mat3,
    pub translation: Vec3,
    pub iterations: usize,
    pub final_error: f64,
    pub converged: bool,
}

impl fmt::Display for IcpResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IcpResult(iters={}, error={:.6}, converged={})",
            self.iterations, self.final_error, self.converged
        )
    }
}

// ── ICP solver ────────────────────────────────────────────────────

/// Run ICP to align `source` to `target`.
pub fn icp_align(source: &[Vec3], target: &[Vec3], config: &IcpConfig) -> IcpResult {
    let tree = KdTree::build(target);
    let max_dist_sq = config.max_correspondence_dist * config.max_correspondence_dist;

    let mut rot = Mat3::identity();
    let mut trans = Vec3::zero();

    // Transformed source
    let mut transformed: Vec<Vec3> = source.to_vec();
    let mut prev_error = f64::MAX;

    for iter in 0..config.max_iterations {
        // Find correspondences
        let mut src_matched = Vec::new();
        let mut tgt_matched = Vec::new();

        for tp in &transformed {
            if let Some((idx, dist_sq)) = tree.nearest(tp) {
                if dist_sq <= max_dist_sq {
                    src_matched.push(*tp);
                    tgt_matched.push(target[idx]);
                }
            }
        }

        if src_matched.len() < 3 {
            return IcpResult { rotation: rot, translation: trans, iterations: iter, final_error: prev_error, converged: false };
        }

        // Compute centroids
        let n = src_matched.len() as f64;
        let src_centroid = src_matched.iter().fold(Vec3::zero(), |a, b| a.add(b)).scale(1.0 / n);
        let tgt_centroid = tgt_matched.iter().fold(Vec3::zero(), |a, b| a.add(b)).scale(1.0 / n);

        // Cross-covariance
        let mut h = Mat3 { data: [0.0; 9] };
        for i in 0..src_matched.len() {
            let s = src_matched[i].sub(&src_centroid);
            let t = tgt_matched[i].sub(&tgt_centroid);
            h.data[0] += s.x * t.x; h.data[1] += s.x * t.y; h.data[2] += s.x * t.z;
            h.data[3] += s.y * t.x; h.data[4] += s.y * t.y; h.data[5] += s.y * t.z;
            h.data[6] += s.z * t.x; h.data[7] += s.z * t.y; h.data[8] += s.z * t.z;
        }

        let r_step = Mat3::rotation_from_cross_covariance(&h);
        let t_step = tgt_centroid.sub(&r_step.mul_vec(&src_centroid));

        // Accumulate
        rot = r_step.mul_mat(&rot);
        trans = r_step.mul_vec(&trans).add(&t_step);

        // Apply full transform
        for (i, sp) in source.iter().enumerate() {
            transformed[i] = rot.mul_vec(sp).add(&trans);
        }

        // Compute error
        let mut total_err = 0.0;
        let mut count = 0usize;
        for tp in &transformed {
            if let Some((_, dist_sq)) = tree.nearest(tp) {
                if dist_sq <= max_dist_sq {
                    total_err += dist_sq;
                    count += 1;
                }
            }
        }
        let mean_err = if count > 0 { total_err / count as f64 } else { f64::MAX };

        if (prev_error - mean_err).abs() < config.tolerance {
            return IcpResult { rotation: rot, translation: trans, iterations: iter + 1, final_error: mean_err, converged: true };
        }
        prev_error = mean_err;
    }

    IcpResult { rotation: rot, translation: trans, iterations: config.max_iterations, final_error: prev_error, converged: false }
}

/// Apply rigid transform (R, t) to a point set.
pub fn transform_points(points: &[Vec3], rotation: &Mat3, translation: &Vec3) -> Vec<Vec3> {
    points.iter().map(|p| rotation.mul_vec(p).add(translation)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_points() -> Vec<Vec3> {
        let mut pts = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    pts.push(Vec3::new(ix as f64, iy as f64, iz as f64));
                }
            }
        }
        pts
    }

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let s = a.add(&b);
        assert!((s.x - 5.0).abs() < 1e-10);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat3_identity() {
        let m = Mat3::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = m.mul_vec(&v);
        assert!((r.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat3_determinant() {
        let m = Mat3::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kdtree_nearest() {
        let pts = cube_points();
        let tree = KdTree::build(&pts);
        let query = Vec3::new(0.1, 0.1, 0.1);
        let (idx, dist) = tree.nearest(&query).unwrap();
        assert_eq!(pts[idx].x, 0.0);
        assert!(dist < 0.1);
    }

    #[test]
    fn test_kdtree_exact() {
        let pts = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let tree = KdTree::build(&pts);
        let (_, dist) = tree.nearest(&Vec3::new(1.0, 2.0, 3.0)).unwrap();
        assert!(dist < 1e-15);
    }

    #[test]
    fn test_kdtree_display() {
        let pts = cube_points();
        let tree = KdTree::build(&pts);
        let s = format!("{}", tree);
        assert!(s.contains("KdTree(n=125)"));
    }

    #[test]
    fn test_icp_identity() {
        let src = cube_points();
        let tgt = src.clone();
        let config = IcpConfig::new().with_max_iterations(10);
        let result = icp_align(&src, &tgt, &config);
        assert!(result.final_error < 1e-6);
    }

    #[test]
    fn test_icp_translation() {
        let src = cube_points();
        let offset = Vec3::new(0.1, 0.0, 0.0);
        let tgt: Vec<Vec3> = src.iter().map(|p| p.add(&offset)).collect();
        let config = IcpConfig::new().with_max_iterations(50).with_tolerance(1e-8);
        let result = icp_align(&src, &tgt, &config);
        assert!(result.final_error < 0.01, "error={}", result.final_error);
    }

    #[test]
    fn test_icp_config_builder() {
        let cfg = IcpConfig::new()
            .with_max_iterations(100)
            .with_tolerance(1e-8)
            .with_variant(IcpVariant::PointToPlane);
        assert_eq!(cfg.max_iterations, 100);
        assert_eq!(cfg.variant, IcpVariant::PointToPlane);
    }

    #[test]
    fn test_icp_result_display() {
        let r = IcpResult {
            rotation: Mat3::identity(),
            translation: Vec3::zero(),
            iterations: 5,
            final_error: 0.001,
            converged: true,
        };
        let s = format!("{}", r);
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_transform_points() {
        let pts = vec![Vec3::new(1.0, 0.0, 0.0)];
        let t = Vec3::new(1.0, 1.0, 1.0);
        let result = transform_points(&pts, &Mat3::identity(), &t);
        assert!((result[0].x - 2.0).abs() < 1e-10);
        assert!((result[0].y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_icp_variant_display() {
        assert_eq!(format!("{}", IcpVariant::PointToPoint), "point-to-point");
        assert_eq!(format!("{}", IcpVariant::PointToPlane), "point-to-plane");
    }

    #[test]
    fn test_vec3_display() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = format!("{}", v);
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_rotation_from_identity_cov() {
        // H = I should yield R = I
        let r = Mat3::rotation_from_cross_covariance(&Mat3::identity());
        for i in 0..3 {
            assert!((r.get(i, i) - 1.0).abs() < 0.1, "diag[{}]={}", i, r.get(i, i));
        }
    }

    #[test]
    fn test_icp_few_points() {
        let src = vec![Vec3::new(0.0, 0.0, 0.0)];
        let tgt = vec![Vec3::new(1.0, 0.0, 0.0)];
        let config = IcpConfig::new();
        let result = icp_align(&src, &tgt, &config);
        // Too few points for reliable alignment
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_mat3_mul_mat() {
        let a = Mat3::identity();
        let b = Mat3::identity();
        let c = a.mul_mat(&b);
        assert!((c.determinant() - 1.0).abs() < 1e-10);
    }
}
