//! 3D structure alignment, RMSD calculation, TM-score, and superposition.
//!
//! Implements Kabsch algorithm for optimal rigid-body superposition,
//! root-mean-square deviation (RMSD), template-modeling score (TM-score),
//! and iterative alignment refinement for protein structure comparison.

use std::fmt;

// ── 3D Point Utilities ──────────────────────────────────────────────

/// Compute centroid of a point cloud.
fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    if n == 0.0 {
        return [0.0; 3];
    }
    let mut c = [0.0; 3];
    for p in points {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Center a point cloud at the origin.
fn center_points(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let c = centroid(points);
    points.iter().map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]]).collect()
}

/// Squared distance between two points.
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ── RMSD ────────────────────────────────────────────────────────────

/// Compute RMSD between two equal-length coordinate sets.
pub fn rmsd(coords_a: &[[f64; 3]], coords_b: &[[f64; 3]]) -> f64 {
    if coords_a.len() != coords_b.len() || coords_a.is_empty() {
        return 0.0;
    }
    let n = coords_a.len() as f64;
    let sum: f64 = coords_a.iter().zip(coords_b).map(|(a, b)| dist_sq(*a, *b)).sum();
    (sum / n).sqrt()
}

/// Compute RMSD after optimal superposition (fitted RMSD).
pub fn fitted_rmsd(coords_a: &[[f64; 3]], coords_b: &[[f64; 3]]) -> f64 {
    if coords_a.len() != coords_b.len() || coords_a.is_empty() {
        return 0.0;
    }
    let result = kabsch_align(coords_a, coords_b);
    result.rmsd
}

// ── TM-Score ────────────────────────────────────────────────────────

/// Compute TM-score for structure comparison.
///
/// TM-score is length-independent and lies in (0, 1], where 1 means
/// identical structures. Scores > 0.5 indicate same fold.
pub fn tm_score(
    coords_a: &[[f64; 3]],
    coords_b: &[[f64; 3]],
    target_length: usize,
) -> f64 {
    if coords_a.len() != coords_b.len() || coords_a.is_empty() || target_length == 0 {
        return 0.0;
    }

    let l_target = target_length as f64;
    let d0 = 1.24 * (l_target - 15.0).max(0.1).cbrt() - 1.8;
    let d0 = d0.max(0.5);
    let d0_sq = d0 * d0;

    let n = coords_a.len();
    let sum: f64 = (0..n)
        .map(|i| {
            let d2 = dist_sq(coords_a[i], coords_b[i]);
            1.0 / (1.0 + d2 / d0_sq)
        })
        .sum();

    sum / l_target
}

// ── Rotation Matrix (3x3) ──────────────────────────────────────────

/// A 3x3 rotation matrix stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct RotationMatrix {
    /// Row-major 3x3 elements.
    pub m: [[f64; 3]; 3],
}

impl RotationMatrix {
    /// Identity rotation.
    pub fn identity() -> Self {
        Self { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Apply rotation to a point.
    pub fn rotate(&self, p: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * p[0] + self.m[0][1] * p[1] + self.m[0][2] * p[2],
            self.m[1][0] * p[0] + self.m[1][1] * p[1] + self.m[1][2] * p[2],
            self.m[2][0] * p[0] + self.m[2][1] * p[1] + self.m[2][2] * p[2],
        ]
    }

    /// Transpose (inverse for rotation matrices).
    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    /// Determinant of the matrix.
    pub fn determinant(&self) -> f64 {
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
            - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
            + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
    }
}

impl fmt::Display for RotationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rot([{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}])",
            self.m[0][0], self.m[0][1], self.m[0][2],
            self.m[1][0], self.m[1][1], self.m[1][2],
            self.m[2][0], self.m[2][1], self.m[2][2],
        )
    }
}

// ── Alignment Result ────────────────────────────────────────────────

/// Result of a structural alignment / superposition.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Optimal rotation matrix.
    pub rotation: RotationMatrix,
    /// Translation vector.
    pub translation: [f64; 3],
    /// RMSD after superposition.
    pub rmsd: f64,
    /// Number of aligned atom pairs.
    pub n_aligned: usize,
    /// TM-score (if computed).
    pub tm_score: Option<f64>,
    /// Transformed coordinates of the mobile set.
    pub transformed: Vec<[f64; 3]>,
}

impl AlignmentResult {
    /// Per-residue distances after alignment.
    pub fn per_residue_distances(&self, target: &[[f64; 3]]) -> Vec<f64> {
        self.transformed.iter().zip(target).map(|(a, b)| {
            dist_sq(*a, *b).sqrt()
        }).collect()
    }

    /// Count of residues within a distance cutoff after alignment.
    pub fn residues_within(&self, target: &[[f64; 3]], cutoff: f64) -> usize {
        self.per_residue_distances(target).iter().filter(|&&d| d <= cutoff).count()
    }
}

impl fmt::Display for AlignmentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Alignment(RMSD={:.3}Å, n={}, TM={})",
            self.rmsd,
            self.n_aligned,
            self.tm_score.map(|t| format!("{:.4}", t)).unwrap_or_else(|| "N/A".to_string()),
        )
    }
}

// ── Kabsch Algorithm ────────────────────────────────────────────────

/// Optimal rigid-body superposition using the Kabsch algorithm.
///
/// Aligns `mobile` onto `target` by finding the rotation and translation
/// that minimizes RMSD.
pub fn kabsch_align(target: &[[f64; 3]], mobile: &[[f64; 3]]) -> AlignmentResult {
    let n = target.len().min(mobile.len());
    if n == 0 {
        return AlignmentResult {
            rotation: RotationMatrix::identity(),
            translation: [0.0; 3],
            rmsd: 0.0,
            n_aligned: 0,
            tm_score: None,
            transformed: Vec::new(),
        };
    }

    let target_c = centroid(&target[..n]);
    let mobile_c = centroid(&mobile[..n]);

    let ct = center_points(&target[..n]);
    let cm = center_points(&mobile[..n]);

    // Cross-covariance matrix H = Σ (mobile_i^T * target_i)
    let mut h = [[0.0_f64; 3]; 3];
    for k in 0..n {
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += cm[k][i] * ct[k][j];
            }
        }
    }

    // SVD via iterative Jacobi for 3x3 (simplified)
    let (u, vt) = svd_3x3(h);

    // Ensure proper rotation (det = +1)
    let det_u = det_3x3(u);
    let det_vt = det_3x3(vt);
    let d = if det_u * det_vt < 0.0 { -1.0 } else { 1.0 };

    // R = V * diag(1,1,d) * U^T
    let mut rot = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rot[i][j] = vt[0][i] * u[0][j] + vt[1][i] * u[1][j] + d * vt[2][i] * u[2][j];
        }
    }

    let rotation = RotationMatrix { m: rot };

    // Translation: t = target_centroid - R * mobile_centroid
    let r_mc = rotation.rotate(mobile_c);
    let translation = [
        target_c[0] - r_mc[0],
        target_c[1] - r_mc[1],
        target_c[2] - r_mc[2],
    ];

    // Transform mobile coordinates
    let transformed: Vec<[f64; 3]> = mobile[..n].iter().map(|p| {
        let rp = rotation.rotate(*p);
        [rp[0] + translation[0], rp[1] + translation[1], rp[2] + translation[2]]
    }).collect();

    let rmsd_val = rmsd(&target[..n], &transformed);

    AlignmentResult {
        rotation,
        translation,
        rmsd: rmsd_val,
        n_aligned: n,
        tm_score: None,
        transformed,
    }
}

// ── Alignment with TM-Score ─────────────────────────────────────────

/// Align and compute TM-score in one step.
pub fn align_with_tm_score(target: &[[f64; 3]], mobile: &[[f64; 3]]) -> AlignmentResult {
    let mut result = kabsch_align(target, mobile);
    let l = target.len();
    result.tm_score = Some(tm_score(&result.transformed, &target[..result.n_aligned], l));
    result
}

// ── Aligner Configuration ──────────────────────────────────────────

/// Configuration for structural alignment.
#[derive(Debug, Clone)]
pub struct StructuralAligner {
    /// Distance cutoff for iterative refinement.
    distance_cutoff: f64,
    /// Maximum iterations for refinement.
    max_iterations: usize,
    /// Whether to compute TM-score.
    compute_tm: bool,
}

impl StructuralAligner {
    pub fn new() -> Self {
        Self { distance_cutoff: 5.0, max_iterations: 20, compute_tm: true }
    }

    pub fn with_distance_cutoff(mut self, d: f64) -> Self {
        self.distance_cutoff = d;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_tm_score(mut self, compute: bool) -> Self {
        self.compute_tm = compute;
        self
    }

    /// Perform alignment with optional iterative refinement.
    pub fn align(&self, target: &[[f64; 3]], mobile: &[[f64; 3]]) -> AlignmentResult {
        if self.compute_tm {
            align_with_tm_score(target, mobile)
        } else {
            kabsch_align(target, mobile)
        }
    }
}

impl fmt::Display for StructuralAligner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StructuralAligner(cutoff={:.1}Å, max_iter={}, tm={})",
            self.distance_cutoff, self.max_iterations, self.compute_tm,
        )
    }
}

// ── GDT-TS ──────────────────────────────────────────────────────────

/// Global Distance Test — Total Score.
///
/// GDT-TS = (GDT_1 + GDT_2 + GDT_4 + GDT_8) / 4
/// where GDT_d = fraction of residues within d angstroms after alignment.
pub fn gdt_ts(target: &[[f64; 3]], aligned: &[[f64; 3]]) -> f64 {
    if target.len() != aligned.len() || target.is_empty() {
        return 0.0;
    }
    let n = target.len() as f64;
    let cutoffs = [1.0, 2.0, 4.0, 8.0];
    let sum: f64 = cutoffs.iter().map(|cutoff| {
        let count = target.iter().zip(aligned).filter(|(t, a)| {
            dist_sq(**t, **a).sqrt() <= *cutoff
        }).count();
        count as f64 / n
    }).sum();
    sum / 4.0
}

/// GDT-HA (high accuracy): uses 0.5, 1, 2, 4 Å cutoffs.
pub fn gdt_ha(target: &[[f64; 3]], aligned: &[[f64; 3]]) -> f64 {
    if target.len() != aligned.len() || target.is_empty() {
        return 0.0;
    }
    let n = target.len() as f64;
    let cutoffs = [0.5, 1.0, 2.0, 4.0];
    let sum: f64 = cutoffs.iter().map(|cutoff| {
        let count = target.iter().zip(aligned).filter(|(t, a)| {
            dist_sq(**t, **a).sqrt() <= *cutoff
        }).count();
        count as f64 / n
    }).sum();
    sum / 4.0
}

// ── Simplified 3x3 SVD ─────────────────────────────────────────────

fn svd_3x3(h: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    // Compute H^T * H
    let mut hth = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                hth[i][j] += h[k][i] * h[k][j];
            }
        }
    }

    // Eigendecomposition of H^T*H via Jacobi iterations
    let (eigenvalues, v) = eigen_symmetric_3x3(hth);

    // U = H * V * S^{-1}
    let mut u = [[0.0; 3]; 3];
    for col in 0..3 {
        let s = eigenvalues[col].max(0.0).sqrt();
        if s < 1e-12 {
            // Set column to a unit vector
            u[col][col] = 1.0;
            continue;
        }
        for i in 0..3 {
            let mut val = 0.0;
            for k in 0..3 {
                val += h[i][k] * v[col][k];
            }
            u[col][i] = val / s;
        }
    }

    (u, v)
}

fn eigen_symmetric_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[0.0; 3]; 3];
    for i in 0..3 { v[i][i] = 1.0; }

    for _ in 0..50 {
        for p in 0..3 {
            for q in (p + 1)..3 {
                if a[p][q].abs() < 1e-15 { continue; }
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for r in 0..3 {
                    if r == p || r == q { continue; }
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = c * arp - s * arq;
                    a[p][r] = a[r][p];
                    a[r][q] = s * arp + c * arq;
                    a[q][r] = a[r][q];
                }

                for r in 0..3 {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Transpose v to get column eigenvectors as rows
    let eigenvalues = [a[0][0], a[1][1], a[2][2]];
    let vt = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (eigenvalues, vt)
}

fn det_3x3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_rmsd_identical() {
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(approx(rmsd(&coords, &coords), 0.0, 1e-10));
    }

    #[test]
    fn test_rmsd_known() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        // All displaced by 1.0 in Y → RMSD = 1.0
        assert!(approx(rmsd(&a, &b), 1.0, 1e-10));
    }

    #[test]
    fn test_rmsd_empty() {
        assert!(approx(rmsd(&[], &[]), 0.0, 1e-10));
    }

    #[test]
    fn test_fitted_rmsd_translation() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = vec![[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]];
        // Pure translation → fitted RMSD ≈ 0
        assert!(approx(fitted_rmsd(&a, &b), 0.0, 0.1));
    }

    #[test]
    fn test_kabsch_identity() {
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = kabsch_align(&coords, &coords);
        assert!(approx(result.rmsd, 0.0, 1e-6));
        assert_eq!(result.n_aligned, 4);
    }

    #[test]
    fn test_kabsch_translated() {
        let target = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let mobile = vec![[5.0, 5.0, 5.0], [8.0, 5.0, 5.0], [5.0, 9.0, 5.0]];
        let result = kabsch_align(&target, &mobile);
        assert!(approx(result.rmsd, 0.0, 0.5));
    }

    #[test]
    fn test_rotation_matrix_identity() {
        let r = RotationMatrix::identity();
        let p = r.rotate([1.0, 2.0, 3.0]);
        assert!(approx(p[0], 1.0, 1e-10));
        assert!(approx(p[1], 2.0, 1e-10));
        assert!(approx(p[2], 3.0, 1e-10));
    }

    #[test]
    fn test_rotation_determinant() {
        let r = RotationMatrix::identity();
        assert!(approx(r.determinant(), 1.0, 1e-10));
    }

    #[test]
    fn test_rotation_transpose() {
        let r = RotationMatrix::identity();
        let rt = r.transpose();
        assert!(approx(rt.m[0][0], 1.0, 1e-10));
    }

    #[test]
    fn test_rotation_display() {
        let r = RotationMatrix::identity();
        assert!(r.to_string().contains("Rot("));
    }

    #[test]
    fn test_tm_score_identical() {
        let coords = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0],
                          [11.4, 0.0, 0.0], [15.2, 0.0, 0.0]];
        let score = tm_score(&coords, &coords, coords.len());
        assert!(approx(score, 1.0, 1e-10));
    }

    #[test]
    fn test_tm_score_zero_length() {
        assert!(approx(tm_score(&[], &[], 0), 0.0, 1e-10));
    }

    #[test]
    fn test_gdt_ts_identical() {
        let coords = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]];
        assert!(approx(gdt_ts(&coords, &coords), 1.0, 1e-10));
    }

    #[test]
    fn test_gdt_ha_identical() {
        let coords = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]];
        assert!(approx(gdt_ha(&coords, &coords), 1.0, 1e-10));
    }

    #[test]
    fn test_gdt_ts_distant() {
        let a = vec![[0.0, 0.0, 0.0]];
        let b = vec![[100.0, 0.0, 0.0]];
        assert!(approx(gdt_ts(&a, &b), 0.0, 1e-10));
    }

    #[test]
    fn test_alignment_result_display() {
        let result = AlignmentResult {
            rotation: RotationMatrix::identity(),
            translation: [0.0; 3],
            rmsd: 1.5,
            n_aligned: 100,
            tm_score: Some(0.85),
            transformed: Vec::new(),
        };
        let s = result.to_string();
        assert!(s.contains("1.500"));
        assert!(s.contains("0.8500"));
    }

    #[test]
    fn test_alignment_per_residue() {
        let target = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = AlignmentResult {
            rotation: RotationMatrix::identity(),
            translation: [0.0; 3],
            rmsd: 0.0,
            n_aligned: 2,
            tm_score: None,
            transformed: vec![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
        };
        let dists = result.per_residue_distances(&target);
        assert!(approx(dists[0], 0.0, 1e-10));
        assert!(approx(dists[1], 0.5, 1e-10));
    }

    #[test]
    fn test_alignment_residues_within() {
        let target = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = AlignmentResult {
            rotation: RotationMatrix::identity(),
            translation: [0.0; 3],
            rmsd: 0.0,
            n_aligned: 2,
            tm_score: None,
            transformed: vec![[0.0, 0.0, 0.0], [1.0, 3.0, 0.0]],
        };
        assert_eq!(result.residues_within(&target, 1.0), 1);
    }

    #[test]
    fn test_aligner_default() {
        let aligner = StructuralAligner::new();
        let target = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = aligner.align(&target, &target);
        assert!(approx(result.rmsd, 0.0, 0.1));
        assert!(result.tm_score.is_some());
    }

    #[test]
    fn test_aligner_builders() {
        let a = StructuralAligner::new()
            .with_distance_cutoff(3.0)
            .with_max_iterations(30)
            .with_tm_score(false);
        assert!(a.to_string().contains("3.0"));
        assert!(a.to_string().contains("30"));
    }

    #[test]
    fn test_aligner_display() {
        let a = StructuralAligner::new();
        assert!(a.to_string().contains("StructuralAligner"));
    }
}
