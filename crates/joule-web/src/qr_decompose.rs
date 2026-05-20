//! QR decomposition — Householder reflections and Gram-Schmidt.
//!
//! Solve Ax = b, least-squares for overdetermined systems, rank estimation,
//! column pivoting for rank-revealing QR, and thin (economy) QR.

use std::fmt;

// ── Dense matrix ──────────────────────────────────────────────

/// Row-major dense matrix for QR operations.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn col_vec(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Matrix multiply.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut c = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols {
                    s += self.get(i, k) * other.get(k, j);
                }
                c.set(i, j, s);
            }
        }
        c
    }
}

impl fmt::Display for DenseMat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:10.6}", self.get(i, j))?;
            }
            if i + 1 < self.rows {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

// ── QR result ─────────────────────────────────────────────────

/// Result of QR decomposition: A = Q * R.
#[derive(Debug, Clone, PartialEq)]
pub struct QrDecomposition {
    /// Orthogonal matrix Q (m x m for full, m x n for thin when m > n).
    pub q: DenseMat,
    /// Upper triangular matrix R (m x n for full, n x n for thin when m > n).
    pub r: DenseMat,
}

/// Result of column-pivoted QR: A * P = Q * R.
#[derive(Debug, Clone, PartialEq)]
pub struct PivotedQr {
    pub q: DenseMat,
    pub r: DenseMat,
    /// Column permutation: column j of the result corresponds to column perm[j] of A.
    pub perm: Vec<usize>,
    /// Estimated rank (number of R diagonal entries above threshold).
    pub rank: usize,
}

// ── Householder QR ────────────────────────────────────────────

/// Full QR decomposition via Householder reflections.
/// For an m x n matrix A (m >= n), returns Q (m x m) and R (m x n).
pub fn qr_householder(a: &DenseMat) -> QrDecomposition {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = DenseMat::identity(m);

    let k_max = m.min(n);
    for k in 0..k_max {
        // Extract sub-column from r[k..m, k].
        let mut col = Vec::with_capacity(m - k);
        for i in k..m {
            col.push(r.get(i, k));
        }
        let norm_col: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_col < 1e-15 {
            continue;
        }

        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = col.clone();
        v[0] += sign * norm_col;
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        // Apply H = I - 2vv^T/||v||^2 to R from the left.
        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..(m - k) {
                dot_val += v[i] * r.get(i + k, j);
            }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for i in 0..(m - k) {
                let old = r.get(i + k, j);
                r.set(i + k, j, old - coeff * v[i]);
            }
        }

        // Apply H to Q from the right: Q = Q * H.
        for i in 0..m {
            let mut dot_val = 0.0;
            for jj in 0..(m - k) {
                dot_val += q.get(i, jj + k) * v[jj];
            }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for jj in 0..(m - k) {
                let old = q.get(i, jj + k);
                q.set(i, jj + k, old - coeff * v[jj]);
            }
        }
    }

    QrDecomposition { q, r }
}

/// Thin (economy) QR: for m x n with m > n, returns Q (m x n) and R (n x n).
pub fn qr_thin(a: &DenseMat) -> QrDecomposition {
    let m = a.rows;
    let n = a.cols;
    let full = qr_householder(a);
    if m <= n {
        return full;
    }
    // Slice Q to m x n.
    let mut q_thin = DenseMat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            q_thin.set(i, j, full.q.get(i, j));
        }
    }
    // Slice R to n x n.
    let mut r_thin = DenseMat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            r_thin.set(i, j, full.r.get(i, j));
        }
    }
    QrDecomposition {
        q: q_thin,
        r: r_thin,
    }
}

// ── Gram-Schmidt ──────────────────────────────────────────────

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Classical Gram-Schmidt QR.  Less numerically stable than Modified or Householder.
pub fn qr_gram_schmidt_classical(a: &DenseMat) -> QrDecomposition {
    let m = a.rows;
    let n = a.cols;
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r = DenseMat::zeros(n, n);

    for j in 0..n {
        let mut v = a.col_vec(j);
        for i in 0..j {
            let rij = vec_dot(&q_cols[i], &a.col_vec(j));
            r.set(i, j, rij);
            for k in 0..m {
                v[k] -= rij * q_cols[i][k];
            }
        }
        let norm = vec_norm(&v);
        r.set(j, j, norm);
        if norm > 1e-15 {
            for k in 0..m {
                v[k] /= norm;
            }
        }
        q_cols.push(v);
    }

    let mut q = DenseMat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            q.set(i, j, q_cols[j][i]);
        }
    }
    QrDecomposition { q, r }
}

/// Modified Gram-Schmidt QR.  More numerically stable than classical.
pub fn qr_gram_schmidt_modified(a: &DenseMat) -> QrDecomposition {
    let m = a.rows;
    let n = a.cols;
    let mut q_cols: Vec<Vec<f64>> = (0..n).map(|j| a.col_vec(j)).collect();
    let mut r = DenseMat::zeros(n, n);

    for i in 0..n {
        let norm = vec_norm(&q_cols[i]);
        r.set(i, i, norm);
        if norm < 1e-15 {
            continue;
        }
        for k in 0..m {
            q_cols[i][k] /= norm;
        }
        for j in (i + 1)..n {
            let rij = vec_dot(&q_cols[i], &q_cols[j]);
            r.set(i, j, rij);
            for k in 0..m {
                q_cols[j][k] -= rij * q_cols[i][k];
            }
        }
    }

    let mut q = DenseMat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            q.set(i, j, q_cols[j][i]);
        }
    }
    QrDecomposition { q, r }
}

// ── Column-pivoted QR ─────────────────────────────────────────

/// Rank-revealing QR with column pivoting.
pub fn qr_column_pivoted(a: &DenseMat, rank_tol: f64) -> PivotedQr {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = DenseMat::identity(m);
    let mut perm: Vec<usize> = (0..n).collect();

    // Column norms for pivoting.
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| r.get(i, j).powi(2)).sum::<f64>())
        .collect();

    let k_max = m.min(n);
    let mut rank = 0;
    for k in 0..k_max {
        // Find column with largest remaining norm.
        let mut best = k;
        let mut best_norm = col_norms[k];
        for j in (k + 1)..n {
            if col_norms[j] > best_norm {
                best_norm = col_norms[j];
                best = j;
            }
        }

        if best_norm.sqrt() < rank_tol {
            break;
        }

        // Swap columns k and best.
        if best != k {
            perm.swap(k, best);
            col_norms.swap(k, best);
            for i in 0..m {
                let tmp = r.get(i, k);
                r.set(i, k, r.get(i, best));
                r.set(i, best, tmp);
            }
        }

        // Householder on column k.
        let mut col: Vec<f64> = (k..m).map(|i| r.get(i, k)).collect();
        let norm_col: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_col < 1e-15 {
            break;
        }
        rank += 1;

        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        col[0] += sign * norm_col;
        let v_norm_sq: f64 = col.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..col.len() {
                dot_val += col[i] * r.get(i + k, j);
            }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for i in 0..col.len() {
                let old = r.get(i + k, j);
                r.set(i + k, j, old - coeff * col[i]);
            }
        }
        for i in 0..m {
            let mut dot_val = 0.0;
            for jj in 0..col.len() {
                dot_val += q.get(i, jj + k) * col[jj];
            }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for jj in 0..col.len() {
                let old = q.get(i, jj + k);
                q.set(i, jj + k, old - coeff * col[jj]);
            }
        }

        // Update remaining column norms.
        for j in (k + 1)..n {
            let rkj = r.get(k, j);
            col_norms[j] -= rkj * rkj;
            if col_norms[j] < 0.0 {
                col_norms[j] = 0.0;
            }
        }
    }

    PivotedQr {
        q,
        r,
        perm,
        rank,
    }
}

// ── Rank estimation ───────────────────────────────────────────

/// Estimate rank from QR decomposition by counting R diagonal entries above threshold.
pub fn estimate_rank(r: &DenseMat, tol: f64) -> usize {
    let k = r.rows.min(r.cols);
    let mut rank = 0;
    for i in 0..k {
        if r.get(i, i).abs() > tol {
            rank += 1;
        } else {
            break;
        }
    }
    rank
}

// ── Solve via QR ──────────────────────────────────────────────

/// Solve Ax = b via QR for a square system.
pub fn qr_solve(a: &DenseMat, b: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(a.rows, a.cols);
    let n = a.rows;
    let qr = qr_householder(a);
    // x = R^{-1} Q^T b
    let qt = qr.q.transpose();
    let mut qtb = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            qtb[i] += qt.get(i, j) * b[j];
        }
    }
    // Back-substitution on R.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = qr.r.get(i, i);
        if diag.abs() < 1e-14 {
            return None;
        }
        let mut s = qtb[i];
        for j in (i + 1)..n {
            s -= qr.r.get(i, j) * x[j];
        }
        x[i] = s / diag;
    }
    Some(x)
}

/// Least-squares solve: minimize ||Ax - b||^2 for overdetermined A (m > n).
pub fn qr_least_squares(a: &DenseMat, b: &[f64]) -> Vec<f64> {
    let m = a.rows;
    let n = a.cols;
    assert!(m >= n, "least squares requires m >= n");
    let qr = qr_thin(a);
    // x = R^{-1} Q^T b   (Q is m x n, R is n x n)
    let mut qtb = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            qtb[i] += qr.q.get(j, i) * b[j];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = qr.r.get(i, i);
        if diag.abs() < 1e-14 {
            continue;
        }
        let mut s = qtb[i];
        for j in (i + 1)..n {
            s -= qr.r.get(i, j) * x[j];
        }
        x[i] = s / diag;
    }
    x
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn mat_approx_eq(a: &DenseMat, b: &DenseMat, eps: f64) -> bool {
        a.rows == b.rows
            && a.cols == b.cols
            && a.data
                .iter()
                .zip(b.data.iter())
                .all(|(x, y)| approx_eq(*x, *y, eps))
    }

    #[test]
    fn test_householder_2x2() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 1.0, 0.0, 1.0]);
        let qr = qr_householder(&a);
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &a, 1e-10));
    }

    #[test]
    fn test_householder_3x3() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_householder(&a);
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &a, 1e-10));
    }

    #[test]
    fn test_q_orthogonal() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_householder(&a);
        let qtq = qr.q.transpose().mul(&qr.q);
        let id = DenseMat::identity(3);
        assert!(mat_approx_eq(&qtq, &id, 1e-10));
    }

    #[test]
    fn test_r_upper_triangular() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_householder(&a);
        for i in 0..3 {
            for j in 0..i {
                assert!(approx_eq(qr.r.get(i, j), 0.0, 1e-10));
            }
        }
    }

    #[test]
    fn test_thin_qr_rectangular() {
        // 4x2 matrix
        let a = DenseMat::from_data(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let qr = qr_thin(&a);
        assert_eq!(qr.q.rows, 4);
        assert_eq!(qr.q.cols, 2);
        assert_eq!(qr.r.rows, 2);
        assert_eq!(qr.r.cols, 2);
        // Q^T Q = I_2
        let qtq = qr.q.transpose().mul(&qr.q);
        let id2 = DenseMat::identity(2);
        assert!(mat_approx_eq(&qtq, &id2, 1e-10));
        // Q*R = A
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &a, 1e-10));
    }

    #[test]
    fn test_gram_schmidt_classical() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let qr = qr_gram_schmidt_classical(&a);
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &a, 1e-10));
    }

    #[test]
    fn test_gram_schmidt_modified() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let qr = qr_gram_schmidt_modified(&a);
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &a, 1e-10));
    }

    #[test]
    fn test_modified_gs_orthogonality() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_gram_schmidt_modified(&a);
        let qtq = qr.q.transpose().mul(&qr.q);
        let id = DenseMat::identity(3);
        assert!(mat_approx_eq(&qtq, &id, 1e-8));
    }

    #[test]
    fn test_qr_solve_2x2() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let b = vec![5.0, 7.0];
        let x = qr_solve(&a, &b).unwrap();
        // Verify
        let ax0 = 2.0 * x[0] + x[1];
        let ax1 = x[0] + 3.0 * x[1];
        assert!(approx_eq(ax0, 5.0, 1e-10));
        assert!(approx_eq(ax1, 7.0, 1e-10));
    }

    #[test]
    fn test_qr_solve_identity() {
        let id = DenseMat::identity(3);
        let b = vec![1.0, 2.0, 3.0];
        let x = qr_solve(&id, &b).unwrap();
        for i in 0..3 {
            assert!(approx_eq(x[i], b[i], 1e-10));
        }
    }

    #[test]
    fn test_least_squares_exact() {
        // 3x2, rank 2, exact solution exists
        // A = [[1,0],[0,1],[1,1]], b = [1,2,3]
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let b = vec![1.0, 2.0, 3.0];
        let x = qr_least_squares(&a, &b);
        // Verify least-squares: A^T A x = A^T b
        let ata = a.transpose().mul(&a);
        let atb_vec: Vec<f64> = {
            let at = a.transpose();
            (0..2).map(|i| (0..3).map(|j| at.get(i, j) * b[j]).sum()).collect()
        };
        for i in 0..2 {
            let mut s = 0.0;
            for j in 0..2 {
                s += ata.get(i, j) * x[j];
            }
            assert!(approx_eq(s, atb_vec[i], 1e-10));
        }
    }

    #[test]
    fn test_rank_estimation_full() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_householder(&a);
        let rank = estimate_rank(&qr.r, 1e-10);
        assert_eq!(rank, 3);
    }

    #[test]
    fn test_rank_estimation_deficient() {
        // rank 1: second row = 2 * first row
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let qr = qr_householder(&a);
        let rank = estimate_rank(&qr.r, 1e-10);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_column_pivoted_qr() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let pqr = qr_column_pivoted(&a, 1e-10);
        assert_eq!(pqr.rank, 3);
        assert_eq!(pqr.perm.len(), 3);
    }

    #[test]
    fn test_column_pivoted_rank_deficient() {
        // Row 2 = 2*Row 1 -> rank deficient
        let a = DenseMat::from_data(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let pqr = qr_column_pivoted(&a, 1e-10);
        assert_eq!(pqr.rank, 1);
    }

    #[test]
    fn test_qr_solve_singular_returns_none() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let result = qr_solve(&a, &[1.0, 2.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_householder_identity_input() {
        let id = DenseMat::identity(3);
        let qr = qr_householder(&id);
        // Q should be +/- identity, R should be +/- identity.
        let prod = qr.q.mul(&qr.r);
        assert!(mat_approx_eq(&prod, &id, 1e-10));
    }

    #[test]
    fn test_qr_3x2_least_squares_polynomial() {
        // Fit y = a + bx to points (0,1), (1,3), (2,5) -> exact fit y=1+2x
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let x = qr_least_squares(&a, &b);
        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 2.0, 1e-10));
    }

    #[test]
    fn test_display_format() {
        let m = DenseMat::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = format!("{}", m);
        assert!(s.contains("1.000000"));
    }
}
