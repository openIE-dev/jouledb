//! Singular Value Decomposition (SVD) — A = U Sigma V^T.
//!
//! One-sided Jacobi SVD for arbitrary matrices.  Singular values in descending
//! order, truncated SVD, low-rank approximation, pseudoinverse, condition
//! number, and matrix norms.

// ── Dense matrix helper ───────────────────────────────────────

/// Row-major dense matrix for SVD operations.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * self.cols + c] }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * self.cols + c] = v; }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { t.set(j, i, self.get(i, j)); } }
        t
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut c = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols { s += self.get(i, k) * other.get(k, j); }
                c.set(i, j, s);
            }
        }
        c
    }

    pub fn col_vec(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    fn col_norm(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| { let v = self.get(i, j); v * v }).sum::<f64>().sqrt()
    }

    fn col_dot(&self, j1: usize, j2: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j1) * self.get(i, j2)).sum()
    }
}

// ── SVD result ────────────────────────────────────────────────

/// Result of SVD: A = U * diag(sigma) * V^T.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdDecomposition {
    /// Left singular vectors (m x k).
    pub u: DenseMat,
    /// Singular values in descending order.
    pub sigma: Vec<f64>,
    /// Right singular vectors (n x k) — columns of V (NOT V^T).
    pub v: DenseMat,
}

// ── One-sided Jacobi SVD ──────────────────────────────────────

/// Compute the SVD of an m x n matrix via one-sided Jacobi rotations.
///
/// Returns all min(m, n) singular values and corresponding vectors.
pub fn svd(a: &DenseMat) -> SvdDecomposition {
    let m = a.rows;
    let n = a.cols;

    // Work on B = A (copy), accumulate V rotations.
    let mut b = a.clone();
    let mut v = DenseMat::identity(n);

    let max_iter = 100 * n * n;
    let tol = 1e-14;

    for _ in 0..max_iter {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let alpha = b.col_dot(p, p);
                let beta = b.col_dot(q, q);
                let gamma = b.col_dot(p, q);

                if gamma.abs() < tol * (alpha * beta).sqrt().max(1e-30) {
                    continue;
                }
                converged = false;

                // Compute Jacobi rotation angle.
                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to columns p, q of B.
                for i in 0..m {
                    let bp = b.get(i, p);
                    let bq = b.get(i, q);
                    b.set(i, p, c * bp - s * bq);
                    b.set(i, q, s * bp + c * bq);
                }

                // Accumulate V.
                for i in 0..n {
                    let vp = v.get(i, p);
                    let vq = v.get(i, q);
                    v.set(i, p, c * vp - s * vq);
                    v.set(i, q, s * vp + c * vq);
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values and normalize U columns.
    let k = m.min(n);
    let mut sigma = Vec::with_capacity(k);
    let mut u = DenseMat::zeros(m, k);

    for j in 0..k {
        let s = b.col_norm(j);
        sigma.push(s);
        if s > 1e-30 {
            for i in 0..m {
                u.set(i, j, b.get(i, j) / s);
            }
        }
    }

    // Sort by descending singular value.
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a_idx, &b_idx| sigma[b_idx].partial_cmp(&sigma[a_idx]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_sigma: Vec<f64> = indices.iter().map(|i| sigma[*i]).collect();
    let mut sorted_u = DenseMat::zeros(m, k);
    let mut sorted_v = DenseMat::zeros(n, k);
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..m {
            sorted_u.set(i, new_j, u.get(i, old_j));
        }
        for i in 0..n {
            sorted_v.set(i, new_j, v.get(i, old_j));
        }
    }

    SvdDecomposition {
        u: sorted_u,
        sigma: sorted_sigma,
        v: sorted_v,
    }
}

// ── Derived operations ────────────────────────────────────────

/// Truncated SVD: keep only the top k singular values.
pub fn svd_truncated(a: &DenseMat, k: usize) -> SvdDecomposition {
    let full = svd(a);
    let m = full.u.rows;
    let n = full.v.rows;
    let k = k.min(full.sigma.len());

    let mut u = DenseMat::zeros(m, k);
    let mut v = DenseMat::zeros(n, k);
    for j in 0..k {
        for i in 0..m { u.set(i, j, full.u.get(i, j)); }
        for i in 0..n { v.set(i, j, full.v.get(i, j)); }
    }
    SvdDecomposition {
        u,
        sigma: full.sigma[..k].to_vec(),
        v,
    }
}

/// Matrix rank from singular values (count above threshold).
pub fn matrix_rank(a: &DenseMat, tol: f64) -> usize {
    let dec = svd(a);
    dec.sigma.iter().filter(|&&s| s > tol).count()
}

/// Low-rank approximation (Eckart-Young): reconstruct from top k singular values.
pub fn low_rank_approx(a: &DenseMat, k: usize) -> DenseMat {
    let trunc = svd_truncated(a, k);
    let m = trunc.u.rows;
    let n = trunc.v.rows;
    let kk = trunc.sigma.len();
    // A_k = U_k * diag(sigma_k) * V_k^T
    let mut result = DenseMat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..kk {
                s += trunc.u.get(i, t) * trunc.sigma[t] * trunc.v.get(j, t);
            }
            result.set(i, j, s);
        }
    }
    result
}

/// Pseudoinverse via SVD: A^+ = V * diag(1/sigma) * U^T.
pub fn pseudoinverse(a: &DenseMat, tol: f64) -> DenseMat {
    let dec = svd(a);
    let m = a.rows;
    let n = a.cols;
    let k = dec.sigma.len();

    // A^+ is n x m.
    let mut pinv = DenseMat::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for t in 0..k {
                if dec.sigma[t] > tol {
                    s += dec.v.get(i, t) * (1.0 / dec.sigma[t]) * dec.u.get(j, t);
                }
            }
            pinv.set(i, j, s);
        }
    }
    pinv
}

/// Condition number: sigma_max / sigma_min.
pub fn condition_number(a: &DenseMat) -> f64 {
    let dec = svd(a);
    if dec.sigma.is_empty() {
        return f64::INFINITY;
    }
    let max_s = dec.sigma[0];
    let min_s = *dec.sigma.last().unwrap();
    if min_s.abs() < 1e-30 {
        return f64::INFINITY;
    }
    max_s / min_s
}

/// 2-norm (spectral norm) = largest singular value.
pub fn matrix_norm_2(a: &DenseMat) -> f64 {
    let dec = svd(a);
    if dec.sigma.is_empty() { 0.0 } else { dec.sigma[0] }
}

/// Frobenius norm = sqrt(sum of sigma^2).
pub fn frobenius_norm_svd(a: &DenseMat) -> f64 {
    let dec = svd(a);
    dec.sigma.iter().map(|s| s * s).sum::<f64>().sqrt()
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn mat_approx_eq(a: &DenseMat, b: &DenseMat, eps: f64) -> bool {
        a.rows == b.rows && a.cols == b.cols
            && a.data.iter().zip(b.data.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    /// Reconstruct A = U * diag(sigma) * V^T.
    fn reconstruct(dec: &SvdDecomposition, m: usize, n: usize) -> DenseMat {
        let k = dec.sigma.len();
        let mut a = DenseMat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for t in 0..k { s += dec.u.get(i, t) * dec.sigma[t] * dec.v.get(j, t); }
                a.set(i, j, s);
            }
        }
        a
    }

    #[test]
    fn test_svd_identity() {
        let id = DenseMat::identity(3);
        let dec = svd(&id);
        assert_eq!(dec.sigma.len(), 3);
        for s in &dec.sigma {
            assert!(approx_eq(*s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_svd_reconstruct_2x2() {
        let a = DenseMat::from_data(2, 2, vec![3.0, 2.0, 2.0, 3.0]);
        let dec = svd(&a);
        let recon = reconstruct(&dec, 2, 2);
        assert!(mat_approx_eq(&recon, &a, 1e-10));
    }

    #[test]
    fn test_svd_reconstruct_3x3() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ]);
        let dec = svd(&a);
        let recon = reconstruct(&dec, 3, 3);
        assert!(mat_approx_eq(&recon, &a, 1e-8));
    }

    #[test]
    fn test_svd_descending_order() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 0.0, 0.0, 5.0]);
        let dec = svd(&a);
        assert!(dec.sigma[0] >= dec.sigma[1] - 1e-10);
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let a = DenseMat::from_data(3, 3, vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let dec = svd(&a);
        assert!(approx_eq(dec.sigma[0], 5.0, 1e-10));
        assert!(approx_eq(dec.sigma[1], 3.0, 1e-10));
        assert!(approx_eq(dec.sigma[2], 1.0, 1e-10));
    }

    #[test]
    fn test_svd_rectangular() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let dec = svd(&a);
        assert_eq!(dec.sigma.len(), 2);
        let recon = reconstruct(&dec, 3, 2);
        assert!(mat_approx_eq(&recon, &a, 1e-10));
    }

    #[test]
    fn test_u_orthonormal_columns() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let dec = svd(&a);
        let utu = dec.u.transpose().mul(&dec.u);
        let id = DenseMat::identity(3);
        assert!(mat_approx_eq(&utu, &id, 1e-8));
    }

    #[test]
    fn test_v_orthonormal_columns() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let dec = svd(&a);
        let vtv = dec.v.transpose().mul(&dec.v);
        let id = DenseMat::identity(3);
        assert!(mat_approx_eq(&vtv, &id, 1e-8));
    }

    #[test]
    fn test_truncated_svd() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ]);
        let trunc = svd_truncated(&a, 2);
        assert_eq!(trunc.sigma.len(), 2);
        assert_eq!(trunc.u.cols, 2);
        assert_eq!(trunc.v.cols, 2);
    }

    #[test]
    fn test_matrix_rank_full() {
        let a = DenseMat::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        assert_eq!(matrix_rank(&a, 1e-10), 3);
    }

    #[test]
    fn test_matrix_rank_deficient() {
        // Rank 1: rows are multiples
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ]);
        assert_eq!(matrix_rank(&a, 1e-8), 1);
    }

    #[test]
    fn test_low_rank_approx() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ]);
        let approx_a = low_rank_approx(&a, 1);
        assert!(mat_approx_eq(&approx_a, &a, 1e-8));
    }

    #[test]
    fn test_pseudoinverse() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let pinv = pseudoinverse(&a, 1e-10);
        // A * A^+ * A = A (Moore-Penrose condition)
        let a_pinv_a = a.mul(&pinv).mul(&a);
        assert!(mat_approx_eq(&a_pinv_a, &a, 1e-8));
    }

    #[test]
    fn test_pseudoinverse_rectangular() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let pinv = pseudoinverse(&a, 1e-10);
        assert_eq!(pinv.rows, 2);
        assert_eq!(pinv.cols, 3);
        // A^+ * A should be identity (2x2)
        let pa = pinv.mul(&a);
        let id2 = DenseMat::identity(2);
        assert!(mat_approx_eq(&pa, &id2, 1e-8));
    }

    #[test]
    fn test_condition_number_identity() {
        let id = DenseMat::identity(3);
        assert!(approx_eq(condition_number(&id), 1.0, 1e-10));
    }

    #[test]
    fn test_condition_number_ill_conditioned() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 0.0, 0.0, 1e-10]);
        let cond = condition_number(&a);
        assert!(cond > 1e9);
    }

    #[test]
    fn test_matrix_norm_2() {
        let a = DenseMat::from_data(2, 2, vec![3.0, 0.0, 0.0, 4.0]);
        assert!(approx_eq(matrix_norm_2(&a), 4.0, 1e-10));
    }

    #[test]
    fn test_frobenius_norm_svd() {
        let a = DenseMat::from_data(2, 2, vec![3.0, 0.0, 0.0, 4.0]);
        // Frobenius = sqrt(9 + 16) = 5
        assert!(approx_eq(frobenius_norm_svd(&a), 5.0, 1e-10));
    }

    #[test]
    fn test_singular_values_nonnegative() {
        let a = DenseMat::from_data(2, 3, vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let dec = svd(&a);
        for s in &dec.sigma {
            assert!(*s >= -1e-12);
        }
    }

    #[test]
    fn test_zero_matrix() {
        let a = DenseMat::zeros(2, 2);
        let dec = svd(&a);
        for s in &dec.sigma {
            assert!(approx_eq(*s, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_single_element() {
        let a = DenseMat::from_data(1, 1, vec![7.0]);
        let dec = svd(&a);
        assert_eq!(dec.sigma.len(), 1);
        assert!(approx_eq(dec.sigma[0], 7.0, 1e-10));
    }

    #[test]
    fn test_svd_symmetric_matrix() {
        // Symmetric: eigenvalues are absolute values of singular values
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let dec = svd(&a);
        // eigenvalues: 3 and 1 => singular values 3 and 1
        assert!(approx_eq(dec.sigma[0], 3.0, 1e-10));
        assert!(approx_eq(dec.sigma[1], 1.0, 1e-10));
    }
}
