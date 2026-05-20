//! Least squares fitting and regression.
//!
//! OLS via normal equations (Cholesky), QR-based, SVD-based, weighted least
//! squares, ridge regression (Tikhonov), residuals, R-squared, polynomial
//! regression, and multi-output regression.

// ── Dense matrix helper ───────────────────────────────────────

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
        assert_eq!(data.len(), rows * cols); Self { rows, cols, data }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }
    #[inline] pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * self.cols + c] }
    #[inline] pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * self.cols + c] = v; }
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
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        (0..self.rows).map(|i| {
            (0..self.cols).map(|j| self.get(i, j) * v[j]).sum()
        }).collect()
    }
}

// ── Result types ──────────────────────────────────────────────

/// Result of a least-squares fit.
#[derive(Debug, Clone, PartialEq)]
pub struct LsResult {
    /// Coefficient vector x that minimizes ||Ax - b||^2.
    pub coeffs: Vec<f64>,
    /// Residual vector: b - Ax.
    pub residuals: Vec<f64>,
    /// Sum of squared residuals.
    pub ssr: f64,
    /// R-squared statistic (coefficient of determination).
    pub r_squared: f64,
}

/// Result of multi-output regression (AX = B).
#[derive(Debug, Clone, PartialEq)]
pub struct MultiLsResult {
    /// Coefficient matrix X (n x p) where n = features, p = outputs.
    pub coeffs: DenseMat,
}

// ── Cholesky decomposition ────────────────────────────────────

/// Lower-triangular Cholesky: A = L L^T.  Returns None if not SPD.
fn cholesky(a: &DenseMat) -> Option<DenseMat> {
    let n = a.rows;
    assert_eq!(n, a.cols);
    let mut l = DenseMat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l.get(i, k) * l.get(j, k);
            }
            if i == j {
                let val = a.get(i, i) - sum;
                if val <= 0.0 { return None; }
                l.set(i, j, val.sqrt());
            } else {
                let lj = l.get(j, j);
                if lj.abs() < 1e-30 { return None; }
                l.set(i, j, (a.get(i, j) - sum) / lj);
            }
        }
    }
    Some(l)
}

/// Solve L y = b by forward substitution.
fn forward_sub(l: &DenseMat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i { s -= l.get(i, j) * y[j]; }
        y[i] = s / l.get(i, i);
    }
    y
}

/// Solve L^T x = y by backward substitution.
fn backward_sub_lt(l: &DenseMat, y: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n { s -= l.get(j, i) * x[j]; }
        x[i] = s / l.get(i, i);
    }
    x
}

// ── Householder QR (thin) ─────────────────────────────────────

fn qr_thin(a: &DenseMat) -> (DenseMat, DenseMat) {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = DenseMat::identity(m);

    let k_max = m.min(n);
    for k in 0..k_max {
        let mut col: Vec<f64> = (k..m).map(|i| r.get(i, k)).collect();
        let norm_col: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_col < 1e-15 { continue; }
        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        col[0] += sign * norm_col;
        let v_norm_sq: f64 = col.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 { continue; }
        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..col.len() { dot_val += col[i] * r.get(i + k, j); }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for i in 0..col.len() { r.set(i + k, j, r.get(i + k, j) - coeff * col[i]); }
        }
        for i in 0..m {
            let mut dot_val = 0.0;
            for jj in 0..col.len() { dot_val += q.get(i, jj + k) * col[jj]; }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for jj in 0..col.len() { q.set(i, jj + k, q.get(i, jj + k) - coeff * col[jj]); }
        }
    }

    // Thin: Q is m x n, R is n x n.
    if m > n {
        let mut qt = DenseMat::zeros(m, n);
        for i in 0..m { for j in 0..n { qt.set(i, j, q.get(i, j)); } }
        let mut rt = DenseMat::zeros(n, n);
        for i in 0..n { for j in 0..n { rt.set(i, j, r.get(i, j)); } }
        (qt, rt)
    } else {
        (q, r)
    }
}

// ── SVD (one-sided Jacobi) for least squares ──────────────────

fn svd_simple(a: &DenseMat) -> (DenseMat, Vec<f64>, DenseMat) {
    let m = a.rows;
    let n = a.cols;
    let mut b = a.clone();
    let mut v = DenseMat::identity(n);
    let max_iter = 100 * n * n;
    let tol = 1e-14;

    for _ in 0..max_iter {
        let mut converged = true;
        for p in 0..n {
            for q_idx in (p + 1)..n {
                let alpha: f64 = (0..m).map(|i| b.get(i, p) * b.get(i, p)).sum();
                let beta: f64 = (0..m).map(|i| b.get(i, q_idx) * b.get(i, q_idx)).sum();
                let gamma: f64 = (0..m).map(|i| b.get(i, p) * b.get(i, q_idx)).sum();
                if gamma.abs() < tol * (alpha * beta).sqrt().max(1e-30) { continue; }
                converged = false;
                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) }
                    else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..m {
                    let bp = b.get(i, p);
                    let bq = b.get(i, q_idx);
                    b.set(i, p, c * bp - s * bq);
                    b.set(i, q_idx, s * bp + c * bq);
                }
                for i in 0..n {
                    let vp = v.get(i, p);
                    let vq = v.get(i, q_idx);
                    v.set(i, p, c * vp - s * vq);
                    v.set(i, q_idx, s * vp + c * vq);
                }
            }
        }
        if converged { break; }
    }

    let k = m.min(n);
    let mut sigma = Vec::with_capacity(k);
    let mut u = DenseMat::zeros(m, k);
    for j in 0..k {
        let s: f64 = (0..m).map(|i| b.get(i, j) * b.get(i, j)).sum::<f64>().sqrt();
        sigma.push(s);
        if s > 1e-30 {
            for i in 0..m { u.set(i, j, b.get(i, j) / s); }
        }
    }

    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|a, b| sigma[*b].partial_cmp(&sigma[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let ss: Vec<f64> = indices.iter().map(|i| sigma[*i]).collect();
    let mut su = DenseMat::zeros(m, k);
    let mut sv = DenseMat::zeros(n, k);
    for (nj, &oj) in indices.iter().enumerate() {
        for i in 0..m { su.set(i, nj, u.get(i, oj)); }
        for i in 0..n { sv.set(i, nj, v.get(i, oj)); }
    }
    (su, ss, sv)
}

// ── Helpers ───────────────────────────────────────────────────

fn compute_residuals(a: &DenseMat, x: &[f64], b: &[f64]) -> Vec<f64> {
    let ax = a.matvec(x);
    b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect()
}

fn compute_ssr(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum()
}

fn compute_r_squared(b: &[f64], ssr: f64) -> f64 {
    let mean: f64 = b.iter().sum::<f64>() / b.len() as f64;
    let sst: f64 = b.iter().map(|bi| (bi - mean) * (bi - mean)).sum();
    if sst < 1e-30 { return 1.0; }
    1.0 - ssr / sst
}

fn make_ls_result(a: &DenseMat, coeffs: Vec<f64>, b: &[f64]) -> LsResult {
    let residuals = compute_residuals(a, &coeffs, b);
    let ssr = compute_ssr(&residuals);
    let r_squared = compute_r_squared(b, ssr);
    LsResult { coeffs, residuals, ssr, r_squared }
}

// ── OLS via normal equations (Cholesky) ───────────────────────

/// Ordinary least squares via normal equations: (A^T A) x = A^T b.
/// Uses Cholesky factorization.  Returns None if A^T A is not SPD.
pub fn ols_normal(a: &DenseMat, b: &[f64]) -> Option<LsResult> {
    let at = a.transpose();
    let ata = at.mul(a);
    let atb = at.matvec(b);
    let l = cholesky(&ata)?;
    let y = forward_sub(&l, &atb);
    let coeffs = backward_sub_lt(&l, &y);
    Some(make_ls_result(a, coeffs, b))
}

// ── QR-based least squares ────────────────────────────────────

/// Least squares via QR decomposition (more numerically stable than normal equations).
pub fn ols_qr(a: &DenseMat, b: &[f64]) -> LsResult {
    let m = a.rows;
    let n = a.cols;
    let (q, r) = qr_thin(a);
    // Q^T b
    let mut qtb = vec![0.0; n];
    for i in 0..n {
        for j in 0..m { qtb[i] += q.get(j, i) * b[j]; }
    }
    // Back-substitution.
    let mut coeffs = vec![0.0; n];
    for i in (0..n).rev() {
        let d = r.get(i, i);
        if d.abs() < 1e-14 { continue; }
        let mut s = qtb[i];
        for j in (i + 1)..n { s -= r.get(i, j) * coeffs[j]; }
        coeffs[i] = s / d;
    }
    make_ls_result(a, coeffs, b)
}

// ── SVD-based least squares ───────────────────────────────────

/// Least squares via SVD (handles rank-deficient A).
pub fn ols_svd(a: &DenseMat, b: &[f64], sv_tol: f64) -> LsResult {
    let m = a.rows;
    let n = a.cols;
    let (u, sigma, v) = svd_simple(a);
    let k = sigma.len();
    // x = V * diag(1/sigma) * U^T * b
    let mut utb = vec![0.0; k];
    for i in 0..k {
        for j in 0..m { utb[i] += u.get(j, i) * b[j]; }
    }
    let mut coeffs = vec![0.0; n];
    for j in 0..n {
        for i in 0..k {
            if sigma[i] > sv_tol {
                coeffs[j] += v.get(j, i) * utb[i] / sigma[i];
            }
        }
    }
    make_ls_result(a, coeffs, b)
}

// ── Weighted least squares ────────────────────────────────────

/// Weighted least squares: minimize ||W^{1/2}(Ax - b)||^2.
/// `weights` is the diagonal of W (must be positive).
pub fn wls(a: &DenseMat, b: &[f64], weights: &[f64]) -> LsResult {
    let m = a.rows;
    let n = a.cols;
    // Form W^{1/2} A and W^{1/2} b.
    let mut wa = DenseMat::zeros(m, n);
    let mut wb = vec![0.0; m];
    for i in 0..m {
        let w = weights[i].sqrt();
        wb[i] = w * b[i];
        for j in 0..n {
            wa.set(i, j, w * a.get(i, j));
        }
    }
    let result = ols_qr(&wa, &wb);
    // Residuals should be in original space.
    let residuals = compute_residuals(a, &result.coeffs, b);
    let ssr = compute_ssr(&residuals);
    let r_squared = compute_r_squared(b, ssr);
    LsResult { coeffs: result.coeffs, residuals, ssr, r_squared }
}

// ── Ridge regression (Tikhonov) ───────────────────────────────

/// Ridge regression: minimize ||Ax - b||^2 + lambda * ||x||^2.
pub fn ridge(a: &DenseMat, b: &[f64], lambda: f64) -> Option<LsResult> {
    let n = a.cols;
    let at = a.transpose();
    let mut ata = at.mul(a);
    // Add lambda * I.
    for i in 0..n { ata.set(i, i, ata.get(i, i) + lambda); }
    let atb = at.matvec(b);
    let l = cholesky(&ata)?;
    let y = forward_sub(&l, &atb);
    let coeffs = backward_sub_lt(&l, &y);
    Some(make_ls_result(a, coeffs, b))
}

// ── Polynomial regression ─────────────────────────────────────

/// Build a Vandermonde matrix for polynomial regression of given degree.
pub fn vandermonde(x: &[f64], degree: usize) -> DenseMat {
    let m = x.len();
    let n = degree + 1;
    let mut v = DenseMat::zeros(m, n);
    for i in 0..m {
        let mut xi = 1.0;
        for j in 0..n {
            v.set(i, j, xi);
            xi *= x[i];
        }
    }
    v
}

/// Fit a polynomial of given degree to (x, y) data.
pub fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> LsResult {
    let v = vandermonde(x, degree);
    ols_qr(&v, y)
}

// ── Multi-output regression ───────────────────────────────────

/// Multi-output least squares: solve AX = B where B is m x p.
/// Returns X as n x p.
pub fn multi_output_ls(a: &DenseMat, b: &DenseMat) -> MultiLsResult {
    let n = a.cols;
    let p = b.cols;
    let mut coeffs = DenseMat::zeros(n, p);
    for j in 0..p {
        let b_col = b.col_vec(j);
        let result = ols_qr(a, &b_col);
        for i in 0..n {
            coeffs.set(i, j, result.coeffs[i]);
        }
    }
    MultiLsResult { coeffs }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn approx_vec(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    #[test]
    fn test_ols_normal_exact() {
        // A = I, b = [1,2] => x = [1,2]
        let a = DenseMat::identity(2);
        let b = vec![1.0, 2.0];
        let res = ols_normal(&a, &b).unwrap();
        assert!(approx_vec(&res.coeffs, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_ols_qr_exact() {
        let a = DenseMat::identity(2);
        let b = vec![1.0, 2.0];
        let res = ols_qr(&a, &b);
        assert!(approx_vec(&res.coeffs, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_ols_svd_exact() {
        let a = DenseMat::identity(2);
        let b = vec![1.0, 2.0];
        let res = ols_svd(&a, &b, 1e-10);
        assert!(approx_vec(&res.coeffs, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_ols_overdetermined() {
        // y = 1 + 2x, points (0,1), (1,3), (2,5)
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let res = ols_qr(&a, &b);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-10));
        assert!(approx_eq(res.r_squared, 1.0, 1e-10));
    }

    #[test]
    fn test_ols_normal_overdetermined() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let res = ols_normal(&a, &b).unwrap();
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-10));
    }

    #[test]
    fn test_ols_svd_overdetermined() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let res = ols_svd(&a, &b, 1e-10);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-8));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-8));
    }

    #[test]
    fn test_residuals() {
        let a = DenseMat::from_data(3, 1, vec![1.0, 1.0, 1.0]);
        let b = vec![1.0, 2.0, 3.0];
        let res = ols_qr(&a, &b);
        // best fit: x = mean(b) = 2
        assert!(approx_eq(res.coeffs[0], 2.0, 1e-10));
        assert!(approx_eq(res.residuals[0], -1.0, 1e-10));
        assert!(approx_eq(res.residuals[1], 0.0, 1e-10));
        assert!(approx_eq(res.residuals[2], 1.0, 1e-10));
    }

    #[test]
    fn test_r_squared_perfect() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let res = ols_qr(&a, &b);
        assert!(approx_eq(res.r_squared, 1.0, 1e-10));
    }

    #[test]
    fn test_r_squared_poor() {
        // Fitting constant to variable data.
        let a = DenseMat::from_data(3, 1, vec![1.0, 1.0, 1.0]);
        let b = vec![1.0, 2.0, 3.0];
        let res = ols_qr(&a, &b);
        assert!(res.r_squared < 0.1);
    }

    #[test]
    fn test_wls() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let w = vec![1.0, 1.0, 1.0]; // uniform weights = OLS
        let res = wls(&a, &b, &w);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-10));
    }

    #[test]
    fn test_wls_nonuniform() {
        // Heavy weight on first point.
        let a = DenseMat::from_data(2, 1, vec![1.0, 1.0]);
        let b = vec![1.0, 3.0];
        let w = vec![100.0, 1.0];
        let res = wls(&a, &b, &w);
        // Should be close to 1.0 (heavily weighted first point).
        assert!(res.coeffs[0] < 1.5);
    }

    #[test]
    fn test_ridge() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = vec![1.0, 3.0, 5.0];
        let res = ridge(&a, &b, 0.001).unwrap();
        // With small lambda, should be close to OLS.
        assert!(approx_eq(res.coeffs[0], 1.0, 0.1));
        assert!(approx_eq(res.coeffs[1], 2.0, 0.1));
    }

    #[test]
    fn test_ridge_high_lambda() {
        let a = DenseMat::identity(2);
        let b = vec![10.0, 10.0];
        let res_small = ridge(&a, &b, 0.0001).unwrap();
        let res_big = ridge(&a, &b, 100.0).unwrap();
        // With large lambda, coefficients are shrunk toward zero.
        assert!(res_big.coeffs[0].abs() < res_small.coeffs[0].abs());
    }

    #[test]
    fn test_polynomial_regression_linear() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0]; // y = 1 + 2x
        let res = polynomial_regression(&x, &y, 1);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-10));
    }

    #[test]
    fn test_polynomial_regression_quadratic() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| 1.0 + 2.0 * xi + 0.5 * xi * xi).collect();
        let res = polynomial_regression(&x, &y, 2);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-8));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-8));
        assert!(approx_eq(res.coeffs[2], 0.5, 1e-8));
    }

    #[test]
    fn test_vandermonde() {
        let x = vec![1.0, 2.0, 3.0];
        let v = vandermonde(&x, 2);
        assert_eq!(v.rows, 3);
        assert_eq!(v.cols, 3);
        assert!(approx_eq(v.get(1, 0), 1.0, 1e-12));
        assert!(approx_eq(v.get(1, 1), 2.0, 1e-12));
        assert!(approx_eq(v.get(1, 2), 4.0, 1e-12));
    }

    #[test]
    fn test_multi_output() {
        // A = I, B = [[1,2],[3,4]] => X = B
        let a = DenseMat::identity(2);
        let b = DenseMat::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let res = multi_output_ls(&a, &b);
        assert!(approx_eq(res.coeffs.get(0, 0), 1.0, 1e-10));
        assert!(approx_eq(res.coeffs.get(1, 1), 4.0, 1e-10));
    }

    #[test]
    fn test_multi_output_overdetermined() {
        let a = DenseMat::from_data(3, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = DenseMat::from_data(3, 2, vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let res = multi_output_ls(&a, &b);
        assert_eq!(res.coeffs.rows, 2);
        assert_eq!(res.coeffs.cols, 2);
    }

    #[test]
    fn test_cholesky_not_spd() {
        let a = DenseMat::from_data(2, 2, vec![-1.0, 0.0, 0.0, 1.0]);
        assert!(cholesky(&a).is_none());
    }

    #[test]
    fn test_ols_3_features() {
        // y = 1*x1 + 2*x2 + 3*x3
        let a = DenseMat::from_data(4, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        let b = vec![1.0, 2.0, 3.0, 6.0];
        let res = ols_qr(&a, &b);
        assert!(approx_eq(res.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(res.coeffs[1], 2.0, 1e-10));
        assert!(approx_eq(res.coeffs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_ssr_zero_for_exact() {
        let a = DenseMat::identity(2);
        let b = vec![3.0, 7.0];
        let res = ols_qr(&a, &b);
        assert!(approx_eq(res.ssr, 0.0, 1e-10));
    }
}
