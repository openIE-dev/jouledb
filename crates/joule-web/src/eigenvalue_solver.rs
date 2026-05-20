//! Eigenvalue computation — power iteration, inverse iteration, QR algorithm,
//! Hessenberg reduction, symmetric tridiagonal QR, spectral radius,
//! SPD check, and characteristic polynomial for small matrices.

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

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { t.set(j, i, self.get(i, j)); } }
        t
    }

    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.rows;
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..self.cols {
                y[i] += self.get(i, j) * v[j];
            }
        }
        y
    }
}

// ── Vector helpers ────────────────────────────────────────────

fn vec_norm(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }
fn vec_dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() }
fn vec_scale(s: f64, v: &[f64]) -> Vec<f64> { v.iter().map(|x| s * x).collect() }
fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

// ── Power iteration ───────────────────────────────────────────

/// Result of eigenvalue computation.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenPair {
    /// Eigenvalue (real part).
    pub value: f64,
    /// Eigenvector.
    pub vector: Vec<f64>,
}

/// Power iteration: find the dominant eigenvalue and eigenvector.
pub fn power_iteration(a: &DenseMat, max_iter: usize, tol: f64) -> EigenPair {
    let n = a.rows;
    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        let w = a.matvec(&v);
        let new_ev = vec_dot(&w, &v);
        let nw = vec_norm(&w);
        if nw < 1e-30 { break; }
        v = vec_scale(1.0 / nw, &w);
        if (new_ev - eigenvalue).abs() < tol {
            eigenvalue = new_ev;
            break;
        }
        eigenvalue = new_ev;
    }
    EigenPair { value: eigenvalue, vector: v }
}

/// Inverse iteration: find eigenvalue nearest to `shift` and its eigenvector.
pub fn inverse_iteration(a: &DenseMat, shift: f64, max_iter: usize, tol: f64) -> EigenPair {
    let n = a.rows;
    // Build (A - shift*I).
    let mut shifted = a.clone();
    for i in 0..n {
        shifted.set(i, i, shifted.get(i, i) - shift);
    }

    // Use an asymmetric starting vector to avoid missing eigenvectors
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / (n as f64)).collect();
    let nv = vec_norm(&v);
    if nv > 1e-30 {
        v = vec_scale(1.0 / nv, &v);
    }
    let mut eigenvalue = shift;

    for _ in 0..max_iter {
        // Solve (A - shift*I) w = v  via simple Gaussian elimination.
        let w = solve_linear(&shifted, &v);
        let nw = vec_norm(&w);
        if nw < 1e-30 { break; }
        let new_v = vec_scale(1.0 / nw, &w);
        let av = a.matvec(&new_v);
        let new_ev = vec_dot(&av, &new_v);
        if (new_ev - eigenvalue).abs() < tol {
            eigenvalue = new_ev;
            v = new_v;
            break;
        }
        eigenvalue = new_ev;
        v = new_v;
    }

    EigenPair { value: eigenvalue, vector: v }
}

/// Simple Gaussian elimination (no pivoting) for inverse iteration.
fn solve_linear(a: &DenseMat, b: &[f64]) -> Vec<f64> {
    let n = a.rows;
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a.get(i, j);
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for k in 0..n {
        // Partial pivoting.
        let mut max_row = k;
        let mut max_val = aug[k * (n + 1) + k].abs();
        for i in (k + 1)..n {
            let v = aug[i * (n + 1) + k].abs();
            if v > max_val { max_val = v; max_row = i; }
        }
        if max_row != k {
            for j in 0..=(n) {
                let tmp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }
        let pivot = aug[k * (n + 1) + k];
        if pivot.abs() < 1e-30 { continue; }
        for i in (k + 1)..n {
            let factor = aug[i * (n + 1) + k] / pivot;
            for j in k..=(n) {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = aug[i * (n + 1) + i];
        if diag.abs() < 1e-30 { continue; }
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / diag;
    }
    x
}

// ── Hessenberg reduction ──────────────────────────────────────

/// Reduce A to upper Hessenberg form: H = Q^T A Q.
/// Returns (H, Q).
pub fn hessenberg(a: &DenseMat) -> (DenseMat, DenseMat) {
    let n = a.rows;
    let mut h = a.clone();
    let mut q = DenseMat::identity(n);

    for k in 0..(n.saturating_sub(2)) {
        let mut col: Vec<f64> = ((k + 1)..n).map(|i| h.get(i, k)).collect();
        let norm_col = vec_norm(&col);
        if norm_col < 1e-15 { continue; }

        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        col[0] += sign * norm_col;
        let v_norm_sq: f64 = col.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 { continue; }

        // H <- (I - 2vv^T/||v||^2) * H
        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..col.len() { dot_val += col[i] * h.get(i + k + 1, j); }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for i in 0..col.len() {
                let old = h.get(i + k + 1, j);
                h.set(i + k + 1, j, old - coeff * col[i]);
            }
        }
        // H <- H * (I - 2vv^T/||v||^2)
        for i in 0..n {
            let mut dot_val = 0.0;
            for j in 0..col.len() { dot_val += h.get(i, j + k + 1) * col[j]; }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for j in 0..col.len() {
                let old = h.get(i, j + k + 1);
                h.set(i, j + k + 1, old - coeff * col[j]);
            }
        }
        // Accumulate Q.
        for i in 0..n {
            let mut dot_val = 0.0;
            for j in 0..col.len() { dot_val += q.get(i, j + k + 1) * col[j]; }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for j in 0..col.len() {
                let old = q.get(i, j + k + 1);
                q.set(i, j + k + 1, old - coeff * col[j]);
            }
        }
    }
    (h, q)
}

// ── QR algorithm for all eigenvalues ──────────────────────────

/// QR algorithm with implicit shifts to find all eigenvalues.
/// Returns eigenvalues (real parts only — complex eigenvalues give approximate real part).
pub fn qr_eigenvalues(a: &DenseMat, max_iter: usize) -> Vec<f64> {
    let n = a.rows;
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![a.get(0, 0)]; }

    let (mut h, _) = hessenberg(a);

    for _ in 0..max_iter {
        // Wilkinson shift: eigenvalue of trailing 2x2 closest to h[n-1,n-1].
        let nn = n;
        let a11 = h.get(nn - 2, nn - 2);
        let a12 = h.get(nn - 2, nn - 1);
        let a21 = h.get(nn - 1, nn - 2);
        let a22 = h.get(nn - 1, nn - 1);
        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let disc = trace * trace - 4.0 * det;
        let shift = if disc >= 0.0 {
            let s1 = (trace + disc.sqrt()) / 2.0;
            let s2 = (trace - disc.sqrt()) / 2.0;
            if (s1 - a22).abs() < (s2 - a22).abs() { s1 } else { s2 }
        } else {
            a22
        };

        // Shift.
        for i in 0..n { h.set(i, i, h.get(i, i) - shift); }

        // QR step via Givens rotations.
        let mut cs = Vec::with_capacity(n - 1);
        let mut sn = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let a_val = h.get(i, i);
            let b_val = h.get(i + 1, i);
            let r = (a_val * a_val + b_val * b_val).sqrt();
            let (c, s) = if r > 1e-30 { (a_val / r, -b_val / r) } else { (1.0, 0.0) };
            cs.push(c);
            sn.push(s);
            // Apply G^T from left.
            for j in 0..n {
                let h1 = h.get(i, j);
                let h2 = h.get(i + 1, j);
                h.set(i, j, c * h1 - s * h2);
                h.set(i + 1, j, s * h1 + c * h2);
            }
        }
        // Apply G from right.
        for i in 0..(n - 1) {
            let c = cs[i];
            let s = sn[i];
            for j in 0..n {
                let h1 = h.get(j, i);
                let h2 = h.get(j, i + 1);
                h.set(j, i, c * h1 - s * h2);
                h.set(j, i + 1, s * h1 + c * h2);
            }
        }
        // Unshift.
        for i in 0..n { h.set(i, i, h.get(i, i) + shift); }

        // Check for convergence: sub-diagonal near zero.
        let mut done = true;
        for i in 0..(n - 1) {
            if h.get(i + 1, i).abs() > 1e-12 {
                done = false;
                break;
            }
        }
        if done { break; }
    }

    // Read eigenvalues from diagonal (with 2x2 block detection).
    let mut eigs = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && h.get(i + 1, i).abs() > 1e-10 {
            // 2x2 block: extract real parts of complex conjugate pair.
            let a11 = h.get(i, i);
            let a22 = h.get(i + 1, i + 1);
            let tr = (a11 + a22) / 2.0;
            eigs.push(tr);
            eigs.push(tr);
            i += 2;
        } else {
            eigs.push(h.get(i, i));
            i += 1;
        }
    }
    eigs
}

// ── Symmetric eigenvalue (tridiagonal) ────────────────────────

/// Reduce a symmetric matrix to tridiagonal form.
/// Returns (diagonal, sub-diagonal, Q) such that T = Q^T A Q.
pub fn tridiagonalize(a: &DenseMat) -> (Vec<f64>, Vec<f64>, DenseMat) {
    let n = a.rows;
    let mut t = a.clone();
    let mut q = DenseMat::identity(n);

    for k in 0..(n.saturating_sub(2)) {
        let mut col: Vec<f64> = ((k + 1)..n).map(|i| t.get(i, k)).collect();
        let norm_col = vec_norm(&col);
        if norm_col < 1e-15 { continue; }

        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        col[0] += sign * norm_col;
        let v_norm_sq: f64 = col.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 { continue; }

        // T <- H * T * H (symmetric Householder).
        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..col.len() { dot_val += col[i] * t.get(i + k + 1, j); }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for i in 0..col.len() { t.set(i + k + 1, j, t.get(i + k + 1, j) - coeff * col[i]); }
        }
        for i in 0..n {
            let mut dot_val = 0.0;
            for j in 0..col.len() { dot_val += t.get(i, j + k + 1) * col[j]; }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for j in 0..col.len() { t.set(i, j + k + 1, t.get(i, j + k + 1) - coeff * col[j]); }
        }
        for i in 0..n {
            let mut dot_val = 0.0;
            for j in 0..col.len() { dot_val += q.get(i, j + k + 1) * col[j]; }
            let coeff = 2.0 * dot_val / v_norm_sq;
            for j in 0..col.len() { q.set(i, j + k + 1, q.get(i, j + k + 1) - coeff * col[j]); }
        }
    }

    let diag: Vec<f64> = (0..n).map(|i| t.get(i, i)).collect();
    let sub_diag: Vec<f64> = (0..(n.saturating_sub(1))).map(|i| t.get(i + 1, i)).collect();
    (diag, sub_diag, q)
}

/// Symmetric eigenvalues via tridiagonal QR with Wilkinson shifts.
pub fn symmetric_eigenvalues(a: &DenseMat, max_iter: usize) -> Vec<f64> {
    let n = a.rows;
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![a.get(0, 0)]; }

    let (mut diag, mut sub, _) = tridiagonalize(a);

    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..sub.len() {
            if sub[i].abs() > 1e-12 {
                converged = false;
                break;
            }
        }
        if converged { break; }

        // Wilkinson shift from trailing 2x2.
        let m = diag.len();
        let d = (diag[m - 2] - diag[m - 1]) / 2.0;
        let mu = diag[m - 1]
            - sub[m - 2] * sub[m - 2]
                / (d + d.signum() * (d * d + sub[m - 2] * sub[m - 2]).sqrt());

        // Implicit QR step (Givens).
        let mut x = diag[0] - mu;
        let mut z = sub[0];
        for k in 0..(m - 1) {
            let r = (x * x + z * z).sqrt();
            let (c, s) = if r > 1e-30 { (x / r, -z / r) } else { (1.0, 0.0) };

            // Update tridiagonal.
            if k > 0 {
                sub[k - 1] = r;
            }
            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e = sub[k];
            diag[k] = c * c * d1 + s * s * d2 - 2.0 * c * s * e;
            diag[k + 1] = s * s * d1 + c * c * d2 + 2.0 * c * s * e;
            sub[k] = c * s * (d1 - d2) + (c * c - s * s) * e;

            if k + 1 < m - 1 {
                x = sub[k + 1] * c;
                z = -sub[k + 1] * s;
                // The sub[k+1] gets multiplied by c elsewhere, handle in next iteration via x,z
            }
        }
    }

    diag.sort_by(|a_v, b_v| a_v.partial_cmp(b_v).unwrap_or(std::cmp::Ordering::Equal));
    diag
}

// ── Derived operations ────────────────────────────────────────

/// Spectral radius: max |eigenvalue|.
pub fn spectral_radius(a: &DenseMat) -> f64 {
    let eigs = qr_eigenvalues(a, 200);
    eigs.iter().map(|e| e.abs()).fold(0.0_f64, f64::max)
}

/// Sort eigenvalues by magnitude (descending).
pub fn sort_by_magnitude(eigenvalues: &mut [f64]) {
    eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
}

/// Sort eigenvalues by real value (ascending).
pub fn sort_by_real(eigenvalues: &mut [f64]) {
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Check if matrix is symmetric positive definite (all eigenvalues > 0).
pub fn is_spd(a: &DenseMat, tol: f64) -> bool {
    if a.rows != a.cols { return false; }
    // Check symmetry.
    for i in 0..a.rows {
        for j in 0..i {
            if (a.get(i, j) - a.get(j, i)).abs() > tol { return false; }
        }
    }
    let eigs = symmetric_eigenvalues(a, 200);
    eigs.iter().all(|e| *e > -tol)
}

/// Characteristic polynomial coefficients for a small matrix (up to 4x4).
/// Returns coefficients [c0, c1, ..., cn] such that p(λ) = c0 + c1*λ + ... + cn*λ^n.
pub fn characteristic_polynomial(a: &DenseMat) -> Vec<f64> {
    let n = a.rows;
    assert_eq!(n, a.cols);
    match n {
        0 => vec![1.0],
        1 => vec![-a.get(0, 0), 1.0],
        2 => {
            let tr = a.get(0, 0) + a.get(1, 1);
            let det = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
            vec![det, -tr, 1.0]
        }
        3 => {
            // Use Faddeev-LeVerrier.
            let tr = a.get(0, 0) + a.get(1, 1) + a.get(2, 2);
            let a2 = a.mul(a);
            let tr2 = a2.get(0, 0) + a2.get(1, 1) + a2.get(2, 2);
            let c2 = (tr * tr - tr2) / 2.0;
            // det via Sarrus.
            let det = a.get(0, 0) * (a.get(1, 1) * a.get(2, 2) - a.get(1, 2) * a.get(2, 1))
                - a.get(0, 1) * (a.get(1, 0) * a.get(2, 2) - a.get(1, 2) * a.get(2, 0))
                + a.get(0, 2) * (a.get(1, 0) * a.get(2, 1) - a.get(1, 1) * a.get(2, 0));
            vec![-det, c2, -tr, 1.0]
        }
        _ => {
            // Faddeev-LeVerrier for general n.
            let mut coeffs = vec![0.0; n + 1];
            coeffs[n] = 1.0;
            let mut m = DenseMat::identity(n);
            for k in 1..=n {
                m = a.mul(&m);
                if k > 1 {
                    for i in 0..n { m.set(i, i, m.get(i, i) + coeffs[n - k + 1]); }
                    m = a.mul(&m);
                }
                let mut tr_val = 0.0;
                for i in 0..n { tr_val += m.get(i, i); }
                coeffs[n - k] = -tr_val / k as f64;
                // Reset M = A^k + c_{n-1} A^{k-1} + ... using the recurrence.
                m = a.clone();
                let mut acc = a.clone();
                for i in 0..n { acc.set(i, i, acc.get(i, i) + coeffs[n - 1]); }
                for j in 2..=k {
                    acc = a.mul(&acc);
                    for i in 0..n { acc.set(i, i, acc.get(i, i) + coeffs[n - j]); }
                }
                m = acc;
            }
            coeffs
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_power_iteration_diagonal() {
        let a = DenseMat::from_data(3, 3, vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let ep = power_iteration(&a, 1000, 1e-12);
        assert!(approx_eq(ep.value, 5.0, 1e-6));
    }

    #[test]
    fn test_power_iteration_symmetric() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let ep = power_iteration(&a, 1000, 1e-12);
        assert!(approx_eq(ep.value, 3.0, 1e-6));
    }

    #[test]
    fn test_inverse_iteration() {
        let a = DenseMat::from_data(3, 3, vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let ep = inverse_iteration(&a, 0.9, 1000, 1e-10);
        assert!(approx_eq(ep.value, 1.0, 1e-4));
    }

    #[test]
    fn test_inverse_iteration_middle_eigenvalue() {
        let a = DenseMat::from_data(3, 3, vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let ep = inverse_iteration(&a, 2.9, 1000, 1e-10);
        assert!(approx_eq(ep.value, 3.0, 1e-4));
    }

    #[test]
    fn test_hessenberg_form() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let (h, _q) = hessenberg(&a);
        // Below sub-diagonal should be zero.
        for i in 2..3 {
            for j in 0..(i - 1) {
                assert!(approx_eq(h.get(i, j), 0.0, 1e-10));
            }
        }
    }

    #[test]
    fn test_hessenberg_preserves_eigenvalues() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let (h, q) = hessenberg(&a);
        // Q^T A Q should equal H.
        let qt_a_q = q.transpose().mul(&a).mul(&q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(qt_a_q.get(i, j), h.get(i, j), 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_eigenvalues_diagonal() {
        let a = DenseMat::from_data(3, 3, vec![
            7.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let mut eigs = qr_eigenvalues(&a, 200);
        sort_by_magnitude(&mut eigs);
        assert!(approx_eq(eigs[0], 7.0, 1e-6));
        assert!(approx_eq(eigs[1], 3.0, 1e-6));
        assert!(approx_eq(eigs[2], 1.0, 1e-6));
    }

    #[test]
    fn test_qr_eigenvalues_symmetric() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let mut eigs = qr_eigenvalues(&a, 200);
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(approx_eq(eigs[0], 3.0, 1e-6));
        assert!(approx_eq(eigs[1], 1.0, 1e-6));
    }

    #[test]
    fn test_symmetric_eigenvalues() {
        let a = DenseMat::from_data(3, 3, vec![
            4.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let eigs = symmetric_eigenvalues(&a, 300);
        // Sum of eigenvalues = trace = 9.
        let sum: f64 = eigs.iter().sum();
        assert!(approx_eq(sum, 9.0, 1e-4));
    }

    #[test]
    fn test_tridiagonalize_preserves_structure() {
        let a = DenseMat::from_data(3, 3, vec![
            4.0, 1.0, 2.0,
            1.0, 3.0, 1.0,
            2.0, 1.0, 2.0,
        ]);
        let (diag, sub, _) = tridiagonalize(&a);
        assert_eq!(diag.len(), 3);
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn test_spectral_radius() {
        let a = DenseMat::from_data(2, 2, vec![3.0, 0.0, 0.0, -5.0]);
        let sr = spectral_radius(&a);
        assert!(approx_eq(sr, 5.0, 1e-4));
    }

    #[test]
    fn test_sort_by_magnitude() {
        let mut eigs = vec![1.0, -5.0, 3.0];
        sort_by_magnitude(&mut eigs);
        assert!(approx_eq(eigs[0], -5.0, 1e-12));
    }

    #[test]
    fn test_sort_by_real() {
        let mut eigs = vec![3.0, 1.0, 2.0];
        sort_by_real(&mut eigs);
        assert!(approx_eq(eigs[0], 1.0, 1e-12));
        assert!(approx_eq(eigs[2], 3.0, 1e-12));
    }

    #[test]
    fn test_is_spd_positive() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        assert!(is_spd(&a, 1e-6));
    }

    #[test]
    fn test_is_spd_negative() {
        let a = DenseMat::from_data(2, 2, vec![-1.0, 0.0, 0.0, 1.0]);
        assert!(!is_spd(&a, 1e-6));
    }

    #[test]
    fn test_is_spd_not_symmetric() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        assert!(!is_spd(&a, 1e-6));
    }

    #[test]
    fn test_char_poly_1x1() {
        let a = DenseMat::from_data(1, 1, vec![3.0]);
        let p = characteristic_polynomial(&a);
        // p(lambda) = lambda - 3  => [-3, 1]
        assert!(approx_eq(p[0], -3.0, 1e-12));
        assert!(approx_eq(p[1], 1.0, 1e-12));
    }

    #[test]
    fn test_char_poly_2x2() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let p = characteristic_polynomial(&a);
        // p(lambda) = lambda^2 - 4*lambda + 3 => [3, -4, 1]
        assert!(approx_eq(p[0], 3.0, 1e-10));
        assert!(approx_eq(p[1], -4.0, 1e-10));
        assert!(approx_eq(p[2], 1.0, 1e-10));
    }

    #[test]
    fn test_char_poly_3x3_trace() {
        let a = DenseMat::from_data(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]);
        let p = characteristic_polynomial(&a);
        // Coefficient of lambda^2 = -(trace) = -6
        assert!(approx_eq(p[2], -6.0, 1e-8));
    }

    #[test]
    fn test_eigenvector_orthogonality() {
        let a = DenseMat::from_data(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let ep1 = power_iteration(&a, 1000, 1e-12);
        let ep2 = inverse_iteration(&a, 0.9, 1000, 1e-10);
        let d = vec_dot(&ep1.vector, &ep2.vector);
        assert!(d.abs() < 0.2); // roughly orthogonal
    }

    #[test]
    fn test_qr_eigenvalues_empty() {
        let a = DenseMat::zeros(0, 0);
        let eigs = qr_eigenvalues(&a, 100);
        assert!(eigs.is_empty());
    }

    #[test]
    fn test_qr_eigenvalues_1x1() {
        let a = DenseMat::from_data(1, 1, vec![42.0]);
        let eigs = qr_eigenvalues(&a, 100);
        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 42.0, 1e-12));
    }

    #[test]
    fn test_symmetric_eigenvalues_identity() {
        let a = DenseMat::identity(3);
        let eigs = symmetric_eigenvalues(&a, 200);
        for e in &eigs {
            assert!(approx_eq(*e, 1.0, 1e-6));
        }
    }
}
