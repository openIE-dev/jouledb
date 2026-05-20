//! LU decomposition with partial pivoting.
//!
//! PA = LU factorization, forward/backward substitution, solve Ax = b,
//! determinant, matrix inverse, singular detection, and multiple RHS.

use std::fmt;

// ── Dense matrix helper ───────────────────────────────────────

/// Row-major dense matrix for LU operations.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    /// Create a matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create from row-major data.
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Self { rows, cols, data }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
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

    /// Extract column j as a vector.
    pub fn col_vec(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
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

// ── LU result ─────────────────────────────────────────────────

/// Result of LU decomposition with partial pivoting.
#[derive(Debug, Clone, PartialEq)]
pub struct LuDecomposition {
    /// Combined L (lower, unit diagonal) and U (upper) in a single matrix.
    /// L entries are below diagonal, U entries are on and above diagonal.
    pub lu: DenseMat,
    /// Pivot permutation: row i of the original matrix is row pivot[i].
    pub pivot: Vec<usize>,
    /// Number of row swaps (for determinant sign).
    pub swaps: usize,
    /// True if a near-zero pivot was encountered (matrix is singular or
    /// nearly singular).
    pub singular: bool,
}

// ── Decomposition ─────────────────────────────────────────────

/// Threshold below which a pivot is considered zero.
const PIVOT_TOL: f64 = 1e-14;

/// Compute the LU decomposition of an n×n matrix with partial pivoting.
///
/// Returns `LuDecomposition` with the combined LU matrix, pivot vector,
/// swap count, and singularity flag.
pub fn lu_decompose(a: &DenseMat) -> LuDecomposition {
    assert_eq!(a.rows, a.cols, "LU requires a square matrix");
    let n = a.rows;
    let mut lu = a.clone();
    let mut pivot: Vec<usize> = (0..n).collect();
    let mut swaps = 0usize;
    let mut singular = false;

    for k in 0..n {
        // Partial pivoting: find row with largest absolute value in column k.
        let mut max_val = lu.get(k, k).abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu.get(i, k).abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < PIVOT_TOL {
            singular = true;
            continue;
        }

        // Swap rows if needed.
        if max_row != k {
            pivot.swap(k, max_row);
            for j in 0..n {
                let tmp = lu.get(k, j);
                lu.set(k, j, lu.get(max_row, j));
                lu.set(max_row, j, tmp);
            }
            swaps += 1;
        }

        // Eliminate below.
        let pivot_val = lu.get(k, k);
        for i in (k + 1)..n {
            let factor = lu.get(i, k) / pivot_val;
            lu.set(i, k, factor); // Store L entry.
            for j in (k + 1)..n {
                let val = lu.get(i, j) - factor * lu.get(k, j);
                lu.set(i, j, val);
            }
        }
    }

    LuDecomposition {
        lu,
        pivot,
        swaps,
        singular,
    }
}

/// In-place LU decomposition: modifies the input matrix directly.
pub fn lu_decompose_inplace(a: &mut DenseMat) -> (Vec<usize>, usize, bool) {
    assert_eq!(a.rows, a.cols);
    let n = a.rows;
    let mut pivot: Vec<usize> = (0..n).collect();
    let mut swaps = 0usize;
    let mut singular = false;

    for k in 0..n {
        let mut max_val = a.get(k, k).abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a.get(i, k).abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < PIVOT_TOL {
            singular = true;
            continue;
        }

        if max_row != k {
            pivot.swap(k, max_row);
            for j in 0..n {
                let tmp = a.get(k, j);
                a.set(k, j, a.get(max_row, j));
                a.set(max_row, j, tmp);
            }
            swaps += 1;
        }

        let pivot_val = a.get(k, k);
        for i in (k + 1)..n {
            let factor = a.get(i, k) / pivot_val;
            a.set(i, k, factor);
            for j in (k + 1)..n {
                let val = a.get(i, j) - factor * a.get(k, j);
                a.set(i, j, val);
            }
        }
    }
    (pivot, swaps, singular)
}

// ── Substitution ──────────────────────────────────────────────

/// Forward substitution: solve Ly = Pb.
/// `lu` is the combined LU matrix, `pivot` is the permutation, `b` is the RHS.
pub fn forward_substitute(lu: &DenseMat, pivot: &[usize], b: &[f64]) -> Vec<f64> {
    let n = lu.rows;
    let mut y = vec![0.0; n];
    // Apply permutation.
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[pivot[i]];
    }
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu.get(i, j) * y[j];
        }
        y[i] = sum;
    }
    y
}

/// Backward substitution: solve Ux = y.
pub fn backward_substitute(lu: &DenseMat, y: &[f64]) -> Vec<f64> {
    let n = lu.rows;
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu.get(i, j) * x[j];
        }
        let diag = lu.get(i, i);
        x[i] = if diag.abs() > PIVOT_TOL {
            sum / diag
        } else {
            0.0
        };
    }
    x
}

// ── Solve ─────────────────────────────────────────────────────

/// Solve Ax = b using LU decomposition with partial pivoting.
pub fn lu_solve(a: &DenseMat, b: &[f64]) -> Option<Vec<f64>> {
    let dec = lu_decompose(a);
    if dec.singular {
        return None;
    }
    let y = forward_substitute(&dec.lu, &dec.pivot, b);
    Some(backward_substitute(&dec.lu, &y))
}

/// Solve AX = B for multiple right-hand sides (columns of B).
pub fn lu_solve_multi(a: &DenseMat, b_mat: &DenseMat) -> Option<DenseMat> {
    assert_eq!(a.rows, b_mat.rows);
    let dec = lu_decompose(a);
    if dec.singular {
        return None;
    }
    let n = a.rows;
    let m = b_mat.cols;
    let mut result = DenseMat::zeros(n, m);
    for j in 0..m {
        let b_col = b_mat.col_vec(j);
        let y = forward_substitute(&dec.lu, &dec.pivot, &b_col);
        let x = backward_substitute(&dec.lu, &y);
        for i in 0..n {
            result.set(i, j, x[i]);
        }
    }
    Some(result)
}

// ── Determinant ───────────────────────────────────────────────

/// Compute the determinant of a square matrix using LU decomposition.
pub fn determinant(a: &DenseMat) -> f64 {
    let dec = lu_decompose(a);
    if dec.singular {
        return 0.0;
    }
    let n = a.rows;
    let mut det = if dec.swaps % 2 == 0 { 1.0 } else { -1.0 };
    for i in 0..n {
        det *= dec.lu.get(i, i);
    }
    det
}

// ── Inverse ───────────────────────────────────────────────────

/// Compute the inverse of a square matrix using LU decomposition.
/// Returns `None` if the matrix is singular.
pub fn inverse(a: &DenseMat) -> Option<DenseMat> {
    let n = a.rows;
    let id = DenseMat::identity(n);
    lu_solve_multi(a, &id)
}

/// Extract the L matrix from a combined LU matrix.
pub fn extract_l(lu: &LuDecomposition) -> DenseMat {
    let n = lu.lu.rows;
    let mut l = DenseMat::identity(n);
    for i in 1..n {
        for j in 0..i {
            l.set(i, j, lu.lu.get(i, j));
        }
    }
    l
}

/// Extract the U matrix from a combined LU matrix.
pub fn extract_u(lu: &LuDecomposition) -> DenseMat {
    let n = lu.lu.rows;
    let mut u = DenseMat::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            u.set(i, j, lu.lu.get(i, j));
        }
    }
    u
}

/// Build the permutation matrix P from the pivot vector.
pub fn permutation_matrix(pivot: &[usize], n: usize) -> DenseMat {
    let mut p = DenseMat::zeros(n, n);
    for i in 0..n {
        p.set(i, pivot[i], 1.0);
    }
    p
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

    /// Multiply two dense matrices.
    fn mat_mul(a: &DenseMat, b: &DenseMat) -> DenseMat {
        assert_eq!(a.cols, b.rows);
        let mut c = DenseMat::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut s = 0.0;
                for k in 0..a.cols {
                    s += a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, s);
            }
        }
        c
    }

    fn test_mat_2x2() -> DenseMat {
        DenseMat::from_data(2, 2, vec![4.0, 3.0, 6.0, 3.0])
    }

    fn test_mat_3x3() -> DenseMat {
        DenseMat::from_data(
            3,
            3,
            vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
        )
    }

    #[test]
    fn test_lu_decompose_2x2() {
        let a = test_mat_2x2();
        let dec = lu_decompose(&a);
        assert!(!dec.singular);
    }

    #[test]
    fn test_lu_pa_equals_lu() {
        let a = test_mat_3x3();
        let dec = lu_decompose(&a);
        let l = extract_l(&dec);
        let u = extract_u(&dec);
        let p = permutation_matrix(&dec.pivot, 3);
        let pa = mat_mul(&p, &a);
        let lu_product = mat_mul(&l, &u);
        assert!(mat_approx_eq(&pa, &lu_product, 1e-10));
    }

    #[test]
    fn test_solve_2x2() {
        let a = test_mat_2x2();
        let b = vec![1.0, 2.0];
        let x = lu_solve(&a, &b).unwrap();
        // Verify: A*x = b
        let ax0 = a.get(0, 0) * x[0] + a.get(0, 1) * x[1];
        let ax1 = a.get(1, 0) * x[0] + a.get(1, 1) * x[1];
        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
    }

    #[test]
    fn test_solve_3x3() {
        let a = test_mat_3x3();
        let b = vec![1.0, 2.0, 3.0];
        let x = lu_solve(&a, &b).unwrap();
        // Verify A*x = b
        for i in 0..3 {
            let mut s = 0.0;
            for j in 0..3 {
                s += a.get(i, j) * x[j];
            }
            assert!(approx_eq(s, b[i], 1e-10));
        }
    }

    #[test]
    fn test_determinant_2x2() {
        let a = test_mat_2x2();
        let det = determinant(&a);
        // det([[4,3],[6,3]]) = 4*3 - 3*6 = -6
        assert!(approx_eq(det, -6.0, 1e-10));
    }

    #[test]
    fn test_determinant_3x3() {
        let a = test_mat_3x3();
        let det = determinant(&a);
        // det = 2(27-21) - 1(36-24) + 1(28-24) = 12 - 12 + 4 = 4
        assert!(approx_eq(det, 4.0, 1e-10));
    }

    #[test]
    fn test_determinant_identity() {
        let id = DenseMat::identity(4);
        assert!(approx_eq(determinant(&id), 1.0, 1e-10));
    }

    #[test]
    fn test_inverse_2x2() {
        let a = test_mat_2x2();
        let inv = inverse(&a).unwrap();
        let prod = mat_mul(&a, &inv);
        let id = DenseMat::identity(2);
        assert!(mat_approx_eq(&prod, &id, 1e-10));
    }

    #[test]
    fn test_inverse_3x3() {
        let a = test_mat_3x3();
        let inv = inverse(&a).unwrap();
        let prod = mat_mul(&a, &inv);
        let id = DenseMat::identity(3);
        assert!(mat_approx_eq(&prod, &id, 1e-10));
    }

    #[test]
    fn test_singular_matrix() {
        // Singular: rows are linearly dependent
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let result = lu_solve(&a, &[1.0, 2.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_inverse_singular() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let inv = inverse(&a);
        assert!(inv.is_none());
    }

    #[test]
    fn test_solve_identity() {
        let id = DenseMat::identity(3);
        let b = vec![5.0, 6.0, 7.0];
        let x = lu_solve(&id, &b).unwrap();
        for i in 0..3 {
            assert!(approx_eq(x[i], b[i], 1e-10));
        }
    }

    #[test]
    fn test_multiple_rhs() {
        let a = test_mat_3x3();
        let b = DenseMat::from_data(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let x = lu_solve_multi(&a, &b).unwrap();
        // Verify each column: A * x_j = b_j
        for j in 0..2 {
            for i in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a.get(i, k) * x.get(k, j);
                }
                assert!(approx_eq(s, b.get(i, j), 1e-10));
            }
        }
    }

    #[test]
    fn test_inplace_decompose() {
        let a = test_mat_3x3();
        let mut a_copy = a.clone();
        let (pivot, swaps, singular) = lu_decompose_inplace(&mut a_copy);
        assert!(!singular);
        // pivot and swaps should be consistent with the non-inplace version
        let dec = lu_decompose(&a);
        assert_eq!(pivot, dec.pivot);
        assert_eq!(swaps, dec.swaps);
    }

    #[test]
    fn test_extract_l_unit_diagonal() {
        let a = test_mat_3x3();
        let dec = lu_decompose(&a);
        let l = extract_l(&dec);
        for i in 0..3 {
            assert!(approx_eq(l.get(i, i), 1.0, 1e-12));
        }
    }

    #[test]
    fn test_extract_u_upper_triangular() {
        let a = test_mat_3x3();
        let dec = lu_decompose(&a);
        let u = extract_u(&dec);
        for i in 0..3 {
            for j in 0..i {
                assert!(approx_eq(u.get(i, j), 0.0, 1e-12));
            }
        }
    }

    #[test]
    fn test_permutation_matrix_is_valid() {
        let a = test_mat_3x3();
        let dec = lu_decompose(&a);
        let p = permutation_matrix(&dec.pivot, 3);
        // Each row and column should sum to 1
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| p.get(i, j)).sum();
            let col_sum: f64 = (0..3).map(|j| p.get(j, i)).sum();
            assert!(approx_eq(row_sum, 1.0, 1e-12));
            assert!(approx_eq(col_sum, 1.0, 1e-12));
        }
    }

    #[test]
    fn test_det_singular_is_zero() {
        let a = DenseMat::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        assert!(approx_eq(determinant(&a), 0.0, 1e-10));
    }

    #[test]
    fn test_4x4_solve() {
        let a = DenseMat::from_data(
            4,
            4,
            vec![
                2.0, 1.0, 1.0, 0.0,
                4.0, 3.0, 3.0, 1.0,
                8.0, 7.0, 9.0, 5.0,
                6.0, 7.0, 9.0, 8.0,
            ],
        );
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = lu_solve(&a, &b).unwrap();
        for i in 0..4 {
            let mut s = 0.0;
            for j in 0..4 {
                s += a.get(i, j) * x[j];
            }
            assert!(approx_eq(s, b[i], 1e-10));
        }
    }

    #[test]
    fn test_dense_mat_display() {
        let m = DenseMat::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = format!("{}", m);
        assert!(s.contains("1.000000"));
        assert!(s.contains("4.000000"));
    }
}
