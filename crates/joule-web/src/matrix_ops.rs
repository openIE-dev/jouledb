//! Matrix operations library — pure-Rust replacement for mathjs, numpy linear algebra.
//!
//! Dense matrix with add, multiply, transpose, determinant, inverse, LU decomposition,
//! and eigenvalue estimation (power iteration). Supports f32/f64 via generic trait.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ── Scalar trait ──────────────────────────────────────────────

/// Trait bounding the scalar types usable with `Matrix<T>`.
pub trait Scalar:
    Copy
    + Default
    + fmt::Debug
    + fmt::Display
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn epsilon() -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn abs(self) -> Self { f64::abs(self) }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    fn epsilon() -> Self { 1e-12 }
}

impl Scalar for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn abs(self) -> Self { f32::abs(self) }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn epsilon() -> Self { 1e-6 }
}

// ── Matrix ────────────────────────────────────────────────────

/// A dense matrix stored in row-major order.
#[derive(Debug, Clone)]
pub struct Matrix<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// Create a matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    /// Create a matrix from a 2D slice of rows.
    pub fn from_rows(rows: &[&[T]]) -> Self {
        if rows.is_empty() {
            return Self::zeros(0, 0);
        }
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in rows {
            assert_eq!(row.len(), ncols, "All rows must have equal length");
            data.extend_from_slice(row);
        }
        Self { rows: nrows, cols: ncols, data }
    }

    /// Create from a flat vec in row-major order.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "Data length must match rows*cols");
        Self { rows, cols, data }
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row * self.cols + col] = val;
    }

    /// Whether the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        result
    }

    /// Matrix addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<T> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Matrix subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<T> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|v| *v * s).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "Incompatible dimensions for multiplication");
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.data[i * self.cols + k];
                for j in 0..other.cols {
                    let cur = result.data[i * other.cols + j];
                    result.data[i * other.cols + j] = cur + a_ik * other.data[k * other.cols + j];
                }
            }
        }
        result
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> T {
        let mut sum = T::zero();
        for &v in &self.data {
            sum = sum + v * v;
        }
        sum.sqrt()
    }

    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> T {
        assert!(self.is_square(), "Trace requires a square matrix");
        let mut sum = T::zero();
        for i in 0..self.rows {
            sum = sum + self.data[i * self.cols + i];
        }
        sum
    }

    /// LU decomposition with partial pivoting.
    /// Returns (L, U, P) where P*A = L*U, P is a permutation vector.
    pub fn lu(&self) -> (Self, Self, Vec<usize>) {
        assert!(self.is_square(), "LU requires a square matrix");
        let n = self.rows;
        let mut a = self.data.clone();
        let mut perm: Vec<usize> = (0..n).collect();

        for col in 0..n {
            // Find pivot
            let mut max_val = T::zero();
            let mut max_row = col;
            for row in col..n {
                let val = a[row * n + col].abs();
                if val.to_f64() > max_val.to_f64() {
                    max_val = val;
                    max_row = row;
                }
            }
            // Swap rows
            if max_row != col {
                perm.swap(col, max_row);
                for j in 0..n {
                    let tmp = a[col * n + j];
                    a[col * n + j] = a[max_row * n + j];
                    a[max_row * n + j] = tmp;
                }
            }
            let pivot = a[col * n + col];
            if pivot.abs().to_f64() < T::epsilon().to_f64() {
                continue;
            }
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                a[row * n + col] = factor; // store L factor
                for j in (col + 1)..n {
                    let val = a[col * n + j];
                    a[row * n + j] = a[row * n + j] - factor * val;
                }
            }
        }

        let mut l = Self::identity(n);
        let mut u = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if j >= i {
                    u.data[i * n + j] = a[i * n + j];
                } else {
                    l.data[i * n + j] = a[i * n + j];
                }
            }
        }
        (l, u, perm)
    }

    /// Determinant via LU decomposition.
    pub fn determinant(&self) -> T {
        assert!(self.is_square(), "Determinant requires a square matrix");
        let n = self.rows;
        if n == 0 {
            return T::one();
        }
        if n == 1 {
            return self.data[0];
        }
        if n == 2 {
            return self.data[0] * self.data[3] - self.data[1] * self.data[2];
        }

        let (_l, u, perm) = self.lu();
        let mut det = T::one();
        for i in 0..n {
            det = det * u.data[i * n + i];
        }
        // Count parity of permutation
        let mut swaps = 0usize;
        let mut visited = vec![false; n];
        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut cycle_len = 0usize;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
                cycle_len += 1;
            }
            if cycle_len > 1 {
                swaps += cycle_len - 1;
            }
        }
        if swaps % 2 == 1 {
            det = T::zero() - det;
        }
        det
    }

    /// Inverse via Gauss-Jordan elimination.
    /// Returns None if the matrix is singular.
    pub fn inverse(&self) -> Option<Self> {
        assert!(self.is_square(), "Inverse requires a square matrix");
        let n = self.rows;
        if n == 0 {
            return Some(Self::zeros(0, 0));
        }

        // Augmented matrix [A | I]
        let mut aug = vec![T::zero(); n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.data[i * n + j];
            }
            aug[i * 2 * n + n + i] = T::one();
        }

        for col in 0..n {
            // Find pivot
            let mut max_val = T::zero();
            let mut max_row = col;
            for row in col..n {
                let val = aug[row * 2 * n + col].abs();
                if val.to_f64() > max_val.to_f64() {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val.to_f64() < T::epsilon().to_f64() {
                return None; // Singular
            }
            // Swap rows
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug[col * 2 * n + j];
                    aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                    aug[max_row * 2 * n + j] = tmp;
                }
            }
            // Scale pivot row
            let pivot = aug[col * 2 * n + col];
            for j in 0..(2 * n) {
                aug[col * 2 * n + j] = aug[col * 2 * n + j] / pivot;
            }
            // Eliminate column
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n + col];
                for j in 0..(2 * n) {
                    let pivot_val = aug[col * 2 * n + j];
                    aug[row * 2 * n + j] = aug[row * 2 * n + j] - factor * pivot_val;
                }
            }
        }

        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.data[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        Some(inv)
    }

    /// Dominant eigenvalue via power iteration.
    /// Returns (eigenvalue, eigenvector) or None if it doesn't converge.
    pub fn power_iteration(&self, max_iter: usize, tol: f64) -> Option<(T, Vec<T>)> {
        assert!(self.is_square(), "Power iteration requires a square matrix");
        let n = self.rows;
        if n == 0 {
            return None;
        }

        // Start with a random-ish vector
        let mut b: Vec<T> = (0..n).map(|i| T::from_f64((i as f64 + 1.0) / n as f64)).collect();

        let mut eigenvalue = T::zero();

        for _ in 0..max_iter {
            // Multiply A * b
            let mut ab = vec![T::zero(); n];
            for i in 0..n {
                let mut sum = T::zero();
                for j in 0..n {
                    sum = sum + self.data[i * n + j] * b[j];
                }
                ab[i] = sum;
            }

            // Find max magnitude element
            let mut max_val = T::zero();
            for &v in &ab {
                if v.abs().to_f64() > max_val.abs().to_f64() {
                    max_val = v;
                }
            }
            if max_val.abs().to_f64() < T::epsilon().to_f64() {
                return None;
            }

            // Normalize
            for v in &mut ab {
                *v = *v / max_val;
            }

            // Check convergence
            let diff = (max_val.to_f64() - eigenvalue.to_f64()).abs();
            eigenvalue = max_val;
            b = ab;

            if diff < tol {
                return Some((eigenvalue, b));
            }
        }

        Some((eigenvalue, b))
    }

    /// Extract a row as a vector.
    pub fn row(&self, r: usize) -> Vec<T> {
        self.data[r * self.cols..(r + 1) * self.cols].to_vec()
    }

    /// Extract a column as a vector.
    pub fn col(&self, c: usize) -> Vec<T> {
        (0..self.rows).map(|r| self.data[r * self.cols + c]).collect()
    }

    /// Submatrix: remove row `skip_row` and column `skip_col`.
    pub fn minor(&self, skip_row: usize, skip_col: usize) -> Self {
        assert!(self.rows > 0 && self.cols > 0);
        let mut data = Vec::with_capacity((self.rows - 1) * (self.cols - 1));
        for r in 0..self.rows {
            if r == skip_row { continue; }
            for c in 0..self.cols {
                if c == skip_col { continue; }
                data.push(self.data[r * self.cols + c]);
            }
        }
        Self { rows: self.rows - 1, cols: self.cols - 1, data }
    }

    /// Element-wise comparison within tolerance.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter())
            .all(|(a, b)| (a.to_f64() - b.to_f64()).abs() < tol)
    }
}

impl<T: Scalar> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            write!(f, "[")?;
            for c in 0..self.cols {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{}", self.data[r * self.cols + c])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mat2x2(a: f64, b: f64, c: f64, d: f64) -> Matrix<f64> {
        Matrix::from_vec(2, 2, vec![a, b, c, d])
    }

    #[test]
    fn zeros_and_identity() {
        let z = Matrix::<f64>::zeros(3, 3);
        assert_eq!(z.data.len(), 9);
        assert_eq!(z.get(1, 1), 0.0);

        let id = Matrix::<f64>::identity(3);
        assert_eq!(id.get(0, 0), 1.0);
        assert_eq!(id.get(0, 1), 0.0);
        assert_eq!(id.get(1, 1), 1.0);
    }

    #[test]
    fn from_rows() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.get(1, 0), 3.0);
    }

    #[test]
    fn transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(2, 0), 3.0);
    }

    #[test]
    fn add_and_sub() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = mat2x2(5.0, 6.0, 7.0, 8.0);
        let sum = a.add(&b);
        assert_eq!(sum.get(0, 0), 6.0);
        assert_eq!(sum.get(1, 1), 12.0);

        let diff = b.sub(&a);
        assert_eq!(diff.get(0, 0), 4.0);
        assert_eq!(diff.get(1, 1), 4.0);
    }

    #[test]
    fn scale() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        let s = m.scale(2.0);
        assert_eq!(s.get(0, 0), 2.0);
        assert_eq!(s.get(1, 1), 8.0);
    }

    #[test]
    fn multiply() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = mat2x2(5.0, 6.0, 7.0, 8.0);
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn multiply_non_square() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.mul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.get(0, 0), 58.0);
    }

    #[test]
    fn determinant_2x2() {
        let m = mat2x2(3.0, 8.0, 4.0, 6.0);
        let d = m.determinant();
        assert!((d - (-14.0)).abs() < 1e-10);
    }

    #[test]
    fn determinant_3x3() {
        let m = Matrix::from_vec(3, 3, vec![
            6.0, 1.0, 1.0,
            4.0, -2.0, 5.0,
            2.0, 8.0, 7.0,
        ]);
        let d = m.determinant();
        assert!((d - (-306.0)).abs() < 1e-8);
    }

    #[test]
    fn lu_decomposition() {
        let a = Matrix::from_vec(3, 3, vec![
            2.0, 1.0, 1.0,
            4.0, 3.0, 3.0,
            8.0, 7.0, 9.0,
        ]);
        let (l, u, perm) = a.lu();
        // Verify L*U = P*A
        let lu = l.mul(&u);
        // Build PA
        let n = a.rows;
        let mut pa = Matrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                pa.set(i, j, a.get(perm[i], j));
            }
        }
        assert!(lu.approx_eq(&pa, 1e-10));
    }

    #[test]
    fn inverse_2x2() {
        let m = mat2x2(4.0, 7.0, 2.0, 6.0);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        let id = Matrix::<f64>::identity(2);
        assert!(product.approx_eq(&id, 1e-10));
    }

    #[test]
    fn inverse_3x3() {
        let m = Matrix::from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0,
        ]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        let id = Matrix::<f64>::identity(3);
        assert!(product.approx_eq(&id, 1e-10));
    }

    #[test]
    fn singular_matrix_no_inverse() {
        let m = mat2x2(1.0, 2.0, 2.0, 4.0);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn trace() {
        let m = Matrix::from_vec(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 9.0,
        ]);
        assert_eq!(m.trace(), 15.0);
    }

    #[test]
    fn frobenius_norm() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((m.frobenius_norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn power_iteration_dominant_eigenvalue() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1
        let m = mat2x2(2.0, 1.0, 1.0, 2.0);
        let (eigenvalue, _eigenvector) = m.power_iteration(100, 1e-10).unwrap();
        assert!((eigenvalue - 3.0).abs() < 1e-6);
    }

    #[test]
    fn row_and_col() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(m.col(1), vec![2.0, 5.0]);
    }

    #[test]
    fn minor_matrix() {
        let m = Matrix::from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let sub = m.minor(0, 0);
        assert_eq!(sub.rows, 2);
        assert_eq!(sub.get(0, 0), 5.0);
        assert_eq!(sub.get(1, 1), 9.0);
    }

    #[test]
    fn identity_is_own_inverse() {
        let id = Matrix::<f64>::identity(4);
        let inv = id.inverse().unwrap();
        assert!(id.approx_eq(&inv, 1e-12));
    }

    #[test]
    fn f32_support() {
        let m = Matrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let det = m.determinant();
        assert!((det - (-2.0_f32)).abs() < 1e-5);
    }

    #[test]
    fn display_format() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        let s = format!("{}", m);
        assert!(s.contains("1"));
        assert!(s.contains("4"));
    }
}
