//! Sparse matrix in CSR (Compressed Sparse Row) format.
//!
//! Pure-Rust sparse matrix: construct from triplets, SpMV, SpMM, transpose,
//! diagonal extraction, row/column slicing, add/subtract, scale, identity,
//! symmetric check, density, and dense conversion.

use std::fmt;

// ── CSR Sparse Matrix ─────────────────────────────────────────

/// Sparse matrix stored in Compressed Sparse Row (CSR) format.
#[derive(Clone, PartialEq)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointer array (length nrows + 1).  `row_ptr[i]..row_ptr[i+1]` gives
    /// the index range into `col_idx` and `values` for row i.
    pub row_ptr: Vec<usize>,
    /// Column indices of non-zero entries.
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
}

impl fmt::Debug for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsrMatrix({}x{}, nnz={})",
            self.nrows,
            self.ncols,
            self.nnz()
        )
    }
}

/// A triplet (row, col, value) used for constructing a sparse matrix.
#[derive(Debug, Clone, Copy)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

// ── Construction ──────────────────────────────────────────────

impl CsrMatrix {
    /// Create an empty matrix of the given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Build a CSR matrix from triplets.  Duplicate entries for the same
    /// (row, col) are summed.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[Triplet]) -> Self {
        // Count entries per row.
        let mut row_counts = vec![0usize; nrows];
        for t in triplets {
            assert!(t.row < nrows && t.col < ncols, "triplet out of bounds");
            row_counts[t.row] += 1;
        }

        // Build row_ptr.
        let mut row_ptr = vec![0usize; nrows + 1];
        for i in 0..nrows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        let nnz = row_ptr[nrows];
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut cursor = row_ptr.clone();

        for t in triplets {
            let pos = cursor[t.row];
            col_idx[pos] = t.col;
            values[pos] = t.val;
            cursor[t.row] += 1;
        }

        // Sort each row by column index and merge duplicates.
        let mut sorted = Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        };
        sorted.sort_and_merge();
        sorted
    }

    /// Sort each row by column index and sum duplicate entries.
    fn sort_and_merge(&mut self) {
        let mut new_col: Vec<usize> = Vec::with_capacity(self.col_idx.len());
        let mut new_val: Vec<f64> = Vec::with_capacity(self.values.len());
        let mut new_ptr = vec![0usize; self.nrows + 1];

        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut pairs: Vec<(usize, f64)> = (start..end)
                .map(|k| (self.col_idx[k], self.values[k]))
                .collect();
            pairs.sort_by_key(|&(c, _)| c);

            // Merge duplicates.
            for &(c, v) in &pairs {
                if let Some(last_c) = new_col.last() {
                    if *last_c == c && new_ptr[i + 1] + new_col.len() - new_ptr[i] > 0 {
                        // Same column — check it is in the current row.
                        let row_start = new_ptr[i];
                        let cur_len = new_col.len();
                        if cur_len > row_start && new_col[cur_len - 1] == c {
                            *new_val.last_mut().unwrap() += v;
                            continue;
                        }
                    }
                }
                new_col.push(c);
                new_val.push(v);
            }
            new_ptr[i + 1] = new_col.len();
        }
        self.col_idx = new_col;
        self.values = new_val;
        self.row_ptr = new_ptr;
    }

    /// Create a sparse identity matrix of size n.
    pub fn identity(n: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            row_ptr.push(i);
            col_idx.push(i);
            values.push(1.0);
        }
        row_ptr.push(n);
        Self {
            nrows: n,
            ncols: n,
            row_ptr,
            col_idx,
            values,
        }
    }

    // ── Queries ───────────────────────────────────────────────

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Density = nnz / (nrows * ncols), or 0 if matrix is empty.
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Get element (i, j).  Returns 0 if not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] == col {
                return self.values[k];
            }
            if self.col_idx[k] > col {
                break;
            }
        }
        0.0
    }

    // ── Operations ────────────────────────────────────────────

    /// Sparse matrix-vector multiply: y = A * x.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols, "spmv: dimension mismatch");
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let mut sum = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = sum;
        }
        y
    }

    /// Sparse-sparse multiply: C = A * B, both in CSR.
    pub fn spmm(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(
            self.ncols, other.nrows,
            "spmm: inner dimensions must match"
        );
        let mut triplets = Vec::new();
        for i in 0..self.nrows {
            // Accumulate row i of C in a dense scratch vector.
            let mut acc = vec![0.0f64; other.ncols];
            for ka in self.row_ptr[i]..self.row_ptr[i + 1] {
                let k = self.col_idx[ka];
                let a_ik = self.values[ka];
                for kb in other.row_ptr[k]..other.row_ptr[k + 1] {
                    acc[other.col_idx[kb]] += a_ik * other.values[kb];
                }
            }
            for (j, &v) in acc.iter().enumerate() {
                if v.abs() > 1e-15 {
                    triplets.push(Triplet {
                        row: i,
                        col: j,
                        val: v,
                    });
                }
            }
        }
        CsrMatrix::from_triplets(self.nrows, other.ncols, &triplets)
    }

    /// Transpose: returns A^T in CSR.
    pub fn transpose(&self) -> CsrMatrix {
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.nrows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push(Triplet {
                    row: self.col_idx[k],
                    col: i,
                    val: self.values[k],
                });
            }
        }
        CsrMatrix::from_triplets(self.ncols, self.nrows, &triplets)
    }

    /// Extract the diagonal as a dense vector.
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.nrows.min(self.ncols);
        let mut diag = vec![0.0; n];
        for i in 0..n {
            diag[i] = self.get(i, i);
        }
        diag
    }

    /// Extract row `i` as a dense vector.
    pub fn row_dense(&self, i: usize) -> Vec<f64> {
        assert!(i < self.nrows, "row out of bounds");
        let mut row = vec![0.0; self.ncols];
        for k in self.row_ptr[i]..self.row_ptr[i + 1] {
            row[self.col_idx[k]] = self.values[k];
        }
        row
    }

    /// Extract column `j` as a dense vector.
    pub fn col_dense(&self, j: usize) -> Vec<f64> {
        assert!(j < self.ncols, "column out of bounds");
        let mut col = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            col[i] = self.get(i, j);
        }
        col
    }

    /// Extract a sub-matrix defined by row and column index slices.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> CsrMatrix {
        let col_set: std::collections::HashMap<usize, usize> =
            cols.iter().enumerate().map(|(new_j, &old_j)| (old_j, new_j)).collect();
        let mut triplets = Vec::new();
        for (new_i, &old_i) in rows.iter().enumerate() {
            for k in self.row_ptr[old_i]..self.row_ptr[old_i + 1] {
                if let Some(&new_j) = col_set.get(&self.col_idx[k]) {
                    triplets.push(Triplet {
                        row: new_i,
                        col: new_j,
                        val: self.values[k],
                    });
                }
            }
        }
        CsrMatrix::from_triplets(rows.len(), cols.len(), &triplets)
    }

    /// Convert to a dense row-major matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for i in 0..self.nrows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                dense[i][self.col_idx[k]] = self.values[k];
            }
        }
        dense
    }

    /// Add two sparse matrices of the same dimensions.
    pub fn add(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        let mut triplets = Vec::with_capacity(self.nnz() + other.nnz());
        for i in 0..self.nrows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push(Triplet {
                    row: i,
                    col: self.col_idx[k],
                    val: self.values[k],
                });
            }
            for k in other.row_ptr[i]..other.row_ptr[i + 1] {
                triplets.push(Triplet {
                    row: i,
                    col: other.col_idx[k],
                    val: other.values[k],
                });
            }
        }
        CsrMatrix::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Subtract: self - other.
    pub fn sub(&self, other: &CsrMatrix) -> CsrMatrix {
        self.add(&other.scale(-1.0))
    }

    /// Scale every value by a scalar.
    pub fn scale(&self, s: f64) -> CsrMatrix {
        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values: self.values.iter().map(|v| v * s).collect(),
        }
    }

    /// Check if the matrix is symmetric (A == A^T) within tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        for i in 0..self.nrows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                let a_ij = self.values[k];
                let a_ji = self.get(j, i);
                if (a_ij - a_ji).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Frobenius norm: sqrt(sum of squares of all entries).
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// ── Display ───────────────────────────────────────────────────

impl fmt::Display for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:8.4}", self.get(i, j))?;
            }
            if i + 1 < self.nrows {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn triplet(r: usize, c: usize, v: f64) -> Triplet {
        Triplet {
            row: r,
            col: c,
            val: v,
        }
    }

    #[test]
    fn test_zeros() {
        let m = CsrMatrix::zeros(3, 4);
        assert_eq!(m.nrows, 3);
        assert_eq!(m.ncols, 4);
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn test_identity() {
        let id = CsrMatrix::identity(4);
        assert_eq!(id.nnz(), 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(id.get(i, j), expected, 1e-12));
            }
        }
    }

    #[test]
    fn test_from_triplets_basic() {
        let t = vec![triplet(0, 0, 1.0), triplet(0, 2, 3.0), triplet(1, 1, 5.0)];
        let m = CsrMatrix::from_triplets(2, 3, &t);
        assert_eq!(m.nnz(), 3);
        assert!(approx_eq(m.get(0, 0), 1.0, 1e-12));
        assert!(approx_eq(m.get(0, 1), 0.0, 1e-12));
        assert!(approx_eq(m.get(0, 2), 3.0, 1e-12));
        assert!(approx_eq(m.get(1, 1), 5.0, 1e-12));
    }

    #[test]
    fn test_duplicate_triplets_sum() {
        let t = vec![
            triplet(0, 0, 1.0),
            triplet(0, 0, 2.0),
            triplet(0, 0, 3.0),
        ];
        let m = CsrMatrix::from_triplets(1, 1, &t);
        assert!(approx_eq(m.get(0, 0), 6.0, 1e-12));
    }

    #[test]
    fn test_spmv() {
        // [[2, 0, 1], [0, 3, 0]] * [1, 2, 3] = [5, 6]
        let t = vec![triplet(0, 0, 2.0), triplet(0, 2, 1.0), triplet(1, 1, 3.0)];
        let m = CsrMatrix::from_triplets(2, 3, &t);
        let y = m.spmv(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(y[0], 5.0, 1e-12));
        assert!(approx_eq(y[1], 6.0, 1e-12));
    }

    #[test]
    fn test_spmv_identity() {
        let id = CsrMatrix::identity(3);
        let x = vec![7.0, 8.0, 9.0];
        let y = id.spmv(&x);
        for i in 0..3 {
            assert!(approx_eq(y[i], x[i], 1e-12));
        }
    }

    #[test]
    fn test_spmm_identity() {
        let t = vec![triplet(0, 0, 2.0), triplet(0, 1, 3.0), triplet(1, 0, 4.0)];
        let a = CsrMatrix::from_triplets(2, 2, &t);
        let id = CsrMatrix::identity(2);
        let c = a.spmm(&id);
        assert!(approx_eq(c.get(0, 0), 2.0, 1e-12));
        assert!(approx_eq(c.get(0, 1), 3.0, 1e-12));
        assert!(approx_eq(c.get(1, 0), 4.0, 1e-12));
    }

    #[test]
    fn test_spmm_product() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[19,22],[43,50]]
        let ta = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0),
            triplet(1, 0, 3.0), triplet(1, 1, 4.0),
        ];
        let tb = vec![
            triplet(0, 0, 5.0), triplet(0, 1, 6.0),
            triplet(1, 0, 7.0), triplet(1, 1, 8.0),
        ];
        let a = CsrMatrix::from_triplets(2, 2, &ta);
        let b = CsrMatrix::from_triplets(2, 2, &tb);
        let c = a.spmm(&b);
        assert!(approx_eq(c.get(0, 0), 19.0, 1e-10));
        assert!(approx_eq(c.get(0, 1), 22.0, 1e-10));
        assert!(approx_eq(c.get(1, 0), 43.0, 1e-10));
        assert!(approx_eq(c.get(1, 1), 50.0, 1e-10));
    }

    #[test]
    fn test_transpose() {
        let t = vec![triplet(0, 1, 5.0), triplet(1, 0, 7.0), triplet(1, 2, 9.0)];
        let a = CsrMatrix::from_triplets(2, 3, &t);
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert!(approx_eq(at.get(1, 0), 5.0, 1e-12));
        assert!(approx_eq(at.get(0, 1), 7.0, 1e-12));
        assert!(approx_eq(at.get(2, 1), 9.0, 1e-12));
    }

    #[test]
    fn test_diagonal() {
        let t = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0),
            triplet(1, 1, 3.0), triplet(2, 2, 5.0),
        ];
        let m = CsrMatrix::from_triplets(3, 3, &t);
        let d = m.diagonal();
        assert!(approx_eq(d[0], 1.0, 1e-12));
        assert!(approx_eq(d[1], 3.0, 1e-12));
        assert!(approx_eq(d[2], 5.0, 1e-12));
    }

    #[test]
    fn test_row_dense() {
        let t = vec![triplet(0, 0, 1.0), triplet(0, 2, 3.0), triplet(1, 1, 5.0)];
        let m = CsrMatrix::from_triplets(2, 3, &t);
        let r = m.row_dense(0);
        assert!(approx_eq(r[0], 1.0, 1e-12));
        assert!(approx_eq(r[1], 0.0, 1e-12));
        assert!(approx_eq(r[2], 3.0, 1e-12));
    }

    #[test]
    fn test_col_dense() {
        let t = vec![triplet(0, 1, 2.0), triplet(1, 1, 4.0), triplet(2, 0, 6.0)];
        let m = CsrMatrix::from_triplets(3, 2, &t);
        let c = m.col_dense(1);
        assert!(approx_eq(c[0], 2.0, 1e-12));
        assert!(approx_eq(c[1], 4.0, 1e-12));
        assert!(approx_eq(c[2], 0.0, 1e-12));
    }

    #[test]
    fn test_to_dense() {
        let t = vec![triplet(0, 0, 1.0), triplet(1, 1, 2.0)];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        let d = m.to_dense();
        assert!(approx_eq(d[0][0], 1.0, 1e-12));
        assert!(approx_eq(d[0][1], 0.0, 1e-12));
        assert!(approx_eq(d[1][0], 0.0, 1e-12));
        assert!(approx_eq(d[1][1], 2.0, 1e-12));
    }

    #[test]
    fn test_add() {
        let t1 = vec![triplet(0, 0, 1.0), triplet(1, 1, 2.0)];
        let t2 = vec![triplet(0, 0, 3.0), triplet(1, 0, 4.0)];
        let a = CsrMatrix::from_triplets(2, 2, &t1);
        let b = CsrMatrix::from_triplets(2, 2, &t2);
        let c = a.add(&b);
        assert!(approx_eq(c.get(0, 0), 4.0, 1e-12));
        assert!(approx_eq(c.get(1, 0), 4.0, 1e-12));
        assert!(approx_eq(c.get(1, 1), 2.0, 1e-12));
    }

    #[test]
    fn test_sub() {
        let t1 = vec![triplet(0, 0, 5.0), triplet(1, 1, 3.0)];
        let t2 = vec![triplet(0, 0, 2.0), triplet(1, 1, 1.0)];
        let a = CsrMatrix::from_triplets(2, 2, &t1);
        let b = CsrMatrix::from_triplets(2, 2, &t2);
        let c = a.sub(&b);
        assert!(approx_eq(c.get(0, 0), 3.0, 1e-12));
        assert!(approx_eq(c.get(1, 1), 2.0, 1e-12));
    }

    #[test]
    fn test_scale() {
        let t = vec![triplet(0, 0, 2.0), triplet(1, 1, 4.0)];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        let s = m.scale(3.0);
        assert!(approx_eq(s.get(0, 0), 6.0, 1e-12));
        assert!(approx_eq(s.get(1, 1), 12.0, 1e-12));
    }

    #[test]
    fn test_symmetric() {
        let t = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0),
            triplet(1, 0, 2.0), triplet(1, 1, 3.0),
        ];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        assert!(m.is_symmetric(1e-12));
    }

    #[test]
    fn test_not_symmetric() {
        let t = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0),
            triplet(1, 0, 3.0), triplet(1, 1, 4.0),
        ];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        assert!(!m.is_symmetric(1e-12));
    }

    #[test]
    fn test_density() {
        let t = vec![triplet(0, 0, 1.0), triplet(1, 1, 2.0)];
        let m = CsrMatrix::from_triplets(3, 3, &t);
        assert!(approx_eq(m.density(), 2.0 / 9.0, 1e-12));
    }

    #[test]
    fn test_frobenius_norm() {
        let t = vec![triplet(0, 0, 3.0), triplet(1, 1, 4.0)];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        assert!(approx_eq(m.frobenius_norm(), 5.0, 1e-12));
    }

    #[test]
    fn test_submatrix() {
        // 3x3 matrix, extract rows [0,2], cols [1,2]
        let t = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0), triplet(0, 2, 3.0),
            triplet(1, 0, 4.0), triplet(1, 1, 5.0), triplet(1, 2, 6.0),
            triplet(2, 0, 7.0), triplet(2, 1, 8.0), triplet(2, 2, 9.0),
        ];
        let m = CsrMatrix::from_triplets(3, 3, &t);
        let s = m.submatrix(&[0, 2], &[1, 2]);
        assert_eq!(s.nrows, 2);
        assert_eq!(s.ncols, 2);
        assert!(approx_eq(s.get(0, 0), 2.0, 1e-12));
        assert!(approx_eq(s.get(0, 1), 3.0, 1e-12));
        assert!(approx_eq(s.get(1, 0), 8.0, 1e-12));
        assert!(approx_eq(s.get(1, 1), 9.0, 1e-12));
    }

    #[test]
    fn test_transpose_of_transpose() {
        let t = vec![
            triplet(0, 0, 1.0), triplet(0, 1, 2.0),
            triplet(1, 0, 3.0), triplet(1, 1, 4.0),
        ];
        let a = CsrMatrix::from_triplets(2, 2, &t);
        let att = a.transpose().transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(a.get(i, j), att.get(i, j), 1e-12));
            }
        }
    }

    #[test]
    fn test_display_format() {
        let t = vec![triplet(0, 0, 1.0), triplet(1, 1, 2.0)];
        let m = CsrMatrix::from_triplets(2, 2, &t);
        let s = format!("{}", m);
        assert!(s.contains("1.0000"));
        assert!(s.contains("2.0000"));
    }

    #[test]
    fn test_empty_matrix_density() {
        let m = CsrMatrix::zeros(0, 0);
        assert!(approx_eq(m.density(), 0.0, 1e-12));
    }

    #[test]
    fn test_rectangular_transpose() {
        let t = vec![triplet(0, 0, 1.0), triplet(0, 2, 2.0), triplet(1, 1, 3.0)];
        let m = CsrMatrix::from_triplets(2, 3, &t);
        let mt = m.transpose();
        assert_eq!(mt.nrows, 3);
        assert_eq!(mt.ncols, 2);
        assert!(approx_eq(mt.get(0, 0), 1.0, 1e-12));
        assert!(approx_eq(mt.get(2, 0), 2.0, 1e-12));
        assert!(approx_eq(mt.get(1, 1), 3.0, 1e-12));
    }
}
