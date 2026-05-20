//! Sparse matrix — CSR/CSC formats, addition, multiplication, transpose,
//! diagonal extraction, conversion between formats, iterators.
//!
//! Pure-Rust sparse matrix library supporting Compressed Sparse Row (CSR) and
//! Compressed Sparse Column (CSC) formats with arithmetic, SpMV, SpMM,
//! transpose, format conversion, diagonal extraction, and iterators.

use std::fmt;

// ── COO format ───────────────────────────────────────────────────

/// Coordinate (COO) sparse matrix — stores (row, col, value) triples.
#[derive(Clone, PartialEq)]
pub struct CooMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl fmt::Debug for CooMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CooMatrix({}x{}, nnz={})", self.rows, self.cols, self.values.len())
    }
}

impl CooMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_indices: Vec::new(), col_indices: Vec::new(), values: Vec::new() }
    }

    pub fn from_triplets(
        rows: usize, cols: usize,
        row_indices: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>,
    ) -> Self {
        assert_eq!(row_indices.len(), col_indices.len());
        assert_eq!(row_indices.len(), values.len());
        for (&r, &c) in row_indices.iter().zip(col_indices.iter()) {
            assert!(r < rows, "row index {} >= rows {}", r, rows);
            assert!(c < cols, "col index {} >= cols {}", c, cols);
        }
        Self { rows, cols, row_indices, col_indices, values }
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows && col < self.cols);
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    pub fn nnz(&self) -> usize { self.values.len() }

    /// Convert to CSR, summing duplicate entries.
    pub fn to_csr(&self) -> CsrMatrix {
        let nnz = self.nnz();
        let mut order: Vec<usize> = (0..nnz).collect();
        order.sort_by(|&a, &b| {
            self.row_indices[a].cmp(&self.row_indices[b])
                .then(self.col_indices[a].cmp(&self.col_indices[b]))
        });

        let mut row_ptr = vec![0usize; self.rows + 1];
        let mut col_idx = Vec::with_capacity(nnz);
        let mut vals = Vec::with_capacity(nnz);

        for &i in &order {
            let r = self.row_indices[i];
            let c = self.col_indices[i];
            let v = self.values[i];
            // Sum duplicates
            if !col_idx.is_empty() && row_ptr[r + 1] == 0 {
                // first entry in this row (row_ptr not yet set)
            }
            if !vals.is_empty() {
                let prev_start = row_ptr[r];
                let cur_len = col_idx.len();
                if cur_len > prev_start && col_idx[cur_len - 1] == c {
                    // check previous entry is in same row
                    let mut same_row = true;
                    for rr in r + 1..=self.rows {
                        if row_ptr[rr] > 0 && row_ptr[rr] <= cur_len - 1 {
                            same_row = false;
                            break;
                        }
                        if rr == self.rows { break; }
                        if row_ptr[rr] > 0 { break; }
                    }
                    if same_row {
                        vals[cur_len - 1] += v;
                        continue;
                    }
                }
            }
            col_idx.push(c);
            vals.push(v);
            row_ptr[r + 1] = col_idx.len();
        }

        // Make row_ptr monotonically non-decreasing
        for i in 1..=self.rows {
            if row_ptr[i] < row_ptr[i - 1] {
                row_ptr[i] = row_ptr[i - 1];
            }
        }
        row_ptr[self.rows] = col_idx.len();
        for i in (0..self.rows).rev() {
            if row_ptr[i] > row_ptr[i + 1] {
                row_ptr[i] = row_ptr[i + 1];
            }
        }

        CsrMatrix { rows: self.rows, cols: self.cols, row_ptr, col_indices: col_idx, values: vals }
    }

    /// Convert to CSC, summing duplicate entries.
    pub fn to_csc(&self) -> CscMatrix {
        let transposed = CooMatrix::from_triplets(
            self.cols, self.rows,
            self.col_indices.clone(), self.row_indices.clone(), self.values.clone(),
        );
        let csr = transposed.to_csr();
        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            col_ptr: csr.row_ptr,
            row_indices: csr.col_indices,
            values: csr.values,
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.push(i, i, 1.0); }
        m
    }
}

// ── CSR format ───────────────────────────────────────────────────

/// Compressed Sparse Row matrix.
#[derive(Clone, PartialEq)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl fmt::Debug for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CsrMatrix({}x{}, nnz={})", self.rows, self.cols, self.values.len())
    }
}

impl CsrMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_ptr: vec![0; rows + 1], col_indices: Vec::new(), values: Vec::new() }
    }

    pub fn from_raw(
        rows: usize, cols: usize,
        row_ptr: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>,
    ) -> Self {
        assert_eq!(row_ptr.len(), rows + 1);
        assert_eq!(col_indices.len(), values.len());
        assert_eq!(*row_ptr.last().unwrap(), values.len());
        Self { rows, cols, row_ptr, col_indices, values }
    }

    pub fn nnz(&self) -> usize { self.values.len() }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols);
        for i in self.row_ptr[row]..self.row_ptr[row + 1] {
            if self.col_indices[i] == col { return self.values[i]; }
        }
        0.0
    }

    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for i in self.row_ptr[r]..self.row_ptr[r + 1] {
                dense[r * self.cols + self.col_indices[i]] = self.values[i];
            }
        }
        dense
    }

    pub fn to_coo(&self) -> CooMatrix {
        let mut ri = Vec::with_capacity(self.nnz());
        let mut ci = Vec::with_capacity(self.nnz());
        let mut vs = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            for i in self.row_ptr[r]..self.row_ptr[r + 1] {
                ri.push(r);
                ci.push(self.col_indices[i]);
                vs.push(self.values[i]);
            }
        }
        CooMatrix { rows: self.rows, cols: self.cols, row_indices: ri, col_indices: ci, values: vs }
    }

    /// Convert CSR to CSC.
    pub fn to_csc(&self) -> CscMatrix {
        self.to_coo().to_csc()
    }

    /// Sparse matrix-vector multiply: y = A * x.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut s = 0.0;
            for i in self.row_ptr[r]..self.row_ptr[r + 1] {
                s += self.values[i] * x[self.col_indices[i]];
            }
            y[r] = s;
        }
        y
    }

    /// Sparse matrix-matrix multiply: C = A * B.
    pub fn spmm(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.cols, other.rows);
        let mut coo = CooMatrix::new(self.rows, other.cols);
        for r in 0..self.rows {
            // Accumulate row r of C
            let mut row_acc = vec![0.0; other.cols];
            let mut touched = Vec::new();
            for i in self.row_ptr[r]..self.row_ptr[r + 1] {
                let k = self.col_indices[i];
                let a_val = self.values[i];
                for j in other.row_ptr[k]..other.row_ptr[k + 1] {
                    let c = other.col_indices[j];
                    if row_acc[c] == 0.0 { touched.push(c); }
                    row_acc[c] += a_val * other.values[j];
                }
            }
            for &c in &touched {
                if row_acc[c].abs() > 1e-15 {
                    coo.push(r, c, row_acc[c]);
                }
                row_acc[c] = 0.0;
            }
        }
        coo.to_csr()
    }

    pub fn transpose(&self) -> CsrMatrix {
        let coo = self.to_coo();
        CooMatrix::from_triplets(self.cols, self.rows, coo.col_indices, coo.row_indices, coo.values)
            .to_csr()
    }

    pub fn add(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.elementwise_op(other, |a, b| a + b)
    }

    pub fn subtract(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.elementwise_op(other, |a, b| a - b)
    }

    pub fn hadamard(&self, other: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut coo = CooMatrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            let (mut ai, mut bi) = (self.row_ptr[r], other.row_ptr[r]);
            let (ae, be) = (self.row_ptr[r + 1], other.row_ptr[r + 1]);
            while ai < ae && bi < be {
                let ac = self.col_indices[ai];
                let bc = other.col_indices[bi];
                if ac == bc {
                    let v = self.values[ai] * other.values[bi];
                    if v.abs() > 1e-15 { coo.push(r, ac, v); }
                    ai += 1; bi += 1;
                } else if ac < bc { ai += 1; } else { bi += 1; }
            }
        }
        coo.to_csr()
    }

    fn elementwise_op(&self, other: &CsrMatrix, op: impl Fn(f64, f64) -> f64) -> CsrMatrix {
        let mut coo = CooMatrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            let (mut ai, mut bi) = (self.row_ptr[r], other.row_ptr[r]);
            let (ae, be) = (self.row_ptr[r + 1], other.row_ptr[r + 1]);
            while ai < ae && bi < be {
                let ac = self.col_indices[ai];
                let bc = other.col_indices[bi];
                if ac == bc {
                    let v = op(self.values[ai], other.values[bi]);
                    if v.abs() > 1e-15 { coo.push(r, ac, v); }
                    ai += 1; bi += 1;
                } else if ac < bc {
                    let v = op(self.values[ai], 0.0);
                    if v.abs() > 1e-15 { coo.push(r, ac, v); }
                    ai += 1;
                } else {
                    let v = op(0.0, other.values[bi]);
                    if v.abs() > 1e-15 { coo.push(r, bc, v); }
                    bi += 1;
                }
            }
            while ai < ae {
                let v = op(self.values[ai], 0.0);
                if v.abs() > 1e-15 { coo.push(r, self.col_indices[ai], v); }
                ai += 1;
            }
            while bi < be {
                let v = op(0.0, other.values[bi]);
                if v.abs() > 1e-15 { coo.push(r, other.col_indices[bi], v); }
                bi += 1;
            }
        }
        coo.to_csr()
    }

    pub fn scale(&self, scalar: f64) -> CsrMatrix {
        CsrMatrix {
            rows: self.rows, cols: self.cols,
            row_ptr: self.row_ptr.clone(), col_indices: self.col_indices.clone(),
            values: self.values.iter().map(|v| v * scalar).collect(),
        }
    }

    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    pub fn diagonal_k(&self, k: isize) -> Vec<f64> {
        let n = if k >= 0 {
            self.rows.min(self.cols.saturating_sub(k as usize))
        } else {
            self.rows.saturating_sub((-k) as usize).min(self.cols)
        };
        (0..n).map(|i| {
            let (r, c) = if k >= 0 { (i, i + k as usize) } else { (i + (-k) as usize, i) };
            if r < self.rows && c < self.cols { self.get(r, c) } else { 0.0 }
        }).collect()
    }

    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 { 0.0 } else { self.nnz() as f64 / total as f64 }
    }

    pub fn sparsity(&self) -> f64 { 1.0 - self.density() }

    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }

    pub fn identity(n: usize) -> Self {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        Self { rows: n, cols: n, row_ptr, col_indices, values }
    }

    pub fn from_dense(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut coo = CooMatrix::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v.abs() > 1e-15 { coo.push(r, c, v); }
            }
        }
        coo.to_csr()
    }

    pub fn nnz_per_row(&self) -> Vec<usize> {
        (0..self.rows).map(|r| self.row_ptr[r + 1] - self.row_ptr[r]).collect()
    }

    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        assert!(row < self.rows);
        self.row_ptr[row]..self.row_ptr[row + 1]
    }

    /// Iterate over non-zero entries as (row, col, value).
    pub fn iter_nonzero(&self) -> CsrIter<'_> {
        CsrIter { matrix: self, row: 0, idx: 0 }
    }

    pub fn stats(&self) -> SparseStats {
        let nnz_pr = self.nnz_per_row();
        SparseStats {
            rows: self.rows, cols: self.cols, nnz: self.nnz(),
            density: self.density(), sparsity: self.sparsity(),
            frobenius_norm: self.frobenius_norm(),
            max_nnz_per_row: nnz_pr.iter().copied().max().unwrap_or(0),
            min_nnz_per_row: nnz_pr.iter().copied().min().unwrap_or(0),
            avg_nnz_per_row: if self.rows > 0 { self.nnz() as f64 / self.rows as f64 } else { 0.0 },
        }
    }
}

/// Iterator over non-zero entries of a CSR matrix.
pub struct CsrIter<'a> {
    matrix: &'a CsrMatrix,
    row: usize,
    idx: usize,
}

impl<'a> Iterator for CsrIter<'a> {
    type Item = (usize, usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.row < self.matrix.rows && self.idx >= self.matrix.row_ptr[self.row + 1] {
            self.row += 1;
        }
        if self.row >= self.matrix.rows { return None; }
        let col = self.matrix.col_indices[self.idx];
        let val = self.matrix.values[self.idx];
        self.idx += 1;
        Some((self.row, col, val))
    }
}

// ── CSC format ───────────────────────────────────────────────────

/// Compressed Sparse Column matrix.
#[derive(Clone, PartialEq)]
pub struct CscMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Column pointers — length `cols + 1`. Column `j` spans `col_ptr[j]..col_ptr[j+1]`.
    pub col_ptr: Vec<usize>,
    /// Row indices for each non-zero.
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl fmt::Debug for CscMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CscMatrix({}x{}, nnz={})", self.rows, self.cols, self.values.len())
    }
}

impl CscMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, col_ptr: vec![0; cols + 1], row_indices: Vec::new(), values: Vec::new() }
    }

    pub fn nnz(&self) -> usize { self.values.len() }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols);
        for i in self.col_ptr[col]..self.col_ptr[col + 1] {
            if self.row_indices[i] == row { return self.values[i]; }
        }
        0.0
    }

    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.rows * self.cols];
        for c in 0..self.cols {
            for i in self.col_ptr[c]..self.col_ptr[c + 1] {
                dense[self.row_indices[i] * self.cols + c] = self.values[i];
            }
        }
        dense
    }

    pub fn to_coo(&self) -> CooMatrix {
        let mut ri = Vec::with_capacity(self.nnz());
        let mut ci = Vec::with_capacity(self.nnz());
        let mut vs = Vec::with_capacity(self.nnz());
        for c in 0..self.cols {
            for i in self.col_ptr[c]..self.col_ptr[c + 1] {
                ri.push(self.row_indices[i]);
                ci.push(c);
                vs.push(self.values[i]);
            }
        }
        CooMatrix { rows: self.rows, cols: self.cols, row_indices: ri, col_indices: ci, values: vs }
    }

    pub fn to_csr(&self) -> CsrMatrix {
        self.to_coo().to_csr()
    }

    /// Column-oriented SpMV: y = A * x.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        for c in 0..self.cols {
            let xc = x[c];
            for i in self.col_ptr[c]..self.col_ptr[c + 1] {
                y[self.row_indices[i]] += self.values[i] * xc;
            }
        }
        y
    }

    pub fn transpose(&self) -> CscMatrix {
        let coo = self.to_coo();
        CooMatrix::from_triplets(self.cols, self.rows, coo.col_indices, coo.row_indices, coo.values)
            .to_csc()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    pub fn nnz_per_col(&self) -> Vec<usize> {
        (0..self.cols).map(|c| self.col_ptr[c + 1] - self.col_ptr[c]).collect()
    }

    /// Iterate over non-zero entries as (row, col, value).
    pub fn iter_nonzero(&self) -> CscIter<'_> {
        CscIter { matrix: self, col: 0, idx: 0 }
    }

    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 { 0.0 } else { self.nnz() as f64 / total as f64 }
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

/// Iterator over non-zero entries of a CSC matrix.
pub struct CscIter<'a> {
    matrix: &'a CscMatrix,
    col: usize,
    idx: usize,
}

impl<'a> Iterator for CscIter<'a> {
    type Item = (usize, usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.col < self.matrix.cols && self.idx >= self.matrix.col_ptr[self.col + 1] {
            self.col += 1;
        }
        if self.col >= self.matrix.cols { return None; }
        let row = self.matrix.row_indices[self.idx];
        let val = self.matrix.values[self.idx];
        self.idx += 1;
        Some((row, self.col, val))
    }
}

/// Sparse matrix statistics.
#[derive(Debug, Clone)]
pub struct SparseStats {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub density: f64,
    pub sparsity: f64,
    pub frobenius_norm: f64,
    pub max_nnz_per_row: usize,
    pub min_nnz_per_row: usize,
    pub avg_nnz_per_row: f64,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn test_coo_create_push() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0); coo.push(1, 2, 5.0); coo.push(2, 1, 3.0);
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_coo_from_triplets() {
        let coo = CooMatrix::from_triplets(3, 3, vec![0,1,2], vec![0,1,2], vec![1.0,2.0,3.0]);
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_coo_to_csr_basic() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0); coo.push(0, 2, 2.0); coo.push(1, 1, 3.0);
        coo.push(2, 0, 4.0); coo.push(2, 2, 5.0);
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 5);
        assert!(approx_eq(csr.get(0, 0), 1.0));
        assert!(approx_eq(csr.get(0, 2), 2.0));
        assert!(approx_eq(csr.get(1, 1), 3.0));
        assert!(approx_eq(csr.get(2, 0), 4.0));
        assert!(approx_eq(csr.get(2, 2), 5.0));
        assert!(approx_eq(csr.get(0, 1), 0.0));
    }

    #[test]
    fn test_csr_to_dense() {
        let csr = CsrMatrix::from_dense(2, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let dense = csr.to_dense();
        assert!(approx_eq(dense[0], 1.0));
        assert!(approx_eq(dense[2], 2.0));
        assert!(approx_eq(dense[4], 3.0));
    }

    #[test]
    fn test_csr_coo_roundtrip() {
        let csr = CsrMatrix::from_dense(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let coo = csr.to_coo();
        let csr2 = coo.to_csr();
        assert_eq!(csr.to_dense(), csr2.to_dense());
    }

    #[test]
    fn test_spmv() {
        let csr = CsrMatrix::from_dense(3, 3, &[1.0,0.0,2.0, 0.0,3.0,0.0, 4.0,0.0,5.0]);
        let y = csr.spmv(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(y[0], 7.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 19.0));
    }

    #[test]
    fn test_spmm() {
        let a = CsrMatrix::from_dense(2, 3, &[1.0,2.0,0.0, 0.0,3.0,4.0]);
        let b = CsrMatrix::from_dense(3, 2, &[1.0,0.0, 0.0,1.0, 2.0,0.0]);
        let c = a.spmm(&b);
        // [1*1+2*0+0*2, 1*0+2*1+0*0] = [1, 2]
        // [0*1+3*0+4*2, 0*0+3*1+4*0] = [8, 3]
        assert!(approx_eq(c.get(0, 0), 1.0));
        assert!(approx_eq(c.get(0, 1), 2.0));
        assert!(approx_eq(c.get(1, 0), 8.0));
        assert!(approx_eq(c.get(1, 1), 3.0));
    }

    #[test]
    fn test_transpose() {
        let csr = CsrMatrix::from_dense(2, 3, &[1.0,2.0,3.0, 4.0,5.0,6.0]);
        let t = csr.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!(approx_eq(t.get(0, 0), 1.0));
        assert!(approx_eq(t.get(0, 1), 4.0));
        assert!(approx_eq(t.get(2, 1), 6.0));
    }

    #[test]
    fn test_add_subtract() {
        let a = CsrMatrix::from_dense(2, 2, &[1.0,2.0,3.0,4.0]);
        let b = CsrMatrix::from_dense(2, 2, &[5.0,6.0,7.0,8.0]);
        let s = a.add(&b);
        assert!(approx_eq(s.get(0, 0), 6.0));
        assert!(approx_eq(s.get(1, 1), 12.0));
        let d = b.subtract(&a);
        assert!(approx_eq(d.get(0, 0), 4.0));
    }

    #[test]
    fn test_hadamard() {
        let a = CsrMatrix::from_dense(2, 2, &[1.0,2.0,0.0,4.0]);
        let b = CsrMatrix::from_dense(2, 2, &[5.0,0.0,7.0,8.0]);
        let h = a.hadamard(&b);
        assert!(approx_eq(h.get(0, 0), 5.0));
        assert!(approx_eq(h.get(0, 1), 0.0));
        assert!(approx_eq(h.get(1, 1), 32.0));
    }

    #[test]
    fn test_scale() {
        let a = CsrMatrix::from_dense(2, 2, &[1.0,0.0,0.0,3.0]);
        let s = a.scale(2.5);
        assert!(approx_eq(s.get(0, 0), 2.5));
        assert!(approx_eq(s.get(1, 1), 7.5));
    }

    #[test]
    fn test_diagonal_and_k() {
        let m = CsrMatrix::from_dense(3, 3, &[1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,9.0]);
        assert!(approx_eq(m.diagonal()[1], 5.0));
        let sup = m.diagonal_k(1);
        assert_eq!(sup.len(), 2);
        assert!(approx_eq(sup[0], 2.0));
        let sub = m.diagonal_k(-1);
        assert!(approx_eq(sub[0], 4.0));
    }

    #[test]
    fn test_density_sparsity() {
        let m = CsrMatrix::from_dense(3, 3, &[1.0,0.0,0.0, 0.0,2.0,0.0, 0.0,0.0,3.0]);
        assert!(approx_eq(m.density(), 3.0 / 9.0));
        assert!(approx_eq(m.sparsity(), 1.0 - 3.0 / 9.0));
    }

    #[test]
    fn test_frobenius_trace() {
        let m = CsrMatrix::from_dense(2, 2, &[3.0,0.0,0.0,4.0]);
        assert!(approx_eq(m.frobenius_norm(), 5.0));
        assert!(approx_eq(m.trace(), 7.0));
    }

    #[test]
    fn test_identity() {
        let id = CsrMatrix::identity(4);
        assert_eq!(id.nnz(), 4);
        for i in 0..4 { assert!(approx_eq(id.get(i, i), 1.0)); }
    }

    #[test]
    fn test_csr_iter_nonzero() {
        let m = CsrMatrix::from_dense(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let entries: Vec<_> = m.iter_nonzero().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (0, 0, 1.0));
        assert_eq!(entries[1], (1, 1, 2.0));
    }

    #[test]
    fn test_empty_matrix() {
        let m = CsrMatrix::new(3, 3);
        assert_eq!(m.nnz(), 0);
        assert!(approx_eq(m.density(), 0.0));
        assert!(approx_eq(m.get(1, 1), 0.0));
    }

    #[test]
    fn test_stats() {
        let m = CsrMatrix::from_dense(3, 3, &[1.0,2.0,0.0, 0.0,3.0,0.0, 0.0,0.0,4.0]);
        let s = m.stats();
        assert_eq!(s.nnz, 4);
        assert_eq!(s.max_nnz_per_row, 2);
    }

    // ── CSC tests ──────────────────────────────────────────────

    #[test]
    fn test_coo_to_csc() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0); coo.push(1, 1, 2.0); coo.push(2, 2, 3.0);
        let csc = coo.to_csc();
        assert_eq!(csc.nnz(), 3);
        assert!(approx_eq(csc.get(0, 0), 1.0));
        assert!(approx_eq(csc.get(1, 1), 2.0));
        assert!(approx_eq(csc.get(2, 2), 3.0));
        assert!(approx_eq(csc.get(0, 1), 0.0));
    }

    #[test]
    fn test_csc_to_dense() {
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0); coo.push(0, 2, 2.0); coo.push(1, 1, 3.0);
        let csc = coo.to_csc();
        let dense = csc.to_dense();
        assert!(approx_eq(dense[0], 1.0));
        assert!(approx_eq(dense[1], 0.0));
        assert!(approx_eq(dense[2], 2.0));
        assert!(approx_eq(dense[3], 0.0));
        assert!(approx_eq(dense[4], 3.0));
    }

    #[test]
    fn test_csc_spmv() {
        let csr = CsrMatrix::from_dense(3, 3, &[1.0,0.0,2.0, 0.0,3.0,0.0, 4.0,0.0,5.0]);
        let csc = csr.to_csc();
        let y = csc.spmv(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(y[0], 7.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 19.0));
    }

    #[test]
    fn test_csc_transpose() {
        let csr = CsrMatrix::from_dense(2, 3, &[1.0,2.0,3.0, 4.0,5.0,6.0]);
        let csc = csr.to_csc();
        let t = csc.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!(approx_eq(t.get(0, 0), 1.0));
        assert!(approx_eq(t.get(0, 1), 4.0));
    }

    #[test]
    fn test_csc_diagonal() {
        let csr = CsrMatrix::from_dense(3, 3, &[1.0,0.0,0.0, 0.0,5.0,0.0, 0.0,0.0,9.0]);
        let csc = csr.to_csc();
        let d = csc.diagonal();
        assert!(approx_eq(d[0], 1.0));
        assert!(approx_eq(d[1], 5.0));
        assert!(approx_eq(d[2], 9.0));
    }

    #[test]
    fn test_csc_iter_nonzero() {
        let csr = CsrMatrix::from_dense(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let csc = csr.to_csc();
        let entries: Vec<_> = csc.iter_nonzero().collect();
        assert_eq!(entries.len(), 2);
        // CSC iterates column-major
        assert_eq!(entries[0].0, 0); // row=0, col=0
        assert_eq!(entries[0].1, 0);
        assert_eq!(entries[1].0, 1); // row=1, col=1
        assert_eq!(entries[1].1, 1);
    }

    #[test]
    fn test_csr_to_csc_roundtrip() {
        let csr = CsrMatrix::from_dense(3, 4, &[
            1.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 4.0,
            5.0, 0.0, 6.0, 0.0,
        ]);
        let csc = csr.to_csc();
        let csr2 = csc.to_csr();
        assert_eq!(csr.to_dense(), csr2.to_dense());
    }

    #[test]
    fn test_csc_nnz_per_col() {
        let csr = CsrMatrix::from_dense(3, 3, &[1.0,2.0,0.0, 0.0,0.0,0.0, 0.0,0.0,3.0]);
        let csc = csr.to_csc();
        let counts = csc.nnz_per_col();
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 1);
    }

    #[test]
    fn test_csc_frobenius() {
        let csr = CsrMatrix::from_dense(2, 2, &[3.0, 0.0, 0.0, 4.0]);
        let csc = csr.to_csc();
        assert!(approx_eq(csc.frobenius_norm(), 5.0));
    }

    #[test]
    fn test_spmv_identity() {
        let id = CsrMatrix::identity(3);
        let x = vec![7.0, 8.0, 9.0];
        assert_eq!(id.spmv(&x), x);
    }

    #[test]
    fn test_nnz_per_row() {
        let m = CsrMatrix::from_dense(3, 3, &[1.0,2.0,0.0, 0.0,0.0,0.0, 0.0,0.0,3.0]);
        let c = m.nnz_per_row();
        assert_eq!(c[0], 2);
        assert_eq!(c[1], 0);
        assert_eq!(c[2], 1);
    }

    #[test]
    fn test_coo_identity_to_csr() {
        let id_coo = CooMatrix::identity(4);
        let id_csr = id_coo.to_csr();
        for i in 0..4 { assert!(approx_eq(id_csr.get(i, i), 1.0)); }
    }
}
