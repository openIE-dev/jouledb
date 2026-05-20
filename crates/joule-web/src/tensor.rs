//! N-dimensional tensor operations.
//!
//! Dense f64 tensor with shape metadata, element-wise arithmetic,
//! matrix multiply, broadcasting, reductions, activations, slicing,
//! reshape, and creation helpers. Pure Rust — no external BLAS or GPU deps.

use std::fmt;

// ── Tensor ──────────────────────────────────────────────────────

/// N-dimensional dense tensor of f64 values stored in row-major order.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    /// Shape of each dimension, e.g. `[2, 3]` for a 2x3 matrix.
    pub shape: Vec<usize>,
    /// Flat data in row-major order.
    pub data: Vec<f64>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, len={})", self.shape, self.data.len())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ndim() == 0 {
            write!(f, "Tensor(scalar={})", self.data[0])
        } else if self.ndim() == 1 {
            write!(f, "Tensor({:?})", self.data)
        } else {
            write!(f, "Tensor(shape={:?})", self.shape)
        }
    }
}

impl Tensor {
    /// Create a new tensor. Panics if `data.len() != product(shape)`.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let expected: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        assert_eq!(
            data.len(),
            expected,
            "data length {} != shape product {}",
            data.len(),
            expected,
        );
        Self { shape, data }
    }

    /// Tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        Self { shape, data: vec![0.0; n] }
    }

    /// Tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        Self { shape, data: vec![1.0; n] }
    }

    /// Tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let n: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        Self { shape, data: vec![value; n] }
    }

    /// Scalar tensor (shape = []).
    pub fn scalar(value: f64) -> Self {
        Self { shape: vec![], data: vec![value] }
    }

    /// 1D tensor from a range [start, stop) with given step.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut v = start;
        while v < stop {
            data.push(v);
            v += step;
        }
        let n = data.len();
        Self { shape: vec![n], data }
    }

    /// 1D tensor with n evenly spaced values from start to stop (inclusive).
    pub fn linspace(start: f64, stop: f64, n: usize) -> Self {
        if n == 0 {
            return Self { shape: vec![0], data: vec![] };
        }
        if n == 1 {
            return Self { shape: vec![1], data: vec![start] };
        }
        let step = (stop - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self { shape: vec![n], data }
    }

    /// Identity matrix of size n x n.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self { shape: vec![n, n], data }
    }

    /// Diagonal matrix from a 1D tensor.
    pub fn diag(values: &[f64]) -> Self {
        let n = values.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = values[i];
        }
        Self { shape: vec![n, n], data }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Compute strides for row-major layout.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len()).rev() {
            if i + 1 < self.shape.len() {
                s[i] = s[i + 1] * self.shape[i + 1];
            }
        }
        s
    }

    // ── Element-wise ops ────────────────────────────────────────

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.broadcast_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.broadcast_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.broadcast_op(other, |a, b| a * b)
    }

    /// Element-wise division with broadcasting.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.broadcast_op(other, |a, b| a / b)
    }

    /// Apply a unary function element-wise.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|v| f(*v)).collect(),
        }
    }

    /// Scalar addition.
    pub fn add_scalar(&self, s: f64) -> Tensor {
        self.map(|v| v + s)
    }

    /// Scalar multiplication.
    pub fn mul_scalar(&self, s: f64) -> Tensor {
        self.map(|v| v * s)
    }

    /// Element-wise power.
    pub fn pow(&self, exp: f64) -> Tensor {
        self.map(|v| v.powf(exp))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor {
        self.map(|v| v.abs())
    }

    // ── Broadcasting ────────────────────────────────────────────

    fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let max_rank = a.len().max(b.len());
        let mut result = vec![0usize; max_rank];
        for i in 0..max_rank {
            let da = if i < max_rank - a.len() { 1 } else { a[i - (max_rank - a.len())] };
            let db = if i < max_rank - b.len() { 1 } else { b[i - (max_rank - b.len())] };
            assert!(
                da == db || da == 1 || db == 1,
                "incompatible broadcast dims {} vs {}",
                da,
                db,
            );
            result[i] = da.max(db);
        }
        result
    }

    fn broadcast_index(shape: &[usize], out_shape: &[usize], flat_idx: usize) -> usize {
        let rank_diff = out_shape.len() - shape.len();
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            let out_i = i + rank_diff;
            let coord = (flat_idx / Self::stride_at(out_shape, out_i)) % out_shape[out_i];
            let actual = if shape[i] == 1 { 0 } else { coord };
            idx += actual * stride;
            stride *= shape[i];
        }
        idx
    }

    fn stride_at(shape: &[usize], dim: usize) -> usize {
        shape[dim + 1..].iter().product()
    }

    fn broadcast_op(&self, other: &Tensor, op: impl Fn(f64, f64) -> f64) -> Tensor {
        let out_shape = Self::broadcast_shape(&self.shape, &other.shape);
        let n: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let ai = Self::broadcast_index(&self.shape, &out_shape, i);
            let bi = Self::broadcast_index(&other.shape, &out_shape, i);
            data.push(op(self.data[ai], other.data[bi]));
        }
        Tensor { shape: out_shape, data }
    }

    // ── Matrix multiply (2D) ────────────────────────────────────

    /// Matrix multiply. Both tensors must be 2D with compatible inner dims.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensor");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensor");
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k1, k2, "inner dimensions must match: {} vs {}", k1, k2);

        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k1 {
                    sum += self.data[i * k1 + p] * other.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Tensor { shape: vec![m, n], data }
    }

    // ── Reshape / transpose / squeeze / unsqueeze ───────────────

    /// Reshape to a new shape. Total elements must be the same.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: element count mismatch");
        Tensor { shape: new_shape, data: self.data.clone() }
    }

    /// Transpose a 2D tensor.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }
        Tensor { shape: vec![cols, rows], data }
    }

    /// Remove all dimensions of size 1.
    pub fn squeeze(&self) -> Tensor {
        let new_shape: Vec<usize> = self.shape.iter().copied().filter(|d| *d != 1).collect();
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        Tensor { shape: new_shape, data: self.data.clone() }
    }

    /// Insert a dimension of size 1 at the given axis.
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        assert!(axis <= self.ndim(), "axis out of range");
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Tensor { shape: new_shape, data: self.data.clone() }
    }

    /// Flatten to 1D.
    pub fn flatten(&self) -> Tensor {
        Tensor { shape: vec![self.numel()], data: self.data.clone() }
    }

    // ── Reductions ──────────────────────────────────────────────

    /// Sum along an axis. Removes that dimension.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, |slice| slice.iter().sum())
    }

    /// Mean along an axis.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, |slice| {
            let s: f64 = slice.iter().sum();
            s / slice.len() as f64
        })
    }

    /// Max along an axis.
    pub fn max_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, |slice| {
            slice.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        })
    }

    /// Min along an axis.
    pub fn min_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, |slice| {
            slice.iter().copied().fold(f64::INFINITY, f64::min)
        })
    }

    /// Global sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Global mean.
    pub fn mean(&self) -> f64 {
        self.sum() / self.numel() as f64
    }

    /// Global max.
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Global min.
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn reduce_axis(&self, axis: usize, reducer: impl Fn(&[f64]) -> f64) -> Tensor {
        assert!(axis < self.ndim(), "axis out of range");
        let dim_size = self.shape[axis];
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();

        let mut data = Vec::with_capacity(outer * inner);
        let mut tmp = vec![0.0; dim_size];
        for o in 0..outer {
            for i in 0..inner {
                for d in 0..dim_size {
                    tmp[d] = self.data[o * dim_size * inner + d * inner + i];
                }
                data.push(reducer(&tmp));
            }
        }
        Tensor { shape: new_shape, data }
    }

    // ── Activations ─────────────────────────────────────────────

    /// Softmax along the last axis.
    pub fn softmax(&self) -> Tensor {
        let last_dim = *self.shape.last().expect("empty shape");
        let n = self.numel();
        let mut data = self.data.clone();
        for chunk in 0..(n / last_dim) {
            let start = chunk * last_dim;
            let end = start + last_dim;
            let max_val = data[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in &mut data[start..end] {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in &mut data[start..end] {
                *v /= sum;
            }
        }
        Tensor { shape: self.shape.clone(), data }
    }

    /// ReLU activation (max(0, x)).
    pub fn relu(&self) -> Tensor {
        self.map(|v| v.max(0.0))
    }

    /// Sigmoid activation (1 / (1 + exp(-x))).
    pub fn sigmoid(&self) -> Tensor {
        self.map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> Tensor {
        self.map(|v| v.tanh())
    }

    // ── Slice / index ───────────────────────────────────────────

    /// Slice along the first axis: returns tensor[start..end].
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Tensor {
        assert!(axis < self.ndim(), "axis out of range");
        assert!(start < end && end <= self.shape[axis], "invalid slice range");

        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let dim_size = self.shape[axis];
        let slice_len = end - start;

        let mut data = Vec::with_capacity(outer * slice_len * inner);
        for o in 0..outer {
            for d in start..end {
                let base = o * dim_size * inner + d * inner;
                data.extend_from_slice(&self.data[base..base + inner]);
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = slice_len;
        Tensor { shape: new_shape, data }
    }

    /// Get a single element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f64 {
        assert_eq!(indices.len(), self.ndim(), "index rank mismatch");
        let strides = self.strides();
        let flat: usize = indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat]
    }

    /// Set a single element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f64) {
        assert_eq!(indices.len(), self.ndim(), "index rank mismatch");
        let strides = self.strides();
        let flat: usize = indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat] = value;
    }

    /// Concatenate tensors along an axis.
    pub fn concat(tensors: &[&Tensor], axis: usize) -> Tensor {
        assert!(!tensors.is_empty(), "need at least one tensor");
        let ndim = tensors[0].ndim();
        for t in tensors {
            assert_eq!(t.ndim(), ndim, "all tensors must have same ndim");
        }

        // Check all dims match except axis
        for i in 0..ndim {
            if i != axis {
                for t in tensors.iter().skip(1) {
                    assert_eq!(
                        t.shape[i], tensors[0].shape[i],
                        "shape mismatch on dim {} for concat",
                        i
                    );
                }
            }
        }

        let mut new_shape = tensors[0].shape.clone();
        let total_axis: usize = tensors.iter().map(|t| t.shape[axis]).sum();
        new_shape[axis] = total_axis;

        let outer: usize = new_shape[..axis].iter().product();
        let inner: usize = new_shape[axis + 1..].iter().product();

        let total_elements: usize = new_shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        for o in 0..outer {
            for t in tensors {
                let t_axis = t.shape[axis];
                let t_inner: usize = t.shape[axis + 1..].iter().product();
                let base = o * t_axis * t_inner;
                data.extend_from_slice(&t.data[base..base + t_axis * t_inner]);
            }
        }

        Tensor { shape: new_shape, data }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn tensors_close(a: &Tensor, b: &Tensor) -> bool {
        a.shape == b.shape && a.data.iter().zip(b.data.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_create_and_numel() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_zeros_ones_full() {
        let z = Tensor::zeros(vec![3, 2]);
        assert!(z.data.iter().all(|v| *v == 0.0));
        let o = Tensor::ones(vec![2, 2]);
        assert!(o.data.iter().all(|v| *v == 1.0));
        let f = Tensor::full(vec![2, 3], 7.0);
        assert!(f.data.iter().all(|v| *v == 7.0));
    }

    #[test]
    fn test_arange_linspace() {
        let a = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(a.shape, vec![5]);
        assert_eq!(a.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let l = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(l.shape, vec![5]);
        assert!(approx_eq(l.data[0], 0.0));
        assert!(approx_eq(l.data[4], 1.0));
        assert!(approx_eq(l.data[2], 0.5));
    }

    #[test]
    fn test_eye_diag() {
        let eye = Tensor::eye(3);
        assert_eq!(eye.shape, vec![3, 3]);
        assert!(approx_eq(eye.get(&[0, 0]), 1.0));
        assert!(approx_eq(eye.get(&[0, 1]), 0.0));
        assert!(approx_eq(eye.get(&[2, 2]), 1.0));

        let d = Tensor::diag(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(d.get(&[1, 1]), 2.0));
        assert!(approx_eq(d.get(&[0, 1]), 0.0));
    }

    #[test]
    fn test_element_wise_add() {
        let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_element_wise_mul_sub_div() {
        let a = Tensor::new(vec![2], vec![6.0, 8.0]);
        let b = Tensor::new(vec![2], vec![2.0, 4.0]);
        assert_eq!(a.sub(&b).data, vec![4.0, 4.0]);
        assert_eq!(a.mul(&b).data, vec![12.0, 32.0]);
        assert_eq!(a.div(&b).data, vec![3.0, 2.0]);
    }

    #[test]
    fn test_scalar_ops() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let s = t.add_scalar(10.0);
        assert_eq!(s.data, vec![11.0, 12.0, 13.0]);
        let m = t.mul_scalar(3.0);
        assert_eq!(m.data, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![1, 3], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b);
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_reshape_and_transpose() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(vec![3, 2]);
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let tr = t.transpose();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let t = Tensor::new(vec![1, 3, 1], vec![1.0, 2.0, 3.0]);
        let s = t.squeeze();
        assert_eq!(s.shape, vec![3]);

        let u = s.unsqueeze(0);
        assert_eq!(u.shape, vec![1, 3]);
        let u2 = s.unsqueeze(1);
        assert_eq!(u2.shape, vec![3, 1]);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let f = t.flatten();
        assert_eq!(f.shape, vec![6]);
        assert_eq!(f.data, t.data);
    }

    #[test]
    fn test_reductions() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = t.sum_axis(1);
        assert_eq!(s.shape, vec![2]);
        assert_eq!(s.data, vec![6.0, 15.0]);

        let m = t.mean_axis(0);
        assert_eq!(m.shape, vec![3]);
        assert!(tensors_close(&m, &Tensor::new(vec![3], vec![2.5, 3.5, 4.5])));

        let mx = t.max_axis(1);
        assert_eq!(mx.data, vec![3.0, 6.0]);

        let mn = t.min_axis(0);
        assert_eq!(mn.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_global_reductions() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(approx_eq(t.sum(), 21.0));
        assert!(approx_eq(t.mean(), 3.5));
        assert!(approx_eq(t.max(), 6.0));
        assert!(approx_eq(t.min(), 1.0));
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let s = t.softmax();
        let total: f64 = s.data.iter().sum();
        assert!(approx_eq(total, 1.0));
        assert!(s.data[0] < s.data[1]);
        assert!(s.data[1] < s.data[2]);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::new(vec![4], vec![-2.0, -1.0, 0.0, 1.0]);

        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0]);

        let sig = t.sigmoid();
        assert!(approx_eq(sig.data[2], 0.5));
        assert!(sig.data[3] > 0.5);
        assert!(sig.data[0] < 0.5);

        let th = t.tanh_act();
        assert!(approx_eq(th.data[2], 0.0));
    }

    #[test]
    fn test_slice_and_index() {
        let t = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = t.slice(0, 1, 3);
        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(s.data, vec![3.0, 4.0, 5.0, 6.0]);

        assert_eq!(t.get(&[1, 0]), 3.0);
        assert_eq!(t.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_set_element() {
        let mut t = Tensor::zeros(vec![2, 2]);
        t.set(&[0, 1], 42.0);
        assert_eq!(t.get(&[0, 1]), 42.0);
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_pow_abs() {
        let t = Tensor::new(vec![3], vec![-2.0, 3.0, 4.0]);
        let a = t.abs();
        assert_eq!(a.data, vec![2.0, 3.0, 4.0]);

        let p = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).pow(2.0);
        assert_eq!(p.data, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0]);
        let c = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_display() {
        let t = Tensor::scalar(42.0);
        let s = format!("{}", t);
        assert!(s.contains("42"));
    }

    #[test]
    fn test_strides() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        let s = t.strides();
        assert_eq!(s, vec![12, 4, 1]);
    }
}
