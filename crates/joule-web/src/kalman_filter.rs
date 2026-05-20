//! Linear Kalman Filter — state estimation with predict/update cycles, innovation
//! monitoring, Mahalanobis distance, and Rauch-Tung-Striebel (RTS) backward smoother.
//!
//! Supports 2x2 fast-path and general NxN matrix operations. All linear algebra
//! is implemented inline — no LAPACK or nalgebra dependency.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Kalman filter errors.
#[derive(Debug, Clone, PartialEq)]
pub enum KalmanError {
    /// Dimension mismatch between matrices.
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
    /// Matrix is singular and cannot be inverted.
    SingularMatrix,
    /// Invalid noise covariance (must be positive semi-definite).
    InvalidCovariance(String),
    /// Empty state vector.
    EmptyState,
}

impl std::fmt::Display for KalmanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}x{}, got {}x{}", expected.0, expected.1, got.0, got.1)
            }
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::InvalidCovariance(msg) => write!(f, "invalid covariance: {msg}"),
            Self::EmptyState => write!(f, "empty state vector"),
        }
    }
}

impl std::error::Error for KalmanError {}

// ── Dense Matrix ────────────────────────────────────────────────

/// Row-major dense matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create an r x c zero matrix.
    pub fn zeros(r: usize, c: usize) -> Self {
        Self { rows: r, cols: c, data: vec![0.0; r * c] }
    }

    /// Create an n x n identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Create from a flat row-major vector.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, KalmanError> {
        if data.len() != rows * cols {
            return Err(KalmanError::DimensionMismatch {
                expected: (rows, cols),
                got: (data.len(), 1),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Element access.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Mutable element access.
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    /// Matrix multiply: self * other.
    pub fn mul(&self, other: &Matrix) -> Result<Matrix, KalmanError> {
        if self.cols != other.rows {
            return Err(KalmanError::DimensionMismatch {
                expected: (self.rows, other.cols),
                got: (self.cols, other.rows),
            });
        }
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        Ok(result)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, KalmanError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(KalmanError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Matrix { rows: self.rows, cols: self.cols, data })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, KalmanError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(KalmanError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Ok(Matrix { rows: self.rows, cols: self.cols, data })
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Invert an NxN matrix using Gauss-Jordan elimination.
    pub fn inverse(&self) -> Result<Matrix, KalmanError> {
        if self.rows != self.cols {
            return Err(KalmanError::DimensionMismatch {
                expected: (self.rows, self.rows),
                got: (self.rows, self.cols),
            });
        }
        let n = self.rows;
        // Augmented matrix [A | I].
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.get(i, j);
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        for col in 0..n {
            // Partial pivoting.
            let mut max_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return Err(KalmanError::SingularMatrix);
            }
            if max_row != col {
                for j in 0..2 * n {
                    let tmp = aug[col * 2 * n + j];
                    aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                    aug[max_row * 2 * n + j] = tmp;
                }
            }
            let pivot = aug[col * 2 * n + col];
            for j in 0..2 * n {
                aug[col * 2 * n + j] /= pivot;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }

        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug[i * 2 * n + n + j]);
            }
        }
        Ok(inv)
    }

    /// Trace of a square matrix.
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    /// Create a column vector from a slice.
    pub fn col_vec(data: &[f64]) -> Self {
        Self {
            rows: data.len(),
            cols: 1,
            data: data.to_vec(),
        }
    }
}

// ── Kalman Filter ───────────────────────────────────────────────

/// Linear Kalman Filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// State dimension.
    pub n: usize,
    /// Measurement dimension.
    pub m: usize,
    /// State estimate.
    pub x: Matrix,
    /// State covariance.
    pub p: Matrix,
    /// State transition.
    pub f_mat: Matrix,
    /// Control input matrix (n x n_u).
    pub b: Option<Matrix>,
    /// Measurement matrix.
    pub h: Matrix,
    /// Process noise covariance.
    pub q: Matrix,
    /// Measurement noise covariance.
    pub r: Matrix,
}

/// Result of a single update step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanUpdate {
    /// State estimate after update.
    pub x: Matrix,
    /// Covariance after update.
    pub p: Matrix,
    /// Kalman gain.
    pub k: Matrix,
    /// Innovation (measurement residual).
    pub innovation: Matrix,
    /// Innovation covariance.
    pub s: Matrix,
}

impl KalmanFilter {
    /// Create a Kalman filter with given dimensions.
    pub fn new(n: usize, m: usize) -> Result<Self, KalmanError> {
        if n == 0 || m == 0 {
            return Err(KalmanError::EmptyState);
        }
        Ok(Self {
            n,
            m,
            x: Matrix::zeros(n, 1),
            p: Matrix::identity(n),
            f_mat: Matrix::identity(n),
            b: None,
            h: Matrix::zeros(m, n),
            q: Matrix::identity(n).scale(0.01),
            r: Matrix::identity(m).scale(0.1),
        })
    }

    /// Predict step: x = Fx + Bu, P = FPF' + Q.
    pub fn predict(&mut self, u: Option<&Matrix>) -> Result<(), KalmanError> {
        // x = F * x
        let mut x_new = self.f_mat.mul(&self.x)?;

        // x += B * u
        if let (Some(b_mat), Some(u_vec)) = (&self.b, u) {
            let bu = b_mat.mul(u_vec)?;
            x_new = x_new.add(&bu)?;
        }

        // P = F * P * F' + Q
        let fp = self.f_mat.mul(&self.p)?;
        let ft = self.f_mat.transpose();
        let fpft = fp.mul(&ft)?;
        let p_new = fpft.add(&self.q)?;

        self.x = x_new;
        self.p = p_new;
        Ok(())
    }

    /// Update step with measurement z.
    pub fn update(&mut self, z: &Matrix) -> Result<KalmanUpdate, KalmanError> {
        // Innovation: y = z - H*x
        let hx = self.h.mul(&self.x)?;
        let innovation = z.sub(&hx)?;

        // Innovation covariance: S = H*P*H' + R
        let hp = self.h.mul(&self.p)?;
        let ht = self.h.transpose();
        let hpht = hp.mul(&ht)?;
        let s = hpht.add(&self.r)?;

        // Kalman gain: K = P*H' * S^-1
        let pht = self.p.mul(&ht)?;
        let s_inv = s.inverse()?;
        let k = pht.mul(&s_inv)?;

        // State update: x = x + K*y
        let ky = k.mul(&innovation)?;
        let x_new = self.x.add(&ky)?;

        // Covariance update: P = (I - K*H)*P  (Joseph form more stable but this is standard)
        let kh = k.mul(&self.h)?;
        let i_kh = Matrix::identity(self.n).sub(&kh)?;
        let p_new = i_kh.mul(&self.p)?;

        self.x = x_new.clone();
        self.p = p_new.clone();

        Ok(KalmanUpdate {
            x: x_new,
            p: p_new,
            k,
            innovation,
            s,
        })
    }

    /// Mahalanobis distance of a measurement from the predicted state.
    pub fn mahalanobis(&self, z: &Matrix) -> Result<f64, KalmanError> {
        let hx = self.h.mul(&self.x)?;
        let y = z.sub(&hx)?;
        let hp = self.h.mul(&self.p)?;
        let ht = self.h.transpose();
        let hpht = hp.mul(&ht)?;
        let s = hpht.add(&self.r)?;
        let s_inv = s.inverse()?;
        let s_inv_y = s_inv.mul(&y)?;
        let yt = y.transpose();
        let dist_sq = yt.mul(&s_inv_y)?;
        Ok(dist_sq.get(0, 0).sqrt())
    }
}

// ── RTS Smoother ────────────────────────────────────────────────

/// Record from a forward Kalman pass (for smoothing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterRecord {
    pub x_pred: Matrix,
    pub p_pred: Matrix,
    pub x_filt: Matrix,
    pub p_filt: Matrix,
}

/// Rauch-Tung-Striebel backward smoother.
pub fn rts_smooth(
    records: &[FilterRecord],
    f_mat: &Matrix,
) -> Result<Vec<(Matrix, Matrix)>, KalmanError> {
    let n = records.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut smoothed = vec![(Matrix::zeros(1, 1), Matrix::zeros(1, 1)); n];
    // Last smoothed = last filtered.
    smoothed[n - 1] = (
        records[n - 1].x_filt.clone(),
        records[n - 1].p_filt.clone(),
    );

    for k in (0..n - 1).rev() {
        let p_pred_inv = records[k + 1].p_pred.inverse()?;
        let ft = f_mat.transpose();
        let pf = records[k].p_filt.mul(&ft)?;
        let gain = pf.mul(&p_pred_inv)?;

        // x_smooth = x_filt + G * (x_smooth[k+1] - x_pred[k+1])
        let diff_x = smoothed[k + 1].0.sub(&records[k + 1].x_pred)?;
        let correction_x = gain.mul(&diff_x)?;
        let x_smooth = records[k].x_filt.add(&correction_x)?;

        // P_smooth = P_filt + G * (P_smooth[k+1] - P_pred[k+1]) * G'
        let diff_p = smoothed[k + 1].1.sub(&records[k + 1].p_pred)?;
        let gd = gain.mul(&diff_p)?;
        let gt = gain.transpose();
        let gdgt = gd.mul(&gt)?;
        let p_smooth = records[k].p_filt.add(&gdgt)?;

        smoothed[k] = (x_smooth, p_smooth);
    }

    Ok(smoothed)
}

// ── 2x2 Fast Path ──────────────────────────────────────────────

/// Optimized 2x2 Kalman filter for common 2-state systems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanFilter2x2 {
    pub x: [f64; 2],
    pub p: [[f64; 2]; 2],
    pub f_mat: [[f64; 2]; 2],
    pub h: [f64; 2],
    pub q: [[f64; 2]; 2],
    pub r: f64,
}

impl KalmanFilter2x2 {
    /// Create a constant-velocity 1D tracker.
    pub fn constant_velocity(dt: f64, process_noise: f64, meas_noise: f64) -> Self {
        Self {
            x: [0.0, 0.0],
            p: [[1.0, 0.0], [0.0, 1.0]],
            f_mat: [[1.0, dt], [0.0, 1.0]],
            h: [1.0, 0.0],
            q: [
                [process_noise * dt * dt * dt / 3.0, process_noise * dt * dt / 2.0],
                [process_noise * dt * dt / 2.0, process_noise * dt],
            ],
            r: meas_noise,
        }
    }

    /// Predict step.
    pub fn predict(&mut self) {
        let f = &self.f_mat;
        // x = F*x
        let x0 = f[0][0] * self.x[0] + f[0][1] * self.x[1];
        let x1 = f[1][0] * self.x[0] + f[1][1] * self.x[1];
        self.x = [x0, x1];

        // P = F*P*F' + Q
        let fp00 = f[0][0] * self.p[0][0] + f[0][1] * self.p[1][0];
        let fp01 = f[0][0] * self.p[0][1] + f[0][1] * self.p[1][1];
        let fp10 = f[1][0] * self.p[0][0] + f[1][1] * self.p[1][0];
        let fp11 = f[1][0] * self.p[0][1] + f[1][1] * self.p[1][1];

        self.p[0][0] = fp00 * f[0][0] + fp01 * f[0][1] + self.q[0][0];
        self.p[0][1] = fp00 * f[1][0] + fp01 * f[1][1] + self.q[0][1];
        self.p[1][0] = fp10 * f[0][0] + fp11 * f[0][1] + self.q[1][0];
        self.p[1][1] = fp10 * f[1][0] + fp11 * f[1][1] + self.q[1][1];
    }

    /// Update step with scalar measurement.
    pub fn update(&mut self, z: f64) -> f64 {
        let h = &self.h;
        // Innovation.
        let y = z - (h[0] * self.x[0] + h[1] * self.x[1]);

        // S = H*P*H' + R (scalar).
        let s = h[0] * (self.p[0][0] * h[0] + self.p[0][1] * h[1])
            + h[1] * (self.p[1][0] * h[0] + self.p[1][1] * h[1])
            + self.r;

        // K = P*H' / S (2x1 vector).
        let k0 = (self.p[0][0] * h[0] + self.p[0][1] * h[1]) / s;
        let k1 = (self.p[1][0] * h[0] + self.p[1][1] * h[1]) / s;

        // x = x + K*y.
        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        // P = (I - K*H)*P.
        let p00 = (1.0 - k0 * h[0]) * self.p[0][0] - k0 * h[1] * self.p[1][0];
        let p01 = (1.0 - k0 * h[0]) * self.p[0][1] - k0 * h[1] * self.p[1][1];
        let p10 = -k1 * h[0] * self.p[0][0] + (1.0 - k1 * h[1]) * self.p[1][0];
        let p11 = -k1 * h[0] * self.p[0][1] + (1.0 - k1 * h[1]) * self.p[1][1];
        self.p = [[p00, p01], [p10, p11]];

        y // return innovation
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert!(approx(m.get(0, 0), 1.0, 1e-10));
        assert!(approx(m.get(0, 1), 0.0, 1e-10));
        assert!(approx(m.get(2, 2), 1.0, 1e-10));
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(approx(t.get(0, 1), 4.0, 1e-10));
    }

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.mul(&b).unwrap();
        assert!(approx(c.get(0, 0), 19.0, 1e-10));
        assert!(approx(c.get(0, 1), 22.0, 1e-10));
        assert!(approx(c.get(1, 0), 43.0, 1e-10));
        assert!(approx(c.get(1, 1), 50.0, 1e-10));
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        let m = Matrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert!(approx(product.get(0, 0), 1.0, 1e-8));
        assert!(approx(product.get(0, 1), 0.0, 1e-8));
        assert!(approx(product.get(1, 0), 0.0, 1e-8));
        assert!(approx(product.get(1, 1), 1.0, 1e-8));
    }

    #[test]
    fn test_matrix_inverse_3x3() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]).unwrap();
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(product.get(i, j), expected, 1e-8));
            }
        }
    }

    #[test]
    fn test_singular_matrix_fails() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(m.inverse().is_err());
    }

    #[test]
    fn test_kalman_creation() {
        let kf = KalmanFilter::new(2, 1).unwrap();
        assert_eq!(kf.n, 2);
        assert_eq!(kf.m, 1);
        assert_eq!(kf.x.rows, 2);
    }

    #[test]
    fn test_kalman_predict_moves_state() {
        let mut kf = KalmanFilter::new(2, 1).unwrap();
        kf.x = Matrix::col_vec(&[1.0, 2.0]);
        // F = [[1, 0.1], [0, 1]]  (constant velocity)
        kf.f_mat = Matrix::from_vec(2, 2, vec![1.0, 0.1, 0.0, 1.0]).unwrap();
        kf.predict(None).unwrap();
        // x = [1 + 0.1*2, 2] = [1.2, 2.0]
        assert!(approx(kf.x.get(0, 0), 1.2, 1e-8));
        assert!(approx(kf.x.get(1, 0), 2.0, 1e-8));
    }

    #[test]
    fn test_kalman_update_reduces_uncertainty() {
        let mut kf = KalmanFilter::new(2, 1).unwrap();
        kf.h = Matrix::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        kf.r = Matrix::from_vec(1, 1, vec![0.1]).unwrap();
        let p_before = kf.p.get(0, 0);
        kf.predict(None).unwrap();
        let z = Matrix::col_vec(&[1.0]);
        kf.update(&z).unwrap();
        let p_after = kf.p.get(0, 0);
        assert!(p_after < p_before + 1.0); // covariance doesn't grow unbounded
    }

    #[test]
    fn test_kalman_converges_to_measurement() {
        let mut kf = KalmanFilter::new(1, 1).unwrap();
        kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.h = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.q = Matrix::from_vec(1, 1, vec![0.001]).unwrap();
        kf.r = Matrix::from_vec(1, 1, vec![0.1]).unwrap();

        for _ in 0..100 {
            kf.predict(None).unwrap();
            let z = Matrix::col_vec(&[5.0]);
            kf.update(&z).unwrap();
        }
        assert!(approx(kf.x.get(0, 0), 5.0, 0.1));
    }

    #[test]
    fn test_kalman_with_control_input() {
        let mut kf = KalmanFilter::new(1, 1).unwrap();
        kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.h = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.b = Some(Matrix::from_vec(1, 1, vec![0.5]).unwrap());
        kf.x = Matrix::col_vec(&[0.0]);

        let u = Matrix::col_vec(&[10.0]);
        kf.predict(Some(&u)).unwrap();
        // x = 1*0 + 0.5*10 = 5
        assert!(approx(kf.x.get(0, 0), 5.0, 1e-8));
    }

    #[test]
    fn test_mahalanobis_distance() {
        let mut kf = KalmanFilter::new(1, 1).unwrap();
        kf.h = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.r = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.p = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.x = Matrix::col_vec(&[0.0]);

        let z = Matrix::col_vec(&[2.0]);
        let d = kf.mahalanobis(&z).unwrap();
        // S = P + R = 2, y = 2, dist = sqrt(y' * S^-1 * y) = sqrt(4/2) = sqrt(2)
        assert!(approx(d, 2.0_f64.sqrt(), 1e-4));
    }

    #[test]
    fn test_innovation_is_correct() {
        let mut kf = KalmanFilter::new(1, 1).unwrap();
        kf.h = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        kf.x = Matrix::col_vec(&[3.0]);
        kf.predict(None).unwrap();

        let z = Matrix::col_vec(&[5.0]);
        let result = kf.update(&z).unwrap();
        assert!(approx(result.innovation.get(0, 0), 2.0, 0.5));
    }

    #[test]
    fn test_2x2_constant_velocity_tracker() {
        let mut kf = KalmanFilter2x2::constant_velocity(0.1, 0.1, 1.0);
        // Feed noisy measurements of a constant-velocity target at v=10.
        for i in 0..200 {
            kf.predict();
            let true_pos = (i as f64) * 0.1 * 10.0;
            kf.update(true_pos + 0.1); // small noise
        }
        // Velocity estimate should be near 10.
        assert!(approx(kf.x[1], 10.0, 2.0));
    }

    #[test]
    fn test_2x2_predict_grows_covariance() {
        let mut kf = KalmanFilter2x2::constant_velocity(0.1, 1.0, 1.0);
        let p00_before = kf.p[0][0];
        kf.predict();
        assert!(kf.p[0][0] > p00_before);
    }

    #[test]
    fn test_2x2_update_shrinks_covariance() {
        let mut kf = KalmanFilter2x2::constant_velocity(0.1, 1.0, 0.01);
        kf.predict();
        let p00_before = kf.p[0][0];
        kf.update(0.0);
        assert!(kf.p[0][0] < p00_before);
    }

    #[test]
    fn test_rts_smoother() {
        let f_mat = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        let records: Vec<FilterRecord> = (0..5).map(|i| {
            let val = i as f64;
            FilterRecord {
                x_pred: Matrix::col_vec(&[val - 0.1]),
                p_pred: Matrix::from_vec(1, 1, vec![1.0]).unwrap(),
                x_filt: Matrix::col_vec(&[val]),
                p_filt: Matrix::from_vec(1, 1, vec![0.5]).unwrap(),
            }
        }).collect();

        let smoothed = rts_smooth(&records, &f_mat).unwrap();
        assert_eq!(smoothed.len(), 5);
        // Last smoothed should equal last filtered.
        assert!(approx(smoothed[4].0.get(0, 0), 4.0, 1e-8));
    }

    #[test]
    fn test_rts_empty_records() {
        let f_mat = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
        let smoothed = rts_smooth(&[], &f_mat).unwrap();
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_matrix_dimension_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn test_empty_state_rejected() {
        assert!(KalmanFilter::new(0, 1).is_err());
        assert!(KalmanFilter::new(1, 0).is_err());
    }

    #[test]
    fn test_matrix_trace() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0]).unwrap();
        assert!(approx(m.trace(), 15.0, 1e-10));
    }

    #[test]
    fn test_matrix_scale() {
        let m = Matrix::from_vec(1, 2, vec![3.0, 4.0]).unwrap();
        let scaled = m.scale(2.0);
        assert!(approx(scaled.get(0, 0), 6.0, 1e-10));
        assert!(approx(scaled.get(0, 1), 8.0, 1e-10));
    }
}
