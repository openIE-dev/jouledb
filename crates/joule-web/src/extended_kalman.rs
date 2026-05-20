//! Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) — nonlinear
//! state estimation with Jacobian-based linearization or sigma-point transforms.
//!
//! Includes numerical Jacobian computation, sigma point generation via Cholesky
//! decomposition, weighted mean/covariance, and a 2D tracking example with
//! range/bearing measurements.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// EKF/UKF errors.
#[derive(Debug, Clone, PartialEq)]
pub enum EkfError {
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Singular matrix during inversion.
    SingularMatrix,
    /// Cholesky decomposition failed (matrix not positive definite).
    CholeskyFailed,
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for EkfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::CholeskyFailed => write!(f, "Cholesky decomposition failed"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for EkfError {}

// ── Dense Vector / Matrix Helpers ───────────────────────────────

/// Column vector (thin wrapper over Vec<f64>).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vec64(pub Vec<f64>);

impl Vec64 {
    pub fn zeros(n: usize) -> Self {
        Self(vec![0.0; n])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn add(&self, other: &Vec64) -> Vec64 {
        Vec64(self.0.iter().zip(&other.0).map(|(a, b)| a + b).collect())
    }

    pub fn sub(&self, other: &Vec64) -> Vec64 {
        Vec64(self.0.iter().zip(&other.0).map(|(a, b)| a - b).collect())
    }

    pub fn scale(&self, s: f64) -> Vec64 {
        Vec64(self.0.iter().map(|v| v * s).collect())
    }
}

/// Row-major square matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mat {
    pub n: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(n: usize) -> Self {
        Self { n, data: vec![0.0; n * n] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.n + c] = val;
    }

    /// Rectangular matrix (rows x cols).
    pub fn zeros_rect(rows: usize, cols: usize) -> MatRect {
        MatRect { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.n);
        for r in 0..self.n {
            for c in 0..self.n {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    pub fn mul(&self, other: &Mat) -> Mat {
        let n = self.n;
        let mut result = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn mul_vec(&self, v: &Vec64) -> Vec64 {
        let n = self.n;
        let mut result = Vec64::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.get(i, j) * v.0[j];
            }
            result.0[i] = sum;
        }
        result
    }

    pub fn add(&self, other: &Mat) -> Mat {
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Mat { n: self.n, data }
    }

    pub fn sub(&self, other: &Mat) -> Mat {
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Mat { n: self.n, data }
    }

    pub fn scale(&self, s: f64) -> Mat {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        Mat { n: self.n, data }
    }

    /// Cholesky decomposition: returns lower triangular L such that A = L * L'.
    pub fn cholesky(&self) -> Result<Mat, EkfError> {
        let n = self.n;
        let mut l = Mat::zeros(n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    let diag = self.get(i, i) - sum;
                    if diag <= 0.0 {
                        return Err(EkfError::CholeskyFailed);
                    }
                    l.set(i, j, diag.sqrt());
                } else {
                    let ljj = l.get(j, j);
                    if ljj.abs() < 1e-14 {
                        return Err(EkfError::CholeskyFailed);
                    }
                    l.set(i, j, (self.get(i, j) - sum) / ljj);
                }
            }
        }
        Ok(l)
    }

    /// Gauss-Jordan inverse.
    pub fn inverse(&self) -> Result<Mat, EkfError> {
        let n = self.n;
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.get(i, j);
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
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
                return Err(EkfError::SingularMatrix);
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
                if row == col { continue; }
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }
        let mut inv = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug[i * 2 * n + n + j]);
            }
        }
        Ok(inv)
    }

    /// Outer product of two vectors: v * w'.
    pub fn outer(v: &Vec64, w: &Vec64) -> Mat {
        let n = v.len();
        let mut m = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                m.set(i, j, v.0[i] * w.0[j]);
            }
        }
        m
    }
}

/// Non-square matrix for Jacobians (m x n).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatRect {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MatRect {
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /// Multiply: (m x n) * Vec64(n) -> Vec64(m).
    pub fn mul_vec(&self, v: &Vec64) -> Vec64 {
        let mut result = Vec64::zeros(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j) * v.0[j];
            }
            result.0[i] = sum;
        }
        result
    }

    /// Transpose: (m x n) -> (n x m).
    pub fn transpose(&self) -> MatRect {
        let mut t = MatRect { rows: self.cols, cols: self.rows, data: vec![0.0; self.rows * self.cols] };
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    /// (m x n) * (n x n) -> (m x n)  (right-multiply by square matrix).
    pub fn mul_sq(&self, sq: &Mat) -> MatRect {
        let mut result = MatRect { rows: self.rows, cols: self.cols, data: vec![0.0; self.rows * self.cols] };
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * sq.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// (n x m)' * (n x m) isn't directly needed; we use specific mul patterns instead.
    /// Compute (m x n) * (n x m) -> (m x m) as square.
    pub fn mul_rect_to_sq(&self, other: &MatRect) -> Mat {
        assert_eq!(self.cols, other.rows);
        let m = self.rows;
        let mut result = Mat::zeros(m);
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// (n x n_state) * (n_state x m) -> specialized for gain: Sq * Rect' -> Rect.
    pub fn sq_mul_rect_t(sq: &Mat, rect: &MatRect) -> MatRect {
        // sq is (n x n), rect is (m x n), rect' is (n x m), result is (n x m)
        let rt = rect.transpose();
        let mut result = MatRect { rows: sq.n, cols: rt.cols, data: vec![0.0; sq.n * rt.cols] };
        for i in 0..sq.n {
            for j in 0..rt.cols {
                let mut sum = 0.0;
                for k in 0..sq.n {
                    sum += sq.get(i, k) * rt.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

// ── Numerical Jacobian ──────────────────────────────────────────

/// Compute numerical Jacobian of f(x) using central differences.
pub fn numerical_jacobian(
    f: &dyn Fn(&Vec64) -> Vec64,
    x: &Vec64,
    eps: f64,
) -> MatRect {
    let y0 = f(x);
    let m = y0.len();
    let n = x.len();
    let mut jac = MatRect { rows: m, cols: n, data: vec![0.0; m * n] };
    for j in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus.0[j] += eps;
        x_minus.0[j] -= eps;
        let y_plus = f(&x_plus);
        let y_minus = f(&x_minus);
        for i in 0..m {
            jac.set(i, j, (y_plus.0[i] - y_minus.0[i]) / (2.0 * eps));
        }
    }
    jac
}

// ── Extended Kalman Filter ──────────────────────────────────────

/// Extended Kalman Filter with function-pointer dynamics.
#[derive(Clone)]
pub struct ExtendedKalmanFilter {
    /// State dimension.
    pub n: usize,
    /// Measurement dimension.
    pub m: usize,
    /// State estimate.
    pub x: Vec64,
    /// State covariance.
    pub p: Mat,
    /// Process noise covariance.
    pub q: Mat,
    /// Measurement noise covariance (m x m stored as Mat with m as n).
    pub r_cov: Mat,
    /// Finite difference epsilon for numerical Jacobians.
    pub jac_eps: f64,
}

impl ExtendedKalmanFilter {
    /// Create an EKF with given dimensions.
    pub fn new(n: usize, m: usize) -> Result<Self, EkfError> {
        if n == 0 || m == 0 {
            return Err(EkfError::InvalidConfig("dimensions must be positive".into()));
        }
        Ok(Self {
            n,
            m,
            x: Vec64::zeros(n),
            p: Mat::identity(n),
            q: Mat::identity(n).scale(0.01),
            r_cov: Mat::identity(m).scale(0.1),
            jac_eps: 1e-6,
        })
    }

    /// Predict step with nonlinear f(x) and its Jacobian.
    pub fn predict(
        &mut self,
        f: &dyn Fn(&Vec64) -> Vec64,
        f_jac: Option<&dyn Fn(&Vec64) -> MatRect>,
    ) -> Result<(), EkfError> {
        let jac = match f_jac {
            Some(fj) => fj(&self.x),
            None => numerical_jacobian(f, &self.x, self.jac_eps),
        };

        self.x = f(&self.x);

        // Convert Jacobian (n x n) to Mat for P update.
        if jac.rows != self.n || jac.cols != self.n {
            return Err(EkfError::DimensionMismatch { expected: self.n, got: jac.rows });
        }
        // P = F * P * F' + Q
        // F is jac as MatRect(n x n), P is Mat(n x n).
        let fp = jac.mul_sq(&self.p); // (n x n) * (n x n) -> (n x n) as MatRect
        let ft = jac.transpose();
        let fpft = fp.mul_rect_to_sq(&ft); // result is (n x n)
        self.p = fpft.add(&self.q);

        Ok(())
    }

    /// Update step with nonlinear h(x) and its Jacobian.
    pub fn update(
        &mut self,
        z: &Vec64,
        h: &dyn Fn(&Vec64) -> Vec64,
        h_jac: Option<&dyn Fn(&Vec64) -> MatRect>,
    ) -> Result<Vec64, EkfError> {
        let hjac = match h_jac {
            Some(hj) => hj(&self.x),
            None => numerical_jacobian(h, &self.x, self.jac_eps),
        };

        // Innovation.
        let z_pred = h(&self.x);
        let innovation = z.sub(&z_pred);

        // S = H*P*H' + R.
        let hp = hjac.mul_sq(&self.p);
        let ht = hjac.transpose();
        let hpht = hp.mul_rect_to_sq(&ht);
        let s = hpht.add(&self.r_cov);
        let s_inv = s.inverse()?;

        // K = P*H'*S^-1 => (n x m).
        let pht = MatRect::sq_mul_rect_t(&self.p, &hjac); // (n x m)
        // pht * s_inv: (n x m) * (m x m)
        let k = pht.mul_sq(&s_inv);

        // x = x + K*innovation.
        let correction = k.mul_vec(&innovation);
        self.x = self.x.add(&correction);

        // P = (I - K*H) * P.
        let kh = k.mul_rect_to_sq(&hjac); // this gives (n x n) if k is (n x m) and hjac is (m x n)
        // Wait: k is (n x m), hjac is (m x n). k.mul_rect_to_sq needs self.cols == other.rows.
        // k: rows=n, cols=m. hjac: rows=m, cols=n. So k.cols(m) == hjac.rows(m). Result: (n x n). Correct.
        let i_kh = Mat::identity(self.n).sub(&kh);
        self.p = i_kh.mul(&self.p);

        Ok(innovation)
    }
}

// ── Unscented Kalman Filter ─────────────────────────────────────

/// UKF configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UkfConfig {
    /// State dimension.
    pub n: usize,
    /// Measurement dimension.
    pub m: usize,
    /// Spread parameter (alpha), typically 1e-3.
    pub alpha: f64,
    /// Secondary scaling (kappa), typically 0.
    pub kappa: f64,
    /// Distribution parameter (beta), 2 is optimal for Gaussian.
    pub beta: f64,
}

impl UkfConfig {
    /// Compute lambda = alpha^2 * (n + kappa) - n.
    pub fn lambda(&self) -> f64 {
        self.alpha * self.alpha * (self.n as f64 + self.kappa) - self.n as f64
    }

    /// Compute mean weights for 2n+1 sigma points.
    pub fn weights_mean(&self) -> Vec<f64> {
        let n = self.n as f64;
        let lam = self.lambda();
        let mut w = Vec::with_capacity(2 * self.n + 1);
        w.push(lam / (n + lam));
        let w_i = 1.0 / (2.0 * (n + lam));
        for _ in 0..2 * self.n {
            w.push(w_i);
        }
        w
    }

    /// Compute covariance weights.
    pub fn weights_cov(&self) -> Vec<f64> {
        let n = self.n as f64;
        let lam = self.lambda();
        let mut w = Vec::with_capacity(2 * self.n + 1);
        w.push(lam / (n + lam) + (1.0 - self.alpha * self.alpha + self.beta));
        let w_i = 1.0 / (2.0 * (n + lam));
        for _ in 0..2 * self.n {
            w.push(w_i);
        }
        w
    }
}

/// Generate sigma points from mean and covariance.
pub fn sigma_points(x: &Vec64, p: &Mat, config: &UkfConfig) -> Result<Vec<Vec64>, EkfError> {
    let n = x.len();
    let lam = config.lambda();
    let scale = ((n as f64 + lam) as f64).sqrt();

    let l = p.cholesky()?;

    let mut points = Vec::with_capacity(2 * n + 1);
    points.push(x.clone());

    for i in 0..n {
        // Extract i-th column of L, scaled.
        let mut col = Vec64::zeros(n);
        for r in 0..n {
            col.0[r] = l.get(r, i) * scale;
        }
        points.push(x.add(&col));
        points.push(x.sub(&col));
    }

    Ok(points)
}

/// Compute weighted mean from sigma points.
pub fn weighted_mean(points: &[Vec64], weights: &[f64]) -> Vec64 {
    let n = points[0].len();
    let mut mean = Vec64::zeros(n);
    for (pt, &w) in points.iter().zip(weights) {
        for i in 0..n {
            mean.0[i] += w * pt.0[i];
        }
    }
    mean
}

/// Compute weighted covariance from sigma points.
pub fn weighted_covariance(points: &[Vec64], mean: &Vec64, weights: &[f64]) -> Mat {
    let n = mean.len();
    let mut cov = Mat::zeros(n);
    for (pt, &w) in points.iter().zip(weights) {
        let diff = pt.sub(mean);
        for i in 0..n {
            for j in 0..n {
                cov.data[i * n + j] += w * diff.0[i] * diff.0[j];
            }
        }
    }
    cov
}

/// Compute weighted cross-covariance between two sets of sigma points.
pub fn weighted_cross_covariance(
    x_points: &[Vec64],
    x_mean: &Vec64,
    z_points: &[Vec64],
    z_mean: &Vec64,
    weights: &[f64],
) -> MatRect {
    let nx = x_mean.len();
    let nz = z_mean.len();
    let mut cc = MatRect { rows: nx, cols: nz, data: vec![0.0; nx * nz] };
    for ((xp, zp), &w) in x_points.iter().zip(z_points).zip(weights) {
        let dx = xp.sub(x_mean);
        let dz = zp.sub(z_mean);
        for i in 0..nx {
            for j in 0..nz {
                cc.data[i * nz + j] += w * dx.0[i] * dz.0[j];
            }
        }
    }
    cc
}

// ── 2D Tracking Example ─────────────────────────────────────────

/// Range-bearing measurement from a sensor at origin to a 2D target.
pub fn range_bearing(x: &Vec64) -> Vec64 {
    let px = x.0[0];
    let py = x.0[1];
    let range = (px * px + py * py).sqrt();
    let bearing = py.atan2(px);
    Vec64(vec![range, bearing])
}

/// Jacobian of range-bearing measurement.
pub fn range_bearing_jacobian(x: &Vec64) -> MatRect {
    let px = x.0[0];
    let py = x.0[1];
    let r = (px * px + py * py).sqrt();
    let r2 = px * px + py * py;
    let n = x.len();
    let mut jac = MatRect { rows: 2, cols: n, data: vec![0.0; 2 * n] };
    if r > 1e-12 {
        jac.set(0, 0, px / r);   // dr/dx
        jac.set(0, 1, py / r);   // dr/dy
        jac.set(1, 0, -py / r2); // dbearing/dx
        jac.set(1, 1, px / r2);  // dbearing/dy
    }
    jac
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec64_operations() {
        let a = Vec64(vec![1.0, 2.0]);
        let b = Vec64(vec![3.0, 4.0]);
        let sum = a.add(&b);
        assert!(approx(sum.0[0], 4.0, 1e-10));
        let diff = a.sub(&b);
        assert!(approx(diff.0[0], -2.0, 1e-10));
        let scaled = a.scale(3.0);
        assert!(approx(scaled.0[0], 3.0, 1e-10));
    }

    #[test]
    fn test_cholesky_2x2() {
        let m = Mat { n: 2, data: vec![4.0, 2.0, 2.0, 3.0] };
        let l = m.cholesky().unwrap();
        let lt = l.transpose();
        let product = l.mul(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(product.get(i, j), m.get(i, j), 1e-8));
            }
        }
    }

    #[test]
    fn test_cholesky_3x3() {
        // Positive definite.
        let m = Mat { n: 3, data: vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0] };
        let l = m.cholesky().unwrap();
        let lt = l.transpose();
        let product = l.mul(&lt);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(product.get(i, j), m.get(i, j), 1e-8));
            }
        }
    }

    #[test]
    fn test_cholesky_not_pd() {
        let m = Mat { n: 2, data: vec![-1.0, 0.0, 0.0, 1.0] };
        assert!(m.cholesky().is_err());
    }

    #[test]
    fn test_numerical_jacobian_linear() {
        let f = |x: &Vec64| Vec64(vec![2.0 * x.0[0] + 3.0 * x.0[1], x.0[0] - x.0[1]]);
        let x = Vec64(vec![1.0, 2.0]);
        let jac = numerical_jacobian(&f, &x, 1e-6);
        assert!(approx(jac.get(0, 0), 2.0, 1e-4));
        assert!(approx(jac.get(0, 1), 3.0, 1e-4));
        assert!(approx(jac.get(1, 0), 1.0, 1e-4));
        assert!(approx(jac.get(1, 1), -1.0, 1e-4));
    }

    #[test]
    fn test_numerical_jacobian_nonlinear() {
        let f = |x: &Vec64| Vec64(vec![x.0[0] * x.0[0], x.0[0] * x.0[1]]);
        let x = Vec64(vec![3.0, 2.0]);
        let jac = numerical_jacobian(&f, &x, 1e-6);
        // df0/dx0 = 2x0 = 6, df0/dx1 = 0
        assert!(approx(jac.get(0, 0), 6.0, 1e-4));
        assert!(approx(jac.get(0, 1), 0.0, 1e-4));
        // df1/dx0 = x1 = 2, df1/dx1 = x0 = 3
        assert!(approx(jac.get(1, 0), 2.0, 1e-4));
        assert!(approx(jac.get(1, 1), 3.0, 1e-4));
    }

    #[test]
    fn test_ekf_creation() {
        let ekf = ExtendedKalmanFilter::new(4, 2).unwrap();
        assert_eq!(ekf.n, 4);
        assert_eq!(ekf.m, 2);
    }

    #[test]
    fn test_ekf_predict_linear() {
        let mut ekf = ExtendedKalmanFilter::new(2, 1).unwrap();
        ekf.x = Vec64(vec![1.0, 0.5]);
        let f = |x: &Vec64| Vec64(vec![x.0[0] + 0.1 * x.0[1], x.0[1]]);
        ekf.predict(&f, None).unwrap();
        assert!(approx(ekf.x.0[0], 1.05, 1e-4));
        assert!(approx(ekf.x.0[1], 0.5, 1e-4));
    }

    #[test]
    fn test_ekf_update_pulls_toward_measurement() {
        let mut ekf = ExtendedKalmanFilter::new(1, 1).unwrap();
        ekf.x = Vec64(vec![0.0]);
        ekf.p = Mat::identity(1);
        ekf.r_cov = Mat::identity(1).scale(0.1);

        let h = |x: &Vec64| Vec64(vec![x.0[0]]);
        let z = Vec64(vec![5.0]);
        ekf.update(&z, &h, None).unwrap();
        // Should move toward 5.
        assert!(ekf.x.0[0] > 0.0);
    }

    #[test]
    fn test_range_bearing_at_origin() {
        let x = Vec64(vec![3.0, 4.0]);
        let meas = range_bearing(&x);
        assert!(approx(meas.0[0], 5.0, 1e-8)); // range
        assert!(approx(meas.0[1], (4.0_f64).atan2(3.0), 1e-8)); // bearing
    }

    #[test]
    fn test_range_bearing_jacobian_numerical() {
        let x = Vec64(vec![3.0, 4.0]);
        let analytical = range_bearing_jacobian(&x);
        let numerical = numerical_jacobian(&range_bearing, &x, 1e-7);
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(analytical.get(i, j), numerical.get(i, j), 1e-4));
            }
        }
    }

    #[test]
    fn test_sigma_points_count() {
        let config = UkfConfig { n: 3, m: 2, alpha: 1e-3, kappa: 0.0, beta: 2.0 };
        let x = Vec64(vec![1.0, 2.0, 3.0]);
        let p = Mat::identity(3);
        let pts = sigma_points(&x, &p, &config).unwrap();
        assert_eq!(pts.len(), 2 * 3 + 1);
    }

    #[test]
    fn test_sigma_points_mean_recovers_state() {
        let config = UkfConfig { n: 2, m: 1, alpha: 1.0, kappa: 0.0, beta: 2.0 };
        let x = Vec64(vec![5.0, 10.0]);
        let p = Mat { n: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
        let pts = sigma_points(&x, &p, &config).unwrap();
        let w = config.weights_mean();
        let mean = weighted_mean(&pts, &w);
        assert!(approx(mean.0[0], 5.0, 1e-4));
        assert!(approx(mean.0[1], 10.0, 1e-4));
    }

    #[test]
    fn test_weighted_covariance_identity() {
        let config = UkfConfig { n: 2, m: 1, alpha: 1.0, kappa: 0.0, beta: 2.0 };
        let x = Vec64(vec![0.0, 0.0]);
        let p = Mat::identity(2);
        let pts = sigma_points(&x, &p, &config).unwrap();
        let wm = config.weights_mean();
        let wc = config.weights_cov();
        let mean = weighted_mean(&pts, &wm);
        let cov = weighted_covariance(&pts, &mean, &wc);
        // Covariance should approximate identity.
        assert!(approx(cov.get(0, 0), 1.0, 0.5));
        assert!(approx(cov.get(1, 1), 1.0, 0.5));
    }

    #[test]
    fn test_mat_inverse() {
        let m = Mat { n: 2, data: vec![4.0, 7.0, 2.0, 6.0] };
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        assert!(approx(product.get(0, 0), 1.0, 1e-8));
        assert!(approx(product.get(1, 1), 1.0, 1e-8));
    }

    #[test]
    fn test_mat_outer_product() {
        let v = Vec64(vec![1.0, 2.0]);
        let w = Vec64(vec![3.0, 4.0]);
        let outer = Mat::outer(&v, &w);
        assert!(approx(outer.get(0, 0), 3.0, 1e-10));
        assert!(approx(outer.get(0, 1), 4.0, 1e-10));
        assert!(approx(outer.get(1, 0), 6.0, 1e-10));
        assert!(approx(outer.get(1, 1), 8.0, 1e-10));
    }

    #[test]
    fn test_ekf_invalid_dimensions() {
        assert!(ExtendedKalmanFilter::new(0, 1).is_err());
    }

    #[test]
    fn test_cross_covariance_dimensions() {
        let xp = vec![Vec64(vec![1.0, 2.0]), Vec64(vec![3.0, 4.0])];
        let zp = vec![Vec64(vec![5.0]), Vec64(vec![6.0])];
        let xm = Vec64(vec![2.0, 3.0]);
        let zm = Vec64(vec![5.5]);
        let w = vec![0.5, 0.5];
        let cc = weighted_cross_covariance(&xp, &xm, &zp, &zm, &w);
        assert_eq!(cc.rows, 2);
        assert_eq!(cc.cols, 1);
    }

    #[test]
    fn test_ukf_weights_sum_to_one() {
        let config = UkfConfig { n: 3, m: 2, alpha: 1e-3, kappa: 0.0, beta: 2.0 };
        let wm = config.weights_mean();
        let sum: f64 = wm.iter().sum();
        assert!(approx(sum, 1.0, 1e-8));
    }

    #[test]
    fn test_ekf_2d_tracking_converges() {
        let mut ekf = ExtendedKalmanFilter::new(2, 2).unwrap();
        ekf.x = Vec64(vec![10.0, 10.0]);
        ekf.p = Mat::identity(2).scale(10.0);
        ekf.q = Mat::identity(2).scale(0.01);
        ekf.r_cov = Mat::identity(2).scale(0.1);

        let true_pos = Vec64(vec![5.0, 5.0]);
        let f_identity = |x: &Vec64| x.clone();

        for _ in 0..50 {
            ekf.predict(&f_identity, None).unwrap();
            let z = range_bearing(&true_pos);
            ekf.update(&z, &range_bearing, Some(&range_bearing_jacobian)).unwrap();
        }

        // Should converge near true position.
        assert!(approx(ekf.x.0[0], 5.0, 2.0));
        assert!(approx(ekf.x.0[1], 5.0, 2.0));
    }
}
