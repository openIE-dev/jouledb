//! State Observer — Luenberger observer, reduced-order observer, disturbance
//! observer, dead-beat observer, observer canonical form, detectability checks,
//! and estimation error dynamics analysis.
//!
//! Pure-Rust state estimation for control systems, replacing ad-hoc
//! observer patterns with verified estimation theory.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// State observer errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserverError {
    /// Dimension mismatch.
    DimensionMismatch(String),
    /// System is not detectable (unobservable modes are unstable).
    NotDetectable,
    /// Singular matrix.
    SingularMatrix,
    /// Invalid pole placement.
    InvalidPoles(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for ObserverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::NotDetectable => write!(f, "system is not detectable"),
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::InvalidPoles(msg) => write!(f, "invalid poles: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ObserverError {}

// ── Dense Matrix (Observer-local) ───────────────────────────────

/// Row-major dense matrix for observer computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl ObsMat {
    pub fn zeros(r: usize, c: usize) -> Self {
        Self { rows: r, cols: c, data: vec![0.0; r * c] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, ObserverError> {
        if data.len() != rows * cols {
            return Err(ObserverError::DimensionMismatch(format!(
                "expected {} elements, got {}", rows * cols, data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    pub fn mul(&self, other: &ObsMat) -> Result<ObsMat, ObserverError> {
        if self.cols != other.rows {
            return Err(ObserverError::DimensionMismatch(format!(
                "{}x{} * {}x{}", self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut result = ObsMat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        Ok(result)
    }

    pub fn add(&self, other: &ObsMat) -> Result<ObsMat, ObserverError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(ObserverError::DimensionMismatch("add size mismatch".into()));
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(ObsMat { rows: self.rows, cols: self.cols, data })
    }

    pub fn sub(&self, other: &ObsMat) -> Result<ObsMat, ObserverError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(ObserverError::DimensionMismatch("sub size mismatch".into()));
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Ok(ObsMat { rows: self.rows, cols: self.cols, data })
    }

    pub fn scale(&self, s: f64) -> ObsMat {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        ObsMat { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_vec(&self, v: &[f64]) -> Result<Vec<f64>, ObserverError> {
        if self.cols != v.len() {
            return Err(ObserverError::DimensionMismatch(format!(
                "mat {}x{} * vec {}", self.rows, self.cols, v.len()
            )));
        }
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols { result[i] += self.get(i, j) * v[j]; }
        }
        Ok(result)
    }

    pub fn inverse(&self) -> Result<ObsMat, ObserverError> {
        if self.rows != self.cols {
            return Err(ObserverError::DimensionMismatch("non-square".into()));
        }
        let n = self.rows;
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n { aug[i * 2 * n + j] = self.get(i, j); }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val { max_val = v; max_row = row; }
            }
            if max_val < 1e-14 { return Err(ObserverError::SingularMatrix); }
            if max_row != col {
                for j in 0..2 * n {
                    let tmp = aug[col * 2 * n + j];
                    aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                    aug[max_row * 2 * n + j] = tmp;
                }
            }
            let pivot = aug[col * 2 * n + col];
            for j in 0..2 * n { aug[col * 2 * n + j] /= pivot; }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n { aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j]; }
            }
        }
        let mut inv = ObsMat::zeros(n, n);
        for i in 0..n {
            for j in 0..n { inv.set(i, j, aug[i * 2 * n + n + j]); }
        }
        Ok(inv)
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// ── Observability Analysis ──────────────────────────────────────

/// Compute observability matrix: O = [C; CA; CA^2; ... CA^(n-1)].
pub fn observability_matrix(a: &ObsMat, c: &ObsMat) -> Result<ObsMat, ObserverError> {
    let n = a.rows;
    let p = c.rows; // number of outputs
    let mut obs = ObsMat::zeros(n * p, n);

    let mut ca_power = c.clone(); // C * A^0 = C
    for i in 0..n {
        for r in 0..p {
            for col in 0..n {
                obs.set(i * p + r, col, ca_power.get(r, col));
            }
        }
        if i < n - 1 {
            ca_power = ca_power.mul(a)?;
        }
    }
    Ok(obs)
}

/// Check if system (A, C) is observable (rank of observability matrix == n).
pub fn is_observable(a: &ObsMat, c: &ObsMat) -> Result<bool, ObserverError> {
    let obs = observability_matrix(a, c)?;
    let rank = numerical_rank(&obs, 1e-8);
    Ok(rank >= a.rows)
}

/// Numerical rank via column pivoted Gaussian elimination.
fn numerical_rank(m: &ObsMat, tol: f64) -> usize {
    let rows = m.rows;
    let cols = m.cols;
    let mut work = m.data.clone();
    let mut rank = 0;
    let mut pivot_col = 0;

    for row in 0..rows.min(cols) {
        if pivot_col >= cols { break; }

        // Find pivot.
        let mut max_val = 0.0;
        let mut max_row = row;
        for r in row..rows {
            let v = work[r * cols + pivot_col].abs();
            if v > max_val { max_val = v; max_row = r; }
        }

        if max_val < tol {
            pivot_col += 1;
            continue;
        }

        // Swap rows.
        if max_row != row {
            for c in 0..cols {
                let tmp = work[row * cols + c];
                work[row * cols + c] = work[max_row * cols + c];
                work[max_row * cols + c] = tmp;
            }
        }

        // Eliminate.
        let pivot = work[row * cols + pivot_col];
        for r in (row + 1)..rows {
            let factor = work[r * cols + pivot_col] / pivot;
            for c in pivot_col..cols {
                work[r * cols + c] -= factor * work[row * cols + c];
            }
        }

        rank += 1;
        pivot_col += 1;
    }
    rank
}

// ── Luenberger Observer ─────────────────────────────────────────

/// Discrete-time Luenberger observer.
/// x̂[k+1] = A*x̂[k] + B*u[k] + L*(y[k] - C*x̂[k])
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LuenbergerObserver {
    /// State dimension.
    pub n: usize,
    /// Input dimension.
    pub nu: usize,
    /// Output dimension.
    pub ny: usize,
    /// State estimate.
    pub x_hat: Vec<f64>,
    /// System matrix (n x n).
    pub a: ObsMat,
    /// Input matrix (n x nu).
    pub b: ObsMat,
    /// Output matrix (ny x n).
    pub c: ObsMat,
    /// Observer gain (n x ny).
    pub l: ObsMat,
}

impl LuenbergerObserver {
    /// Create with given system matrices and observer gain.
    pub fn new(
        a: ObsMat, b: ObsMat, c: ObsMat, l: ObsMat,
    ) -> Result<Self, ObserverError> {
        let n = a.rows;
        if a.cols != n { return Err(ObserverError::DimensionMismatch("A not square".into())); }
        let nu = b.cols;
        let ny = c.rows;
        if b.rows != n { return Err(ObserverError::DimensionMismatch("B row mismatch".into())); }
        if c.cols != n { return Err(ObserverError::DimensionMismatch("C col mismatch".into())); }
        if l.rows != n || l.cols != ny {
            return Err(ObserverError::DimensionMismatch("L dimension mismatch".into()));
        }
        Ok(Self {
            n, nu, ny,
            x_hat: vec![0.0; n],
            a, b, c, l,
        })
    }

    /// One-step update: x̂[k+1] = A*x̂ + B*u + L*(y - C*x̂).
    pub fn update(&mut self, u: &[f64], y: &[f64]) -> Result<Vec<f64>, ObserverError> {
        // Predicted output.
        let y_hat = self.c.mul_vec(&self.x_hat)?;

        // Innovation.
        let innovation: Vec<f64> = y.iter().zip(&y_hat).map(|(a, b)| a - b).collect();

        // State prediction.
        let ax = self.a.mul_vec(&self.x_hat)?;
        let bu = self.b.mul_vec(u)?;
        let l_innov = self.l.mul_vec(&innovation)?;

        self.x_hat = ax.iter()
            .zip(&bu)
            .zip(&l_innov)
            .map(|((a, b), c)| a + b + c)
            .collect();

        Ok(self.x_hat.clone())
    }

    /// Get current estimate.
    pub fn estimate(&self) -> &[f64] {
        &self.x_hat
    }

    /// Reset estimate to specific value.
    pub fn reset(&mut self, x0: &[f64]) {
        self.x_hat = x0.to_vec();
    }

    /// Compute error dynamics matrix: A - L*C.
    pub fn error_dynamics(&self) -> Result<ObsMat, ObserverError> {
        let lc = self.l.mul(&self.c)?;
        self.a.sub(&lc)
    }
}

// ── Observer Gain Design ────────────────────────────────────────

/// Design observer gain L for a SISO system using Ackermann's formula approach.
/// For a 2x2 system with desired poles p1, p2:
/// Desired characteristic: (z - p1)(z - p2) = z^2 - (p1+p2)z + p1*p2
pub fn design_gain_2x2(
    a: &ObsMat,
    c: &ObsMat,
    p1: f64,
    p2: f64,
) -> Result<ObsMat, ObserverError> {
    if a.rows != 2 || a.cols != 2 {
        return Err(ObserverError::InvalidPoles("only supports 2x2 systems".into()));
    }
    if c.rows != 1 || c.cols != 2 {
        return Err(ObserverError::DimensionMismatch("C must be 1x2".into()));
    }

    // Check observability first.
    if !is_observable(a, c)? {
        return Err(ObserverError::NotDetectable);
    }

    // We need L = [l1, l2]^T (2x1) such that eig(A - L*C) = {p1, p2}.
    // A - L*C = [[a00 - l1*c0, a01 - l1*c1],
    //            [a10 - l2*c0, a11 - l2*c1]]
    //
    // Constraints:
    //   trace(A - LC) = p1 + p2
    //   det(A - LC)   = p1 * p2
    //
    // Equation 1 (trace): (a00 - l1*c0) + (a11 - l2*c1) = p1 + p2
    //   => l1*c0 + l2*c1 = trace(A) - (p1 + p2)
    //
    // Equation 2 (det):
    //   (a00 - l1*c0)*(a11 - l2*c1) - (a01 - l1*c1)*(a10 - l2*c0) = p1*p2

    let a00 = a.get(0, 0); let a01 = a.get(0, 1);
    let a10 = a.get(1, 0); let a11 = a.get(1, 1);
    let c0 = c.get(0, 0); let c1 = c.get(0, 1);
    let trace_a = a00 + a11;
    let desired_trace = p1 + p2;
    let desired_det = p1 * p2;

    // Solve the 2x2 linear system from the trace and det constraints.
    // After expanding det(A - LC), the determinant constraint is also linear in l1, l2:
    //   det(A-LC) = a00*a11 - a00*l2*c1 - l1*c0*a11 + l1*c0*l2*c1
    //             - a01*a10 + a01*l2*c0 + l1*c1*a10 - l1*c1*l2*c0
    //   The l1*l2 terms cancel: l1*c0*l2*c1 - l1*c1*l2*c0 = 0.
    //   So: det(A-LC) = det(A) - l2*(a00*c1 - a01*c0) - l1*(c0*a11 - c1*a10)
    //
    // System:  [c0,  c1            ] [l1]   [trace(A) - (p1+p2)    ]
    //          [c0*a11 - c1*a10,  a00*c1 - a01*c0] [l2] = [det(A) - p1*p2]

    let det_a = a00 * a11 - a01 * a10;

    let m00 = c0;
    let m01 = c1;
    let rhs0 = trace_a - desired_trace;

    let m10 = c0 * a11 - c1 * a10;
    let m11 = a00 * c1 - a01 * c0;
    let rhs1 = det_a - desired_det;

    let det_m = m00 * m11 - m01 * m10;
    if det_m.abs() < 1e-14 {
        return Err(ObserverError::NotDetectable);
    }

    let l1 = (rhs0 * m11 - rhs1 * m01) / det_m;
    let l2 = (rhs1 * m00 - rhs0 * m10) / det_m;

    ObsMat::from_vec(2, 1, vec![l1, l2])
}

// ── Reduced-Order Observer ──────────────────────────────────────

/// Reduced-order observer: estimates only the unmeasured states.
/// Assumes y = [I 0] * x (first ny states are measured).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedOrderObserver {
    /// Measured state count.
    pub ny: usize,
    /// Unmeasured state count.
    pub nz: usize,
    /// Internal state.
    pub z_hat: Vec<f64>,
    /// Partitioned matrices: A = [[A11, A12], [A21, A22]]
    pub a11: ObsMat, pub a12: ObsMat,
    pub a21: ObsMat, pub a22: ObsMat,
    /// Input partition: B = [B1; B2]
    pub b1: ObsMat, pub b2: ObsMat,
    /// Observer gain for reduced system.
    pub l_r: ObsMat,
}

impl ReducedOrderObserver {
    /// Create from full system partitioned around measured outputs.
    pub fn new(
        a11: ObsMat, a12: ObsMat, a21: ObsMat, a22: ObsMat,
        b1: ObsMat, b2: ObsMat, l_r: ObsMat,
    ) -> Result<Self, ObserverError> {
        let ny = a11.rows;
        let nz = a22.rows;
        if l_r.rows != nz || l_r.cols != ny {
            return Err(ObserverError::DimensionMismatch("L_r dimension mismatch".into()));
        }
        Ok(Self {
            ny, nz,
            z_hat: vec![0.0; nz],
            a11, a12, a21, a22,
            b1, b2, l_r,
        })
    }

    /// Update with input u and measured output y.
    pub fn update(&mut self, u: &[f64], y: &[f64]) -> Result<Vec<f64>, ObserverError> {
        // z_hat[k+1] = (A22 - L*A12)*z_hat + (A21 - L*A11)*y + (B2 - L*B1)*u + L*y[k+1]
        // Simplified single-step: use current y for correction.
        let a22z = self.a22.mul_vec(&self.z_hat)?;
        let la12 = self.l_r.mul(&self.a12)?;
        let la12z = la12.mul_vec(&self.z_hat)?;

        let a21y = self.a21.mul_vec(y)?;
        let la11 = self.l_r.mul(&self.a11)?;
        let la11y = la11.mul_vec(y)?;

        let b2u = self.b2.mul_vec(u)?;
        let lb1 = self.l_r.mul(&self.b1)?;
        let lb1u = lb1.mul_vec(u)?;

        let ly = self.l_r.mul_vec(y)?;

        self.z_hat = (0..self.nz).map(|i| {
            a22z[i] - la12z[i] + a21y[i] - la11y[i] + b2u[i] - lb1u[i] + ly[i]
        }).collect();

        Ok(self.z_hat.clone())
    }

    /// Full state estimate: [y; z_hat].
    pub fn full_estimate(&self, y: &[f64]) -> Vec<f64> {
        let mut x = y.to_vec();
        x.extend_from_slice(&self.z_hat);
        x
    }
}

// ── Disturbance Observer ────────────────────────────────────────

/// Estimates an unknown disturbance d from the model x[k+1] = Ax + Bu + d.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisturbanceObserver {
    /// State dimension.
    pub n: usize,
    /// State estimate.
    pub x_hat: Vec<f64>,
    /// Disturbance estimate.
    pub d_hat: Vec<f64>,
    /// System matrix.
    pub a: ObsMat,
    /// Input matrix.
    pub b: ObsMat,
    /// Output matrix.
    pub c: ObsMat,
    /// Observer gain for state.
    pub l_x: ObsMat,
    /// Observer gain for disturbance (learning rate).
    pub l_d: ObsMat,
}

impl DisturbanceObserver {
    /// Create a disturbance observer.
    pub fn new(
        a: ObsMat, b: ObsMat, c: ObsMat, l_x: ObsMat, l_d: ObsMat,
    ) -> Result<Self, ObserverError> {
        let n = a.rows;
        Ok(Self {
            n,
            x_hat: vec![0.0; n],
            d_hat: vec![0.0; n],
            a, b, c, l_x, l_d,
        })
    }

    /// Update with input u and measured output y.
    pub fn update(&mut self, u: &[f64], y: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ObserverError> {
        let y_hat = self.c.mul_vec(&self.x_hat)?;
        let innovation: Vec<f64> = y.iter().zip(&y_hat).map(|(a, b)| a - b).collect();

        let ax = self.a.mul_vec(&self.x_hat)?;
        let bu = self.b.mul_vec(u)?;
        let lx_inn = self.l_x.mul_vec(&innovation)?;

        // State update: x_hat = Ax + Bu + d_hat + Lx*(y - Cx_hat)
        self.x_hat = (0..self.n).map(|i| {
            ax[i] + bu[i] + self.d_hat[i] + lx_inn[i]
        }).collect();

        // Disturbance update: d_hat += Ld*(y - Cx_hat)
        let ld_inn = self.l_d.mul_vec(&innovation)?;
        for i in 0..self.n {
            self.d_hat[i] += ld_inn[i];
        }

        Ok((self.x_hat.clone(), self.d_hat.clone()))
    }
}

// ── Dead-Beat Observer ──────────────────────────────────────────

/// Dead-beat observer: converges in exactly n steps (discrete-time).
/// Achieved by placing all observer poles at z = 0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeadBeatObserver {
    /// Underlying Luenberger observer with dead-beat gain.
    pub observer: LuenbergerObserver,
    /// Steps since last reset.
    pub steps: usize,
}

impl DeadBeatObserver {
    /// Create dead-beat observer for a 2x2 SISO system.
    /// Poles at z=0 for fastest convergence.
    pub fn new_2x2(
        a: ObsMat, b: ObsMat, c: ObsMat,
    ) -> Result<Self, ObserverError> {
        // Place both poles at 0.
        let l = design_gain_2x2(&a, &c, 0.0, 0.0)?;
        let observer = LuenbergerObserver::new(a, b, c, l)?;
        Ok(Self { observer, steps: 0 })
    }

    /// Update and track convergence.
    pub fn update(&mut self, u: &[f64], y: &[f64]) -> Result<Vec<f64>, ObserverError> {
        self.steps += 1;
        self.observer.update(u, y)
    }

    /// Whether the observer has converged (n steps elapsed).
    pub fn converged(&self) -> bool {
        self.steps >= self.observer.n
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.observer.x_hat = vec![0.0; self.observer.n];
        self.steps = 0;
    }
}

// ── Observer Canonical Form ─────────────────────────────────────

/// Convert a 2x2 SISO system to observer canonical form.
/// OCF: A = [[0, 1], [-a0, -a1]], C = [1, 0].
pub fn observer_canonical_form_2x2(
    a: &ObsMat,
    c: &ObsMat,
) -> Result<(ObsMat, ObsMat), ObserverError> {
    // Characteristic polynomial coefficients.
    let trace_a = a.get(0, 0) + a.get(1, 1);
    let det_a = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);

    // Check observability.
    if !is_observable(a, c)? {
        return Err(ObserverError::NotDetectable);
    }

    // OCF form.
    let a_ocf = ObsMat::from_vec(2, 2, vec![0.0, 1.0, -det_a, -trace_a])?;
    let c_ocf = ObsMat::from_vec(1, 2, vec![1.0, 0.0])?;

    Ok((a_ocf, c_ocf))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn simple_system() -> (ObsMat, ObsMat, ObsMat) {
        // x[k+1] = [[0.9, 0.1], [0, 0.8]] * x + [[0], [1]] * u
        // y = [1, 0] * x
        let a = ObsMat::from_vec(2, 2, vec![0.9, 0.1, 0.0, 0.8]).unwrap();
        let b = ObsMat::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
        let c = ObsMat::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        (a, b, c)
    }

    #[test]
    fn test_observability_matrix_size() {
        let (a, _, c) = simple_system();
        let obs = observability_matrix(&a, &c).unwrap();
        assert_eq!(obs.rows, 2); // n * p = 2 * 1
        assert_eq!(obs.cols, 2);
    }

    #[test]
    fn test_system_is_observable() {
        let (a, _, c) = simple_system();
        assert!(is_observable(&a, &c).unwrap());
    }

    #[test]
    fn test_unobservable_system() {
        let a = ObsMat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let c = ObsMat::from_vec(1, 2, vec![1.0, 0.0]).unwrap(); // Only measures x1
        assert!(!is_observable(&a, &c).unwrap());
    }

    #[test]
    fn test_luenberger_creation() {
        let (a, b, c) = simple_system();
        let l = ObsMat::from_vec(2, 1, vec![0.5, 0.3]).unwrap();
        let obs = LuenbergerObserver::new(a, b, c, l).unwrap();
        assert_eq!(obs.n, 2);
    }

    #[test]
    fn test_luenberger_converges() {
        let (a, b, c) = simple_system();
        let l = ObsMat::from_vec(2, 1, vec![0.5, 0.3]).unwrap();
        let mut obs = LuenbergerObserver::new(a.clone(), b.clone(), c.clone(), l).unwrap();

        // True state.
        let mut x_true = vec![5.0, 3.0];
        let u = vec![0.0];

        for _ in 0..50 {
            let y_true = c.mul_vec(&x_true).unwrap();
            obs.update(&u, &y_true).unwrap();
            // Propagate true state.
            let ax = a.mul_vec(&x_true).unwrap();
            let bu = b.mul_vec(&u).unwrap();
            x_true = ax.iter().zip(&bu).map(|(a, b)| a + b).collect();
        }

        // Estimate should be close to true state.
        assert!(approx(obs.x_hat[0], x_true[0], 0.5));
    }

    #[test]
    fn test_error_dynamics_matrix() {
        let (a, b, c) = simple_system();
        let l = ObsMat::from_vec(2, 1, vec![0.5, 0.3]).unwrap();
        let obs = LuenbergerObserver::new(a, b, c, l).unwrap();
        let err_dyn = obs.error_dynamics().unwrap();
        assert_eq!(err_dyn.rows, 2);
        assert_eq!(err_dyn.cols, 2);
    }

    #[test]
    fn test_design_gain_2x2() {
        let (a, _, c) = simple_system();
        let l = design_gain_2x2(&a, &c, 0.3, 0.2).unwrap();
        assert_eq!(l.rows, 2);
        assert_eq!(l.cols, 1);
        // Verify error dynamics has desired eigenvalues.
        let lc = l.mul(&c).unwrap();
        let a_lc = a.sub(&lc).unwrap();
        let trace = a_lc.get(0, 0) + a_lc.get(1, 1);
        let det = a_lc.get(0, 0) * a_lc.get(1, 1) - a_lc.get(0, 1) * a_lc.get(1, 0);
        assert!(approx(trace, 0.3 + 0.2, 1e-8));
        assert!(approx(det, 0.3 * 0.2, 1e-8));
    }

    #[test]
    fn test_dead_beat_observer() {
        let (a, b, c) = simple_system();
        let mut dbo = DeadBeatObserver::new_2x2(a.clone(), b.clone(), c.clone()).unwrap();
        assert!(!dbo.converged());

        let mut x_true = vec![5.0, 3.0];
        let u = vec![0.0];
        for _ in 0..2 {
            let y = c.mul_vec(&x_true).unwrap();
            dbo.update(&u, &y).unwrap();
            let ax = a.mul_vec(&x_true).unwrap();
            x_true = ax.iter().zip(b.mul_vec(&u).unwrap().iter()).map(|(a, b)| a + b).collect();
        }
        assert!(dbo.converged());
    }

    #[test]
    fn test_dead_beat_reset() {
        let (a, b, c) = simple_system();
        let mut dbo = DeadBeatObserver::new_2x2(a, b, c).unwrap();
        dbo.steps = 5;
        dbo.reset();
        assert_eq!(dbo.steps, 0);
        assert!(approx(dbo.observer.x_hat[0], 0.0, 1e-10));
    }

    #[test]
    fn test_observer_canonical_form() {
        let (a, _, c) = simple_system();
        let (a_ocf, c_ocf) = observer_canonical_form_2x2(&a, &c).unwrap();
        assert!(approx(c_ocf.get(0, 0), 1.0, 1e-10));
        assert!(approx(c_ocf.get(0, 1), 0.0, 1e-10));
        // A_OCF should have [0, 1] in first row.
        assert!(approx(a_ocf.get(0, 0), 0.0, 1e-10));
        assert!(approx(a_ocf.get(0, 1), 1.0, 1e-10));
    }

    #[test]
    fn test_disturbance_observer() {
        let (a, b, c) = simple_system();
        let l_x = ObsMat::from_vec(2, 1, vec![0.5, 0.3]).unwrap();
        let l_d = ObsMat::from_vec(2, 1, vec![0.01, 0.01]).unwrap();
        let mut dobs = DisturbanceObserver::new(a, b, c, l_x, l_d).unwrap();

        let u = vec![0.0];
        let y = vec![1.0];
        let (x_est, d_est) = dobs.update(&u, &y).unwrap();
        assert_eq!(x_est.len(), 2);
        assert_eq!(d_est.len(), 2);
    }

    #[test]
    fn test_reduced_order_observer() {
        // System with 1 measured state and 1 unmeasured.
        let a11 = ObsMat::from_vec(1, 1, vec![0.9]).unwrap();
        let a12 = ObsMat::from_vec(1, 1, vec![0.1]).unwrap();
        let a21 = ObsMat::from_vec(1, 1, vec![0.0]).unwrap();
        let a22 = ObsMat::from_vec(1, 1, vec![0.8]).unwrap();
        let b1 = ObsMat::from_vec(1, 1, vec![0.0]).unwrap();
        let b2 = ObsMat::from_vec(1, 1, vec![1.0]).unwrap();
        let l_r = ObsMat::from_vec(1, 1, vec![0.5]).unwrap();

        let mut robs = ReducedOrderObserver::new(a11, a12, a21, a22, b1, b2, l_r).unwrap();
        let y = vec![1.0];
        let u = vec![0.0];
        robs.update(&u, &y).unwrap();
        let full = robs.full_estimate(&y);
        assert_eq!(full.len(), 2);
        assert!(approx(full[0], 1.0, 1e-10)); // measured state
    }

    #[test]
    fn test_luenberger_reset() {
        let (a, b, c) = simple_system();
        let l = ObsMat::from_vec(2, 1, vec![0.5, 0.3]).unwrap();
        let mut obs = LuenbergerObserver::new(a, b, c, l).unwrap();
        obs.x_hat = vec![10.0, 20.0];
        obs.reset(&[0.0, 0.0]);
        assert!(approx(obs.x_hat[0], 0.0, 1e-10));
    }

    #[test]
    fn test_obs_mat_frobenius_norm() {
        let m = ObsMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(approx(m.frobenius_norm(), expected, 1e-10));
    }

    #[test]
    fn test_obs_mat_inverse() {
        let m = ObsMat::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert!(approx(product.get(0, 0), 1.0, 1e-8));
        assert!(approx(product.get(1, 1), 1.0, 1e-8));
    }

    #[test]
    fn test_numerical_rank_full() {
        let m = ObsMat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(numerical_rank(&m, 1e-8), 2);
    }

    #[test]
    fn test_numerical_rank_deficient() {
        let m = ObsMat::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(numerical_rank(&m, 1e-8), 1);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let a = ObsMat::zeros(2, 2);
        let b = ObsMat::zeros(3, 1);
        let c = ObsMat::zeros(1, 2);
        let l = ObsMat::zeros(2, 1);
        assert!(LuenbergerObserver::new(a, b, c, l).is_err());
    }

    #[test]
    fn test_design_gain_unobservable_fails() {
        let a = ObsMat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let c = ObsMat::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        assert!(design_gain_2x2(&a, &c, 0.3, 0.2).is_err());
    }
}
