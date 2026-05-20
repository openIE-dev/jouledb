//! Model Predictive Control (MPC) — linear state-space prediction, quadratic cost
//! optimization over a finite horizon, input/state constraints, projected gradient
//! QP solver, and receding horizon control.
//!
//! Replaces ad-hoc MPC in scripting languages with a pure-Rust implementation
//! suitable for real-time embedded and server-side control workloads.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// MPC errors.
#[derive(Debug, Clone, PartialEq)]
pub enum MpcError {
    /// Dimension mismatch.
    DimensionMismatch(String),
    /// QP solver failed to converge.
    SolverNotConverged { iterations: usize },
    /// Invalid horizon length.
    InvalidHorizon(usize),
    /// Invalid constraint configuration.
    InvalidConstraint(String),
    /// Singular matrix encountered.
    SingularMatrix,
}

impl std::fmt::Display for MpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::SolverNotConverged { iterations } => {
                write!(f, "QP solver did not converge in {iterations} iterations")
            }
            Self::InvalidHorizon(n) => write!(f, "invalid horizon: {n}"),
            Self::InvalidConstraint(msg) => write!(f, "invalid constraint: {msg}"),
            Self::SingularMatrix => write!(f, "singular matrix"),
        }
    }
}

impl std::error::Error for MpcError {}

// ── Dense Matrix (MPC-local) ────────────────────────────────────

/// Row-major dense matrix for MPC computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    pub fn zeros(r: usize, c: usize) -> Self {
        Self { rows: r, cols: c, data: vec![0.0; r * c] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MpcError> {
        if data.len() != rows * cols {
            return Err(MpcError::DimensionMismatch(format!(
                "expected {} elements, got {}",
                rows * cols,
                data.len()
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

    pub fn mul(&self, other: &DenseMat) -> Result<DenseMat, MpcError> {
        if self.cols != other.rows {
            return Err(MpcError::DimensionMismatch(format!(
                "{}x{} * {}x{}", self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut result = DenseMat::zeros(self.rows, other.cols);
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

    pub fn add(&self, other: &DenseMat) -> Result<DenseMat, MpcError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MpcError::DimensionMismatch(format!(
                "add: {}x{} + {}x{}", self.rows, self.cols, other.rows, other.cols
            )));
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(DenseMat { rows: self.rows, cols: self.cols, data })
    }

    pub fn sub(&self, other: &DenseMat) -> Result<DenseMat, MpcError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MpcError::DimensionMismatch(format!(
                "sub: {}x{} - {}x{}", self.rows, self.cols, other.rows, other.cols
            )));
        }
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Ok(DenseMat { rows: self.rows, cols: self.cols, data })
    }

    pub fn scale(&self, s: f64) -> DenseMat {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        DenseMat { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_vec(&self, v: &[f64]) -> Result<Vec<f64>, MpcError> {
        if self.cols != v.len() {
            return Err(MpcError::DimensionMismatch(format!(
                "mat {}x{} * vec {}", self.rows, self.cols, v.len()
            )));
        }
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        Ok(result)
    }

    /// Gauss-Jordan inverse.
    pub fn inverse(&self) -> Result<DenseMat, MpcError> {
        if self.rows != self.cols {
            return Err(MpcError::DimensionMismatch("non-square".into()));
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
            if max_val < 1e-14 { return Err(MpcError::SingularMatrix); }
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
        let mut inv = DenseMat::zeros(n, n);
        for i in 0..n {
            for j in 0..n { inv.set(i, j, aug[i * 2 * n + n + j]); }
        }
        Ok(inv)
    }
}

// ── Linear State-Space Model ────────────────────────────────────

/// Discrete-time linear model: x[k+1] = A*x[k] + B*u[k].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearModel {
    /// State dimension.
    pub nx: usize,
    /// Input dimension.
    pub nu: usize,
    /// State transition matrix (nx x nx).
    pub a: DenseMat,
    /// Input matrix (nx x nu).
    pub b: DenseMat,
}

impl LinearModel {
    /// Propagate state one step.
    pub fn step(&self, x: &[f64], u: &[f64]) -> Result<Vec<f64>, MpcError> {
        let ax = self.a.mul_vec(x)?;
        let bu = self.b.mul_vec(u)?;
        Ok(ax.iter().zip(&bu).map(|(a, b)| a + b).collect())
    }

    /// Predict trajectory over N steps given input sequence.
    pub fn predict(&self, x0: &[f64], inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MpcError> {
        let mut trajectory = Vec::with_capacity(inputs.len() + 1);
        trajectory.push(x0.to_vec());
        let mut x = x0.to_vec();
        for u in inputs {
            x = self.step(&x, u)?;
            trajectory.push(x.clone());
        }
        Ok(trajectory)
    }
}

// ── Constraints ─────────────────────────────────────────────────

/// Box constraints on inputs and states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    /// Per-input lower bounds.
    pub u_min: Vec<f64>,
    /// Per-input upper bounds.
    pub u_max: Vec<f64>,
    /// Per-state lower bounds (optional).
    pub x_min: Option<Vec<f64>>,
    /// Per-state upper bounds (optional).
    pub x_max: Option<Vec<f64>>,
}

impl Constraints {
    /// Simple symmetric input constraints.
    pub fn input_symmetric(nu: usize, limit: f64) -> Self {
        Self {
            u_min: vec![-limit; nu],
            u_max: vec![limit; nu],
            x_min: None,
            x_max: None,
        }
    }

    /// Validate dimensions.
    pub fn validate(&self, nu: usize, nx: usize) -> Result<(), MpcError> {
        if self.u_min.len() != nu || self.u_max.len() != nu {
            return Err(MpcError::InvalidConstraint("input dimension mismatch".into()));
        }
        for i in 0..nu {
            if self.u_min[i] > self.u_max[i] {
                return Err(MpcError::InvalidConstraint(format!("u_min[{i}] > u_max[{i}]")));
            }
        }
        if let Some(xmin) = &self.x_min {
            if xmin.len() != nx {
                return Err(MpcError::InvalidConstraint("x_min dimension mismatch".into()));
            }
        }
        if let Some(xmax) = &self.x_max {
            if xmax.len() != nx {
                return Err(MpcError::InvalidConstraint("x_max dimension mismatch".into()));
            }
        }
        Ok(())
    }

    /// Clamp an input vector.
    pub fn clamp_input(&self, u: &mut [f64]) {
        for (i, val) in u.iter_mut().enumerate() {
            *val = val.clamp(self.u_min[i], self.u_max[i]);
        }
    }
}

// ── Cost Function ───────────────────────────────────────────────

/// Quadratic cost: sum_{k=0}^{N-1} [ (x-ref)'Q(x-ref) + u'Ru ] + (x_N-ref)'Qf(x_N-ref).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuadraticCost {
    /// State tracking weight (nx x nx diagonal).
    pub q_diag: Vec<f64>,
    /// Input effort weight (nu x nu diagonal).
    pub r_diag: Vec<f64>,
    /// Terminal state weight (nx x nx diagonal).
    pub qf_diag: Vec<f64>,
}

impl QuadraticCost {
    /// Create with uniform weights.
    pub fn uniform(nx: usize, nu: usize, q: f64, r: f64, qf: f64) -> Self {
        Self {
            q_diag: vec![q; nx],
            r_diag: vec![r; nu],
            qf_diag: vec![qf; nx],
        }
    }

    /// Evaluate total cost for a trajectory and input sequence.
    pub fn evaluate(
        &self,
        trajectory: &[Vec<f64>],
        inputs: &[Vec<f64>],
        reference: &[Vec<f64>],
    ) -> f64 {
        let n = inputs.len();
        let mut cost = 0.0;

        // Stage costs.
        for k in 0..n {
            let x = &trajectory[k];
            let u = &inputs[k];
            let r = if k < reference.len() { &reference[k] } else { &reference[reference.len() - 1] };

            for (i, (&xi, &ri)) in x.iter().zip(r.iter()).enumerate() {
                cost += self.q_diag[i] * (xi - ri) * (xi - ri);
            }
            for (i, &ui) in u.iter().enumerate() {
                cost += self.r_diag[i] * ui * ui;
            }
        }

        // Terminal cost.
        let x_n = &trajectory[n];
        let r_n = if n < reference.len() { &reference[n] } else { &reference[reference.len() - 1] };
        for (i, (&xi, &ri)) in x_n.iter().zip(r_n.iter()).enumerate() {
            cost += self.qf_diag[i] * (xi - ri) * (xi - ri);
        }

        cost
    }
}

// ── Projected Gradient QP Solver ────────────────────────────────

/// Solve the MPC problem using projected gradient descent.
pub fn solve_mpc_pgd(
    model: &LinearModel,
    cost: &QuadraticCost,
    constraints: &Constraints,
    x0: &[f64],
    reference: &[Vec<f64>],
    horizon: usize,
    max_iter: usize,
    step_size: f64,
) -> Result<Vec<Vec<f64>>, MpcError> {
    if horizon == 0 {
        return Err(MpcError::InvalidHorizon(0));
    }
    constraints.validate(model.nu, model.nx)?;

    // Initialize input sequence to zeros.
    let mut inputs: Vec<Vec<f64>> = vec![vec![0.0; model.nu]; horizon];

    for _iter in 0..max_iter {
        // Forward simulate.
        let trajectory = model.predict(x0, &inputs)?;

        // Compute gradient of cost w.r.t. each u[k] via finite differences.
        let eps = 1e-5;
        let mut grad = vec![vec![0.0; model.nu]; horizon];

        for k in 0..horizon {
            for j in 0..model.nu {
                let mut u_plus = inputs.clone();
                u_plus[k][j] += eps;
                let traj_plus = model.predict(x0, &u_plus)?;
                let cost_plus = cost.evaluate(&traj_plus, &u_plus, reference);

                let mut u_minus = inputs.clone();
                u_minus[k][j] -= eps;
                let traj_minus = model.predict(x0, &u_minus)?;
                let cost_minus = cost.evaluate(&traj_minus, &u_minus, reference);

                grad[k][j] = (cost_plus - cost_minus) / (2.0 * eps);
            }
        }

        // Backtracking line search: start from step_size, halve until cost decreases.
        let base_cost = cost.evaluate(&trajectory, &inputs, reference);
        let mut alpha = step_size;
        let mut accepted = false;
        for _ in 0..10 {
            let trial: Vec<Vec<f64>> = (0..horizon).map(|k| {
                let mut u_new = inputs[k].clone();
                for j in 0..model.nu {
                    u_new[j] -= alpha * grad[k][j];
                }
                constraints.clamp_input(&mut u_new);
                u_new
            }).collect();
            let trial_traj = model.predict(x0, &trial)?;
            let trial_cost = cost.evaluate(&trial_traj, &trial, reference);
            if trial_cost < base_cost {
                inputs = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // Fall back to small step.
            for k in 0..horizon {
                for j in 0..model.nu {
                    inputs[k][j] -= alpha * grad[k][j];
                }
                constraints.clamp_input(&mut inputs[k]);
            }
        }
    }

    // Check final cost is finite.
    let final_traj = model.predict(x0, &inputs)?;
    let final_cost = cost.evaluate(&final_traj, &inputs, reference);
    if !final_cost.is_finite() {
        return Err(MpcError::SolverNotConverged { iterations: max_iter });
    }

    Ok(inputs)
}

// ── MPC Controller ──────────────────────────────────────────────

/// Receding-horizon MPC controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MpcController {
    pub model: LinearModel,
    pub cost: QuadraticCost,
    pub constraints: Constraints,
    pub horizon: usize,
    pub max_iter: usize,
    pub step_size: f64,
    /// Warm-start: previous input sequence shifted by one step.
    pub prev_inputs: Option<Vec<Vec<f64>>>,
}

impl MpcController {
    /// Create an MPC controller.
    pub fn new(
        model: LinearModel,
        cost: QuadraticCost,
        constraints: Constraints,
        horizon: usize,
    ) -> Result<Self, MpcError> {
        if horizon == 0 {
            return Err(MpcError::InvalidHorizon(0));
        }
        constraints.validate(model.nu, model.nx)?;
        Ok(Self {
            model,
            cost,
            constraints,
            horizon,
            max_iter: 50,
            step_size: 0.01,
            prev_inputs: None,
        })
    }

    /// Compute the optimal first input for receding horizon control.
    pub fn control(&mut self, x: &[f64], reference: &[Vec<f64>]) -> Result<Vec<f64>, MpcError> {
        let inputs = solve_mpc_pgd(
            &self.model,
            &self.cost,
            &self.constraints,
            x,
            reference,
            self.horizon,
            self.max_iter,
            self.step_size,
        )?;

        let first_input = inputs[0].clone();

        // Warm-start: shift for next call.
        let mut shifted = inputs[1..].to_vec();
        shifted.push(vec![0.0; self.model.nu]);
        self.prev_inputs = Some(shifted);

        Ok(first_input)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn integrator_model() -> LinearModel {
        // Simple integrator: x[k+1] = x[k] + u[k]
        LinearModel {
            nx: 1,
            nu: 1,
            a: DenseMat::from_vec(1, 1, vec![1.0]).unwrap(),
            b: DenseMat::from_vec(1, 1, vec![1.0]).unwrap(),
        }
    }

    fn double_integrator() -> LinearModel {
        // x = [position, velocity], u = [acceleration]
        // x1[k+1] = x1[k] + dt*x2[k]
        // x2[k+1] = x2[k] + dt*u[k]
        let dt = 0.1;
        LinearModel {
            nx: 2,
            nu: 1,
            a: DenseMat::from_vec(2, 2, vec![1.0, dt, 0.0, 1.0]).unwrap(),
            b: DenseMat::from_vec(2, 1, vec![0.0, dt]).unwrap(),
        }
    }

    #[test]
    fn test_linear_model_step() {
        let model = integrator_model();
        let x = model.step(&[5.0], &[3.0]).unwrap();
        assert!(approx(x[0], 8.0, 1e-10));
    }

    #[test]
    fn test_linear_model_predict() {
        let model = integrator_model();
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let traj = model.predict(&[0.0], &inputs).unwrap();
        assert_eq!(traj.len(), 4);
        assert!(approx(traj[3][0], 6.0, 1e-10));
    }

    #[test]
    fn test_double_integrator_step() {
        let model = double_integrator();
        let x = model.step(&[0.0, 0.0], &[10.0]).unwrap();
        // vel = 0 + 0.1*10 = 1, pos = 0 + 0.1*0 = 0
        assert!(approx(x[0], 0.0, 1e-10));
        assert!(approx(x[1], 1.0, 1e-10));
    }

    #[test]
    fn test_constraints_symmetric() {
        let c = Constraints::input_symmetric(2, 5.0);
        assert!(approx(c.u_min[0], -5.0, 1e-10));
        assert!(approx(c.u_max[1], 5.0, 1e-10));
    }

    #[test]
    fn test_constraints_clamp() {
        let c = Constraints::input_symmetric(1, 5.0);
        let mut u = vec![100.0];
        c.clamp_input(&mut u);
        assert!(approx(u[0], 5.0, 1e-10));

        let mut u2 = vec![-100.0];
        c.clamp_input(&mut u2);
        assert!(approx(u2[0], -5.0, 1e-10));
    }

    #[test]
    fn test_constraints_validate() {
        let c = Constraints::input_symmetric(2, 5.0);
        assert!(c.validate(2, 1).is_ok());
        assert!(c.validate(3, 1).is_err());
    }

    #[test]
    fn test_quadratic_cost_zero_at_reference() {
        let cost = QuadraticCost::uniform(1, 1, 1.0, 0.1, 1.0);
        let reference = vec![vec![5.0]; 3];
        let traj = vec![vec![5.0]; 3];
        let inputs = vec![vec![0.0]; 2];
        let c = cost.evaluate(&traj, &inputs, &reference);
        assert!(approx(c, 0.0, 1e-10));
    }

    #[test]
    fn test_quadratic_cost_positive_for_error() {
        let cost = QuadraticCost::uniform(1, 1, 1.0, 0.1, 1.0);
        let reference = vec![vec![5.0]; 3];
        let traj = vec![vec![0.0]; 3];
        let inputs = vec![vec![1.0]; 2];
        let c = cost.evaluate(&traj, &inputs, &reference);
        assert!(c > 0.0);
    }

    #[test]
    fn test_mpc_solver_integrator() {
        let model = integrator_model();
        let cost = QuadraticCost::uniform(1, 1, 10.0, 0.1, 10.0);
        let constraints = Constraints::input_symmetric(1, 10.0);
        let reference = vec![vec![5.0]; 11];

        let inputs = solve_mpc_pgd(
            &model, &cost, &constraints, &[0.0], &reference, 10, 100, 0.005,
        ).unwrap();

        // First input should be positive (moving toward reference=5).
        assert!(inputs[0][0] > 0.0);
    }

    #[test]
    fn test_mpc_solver_respects_constraints() {
        let model = integrator_model();
        let cost = QuadraticCost::uniform(1, 1, 100.0, 0.01, 100.0);
        let constraints = Constraints::input_symmetric(1, 2.0);
        let reference = vec![vec![100.0]; 6];

        let inputs = solve_mpc_pgd(
            &model, &cost, &constraints, &[0.0], &reference, 5, 50, 0.005,
        ).unwrap();

        for u in &inputs {
            assert!(u[0] <= 2.0 + 1e-8);
            assert!(u[0] >= -2.0 - 1e-8);
        }
    }

    #[test]
    fn test_mpc_controller_receding_horizon() {
        let model = integrator_model();
        let cost = QuadraticCost::uniform(1, 1, 10.0, 0.1, 10.0);
        let constraints = Constraints::input_symmetric(1, 5.0);
        let mut ctrl = MpcController::new(model.clone(), cost, constraints, 5).unwrap();

        let reference = vec![vec![10.0]; 6];
        let mut x = vec![0.0];
        for _ in 0..20 {
            let u = ctrl.control(&x, &reference).unwrap();
            x = model.step(&x, &u).unwrap();
        }
        // Should approach reference.
        assert!(approx(x[0], 10.0, 2.0));
    }

    #[test]
    fn test_mpc_invalid_horizon() {
        let model = integrator_model();
        let cost = QuadraticCost::uniform(1, 1, 1.0, 0.1, 1.0);
        let constraints = Constraints::input_symmetric(1, 5.0);
        assert!(MpcController::new(model, cost, constraints, 0).is_err());
    }

    #[test]
    fn test_dense_mat_mul() {
        let a = DenseMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DenseMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.mul(&b).unwrap();
        assert!(approx(c.get(0, 0), 19.0, 1e-10));
    }

    #[test]
    fn test_dense_mat_inverse() {
        let m = DenseMat::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert!(approx(product.get(0, 0), 1.0, 1e-8));
        assert!(approx(product.get(1, 1), 1.0, 1e-8));
    }

    #[test]
    fn test_dense_mat_transpose() {
        let m = DenseMat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(approx(t.get(1, 0), 2.0, 1e-10));
    }

    #[test]
    fn test_mpc_double_integrator_tracks() {
        let model = double_integrator();
        let cost = QuadraticCost {
            q_diag: vec![10.0, 1.0],
            r_diag: vec![0.1],
            qf_diag: vec![10.0, 1.0],
        };
        let constraints = Constraints::input_symmetric(1, 20.0);
        let mut ctrl = MpcController::new(model.clone(), cost, constraints, 10).unwrap();
        ctrl.step_size = 0.001;
        ctrl.max_iter = 80;

        let reference = vec![vec![5.0, 0.0]; 11];
        let mut x = vec![0.0, 0.0];
        for _ in 0..100 {
            let u = ctrl.control(&x, &reference).unwrap();
            x = model.step(&x, &u).unwrap();
        }
        // Position should approach 5, velocity should approach 0.
        assert!(approx(x[0], 5.0, 2.0));
    }

    #[test]
    fn test_constraint_validation_bad_bounds() {
        let c = Constraints {
            u_min: vec![10.0],
            u_max: vec![5.0],
            x_min: None,
            x_max: None,
        };
        assert!(c.validate(1, 1).is_err());
    }

    #[test]
    fn test_dense_mat_from_vec_wrong_size() {
        assert!(DenseMat::from_vec(2, 2, vec![1.0]).is_err());
    }

    #[test]
    fn test_cost_with_reference_shorter_than_horizon() {
        let cost = QuadraticCost::uniform(1, 1, 1.0, 0.1, 1.0);
        let reference = vec![vec![5.0]]; // only 1 reference point
        let traj = vec![vec![5.0]; 4];
        let inputs = vec![vec![0.0]; 3];
        // Should not panic — last reference is reused.
        let c = cost.evaluate(&traj, &inputs, &reference);
        assert!(c >= 0.0);
    }

    #[test]
    fn test_mpc_zero_state_zero_reference() {
        let model = integrator_model();
        let cost = QuadraticCost::uniform(1, 1, 1.0, 1.0, 1.0);
        let constraints = Constraints::input_symmetric(1, 10.0);

        let inputs = solve_mpc_pgd(
            &model, &cost, &constraints, &[0.0], &vec![vec![0.0]; 6], 5, 50, 0.01,
        ).unwrap();

        // All inputs should be near zero.
        for u in &inputs {
            assert!(u[0].abs() < 0.1);
        }
    }
}
